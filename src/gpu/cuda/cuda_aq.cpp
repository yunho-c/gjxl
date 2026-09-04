// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/codestream.h"
#include "codec/gaborish_internal.h"
#include "codec/quantization_tables_generated.h"
#include "codec/vardct_frame_internal.h"
#include "core/block_grid.h"
#include "core/frame_geometry.h"
#include "core/quantizer.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_kernels.h"
#include "gpu/ops/aq_evaluation_internal.h"
#include "gpu/scratch.h"
#include "gpu/submission.h"

namespace gjxl::cuda_internal {
namespace {

constexpr size_t kArenaAlignment = 256;

bool HasResidentImage(ConstDeviceImage3View image) noexcept {
  return image.plane[0].buffer != nullptr || image.plane[1].buffer != nullptr ||
         image.plane[2].buffer != nullptr;
}

Status ValidateFiniteImage(ConstImage3FView image, const char* name) {
  if (!image.valid()) {
    return Status::InvalidArgument(std::string(name) + " image is invalid");
  }
  for (ConstPlaneF32View plane : image.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.Row(y)[x])) {
          return Status::InvalidArgument(
              std::string(name) + " image must contain only finite samples");
        }
      }
    }
  }
  return Status::Ok();
}

template <typename T>
void CopyPlaneToContiguous(PlaneView<const T> source, std::vector<T>* out) {
  size_t count = 0;
  (void)source.extent.try_area(&count);
  out->resize(count);
  for (size_t y = 0; y < source.extent.height; ++y) {
    std::copy_n(source.Row(y), source.extent.width,
                out->data() + y * source.extent.width);
  }
}

template <typename T>
void CopyContiguousToPlane(const std::vector<T>& source, PlaneView<T> out) {
  for (size_t y = 0; y < out.extent.height; ++y) {
    std::copy_n(source.data() + y * out.extent.width, out.extent.width,
                out.Row(y));
  }
}

bool ValidInitialOutput(InitialQuantFieldOutput output, Extent2D blocks,
                        Extent2D pixels) noexcept {
  return output.quant_field.valid() && output.quant_field.extent == blocks &&
         output.strategy_mask.valid() &&
         output.strategy_mask.extent == blocks && output.pixel_mask.valid() &&
         output.pixel_mask.extent == pixels;
}

Status PlanPlane(DeviceElementType type, Extent2D extent, size_t row_stride,
                 size_t* bytes) {
  if (bytes == nullptr || extent.empty() || row_stride < extent.width) {
    return Status::InvalidArgument("CUDA AQ arena plan is invalid");
  }
  if (*bytes > std::numeric_limits<size_t>::max() - (kArenaAlignment - 1)) {
    return Status::InvalidArgument("CUDA AQ arena alignment overflows");
  }
  const size_t aligned =
      (*bytes + kArenaAlignment - 1) & ~(kArenaAlignment - 1);
  if (extent.height - 1 >
      (std::numeric_limits<size_t>::max() - extent.width) / row_stride) {
    return Status::InvalidArgument("CUDA AQ plane geometry overflows");
  }
  const size_t elements = (extent.height - 1) * row_stride + extent.width;
  const size_t element_size = DeviceElementSize(type);
  if (element_size == 0 ||
      elements > std::numeric_limits<size_t>::max() / element_size ||
      aligned > std::numeric_limits<size_t>::max() - elements * element_size) {
    return Status::InvalidArgument("CUDA AQ plane size overflows");
  }
  *bytes = aligned + elements * element_size;
  return Status::Ok();
}

}  // namespace

class CudaPreparedAqEvaluation final
    : public PreparedAqEvaluation,
      public aq_evaluation_internal::PreparedAqScaleReconfiguration,
      public aq_evaluation_internal::PreparedAqEncodingInitialQuantization {
 public:
  explicit CudaPreparedAqEvaluation(CudaBackend& backend)
      : backend_(&backend) {}

  Status Prepare(const AqEvaluationPreparation& preparation) {
    if (!preparation.frame_only ||
        !preparation.frame_only_resident_initial_quant ||
        !preparation.frame_only_resident_quantizer ||
        !preparation.resident_initial_cfl ||
        preparation.resident_ac_strategy_inputs ||
        preparation.resident_quantization ||
        HasResidentImage(preparation.resident_coding_opsin) ||
        preparation.coefficient_decision_mode !=
            AcCoefficientDecisionMode::kAdjustedSharedQuant) {
      return Status::Unavailable(
          "CUDA currently supports only the DCT8 maximum-throughput AQ "
          "preparation");
    }
    if (preparation.strategies == nullptr ||
        !preparation.strategies->complete() ||
        !preparation.epf_sharpness.valid() ||
        !preparation.options.profile.valid()) {
      return Status::InvalidArgument("CUDA AQ preparation metadata is invalid");
    }
    Status status = ValidateFiniteImage(preparation.original_linear_rgb,
                                        "CUDA AQ original");
    if (!status.ok()) return status;
    status = ValidateFiniteImage(preparation.coding_opsin, "CUDA AQ coding");
    if (!status.ok()) return status;
    if (!BlockGrid::IsPaddedPixelExtent(preparation.coding_opsin.extent())) {
      return Status::InvalidArgument(
          "CUDA AQ coding image must be padded to complete 8x8 blocks");
    }
    FrameGeometry geometry;
    status = FrameGeometry::Create(preparation.original_linear_rgb.extent(),
                                   &geometry);
    if (!status.ok()) return status;
    const Extent2D pixels = preparation.coding_opsin.extent();
    const Extent2D blocks = geometry.block_grid().blocks;
    if (geometry.padded_frame() != pixels ||
        preparation.strategies->extent() != blocks ||
        preparation.epf_sharpness.extent != blocks ||
        preparation.frame_only_inverse_gaborish !=
            preparation.options.profile.loop_filter.gaborish) {
      return Status::InvalidArgument(
          "CUDA AQ preparation geometry is inconsistent");
    }
    size_t anchors = 0;
    status = preparation.strategies->ForEachAnchor([&](size_t, size_t,
                                                       AcStrategyType
                                                           strategy) {
      if (strategy != AcStrategyType::kDct8) {
        return Status::Unavailable(
            "CUDA maximum-throughput AQ requires an all-DCT8 strategy grid");
      }
      ++anchors;
      return Status::Ok();
    });
    if (!status.ok()) return status;

    size_t pixel_count = 0;
    size_t block_count = 0;
    size_t tile_count = 0;
    const Extent2D tiles = ColorTileExtent(pixels);
    if (!pixels.try_area(&pixel_count) || !blocks.try_area(&block_count) ||
        !tiles.try_area(&tile_count) || anchors != block_count ||
        pixels.width > std::numeric_limits<unsigned int>::max() ||
        pixels.height > std::numeric_limits<unsigned int>::max() ||
        blocks.width > std::numeric_limits<unsigned int>::max() ||
        blocks.height > std::numeric_limits<unsigned int>::max() ||
        tiles.width > std::numeric_limits<unsigned int>::max() ||
        tiles.height > std::numeric_limits<unsigned int>::max() ||
        block_count > std::numeric_limits<unsigned int>::max() / 3 ||
        pixel_count >
            std::numeric_limits<size_t>::max() / (3 * sizeof(float)) ||
        3 * block_count > backend_->state_->maximum_grid_x ||
        (3 * pixel_count + 255) / 256 > backend_->state_->maximum_grid_x) {
      return Status::InvalidArgument(
          "CUDA AQ frame dimensions exceed backend limits");
    }
    for (size_t y = 0; y < blocks.height; ++y) {
      for (size_t x = 0; x < blocks.width; ++x) {
        if (preparation.epf_sharpness.Row(y)[x] >= 8) {
          return Status::InvalidArgument(
              "CUDA AQ EPF sharpness must be in the JPEG XL range");
        }
      }
    }

    source_extent_ = preparation.original_linear_rgb.extent();
    pixel_extent_ = pixels;
    block_extent_ = blocks;
    tile_extent_ = tiles;
    pixel_count_ = pixel_count;
    block_count_ = block_count;
    tile_count_ = tile_count;
    strategies_ = *preparation.strategies;
    evaluation_options_ = preparation.options;
    profile_ = preparation.options.profile;
    inverse_gaborish_ = preparation.frame_only_inverse_gaborish;
    geometry_ = {
        static_cast<unsigned int>(pixels.width),
        static_cast<unsigned int>(pixels.height),
        static_cast<unsigned int>(blocks.width),
        static_cast<unsigned int>(blocks.height),
        static_cast<unsigned int>(tiles.width),
        static_cast<unsigned int>(tiles.height),
    };

    for (size_t channel = 0; channel < 3; ++channel) {
      CopyPlaneToContiguous(preparation.coding_opsin.plane[channel],
                            &coding_host_[channel]);
    }
    CopyPlaneToContiguous(preparation.epf_sharpness, &epf_sharpness_);
    raw_quant_host_.resize(block_count_);
    y_to_x_host_.resize(tile_count_);
    y_to_b_host_.resize(tile_count_);
    quantized_ac_host_.resize(3 * pixel_count_);
    quantized_dc_host_.resize(3 * block_count_);
    transforms_.resize(block_count_);
    for (size_t block = 0; block < block_count_; ++block) {
      transforms_[block] = {
          .block_x = block % block_extent_.width,
          .block_y = block / block_extent_.width,
          .strategy = AcStrategyType::kDct8,
          .coefficient_count = 64,
          .coefficient_offsets =
              {
                  block * 64,
                  pixel_count_ + block * 64,
                  2 * pixel_count_ + block * 64,
              },
      };
    }

    std::array<float, 384> quant_tables{};
    std::copy(quantization_internal::kDct8Dequant.begin(),
              quantization_internal::kDct8Dequant.end(), quant_tables.begin());
    std::copy(quantization_internal::kDct8InverseDequant.begin(),
              quantization_internal::kDct8InverseDequant.end(),
              quant_tables.begin() + 192);
    status = PlanArenas();
    if (status.ok()) status = AllocateArenas();
    if (!status.ok()) return status;
    const std::array<CudaHostToDeviceCopy, 5> uploads{{
        {coding_[0].buffer, coding_host_[0].data(),
         pixel_count_ * sizeof(float), coding_[0].offset_bytes},
        {coding_[1].buffer, coding_host_[1].data(),
         pixel_count_ * sizeof(float), coding_[1].offset_bytes},
        {coding_[2].buffer, coding_host_[2].data(),
         pixel_count_ * sizeof(float), coding_[2].offset_bytes},
        {epf_device_.buffer, epf_sharpness_.data(), block_count_,
         epf_device_.offset_bytes},
        {quant_tables_.buffer, quant_tables.data(), sizeof(quant_tables),
         quant_tables_.offset_bytes},
    }};
    status = backend_->CopyHostToDeviceBatch(uploads);
    if (!status.ok()) return status;

    memory_stats_.persistent_bytes = persistent_.capacity_bytes();
    memory_stats_.staging_bytes = staging_.capacity_bytes();
    memory_stats_.peak_scratch_bytes = staging_.capacity_bytes();
    return Status::Ok();
  }

  Status Evaluate(AqEvaluationInput, AqEvaluationOutput) override {
    return Status::Unavailable(
        "CUDA perceptual AQ evaluation is not implemented yet");
  }

  Status Reconfigure(const AcStrategyGrid& strategies,
                     ConstPlaneU8View epf_sharpness) override {
    if (!strategies.complete() || strategies.extent() != block_extent_ ||
        !epf_sharpness.valid() || epf_sharpness.extent != block_extent_) {
      return Status::InvalidArgument(
          "CUDA AQ reconfiguration geometry is invalid");
    }
    size_t count = 0;
    Status status =
        strategies.ForEachAnchor([&](size_t, size_t, AcStrategyType strategy) {
          if (strategy != AcStrategyType::kDct8) {
            return Status::Unavailable(
                "CUDA maximum-throughput AQ reconfiguration requires DCT8");
          }
          ++count;
          return Status::Ok();
        });
    if (!status.ok()) return status;
    if (count != block_count_) {
      return Status::InvalidArgument("CUDA AQ DCT8 grid is incomplete");
    }
    std::vector<uint8_t> candidate;
    CopyPlaneToContiguous(epf_sharpness, &candidate);
    if (std::any_of(candidate.begin(), candidate.end(),
                    [](uint8_t value) { return value >= 8; })) {
      return Status::InvalidArgument("CUDA AQ EPF sharpness is invalid");
    }
    std::lock_guard lock(mutex_);
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA maximum-throughput AQ evaluation was invalidated");
    }
    status = backend_->CopyHostToDevice(
        *epf_device_.buffer, candidate.data(), candidate.size(),
        epf_device_.offset_bytes);
    if (!status.ok()) return Invalidate(status);
    strategies_ = strategies;
    epf_sharpness_ = std::move(candidate);
    initial_ready_ = false;
    return Status::Ok();
  }

  Status ReconfigureScaleSelectors(AqEvaluationOptions options) override {
    if (!options.profile.valid()) {
      return Status::InvalidArgument(
          "CUDA frame-only AQ scale reconfiguration is invalid");
    }
    std::lock_guard lock(mutex_);
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA maximum-throughput AQ evaluation was invalidated");
    }
    AqEvaluationOptions normalized_previous = evaluation_options_;
    AqEvaluationOptions normalized_current = options;
    normalized_previous.profile.x_qm_scale = 0;
    normalized_previous.profile.b_qm_scale = 0;
    normalized_current.profile.x_qm_scale = 0;
    normalized_current.profile.b_qm_scale = 0;
    if (normalized_previous != normalized_current) {
      return Status::InvalidArgument(
          "CUDA frame-only AQ reconfiguration changes non-scale options");
    }
    evaluation_options_ = options;
    profile_ = options.profile;
    return Status::Ok();
  }

  Status ComputeInitialQuantization(
      InitialQuantizationOptions options, InitialQuantFieldOutput output,
      QuantizerParams* quantizer, float quant_dc,
      ColorCorrelationMap* initial_color_correlation) override {
    if (!ValidInitialOutput(output, block_extent_, pixel_extent_) ||
        !std::isfinite(options.butteraugli_target) ||
        options.butteraugli_target <= 0.0f || !std::isfinite(options.rescale) ||
        options.rescale <= 0.0f || quantizer == nullptr ||
        !std::isfinite(quant_dc) || quant_dc <= 0.0f ||
        quant_dc > static_cast<float>(kMaxQuantDc)) {
      return Status::InvalidArgument(
          "CUDA resident initial-quantization request is invalid");
    }
    if (initial_color_correlation != nullptr) {
      return Status::Unavailable(
          "CUDA initial CfL is materialized by EncodeFrame");
    }
    return ComputeInitialQuantizationImpl(
        options, quantizer, quant_dc, &output);
  }

  Status ComputeInitialQuantizationForEncoding(
      InitialQuantizationOptions options, QuantizerParams* quantizer,
      float quant_dc) override {
    if (!std::isfinite(options.butteraugli_target) ||
        options.butteraugli_target <= 0.0f ||
        !std::isfinite(options.rescale) || options.rescale <= 0.0f ||
        quantizer == nullptr || !std::isfinite(quant_dc) ||
        quant_dc <= 0.0f || quant_dc > static_cast<float>(kMaxQuantDc)) {
      return Status::InvalidArgument(
          "CUDA encoding initial-quantization request is invalid");
    }
    return ComputeInitialQuantizationImpl(options, quantizer, quant_dc,
                                          nullptr);
  }

 private:
  Status ComputeInitialQuantizationImpl(
      InitialQuantizationOptions options, QuantizerParams* quantizer,
      float quant_dc, InitialQuantFieldOutput* output) {
    if (output != nullptr) {
      try {
        quant_field_host_.resize(block_count_);
        strategy_mask_host_.resize(block_count_);
        pixel_mask_host_.resize(pixel_count_);
      } catch (const std::bad_alloc&) {
        return Status::OutOfMemory(
            "Unable to allocate CUDA initial-quantization readback");
      } catch (const std::length_error&) {
        return Status::InvalidArgument(
            "CUDA initial-quantization readback is too large");
      }
    }
    std::lock_guard lock(mutex_);
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA maximum-throughput AQ evaluation was invalidated");
    }
    // A committed initialization writes the resident quantizer and raw-quant
    // state in place. Do not allow an earlier successful initialization to
    // remain observable while replacement work is in flight or after it
    // fails.
    initial_ready_ = false;
    initial_options_ = options;
    quant_dc_ = quant_dc;
    std::unique_ptr<GpuSubmission> submission;
    Status status = backend_->SubmitCompute(
        &CudaPreparedAqEvaluation::EncodeInitial, this, &submission);
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Invalidate(Status::Internal(
          "CUDA initial quantization returned no submission"));
    }
    status = submission->Wait();
    if (!status.ok()) return Invalidate(status);

    unsigned int device_error = 0;
    std::array<unsigned int, 2> params{};
    std::array<CudaDeviceToHostCopy, 5> readbacks{{
        {error_.buffer, &device_error, sizeof(device_error),
         error_.offset_bytes},
        {quantizer_params_.buffer, params.data(), sizeof(params),
         quantizer_params_.offset_bytes},
    }};
    size_t readback_count = 2;
    if (output != nullptr) {
      readbacks[readback_count++] = {
          quant_field_.buffer, quant_field_host_.data(),
          quant_field_host_.size() * sizeof(float), quant_field_.offset_bytes};
      readbacks[readback_count++] = {
          strategy_mask_.buffer, strategy_mask_host_.data(),
          strategy_mask_host_.size() * sizeof(float),
          strategy_mask_.offset_bytes};
      readbacks[readback_count++] = {
          pixel_mask_.buffer, pixel_mask_host_.data(),
          pixel_mask_host_.size() * sizeof(float), pixel_mask_.offset_bytes};
    }
    status = backend_->CopyDeviceToHostBatch(
        std::span<const CudaDeviceToHostCopy>(readbacks).first(
          readback_count));
    if (!status.ok()) return Invalidate(status);
    if (device_error != 0) {
      return Invalidate(Status::DeviceError(
          "CUDA initial quantization detected invalid numeric data"));
    }
    const auto positive = [](const std::vector<float>& values) {
      return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value) && value > 0.0f;
      });
    };
    Quantizer checked;
    const QuantizerParams candidate{params[0], params[1]};
    status = Quantizer::Create(candidate, &checked);
    if (!status.ok() ||
        (output != nullptr &&
         (!positive(quant_field_host_) || !positive(strategy_mask_host_) ||
          !positive(pixel_mask_host_)))) {
      return Invalidate(Status::DeviceError(
          "CUDA initial quantization readback is invalid"));
    }
    if (output != nullptr) {
      CopyContiguousToPlane(quant_field_host_, output->quant_field);
      CopyContiguousToPlane(strategy_mask_host_, output->strategy_mask);
      CopyContiguousToPlane(pixel_mask_host_, output->pixel_mask);
    }
    quantizer_ = candidate;
    *quantizer = candidate;
    initial_ready_ = true;
    return Status::Ok();
  }

  Status EncodeFrame(AqEvaluationInput input,
                     VarDctEncoderFrame* frame) override {
    if (frame == nullptr) {
      return Status::InvalidArgument("CUDA AQ frame output is null");
    }
    if (input.raw_quant_field.valid() || input.y_to_x.valid() ||
        input.y_to_b.valid() || input.epf_inverse_sigma.valid() ||
        input.quant_field.valid() || input.exact_coefficients != nullptr ||
        input.exact_reconstructed_linear_rgb.valid()) {
      return Status::InvalidArgument(
          "CUDA maximum-throughput EncodeFrame requires resident inputs");
    }
    Quantizer checked;
    Status status = Quantizer::Create(input.quantizer, &checked);
    if (!status.ok()) return status;
    std::lock_guard lock(mutex_);
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA maximum-throughput AQ evaluation was invalidated");
    }
    if (!initial_ready_) {
      return Status::FailedPrecondition(
          "CUDA initial quantization must complete before EncodeFrame");
    }
    if (input.quantizer.global_scale != quantizer_.global_scale ||
        input.quantizer.quant_dc != quantizer_.quant_dc) {
      return Status::InvalidArgument(
          "CUDA EncodeFrame quantizer does not match resident initial "
          "quantization");
    }

    std::unique_ptr<GpuSubmission> submission;
    status = backend_->SubmitCompute(
        &CudaPreparedAqEvaluation::EncodeFrameSubmission, this, &submission);
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Status::Internal("CUDA AQ frame encoding returned no submission");
    }
    status = submission->Wait();
    if (!status.ok()) return status;
    unsigned int device_error = 0;
    const std::array<CudaDeviceToHostCopy, 6> readbacks{{
        {error_.buffer, &device_error, sizeof(device_error),
         error_.offset_bytes},
        {raw_quant_work_.buffer, raw_quant_host_.data(),
         raw_quant_host_.size() * sizeof(int32_t),
         raw_quant_work_.offset_bytes},
        {y_to_x_.buffer, y_to_x_host_.data(), tile_count_ * sizeof(int8_t),
         y_to_x_.offset_bytes},
        {y_to_b_.buffer, y_to_b_host_.data(), tile_count_ * sizeof(int8_t),
         y_to_b_.offset_bytes},
        {quantized_ac_.buffer, quantized_ac_host_.data(),
         quantized_ac_host_.size() * sizeof(int32_t),
         quantized_ac_.offset_bytes},
        {quantized_dc_.buffer, quantized_dc_host_.data(),
         quantized_dc_host_.size() * sizeof(int32_t),
         quantized_dc_.offset_bytes},
    }};
    status = backend_->CopyDeviceToHostBatch(readbacks);
    if (!status.ok()) return status;
    if (device_error != 0 ||
        std::any_of(
            raw_quant_host_.begin(), raw_quant_host_.end(),
            [](int32_t value) { return value < 1 || value > kMaxRawQuant; })) {
      return Status::DeviceError("CUDA AQ frame encoding readback is invalid");
    }

    FrameGeometry frame_geometry;
    status = FrameGeometry::Create(source_extent_, &frame_geometry);
    if (!status.ok()) return status;
    ConstImage3I32View quantized_dc;
    quantized_dc.plane[0] = {quantized_dc_host_.data(), block_extent_,
                             block_extent_.width};
    quantized_dc.plane[1] = {quantized_dc_host_.data() + block_count_,
                             block_extent_, block_extent_.width};
    quantized_dc.plane[2] = {quantized_dc_host_.data() + 2 * block_count_,
                             block_extent_, block_extent_.width};
    VarDctEncoderFrame candidate;
    status = vardct_frame_internal::AssembleVarDctEncoderFrame(
        {
            .geometry = frame_geometry,
            .strategies = &strategies_,
            .raw_quant_field = {raw_quant_host_.data(), block_extent_,
                                block_extent_.width},
            .quantizer = &checked,
            .y_to_x = {y_to_x_host_.data(), tile_extent_, tile_extent_.width},
            .y_to_b = {y_to_b_host_.data(), tile_extent_, tile_extent_.width},
            .epf_sharpness = {epf_sharpness_.data(), block_extent_,
                              block_extent_.width},
            .profile = profile_,
            .quantized_dc = quantized_dc,
            .quantized_ac = quantized_ac_host_,
            .transforms = transforms_,
            .reject_unwritten_coefficients = true,
        },
        &candidate);
    if (!status.ok()) return status;
    *frame = std::move(candidate);
    return Status::Ok();
  }

 public:
  AqEvaluationMemoryStats memory_stats() const noexcept override {
    return memory_stats_;
  }

 private:
  Status PlanArenas() {
    size_t persistent_bytes = 0;
    size_t staging_bytes = 0;
    Status status = Status::Ok();
    for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
      status = PlanPlane(DeviceElementType::kF32, pixel_extent_,
                         pixel_extent_.width, &persistent_bytes);
    }
    if (status.ok()) {
      status = PlanPlane(DeviceElementType::kU8, block_extent_,
                         block_extent_.width, &persistent_bytes);
    }
    if (status.ok()) {
      status = PlanPlane(DeviceElementType::kF32, {384, 1}, 384,
                         &persistent_bytes);
    }
    if (!status.ok()) return status;

    const auto plan_staging = [&](DeviceElementType type, Extent2D extent,
                                  size_t stride) {
      return PlanPlane(type, extent, stride, &staging_bytes);
    };
    if (inverse_gaborish_) {
      for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
        status = plan_staging(DeviceElementType::kF32, pixel_extent_,
                              pixel_extent_.width);
      }
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kF32, pixel_extent_,
                            pixel_extent_.width);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kF32, pixel_extent_,
                            pixel_extent_.width);
    }
    if (status.ok()) {
      const Extent2D pre_erosion_extent{pixel_extent_.width / 4,
                                        pixel_extent_.height / 4};
      status = plan_staging(DeviceElementType::kF32, pre_erosion_extent,
                            pre_erosion_extent.width);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kF32, block_extent_,
                            block_extent_.width);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kF32, block_extent_,
                            block_extent_.width);
    }
    if (status.ok()) status = plan_staging(DeviceElementType::kI32, {3, 1}, 3);
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kI32, {256, 1}, 256);
    }
    if (status.ok()) status = plan_staging(DeviceElementType::kF32, {2, 1}, 2);
    if (status.ok()) status = plan_staging(DeviceElementType::kI32, {2, 1}, 2);
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kI32, block_extent_,
                            block_extent_.width);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kI32, block_extent_,
                            block_extent_.width);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kI8, tile_extent_,
                            tile_extent_.width);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kI8, tile_extent_,
                            tile_extent_.width);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kF32, {3 * pixel_count_, 1},
                            3 * pixel_count_);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kF32, {3 * pixel_count_, 1},
                            3 * pixel_count_);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kF32, {4 * block_count_, 1},
                            4 * block_count_);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kI32, {3 * pixel_count_, 1},
                            3 * pixel_count_);
    }
    if (status.ok()) {
      status = plan_staging(DeviceElementType::kI32, {3 * block_count_, 1},
                            3 * block_count_);
    }
    if (status.ok()) status = plan_staging(DeviceElementType::kI32, {1, 1}, 1);
    if (!status.ok()) return status;
    status = persistent_.Prepare(*backend_, persistent_bytes);
    if (!status.ok()) return status;
    return staging_.Prepare(*backend_, staging_bytes);
  }

  Status AllocateArenas() {
    Status status = Status::Ok();
    for (DevicePlaneView& plane : coding_) {
      status = AllocatePlane(persistent_, DeviceElementType::kF32,
                             pixel_extent_, pixel_extent_.width, &plane);
      if (!status.ok()) return status;
    }
    status = AllocatePlane(persistent_, DeviceElementType::kU8, block_extent_,
                           block_extent_.width, &epf_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(persistent_, DeviceElementType::kF32, {384, 1}, 384,
                           &quant_tables_);
    if (!status.ok()) return status;

    if (inverse_gaborish_) {
      for (DevicePlaneView& plane : filtered_) {
        status = AllocatePlane(staging_, DeviceElementType::kF32,
                               pixel_extent_, pixel_extent_.width, &plane);
        if (!status.ok()) return status;
      }
    }
    status = AllocatePlane(staging_, DeviceElementType::kF32, pixel_extent_,
                           pixel_extent_.width, &unblurred_pixel_mask_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32, pixel_extent_,
                           pixel_extent_.width, &pixel_mask_);
    if (!status.ok()) return status;
    const Extent2D pre_erosion_extent{pixel_extent_.width / 4,
                                      pixel_extent_.height / 4};
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           pre_erosion_extent, pre_erosion_extent.width,
                           &pre_erosion_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32, block_extent_,
                           block_extent_.width, &quant_field_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32, block_extent_,
                           block_extent_.width, &strategy_mask_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32, {3, 1}, 3,
                           &selection_state_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32, {256, 1}, 256,
                           &histogram_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32, {2, 1}, 2,
                           &statistics_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32, {2, 1}, 2,
                           &quantizer_params_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32, block_extent_,
                           block_extent_.width, &raw_quant_initial_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32, block_extent_,
                           block_extent_.width, &raw_quant_work_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI8, tile_extent_,
                           tile_extent_.width, &y_to_x_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI8, tile_extent_,
                           tile_extent_.width, &y_to_b_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {3 * pixel_count_, 1}, 3 * pixel_count_, &gathered_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {3 * pixel_count_, 1}, 3 * pixel_count_, &forward_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {4 * block_count_, 1}, 4 * block_count_,
                           &y_thresholds_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32,
                           {3 * pixel_count_, 1}, 3 * pixel_count_,
                           &quantized_ac_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32,
                           {3 * block_count_, 1}, 3 * block_count_,
                           &quantized_dc_);
    if (!status.ok()) return status;
    return AllocatePlane(staging_, DeviceElementType::kI32, {1, 1}, 1,
                         &error_);
  }

  Status AllocatePlane(DeviceScratchArena& arena, DeviceElementType type,
                       Extent2D extent, size_t row_stride,
                       DevicePlaneView* plane) {
    return arena.AllocatePlane(type, extent, row_stride, kArenaAlignment,
                               plane);
  }

  template <typename T>
  Status Read(ConstDevicePlaneView source, T* out, size_t count) {
    return backend_->CopyDeviceToHost(*source.buffer, out, count * sizeof(T),
                                      source.offset_bytes);
  }

  Status Invalidate(Status status) {
    initial_ready_ = false;
    invalid_ = true;
    return status;
  }

  template <typename T>
  static T* Pointer(DevicePlaneView view) {
    CudaBuffer* buffer = CudaBackend::AsCudaBuffer(*view.buffer);
    return reinterpret_cast<T*>(static_cast<std::byte*>(buffer->pointer()) +
                                view.offset_bytes);
  }

  template <typename T>
  static const T* Pointer(ConstDevicePlaneView view) {
    const CudaBuffer* buffer = CudaBackend::AsCudaBuffer(*view.buffer);
    return reinterpret_cast<const T*>(
        static_cast<const std::byte*>(buffer->pointer()) + view.offset_bytes);
  }

  static cudaError_t EncodeInitial(CudaBackend& backend, const void* context) {
    auto& self =
        *static_cast<CudaPreparedAqEvaluation*>(const_cast<void*>(context));
    return LaunchCudaAqInitialQuantization(
        Pointer<const float>(self.coding_[0]),
        Pointer<const float>(self.coding_[1]),
        Pointer<const float>(self.coding_[2]),
        Pointer<float>(self.unblurred_pixel_mask_),
        Pointer<float>(self.pixel_mask_), Pointer<float>(self.pre_erosion_),
        Pointer<float>(self.quant_field_), Pointer<float>(self.strategy_mask_),
        Pointer<unsigned int>(self.selection_state_),
        Pointer<unsigned int>(self.histogram_),
        Pointer<float>(self.statistics_),
        Pointer<unsigned int>(self.quantizer_params_),
        Pointer<int>(self.raw_quant_initial_),
        Pointer<unsigned int>(self.error_),
        self.geometry_, self.initial_options_.butteraugli_target,
        self.initial_options_.rescale, self.quant_dc_, backend.state_->stream);
  }

  static cudaError_t EncodeFrameSubmission(CudaBackend& backend,
                                           const void* context) {
    auto& self =
        *static_cast<CudaPreparedAqEvaluation*>(const_cast<void*>(context));
    cudaError_t status =
        cudaMemcpyAsync(Pointer<int32_t>(self.raw_quant_work_),
                        Pointer<const int32_t>(self.raw_quant_initial_),
                        self.block_count_ * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, backend.state_->stream);
    if (status != cudaSuccess) return status;
    std::array<const float*, 3> transform{};
    for (size_t channel = 0; channel < 3; ++channel) {
      const auto* coding = Pointer<const float>(self.coding_[channel]);
      transform[channel] = coding;
      if (self.inverse_gaborish_) {
        const Symmetric5Weights weights =
            gaborish_internal::GaborishInverseWeights(
                self.profile_.gaborish_inverse_multipliers[channel]);
        auto* filtered = Pointer<float>(self.filtered_[channel]);
        status = LaunchCudaSymmetric5Convolution(
            coding, filtered, self.geometry_.width, self.geometry_.height,
            self.geometry_.width, self.geometry_.width, weights.distance0,
            weights.distance1, weights.distance2, weights.distance4,
            weights.distance8, weights.distance5, backend.state_->stream);
        if (status != cudaSuccess) return status;
        transform[channel] = filtered;
      }
    }
    return LaunchCudaAqEncodeFrame(
        Pointer<const float>(self.coding_[0]),
        Pointer<const float>(self.coding_[1]),
        Pointer<const float>(self.coding_[2]),
        transform[0], transform[1], transform[2],
        Pointer<float>(self.gathered_), Pointer<float>(self.forward_),
        Pointer<const float>(self.quant_tables_),
        Pointer<int>(self.raw_quant_work_),
        Pointer<const unsigned int>(self.quantizer_params_),
        Pointer<signed char>(self.y_to_x_), Pointer<signed char>(self.y_to_b_),
        Pointer<float>(self.y_thresholds_), Pointer<int>(self.quantized_ac_),
        Pointer<int>(self.quantized_dc_), Pointer<unsigned int>(self.error_),
        self.geometry_, QuantizationMatrixMultiplier(self.profile_.x_qm_scale),
        QuantizationMatrixMultiplier(self.profile_.b_qm_scale),
        backend.state_->stream);
  }

  CudaBackend* backend_ = nullptr;
  std::mutex mutex_;
  Extent2D source_extent_;
  Extent2D pixel_extent_;
  Extent2D block_extent_;
  Extent2D tile_extent_;
  size_t pixel_count_ = 0;
  size_t block_count_ = 0;
  size_t tile_count_ = 0;
  CudaAqGeometry geometry_;
  AcStrategyGrid strategies_;
  AqEvaluationOptions evaluation_options_;
  SimpleVarDctCodestreamProfile profile_;
  bool inverse_gaborish_ = false;
  bool initial_ready_ = false;
  bool invalid_ = false;
  InitialQuantizationOptions initial_options_;
  float quant_dc_ = 0.0f;
  QuantizerParams quantizer_;

  std::array<std::vector<float>, 3> coding_host_;
  std::vector<uint8_t> epf_sharpness_;
  std::vector<float> quant_field_host_;
  std::vector<float> strategy_mask_host_;
  std::vector<float> pixel_mask_host_;
  std::vector<int32_t> raw_quant_host_;
  std::vector<int8_t> y_to_x_host_;
  std::vector<int8_t> y_to_b_host_;
  std::vector<int32_t> quantized_ac_host_;
  std::vector<int32_t> quantized_dc_host_;
  std::vector<vardct_frame_internal::QuantizedAcTransformLayout> transforms_;

  DeviceScratchArena persistent_;
  DeviceScratchArena staging_;
  std::array<DevicePlaneView, 3> coding_{};
  std::array<DevicePlaneView, 3> filtered_{};
  DevicePlaneView epf_device_{};
  DevicePlaneView quant_tables_{};
  DevicePlaneView unblurred_pixel_mask_{};
  DevicePlaneView pixel_mask_{};
  DevicePlaneView pre_erosion_{};
  DevicePlaneView quant_field_{};
  DevicePlaneView strategy_mask_{};
  DevicePlaneView selection_state_{};
  DevicePlaneView histogram_{};
  DevicePlaneView statistics_{};
  DevicePlaneView quantizer_params_{};
  DevicePlaneView raw_quant_initial_{};
  DevicePlaneView raw_quant_work_{};
  DevicePlaneView y_to_x_{};
  DevicePlaneView y_to_b_{};
  DevicePlaneView gathered_{};
  DevicePlaneView forward_{};
  DevicePlaneView y_thresholds_{};
  DevicePlaneView quantized_ac_{};
  DevicePlaneView quantized_dc_{};
  DevicePlaneView error_{};

  AqEvaluationMemoryStats memory_stats_;
};

Status CudaBackend::PrepareAqEvaluation(
    const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared) {
  if (prepared == nullptr) {
    return Status::InvalidArgument("CUDA prepared AQ output pointer is null");
  }
  prepared->reset();
  if (!preparation.frame_only && preparation.resident_quantization) {
    return PrepareCudaResidentAqEvaluation(*this, preparation, prepared);
  }
  if (!preparation.frame_only) {
    return PrepareCudaExactAqEvaluation(*this, preparation, prepared);
  }
  try {
    auto result = std::make_unique<CudaPreparedAqEvaluation>(*this);
    Status status = result->Prepare(preparation);
    if (!status.ok()) return status;
    *prepared = std::move(result);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
        "Unable to allocate CUDA prepared AQ evaluation");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("CUDA prepared AQ dimensions are too large");
  }
}

}  // namespace gjxl::cuda_internal
