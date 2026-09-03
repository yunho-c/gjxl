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
#include "gpu/submission.h"

namespace gjxl::cuda_internal {
namespace {

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

}  // namespace

class CudaPreparedAqEvaluation final : public PreparedAqEvaluation {
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
    quant_field_host_.resize(block_count_);
    strategy_mask_host_.resize(block_count_);
    pixel_mask_host_.resize(pixel_count_);
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

    for (size_t channel = 0; channel < 3; ++channel) {
      status = Allocate<float>(pixel_count_, true, &coding_[channel]);
      if (!status.ok()) return status;
      status = backend_->CopyHostToDevice(*coding_[channel],
                                          coding_host_[channel].data(),
                                          pixel_count_ * sizeof(float), 0);
      if (!status.ok()) return status;
      if (inverse_gaborish_) {
        status = Allocate<float>(pixel_count_, false, &filtered_[channel]);
        if (!status.ok()) return status;
      }
    }
    status = Allocate<uint8_t>(block_count_, true, &epf_device_);
    if (!status.ok()) return status;
    status = backend_->CopyHostToDevice(*epf_device_, epf_sharpness_.data(),
                                        block_count_, 0);
    if (!status.ok()) return status;

    std::array<float, 384> quant_tables{};
    std::copy(quantization_internal::kDct8Dequant.begin(),
              quantization_internal::kDct8Dequant.end(), quant_tables.begin());
    std::copy(quantization_internal::kDct8InverseDequant.begin(),
              quantization_internal::kDct8InverseDequant.end(),
              quant_tables.begin() + 192);
    status = Allocate<float>(quant_tables.size(), true, &quant_tables_);
    if (!status.ok()) return status;
    status = backend_->CopyHostToDevice(*quant_tables_, quant_tables.data(),
                                        sizeof(quant_tables), 0);
    if (!status.ok()) return status;

    status = Allocate<float>(pixel_count_, false, &unblurred_pixel_mask_);
    if (!status.ok()) return status;
    status = Allocate<float>(pixel_count_, false, &pixel_mask_);
    if (!status.ok()) return status;
    status = Allocate<float>(pixel_count_ / 16, false, &pre_erosion_);
    if (!status.ok()) return status;
    status = Allocate<float>(block_count_, false, &quant_field_);
    if (!status.ok()) return status;
    status = Allocate<float>(block_count_, false, &strategy_mask_);
    if (!status.ok()) return status;
    status = Allocate<unsigned int>(3, false, &selection_state_);
    if (!status.ok()) return status;
    status = Allocate<unsigned int>(256, false, &histogram_);
    if (!status.ok()) return status;
    status = Allocate<float>(2, false, &statistics_);
    if (!status.ok()) return status;
    status = Allocate<unsigned int>(2, false, &quantizer_params_);
    if (!status.ok()) return status;
    status = Allocate<int32_t>(block_count_, false, &raw_quant_initial_);
    if (!status.ok()) return status;
    status = Allocate<int32_t>(block_count_, false, &raw_quant_work_);
    if (!status.ok()) return status;
    status = Allocate<int8_t>(tile_count_, false, &y_to_x_);
    if (!status.ok()) return status;
    status = Allocate<int8_t>(tile_count_, false, &y_to_b_);
    if (!status.ok()) return status;
    status = Allocate<float>(3 * pixel_count_, false, &gathered_);
    if (!status.ok()) return status;
    status = Allocate<float>(3 * pixel_count_, false, &forward_);
    if (!status.ok()) return status;
    status = Allocate<float>(4 * block_count_, false, &y_thresholds_);
    if (!status.ok()) return status;
    status = Allocate<int32_t>(3 * pixel_count_, false, &quantized_ac_);
    if (!status.ok()) return status;
    status = Allocate<int32_t>(3 * block_count_, false, &quantized_dc_);
    if (!status.ok()) return status;
    status = Allocate<unsigned int>(1, false, &error_);
    if (!status.ok()) return status;

    memory_stats_.persistent_bytes = persistent_bytes_;
    memory_stats_.staging_bytes = staging_bytes_;
    memory_stats_.peak_scratch_bytes = staging_bytes_;
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
    status = backend_->CopyHostToDevice(*epf_device_, candidate.data(),
                                        candidate.size(), 0);
    if (!status.ok()) return Invalidate(status);
    strategies_ = strategies;
    epf_sharpness_ = std::move(candidate);
    initial_ready_ = false;
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
    status = Read(*error_, &device_error, 1);
    if (status.ok())
      status = Read(*quantizer_params_, params.data(), params.size());
    if (status.ok())
      status = Read(*quant_field_, quant_field_host_.data(),
                    quant_field_host_.size());
    if (status.ok())
      status = Read(*strategy_mask_, strategy_mask_host_.data(),
                    strategy_mask_host_.size());
    if (status.ok())
      status =
          Read(*pixel_mask_, pixel_mask_host_.data(), pixel_mask_host_.size());
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
    if (!status.ok() || !positive(quant_field_host_) ||
        !positive(strategy_mask_host_) || !positive(pixel_mask_host_)) {
      return Invalidate(Status::DeviceError(
          "CUDA initial quantization readback is invalid"));
    }
    CopyContiguousToPlane(quant_field_host_, output.quant_field);
    CopyContiguousToPlane(strategy_mask_host_, output.strategy_mask);
    CopyContiguousToPlane(pixel_mask_host_, output.pixel_mask);
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
    status = Read(*error_, &device_error, 1);
    if (status.ok())
      status = Read(*raw_quant_work_, raw_quant_host_.data(),
                    raw_quant_host_.size());
    if (status.ok()) status = Read(*y_to_x_, y_to_x_host_.data(), tile_count_);
    if (status.ok()) status = Read(*y_to_b_, y_to_b_host_.data(), tile_count_);
    if (status.ok())
      status = Read(*quantized_ac_, quantized_ac_host_.data(),
                    quantized_ac_host_.size());
    if (status.ok())
      status = Read(*quantized_dc_, quantized_dc_host_.data(),
                    quantized_dc_host_.size());
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

  AqEvaluationMemoryStats memory_stats() const noexcept override {
    return memory_stats_;
  }

 private:
  template <typename T>
  Status Allocate(size_t count, bool persistent,
                  std::unique_ptr<DeviceBuffer>* out) {
    if (count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      return Status::InvalidArgument("CUDA AQ buffer dimensions are too large");
    }
    const size_t bytes = count * sizeof(T);
    Status status = backend_->Allocate(bytes, out);
    if (status.ok()) {
      if (persistent)
        persistent_bytes_ += bytes;
      else
        staging_bytes_ += bytes;
    }
    return status;
  }

  template <typename T>
  Status Read(const DeviceBuffer& source, T* out, size_t count) {
    return backend_->CopyDeviceToHost(source, out, count * sizeof(T), 0);
  }

  Status Invalidate(Status status) {
    initial_ready_ = false;
    invalid_ = true;
    return status;
  }

  static CudaBuffer* Buffer(std::unique_ptr<DeviceBuffer>& buffer) {
    return dynamic_cast<CudaBuffer*>(buffer.get());
  }

  static const CudaBuffer* Buffer(const std::unique_ptr<DeviceBuffer>& buffer) {
    return dynamic_cast<const CudaBuffer*>(buffer.get());
  }

  static cudaError_t EncodeInitial(CudaBackend& backend, const void* context) {
    auto& self =
        *static_cast<CudaPreparedAqEvaluation*>(const_cast<void*>(context));
    return LaunchCudaAqInitialQuantization(
        static_cast<const float*>(Buffer(self.coding_[0])->pointer()),
        static_cast<const float*>(Buffer(self.coding_[1])->pointer()),
        static_cast<const float*>(Buffer(self.coding_[2])->pointer()),
        static_cast<float*>(Buffer(self.unblurred_pixel_mask_)->pointer()),
        static_cast<float*>(Buffer(self.pixel_mask_)->pointer()),
        static_cast<float*>(Buffer(self.pre_erosion_)->pointer()),
        static_cast<float*>(Buffer(self.quant_field_)->pointer()),
        static_cast<float*>(Buffer(self.strategy_mask_)->pointer()),
        static_cast<unsigned int*>(Buffer(self.selection_state_)->pointer()),
        static_cast<unsigned int*>(Buffer(self.histogram_)->pointer()),
        static_cast<float*>(Buffer(self.statistics_)->pointer()),
        static_cast<unsigned int*>(Buffer(self.quantizer_params_)->pointer()),
        static_cast<int*>(Buffer(self.raw_quant_initial_)->pointer()),
        static_cast<unsigned int*>(Buffer(self.error_)->pointer()),
        self.geometry_, self.initial_options_.butteraugli_target,
        self.initial_options_.rescale, self.quant_dc_, backend.state_->stream);
  }

  static cudaError_t EncodeFrameSubmission(CudaBackend& backend,
                                           const void* context) {
    auto& self =
        *static_cast<CudaPreparedAqEvaluation*>(const_cast<void*>(context));
    cudaError_t status =
        cudaMemcpyAsync(Buffer(self.raw_quant_work_)->pointer(),
                        Buffer(self.raw_quant_initial_)->pointer(),
                        self.block_count_ * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, backend.state_->stream);
    if (status != cudaSuccess) return status;
    std::array<const float*, 3> transform{};
    for (size_t channel = 0; channel < 3; ++channel) {
      const auto* coding =
          static_cast<const float*>(Buffer(self.coding_[channel])->pointer());
      transform[channel] = coding;
      if (self.inverse_gaborish_) {
        const Symmetric5Weights weights =
            gaborish_internal::GaborishInverseWeights(
                self.profile_.gaborish_inverse_multipliers[channel]);
        auto* filtered =
            static_cast<float*>(Buffer(self.filtered_[channel])->pointer());
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
        static_cast<const float*>(Buffer(self.coding_[0])->pointer()),
        static_cast<const float*>(Buffer(self.coding_[1])->pointer()),
        static_cast<const float*>(Buffer(self.coding_[2])->pointer()),
        transform[0], transform[1], transform[2],
        static_cast<float*>(Buffer(self.gathered_)->pointer()),
        static_cast<float*>(Buffer(self.forward_)->pointer()),
        static_cast<const float*>(Buffer(self.quant_tables_)->pointer()),
        static_cast<int*>(Buffer(self.raw_quant_work_)->pointer()),
        static_cast<const unsigned int*>(
            Buffer(self.quantizer_params_)->pointer()),
        static_cast<signed char*>(Buffer(self.y_to_x_)->pointer()),
        static_cast<signed char*>(Buffer(self.y_to_b_)->pointer()),
        static_cast<float*>(Buffer(self.y_thresholds_)->pointer()),
        static_cast<int*>(Buffer(self.quantized_ac_)->pointer()),
        static_cast<int*>(Buffer(self.quantized_dc_)->pointer()),
        static_cast<unsigned int*>(Buffer(self.error_)->pointer()),
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

  std::array<std::unique_ptr<DeviceBuffer>, 3> coding_;
  std::array<std::unique_ptr<DeviceBuffer>, 3> filtered_;
  std::unique_ptr<DeviceBuffer> epf_device_;
  std::unique_ptr<DeviceBuffer> quant_tables_;
  std::unique_ptr<DeviceBuffer> unblurred_pixel_mask_;
  std::unique_ptr<DeviceBuffer> pixel_mask_;
  std::unique_ptr<DeviceBuffer> pre_erosion_;
  std::unique_ptr<DeviceBuffer> quant_field_;
  std::unique_ptr<DeviceBuffer> strategy_mask_;
  std::unique_ptr<DeviceBuffer> selection_state_;
  std::unique_ptr<DeviceBuffer> histogram_;
  std::unique_ptr<DeviceBuffer> statistics_;
  std::unique_ptr<DeviceBuffer> quantizer_params_;
  std::unique_ptr<DeviceBuffer> raw_quant_initial_;
  std::unique_ptr<DeviceBuffer> raw_quant_work_;
  std::unique_ptr<DeviceBuffer> y_to_x_;
  std::unique_ptr<DeviceBuffer> y_to_b_;
  std::unique_ptr<DeviceBuffer> gathered_;
  std::unique_ptr<DeviceBuffer> forward_;
  std::unique_ptr<DeviceBuffer> y_thresholds_;
  std::unique_ptr<DeviceBuffer> quantized_ac_;
  std::unique_ptr<DeviceBuffer> quantized_dc_;
  std::unique_ptr<DeviceBuffer> error_;

  size_t persistent_bytes_ = 0;
  size_t staging_bytes_ = 0;
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
