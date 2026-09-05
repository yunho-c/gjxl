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
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "codec/chroma_from_luma_internal.h"
#include "codec/adaptive_quantization_internal.h"
#include "codec/codestream.h"
#include "codec/gaborish_internal.h"
#include "codec/quantization_tables_generated.h"
#include "codec/vardct_frame_internal.h"
#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/quantizer.h"
#include "gpu/cuda/cuda_aq_exact_kernels.h"
#include "gpu/cuda/cuda_aq_resident_kernels.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_butteraugli_internal.h"
#include "gpu/cuda/cuda_kernels.h"
#include "gpu/ops/aq_evaluation_internal.h"
#include "gpu/scratch.h"
#include "gpu/submission.h"

namespace gjxl::cuda_internal {
namespace {

constexpr size_t kArenaAlignment = 256;
constexpr size_t kQuantTableValueCount = 11904;
constexpr std::array<AcStrategyType, 7> kSupportedStrategies = {
    AcStrategyType::kDct8,     AcStrategyType::kDct16x16,
    AcStrategyType::kDct32x32, AcStrategyType::kDct16x8,
    AcStrategyType::kDct8x16,  AcStrategyType::kDct32x16,
    AcStrategyType::kDct16x32};

static_assert(std::is_standard_layout_v<CudaAqResidentParams>);
static_assert(std::is_trivially_copyable_v<CudaAqResidentParams>);
static_assert(sizeof(CudaAqResidentParams) == 68);
static_assert(std::is_standard_layout_v<CudaAqColorTransformRecord>);
static_assert(std::is_trivially_copyable_v<CudaAqColorTransformRecord>);
static_assert(sizeof(CudaAqColorTransformRecord) == 24);
static_assert(std::is_standard_layout_v<CudaAqResidentPolicyParams>);
static_assert(std::is_trivially_copyable_v<CudaAqResidentPolicyParams>);
static_assert(sizeof(CudaAqResidentPolicyParams) == 32);

template <typename T>
bool ValidHostPlaneLayout(PlaneView<T> plane) noexcept {
  if (!plane.valid()) return false;
  if (plane.extent.height - 1 >
      (std::numeric_limits<size_t>::max() - plane.extent.width) /
          plane.stride) {
    return false;
  }
  const size_t elements =
      (plane.extent.height - 1) * plane.stride + plane.extent.width;
  return elements <=
         std::numeric_limits<size_t>::max() / sizeof(std::remove_const_t<T>);
}

bool FinitePositive(float value) {
  return std::isfinite(value) && value > 0.0f;
}

bool HostPlaneSpecified(ConstPlaneF32View plane) noexcept {
  return plane.data != nullptr || !plane.extent.empty() || plane.stride != 0;
}

bool HostImageSpecified(ConstImage3FView image) noexcept {
  return HostPlaneSpecified(image.plane[0]) ||
         HostPlaneSpecified(image.plane[1]) ||
         HostPlaneSpecified(image.plane[2]);
}

bool MutableImageSpecified(Image3FView image) noexcept {
  return image.plane[0].data != nullptr || image.plane[1].data != nullptr ||
         image.plane[2].data != nullptr || !image.extent().empty();
}

Status ValidateFiniteImage(ConstImage3FView image, const char* name) {
  if (!image.valid() ||
      !std::ranges::all_of(image.plane, [](ConstPlaneF32View plane) {
        return ValidHostPlaneLayout(plane);
      })) {
    return Status::InvalidArgument(std::string(name) + " image is invalid");
  }
  for (ConstPlaneF32View plane : image.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.Row(y)[x])) {
          return Status::InvalidArgument(std::string(name) +
                                         " image contains a non-finite sample");
        }
      }
    }
  }
  return Status::Ok();
}

Status ValidateOptions(const AqEvaluationOptions& options) {
  if (!options.profile.valid()) {
    return Status::InvalidArgument("CUDA resident AQ profile is invalid");
  }
  if (options.metric == AqEvaluationMetric::kButteraugli) {
    if (!FinitePositive(options.butteraugli.hf_asymmetry) ||
        !FinitePositive(options.butteraugli.x_multiplier) ||
        !FinitePositive(options.butteraugli.intensity_target)) {
      return Status::InvalidArgument(
          "CUDA resident AQ Butteraugli options are invalid");
    }
  } else if (options.metric == AqEvaluationMetric::kMaximumError) {
    if (!std::ranges::all_of(options.maximum_error, FinitePositive)) {
      return Status::InvalidArgument(
          "CUDA resident AQ maximum-error limits are invalid");
    }
  } else {
    return Status::InvalidArgument("CUDA resident AQ metric is invalid");
  }
  return Status::Ok();
}

Status PlanPlane(DeviceElementType type, Extent2D extent, size_t row_stride,
                 size_t* bytes) {
  if (bytes == nullptr || extent.empty() || row_stride < extent.width) {
    return Status::InvalidArgument("CUDA resident AQ arena plan is invalid");
  }
  if (*bytes > std::numeric_limits<size_t>::max() - (kArenaAlignment - 1)) {
    return Status::InvalidArgument(
        "CUDA resident AQ arena alignment overflows");
  }
  const size_t aligned =
      (*bytes + kArenaAlignment - 1) & ~(kArenaAlignment - 1);
  if (extent.height - 1 >
      (std::numeric_limits<size_t>::max() - extent.width) / row_stride) {
    return Status::InvalidArgument("CUDA resident AQ plane geometry overflows");
  }
  const size_t elements = (extent.height - 1) * row_stride + extent.width;
  const size_t element_size = DeviceElementSize(type);
  if (element_size == 0 ||
      elements > std::numeric_limits<size_t>::max() / element_size ||
      aligned > std::numeric_limits<size_t>::max() - elements * element_size) {
    return Status::InvalidArgument("CUDA resident AQ plane size overflows");
  }
  *bytes = aligned + elements * element_size;
  return Status::Ok();
}

template <typename T>
Status UploadPlane(CudaBackend& backend, PlaneView<const T> source,
                   DevicePlaneView destination) {
  const size_t row_bytes = source.extent.width * sizeof(T);
  const size_t source_stride_bytes = source.extent.height == 1
    ? row_bytes
    : source.stride * sizeof(T);
  return backend.CopyHostToDevice2D(
    *destination.buffer, source.data, source_stride_bytes, row_bytes,
    source.extent.height, destination.row_stride * sizeof(T),
    destination.offset_bytes);
}

template <typename Range>
void AppendQuantTable(const Range& values, std::vector<float>* packed) {
  packed->insert(packed->end(), values.begin(), values.end());
}

Status PackQuantTables(std::vector<float>* packed) {
  try {
    packed->clear();
    packed->reserve(kQuantTableValueCount);
    AppendQuantTable(quantization_internal::kDct8Dequant, packed);
    AppendQuantTable(quantization_internal::kDct8InverseDequant, packed);
    AppendQuantTable(quantization_internal::kDct16Dequant, packed);
    AppendQuantTable(quantization_internal::kDct16InverseDequant, packed);
    AppendQuantTable(quantization_internal::kDct32Dequant, packed);
    AppendQuantTable(quantization_internal::kDct32InverseDequant, packed);
    AppendQuantTable(quantization_internal::kDct8x16Dequant, packed);
    AppendQuantTable(quantization_internal::kDct8x16InverseDequant, packed);
    AppendQuantTable(quantization_internal::kDct16x32Dequant, packed);
    AppendQuantTable(quantization_internal::kDct16x32InverseDequant, packed);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
        "Unable to pack CUDA resident AQ quantization tables");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
        "CUDA resident AQ quantization table pack is too large");
  }
  return packed->size() == kQuantTableValueCount
             ? Status::Ok()
             : Status::Internal(
                   "CUDA resident AQ quantization table layout changed");
}

size_t StrategyBatchIndex(AcStrategyType strategy) noexcept {
  const auto found = std::ranges::find(kSupportedStrategies, strategy);
  return found == kSupportedStrategies.end()
             ? kSupportedStrategies.size()
             : static_cast<size_t>(found - kSupportedStrategies.begin());
}

}  // namespace

class CudaPreparedResidentAqEvaluation final
    : public PreparedAqEvaluation,
      public aq_evaluation_internal::PreparedAqScaleReconfiguration,
      public aq_evaluation_internal::PreparedAqEncodingInitialQuantization {
 public:
  explicit CudaPreparedResidentAqEvaluation(CudaBackend& backend)
      : backend_(&backend) {}

  size_t reconstruction_staging_bytes_for_test() const noexcept {
    size_t bytes = 0;
    for (const auto& plane : linear_readback_)
      bytes += plane.capacity() * sizeof(float);
    return bytes;
  }

  void PoisonCoefficientReadbackForTest(int32_t value) {
    std::fill_n(quantized_readback_.get(), coefficient_count_, value);
  }

  Status Prepare(const AqEvaluationPreparation& preparation) {
    const bool has_resident_original =
      preparation.resident_original_linear_rgb.plane[0].buffer != nullptr ||
      preparation.resident_original_linear_rgb.plane[1].buffer != nullptr ||
      preparation.resident_original_linear_rgb.plane[2].buffer != nullptr;
    const bool has_resident_coding =
      preparation.resident_coding_opsin.plane[0].buffer != nullptr ||
      preparation.resident_coding_opsin.plane[1].buffer != nullptr ||
      preparation.resident_coding_opsin.plane[2].buffer != nullptr;
    const bool resident_input = has_resident_original && has_resident_coding;
    const bool resident_frontend =
        preparation.resident_initial_cfl &&
        preparation.frame_only_resident_initial_quant &&
        preparation.resident_ac_strategy_inputs;
    const bool partial_resident_frontend =
        preparation.resident_initial_cfl ||
        preparation.frame_only_resident_initial_quant ||
        preparation.resident_ac_strategy_inputs;
    if (preparation.frame_only || !preparation.resident_quantization) {
      return Status::Unavailable(
          "CUDA resident AQ requires complete resident preparation");
    }
    if (preparation.frame_only_inverse_gaborish ||
        preparation.frame_only_resident_quantizer) {
      return Status::Unavailable(
          "CUDA resident AQ received frame-only preparation features");
    }
    if (partial_resident_frontend && !resident_frontend) {
      return Status::Unavailable(
          "CUDA resident AQ requires the complete resident frontend");
    }
    if (has_resident_original != has_resident_coding) {
      return Status::InvalidArgument(
        "CUDA resident AQ external input is incomplete");
    }
    if (preparation.coefficient_decision_mode !=
        AcCoefficientDecisionMode::kAdjustedSharedQuant) {
      return Status::Unavailable(
          "CUDA resident AQ requires adjusted shared quantization");
    }
    Status status = ValidateOptions(preparation.options);
    if (!status.ok()) return status;
    if (resident_input) {
      status = ValidateDeviceImage3View(
        preparation.resident_original_linear_rgb, backend_->id());
      if (status.ok()) {
        status = ValidateDeviceImage3View(
          preparation.resident_coding_opsin, backend_->id());
      }
      if (!status.ok()) return status;
      if (std::ranges::any_of(preparation.resident_original_linear_rgb.plane,
            [](ConstDevicePlaneView plane) {
              return plane.element_type != DeviceElementType::kF32;
            }) ||
          std::ranges::any_of(preparation.resident_coding_opsin.plane,
            [](ConstDevicePlaneView plane) {
              return plane.element_type != DeviceElementType::kF32;
            })) {
        return Status::InvalidArgument(
          "CUDA resident AQ external images must contain floats");
      }
    } else {
      status = ValidateFiniteImage(
        preparation.original_linear_rgb, "CUDA resident AQ original");
      if (status.ok()) {
        status = ValidateFiniteImage(
          preparation.coding_opsin, "CUDA resident AQ coding");
      }
      if (!status.ok()) return status;
    }

    source_extent_ = preparation.original_linear_rgb.extent();
    coding_extent_ = resident_input
                       ? preparation.resident_coding_opsin.plane[0].extent
                       : preparation.coding_opsin.extent();
    if (source_extent_.empty() || coding_extent_.empty() ||
        coding_extent_.width % 8 != 0 || coding_extent_.height % 8 != 0 ||
        source_extent_.width > coding_extent_.width ||
        source_extent_.height > coding_extent_.height ||
        coding_extent_.width - source_extent_.width >= 8 ||
        coding_extent_.height - source_extent_.height >= 8) {
      return Status::InvalidArgument(
          "CUDA resident AQ source and coding geometry are incompatible");
    }
    if (resident_input &&
        preparation.resident_original_linear_rgb.plane[0].extent !=
          source_extent_) {
      return Status::InvalidArgument(
        "CUDA resident AQ external source geometry is inconsistent");
    }
    borrowed_original_ = resident_input
                           ? preparation.resident_original_linear_rgb
                           : ConstDeviceImage3View{};
    borrowed_coding_ = resident_input ? preparation.resident_coding_opsin
                                      : ConstDeviceImage3View{};
    block_extent_ = {coding_extent_.width / 8, coding_extent_.height / 8};
    tile_extent_ = {(coding_extent_.width + 63) / 64,
                    (coding_extent_.height + 63) / 64};
    if (preparation.strategies == nullptr ||
        !preparation.strategies->complete() ||
        preparation.strategies->extent() != block_extent_ ||
        !ValidHostPlaneLayout(preparation.epf_sharpness) ||
        preparation.epf_sharpness.extent != block_extent_) {
      return Status::InvalidArgument(
          "CUDA resident AQ strategy or EPF grid is invalid");
    }
    if (!source_extent_.try_area(&source_count_) ||
        !coding_extent_.try_area(&coding_count_) ||
        !block_extent_.try_area(&block_count_) ||
        !tile_extent_.try_area(&tile_count_) ||
        coding_count_ > std::numeric_limits<size_t>::max() / 3 ||
        block_count_ > std::numeric_limits<uint32_t>::max() ||
        tile_count_ > std::numeric_limits<uint32_t>::max() ||
        coding_extent_.width > std::numeric_limits<uint32_t>::max() ||
        source_extent_.width > std::numeric_limits<uint32_t>::max() ||
        source_extent_.height > std::numeric_limits<uint32_t>::max()) {
      return Status::InvalidArgument(
          "CUDA resident AQ geometry exceeds backend limits");
    }
    coefficient_count_ = 3 * coding_count_;
    options_ = preparation.options;
    resident_frontend_ = resident_frontend;
    filter_stage_count_ =
        (options_.profile.loop_filter.gaborish ? size_t{1} : size_t{0}) +
        options_.profile.loop_filter.epf_options.iterations;
    filter_scratch_count_ = std::min<size_t>(2, filter_stage_count_);
    final_filter_index_ = filter_stage_count_ == 0
                              ? -1
                              : static_cast<int>((filter_stage_count_ - 1) % 2);

    Metadata metadata;
    status = BuildMetadata(*preparation.strategies, preparation.epf_sharpness,
                           &metadata);
    if (!status.ok()) return status;
    const size_t source_dispatches =
        source_count_ / 256 + (source_count_ % 256 != 0);
    const size_t block_dispatches =
        block_count_ / 256 + (block_count_ % 256 != 0);
    if (source_dispatches > backend_->state_->maximum_grid_x ||
        block_dispatches > backend_->state_->maximum_grid_x ||
        block_count_ > backend_->state_->maximum_grid_x ||
        tile_count_ > backend_->state_->maximum_grid_x) {
      return Status::InvalidArgument(
          "CUDA resident AQ geometry exceeds device launch limits");
    }
    for (const CudaAqExactBatch& batch : metadata.batches) {
      const size_t transforms = 3 * static_cast<size_t>(batch.anchor_count);
      const size_t values = transforms * batch.coefficient_count;
      const size_t value_dispatches = values / 256 + (values % 256 != 0);
      if (transforms > backend_->state_->maximum_grid_x ||
          value_dispatches > backend_->state_->maximum_grid_x) {
        return Status::InvalidArgument(
            "CUDA resident AQ strategy batch exceeds device launch limits");
      }
    }
    std::vector<float> quant_tables;
    status = PackQuantTables(&quant_tables);
    if (!status.ok()) return status;
    try {
      block_readback_.resize(block_count_);
      maximum_readback_.resize(3 * block_count_);
      raw_readback_.resize(block_count_);
      // The synchronous readback overwrites every coefficient before any host
      // consumer can observe it. Avoid clearing this full-image array first.
      quantized_readback_ =
          std::make_unique_for_overwrite<int32_t[]>(coefficient_count_);
      quantized_dc_readback_.resize(3 * block_count_);
      y_to_x_readback_.resize(tile_count_);
      y_to_b_readback_.resize(tile_count_);
      quant_field_readback_.resize(block_count_);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
          "Unable to allocate CUDA resident AQ host staging");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
          "CUDA resident AQ host staging is too large");
    }

    status = PlanArenas();
    if (!status.ok()) return status;
    status = AllocateArenas();
    if (!status.ok()) return status;
    if (!resident_input) {
      for (size_t channel = 0; channel < 3; ++channel) {
        status = UploadPlane(*backend_,
          preparation.original_linear_rgb.plane[channel],
          original_[channel]);
        if (status.ok()) {
          status = UploadPlane(*backend_,
            preparation.coding_opsin.plane[channel],
            coding_[channel]);
        }
        if (!status.ok()) return status;
      }
    }
    status = UploadMetadata(metadata, quant_tables);
    if (!status.ok()) return status;
    CommitMetadata(std::move(metadata));
    InitializeKernelParams();

    if (options_.metric == AqEvaluationMetric::kButteraugli) {
      status = PrepareDeviceButteraugli(*backend_,
        {.reference_linear_rgb = OriginalImage(),
          .options = options_.butteraugli},
        &butteraugli_);
      if (!status.ok()) return status;
    }
    const DeviceButteraugliMemoryStats butter_memory =
        butteraugli_ == nullptr ? DeviceButteraugliMemoryStats{}
                                : butteraugli_->memory_stats();
    memory_stats_ = {
        persistent_.capacity_bytes(),
        staging_.capacity_bytes() + butter_memory.prepared_allocation_bytes,
        staging_.capacity_bytes() +
            butter_memory.peak_comparison_scratch_bytes};
    return Status::Ok();
  }

  Status AdjustQuantFieldResident(float butteraugli_target,
                                  ConstPlaneF32View input,
                                  PlaneF32View output) override {
    if (!ValidHostPlaneLayout(input) || !ValidHostPlaneLayout(output) ||
        input.extent != block_extent_ || output.extent != block_extent_ ||
        !FinitePositive(butteraugli_target)) {
      return Status::InvalidArgument(
          "CUDA resident quant-field adjustment input is invalid");
    }
    for (size_t y = 0; y < block_extent_.height; ++y) {
      for (size_t x = 0; x < block_extent_.width; ++x) {
        if (!FinitePositive(input.Row(y)[x])) {
          return Status::InvalidArgument(
              "CUDA resident quant field must contain positive finite values");
        }
      }
    }
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    Status status = UploadPlane(*backend_, input, quant_field_device_);
    if (!status.ok()) return Invalidate(status);
    float mean_max_mixer = 1.0f;
    constexpr float kMixerLimit = 1.54138f;
    constexpr float kMixerSlope = 0.56391f;
    if (butteraugli_target > kMixerLimit) {
      mean_max_mixer =
          std::max(0.0f, mean_max_mixer -
                             (butteraugli_target - kMixerLimit) * kMixerSlope);
    }
    AdjustmentContext context{this, mean_max_mixer};
    std::unique_ptr<GpuSubmission> submission;
    status = backend_->SubmitCompute(
        &CudaPreparedResidentAqEvaluation::EncodeAdjustment, &context,
        &submission);
    if (!status.ok() || submission == nullptr) {
      return Invalidate(
          status.ok() ? Status::Internal("CUDA resident quant-field adjustment "
                                         "returned no submission")
                      : status);
    }
    status = submission->Wait();
    if (!status.ok()) return Invalidate(status);
    status = ReadAndCheckDeviceError();
    if (!status.ok()) return Invalidate(status);
    status = backend_->CopyDeviceToHost(
        *quant_field_device_.buffer, quant_field_readback_.data(),
        block_count_ * sizeof(float), quant_field_device_.offset_bytes);
    if (!status.ok()) return Invalidate(status);
    if (!std::ranges::all_of(quant_field_readback_, FinitePositive)) {
      return Invalidate(Status::DeviceError(
          "CUDA resident quant-field adjustment readback is invalid"));
    }
    for (size_t y = 0; y < block_extent_.height; ++y) {
      std::copy_n(quant_field_readback_.data() + y * block_extent_.width,
                  block_extent_.width, output.Row(y));
    }
    return Status::Ok();
  }

  Status ComputeInitialQuantization(
      InitialQuantizationOptions options, InitialQuantFieldOutput output,
      QuantizerParams* quantizer, float quant_dc,
      ColorCorrelationMap* initial_color_correlation) override {
    if (!resident_frontend_) {
      return Status::FailedPrecondition(
          "CUDA resident initial quantization was not prepared");
    }
    if (!FinitePositive(options.butteraugli_target) ||
        !FinitePositive(options.rescale) ||
        !ValidHostPlaneLayout(output.quant_field) ||
        output.quant_field.extent != block_extent_ ||
        !ValidHostPlaneLayout(output.strategy_mask) ||
        output.strategy_mask.extent != block_extent_ ||
        !ValidHostPlaneLayout(output.pixel_mask) ||
        output.pixel_mask.extent != coding_extent_) {
      return Status::InvalidArgument(
          "CUDA resident initial-quantization request is invalid");
    }
    (void)quantizer;
    (void)quant_dc;
    return ComputeInitialQuantizationImpl(
        options, &output, initial_color_correlation);
  }

  Status ComputeInitialQuantizationForEncoding(
      InitialQuantizationOptions options, QuantizerParams* quantizer,
      float quant_dc) override {
    if (!resident_frontend_ || !FinitePositive(options.butteraugli_target) ||
        !FinitePositive(options.rescale)) {
      return Status::InvalidArgument(
          "CUDA resident encoding initial-quantization request is invalid");
    }
    (void)quantizer;
    (void)quant_dc;
    return ComputeInitialQuantizationImpl(options, nullptr, nullptr);
  }

  Status PrepareResidentEncodingPolicy(
      float butteraugli_target,
      aq_evaluation_internal::ResidentEncodingPolicySetup* setup) override {
    if (setup == nullptr || !FinitePositive(butteraugli_target) ||
        options_.metric != AqEvaluationMetric::kButteraugli) {
      return Status::InvalidArgument(
          "CUDA resident encoding policy request is invalid");
    }
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    if (!resident_initial_ready_) {
      return Status::FailedPrecondition(
          "CUDA resident initial quantization is unavailable");
    }
    resident_encoding_policy_ready_ = false;
    invariant_color_correlation_ready_ = false;
    forward_coefficients_ready_ = false;
    color_correlation_pending_ = false;
    float mean_max_mixer = 1.0f;
    constexpr float kMixerLimit = 1.54138f;
    constexpr float kMixerSlope = 0.56391f;
    if (butteraugli_target > kMixerLimit) {
      mean_max_mixer = std::max(
          0.0f, mean_max_mixer -
                    (butteraugli_target - kMixerLimit) * kMixerSlope);
    }
    AdjustmentContext context{this, mean_max_mixer, true};
    std::unique_ptr<GpuSubmission> submission;
    Status status = backend_->SubmitCompute(
        &CudaPreparedResidentAqEvaluation::EncodeAdjustment, &context,
        &submission);
    if (!status.ok() || submission == nullptr) {
      return Invalidate(
          status.ok()
              ? Status::Internal(
                    "CUDA resident policy preparation returned no submission")
              : status);
    }
    status = submission->Wait();
    if (!status.ok()) return Invalidate(status);
    uint32_t device_error = 0;
    std::array<float, 2> range{};
    const std::array<CudaDeviceToHostCopy, 2> readbacks{{
        {error_device_.buffer, &device_error, sizeof(device_error),
         error_device_.offset_bytes},
        {statistics_device_.buffer, range.data(), sizeof(range),
         statistics_device_.offset_bytes},
    }};
    status = backend_->CopyDeviceToHostBatch(readbacks);
    if (!status.ok()) return Invalidate(status);
    if (device_error != 0) {
      return Invalidate(Status::DeviceError(
          "CUDA resident policy preparation detected invalid numerics"));
    }
    adaptive_quantization_internal::ButteraugliPolicySetup policy_setup;
    status = adaptive_quantization_internal::PrepareButteraugliPolicyFromRange(
        range[0], range[1], butteraugli_target, &policy_setup);
    if (!status.ok()) return Invalidate(status);
    resident_policy_setup_ = {
        policy_setup.quant_dc, policy_setup.lower_bound,
        policy_setup.upper_bound};
    invariant_quant_dc_ = policy_setup.quant_dc;
    invariant_color_correlation_ready_ = true;
    color_correlation_pending_ = true;
    resident_encoding_policy_ready_ = true;
    *setup = resident_policy_setup_;
    return Status::Ok();
  }

  Status GetResidentAcStrategyInputs(
      ResidentAcStrategyInputs* inputs) override {
    if (inputs == nullptr) {
      return Status::InvalidArgument(
          "CUDA resident AC-strategy input output is null");
    }
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    if (!resident_frontend_ || !resident_initial_ready_) {
      return Status::FailedPrecondition(
          "CUDA resident initial quantization is unavailable");
    }
    const std::array<DevicePlaneView, 3>& search_opsin =
        options_.profile.loop_filter.gaborish ? reconstructed_ : coding_;
    *inputs = {.opsin = ConstImage(search_opsin),
               .quant_field = quant_field_device_,
               .pixel_mask = initial_pixel_device_,
               .y_to_x = y_to_x_device_,
               .y_to_b = y_to_b_device_};
    return Status::Ok();
  }

  Status PrepareInvariantColorCorrelationResident(ConstPlaneF32View quant_field,
                                                  float quant_dc) override {
    if (!ValidHostPlaneLayout(quant_field) ||
        quant_field.extent != block_extent_ || !FinitePositive(quant_dc) ||
        quant_dc > static_cast<float>(kMaxQuantDc)) {
      return Status::InvalidArgument(
          "CUDA resident final color-correlation input is invalid");
    }
    for (size_t y = 0; y < block_extent_.height; ++y) {
      for (size_t x = 0; x < block_extent_.width; ++x) {
        if (!FinitePositive(quant_field.Row(y)[x])) {
          return Status::InvalidArgument(
              "CUDA resident final color-correlation field is invalid");
        }
      }
    }
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    invariant_color_correlation_ready_ = false;
    resident_encoding_policy_ready_ = false;
    forward_coefficients_ready_ = false;
    color_correlation_pending_ = false;
    // The block-distance scratch is unused until evaluation. Retain the
    // preparation field there so the first evaluation can derive invariant
    // CfL before reusing the same storage for its output.
    Status status = UploadPlane(*backend_, quant_field, block_device_);
    if (!status.ok()) return Invalidate(status);
    invariant_quant_dc_ = quant_dc;
    invariant_color_correlation_ready_ = true;
    color_correlation_pending_ = true;
    return Status::Ok();
  }

  Status Evaluate(AqEvaluationInput input, AqEvaluationOutput output) override {
    Status status = ValidateInput(input);
    if (status.ok()) status = ValidateOutput(output);
    if (!status.ok()) return status;
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    if (!invariant_color_correlation_ready_) {
      return Status::FailedPrecondition(
          "CUDA resident final color correlation was not prepared");
    }
    const bool reconstruction_requested =
        output.final != nullptr &&
        MutableImageSpecified(output.final->reconstructed_linear_rgb);
    if (reconstruction_requested) {
      status = PrepareReconstructionReadback();
      if (!status.ok()) return status;
    }
    status = UploadPlane(*backend_, input.quant_field, quant_field_device_);
    if (!status.ok()) return Invalidate(status);

    EvaluationContext context{this, input.quant_dc,
                              !forward_coefficients_ready_,
                              color_correlation_pending_};
    std::unique_ptr<GpuSubmission> submission;
    status = backend_->SubmitCompute(
        &CudaPreparedResidentAqEvaluation::EncodeReconstruction, &context,
        &submission);
    if (!status.ok() || submission == nullptr) {
      return Invalidate(
          status.ok()
              ? Status::Internal("CUDA resident AQ returned no submission")
              : status);
    }
    status = submission->Wait();
    if (!status.ok()) return Invalidate(status);
    status = ReadAndCheckDeviceError();
    if (!status.ok()) return Invalidate(status);

    QuantizerParams candidate_params;
    status = backend_->CopyDeviceToHost(
        *quantizer_device_.buffer, &candidate_params, sizeof(candidate_params),
        quantizer_device_.offset_bytes);
    Quantizer candidate_quantizer;
    if (status.ok())
      status = Quantizer::Create(candidate_params, &candidate_quantizer);
    if (!status.ok()) return Invalidate(status);

    double candidate_score = 0.0;
    MaximumErrorReduction candidate_maximum;
    if (options_.metric == AqEvaluationMetric::kButteraugli) {
      status = butteraugli_->Compare(
          {.distorted_linear_rgb = ConstImage(reconstructed_linear_),
           .distance_map = distance_device_,
           .score = score_device_});
      if (!status.ok()) return Invalidate(status);
      status = butteraugli_->ReadScore(&candidate_score);
      if (!status.ok()) return Invalidate(status);
      submission.reset();
      status = backend_->SubmitCompute(
          &CudaPreparedResidentAqEvaluation::EncodeBlockReduction, this,
          &submission);
      if (!status.ok() || submission == nullptr) {
        return Invalidate(
            status.ok()
                ? Status::Internal(
                      "CUDA resident AQ block reduction returned no submission")
                : status);
      }
      status = submission->Wait();
      if (!status.ok()) return Invalidate(status);
      status = ReadAndCheckDeviceError();
      if (!status.ok()) return Invalidate(status);
    }

    status = backend_->CopyDeviceToHost(
        *block_device_.buffer, block_readback_.data(),
        block_count_ * sizeof(float), block_device_.offset_bytes);
    if (!status.ok()) return Invalidate(status);
    if (options_.metric == AqEvaluationMetric::kMaximumError) {
      status = backend_->CopyDeviceToHost(
          *maximum_device_.buffer, maximum_readback_.data(),
          3 * anchor_count_ * sizeof(float), maximum_device_.offset_bytes);
      if (!status.ok()) return Invalidate(status);
      for (size_t anchor = 0; anchor < anchor_count_; ++anchor) {
        for (size_t channel = 0; channel < 3; ++channel) {
          const float value = maximum_readback_[3 * anchor + channel];
          if (!std::isfinite(value) || value < 0.0f) {
            return Invalidate(Status::DeviceError(
                "CUDA resident AQ maximum-error readback is invalid"));
          }
          candidate_maximum.channel_maximum[channel] =
              std::max(candidate_maximum.channel_maximum[channel], value);
        }
      }
      candidate_maximum.normalized_maximum =
          *std::ranges::max_element(block_readback_);
      candidate_score = candidate_maximum.normalized_maximum;
    }
    if (!std::ranges::all_of(block_readback_,
                             [](float value) {
                               return std::isfinite(value) && value >= 0.0f;
                             }) ||
        !std::isfinite(candidate_score) || candidate_score < 0.0) {
      return Invalidate(
          Status::DeviceError("CUDA resident AQ bounded readback is invalid"));
    }

    VarDctEncoderFrame candidate_frame;
    if (output.final != nullptr) {
      status = AssembleFrame(candidate_quantizer, &candidate_frame);
      if (!status.ok()) return Invalidate(status);
    }
    if (reconstruction_requested) {
      const std::array<CudaDeviceToHostCopy, 3> readbacks{{
          {reconstructed_linear_[0].buffer, linear_readback_[0].data(),
           source_count_ * sizeof(float), reconstructed_linear_[0].offset_bytes},
          {reconstructed_linear_[1].buffer, linear_readback_[1].data(),
           source_count_ * sizeof(float), reconstructed_linear_[1].offset_bytes},
          {reconstructed_linear_[2].buffer, linear_readback_[2].data(),
           source_count_ * sizeof(float), reconstructed_linear_[2].offset_bytes},
      }};
      status = backend_->CopyDeviceToHostBatch(readbacks);
      if (!status.ok()) return Invalidate(status);
      for (size_t channel = 0; channel < 3; ++channel) {
        if (!std::ranges::all_of(linear_readback_[channel], [](float value) {
              return std::isfinite(value);
            })) {
          return Invalidate(Status::DeviceError(
              "CUDA resident AQ reconstruction readback is invalid"));
        }
      }
    }

    for (size_t y = 0; y < block_extent_.height; ++y) {
      std::copy_n(block_readback_.data() + y * block_extent_.width,
                  block_extent_.width, output.block_distance_map.Row(y));
    }
    *output.score = candidate_score;
    *output.quantizer = candidate_params;
    if (options_.metric == AqEvaluationMetric::kMaximumError) {
      *output.maximum_error = candidate_maximum;
    }
    if (reconstruction_requested) {
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t y = 0; y < source_extent_.height; ++y) {
          std::copy_n(
              linear_readback_[channel].data() + y * source_extent_.width,
              source_extent_.width,
              output.final->reconstructed_linear_rgb.plane[channel].Row(y));
        }
      }
    }
    if (output.final != nullptr) {
      *output.final->frame = std::move(candidate_frame);
    }
    forward_coefficients_ready_ = true;
    color_correlation_pending_ = false;
    return Status::Ok();
  }

  Status EvaluateResidentButteraugliPolicy(
      AqResidentButteraugliPolicyInput input,
      AqResidentButteraugliPolicyOutput output) override {
    if (options_.metric != AqEvaluationMetric::kButteraugli) {
      return Status::Unavailable(
          "CUDA resident Butteraugli policy was not prepared");
    }
    const bool quant_requested = output.quant_field.data != nullptr ||
                                 !output.quant_field.extent.empty() ||
                                 output.quant_field.stride != 0;
    const bool block_requested = output.block_distance_map.data != nullptr ||
                                 !output.block_distance_map.extent.empty() ||
                                 output.block_distance_map.stride != 0;
    const bool reconstruction_requested =
        MutableImageSpecified(output.reconstructed_linear_rgb);
    if (input.iterations > 4 || !FinitePositive(input.butteraugli_target) ||
        !FinitePositive(input.lower_bound) ||
        !std::isfinite(input.upper_bound) ||
        input.upper_bound < input.lower_bound ||
        input.upper_bound / input.lower_bound >= 253.0f ||
        (quant_requested && (!ValidHostPlaneLayout(output.quant_field) ||
                             output.quant_field.extent != block_extent_)) ||
        (block_requested &&
         (!ValidHostPlaneLayout(output.block_distance_map) ||
          output.block_distance_map.extent != block_extent_)) ||
        output.score_history == nullptr ||
        (reconstruction_requested &&
         (!output.reconstructed_linear_rgb.valid() ||
          output.reconstructed_linear_rgb.extent() != source_extent_ ||
          !std::ranges::all_of(
              output.reconstructed_linear_rgb.plane,
              [](PlaneF32View plane) {
                return ValidHostPlaneLayout(plane);
              }))) ||
        (!input.evaluate_final_field &&
         (input.iterations == 0 || output.frame == nullptr || block_requested ||
          reconstruction_requested))) {
      return Status::InvalidArgument(
          "CUDA resident Butteraugli policy request is invalid");
    }
    const bool resident_policy_input =
        !HostPlaneSpecified(input.adjusted_initial_quant_field);
    Status status = resident_policy_input
      ? (FinitePositive(input.quant_dc) &&
         input.quant_dc <= static_cast<float>(kMaxQuantDc)
           ? Status::Ok()
           : Status::InvalidArgument(
               "CUDA resident policy quantizer is invalid"))
      : ValidateInput({.quant_field = input.adjusted_initial_quant_field,
                       .quant_dc = input.quant_dc});
    if (!status.ok()) return status;

    const size_t score_count =
        input.iterations + static_cast<size_t>(input.evaluate_final_field);
    std::vector<double> candidate_scores;
    try {
      candidate_scores.resize(score_count);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
          "Unable to allocate CUDA resident policy score staging");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
          "CUDA resident policy score staging is too large");
    }

    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    if (!invariant_color_correlation_ready_ || butteraugli_ == nullptr) {
      return Status::FailedPrecondition(
          "CUDA resident policy invariant state is unavailable");
    }
    if (reconstruction_requested) {
      status = PrepareReconstructionReadback();
      if (!status.ok()) return status;
    }
    if (resident_policy_input) {
      if (!resident_encoding_policy_ready_ ||
          input.quant_dc != resident_policy_setup_.quant_dc ||
          input.lower_bound != resident_policy_setup_.lower_bound ||
          input.upper_bound != resident_policy_setup_.upper_bound) {
        return Status::FailedPrecondition(
            "CUDA resident encoding policy setup does not match");
      }
    } else {
      status = UploadPlane(*backend_, input.adjusted_initial_quant_field,
                           quant_field_device_);
      if (!status.ok()) return Invalidate(status);
    }

    PolicyContext context{this, input, score_count,
                          !forward_coefficients_ready_,
                          color_correlation_pending_};
    std::unique_ptr<GpuSubmission> submission;
    status = backend_->SubmitCompute(
        &CudaPreparedResidentAqEvaluation::EncodeResidentPolicy, &context,
        &submission);
    if (!status.ok() || submission == nullptr) {
      return Invalidate(
          status.ok()
              ? Status::Internal(
                    "CUDA resident Butteraugli policy returned no submission")
              : status);
    }
    status = submission->Wait();
    if (!status.ok()) return Invalidate(status);
    status = ReadAndCheckDeviceError();
    if (!status.ok()) return Invalidate(status);

    std::array<CudaDeviceToHostCopy, 3> policy_readbacks{{
        {policy_scores_device_.buffer, policy_score_readback_.data(),
         score_count * sizeof(float), policy_scores_device_.offset_bytes},
    }};
    size_t policy_readback_count = 1;
    if (quant_requested) {
      policy_readbacks[policy_readback_count++] = {
          quant_field_device_.buffer, quant_field_readback_.data(),
          block_count_ * sizeof(float), quant_field_device_.offset_bytes};
    }
    if (block_requested) {
      policy_readbacks[policy_readback_count++] = {
          block_device_.buffer, block_readback_.data(),
          block_count_ * sizeof(float), block_device_.offset_bytes};
    }
    status = backend_->CopyDeviceToHostBatch(
        std::span<const CudaDeviceToHostCopy>(policy_readbacks).first(
          policy_readback_count));
    if (!status.ok()) return Invalidate(status);

    for (size_t index = 0; index < score_count; ++index) {
      const float score = policy_score_readback_[index];
      if (!std::isfinite(score) || score < 0.0f) {
        return Invalidate(Status::DeviceError(
            "CUDA resident policy score readback is invalid"));
      }
      candidate_scores[index] = score;
    }
    if (quant_requested &&
        !std::ranges::all_of(quant_field_readback_, FinitePositive)) {
      return Invalidate(Status::DeviceError(
          "CUDA resident policy quant-field readback is invalid"));
    }
    if (block_requested &&
        !std::ranges::all_of(block_readback_, [](float value) {
          return std::isfinite(value) && value >= 0.0f;
        })) {
      return Invalidate(Status::DeviceError(
          "CUDA resident policy block-map readback is invalid"));
    }

    VarDctEncoderFrame candidate_frame;
    if (output.frame != nullptr) {
      QuantizerParams candidate_params;
      status = backend_->CopyDeviceToHost(
          *quantizer_device_.buffer, &candidate_params,
          sizeof(candidate_params), quantizer_device_.offset_bytes);
      Quantizer candidate_quantizer;
      if (status.ok()) {
        status = Quantizer::Create(candidate_params, &candidate_quantizer);
      }
      if (status.ok())
        status = AssembleFrame(candidate_quantizer, &candidate_frame);
      if (!status.ok()) return Invalidate(status);
    }
    if (reconstruction_requested) {
      const std::array<CudaDeviceToHostCopy, 3> readbacks{{
          {reconstructed_linear_[0].buffer, linear_readback_[0].data(),
           source_count_ * sizeof(float), reconstructed_linear_[0].offset_bytes},
          {reconstructed_linear_[1].buffer, linear_readback_[1].data(),
           source_count_ * sizeof(float), reconstructed_linear_[1].offset_bytes},
          {reconstructed_linear_[2].buffer, linear_readback_[2].data(),
           source_count_ * sizeof(float), reconstructed_linear_[2].offset_bytes},
      }};
      status = backend_->CopyDeviceToHostBatch(readbacks);
      if (!status.ok()) return Invalidate(status);
      for (size_t channel = 0; channel < 3; ++channel) {
        if (!std::ranges::all_of(linear_readback_[channel], [](float value) {
              return std::isfinite(value);
            })) {
          return Invalidate(Status::DeviceError(
              "CUDA resident policy reconstruction readback is invalid"));
        }
      }
    }

    if (quant_requested) {
      for (size_t y = 0; y < block_extent_.height; ++y) {
        std::copy_n(quant_field_readback_.data() + y * block_extent_.width,
                    block_extent_.width, output.quant_field.Row(y));
      }
    }
    if (block_requested) {
      for (size_t y = 0; y < block_extent_.height; ++y) {
        std::copy_n(block_readback_.data() + y * block_extent_.width,
                    block_extent_.width, output.block_distance_map.Row(y));
      }
    }
    *output.score_history = std::move(candidate_scores);
    if (reconstruction_requested) {
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t y = 0; y < source_extent_.height; ++y) {
          std::copy_n(
              linear_readback_[channel].data() + y * source_extent_.width,
              source_extent_.width,
              output.reconstructed_linear_rgb.plane[channel].Row(y));
        }
      }
    }
    if (output.frame != nullptr) *output.frame = std::move(candidate_frame);
    forward_coefficients_ready_ = true;
    color_correlation_pending_ = false;
    return Status::Ok();
  }

  Status Reconfigure(const AcStrategyGrid& strategies,
                     ConstPlaneU8View epf_sharpness) override {
    Metadata metadata;
    Status status = BuildMetadata(strategies, epf_sharpness, &metadata);
    if (!status.ok()) return status;
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    status = UploadMetadata(metadata);
    if (!status.ok()) return Invalidate(status);
    CommitMetadata(std::move(metadata));
    invariant_color_correlation_ready_ = false;
    resident_encoding_policy_ready_ = false;
    forward_coefficients_ready_ = false;
    color_correlation_pending_ = false;
    return Status::Ok();
  }

  Status ReconfigureScaleSelectors(AqEvaluationOptions options) override {
    Status status = ValidateOptions(options);
    if (!status.ok()) return status;
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    AqEvaluationOptions normalized_previous = options_;
    AqEvaluationOptions normalized_current = options;
    normalized_previous.profile.x_qm_scale = 0;
    normalized_previous.profile.b_qm_scale = 0;
    normalized_current.profile.x_qm_scale = 0;
    normalized_current.profile.b_qm_scale = 0;
    if (normalized_previous != normalized_current) {
      return Status::InvalidArgument(
          "CUDA resident AQ reconfiguration changes non-scale options");
    }
    options_ = options;
    forward_coefficients_ready_ = false;
    color_correlation_pending_ = false;
    return Status::Ok();
  }

  AqEvaluationMemoryStats memory_stats() const noexcept override {
    return memory_stats_;
  }

 private:
  struct HostAnchor {
    size_t block_x = 0;
    size_t block_y = 0;
    AcStrategyType strategy = AcStrategyType::kDct8;
    size_t batch_index = 0;
    size_t index_in_batch = 0;
  };

  struct Metadata {
    AcStrategyGrid strategies;
    std::array<CudaAqExactBatch, 7> batches{};
    std::vector<CudaAqAnchor> device_anchors;
    std::vector<HostAnchor> row_major_anchors;
    std::vector<uint8_t> epf_sharpness;
    std::vector<CudaAqColorTransformRecord> color_transforms;
    std::vector<uint32_t> color_tile_offsets;
    std::vector<vardct_frame_internal::QuantizedAcTransformLayout> layouts;
  };

  struct AdjustmentContext {
    CudaPreparedResidentAqEvaluation* self = nullptr;
    float mean_max_mixer = 1.0f;
    bool prepare_encoding_policy = false;
  };

  struct InitialContext {
    CudaPreparedResidentAqEvaluation* self = nullptr;
    InitialQuantizationOptions options;
  };

  struct EvaluationContext {
    CudaPreparedResidentAqEvaluation* self = nullptr;
    float quant_dc = 0.0f;
    bool compute_forward = false;
    bool compute_color_correlation = false;
    bool reset_error = true;
  };

  struct PolicyContext {
    CudaPreparedResidentAqEvaluation* self = nullptr;
    AqResidentButteraugliPolicyInput input;
    size_t score_count = 0;
    bool compute_forward = false;
    bool compute_color_correlation = false;
  };

  Status ComputeInitialQuantizationImpl(
      InitialQuantizationOptions options, InitialQuantFieldOutput* output,
      ColorCorrelationMap* initial_color_correlation) {
    if (output != nullptr) {
      try {
        initial_strategy_readback_.resize(block_count_);
        initial_pixel_readback_.resize(coding_count_);
      } catch (const std::bad_alloc&) {
        return Status::OutOfMemory(
            "Unable to allocate CUDA resident initial readback");
      } catch (const std::length_error&) {
        return Status::InvalidArgument(
            "CUDA resident initial readback is too large");
      }
    }
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA resident AQ evaluation was invalidated");
    }
    resident_initial_ready_ = false;
    resident_encoding_policy_ready_ = false;
    invariant_color_correlation_ready_ = false;
    forward_coefficients_ready_ = false;
    color_correlation_pending_ = false;
    InitialContext context{this, options};
    std::unique_ptr<GpuSubmission> submission;
    Status status = backend_->SubmitCompute(
        &CudaPreparedResidentAqEvaluation::EncodeInitialQuantization, &context,
        &submission);
    if (!status.ok() || submission == nullptr) {
      return Invalidate(
          status.ok()
              ? Status::Internal(
                    "CUDA resident initial quantization returned no submission")
              : status);
    }
    status = submission->Wait();
    if (!status.ok()) return Invalidate(status);
    status = ReadAndCheckDeviceError();
    if (!status.ok()) return Invalidate(status);
    if (output == nullptr) {
      resident_initial_ready_ = true;
      return Status::Ok();
    }

    std::array<CudaDeviceToHostCopy, 5> initial_readbacks{{
        {quant_field_device_.buffer, quant_field_readback_.data(),
         block_count_ * sizeof(float), quant_field_device_.offset_bytes},
        {initial_strategy_device_.buffer, initial_strategy_readback_.data(),
         block_count_ * sizeof(float), initial_strategy_device_.offset_bytes},
        {initial_pixel_device_.buffer, initial_pixel_readback_.data(),
         coding_count_ * sizeof(float), initial_pixel_device_.offset_bytes},
    }};
    size_t initial_readback_count = 3;
    if (initial_color_correlation != nullptr) {
      initial_readbacks[initial_readback_count++] = {
          y_to_x_device_.buffer, y_to_x_readback_.data(),
          tile_count_ * sizeof(int8_t), y_to_x_device_.offset_bytes};
      initial_readbacks[initial_readback_count++] = {
          y_to_b_device_.buffer, y_to_b_readback_.data(),
          tile_count_ * sizeof(int8_t), y_to_b_device_.offset_bytes};
    }
    status = backend_->CopyDeviceToHostBatch(
        std::span<const CudaDeviceToHostCopy>(initial_readbacks).first(
          initial_readback_count));
    if (!status.ok()) return Invalidate(status);
    if (!std::ranges::all_of(quant_field_readback_, FinitePositive) ||
        !std::ranges::all_of(initial_strategy_readback_, FinitePositive) ||
        !std::ranges::all_of(initial_pixel_readback_, FinitePositive)) {
      return Invalidate(Status::DeviceError(
          "CUDA resident initial-quantization readback is invalid"));
    }

    ColorCorrelationMap candidate_color;
    if (initial_color_correlation != nullptr) {
      status = chroma_from_luma_internal::CreateColorCorrelationMap(
          {y_to_x_readback_.data(), tile_extent_, tile_extent_.width},
          {y_to_b_readback_.data(), tile_extent_, tile_extent_.width},
          &candidate_color);
      if (!status.ok()) return Invalidate(status);
    }
    for (size_t y = 0; y < block_extent_.height; ++y) {
      std::copy_n(quant_field_readback_.data() + y * block_extent_.width,
                  block_extent_.width, output->quant_field.Row(y));
      std::copy_n(initial_strategy_readback_.data() + y * block_extent_.width,
                  block_extent_.width, output->strategy_mask.Row(y));
    }
    for (size_t y = 0; y < coding_extent_.height; ++y) {
      std::copy_n(initial_pixel_readback_.data() + y * coding_extent_.width,
                  coding_extent_.width, output->pixel_mask.Row(y));
    }
    if (initial_color_correlation != nullptr) {
      *initial_color_correlation = std::move(candidate_color);
    }
    resident_initial_ready_ = true;
    return Status::Ok();
  }

  Status BuildMetadata(const AcStrategyGrid& strategies,
                       ConstPlaneU8View epf_sharpness,
                       Metadata* metadata) const {
    if (metadata == nullptr || !strategies.complete() ||
        strategies.extent() != block_extent_ ||
        !ValidHostPlaneLayout(epf_sharpness) ||
        epf_sharpness.extent != block_extent_) {
      return Status::InvalidArgument(
          "CUDA resident AQ reconfiguration metadata is invalid");
    }
    try {
      Metadata candidate;
      candidate.strategies = strategies;
      std::array<std::vector<CudaAqAnchor>, 7> grouped;
      candidate.row_major_anchors.reserve(block_count_);
      candidate.device_anchors.reserve(block_count_);
      candidate.epf_sharpness.resize(block_count_);
      for (size_t y = 0; y < block_extent_.height; ++y) {
        std::copy_n(epf_sharpness.Row(y), block_extent_.width,
                    candidate.epf_sharpness.data() + y * block_extent_.width);
        for (size_t x = 0; x < block_extent_.width; ++x) {
          AcStrategyCell cell;
          Status status = strategies.Get(x, y, &cell);
          if (!status.ok()) return status;
          const size_t batch_index = StrategyBatchIndex(cell.strategy);
          if (batch_index >= grouped.size() || epf_sharpness.Row(y)[x] >= 8) {
            return Status::InvalidArgument(
                "CUDA resident AQ strategy or EPF value is unsupported");
          }
          if (cell.is_anchor &&
              !chroma_from_luma_internal::StrategyFitsColorTile(
                  x, y, cell.strategy)) {
            return Status::InvalidArgument(
                "CUDA resident AQ strategy crosses a color tile");
          }
          if (cell.is_anchor) {
            const size_t index_in_batch = grouped[batch_index].size();
            grouped[batch_index].push_back(
                {static_cast<uint32_t>(x), static_cast<uint32_t>(y)});
            candidate.row_major_anchors.push_back(
                {x, y, cell.strategy, batch_index, index_in_batch});
          }
        }
      }
      size_t anchor_offset = 0;
      size_t coefficient_offset = 0;
      for (size_t index = 0; index < grouped.size(); ++index) {
        const AcStrategyInfo* info =
            GetAcStrategyInfo(kSupportedStrategies[index]);
        if (info == nullptr) {
          return Status::Internal(
              "CUDA resident AQ strategy metadata disappeared");
        }
        const size_t count = grouped[index].size();
        const size_t coefficient_count = info->coefficient_count();
        if (anchor_offset > std::numeric_limits<uint32_t>::max() ||
            count > std::numeric_limits<uint32_t>::max() ||
            coefficient_offset > std::numeric_limits<uint32_t>::max() ||
            coefficient_count > std::numeric_limits<uint32_t>::max() ||
            info->pixel_extent().width > std::numeric_limits<uint32_t>::max() ||
            info->pixel_extent().height >
                std::numeric_limits<uint32_t>::max()) {
          return Status::InvalidArgument(
              "CUDA resident AQ strategy metadata exceeds kernel limits");
        }
        candidate.batches[index] = {
            static_cast<uint32_t>(anchor_offset),
            static_cast<uint32_t>(count),
            static_cast<uint32_t>(coefficient_offset),
            static_cast<uint32_t>(coefficient_count),
            static_cast<uint32_t>(info->pixel_extent().width),
            static_cast<uint32_t>(info->pixel_extent().height),
            static_cast<uint32_t>(info->covered_blocks.width),
            static_cast<uint32_t>(info->covered_blocks.height)};
        candidate.device_anchors.insert(candidate.device_anchors.end(),
                                        grouped[index].begin(),
                                        grouped[index].end());
        anchor_offset += count;
        if (count != 0 &&
            coefficient_count >
                (std::numeric_limits<size_t>::max() - coefficient_offset) /
                    (3 * count)) {
          return Status::InvalidArgument(
              "CUDA resident AQ coefficient metadata overflows");
        }
        coefficient_offset += 3 * count * coefficient_count;
      }
      if (anchor_offset != candidate.row_major_anchors.size() ||
          coefficient_offset != coefficient_count_ ||
          anchor_offset > block_count_) {
        return Status::Internal(
            "CUDA resident AQ strategies do not cover the coding image");
      }

      candidate.layouts.reserve(candidate.row_major_anchors.size());
      for (const HostAnchor& anchor : candidate.row_major_anchors) {
        const CudaAqExactBatch& batch = candidate.batches[anchor.batch_index];
        const size_t channel_stride =
            static_cast<size_t>(batch.anchor_count) * batch.coefficient_count;
        vardct_frame_internal::QuantizedAcTransformLayout layout{
            .block_x = anchor.block_x,
            .block_y = anchor.block_y,
            .strategy = anchor.strategy,
            .coefficient_count = batch.coefficient_count};
        for (size_t channel = 0; channel < 3; ++channel) {
          layout.coefficient_offsets[channel] =
              batch.coefficient_offset + channel * channel_stride +
              anchor.index_in_batch * batch.coefficient_count;
        }
        candidate.layouts.push_back(layout);
      }

      candidate.color_tile_offsets.assign(tile_count_ + 1, 0);
      for (const HostAnchor& anchor : candidate.row_major_anchors) {
        const size_t tile =
            (anchor.block_y / 8) * tile_extent_.width + anchor.block_x / 8;
        if (tile >= tile_count_ || candidate.color_tile_offsets[tile + 1] ==
                                       std::numeric_limits<uint32_t>::max()) {
          return Status::InvalidArgument(
              "CUDA resident AQ color transform metadata is too large");
        }
        ++candidate.color_tile_offsets[tile + 1];
      }
      for (size_t tile = 0; tile < tile_count_; ++tile) {
        candidate.color_tile_offsets[tile + 1] +=
            candidate.color_tile_offsets[tile];
      }
      candidate.color_transforms.resize(anchor_offset);
      std::vector<uint32_t> positions = candidate.color_tile_offsets;
      std::vector<uint32_t> tile_value_offsets(tile_count_, 0);
      for (const HostAnchor& anchor : candidate.row_major_anchors) {
        const size_t tile =
            (anchor.block_y / 8) * tile_extent_.width + anchor.block_x / 8;
        const CudaAqExactBatch& batch = candidate.batches[anchor.batch_index];
        const size_t channel_stride =
            static_cast<size_t>(batch.anchor_count) * batch.coefficient_count;
        const size_t coefficient_offset =
            batch.coefficient_offset +
            anchor.index_in_batch * batch.coefficient_count;
        const size_t raw_quant_index =
            anchor.block_y * block_extent_.width + anchor.block_x;
        if (channel_stride > std::numeric_limits<uint32_t>::max() ||
            coefficient_offset > std::numeric_limits<uint32_t>::max() ||
            raw_quant_index > std::numeric_limits<uint32_t>::max()) {
          return Status::InvalidArgument(
              "CUDA resident AQ color transform record exceeds limits");
        }
        candidate.color_transforms[positions[tile]++] = {
            static_cast<uint32_t>(coefficient_offset),
            static_cast<uint32_t>(channel_stride),
            batch.coefficient_count,
            static_cast<uint32_t>(anchor.strategy),
            static_cast<uint32_t>(raw_quant_index),
            tile_value_offsets[tile]};
        tile_value_offsets[tile] += batch.coefficient_count;
      }
      *metadata = std::move(candidate);
      return Status::Ok();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
          "Unable to allocate CUDA resident AQ strategy metadata");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
          "CUDA resident AQ strategy metadata is too large");
    }
  }

  Status PlanArenas() {
    size_t persistent_bytes = 0;
    size_t staging_bytes = 0;
    Status status = Status::Ok();
    if (!has_borrowed_input()) {
      for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
        status = PlanPlane(DeviceElementType::kF32,
          source_extent_,
          source_extent_.width,
          &persistent_bytes);
      }
    }
    const size_t coding_image_count = has_borrowed_input() ? 1 : 2;
    for (size_t image = 0; image < coding_image_count && status.ok(); ++image) {
      for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
        status = PlanPlane(DeviceElementType::kF32, coding_extent_,
                           coding_extent_.width, &persistent_bytes);
      }
    }
    for (size_t image = 0; image < filter_scratch_count_ && status.ok();
         ++image) {
      for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
        status = PlanPlane(DeviceElementType::kF32, coding_extent_,
                           coding_extent_.width, &persistent_bytes);
      }
    }
    for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
      status = PlanPlane(DeviceElementType::kF32, source_extent_,
                         source_extent_.width, &persistent_bytes);
    }
    if (status.ok()) {
      status = PlanPlane(DeviceElementType::kI32, {2 * block_count_, 1},
                         2 * block_count_, &persistent_bytes);
    }
    if (status.ok()) {
      status = PlanPlane(DeviceElementType::kU8, block_extent_,
                         block_extent_.width, &persistent_bytes);
    }
    if (status.ok()) {
      status = PlanPlane(DeviceElementType::kF32, {kQuantTableValueCount, 1},
                         kQuantTableValueCount, &persistent_bytes);
    }
    if (status.ok()) {
      status = PlanPlane(DeviceElementType::kI32, {6 * block_count_, 1},
                         6 * block_count_, &persistent_bytes);
    }
    if (status.ok()) {
      status = PlanPlane(DeviceElementType::kI32, {tile_count_ + 1, 1},
                         tile_count_ + 1, &persistent_bytes);
    }
    for (size_t map = 0; map < 2 && status.ok(); ++map) {
      status = PlanPlane(DeviceElementType::kI8, tile_extent_,
                         tile_extent_.width, &persistent_bytes);
    }
    if (!status.ok()) return status;

    const auto plan_staging = [&](DeviceElementType type, Extent2D extent,
                                  size_t stride) {
      return PlanPlane(type, extent, stride, &staging_bytes);
    };
    status = plan_staging(DeviceElementType::kF32, block_extent_,
                          block_extent_.width);
    if (resident_frontend_ && status.ok()) {
      status = plan_staging(DeviceElementType::kF32, coding_extent_,
                            coding_extent_.width);
      if (status.ok()) {
        status = plan_staging(DeviceElementType::kF32, coding_extent_,
                              coding_extent_.width);
      }
      if (status.ok()) {
        status =
            plan_staging(DeviceElementType::kF32,
                         {coding_extent_.width / 4, coding_extent_.height / 4},
                         coding_extent_.width / 4);
      }
      if (status.ok()) {
        status = plan_staging(DeviceElementType::kF32, block_extent_,
                              block_extent_.width);
      }
    }
    if (options_.metric == AqEvaluationMetric::kButteraugli && status.ok()) {
      status = plan_staging(DeviceElementType::kF32, block_extent_,
                            block_extent_.width);
      if (status.ok()) {
        status = plan_staging(DeviceElementType::kF32, {5, 1}, 5);
      }
    }
    if (status.ok()) status = plan_staging(DeviceElementType::kI32, {3, 1}, 3);
    if (status.ok())
      status = plan_staging(DeviceElementType::kI32, {256, 1}, 256);
    if (status.ok()) status = plan_staging(DeviceElementType::kF32, {2, 1}, 2);
    if (status.ok()) status = plan_staging(DeviceElementType::kI32, {2, 1}, 2);
    if (status.ok())
      status = plan_staging(DeviceElementType::kI32, block_extent_,
                            block_extent_.width);
    for (size_t index = 0; index < 2 && status.ok(); ++index) {
      status = plan_staging(DeviceElementType::kF32, {coefficient_count_, 1},
                            coefficient_count_);
    }
    if (status.ok())
      status = plan_staging(DeviceElementType::kF32, {coefficient_count_, 1},
                            coefficient_count_);
    if (status.ok())
      status = plan_staging(DeviceElementType::kI32, {coefficient_count_, 1},
                            coefficient_count_);
    if (status.ok())
      status = plan_staging(DeviceElementType::kF32, {coefficient_count_, 1},
                            coefficient_count_);
    if (status.ok())
      status = plan_staging(DeviceElementType::kF32, {3 * block_count_, 1},
                            3 * block_count_);
    if (status.ok())
      status = plan_staging(DeviceElementType::kI32, {3 * block_count_, 1},
                            3 * block_count_);
    if (status.ok())
      status = plan_staging(DeviceElementType::kF32, {coefficient_count_, 1},
                            coefficient_count_);
    if (status.ok())
      status = plan_staging(DeviceElementType::kF32, block_extent_,
                            block_extent_.width);
    if (options_.metric == AqEvaluationMetric::kButteraugli && status.ok()) {
      status = plan_staging(DeviceElementType::kF32, source_extent_,
                            source_extent_.width);
      if (status.ok())
        status = plan_staging(DeviceElementType::kF32, {1, 1}, 1);
    }
    if (status.ok())
      status = plan_staging(DeviceElementType::kF32, block_extent_,
                            block_extent_.width);
    if (options_.metric == AqEvaluationMetric::kMaximumError && status.ok()) {
      status = plan_staging(DeviceElementType::kF32, {3 * block_count_, 1},
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
    if (!has_borrowed_input()) {
      for (DevicePlaneView& plane : original_) {
        status = AllocatePlane(persistent_,
          DeviceElementType::kF32,
          source_extent_,
          source_extent_.width,
          &plane);
        if (!status.ok()) return status;
      }
    }
    if (!has_borrowed_input()) {
      for (DevicePlaneView& plane : coding_) {
        status = AllocatePlane(persistent_,
          DeviceElementType::kF32,
          coding_extent_,
          coding_extent_.width,
          &plane);
        if (!status.ok()) return status;
      }
    }
    for (DevicePlaneView& plane : reconstructed_) {
      status = AllocatePlane(persistent_, DeviceElementType::kF32,
                             coding_extent_, coding_extent_.width, &plane);
      if (!status.ok()) return status;
    }
    for (size_t image = 0; image < filter_scratch_count_; ++image) {
      for (DevicePlaneView& plane : filter_scratch_[image]) {
        status = AllocatePlane(persistent_, DeviceElementType::kF32,
                               coding_extent_, coding_extent_.width, &plane);
        if (!status.ok()) return status;
      }
    }
    for (DevicePlaneView& plane : reconstructed_linear_) {
      status = AllocatePlane(persistent_, DeviceElementType::kF32,
                             source_extent_, source_extent_.width, &plane);
      if (!status.ok()) return status;
    }
    status = AllocatePlane(persistent_, DeviceElementType::kI32,
                           {2 * block_count_, 1}, 2 * block_count_,
                           &anchors_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(persistent_, DeviceElementType::kU8, block_extent_,
                           block_extent_.width, &epf_sharpness_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(persistent_, DeviceElementType::kF32,
                           {kQuantTableValueCount, 1}, kQuantTableValueCount,
                           &quant_tables_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(persistent_, DeviceElementType::kI32,
                           {6 * block_count_, 1}, 6 * block_count_,
                           &color_transforms_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(persistent_, DeviceElementType::kI32,
                           {tile_count_ + 1, 1}, tile_count_ + 1,
                           &color_tile_offsets_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(persistent_, DeviceElementType::kI8, tile_extent_,
                           tile_extent_.width, &y_to_x_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(persistent_, DeviceElementType::kI8, tile_extent_,
                           tile_extent_.width, &y_to_b_device_);
    if (!status.ok()) return status;

    status = AllocatePlane(staging_, DeviceElementType::kF32, block_extent_,
                           block_extent_.width, &quant_field_device_);
    if (!status.ok()) return status;
    if (resident_frontend_) {
      status =
          AllocatePlane(staging_, DeviceElementType::kF32, coding_extent_,
                        coding_extent_.width, &initial_unblurred_pixel_device_);
      if (!status.ok()) return status;
      status = AllocatePlane(staging_, DeviceElementType::kF32, coding_extent_,
                             coding_extent_.width, &initial_pixel_device_);
      if (!status.ok()) return status;
      const Extent2D pre_erosion_extent{coding_extent_.width / 4,
                                        coding_extent_.height / 4};
      status =
          AllocatePlane(staging_, DeviceElementType::kF32, pre_erosion_extent,
                        pre_erosion_extent.width, &initial_pre_erosion_device_);
      if (!status.ok()) return status;
      status = AllocatePlane(staging_, DeviceElementType::kF32, block_extent_,
                             block_extent_.width, &initial_strategy_device_);
      if (!status.ok()) return status;
    }
    if (options_.metric == AqEvaluationMetric::kButteraugli) {
      status =
          AllocatePlane(staging_, DeviceElementType::kF32, block_extent_,
                        block_extent_.width, &policy_initial_field_device_);
      if (!status.ok()) return status;
      status = AllocatePlane(staging_, DeviceElementType::kF32, {5, 1}, 5,
                             &policy_scores_device_);
      if (!status.ok()) return status;
    }
    status = AllocatePlane(staging_, DeviceElementType::kI32, {3, 1}, 3,
                           &selection_state_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32, {256, 1}, 256,
                           &histogram_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32, {2, 1}, 2,
                           &statistics_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32, {2, 1}, 2,
                           &quantizer_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32, block_extent_,
                           block_extent_.width, &raw_quant_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {coefficient_count_, 1}, coefficient_count_,
                           &gathered_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {coefficient_count_, 1}, coefficient_count_,
                           &forward_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {coefficient_count_, 1}, coefficient_count_,
                           &thresholds_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kI32,
                           {coefficient_count_, 1}, coefficient_count_,
                           &quantized_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {coefficient_count_, 1}, coefficient_count_,
                           &reconstruction_coefficients_device_);
    if (!status.ok()) return status;
    status =
        AllocatePlane(staging_, DeviceElementType::kF32, {3 * block_count_, 1},
                      3 * block_count_, &dc_device_);
    if (!status.ok()) return status;
    status =
        AllocatePlane(staging_, DeviceElementType::kI32, {3 * block_count_, 1},
                      3 * block_count_, &quantized_dc_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {coefficient_count_, 1}, coefficient_count_,
                           &inverse_device_);
    if (!status.ok()) return status;
    status = AllocatePlane(staging_, DeviceElementType::kF32, block_extent_,
                           block_extent_.width, &inverse_sigma_device_);
    if (!status.ok()) return status;
    if (options_.metric == AqEvaluationMetric::kButteraugli) {
      status = AllocatePlane(staging_, DeviceElementType::kF32, source_extent_,
                             source_extent_.width, &distance_device_);
      if (!status.ok()) return status;
      status = AllocatePlane(staging_, DeviceElementType::kF32, {1, 1}, 1,
                             &score_device_);
      if (!status.ok()) return status;
    }
    status = AllocatePlane(staging_, DeviceElementType::kF32, block_extent_,
                           block_extent_.width, &block_device_);
    if (!status.ok()) return status;
    if (options_.metric == AqEvaluationMetric::kMaximumError) {
      status = AllocatePlane(staging_, DeviceElementType::kF32,
                             {3 * block_count_, 1}, 3 * block_count_,
                             &maximum_device_);
      if (!status.ok()) return status;
    }
    return AllocatePlane(staging_, DeviceElementType::kI32, {1, 1}, 1,
                         &error_device_);
  }

  Status AllocatePlane(DeviceScratchArena& arena, DeviceElementType type,
                       Extent2D extent, size_t row_stride,
                       DevicePlaneView* plane) {
    return arena.AllocatePlane(type, extent, row_stride, kArenaAlignment,
                               plane);
  }

  Status UploadMetadata(
      const Metadata& metadata,
      std::span<const float> quant_tables = {}) {
    std::array<CudaHostToDeviceCopy, 5> uploads{{
        {anchors_device_.buffer, metadata.device_anchors.data(),
         metadata.device_anchors.size() * sizeof(CudaAqAnchor),
         anchors_device_.offset_bytes},
        {epf_sharpness_device_.buffer, metadata.epf_sharpness.data(),
         metadata.epf_sharpness.size(), epf_sharpness_device_.offset_bytes},
        {color_transforms_device_.buffer, metadata.color_transforms.data(),
         metadata.color_transforms.size() * sizeof(CudaAqColorTransformRecord),
         color_transforms_device_.offset_bytes},
        {color_tile_offsets_device_.buffer, metadata.color_tile_offsets.data(),
         metadata.color_tile_offsets.size() * sizeof(uint32_t),
         color_tile_offsets_device_.offset_bytes},
    }};
    size_t upload_count = 4;
    if (!quant_tables.empty()) {
      uploads[upload_count++] = {
          quant_tables_device_.buffer, quant_tables.data(),
          quant_tables.size_bytes(), quant_tables_device_.offset_bytes};
    }
    return backend_->CopyHostToDeviceBatch(
        std::span<const CudaHostToDeviceCopy>(uploads).first(upload_count));
  }

  void CommitMetadata(Metadata metadata) {
    batches_ = metadata.batches;
    row_major_anchors_ = std::move(metadata.row_major_anchors);
    epf_sharpness_ = std::move(metadata.epf_sharpness);
    layouts_ = std::move(metadata.layouts);
    strategies_ = std::move(metadata.strategies);
    anchor_count_ = row_major_anchors_.size();
  }

  Status ValidateInput(AqEvaluationInput input) const {
    const auto plane_i32_specified = [](ConstPlaneI32View plane) {
      return plane.data != nullptr || !plane.extent.empty() ||
             plane.stride != 0;
    };
    const auto plane_i8_specified = [](ConstPlaneI8View plane) {
      return plane.data != nullptr || !plane.extent.empty() ||
             plane.stride != 0;
    };
    if (!ValidHostPlaneLayout(input.quant_field) ||
        input.quant_field.extent != block_extent_ ||
        !FinitePositive(input.quant_dc) ||
        input.quant_dc > static_cast<float>(kMaxQuantDc) ||
        plane_i32_specified(input.raw_quant_field) ||
        plane_i8_specified(input.y_to_x) || plane_i8_specified(input.y_to_b) ||
        HostPlaneSpecified(input.epf_inverse_sigma) ||
        input.exact_coefficients != nullptr ||
        HostImageSpecified(input.exact_reconstructed_linear_rgb)) {
      return Status::InvalidArgument("CUDA resident AQ input is invalid");
    }
    for (size_t y = 0; y < block_extent_.height; ++y) {
      for (size_t x = 0; x < block_extent_.width; ++x) {
        if (!FinitePositive(input.quant_field.Row(y)[x])) {
          return Status::InvalidArgument(
              "CUDA resident AQ quant field is invalid");
        }
      }
    }
    return Status::Ok();
  }

  Status ValidateOutput(AqEvaluationOutput output) const {
    if (!ValidHostPlaneLayout(output.block_distance_map) ||
        output.block_distance_map.extent != block_extent_ ||
        output.score == nullptr || output.quantizer == nullptr) {
      return Status::InvalidArgument("CUDA resident AQ output is invalid");
    }
    if ((options_.metric == AqEvaluationMetric::kMaximumError) !=
        (output.maximum_error != nullptr)) {
      return Status::InvalidArgument(
          "CUDA resident AQ maximum-error output is inconsistent");
    }
    if (output.final == nullptr) return Status::Ok();
    if (output.final->frame == nullptr) {
      return Status::InvalidArgument(
          "CUDA resident AQ final frame output is null");
    }
    if (MutableImageSpecified(output.final->reconstructed_linear_rgb) &&
        (!output.final->reconstructed_linear_rgb.valid() ||
         output.final->reconstructed_linear_rgb.extent() != source_extent_ ||
         !std::ranges::all_of(
             output.final->reconstructed_linear_rgb.plane,
             [](PlaneF32View plane) { return ValidHostPlaneLayout(plane); }))) {
      return Status::InvalidArgument(
          "CUDA resident AQ reconstruction output is invalid");
    }
    return Status::Ok();
  }

  Status PrepareReconstructionReadback() {
    // Encoding-only requests never read reconstructed RGB back to the host.
    // Keep transactional staging for diagnostic outputs, but allocate it only
    // on first use and retain it for subsequent evaluations of this object.
    if (linear_readback_[0].size() == source_count_) return Status::Ok();
    try {
      std::array<std::vector<float>, 3> candidate;
      for (auto& plane : candidate) plane.resize(source_count_);
      linear_readback_ = std::move(candidate);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
          "Unable to allocate CUDA resident reconstruction host staging");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
          "CUDA resident reconstruction host staging is too large");
    }
    return Status::Ok();
  }

  Status AssembleFrame(const Quantizer& quantizer, VarDctEncoderFrame* frame) {
    const std::array<CudaDeviceToHostCopy, 5> readbacks{{
        {raw_quant_device_.buffer, raw_readback_.data(),
         block_count_ * sizeof(int32_t), raw_quant_device_.offset_bytes},
        {quantized_device_.buffer, quantized_readback_.get(),
         coefficient_count_ * sizeof(int32_t), quantized_device_.offset_bytes},
        {quantized_dc_device_.buffer, quantized_dc_readback_.data(),
         3 * block_count_ * sizeof(int32_t),
         quantized_dc_device_.offset_bytes},
        {y_to_x_device_.buffer, y_to_x_readback_.data(),
         tile_count_ * sizeof(int8_t), y_to_x_device_.offset_bytes},
        {y_to_b_device_.buffer, y_to_b_readback_.data(),
         tile_count_ * sizeof(int8_t), y_to_b_device_.offset_bytes},
    }};
    Status status = backend_->CopyDeviceToHostBatch(readbacks);
    if (!status.ok()) return status;
    if (!std::ranges::all_of(raw_readback_, [](int32_t value) {
          return value >= 1 && value <= kMaxRawQuant;
        })) {
      return Status::DeviceError(
          "CUDA resident AQ raw-quant readback is invalid");
    }
    FrameGeometry geometry;
    status = FrameGeometry::Create(source_extent_, &geometry);
    if (!status.ok()) return status;
    ConstImage3I32View quantized_dc;
    quantized_dc.plane[0] = {quantized_dc_readback_.data(), block_extent_,
                             block_extent_.width};
    quantized_dc.plane[1] = {quantized_dc_readback_.data() + block_count_,
                             block_extent_, block_extent_.width};
    quantized_dc.plane[2] = {quantized_dc_readback_.data() + 2 * block_count_,
                             block_extent_, block_extent_.width};
    return vardct_frame_internal::AssembleVarDctEncoderFrame(
        {.geometry = geometry,
         .strategies = &strategies_,
         .raw_quant_field = {raw_readback_.data(), block_extent_,
                             block_extent_.width},
         .quantizer = &quantizer,
         .y_to_x = {y_to_x_readback_.data(), tile_extent_, tile_extent_.width},
         .y_to_b = {y_to_b_readback_.data(), tile_extent_, tile_extent_.width},
         .epf_sharpness = {epf_sharpness_.data(), block_extent_,
                           block_extent_.width},
         .profile = options_.profile,
         .quantized_dc = quantized_dc,
         .quantized_ac = {quantized_readback_.get(), coefficient_count_},
         .transforms = layouts_,
         .reject_unwritten_coefficients = true},
        frame);
  }

  void InitializeKernelParams() {
    gaborish_params_.width = static_cast<uint32_t>(source_extent_.width);
    gaborish_params_.height = static_cast<uint32_t>(source_extent_.height);
    gaborish_params_.input_stride = static_cast<uint32_t>(coding_extent_.width);
    gaborish_params_.output_stride =
        static_cast<uint32_t>(coding_extent_.width);
    for (size_t channel = 0; channel < 3; ++channel) {
      const float weight1 =
          options_.profile.loop_filter.gaborish_options.weight1[channel];
      const float weight2 =
          options_.profile.loop_filter.gaborish_options.weight2[channel];
      const float divisor = 1.0f + 4.0f * (weight1 + weight2);
      gaborish_params_.center_weight[channel] = 1.0f / divisor;
      gaborish_params_.axis_weight[channel] = weight1 / divisor;
      gaborish_params_.diagonal_weight[channel] = weight2 / divisor;
    }
    for (uint32_t pass = 0; pass < 3; ++pass) {
      const float pass_scale =
          pass == 0 ? options_.profile.loop_filter.epf_options.pass0_sigma_scale
          : pass == 2
              ? options_.profile.loop_filter.epf_options.pass2_sigma_scale
              : 1.0f;
      CudaAqEpfParams& params = epf_params_[pass];
      params.width = static_cast<uint32_t>(source_extent_.width);
      params.height = static_cast<uint32_t>(source_extent_.height);
      params.input_stride = static_cast<uint32_t>(coding_extent_.width);
      params.output_stride = static_cast<uint32_t>(coding_extent_.width);
      params.inverse_sigma_stride = static_cast<uint32_t>(block_extent_.width);
      params.pass = pass;
      params.sigma_scale = 1.65f * pass_scale;
      params.border_sad_multiplier =
          options_.profile.loop_filter.epf_options.border_sad_multiplier;
      for (size_t channel = 0; channel < 3; ++channel) {
        params.channel_scale[channel] =
            options_.profile.loop_filter.epf_options.channel_scale[channel];
      }
    }
    color_params_ = {static_cast<uint32_t>(source_extent_.width),
                     static_cast<uint32_t>(source_extent_.height),
                     static_cast<uint32_t>(coding_extent_.width),
                     static_cast<uint32_t>(source_extent_.width),
                     255.0f / options_.profile.intensity_target};
  }

  CudaAqResidentParams ResidentParams(size_t batch_index) const {
    CudaAqResidentParams params;
    params.coding_stride = static_cast<uint32_t>(coding_extent_.width);
    params.block_width = static_cast<uint32_t>(block_extent_.width);
    params.block_height = static_cast<uint32_t>(block_extent_.height);
    params.color_stride = static_cast<uint32_t>(tile_extent_.width);
    params.strategy = static_cast<uint32_t>(kSupportedStrategies[batch_index]);
    params.x_matrix_multiplier =
        QuantizationMatrixMultiplier(options_.profile.x_qm_scale);
    params.b_matrix_multiplier =
        QuantizationMatrixMultiplier(options_.profile.b_qm_scale);
    params.adjust_ac_quant = 1;
    params.epf_quant_multiplier = options_.profile.epf_sigma.quant_multiplier;
    for (size_t index = 0; index < 8; ++index) {
      params.epf_sharpness_lut[index] =
          options_.profile.epf_sigma.sharpness_lut[index];
    }
    return params;
  }

  static cudaError_t EncodeAdjustment(CudaBackend& backend,
                                      const void* opaque) {
    const auto& context = *static_cast<const AdjustmentContext*>(opaque);
    CudaPreparedResidentAqEvaluation& self = *context.self;
    cudaError_t status =
        cudaMemsetAsync(Pointer<unsigned int>(self.error_device_), 0,
                        sizeof(uint32_t), backend.state_->stream);
    if (status != cudaSuccess) return status;
    for (const CudaAqExactBatch& batch : self.batches_) {
      if (batch.anchor_count == 0) continue;
      status = LaunchCudaAqAdjustQuantField(
          Pointer<CudaAqAnchor>(self.anchors_device_),
          Pointer<float>(self.quant_field_device_),
          Pointer<unsigned int>(self.error_device_),
          static_cast<uint32_t>(self.block_extent_.width), batch,
          context.mean_max_mixer, backend.state_->stream);
      if (status != cudaSuccess) return status;
    }
    if (context.prepare_encoding_policy) {
      status = LaunchCudaAqPositiveRange(
          Pointer<const float>(self.quant_field_device_),
          static_cast<uint32_t>(self.block_count_),
          Pointer<float>(self.statistics_device_),
          Pointer<unsigned int>(self.error_device_), backend.state_->stream);
      if (status != cudaSuccess) return status;
      status = cudaMemcpyAsync(
          Pointer<float>(self.block_device_),
          Pointer<const float>(self.quant_field_device_),
          self.block_count_ * sizeof(float), cudaMemcpyDeviceToDevice,
          backend.state_->stream);
      if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
  }

  static cudaError_t EncodeInitialQuantization(CudaBackend& backend,
                                               const void* opaque) {
    const auto& context = *static_cast<const InitialContext*>(opaque);
    CudaPreparedResidentAqEvaluation& self = *context.self;
    const CudaAqGeometry geometry{
        static_cast<uint32_t>(self.coding_extent_.width),
        static_cast<uint32_t>(self.coding_extent_.height),
        static_cast<uint32_t>(self.block_extent_.width),
        static_cast<uint32_t>(self.block_extent_.height),
        static_cast<uint32_t>(self.tile_extent_.width),
        static_cast<uint32_t>(self.tile_extent_.height)};
    cudaError_t status = LaunchCudaAqInitialField(self.CodingPointer(0),
      self.CodingPointer(1),
      self.CodingPointer(2),
      Pointer<float>(self.initial_unblurred_pixel_device_),
      Pointer<float>(self.initial_pixel_device_),
      Pointer<float>(self.initial_pre_erosion_device_),
      Pointer<float>(self.quant_field_device_),
      Pointer<float>(self.initial_strategy_device_),
      Pointer<unsigned int>(self.error_device_),
      geometry,
      context.options.butteraugli_target,
      context.options.rescale,
      backend.state_->stream);
    if (status != cudaSuccess) return status;

    std::array<const float*, 3> cfl_source = self.CodingPointers();
    if (self.options_.profile.loop_filter.gaborish) {
      for (size_t channel = 0; channel < 3; ++channel) {
        const Symmetric5Weights weights =
            gaborish_internal::GaborishInverseWeights(
                self.options_.profile.gaborish_inverse_multipliers[channel]);
        status = LaunchCudaSymmetric5Convolution(self.CodingPointer(channel),
          Pointer<float>(self.reconstructed_[channel]),
          geometry.width,
          geometry.height,
          geometry.width,
          geometry.width,
          weights.distance0,
          weights.distance1,
          weights.distance2,
          weights.distance4,
          weights.distance8,
          weights.distance5,
          backend.state_->stream);
        if (status != cudaSuccess) return status;
      }
      cfl_source = ConstPointers(self.reconstructed_);
    }
    return LaunchCudaAqInitialCfl(cfl_source[0], cfl_source[1], cfl_source[2],
                                  Pointer<signed char>(self.y_to_x_device_),
                                  Pointer<signed char>(self.y_to_b_device_),
                                  Pointer<unsigned int>(self.error_device_),
                                  geometry, backend.state_->stream);
  }

  static cudaError_t EncodeResidentQuantizer(
      CudaBackend& backend, CudaPreparedResidentAqEvaluation& self,
      const DevicePlaneView& quant_field, float quant_dc) {
    return LaunchCudaAqSelectResidentQuantizer(
        Pointer<const float>(quant_field),
        static_cast<uint32_t>(self.block_count_),
        Pointer<unsigned int>(self.selection_state_device_),
        Pointer<unsigned int>(self.histogram_device_),
        Pointer<float>(self.statistics_device_),
        Pointer<unsigned int>(self.quantizer_device_),
        Pointer<int>(self.raw_quant_device_),
        Pointer<unsigned int>(self.error_device_), quant_dc,
        backend.state_->stream);
  }

  static cudaError_t EncodeForwardCoefficients(
      CudaBackend& backend, CudaPreparedResidentAqEvaluation& self) {
    std::array<const float*, 3> coding_source = self.CodingPointers();
    if (self.resident_frontend_ && self.options_.profile.loop_filter.gaborish) {
      coding_source = ConstPointers(self.reconstructed_);
    }
    for (const CudaAqExactBatch& batch : self.batches_) {
      if (batch.anchor_count == 0) continue;
      cudaError_t status = LaunchCudaAqGatherTransformPixels(coding_source[0],
        coding_source[1],
        coding_source[2],
        Pointer<CudaAqAnchor>(self.anchors_device_),
        Pointer<float>(self.gathered_device_),
        batch,
        static_cast<uint32_t>(self.coding_extent_.width),
        backend.state_->stream);
      if (status != cudaSuccess) return status;
      status = LaunchCudaDct(
          true,
          Pointer<const float>(self.gathered_device_) +
              batch.coefficient_offset,
          Pointer<float>(self.forward_device_) + batch.coefficient_offset,
          3 * static_cast<size_t>(batch.anchor_count), batch.pixel_width,
          batch.pixel_height, backend.state_->stream);
      if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
  }

  static cudaError_t EncodeFinalColorCorrelation(
      CudaBackend& backend, CudaPreparedResidentAqEvaluation& self) {
    return LaunchCudaAqFinalColorCorrelation(
        Pointer<CudaAqColorTransformRecord>(self.color_transforms_device_),
        Pointer<uint32_t>(self.color_tile_offsets_device_),
        Pointer<const float>(self.quant_tables_device_),
        Pointer<const float>(self.forward_device_),
        Pointer<const int>(self.raw_quant_device_),
        Pointer<const unsigned int>(self.quantizer_device_),
        Pointer<signed char>(self.y_to_x_device_),
        Pointer<signed char>(self.y_to_b_device_),
        Pointer<unsigned int>(self.error_device_),
        static_cast<uint32_t>(self.tile_count_), backend.state_->stream);
  }

  static cudaError_t EncodeReconstruction(CudaBackend& backend,
                                          const void* opaque) {
    const auto& context = *static_cast<const EvaluationContext*>(opaque);
    CudaPreparedResidentAqEvaluation& self = *context.self;
    cudaError_t status = cudaSuccess;
    if (context.reset_error) {
      status = cudaMemsetAsync(Pointer<unsigned int>(self.error_device_), 0,
                               sizeof(uint32_t), backend.state_->stream);
      if (status != cudaSuccess) return status;
    }
    if (context.compute_forward) {
      status = EncodeForwardCoefficients(backend, self);
      if (status != cudaSuccess) return status;
    }
    if (context.compute_color_correlation) {
      // block_device_ still contains the field retained by preparation. The
      // reconstruction below is the first operation that may overwrite it.
      status = EncodeResidentQuantizer(backend, self, self.block_device_,
                                       self.invariant_quant_dc_);
      if (status != cudaSuccess) return status;
      status = EncodeFinalColorCorrelation(backend, self);
      if (status != cudaSuccess) return status;
    }
    status = EncodeResidentQuantizer(backend, self, self.quant_field_device_,
                                     context.quant_dc);
    if (status != cudaSuccess) return status;

    for (size_t batch_index = 0; batch_index < self.batches_.size();
         ++batch_index) {
      const CudaAqExactBatch& batch = self.batches_[batch_index];
      if (batch.anchor_count == 0) continue;
      const CudaAqResidentParams params = self.ResidentParams(batch_index);
      status = LaunchCudaAqSelectAdjustedQuantization(
          Pointer<CudaAqAnchor>(self.anchors_device_),
          Pointer<const float>(self.quant_tables_device_),
          Pointer<int>(self.raw_quant_device_),
          Pointer<const float>(self.forward_device_),
          Pointer<float>(self.thresholds_device_),
          Pointer<const unsigned int>(self.quantizer_device_),
          Pointer<unsigned int>(self.error_device_), batch, params,
          backend.state_->stream);
      if (status != cudaSuccess) return status;
      status = LaunchCudaAqEncodeResidentCoefficients(
          Pointer<CudaAqAnchor>(self.anchors_device_),
          Pointer<const float>(self.quant_tables_device_),
          Pointer<const int>(self.raw_quant_device_),
          Pointer<const signed char>(self.y_to_x_device_),
          Pointer<const signed char>(self.y_to_b_device_),
          Pointer<const float>(self.forward_device_),
          Pointer<int>(self.quantized_device_),
          Pointer<float>(self.reconstruction_coefficients_device_),
          Pointer<float>(self.dc_device_),
          Pointer<int>(self.quantized_dc_device_),
          Pointer<float>(self.inverse_sigma_device_),
          Pointer<const unsigned char>(self.epf_sharpness_device_),
          Pointer<const unsigned int>(self.quantizer_device_),
          Pointer<const float>(self.thresholds_device_),
          Pointer<unsigned int>(self.error_device_), batch, params,
          backend.state_->stream);
      if (status != cudaSuccess) return status;
      status = LaunchCudaDct(
          false,
          Pointer<const float>(self.reconstruction_coefficients_device_) +
              batch.coefficient_offset,
          Pointer<float>(self.inverse_device_) + batch.coefficient_offset,
          3 * static_cast<size_t>(batch.anchor_count), batch.pixel_width,
          batch.pixel_height, backend.state_->stream);
      if (status != cudaSuccess) return status;
      status = LaunchCudaAqScatterReconstruction(
          Pointer<CudaAqAnchor>(self.anchors_device_),
          Pointer<const float>(self.inverse_device_),
          MutablePointers(self.reconstructed_),
          static_cast<uint32_t>(self.coding_extent_.width), batch,
          backend.state_->stream);
      if (status != cudaSuccess) return status;
    }
    status = self.EncodePostprocess(backend);
    if (status != cudaSuccess) return status;
    if (self.options_.metric == AqEvaluationMetric::kMaximumError) {
      const std::array<DevicePlaneView, 3> filtered = self.FinalFilteredImage();
      for (const CudaAqExactBatch& batch : self.batches_) {
        if (batch.anchor_count == 0) continue;
        status = LaunchCudaAqReduceMaximumError(self.CodingPointers(),
          ConstPointers(filtered),
          static_cast<uint32_t>(self.coding_extent_.width),
          static_cast<uint32_t>(self.coding_extent_.width),
          Pointer<CudaAqAnchor>(self.anchors_device_),
          Pointer<float>(self.block_device_),
          static_cast<uint32_t>(self.block_extent_.width),
          Pointer<float>(self.maximum_device_),
          Pointer<unsigned int>(self.error_device_),
          static_cast<uint32_t>(self.source_extent_.width),
          static_cast<uint32_t>(self.source_extent_.height),
          self.options_.maximum_error,
          batch,
          backend.state_->stream);
        if (status != cudaSuccess) return status;
      }
    }
    return cudaSuccess;
  }

  cudaError_t EncodePostprocess(CudaBackend& backend) {
    std::array<DevicePlaneView, 3> current = reconstructed_;
    size_t stage = 0;
    cudaError_t status = cudaSuccess;
    if (options_.profile.loop_filter.gaborish) {
      status = LaunchCudaAqGaborish(ConstPointers(current),
                                    MutablePointers(filter_scratch_[0]),
                                    Pointer<unsigned int>(error_device_),
                                    gaborish_params_, backend.state_->stream);
      if (status != cudaSuccess) return status;
      current = filter_scratch_[0];
      ++stage;
    }
    const uint32_t iterations =
        options_.profile.loop_filter.epf_options.iterations;
    const uint32_t first_pass = iterations == 3 ? 0 : 1;
    for (uint32_t pass = first_pass; pass < first_pass + iterations; ++pass) {
      std::array<DevicePlaneView, 3>& destination = filter_scratch_[stage % 2];
      status = LaunchCudaAqEpf(
          ConstPointers(current), Pointer<const float>(inverse_sigma_device_),
          MutablePointers(destination), Pointer<unsigned int>(error_device_),
          epf_params_[pass], backend.state_->stream);
      if (status != cudaSuccess) return status;
      current = destination;
      ++stage;
    }
    return LaunchCudaAqOpsinToLinear(ConstPointers(current),
                                     MutablePointers(reconstructed_linear_),
                                     Pointer<unsigned int>(error_device_),
                                     color_params_, backend.state_->stream);
  }

  static cudaError_t EncodeResidentPolicy(CudaBackend& backend,
                                          const void* opaque) {
    const auto& context = *static_cast<const PolicyContext*>(opaque);
    CudaPreparedResidentAqEvaluation& self = *context.self;
    for (size_t iteration = 0; iteration < context.score_count; ++iteration) {
      EvaluationContext evaluation{
          &self, context.input.quant_dc,
          iteration == 0 && context.compute_forward,
          iteration == 0 && context.compute_color_correlation, iteration == 0};
      cudaError_t status = EncodeReconstruction(backend, &evaluation);
      if (status != cudaSuccess) return status;

      CudaAqResidentPolicyParams policy_params{
          static_cast<uint32_t>(self.block_count_),
          static_cast<uint32_t>(context.score_count),
          static_cast<uint32_t>(iteration),
          static_cast<uint32_t>(iteration),
          static_cast<uint32_t>(iteration < context.input.iterations),
          context.input.butteraugli_target,
          context.input.lower_bound,
          context.input.upper_bound};
      if (iteration == 0) {
        status = LaunchCudaAqResidentPolicyInitialize(
            Pointer<const float>(self.quant_field_device_),
            Pointer<float>(self.policy_initial_field_device_),
            Pointer<float>(self.policy_scores_device_),
            Pointer<unsigned int>(self.error_device_), policy_params,
            backend.state_->stream);
        if (status != cudaSuccess) return status;
      }

      status = EncodePreparedCudaButteraugli(
          *self.butteraugli_,
          {.distorted_linear_rgb = ConstImage(self.reconstructed_linear_),
           .distance_map = self.distance_device_,
           .score = self.score_device_},
          backend.state_->stream);
      if (status != cudaSuccess) return status;
      status = EncodeBlockReductionOnStream(backend, self, false);
      if (status != cudaSuccess) return status;
      status = LaunchCudaAqResidentPolicyUpdate(
          Pointer<float>(self.quant_field_device_),
          Pointer<const float>(self.policy_initial_field_device_),
          Pointer<const float>(self.block_device_),
          Pointer<const float>(self.score_device_),
          Pointer<float>(self.policy_scores_device_),
          Pointer<const unsigned int>(self.quantizer_device_),
          Pointer<unsigned int>(self.error_device_), policy_params,
          backend.state_->stream);
      if (status != cudaSuccess) return status;
    }

    if (!context.input.evaluate_final_field) {
      cudaError_t status = EncodeResidentQuantizer(
          backend, self, self.quant_field_device_, context.input.quant_dc);
      if (status != cudaSuccess) return status;
      for (size_t batch_index = 0; batch_index < self.batches_.size();
           ++batch_index) {
        const CudaAqExactBatch& batch = self.batches_[batch_index];
        if (batch.anchor_count == 0) continue;
        const CudaAqResidentParams params = self.ResidentParams(batch_index);
        status = LaunchCudaAqSelectAdjustedQuantization(
            Pointer<CudaAqAnchor>(self.anchors_device_),
            Pointer<const float>(self.quant_tables_device_),
            Pointer<int>(self.raw_quant_device_),
            Pointer<const float>(self.forward_device_),
            Pointer<float>(self.thresholds_device_),
            Pointer<const unsigned int>(self.quantizer_device_),
            Pointer<unsigned int>(self.error_device_), batch, params,
            backend.state_->stream);
        if (status != cudaSuccess) return status;
        status = LaunchCudaAqEncodeResidentCoefficients(
            Pointer<CudaAqAnchor>(self.anchors_device_),
            Pointer<const float>(self.quant_tables_device_),
            Pointer<const int>(self.raw_quant_device_),
            Pointer<const signed char>(self.y_to_x_device_),
            Pointer<const signed char>(self.y_to_b_device_),
            Pointer<const float>(self.forward_device_),
            Pointer<int>(self.quantized_device_),
            Pointer<float>(self.reconstruction_coefficients_device_),
            Pointer<float>(self.dc_device_),
            Pointer<int>(self.quantized_dc_device_),
            Pointer<float>(self.inverse_sigma_device_),
            Pointer<const unsigned char>(self.epf_sharpness_device_),
            Pointer<const unsigned int>(self.quantizer_device_),
            Pointer<const float>(self.thresholds_device_),
            Pointer<unsigned int>(self.error_device_), batch, params,
            backend.state_->stream);
        if (status != cudaSuccess) return status;
      }
    }
    return cudaSuccess;
  }

  static cudaError_t EncodeBlockReduction(CudaBackend& backend,
                                          const void* opaque) {
    auto& self = *static_cast<CudaPreparedResidentAqEvaluation*>(
        const_cast<void*>(opaque));
    return EncodeBlockReductionOnStream(backend, self, true);
  }

  static cudaError_t EncodeBlockReductionOnStream(
      CudaBackend& backend, CudaPreparedResidentAqEvaluation& self,
      bool reset_error) {
    cudaError_t status = cudaSuccess;
    if (reset_error) {
      status = cudaMemsetAsync(Pointer<unsigned int>(self.error_device_), 0,
                               sizeof(uint32_t), backend.state_->stream);
      if (status != cudaSuccess) return status;
    }
    for (const CudaAqExactBatch& batch : self.batches_) {
      if (batch.anchor_count == 0) continue;
      status = LaunchCudaAqReduceButteraugli(
          Pointer<const float>(self.distance_device_),
          static_cast<uint32_t>(self.source_extent_.width),
          Pointer<CudaAqAnchor>(self.anchors_device_),
          Pointer<float>(self.block_device_),
          static_cast<uint32_t>(self.block_extent_.width),
          Pointer<unsigned int>(self.error_device_),
          static_cast<uint32_t>(self.source_extent_.width),
          static_cast<uint32_t>(self.source_extent_.height), batch,
          backend.state_->stream);
      if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
  }

  Status ReadAndCheckDeviceError() {
    uint32_t error = 0;
    Status status =
        backend_->CopyDeviceToHost(*error_device_.buffer, &error, sizeof(error),
                                   error_device_.offset_bytes);
    if (!status.ok()) return status;
    return error == 0 ? Status::Ok()
                      : Status::DeviceError(
                            "CUDA resident AQ detected invalid device numerics "
                            "(flag " +
                            std::to_string(error) + ")");
  }

  Status Invalidate(Status status) {
    invalid_ = true;
    return status;
  }

  std::array<DevicePlaneView, 3> FinalFilteredImage() const noexcept {
    return final_filter_index_ < 0
               ? reconstructed_
               : filter_scratch_[static_cast<size_t>(final_filter_index_)];
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

  static std::array<float*, 3> MutablePointers(
      const std::array<DevicePlaneView, 3>& image) {
    return {Pointer<float>(image[0]), Pointer<float>(image[1]),
            Pointer<float>(image[2])};
  }

  static std::array<const float*, 3> ConstPointers(
      const std::array<DevicePlaneView, 3>& image) {
    return {Pointer<const float>(image[0]), Pointer<const float>(image[1]),
            Pointer<const float>(image[2])};
  }

  static ConstDeviceImage3View ConstImage(
      const std::array<DevicePlaneView, 3>& image) {
    return {{{static_cast<ConstDevicePlaneView>(image[0]),
              static_cast<ConstDevicePlaneView>(image[1]),
              static_cast<ConstDevicePlaneView>(image[2])}}};
  }

  bool has_borrowed_input() const noexcept {
    return borrowed_coding_.plane[0].buffer != nullptr;
  }

  const float* CodingPointer(
    size_t channel) const {
    return has_borrowed_input()
             ? Pointer<const float>(borrowed_coding_.plane[channel])
             : Pointer<const float>(coding_[channel]);
  }

  std::array<const float*, 3> CodingPointers() const {
    return {CodingPointer(0), CodingPointer(1), CodingPointer(2)};
  }

  ConstDeviceImage3View OriginalImage() const {
    return has_borrowed_input() ? borrowed_original_ : ConstImage(original_);
  }

  CudaBackend* backend_ = nullptr;
  DeviceScratchArena persistent_;
  DeviceScratchArena staging_;
  std::array<DevicePlaneView, 3> original_{};
  std::array<DevicePlaneView, 3> coding_{};
  ConstDeviceImage3View borrowed_original_{};
  ConstDeviceImage3View borrowed_coding_{};
  std::array<DevicePlaneView, 3> reconstructed_{};
  std::array<std::array<DevicePlaneView, 3>, 2> filter_scratch_{};
  std::array<DevicePlaneView, 3> reconstructed_linear_{};
  DevicePlaneView anchors_device_{};
  DevicePlaneView epf_sharpness_device_{};
  DevicePlaneView quant_tables_device_{};
  DevicePlaneView color_transforms_device_{};
  DevicePlaneView color_tile_offsets_device_{};
  DevicePlaneView y_to_x_device_{};
  DevicePlaneView y_to_b_device_{};
  DevicePlaneView quant_field_device_{};
  DevicePlaneView initial_unblurred_pixel_device_{};
  DevicePlaneView initial_pixel_device_{};
  DevicePlaneView initial_pre_erosion_device_{};
  DevicePlaneView initial_strategy_device_{};
  DevicePlaneView policy_initial_field_device_{};
  DevicePlaneView policy_scores_device_{};
  DevicePlaneView selection_state_device_{};
  DevicePlaneView histogram_device_{};
  DevicePlaneView statistics_device_{};
  DevicePlaneView quantizer_device_{};
  DevicePlaneView raw_quant_device_{};
  DevicePlaneView gathered_device_{};
  DevicePlaneView forward_device_{};
  DevicePlaneView thresholds_device_{};
  DevicePlaneView quantized_device_{};
  DevicePlaneView reconstruction_coefficients_device_{};
  DevicePlaneView dc_device_{};
  DevicePlaneView quantized_dc_device_{};
  DevicePlaneView inverse_device_{};
  DevicePlaneView inverse_sigma_device_{};
  DevicePlaneView distance_device_{};
  DevicePlaneView score_device_{};
  DevicePlaneView block_device_{};
  DevicePlaneView maximum_device_{};
  DevicePlaneView error_device_{};
  Extent2D source_extent_{};
  Extent2D coding_extent_{};
  Extent2D block_extent_{};
  Extent2D tile_extent_{};
  size_t source_count_ = 0;
  size_t coding_count_ = 0;
  size_t block_count_ = 0;
  size_t tile_count_ = 0;
  size_t coefficient_count_ = 0;
  size_t anchor_count_ = 0;
  size_t filter_stage_count_ = 0;
  size_t filter_scratch_count_ = 0;
  int final_filter_index_ = -1;
  AqEvaluationOptions options_{};
  AcStrategyGrid strategies_{};
  std::array<CudaAqExactBatch, 7> batches_{};
  std::vector<HostAnchor> row_major_anchors_;
  std::vector<uint8_t> epf_sharpness_;
  std::vector<vardct_frame_internal::QuantizedAcTransformLayout> layouts_;
  std::vector<float> block_readback_;
  std::vector<float> maximum_readback_;
  std::vector<int32_t> raw_readback_;
  std::unique_ptr<int32_t[]> quantized_readback_;
  std::vector<int32_t> quantized_dc_readback_;
  std::vector<int8_t> y_to_x_readback_;
  std::vector<int8_t> y_to_b_readback_;
  std::vector<float> quant_field_readback_;
  std::vector<float> initial_strategy_readback_;
  std::vector<float> initial_pixel_readback_;
  std::array<float, 5> policy_score_readback_{};
  std::array<std::vector<float>, 3> linear_readback_;
  std::unique_ptr<PreparedDeviceButteraugli> butteraugli_;
  CudaAqGaborishParams gaborish_params_{};
  std::array<CudaAqEpfParams, 3> epf_params_{};
  CudaAqColorParams color_params_{};
  AqEvaluationMemoryStats memory_stats_{};
  std::mutex mutex_;
  bool invariant_color_correlation_ready_ = false;
  bool forward_coefficients_ready_ = false;
  bool color_correlation_pending_ = false;
  bool resident_frontend_ = false;
  bool resident_initial_ready_ = false;
  bool resident_encoding_policy_ready_ = false;
  aq_evaluation_internal::ResidentEncodingPolicySetup resident_policy_setup_{};
  float invariant_quant_dc_ = 0.0f;
  bool invalid_ = false;
};

Status GetCudaResidentReconstructionStagingBytesForTest(
    const PreparedAqEvaluation& prepared, size_t* bytes) {
  const auto* resident =
      dynamic_cast<const CudaPreparedResidentAqEvaluation*>(&prepared);
  if (resident == nullptr || bytes == nullptr) {
    return Status::InvalidArgument(
        "CUDA resident reconstruction staging query is invalid");
  }
  *bytes = resident->reconstruction_staging_bytes_for_test();
  return Status::Ok();
}

Status PoisonCudaResidentCoefficientReadbackForTest(
    PreparedAqEvaluation& prepared, int32_t value) {
  auto* resident = dynamic_cast<CudaPreparedResidentAqEvaluation*>(&prepared);
  if (resident == nullptr) {
    return Status::InvalidArgument(
        "CUDA resident coefficient readback poison target is invalid");
  }
  resident->PoisonCoefficientReadbackForTest(value);
  return Status::Ok();
}

Status PrepareCudaResidentAqEvaluation(
    CudaBackend& backend, const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared) {
  if (prepared == nullptr) {
    return Status::InvalidArgument(
        "CUDA resident AQ prepared output pointer is null");
  }
  prepared->reset();
  try {
    auto candidate =
        std::make_unique<CudaPreparedResidentAqEvaluation>(backend);
    Status status = candidate->Prepare(preparation);
    if (!status.ok()) return status;
    *prepared = std::move(candidate);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
        "Unable to allocate CUDA resident AQ prepared state");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
        "CUDA resident AQ prepared dimensions are too large");
  }
}

}  // namespace gjxl::cuda_internal
