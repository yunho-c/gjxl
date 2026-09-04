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
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/dc_conversion.h"
#include "codec/quantization.h"
#include "codec/vardct_frame.h"
#include "core/ac_strategy.h"
#include "core/block_grid.h"
#include "core/image_ops.h"
#include "core/quantizer.h"
#include "gpu/cuda/cuda_aq_exact_kernels.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_kernels.h"
#include "gpu/ops/butteraugli.h"
#include "gpu/scratch.h"
#include "gpu/submission.h"

namespace gjxl::cuda_internal {
namespace {

constexpr size_t kArenaAlignment = 256;
constexpr std::array<AcStrategyType, 7> kSupportedStrategies = {
    AcStrategyType::kDct8,     AcStrategyType::kDct16x16,
    AcStrategyType::kDct32x32, AcStrategyType::kDct16x8,
    AcStrategyType::kDct8x16,  AcStrategyType::kDct32x16,
    AcStrategyType::kDct16x32,
};

static_assert(std::is_standard_layout_v<CudaAqAnchor>);
static_assert(std::is_trivially_copyable_v<CudaAqAnchor>);
static_assert(std::is_standard_layout_v<CudaAqExactBatch>);
static_assert(std::is_trivially_copyable_v<CudaAqExactBatch>);
static_assert(std::is_standard_layout_v<CudaAqGaborishParams>);
static_assert(std::is_trivially_copyable_v<CudaAqGaborishParams>);
static_assert(std::is_standard_layout_v<CudaAqEpfParams>);
static_assert(std::is_trivially_copyable_v<CudaAqEpfParams>);
static_assert(std::is_standard_layout_v<CudaAqColorParams>);
static_assert(std::is_trivially_copyable_v<CudaAqColorParams>);

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

template <typename T>
bool PlaneDescriptorSpecified(PlaneView<T> plane) noexcept {
  return plane.data != nullptr || !plane.extent.empty() || plane.stride != 0;
}

template <typename T>
bool ImageDescriptorSpecified(Image3View<T> image) noexcept {
  return std::ranges::any_of(image.plane, [](PlaneView<T> plane) {
    return PlaneDescriptorSpecified(plane);
  });
}

bool FinitePositive(float value) noexcept {
  return std::isfinite(value) && value > 0.0f;
}

bool SupportedStrategy(AcStrategyType strategy) noexcept {
  return std::ranges::find(kSupportedStrategies, strategy) !=
         kSupportedStrategies.end();
}

size_t StrategyBatchIndex(AcStrategyType strategy) noexcept {
  const auto found = std::ranges::find(kSupportedStrategies, strategy);
  return found == kSupportedStrategies.end()
             ? kSupportedStrategies.size()
             : static_cast<size_t>(found - kSupportedStrategies.begin());
}

Status ValidateFiniteImage(ConstImage3FView image, std::string_view name) {
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
    return Status::InvalidArgument("CUDA exact AQ profile is invalid");
  }
  switch (options.metric) {
    case AqEvaluationMetric::kButteraugli:
      if (!FinitePositive(options.butteraugli.hf_asymmetry) ||
          !FinitePositive(options.butteraugli.x_multiplier) ||
          !FinitePositive(options.butteraugli.intensity_target)) {
        return Status::InvalidArgument(
            "CUDA exact AQ Butteraugli options are invalid");
      }
      break;
    case AqEvaluationMetric::kMaximumError:
      if (!std::ranges::all_of(options.maximum_error, FinitePositive)) {
        return Status::InvalidArgument(
            "CUDA exact AQ maximum-error limits are invalid");
      }
      break;
    default:
      return Status::InvalidArgument("CUDA exact AQ metric is invalid");
  }
  if (options.profile.loop_filter.gaborish) {
    for (size_t channel = 0; channel < 3; ++channel) {
      const float weight1 =
          options.profile.loop_filter.gaborish_options.weight1[channel];
      const float weight2 =
          options.profile.loop_filter.gaborish_options.weight2[channel];
      const float divisor = 1.0f + 4.0f * (weight1 + weight2);
      if (!std::isfinite(weight1) || !std::isfinite(weight2) ||
          !std::isfinite(divisor) || std::abs(divisor) < 1.0e-8f) {
        return Status::InvalidArgument(
            "CUDA exact AQ Gaborish options are invalid");
      }
    }
  }
  const EpfFilterOptions& epf = options.profile.loop_filter.epf_options;
  if (epf.iterations > 3 || !FinitePositive(epf.pass0_sigma_scale) ||
      !FinitePositive(epf.pass2_sigma_scale) ||
      !FinitePositive(epf.border_sad_multiplier) ||
      !std::ranges::all_of(epf.channel_scale, [](float value) {
        return std::isfinite(value) && value >= 0.0f;
      })) {
    return Status::InvalidArgument("CUDA exact AQ EPF options are invalid");
  }
  return Status::Ok();
}

Status PlanPlane(DeviceElementType type, Extent2D extent, size_t row_stride,
                 size_t* bytes) {
  if (bytes == nullptr || extent.empty() || row_stride < extent.width) {
    return Status::InvalidArgument("CUDA exact AQ arena plan is invalid");
  }
  if (*bytes > std::numeric_limits<size_t>::max() - (kArenaAlignment - 1)) {
    return Status::InvalidArgument("CUDA exact AQ arena alignment overflows");
  }
  const size_t aligned =
      (*bytes + kArenaAlignment - 1) & ~(kArenaAlignment - 1);
  if (extent.height - 1 >
      (std::numeric_limits<size_t>::max() - extent.width) / row_stride) {
    return Status::InvalidArgument("CUDA exact AQ plane geometry overflows");
  }
  const size_t elements = (extent.height - 1) * row_stride + extent.width;
  const size_t element_size = DeviceElementSize(type);
  if (element_size == 0 ||
      elements > std::numeric_limits<size_t>::max() / element_size ||
      aligned > std::numeric_limits<size_t>::max() - elements * element_size) {
    return Status::InvalidArgument("CUDA exact AQ plane size overflows");
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

}  // namespace

class CudaPreparedExactAqEvaluation final : public PreparedAqEvaluation {
 public:
  explicit CudaPreparedExactAqEvaluation(CudaBackend& backend)
      : backend_(&backend) {}

  Status Prepare(const AqEvaluationPreparation& preparation) {
    if (preparation.frame_only || preparation.frame_only_inverse_gaborish ||
        preparation.resident_initial_cfl ||
        preparation.frame_only_resident_initial_quant ||
        preparation.resident_ac_strategy_inputs ||
        preparation.frame_only_resident_quantizer ||
        preparation.resident_quantization ||
        preparation.resident_coding_opsin.plane[0].buffer != nullptr ||
        preparation.resident_coding_opsin.plane[1].buffer != nullptr ||
        preparation.resident_coding_opsin.plane[2].buffer != nullptr) {
      return Status::Unavailable(
          "CUDA exact AQ does not support resident preparation inputs");
    }
    if (preparation.coefficient_decision_mode !=
            AcCoefficientDecisionMode::kAdjustedSharedQuant &&
        preparation.coefficient_decision_mode !=
            AcCoefficientDecisionMode::kFixedRawQuant) {
      return Status::InvalidArgument(
          "CUDA exact AQ coefficient decision mode is invalid");
    }
    Status status = ValidateOptions(preparation.options);
    if (!status.ok()) return status;
    status = ValidateFiniteImage(preparation.original_linear_rgb,
                                 "CUDA exact AQ original");
    if (!status.ok()) return status;
    status =
        ValidateFiniteImage(preparation.coding_opsin, "CUDA exact AQ coding");
    if (!status.ok()) return status;
    source_extent_ = preparation.original_linear_rgb.extent();
    coding_extent_ = preparation.coding_opsin.extent();
    if (source_extent_.empty() || coding_extent_.empty() ||
        coding_extent_.width % kJxlBlockDimension != 0 ||
        coding_extent_.height % kJxlBlockDimension != 0 ||
        source_extent_.width > coding_extent_.width ||
        source_extent_.height > coding_extent_.height ||
        coding_extent_.width - source_extent_.width >= kJxlBlockDimension ||
        coding_extent_.height - source_extent_.height >= kJxlBlockDimension) {
      return Status::InvalidArgument(
          "CUDA exact AQ source and coding geometry are incompatible");
    }
    block_extent_ = {coding_extent_.width / kJxlBlockDimension,
                     coding_extent_.height / kJxlBlockDimension};
    tile_extent_ = {
        (coding_extent_.width + kColorTileDimension - 1) / kColorTileDimension,
        (coding_extent_.height + kColorTileDimension - 1) /
            kColorTileDimension};
    if (preparation.strategies == nullptr ||
        !preparation.strategies->complete() ||
        preparation.strategies->extent() != block_extent_ ||
        !ValidHostPlaneLayout(preparation.epf_sharpness) ||
        preparation.epf_sharpness.extent != block_extent_) {
      return Status::InvalidArgument(
          "CUDA exact AQ strategy or EPF grid is invalid");
    }
    if (source_extent_.width > std::numeric_limits<uint32_t>::max() ||
        source_extent_.height > std::numeric_limits<uint32_t>::max() ||
        coding_extent_.width > std::numeric_limits<uint32_t>::max() ||
        coding_extent_.height > std::numeric_limits<uint32_t>::max() ||
        block_extent_.width > std::numeric_limits<uint32_t>::max() ||
        block_extent_.height > std::numeric_limits<uint32_t>::max()) {
      return Status::InvalidArgument(
          "CUDA exact AQ geometry exceeds kernel limits");
    }
    if (!source_extent_.try_area(&source_count_) ||
        !coding_extent_.try_area(&coding_count_) ||
        !block_extent_.try_area(&block_count_) ||
        coding_count_ > std::numeric_limits<size_t>::max() / 3 ||
        source_count_ > std::numeric_limits<size_t>::max() / sizeof(float) ||
        block_count_ > std::numeric_limits<size_t>::max() / 3) {
      return Status::InvalidArgument("CUDA exact AQ geometry overflows");
    }
    coefficient_count_ = 3 * coding_count_;
    const size_t pixel_dispatches = (source_count_ + 255) / 256;
    const size_t coefficient_dispatches = (coefficient_count_ + 255) / 256;
    if (block_count_ > backend_->state_->maximum_grid_x ||
        pixel_dispatches > backend_->state_->maximum_grid_x ||
        coefficient_dispatches > backend_->state_->maximum_grid_x) {
      return Status::InvalidArgument(
          "CUDA exact AQ geometry exceeds device launch limits");
    }

    options_ = preparation.options;
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

    size_t group_count = 0;
    const size_t group_width =
        (block_extent_.width + kVarDctAcGroupBlockDimension - 1) /
        kVarDctAcGroupBlockDimension;
    const size_t group_height =
        (block_extent_.height + kVarDctAcGroupBlockDimension - 1) /
        kVarDctAcGroupBlockDimension;
    if (group_width != 0 &&
        group_height > std::numeric_limits<size_t>::max() / group_width) {
      return Status::InvalidArgument("CUDA exact AQ group geometry overflows");
    }
    group_count = group_width * group_height;
    try {
      coefficient_staging_.resize(coefficient_count_);
      group_offsets_.resize(group_count);
      block_readback_.resize(block_count_);
      maximum_readback_.resize(3 * block_count_);
      for (std::vector<float>& plane : linear_readback_) {
        plane.resize(source_count_);
      }
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
          "Unable to allocate CUDA exact AQ host staging");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
          "CUDA exact AQ host staging dimensions are too large");
    }

    size_t persistent_bytes = 0;
    for (size_t channel = 0; channel < 3; ++channel) {
      status = PlanPlane(DeviceElementType::kF32, source_extent_,
                         source_extent_.width, &persistent_bytes);
      if (!status.ok()) return status;
    }
    for (size_t image = 0; image < 2; ++image) {
      for (size_t channel = 0; channel < 3; ++channel) {
        status = PlanPlane(DeviceElementType::kF32, coding_extent_,
                           coding_extent_.width, &persistent_bytes);
        if (!status.ok()) return status;
      }
    }
    for (size_t image = 0; image < filter_scratch_count_; ++image) {
      for (size_t channel = 0; channel < 3; ++channel) {
        status = PlanPlane(DeviceElementType::kF32, coding_extent_,
                           coding_extent_.width, &persistent_bytes);
        if (!status.ok()) return status;
      }
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      status = PlanPlane(DeviceElementType::kF32, source_extent_,
                         source_extent_.width, &persistent_bytes);
      if (!status.ok()) return status;
    }
    status = PlanPlane(DeviceElementType::kI32, {2 * block_count_, 1},
                       2 * block_count_, &persistent_bytes);
    if (!status.ok()) return status;

    size_t staging_bytes = 0;
    status = PlanPlane(DeviceElementType::kF32, {coefficient_count_, 1},
                       coefficient_count_, &staging_bytes);
    if (!status.ok()) return status;
    status = PlanPlane(DeviceElementType::kF32, {coefficient_count_, 1},
                       coefficient_count_, &staging_bytes);
    if (!status.ok()) return status;
    status = PlanPlane(DeviceElementType::kF32, block_extent_,
                       block_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    if (options_.metric == AqEvaluationMetric::kButteraugli) {
      status = PlanPlane(DeviceElementType::kF32, source_extent_,
                         source_extent_.width, &staging_bytes);
      if (!status.ok()) return status;
      status = PlanPlane(DeviceElementType::kF32, {1, 1}, 1, &staging_bytes);
      if (!status.ok()) return status;
    }
    status = PlanPlane(DeviceElementType::kF32, block_extent_,
                       block_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    if (options_.metric == AqEvaluationMetric::kMaximumError) {
      status = PlanPlane(DeviceElementType::kF32, {3 * block_count_, 1},
                         3 * block_count_, &staging_bytes);
      if (!status.ok()) return status;
    }
    status = PlanPlane(DeviceElementType::kI32, {1, 1}, 1, &staging_bytes);
    if (!status.ok()) return status;

    status = persistent_.Prepare(*backend_, persistent_bytes);
    if (!status.ok()) return status;
    status = staging_.Prepare(*backend_, staging_bytes);
    if (!status.ok()) return status;
    for (DevicePlaneView& plane : original_) {
      status = AllocatePlane(persistent_, DeviceElementType::kF32,
                             source_extent_, source_extent_.width, &plane);
      if (!status.ok()) return status;
    }
    for (DevicePlaneView& plane : coding_) {
      status = AllocatePlane(persistent_, DeviceElementType::kF32,
                             coding_extent_, coding_extent_.width, &plane);
      if (!status.ok()) return status;
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
    status = AllocatePlane(staging_, DeviceElementType::kF32,
                           {coefficient_count_, 1}, coefficient_count_,
                           &coefficients_device_);
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
    status = AllocatePlane(staging_, DeviceElementType::kI32, {1, 1}, 1,
                           &error_device_);
    if (!status.ok()) return status;

    for (size_t channel = 0; channel < 3; ++channel) {
      status =
          UploadPlane(*backend_, preparation.original_linear_rgb.plane[channel],
                      original_[channel]);
      if (!status.ok()) return status;
      status = UploadPlane(*backend_, preparation.coding_opsin.plane[channel],
                           coding_[channel]);
      if (!status.ok()) return status;
    }
    status = UploadAnchors(metadata.device_anchors);
    if (!status.ok()) return status;

    strategies_ = *preparation.strategies;
    epf_sharpness_ = std::move(metadata.epf_sharpness);
    batches_ = metadata.batches;
    row_major_anchors_ = std::move(metadata.row_major_anchors);
    anchor_count_ = metadata.device_anchors.size();

    if (options_.metric == AqEvaluationMetric::kButteraugli) {
      status = PrepareDeviceButteraugli(
          *backend_,
          {.reference_linear_rgb = ConstImage(original_),
           .options = options_.butteraugli},
          &butteraugli_);
      if (!status.ok()) return status;
    }
    const DeviceButteraugliMemoryStats butteraugli_memory =
        butteraugli_ == nullptr ? DeviceButteraugliMemoryStats{}
                                : butteraugli_->memory_stats();
    if (staging_.capacity_bytes() >
            std::numeric_limits<size_t>::max() -
                butteraugli_memory.prepared_allocation_bytes ||
        staging_.capacity_bytes() >
            std::numeric_limits<size_t>::max() -
                butteraugli_memory.peak_comparison_scratch_bytes) {
      return Status::InvalidArgument(
          "CUDA exact AQ memory accounting overflows");
    }
    memory_stats_ = {
        persistent_.capacity_bytes(),
        staging_.capacity_bytes() +
            butteraugli_memory.prepared_allocation_bytes,
        staging_.capacity_bytes() +
            butteraugli_memory.peak_comparison_scratch_bytes,
    };
    InitializeKernelParams();
    return Status::Ok();
  }

  Status Evaluate(AqEvaluationInput input, AqEvaluationOutput output) override {
    Status status = ValidateOutput(output);
    if (!status.ok()) return status;
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return Status::FailedPrecondition(
          "CUDA exact AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA exact AQ evaluation was invalidated");
    }
    status = ValidateInput(input);
    if (!status.ok()) return status;

    VarDctEncoderFrame candidate_frame;
    try {
      if (output.final != nullptr) {
        candidate_frame = *input.exact_coefficients;
      }
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("Unable to stage CUDA exact AQ final frame");
    } catch (const std::length_error&) {
      return Status::InvalidArgument("CUDA exact AQ final frame is too large");
    }

    const bool exact_linear = input.exact_reconstructed_linear_rgb.valid();
    if (!exact_linear) {
      status = StageCoefficients(input);
      if (!status.ok()) return status;
    }

    if (!exact_linear) {
      status = backend_->CopyHostToDevice(*coefficients_device_.buffer,
                                          coefficient_staging_.data(),
                                          coefficient_count_ * sizeof(float),
                                          coefficients_device_.offset_bytes);
      if (status.ok()) {
        status = UploadPlane(*backend_, input.epf_inverse_sigma,
                             inverse_sigma_device_);
      }
    } else {
      for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
        status = UploadPlane(
            *backend_, input.exact_reconstructed_linear_rgb.plane[channel],
            reconstructed_linear_[channel]);
      }
    }
    if (!status.ok()) return Invalidate(status);

    EvaluationContext context{this, exact_linear};
    std::unique_ptr<GpuSubmission> submission;
    status = backend_->SubmitCompute(
        &CudaPreparedExactAqEvaluation::EncodeReconstruction, &context,
        &submission);
    if (!status.ok() || submission == nullptr) {
      return Invalidate(
          status.ok() ? Status::Internal("CUDA exact AQ returned no submission")
                      : status);
    }
    status = submission->Wait();
    if (!status.ok()) return Invalidate(status);
    status = ReadAndCheckDeviceError();
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
          &CudaPreparedExactAqEvaluation::EncodeBlockReduction, this,
          &submission);
      if (!status.ok() || submission == nullptr) {
        return Invalidate(
            status.ok()
                ? Status::Internal("CUDA exact AQ block reduction returned "
                                   "no submission")
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
                "CUDA exact AQ maximum-error readback is invalid"));
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
          Status::DeviceError("CUDA exact AQ bounded readback is invalid"));
    }

    const bool reconstruction_requested =
        output.final != nullptr &&
        ImageDescriptorSpecified(output.final->reconstructed_linear_rgb);
    if (reconstruction_requested) {
      for (size_t channel = 0; channel < 3; ++channel) {
        status = backend_->CopyDeviceToHost(
            *reconstructed_linear_[channel].buffer,
            linear_readback_[channel].data(), source_count_ * sizeof(float),
            reconstructed_linear_[channel].offset_bytes);
        if (!status.ok()) return Invalidate(status);
        if (!std::ranges::all_of(linear_readback_[channel], [](float value) {
              return std::isfinite(value);
            })) {
          return Invalidate(Status::DeviceError(
              "CUDA exact AQ reconstruction readback is invalid"));
        }
      }
    }

    for (size_t y = 0; y < block_extent_.height; ++y) {
      std::copy_n(block_readback_.data() + y * block_extent_.width,
                  block_extent_.width, output.block_distance_map.Row(y));
    }
    *output.score = candidate_score;
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
          "CUDA exact AQ evaluation is already in use");
    }
    if (invalid_) {
      return Status::FailedPrecondition(
          "CUDA exact AQ evaluation was invalidated");
    }
    status = UploadAnchors(metadata.device_anchors);
    if (!status.ok()) return Invalidate(status);
    strategies_ = strategies;
    epf_sharpness_ = std::move(metadata.epf_sharpness);
    batches_ = metadata.batches;
    row_major_anchors_ = std::move(metadata.row_major_anchors);
    anchor_count_ = metadata.device_anchors.size();
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
    std::array<CudaAqExactBatch, 7> batches{};
    std::vector<CudaAqAnchor> device_anchors;
    std::vector<HostAnchor> row_major_anchors;
    std::vector<uint8_t> epf_sharpness;
  };

  struct EvaluationContext {
    CudaPreparedExactAqEvaluation* self = nullptr;
    bool exact_linear = false;
  };

  Status BuildMetadata(const AcStrategyGrid& strategies,
                       ConstPlaneU8View epf_sharpness,
                       Metadata* metadata) const {
    if (metadata == nullptr || !strategies.complete() ||
        strategies.extent() != block_extent_ ||
        !ValidHostPlaneLayout(epf_sharpness) ||
        epf_sharpness.extent != block_extent_) {
      return Status::InvalidArgument(
          "CUDA exact AQ reconfiguration metadata is invalid");
    }
    try {
      Metadata candidate;
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
          if (!status.ok() || !SupportedStrategy(cell.strategy) ||
              epf_sharpness.Row(y)[x] >= 8) {
            return Status::InvalidArgument(
                "CUDA exact AQ strategy or EPF value is unsupported");
          }
          if (cell.is_anchor) {
            const size_t batch_index = StrategyBatchIndex(cell.strategy);
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
              "CUDA exact AQ strategy metadata disappeared");
        }
        const size_t count = grouped[index].size();
        const size_t coefficient_count = info->coefficient_count();
        if (anchor_offset > std::numeric_limits<uint32_t>::max() ||
            count > std::numeric_limits<uint32_t>::max() ||
            count > std::numeric_limits<uint32_t>::max() / 3 ||
            3 * count > backend_->state_->maximum_grid_x ||
            coefficient_offset > std::numeric_limits<uint32_t>::max() ||
            coefficient_count > std::numeric_limits<uint32_t>::max() ||
            info->pixel_extent().width > std::numeric_limits<uint32_t>::max() ||
            info->pixel_extent().height >
                std::numeric_limits<uint32_t>::max()) {
          return Status::InvalidArgument(
              "CUDA exact AQ strategy metadata exceeds kernel limits");
        }
        candidate.batches[index] = {
            static_cast<uint32_t>(anchor_offset),
            static_cast<uint32_t>(count),
            static_cast<uint32_t>(coefficient_offset),
            static_cast<uint32_t>(coefficient_count),
            static_cast<uint32_t>(info->pixel_extent().width),
            static_cast<uint32_t>(info->pixel_extent().height),
            static_cast<uint32_t>(info->covered_blocks.width),
            static_cast<uint32_t>(info->covered_blocks.height),
        };
        candidate.device_anchors.insert(candidate.device_anchors.end(),
                                        grouped[index].begin(),
                                        grouped[index].end());
        anchor_offset += count;
        if (count != 0 &&
            coefficient_count >
                (std::numeric_limits<size_t>::max() - coefficient_offset) /
                    (3 * count)) {
          return Status::InvalidArgument(
              "CUDA exact AQ coefficient metadata overflows");
        }
        coefficient_offset += 3 * count * coefficient_count;
      }
      if (anchor_offset != candidate.row_major_anchors.size() ||
          coefficient_offset != coefficient_count_ ||
          anchor_offset > block_count_) {
        return Status::Internal(
            "CUDA exact AQ strategies do not cover the coding image");
      }
      *metadata = std::move(candidate);
      return Status::Ok();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
          "Unable to allocate CUDA exact AQ strategy metadata");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
          "CUDA exact AQ strategy metadata is too large");
    }
  }

  Status ValidateInput(AqEvaluationInput input) const {
    if (input.exact_coefficients == nullptr ||
        !ValidHostPlaneLayout(input.raw_quant_field) ||
        input.raw_quant_field.extent != block_extent_ ||
        !ValidHostPlaneLayout(input.epf_inverse_sigma) ||
        input.epf_inverse_sigma.extent != block_extent_ ||
        PlaneDescriptorSpecified(input.quant_field)) {
      return Status::InvalidArgument("CUDA exact AQ input geometry is invalid");
    }
    const bool exact_linear_specified =
        ImageDescriptorSpecified(input.exact_reconstructed_linear_rgb);
    const bool exact_linear = input.exact_reconstructed_linear_rgb.valid();
    if ((exact_linear_specified && !exact_linear) ||
        (exact_linear &&
         input.exact_reconstructed_linear_rgb.extent() != source_extent_) ||
        (exact_linear &&
         options_.metric == AqEvaluationMetric::kMaximumError)) {
      return Status::InvalidArgument("CUDA exact AQ linear handoff is invalid");
    }
    Quantizer quantizer;
    Status status = Quantizer::Create(input.quantizer, &quantizer);
    if (!status.ok()) return status;
    for (size_t y = 0; y < block_extent_.height; ++y) {
      for (size_t x = 0; x < block_extent_.width; ++x) {
        const int32_t raw = input.raw_quant_field.Row(y)[x];
        const float inverse_sigma = input.epf_inverse_sigma.Row(y)[x];
        if (raw < 1 || raw > kMaxRawQuant || !std::isfinite(inverse_sigma) ||
            inverse_sigma >= 0.0f) {
          return Status::InvalidArgument(
              "CUDA exact AQ quantization or EPF value is invalid");
        }
      }
    }
    if (exact_linear) {
      status = ValidateFiniteImage(input.exact_reconstructed_linear_rgb,
                                   "CUDA exact AQ linear handoff");
      if (!status.ok()) return status;
    }
    const bool valid_cfl = ValidHostPlaneLayout(input.y_to_x) &&
                           input.y_to_x.extent == tile_extent_ &&
                           ValidHostPlaneLayout(input.y_to_b) &&
                           input.y_to_b.extent == tile_extent_;
    if (!valid_cfl) {
      return Status::InvalidArgument(
          "CUDA exact AQ color-correlation maps are invalid");
    }
    const VarDctEncoderFrame& frame = *input.exact_coefficients;
    if (!frame.valid() || frame.geometry().frame() != source_extent_ ||
        frame.geometry().padded_frame() != coding_extent_ ||
        frame.strategies().extent() != block_extent_ ||
        frame.raw_quant_field().extent != block_extent_ ||
        frame.epf_sharpness().extent != block_extent_ ||
        frame.color_correlation().tile_extent() != tile_extent_ ||
        frame.quantizer().params().global_scale !=
            input.quantizer.global_scale ||
        frame.quantizer().params().quant_dc != input.quantizer.quant_dc ||
        frame.profile() != options_.profile) {
      return Status::InvalidArgument(
          "CUDA exact AQ coefficient frame does not match prepared state");
    }
    for (size_t y = 0; y < block_extent_.height; ++y) {
      for (size_t x = 0; x < block_extent_.width; ++x) {
        AcStrategyCell frame_cell;
        AcStrategyCell prepared_cell;
        if (!frame.strategies().Get(x, y, &frame_cell).ok() ||
            !strategies_.Get(x, y, &prepared_cell).ok() ||
            frame_cell.strategy != prepared_cell.strategy ||
            frame_cell.is_anchor != prepared_cell.is_anchor ||
            frame.raw_quant_field().Row(y)[x] !=
                input.raw_quant_field.Row(y)[x] ||
            frame.epf_sharpness().Row(y)[x] !=
                epf_sharpness_[y * block_extent_.width + x]) {
          return Status::InvalidArgument(
              "CUDA exact AQ coefficient decisions do not match input");
        }
      }
    }
    const ConstPlaneI8View frame_x = frame.color_correlation().y_to_x_map();
    const ConstPlaneI8View frame_b = frame.color_correlation().y_to_b_map();
    for (size_t y = 0; y < tile_extent_.height; ++y) {
      for (size_t x = 0; x < tile_extent_.width; ++x) {
        if (frame_x.Row(y)[x] != input.y_to_x.Row(y)[x] ||
            frame_b.Row(y)[x] != input.y_to_b.Row(y)[x]) {
          return Status::InvalidArgument(
              "CUDA exact AQ color factors do not match coefficient frame");
        }
      }
    }
    return Status::Ok();
  }

  Status ValidateOutput(AqEvaluationOutput output) const {
    if (!ValidHostPlaneLayout(output.block_distance_map) ||
        output.block_distance_map.extent != block_extent_ ||
        output.score == nullptr ||
        (options_.metric == AqEvaluationMetric::kMaximumError &&
         output.maximum_error == nullptr)) {
      return Status::InvalidArgument("CUDA exact AQ output is invalid");
    }
    if (output.final != nullptr) {
      const bool reconstruction =
          ImageDescriptorSpecified(output.final->reconstructed_linear_rgb);
      if (output.final->frame == nullptr ||
          (reconstruction &&
           (!output.final->reconstructed_linear_rgb.valid() ||
            output.final->reconstructed_linear_rgb.extent() != source_extent_ ||
            !std::ranges::all_of(output.final->reconstructed_linear_rgb.plane,
                                 [](PlaneF32View plane) {
                                   return ValidHostPlaneLayout(plane);
                                 })))) {
        return Status::InvalidArgument("CUDA exact AQ final output is invalid");
      }
    }
    return Status::Ok();
  }

  Status StageCoefficients(AqEvaluationInput input) {
    const VarDctEncoderFrame& frame = *input.exact_coefficients;
    std::ranges::fill(group_offsets_, 0);
    for (const HostAnchor& anchor : row_major_anchors_) {
      const AcStrategyInfo* info = GetAcStrategyInfo(anchor.strategy);
      if (info == nullptr) {
        return Status::Internal(
            "CUDA exact AQ strategy disappeared during staging");
      }
      const size_t group_x = anchor.block_x / kVarDctAcGroupBlockDimension;
      const size_t group_y = anchor.block_y / kVarDctAcGroupBlockDimension;
      const size_t group_index =
          group_y * frame.ac_group_extent().width + group_x;
      VarDctAcGroupView group;
      Status status = frame.GetAcGroup(group_index, &group);
      if (!status.ok()) return status;
      const CudaAqExactBatch& batch = batches_[anchor.batch_index];
      const size_t source_offset = group_offsets_[group_index];
      const size_t channel_stride =
          static_cast<size_t>(batch.anchor_count) * batch.coefficient_count;
      if (batch.coefficient_count != info->coefficient_count() ||
          source_offset > group.used_coefficient_count ||
          batch.coefficient_count >
              group.used_coefficient_count - source_offset) {
        return Status::InvalidArgument(
            "CUDA exact AQ coefficient group layout is inconsistent");
      }
      constexpr std::array<XybChannel, 3> kChannels = {
          XybChannel::kX, XybChannel::kY, XybChannel::kB};
      for (size_t channel = 0; channel < 3; ++channel) {
        const size_t destination_offset =
            batch.coefficient_offset + channel * channel_stride +
            anchor.index_in_batch * batch.coefficient_count;
        const float matrix_multiplier =
            channel == 0
                ? QuantizationMatrixMultiplier(options_.profile.x_qm_scale)
                : (channel == 2 ? QuantizationMatrixMultiplier(
                                      options_.profile.b_qm_scale)
                                : 1.0f);
        status = DequantizeAcBlock(
            anchor.strategy, frame.quantizer(),
            input.raw_quant_field.Row(anchor.block_y)[anchor.block_x],
            {.channel = kChannels[channel],
             .matrix_multiplier = matrix_multiplier},
            std::span<const int32_t>(
                group.coefficients[channel].data() + source_offset,
                batch.coefficient_count),
            std::span<float>(coefficient_staging_.data() + destination_offset,
                             batch.coefficient_count));
        if (!status.ok()) return status;
      }

      const std::array<float, 3> factors = frame.color_correlation().AcFactors(
          anchor.block_x / (kColorTileDimension / kJxlBlockDimension),
          anchor.block_y / (kColorTileDimension / kJxlBlockDimension));
      const size_t x_offset = batch.coefficient_offset +
                              anchor.index_in_batch * batch.coefficient_count;
      const size_t y_offset = x_offset + channel_stride;
      const size_t b_offset = y_offset + channel_stride;
      for (size_t coefficient = 0; coefficient < batch.coefficient_count;
           ++coefficient) {
        const float reconstructed_y =
            coefficient_staging_[y_offset + coefficient];
        coefficient_staging_[x_offset + coefficient] +=
            factors[0] * reconstructed_y;
        coefficient_staging_[b_offset + coefficient] +=
            factors[2] * reconstructed_y;
      }

      const ConstImage3FView dc = frame.dc();
      for (size_t channel = 0; channel < 3; ++channel) {
        std::array<float, 16> dc_workspace{};
        const size_t dc_count =
            info->covered_blocks.width * info->covered_blocks.height;
        if (dc_count > dc_workspace.size()) {
          return Status::Internal(
              "CUDA exact AQ strategy exceeds DC staging capacity");
        }
        for (size_t y = 0; y < info->covered_blocks.height; ++y) {
          std::copy_n(
              dc.plane[channel].Row(anchor.block_y + y) + anchor.block_x,
              info->covered_blocks.width,
              dc_workspace.data() + y * info->covered_blocks.width);
        }
        const size_t destination_offset =
            batch.coefficient_offset + channel * channel_stride +
            anchor.index_in_batch * batch.coefficient_count;
        status = ConvertDcToLowFrequencies(
            anchor.strategy,
            {dc_workspace.data(), info->covered_blocks,
             info->covered_blocks.width},
            std::span<float>(coefficient_staging_.data() + destination_offset,
                             batch.coefficient_count));
        if (!status.ok()) return status;
      }
      group_offsets_[group_index] += batch.coefficient_count;
    }
    for (size_t group_index = 0; group_index < group_offsets_.size();
         ++group_index) {
      VarDctAcGroupView group;
      Status status = frame.GetAcGroup(group_index, &group);
      if (!status.ok()) return status;
      if (group_offsets_[group_index] != group.used_coefficient_count) {
        return Status::InvalidArgument(
            "CUDA exact AQ coefficient group has unconsumed values");
      }
    }
    return Status::Ok();
  }

  static cudaError_t EncodeReconstruction(CudaBackend& backend,
                                          const void* opaque) {
    const auto& context = *static_cast<const EvaluationContext*>(opaque);
    CudaPreparedExactAqEvaluation& self = *context.self;
    cudaError_t status =
        cudaMemsetAsync(Pointer(self.error_device_), 0, sizeof(uint32_t),
                        backend.state_->stream);
    if (status != cudaSuccess) return status;
    if (!context.exact_linear) {
      for (const CudaAqExactBatch& batch : self.batches_) {
        if (batch.anchor_count == 0) continue;
        status = LaunchCudaDct(
            false,
            Pointer(self.coefficients_device_) + batch.coefficient_offset,
            Pointer(self.inverse_device_) + batch.coefficient_offset,
            3 * static_cast<size_t>(batch.anchor_count), batch.pixel_width,
            batch.pixel_height, backend.state_->stream);
        if (status != cudaSuccess) return status;
        status = LaunchCudaAqScatterReconstruction(
            AnchorPointer(self.anchors_device_), Pointer(self.inverse_device_),
            MutablePointers(self.reconstructed_),
            static_cast<uint32_t>(self.coding_extent_.width), batch,
            backend.state_->stream);
        if (status != cudaSuccess) return status;
      }
      status = self.EncodePostprocess(backend);
      if (status != cudaSuccess) return status;
    }
    if (self.options_.metric == AqEvaluationMetric::kMaximumError) {
      const std::array<DevicePlaneView, 3> filtered = self.FinalFilteredImage();
      for (const CudaAqExactBatch& batch : self.batches_) {
        status = LaunchCudaAqReduceMaximumError(
            ConstPointers(self.coding_), ConstPointers(filtered),
            static_cast<uint32_t>(self.coding_extent_.width),
            static_cast<uint32_t>(self.coding_extent_.width),
            AnchorPointer(self.anchors_device_), Pointer(self.block_device_),
            static_cast<uint32_t>(self.block_extent_.width),
            Pointer(self.maximum_device_), ErrorPointer(self.error_device_),
            static_cast<uint32_t>(self.source_extent_.width),
            static_cast<uint32_t>(self.source_extent_.height),
            self.options_.maximum_error, batch, backend.state_->stream);
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
                                    ErrorPointer(error_device_),
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
          ConstPointers(current), Pointer(inverse_sigma_device_),
          MutablePointers(destination), ErrorPointer(error_device_),
          epf_params_[pass], backend.state_->stream);
      if (status != cudaSuccess) return status;
      current = destination;
      ++stage;
    }
    return LaunchCudaAqOpsinToLinear(
        ConstPointers(current), MutablePointers(reconstructed_linear_),
        ErrorPointer(error_device_), color_params_, backend.state_->stream);
  }

  static cudaError_t EncodeBlockReduction(CudaBackend& backend,
                                          const void* opaque) {
    auto& self =
        *static_cast<CudaPreparedExactAqEvaluation*>(const_cast<void*>(opaque));
    cudaError_t status =
        cudaMemsetAsync(Pointer(self.error_device_), 0, sizeof(uint32_t),
                        backend.state_->stream);
    if (status != cudaSuccess) return status;
    for (const CudaAqExactBatch& batch : self.batches_) {
      status = LaunchCudaAqReduceButteraugli(
          Pointer(self.distance_device_),
          static_cast<uint32_t>(self.source_extent_.width),
          AnchorPointer(self.anchors_device_), Pointer(self.block_device_),
          static_cast<uint32_t>(self.block_extent_.width),
          ErrorPointer(self.error_device_),
          static_cast<uint32_t>(self.source_extent_.width),
          static_cast<uint32_t>(self.source_extent_.height), batch,
          backend.state_->stream);
      if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
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
                    : (pass == 2 ? options_.profile.loop_filter.epf_options
                                       .pass2_sigma_scale
                                 : 1.0f);
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
    color_params_ = {
        static_cast<uint32_t>(source_extent_.width),
        static_cast<uint32_t>(source_extent_.height),
        static_cast<uint32_t>(coding_extent_.width),
        static_cast<uint32_t>(source_extent_.width),
        255.0f / options_.profile.intensity_target,
    };
  }

  Status AllocatePlane(DeviceScratchArena& arena, DeviceElementType type,
                       Extent2D extent, size_t row_stride,
                       DevicePlaneView* plane) {
    return arena.AllocatePlane(type, extent, row_stride, kArenaAlignment,
                               plane);
  }

  Status UploadAnchors(std::span<const CudaAqAnchor> anchors) {
    return backend_->CopyHostToDevice(*anchors_device_.buffer, anchors.data(),
                                      anchors.size_bytes(),
                                      anchors_device_.offset_bytes);
  }

  Status ReadAndCheckDeviceError() {
    uint32_t error = 0;
    Status status =
        backend_->CopyDeviceToHost(*error_device_.buffer, &error, sizeof(error),
                                   error_device_.offset_bytes);
    if (!status.ok()) return status;
    return error == 0
               ? Status::Ok()
               : Status::DeviceError(
                     "CUDA exact AQ detected invalid device numerics (flag " +
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

  static float* Pointer(DevicePlaneView view) {
    CudaBuffer* buffer = CudaBackend::AsCudaBuffer(*view.buffer);
    return reinterpret_cast<float*>(static_cast<std::byte*>(buffer->pointer()) +
                                    view.offset_bytes);
  }

  static const float* Pointer(ConstDevicePlaneView view) {
    const CudaBuffer* buffer = CudaBackend::AsCudaBuffer(*view.buffer);
    return reinterpret_cast<const float*>(
        static_cast<const std::byte*>(buffer->pointer()) + view.offset_bytes);
  }

  static CudaAqAnchor* AnchorPointer(DevicePlaneView view) {
    CudaBuffer* buffer = CudaBackend::AsCudaBuffer(*view.buffer);
    return reinterpret_cast<CudaAqAnchor*>(
        static_cast<std::byte*>(buffer->pointer()) + view.offset_bytes);
  }

  static unsigned int* ErrorPointer(DevicePlaneView view) {
    CudaBuffer* buffer = CudaBackend::AsCudaBuffer(*view.buffer);
    return reinterpret_cast<unsigned int*>(
        static_cast<std::byte*>(buffer->pointer()) + view.offset_bytes);
  }

  static std::array<float*, 3> MutablePointers(
      const std::array<DevicePlaneView, 3>& image) {
    return {Pointer(image[0]), Pointer(image[1]), Pointer(image[2])};
  }

  static std::array<const float*, 3> ConstPointers(
      const std::array<DevicePlaneView, 3>& image) {
    return {Pointer(static_cast<ConstDevicePlaneView>(image[0])),
            Pointer(static_cast<ConstDevicePlaneView>(image[1])),
            Pointer(static_cast<ConstDevicePlaneView>(image[2]))};
  }

  static ConstDeviceImage3View ConstImage(
      const std::array<DevicePlaneView, 3>& image) {
    return {{{static_cast<ConstDevicePlaneView>(image[0]),
              static_cast<ConstDevicePlaneView>(image[1]),
              static_cast<ConstDevicePlaneView>(image[2])}}};
  }

  CudaBackend* backend_ = nullptr;
  DeviceScratchArena persistent_;
  DeviceScratchArena staging_;
  std::array<DevicePlaneView, 3> original_{};
  std::array<DevicePlaneView, 3> coding_{};
  std::array<DevicePlaneView, 3> reconstructed_{};
  std::array<std::array<DevicePlaneView, 3>, 2> filter_scratch_{};
  std::array<DevicePlaneView, 3> reconstructed_linear_{};
  DevicePlaneView anchors_device_{};
  DevicePlaneView coefficients_device_{};
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
  size_t coefficient_count_ = 0;
  size_t anchor_count_ = 0;
  size_t filter_stage_count_ = 0;
  size_t filter_scratch_count_ = 0;
  int final_filter_index_ = -1;
  AqEvaluationOptions options_{};
  AcStrategyGrid strategies_{};
  std::vector<uint8_t> epf_sharpness_;
  std::array<CudaAqExactBatch, 7> batches_{};
  std::vector<HostAnchor> row_major_anchors_;
  std::vector<float> coefficient_staging_;
  std::vector<size_t> group_offsets_;
  std::vector<float> block_readback_;
  std::vector<float> maximum_readback_;
  std::array<std::vector<float>, 3> linear_readback_;
  std::unique_ptr<PreparedDeviceButteraugli> butteraugli_;
  CudaAqGaborishParams gaborish_params_{};
  std::array<CudaAqEpfParams, 3> epf_params_{};
  CudaAqColorParams color_params_{};
  AqEvaluationMemoryStats memory_stats_{};
  std::mutex mutex_;
  bool invalid_ = false;
};

Status PrepareCudaExactAqEvaluation(
    CudaBackend& backend, const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared) {
  if (prepared == nullptr) {
    return Status::InvalidArgument(
        "CUDA exact AQ prepared output pointer is null");
  }
  prepared->reset();
  try {
    auto candidate = std::make_unique<CudaPreparedExactAqEvaluation>(backend);
    Status status = candidate->Prepare(preparation);
    if (!status.ok()) return status;
    *prepared = std::move(candidate);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
        "Unable to allocate CUDA exact AQ prepared state");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
        "CUDA exact AQ prepared dimensions are too large");
  }
}

}  // namespace gjxl::cuda_internal
