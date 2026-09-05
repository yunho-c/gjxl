// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_backend_internal.h"
#include "gpu/metal/metal_butteraugli_encoding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "gpu/metal/metal_status.h"
#include "gpu/metal/metal_butteraugli_test.h"
#include "gpu/scratch.h"

#define setComputePipelineState(state)                                    \
  setComputePipelineState(state);                                         \
  ::gjxl::metal_internal::RecordMetalComputePipelineState(state)

namespace gjxl::metal_internal {
namespace {

constexpr size_t kPlaneAlignment = 64;
constexpr size_t kReductionWidth = 256;
constexpr std::array<float, 5> kBlurSigmas{
  1.2f,
  7.15593339443f,
  3.22489901262f,
  1.56416327805f,
  2.7f,
};
constexpr std::array<size_t, 5> kKernelSizes{5, 33, 15, 7, 13};

constexpr size_t kPsychoReference = 0;
constexpr size_t kPsychoDistorted = 10;
constexpr size_t kPsychoPlaneCount = 10;
constexpr size_t kReferenceMask = 20;

// Psycho-image encoding finishes before difference encoding begins. Reuse its
// six image/blur planes for the three AC and three DC accumulators, and reuse
// its convolution intermediates for difference scratch. The final staging
// plane remains distinct because multiscale composition and diagnostic capture
// can keep it live alongside the difference scratch.
constexpr size_t kImage = 21;
constexpr size_t kAc = kImage;
constexpr size_t kDc = kImage + 3;
constexpr size_t kPsychoWork = 27;
constexpr size_t kWork = kPsychoWork;
// Psycho construction uses kWork through kWork + 2 and raw masks use + 4.
// Packed subscale caches leave this full-size mask slot invariant after prep.
constexpr size_t kReferenceErodedMask = kWork + 3;
constexpr size_t kFinalStaging = 32;
constexpr size_t kWorkingPlaneCount = 33;
constexpr size_t kMaltaTileWidth = 32;
constexpr size_t kMaltaTileHeight = 8;
constexpr size_t kMaltaRadius = 4;
constexpr size_t kOpsinBlur5TileWidth = 16;
constexpr size_t kOpsinBlur5TileHeight = 8;
constexpr size_t kOpsinBlur5Radius = 2;
constexpr size_t kOpsinBlur5RawPlaneElements =
  (kOpsinBlur5TileWidth + 2 * kOpsinBlur5Radius) *
  (kOpsinBlur5TileHeight + 2 * kOpsinBlur5Radius);
constexpr size_t kOpsinBlur5HorizontalPlaneElements =
  kOpsinBlur5TileWidth *
  (kOpsinBlur5TileHeight + 2 * kOpsinBlur5Radius);
constexpr size_t kOpsinBlur5ThreadgroupMemoryBytes =
  3 * (kOpsinBlur5RawPlaneElements +
       kOpsinBlur5HorizontalPlaneElements) * sizeof(float);
static_assert(kOpsinBlur5TileWidth > 0 && kOpsinBlur5TileHeight > 0);
static_assert(kOpsinBlur5TileWidth * kOpsinBlur5TileHeight <= 1024);
constexpr size_t kLowMediumTileWidth = 16;
constexpr size_t kLowMediumTileHeight = 64;
constexpr size_t kLowMediumRadius = 16;
constexpr size_t kLowMediumHorizontalPlaneElements =
  kLowMediumTileWidth *
  (kLowMediumTileHeight + 2 * kLowMediumRadius);
constexpr size_t kLowMediumThreadgroupMemoryBytes =
  3 * kLowMediumHorizontalPlaneElements * sizeof(float);
static_assert(kLowMediumTileWidth > 0 && kLowMediumTileHeight > 0);
static_assert(kLowMediumTileWidth * kLowMediumTileHeight <= 1024);

using PsychoPlanes = std::array<DevicePlaneView, kPsychoPlaneCount>;

constexpr std::array<double, 6> kMaltaWeights{
  37.0819870399,
  8246.75321353,
  18.7237414387,
  6923.99476109,
  1.10039032555,
  173.5,
};
constexpr std::array<double, 6> kMaltaNorms{
  130262059.556,
  1009002.70582,
  4498534.45232,
  8051.15833247,
  71.7800275169,
  5.0,
};
constexpr std::array<size_t, 6> kMaltaAccumulationOrder{4, 5, 2, 3, 0, 1};
constexpr std::array<size_t, 6> kMaltaPsychoPlane{4, 3, 7, 6, 9, 8};

struct PlaneParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
};

struct ExpandParams {
  uint32_t input_width;
  uint32_t input_height;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t xborder;
  uint32_t yborder;
};

struct SubsampleParams {
  uint32_t input_width;
  uint32_t input_height;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t input_stride;
  uint32_t output_stride;
};

struct ConvolutionParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t kernel_size;
};

struct OpsinParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride0;
  uint32_t input_stride1;
  uint32_t input_stride2;
  uint32_t output_stride;
  float intensity_target;
};

struct FrequencyLowMediumTiledParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
};

struct FrequencyConvolutionChannelParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t intermediate_stride;
  uint32_t output_stride;
  uint32_t channel;
  uint32_t kernel_size;
};

struct MaltaScaleParams {
  uint32_t width;
  uint32_t height;
  uint32_t reference_stride;
  uint32_t distorted_stride;
  uint32_t output_stride;
  uint32_t low_frequency;
  float norm2_0_gt_1;
  float norm2_0_lt_1;
  float norm;
};

struct MaltaResponseParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t accumulation_stride;
  uint32_t low_frequency;
  uint32_t initialize_accumulation;
  uint32_t write_response;
};

struct MaltaFusedParams {
  uint32_t width;
  uint32_t height;
  uint32_t reference_stride;
  uint32_t distorted_stride;
  uint32_t response_stride;
  uint32_t accumulation_stride;
  uint32_t low_frequency;
  uint32_t initialize_accumulation;
  uint32_t write_response;
  float norm2_0_gt_1;
  float norm2_0_lt_1;
  float norm;
};

struct DifferenceParams {
  uint32_t width;
  uint32_t height;
  uint32_t reference_stride;
  uint32_t distorted_stride;
  uint32_t work_stride;
  float asymmetry;
};

struct FinalParams {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t output_stride;
  uint32_t reference_mask_stride;
  uint32_t mask_stride;
  float x_multiplier;
};

struct FinalL2Params {
  uint32_t width;
  uint32_t height;
  uint32_t reference_stride;
  uint32_t distorted_stride;
  uint32_t work_stride;
  uint32_t output_stride;
  uint32_t reference_mask_stride;
  uint32_t mask_stride;
  float asymmetry;
  float x_multiplier;
};

struct CropParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t xborder;
  uint32_t yborder;
};

struct ComposeParams {
  uint32_t width;
  uint32_t height;
  uint32_t main_stride;
  uint32_t sub_stride;
  uint32_t output_stride;
};

struct ReductionParams {
  uint32_t width;
  uint32_t input_stride;
  uint32_t input_count;
};

struct ResidentReductionParams {
  uint32_t source_width;
  uint32_t source_height;
  uint32_t work_stride;
  uint32_t sub_stride;
  uint32_t block_stride;
  uint32_t anchor_offset;
  uint32_t anchor_count;
  uint32_t pixel_width;
  uint32_t pixel_height;
  uint32_t covered_width;
  uint32_t covered_height;
  float x_multiplier;
  float asymmetry;
};

static_assert(std::is_trivially_copyable_v<PlaneParams>);
static_assert(sizeof(PlaneParams) == 16);
static_assert(sizeof(ExpandParams) == 32);
static_assert(sizeof(SubsampleParams) == 24);
static_assert(sizeof(ConvolutionParams) == 20);
static_assert(sizeof(OpsinParams) == 28);
static_assert(sizeof(FrequencyLowMediumTiledParams) == 16);
static_assert(sizeof(FrequencyConvolutionChannelParams) == 28);
static_assert(sizeof(MaltaScaleParams) == 36);
static_assert(sizeof(MaltaResponseParams) == 32);
static_assert(sizeof(MaltaFusedParams) == 48);
static_assert(sizeof(DifferenceParams) == 24);
static_assert(sizeof(FinalParams) == 28);
static_assert(sizeof(FinalL2Params) == 40);
static_assert(std::is_standard_layout_v<ResidentReductionParams>);
static_assert(std::is_trivially_copyable_v<ResidentReductionParams>);
static_assert(sizeof(ResidentReductionParams) == 52);
static_assert(sizeof(CropParams) == 24);
static_assert(sizeof(ComposeParams) == 20);
static_assert(sizeof(ReductionParams) == 12);

[[nodiscard]] bool AddAlignedAllocation(
  size_t size_bytes,
  size_t* capacity) noexcept {

  if (capacity == nullptr ||
      *capacity > std::numeric_limits<size_t>::max() - (kPlaneAlignment - 1)) {
    return false;
  }
  const size_t aligned =
    (*capacity + kPlaneAlignment - 1) & ~(kPlaneAlignment - 1);
  if (aligned > std::numeric_limits<size_t>::max() - size_bytes) {
    return false;
  }
  *capacity = aligned + size_bytes;
  return true;
}

[[nodiscard]] Status CreatePipeline(
  MTL::Device* device,
  MTL::Library* library,
  std::string_view function_name,
  NS::SharedPtr<MTL::ComputePipelineState>* out) {

  if (device == nullptr || library == nullptr || function_name.empty() ||
      out == nullptr) {
    return Status::InvalidArgument(
      "CreateButteraugliPipeline received invalid argument");
  }
  const std::string name_string(function_name);
  NS::String* name = NS::String::string(
    name_string.c_str(), NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(library->newFunction(name));
  if (!function) {
    return Status::Internal(
      std::string("Metal function not found: ") + name_string);
  }
  NS::Error* error = nullptr;
  auto pipeline = NS::TransferPtr(
    device->newComputePipelineState(function.get(), &error));
  if (!pipeline) {
    return metal::ErrorToStatus(error, "newComputePipelineState");
  }
  RegisterMetalComputePipeline(pipeline.get(), function_name);
  *out = std::move(pipeline);
  return Status::Ok();
}

[[nodiscard]] std::vector<float> MakeGaussianKernel(float sigma) {
  const int radius = std::max(1, static_cast<int>(2.25f * std::abs(sigma)));
  const size_t size = 2 * static_cast<size_t>(radius) + 1;
  std::vector<float> result(size);
  const double exponent_scale = -1.0 / (2.0 * sigma * sigma);
  for (int index = -radius; index <= radius; ++index) {
    result[static_cast<size_t>(index + radius)] =
      static_cast<float>(std::exp(exponent_scale * index * index));
  }
  return result;
}

[[nodiscard]] ConstDevicePlaneView AsConst(DevicePlaneView view) noexcept {
  return view;
}

}  // namespace

class MetalPreparedDeviceButteraugli final
    : public PreparedDeviceButteraugli {
public:
  MetalPreparedDeviceButteraugli(
    MetalBackend& backend,
    DeviceButteraugliPrepareDescriptor descriptor)
    : PreparedDeviceButteraugli(backend, descriptor),
      metal_(backend) {}

  [[nodiscard]] Status PrepareStorage() {
    const Extent2D requested = extent();
    if (requested.width > std::numeric_limits<uint32_t>::max() ||
        requested.height > std::numeric_limits<uint32_t>::max()) {
      return Status::InvalidArgument(
        "Device Butteraugli extent exceeds Metal shader limits");
    }
    expanded_ = requested.width < 8 || requested.height < 8;
    working_extent_ = expanded_
      ? Extent2D{std::max<size_t>(8, requested.width),
                 std::max<size_t>(8, requested.height)}
      : requested;
    xborder_ = requested.width < 8 ? (8 - requested.width) / 2 : 0;
    yborder_ = requested.height < 8 ? (8 - requested.height) / 2 : 0;
    multiscale_ = !expanded_ && requested.width >= 15 &&
                  requested.height >= 15;
    if (multiscale_) {
      sub_extent_ = {(requested.width + 1) / 2,
                     (requested.height + 1) / 2};
    }

    size_t working_area = 0;
    size_t requested_area = 0;
    if (!working_extent_.try_area(&working_area) ||
        !requested.try_area(&requested_area) ||
        requested_area > std::numeric_limits<uint32_t>::max() ||
        working_area > std::numeric_limits<size_t>::max() / sizeof(float)) {
      return Status::InvalidArgument(
        "Device Butteraugli scratch geometry overflows");
    }
    const size_t plane_bytes = working_area * sizeof(float);
    size_t sub_area = 0;
    size_t sub_plane_bytes = 0;
    if (multiscale_) {
      if (!sub_extent_.try_area(&sub_area) ||
          sub_area >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
        return Status::InvalidArgument(
          "Device Butteraugli cached-reference geometry overflows");
      }
      sub_plane_bytes = sub_area * sizeof(float);
    }
    const size_t partial_count =
      requested_area / kReductionWidth +
      static_cast<size_t>(requested_area % kReductionWidth != 0);
    if (partial_count >
        std::numeric_limits<size_t>::max() / sizeof(float)) {
      return Status::InvalidArgument(
        "Device Butteraugli reduction scratch overflows");
    }

    size_t capacity = 0;
    for (size_t index = 0; index < kWorkingPlaneCount; ++index) {
      if (!AddAlignedAllocation(plane_bytes, &capacity)) {
        return Status::InvalidArgument(
          "Device Butteraugli scratch capacity overflows");
      }
    }
    if (multiscale_) {
      for (size_t index = 0; index < reference_sub_.size() + 2; ++index) {
        if (!AddAlignedAllocation(sub_plane_bytes, &capacity)) {
          return Status::InvalidArgument(
            "Device Butteraugli cached-reference capacity overflows");
        }
      }
    }
    for (size_t index = 0; index < 2; ++index) {
      if (!AddAlignedAllocation(partial_count * sizeof(float), &capacity)) {
        return Status::InvalidArgument(
          "Device Butteraugli reduction capacity overflows");
      }
    }
    for (size_t kernel_size : kKernelSizes) {
      if (!AddAlignedAllocation(kernel_size * sizeof(float), &capacity)) {
        return Status::InvalidArgument(
          "Device Butteraugli kernel capacity overflows");
      }
    }

    cached_reference_bytes_ =
      (kPsychoPlaneCount + 2) * plane_bytes +
      (multiscale_ ? (kPsychoPlaneCount + 2) * sub_plane_bytes : 0);
    gaussian_kernel_bytes_ = 0;
    for (size_t kernel_size : kKernelSizes) {
      gaussian_kernel_bytes_ += kernel_size * sizeof(float);
    }
    peak_comparison_scratch_bytes_ =
      (kWorkingPlaneCount - (kPsychoPlaneCount + 2)) * plane_bytes +
      2 * partial_count * sizeof(float);

    Status status = scratch_.Prepare(metal_, capacity);
    if (!status.ok()) return status;
    for (DevicePlaneView& plane : planes_) {
      status = scratch_.AllocatePlane(
        DeviceElementType::kF32,
        working_extent_,
        working_extent_.width,
        kPlaneAlignment,
        &plane);
      if (!status.ok()) return status;
    }
    reference_eroded_mask_ = Plane(kReferenceErodedMask, working_extent_);
    if (multiscale_) {
      for (DevicePlaneView& plane : reference_sub_) {
        status = scratch_.AllocatePlane(
          DeviceElementType::kF32,
          sub_extent_,
          sub_extent_.width,
          kPlaneAlignment,
          &plane);
        if (!status.ok()) return status;
      }
      status = scratch_.AllocatePlane(
        DeviceElementType::kF32, sub_extent_, sub_extent_.width,
        kPlaneAlignment, &reference_sub_mask_);
      if (!status.ok()) return status;
      status = scratch_.AllocatePlane(
        DeviceElementType::kF32, sub_extent_, sub_extent_.width,
        kPlaneAlignment, &reference_sub_eroded_mask_);
      if (!status.ok()) return status;
    }
    const Extent2D reduction_extent{partial_count, 1};
    status = scratch_.AllocatePlane(
      DeviceElementType::kF32, reduction_extent, partial_count,
      kPlaneAlignment, &reduction_a_);
    if (!status.ok()) return status;
    status = scratch_.AllocatePlane(
      DeviceElementType::kF32, reduction_extent, partial_count,
      kPlaneAlignment, &reduction_b_);
    if (!status.ok()) return status;

    for (size_t index = 0; index < kernels_.size(); ++index) {
      status = scratch_.AllocatePlane(
        DeviceElementType::kF32,
        {kKernelSizes[index], 1},
        kKernelSizes[index],
        kPlaneAlignment,
        &kernels_[index]);
      if (!status.ok()) return status;
      const std::vector<float> kernel = MakeGaussianKernel(kBlurSigmas[index]);
      if (kernel.size() != kKernelSizes[index]) {
        return Status::Internal(
          "Device Butteraugli Gaussian kernel size is inconsistent");
      }
      status = metal_.CopyHostToDevice(
        *kernels_[index].buffer,
        kernel.data(),
        kernel.size() * sizeof(float),
        kernels_[index].offset_bytes);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  [[nodiscard]] Status PrepareReference(
    gpu_profile_internal::GpuProfilingMode profiling_mode =
      gpu_profile_internal::GpuProfilingMode::kDisabled,
    gpu_profile_internal::GpuExecutionProfile* profile = nullptr) {

    const bool profiling = profiling_mode !=
      gpu_profile_internal::GpuProfilingMode::kDisabled;
    if (profiling != (profile != nullptr)) {
      return Status::InvalidArgument(
        "Metal Butteraugli preparation profile request is invalid");
    }
    const PreparationContext context{this};
    std::unique_ptr<GpuSubmission> submission;
    Status status;
    if (profiling) {
      const MetalProfiledComputeStage stage{
        .stage_id = "frontend.prepare_aq.reference",
        .encode =
          &MetalPreparedDeviceButteraugli::EncodePreparationSubmission,
        .context = &context,
      };
      status = metal_.SubmitComputeProfiled(
        "gjxl Butteraugli reference preparation profile",
        std::span<const MetalProfiledComputeStage>(&stage, 1),
        profiling_mode, &submission);
    } else {
      status = metal_.SubmitCompute(
        "gjxl Butteraugli reference preparation",
        &MetalPreparedDeviceButteraugli::EncodePreparationSubmission,
        &context, &submission);
    }
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Status::Internal(
        "Metal Butteraugli preparation submission is null");
    }
    status = submission->Wait();
    if (!status.ok() || !profiling) return status;
    return metal_.ResolveGpuSubmissionProfile(
      *submission, "frontend.prepare_aq.reference", profiling_mode,
      profile);
  }

  [[nodiscard]] DeviceButteraugliMemoryStats memory_stats()
    const noexcept override {
    return {
      scratch_.capacity_bytes(),
      cached_reference_bytes_,
      gaussian_kernel_bytes_,
      peak_comparison_scratch_bytes_,
    };
  }

  [[nodiscard]] Status ConfigureCapture(MetalButteraugliStage stage) {
    if (expanded_ || multiscale_ ||
        static_cast<size_t>(stage) >=
          static_cast<size_t>(MetalButteraugliStage::kCount)) {
      return Status::InvalidArgument(
        "Metal Butteraugli stage capture requires one unexpanded scale");
    }
    capture_stage_ = stage;
    capture_ready_ = false;
    return Status::Ok();
  }

  [[nodiscard]] Status ReadCapture(PlaneF32View output) {
    if (!capture_stage_.has_value() || !capture_ready_ || !output.valid() ||
        output.extent != extent()) {
      return Status::InvalidArgument(
        "Metal Butteraugli stage capture is not readable");
    }
    size_t area = 0;
    if (!extent().try_area(&area)) {
      return Status::InvalidArgument(
        "Metal Butteraugli stage capture extent overflows");
    }
    try {
      std::vector<float> candidate(area);
      const DevicePlaneView capture = Plane(kFinalStaging, extent());
      const size_t row_bytes = extent().width * sizeof(float);
      for (size_t y = 0; y < extent().height; ++y) {
        Status status = metal_.CopyDeviceToHost(
          *capture.buffer,
          candidate.data() + y * extent().width,
          row_bytes,
          capture.offset_bytes + y * capture.row_stride * sizeof(float));
        if (!status.ok()) return status;
      }
      for (float value : candidate) {
        if (!std::isfinite(value)) {
          return Status::Internal(
            "Metal Butteraugli stage capture is non-finite");
        }
      }
      for (size_t y = 0; y < extent().height; ++y) {
        std::copy_n(
          candidate.data() + y * extent().width,
          extent().width,
          output.Row(y));
      }
      return Status::Ok();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate Metal Butteraugli stage readback");
    }
  }

  [[nodiscard]] Status ValidateEncodingDescriptor(
    const DeviceButteraugliComparisonDescriptor& descriptor) const {

    if (!valid()) {
      return Status::FailedPrecondition(
        "Prepared Metal Butteraugli state is invalid");
    }
    return ValidateDeviceButteraugliComparisonDescriptor(
      metal_, reference_linear_rgb(), extent(), descriptor);
  }

  [[nodiscard]] Status ValidateResidentEncodingDescriptor(
    const MetalButteraugliResidentComparisonDescriptor& descriptor) const {

    if (!valid()) {
      return Status::FailedPrecondition(
        "Prepared Metal Butteraugli state is invalid");
    }
    if (!multiscale_ || expanded_ || capture_stage_.has_value()) {
      return Status::Unavailable(
        "Resident Metal Butteraugli sinks require an uncaptured multiscale "
        "comparison");
    }
    Status status = ValidateDeviceImage3View(
      descriptor.distorted_linear_rgb, metal_.id());
    if (!status.ok()) return status;
    if (descriptor.distorted_linear_rgb.plane[0].element_type !=
          DeviceElementType::kF32 ||
        descriptor.distorted_linear_rgb.plane[0].extent != extent()) {
      return Status::InvalidArgument(
        "Resident Metal Butteraugli image geometry or type is invalid");
    }

    if (descriptor.batches.size() != 7) {
      return Status::InvalidArgument(
        "Resident Metal Butteraugli requires seven strategy batches");
    }
    size_t anchor_count = 0;
    for (const MetalButteraugliResidentBatch& batch : descriptor.batches) {
      if (batch.anchor_offset != anchor_count ||
          batch.pixel_width == 0 || batch.pixel_height == 0 ||
          batch.covered_width == 0 || batch.covered_height == 0 ||
          batch.pixel_width != 8u * batch.covered_width ||
          batch.pixel_height != 8u * batch.covered_height ||
          batch.anchor_count >
            std::numeric_limits<uint32_t>::max() - batch.anchor_offset) {
        return Status::InvalidArgument(
          "Resident Metal Butteraugli batch metadata is invalid");
      }
      anchor_count += batch.anchor_count;
    }
    if (anchor_count == 0 ||
        anchor_count > std::numeric_limits<size_t>::max() / 2) {
      return Status::InvalidArgument(
        "Resident Metal Butteraugli anchor count is invalid");
    }
    const Extent2D block_extent{
      (extent().width + 7) / 8, (extent().height + 7) / 8};
    if (descriptor.anchors.element_type != DeviceElementType::kI32 ||
        descriptor.anchors.extent != Extent2D{2 * anchor_count, 1} ||
        descriptor.block_distance.element_type != DeviceElementType::kF32 ||
        descriptor.block_distance.extent != block_extent ||
        descriptor.score_partials.element_type != DeviceElementType::kF32 ||
        descriptor.score_partials.extent != Extent2D{anchor_count, 1} ||
        descriptor.score.element_type != DeviceElementType::kF32 ||
        descriptor.score.extent != Extent2D{1, 1} ||
        descriptor.error.element_type != DeviceElementType::kI32 ||
        descriptor.error.extent != Extent2D{1, 1}) {
      return Status::InvalidArgument(
        "Resident Metal Butteraugli sink geometry or type is invalid");
    }

    DeviceMemoryRange anchors_range;
    status = ComputeDevicePlaneRange(
      descriptor.anchors, metal_.id(), &anchors_range);
    if (!status.ok()) return status;
    std::array<DeviceMemoryRange, 4> output_ranges;
    const std::array<DevicePlaneView, 4> outputs{
      descriptor.block_distance,
      descriptor.score_partials,
      descriptor.score,
      descriptor.error,
    };
    for (size_t index = 0; index < outputs.size(); ++index) {
      status = ComputeDevicePlaneRange(
        outputs[index], metal_.id(), &output_ranges[index]);
      if (!status.ok()) return status;
      if (DeviceRangesOverlap(anchors_range, output_ranges[index])) {
        return Status::InvalidArgument(
          "Resident Metal Butteraugli sink overlaps anchor metadata");
      }
      for (size_t previous = 0; previous < index; ++previous) {
        if (DeviceRangesOverlap(
              output_ranges[previous], output_ranges[index])) {
          return Status::InvalidArgument(
            "Resident Metal Butteraugli sinks overlap");
        }
      }
    }
    const std::array<ConstDeviceImage3View, 2> images{
      reference_linear_rgb(), descriptor.distorted_linear_rgb};
    for (ConstDeviceImage3View image : images) {
      for (ConstDevicePlaneView plane : image.plane) {
        DeviceMemoryRange input_range;
        status = ComputeDevicePlaneRange(
          plane, metal_.id(), &input_range);
        if (!status.ok()) return status;
        for (DeviceMemoryRange output_range : output_ranges) {
          if (DeviceRangesOverlap(input_range, output_range)) {
            return Status::InvalidArgument(
              "Resident Metal Butteraugli sink overlaps an input image");
          }
        }
      }
    }
    return Status::Ok();
  }

  void EncodeValidatedComparison(
    MTL::ComputeCommandEncoder* encoder,
    const DeviceButteraugliComparisonDescriptor& descriptor) {

    capture_ready_ = false;
    EncodeComparison(encoder, descriptor);
  }

  void EncodeValidatedResidentComparison(
    MTL::ComputeCommandEncoder* encoder,
    const MetalButteraugliResidentComparisonDescriptor& descriptor) {

    capture_ready_ = false;
    EncodeResidentComparison(encoder, descriptor);
  }

  void EncodeValidatedResidentComparisonProfileStage(
    MTL::ComputeCommandEncoder* encoder,
    const MetalButteraugliResidentComparisonDescriptor& descriptor,
    MetalButteraugliProfileStage stage) {

    EncodeResidentComparisonProfileStage(encoder, descriptor, stage);
  }

  void EncodeValidatedComparisonProfileStage(
    MTL::ComputeCommandEncoder* encoder,
    const DeviceButteraugliComparisonDescriptor& descriptor,
    MetalButteraugliProfileStage stage) {

    EncodeComparisonProfileStage(encoder, descriptor, stage);
  }

private:
  enum class DifferenceProfileStage : uint8_t {
    kAll,
    kMalta,
    kL2,
    kMaskAndFinal,
  };
  struct PreparationContext {
    MetalPreparedDeviceButteraugli* prepared = nullptr;
  };

  struct ComparisonContext {
    MetalPreparedDeviceButteraugli* prepared = nullptr;
    DeviceButteraugliComparisonDescriptor descriptor;
  };

  [[nodiscard]] Status CompareValidated(
    const DeviceButteraugliComparisonDescriptor& descriptor) override {

    capture_ready_ = false;
    const ComparisonContext context{this, descriptor};
    std::unique_ptr<GpuSubmission> submission;
    Status status = metal_.SubmitCompute(
      "gjxl Butteraugli comparison",
      &MetalPreparedDeviceButteraugli::EncodeSubmission,
      &context,
      &submission);
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Status::Internal(
        "Metal Butteraugli submission is null");
    }
    status = submission->Wait();
    if (status.ok() && capture_stage_.has_value()) capture_ready_ = true;
    return status;
  }

  [[nodiscard]] DevicePlaneView Plane(
    size_t index,
    Extent2D plane_extent) const noexcept {

    DevicePlaneView result = planes_[index];
    result.extent = plane_extent;
    return result;
  }

  [[nodiscard]] PsychoPlanes PsychoSlots(
    size_t base,
    Extent2D plane_extent) const noexcept {

    PsychoPlanes result;
    for (size_t index = 0; index < result.size(); ++index) {
      result[index] = Plane(base + index, plane_extent);
    }
    return result;
  }

  [[nodiscard]] PsychoPlanes ReferenceSubSlots() const noexcept {
    PsychoPlanes result = reference_sub_;
    for (DevicePlaneView& plane : result) plane.extent = sub_extent_;
    return result;
  }

  [[nodiscard]] DevicePlaneView TransposedPlane(
    size_t index,
    Extent2D source_extent) const noexcept {

    DevicePlaneView result = planes_[index];
    result.extent = {source_extent.height, source_extent.width};
    result.row_stride = working_extent_.height;
    return result;
  }

  [[nodiscard]] static MTL::Buffer* Handle(
    MetalBackend& backend,
    DevicePlaneView view) noexcept {

    return backend.AsMetalBuffer(*view.buffer)->handle();
  }

  [[nodiscard]] static MTL::Buffer* Handle(
    MetalBackend& backend,
    ConstDevicePlaneView view) noexcept {

    return const_cast<MTL::Buffer*>(
      backend.AsMetalBuffer(*view.buffer)->handle());
  }

  static void Bind(
    MTL::ComputeCommandEncoder* encoder,
    MTL::Buffer* buffer,
    size_t offset,
    NS::UInteger index) {

    encoder->setBuffer(buffer, static_cast<NS::UInteger>(offset), index);
  }

  static void EncodeSubmission(
    MetalBackend&,
    MTL::ComputeCommandEncoder* encoder,
    const void* opaque) {

    const auto& context = *static_cast<const ComparisonContext*>(opaque);
    context.prepared->EncodeComparison(encoder, context.descriptor);
  }

  static void EncodePreparationSubmission(
    MetalBackend&,
    MTL::ComputeCommandEncoder* encoder,
    const void* opaque) {

    const auto& context = *static_cast<const PreparationContext*>(opaque);
    context.prepared->EncodePreparation(encoder);
  }

  void EncodeCopy(
    MTL::ComputeCommandEncoder* encoder,
    ConstDevicePlaneView input,
    DevicePlaneView output,
    Extent2D plane_extent) {

    const PlaneParams params{
      static_cast<uint32_t>(plane_extent.width),
      static_cast<uint32_t>(plane_extent.height),
      static_cast<uint32_t>(input.row_stride),
      static_cast<uint32_t>(output.row_stride),
    };
    encoder->setComputePipelineState(metal_.butteraugli_pipelines_.copy.get());
    Bind(encoder, Handle(metal_, input), input.offset_bytes, 0);
    Bind(encoder, Handle(metal_, output), output.offset_bytes, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    metal_.DispatchPlane(encoder, plane_extent);
  }

  void MaybeCapture(
    MTL::ComputeCommandEncoder* encoder,
    MetalButteraugliStage stage,
    ConstDevicePlaneView source,
    Extent2D plane_extent) {

    if (capture_stage_ == stage) {
      EncodeCopy(
        encoder, source, Plane(kFinalStaging, plane_extent), plane_extent);
    }
  }

  void EncodeBlur(
    MTL::ComputeCommandEncoder* encoder,
    ConstDevicePlaneView input,
    size_t kernel_index,
    size_t intermediate_index,
    DevicePlaneView output,
    Extent2D plane_extent) {

    const ConstDevicePlaneView kernel = kernels_[kernel_index];
    if (kKernelSizes[kernel_index] == 5) {
      DevicePlaneView intermediate = Plane(intermediate_index, plane_extent);
      const ConvolutionParams horizontal{
        static_cast<uint32_t>(plane_extent.width),
        static_cast<uint32_t>(plane_extent.height),
        static_cast<uint32_t>(input.row_stride),
        static_cast<uint32_t>(intermediate.row_stride),
        5,
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.blur5_horizontal.get());
      Bind(encoder, Handle(metal_, input), input.offset_bytes, 0);
      Bind(encoder, Handle(metal_, kernel), kernel.offset_bytes, 1);
      Bind(encoder, Handle(metal_, intermediate), intermediate.offset_bytes, 2);
      encoder->setBytes(&horizontal, sizeof(horizontal), 3);
      metal_.DispatchPlane(encoder, plane_extent);

      const ConvolutionParams vertical{
        static_cast<uint32_t>(plane_extent.width),
        static_cast<uint32_t>(plane_extent.height),
        static_cast<uint32_t>(intermediate.row_stride),
        static_cast<uint32_t>(output.row_stride),
        5,
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.blur5_vertical.get());
      Bind(encoder, Handle(metal_, intermediate), intermediate.offset_bytes, 0);
      Bind(encoder, Handle(metal_, kernel), kernel.offset_bytes, 1);
      Bind(encoder, Handle(metal_, output), output.offset_bytes, 2);
      encoder->setBytes(&vertical, sizeof(vertical), 3);
      metal_.DispatchPlane(encoder, plane_extent);
      return;
    }

    DevicePlaneView intermediate =
      TransposedPlane(intermediate_index, plane_extent);
    const ConvolutionParams first{
      static_cast<uint32_t>(plane_extent.width),
      static_cast<uint32_t>(plane_extent.height),
      static_cast<uint32_t>(input.row_stride),
      static_cast<uint32_t>(intermediate.row_stride),
      static_cast<uint32_t>(kKernelSizes[kernel_index]),
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.convolution_transpose.get());
    Bind(encoder, Handle(metal_, input), input.offset_bytes, 0);
    Bind(encoder, Handle(metal_, kernel), kernel.offset_bytes, 1);
    Bind(encoder, Handle(metal_, intermediate), intermediate.offset_bytes, 2);
    encoder->setBytes(&first, sizeof(first), 3);
    metal_.DispatchPlane(encoder, plane_extent);

    const ConvolutionParams second{
      static_cast<uint32_t>(plane_extent.height),
      static_cast<uint32_t>(plane_extent.width),
      static_cast<uint32_t>(intermediate.row_stride),
      static_cast<uint32_t>(output.row_stride),
      static_cast<uint32_t>(kKernelSizes[kernel_index]),
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.convolution_transpose.get());
    Bind(encoder, Handle(metal_, intermediate), intermediate.offset_bytes, 0);
    Bind(encoder, Handle(metal_, kernel), kernel.offset_bytes, 1);
    Bind(encoder, Handle(metal_, output), output.offset_bytes, 2);
    encoder->setBytes(&second, sizeof(second), 3);
    metal_.DispatchPlane(encoder, {plane_extent.height, plane_extent.width});
  }

  void EncodeMaskPrecompute(
    MTL::ComputeCommandEncoder* encoder,
    const PsychoPlanes& psycho,
    DevicePlaneView destination,
    Extent2D scale_extent) {

    const PlaneParams params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(psycho[0].row_stride),
      static_cast<uint32_t>(destination.row_stride),
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.mask_precompute.get());
    for (size_t index = 0; index < 4; ++index) {
      const size_t psycho_index = index < 2 ? 6 + index : 8 + index - 2;
      DevicePlaneView plane = psycho[psycho_index];
      Bind(encoder, Handle(metal_, plane), plane.offset_bytes, index);
    }
    Bind(encoder, Handle(metal_, destination), destination.offset_bytes, 4);
    encoder->setBytes(&params, sizeof(params), 5);
    metal_.DispatchPlane(encoder, scale_extent);
  }

  void EncodeReferenceMask(
    MTL::ComputeCommandEncoder* encoder,
    DevicePlaneView destination,
    Extent2D scale_extent) {

    DevicePlaneView precomputed = Plane(kWork + 4, scale_extent);
    // The fused ultra-Y producer uses the psycho output stride. Reference
    // caches use that same stride, including the packed subscale planes.
    precomputed.row_stride = destination.row_stride;
    EncodeBlur(
      encoder, AsConst(precomputed), 4, kWork + 1, destination,
      scale_extent);
  }

  void EncodeReferenceErosion(
    MTL::ComputeCommandEncoder* encoder,
    ConstDevicePlaneView source,
    DevicePlaneView destination,
    Extent2D scale_extent) {

    const PlaneParams fuzzy_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(source.row_stride),
      static_cast<uint32_t>(destination.row_stride),
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.fuzzy_erosion.get());
    Bind(encoder, Handle(metal_, source),
         source.offset_bytes, 0);
    Bind(encoder, Handle(metal_, destination), destination.offset_bytes, 1);
    encoder->setBytes(&fuzzy_params, sizeof(fuzzy_params), 2);
    metal_.DispatchPlane(encoder, scale_extent);
  }

  void EncodePsychoImage(
    MTL::ComputeCommandEncoder* encoder,
    ConstDeviceImage3View input,
    const PsychoPlanes& psycho,
    Extent2D scale_extent,
    bool capture_reference,
    bool prepare_mask = true) {

    const OpsinParams opsin_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(input.plane[0].row_stride),
      static_cast<uint32_t>(input.plane[1].row_stride),
      static_cast<uint32_t>(input.plane[2].row_stride),
      static_cast<uint32_t>(working_extent_.width),
      options().intensity_target,
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.opsin_blur5_tiled.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      Bind(encoder, Handle(metal_, input.plane[channel]),
           input.plane[channel].offset_bytes, channel);
      DevicePlaneView xyb = Plane(kImage + channel, scale_extent);
      Bind(encoder, Handle(metal_, xyb), xyb.offset_bytes, 4 + channel);
    }
    Bind(encoder, Handle(metal_, kernels_[0]), kernels_[0].offset_bytes, 3);
    encoder->setBytes(&opsin_params, sizeof(opsin_params), 7);
    encoder->setThreadgroupMemoryLength(
      kOpsinBlur5ThreadgroupMemoryBytes, 0);
    DispatchMetalThreadgroups(
      encoder,
      MTL::Size(
        (scale_extent.width + kOpsinBlur5TileWidth - 1) /
          kOpsinBlur5TileWidth,
        (scale_extent.height + kOpsinBlur5TileHeight - 1) /
          kOpsinBlur5TileHeight,
        1),
      MTL::Size(kOpsinBlur5TileWidth, kOpsinBlur5TileHeight, 1));
    if (capture_reference) {
      for (size_t channel = 0; channel < 3; ++channel) {
        MaybeCapture(
          encoder,
          static_cast<MetalButteraugliStage>(
            static_cast<size_t>(MetalButteraugliStage::kOpsinX) + channel),
          AsConst(Plane(kImage + channel, scale_extent)),
          scale_extent);
      }
    }

    const FrequencyLowMediumTiledParams frequency_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(working_extent_.width),
      static_cast<uint32_t>(psycho[0].row_stride),
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.frequency_low_medium_tiled.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      DevicePlaneView xyb = Plane(kImage + channel, scale_extent);
      DevicePlaneView low = psycho[channel];
      DevicePlaneView medium = psycho[3 + channel];
      Bind(encoder, Handle(metal_, xyb), xyb.offset_bytes, channel);
      Bind(encoder, Handle(metal_, low), low.offset_bytes, 4 + channel);
      Bind(encoder, Handle(metal_, medium),
           medium.offset_bytes, 7 + channel);
    }
    Bind(encoder, Handle(metal_, kernels_[1]), kernels_[1].offset_bytes, 3);
    encoder->setBytes(&frequency_params, sizeof(frequency_params), 10);
    encoder->setThreadgroupMemoryLength(
      kLowMediumThreadgroupMemoryBytes, 0);
    DispatchMetalThreadgroups(
      encoder,
      MTL::Size(
        (scale_extent.width + kLowMediumTileWidth - 1) /
          kLowMediumTileWidth,
        (scale_extent.height + kLowMediumTileHeight - 1) /
          kLowMediumTileHeight,
        1),
      MTL::Size(kLowMediumTileWidth, kLowMediumTileHeight, 1));
    if (capture_reference) {
      for (size_t channel = 0; channel < 3; ++channel) {
        MaybeCapture(
          encoder,
          static_cast<MetalButteraugliStage>(
            static_cast<size_t>(MetalButteraugliStage::kLowFrequencyX) +
            channel),
          AsConst(psycho[channel]),
          scale_extent);
      }
    }

    for (size_t channel = 0; channel < 2; ++channel) {
      DevicePlaneView medium = psycho[3 + channel];
      DevicePlaneView high = psycho[6 + channel];
      DevicePlaneView intermediate =
        TransposedPlane(kPsychoWork, scale_extent);
      const ConvolutionParams convolution_params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(medium.row_stride),
        static_cast<uint32_t>(intermediate.row_stride),
        static_cast<uint32_t>(kKernelSizes[2]),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.convolution_transpose.get());
      Bind(encoder, Handle(metal_, medium), medium.offset_bytes, 0);
      Bind(encoder, Handle(metal_, kernels_[2]),
           kernels_[2].offset_bytes, 1);
      Bind(encoder, Handle(metal_, intermediate),
           intermediate.offset_bytes, 2);
      encoder->setBytes(
        &convolution_params, sizeof(convolution_params), 3);
      metal_.DispatchPlane(encoder, scale_extent);

      const FrequencyConvolutionChannelParams channel_params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(medium.row_stride),
        static_cast<uint32_t>(intermediate.row_stride),
        static_cast<uint32_t>(high.row_stride),
        static_cast<uint32_t>(channel),
        static_cast<uint32_t>(kKernelSizes[2]),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.frequency_high_convolve.get());
      Bind(encoder, Handle(metal_, intermediate),
           intermediate.offset_bytes, 0);
      Bind(encoder, Handle(metal_, kernels_[2]),
           kernels_[2].offset_bytes, 1);
      Bind(encoder, Handle(metal_, medium), medium.offset_bytes, 2);
      Bind(encoder, Handle(metal_, high), high.offset_bytes, 3);
      encoder->setBytes(&channel_params, sizeof(channel_params), 4);
      metal_.DispatchPlane(encoder, scale_extent);
    }
    DevicePlaneView medium_b = psycho[5];
    EncodeBlur(
      encoder, AsConst(medium_b), 2, kPsychoWork, medium_b, scale_extent);

    DevicePlaneView high_x = psycho[6];
    DevicePlaneView high_y = psycho[7];
    const PlaneParams suppress_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(high_y.row_stride),
      static_cast<uint32_t>(high_x.row_stride),
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.frequency_suppress_x.get());
    Bind(encoder, Handle(metal_, high_x), high_x.offset_bytes, 0);
    Bind(encoder, Handle(metal_, high_y), high_y.offset_bytes, 1);
    encoder->setBytes(&suppress_params, sizeof(suppress_params), 2);
    metal_.DispatchPlane(encoder, scale_extent);

    for (size_t channel = 0; channel < 2; ++channel) {
      DevicePlaneView high = psycho[6 + channel];
      DevicePlaneView ultra = psycho[8 + channel];
      DevicePlaneView intermediate =
        TransposedPlane(kPsychoWork, scale_extent);
      const ConvolutionParams convolution_params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(high.row_stride),
        static_cast<uint32_t>(intermediate.row_stride),
        static_cast<uint32_t>(kKernelSizes[3]),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.convolution_transpose.get());
      Bind(encoder, Handle(metal_, high), high.offset_bytes, 0);
      Bind(encoder, Handle(metal_, kernels_[3]),
           kernels_[3].offset_bytes, 1);
      Bind(encoder, Handle(metal_, intermediate),
           intermediate.offset_bytes, 2);
      encoder->setBytes(
        &convolution_params, sizeof(convolution_params), 3);
      metal_.DispatchPlane(encoder, scale_extent);

      const FrequencyConvolutionChannelParams channel_params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(high.row_stride),
        static_cast<uint32_t>(intermediate.row_stride),
        static_cast<uint32_t>(ultra.row_stride),
        static_cast<uint32_t>(channel),
        static_cast<uint32_t>(kKernelSizes[3]),
      };
      const bool fuse_mask = channel == 1 && prepare_mask;
      encoder->setComputePipelineState(
        fuse_mask
          ? metal_.butteraugli_pipelines_.frequency_ultra_mask_convolve.get()
          : metal_.butteraugli_pipelines_.frequency_ultra_convolve.get());
      Bind(encoder, Handle(metal_, intermediate),
           intermediate.offset_bytes, 0);
      Bind(encoder, Handle(metal_, kernels_[3]),
           kernels_[3].offset_bytes, 1);
      Bind(encoder, Handle(metal_, high), high.offset_bytes, 2);
      Bind(encoder, Handle(metal_, ultra), ultra.offset_bytes, 3);
      encoder->setBytes(&channel_params, sizeof(channel_params), 4);
      if (fuse_mask) {
        const DevicePlaneView high_x = psycho[6];
        const DevicePlaneView ultra_x = psycho[8];
        DevicePlaneView mask = Plane(kWork + 4, scale_extent);
        mask.row_stride = ultra.row_stride;
        Bind(encoder, Handle(metal_, high_x), high_x.offset_bytes, 5);
        Bind(encoder, Handle(metal_, ultra_x), ultra_x.offset_bytes, 6);
        Bind(encoder, Handle(metal_, mask), mask.offset_bytes, 7);
      }
      metal_.DispatchPlane(encoder, scale_extent);
    }
    if (capture_reference) {
      for (size_t channel = 0; channel < 3; ++channel) {
        MaybeCapture(
          encoder,
          static_cast<MetalButteraugliStage>(
            static_cast<size_t>(MetalButteraugliStage::kMediumFrequencyX) +
            channel),
          AsConst(psycho[3 + channel]),
          scale_extent);
      }
      for (size_t channel = 0; channel < 2; ++channel) {
        MaybeCapture(
          encoder,
          static_cast<MetalButteraugliStage>(
            static_cast<size_t>(MetalButteraugliStage::kHighFrequencyX) +
            channel),
          AsConst(psycho[6 + channel]),
          scale_extent);
        MaybeCapture(
          encoder,
          static_cast<MetalButteraugliStage>(
            static_cast<size_t>(MetalButteraugliStage::kUltraHighFrequencyX) +
            channel),
          AsConst(psycho[8 + channel]),
          scale_extent);
      }
    }
  }

  void EncodeDifference(
    MTL::ComputeCommandEncoder* encoder,
    const PsychoPlanes& reference,
    const PsychoPlanes& distorted,
    ConstDevicePlaneView mask_blurred_reference,
    ConstDevicePlaneView mask,
    Extent2D scale_extent,
    DevicePlaneView output,
    DifferenceProfileStage profile_stage = DifferenceProfileStage::kAll,
    bool emit_final = true,
    bool defer_l2 = false) {

    const float asymmetry = options().hf_asymmetry;
    const float sqrt_asymmetry = std::sqrt(asymmetry);
    if (profile_stage == DifferenceProfileStage::kAll ||
        profile_stage == DifferenceProfileStage::kMalta) {
      for (size_t stage_index : kMaltaAccumulationOrder) {
      const double weight_up = stage_index < 2
        ? kMaltaWeights[stage_index]
        : stage_index < 4
          ? kMaltaWeights[stage_index] * sqrt_asymmetry
          : kMaltaWeights[stage_index] * asymmetry;
      const double weight_down = stage_index < 2
        ? kMaltaWeights[stage_index]
        : stage_index < 4
          ? kMaltaWeights[stage_index] / sqrt_asymmetry
          : kMaltaWeights[stage_index] / asymmetry;
      constexpr double kWeight0 = 0.5;
      constexpr double kWeight1 = 0.33;
      constexpr double kLength = 3.75;
      const bool low_frequency = stage_index < 4;
      const double multiplier = low_frequency
        ? 0.611612573796
        : 0.39905817637;
      const double pre_up =
        multiplier * std::sqrt(kWeight0 * weight_up) / (kLength * 2.0 + 1.0);
      const double pre_down =
        multiplier * std::sqrt(kWeight1 * weight_down) / (kLength * 2.0 + 1.0);
      DevicePlaneView reference_plane =
        reference[kMaltaPsychoPlane[stage_index]];
      DevicePlaneView distorted_plane =
        distorted[kMaltaPsychoPlane[stage_index]];
      DevicePlaneView response = Plane(kWork + 1, scale_extent);
      const size_t channel = stage_index % 2 == 0 ? 1 : 0;
      DevicePlaneView accumulation = Plane(kAc + channel, scale_extent);
      const auto response_stage =
        static_cast<MetalButteraugliStage>(
          static_cast<size_t>(
            MetalButteraugliStage::kMaltaMediumFrequencyY) + stage_index);
      const uint32_t write_response =
        capture_stage_ == response_stage ? 1u : 0u;
      MTL::ComputePipelineState* fused_pipeline =
        metal_.butteraugli_pipelines_.malta_fused.get();
      if (fused_pipeline->maxTotalThreadsPerThreadgroup() >=
          kMaltaTileWidth * kMaltaTileHeight) {
        const MaltaFusedParams params{
          static_cast<uint32_t>(scale_extent.width),
          static_cast<uint32_t>(scale_extent.height),
          static_cast<uint32_t>(reference_plane.row_stride),
          static_cast<uint32_t>(distorted_plane.row_stride),
          static_cast<uint32_t>(response.row_stride),
          static_cast<uint32_t>(accumulation.row_stride),
          static_cast<uint32_t>(low_frequency),
          static_cast<uint32_t>(stage_index >= 4),
          write_response,
          static_cast<float>(pre_up * kMaltaNorms[stage_index]),
          static_cast<float>(pre_down * kMaltaNorms[stage_index]),
          static_cast<float>(kMaltaNorms[stage_index]),
        };
        encoder->setComputePipelineState(fused_pipeline);
        Bind(encoder, Handle(metal_, reference_plane),
             reference_plane.offset_bytes, 0);
        Bind(encoder, Handle(metal_, distorted_plane),
             distorted_plane.offset_bytes, 1);
        Bind(encoder, Handle(metal_, response), response.offset_bytes, 2);
        Bind(encoder, Handle(metal_, accumulation),
             accumulation.offset_bytes, 3);
        encoder->setBytes(&params, sizeof(params), 4);
        constexpr size_t kThreadgroupMemoryBytes =
          (kMaltaTileWidth + 2 * kMaltaRadius) *
          (kMaltaTileHeight + 2 * kMaltaRadius) * sizeof(float);
        encoder->setThreadgroupMemoryLength(kThreadgroupMemoryBytes, 0);
        DispatchMetalThreadgroups(
          encoder,
          MTL::Size(
            (scale_extent.width + kMaltaTileWidth - 1) / kMaltaTileWidth,
            (scale_extent.height + kMaltaTileHeight - 1) / kMaltaTileHeight,
            1),
          MTL::Size(kMaltaTileWidth, kMaltaTileHeight, 1));
      } else {
        DevicePlaneView scaled = Plane(kWork, scale_extent);
        const MaltaScaleParams scale_params{
          static_cast<uint32_t>(scale_extent.width),
          static_cast<uint32_t>(scale_extent.height),
          static_cast<uint32_t>(reference_plane.row_stride),
          static_cast<uint32_t>(distorted_plane.row_stride),
          static_cast<uint32_t>(scaled.row_stride),
          static_cast<uint32_t>(low_frequency),
          static_cast<float>(pre_up * kMaltaNorms[stage_index]),
          static_cast<float>(pre_down * kMaltaNorms[stage_index]),
          static_cast<float>(kMaltaNorms[stage_index]),
        };
        encoder->setComputePipelineState(
          metal_.butteraugli_pipelines_.malta_scale.get());
        Bind(encoder, Handle(metal_, reference_plane),
             reference_plane.offset_bytes, 0);
        Bind(encoder, Handle(metal_, distorted_plane),
             distorted_plane.offset_bytes, 1);
        Bind(encoder, Handle(metal_, scaled), scaled.offset_bytes, 2);
        encoder->setBytes(&scale_params, sizeof(scale_params), 3);
        metal_.DispatchPlane(encoder, scale_extent);

        const MaltaResponseParams response_params{
          static_cast<uint32_t>(scale_extent.width),
          static_cast<uint32_t>(scale_extent.height),
          static_cast<uint32_t>(scaled.row_stride),
          static_cast<uint32_t>(response.row_stride),
          static_cast<uint32_t>(accumulation.row_stride),
          static_cast<uint32_t>(low_frequency),
          static_cast<uint32_t>(stage_index >= 4),
          write_response,
        };
        encoder->setComputePipelineState(
          metal_.butteraugli_pipelines_.malta_response.get());
        Bind(encoder, Handle(metal_, scaled), scaled.offset_bytes, 0);
        Bind(encoder, Handle(metal_, response), response.offset_bytes, 1);
        Bind(encoder, Handle(metal_, accumulation),
             accumulation.offset_bytes, 2);
        encoder->setBytes(&response_params, sizeof(response_params), 3);
        metal_.DispatchPlane(encoder, scale_extent);
      }
      MaybeCapture(encoder, response_stage, AsConst(response), scale_extent);
      }
    }
    if (profile_stage == DifferenceProfileStage::kMalta) return;

    if (!defer_l2 &&
        (profile_stage == DifferenceProfileStage::kAll ||
         profile_stage == DifferenceProfileStage::kL2)) {
      const DifferenceParams difference_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(reference[0].row_stride),
      static_cast<uint32_t>(distorted[0].row_stride),
      static_cast<uint32_t>(Plane(kAc, scale_extent).row_stride),
      asymmetry,
    };
      encoder->setComputePipelineState(metal_.butteraugli_pipelines_.l2.get());
      for (size_t index = 0; index < 8; ++index) {
        DevicePlaneView plane = reference[index];
        Bind(encoder, Handle(metal_, plane), plane.offset_bytes, index);
      }
      for (size_t index = 0; index < 8; ++index) {
        DevicePlaneView plane = distorted[index];
        Bind(encoder, Handle(metal_, plane), plane.offset_bytes, 8 + index);
      }
      for (size_t channel = 0; channel < 3; ++channel) {
        DevicePlaneView ac = Plane(kAc + channel, scale_extent);
        DevicePlaneView dc = Plane(kDc + channel, scale_extent);
        Bind(encoder, Handle(metal_, ac), ac.offset_bytes, 16 + channel);
        Bind(encoder, Handle(metal_, dc), dc.offset_bytes, 19 + channel);
      }
      encoder->setBytes(&difference_params, sizeof(difference_params), 22);
      metal_.DispatchPlane(encoder, scale_extent);
    }
    if (profile_stage == DifferenceProfileStage::kL2) return;

    DevicePlaneView precomputed = Plane(kWork, scale_extent);
    DevicePlaneView mask_blurred_distorted = Plane(kWork + 4, scale_extent);
    DevicePlaneView uncached_reference_mask =
      Plane(kWork + 2, scale_extent);
    if (mask_blurred_reference.buffer == nullptr) {
      EncodeMaskPrecompute(encoder, reference, precomputed, scale_extent);
      EncodeBlur(
        encoder, AsConst(precomputed), 4, kWork + 1,
        uncached_reference_mask, scale_extent);
      mask_blurred_reference = AsConst(uncached_reference_mask);
    }

    MaybeCapture(
      encoder, MetalButteraugliStage::kMask, mask, scale_extent);

    // Ultra Y emitted raw activity here. Malta/L2 and reference-mask work use
    // other planes. The first blur pass consumes it before the second pass
    // overwrites this same plane with the completed distorted mask.
    EncodeBlur(
      encoder, AsConst(mask_blurred_distorted), 4, kWork + 1,
      mask_blurred_distorted, scale_extent);
    DevicePlaneView ac_y = Plane(kAc + 1, scale_extent);
    const bool capture_masked_ac =
      capture_stage_ == MetalButteraugliStage::kMaskedAcY;
    if (capture_masked_ac) {
      const PlaneParams masked_params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(mask_blurred_reference.row_stride),
        static_cast<uint32_t>(ac_y.row_stride),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.masked_ac.get());
      Bind(encoder, Handle(metal_, mask_blurred_reference),
           mask_blurred_reference.offset_bytes, 0);
      Bind(encoder, Handle(metal_, mask_blurred_distorted),
           mask_blurred_distorted.offset_bytes, 1);
      Bind(encoder, Handle(metal_, ac_y), ac_y.offset_bytes, 2);
      encoder->setBytes(&masked_params, sizeof(masked_params), 3);
      metal_.DispatchPlane(encoder, scale_extent);
      MaybeCapture(
        encoder, MetalButteraugliStage::kMaskedAcY, AsConst(ac_y),
        scale_extent);
    }
    if (!emit_final) return;

    if (defer_l2) {
      const FinalL2Params params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(reference[0].row_stride),
        static_cast<uint32_t>(distorted[0].row_stride),
        static_cast<uint32_t>(working_extent_.width),
        static_cast<uint32_t>(output.row_stride),
        static_cast<uint32_t>(mask_blurred_reference.row_stride),
        static_cast<uint32_t>(mask.row_stride),
        asymmetry,
        options().x_multiplier,
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.final_l2_masked_ac.get());
      for (size_t index = 0; index < 8; ++index) {
        Bind(encoder, Handle(metal_, reference[index]),
             reference[index].offset_bytes, index);
        Bind(encoder, Handle(metal_, distorted[index]),
             distorted[index].offset_bytes, 8 + index);
      }
      for (size_t channel = 0; channel < 2; ++channel) {
        const DevicePlaneView ac = Plane(kAc + channel, scale_extent);
        Bind(encoder, Handle(metal_, ac), ac.offset_bytes, 16 + channel);
      }
      Bind(encoder, Handle(metal_, mask), mask.offset_bytes, 18);
      Bind(encoder, Handle(metal_, mask_blurred_reference),
           mask_blurred_reference.offset_bytes, 19);
      Bind(encoder, Handle(metal_, mask_blurred_distorted),
           mask_blurred_distorted.offset_bytes, 20);
      Bind(encoder, Handle(metal_, output), output.offset_bytes, 21);
      encoder->setBytes(&params, sizeof(params), 22);
      metal_.DispatchPlane(encoder, scale_extent);
      return;
    }

    const FinalParams final_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(working_extent_.width),
      static_cast<uint32_t>(output.row_stride),
      static_cast<uint32_t>(mask_blurred_reference.row_stride),
      static_cast<uint32_t>(mask.row_stride),
      options().x_multiplier,
    };
    encoder->setComputePipelineState(
      capture_masked_ac
        ? metal_.butteraugli_pipelines_.final.get()
        : metal_.butteraugli_pipelines_.final_masked_ac.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      DevicePlaneView dc = Plane(kDc + channel, scale_extent);
      DevicePlaneView ac = Plane(kAc + channel, scale_extent);
      Bind(encoder, Handle(metal_, dc), dc.offset_bytes, channel);
      Bind(encoder, Handle(metal_, ac), ac.offset_bytes, 3 + channel);
    }
    Bind(encoder, Handle(metal_, mask), mask.offset_bytes, 6);
    if (capture_masked_ac) {
      Bind(encoder, Handle(metal_, output), output.offset_bytes, 7);
      encoder->setBytes(&final_params, sizeof(final_params), 8);
    } else {
      Bind(encoder, Handle(metal_, mask_blurred_reference),
           mask_blurred_reference.offset_bytes, 7);
      Bind(encoder, Handle(metal_, mask_blurred_distorted),
           mask_blurred_distorted.offset_bytes, 8);
      Bind(encoder, Handle(metal_, output), output.offset_bytes, 9);
      encoder->setBytes(&final_params, sizeof(final_params), 10);
    }
    metal_.DispatchPlane(encoder, scale_extent);
    MaybeCapture(
      encoder, MetalButteraugliStage::kFinalComposition, AsConst(output),
      scale_extent);
  }

  void EncodeExpand(
    MTL::ComputeCommandEncoder* encoder,
    ConstDeviceImage3View input,
    Extent2D requested,
    Extent2D expanded) {

    for (size_t channel = 0; channel < 3; ++channel) {
      DevicePlaneView output = Plane(kPsychoWork + channel, expanded);
      const ExpandParams params{
        static_cast<uint32_t>(requested.width),
        static_cast<uint32_t>(requested.height),
        static_cast<uint32_t>(expanded.width),
        static_cast<uint32_t>(expanded.height),
        static_cast<uint32_t>(input.plane[channel].row_stride),
        static_cast<uint32_t>(output.row_stride),
        static_cast<uint32_t>(xborder_),
        static_cast<uint32_t>(yborder_),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.expand.get());
      Bind(encoder, Handle(metal_, input.plane[channel]),
           input.plane[channel].offset_bytes, 0);
      Bind(encoder, Handle(metal_, output), output.offset_bytes, 1);
      encoder->setBytes(&params, sizeof(params), 2);
      metal_.DispatchPlane(encoder, expanded);
    }
  }

  void EncodeSubsample(
    MTL::ComputeCommandEncoder* encoder,
    ConstDeviceImage3View input,
    Extent2D requested,
    Extent2D subsampled) {

    for (size_t channel = 0; channel < 3; ++channel) {
      DevicePlaneView output = Plane(kPsychoWork + channel, subsampled);
      const SubsampleParams params{
        static_cast<uint32_t>(requested.width),
        static_cast<uint32_t>(requested.height),
        static_cast<uint32_t>(subsampled.width),
        static_cast<uint32_t>(subsampled.height),
        static_cast<uint32_t>(input.plane[channel].row_stride),
        static_cast<uint32_t>(output.row_stride),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.subsample.get());
      Bind(encoder, Handle(metal_, input.plane[channel]),
           input.plane[channel].offset_bytes, 0);
      Bind(encoder, Handle(metal_, output), output.offset_bytes, 1);
      encoder->setBytes(&params, sizeof(params), 2);
      metal_.DispatchPlane(encoder, subsampled);
    }
  }

  [[nodiscard]] ConstDeviceImage3View PsychoInputSlots(
    Extent2D image_extent) const noexcept {

    // Resampled RGB must remain disjoint from kImage: neighboring tiled Opsin
    // threadgroups can still be loading halo pixels while another group writes
    // its output.
    return {{{
      AsConst(Plane(kPsychoWork, image_extent)),
      AsConst(Plane(kPsychoWork + 1, image_extent)),
      AsConst(Plane(kPsychoWork + 2, image_extent)),
    }}};
  }

  void EncodeReductionPass(
    MTL::ComputeCommandEncoder* encoder,
    ConstDevicePlaneView input,
    size_t input_count,
    DevicePlaneView output) {

    const ReductionParams params{
      static_cast<uint32_t>(input.extent.width),
      static_cast<uint32_t>(input.row_stride),
      static_cast<uint32_t>(input_count),
    };
    const size_t output_count =
      input_count / kReductionWidth +
      static_cast<size_t>(input_count % kReductionWidth != 0);
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.maximum_reduction.get());
    Bind(encoder, Handle(metal_, input), input.offset_bytes, 0);
    Bind(encoder, Handle(metal_, output), output.offset_bytes, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    DispatchMetalThreadgroups(
      encoder,
      MTL::Size(static_cast<NS::UInteger>(output_count), 1, 1),
      MTL::Size(kReductionWidth, 1, 1));
  }

  void EncodeMaximumReduction(
    MTL::ComputeCommandEncoder* encoder,
    ConstDevicePlaneView input,
    DevicePlaneView output) {

    size_t input_count = 0;
    (void)input.extent.try_area(&input_count);
    bool use_a = true;
    while (true) {
      const size_t output_count =
        input_count / kReductionWidth +
        static_cast<size_t>(input_count % kReductionWidth != 0);
      DevicePlaneView destination = output_count == 1
        ? output
        : (use_a ? reduction_a_ : reduction_b_);
      EncodeReductionPass(encoder, input, input_count, destination);
      if (output_count == 1) break;
      input = destination;
      input.extent = {output_count, 1};
      input.row_stride = output_count;
      input_count = output_count;
      use_a = !use_a;
    }
  }

  void EncodeResidentReduction(
    MTL::ComputeCommandEncoder* encoder,
    ConstDevicePlaneView sub_map,
    const MetalButteraugliResidentComparisonDescriptor& descriptor) {

    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.resident_reduction.get());
    const PsychoPlanes reference = PsychoSlots(kPsychoReference, working_extent_);
    const PsychoPlanes distorted = PsychoSlots(kPsychoDistorted, working_extent_);
    for (size_t index = 0; index < 8; ++index) {
      Bind(encoder, Handle(metal_, reference[index]),
           reference[index].offset_bytes, index);
      Bind(encoder, Handle(metal_, distorted[index]),
           distorted[index].offset_bytes, 8 + index);
    }
    for (size_t channel = 0; channel < 2; ++channel) {
      const DevicePlaneView ac = Plane(kAc + channel, extent());
      Bind(encoder, Handle(metal_, ac), ac.offset_bytes, 16 + channel);
    }
    const DevicePlaneView mask = reference_eroded_mask_;
    const DevicePlaneView mask_blurred_distorted =
      Plane(kWork + 4, extent());
    const DevicePlaneView mask_blurred_reference =
      Plane(kReferenceMask, extent());
    Bind(encoder, Handle(metal_, mask), mask.offset_bytes, 18);
    Bind(encoder, Handle(metal_, mask_blurred_reference),
         mask_blurred_reference.offset_bytes, 19);
    Bind(encoder, Handle(metal_, mask_blurred_distorted),
         mask_blurred_distorted.offset_bytes, 20);
    Bind(encoder, Handle(metal_, sub_map), sub_map.offset_bytes, 21);
    Bind(encoder, Handle(metal_, descriptor.anchors),
         descriptor.anchors.offset_bytes, 22);
    Bind(encoder, Handle(metal_, descriptor.block_distance),
         descriptor.block_distance.offset_bytes, 23);
    Bind(encoder, Handle(metal_, descriptor.score_partials),
         descriptor.score_partials.offset_bytes, 24);
    Bind(encoder, Handle(metal_, descriptor.error),
         descriptor.error.offset_bytes, 25);

    for (const MetalButteraugliResidentBatch& batch : descriptor.batches) {
      if (batch.anchor_count == 0) continue;
      const ResidentReductionParams params{
        static_cast<uint32_t>(extent().width),
        static_cast<uint32_t>(extent().height),
        static_cast<uint32_t>(working_extent_.width),
        static_cast<uint32_t>(sub_map.row_stride),
        static_cast<uint32_t>(descriptor.block_distance.row_stride),
        batch.anchor_offset,
        batch.anchor_count,
        batch.pixel_width,
        batch.pixel_height,
        batch.covered_width,
        batch.covered_height,
        options().x_multiplier,
        options().hf_asymmetry,
      };
      encoder->setBytes(&params, sizeof(params), 26);
      DispatchMetalThreadgroups(
        encoder,
        MTL::Size(static_cast<NS::UInteger>(batch.anchor_count), 1, 1),
        MTL::Size(kReductionWidth, 1, 1));
    }
    EncodeMaximumReduction(
      encoder, descriptor.score_partials, descriptor.score);
  }

  void EncodeResidentComparison(
    MTL::ComputeCommandEncoder* encoder,
    const MetalButteraugliResidentComparisonDescriptor& descriptor) {

    const Extent2D requested = extent();
    const PsychoPlanes reference_main =
      PsychoSlots(kPsychoReference, working_extent_);
    const PsychoPlanes distorted =
      PsychoSlots(kPsychoDistorted, working_extent_);

    // Produce the quarter-area map first. The main-scale psycho and difference
    // preparation then reuse all ordinary working planes while kFinalStaging
    // keeps this subscale result live for the resident reducer.
    EncodeSubsample(
      encoder, descriptor.distorted_linear_rgb, requested, sub_extent_);
    EncodePsychoImage(
      encoder, PsychoInputSlots(sub_extent_), distorted, sub_extent_, false);
    const DevicePlaneView sub_map = Plane(kFinalStaging, sub_extent_);
    EncodeDifference(
      encoder, ReferenceSubSlots(), distorted,
      AsConst(reference_sub_mask_), reference_sub_eroded_mask_,
      sub_extent_, sub_map,
      DifferenceProfileStage::kAll, true, true);

    EncodePsychoImage(
      encoder, descriptor.distorted_linear_rgb, distorted, requested, false);
    EncodeDifference(
      encoder, reference_main, distorted,
      AsConst(Plane(kReferenceMask, requested)), reference_eroded_mask_,
      requested, {},
      DifferenceProfileStage::kAll, false, true);
    EncodeResidentReduction(encoder, AsConst(sub_map), descriptor);
  }

  void EncodeResidentComparisonProfileStage(
    MTL::ComputeCommandEncoder* encoder,
    const MetalButteraugliResidentComparisonDescriptor& descriptor,
    MetalButteraugliProfileStage stage) {

    const Extent2D requested = extent();
    const PsychoPlanes reference_main =
      PsychoSlots(kPsychoReference, working_extent_);
    const PsychoPlanes distorted =
      PsychoSlots(kPsychoDistorted, working_extent_);
    if (stage == MetalButteraugliProfileStage::kDistortedPsychoSub) {
      EncodeSubsample(
        encoder, descriptor.distorted_linear_rgb, requested, sub_extent_);
      EncodePsychoImage(
        encoder, PsychoInputSlots(sub_extent_), distorted, sub_extent_, false);
      return;
    }
    if (stage == MetalButteraugliProfileStage::kDistortedPsychoMain) {
      EncodePsychoImage(
        encoder, descriptor.distorted_linear_rgb, distorted, requested,
        false);
      return;
    }
    if (stage == MetalButteraugliProfileStage::kResidentReduction) {
      EncodeResidentReduction(
        encoder, AsConst(Plane(kFinalStaging, sub_extent_)), descriptor);
      return;
    }

    const bool sub_stage =
      stage == MetalButteraugliProfileStage::kMaltaSub ||
      stage == MetalButteraugliProfileStage::kL2Sub ||
      stage == MetalButteraugliProfileStage::kMaskAndFinalSub;
    const DifferenceProfileStage difference_stage =
      stage == MetalButteraugliProfileStage::kMaltaMain ||
          stage == MetalButteraugliProfileStage::kMaltaSub
        ? DifferenceProfileStage::kMalta
        : stage == MetalButteraugliProfileStage::kL2Main ||
              stage == MetalButteraugliProfileStage::kL2Sub
          ? DifferenceProfileStage::kL2
          : DifferenceProfileStage::kMaskAndFinal;
    if (sub_stage) {
      EncodeDifference(
        encoder, ReferenceSubSlots(), distorted,
        AsConst(reference_sub_mask_), reference_sub_eroded_mask_,
        sub_extent_,
        Plane(kFinalStaging, sub_extent_), difference_stage, true, true);
      return;
    }
    EncodeDifference(
      encoder, reference_main, distorted,
      AsConst(Plane(kReferenceMask, requested)), reference_eroded_mask_,
      requested, {},
      difference_stage,
      stage != MetalButteraugliProfileStage::kMaskAndFinalMain, true);
  }

  [[nodiscard]] bool NeedsReferencePsychoCapture() const noexcept {
    if (!capture_stage_.has_value()) return false;
    const size_t stage = static_cast<size_t>(*capture_stage_);
    return stage >= static_cast<size_t>(MetalButteraugliStage::kOpsinX) &&
           stage <=
             static_cast<size_t>(MetalButteraugliStage::kUltraHighFrequencyY);
  }

  void EncodePreparation(MTL::ComputeCommandEncoder* encoder) {
    const Extent2D requested = extent();
    const PsychoPlanes reference_main =
      PsychoSlots(kPsychoReference, working_extent_);
    if (expanded_) {
      EncodeExpand(
        encoder, reference_linear_rgb(), requested, working_extent_);
      EncodePsychoImage(
        encoder, PsychoInputSlots(working_extent_), reference_main,
        working_extent_, false);
      EncodeReferenceMask(
        encoder, Plane(kReferenceMask, working_extent_),
        working_extent_);
      EncodeReferenceErosion(
        encoder, AsConst(Plane(kReferenceMask, working_extent_)),
        reference_eroded_mask_, working_extent_);
      return;
    }

    EncodePsychoImage(
      encoder, reference_linear_rgb(), reference_main, requested, false);
    EncodeReferenceMask(
      encoder, Plane(kReferenceMask, requested), requested);
    EncodeReferenceErosion(
      encoder, AsConst(Plane(kReferenceMask, requested)),
      reference_eroded_mask_, requested);
    if (multiscale_) {
      EncodeSubsample(
        encoder, reference_linear_rgb(), requested, sub_extent_);
      EncodePsychoImage(
        encoder, PsychoInputSlots(sub_extent_), ReferenceSubSlots(),
        sub_extent_, false);
      EncodeReferenceMask(encoder, reference_sub_mask_, sub_extent_);
      EncodeReferenceErosion(
        encoder, AsConst(reference_sub_mask_), reference_sub_eroded_mask_,
        sub_extent_);
    }
  }

  void EncodeComparison(
    MTL::ComputeCommandEncoder* encoder,
    const DeviceButteraugliComparisonDescriptor& descriptor) {

    const Extent2D requested = extent();
    const PsychoPlanes reference_main =
      PsychoSlots(kPsychoReference, working_extent_);
    const PsychoPlanes distorted_main =
      PsychoSlots(kPsychoDistorted, working_extent_);
    if (capture_stage_.has_value() &&
        static_cast<size_t>(*capture_stage_) < kBlurSigmas.size()) {
      EncodeBlur(
        encoder,
        reference_linear_rgb().plane[0],
        static_cast<size_t>(*capture_stage_),
        kWork,
        Plane(kFinalStaging, requested),
        requested);
    }
    if (NeedsReferencePsychoCapture()) {
      EncodePsychoImage(
        encoder, reference_linear_rgb(), distorted_main, requested, true,
        false);
    }
    if (expanded_) {
      EncodeExpand(
        encoder, descriptor.distorted_linear_rgb, requested, working_extent_);
      EncodePsychoImage(
        encoder, PsychoInputSlots(working_extent_), distorted_main,
        working_extent_, false);
      DevicePlaneView staging = Plane(kFinalStaging, working_extent_);
      EncodeDifference(
        encoder, reference_main, distorted_main,
        AsConst(Plane(kReferenceMask, working_extent_)), reference_eroded_mask_,
        working_extent_,
        staging);
      const CropParams params{
        static_cast<uint32_t>(requested.width),
        static_cast<uint32_t>(requested.height),
        static_cast<uint32_t>(staging.row_stride),
        static_cast<uint32_t>(descriptor.distance_map.row_stride),
        static_cast<uint32_t>(xborder_),
        static_cast<uint32_t>(yborder_),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.crop.get());
      Bind(encoder, Handle(metal_, staging), staging.offset_bytes, 0);
      Bind(encoder, Handle(metal_, descriptor.distance_map),
           descriptor.distance_map.offset_bytes, 1);
      encoder->setBytes(&params, sizeof(params), 2);
      metal_.DispatchPlane(encoder, requested);
    } else {
      EncodePsychoImage(
        encoder, descriptor.distorted_linear_rgb, distorted_main, requested,
        false);
      EncodeDifference(
        encoder, reference_main, distorted_main,
        AsConst(Plane(kReferenceMask, requested)), reference_eroded_mask_,
        requested,
        descriptor.distance_map);

      if (multiscale_) {
        EncodeSubsample(
          encoder, descriptor.distorted_linear_rgb, requested, sub_extent_);
        EncodePsychoImage(
          encoder, PsychoInputSlots(sub_extent_), distorted_main,
          sub_extent_, false);
        DevicePlaneView sub_map = Plane(kFinalStaging, sub_extent_);
        EncodeDifference(
          encoder, ReferenceSubSlots(), distorted_main,
          AsConst(reference_sub_mask_), reference_sub_eroded_mask_,
          sub_extent_,
          sub_map);
        const ComposeParams params{
          static_cast<uint32_t>(requested.width),
          static_cast<uint32_t>(requested.height),
          static_cast<uint32_t>(descriptor.distance_map.row_stride),
          static_cast<uint32_t>(sub_map.row_stride),
          static_cast<uint32_t>(descriptor.distance_map.row_stride),
        };
        encoder->setComputePipelineState(
          metal_.butteraugli_pipelines_.compose.get());
        Bind(encoder, Handle(metal_, descriptor.distance_map),
             descriptor.distance_map.offset_bytes, 0);
        Bind(encoder, Handle(metal_, sub_map), sub_map.offset_bytes, 1);
        Bind(encoder, Handle(metal_, descriptor.distance_map),
             descriptor.distance_map.offset_bytes, 2);
        encoder->setBytes(&params, sizeof(params), 3);
        metal_.DispatchPlane(encoder, requested);
      }
    }
    EncodeMaximumReduction(
      encoder, descriptor.distance_map, descriptor.score);
  }

  void EncodeComparisonProfileStage(
    MTL::ComputeCommandEncoder* encoder,
    const DeviceButteraugliComparisonDescriptor& descriptor,
    MetalButteraugliProfileStage stage) {

    const Extent2D requested = extent();
    const PsychoPlanes reference_main =
      PsychoSlots(kPsychoReference, working_extent_);
    const PsychoPlanes distorted_main =
      PsychoSlots(kPsychoDistorted, working_extent_);

    if (stage == MetalButteraugliProfileStage::kDistortedPsychoMain) {
      if (expanded_) {
        EncodeExpand(
          encoder, descriptor.distorted_linear_rgb, requested,
          working_extent_);
        EncodePsychoImage(
          encoder, PsychoInputSlots(working_extent_), distorted_main,
          working_extent_, false);
      } else {
        EncodePsychoImage(
          encoder, descriptor.distorted_linear_rgb, distorted_main,
          requested, false);
      }
      return;
    }

    if (stage == MetalButteraugliProfileStage::kDistortedPsychoSub) {
      if (multiscale_) {
        EncodeSubsample(
          encoder, descriptor.distorted_linear_rgb, requested, sub_extent_);
        EncodePsychoImage(
          encoder, PsychoInputSlots(sub_extent_), distorted_main,
          sub_extent_, false);
      }
      return;
    }

    if (stage == MetalButteraugliProfileStage::kScoreReduction) {
      EncodeMaximumReduction(
        encoder, descriptor.distance_map, descriptor.score);
      return;
    }

    const bool sub_stage =
      stage == MetalButteraugliProfileStage::kMaltaSub ||
      stage == MetalButteraugliProfileStage::kL2Sub ||
      stage == MetalButteraugliProfileStage::kMaskAndFinalSub;
    const DifferenceProfileStage difference_stage =
      stage == MetalButteraugliProfileStage::kMaltaMain ||
          stage == MetalButteraugliProfileStage::kMaltaSub
        ? DifferenceProfileStage::kMalta
        : stage == MetalButteraugliProfileStage::kL2Main ||
              stage == MetalButteraugliProfileStage::kL2Sub
          ? DifferenceProfileStage::kL2
          : DifferenceProfileStage::kMaskAndFinal;
    if (sub_stage) {
      if (!multiscale_) return;
      DevicePlaneView sub_map = Plane(kFinalStaging, sub_extent_);
      EncodeDifference(
        encoder, ReferenceSubSlots(), distorted_main,
        AsConst(reference_sub_mask_), reference_sub_eroded_mask_,
        sub_extent_, sub_map,
        difference_stage);
      if (stage == MetalButteraugliProfileStage::kMaskAndFinalSub) {
        const ComposeParams params{
          static_cast<uint32_t>(requested.width),
          static_cast<uint32_t>(requested.height),
          static_cast<uint32_t>(descriptor.distance_map.row_stride),
          static_cast<uint32_t>(sub_map.row_stride),
          static_cast<uint32_t>(descriptor.distance_map.row_stride),
        };
        encoder->setComputePipelineState(
          metal_.butteraugli_pipelines_.compose.get());
        Bind(encoder, Handle(metal_, descriptor.distance_map),
             descriptor.distance_map.offset_bytes, 0);
        Bind(encoder, Handle(metal_, sub_map), sub_map.offset_bytes, 1);
        Bind(encoder, Handle(metal_, descriptor.distance_map),
             descriptor.distance_map.offset_bytes, 2);
        encoder->setBytes(&params, sizeof(params), 3);
        metal_.DispatchPlane(encoder, requested);
      }
      return;
    }
    if (expanded_) {
      DevicePlaneView staging = Plane(kFinalStaging, working_extent_);
      EncodeDifference(
        encoder, reference_main, distorted_main,
        AsConst(Plane(kReferenceMask, working_extent_)), reference_eroded_mask_,
        working_extent_,
        staging, difference_stage);
      if (stage == MetalButteraugliProfileStage::kMaskAndFinalMain) {
        const CropParams params{
          static_cast<uint32_t>(requested.width),
          static_cast<uint32_t>(requested.height),
          static_cast<uint32_t>(staging.row_stride),
          static_cast<uint32_t>(descriptor.distance_map.row_stride),
          static_cast<uint32_t>(xborder_),
          static_cast<uint32_t>(yborder_),
        };
        encoder->setComputePipelineState(
          metal_.butteraugli_pipelines_.crop.get());
        Bind(encoder, Handle(metal_, staging), staging.offset_bytes, 0);
        Bind(encoder, Handle(metal_, descriptor.distance_map),
             descriptor.distance_map.offset_bytes, 1);
        encoder->setBytes(&params, sizeof(params), 2);
        metal_.DispatchPlane(encoder, requested);
      }
      return;
    }

    EncodeDifference(
      encoder, reference_main, distorted_main,
      AsConst(Plane(kReferenceMask, requested)), reference_eroded_mask_,
      requested,
      descriptor.distance_map, difference_stage);
  }

  MetalBackend& metal_;
  DeviceScratchArena scratch_;
  std::array<DevicePlaneView, kWorkingPlaneCount> planes_;
  PsychoPlanes reference_sub_;
  DevicePlaneView reference_sub_mask_;
  DevicePlaneView reference_eroded_mask_;
  DevicePlaneView reference_sub_eroded_mask_;
  std::array<DevicePlaneView, kBlurSigmas.size()> kernels_;
  DevicePlaneView reduction_a_;
  DevicePlaneView reduction_b_;
  Extent2D working_extent_;
  Extent2D sub_extent_;
  size_t xborder_ = 0;
  size_t yborder_ = 0;
  bool expanded_ = false;
  bool multiscale_ = false;
  size_t cached_reference_bytes_ = 0;
  size_t gaussian_kernel_bytes_ = 0;
  size_t peak_comparison_scratch_bytes_ = 0;
  std::optional<MetalButteraugliStage> capture_stage_;
  bool capture_ready_ = false;
};

Status ValidatePreparedMetalButteraugliEncoding(
  PreparedDeviceButteraugli& prepared,
  const DeviceButteraugliComparisonDescriptor& descriptor) {

  auto* metal = dynamic_cast<MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "Prepared Butteraugli state is not a Metal implementation");
  }
  return metal->ValidateEncodingDescriptor(descriptor);
}

Status ValidatePreparedMetalButteraugliResidentEncoding(
  PreparedDeviceButteraugli& prepared,
  const MetalButteraugliResidentComparisonDescriptor& descriptor) {

  auto* metal = dynamic_cast<MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "Prepared Butteraugli state is not a Metal implementation");
  }
  return metal->ValidateResidentEncodingDescriptor(descriptor);
}

void EncodePreparedMetalButteraugli(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const DeviceButteraugliComparisonDescriptor& descriptor) {

  auto* metal = dynamic_cast<MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal != nullptr && encoder != nullptr) {
    metal->EncodeValidatedComparison(encoder, descriptor);
  }
}

void EncodePreparedMetalButteraugliResident(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const MetalButteraugliResidentComparisonDescriptor& descriptor) {

  auto* metal = dynamic_cast<MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal != nullptr && encoder != nullptr) {
    metal->EncodeValidatedResidentComparison(encoder, descriptor);
  }
}

void EncodePreparedMetalButteraugliResidentProfileStage(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const MetalButteraugliResidentComparisonDescriptor& descriptor,
  MetalButteraugliProfileStage stage) {

  auto* metal = dynamic_cast<MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal != nullptr && encoder != nullptr) {
    metal->EncodeValidatedResidentComparisonProfileStage(
      encoder, descriptor, stage);
  }
}

void EncodePreparedMetalButteraugliProfileStage(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const DeviceButteraugliComparisonDescriptor& descriptor,
  MetalButteraugliProfileStage stage) {

  auto* metal = dynamic_cast<MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal != nullptr && encoder != nullptr) {
    metal->EncodeValidatedComparisonProfileStage(encoder, descriptor, stage);
  }
}

Status CreateButteraugliPipelines(
  MTL::Device* device,
  MTL::Library* library,
  ButteraugliPipelines* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Butteraugli pipeline output is null");
  }
  ButteraugliPipelines pipelines;
  const std::array<std::pair<
    std::string_view,
    NS::SharedPtr<MTL::ComputePipelineState>*>, 26> bindings{{
    {"gjxl_butteraugli_copy_f32", &pipelines.copy},
    {"gjxl_butteraugli_expand_f32", &pipelines.expand},
    {"gjxl_butteraugli_subsample2x_f32", &pipelines.subsample},
    {"gjxl_butteraugli_blur5_horizontal_f32", &pipelines.blur5_horizontal},
    {"gjxl_butteraugli_blur5_vertical_f32", &pipelines.blur5_vertical},
    {"gjxl_butteraugli_convolve_transpose_f32", &pipelines.convolution_transpose},
    {"gjxl_butteraugli_opsin_blur5_tiled_f32",
     &pipelines.opsin_blur5_tiled},
    {"gjxl_butteraugli_frequency_low_medium_tiled_f32",
     &pipelines.frequency_low_medium_tiled},
    {"gjxl_butteraugli_frequency_high_convolve_f32",
     &pipelines.frequency_high_convolve},
    {"gjxl_butteraugli_frequency_suppress_x_f32", &pipelines.frequency_suppress_x},
    {"gjxl_butteraugli_frequency_ultra_convolve_f32",
     &pipelines.frequency_ultra_convolve},
    {"gjxl_butteraugli_frequency_ultra_mask_convolve_f32",
     &pipelines.frequency_ultra_mask_convolve},
    {"gjxl_butteraugli_malta_scale_f32", &pipelines.malta_scale},
    {"gjxl_butteraugli_malta_response_f32", &pipelines.malta_response},
    {"gjxl_butteraugli_malta_fused_f32", &pipelines.malta_fused},
    {"gjxl_butteraugli_l2_f32", &pipelines.l2},
    {"gjxl_butteraugli_mask_precompute_f32", &pipelines.mask_precompute},
    {"gjxl_butteraugli_fuzzy_erosion_f32", &pipelines.fuzzy_erosion},
    {"gjxl_butteraugli_masked_ac_f32", &pipelines.masked_ac},
    {"gjxl_butteraugli_final_f32", &pipelines.final},
    {"gjxl_butteraugli_final_masked_ac_f32", &pipelines.final_masked_ac},
    {"gjxl_butteraugli_final_l2_masked_ac_f32",
     &pipelines.final_l2_masked_ac},
    {"gjxl_butteraugli_crop_f32", &pipelines.crop},
    {"gjxl_butteraugli_compose_f32", &pipelines.compose},
    {"gjxl_butteraugli_resident_l2_reduce_f32",
     &pipelines.resident_reduction},
    {"gjxl_butteraugli_reduce_max_f32", &pipelines.maximum_reduction},
  }};
  for (const auto& [name, pipeline] : bindings) {
    Status status = CreatePipeline(device, library, name, pipeline);
    if (!status.ok()) {
      return {
        status.code(),
        std::string("Failed to create required Butteraugli pipeline ") +
          std::string(name) + ": " + std::string(status.message()),
      };
    }
  }
  constexpr NS::UInteger kPlaneThreads = 8 * 8;
  for (size_t index = 0; index + 1 < bindings.size(); ++index) {
    if (bindings[index].second->get()->maxTotalThreadsPerThreadgroup() <
        kPlaneThreads) {
      return Status::Unavailable(
        "Metal cannot launch required Butteraugli plane threadgroups");
    }
  }
  if (pipelines.maximum_reduction->maxTotalThreadsPerThreadgroup() <
      kReductionWidth) {
    return Status::Unavailable(
      "Metal cannot launch the Butteraugli reduction threadgroup");
  }
  if (pipelines.resident_reduction->maxTotalThreadsPerThreadgroup() <
      kReductionWidth) {
    return Status::Unavailable(
      "Metal cannot launch the resident Butteraugli reduction threadgroup");
  }
  if (pipelines.opsin_blur5_tiled->maxTotalThreadsPerThreadgroup() <
      kOpsinBlur5TileWidth * kOpsinBlur5TileHeight) {
    return Status::Unavailable(
      "Metal cannot launch the tiled Butteraugli Opsin threadgroup");
  }
  if (device->maxThreadgroupMemoryLength() <
      kOpsinBlur5ThreadgroupMemoryBytes) {
    return Status::Unavailable(
      "Metal lacks memory for the tiled Butteraugli Opsin threadgroup");
  }
  MTL::ComputePipelineState* low_medium_tiled =
    pipelines.frequency_low_medium_tiled.get();
  if (low_medium_tiled->maxTotalThreadsPerThreadgroup() <
      kLowMediumTileWidth * kLowMediumTileHeight) {
    return Status::Unavailable(
      "Metal cannot launch the tiled Butteraugli low/medium threadgroup");
  }
  if (device->maxThreadgroupMemoryLength() <
      kLowMediumThreadgroupMemoryBytes) {
    return Status::Unavailable(
      "Metal lacks memory for the tiled Butteraugli low/medium threadgroup");
  }
  *out = std::move(pipelines);
  return Status::Ok();
}

Status MetalBackend::Prepare(
  GpuBackend& backend,
  const DeviceButteraugliPrepareDescriptor& descriptor,
  std::unique_ptr<PreparedDeviceButteraugli>* prepared) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Metal Butteraugli prepared output is null");
  }
  prepared->reset();
  if (&backend != this) {
    return Status::InvalidArgument(
      "Metal Butteraugli operation received another backend");
  }
  return PrepareDeviceButteraugliImpl(
    descriptor, gpu_profile_internal::GpuProfilingMode::kDisabled,
    prepared, nullptr);
}

Status MetalBackend::PrepareDeviceButteraugliImpl(
  const DeviceButteraugliPrepareDescriptor& descriptor,
  gpu_profile_internal::GpuProfilingMode mode,
  std::unique_ptr<PreparedDeviceButteraugli>* prepared,
  gpu_profile_internal::GpuExecutionProfile* profile) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Metal Butteraugli prepared output is null");
  }
  prepared->reset();
  const bool profiling =
    mode != gpu_profile_internal::GpuProfilingMode::kDisabled;
  if (profiling != (profile != nullptr)) {
    return Status::InvalidArgument(
      "Metal Butteraugli preparation profile request is invalid");
  }
  Status status = ValidateDeviceButteraugliPrepareDescriptor(
    *this, descriptor);
  if (!status.ok()) return status;
  try {
    auto candidate = std::make_unique<MetalPreparedDeviceButteraugli>(
      *this, descriptor);
    status = candidate->PrepareStorage();
    if (!status.ok()) return status;
    gpu_profile_internal::GpuExecutionProfile candidate_profile;
    status = candidate->PrepareReference(
      mode, profiling ? &candidate_profile : nullptr);
    if (!status.ok()) return status;
    *prepared = std::move(candidate);
    if (profiling) *profile = std::move(candidate_profile);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate Metal Butteraugli prepared state");
  }
}

}  // namespace gjxl::metal_internal

namespace gjxl {

Status ConfigureMetalButteraugliStageCaptureForTest(
  PreparedDeviceButteraugli& prepared,
  MetalButteraugliStage stage) {

  auto* metal =
    dynamic_cast<metal_internal::MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "Butteraugli stage capture requires a Metal prepared state");
  }
  return metal->ConfigureCapture(stage);
}

Status ReadMetalButteraugliStageCaptureForTest(
  PreparedDeviceButteraugli& prepared,
  PlaneF32View output) {

  auto* metal =
    dynamic_cast<metal_internal::MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "Butteraugli stage capture requires a Metal prepared state");
  }
  return metal->ReadCapture(output);
}

Status QueryMetalButteraugliResourceUsageForTest(
  PreparedDeviceButteraugli& prepared,
  MetalButteraugliResourceUsage* usage) {

  if (usage == nullptr) {
    return Status::InvalidArgument(
      "Metal Butteraugli resource-usage output is null");
  }
  auto* metal =
    dynamic_cast<metal_internal::MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "Butteraugli resource usage requires a Metal prepared state");
  }
  *usage = metal->memory_stats();
  return Status::Ok();
}

}  // namespace gjxl

#undef setComputePipelineState
