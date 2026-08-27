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

namespace gjxl::metal_internal {
namespace {

constexpr size_t kPlaneAlignment = 64;
constexpr size_t kWorkingPlaneCount = 38;
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
constexpr size_t kImage = 20;
constexpr size_t kAc = 26;
constexpr size_t kDc = 29;
constexpr size_t kWork = 32;
constexpr size_t kFinalStaging = 37;
constexpr size_t kPsychoPlaneCount = 10;

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
  uint32_t blurred_stride;
  uint32_t output_stride;
  float intensity_target;
};

struct FrequencyParams {
  uint32_t width;
  uint32_t height;
  uint32_t xyb_stride;
  uint32_t psycho_stride;
};

struct FrequencyChannelParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t blurred_stride;
  uint32_t output_stride;
  uint32_t channel;
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
  uint32_t low_frequency;
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

static_assert(std::is_trivially_copyable_v<PlaneParams>);
static_assert(sizeof(PlaneParams) == 16);
static_assert(sizeof(ExpandParams) == 32);
static_assert(sizeof(SubsampleParams) == 24);
static_assert(sizeof(ConvolutionParams) == 20);
static_assert(sizeof(OpsinParams) == 32);
static_assert(sizeof(FrequencyParams) == 16);
static_assert(sizeof(FrequencyChannelParams) == 24);
static_assert(sizeof(MaltaScaleParams) == 36);
static_assert(sizeof(MaltaResponseParams) == 20);
static_assert(sizeof(DifferenceParams) == 24);
static_assert(sizeof(FinalParams) == 20);
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
      for (size_t index = 0; index < reference_sub_.size(); ++index) {
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
      kPsychoPlaneCount * plane_bytes +
      (multiscale_ ? kPsychoPlaneCount * sub_plane_bytes : 0);
    gaussian_kernel_bytes_ = 0;
    for (size_t kernel_size : kKernelSizes) {
      gaussian_kernel_bytes_ += kernel_size * sizeof(float);
    }
    peak_comparison_scratch_bytes_ =
      (kWorkingPlaneCount - kPsychoPlaneCount) * plane_bytes +
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

  [[nodiscard]] Status PrepareReference() {
    const PreparationContext context{this};
    std::unique_ptr<GpuSubmission> submission;
    Status status = metal_.SubmitCompute(
      "gjxl Butteraugli reference preparation",
      &MetalPreparedDeviceButteraugli::EncodePreparationSubmission,
      &context,
      &submission);
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Status::Internal(
        "Metal Butteraugli preparation submission is null");
    }
    return submission->Wait();
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

  void EncodeValidatedComparison(
    MTL::ComputeCommandEncoder* encoder,
    const DeviceButteraugliComparisonDescriptor& descriptor) {

    capture_ready_ = false;
    EncodeComparison(encoder, descriptor);
  }

private:
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

  void EncodeClear(
    MTL::ComputeCommandEncoder* encoder,
    DevicePlaneView output,
    Extent2D plane_extent) {

    const PlaneParams params{
      static_cast<uint32_t>(plane_extent.width),
      static_cast<uint32_t>(plane_extent.height),
      0,
      static_cast<uint32_t>(output.row_stride),
    };
    encoder->setComputePipelineState(metal_.butteraugli_pipelines_.clear.get());
    Bind(encoder, Handle(metal_, output), output.offset_bytes, 0);
    encoder->setBytes(&params, sizeof(params), 1);
    metal_.DispatchPlane(encoder, plane_extent);
  }

  void EncodeAdd(
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
    encoder->setComputePipelineState(metal_.butteraugli_pipelines_.add.get());
    Bind(encoder, Handle(metal_, input), input.offset_bytes, 0);
    Bind(encoder, Handle(metal_, output), output.offset_bytes, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    metal_.DispatchPlane(encoder, plane_extent);
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

  void EncodePsychoImage(
    MTL::ComputeCommandEncoder* encoder,
    ConstDeviceImage3View input,
    const PsychoPlanes& psycho,
    Extent2D scale_extent,
    bool capture_reference) {

    for (size_t channel = 0; channel < 3; ++channel) {
      EncodeBlur(
        encoder, input.plane[channel], 0, kWork,
        Plane(kImage + 3 + channel, scale_extent), scale_extent);
    }

    const OpsinParams opsin_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(input.plane[0].row_stride),
      static_cast<uint32_t>(input.plane[1].row_stride),
      static_cast<uint32_t>(input.plane[2].row_stride),
      static_cast<uint32_t>(working_extent_.width),
      static_cast<uint32_t>(working_extent_.width),
      options().intensity_target,
    };
    encoder->setComputePipelineState(metal_.butteraugli_pipelines_.opsin.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      Bind(encoder, Handle(metal_, input.plane[channel]),
           input.plane[channel].offset_bytes, channel);
      DevicePlaneView blurred = Plane(kImage + 3 + channel, scale_extent);
      Bind(encoder, Handle(metal_, blurred), blurred.offset_bytes, 3 + channel);
      DevicePlaneView xyb = Plane(kImage + channel, scale_extent);
      Bind(encoder, Handle(metal_, xyb), xyb.offset_bytes, 6 + channel);
    }
    encoder->setBytes(&opsin_params, sizeof(opsin_params), 9);
    metal_.DispatchPlane(encoder, scale_extent);
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

    for (size_t channel = 0; channel < 3; ++channel) {
      EncodeBlur(
        encoder,
        AsConst(Plane(kImage + channel, scale_extent)),
        1,
        kWork,
        psycho[channel],
        scale_extent);
    }
    const FrequencyParams frequency_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(working_extent_.width),
      static_cast<uint32_t>(psycho[0].row_stride),
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.frequency_low_medium.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      DevicePlaneView xyb = Plane(kImage + channel, scale_extent);
      DevicePlaneView low = psycho[channel];
      DevicePlaneView medium = psycho[3 + channel];
      Bind(encoder, Handle(metal_, xyb), xyb.offset_bytes, channel);
      Bind(encoder, Handle(metal_, low), low.offset_bytes, 3 + channel);
      Bind(encoder, Handle(metal_, medium), medium.offset_bytes, 6 + channel);
    }
    encoder->setBytes(&frequency_params, sizeof(frequency_params), 9);
    metal_.DispatchPlane(encoder, scale_extent);
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
      DevicePlaneView blurred = Plane(kWork + 1, scale_extent);
      DevicePlaneView high = psycho[6 + channel];
      EncodeBlur(encoder, AsConst(medium), 2, kWork, blurred, scale_extent);
      const FrequencyChannelParams channel_params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(medium.row_stride),
        static_cast<uint32_t>(blurred.row_stride),
        static_cast<uint32_t>(high.row_stride),
        static_cast<uint32_t>(channel),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.frequency_high.get());
      Bind(encoder, Handle(metal_, medium), medium.offset_bytes, 0);
      Bind(encoder, Handle(metal_, blurred), blurred.offset_bytes, 1);
      Bind(encoder, Handle(metal_, high), high.offset_bytes, 2);
      encoder->setBytes(&channel_params, sizeof(channel_params), 3);
      metal_.DispatchPlane(encoder, scale_extent);
    }
    DevicePlaneView medium_b = psycho[5];
    DevicePlaneView blurred_b = Plane(kWork + 1, scale_extent);
    EncodeBlur(encoder, AsConst(medium_b), 2, kWork, blurred_b, scale_extent);
    EncodeCopy(encoder, AsConst(blurred_b), medium_b, scale_extent);

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
      DevicePlaneView blurred = Plane(kWork + 1, scale_extent);
      DevicePlaneView ultra = psycho[8 + channel];
      EncodeBlur(encoder, AsConst(high), 3, kWork, blurred, scale_extent);
      const FrequencyChannelParams channel_params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(high.row_stride),
        static_cast<uint32_t>(blurred.row_stride),
        static_cast<uint32_t>(ultra.row_stride),
        static_cast<uint32_t>(channel),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.frequency_ultra.get());
      Bind(encoder, Handle(metal_, high), high.offset_bytes, 0);
      Bind(encoder, Handle(metal_, blurred), blurred.offset_bytes, 1);
      Bind(encoder, Handle(metal_, ultra), ultra.offset_bytes, 2);
      encoder->setBytes(&channel_params, sizeof(channel_params), 3);
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
    Extent2D scale_extent,
    DevicePlaneView output) {

    for (size_t channel = 0; channel < 3; ++channel) {
      EncodeClear(encoder, Plane(kAc + channel, scale_extent), scale_extent);
      EncodeClear(encoder, Plane(kDc + channel, scale_extent), scale_extent);
    }

    const float asymmetry = options().hf_asymmetry;
    const float sqrt_asymmetry = std::sqrt(asymmetry);
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

      DevicePlaneView response = Plane(kWork + 1, scale_extent);
      const MaltaResponseParams response_params{
        static_cast<uint32_t>(scale_extent.width),
        static_cast<uint32_t>(scale_extent.height),
        static_cast<uint32_t>(scaled.row_stride),
        static_cast<uint32_t>(response.row_stride),
        static_cast<uint32_t>(low_frequency),
      };
      encoder->setComputePipelineState(
        metal_.butteraugli_pipelines_.malta_response.get());
      Bind(encoder, Handle(metal_, scaled), scaled.offset_bytes, 0);
      Bind(encoder, Handle(metal_, response), response.offset_bytes, 1);
      encoder->setBytes(&response_params, sizeof(response_params), 2);
      metal_.DispatchPlane(encoder, scale_extent);
      MaybeCapture(
        encoder,
        static_cast<MetalButteraugliStage>(
          static_cast<size_t>(
            MetalButteraugliStage::kMaltaMediumFrequencyY) + stage_index),
        AsConst(response),
        scale_extent);
      const size_t channel = stage_index % 2 == 0 ? 1 : 0;
      EncodeAdd(
        encoder, AsConst(response), Plane(kAc + channel, scale_extent),
        scale_extent);
    }

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

    const auto encode_mask_precompute =
      [&](const PsychoPlanes& psycho, DevicePlaneView destination) {
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
      };

    DevicePlaneView precomputed = Plane(kWork, scale_extent);
    DevicePlaneView mask_blurred_reference = Plane(kWork + 2, scale_extent);
    DevicePlaneView mask = Plane(kWork + 3, scale_extent);
    DevicePlaneView mask_blurred_distorted = Plane(kWork + 4, scale_extent);
    encode_mask_precompute(reference, precomputed);
    EncodeBlur(
      encoder, AsConst(precomputed), 4, kWork + 1,
      mask_blurred_reference, scale_extent);

    const PlaneParams fuzzy_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(mask_blurred_reference.row_stride),
      static_cast<uint32_t>(mask.row_stride),
    };
    encoder->setComputePipelineState(
      metal_.butteraugli_pipelines_.fuzzy_erosion.get());
    Bind(encoder, Handle(metal_, mask_blurred_reference),
         mask_blurred_reference.offset_bytes, 0);
    Bind(encoder, Handle(metal_, mask), mask.offset_bytes, 1);
    encoder->setBytes(&fuzzy_params, sizeof(fuzzy_params), 2);
    metal_.DispatchPlane(encoder, scale_extent);
    MaybeCapture(
      encoder, MetalButteraugliStage::kMask, AsConst(mask), scale_extent);

    encode_mask_precompute(distorted, precomputed);
    EncodeBlur(
      encoder, AsConst(precomputed), 4, kWork + 1,
      mask_blurred_distorted, scale_extent);
    DevicePlaneView ac_y = Plane(kAc + 1, scale_extent);
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

    const FinalParams final_params{
      static_cast<uint32_t>(scale_extent.width),
      static_cast<uint32_t>(scale_extent.height),
      static_cast<uint32_t>(working_extent_.width),
      static_cast<uint32_t>(output.row_stride),
      options().x_multiplier,
    };
    encoder->setComputePipelineState(metal_.butteraugli_pipelines_.final.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      DevicePlaneView dc = Plane(kDc + channel, scale_extent);
      DevicePlaneView ac = Plane(kAc + channel, scale_extent);
      Bind(encoder, Handle(metal_, dc), dc.offset_bytes, channel);
      Bind(encoder, Handle(metal_, ac), ac.offset_bytes, 3 + channel);
    }
    Bind(encoder, Handle(metal_, mask), mask.offset_bytes, 6);
    Bind(encoder, Handle(metal_, output), output.offset_bytes, 7);
    encoder->setBytes(&final_params, sizeof(final_params), 8);
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
      DevicePlaneView output = Plane(kImage + channel, expanded);
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
      DevicePlaneView output = Plane(kImage + channel, subsampled);
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

  [[nodiscard]] ConstDeviceImage3View ImageSlots(
    Extent2D image_extent) const noexcept {

    return {{{
      AsConst(Plane(kImage, image_extent)),
      AsConst(Plane(kImage + 1, image_extent)),
      AsConst(Plane(kImage + 2, image_extent)),
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
    encoder->dispatchThreadgroups(
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
        encoder, ImageSlots(working_extent_), reference_main,
        working_extent_, false);
      return;
    }

    EncodePsychoImage(
      encoder, reference_linear_rgb(), reference_main, requested, false);
    if (multiscale_) {
      EncodeSubsample(
        encoder, reference_linear_rgb(), requested, sub_extent_);
      EncodePsychoImage(
        encoder, ImageSlots(sub_extent_), ReferenceSubSlots(), sub_extent_,
        false);
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
        encoder, reference_linear_rgb(), distorted_main, requested, true);
    }
    if (expanded_) {
      EncodeExpand(
        encoder, descriptor.distorted_linear_rgb, requested, working_extent_);
      EncodePsychoImage(
        encoder, ImageSlots(working_extent_), distorted_main,
        working_extent_, false);
      DevicePlaneView staging = Plane(kFinalStaging, working_extent_);
      EncodeDifference(
        encoder, reference_main, distorted_main, working_extent_, staging);
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
        encoder, reference_main, distorted_main, requested,
        descriptor.distance_map);

      if (multiscale_) {
        EncodeSubsample(
          encoder, descriptor.distorted_linear_rgb, requested, sub_extent_);
        EncodePsychoImage(
          encoder, ImageSlots(sub_extent_), distorted_main, sub_extent_,
          false);
        DevicePlaneView sub_map = Plane(kFinalStaging, sub_extent_);
        EncodeDifference(
          encoder, ReferenceSubSlots(), distorted_main, sub_extent_, sub_map);
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

  MetalBackend& metal_;
  DeviceScratchArena scratch_;
  std::array<DevicePlaneView, kWorkingPlaneCount> planes_;
  PsychoPlanes reference_sub_;
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

void EncodePreparedMetalButteraugli(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const DeviceButteraugliComparisonDescriptor& descriptor) {

  auto* metal = dynamic_cast<MetalPreparedDeviceButteraugli*>(&prepared);
  if (metal != nullptr && encoder != nullptr) {
    metal->EncodeValidatedComparison(encoder, descriptor);
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
    NS::SharedPtr<MTL::ComputePipelineState>*>, 23> bindings{{
    {"gjxl_butteraugli_copy_f32", &pipelines.copy},
    {"gjxl_butteraugli_clear_f32", &pipelines.clear},
    {"gjxl_butteraugli_add_f32", &pipelines.add},
    {"gjxl_butteraugli_expand_f32", &pipelines.expand},
    {"gjxl_butteraugli_subsample2x_f32", &pipelines.subsample},
    {"gjxl_butteraugli_blur5_horizontal_f32", &pipelines.blur5_horizontal},
    {"gjxl_butteraugli_blur5_vertical_f32", &pipelines.blur5_vertical},
    {"gjxl_butteraugli_convolve_transpose_f32", &pipelines.convolution_transpose},
    {"gjxl_butteraugli_opsin_f32", &pipelines.opsin},
    {"gjxl_butteraugli_frequency_low_medium_f32", &pipelines.frequency_low_medium},
    {"gjxl_butteraugli_frequency_high_f32", &pipelines.frequency_high},
    {"gjxl_butteraugli_frequency_suppress_x_f32", &pipelines.frequency_suppress_x},
    {"gjxl_butteraugli_frequency_ultra_f32", &pipelines.frequency_ultra},
    {"gjxl_butteraugli_malta_scale_f32", &pipelines.malta_scale},
    {"gjxl_butteraugli_malta_response_f32", &pipelines.malta_response},
    {"gjxl_butteraugli_l2_f32", &pipelines.l2},
    {"gjxl_butteraugli_mask_precompute_f32", &pipelines.mask_precompute},
    {"gjxl_butteraugli_fuzzy_erosion_f32", &pipelines.fuzzy_erosion},
    {"gjxl_butteraugli_masked_ac_f32", &pipelines.masked_ac},
    {"gjxl_butteraugli_final_f32", &pipelines.final},
    {"gjxl_butteraugli_crop_f32", &pipelines.crop},
    {"gjxl_butteraugli_compose_f32", &pipelines.compose},
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
  Status status = ValidateDeviceButteraugliPrepareDescriptor(
    backend, descriptor);
  if (!status.ok()) return status;
  try {
    auto candidate = std::make_unique<MetalPreparedDeviceButteraugli>(
      *this, descriptor);
    status = candidate->PrepareStorage();
    if (!status.ok()) return status;
    status = candidate->PrepareReference();
    if (!status.ok()) return status;
    *prepared = std::move(candidate);
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
