// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_butteraugli_internal.h"
#include "gpu/cuda/cuda_butteraugli_kernels.h"
#include "gpu/scratch.h"
#include "gpu/submission.h"

namespace gjxl::cuda_internal {
namespace {

constexpr size_t kPlaneAlignment = 64;
constexpr size_t kReductionWidth = 256;
constexpr std::array<float, kCudaButteraugliKernelCount> kBlurSigmas{
    1.2f, 7.15593339443f, 3.22489901262f, 1.56416327805f, 2.7f,
};
constexpr std::array<size_t, kCudaButteraugliKernelCount> kKernelSizes{
    5, 33, 15, 7, 13,
};

[[nodiscard]] bool AddAlignedAllocation(size_t size_bytes,
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

}  // namespace

class CudaPreparedDeviceButteraugli final : public PreparedDeviceButteraugli {
 public:
  CudaPreparedDeviceButteraugli(CudaBackend& backend,
                                DeviceButteraugliPrepareDescriptor descriptor)
      : PreparedDeviceButteraugli(backend, descriptor), backend_(backend) {}

  [[nodiscard]] Status PrepareStorage() {
    const Extent2D requested = extent();
    if (requested.width > std::numeric_limits<uint32_t>::max() ||
        requested.height > std::numeric_limits<uint32_t>::max()) {
      return Status::InvalidArgument(
          "Device Butteraugli extent exceeds CUDA kernel limits");
    }
    expanded_ = requested.width < 8 || requested.height < 8;
    working_extent_ = expanded_
                          ? Extent2D{std::max<size_t>(8, requested.width),
                                     std::max<size_t>(8, requested.height)}
                          : requested;
    xborder_ = requested.width < 8 ? (8 - requested.width) / 2 : 0;
    yborder_ = requested.height < 8 ? (8 - requested.height) / 2 : 0;
    multiscale_ = !expanded_ && requested.width >= 15 && requested.height >= 15;
    if (multiscale_) {
      sub_extent_ = {
          (requested.width + 1) / 2,
          (requested.height + 1) / 2,
      };
    }

    size_t working_area = 0;
    size_t requested_area = 0;
    size_t sub_area = 0;
    if (!working_extent_.try_area(&working_area) ||
        !requested.try_area(&requested_area) ||
        requested_area > std::numeric_limits<uint32_t>::max() ||
        working_area > std::numeric_limits<uint32_t>::max() ||
        working_area > std::numeric_limits<size_t>::max() / sizeof(float) ||
        working_extent_.width > std::numeric_limits<uint32_t>::max() ||
        working_extent_.height > std::numeric_limits<uint32_t>::max()) {
      return Status::InvalidArgument(
          "Device Butteraugli scratch geometry overflows");
    }
    if (multiscale_ &&
        (!sub_extent_.try_area(&sub_area) ||
         sub_area > std::numeric_limits<uint32_t>::max() ||
         sub_area > std::numeric_limits<size_t>::max() / sizeof(float))) {
      return Status::InvalidArgument(
          "Device Butteraugli subscale geometry overflows");
    }
    for (ConstDevicePlaneView plane : reference_linear_rgb().plane) {
      if (plane.row_stride > std::numeric_limits<uint32_t>::max()) {
        return Status::InvalidArgument(
            "Device Butteraugli reference stride exceeds CUDA limits");
      }
    }
    const size_t maximum_area = std::max(working_area, requested_area);
    const size_t plane_blocks = (maximum_area + 255) / 256;
    const size_t partial_count =
        (requested_area + kReductionWidth - 1) / kReductionWidth;
    if (plane_blocks > backend_.state_->maximum_grid_x ||
        partial_count > backend_.state_->maximum_grid_x) {
      return Status::InvalidArgument(
          "Device Butteraugli extent exceeds CUDA launch limits");
    }

    const size_t plane_bytes = working_area * sizeof(float);
    const size_t sub_plane_bytes = sub_area * sizeof(float);
    size_t capacity = 0;
    for (size_t index = 0; index < kCudaButteraugliWorkingPlaneCount; ++index) {
      if (!AddAlignedAllocation(plane_bytes, &capacity)) {
        return Status::InvalidArgument(
            "Device Butteraugli scratch capacity overflows");
      }
    }
    if (multiscale_) {
      for (size_t index = 0; index < kCudaButteraugliPsychoPlaneCount;
           ++index) {
        if (!AddAlignedAllocation(sub_plane_bytes, &capacity)) {
          return Status::InvalidArgument(
              "Device Butteraugli cached subscale capacity overflows");
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
            "Device Butteraugli Gaussian-kernel capacity overflows");
      }
    }

    cached_reference_bytes_ =
        (kCudaButteraugliPsychoPlaneCount + 1) * plane_bytes +
        (multiscale_ ? kCudaButteraugliPsychoPlaneCount * sub_plane_bytes : 0);
    gaussian_kernel_bytes_ = 0;
    for (size_t kernel_size : kKernelSizes) {
      gaussian_kernel_bytes_ += kernel_size * sizeof(float);
    }
    peak_comparison_scratch_bytes_ = (kCudaButteraugliWorkingPlaneCount -
                                      (kCudaButteraugliPsychoPlaneCount + 1)) *
                                         plane_bytes +
                                     2 * partial_count * sizeof(float);

    Status status = scratch_.Prepare(backend_, capacity);
    if (!status.ok()) return status;
    for (DevicePlaneView& plane : planes_) {
      status = scratch_.AllocatePlane(DeviceElementType::kF32, working_extent_,
                                      working_extent_.width, kPlaneAlignment,
                                      &plane);
      if (!status.ok()) return status;
    }
    if (multiscale_) {
      for (DevicePlaneView& plane : reference_sub_) {
        status =
            scratch_.AllocatePlane(DeviceElementType::kF32, sub_extent_,
                                   sub_extent_.width, kPlaneAlignment, &plane);
        if (!status.ok()) return status;
      }
    }
    const Extent2D reduction_extent{partial_count, 1};
    for (DevicePlaneView& reduction : reduction_) {
      status =
          scratch_.AllocatePlane(DeviceElementType::kF32, reduction_extent,
                                 partial_count, kPlaneAlignment, &reduction);
      if (!status.ok()) return status;
    }
    for (size_t index = 0; index < kernels_.size(); ++index) {
      status = scratch_.AllocatePlane(
          DeviceElementType::kF32, {kKernelSizes[index], 1},
          kKernelSizes[index], kPlaneAlignment, &kernels_[index]);
      if (!status.ok()) return status;
      const std::vector<float> kernel = MakeGaussianKernel(kBlurSigmas[index]);
      if (kernel.size() != kKernelSizes[index]) {
        return Status::Internal(
            "Device Butteraugli Gaussian kernel size is inconsistent");
      }
      status = backend_.CopyHostToDevice(*kernels_[index].buffer, kernel.data(),
                                         kernel.size() * sizeof(float),
                                         kernels_[index].offset_bytes);
      if (!status.ok()) return status;
    }
    PopulatePlan();
    return Status::Ok();
  }

  [[nodiscard]] Status PrepareReference() {
    std::unique_ptr<GpuSubmission> submission;
    Status status = backend_.SubmitCompute(
        &CudaPreparedDeviceButteraugli::EncodePreparation, this, &submission);
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Status::Internal(
          "CUDA Butteraugli reference preparation returned no submission");
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

  [[nodiscard]] cudaError_t EncodeComparisonOnStream(
      const DeviceButteraugliComparisonDescriptor& descriptor,
      cudaStream_t stream) {
    std::array<const float*, 3> distorted{};
    std::array<uint32_t, 3> distorted_stride{};
    for (size_t channel = 0; channel < 3; ++channel) {
      distorted[channel] =
          Pointer(descriptor.distorted_linear_rgb.plane[channel]);
      distorted_stride[channel] = static_cast<uint32_t>(
          descriptor.distorted_linear_rgb.plane[channel].row_stride);
    }
    return LaunchCudaButteraugliCompare(
        plan_, distorted, distorted_stride, Pointer(descriptor.distance_map),
        static_cast<uint32_t>(descriptor.distance_map.row_stride),
        Pointer(descriptor.score), stream);
  }

 private:
  struct ComparisonContext {
    CudaPreparedDeviceButteraugli* prepared = nullptr;
    std::array<const float*, 3> distorted{};
    std::array<uint32_t, 3> distorted_stride{};
    float* distance_map = nullptr;
    uint32_t distance_stride = 0;
    float* score = nullptr;
  };

  [[nodiscard]] static float* Pointer(DevicePlaneView view) {
    CudaBuffer* buffer = CudaBackend::AsCudaBuffer(*view.buffer);
    return reinterpret_cast<float*>(static_cast<std::byte*>(buffer->pointer()) +
                                    view.offset_bytes);
  }

  [[nodiscard]] static const float* Pointer(ConstDevicePlaneView view) {
    const CudaBuffer* buffer = CudaBackend::AsCudaBuffer(*view.buffer);
    return reinterpret_cast<const float*>(
        static_cast<const std::byte*>(buffer->pointer()) + view.offset_bytes);
  }

  void PopulatePlan() {
    for (size_t channel = 0; channel < 3; ++channel) {
      plan_.reference[channel] = Pointer(reference_linear_rgb().plane[channel]);
      plan_.reference_stride[channel] = static_cast<uint32_t>(
          reference_linear_rgb().plane[channel].row_stride);
    }
    for (size_t index = 0; index < planes_.size(); ++index) {
      plan_.planes[index] = Pointer(planes_[index]);
    }
    if (multiscale_) {
      for (size_t index = 0; index < reference_sub_.size(); ++index) {
        plan_.reference_sub[index] = Pointer(reference_sub_[index]);
      }
    }
    for (size_t index = 0; index < kernels_.size(); ++index) {
      plan_.kernels[index] = Pointer(kernels_[index]);
    }
    for (size_t index = 0; index < reduction_.size(); ++index) {
      plan_.reduction[index] = Pointer(reduction_[index]);
    }
    plan_.width = static_cast<uint32_t>(extent().width);
    plan_.height = static_cast<uint32_t>(extent().height);
    plan_.working_width = static_cast<uint32_t>(working_extent_.width);
    plan_.working_height = static_cast<uint32_t>(working_extent_.height);
    plan_.sub_width = static_cast<uint32_t>(sub_extent_.width);
    plan_.sub_height = static_cast<uint32_t>(sub_extent_.height);
    plan_.xborder = static_cast<uint32_t>(xborder_);
    plan_.yborder = static_cast<uint32_t>(yborder_);
    plan_.expanded = static_cast<uint32_t>(expanded_);
    plan_.multiscale = static_cast<uint32_t>(multiscale_);
    plan_.hf_asymmetry = options().hf_asymmetry;
    plan_.x_multiplier = options().x_multiplier;
    plan_.intensity_target = options().intensity_target;
  }

  [[nodiscard]] static cudaError_t EncodePreparation(CudaBackend& backend,
                                                     const void* opaque) {
    const auto* prepared =
        static_cast<const CudaPreparedDeviceButteraugli*>(opaque);
    return LaunchCudaButteraugliPrepare(prepared->plan_,
                                        backend.state_->stream);
  }

  [[nodiscard]] static cudaError_t EncodeComparison(CudaBackend& backend,
                                                    const void* opaque) {
    const auto& context = *static_cast<const ComparisonContext*>(opaque);
    return LaunchCudaButteraugliCompare(
        context.prepared->plan_, context.distorted, context.distorted_stride,
        context.distance_map, context.distance_stride, context.score,
        backend.state_->stream);
  }

  [[nodiscard]] Status CompareValidated(
      const DeviceButteraugliComparisonDescriptor& descriptor) override {
    ComparisonContext context;
    context.prepared = this;
    for (size_t channel = 0; channel < 3; ++channel) {
      if (descriptor.distorted_linear_rgb.plane[channel].row_stride >
          std::numeric_limits<uint32_t>::max()) {
        return Status::InvalidArgument(
            "Device Butteraugli distorted stride exceeds CUDA limits");
      }
      context.distorted[channel] =
          Pointer(descriptor.distorted_linear_rgb.plane[channel]);
      context.distorted_stride[channel] = static_cast<uint32_t>(
          descriptor.distorted_linear_rgb.plane[channel].row_stride);
    }
    if (descriptor.distance_map.row_stride >
        std::numeric_limits<uint32_t>::max()) {
      return Status::InvalidArgument(
          "Device Butteraugli output stride exceeds CUDA limits");
    }
    context.distance_map = Pointer(descriptor.distance_map);
    context.distance_stride =
        static_cast<uint32_t>(descriptor.distance_map.row_stride);
    context.score = Pointer(descriptor.score);

    std::unique_ptr<GpuSubmission> submission;
    Status status =
        backend_.SubmitCompute(&CudaPreparedDeviceButteraugli::EncodeComparison,
                               &context, &submission);
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Status::Internal(
          "CUDA Butteraugli comparison returned no submission");
    }
    return submission->Wait();
  }

  CudaBackend& backend_;
  DeviceScratchArena scratch_;
  std::array<DevicePlaneView, kCudaButteraugliWorkingPlaneCount> planes_{};
  std::array<DevicePlaneView, kCudaButteraugliPsychoPlaneCount>
      reference_sub_{};
  std::array<DevicePlaneView, kCudaButteraugliKernelCount> kernels_{};
  std::array<DevicePlaneView, 2> reduction_{};
  CudaButteraugliPlan plan_;
  Extent2D working_extent_;
  Extent2D sub_extent_;
  size_t xborder_ = 0;
  size_t yborder_ = 0;
  bool expanded_ = false;
  bool multiscale_ = false;
  size_t cached_reference_bytes_ = 0;
  size_t gaussian_kernel_bytes_ = 0;
  size_t peak_comparison_scratch_bytes_ = 0;
};

Status CudaBackend::Prepare(
    GpuBackend& backend, const DeviceButteraugliPrepareDescriptor& descriptor,
    std::unique_ptr<PreparedDeviceButteraugli>* prepared) {
  if (prepared == nullptr) {
    return Status::InvalidArgument("CUDA Butteraugli prepared output is null");
  }
  prepared->reset();
  if (&backend != this) {
    return Status::InvalidArgument(
        "CUDA Butteraugli operation received another backend");
  }
  Status status =
      ValidateDeviceButteraugliPrepareDescriptor(backend, descriptor);
  if (!status.ok()) return status;
  try {
    auto candidate =
        std::make_unique<CudaPreparedDeviceButteraugli>(*this, descriptor);
    status = candidate->PrepareStorage();
    if (!status.ok()) return status;
    status = candidate->PrepareReference();
    if (!status.ok()) return status;
    *prepared = std::move(candidate);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
        "Unable to allocate CUDA Butteraugli prepared state");
  }
}

cudaError_t EncodePreparedCudaButteraugli(
    PreparedDeviceButteraugli& prepared,
    const DeviceButteraugliComparisonDescriptor& descriptor,
    cudaStream_t stream) {
  auto* cuda = dynamic_cast<CudaPreparedDeviceButteraugli*>(&prepared);
  return cuda == nullptr ? cudaErrorInvalidResourceHandle
                         : cuda->EncodeComparisonOnStream(descriptor, stream);
}

}  // namespace gjxl::cuda_internal
