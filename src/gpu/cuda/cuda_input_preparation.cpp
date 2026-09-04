// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

#include "core/frame_geometry.h"
#include "gpu/cuda/cuda_aq_exact_kernels.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/scratch.h"
#include "gpu/submission.h"

namespace gjxl::cuda_internal {
namespace {

constexpr size_t kArenaAlignment = 256;

Status AddPlaneBytes(
  Extent2D extent, size_t* bytes) {
  size_t count = 0;
  if (bytes == nullptr || !extent.try_area(&count) ||
      count > std::numeric_limits<size_t>::max() / sizeof(float) ||
      *bytes > std::numeric_limits<size_t>::max() - (kArenaAlignment - 1)) {
    return Status::InvalidArgument(
      "CUDA input-preparation plane dimensions are too large");
  }
  const size_t aligned =
    (*bytes + kArenaAlignment - 1) & ~(kArenaAlignment - 1);
  const size_t plane_bytes = count * sizeof(float);
  if (aligned > std::numeric_limits<size_t>::max() - plane_bytes) {
    return Status::InvalidArgument(
      "CUDA input-preparation arena size overflows");
  }
  *bytes = aligned + plane_bytes;
  return Status::Ok();
}

template <typename T>
T* Pointer(
  DevicePlaneView view) {
  auto* buffer = dynamic_cast<CudaBuffer*>(view.buffer);
  return reinterpret_cast<T*>(
    static_cast<std::byte*>(buffer->pointer()) + view.offset_bytes);
}

}  // namespace

class CudaPreparedLinearRgbOpsin final : public PreparedGpuLinearRgbOpsin {
 public:
  explicit CudaPreparedLinearRgbOpsin(
    CudaBackend& backend)
      : backend_(&backend) {}

  Status Prepare(
    ConstImage3FView linear_rgb, LinearRgbOpsinPreparationOptions options) {
    if (!linear_rgb.valid() || !std::isfinite(options.intensity_target) ||
        options.intensity_target <= 0.0f) {
      return Status::InvalidArgument(
        "CUDA linear-RGB input preparation is invalid");
    }
    FrameGeometry geometry;
    Status status = FrameGeometry::Create(linear_rgb.extent(), &geometry);
    if (!status.ok()) return status;
    if (options.padded_extent != geometry.padded_frame() ||
        linear_rgb.width() > std::numeric_limits<uint32_t>::max() ||
        linear_rgb.height() > std::numeric_limits<uint32_t>::max() ||
        options.padded_extent.width > std::numeric_limits<uint32_t>::max() ||
        options.padded_extent.height > std::numeric_limits<uint32_t>::max()) {
      return Status::InvalidArgument(
        "CUDA input-preparation geometry is invalid");
    }
    size_t source_count = 0;
    size_t padded_count = 0;
    if (!linear_rgb.extent().try_area(&source_count) ||
        !options.padded_extent.try_area(&padded_count) ||
        (padded_count + 255) / 256 > backend_->state_->maximum_grid_x ||
        (source_count + 255) / 256 > backend_->state_->maximum_grid_x) {
      return Status::InvalidArgument(
        "CUDA input-preparation geometry exceeds device limits");
    }

    size_t capacity = 0;
    for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
      status = AddPlaneBytes(linear_rgb.extent(), &capacity);
    }
    for (size_t channel = 0; channel < 3 && status.ok(); ++channel) {
      status = AddPlaneBytes(options.padded_extent, &capacity);
    }
    if (status.ok()) status = AddPlaneBytes({4, 1}, &capacity);
    if (!status.ok()) return status;
    status = arena_.Prepare(*backend_, capacity);
    if (!status.ok()) return status;
    for (DevicePlaneView& plane : original_) {
      status = arena_.AllocatePlane(DeviceElementType::kF32,
        linear_rgb.extent(),
        linear_rgb.width(),
        kArenaAlignment,
        &plane);
      if (!status.ok()) return status;
    }
    for (DevicePlaneView& plane : coding_) {
      status = arena_.AllocatePlane(DeviceElementType::kF32,
        options.padded_extent,
        options.padded_extent.width,
        kArenaAlignment,
        &plane);
      if (!status.ok()) return status;
    }
    status = arena_.AllocatePlane(
      DeviceElementType::kI32, {4, 1}, 4, kArenaAlignment, &results_);
    if (!status.ok()) return status;

    Context context{
      .owner = this,
      .linear_rgb = linear_rgb,
      .params =
        {
          .source_width = static_cast<uint32_t>(linear_rgb.width()),
          .source_height = static_cast<uint32_t>(linear_rgb.height()),
          .source_stride = static_cast<uint32_t>(linear_rgb.width()),
          .padded_width = static_cast<uint32_t>(options.padded_extent.width),
          .padded_height = static_cast<uint32_t>(options.padded_extent.height),
          .output_stride = static_cast<uint32_t>(options.padded_extent.width),
          .intensity_target = options.intensity_target,
          .compute_matrix_scale_stats =
            options.compute_matrix_scale_stats ? 1u : 0u,
        },
    };
    std::unique_ptr<GpuSubmission> submission;
    status = backend_->SubmitCompute(&Encode, &context, &submission);
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Status::Internal("CUDA input preparation returned no submission");
    }
    status = submission->Wait();
    if (!status.ok()) return status;
    if (context.host_results[3] != 0) {
      return Status::InvalidArgument(
        "CUDA input preparation encountered a non-finite sample");
    }
    for (size_t index = 0; index < matrix_scale_stats_.size(); ++index) {
      matrix_scale_stats_[index] =
        std::bit_cast<float>(context.host_results[index]);
    }
    return Status::Ok();
  }

  ConstDeviceImage3View original_linear_rgb() const noexcept override {
    return ConstImage(original_);
  }

  ConstDeviceImage3View coding_opsin() const noexcept override {
    return ConstImage(coding_);
  }

  std::array<float, 3> matrix_scale_stats() const noexcept override {
    return matrix_scale_stats_;
  }

 private:
  struct Context {
    CudaPreparedLinearRgbOpsin* owner = nullptr;
    ConstImage3FView linear_rgb;
    CudaLinearRgbToOpsinParams params;
    std::array<uint32_t, 4> host_results{};
  };

  static cudaError_t Encode(
    CudaBackend& backend, const void* opaque) {
    auto& context = *static_cast<Context*>(const_cast<void*>(opaque));
    CudaPreparedLinearRgbOpsin& self = *context.owner;
    cudaError_t status = cudaSuccess;
    for (size_t channel = 0; channel < 3 && status == cudaSuccess; ++channel) {
      const ConstPlaneF32View source = context.linear_rgb.plane[channel];
      status = cudaMemcpy2DAsync(Pointer<float>(self.original_[channel]),
        self.original_[channel].row_stride * sizeof(float),
        source.data,
        source.stride * sizeof(float),
        source.extent.width * sizeof(float),
        source.extent.height,
        cudaMemcpyHostToDevice,
        backend.state_->stream);
    }
    if (status == cudaSuccess) {
      status = cudaMemsetAsync(Pointer<unsigned int>(self.results_),
        0,
        4 * sizeof(uint32_t),
        backend.state_->stream);
    }
    if (status == cudaSuccess) {
      status =
        LaunchCudaLinearRgbToOpsin({Pointer<const float>(self.original_[0]),
                                     Pointer<const float>(self.original_[1]),
                                     Pointer<const float>(self.original_[2])},
          {Pointer<float>(self.coding_[0]),
            Pointer<float>(self.coding_[1]),
            Pointer<float>(self.coding_[2])},
          Pointer<unsigned int>(self.results_),
          Pointer<unsigned int>(self.results_) + 3,
          context.params,
          backend.state_->stream);
    }
    if (status == cudaSuccess) {
      status = cudaMemcpyAsync(context.host_results.data(),
        Pointer<unsigned int>(self.results_),
        4 * sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        backend.state_->stream);
    }
    return status;
  }

  static ConstDeviceImage3View ConstImage(
    const std::array<DevicePlaneView, 3>& image) noexcept {
    return {{{image[0], image[1], image[2]}}};
  }

  CudaBackend* backend_ = nullptr;
  DeviceScratchArena arena_;
  std::array<DevicePlaneView, 3> original_{};
  std::array<DevicePlaneView, 3> coding_{};
  DevicePlaneView results_{};
  std::array<float, 3> matrix_scale_stats_{};
};

Status CudaBackend::PrepareLinearRgbOpsin(
  ConstImage3FView linear_rgb,
  LinearRgbOpsinPreparationOptions options,
  std::unique_ptr<PreparedGpuLinearRgbOpsin>* prepared) {
  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "CUDA prepared input output pointer is null");
  }
  prepared->reset();
  try {
    auto candidate = std::make_unique<CudaPreparedLinearRgbOpsin>(*this);
    Status status = candidate->Prepare(linear_rgb, options);
    if (!status.ok()) return status;
    *prepared = std::move(candidate);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Unable to allocate CUDA prepared input owner");
  }
}

}  // namespace gjxl::cuda_internal
