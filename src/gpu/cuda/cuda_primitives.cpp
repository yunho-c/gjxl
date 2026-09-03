// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_backend_internal.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "gpu/cuda/cuda_kernels.h"

namespace gjxl::cuda_internal {
namespace {

constexpr size_t kReductionThreadCount = 256;

template <typename T>
  requires (!std::is_const_v<T>)
T* OffsetPointer(T* pointer, size_t offset_bytes) {
  return reinterpret_cast<T*>(
    reinterpret_cast<std::byte*>(pointer) + offset_bytes);
}

template <typename T>
const T* OffsetPointer(const T* pointer, size_t offset_bytes) {
  return reinterpret_cast<const T*>(
    reinterpret_cast<const std::byte*>(pointer) + offset_bytes);
}

}  // namespace

Status CudaBackend::ResolvePlane(
  ConstDevicePlaneView view,
  ResolvedConstPlane* out) const {
  if (out == nullptr || view.element_type != DeviceElementType::kF32) {
    return Status::InvalidArgument(
      "CUDA primitive requires a float32 input plane");
  }
  DeviceMemoryRange range;
  Status status = ComputeDevicePlaneRange(view, id(), &range);
  if (!status.ok()) return status;
  const CudaBuffer* buffer = AsCudaBuffer(*view.buffer);
  if (buffer == nullptr || buffer->state() != state_.get()) {
    return Status::InvalidArgument(
      "Primitive input is not owned by this CUDA backend");
  }
  if (view.extent.width > std::numeric_limits<uint32_t>::max() ||
      view.extent.height > std::numeric_limits<uint32_t>::max() ||
      view.row_stride > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument(
      "CUDA primitive input geometry exceeds kernel limits");
  }
  size_t elements = 0;
  if (!view.extent.try_area(&elements) ||
      (elements + 255) / 256 > state_->maximum_grid_x) {
    return Status::InvalidArgument(
      "CUDA primitive input exceeds grid limits");
  }
  *out = {view, range, buffer};
  return Status::Ok();
}

Status CudaBackend::ResolvePlane(
  DevicePlaneView view,
  ResolvedPlane* out) const {
  if (out == nullptr) {
    return Status::InvalidArgument(
      "CUDA primitive output resolver is null");
  }
  ResolvedConstPlane resolved;
  Status status = ResolvePlane(
    static_cast<ConstDevicePlaneView>(view), &resolved);
  if (!status.ok()) return status;
  CudaBuffer* buffer = AsCudaBuffer(*view.buffer);
  if (buffer == nullptr) {
    return Status::InvalidArgument(
      "Primitive output is not a CUDA buffer");
  }
  *out = {view, resolved.range, buffer};
  return Status::Ok();
}

bool CudaBackend::SamePlaneLayout(
  ConstDevicePlaneView left,
  ConstDevicePlaneView right) noexcept {
  return left.buffer == right.buffer &&
    left.offset_bytes == right.offset_bytes &&
    left.element_type == right.element_type &&
    left.extent == right.extent &&
    left.row_stride == right.row_stride;
}

Status CudaBackend::RejectOverlap(
  DeviceMemoryRange left,
  DeviceMemoryRange right,
  std::string_view message) {
  return DeviceRangesOverlap(left, right)
    ? Status::InvalidArgument(std::string(message))
    : Status::Ok();
}

Status CudaBackend::ValidatePrimitive(
  const PointwiseAffineCommand& command) const {
  ResolvedConstPlane input;
  ResolvedPlane output;
  Status status = ResolvePlane(command.input, &input);
  if (!status.ok()) return status;
  status = ResolvePlane(command.output, &output);
  if (!status.ok()) return status;
  if (input.view.extent != output.view.extent ||
      !std::isfinite(command.scale) || !std::isfinite(command.bias)) {
    return Status::InvalidArgument(
      "Pointwise affine geometry or parameters are invalid");
  }
  if (DeviceRangesOverlap(input.range, output.range) &&
      !SamePlaneLayout(input.view, output.view)) {
    return Status::InvalidArgument(
      "Pointwise affine planes partially overlap");
  }
  return Status::Ok();
}

Status CudaBackend::ValidatePrimitive(
  const SeparableConvolutionCommand& command) const {
  ResolvedConstPlane input;
  ResolvedConstPlane kernel;
  ResolvedPlane intermediate;
  ResolvedPlane output;
  Status status = ResolvePlane(command.input, &input);
  if (!status.ok()) return status;
  status = ResolvePlane(command.kernel, &kernel);
  if (!status.ok()) return status;
  status = ResolvePlane(command.intermediate, &intermediate);
  if (!status.ok()) return status;
  status = ResolvePlane(command.output, &output);
  if (!status.ok()) return status;
  if (input.view.extent != intermediate.view.extent ||
      input.view.extent != output.view.extent ||
      kernel.view.extent.height != 1 ||
      kernel.view.extent.width > 33 ||
      (kernel.view.extent.width & 1u) == 0) {
    return Status::InvalidArgument(
      "Separable convolution geometry or kernel size is invalid");
  }
  if (DeviceRangesOverlap(input.range, output.range) &&
      !SamePlaneLayout(input.view, output.view)) {
    return Status::InvalidArgument(
      "Separable convolution input and output partially overlap");
  }
  const std::array<std::pair<DeviceMemoryRange, DeviceMemoryRange>, 5>
    forbidden{{
      {intermediate.range, input.range},
      {intermediate.range, kernel.range},
      {intermediate.range, output.range},
      {kernel.range, input.range},
      {kernel.range, output.range},
    }};
  for (const auto& [left, right] : forbidden) {
    status = RejectOverlap(
      left, right, "Separable convolution has unsupported aliasing");
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status CudaBackend::ValidatePrimitive(
  const Symmetric5ConvolutionCommand& command) const {
  ResolvedConstPlane input;
  ResolvedPlane output;
  Status status = ResolvePlane(command.input, &input);
  if (!status.ok()) return status;
  status = ResolvePlane(command.output, &output);
  if (!status.ok()) return status;
  const std::array weights{
    command.weights.distance0,
    command.weights.distance1,
    command.weights.distance2,
    command.weights.distance4,
    command.weights.distance8,
    command.weights.distance5,
  };
  if (input.view.extent != output.view.extent) {
    return Status::InvalidArgument(
      "Symmetric5 convolution planes have different geometry");
  }
  for (float weight : weights) {
    if (!std::isfinite(weight)) {
      return Status::InvalidArgument(
        "Symmetric5 convolution weights must be finite");
    }
  }
  return RejectOverlap(
    input.range, output.range,
    "Symmetric5 convolution planes overlap");
}

Status CudaBackend::ValidatePrimitive(
  const MaximumReductionCommand& command) const {
  ResolvedConstPlane input;
  ResolvedPlane scratch_a;
  ResolvedPlane scratch_b;
  ResolvedPlane output;
  Status status = ResolvePlane(command.input, &input);
  if (!status.ok()) return status;
  status = ResolvePlane(command.scratch_a, &scratch_a);
  if (!status.ok()) return status;
  status = ResolvePlane(command.scratch_b, &scratch_b);
  if (!status.ok()) return status;
  status = ResolvePlane(command.output, &output);
  if (!status.ok()) return status;
  size_t input_count = 0;
  if (!input.view.extent.try_area(&input_count) ||
      input_count > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument("Maximum reduction input is too large");
  }
  const size_t partial_count =
    (input_count + kReductionThreadCount - 1) / kReductionThreadCount;
  if (scratch_a.view.extent.height != 1 ||
      scratch_b.view.extent.height != 1 ||
      scratch_a.view.extent.width < partial_count ||
      scratch_b.view.extent.width < partial_count ||
      output.view.extent != Extent2D{1, 1}) {
    return Status::InvalidArgument(
      "Maximum reduction scratch or output geometry is invalid");
  }
  const std::array<DeviceMemoryRange, 4> ranges{
    input.range, scratch_a.range, scratch_b.range, output.range};
  for (size_t left = 0; left < ranges.size(); ++left) {
    for (size_t right = left + 1; right < ranges.size(); ++right) {
      if (DeviceRangesOverlap(ranges[left], ranges[right])) {
        return Status::InvalidArgument(
          "Maximum reduction buffers overlap");
      }
    }
  }
  return Status::Ok();
}

Status CudaBackend::ValidatePrimitiveCommand(
  const ImagePrimitiveCommand& command) const {
  return std::visit(
    [this](const auto& concrete) { return ValidatePrimitive(concrete); },
    command);
}

cudaError_t CudaBackend::EncodePrimitive(
  const PointwiseAffineCommand& command) {
  const CudaBuffer* input = AsCudaBuffer(*command.input.buffer);
  CudaBuffer* output = AsCudaBuffer(*command.output.buffer);
  return LaunchCudaPointwiseAffine(
    OffsetPointer(static_cast<const float*>(input->pointer()),
                  command.input.offset_bytes),
    OffsetPointer(static_cast<float*>(output->pointer()),
                  command.output.offset_bytes),
    static_cast<unsigned int>(command.input.extent.width),
    static_cast<unsigned int>(command.input.extent.height),
    static_cast<unsigned int>(command.input.row_stride),
    static_cast<unsigned int>(command.output.row_stride),
    command.scale, command.bias, state_->stream);
}

cudaError_t CudaBackend::EncodePrimitive(
  const SeparableConvolutionCommand& command) {
  const CudaBuffer* input = AsCudaBuffer(*command.input.buffer);
  const CudaBuffer* kernel = AsCudaBuffer(*command.kernel.buffer);
  CudaBuffer* intermediate = AsCudaBuffer(*command.intermediate.buffer);
  CudaBuffer* output = AsCudaBuffer(*command.output.buffer);
  const float* weights = OffsetPointer(
    static_cast<const float*>(kernel->pointer()), command.kernel.offset_bytes);
  cudaError_t error = LaunchCudaSeparableConvolutionPass(
    true,
    OffsetPointer(static_cast<const float*>(input->pointer()),
                  command.input.offset_bytes),
    weights,
    OffsetPointer(static_cast<float*>(intermediate->pointer()),
                  command.intermediate.offset_bytes),
    static_cast<unsigned int>(command.input.extent.width),
    static_cast<unsigned int>(command.input.extent.height),
    static_cast<unsigned int>(command.input.row_stride),
    static_cast<unsigned int>(command.intermediate.row_stride),
    static_cast<unsigned int>(command.kernel.extent.width), state_->stream);
  if (error != cudaSuccess) return error;
  return LaunchCudaSeparableConvolutionPass(
    false,
    OffsetPointer(static_cast<const float*>(intermediate->pointer()),
                  command.intermediate.offset_bytes),
    weights,
    OffsetPointer(static_cast<float*>(output->pointer()),
                  command.output.offset_bytes),
    static_cast<unsigned int>(command.input.extent.width),
    static_cast<unsigned int>(command.input.extent.height),
    static_cast<unsigned int>(command.intermediate.row_stride),
    static_cast<unsigned int>(command.output.row_stride),
    static_cast<unsigned int>(command.kernel.extent.width), state_->stream);
}

cudaError_t CudaBackend::EncodePrimitive(
  const Symmetric5ConvolutionCommand& command) {
  const CudaBuffer* input = AsCudaBuffer(*command.input.buffer);
  CudaBuffer* output = AsCudaBuffer(*command.output.buffer);
  return LaunchCudaSymmetric5Convolution(
    OffsetPointer(static_cast<const float*>(input->pointer()),
                  command.input.offset_bytes),
    OffsetPointer(static_cast<float*>(output->pointer()),
                  command.output.offset_bytes),
    static_cast<unsigned int>(command.input.extent.width),
    static_cast<unsigned int>(command.input.extent.height),
    static_cast<unsigned int>(command.input.row_stride),
    static_cast<unsigned int>(command.output.row_stride),
    command.weights.distance0,
    command.weights.distance1,
    command.weights.distance2,
    command.weights.distance4,
    command.weights.distance8,
    command.weights.distance5,
    state_->stream);
}

cudaError_t CudaBackend::EncodePrimitive(
  const MaximumReductionCommand& command) {
  const CudaBuffer* input_buffer = AsCudaBuffer(*command.input.buffer);
  CudaBuffer* scratch_a = AsCudaBuffer(*command.scratch_a.buffer);
  CudaBuffer* scratch_b = AsCudaBuffer(*command.scratch_b.buffer);
  CudaBuffer* output_buffer = AsCudaBuffer(*command.output.buffer);
  const float* input = OffsetPointer(
    static_cast<const float*>(input_buffer->pointer()),
    command.input.offset_bytes);
  size_t input_count = 0;
  (void)command.input.extent.try_area(&input_count);
  unsigned int width = static_cast<unsigned int>(command.input.extent.width);
  unsigned int stride = static_cast<unsigned int>(command.input.row_stride);
  bool use_a = true;
  while (true) {
    const size_t output_count =
      (input_count + kReductionThreadCount - 1) / kReductionThreadCount;
    DevicePlaneView destination = output_count == 1
      ? command.output
      : (use_a ? command.scratch_a : command.scratch_b);
    CudaBuffer* destination_buffer = output_count == 1
      ? output_buffer
      : (use_a ? scratch_a : scratch_b);
    float* output = OffsetPointer(
      static_cast<float*>(destination_buffer->pointer()),
      destination.offset_bytes);
    const cudaError_t error = LaunchCudaMaximumReduction(
      input, output, width, stride, static_cast<unsigned int>(input_count),
      state_->stream);
    if (error != cudaSuccess || output_count == 1) return error;
    input = output;
    input_count = output_count;
    width = static_cast<unsigned int>(output_count);
    stride = width;
    use_a = !use_a;
  }
}

cudaError_t CudaBackend::EncodePrimitiveCommand(
  const ImagePrimitiveCommand& command) {
  return std::visit(
    [this](const auto& concrete) { return EncodePrimitive(concrete); },
    command);
}

cudaError_t CudaBackend::EncodePrimitiveSequence(
  CudaBackend& backend,
  const void* context) {
  const auto commands =
    *static_cast<const std::span<const ImagePrimitiveCommand>*>(context);
  for (const ImagePrimitiveCommand& command : commands) {
    const cudaError_t error = backend.EncodePrimitiveCommand(command);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

Status CudaBackend::SubmitImagePrimitiveSequence(
  std::span<const ImagePrimitiveCommand> commands,
  std::unique_ptr<GpuSubmission>* submission) {
  if (submission == nullptr) {
    return Status::InvalidArgument(
      "Image primitive submission output pointer is null");
  }
  submission->reset();
  if (commands.empty()) {
    return Status::InvalidArgument("Image primitive sequence is empty");
  }
  for (const ImagePrimitiveCommand& command : commands) {
    Status status = ValidatePrimitiveCommand(command);
    if (!status.ok()) return status;
  }
  return SubmitCompute(
    &CudaBackend::EncodePrimitiveSequence, &commands, submission);
}

}  // namespace gjxl::cuda_internal
