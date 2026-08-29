// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_backend_internal.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "gpu/metal/metal_status.h"

namespace gjxl::metal_internal {
namespace {

struct AffineParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  float scale;
  float bias;
};

struct ConvolutionParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t kernel_size;
};

struct Symmetric5Params {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  float distance0;
  float distance1;
  float distance2;
  float distance4;
  float distance8;
  float distance5;
};

struct ReductionParams {
  uint32_t width;
  uint32_t input_stride;
  uint32_t input_count;
};

static_assert(std::is_standard_layout_v<AffineParams>);
static_assert(std::is_standard_layout_v<ConvolutionParams>);
static_assert(std::is_standard_layout_v<Symmetric5Params>);
static_assert(std::is_standard_layout_v<ReductionParams>);
static_assert(std::is_trivially_copyable_v<AffineParams>);
static_assert(std::is_trivially_copyable_v<ConvolutionParams>);
static_assert(std::is_trivially_copyable_v<Symmetric5Params>);
static_assert(std::is_trivially_copyable_v<ReductionParams>);
static_assert(sizeof(AffineParams) == 24);
static_assert(sizeof(ConvolutionParams) == 20);
static_assert(sizeof(Symmetric5Params) == 40);
static_assert(sizeof(ReductionParams) == 12);

inline constexpr size_t kReductionThreadCount = 256;

Status CreatePrimitivePipeline(
  MTL::Device* device,
  MTL::Library* library,
  std::string_view function_name,
  NS::SharedPtr<MTL::ComputePipelineState>* out) {

  if (device == nullptr || library == nullptr || function_name.empty() ||
      out == nullptr) {
    return Status::InvalidArgument(
      "CreatePrimitivePipeline received invalid argument");
  }
  const std::string function_name_string(function_name);
  NS::String* name = NS::String::string(
    function_name_string.c_str(), NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(library->newFunction(name));
  if (!function) {
    return Status::Internal(
      std::string("Metal function not found: ") + function_name_string);
  }
  if (function->functionType() != MTL::FunctionTypeKernel) {
    return Status::InvalidArgument(
      std::string("Metal function is not a kernel: ") + function_name_string);
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

}  // namespace

Status CreatePrimitivePipelines(
  MTL::Device* device,
  MTL::Library* library,
  PrimitivePipelines* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Primitive pipeline output is null");
  }
  PrimitivePipelines pipelines;
  const std::array<std::pair<
    std::string_view,
    NS::SharedPtr<MTL::ComputePipelineState>*>, 5> bindings{{
    {"gjxl_pointwise_affine_f32", &pipelines.affine},
    {"gjxl_convolve_horizontal_f32", &pipelines.convolution_horizontal},
    {"gjxl_convolve_vertical_f32", &pipelines.convolution_vertical},
    {"gjxl_convolve_symmetric5_f32", &pipelines.symmetric5_convolution},
    {"gjxl_reduce_max_f32", &pipelines.maximum_reduction},
  }};
  for (const auto& [name, pipeline] : bindings) {
    Status status = CreatePrimitivePipeline(device, library, name, pipeline);
    if (!status.ok()) {
      return {
        status.code(),
        std::string("Failed to create required primitive pipeline ") +
          std::string(name) + ": " + std::string(status.message()),
      };
    }
  }
  constexpr NS::UInteger kPointwiseThreads = 8 * 8;
  if (pipelines.affine->maxTotalThreadsPerThreadgroup() < kPointwiseThreads ||
      pipelines.convolution_horizontal->maxTotalThreadsPerThreadgroup() <
        kPointwiseThreads ||
      pipelines.convolution_vertical->maxTotalThreadsPerThreadgroup() <
        kPointwiseThreads ||
      pipelines.symmetric5_convolution->maxTotalThreadsPerThreadgroup() <
        kPointwiseThreads ||
      pipelines.maximum_reduction->maxTotalThreadsPerThreadgroup() <
        kReductionThreadCount) {
    return Status::Unavailable(
      "Metal cannot launch the required primitive threadgroups");
  }
  *out = std::move(pipelines);
  return Status::Ok();
}

Status MetalBackend::ResolvePlane(
  ConstDevicePlaneView view,
  ResolvedConstPlane* out) const {

  if (out == nullptr || view.element_type != DeviceElementType::kF32) {
    return Status::InvalidArgument(
      "Metal primitive requires a float32 input plane");
  }
  DeviceMemoryRange range;
  Status status = ComputeDevicePlaneRange(view, id(), &range);
  if (!status.ok()) {
    return status;
  }
  const MetalBuffer* buffer = AsMetalBuffer(*view.buffer);
  if (buffer == nullptr || buffer->device() != device_.get()) {
    return Status::InvalidArgument(
      "Primitive input is not owned by this Metal backend");
  }
  if (view.extent.width > std::numeric_limits<uint32_t>::max() ||
      view.extent.height > std::numeric_limits<uint32_t>::max() ||
      view.row_stride > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument(
      "Metal primitive input geometry exceeds shader limits");
  }
  *out = {view, range, buffer};
  return Status::Ok();
}

Status MetalBackend::ResolvePlane(
  DevicePlaneView view,
  ResolvedPlane* out) const {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Metal primitive output resolver is null");
  }
  ResolvedConstPlane resolved;
  Status status = ResolvePlane(
    static_cast<ConstDevicePlaneView>(view), &resolved);
  if (!status.ok()) {
    return status;
  }
  MetalBuffer* buffer = AsMetalBuffer(*view.buffer);
  if (buffer == nullptr) {
    return Status::InvalidArgument(
      "Primitive output is not a Metal buffer");
  }
  *out = {view, resolved.range, buffer};
  return Status::Ok();
}

bool MetalBackend::SamePlaneLayout(
  ConstDevicePlaneView left,
  ConstDevicePlaneView right) noexcept {

  return left.buffer == right.buffer &&
         left.offset_bytes == right.offset_bytes &&
         left.element_type == right.element_type &&
         left.extent == right.extent &&
         left.row_stride == right.row_stride;
}

Status MetalBackend::RejectOverlap(
  DeviceMemoryRange left,
  DeviceMemoryRange right,
  std::string_view message) {

  return DeviceRangesOverlap(left, right)
    ? Status::InvalidArgument(std::string(message))
    : Status::Ok();
}

Status MetalBackend::ValidatePrimitive(
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

Status MetalBackend::ValidatePrimitive(
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

Status MetalBackend::ValidatePrimitive(
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
  status = RejectOverlap(
    input.range,
    output.range,
    "Symmetric5 convolution planes overlap");
  if (!status.ok()) return status;
  return Status::Ok();
}

Status MetalBackend::ValidatePrimitive(
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
    return Status::InvalidArgument(
      "Maximum reduction input is too large");
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

Status MetalBackend::ValidatePrimitiveCommand(
  const ImagePrimitiveCommand& command) const {

  return std::visit(
    [this](const auto& concrete) {
      return ValidatePrimitive(concrete);
    },
    command);
}

void MetalBackend::DispatchPlane(
  MTL::ComputeCommandEncoder* encoder,
  Extent2D extent) {

  DispatchMetalThreads(
    encoder,
    MTL::Size(
      static_cast<NS::UInteger>(extent.width),
      static_cast<NS::UInteger>(extent.height),
      1),
    MTL::Size(8, 8, 1));
}

void MetalBackend::EncodePrimitive(
  MTL::ComputeCommandEncoder* encoder,
  const PointwiseAffineCommand& command) {

  const MetalBuffer* input = AsMetalBuffer(*command.input.buffer);
  MetalBuffer* output = AsMetalBuffer(*command.output.buffer);
  const AffineParams params{
    static_cast<uint32_t>(command.input.extent.width),
    static_cast<uint32_t>(command.input.extent.height),
    static_cast<uint32_t>(command.input.row_stride),
    static_cast<uint32_t>(command.output.row_stride),
    command.scale,
    command.bias,
  };
  encoder->setComputePipelineState(primitive_pipelines_.affine.get());
  encoder->setBuffer(input->handle(), command.input.offset_bytes, 0);
  encoder->setBuffer(output->handle(), command.output.offset_bytes, 1);
  encoder->setBytes(&params, sizeof(params), 2);
  DispatchPlane(encoder, command.input.extent);
}

void MetalBackend::EncodeConvolutionPass(
  MTL::ComputeCommandEncoder* encoder,
  MTL::ComputePipelineState* pipeline,
  ConstDevicePlaneView input_view,
  ConstDevicePlaneView kernel_view,
  DevicePlaneView output_view) {

  const MetalBuffer* input = AsMetalBuffer(*input_view.buffer);
  const MetalBuffer* kernel = AsMetalBuffer(*kernel_view.buffer);
  MetalBuffer* output = AsMetalBuffer(*output_view.buffer);
  const ConvolutionParams params{
    static_cast<uint32_t>(input_view.extent.width),
    static_cast<uint32_t>(input_view.extent.height),
    static_cast<uint32_t>(input_view.row_stride),
    static_cast<uint32_t>(output_view.row_stride),
    static_cast<uint32_t>(kernel_view.extent.width),
  };
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(input->handle(), input_view.offset_bytes, 0);
  encoder->setBuffer(kernel->handle(), kernel_view.offset_bytes, 1);
  encoder->setBuffer(output->handle(), output_view.offset_bytes, 2);
  encoder->setBytes(&params, sizeof(params), 3);
  DispatchPlane(encoder, input_view.extent);
}

void MetalBackend::EncodePrimitive(
  MTL::ComputeCommandEncoder* encoder,
  const SeparableConvolutionCommand& command) {

  EncodeConvolutionPass(
    encoder,
    primitive_pipelines_.convolution_horizontal.get(),
    command.input,
    command.kernel,
    command.intermediate);
  EncodeConvolutionPass(
    encoder,
    primitive_pipelines_.convolution_vertical.get(),
    command.intermediate,
    command.kernel,
    command.output);
}

void MetalBackend::EncodePrimitive(
  MTL::ComputeCommandEncoder* encoder,
  const Symmetric5ConvolutionCommand& command) {

  const MetalBuffer* input = AsMetalBuffer(*command.input.buffer);
  MetalBuffer* output = AsMetalBuffer(*command.output.buffer);
  const Symmetric5Params params{
    static_cast<uint32_t>(command.input.extent.width),
    static_cast<uint32_t>(command.input.extent.height),
    static_cast<uint32_t>(command.input.row_stride),
    static_cast<uint32_t>(command.output.row_stride),
    command.weights.distance0,
    command.weights.distance1,
    command.weights.distance2,
    command.weights.distance4,
    command.weights.distance8,
    command.weights.distance5,
  };
  encoder->setComputePipelineState(
    primitive_pipelines_.symmetric5_convolution.get());
  encoder->setBuffer(input->handle(), command.input.offset_bytes, 0);
  encoder->setBuffer(output->handle(), command.output.offset_bytes, 1);
  encoder->setBytes(&params, sizeof(params), 2);
  DispatchPlane(encoder, command.input.extent);
}

void MetalBackend::EncodeReductionPass(
  MTL::ComputeCommandEncoder* encoder,
  ConstDevicePlaneView input_view,
  size_t input_count,
  DevicePlaneView output_view) {

  const MetalBuffer* input = AsMetalBuffer(*input_view.buffer);
  MetalBuffer* output = AsMetalBuffer(*output_view.buffer);
  const ReductionParams params{
    static_cast<uint32_t>(input_view.extent.width),
    static_cast<uint32_t>(input_view.row_stride),
    static_cast<uint32_t>(input_count),
  };
  const size_t output_count =
    (input_count + kReductionThreadCount - 1) / kReductionThreadCount;
  encoder->setComputePipelineState(
    primitive_pipelines_.maximum_reduction.get());
  encoder->setBuffer(input->handle(), input_view.offset_bytes, 0);
  encoder->setBuffer(output->handle(), output_view.offset_bytes, 1);
  encoder->setBytes(&params, sizeof(params), 2);
  DispatchMetalThreadgroups(
    encoder,
    MTL::Size(static_cast<NS::UInteger>(output_count), 1, 1),
    MTL::Size(kReductionThreadCount, 1, 1));
}

void MetalBackend::EncodePrimitive(
  MTL::ComputeCommandEncoder* encoder,
  const MaximumReductionCommand& command) {

  size_t input_count = 0;
  (void)command.input.extent.try_area(&input_count);
  ConstDevicePlaneView input = command.input;
  bool use_a = true;
  while (true) {
    const size_t output_count =
      (input_count + kReductionThreadCount - 1) / kReductionThreadCount;
    DevicePlaneView destination = output_count == 1
      ? command.output
      : (use_a ? command.scratch_a : command.scratch_b);
    EncodeReductionPass(encoder, input, input_count, destination);
    if (output_count == 1) {
      break;
    }
    input = destination;
    input.extent = {output_count, 1};
    input.row_stride = output_count;
    input_count = output_count;
    use_a = !use_a;
  }
}

void MetalBackend::EncodePrimitiveCommand(
  MTL::ComputeCommandEncoder* encoder,
  const ImagePrimitiveCommand& command) {

  std::visit(
    [this, encoder](const auto& concrete) {
      EncodePrimitive(encoder, concrete);
    },
    command);
}

void MetalBackend::EncodePrimitiveSubmission(
  MetalBackend& backend,
  MTL::ComputeCommandEncoder* encoder,
  const void* context) {

  const auto& commands =
    *static_cast<const std::span<const ImagePrimitiveCommand>*>(context);
  for (const ImagePrimitiveCommand& command : commands) {
    backend.EncodePrimitiveCommand(encoder, command);
  }
}

Status MetalBackend::SubmitImagePrimitiveSequence(
  std::span<const ImagePrimitiveCommand> commands,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr) {
    return Status::InvalidArgument(
      "Image primitive submission output pointer is null");
  }
  submission->reset();
  if (commands.empty()) {
    return Status::InvalidArgument(
      "Image primitive sequence is empty");
  }
  for (const ImagePrimitiveCommand& command : commands) {
    Status status = ValidatePrimitiveCommand(command);
    if (!status.ok()) {
      return status;
    }
  }
  return SubmitCompute(
    "gjxl image primitive sequence",
    &MetalBackend::EncodePrimitiveSubmission,
    &commands,
    submission);
}

Status MetalBackend::SubmitImagePrimitiveSequenceProfiled(
  std::span<const ImagePrimitiveCommand> commands,
  std::string_view stage_id,
  gpu_profile_internal::GpuProfilingMode mode,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr || commands.empty() || stage_id.empty() ||
      mode == gpu_profile_internal::GpuProfilingMode::kDisabled) {
    return Status::InvalidArgument(
      "Profiled image primitive submission is invalid");
  }
  submission->reset();
  for (const ImagePrimitiveCommand& command : commands) {
    Status status = ValidatePrimitiveCommand(command);
    if (!status.ok()) return status;
  }
  try {
    const std::string stage_id_storage(stage_id);
    const MetalProfiledComputeStage stage{
      .stage_id = stage_id_storage.c_str(),
      .encode = &MetalBackend::EncodePrimitiveSubmission,
      .context = &commands,
    };
    return SubmitComputeProfiled(
      "gjxl image primitive sequence profile",
      std::span<const MetalProfiledComputeStage>(&stage, 1), mode,
      submission);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate image primitive profile metadata");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Image primitive profile metadata is too large");
  }
}

}  // namespace gjxl::metal_internal
