#include "gpu/metal/metal_backend.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/buffer.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/metal/metal_status.h"

namespace gjxl {
namespace {

using metal_internal::ButteraugliPipelines;
using metal_internal::MetalBackend;
using metal_internal::MetalBuffer;
using metal_internal::PrimitivePipelines;
using metal_internal::TransformDirection;
using metal_internal::TransformPipeline;
using metal_internal::TransformPipelinePair;
using metal_internal::TransformPipelineRegistry;

enum class TransformDispatchMode {
  kOneThreadPerElement,
  kFixedSimdgroupCount,
};

struct DctImplementationSpec {
  AcStrategyType strategy;
  MetalDctImplementation implementation;
  std::string_view display_name;
  std::string_view forward_function_name;
  std::string_view inverse_function_name;
  TransformDispatchMode dispatch_mode;
  size_t simdgroups_per_threadgroup = 0;
};

struct DctSelection {
  AcStrategyType strategy;
  MetalDctImplementation forward;
  MetalDctImplementation inverse;
};

constexpr std::array<DctImplementationSpec, 14>
kDctImplementationSpecs{{
  {
    .strategy = AcStrategyType::kDct8,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name =
      "gjxl_dct8_forward_scalar_2d_matmul",
    .inverse_function_name =
      "gjxl_dct8_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct8,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name =
      "gjxl_dct8_forward_simdgroup_2d_matmul",
    .inverse_function_name =
      "gjxl_dct8_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct16x16,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name =
      "gjxl_dct16_forward_scalar_2d_matmul",
    .inverse_function_name =
      "gjxl_dct16_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct16x16,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name =
      "gjxl_dct16_forward_simdgroup_2d_matmul",
    .inverse_function_name =
      "gjxl_dct16_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 2,
  },
  {
    .strategy = AcStrategyType::kDct32x32,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name =
      "gjxl_dct32_forward_scalar_2d_matmul",
    .inverse_function_name =
      "gjxl_dct32_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct32x32,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name =
      "gjxl_dct32_forward_simdgroup_2d_matmul",
    .inverse_function_name =
      "gjxl_dct32_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 4,
  },
  {
    .strategy = AcStrategyType::kDct16x8,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct16x8_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct16x8_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct16x8,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct16x8_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct16x8_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 2,
  },
  {
    .strategy = AcStrategyType::kDct8x16,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct8x16_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct8x16_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct8x16,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct8x16_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct8x16_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct32x16,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct32x16_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct32x16_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct32x16,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct32x16_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct32x16_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 4,
  },
  {
    .strategy = AcStrategyType::kDct16x32,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct16x32_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct16x32_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct16x32,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct16x32_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct16x32_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 2,
  },
}};

const DctImplementationSpec* FindDctImplementationSpec(
  AcStrategyType strategy,
  MetalDctImplementation implementation) {

  for (const DctImplementationSpec& spec : kDctImplementationSpecs) {
    if (spec.strategy == strategy &&
        spec.implementation == implementation) {
      return &spec;
    }
  }

  return nullptr;
}

[[nodiscard]] constexpr size_t StrategyIndex(
  AcStrategyType strategy) noexcept {

  return static_cast<size_t>(strategy);
}

// Pipeline creation
Status CreatePipeline(
  MTL::Device* device,
  MTL::Library* library,
  std::string_view function_name,
  NS::SharedPtr<MTL::ComputePipelineState>* out) {

  if (device == nullptr ||
      library == nullptr ||
      function_name.empty() ||
      out == nullptr) {

    return Status::InvalidArgument(
      "CreatePipeline received invalid argument");
  }

  const std::string function_name_string(function_name);

  NS::String* name =
    NS::String::string(
      function_name_string.c_str(),
      NS::UTF8StringEncoding);

  auto function =
    NS::TransferPtr(
      library->newFunction(name));

  if (!function) {
    return Status::Internal(
      std::string("Metal function not found: ") +
      function_name_string);
  }

  if (function->functionType() != MTL::FunctionTypeKernel) {
    return Status::InvalidArgument(
      std::string("Metal function is not a kernel: ") +
      function_name_string);
  }

  NS::Error* error = nullptr;

  auto pipeline =
    NS::TransferPtr(
      device->newComputePipelineState(
        function.get(),
        &error));

  if (!pipeline) {
    return metal::ErrorToStatus(
      error,
      "newComputePipelineState");
  }

  *out = std::move(pipeline);

  return Status::Ok();
}

Status CreateTransformPipeline(
  MTL::Device* device,
  MTL::Library* library,
  AcStrategyType strategy,
  std::string_view implementation_name,
  TransformDispatchMode dispatch_mode,
  size_t simdgroups_per_threadgroup,
  std::string_view function_name,
  std::string_view operation,
  TransformPipeline* out) {

  const AcStrategyInfo* strategy_info =
    GetAcStrategyInfo(strategy);

  if (strategy_info == nullptr || out == nullptr) {
    return Status::InvalidArgument(
      "CreateTransformPipeline received invalid argument");
  }

  const size_t coefficient_count =
    strategy_info->coefficient_count();

  NS::SharedPtr<MTL::ComputePipelineState> state;

  Status status =
    CreatePipeline(
      device,
      library,
      function_name,
      &state);

  if (!status.ok()) {
    return {
      status.code(),
      std::string("Failed to create ") +
      std::string(operation) +
      " " +
      std::string(strategy_info->name) +
      " pipeline for " +
      std::string(implementation_name) +
      ": " +
      std::string(status.message()),
    };
  }

  NS::UInteger threads_per_threadgroup = 0;

  switch (dispatch_mode) {
    case TransformDispatchMode::kOneThreadPerElement:
      threads_per_threadgroup =
        static_cast<NS::UInteger>(coefficient_count);
      break;

    case TransformDispatchMode::kFixedSimdgroupCount: {
      const NS::UInteger simd_width = state->threadExecutionWidth();

      if (simd_width == 0 || simdgroups_per_threadgroup == 0 ||
          simdgroups_per_threadgroup >
            std::numeric_limits<NS::UInteger>::max() / simd_width) {
        return Status::Unavailable(
          std::string("Metal reported invalid SIMD-group dispatch data for ") +
          std::string(implementation_name));
      }

      threads_per_threadgroup =
        simd_width *
        static_cast<NS::UInteger>(simdgroups_per_threadgroup);
      break;
    }
  }

  if (threads_per_threadgroup == 0) {
    return Status::Unavailable(
      std::string("Metal reported an invalid threadgroup size for ") +
      std::string(implementation_name));
  }

  if (state->maxTotalThreadsPerThreadgroup() <
      threads_per_threadgroup) {

    return Status::Unavailable(
      std::string("Metal GPU cannot launch the required threadgroup for ") +
      std::string(implementation_name));
  }

  out->state = std::move(state);
  out->threads_per_threadgroup = threads_per_threadgroup;
  out->strategy = strategy;
  out->label =
    std::string("gjxl ") +
    std::string(operation) +
    " " +
    std::string(strategy_info->name) +
    " (" +
    std::string(implementation_name) +
    ")";

  return Status::Ok();
}


}  // namespace

namespace metal_internal {
namespace {

struct TransformSubmissionContext {
  const TransformPipeline* pipeline = nullptr;
  const MetalBuffer* input = nullptr;
  MetalBuffer* output = nullptr;
  size_t transform_count = 0;
};

}  // namespace

MetalBackend::MetalBackend(
  NS::SharedPtr<MTL::Device> device,
  NS::SharedPtr<MTL::CommandQueue> command_queue,
  NS::SharedPtr<MTL::Library> library,
  TransformPipelineRegistry transform_pipelines,
  PrimitivePipelines primitive_pipelines,
  ButteraugliPipelines butteraugli_pipelines,
  bool test_fail_submission,
  bool test_fail_completion)
  : device_(std::move(device)),
    command_queue_(std::move(command_queue)),
    library_(std::move(library)),
    transform_pipelines_(std::move(transform_pipelines)),
    primitive_pipelines_(std::move(primitive_pipelines)),
    butteraugli_pipelines_(std::move(butteraugli_pipelines)),
    test_fail_submission_(test_fail_submission),
    test_fail_completion_(test_fail_completion) {

  NS::String* device_name = device_->name();
  if (device_name != nullptr) {
    const char* utf8 = device_name->utf8String();
    if (utf8 != nullptr) {
      name_ = std::string("Metal: ") + utf8;
    }
  }
  if (name_.empty()) {
    name_ = "Metal";
  }
}

BackendKind MetalBackend::kind() const noexcept {
  return BackendKind::kMetal;
}

std::string_view MetalBackend::name() const noexcept {
  return name_;
}

Status MetalBackend::Allocate(
  size_t size_bytes,
  std::unique_ptr<DeviceBuffer>* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Allocate output pointer is null");
  }
  if (size_bytes == 0) {
    return Status::InvalidArgument(
      "Cannot allocate zero-sized Metal buffer");
  }
  if (size_bytes > std::numeric_limits<NS::UInteger>::max()) {
    return Status::InvalidArgument(
      "Requested Metal buffer is too large");
  }

  auto buffer = NS::TransferPtr(
    device_->newBuffer(
      static_cast<NS::UInteger>(size_bytes),
      MTL::ResourceStorageModeShared));
  if (!buffer) {
    return Status::OutOfMemory(
      "Metal failed to allocate MTL::Buffer");
  }

  out->reset(new MetalBuffer(std::move(buffer), id(), size_bytes));
  RecordSuccessfulAllocation();
  return Status::Ok();
}

Status MetalBackend::CopyHostToDevice(
  DeviceBuffer& dst,
  const void* src,
  size_t size_bytes,
  size_t dst_offset_bytes) {

  if (src == nullptr && size_bytes != 0) {
    return Status::InvalidArgument(
      "Host source pointer is null");
  }
  MetalBuffer* metal_dst = AsMetalBuffer(dst);
  if (metal_dst == nullptr) {
    return Status::InvalidArgument(
      "Destination is not a Metal buffer");
  }
  if (!owns(dst) || metal_dst->device() != device_.get()) {
    return Status::InvalidArgument(
      "Destination belongs to another Metal backend");
  }
  if (dst_offset_bytes > dst.size_bytes() ||
      size_bytes > dst.size_bytes() - dst_offset_bytes) {
    return Status::InvalidArgument(
      "Host to device copy exceeds destination buffer");
  }

  auto* destination =
    static_cast<std::byte*>(metal_dst->contents()) + dst_offset_bytes;
  std::memcpy(destination, src, size_bytes);
  return Status::Ok();
}

Status MetalBackend::CopyDeviceToHost(
  const DeviceBuffer& src,
  void* dst,
  size_t size_bytes,
  size_t src_offset_bytes) {

  if (dst == nullptr && size_bytes != 0) {
    return Status::InvalidArgument(
      "Host destination pointer is null");
  }
  const MetalBuffer* metal_src = AsMetalBuffer(src);
  if (metal_src == nullptr) {
    return Status::InvalidArgument(
      "Source is not a Metal buffer");
  }
  if (!owns(src) || metal_src->device() != device_.get()) {
    return Status::InvalidArgument(
      "Source belongs to another Metal backend");
  }
  if (src_offset_bytes > src.size_bytes() ||
      size_bytes > src.size_bytes() - src_offset_bytes) {
    return Status::InvalidArgument(
      "Device-to-host copy exceeds source buffer");
  }

  const auto* source =
    static_cast<const std::byte*>(metal_src->contents()) + src_offset_bytes;
  std::memcpy(dst, source, size_bytes);
  return Status::Ok();
}

Status MetalBackend::ForwardTransform(
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {

  return SubmitTransform(
    TransformDirection::kForward, batch, submission);
}

Status MetalBackend::InverseTransform(
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {

  return SubmitTransform(
    TransformDirection::kInverse, batch, submission);
}

MetalBuffer* MetalBackend::AsMetalBuffer(DeviceBuffer& buffer) {
  if (buffer.backend() != BackendKind::kMetal) {
    return nullptr;
  }
  return dynamic_cast<MetalBuffer*>(&buffer);
}

const MetalBuffer* MetalBackend::AsMetalBuffer(
  const DeviceBuffer& buffer) {

  if (buffer.backend() != BackendKind::kMetal) {
    return nullptr;
  }
  return dynamic_cast<const MetalBuffer*>(&buffer);
}

Status MetalBackend::ValidateTransformBatch(
  const AcStrategyInfo& strategy_info,
  const TransformBatch& batch,
  const MetalBuffer** input,
  MetalBuffer** output) const {

  if (input == nullptr || output == nullptr) {
    return Status::Internal(
      "ValidateTransformBatch output is null");
  }
  if (batch.transform_count == 0) {
    *input = nullptr;
    *output = nullptr;
    return Status::Ok();
  }
  if (batch.input == nullptr || batch.output == nullptr) {
    return Status::InvalidArgument(
      "Transform input/output buffer is null");
  }
  if (batch.input == batch.output) {
    return Status::InvalidArgument(
      "In-place transforms are not supported yet");
  }

  const size_t coefficient_count = strategy_info.coefficient_count();
  if (coefficient_count >
      std::numeric_limits<size_t>::max() / sizeof(float)) {
    return Status::Internal(
      "Transform coefficient count is too large");
  }
  const size_t bytes_per_transform =
    coefficient_count * sizeof(float);
  if (batch.transform_count >
      std::numeric_limits<size_t>::max() / bytes_per_transform) {
    return Status::InvalidArgument(
      "Transform batch is too large");
  }
  const size_t required_bytes =
    batch.transform_count * bytes_per_transform;
  if (batch.input->size_bytes() < required_bytes ||
      batch.output->size_bytes() < required_bytes) {
    return Status::InvalidArgument(
      "Transform buffer is too small");
  }

  *input = AsMetalBuffer(*batch.input);
  *output = AsMetalBuffer(*batch.output);
  if (*input == nullptr || *output == nullptr) {
    return Status::InvalidArgument(
      "Transform buffers are not Metal buffers");
  }
  if (!owns(*batch.input) || !owns(*batch.output) ||
      (*input)->device() != device_.get() ||
      (*output)->device() != device_.get()) {
    return Status::InvalidArgument(
      "Transform buffer belongs to another Metal backend");
  }
  return Status::Ok();
}

void MetalBackend::EncodeTransformSubmission(
  MetalBackend&,
  MTL::ComputeCommandEncoder* encoder,
  const void* context) {

  const auto& transform =
    *static_cast<const TransformSubmissionContext*>(context);
  encoder->setComputePipelineState(transform.pipeline->state.get());
  encoder->setBuffer(transform.input->handle(), 0, 0);
  encoder->setBuffer(transform.output->handle(), 0, 1);
  encoder->dispatchThreadgroups(
    MTL::Size(
      static_cast<NS::UInteger>(transform.transform_count), 1, 1),
    MTL::Size(transform.pipeline->threads_per_threadgroup, 1, 1));
}

Status MetalBackend::SubmitTransform(
  TransformDirection direction,
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr) {
    return Status::InvalidArgument(
      "Transform submission output pointer is null");
  }
  submission->reset();

  const AcStrategyInfo* strategy_info =
    GetAcStrategyInfo(batch.strategy);
  if (strategy_info == nullptr) {
    return Status::InvalidArgument(
      "Unknown JPEG XL AC strategy");
  }

  const TransformPipelinePair& pair =
    transform_pipelines_[StrategyIndex(batch.strategy)];
  const TransformPipeline& pipeline =
    direction == TransformDirection::kForward
      ? pair.forward
      : pair.inverse;
  if (!pipeline.state) {
    return Status::Unavailable(
      std::string("Metal backend does not support ") +
      std::string(strategy_info->name));
  }
  if (pipeline.strategy != batch.strategy) {
    return Status::Internal(
      "Transform pipeline strategy does not match batch strategy");
  }

  const MetalBuffer* input = nullptr;
  MetalBuffer* output = nullptr;
  Status status = ValidateTransformBatch(
    *strategy_info, batch, &input, &output);
  if (!status.ok()) {
    return status;
  }
  if (batch.transform_count == 0) {
    return Status::Ok();
  }
  if (batch.transform_count >
      std::numeric_limits<NS::UInteger>::max()) {
    return Status::InvalidArgument(
      "Transform batch exceeds Metal grid range");
  }

  const TransformSubmissionContext context{
    &pipeline, input, output, batch.transform_count};
  return SubmitCompute(
    pipeline.label.c_str(),
    &MetalBackend::EncodeTransformSubmission,
    &context,
    submission);
}

}  // namespace metal_internal

// Factory

Status CreateMetalBackend(
  std::string_view metallib_path,
  std::unique_ptr<GpuBackend>* out) {

  return CreateMetalBackend(
    metallib_path,
    MetalBackendOptions{},
    out);
}

Status CreateMetalBackend(
  std::string_view metallib_path,
  const MetalBackendOptions& options,
  std::unique_ptr<GpuBackend>* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "CreateMetalBackend output pointer is null");
  }

  if (metallib_path.empty()) {
    return Status::InvalidArgument(
      "Metal library path is empty");
  }

  const std::array<DctSelection, 7> dct_selections{{
    {
      .strategy = AcStrategyType::kDct8,
      .forward = options.forward_dct8,
      .inverse = options.inverse_dct8,
    },
    {
      .strategy = AcStrategyType::kDct16x16,
      .forward = options.forward_dct16x16,
      .inverse = options.inverse_dct16x16,
    },
    {
      .strategy = AcStrategyType::kDct32x32,
      .forward = options.forward_dct32x32,
      .inverse = options.inverse_dct32x32,
    },
    {
      .strategy = AcStrategyType::kDct16x8,
      .forward = options.forward_dct16x8,
      .inverse = options.inverse_dct16x8,
    },
    {
      .strategy = AcStrategyType::kDct8x16,
      .forward = options.forward_dct8x16,
      .inverse = options.inverse_dct8x16,
    },
    {
      .strategy = AcStrategyType::kDct32x16,
      .forward = options.forward_dct32x16,
      .inverse = options.inverse_dct32x16,
    },
    {
      .strategy = AcStrategyType::kDct16x32,
      .forward = options.forward_dct16x32,
      .inverse = options.inverse_dct16x32,
    },
  }};

  for (const DctSelection& selection : dct_selections) {
    const AcStrategyInfo* strategy_info =
      GetAcStrategyInfo(selection.strategy);

    if (strategy_info == nullptr) {
      return Status::Internal(
        "Metal DCT selection has an invalid strategy");
    }

    if (FindDctImplementationSpec(
          selection.strategy,
          selection.forward) == nullptr) {
      return Status::InvalidArgument(
        std::string("Unknown forward Metal ") +
        std::string(strategy_info->name) +
        " implementation");
    }

    if (FindDctImplementationSpec(
          selection.strategy,
          selection.inverse) == nullptr) {
      return Status::InvalidArgument(
        std::string("Unknown inverse Metal ") +
        std::string(strategy_info->name) +
        " implementation");
    }
  }

  auto pool =
    NS::TransferPtr(
      NS::AutoreleasePool::alloc()->init());

  auto device =
    NS::TransferPtr(
      MTL::CreateSystemDefaultDevice());

  if (!device) {
    return Status::Unavailable(
      "No Metal device is available");
  }

  auto command_queue =
    NS::TransferPtr(
      device->newCommandQueue());

  if (!command_queue) {
    return Status::Internal(
      "Failed to create Metal command queue");
  }

  const std::string path(metallib_path);

  NS::String* ns_path =
    NS::String::string(
      path.c_str(),
      NS::UTF8StringEncoding);

  NS::Error* error = nullptr;

  auto library =
    NS::TransferPtr(
      device->newLibrary(
        ns_path,
        &error));

  if (!library) {
    return metal::ErrorToStatus(
      error,
      "Loading gjxl.metallib");
  }

  TransformPipelineRegistry transform_pipelines;
  Status status = Status::Ok();

  for (const DctSelection& selection : dct_selections) {
    const DctImplementationSpec* forward_spec =
      FindDctImplementationSpec(
        selection.strategy,
        selection.forward);
    const DctImplementationSpec* inverse_spec =
      FindDctImplementationSpec(
        selection.strategy,
        selection.inverse);

    if (forward_spec == nullptr || inverse_spec == nullptr) {
      return Status::Internal(
        "Validated Metal DCT implementation disappeared");
    }

    TransformPipelinePair& pipelines =
      transform_pipelines[StrategyIndex(selection.strategy)];

    status =
      CreateTransformPipeline(
        device.get(),
        library.get(),
        selection.strategy,
        forward_spec->display_name,
        forward_spec->dispatch_mode,
        forward_spec->simdgroups_per_threadgroup,
        forward_spec->forward_function_name,
        "forward",
        &pipelines.forward);

    if (!status.ok()) {
      return status;
    }

    status =
      CreateTransformPipeline(
        device.get(),
        library.get(),
        selection.strategy,
        inverse_spec->display_name,
        inverse_spec->dispatch_mode,
        inverse_spec->simdgroups_per_threadgroup,
        inverse_spec->inverse_function_name,
        "inverse",
        &pipelines.inverse);

    if (!status.ok()) {
      return status;
    }
  }

  PrimitivePipelines primitive_pipelines;
  status = CreatePrimitivePipelines(
    device.get(), library.get(), &primitive_pipelines);
  if (!status.ok()) {
    return status;
  }

  ButteraugliPipelines butteraugli_pipelines;
  status = CreateButteraugliPipelines(
    device.get(), library.get(), &butteraugli_pipelines);
  if (!status.ok()) {
    return status;
  }

  out->reset(
    new MetalBackend(
      std::move(device),
      std::move(command_queue),
      std::move(library),
      std::move(transform_pipelines),
      std::move(primitive_pipelines),
      std::move(butteraugli_pipelines),
      options.test_fail_submission,
      options.test_fail_completion));

  return Status::Ok();
}

}  // namespace gjxl
