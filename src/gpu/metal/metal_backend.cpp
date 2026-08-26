#include "gpu/metal/metal_backend.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/buffer.h"
#include "gpu/image.h"
#include "gpu/metal/metal_status.h"
#include "gpu/ops/primitives.h"

namespace gjxl {
namespace {

enum class TransformDirection {
  kForward,
  kInverse,
};

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

struct TransformPipeline {
  NS::SharedPtr<MTL::ComputePipelineState> state;
  NS::UInteger threads_per_threadgroup = 0;
  AcStrategyType strategy = AcStrategyType::kCount;
  std::string label;
};

struct TransformPipelinePair {
  TransformPipeline forward;
  TransformPipeline inverse;
};

using TransformPipelineRegistry =
  std::array<TransformPipelinePair, kAcStrategyCount>;

struct PrimitivePipelines {
  NS::SharedPtr<MTL::ComputePipelineState> affine;
  NS::SharedPtr<MTL::ComputePipelineState> convolution_horizontal;
  NS::SharedPtr<MTL::ComputePipelineState> convolution_vertical;
  NS::SharedPtr<MTL::ComputePipelineState> maximum_reduction;
};

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

struct ReductionParams {
  uint32_t width;
  uint32_t input_stride;
  uint32_t input_count;
};

static_assert(std::is_standard_layout_v<AffineParams>);
static_assert(std::is_standard_layout_v<ConvolutionParams>);
static_assert(std::is_standard_layout_v<ReductionParams>);
static_assert(std::is_trivially_copyable_v<AffineParams>);
static_assert(std::is_trivially_copyable_v<ConvolutionParams>);
static_assert(std::is_trivially_copyable_v<ReductionParams>);
static_assert(sizeof(AffineParams) == 24);
static_assert(sizeof(ConvolutionParams) == 20);
static_assert(sizeof(ReductionParams) == 12);

inline constexpr size_t kReductionThreadCount = 256;

[[nodiscard]] constexpr size_t StrategyIndex(
  AcStrategyType strategy) noexcept {

  return static_cast<size_t>(strategy);
}

// MetalBuffer
class MetalBuffer final : public DeviceBuffer {
public:
  MetalBuffer(
    NS::SharedPtr<MTL::Buffer> buffer,
    BackendId backend_id,
    size_t size_bytes)
    : DeviceBuffer(
      BackendKind::kMetal,
      backend_id,
      size_bytes),
    buffer_(std::move(buffer)) {}

  ~MetalBuffer() override = default;

  [[nodiscard]]
  MTL::Buffer* handle() const noexcept {
    return buffer_.get();
  }

  [[nodiscard]]
  MTL::Device* device() const noexcept {
    return buffer_->device();
  }

  [[nodiscard]]
  void* contents() noexcept {
    return buffer_->contents();
  }

  [[nodiscard]]
  const void* contents() const noexcept {
    return buffer_->contents();
  }

private:
  NS::SharedPtr<MTL::Buffer> buffer_;
};

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
    NS::SharedPtr<MTL::ComputePipelineState>*>, 4> bindings{{
    {"gjxl_pointwise_affine_f32", &pipelines.affine},
    {"gjxl_convolve_horizontal_f32", &pipelines.convolution_horizontal},
    {"gjxl_convolve_vertical_f32", &pipelines.convolution_vertical},
    {"gjxl_reduce_max_f32", &pipelines.maximum_reduction},
  }};
  for (const auto& [name, pipeline] : bindings) {
    Status status = CreatePipeline(device, library, name, pipeline);
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
      pipelines.maximum_reduction->maxTotalThreadsPerThreadgroup() <
        kReductionThreadCount) {
    return Status::Unavailable(
      "Metal cannot launch the required primitive threadgroups");
  }
  *out = std::move(pipelines);
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

// MetalBackend
class MetalBackend final : public GpuBackend {
public:
  MetalBackend(
    NS::SharedPtr<MTL::Device> device,
    NS::SharedPtr<MTL::CommandQueue> command_queue,
    NS::SharedPtr<MTL::Library> library,
    TransformPipelineRegistry transform_pipelines,
    PrimitivePipelines primitive_pipelines,
    bool test_fail_submission,
    bool test_fail_completion)
    : device_(std::move(device)),
      command_queue_(std::move(command_queue)),
      library_(std::move(library)),
      transform_pipelines_(std::move(transform_pipelines)),
      primitive_pipelines_(std::move(primitive_pipelines)),
      test_fail_submission_(test_fail_submission),
      test_fail_completion_(test_fail_completion) {

    NS::String* device_name = device_->name();

    if (device_name != nullptr) {
      const char* utf8 =
        device_name->utf8String();

      if (utf8 != nullptr) {
        name_ = std::string("Metal: ") + utf8;
      }
    }

    if (name_.empty()) {
      name_ = "Metal";
    }
  }

  ~MetalBackend() override {
    // Avoid destroying resources while GPU work is still running.
    (void)Synchronize();
  }

  // Identity
  [[nodiscard]]
  BackendKind kind() const noexcept override {
    return BackendKind::kMetal;
  }

  [[nodiscard]]
  std::string_view name() const noexcept override {
    return name_;
  }

  [[nodiscard]]
  GpuBackendStats stats() const noexcept override {
    return stats_;
  }

  // Memory
  Status Allocate(
    size_t size_bytes,
    std::unique_ptr<DeviceBuffer>* out) override {

    if (out == nullptr) {
      return Status::InvalidArgument(
        "Allocate output pointer is null");
    }

    if (size_bytes == 0) {
      return Status::InvalidArgument(
        "Cannot allocate zero-sized Metal buffer");
    }

    if (size_bytes >
      std::numeric_limits<NS::UInteger>::max()) {

      return Status::InvalidArgument(
        "Requested Metal buffer is too large");
    }

    auto buffer =
      NS::TransferPtr(
        device_->newBuffer(
          static_cast<NS::UInteger>(size_bytes),
          MTL::ResourceStorageModeShared));

    if (!buffer) {
      return Status::OutOfMemory(
        "Metal failed to allocate MTL::Buffer");
    }

    out->reset(
      new MetalBuffer(
        std::move(buffer),
        id(),
        size_bytes));

    ++stats_.successful_allocations;

    return Status::Ok();
  }

  // Host → GPU
  // On Apple Silicon, these are shared buffers, so this is simply a memcpy
  Status CopyHostToDevice(
    DeviceBuffer& dst,
    const void* src,
    size_t size_bytes,
    size_t dst_offset_bytes) override {

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
      static_cast<std::byte*>(
        metal_dst->contents()) +
      dst_offset_bytes;

    std::memcpy(
      destination,
      src,
      size_bytes);

    return Status::Ok();
  }

  // GPU → Host
  Status CopyDeviceToHost(
    const DeviceBuffer& src,
    void* dst,
    size_t size_bytes,
    size_t src_offset_bytes) override {

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
      static_cast<const std::byte*>(
        metal_src->contents()) + src_offset_bytes;

    std::memcpy(
      dst,
      source,
      size_bytes);

    return Status::Ok();
  }

  // Transforms
  Status ForwardTransform(
    const TransformBatch& batch) override {

    return SubmitTransform(
      TransformDirection::kForward,
      batch);
  }

  Status InverseTransform(
    const TransformBatch& batch) override {

    return SubmitTransform(
      TransformDirection::kInverse,
      batch);
  }

  Status SubmitPrimitiveSequence(
    std::span<const PrimitiveCommand> commands) override {

    return SubmitPrimitives(commands);
  }

  // Synchronization
  Status Synchronize() override {
    if (!last_command_buffer_) {
      return Status::Ok();
    }

    last_command_buffer_->waitUntilCompleted();

    if (test_fail_completion_) {
      last_command_buffer_.reset();
      return Status::DeviceError(
        "Injected Metal command-buffer completion failure");
    }

    Status result = Status::Ok();

    if (last_command_buffer_->status() ==
      MTL::CommandBufferStatusError) {

      result =
        metal::ErrorToDeviceStatus(
          last_command_buffer_->error(),
          "Metal command buffer");
    }

    last_command_buffer_.reset();

    return result;
  }


private:
  struct ResolvedConstPlane {
    ConstDevicePlaneView view;
    DeviceMemoryRange range;
    const MetalBuffer* buffer = nullptr;
  };

  struct ResolvedPlane {
    DevicePlaneView view;
    DeviceMemoryRange range;
    MetalBuffer* buffer = nullptr;
  };

  static MetalBuffer* AsMetalBuffer(
    DeviceBuffer& buffer) {

    if (buffer.backend() != BackendKind::kMetal) {
      return nullptr;
    }

    return dynamic_cast<MetalBuffer*>(&buffer);
  }

  static const MetalBuffer* AsMetalBuffer(
    const DeviceBuffer& buffer) {

    if (buffer.backend() !=
      BackendKind::kMetal) {
      return nullptr;
    }

    return dynamic_cast<const MetalBuffer*>(&buffer);
  }

  Status ResolvePlane(
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

  Status ResolvePlane(
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

  [[nodiscard]] static bool SamePlaneLayout(
    ConstDevicePlaneView left,
    ConstDevicePlaneView right) noexcept {

    return left.buffer == right.buffer &&
           left.offset_bytes == right.offset_bytes &&
           left.element_type == right.element_type &&
           left.extent == right.extent &&
           left.row_stride == right.row_stride;
  }

  [[nodiscard]] static Status RejectOverlap(
    DeviceMemoryRange left,
    DeviceMemoryRange right,
    std::string_view message) {

    return DeviceRangesOverlap(left, right)
      ? Status::InvalidArgument(std::string(message))
      : Status::Ok();
  }

  Status ValidatePrimitive(
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

  Status ValidatePrimitive(
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

  Status ValidatePrimitive(
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

  Status ValidatePrimitiveCommand(
    const PrimitiveCommand& command) const {

    return std::visit(
      [this](const auto& concrete) {
        return ValidatePrimitive(concrete);
      },
      command);
  }

  static void DispatchPlane(
    MTL::ComputeCommandEncoder* encoder,
    Extent2D extent) {

    encoder->dispatchThreads(
      MTL::Size(
        static_cast<NS::UInteger>(extent.width),
        static_cast<NS::UInteger>(extent.height),
        1),
      MTL::Size(8, 8, 1));
  }

  void EncodePrimitive(
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

  void EncodeConvolutionPass(
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

  void EncodePrimitive(
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

  void EncodeReductionPass(
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
    encoder->dispatchThreadgroups(
      MTL::Size(static_cast<NS::UInteger>(output_count), 1, 1),
      MTL::Size(kReductionThreadCount, 1, 1));
  }

  void EncodePrimitive(
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

  void EncodePrimitiveCommand(
    MTL::ComputeCommandEncoder* encoder,
    const PrimitiveCommand& command) {

    std::visit(
      [this, encoder](const auto& concrete) {
        EncodePrimitive(encoder, concrete);
      },
      command);
  }

  template <typename Encode>
  Status SubmitCompute(
    const char* label,
    Encode&& encode) {

    if (test_fail_submission_) {
      return Status::SubmissionFailed(
        "Injected Metal submission failure");
    }
    auto pool = NS::TransferPtr(
      NS::AutoreleasePool::alloc()->init());
    MTL::CommandBuffer* raw_command_buffer = command_queue_->commandBuffer();
    if (raw_command_buffer == nullptr) {
      return Status::SubmissionFailed(
        "Failed to create Metal command buffer");
    }
    auto command_buffer = NS::RetainPtr(raw_command_buffer);
    raw_command_buffer->setLabel(NS::String::string(
      label, NS::UTF8StringEncoding));
    MTL::ComputeCommandEncoder* encoder =
      raw_command_buffer->computeCommandEncoder();
    if (encoder == nullptr) {
      return Status::SubmissionFailed(
        "Failed to create Metal compute encoder");
    }
    std::forward<Encode>(encode)(encoder);
    encoder->endEncoding();
    raw_command_buffer->commit();
    last_command_buffer_ = std::move(command_buffer);
    ++stats_.committed_submissions;
    return Status::Ok();
  }

  Status SubmitPrimitives(
    std::span<const PrimitiveCommand> commands) {

    if (commands.empty()) {
      return Status::Ok();
    }
    for (const PrimitiveCommand& command : commands) {
      Status status = ValidatePrimitiveCommand(command);
      if (!status.ok()) {
        return status;
      }
    }
    return SubmitCompute(
      "gjxl primitive sequence",
      [this, commands](MTL::ComputeCommandEncoder* encoder) {
        for (const PrimitiveCommand& command : commands) {
          EncodePrimitiveCommand(encoder, command);
        }
      });
  }

  Status ValidateTransformBatch(
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

    const size_t coefficient_count =
      strategy_info.coefficient_count();

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


  Status SubmitTransform(
    TransformDirection direction,
    const TransformBatch& batch) {

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

    Status status =
      ValidateTransformBatch(
        *strategy_info,
        batch,
        &input,
        &output);

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

    return SubmitCompute(
      pipeline.label.c_str(),
      [&](MTL::ComputeCommandEncoder* encoder) {
        encoder->setComputePipelineState(pipeline.state.get());
        encoder->setBuffer(input->handle(), 0, 0);
        encoder->setBuffer(output->handle(), 0, 1);
        // Exactly one threadgroup per complete transform. The selected
        // implementation determines how many threads cooperate within it.
        encoder->dispatchThreadgroups(
          MTL::Size(
            static_cast<NS::UInteger>(batch.transform_count), 1, 1),
          MTL::Size(pipeline.threads_per_threadgroup, 1, 1));
      });
  }

  NS::SharedPtr<MTL::Device> device_;

  NS::SharedPtr<MTL::CommandQueue> command_queue_;

  NS::SharedPtr<MTL::Library> library_;

  TransformPipelineRegistry transform_pipelines_;

  PrimitivePipelines primitive_pipelines_;

  NS::SharedPtr<MTL::CommandBuffer> last_command_buffer_;

  GpuBackendStats stats_;

  bool test_fail_submission_ = false;

  bool test_fail_completion_ = false;

  std::string name_;
};


}  // namespace

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

  out->reset(
    new MetalBackend(
      std::move(device),
      std::move(command_queue),
      std::move(library),
      std::move(transform_pipelines),
      std::move(primitive_pipelines),
      options.test_fail_submission,
      options.test_fail_completion));

  return Status::Ok();
}

}  // namespace gjxl
