#include "gpu/metal/metal_backend.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/buffer.h"
#include "gpu/metal/metal_status.h"
#include "gpu/ops/ac_strategy.h"

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
  size_t transforms_per_threadgroup = 1;
};

struct DctSelection {
  AcStrategyType strategy;
  MetalDctImplementation forward;
  MetalDctImplementation inverse;
};

constexpr std::array<DctImplementationSpec, 21>
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
    .strategy = AcStrategyType::kDct8,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct8_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct8_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
    .transforms_per_threadgroup = 4,
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
    .strategy = AcStrategyType::kDct16x16,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct16_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct16_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
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
    .strategy = AcStrategyType::kDct32x32,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct32_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct32_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
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
    .strategy = AcStrategyType::kDct16x8,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct16x8_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct16x8_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
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
    .strategy = AcStrategyType::kDct8x16,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct8x16_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct8x16_inverse_factored_radix2",
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
    .strategy = AcStrategyType::kDct32x16,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct32x16_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct32x16_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
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
  {
    .strategy = AcStrategyType::kDct16x32,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct16x32_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct16x32_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
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
  size_t transforms_per_threadgroup = 1;
  AcStrategyType strategy = AcStrategyType::kCount;
  std::string label;
};

struct TransformPipelinePair {
  TransformPipeline forward;
  TransformPipeline inverse;
};

using TransformPipelineRegistry =
  std::array<TransformPipelinePair, kAcStrategyCount>;

struct AcStrategyPipelines {
  NS::SharedPtr<MTL::ComputePipelineState> gather;
  NS::SharedPtr<MTL::ComputePipelineState> residual;
  NS::SharedPtr<MTL::ComputePipelineState> cost;
  NS::UInteger gather_threads_per_threadgroup = 0;
};

struct MetalAcStrategyBatchParams {
  uint32_t pixel_width;
  uint32_t pixel_height;
  uint32_t opsin_row_stride;
  uint32_t opsin_plane_stride;
  uint32_t pixel_mask_row_stride;
  uint32_t candidate_count;
  uint32_t coefficient_count;
  uint32_t transform_width;
  uint32_t transform_height;
  uint32_t covered_block_count;
  float info_loss_multiplier;
  float zeros_multiplier;
  float cost_delta;
};

static_assert(std::is_standard_layout_v<MetalAcStrategyBatchParams>);
static_assert(sizeof(MetalAcStrategyBatchParams) == 13 * sizeof(uint32_t));

[[nodiscard]] constexpr size_t StrategyIndex(
  AcStrategyType strategy) noexcept {

  return static_cast<size_t>(strategy);
}

// MetalBuffer
class MetalBuffer final : public DeviceBuffer {
public:
  MetalBuffer(
    NS::SharedPtr<MTL::Buffer> buffer,
    size_t size_bytes)
    : DeviceBuffer(
      BackendKind::kMetal,
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

Status CreateTransformPipeline(
  MTL::Device* device,
  MTL::Library* library,
  AcStrategyType strategy,
  std::string_view implementation_name,
  TransformDispatchMode dispatch_mode,
  size_t simdgroups_per_threadgroup,
  size_t transforms_per_threadgroup,
  std::string_view function_name,
  std::string_view operation,
  TransformPipeline* out) {

  const AcStrategyInfo* strategy_info =
    GetAcStrategyInfo(strategy);

  if (strategy_info == nullptr || transforms_per_threadgroup == 0 ||
      out == nullptr) {
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
  out->transforms_per_threadgroup = transforms_per_threadgroup;
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

Status CreateAcStrategyPipelines(
  MTL::Device* device,
  MTL::Library* library,
  AcStrategyPipelines* out) {

  if (device == nullptr || library == nullptr || out == nullptr) {
    return Status::InvalidArgument(
      "CreateAcStrategyPipelines received invalid argument");
  }

  AcStrategyPipelines pipelines;
  Status status = CreatePipeline(
    device,
    library,
    "gjxl_ac_strategy_gather",
    &pipelines.gather);
  if (!status.ok()) {
    return status;
  }
  status = CreatePipeline(
    device,
    library,
    "gjxl_ac_strategy_residual",
    &pipelines.residual);
  if (!status.ok()) {
    return status;
  }
  status = CreatePipeline(
    device,
    library,
    "gjxl_ac_strategy_cost",
    &pipelines.cost);
  if (!status.ok()) {
    return status;
  }

  constexpr NS::UInteger kPreferredGatherThreads = 256;
  const NS::UInteger execution_width =
    pipelines.gather->threadExecutionWidth();
  const NS::UInteger maximum_threads =
    pipelines.gather->maxTotalThreadsPerThreadgroup();
  if (execution_width == 0 || maximum_threads < execution_width) {
    return Status::Unavailable(
      "Metal reported invalid AC-strategy gather dispatch data");
  }
  const NS::UInteger capped_threads = std::min(
    maximum_threads,
    kPreferredGatherThreads);
  pipelines.gather_threads_per_threadgroup =
    (capped_threads / execution_width) * execution_width;
  if (pipelines.gather_threads_per_threadgroup == 0) {
    return Status::Unavailable(
      "Metal cannot launch the AC-strategy gather kernel");
  }

  *out = std::move(pipelines);
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
    AcStrategyPipelines ac_strategy_pipelines)
    : device_(std::move(device)),
      command_queue_(std::move(command_queue)),
      library_(std::move(library)),
      transform_pipelines_(std::move(transform_pipelines)),
      ac_strategy_pipelines_(std::move(ac_strategy_pipelines)) {

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
        size_bytes));

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

    if (metal_dst->device() != device_.get()) {
      return Status::InvalidArgument(
        "Destination belongs to another Metal device");
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

    if (metal_src->device() != device_.get()) {
      return Status::InvalidArgument(
        "Source belongs to another Metal device");
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

  Status EvaluateAcStrategyCandidates(
    const AcStrategyCandidateBatch& batch) override {

    return SubmitAcStrategyCandidates(
      std::span<const AcStrategyCandidateBatch>(&batch, 1));
  }

  Status EvaluateAcStrategyCandidateBatches(
    std::span<const AcStrategyCandidateBatch> batches) override {

    return SubmitAcStrategyCandidates(batches);
  }

  // Synchronization
  Status Synchronize() override {
    if (!last_command_buffer_) {
      return Status::Ok();
    }

    last_command_buffer_->waitUntilCompleted();

    Status result = Status::Ok();

    if (last_command_buffer_->status() ==
      MTL::CommandBufferStatusError) {

      result =
        metal::ErrorToStatus(
          last_command_buffer_->error(),
          "Metal command buffer");
    }

    last_command_buffer_.reset();

    return result;
  }


private:
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

  struct ValidatedAcStrategyBatch {
    const MetalBuffer* opsin = nullptr;
    const MetalBuffer* pixel_mask = nullptr;
    const MetalBuffer* matrices = nullptr;
    const MetalBuffer* candidates = nullptr;
    MetalBuffer* scratch_a = nullptr;
    MetalBuffer* scratch_b = nullptr;
    MetalBuffer* rate_scratch = nullptr;
    MetalBuffer* costs = nullptr;
    const TransformPipeline* forward = nullptr;
    const TransformPipeline* inverse = nullptr;
    MetalAcStrategyBatchParams params{};
    size_t transform_count = 0;
    size_t packed_element_count = 0;
    size_t forward_threadgroup_count = 0;
    size_t inverse_threadgroup_count = 0;
  };

  [[nodiscard]] static bool TryMultiply(
    size_t left,
    size_t right,
    size_t* result) noexcept {

    if (result == nullptr ||
        (right != 0 &&
         left > std::numeric_limits<size_t>::max() / right)) {
      return false;
    }
    *result = left * right;
    return true;
  }

  static Status TransformThreadgroupCount(
    const TransformPipeline& pipeline,
    size_t transform_count,
    size_t* out) {

    if (out == nullptr || pipeline.transforms_per_threadgroup == 0) {
      return Status::Internal(
        "Transform pipeline has an invalid packing factor");
    }
    if (transform_count == 0) {
      *out = 0;
      return Status::Ok();
    }
    if (pipeline.transforms_per_threadgroup > 1 &&
        transform_count > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument(
        "Packed transform batch exceeds Metal kernel range");
    }

    const size_t threadgroup_count =
      1 +
      (transform_count - 1) /
        pipeline.transforms_per_threadgroup;
    if (threadgroup_count >
        std::numeric_limits<NS::UInteger>::max()) {
      return Status::InvalidArgument(
        "Transform batch exceeds Metal grid range");
    }
    *out = threadgroup_count;
    return Status::Ok();
  }

  Status RequireMetalBuffer(
    const DeviceBuffer* buffer,
    size_t required_bytes,
    std::string_view role,
    const MetalBuffer** out) const {

    if (buffer == nullptr || out == nullptr) {
      return Status::InvalidArgument(
        std::string(role) + " buffer is null");
    }
    if (buffer->size_bytes() < required_bytes) {
      return Status::InvalidArgument(
        std::string(role) + " buffer is too small");
    }
    const MetalBuffer* metal_buffer = AsMetalBuffer(*buffer);
    if (metal_buffer == nullptr || metal_buffer->device() != device_.get()) {
      return Status::InvalidArgument(
        std::string(role) + " buffer does not belong to this Metal device");
    }
    *out = metal_buffer;
    return Status::Ok();
  }

  Status RequireMetalBuffer(
    DeviceBuffer* buffer,
    size_t required_bytes,
    std::string_view role,
    MetalBuffer** out) const {

    const MetalBuffer* validated = nullptr;
    Status status = RequireMetalBuffer(
      static_cast<const DeviceBuffer*>(buffer),
      required_bytes,
      role,
      &validated);
    if (!status.ok()) {
      return status;
    }
    *out = const_cast<MetalBuffer*>(validated);
    return Status::Ok();
  }

  Status ValidateAcStrategyCandidateBatch(
    const AcStrategyCandidateBatch& batch,
    ValidatedAcStrategyBatch* out) const {

    if (out == nullptr) {
      return Status::Internal(
        "AC-strategy batch validation output is null");
    }
    const AcStrategyInfo* strategy_info =
      GetAcStrategyInfo(batch.strategy);
    if (strategy_info == nullptr) {
      return Status::InvalidArgument(
        "Unknown JPEG XL AC strategy");
    }

    const TransformPipelinePair& transform_pair =
      transform_pipelines_[StrategyIndex(batch.strategy)];
    if (!transform_pair.forward.state || !transform_pair.inverse.state) {
      return Status::Unavailable(
        std::string("Metal candidate evaluation does not support ") +
        std::string(strategy_info->name));
    }
    if (batch.candidate_count == 0) {
      *out = {};
      return Status::Ok();
    }

    const Extent2D transform_extent = strategy_info->pixel_extent();
    if (batch.pixel_extent.empty() ||
        batch.pixel_extent.width % kJxlBlockDimension != 0 ||
        batch.pixel_extent.height % kJxlBlockDimension != 0 ||
        transform_extent.width > batch.pixel_extent.width ||
        transform_extent.height > batch.pixel_extent.height ||
        batch.opsin_row_stride < batch.pixel_extent.width ||
        batch.pixel_mask_row_stride < batch.pixel_extent.width) {
      return Status::InvalidArgument(
        "AC-strategy batch image geometry is invalid");
    }

    size_t minimum_plane_stride = 0;
    if (!TryMultiply(
          batch.opsin_row_stride,
          batch.pixel_extent.height,
          &minimum_plane_stride) ||
        batch.opsin_plane_stride < minimum_plane_stride) {
      return Status::InvalidArgument(
        "AC-strategy batch opsin strides are invalid");
    }
    if (!std::isfinite(batch.butteraugli_target) ||
        batch.butteraugli_target <= 0.0f) {
      return Status::InvalidArgument(
        "AC-strategy batch Butteraugli target is invalid");
    }

    constexpr size_t kUint32Maximum =
      std::numeric_limits<uint32_t>::max();
    const std::array<size_t, 8> uint32_values = {
      batch.pixel_extent.width,
      batch.pixel_extent.height,
      batch.opsin_row_stride,
      batch.opsin_plane_stride,
      batch.pixel_mask_row_stride,
      batch.candidate_count,
      transform_extent.width,
      transform_extent.height,
    };
    if (std::ranges::any_of(
          uint32_values,
          [](size_t value) { return value > kUint32Maximum; })) {
      return Status::InvalidArgument(
        "AC-strategy batch exceeds Metal's 32-bit indexing range");
    }

    const size_t coefficient_count = strategy_info->coefficient_count();
    size_t transform_count = 0;
    size_t packed_element_count = 0;
    size_t packed_bytes = 0;
    size_t opsin_floats = 0;
    size_t opsin_bytes = 0;
    size_t mask_floats = 0;
    size_t mask_bytes = 0;
    size_t matrix_floats = 0;
    size_t matrix_bytes = 0;
    size_t candidate_bytes = 0;
    size_t rate_channels = 0;
    size_t rate_bytes = 0;
    size_t cost_bytes = 0;
    if (!TryMultiply(
          batch.candidate_count,
          kAcStrategyCandidateChannelCount,
          &transform_count) ||
        !TryMultiply(
          transform_count,
          coefficient_count,
          &packed_element_count) ||
        !TryMultiply(packed_element_count, sizeof(float), &packed_bytes) ||
        !TryMultiply(
          batch.opsin_plane_stride,
          kAcStrategyCandidateChannelCount,
          &opsin_floats) ||
        !TryMultiply(opsin_floats, sizeof(float), &opsin_bytes) ||
        !TryMultiply(
          batch.pixel_mask_row_stride,
          batch.pixel_extent.height,
          &mask_floats) ||
        !TryMultiply(mask_floats, sizeof(float), &mask_bytes) ||
        !TryMultiply(
          coefficient_count,
          kAcStrategyCostMatrixCount,
          &matrix_floats) ||
        !TryMultiply(matrix_floats, sizeof(float), &matrix_bytes) ||
        !TryMultiply(
          batch.candidate_count,
          sizeof(AcStrategyCandidate),
          &candidate_bytes) ||
        !TryMultiply(
          batch.candidate_count,
          kAcStrategyCandidateChannelCount,
          &rate_channels) ||
        !TryMultiply(
          rate_channels,
          kAcStrategyRateScratchBytesPerChannel,
          &rate_bytes) ||
        !TryMultiply(batch.candidate_count, sizeof(float), &cost_bytes) ||
        transform_count > kUint32Maximum ||
        packed_element_count > kUint32Maximum) {
      return Status::InvalidArgument(
        "AC-strategy batch buffer size overflows");
    }

    const std::array<const DeviceBuffer*, 4> inputs = {
      batch.opsin,
      batch.pixel_mask,
      batch.matrices,
      batch.candidates,
    };
    const std::array<DeviceBuffer*, 4> outputs = {
      batch.scratch_a,
      batch.scratch_b,
      batch.rate_scratch,
      batch.costs,
    };
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i] == nullptr) {
        return Status::InvalidArgument(
          "AC-strategy batch output buffer is null");
      }
      for (size_t j = i + 1; j < outputs.size(); ++j) {
        if (outputs[i] == outputs[j]) {
          return Status::InvalidArgument(
            "AC-strategy batch output buffers must not alias");
        }
      }
      if (std::ranges::find(inputs, outputs[i]) != inputs.end()) {
        return Status::InvalidArgument(
          "AC-strategy batch input and output buffers must not alias");
      }
    }

    ValidatedAcStrategyBatch validated;
    Status status = RequireMetalBuffer(
      batch.opsin, opsin_bytes, "Opsin", &validated.opsin);
    if (!status.ok()) {
      return status;
    }
    status = RequireMetalBuffer(
      batch.pixel_mask, mask_bytes, "Pixel mask", &validated.pixel_mask);
    if (!status.ok()) {
      return status;
    }
    status = RequireMetalBuffer(
      batch.matrices, matrix_bytes, "Quantization matrix", &validated.matrices);
    if (!status.ok()) {
      return status;
    }
    status = RequireMetalBuffer(
      batch.candidates, candidate_bytes, "Candidate", &validated.candidates);
    if (!status.ok()) {
      return status;
    }
    status = RequireMetalBuffer(
      batch.scratch_a, packed_bytes, "Scratch A", &validated.scratch_a);
    if (!status.ok()) {
      return status;
    }
    status = RequireMetalBuffer(
      batch.scratch_b, packed_bytes, "Scratch B", &validated.scratch_b);
    if (!status.ok()) {
      return status;
    }
    status = RequireMetalBuffer(
      batch.rate_scratch, rate_bytes, "Rate scratch", &validated.rate_scratch);
    if (!status.ok()) {
      return status;
    }
    status = RequireMetalBuffer(
      batch.costs, cost_bytes, "Cost", &validated.costs);
    if (!status.ok()) {
      return status;
    }

    if (ac_strategy_pipelines_.residual->maxTotalThreadsPerThreadgroup() <
          coefficient_count ||
        ac_strategy_pipelines_.cost->maxTotalThreadsPerThreadgroup() <
          coefficient_count) {
      return Status::Unavailable(
        "Metal cannot launch the required AC-strategy threadgroup");
    }

    constexpr float kBias = 0.13731742964354549f;
    const float ratio =
      (batch.butteraugli_target + kBias) / (1.0f + kBias);
    validated.params = {
      .pixel_width = static_cast<uint32_t>(batch.pixel_extent.width),
      .pixel_height = static_cast<uint32_t>(batch.pixel_extent.height),
      .opsin_row_stride = static_cast<uint32_t>(batch.opsin_row_stride),
      .opsin_plane_stride = static_cast<uint32_t>(batch.opsin_plane_stride),
      .pixel_mask_row_stride =
        static_cast<uint32_t>(batch.pixel_mask_row_stride),
      .candidate_count = static_cast<uint32_t>(batch.candidate_count),
      .coefficient_count = static_cast<uint32_t>(coefficient_count),
      .transform_width = static_cast<uint32_t>(transform_extent.width),
      .transform_height = static_cast<uint32_t>(transform_extent.height),
      .covered_block_count = static_cast<uint32_t>(
        strategy_info->covered_blocks.width *
        strategy_info->covered_blocks.height),
      .info_loss_multiplier = 1.2f * std::pow(
        ratio,
        0.33677806662454718f),
      .zeros_multiplier = 9.3089059022677905f * std::pow(
        ratio,
        0.50990926717963703f),
      .cost_delta = 10.833273317067883f * std::pow(
        ratio,
        0.36702940662370243f),
    };
    validated.forward = &transform_pair.forward;
    validated.inverse = &transform_pair.inverse;
    validated.transform_count = transform_count;
    validated.packed_element_count = packed_element_count;
    status = TransformThreadgroupCount(
      *validated.forward,
      transform_count,
      &validated.forward_threadgroup_count);
    if (!status.ok()) {
      return status;
    }
    status = TransformThreadgroupCount(
      *validated.inverse,
      transform_count,
      &validated.inverse_threadgroup_count);
    if (!status.ok()) {
      return status;
    }
    *out = validated;
    return Status::Ok();
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

    if ((*input)->device() != device_.get() ||
        (*output)->device() != device_.get()) {

      return Status::InvalidArgument(
        "Transform buffer belongs to another Metal device");
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

    size_t threadgroup_count = 0;
    status = TransformThreadgroupCount(
      pipeline,
      batch.transform_count,
      &threadgroup_count);
    if (!status.ok()) {
      return status;
    }

    // metal-cpp uses autorelease for commandBuffer() & computeCommandEncoder()
    auto pool =
      NS::TransferPtr(
        NS::AutoreleasePool::alloc()->init());

    MTL::CommandBuffer* raw_command_buffer = command_queue_->commandBuffer();

    if (raw_command_buffer == nullptr) {
      return Status::Internal(
        "Failed to create Metal command buffer");
    }

    // commandBuffer() is borrowed/autoreleased. Keep our own reference
    // so Synchronize() can use it safely after this function returns.
    auto command_buffer =
      NS::RetainPtr(
        raw_command_buffer);

    raw_command_buffer->setLabel(
      NS::String::string(
        pipeline.label.c_str(),
        NS::UTF8StringEncoding));

    MTL::ComputeCommandEncoder* encoder =
      raw_command_buffer->computeCommandEncoder();

    if (encoder == nullptr) {
      return Status::Internal(
        "Failed to create Metal compute encoder");
    }

    EncodeTransformDispatch(
      encoder,
      pipeline,
      *input,
      *output,
      batch.transform_count,
      threadgroup_count);

    encoder->endEncoding();

    raw_command_buffer->commit();

    // Synchronizing also waits for previously submitted work.
    last_command_buffer_ = std::move(command_buffer);

    return Status::Ok();
  }

  static void EncodeTransformDispatch(
    MTL::ComputeCommandEncoder* encoder,
    const TransformPipeline& pipeline,
    const MetalBuffer& input,
    MetalBuffer& output,
    size_t transform_count,
    size_t threadgroup_count) {

    encoder->setComputePipelineState(pipeline.state.get());
    encoder->setBuffer(input.handle(), 0, 0);
    encoder->setBuffer(output.handle(), 0, 1);
    if (pipeline.transforms_per_threadgroup > 1) {
      const std::uint32_t packed_transform_count =
        static_cast<std::uint32_t>(transform_count);
      encoder->setBytes(
        &packed_transform_count,
        sizeof(packed_transform_count),
        2);
    }
    encoder->dispatchThreadgroups(
      MTL::Size(
        static_cast<NS::UInteger>(threadgroup_count),
        1,
        1),
      MTL::Size(pipeline.threads_per_threadgroup, 1, 1));
  }

  void EncodeAcStrategyCandidateBatch(
    MTL::ComputeCommandEncoder* encoder,
    const ValidatedAcStrategyBatch& validated) {

    encoder->setComputePipelineState(ac_strategy_pipelines_.gather.get());
    encoder->setBuffer(validated.opsin->handle(), 0, 0);
    encoder->setBuffer(validated.candidates->handle(), 0, 1);
    encoder->setBuffer(validated.scratch_a->handle(), 0, 2);
    encoder->setBytes(
      &validated.params,
      sizeof(validated.params),
      3);
    encoder->dispatchThreads(
      MTL::Size(
        static_cast<NS::UInteger>(validated.packed_element_count),
        1,
        1),
      MTL::Size(
        ac_strategy_pipelines_.gather_threads_per_threadgroup,
        1,
        1));

    EncodeTransformDispatch(
      encoder,
      *validated.forward,
      *validated.scratch_a,
      *validated.scratch_b,
      validated.transform_count,
      validated.forward_threadgroup_count);

    encoder->setComputePipelineState(ac_strategy_pipelines_.residual.get());
    encoder->setBuffer(validated.scratch_b->handle(), 0, 0);
    encoder->setBuffer(validated.matrices->handle(), 0, 1);
    encoder->setBuffer(validated.candidates->handle(), 0, 2);
    encoder->setBuffer(validated.scratch_a->handle(), 0, 3);
    encoder->setBuffer(validated.rate_scratch->handle(), 0, 4);
    encoder->setBytes(
      &validated.params,
      sizeof(validated.params),
      5);
    encoder->dispatchThreadgroups(
      MTL::Size(
        static_cast<NS::UInteger>(validated.transform_count),
        1,
        1),
      MTL::Size(validated.params.coefficient_count, 1, 1));

    EncodeTransformDispatch(
      encoder,
      *validated.inverse,
      *validated.scratch_a,
      *validated.scratch_b,
      validated.transform_count,
      validated.inverse_threadgroup_count);

    encoder->setComputePipelineState(ac_strategy_pipelines_.cost.get());
    encoder->setBuffer(validated.scratch_b->handle(), 0, 0);
    encoder->setBuffer(validated.pixel_mask->handle(), 0, 1);
    encoder->setBuffer(validated.candidates->handle(), 0, 2);
    encoder->setBuffer(validated.rate_scratch->handle(), 0, 3);
    encoder->setBuffer(validated.costs->handle(), 0, 4);
    encoder->setBytes(
      &validated.params,
      sizeof(validated.params),
      5);
    encoder->dispatchThreadgroups(
      MTL::Size(
        static_cast<NS::UInteger>(validated.params.candidate_count),
        1,
        1),
      MTL::Size(validated.params.coefficient_count, 1, 1));
  }

  Status SubmitAcStrategyCandidates(
    std::span<const AcStrategyCandidateBatch> batches) {

    std::vector<ValidatedAcStrategyBatch> validated_batches;
    try {
      validated_batches.reserve(batches.size());
      for (const AcStrategyCandidateBatch& batch : batches) {
        ValidatedAcStrategyBatch validated;
        Status status = ValidateAcStrategyCandidateBatch(batch, &validated);
        if (!status.ok()) {
          return status;
        }
        if (batch.candidate_count != 0) {
          validated_batches.push_back(validated);
        }
      }
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to validate AC-strategy candidate batches");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Too many AC-strategy candidate batches");
    }

    if (validated_batches.empty()) {
      return Status::Ok();
    }

    auto pool = NS::TransferPtr(
      NS::AutoreleasePool::alloc()->init());
    MTL::CommandBuffer* raw_command_buffer = command_queue_->commandBuffer();
    if (raw_command_buffer == nullptr) {
      return Status::Internal(
        "Failed to create Metal command buffer");
    }
    auto command_buffer = NS::RetainPtr(raw_command_buffer);
    raw_command_buffer->setLabel(NS::String::string(
      "gjxl staged AC candidate evaluation",
      NS::UTF8StringEncoding));

    MTL::ComputeCommandEncoder* encoder =
      raw_command_buffer->computeCommandEncoder();
    if (encoder == nullptr) {
      return Status::Internal(
        "Failed to create Metal compute encoder");
    }

    for (const ValidatedAcStrategyBatch& validated : validated_batches) {
      EncodeAcStrategyCandidateBatch(encoder, validated);
    }

    encoder->endEncoding();
    raw_command_buffer->commit();
    last_command_buffer_ = std::move(command_buffer);
    return Status::Ok();
  }

  NS::SharedPtr<MTL::Device> device_;

  NS::SharedPtr<MTL::CommandQueue> command_queue_;

  NS::SharedPtr<MTL::Library> library_;

  TransformPipelineRegistry transform_pipelines_;

  AcStrategyPipelines ac_strategy_pipelines_;

  NS::SharedPtr<MTL::CommandBuffer> last_command_buffer_;

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
        forward_spec->transforms_per_threadgroup,
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
        inverse_spec->transforms_per_threadgroup,
        inverse_spec->inverse_function_name,
        "inverse",
        &pipelines.inverse);

    if (!status.ok()) {
      return status;
    }
  }

  AcStrategyPipelines ac_strategy_pipelines;
  status = CreateAcStrategyPipelines(
    device.get(),
    library.get(),
    &ac_strategy_pipelines);
  if (!status.ok()) {
    return {
      status.code(),
      std::string("Failed to create AC-strategy pipelines: ") +
        std::string(status.message()),
    };
  }

  out->reset(
    new MetalBackend(
      std::move(device),
      std::move(command_queue),
      std::move(library),
      std::move(transform_pipelines),
      std::move(ac_strategy_pipelines)));

  return Status::Ok();
}

}  // namespace gjxl
