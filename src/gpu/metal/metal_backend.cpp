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

#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/buffer.h"
#include "gpu/metal/metal_status.h"

namespace gjxl {
namespace {

enum class DctDispatchMode {
  kOneThreadPerElement,
  kSingleSimdgroup,
};

struct Dct8ImplementationSpec {
  MetalDct8Implementation implementation;
  std::string_view display_name;
  std::string_view forward_function_name;
  std::string_view inverse_function_name;
  DctDispatchMode dispatch_mode;
};

constexpr std::array<Dct8ImplementationSpec, 2>
kDct8ImplementationSpecs{{
  {
    .implementation = MetalDct8Implementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name =
      "gjxl_dct8_forward_scalar_2d_matmul",
    .inverse_function_name =
      "gjxl_dct8_inverse_scalar_2d_matmul",
    .dispatch_mode = DctDispatchMode::kOneThreadPerElement,
  },
  {
    .implementation = MetalDct8Implementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name =
      "gjxl_dct8_forward_simdgroup_2d_matmul",
    .inverse_function_name =
      "gjxl_dct8_inverse_simdgroup_2d_matmul",
    .dispatch_mode = DctDispatchMode::kSingleSimdgroup,
  },
}};

const Dct8ImplementationSpec* FindDct8ImplementationSpec(
  MetalDct8Implementation implementation) {

  for (const Dct8ImplementationSpec& spec :
       kDct8ImplementationSpecs) {

    if (spec.implementation == implementation) {
      return &spec;
    }
  }

  return nullptr;
}

struct DctPipeline {
  NS::SharedPtr<MTL::ComputePipelineState> state;
  NS::UInteger threads_per_threadgroup = 0;
  size_t elements_per_block = 0;
  std::string label;
};

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

Status CreateDctPipeline(
  MTL::Device* device,
  MTL::Library* library,
  size_t dimension,
  std::string_view implementation_name,
  DctDispatchMode dispatch_mode,
  std::string_view function_name,
  std::string_view operation,
  DctPipeline* out) {

  if (dimension == 0 || out == nullptr) {
    return Status::InvalidArgument(
      "CreateDctPipeline received invalid argument");
  }

  const size_t elements_per_block = dimension * dimension;

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
        " DCT" +
        std::to_string(dimension) +
        " pipeline for " +
        std::string(implementation_name) +
        ": " +
        std::string(status.message()),
    };
  }

  NS::UInteger threads_per_threadgroup = 0;

  switch (dispatch_mode) {
    case DctDispatchMode::kOneThreadPerElement:
      threads_per_threadgroup =
        static_cast<NS::UInteger>(elements_per_block);
      break;

    case DctDispatchMode::kSingleSimdgroup:
      threads_per_threadgroup = state->threadExecutionWidth();
      break;
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
  out->elements_per_block = elements_per_block;
  out->label =
    std::string("gjxl ") +
    std::string(operation) +
    " DCT" +
    std::to_string(dimension) +
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
    DctPipeline forward_dct8,
    DctPipeline inverse_dct8,
    DctPipeline forward_dct16,
    DctPipeline inverse_dct16,
    DctPipeline forward_dct32,
    DctPipeline inverse_dct32)
    : device_(std::move(device)),
      command_queue_(std::move(command_queue)),
      library_(std::move(library)),
      forward_dct8_(std::move(forward_dct8)),
      inverse_dct8_(std::move(inverse_dct8)),
      forward_dct16_(std::move(forward_dct16)),
      inverse_dct16_(std::move(inverse_dct16)),
      forward_dct32_(std::move(forward_dct32)),
      inverse_dct32_(std::move(inverse_dct32)) {

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

  // DCT
  Status ForwardDct8(
    const Dct8Batch& batch) override {

    return SubmitDct(
      forward_dct8_,
      batch);
  }

  Status InverseDct8(
    const Dct8Batch& batch) override {

    return SubmitDct(
      inverse_dct8_,
      batch);
  }

  Status ForwardDct16(
    const Dct16Batch& batch) override {

    return SubmitDct(
      forward_dct16_,
      batch);
  }

  Status InverseDct16(
    const Dct16Batch& batch) override {

    return SubmitDct(
      inverse_dct16_,
      batch);
  }

  Status ForwardDct32(
    const Dct32Batch& batch) override {

    return SubmitDct(
      forward_dct32_,
      batch);
  }

  Status InverseDct32(
    const Dct32Batch& batch) override {

    return SubmitDct(
      inverse_dct32_,
      batch);
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

  template <size_t Dimension>
  Status ValidateDctBatch(
    const SquareDctBatch<Dimension>& batch,
    const MetalBuffer** input,
    MetalBuffer** output) const {

    if (input == nullptr || output == nullptr) {

      return Status::Internal(
        "ValidateDctBatch output is null");
    }

    if (batch.block_count == 0) {
      *input = nullptr;
      *output = nullptr;
      return Status::Ok();
    }

    if (batch.input == nullptr || batch.output == nullptr) {
      return Status::InvalidArgument(
        "DCT input/output buffer is null");
    }

    if (batch.input == batch.output) {
      return Status::InvalidArgument(
        "In-place DCT is not supported yet");
    }

    constexpr size_t kBytesPerBlock =
      SquareDctBatch<Dimension>::kElementsPerBlock * sizeof(float);

    if (batch.block_count >
        std::numeric_limits<size_t>::max() / kBytesPerBlock) {

      return Status::InvalidArgument(
        "DCT batch is too large");
    }

    const size_t required_bytes = batch.block_count * kBytesPerBlock;

    if (batch.input->size_bytes() < required_bytes ||
        batch.output->size_bytes() < required_bytes) {

      return Status::InvalidArgument(
        "DCT buffer is too small");
    }

    *input = AsMetalBuffer(*batch.input);
    *output = AsMetalBuffer(*batch.output);

    if (*input == nullptr || *output == nullptr) {
      return Status::InvalidArgument(
        "DCT buffers are not Metal buffers");
    }

    if ((*input)->device() != device_.get() ||
        (*output)->device() != device_.get()) {

      return Status::InvalidArgument(
        "DCT buffer belongs to another Metal device");
    }

    return Status::Ok();
  }


  template <size_t Dimension>
  Status SubmitDct(
    const DctPipeline& pipeline,
    const SquareDctBatch<Dimension>& batch) {

    if (pipeline.elements_per_block !=
        SquareDctBatch<Dimension>::kElementsPerBlock) {

      return Status::Internal(
        "DCT pipeline dimension does not match batch type");
    }

    const MetalBuffer* input = nullptr;
    MetalBuffer* output = nullptr;

    Status status =
      ValidateDctBatch(
        batch,
        &input,
        &output);

    if (!status.ok()) {
      return status;
    }

    if (batch.block_count == 0) {
      return Status::Ok();
    }

    if (batch.block_count > std::numeric_limits<NS::UInteger>::max()) {
      return Status::InvalidArgument(
        "DCT batch exceeds Metal grid range");
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

    encoder->setComputePipelineState(pipeline.state.get());

    encoder->setBuffer(
      input->handle(),
      0,
      0);

    encoder->setBuffer(
      output->handle(),
      0,
      1);

    // Exactly one threadgroup per square DCT. The selected implementation
    // determines how many threads cooperate within that threadgroup.
    const MTL::Size threadgroups(
      static_cast<NS::UInteger>(
        batch.block_count),
        1,
        1);

    const MTL::Size threads_per_threadgroup(
      pipeline.threads_per_threadgroup,
      1,
      1);

    encoder->dispatchThreadgroups(
      threadgroups,
      threads_per_threadgroup);

    encoder->endEncoding();

    raw_command_buffer->commit();

    // Synchronizing also waits for previously submitted work.
    last_command_buffer_ = std::move(command_buffer);

    return Status::Ok();
  }


  NS::SharedPtr<MTL::Device> device_;

  NS::SharedPtr<MTL::CommandQueue> command_queue_;

  NS::SharedPtr<MTL::Library> library_;

  DctPipeline forward_dct8_;

  DctPipeline inverse_dct8_;

  DctPipeline forward_dct16_;

  DctPipeline inverse_dct16_;

  DctPipeline forward_dct32_;

  DctPipeline inverse_dct32_;

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

  const Dct8ImplementationSpec* forward_dct8_spec =
    FindDct8ImplementationSpec(options.forward_dct8);

  if (forward_dct8_spec == nullptr) {
    return Status::InvalidArgument(
      "Unknown forward Metal DCT8 implementation");
  }

  const Dct8ImplementationSpec* inverse_dct8_spec =
    FindDct8ImplementationSpec(options.inverse_dct8);

  if (inverse_dct8_spec == nullptr) {
    return Status::InvalidArgument(
      "Unknown inverse Metal DCT8 implementation");
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

  DctPipeline forward_dct8;

  Status status =
    CreateDctPipeline(
      device.get(),
      library.get(),
      8,
      forward_dct8_spec->display_name,
      forward_dct8_spec->dispatch_mode,
      forward_dct8_spec->forward_function_name,
      "forward",
      &forward_dct8);

  if (!status.ok()) {
    return status;
  }

  DctPipeline inverse_dct8;

  status =
    CreateDctPipeline(
      device.get(),
      library.get(),
      8,
      inverse_dct8_spec->display_name,
      inverse_dct8_spec->dispatch_mode,
      inverse_dct8_spec->inverse_function_name,
      "inverse",
      &inverse_dct8);

  if (!status.ok()) {
    return status;
  }

  DctPipeline forward_dct16;

  status =
    CreateDctPipeline(
      device.get(),
      library.get(),
      16,
      "scalar matmul",
      DctDispatchMode::kOneThreadPerElement,
      "gjxl_dct16_forward_scalar_2d_matmul",
      "forward",
      &forward_dct16);

  if (!status.ok()) {
    return status;
  }

  DctPipeline inverse_dct16;

  status =
    CreateDctPipeline(
      device.get(),
      library.get(),
      16,
      "scalar matmul",
      DctDispatchMode::kOneThreadPerElement,
      "gjxl_dct16_inverse_scalar_2d_matmul",
      "inverse",
      &inverse_dct16);

  if (!status.ok()) {
    return status;
  }

  DctPipeline forward_dct32;

  status =
    CreateDctPipeline(
      device.get(),
      library.get(),
      32,
      "scalar matmul",
      DctDispatchMode::kOneThreadPerElement,
      "gjxl_dct32_forward_scalar_2d_matmul",
      "forward",
      &forward_dct32);

  if (!status.ok()) {
    return status;
  }

  DctPipeline inverse_dct32;

  status =
    CreateDctPipeline(
      device.get(),
      library.get(),
      32,
      "scalar matmul",
      DctDispatchMode::kOneThreadPerElement,
      "gjxl_dct32_inverse_scalar_2d_matmul",
      "inverse",
      &inverse_dct32);

  if (!status.ok()) {
    return status;
  }

  out->reset(
    new MetalBackend(
      std::move(device),
      std::move(command_queue),
      std::move(library),
      std::move(forward_dct8),
      std::move(inverse_dct8),
      std::move(forward_dct16),
      std::move(inverse_dct16),
      std::move(forward_dct32),
      std::move(inverse_dct32)));

  return Status::Ok();
}

}  // namespace gjxl
