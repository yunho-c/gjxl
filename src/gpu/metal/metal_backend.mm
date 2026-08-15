// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu/metal/metal_backend.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/buffer.h"
#include "gpu/metal/metal_status.h"

namespace gjxl {
namespace {

constexpr size_t kDct8Elements = 64;
constexpr size_t kDct8Bytes = kDct8Elements * sizeof(float);

// MetalBuffer
class MetalBuffer final : public DeviceBuffer {
public:
  explicit MetalBuffer(
    id<MTLBuffer> buffer,
    size_t size_bytes)
    : DeviceBuffer(
        BackendKind::kMetal,
        size_bytes),
      buffer_(buffer) {}

  ~MetalBuffer() override = default;

  [[nodiscard]]
  id<MTLBuffer> handle() const noexcept {
    return buffer_;
  }

  [[nodiscard]]
  id<MTLDevice> device() const noexcept {
    return [buffer_ device];
  }

  [[nodiscard]]
  void* contents() noexcept {
    return [buffer_ contents];
  }

  [[nodiscard]]
  const void* contents() const noexcept {
    return [buffer_ contents];
  }

private:
  // Under ARC this is a strong Objective-C reference.
  id<MTLBuffer> __strong buffer_;
};

// Helper: pipeline creation
Status CreatePipeline(
  id<MTLDevice> device,
  id<MTLLibrary> library,
  NSString* function_name,
  id<MTLComputePipelineState>* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "CreatePipeline output is null");
  }

  id<MTLFunction> function = [library newFunctionWithName:function_name];

  if (function == nil) {
    std::string message = "Metal function not found: ";

    const char* name = [function_name UTF8String];

    if (name != nullptr) {
      message += name;
    }

    return Status::Internal(
      std::move(message));
  }

  NSError* error = nil;

  id<MTLComputePipelineState> pipeline =
    [device
      newComputePipelineStateWithFunction:function
      error:&error];

  if (pipeline == nil) {
    return metal::ErrorToStatus(
      error,
      "newComputePipelineStateWithFunction");
  }

  *out = pipeline;

  return Status::Ok();
}

// MetalBackend
class MetalBackend final : public GpuBackend {
public:
  MetalBackend(
    id<MTLDevice> device,
    id<MTLCommandQueue> command_queue,
    id<MTLLibrary> library,
    id<MTLComputePipelineState> forward_dct8,
    id<MTLComputePipelineState> inverse_dct8)
    : device_(device),
      command_queue_(command_queue),
      library_(library),
      forward_dct8_(forward_dct8),
      inverse_dct8_(inverse_dct8) {

    NSString* metal_name = [device_ name];

    if (metal_name != nil) {
      const char* utf8 = [metal_name UTF8String];

      if (utf8 != nullptr) {
        name_ = std:;string("Metal: ") + utf8;
      }
    }

    if (name_.empty()) {
      name_ = "Metal";
    }
  }

  ~MetalBackend() override {
    // Ensure this object's resources aren't destroyed while the
    // GPU still has commands referencing them.
    (void)Synchronize();
  }

  [[nodiscard]]
  BackendKind kind() const noexcept override {
    return BackendKind::kMetal;
  }

  [[nodiscard]]
  std::string_view name() const noexcept override {
    return name_;
  }

  // Allocation
  Status Allocate(
    size_t size_bytes,
    std::unique_ptr<DeviceBuffer>* out) override {

      if (out == nullptr) {
        return Status::InvalidArgument(
          "Allocate output pointer is null");
      }

      if (size_bytes == 0) {
        return Status::InvalidArgument(
          "Cannot allocate a zero-sized Metal buffer");
      }

      @autoreleasepool {

        id<MTLBuffer> buffer =
          [device_
            newBufferWithLength:size_bytes
            options:MTLResourceStorageModeShared];

        if (buffer == nil) {
          return Status::OutOfMemory(
            "Metal failed to allocate MTLBuffer");
        }

        out->reset(
          new MetalBuffer(
            buffer,
            size_bytes));
      }

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

    if (metal_dst->device() != device_) {
      return Status::InvalidArgument(
        "Destination belongs to another Metal device");
    }

    if (dst_offset_bytes > dst.size_bytes() ||
        size_bytes > dst.size_bytes() - dst_offset_bytes) {

      return Status::InvalidArgument(
        "Host-to-device copy exceeds destination buffer");
    }
    auto* destination =
      static_cast<std::byte*>(metal_dst->contents()) + dst_offset_bytes;

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

    if (metal_src->device() != device_) {
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

      return SubmitDct8(
        forward_dct8_,
        batch,
        "forward DCT8");
    }

    Status InverseDct8(
      const Dct8Batch& batch) override {

      return SubmitDct8(
        inverse_dct8_,
        batch,
        "inverse DCT8");
    }

    // Synchronization
    Status Synchronize() override {
      @autoreleasepool {

        if (last_command_buffer_ == nil) {
          return Status::Ok();
        }

        id<MTLCommandBuffer> command_buffer = last_command_buffer_;

        [command_buffer waitUntilCompleted];

        last_command_buffer_ = nil;

        if ([command_buffer status] == MTLCommandBufferStatusError) {
          return metal::ErrorToStatus(
            [command_buffer error],
            "Metal command buffer");
        }
      }

      return Status::Ok();
    }


private:
  // Buffer conversion
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

  // Validation
  Status ValidateDctBatch(
    const Dct8Batch& batch,
    const MetalBuffer** input,
    MetalBuffer** output) const {

    if (input == nullptr || output == nullptr) {
      return Status::Internal("ValidateDctBatch output is null");
    }

    if (batch.block_count == 0) {
      *input = nullptr;
      *output = nullptr;

      return Status::Ok();
    }

    if (batch.input == nullptr || batch.output == nullptr) {
      return Status::InvalidArgument("DCT input/output buffer is null");
    }

    if (batch.input == batch.output) {
      return Status::InvalidArgument("In-place DCT is not supported yet");
    }

    if (batch.block_count >
      std::numeric_limits<size_t>::max() / kDct8Bytes) {
        return Status::InvalidArgument("DCT batch is too large");
    }

    const size_t required_bytes = batch.block_count * kDct8Bytes;

    if (batch.input->size_bytes() < required_bytes ||
      batch.output->size_bytes() < required_bytes) {
        return Status::InvalidArgument("DCT buffer too small");
    }

    *input = AsMetalBuffer(*batch.input);
    *output = AsMetalBuffer(*batch.output);

    if (*input == nullptr || *output == nullptr) {
      return Status::InvalidArgument("DCT buffers are not Metal buffers");
    }

    if ((*input)->device() != device_ ||
        (*output)->device() != device_) {
        return Status::InvalidArgument(
          "DCT buffer belongs to another Metal device");
    }

    return Status::Ok();
  }

  // Kernel dispatch
  Status SubmitDct8(
    id<MTLComputePipelineState> pipeline,
    const Dct8Batch& batch,
    const char* label) {

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

    // Initial naïve design:
    //   one threadgroup = one 8x8 transform
    //   64 threads      = one thread per pixel
    constexpr NSUInteger kThreadsPerDct = 64;

    if ([pipeline maxTotalThreadsPerThreadgroup] < kThreadsPerDct) {
      return Status::Unavailable(
        "Metal GPU cannot launch 64-thread DCT8 threadgroup");
    }

    if (batch.block_count > std::numeric_limits<NSUInteger>::max()) {
      return Status::InvalidArgument(
        "DCT block count exceeds Metal grid limit");
    }

    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];

      if (command_buffer == nil) {
        return Status::Internal(
          "Failed to create MTLCommandBuffer");
      }

      if (label != nullptr) {
        NSString* ns_label = [NSString stringWithUTF8String:label];

        [command_buffer setLabel:ns_label];
      }

      id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];

      if (encoder == nil) {
        return Status::Internal("Failed to create MTLComputeCommandEncoder");
      }

      [encoder setComputePipelineState:pipeline];

      [encoder
        setBuffer:input->handle()
        offset:0
        atIndex:0];

      [encoder
        setBuffer:output->handle()
        offset:0
        atIndex:1];

      // Exactly one threadgroup per 8x8 DCT.
      //
      // kernel:
      //   thread_index_in_threadgroup -> 0...63
      //   threadgroup_position_in_grid.x -> block index
      MTLSize threadgroups = MTLSizeMake(
        static_cast<NSUInteger>(
          batch.block_count),
        1,
        1);

      MTLSize threads_per_threadgroup =
        MTLSizeMake(
          kThreadsPerDct,
          1,
          1);

      [encoder
        dispatchThreadgroups:threadgroups
        threadsPerThreadgroup:
          threads_per_threadgroup];

      [encoder endEncoding];

      [command_buffer commit];

      // Retain only the newest committed bufffer.
      //
      // Command buffers are submitted through one queue, preserving dependency order.
      last_command_buffer_ = command_buffer;
    }

    return Status::Ok();
  }


  id<MTLDevice> __strong device_;

  id<MTLCommandQueue> __strong command_queue_;

  id<MTLLibrary> __strong library_;

  id<MTLComputePipelineState> __strong forward_dct8_;

  id<MTLComputePipelineState> __strong inverse_dct8_;

  id<MTLCommandBuffer> __strong last_command_buffer_ = nil;

  std::string name_;
};


}  // namespace

// Factory

Status CreateMetalBackend(
  std::string_view metallib_path,
  std::unique_ptr<GpuBackend>* out) {

  if (out == nullptr) {
    return Status::InvalidArgument("CreateMetalBackend output pointer is null");
  }

  if (metallib_path.empty()) {
    return Status::InvalidArgument("Metal library path is empty");
  }

  @autoreleasepool {

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (device == nil) {
      return Status::Unavailable("No Metal device is available");
    }

    id<MTLCommandQueue> command_queue = [device newCommandQueue];

    if (command_queue == nil) {
      return Status::Internal("Failed to create MTLCommandQueue");
    }

    std::string path_string(metallib_path);

    NSString* ns_path =
      [NSString
        stringWithUTF8String:
          path_string.c_str()];

    if (ns_path == nil) {
      return Status::InvalidArgument(
        "Metal library path is not valid UTF-8");
    }

    NSURL* library_url =
      [NSURL fileURLWithPath:ns_path];

    NSError* error = nil;

    id<MTLLibrary> library =
      [device
        newLibraryWithURL:library_url
        error:&error];

    if (library == nil) {
      return metal::ErrorToStatus(
        error,
        "newLibraryWithURL");
    }

    id<MTLComputePipelineState> forward_dct8 = nil;

    Status status =
      CreatePipeline(
        device,
        library,
        @"gjxl_forward_dct8",
        &forward_dct8);

    if (!status.ok()) {
      return status;
    }

    id<MTLComputePipelineState> inverse_dct8 = nil;

    status =
      CreatePipeline(
        device,
        library,
        @"gjxl_inverse_dct8",
        &inverse_dct8);

    if (!status.ok()) {
      return status;
    }

    out->reset(
      new MetalBackend(
        device,
        command_queue,
        library,
        forward_dct8,
        inverse_dct8));
  }

  return Status::Ok();
}

}  // namespace gjxl
