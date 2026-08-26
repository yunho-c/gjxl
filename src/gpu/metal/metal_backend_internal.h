// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"
#include "gpu/ops/primitives.h"

namespace gjxl::metal_internal {

enum class TransformDirection {
  kForward,
  kInverse,
};

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

class MetalBuffer final : public DeviceBuffer {
public:
  MetalBuffer(
    NS::SharedPtr<MTL::Buffer> buffer,
    BackendId backend_id,
    size_t size_bytes)
    : DeviceBuffer(BackendKind::kMetal, backend_id, size_bytes),
      buffer_(std::move(buffer)) {}

  ~MetalBuffer() override = default;

  [[nodiscard]] MTL::Buffer* handle() const noexcept {
    return buffer_.get();
  }

  [[nodiscard]] MTL::Device* device() const noexcept {
    return buffer_->device();
  }

  [[nodiscard]] void* contents() noexcept {
    return buffer_->contents();
  }

  [[nodiscard]] const void* contents() const noexcept {
    return buffer_->contents();
  }

private:
  NS::SharedPtr<MTL::Buffer> buffer_;
};

class MetalBackend final : public GpuBackend, public GpuImagePrimitives {
public:
  MetalBackend(
    NS::SharedPtr<MTL::Device> device,
    NS::SharedPtr<MTL::CommandQueue> command_queue,
    NS::SharedPtr<MTL::Library> library,
    TransformPipelineRegistry transform_pipelines,
    PrimitivePipelines primitive_pipelines,
    bool test_fail_submission,
    bool test_fail_completion);

  ~MetalBackend() override = default;

  [[nodiscard]] BackendKind kind() const noexcept override;
  [[nodiscard]] std::string_view name() const noexcept override;

  Status Allocate(
    size_t size_bytes,
    std::unique_ptr<DeviceBuffer>* out) override;

  Status CopyHostToDevice(
    DeviceBuffer& dst,
    const void* src,
    size_t size_bytes,
    size_t dst_offset_bytes) override;

  Status CopyDeviceToHost(
    const DeviceBuffer& src,
    void* dst,
    size_t size_bytes,
    size_t src_offset_bytes) override;

  Status ForwardTransform(
    const TransformBatch& batch,
    std::unique_ptr<GpuSubmission>* submission) override;

  Status InverseTransform(
    const TransformBatch& batch,
    std::unique_ptr<GpuSubmission>* submission) override;

  Status SubmitImagePrimitiveSequence(
    std::span<const ImagePrimitiveCommand> commands,
    std::unique_ptr<GpuSubmission>* submission) override;

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

  using ComputeEncodeCallback = void (*)(
    MetalBackend&,
    MTL::ComputeCommandEncoder*,
    const void*);

  static MetalBuffer* AsMetalBuffer(DeviceBuffer& buffer);
  static const MetalBuffer* AsMetalBuffer(const DeviceBuffer& buffer);

  Status SubmitCompute(
    const char* label,
    ComputeEncodeCallback encode,
    const void* context,
    std::unique_ptr<GpuSubmission>* submission);

  Status ValidateTransformBatch(
    const AcStrategyInfo& strategy_info,
    const TransformBatch& batch,
    const MetalBuffer** input,
    MetalBuffer** output) const;

  Status SubmitTransform(
    TransformDirection direction,
    const TransformBatch& batch,
    std::unique_ptr<GpuSubmission>* submission);

  static void EncodeTransformSubmission(
    MetalBackend& backend,
    MTL::ComputeCommandEncoder* encoder,
    const void* context);

  Status ResolvePlane(
    ConstDevicePlaneView view,
    ResolvedConstPlane* out) const;

  Status ResolvePlane(
    DevicePlaneView view,
    ResolvedPlane* out) const;

  [[nodiscard]] static bool SamePlaneLayout(
    ConstDevicePlaneView left,
    ConstDevicePlaneView right) noexcept;

  [[nodiscard]] static Status RejectOverlap(
    DeviceMemoryRange left,
    DeviceMemoryRange right,
    std::string_view message);

  Status ValidatePrimitive(const PointwiseAffineCommand& command) const;
  Status ValidatePrimitive(const SeparableConvolutionCommand& command) const;
  Status ValidatePrimitive(const MaximumReductionCommand& command) const;
  Status ValidatePrimitiveCommand(
    const ImagePrimitiveCommand& command) const;

  static void DispatchPlane(
    MTL::ComputeCommandEncoder* encoder,
    Extent2D extent);

  void EncodePrimitive(
    MTL::ComputeCommandEncoder* encoder,
    const PointwiseAffineCommand& command);

  void EncodeConvolutionPass(
    MTL::ComputeCommandEncoder* encoder,
    MTL::ComputePipelineState* pipeline,
    ConstDevicePlaneView input_view,
    ConstDevicePlaneView kernel_view,
    DevicePlaneView output_view);

  void EncodePrimitive(
    MTL::ComputeCommandEncoder* encoder,
    const SeparableConvolutionCommand& command);

  void EncodeReductionPass(
    MTL::ComputeCommandEncoder* encoder,
    ConstDevicePlaneView input_view,
    size_t input_count,
    DevicePlaneView output_view);

  void EncodePrimitive(
    MTL::ComputeCommandEncoder* encoder,
    const MaximumReductionCommand& command);

  void EncodePrimitiveCommand(
    MTL::ComputeCommandEncoder* encoder,
    const ImagePrimitiveCommand& command);

  static void EncodePrimitiveSubmission(
    MetalBackend& backend,
    MTL::ComputeCommandEncoder* encoder,
    const void* context);

  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
  NS::SharedPtr<MTL::Library> library_;
  TransformPipelineRegistry transform_pipelines_;
  PrimitivePipelines primitive_pipelines_;
  bool test_fail_submission_ = false;
  bool test_fail_completion_ = false;
  std::string name_;
};

Status CreatePrimitivePipelines(
  MTL::Device* device,
  MTL::Library* library,
  PrimitivePipelines* out);

}  // namespace gjxl::metal_internal
