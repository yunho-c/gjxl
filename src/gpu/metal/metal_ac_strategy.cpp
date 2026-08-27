// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_backend_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/ac_strategy.h"
#include "gpu/ops/ac_strategy.h"
#include "gpu/submission.h"
#include "gpu/metal/metal_status.h"

namespace gjxl::metal_internal {
namespace {

Status CreatePipeline(
  MTL::Device* device,
  MTL::Library* library,
  std::string_view function_name,
  NS::SharedPtr<MTL::ComputePipelineState>* out) {

  if (device == nullptr || library == nullptr || function_name.empty() ||
      out == nullptr) {
    return Status::InvalidArgument(
      "CreatePipeline received invalid argument");
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
      std::string("Metal function is not a kernel: ") +
      function_name_string);
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
    device, library, "gjxl_ac_strategy_gather", &pipelines.gather);
  if (!status.ok()) {
    return status;
  }
  status = CreatePipeline(
    device, library, "gjxl_ac_strategy_residual", &pipelines.residual);
  if (!status.ok()) {
    return status;
  }
  status = CreatePipeline(
    device, library, "gjxl_ac_strategy_cost", &pipelines.cost);
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
    maximum_threads, kPreferredGatherThreads);
  pipelines.gather_threads_per_threadgroup =
    (capped_threads / execution_width) * execution_width;
  if (pipelines.gather_threads_per_threadgroup == 0) {
    return Status::Unavailable(
      "Metal cannot launch the AC-strategy gather kernel");
  }

  *out = std::move(pipelines);
  return Status::Ok();
}

MetalBackend::~MetalBackend() {
  (void)Synchronize();
}

Status MetalBackend::EvaluateAcStrategyCandidates(
  const AcStrategyCandidateBatch& batch) {

  return SubmitAcStrategyCandidates(
    std::span<const AcStrategyCandidateBatch>(&batch, 1));
}

Status MetalBackend::EvaluateAcStrategyCandidateBatches(
  std::span<const AcStrategyCandidateBatch> batches) {

  return SubmitAcStrategyCandidates(batches);
}

Status MetalBackend::Synchronize() {
  if (!pending_ac_submission_) {
    return Status::Ok();
  }
  const Status status = pending_ac_submission_->Wait();
  pending_ac_submission_.reset();
  return status;
}

bool MetalBackend::TryMultiply(
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

Status MetalBackend::RequireMetalBuffer(
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
  if (metal_buffer == nullptr || !owns(*buffer) ||
      metal_buffer->device() != device_.get()) {
    return Status::InvalidArgument(
      std::string(role) + " buffer does not belong to this Metal backend");
  }
  *out = metal_buffer;
  return Status::Ok();
}

Status MetalBackend::RequireMetalBuffer(
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

Status MetalBackend::ValidateAcStrategyCandidateBatch(
  const AcStrategyCandidateBatch& batch,
  ValidatedAcStrategyBatch* out) const {

  if (out == nullptr) {
    return Status::Internal(
      "AC-strategy batch validation output is null");
  }
  const AcStrategyInfo* strategy_info = GetAcStrategyInfo(batch.strategy);
  if (strategy_info == nullptr) {
    return Status::InvalidArgument("Unknown JPEG XL AC strategy");
  }

  const TransformPipelinePair& transform_pair =
    transform_pipelines_[static_cast<size_t>(batch.strategy)];
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
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (outputs[index] == nullptr) {
      return Status::InvalidArgument(
        "AC-strategy batch output buffer is null");
    }
    for (size_t other = index + 1; other < outputs.size(); ++other) {
      if (outputs[index] == outputs[other]) {
        return Status::InvalidArgument(
          "AC-strategy batch output buffers must not alias");
      }
    }
    if (std::ranges::find(inputs, outputs[index]) != inputs.end()) {
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
    batch.matrices,
    matrix_bytes,
    "Quantization matrix",
    &validated.matrices);
  if (!status.ok()) {
    return status;
  }
  status = RequireMetalBuffer(
    batch.candidates,
    candidate_bytes,
    "Candidate",
    &validated.candidates);
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
    batch.rate_scratch,
    rate_bytes,
    "Rate scratch",
    &validated.rate_scratch);
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
  validated.strategy = batch.strategy;
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
      ratio, 0.33677806662454718f),
    .zeros_multiplier = 9.3089059022677905f * std::pow(
      ratio, 0.50990926717963703f),
    .cost_delta = 10.833273317067883f * std::pow(
      ratio, 0.36702940662370243f),
  };
  validated.forward = &transform_pair.forward;
  validated.inverse = &transform_pair.inverse;
  validated.transform_count = transform_count;
  validated.packed_element_count = packed_element_count;
  *out = validated;
  return Status::Ok();
}

void MetalBackend::EncodeAcStrategySubmission(
  MetalBackend& backend,
  MTL::ComputeCommandEncoder* encoder,
  const void* context) {

  const auto& ac = *static_cast<const AcStrategyEncodeContext*>(context);
  for (const ValidatedAcStrategyBatch& batch : ac.batches) {
    backend.EncodeAcStrategyCandidateBatch(encoder, batch);
  }
}

void MetalBackend::EncodeAcStrategyCandidateBatch(
  MTL::ComputeCommandEncoder* encoder,
  const ValidatedAcStrategyBatch& validated) {

  encoder->setComputePipelineState(ac_strategy_pipelines_.gather.get());
  encoder->setBuffer(validated.opsin->handle(), 0, 0);
  encoder->setBuffer(validated.candidates->handle(), 0, 1);
  encoder->setBuffer(validated.scratch_a->handle(), 0, 2);
  encoder->setBytes(&validated.params, sizeof(validated.params), 3);
  encoder->dispatchThreads(
    MTL::Size(
      static_cast<NS::UInteger>(validated.packed_element_count), 1, 1),
    MTL::Size(
      ac_strategy_pipelines_.gather_threads_per_threadgroup, 1, 1));

  EncodeTransformBatch(
    encoder,
    TransformDirection::kForward,
    validated.strategy,
    *validated.scratch_a,
    0,
    *validated.scratch_b,
    0,
    validated.transform_count);

  encoder->setComputePipelineState(ac_strategy_pipelines_.residual.get());
  encoder->setBuffer(validated.scratch_b->handle(), 0, 0);
  encoder->setBuffer(validated.matrices->handle(), 0, 1);
  encoder->setBuffer(validated.candidates->handle(), 0, 2);
  encoder->setBuffer(validated.scratch_a->handle(), 0, 3);
  encoder->setBuffer(validated.rate_scratch->handle(), 0, 4);
  encoder->setBytes(&validated.params, sizeof(validated.params), 5);
  encoder->dispatchThreadgroups(
    MTL::Size(
      static_cast<NS::UInteger>(validated.transform_count), 1, 1),
    MTL::Size(validated.params.coefficient_count, 1, 1));

  EncodeTransformBatch(
    encoder,
    TransformDirection::kInverse,
    validated.strategy,
    *validated.scratch_a,
    0,
    *validated.scratch_b,
    0,
    validated.transform_count);

  encoder->setComputePipelineState(ac_strategy_pipelines_.cost.get());
  encoder->setBuffer(validated.scratch_b->handle(), 0, 0);
  encoder->setBuffer(validated.pixel_mask->handle(), 0, 1);
  encoder->setBuffer(validated.candidates->handle(), 0, 2);
  encoder->setBuffer(validated.rate_scratch->handle(), 0, 3);
  encoder->setBuffer(validated.costs->handle(), 0, 4);
  encoder->setBytes(&validated.params, sizeof(validated.params), 5);
  encoder->dispatchThreadgroups(
    MTL::Size(
      static_cast<NS::UInteger>(validated.params.candidate_count), 1, 1),
    MTL::Size(validated.params.coefficient_count, 1, 1));
}

Status MetalBackend::SubmitAcStrategyCandidates(
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
  Status status = Synchronize();
  if (!status.ok()) {
    return status;
  }

  const AcStrategyEncodeContext context{validated_batches};
  std::unique_ptr<GpuSubmission> submission;
  status = SubmitCompute(
    "gjxl staged AC candidate evaluation",
    &MetalBackend::EncodeAcStrategySubmission,
    &context,
    &submission);
  if (!status.ok()) {
    return status;
  }
  pending_ac_submission_ = std::move(submission);
  return Status::Ok();
}

}  // namespace gjxl::metal_internal
