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

#define setComputePipelineState(state)                                    \
  setComputePipelineState(state);                                         \
  ::gjxl::metal_internal::RecordMetalComputePipelineState(state)

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
  RegisterMetalComputePipeline(pipeline.get(), function_name);
  *out = std::move(pipeline);
  return Status::Ok();
}

const char* AcStrategyProfileStageId(AcStrategyType strategy) {
  switch (strategy) {
    case AcStrategyType::kDct8:
      return "frontend.ac_strategy.dct8";
    case AcStrategyType::kDct16x8:
      return "frontend.ac_strategy.dct16x8";
    case AcStrategyType::kDct8x16:
      return "frontend.ac_strategy.dct8x16";
    case AcStrategyType::kDct16x16:
      return "frontend.ac_strategy.dct16";
    case AcStrategyType::kDct32x16:
      return "frontend.ac_strategy.dct32x16";
    case AcStrategyType::kDct16x32:
      return "frontend.ac_strategy.dct16x32";
    case AcStrategyType::kDct32x32:
      return "frontend.ac_strategy.dct32";
    default:
      return "frontend.ac_strategy.unsupported";
  }
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

Status MetalBackend::EvaluateAcStrategyCandidateBatches(
  std::span<const AcStrategyCandidateBatch> batches,
  std::unique_ptr<GpuSubmission>* submission) {

  return SubmitAcStrategyCandidatesImpl(
    batches, gpu_profile_internal::GpuProfilingMode::kDisabled,
    submission);
}

Status MetalBackend::EvaluateAcStrategyCandidateBatchesProfiled(
  std::span<const AcStrategyCandidateBatch> batches,
  gpu_profile_internal::GpuProfilingMode mode,
  std::unique_ptr<GpuSubmission>* submission) {

  if (mode == gpu_profile_internal::GpuProfilingMode::kDisabled) {
    return Status::InvalidArgument(
      "Profiled AC-strategy mode is disabled");
  }
  return SubmitAcStrategyCandidatesImpl(batches, mode, submission);
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
  const Extent2D block_extent{
      batch.pixel_extent.width / kJxlBlockDimension,
      batch.pixel_extent.height / kJxlBlockDimension,
  };
  const bool use_resident =
      batch.resident_opsin.plane[0].buffer != nullptr;
  if (batch.pixel_extent.empty() ||
      batch.pixel_extent.width % kJxlBlockDimension != 0 ||
      batch.pixel_extent.height % kJxlBlockDimension != 0 ||
      transform_extent.width > batch.pixel_extent.width ||
      transform_extent.height > batch.pixel_extent.height) {
    return Status::InvalidArgument(
      "AC-strategy batch image geometry is invalid");
  }

  std::array<ConstDevicePlaneView, 3> opsin_views;
  ConstDevicePlaneView mask_view;
  ConstDevicePlaneView quant_view;
  bool use_device_quant_norm = false;
  if (use_resident) {
    opsin_views = batch.resident_opsin.plane;
    mask_view = batch.resident_pixel_mask;
    quant_view = batch.resident_quant_field;
    use_device_quant_norm = quant_view.buffer != nullptr;
    if (std::ranges::any_of(opsin_views, [&](ConstDevicePlaneView view) {
          return view.extent != batch.pixel_extent ||
                 view.row_stride < batch.pixel_extent.width;
        }) ||
        mask_view.extent != batch.pixel_extent ||
        mask_view.row_stride < batch.pixel_extent.width ||
        (use_device_quant_norm &&
         (quant_view.extent != block_extent ||
          quant_view.row_stride < block_extent.width))) {
      return Status::InvalidArgument(
          "Resident AC-strategy input geometry is invalid");
    }
  } else {
    size_t minimum_plane_stride = 0;
    if (batch.opsin_row_stride < batch.pixel_extent.width ||
        batch.pixel_mask_row_stride < batch.pixel_extent.width ||
        !TryMultiply(batch.opsin_row_stride, batch.pixel_extent.height,
                     &minimum_plane_stride) ||
        batch.opsin_plane_stride < minimum_plane_stride) {
      return Status::InvalidArgument(
          "AC-strategy batch input strides are invalid");
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      opsin_views[channel] = {
          batch.opsin,
          channel * batch.opsin_plane_stride * sizeof(float),
          DeviceElementType::kF32,
          batch.pixel_extent,
          batch.opsin_row_stride,
      };
    }
    mask_view = {
        batch.pixel_mask,
        0,
        DeviceElementType::kF32,
        batch.pixel_extent,
        batch.pixel_mask_row_stride,
    };
  }
  if (!std::isfinite(batch.butteraugli_target) ||
      batch.butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "AC-strategy batch Butteraugli target is invalid");
  }

  constexpr size_t kUint32Maximum =
    std::numeric_limits<uint32_t>::max();
  const size_t opsin_row_stride = opsin_views[0].row_stride;
  if (opsin_views[1].row_stride != opsin_row_stride ||
      opsin_views[2].row_stride != opsin_row_stride) {
    return Status::InvalidArgument(
        "AC-strategy opsin plane strides differ");
  }
  const std::array<size_t, 9> uint32_values = {
    batch.pixel_extent.width,
    batch.pixel_extent.height,
    opsin_row_stride,
    mask_view.row_stride,
    use_device_quant_norm ? quant_view.row_stride : size_t{0},
    batch.candidate_count,
    transform_extent.width,
    transform_extent.height,
    strategy_info->covered_blocks.width *
      strategy_info->covered_blocks.height,
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

  const std::array<const DeviceBuffer*, 7> inputs = {
    opsin_views[0].buffer,
    opsin_views[1].buffer,
    opsin_views[2].buffer,
    mask_view.buffer,
    quant_view.buffer,
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
  std::array<ResolvedConstPlane, 3> resolved_opsin;
  Status status = Status::Ok();
  for (size_t channel = 0; channel < 3; ++channel) {
    status = ResolvePlane(opsin_views[channel], &resolved_opsin[channel]);
    if (!status.ok()) return status;
    validated.opsin[channel] = resolved_opsin[channel].buffer;
    validated.opsin_offset_bytes[channel] =
        resolved_opsin[channel].view.offset_bytes;
  }
  ResolvedConstPlane resolved_mask;
  status = ResolvePlane(mask_view, &resolved_mask);
  if (!status.ok()) return status;
  validated.pixel_mask = resolved_mask.buffer;
  validated.pixel_mask_offset_bytes = resolved_mask.view.offset_bytes;
  if (use_device_quant_norm) {
    ResolvedConstPlane resolved_quant;
    status = ResolvePlane(quant_view, &resolved_quant);
    if (!status.ok()) return status;
    validated.quant_field = resolved_quant.buffer;
    validated.quant_field_offset_bytes = resolved_quant.view.offset_bytes;
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
  if (!use_device_quant_norm) {
    validated.quant_field = validated.candidates;
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
    .opsin_row_stride = static_cast<uint32_t>(opsin_row_stride),
    .pixel_mask_row_stride =
      static_cast<uint32_t>(mask_view.row_stride),
    .quant_field_row_stride = static_cast<uint32_t>(
      use_device_quant_norm ? quant_view.row_stride : 0),
    .candidate_count = static_cast<uint32_t>(batch.candidate_count),
    .coefficient_count = static_cast<uint32_t>(coefficient_count),
    .transform_width = static_cast<uint32_t>(transform_extent.width),
    .transform_height = static_cast<uint32_t>(transform_extent.height),
    .covered_block_width = static_cast<uint32_t>(
      strategy_info->covered_blocks.width),
    .covered_block_height = static_cast<uint32_t>(
      strategy_info->covered_blocks.height),
    .covered_block_count = static_cast<uint32_t>(
      strategy_info->covered_blocks.width *
      strategy_info->covered_blocks.height),
    .use_device_quant_norm = use_device_quant_norm ? 1u : 0u,
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

void MetalBackend::EncodeAcStrategyProfileStage(
  MetalBackend& backend,
  MTL::ComputeCommandEncoder* encoder,
  const void* context) {

  const auto& profile =
    *static_cast<const AcStrategyProfileContext*>(context);
  backend.EncodeAcStrategyCandidateBatch(encoder, *profile.batch);
}

void MetalBackend::EncodeAcStrategyCandidateBatch(
  MTL::ComputeCommandEncoder* encoder,
  const ValidatedAcStrategyBatch& validated) {

  encoder->setComputePipelineState(ac_strategy_pipelines_.gather.get());
  for (size_t channel = 0; channel < 3; ++channel) {
    encoder->setBuffer(validated.opsin[channel]->handle(),
                       validated.opsin_offset_bytes[channel], channel);
  }
  encoder->setBuffer(validated.candidates->handle(), 0, 3);
  encoder->setBuffer(validated.scratch_a->handle(), 0, 4);
  encoder->setBytes(&validated.params, sizeof(validated.params), 5);
  DispatchMetalThreads(
    encoder,
    MTL::Size(
      static_cast<NS::UInteger>(validated.packed_element_count), 1, 1),
    MTL::Size(
      ac_strategy_pipelines_.gather_threads_per_threadgroup, 1, 1));

  EncodeTransformBatch(
    encoder, TransformDirection::kForward, validated.strategy,
    *validated.scratch_a, 0, *validated.scratch_b, 0,
    validated.transform_count);

  encoder->setComputePipelineState(ac_strategy_pipelines_.residual.get());
  encoder->setBuffer(validated.scratch_b->handle(), 0, 0);
  encoder->setBuffer(validated.matrices->handle(), 0, 1);
  encoder->setBuffer(validated.candidates->handle(), 0, 2);
  encoder->setBuffer(validated.quant_field->handle(),
                     validated.quant_field_offset_bytes, 3);
  encoder->setBuffer(validated.scratch_a->handle(), 0, 4);
  encoder->setBuffer(validated.rate_scratch->handle(), 0, 5);
  encoder->setBytes(&validated.params, sizeof(validated.params), 6);
  const NS::UInteger reduction_bytes =
    validated.params.coefficient_count * sizeof(float);
  encoder->setThreadgroupMemoryLength(reduction_bytes, 0);
  encoder->setThreadgroupMemoryLength(reduction_bytes, 1);
  DispatchMetalThreadgroups(
    encoder,
    MTL::Size(
      static_cast<NS::UInteger>(validated.transform_count), 1, 1),
    MTL::Size(validated.params.coefficient_count, 1, 1));

  EncodeTransformBatch(
    encoder, TransformDirection::kInverse, validated.strategy,
    *validated.scratch_a, 0, *validated.scratch_b, 0,
    validated.transform_count);

  encoder->setComputePipelineState(ac_strategy_pipelines_.cost.get());
  encoder->setBuffer(validated.scratch_b->handle(), 0, 0);
  encoder->setBuffer(validated.pixel_mask->handle(),
                     validated.pixel_mask_offset_bytes, 1);
  encoder->setBuffer(validated.candidates->handle(), 0, 2);
  encoder->setBuffer(validated.rate_scratch->handle(), 0, 3);
  encoder->setBuffer(validated.costs->handle(), 0, 4);
  encoder->setBuffer(validated.quant_field->handle(),
                     validated.quant_field_offset_bytes, 5);
  encoder->setBytes(&validated.params, sizeof(validated.params), 6);
  encoder->setThreadgroupMemoryLength(3 * reduction_bytes, 0);
  DispatchMetalThreadgroups(
    encoder,
    MTL::Size(
      static_cast<NS::UInteger>(validated.params.candidate_count), 1, 1),
    MTL::Size(validated.params.coefficient_count, 1, 1));
}

Status MetalBackend::SubmitAcStrategyCandidatesImpl(
  std::span<const AcStrategyCandidateBatch> batches,
  gpu_profile_internal::GpuProfilingMode mode,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr) {
    return Status::InvalidArgument(
      "AC-strategy submission output pointer is null");
  }
  submission->reset();

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

  const AcStrategyEncodeContext context{validated_batches};
  if (mode == gpu_profile_internal::GpuProfilingMode::kDisabled) {
    return SubmitCompute(
      "gjxl staged AC candidate evaluation",
      &MetalBackend::EncodeAcStrategySubmission,
      &context,
      submission);
  }
  std::vector<AcStrategyProfileContext> contexts;
  std::vector<MetalProfiledComputeStage> stages;
  try {
    contexts.resize(validated_batches.size());
    stages.resize(validated_batches.size());
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate AC-strategy GPU stage metadata");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "AC-strategy GPU stage metadata is too large");
  }
  for (size_t index = 0; index < validated_batches.size(); ++index) {
    contexts[index] = {&validated_batches[index]};
    stages[index] = {
      .stage_id = AcStrategyProfileStageId(validated_batches[index].strategy),
      .group_id = "frontend.ac_strategy",
      .encode = &MetalBackend::EncodeAcStrategyProfileStage,
      .context = &contexts[index],
    };
  }
  return SubmitComputeProfiled(
    "gjxl staged AC candidate evaluation profile",
    stages, mode, submission);
}

}  // namespace gjxl::metal_internal
