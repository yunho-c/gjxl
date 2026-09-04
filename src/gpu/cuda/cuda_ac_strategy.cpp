// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_backend_internal.h"

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
#include "gpu/image.h"
#include "gpu/ops/ac_strategy.h"

namespace gjxl::cuda_internal {
namespace {

bool TryMultiply(size_t left, size_t right, size_t* result) noexcept {
  if (result == nullptr ||
      (right != 0 &&
       left > std::numeric_limits<size_t>::max() / right)) {
    return false;
  }
  *result = left * right;
  return true;
}

template <typename T>
const T* OffsetPointer(const void* pointer, size_t offset_bytes) {
  return reinterpret_cast<const T*>(
    static_cast<const std::byte*>(pointer) + offset_bytes);
}

DeviceMemoryRange BufferRange(
  const DeviceBuffer* buffer,
  size_t offset_bytes,
  size_t size_bytes) noexcept {
  return {buffer, offset_bytes, size_bytes};
}

}  // namespace

bool CudaBackend::IsSupportedAcStrategy(
  AcStrategyType strategy) noexcept {
  switch (strategy) {
    case AcStrategyType::kDct8:
    case AcStrategyType::kDct16x16:
    case AcStrategyType::kDct32x32:
    case AcStrategyType::kDct16x8:
    case AcStrategyType::kDct8x16:
    case AcStrategyType::kDct32x16:
    case AcStrategyType::kDct16x32:
      return true;
    default:
      return false;
  }
}

Status CudaBackend::RequireCudaBuffer(
  const DeviceBuffer* buffer,
  size_t required_bytes,
  size_t offset_bytes,
  std::string_view role,
  const CudaBuffer** out) const {
  if (buffer == nullptr || out == nullptr) {
    return Status::InvalidArgument(std::string(role) + " buffer is null");
  }
  if (offset_bytes > buffer->size_bytes() ||
      required_bytes > buffer->size_bytes() - offset_bytes) {
    return Status::InvalidArgument(
      std::string(role) + " buffer is too small");
  }
  const CudaBuffer* cuda_buffer = AsCudaBuffer(*buffer);
  if (cuda_buffer == nullptr || !owns(*buffer) ||
      cuda_buffer->state() != state_.get()) {
    return Status::InvalidArgument(
      std::string(role) + " buffer does not belong to this CUDA backend");
  }
  *out = cuda_buffer;
  return Status::Ok();
}

Status CudaBackend::RequireCudaBuffer(
  DeviceBuffer* buffer,
  size_t required_bytes,
  size_t offset_bytes,
  std::string_view role,
  CudaBuffer** out) const {
  const CudaBuffer* validated = nullptr;
  Status status = RequireCudaBuffer(
    static_cast<const DeviceBuffer*>(buffer), required_bytes, offset_bytes, role,
    &validated);
  if (!status.ok()) return status;
  *out = const_cast<CudaBuffer*>(validated);
  return Status::Ok();
}

Status CudaBackend::ValidateAcStrategyCandidateBatch(
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
  if (!IsSupportedAcStrategy(batch.strategy)) {
    return Status::Unavailable(
      std::string("CUDA candidate evaluation does not support ") +
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
      transform_extent.height > batch.pixel_extent.height) {
    return Status::InvalidArgument(
      "AC-strategy batch image geometry is invalid");
  }
  const Extent2D block_extent{
    batch.pixel_extent.width / kJxlBlockDimension,
    batch.pixel_extent.height / kJxlBlockDimension,
  };
  const bool use_resident =
    batch.resident_opsin.plane[0].buffer != nullptr;
  std::array<ConstDevicePlaneView, 3> opsin_views;
  ConstDevicePlaneView mask_view;
  ConstDevicePlaneView quant_view;
  bool use_device_quant_norm = false;
  if (use_resident) {
    opsin_views = batch.resident_opsin.plane;
    mask_view = batch.resident_pixel_mask;
    quant_view = batch.resident_quant_field;
    use_device_quant_norm = quant_view.buffer != nullptr;
    if (std::ranges::any_of(
          opsin_views,
          [&](ConstDevicePlaneView view) {
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
        !TryMultiply(
          batch.opsin_row_stride, batch.pixel_extent.height,
          &minimum_plane_stride) ||
        batch.opsin_plane_stride < minimum_plane_stride) {
      return Status::InvalidArgument(
        "AC-strategy batch input strides are invalid");
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      size_t offset_elements = 0;
      if (!TryMultiply(channel, batch.opsin_plane_stride, &offset_elements) ||
          !TryMultiply(offset_elements, sizeof(float), &offset_elements) ||
          batch.opsin_offset_bytes >
            std::numeric_limits<size_t>::max() - offset_elements) {
        return Status::InvalidArgument(
          "AC-strategy packed opsin offset overflows");
      }
      opsin_views[channel] = {
        batch.opsin,
        batch.opsin_offset_bytes + offset_elements,
        DeviceElementType::kF32,
        batch.pixel_extent,
        batch.opsin_row_stride,
      };
    }
    mask_view = {
      batch.pixel_mask,
      batch.pixel_mask_offset_bytes,
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
  const size_t covered_block_count =
    strategy_info->covered_blocks.width *
    strategy_info->covered_blocks.height;
  const std::array<size_t, 9> uint32_values = {
    batch.pixel_extent.width,
    batch.pixel_extent.height,
    opsin_row_stride,
    mask_view.row_stride,
    use_device_quant_norm ? quant_view.row_stride : size_t{0},
    batch.candidate_count,
    transform_extent.width,
    transform_extent.height,
    covered_block_count,
  };
  if (std::ranges::any_of(
        uint32_values,
        [](size_t value) { return value > kUint32Maximum; })) {
    return Status::InvalidArgument(
      "AC-strategy batch exceeds CUDA's 32-bit indexing range");
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
        batch.candidate_count, kAcStrategyCandidateChannelCount,
        &transform_count) ||
      !TryMultiply(
        transform_count, coefficient_count, &packed_element_count) ||
      !TryMultiply(packed_element_count, sizeof(float), &packed_bytes) ||
      !TryMultiply(
        coefficient_count, kAcStrategyCostMatrixCount,
        &matrix_floats) ||
      !TryMultiply(matrix_floats, sizeof(float), &matrix_bytes) ||
      !TryMultiply(
        batch.candidate_count, sizeof(AcStrategyCandidate),
        &candidate_bytes) ||
      !TryMultiply(
        batch.candidate_count, kAcStrategyCandidateChannelCount,
        &rate_channels) ||
      !TryMultiply(
        rate_channels, kAcStrategyRateScratchBytesPerChannel,
        &rate_bytes) ||
      !TryMultiply(batch.candidate_count, sizeof(float), &cost_bytes) ||
      transform_count > kUint32Maximum ||
      packed_element_count > kUint32Maximum) {
    return Status::InvalidArgument(
      "AC-strategy batch buffer size overflows");
  }
  const size_t gather_blocks =
    (packed_element_count + 255) / 256;
  const std::array<size_t, 6> buffer_offsets = {
    batch.matrices_offset_bytes,
    batch.candidates_offset_bytes,
    batch.scratch_a_offset_bytes,
    batch.scratch_b_offset_bytes,
    batch.rate_scratch_offset_bytes,
    batch.costs_offset_bytes,
  };
  if (std::ranges::any_of(buffer_offsets,
        [](size_t offset) { return offset % alignof(float) != 0; })) {
    return Status::InvalidArgument(
      "AC-strategy batch buffer offset alignment is invalid");
  }
  if (!use_resident &&
      (batch.opsin_offset_bytes % alignof(float) != 0 ||
       batch.pixel_mask_offset_bytes % alignof(float) != 0)) {
    return Status::InvalidArgument(
      "AC-strategy packed input offset alignment is invalid");
  }
  if (coefficient_count > state_->maximum_threads_per_block) {
    return Status::Unavailable(
      "CUDA cannot launch the required AC-strategy thread block");
  }
  if (batch.candidate_count > state_->maximum_grid_x ||
      transform_count > state_->maximum_grid_x ||
      gather_blocks > state_->maximum_grid_x) {
    return Status::InvalidArgument(
      "AC-strategy batch exceeds CUDA grid limits");
  }

  ValidatedAcStrategyBatch validated;
  std::array<ResolvedConstPlane, 3> resolved_opsin;
  for (size_t channel = 0; channel < 3; ++channel) {
    Status status = ResolvePlane(
      opsin_views[channel], &resolved_opsin[channel]);
    if (!status.ok()) return status;
    validated.opsin[channel] = OffsetPointer<float>(
      resolved_opsin[channel].buffer->pointer(),
      resolved_opsin[channel].view.offset_bytes);
  }
  ResolvedConstPlane resolved_mask;
  Status status = ResolvePlane(mask_view, &resolved_mask);
  if (!status.ok()) return status;
  validated.pixel_mask = OffsetPointer<float>(
    resolved_mask.buffer->pointer(), resolved_mask.view.offset_bytes);

  ResolvedConstPlane resolved_quant;
  if (use_device_quant_norm) {
    status = ResolvePlane(quant_view, &resolved_quant);
    if (!status.ok()) return status;
    validated.quant_field = OffsetPointer<float>(
      resolved_quant.buffer->pointer(), resolved_quant.view.offset_bytes);
  }

  const CudaBuffer* matrices = nullptr;
  const CudaBuffer* candidates = nullptr;
  CudaBuffer* scratch_a = nullptr;
  CudaBuffer* scratch_b = nullptr;
  CudaBuffer* rate_scratch = nullptr;
  CudaBuffer* costs = nullptr;
  status = RequireCudaBuffer(
    batch.matrices, matrix_bytes, batch.matrices_offset_bytes,
    "Quantization matrix", &matrices);
  if (!status.ok()) return status;
  status = RequireCudaBuffer(
    batch.candidates, candidate_bytes, batch.candidates_offset_bytes,
    "Candidate", &candidates);
  if (!status.ok()) return status;
  status = RequireCudaBuffer(
    batch.scratch_a, packed_bytes, batch.scratch_a_offset_bytes,
    "Scratch A", &scratch_a);
  if (!status.ok()) return status;
  status = RequireCudaBuffer(
    batch.scratch_b, packed_bytes, batch.scratch_b_offset_bytes,
    "Scratch B", &scratch_b);
  if (!status.ok()) return status;
  status = RequireCudaBuffer(
    batch.rate_scratch, rate_bytes, batch.rate_scratch_offset_bytes,
    "Rate scratch", &rate_scratch);
  if (!status.ok()) return status;
  status = RequireCudaBuffer(batch.costs, cost_bytes,
    batch.costs_offset_bytes, "Cost", &costs);
  if (!status.ok()) return status;

  const std::array<DeviceMemoryRange, 7> input_ranges = {
    resolved_opsin[0].range,
    resolved_opsin[1].range,
    resolved_opsin[2].range,
    resolved_mask.range,
    use_device_quant_norm ? resolved_quant.range : DeviceMemoryRange{},
    BufferRange(batch.matrices, batch.matrices_offset_bytes, matrix_bytes),
    BufferRange(batch.candidates, batch.candidates_offset_bytes,
      candidate_bytes),
  };
  const std::array<DeviceMemoryRange, 4> output_ranges = {
    BufferRange(batch.scratch_a, batch.scratch_a_offset_bytes, packed_bytes),
    BufferRange(batch.scratch_b, batch.scratch_b_offset_bytes, packed_bytes),
    BufferRange(batch.rate_scratch, batch.rate_scratch_offset_bytes,
      rate_bytes),
    BufferRange(batch.costs, batch.costs_offset_bytes, cost_bytes),
  };
  for (size_t index = 0; index < output_ranges.size(); ++index) {
    for (size_t other = index + 1; other < output_ranges.size(); ++other) {
      if (DeviceRangesOverlap(output_ranges[index], output_ranges[other])) {
        return Status::InvalidArgument(
          "AC-strategy batch output buffers must not alias");
      }
    }
    for (DeviceMemoryRange input : input_ranges) {
      if (input.buffer != nullptr &&
          DeviceRangesOverlap(output_ranges[index], input)) {
        return Status::InvalidArgument(
          "AC-strategy batch input and output buffers must not alias");
      }
    }
  }

  constexpr float kBias = 0.13731742964354549f;
  const float ratio =
    (batch.butteraugli_target + kBias) / (1.0f + kBias);
  validated.strategy = batch.strategy;
  validated.matrices = OffsetPointer<float>(
    matrices->pointer(), batch.matrices_offset_bytes);
  validated.candidates = OffsetPointer<std::byte>(
    candidates->pointer(), batch.candidates_offset_bytes);
  validated.scratch_a = const_cast<float*>(OffsetPointer<float>(
    scratch_a->pointer(), batch.scratch_a_offset_bytes));
  validated.scratch_b = const_cast<float*>(OffsetPointer<float>(
    scratch_b->pointer(), batch.scratch_b_offset_bytes));
  validated.rate_scratch = const_cast<std::byte*>(OffsetPointer<std::byte>(
    rate_scratch->pointer(), batch.rate_scratch_offset_bytes));
  validated.costs = const_cast<float*>(OffsetPointer<float>(
    costs->pointer(), batch.costs_offset_bytes));
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
    .covered_block_count = static_cast<uint32_t>(covered_block_count),
    .use_device_quant_norm = use_device_quant_norm ? 1u : 0u,
    .info_loss_multiplier = 1.2f * std::pow(
      ratio, 0.33677806662454718f),
    .zeros_multiplier = 9.3089059022677905f * std::pow(
      ratio, 0.50990926717963703f),
    .cost_delta = 10.833273317067883f * std::pow(
      ratio, 0.36702940662370243f),
  };
  *out = validated;
  return Status::Ok();
}

cudaError_t CudaBackend::EncodeAcStrategySubmission(
  CudaBackend& backend,
  const void* context) {
  const auto& ac = *static_cast<const AcStrategyEncodeContext*>(context);
  for (const ValidatedAcStrategyBatch& batch : ac.batches) {
    const cudaError_t error = LaunchCudaAcStrategyBatch(
      batch.opsin[0], batch.opsin[1], batch.opsin[2],
      batch.pixel_mask, batch.quant_field, batch.matrices,
      batch.candidates, batch.scratch_a, batch.scratch_b,
      batch.rate_scratch, batch.costs, batch.params,
      backend.state_->stream);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

Status CudaBackend::EvaluateAcStrategyCandidateBatches(
  std::span<const AcStrategyCandidateBatch> batches,
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
      if (!status.ok()) return status;
      if (batch.candidate_count != 0) {
        validated_batches.push_back(validated);
      }
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to validate CUDA AC-strategy candidate batches");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Too many CUDA AC-strategy candidate batches");
  }
  if (validated_batches.empty()) return Status::Ok();

  const AcStrategyEncodeContext context{validated_batches};
  return SubmitCompute(
    &CudaBackend::EncodeAcStrategySubmission, &context, submission);
}

}  // namespace gjxl::cuda_internal
