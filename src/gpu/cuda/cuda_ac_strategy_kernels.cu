// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_kernels.h"

#include <cstddef>
#include <cstdint>

namespace gjxl::cuda_internal {
namespace {

constexpr unsigned int kGatherThreads = 256;

struct AcStrategyCandidateDevice {
  uint32_t block_x;
  uint32_t block_y;
  float quant_norm;
  float entropy_multiplier;
  float cfl_x;
  float cfl_b;
};

static_assert(sizeof(AcStrategyCandidateDevice) == 6 * sizeof(uint32_t));

struct ChannelRate {
  float magnitude;
  uint32_t nonzero_count;
};

static_assert(sizeof(ChannelRate) == 2 * sizeof(float));

__device__ unsigned int CeilLog2Nonzero(unsigned int value) {
  return value <= 1 ? 0 : 32 - __clz(value - 1);
}

__device__ float RoundAwayFromZero(float value) {
  return copysignf(floorf(fabsf(value) + 0.5f), value);
}

__device__ float FastLog2(float value) {
  const unsigned int value_bits = __float_as_uint(value);
  const int shifted_exponent =
    static_cast<int>(value_bits - 0x3f2aaaabu) >> 23;
  const unsigned int mantissa_bits =
    value_bits - (static_cast<unsigned int>(shifted_exponent) << 23);
  const float x = __uint_as_float(mantissa_bits) - 1.0f;
  float numerator =
    fmaf(0.74245873327820566f, x, 1.4287160470083755f);
  numerator = fmaf(numerator, x, -1.8503833400518310e-06f);
  float denominator =
    fmaf(0.17409343003366853f, x, 1.0096718572241148f);
  denominator = fmaf(denominator, x, 0.99032814277590719f);
  return numerator / denominator + static_cast<float>(shifted_exponent);
}

__device__ float FastPow2(float value) {
  const float floor_value = floorf(value);
  const int exponent = static_cast<int>(floor_value) + 127;
  const float exponent_value =
    __uint_as_float(static_cast<unsigned int>(exponent) << 23);
  const float fraction = value - floor_value;
  float numerator = fraction + 1.01749063e+01f;
  numerator = fmaf(numerator, fraction, 4.88687798e+01f);
  numerator = fmaf(numerator, fraction, 9.85506591e+01f);
  numerator *= exponent_value;
  float denominator =
    fmaf(fraction, 2.10242958e-01f, -2.22328856e-02f);
  denominator = fmaf(denominator, fraction, -1.94414990e+01f);
  denominator = fmaf(denominator, fraction, 9.85506633e+01f);
  return numerator / denominator;
}

__device__ bool CandidateValid(
  AcStrategyCandidateDevice candidate,
  CudaAcStrategyBatchParams params) {
  return params.transform_width <= params.pixel_width &&
    params.transform_height <= params.pixel_height &&
    candidate.block_x <=
      (params.pixel_width - params.transform_width) / 8 &&
    candidate.block_y <=
      (params.pixel_height - params.transform_height) / 8 &&
    isfinite(candidate.quant_norm) && candidate.quant_norm > 0.0f &&
    isfinite(candidate.entropy_multiplier) &&
    candidate.entropy_multiplier > 0.0f &&
    (params.use_device_cfl != 0u ||
     (isfinite(candidate.cfl_x) && isfinite(candidate.cfl_b)));
}

__device__ float ComputeCflFactor(
  const signed char* y_to_x,
  const signed char* y_to_b,
  AcStrategyCandidateDevice candidate,
  unsigned int channel,
  CudaAcStrategyBatchParams params) {
  if (channel == 1u) return 0.0f;
  if (params.use_device_cfl == 0u) {
    return channel == 0u ? candidate.cfl_x : candidate.cfl_b;
  }
  constexpr float kCflScale = 1.0f / 84.0f;
  const size_t tile_index =
    static_cast<size_t>(candidate.block_y / 8u) *
      params.color_tile_row_stride + candidate.block_x / 8u;
  return channel == 0u
    ? static_cast<float>(y_to_x[tile_index]) * kCflScale
    : 1.0f + static_cast<float>(y_to_b[tile_index]) * kCflScale;
}

__device__ float ComputeQuantNorm(
  const float* quant_field,
  AcStrategyCandidateDevice candidate,
  CudaAcStrategyBatchParams params) {
  if (params.use_device_quant_norm == 0u) return candidate.quant_norm;
  if (params.covered_block_count == 1u) {
    return quant_field[
      static_cast<size_t>(candidate.block_y) *
        params.quant_field_row_stride +
      candidate.block_x];
  }
  if (params.covered_block_count == 2u) {
    const float first = quant_field[
      static_cast<size_t>(candidate.block_y) *
        params.quant_field_row_stride +
      candidate.block_x];
    const unsigned int second_x = candidate.block_x +
      (params.covered_block_width == 2u ? 1u : 0u);
    const unsigned int second_y = candidate.block_y +
      (params.covered_block_height == 2u ? 1u : 0u);
    const float second = quant_field[
      static_cast<size_t>(second_y) * params.quant_field_row_stride +
      second_x];
    return fmaxf(first, second);
  }
  float sum = 0.0f;
  for (unsigned int dy = 0; dy < params.covered_block_height; ++dy) {
    for (unsigned int dx = 0; dx < params.covered_block_width; ++dx) {
      float value = quant_field[
        static_cast<size_t>(candidate.block_y + dy) *
          params.quant_field_row_stride +
        candidate.block_x + dx];
      value *= value;
      value *= value;
      value *= value;
      sum += value * value;
    }
  }
  sum /= static_cast<float>(params.covered_block_count);
  return FastPow2(FastLog2(sum) * (1.0f / 16.0f));
}

__global__ void GatherKernel(
  const float* opsin_x,
  const float* opsin_y,
  const float* opsin_b,
  const AcStrategyCandidateDevice* candidates,
  float* packed_pixels,
  CudaAcStrategyBatchParams params) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  const size_t candidate_stride = 3u * params.coefficient_count;
  const size_t element_count =
    static_cast<size_t>(params.candidate_count) * candidate_stride;
  if (index >= element_count) return;

  const unsigned int candidate_index = static_cast<unsigned int>(
    index / candidate_stride);
  const unsigned int candidate_element = static_cast<unsigned int>(
    index % candidate_stride);
  const unsigned int channel =
    candidate_element / params.coefficient_count;
  const unsigned int element =
    candidate_element % params.coefficient_count;
  const AcStrategyCandidateDevice candidate = candidates[candidate_index];
  if (!CandidateValid(candidate, params)) {
    packed_pixels[index] = NAN;
    return;
  }
  const unsigned int row = element / params.transform_width;
  const unsigned int column = element % params.transform_width;
  const unsigned int pixel_x = candidate.block_x * 8 + column;
  const unsigned int pixel_y = candidate.block_y * 8 + row;
  const size_t source_index =
    static_cast<size_t>(pixel_y) * params.opsin_row_stride + pixel_x;
  packed_pixels[index] = channel == 0u ? opsin_x[source_index] :
    channel == 1u ? opsin_y[source_index] : opsin_b[source_index];
}

__global__ void ResidualKernel(
  const float* coefficients,
  const float* matrices,
  const AcStrategyCandidateDevice* candidates,
  const float* quant_field,
  const signed char* y_to_x,
  const signed char* y_to_b,
  float* residual_coefficients,
  ChannelRate* channel_rates,
  CudaAcStrategyBatchParams params) {
  extern __shared__ unsigned char shared_bytes[];
  float* magnitude_reduction = reinterpret_cast<float*>(shared_bytes);
  unsigned int* nonzero_reduction = reinterpret_cast<unsigned int*>(
    magnitude_reduction + params.coefficient_count);
  const unsigned int tid = threadIdx.x;
  const unsigned int transform_index = blockIdx.x;
  const unsigned int candidate_index = transform_index / 3;
  const unsigned int channel = transform_index % 3;
  const size_t base =
    static_cast<size_t>(transform_index) * params.coefficient_count;
  const size_t y_base =
    static_cast<size_t>(candidate_index * 3 + 1) *
    params.coefficient_count;
  const size_t matrix_base =
    static_cast<size_t>(channel) * params.coefficient_count;
  const size_t inverse_matrix_base =
    static_cast<size_t>(3 + channel) * params.coefficient_count;
  const AcStrategyCandidateDevice candidate = candidates[candidate_index];

  float rounded = NAN;
  if (CandidateValid(candidate, params)) {
    const float quant_norm =
      ComputeQuantNorm(quant_field, candidate, params);
    const float cfl_factor = ComputeCflFactor(
      y_to_x, y_to_b, candidate, channel, params);
    const float decorrelated =
      coefficients[base + tid] - coefficients[y_base + tid] * cfl_factor;
    const float scaled = decorrelated *
      matrices[inverse_matrix_base + tid] * quant_norm;
    rounded = RoundAwayFromZero(scaled);
    residual_coefficients[base + tid] =
      matrices[matrix_base + tid] * (scaled - rounded);
  } else {
    residual_coefficients[base + tid] = NAN;
  }
  magnitude_reduction[tid] = sqrtf(fabsf(rounded));
  nonzero_reduction[tid] = rounded != 0.0f ? 1u : 0u;
  __syncthreads();

  for (unsigned int stride = params.coefficient_count / 2;
       stride != 0; stride /= 2) {
    if (tid < stride) {
      magnitude_reduction[tid] += magnitude_reduction[tid + stride];
      nonzero_reduction[tid] += nonzero_reduction[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    channel_rates[transform_index] = {
      magnitude_reduction[0], nonzero_reduction[0]};
  }
}

__global__ void CostKernel(
  const float* residual_pixels,
  const float* pixel_mask,
  const AcStrategyCandidateDevice* candidates,
  const ChannelRate* channel_rates,
  float* costs,
  const float* quant_field,
  CudaAcStrategyBatchParams params) {
  extern __shared__ float loss_reduction[];
  constexpr float kMaskOffset[3] = {12.0f, 0.0f, 4.0f};
  constexpr float kChannelMultiplier[3] = {
    2.0441408586549744e7f, 1.0f, 1.266770081387616f};
  const unsigned int tid = threadIdx.x;
  const unsigned int candidate_index = blockIdx.x;
  const AcStrategyCandidateDevice candidate = candidates[candidate_index];
  const bool candidate_fits =
    candidate.block_x <=
      (params.pixel_width - params.transform_width) / 8 &&
    candidate.block_y <=
      (params.pixel_height - params.transform_height) / 8;
  if (!candidate_fits) {
    if (tid == 0) costs[candidate_index] = NAN;
    return;
  }

  const unsigned int row = tid / params.transform_width;
  const unsigned int column = tid % params.transform_width;
  const unsigned int pixel_x = candidate.block_x * 8 + column;
  const unsigned int pixel_y = candidate.block_y * 8 + row;
  const float mask = pixel_mask[
    static_cast<size_t>(pixel_y) * params.pixel_mask_row_stride + pixel_x];
  for (unsigned int channel = 0; channel < 3; ++channel) {
    const unsigned int transform_index = candidate_index * 3 + channel;
    const size_t base =
      static_cast<size_t>(transform_index) * params.coefficient_count;
    float weighted =
      (mask + kMaskOffset[channel]) * residual_pixels[base + tid];
    weighted *= weighted;
    weighted *= weighted;
    weighted *= weighted;
    loss_reduction[
      static_cast<size_t>(channel) * params.coefficient_count + tid] =
      isfinite(mask) && mask > 0.0f ? weighted : NAN;
  }
  __syncthreads();

  for (unsigned int stride = params.coefficient_count / 2;
       stride != 0; stride /= 2) {
    if (tid < stride) {
      for (unsigned int channel = 0; channel < 3; ++channel) {
        const size_t base =
          static_cast<size_t>(channel) * params.coefficient_count;
        loss_reduction[base + tid] +=
          loss_reduction[base + tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    float entropy = 0.0f;
    float loss = 0.0f;
    for (unsigned int channel = 0; channel < 3; ++channel) {
      const ChannelRate rate =
        channel_rates[candidate_index * 3 + channel];
      entropy += params.cost_delta * rate.magnitude;
      const unsigned int nonzero_bits =
        CeilLog2Nonzero(rate.nonzero_count + 1) + 1;
      entropy += params.zeros_multiplier * static_cast<float>(
        CeilLog2Nonzero(nonzero_bits + 17) + nonzero_bits);
      loss += loss_reduction[
        static_cast<size_t>(channel) * params.coefficient_count] *
        kChannelMultiplier[channel];
      if (channel == 0 && params.covered_block_count >= 2) {
        const float weight = 1.0f + fminf(
          3.0f, static_cast<float>(params.covered_block_count) / 8.0f);
        entropy *= weight;
        loss *= weight;
      }
    }
    const float quant_norm =
      ComputeQuantNorm(quant_field, candidate, params);
    const float normalized_loss =
      loss / static_cast<float>(params.coefficient_count);
    const float loss_cost = powf(normalized_loss, 0.125f) *
      static_cast<float>(params.coefficient_count) / quant_norm;
    const float result = entropy * candidate.entropy_multiplier +
      params.info_loss_multiplier * loss_cost;
    costs[candidate_index] =
      isfinite(result) && result >= 0.0f ? result : NAN;
  }
}

}  // namespace

cudaError_t LaunchCudaAcStrategyBatch(
  const float* opsin_x,
  const float* opsin_y,
  const float* opsin_b,
  const float* pixel_mask,
  const float* quant_field,
  const signed char* y_to_x,
  const signed char* y_to_b,
  const float* matrices,
  const void* candidates,
  float* scratch_a,
  float* scratch_b,
  void* rate_scratch,
  float* costs,
  CudaAcStrategyBatchParams params,
  cudaStream_t stream) {
  const auto* typed_candidates =
    static_cast<const AcStrategyCandidateDevice*>(candidates);
  const size_t packed_element_count =
    static_cast<size_t>(params.candidate_count) * 3 *
    params.coefficient_count;
  const unsigned int gather_blocks = static_cast<unsigned int>(
    (packed_element_count + kGatherThreads - 1) / kGatherThreads);
  GatherKernel<<<gather_blocks, kGatherThreads, 0, stream>>>(
    opsin_x, opsin_y, opsin_b, typed_candidates, scratch_a, params);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  const size_t transform_count =
    static_cast<size_t>(params.candidate_count) * 3;
  error = LaunchCudaDct(
    true, scratch_a, scratch_b, transform_count,
    params.transform_width, params.transform_height, stream);
  if (error != cudaSuccess) return error;

  const size_t residual_shared_bytes =
    static_cast<size_t>(params.coefficient_count) *
    (sizeof(float) + sizeof(unsigned int));
  ResidualKernel<<<
    static_cast<unsigned int>(transform_count),
    params.coefficient_count,
    residual_shared_bytes,
    stream>>>(
      scratch_b, matrices, typed_candidates, quant_field, y_to_x, y_to_b,
      scratch_a,
      static_cast<ChannelRate*>(rate_scratch), params);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  error = LaunchCudaDct(
    false, scratch_a, scratch_b, transform_count,
    params.transform_width, params.transform_height, stream);
  if (error != cudaSuccess) return error;

  const size_t cost_shared_bytes =
    static_cast<size_t>(params.coefficient_count) * 3 * sizeof(float);
  CostKernel<<<
    params.candidate_count,
    params.coefficient_count,
    cost_shared_bytes,
    stream>>>(
      scratch_b, pixel_mask, typed_candidates,
      static_cast<const ChannelRate*>(rate_scratch), costs, quant_field,
      params);
  return cudaGetLastError();
}

}  // namespace gjxl::cuda_internal
