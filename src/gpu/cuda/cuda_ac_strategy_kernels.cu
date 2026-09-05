// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_kernels.h"

#include <cstddef>
#include <cstdint>

#include "gpu/cuda/cuda_ac_strategy_device.cuh"

namespace gjxl::cuda_internal {
namespace {

constexpr unsigned int kQuantNormThreads = 256;
constexpr unsigned int kCostThreads = 256;

using ChannelRate = AcStrategyChannelRateDevice;

__device__ unsigned int CeilLog2Nonzero(unsigned int value) {
  return value <= 1 ? 0 : 32 - __clz(value - 1);
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

__global__ void PrepareQuantNormsKernel(
  const AcStrategyCandidateDevice* candidates,
  const float* quant_field,
  float* quant_norms,
  CudaAcStrategyBatchParams params) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= params.candidate_count) return;
  const AcStrategyCandidateDevice candidate = candidates[index];
  quant_norms[index] = CandidateValid(candidate, params)
    ? ComputeQuantNorm(quant_field, candidate, params) : NAN;
}

__global__ void FinalizeCostKernel(
  const float* channel_losses,
  const AcStrategyCandidateDevice* candidates,
  const ChannelRate* channel_rates,
  float* costs,
  CudaAcStrategyBatchParams params) {
  const unsigned int candidate_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate_index >= params.candidate_count) return;
  constexpr float kChannelMultiplier[3] = {
    2.0441408586549744e7f, 1.0f, 1.266770081387616f};
  const auto candidate = candidates[candidate_index];
  const unsigned int coefficient_count = params.coefficient_count;
  const float totals[3] = {channel_losses[candidate_index * 3],
    channel_losses[candidate_index * 3 + 1], channel_losses[candidate_index * 3 + 2]};
  float entropy = 0.0f;
  float loss = 0.0f;
  for (unsigned int channel = 0; channel < 3; ++channel) {
    const ChannelRate rate = channel_rates[candidate_index * 3 + channel];
    // The parent contracts X weighting with the following Y addition.
    // Preserve that FMA explicitly; register totals otherwise cause NVCC
    // to emit a separately rounded multiply and add across this branch.
    if (channel == 1 && params.covered_block_count >= 2) {
      const float weight = 1.0f + fminf(
        3.0f, static_cast<float>(params.covered_block_count) / 8.0f);
      entropy = fmaf(entropy, weight, params.cost_delta * rate.magnitude);
      loss = fmaf(loss, weight, totals[channel]);
    } else {
      entropy = fmaf(params.cost_delta, rate.magnitude, entropy);
      loss = fmaf(totals[channel], kChannelMultiplier[channel], loss);
    }
    const unsigned int nonzero_bits =
      CeilLog2Nonzero(rate.nonzero_count + 1) + 1;
    entropy = fmaf(params.zeros_multiplier, static_cast<float>(
      CeilLog2Nonzero(nonzero_bits + 17) + nonzero_bits), entropy);
  }
  const float quant_norm = costs[candidate_index];
  const float normalized_loss = loss / static_cast<float>(coefficient_count);
  const float loss_cost = powf(normalized_loss, 0.125f) *
    static_cast<float>(coefficient_count) / quant_norm;
  const float result = entropy * candidate.entropy_multiplier +
    params.info_loss_multiplier * loss_cost;
  costs[candidate_index] =
    isfinite(result) && result >= 0.0f ? result : NAN;
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
  // Costs are not consumed until this ordered batch completes. Reuse that
  // storage for one quant norm per candidate through residual evaluation;
  // FinalizeCostKernel reads the norm before writing its final result.
  const unsigned int norm_blocks =
    (params.candidate_count + kQuantNormThreads - 1) / kQuantNormThreads;
  PrepareQuantNormsKernel<<<norm_blocks, kQuantNormThreads, 0, stream>>>(
    typed_candidates, quant_field, costs, params);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  error = LaunchCudaAcStrategyForward(
    opsin_x, opsin_y, opsin_b, candidates, scratch_b, params, stream);
  if (error != cudaSuccess) return error;

  // The inverse consumes forward coefficients in scratch B while writing
  // compact loss sums to scratch A; those ranges must remain disjoint.
  error = LaunchCudaAcStrategyResidualInverseLoss(
    scratch_b, matrices, costs, y_to_x, y_to_b, pixel_mask, candidates,
    rate_scratch, scratch_a, params, stream);
  if (error != cudaSuccess) return error;

  const unsigned int cost_blocks =
    (params.candidate_count + kCostThreads - 1) / kCostThreads;
  FinalizeCostKernel<<<cost_blocks, kCostThreads, 0, stream>>>(
    scratch_a, typed_candidates, static_cast<const ChannelRate*>(rate_scratch),
    costs, params);
  return cudaGetLastError();
}

}  // namespace gjxl::cuda_internal
