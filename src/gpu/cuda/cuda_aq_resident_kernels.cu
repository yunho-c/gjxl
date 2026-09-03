// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "gpu/cuda/cuda_aq_resident_kernels.h"

namespace gjxl::cuda_internal {
namespace {

constexpr unsigned int kThreads = 256;
constexpr float kPi = 3.14159265358979323846f;

struct TableOffsets {
  uint32_t dequant;
  uint32_t inverse_dequant;
};

__device__ TableOffsets QuantTableOffsets(uint32_t strategy) {
  switch (strategy) {
    case 0:
      return {0, 192};
    case 4:
      return {384, 1152};
    case 5:
      return {1920, 4992};
    case 6:
    case 7:
      return {8064, 8448};
    case 10:
    case 11:
      return {8832, 10368};
    default:
      return {0, 192};
  }
}

__device__ uint32_t CoefficientIndex(CudaAqExactBatch batch, uint32_t v,
                                     uint32_t u) {
  return batch.pixel_height < batch.pixel_width ? v * batch.pixel_width + u
                                                : u * batch.pixel_height + v;
}

__device__ float DownsampleScale(uint32_t length, uint32_t frequency) {
  if (length == 1) return 1.0f;
  if (length == 2) return frequency == 0 ? 1.0f : 0.901764195028874394f;
  constexpr float kScale[4] = {1.0f, 0.974886821136879522f,
                               0.901764195028874394f, 0.787054918159101335f};
  return kScale[frequency];
}

__device__ float UpsampleScale(uint32_t length, uint32_t frequency) {
  if (length == 1) return 1.0f;
  if (length == 2) return frequency == 0 ? 1.0f : 1.108937353592731823f;
  constexpr float kScale[4] = {1.0f, 1.025760096781116015f,
                               1.108937353592731823f, 1.270559368765487251f};
  return kScale[frequency];
}

__device__ float ForwardBasis(uint32_t length, uint32_t frequency,
                              uint32_t sample) {
  const float alpha = frequency == 0 ? 0.7071067811865475244f : 1.0f;
  const float angle = (static_cast<float>(sample) + 0.5f) *
                      static_cast<float>(frequency) * kPi /
                      static_cast<float>(length);
  return 1.4142135623730950488f * alpha * cosf(angle) /
         static_cast<float>(length);
}

__device__ float InverseBasis(uint32_t length, uint32_t frequency,
                              uint32_t sample) {
  const float alpha = frequency == 0 ? 0.7071067811865475244f : 1.0f;
  const float angle = (static_cast<float>(sample) + 0.5f) *
                      static_cast<float>(frequency) * kPi /
                      static_cast<float>(length);
  return 1.4142135623730950488f * alpha * cosf(angle);
}

__device__ signed char QuantizeCfl(float value) {
  constexpr float kTowardsZero = 2.6f;
  if (value >= kTowardsZero) {
    value -= kTowardsZero;
  } else if (value <= -kTowardsZero) {
    value += kTowardsZero;
  } else {
    value = 0.0f;
  }
  return static_cast<signed char>(fminf(fmaxf(roundf(value), -128.0f), 127.0f));
}

__device__ float QuantizationThreshold(uint32_t channel, uint32_t covered_count,
                                       uint32_t coefficient, uint32_t width,
                                       uint32_t height) {
  const uint32_t x = coefficient % width;
  const uint32_t y = coefficient / width;
  const uint32_t quadrant =
      (y >= height / 2 ? 2u : 0u) + (x >= width / 2 ? 1u : 0u);
  float threshold;
  if (channel == 1) {
    threshold = quadrant == 0 ? 0.58f : 0.64f;
  } else {
    threshold = quadrant == 0 ? 0.58f : 0.62f;
    if (covered_count >= 4) {
      threshold = fmaxf(0.5f, threshold - 0.00744f * covered_count);
    }
  }
  return threshold;
}

__device__ int QuantizeCoefficient(float coefficient, float inverse_dequant,
                                   uint32_t global_scale, int raw_quant,
                                   float matrix_multiplier, float threshold,
                                   unsigned int* error) {
  const float quantization_scale =
      static_cast<float>(global_scale) * (1.0f / 65536.0f) *
      static_cast<float>(raw_quant) * matrix_multiplier;
  const float value = inverse_dequant * quantization_scale * coefficient;
  if (!isfinite(coefficient) || !isfinite(value)) {
    atomicOr(error, 1u);
    return 0;
  }
  if (fabsf(value) < threshold) return 0;
  const float rounded = rintf(value);
  if (!isfinite(rounded) || rounded < -2147483648.0f ||
      rounded >= 2147483648.0f) {
    atomicOr(error, 2u);
    return 0;
  }
  return static_cast<int>(rounded);
}

__device__ float AdjustQuantizationBias(int quantized, uint32_t channel) {
  constexpr float kBias[3] = {1.0f - 0.05465007330715401f,
                              1.0f - 0.07005449891748593f,
                              1.0f - 0.049935103337343655f};
  const float value = static_cast<float>(quantized);
  const float absolute = fabsf(value);
  if (absolute < 0.125f) return 0.0f;
  if (absolute < 1.125f) return copysignf(kBias[channel], value);
  return value - 0.145f / value;
}

__device__ float DequantizeCoefficient(int quantized, float dequant,
                                       uint32_t global_scale, int raw_quant,
                                       float matrix_multiplier,
                                       uint32_t channel, unsigned int* error) {
  const float scale = (65536.0f / static_cast<float>(global_scale)) /
                      (static_cast<float>(raw_quant) * matrix_multiplier);
  const float coefficient =
      AdjustQuantizationBias(quantized, channel) * dequant * scale;
  if (!isfinite(coefficient)) {
    atomicOr(error, 4u);
    return 0.0f;
  }
  return coefficient;
}

__device__ int RoundDc(float value, unsigned int* error) {
  const float rounded = roundf(value);
  if (!isfinite(rounded) || rounded < -2147483648.0f ||
      rounded >= 2147483648.0f) {
    atomicOr(error, 8u);
    return 0;
  }
  return static_cast<int>(rounded);
}

__device__ int AdjustQuantForChannel(
    const float* coefficients, const float* quant_tables,
    uint32_t coefficient_count, uint32_t channel_stride,
    uint32_t coefficient_width, uint32_t coefficient_height, uint32_t strategy,
    uint32_t channel, uint32_t global_scale, int initial_raw,
    float matrix_multiplier, float thresholds[4], unsigned int* error) {
  const uint32_t xsize = coefficient_width / 8;
  const uint32_t ysize = coefficient_height / 8;
  const TableOffsets table_offsets = QuantTableOffsets(strategy);
  const float qac = static_cast<float>(global_scale) * (1.0f / 65536.0f) *
                    static_cast<float>(initial_raw);
  if (xsize > 1 || ysize > 1) {
    const float reduction =
        fminf(fmaxf(0.003f * static_cast<float>(xsize * ysize), 0.0f), 0.08f);
    for (uint32_t quadrant = 0; quadrant < 4; ++quadrant) {
      thresholds[quadrant] = fmaxf(0.54f, thresholds[quadrant] - reduction);
    }
  }

  float border_sum = 0.0f;
  float error_sum = 0.0f;
  float value_sum = 0.0f;
  float nonzeros[4] = {};
  float max_error[4] = {};
  for (uint32_t y = 0; y < coefficient_height; ++y) {
    for (uint32_t x = 0; x < coefficient_width; ++x) {
      if (x < xsize && y < ysize) continue;
      const uint32_t index = y * coefficient_width + x;
      const uint32_t quadrant = (y >= coefficient_height / 2 ? 2u : 0u) +
                                (x >= coefficient_width / 2 ? 1u : 0u);
      const uint32_t table = channel * coefficient_count + index;
      const float coefficient = coefficients[channel * channel_stride + index];
      const float value =
          coefficient * (quant_tables[table_offsets.inverse_dequant + table] *
                         qac * matrix_multiplier);
      if (!isfinite(coefficient) || !isfinite(value)) {
        atomicOr(error, 16u);
        continue;
      }
      const float quantized =
          fabsf(value) < thresholds[quadrant] ? 0.0f : rintf(value);
      const float quantization_error = fabsf(value - quantized);
      error_sum += quantization_error;
      value_sum += fabsf(quantized);
      if (channel == 1 && quantized == 0.0f) {
        max_error[quadrant] = fmaxf(max_error[quadrant], quantization_error);
      }
      if (quantized != 0.0f) {
        nonzeros[quadrant] += fabsf(quantized);
        const bool in_corner = y >= 7 * ysize && x >= 7 * xsize;
        const bool on_border =
            y + 1 == coefficient_height || x + 1 == coefficient_width;
        const bool in_larger_corner = x >= 4 * xsize && y >= 4 * ysize;
        if (in_corner || (on_border && in_larger_corner)) {
          border_sum += fabsf(value);
        }
      }
    }
  }

  int raw = initial_raw;
  if (channel == 1 && value_sum * 8.0f < static_cast<float>(xsize * ysize)) {
    constexpr float kLimit = 0.46f;
    int candidate = initial_raw;
    for (uint32_t quadrant = 1; quadrant < 4; ++quadrant) {
      if (nonzeros[quadrant] == 0.0f && max_error[quadrant] > kLimit) {
        candidate = initial_raw + 1;
        break;
      }
    }
    raw = candidate;
    if (nonzeros[3] == 0.0f && max_error[3] > kLimit) {
      thresholds[3] = 0.9999f * max_error[3] * candidate / initial_raw;
    } else if ((nonzeros[1] == 0.0f && max_error[1] > kLimit) ||
               (nonzeros[2] == 0.0f && max_error[2] > kLimit)) {
      thresholds[1] =
          0.9999f * fmaxf(max_error[1], max_error[2]) * candidate / initial_raw;
      thresholds[2] = thresholds[1];
    } else if (nonzeros[0] == 0.0f && max_error[0] > kLimit) {
      thresholds[0] = 0.9999f * max_error[0] * candidate / initial_raw;
    }
  }

  const float all_nonzeros =
      nonzeros[0] + nonzeros[1] + nonzeros[2] + nonzeros[3] + 1.0f;
  constexpr float kBorderMultiplier[3] = {70.0f, 30.0f, 60.0f};
  if (kBorderMultiplier[channel] * border_sum >= all_nonzeros) {
    raw = static_cast<int>(raw + kBorderMultiplier[channel] * border_sum /
                                     all_nonzeros);
    if (raw >= 256) raw = 255;
  }
  if (strategy == 0 &&
      nonzeros[0] + nonzeros[1] + nonzeros[2] + nonzeros[3] < 11.0f) {
    if (++raw >= 256) raw = 255;
  }

  constexpr float kFirstMultiplier[4][3] = {
      {0.22080615753848404f, 0.45797479824262011f, 0.29859235095977965f},
      {0.70109486510286834f, 0.16185281305512639f, 0.14387691730035473f},
      {0.114985964456218638f, 0.44656840441027695f, 0.10587658215149048f},
      {0.46849665264409396f, 0.41239077937781954f, 0.088667407767185444f}};
  constexpr float kSecondMultiplier[4][3] = {
      {0.27450281941822197f, 1.1255766549984996f, 0.98950459134128388f},
      {0.4652168675598285f, 0.40945807983455818f, 0.36581899811751367f},
      {0.28034972424715715f, 0.9182653201929738f, 1.5581531543057416f},
      {0.26873118114033728f, 0.68863712390392484f, 1.2082185408666786f}};
  error_sum *= 2.2942708343284721f;
  value_sum *= 2.2942708343284721f;
  if (strategy >= 4) {
    uint32_t strategy_class = 3;
    if (strategy == 10 || strategy == 11) {
      strategy_class = 1;
    } else if (strategy == 4) {
      strategy_class = 0;
    } else if (strategy == 5) {
      strategy_class = 2;
    }
    const float threshold =
        kFirstMultiplier[strategy_class][channel] *
            static_cast<float>(xsize * ysize * 64) +
        kSecondMultiplier[strategy_class][channel] * value_sum;
    const int step = max(0, min(2, static_cast<int>(error_sum / threshold)));
    if (error_sum > threshold) {
      raw += step;
      if (raw >= 256) raw = 255;
    }
  }

  const int divisor = static_cast<int>(xsize * ysize);
  const float minimum =
      fminf(fminf(nonzeros[0], nonzeros[1]), fminf(nonzeros[2], nonzeros[3]));
  int activity = 15;
  if (minimum < static_cast<float>(15 * divisor)) {
    activity = (static_cast<int>(minimum) + divisor / 2) / divisor;
  }
  int adjusted = raw - activity;
  if (channel == 1) {
    for (uint32_t quadrant = 1; quadrant < 4; ++quadrant) {
      thresholds[quadrant] += 0.01f * activity;
    }
  }
  const int limit = max(4, raw / 2);
  if (adjusted < limit) adjusted = limit;
  return adjusted;
}

__global__ void AdjustQuantFieldKernel(const CudaAqAnchor* anchors,
                                       float* quant_field, unsigned int* error,
                                       uint32_t quant_stride,
                                       CudaAqExactBatch batch,
                                       float mean_max_mixer) {
  const uint32_t anchor_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (anchor_index >= batch.anchor_count) return;
  const CudaAqAnchor anchor = anchors[batch.anchor_offset + anchor_index];
  const uint32_t covered_count = batch.covered_width * batch.covered_height;
  float maximum =
      quant_field[static_cast<size_t>(anchor.y) * quant_stride + anchor.x];
  float mean = 0.0f;
  for (uint32_t y = 0; y < batch.covered_height; ++y) {
    for (uint32_t x = 0; x < batch.covered_width; ++x) {
      const float value =
          quant_field[static_cast<size_t>(anchor.y + y) * quant_stride +
                      anchor.x + x];
      maximum = fmaxf(maximum, value);
      mean += value;
    }
  }
  mean /= static_cast<float>(covered_count);
  const float result = covered_count < 4 ? maximum
                                         : maximum * mean_max_mixer +
                                               mean * (1.0f - mean_max_mixer);
  if (!isfinite(result) || result <= 0.0f) {
    atomicOr(error, 32u);
    return;
  }
  for (uint32_t y = 0; y < batch.covered_height; ++y) {
    for (uint32_t x = 0; x < batch.covered_width; ++x) {
      quant_field[static_cast<size_t>(anchor.y + y) * quant_stride + anchor.x +
                  x] = result;
    }
  }
}

__global__ void GatherTransformPixelsKernel(
    const float* coding_x, const float* coding_y, const float* coding_b,
    const CudaAqAnchor* anchors, float* gathered, CudaAqExactBatch batch,
    uint32_t coding_stride) {
  const size_t values_per_channel =
      static_cast<size_t>(batch.anchor_count) * batch.coefficient_count;
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 3 * values_per_channel) return;
  const uint32_t channel = static_cast<uint32_t>(index / values_per_channel);
  const size_t channel_index = index - channel * values_per_channel;
  const uint32_t anchor_index =
      static_cast<uint32_t>(channel_index / batch.coefficient_count);
  const uint32_t pixel_index =
      static_cast<uint32_t>(channel_index % batch.coefficient_count);
  const uint32_t x = pixel_index % batch.pixel_width;
  const uint32_t y = pixel_index / batch.pixel_width;
  const CudaAqAnchor anchor = anchors[batch.anchor_offset + anchor_index];
  const size_t source_index =
      static_cast<size_t>(anchor.y * 8 + y) * coding_stride + anchor.x * 8 + x;
  const float* source =
      channel == 0 ? coding_x : (channel == 1 ? coding_y : coding_b);
  gathered[batch.coefficient_offset + index] = source[source_index];
}

__global__ void FinalColorCorrelationKernel(
    const CudaAqColorTransformRecord* transforms, const uint32_t* tile_offsets,
    const float* quant_tables, const float* forward_coefficients,
    const int* raw_quant, const unsigned int* quantizer, signed char* y_to_x,
    signed char* y_to_b, unsigned int* error, uint32_t tile_count) {
  const uint32_t tile = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  if (tile >= tile_count || lane >= 4) return;
  const uint32_t begin = tile_offsets[tile];
  const uint32_t end = tile_offsets[tile + 1];
  if (begin >= end) {
    if (lane == 0) atomicOr(error, 64u);
    return;
  }
  const uint32_t global_scale = quantizer[0];
  float quadratic_x = 0.0f;
  float linear_x = 0.0f;
  float quadratic_b = 0.0f;
  float linear_b = 0.0f;
  for (uint32_t transform_index = begin; transform_index < end;
       ++transform_index) {
    const CudaAqColorTransformRecord transform = transforms[transform_index];
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t low_width = 0;
    uint32_t low_height = 0;
    switch (transform.strategy) {
      case 0:
        width = height = 8;
        low_width = low_height = 1;
        break;
      case 4:
        width = height = 16;
        low_width = low_height = 2;
        break;
      case 5:
        width = height = 32;
        low_width = low_height = 4;
        break;
      case 6:
      case 7:
        width = 16;
        height = 8;
        low_width = 2;
        low_height = 1;
        break;
      case 10:
      case 11:
        width = 32;
        height = 16;
        low_width = 4;
        low_height = 2;
        break;
      default:
        atomicOr(error, 64u);
        continue;
    }
    const int raw = raw_quant[transform.raw_quant_index];
    if (width * height != transform.coefficient_count || raw < 1 || raw > 256 ||
        global_scale == 0 || global_scale > 32768) {
      atomicOr(error, 64u);
      continue;
    }
    const float quant_scale =
        static_cast<float>(global_scale) * (1.0f / 65536.0f) * 128.0f * raw;
    const TableOffsets table_offsets = QuantTableOffsets(transform.strategy);
    const uint32_t first = (lane + 4 - (transform.tile_value_offset & 3u)) & 3u;
    for (uint32_t coefficient = first;
         coefficient < transform.coefficient_count; coefficient += 4) {
      const uint32_t x = coefficient % width;
      const uint32_t y = coefficient / width;
      if (x < low_width && y < low_height) continue;
      const size_t base = transform.coefficient_offset + coefficient;
      const float coefficient_y =
          forward_coefficients[base + transform.channel_stride];
      const float coefficient_x = forward_coefficients[base];
      const float coefficient_b =
          forward_coefficients[base + 2 * transform.channel_stride];
      const float value_y_x =
          coefficient_y *
          quant_tables[table_offsets.inverse_dequant + coefficient] *
          quant_scale;
      const float value_x =
          coefficient_x *
          quant_tables[table_offsets.inverse_dequant + coefficient] *
          quant_scale;
      const uint32_t b_table = 2 * transform.coefficient_count + coefficient;
      const float value_y_b =
          coefficient_y *
          quant_tables[table_offsets.inverse_dequant + b_table] * quant_scale;
      const float value_b =
          coefficient_b *
          quant_tables[table_offsets.inverse_dequant + b_table] * quant_scale;
      const float a_x = value_y_x * (1.0f / 84.0f);
      const float a_b = value_y_b * (1.0f / 84.0f);
      quadratic_x = fmaf(a_x, a_x, quadratic_x);
      linear_x = fmaf(a_x, -value_x, linear_x);
      quadratic_b = fmaf(a_b, a_b, quadratic_b);
      linear_b = fmaf(a_b, value_y_b - value_b, linear_b);
    }
  }
  __shared__ float values[16];
  values[lane] = quadratic_x;
  values[4 + lane] = linear_x;
  values[8 + lane] = quadratic_b;
  values[12 + lane] = linear_b;
  __syncthreads();
  if (lane != 0) return;
  const CudaAqColorTransformRecord last = transforms[end - 1];
  const float sample_count =
      static_cast<float>(last.tile_value_offset + last.coefficient_count);
  const float qx = (values[0] + values[1]) + (values[2] + values[3]);
  const float lx = (values[4] + values[5]) + (values[6] + values[7]);
  const float qb = (values[8] + values[9]) + (values[10] + values[11]);
  const float lb = (values[12] + values[13]) + (values[14] + values[15]);
  const float result_x = -lx / (qx + sample_count * 5.0e-10f);
  const float result_b = -lb / (qb + sample_count * 5.0e-10f);
  if (!isfinite(result_x) || !isfinite(result_b)) {
    atomicOr(error, 64u);
    return;
  }
  y_to_x[tile] = QuantizeCfl(result_x);
  y_to_b[tile] = QuantizeCfl(result_b);
}

__global__ void SelectAdjustedQuantizationKernel(
    const CudaAqAnchor* anchors, const float* quant_tables, int* raw_quant,
    const float* forward_coefficients, float* adjustment_thresholds,
    const unsigned int* quantizer, unsigned int* error, CudaAqExactBatch batch,
    CudaAqResidentParams params) {
  const uint32_t anchor_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (anchor_index >= batch.anchor_count) return;
  const CudaAqAnchor anchor = anchors[batch.anchor_offset + anchor_index];
  const uint32_t channel_stride = batch.anchor_count * batch.coefficient_count;
  const size_t transform_offset =
      batch.coefficient_offset +
      static_cast<size_t>(anchor_index) * batch.coefficient_count;
  const uint32_t coefficient_width = max(batch.pixel_width, batch.pixel_height);
  const uint32_t coefficient_height =
      min(batch.pixel_width, batch.pixel_height);
  const size_t raw_index =
      static_cast<size_t>(anchor.y) * params.block_width + anchor.x;
  const uint32_t channels[3] = {1, 0, 2};
  int selected = 0;
  float selected_y[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  for (uint32_t channel_index = 0; channel_index < 3; ++channel_index) {
    const uint32_t channel = channels[channel_index];
    float thresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};
    const float multiplier = channel == 0   ? params.x_matrix_multiplier
                             : channel == 2 ? params.b_matrix_multiplier
                                            : 1.0f;
    const int candidate = AdjustQuantForChannel(
        forward_coefficients + transform_offset, quant_tables,
        batch.coefficient_count, channel_stride, coefficient_width,
        coefficient_height, params.strategy, channel, quantizer[0],
        raw_quant[raw_index], multiplier, thresholds, error);
    if (channel == 1) {
      for (uint32_t quadrant = 0; quadrant < 4; ++quadrant) {
        selected_y[quadrant] = thresholds[quadrant];
      }
    }
    selected = max(selected, candidate);
  }
  raw_quant[raw_index] = selected;
  const size_t threshold_offset =
      batch.coefficient_offset + 4 * static_cast<size_t>(anchor_index);
  for (uint32_t quadrant = 0; quadrant < 4; ++quadrant) {
    adjustment_thresholds[threshold_offset + quadrant] = selected_y[quadrant];
  }
}

__global__ void EncodeResidentCoefficientsKernel(
    const CudaAqAnchor* anchors, const float* quant_tables,
    const int* raw_quant, const signed char* y_to_x, const signed char* y_to_b,
    const float* forward_coefficients, int* quantized_coefficients,
    float* reconstruction_coefficients, float* dc, int* quantized_dc,
    float* inverse_sigma, const unsigned char* epf_sharpness,
    const unsigned int* quantizer, const float* adjustment_thresholds,
    unsigned int* error, CudaAqExactBatch batch, CudaAqResidentParams params) {
  const uint32_t anchor_index = blockIdx.x;
  const uint32_t thread_index = threadIdx.x;
  if (anchor_index >= batch.anchor_count) return;
  const CudaAqAnchor anchor = anchors[batch.anchor_offset + anchor_index];
  const uint32_t block_count = params.block_width * params.block_height;
  const uint32_t covered_count = batch.covered_width * batch.covered_height;
  const uint32_t channel_stride = batch.anchor_count * batch.coefficient_count;
  const size_t transform_offset =
      batch.coefficient_offset +
      static_cast<size_t>(anchor_index) * batch.coefficient_count;
  const TableOffsets table_offsets = QuantTableOffsets(params.strategy);
  const uint32_t coefficient_width = max(batch.pixel_width, batch.pixel_height);
  const uint32_t coefficient_height =
      min(batch.pixel_width, batch.pixel_height);
  const size_t raw_index =
      static_cast<size_t>(anchor.y) * params.block_width + anchor.x;
  const int raw = raw_quant[raw_index];
  const size_t color_index =
      static_cast<size_t>(anchor.y / 8) * params.color_stride + anchor.x / 8;
  const float cfl_x = static_cast<float>(y_to_x[color_index]) * (1.0f / 84.0f);
  const float cfl_b =
      1.0f + static_cast<float>(y_to_b[color_index]) * (1.0f / 84.0f);
  float selected_y[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  if (params.adjust_ac_quant != 0) {
    const size_t threshold_offset =
        batch.coefficient_offset + 4 * static_cast<size_t>(anchor_index);
    for (uint32_t quadrant = 0; quadrant < 4; ++quadrant) {
      selected_y[quadrant] = adjustment_thresholds[threshold_offset + quadrant];
    }
  }

  if (params.adjust_ac_quant != 0) {
    constexpr float kInverseSigmaNumerator = -1.1715728752538099024f;
    const float quantizer_scale =
        static_cast<float>(quantizer[0]) * (1.0f / 65536.0f);
    const float sigma_quant =
        params.epf_quant_multiplier /
        (quantizer_scale * static_cast<float>(raw) * kInverseSigmaNumerator);
    for (uint32_t block = thread_index; block < covered_count;
         block += blockDim.x) {
      const uint32_t x = block % batch.covered_width;
      const uint32_t y = block / batch.covered_width;
      const size_t sharpness_index =
          static_cast<size_t>(anchor.y + y) * params.block_width + anchor.x + x;
      const uint32_t sharpness = epf_sharpness[sharpness_index];
      if (sharpness >= 8) {
        atomicOr(error, 128u);
        continue;
      }
      float sigma = sigma_quant * params.epf_sharpness_lut[sharpness];
      sigma = fminf(-1.0e-4f, sigma);
      const float value = 1.0f / sigma;
      if (!isfinite(value) || value >= 0.0f) {
        atomicOr(error, 128u);
        continue;
      }
      inverse_sigma[sharpness_index] = value;
    }
  }
  __syncthreads();

  for (uint32_t dc_task = thread_index; dc_task < 3 * covered_count;
       dc_task += blockDim.x) {
    const uint32_t channel = dc_task / covered_count;
    const uint32_t small = dc_task - channel * covered_count;
    const uint32_t x = small % batch.covered_width;
    const uint32_t y = small / batch.covered_width;
    float value = 0.0f;
    for (uint32_t v = 0; v < batch.covered_height; ++v) {
      for (uint32_t u = 0; u < batch.covered_width; ++u) {
        const uint32_t coefficient = CoefficientIndex(batch, v, u);
        const float scaled =
            forward_coefficients[transform_offset + channel * channel_stride +
                                 coefficient] *
            DownsampleScale(batch.covered_height, v) *
            DownsampleScale(batch.covered_width, u);
        value += scaled * InverseBasis(batch.covered_height, v, y) *
                 InverseBasis(batch.covered_width, u, x);
      }
    }
    if (!isfinite(value)) {
      atomicOr(error, 256u);
      value = 0.0f;
    }
    dc[static_cast<size_t>(channel) * block_count +
       static_cast<size_t>(anchor.y + y) * params.block_width + anchor.x + x] =
        value;
  }
  __syncthreads();

  for (uint32_t small = thread_index; small < covered_count;
       small += blockDim.x) {
    const uint32_t x = small % batch.covered_width;
    const uint32_t y = small / batch.covered_width;
    const size_t block_index =
        static_cast<size_t>(anchor.y + y) * params.block_width + anchor.x + x;
    const float quant_scale =
        static_cast<float>(quantizer[0]) * (1.0f / 65536.0f) * quantizer[1];
    const float inverse_x = 4096.0f * quant_scale;
    const float inverse_y = 512.0f * quant_scale;
    const float inverse_b = 256.0f * quant_scale;
    const int quantized_y =
        RoundDc(dc[block_count + block_index] * inverse_y, error);
    const float reconstructed_y = quantized_y / inverse_y;
    const int quantized_x = RoundDc(dc[block_index] * inverse_x, error);
    const int quantized_b = RoundDc(
        (dc[2 * block_count + block_index] - reconstructed_y) * inverse_b,
        error);
    quantized_dc[block_index] = quantized_x;
    quantized_dc[block_count + block_index] = quantized_y;
    quantized_dc[2 * block_count + block_index] = quantized_b;
    dc[block_index] = quantized_x / inverse_x;
    dc[block_count + block_index] = reconstructed_y;
    dc[2 * block_count + block_index] =
        quantized_b / inverse_b + reconstructed_y;
  }
  __syncthreads();

  for (uint32_t coefficient = thread_index;
       coefficient < batch.coefficient_count; coefficient += blockDim.x) {
    const uint32_t channel = 1;
    const size_t offset =
        transform_offset + channel * channel_stride + coefficient;
    const uint32_t x = coefficient % coefficient_width;
    const uint32_t y = coefficient / coefficient_width;
    const uint32_t quadrant = (y >= coefficient_height / 2 ? 2u : 0u) +
                              (x >= coefficient_width / 2 ? 1u : 0u);
    const float threshold =
        params.adjust_ac_quant != 0
            ? selected_y[quadrant]
            : QuantizationThreshold(channel, covered_count, coefficient,
                                    coefficient_width, coefficient_height);
    const uint32_t table = channel * batch.coefficient_count + coefficient;
    const int quantized =
        QuantizeCoefficient(forward_coefficients[offset],
                            quant_tables[table_offsets.inverse_dequant + table],
                            quantizer[0], raw, 1.0f, threshold, error);
    quantized_coefficients[offset] = quantized;
    reconstruction_coefficients[offset] = DequantizeCoefficient(
        quantized, quant_tables[table_offsets.dequant + table], quantizer[0],
        raw, 1.0f, channel, error);
  }
  __syncthreads();

  for (uint32_t coefficient = thread_index;
       coefficient < batch.coefficient_count; coefficient += blockDim.x) {
    const float reconstructed_y =
        reconstruction_coefficients[transform_offset + channel_stride +
                                    coefficient];
    for (uint32_t channel_pass = 0; channel_pass < 2; ++channel_pass) {
      const uint32_t channel = channel_pass == 0 ? 0 : 2;
      const size_t offset =
          transform_offset + channel * channel_stride + coefficient;
      const float factor = channel == 0 ? cfl_x : cfl_b;
      const float multiplier = channel == 0 ? params.x_matrix_multiplier
                                            : params.b_matrix_multiplier;
      const float predicted =
          forward_coefficients[offset] - factor * reconstructed_y;
      const float threshold =
          QuantizationThreshold(channel, covered_count, coefficient,
                                coefficient_width, coefficient_height);
      const uint32_t table = channel * batch.coefficient_count + coefficient;
      const int quantized = QuantizeCoefficient(
          predicted, quant_tables[table_offsets.inverse_dequant + table],
          quantizer[0], raw, multiplier, threshold, error);
      quantized_coefficients[offset] = quantized;
      reconstruction_coefficients[offset] = DequantizeCoefficient(
          quantized, quant_tables[table_offsets.dequant + table], quantizer[0],
          raw, multiplier, channel, error);
    }
  }
  __syncthreads();

  for (uint32_t coefficient = thread_index;
       coefficient < batch.coefficient_count; coefficient += blockDim.x) {
    const float reconstructed_y =
        reconstruction_coefficients[transform_offset + channel_stride +
                                    coefficient];
    reconstruction_coefficients[transform_offset + coefficient] +=
        cfl_x * reconstructed_y;
    reconstruction_coefficients[transform_offset + 2 * channel_stride +
                                coefficient] += cfl_b * reconstructed_y;
  }
  __syncthreads();

  for (uint32_t llf_task = thread_index; llf_task < 3 * covered_count;
       llf_task += blockDim.x) {
    const uint32_t channel = llf_task / covered_count;
    const uint32_t small = llf_task - channel * covered_count;
    const uint32_t u = small % batch.covered_width;
    const uint32_t v = small / batch.covered_width;
    float value = 0.0f;
    for (uint32_t y = 0; y < batch.covered_height; ++y) {
      for (uint32_t x = 0; x < batch.covered_width; ++x) {
        value += dc[static_cast<size_t>(channel) * block_count +
                    static_cast<size_t>(anchor.y + y) * params.block_width +
                    anchor.x + x] *
                 ForwardBasis(batch.covered_height, v, y) *
                 ForwardBasis(batch.covered_width, u, x);
      }
    }
    value *= UpsampleScale(batch.covered_height, v) *
             UpsampleScale(batch.covered_width, u);
    reconstruction_coefficients[transform_offset + channel * channel_stride +
                                CoefficientIndex(batch, v, u)] = value;
  }
}

}  // namespace

cudaError_t LaunchCudaAqAdjustQuantField(
    const CudaAqAnchor* anchors, float* quant_field, unsigned int* error,
    uint32_t quant_stride, CudaAqExactBatch batch, float mean_max_mixer,
    cudaStream_t stream) {
  const uint32_t blocks = (batch.anchor_count + kThreads - 1) / kThreads;
  AdjustQuantFieldKernel<<<blocks, kThreads, 0, stream>>>(
      anchors, quant_field, error, quant_stride, batch, mean_max_mixer);
  return cudaPeekAtLastError();
}

cudaError_t LaunchCudaAqGatherTransformPixels(
    const float* coding_x, const float* coding_y, const float* coding_b,
    const CudaAqAnchor* anchors, float* gathered, CudaAqExactBatch batch,
    uint32_t coding_stride, cudaStream_t stream) {
  const size_t count =
      3 * static_cast<size_t>(batch.anchor_count) * batch.coefficient_count;
  const uint32_t blocks =
      static_cast<uint32_t>((count + kThreads - 1) / kThreads);
  GatherTransformPixelsKernel<<<blocks, kThreads, 0, stream>>>(
      coding_x, coding_y, coding_b, anchors, gathered, batch, coding_stride);
  return cudaPeekAtLastError();
}

cudaError_t LaunchCudaAqFinalColorCorrelation(
    const CudaAqColorTransformRecord* transforms, const uint32_t* tile_offsets,
    const float* quant_tables, const float* forward_coefficients,
    const int* raw_quant, const unsigned int* quantizer, signed char* y_to_x,
    signed char* y_to_b, unsigned int* error, uint32_t tile_count,
    cudaStream_t stream) {
  FinalColorCorrelationKernel<<<tile_count, 4, 0, stream>>>(
      transforms, tile_offsets, quant_tables, forward_coefficients, raw_quant,
      quantizer, y_to_x, y_to_b, error, tile_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchCudaAqSelectAdjustedQuantization(
    const CudaAqAnchor* anchors, const float* quant_tables, int* raw_quant,
    const float* forward_coefficients, float* adjustment_thresholds,
    const unsigned int* quantizer, unsigned int* error, CudaAqExactBatch batch,
    CudaAqResidentParams params, cudaStream_t stream) {
  const uint32_t blocks = (batch.anchor_count + kThreads - 1) / kThreads;
  SelectAdjustedQuantizationKernel<<<blocks, kThreads, 0, stream>>>(
      anchors, quant_tables, raw_quant, forward_coefficients,
      adjustment_thresholds, quantizer, error, batch, params);
  return cudaPeekAtLastError();
}

cudaError_t LaunchCudaAqEncodeResidentCoefficients(
    const CudaAqAnchor* anchors, const float* quant_tables,
    const int* raw_quant, const signed char* y_to_x, const signed char* y_to_b,
    const float* forward_coefficients, int* quantized_coefficients,
    float* reconstruction_coefficients, float* dc, int* quantized_dc,
    float* inverse_sigma, const unsigned char* epf_sharpness,
    const unsigned int* quantizer, const float* adjustment_thresholds,
    unsigned int* error, CudaAqExactBatch batch, CudaAqResidentParams params,
    cudaStream_t stream) {
  EncodeResidentCoefficientsKernel<<<batch.anchor_count, kThreads, 0, stream>>>(
      anchors, quant_tables, raw_quant, y_to_x, y_to_b, forward_coefficients,
      quantized_coefficients, reconstruction_coefficients, dc, quantized_dc,
      inverse_sigma, epf_sharpness, quantizer, adjustment_thresholds, error,
      batch, params);
  return cudaPeekAtLastError();
}

}  // namespace gjxl::cuda_internal
