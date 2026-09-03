// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "gpu/cuda/cuda_kernels.h"

namespace gjxl::cuda_internal {
namespace {

constexpr unsigned int kThreads = 256;

__device__ float GammaRatio(float value, bool invert) {
  constexpr float kEpsilon = 1.0e-2f;
  constexpr float kSgMul = 226.77216153508914f;
  constexpr float kSgMul2 = 1.0f / 73.377132366608819f;
  constexpr float kInverseLog2E = 0.6931471805599453f;
  constexpr float kReturn = kSgMul2 * 18.6580932135f * kInverseLog2E;
  constexpr float kOffset = 7.7825991679894591f;
  value = fmaxf(value, 0.0f);
  const float squared = value * value;
  const float numerator = fmaf(kReturn * 3.0f * kSgMul, squared, kEpsilon);
  const float denominator = fmaf(kInverseLog2E * kSgMul * value, squared,
                                 kOffset * kInverseLog2E + kEpsilon);
  return invert ? numerator / denominator : denominator / numerator;
}

__device__ float MaskingSqrt(float value) {
  constexpr float kLogOffset = 27.505837037000106f;
  constexpr float kMul = 211.66567973503678f;
  return 0.25f * sqrtf(fmaf(value, sqrtf(kMul * 1.0e8f), kLogOffset));
}

__device__ float CompensatedLog1p(float value) {
  const float sum = 1.0f + value;
  return logf(sum) + (value - (sum - 1.0f)) / sum;
}

__global__ void InitialGradientKernel(const float* coding_y, float* pixel_mask,
                                      float* pre_erosion, unsigned int* error,
                                      CudaAqGeometry geometry) {
  const unsigned int sx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int sy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int pre_width = geometry.width / 4;
  const unsigned int pre_height = geometry.height / 4;
  if (sx >= pre_width || sy >= pre_height) return;

  constexpr float kGammaOffset = 0.019f;
  constexpr float kDifferenceLimit = 0.2f;
  float columns[4] = {};
  const unsigned int x_begin = sx * 4;
  const unsigned int y_begin = sy * 4;
  for (unsigned int row_index = 0; row_index < 4; ++row_index) {
    const unsigned int y = y_begin + row_index;
    const unsigned int top = y == 0 ? y : y - 1;
    const unsigned int bottom = min(y + 1, geometry.height - 1);
    for (unsigned int column = 0; column < 4; ++column) {
      const unsigned int x = x_begin + column;
      const unsigned int left = x == 0 ? x : x - 1;
      const unsigned int right = min(x + 1, geometry.width - 1);
      const float center =
          coding_y[static_cast<size_t>(y) * geometry.width + x];
      const float base =
          0.25f * (coding_y[static_cast<size_t>(bottom) * geometry.width + x] +
                   coding_y[static_cast<size_t>(top) * geometry.width + x] +
                   coding_y[static_cast<size_t>(y) * geometry.width + left] +
                   coding_y[static_cast<size_t>(y) * geometry.width + right]);
      const float delta =
          GammaRatio(center + kGammaOffset, false) * (center - base);
      const float mask = 1.0f / (CompensatedLog1p(fabsf(delta)) + 0.01f);
      pixel_mask[static_cast<size_t>(y) * geometry.width + x] = mask;
      const float difference =
          MaskingSqrt(fminf(delta * delta, kDifferenceLimit));
      columns[column] += difference;
      if (!isfinite(mask) || mask <= 0.0f || !isfinite(difference)) {
        atomicOr(error, 1u << 0);
      }
    }
  }
  const float value =
      (((columns[0] + columns[1]) + columns[2]) + columns[3]) * 0.25f;
  pre_erosion[static_cast<size_t>(sy) * pre_width + sx] = value;
  if (!isfinite(value) || value <= 0.0f) atomicOr(error, 1u << 0);
}

__device__ void StoreMin4(float value, float& a, float& b, float& c, float& d) {
  if (value >= d) return;
  if (value < a) {
    d = c;
    c = b;
    b = a;
    a = value;
  } else if (value < b) {
    d = c;
    c = b;
    b = value;
  } else if (value < c) {
    d = c;
    c = value;
  } else {
    d = value;
  }
}

__device__ void Sort4(float& a, float& b, float& c, float& d) {
  if (a > b) {
    const float t = a;
    a = b;
    b = t;
  }
  if (a > c) {
    const float t = a;
    a = c;
    c = t;
  }
  if (a > d) {
    const float t = a;
    a = d;
    d = t;
  }
  if (b > c) {
    const float t = b;
    b = c;
    c = t;
  }
  if (b > d) {
    const float t = b;
    b = d;
    d = t;
  }
  if (c > d) {
    const float t = c;
    c = d;
    d = t;
  }
}

__device__ float ErodedValue(const float* source, unsigned int x,
                             unsigned int y, unsigned int width,
                             unsigned int height, const float* weights) {
  const unsigned int top = y == 0 ? y : y - 1;
  const unsigned int bottom = min(y + 1, height - 1);
  const unsigned int left = x == 0 ? x : x - 1;
  const unsigned int right = min(x + 1, width - 1);
  const auto at = [&](unsigned int px, unsigned int py) {
    return source[static_cast<size_t>(py) * width + px];
  };
  float a = at(x, y);
  float b = at(left, y);
  float c = at(right, y);
  float d = at(left, top);
  Sort4(a, b, c, d);
  StoreMin4(at(x, top), a, b, c, d);
  StoreMin4(at(right, top), a, b, c, d);
  StoreMin4(at(left, bottom), a, b, c, d);
  StoreMin4(at(x, bottom), a, b, c, d);
  StoreMin4(at(right, bottom), a, b, c, d);
  return ((weights[0] * a + weights[1] * b) + weights[2] * c) + weights[3] * d;
}

__global__ void FuzzyErosionKernel(const float* pre_erosion, float* quant_field,
                                   float* strategy_mask, unsigned int* error,
                                   CudaAqGeometry geometry, float weight0,
                                   float weight1, float weight2,
                                   float weight3) {
  const unsigned int bx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= geometry.block_width || by >= geometry.block_height) return;
  const float weights[4] = {weight0, weight1, weight2, weight3};
  const unsigned int pre_width = geometry.width / 4;
  const unsigned int pre_height = geometry.height / 4;
  const unsigned int sx = bx * 2;
  const unsigned int sy = by * 2;
  float value =
      ErodedValue(pre_erosion, sx, sy, pre_width, pre_height, weights);
  value += ErodedValue(pre_erosion, sx + 1, sy, pre_width, pre_height, weights);
  value += ErodedValue(pre_erosion, sx, sy + 1, pre_width, pre_height, weights);
  value +=
      ErodedValue(pre_erosion, sx + 1, sy + 1, pre_width, pre_height, weights);
  const size_t index = static_cast<size_t>(by) * geometry.block_width + bx;
  const float strategy = 1.0f / (value + 0.001f);
  quant_field[index] = value;
  strategy_mask[index] = strategy;
  if (!isfinite(value) || value <= 0.0f || !isfinite(strategy) ||
      strategy <= 0.0f) {
    atomicOr(error, 1u << 1);
  }
}

__device__ float ComputeMask(float value) {
  constexpr float kBase = -0.7647f;
  constexpr float kMul4 = 9.4708735624378946f;
  constexpr float kMul2 = 17.35036561631863f;
  constexpr float kOffset2 = 302.59587815579727f;
  constexpr float kMul3 = 6.7943250517376494f;
  constexpr float kOffset3 = 3.7179635626140772f;
  constexpr float kOffset4 = 0.25f * kOffset3;
  constexpr float kMul0 = 0.80061762862741759f;
  const float v1 = fmaxf(value * kMul0, 1.0e-3f);
  const float v2 = 1.0f / (v1 + kOffset2);
  const float v3 = 1.0f / fmaf(v1, v1, kOffset3);
  const float v4 = 1.0f / fmaf(v1, v1, kOffset4);
  return kBase + fmaf(kMul4, v4, fmaf(kMul2, v2, kMul3 * v3));
}

__device__ float FastLog2(float value) {
  constexpr float kP0 = -1.8503833400518310e-06f;
  constexpr float kP1 = 1.4287160470083755f;
  constexpr float kP2 = 0.74245873327820566f;
  constexpr float kQ0 = 0.99032814277590719f;
  constexpr float kQ1 = 1.0096718572241148f;
  constexpr float kQ2 = 0.17409343003366853f;
  const unsigned int bits = __float_as_uint(value);
  const int exponent = static_cast<int>(bits - 0x3f2aaaabu) >> 23;
  const unsigned int mantissa =
      bits - (static_cast<unsigned int>(exponent) << 23);
  const float x = __uint_as_float(mantissa) - 1.0f;
  float numerator = fmaf(kP2, x, kP1);
  numerator = fmaf(numerator, x, kP0);
  float denominator = fmaf(kQ2, x, kQ1);
  denominator = fmaf(denominator, x, kQ0);
  return numerator / denominator + static_cast<float>(exponent);
}

__device__ float FastPow2(float value) {
  const float floored = floorf(value);
  const int exponent = static_cast<int>(floored) + 127;
  const float exponent_value =
      __uint_as_float(static_cast<unsigned int>(exponent) << 23);
  const float fraction = value - floored;
  float numerator = fraction + 1.01749063e+01f;
  numerator = fmaf(numerator, fraction, 4.88687798e+01f);
  numerator = fmaf(numerator, fraction, 9.85506591e+01f);
  numerator *= exponent_value;
  float denominator = fmaf(fraction, 2.10242958e-01f, -2.22328856e-02f);
  denominator = fmaf(denominator, fraction, -1.94414990e+01f);
  denominator = fmaf(denominator, fraction, 9.85506633e+01f);
  return numerator / denominator;
}

__global__ void ModulationKernel(const float* coding_x, const float* coding_y,
                                 const float* coding_b, float* quant_field,
                                 unsigned int* error, CudaAqGeometry geometry,
                                 float multiplier, float addend) {
  const unsigned int bx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= geometry.block_width || by >= geometry.block_height) return;
  constexpr float kGammaBias = 0.16f;
  constexpr float kFrequencyLimit = 0.0206f;
  constexpr float kBlueLimit = 0.010474084867598155f;
  constexpr float kBlueOffset = 0.0031994768654636393f;
  float gamma[4] = {};
  float frequency[4] = {};
  float blue[4] = {};
  const unsigned int pixel_x = bx * 8;
  const unsigned int pixel_y = by * 8;
  for (unsigned int dy = 0; dy < 8; ++dy) {
    const size_t row =
        static_cast<size_t>(pixel_y + dy) * geometry.width + pixel_x;
    const size_t next = dy + 1 < 8 ? row + geometry.width : row;
    for (unsigned int dx = 0; dx < 8; ++dx) {
      const unsigned int lane = dx & 3u;
      const float x = coding_x[row + dx];
      const float y = coding_y[row + dx];
      const float b = coding_b[row + dx];
      const float biased_y = y + kGammaBias;
      gamma[lane] += GammaRatio(biased_y - x, true);
      gamma[lane] += GammaRatio(biased_y + x, true);
      if (dx + 1 < 8) {
        frequency[lane] +=
            fminf(kFrequencyLimit, fabsf(y - coding_y[row + dx + 1]));
      }
      frequency[lane] += fminf(kFrequencyLimit, fabsf(y - coding_y[next + dx]));
      const float effective_y = y + kBlueOffset + fabsf(x);
      if (b > effective_y) blue[lane] += fminf(b - effective_y, kBlueLimit);
    }
  }
  const float overall =
      ((gamma[0] + gamma[1]) + (gamma[2] + gamma[3])) * (0.5f / 64.0f);
  const size_t index = static_cast<size_t>(by) * geometry.block_width + bx;
  const float gamma_value = fmaf(0.1005613337192697f, FastLog2(overall),
                                 ComputeMask(quant_field[index]));
  const float frequency_sum =
      (frequency[0] + frequency[1]) + (frequency[2] + frequency[3]);
  const float frequency_value = gamma_value + frequency_sum * -0.38f + 0.42f;
  float blue_sum = (blue[0] + blue[1]) + (blue[2] + blue[3]);
  if (blue_sum >= 32.0f * kBlueLimit) blue_sum = 64.0f * kBlueLimit - blue_sum;
  blue_sum = fminf(blue_sum, 15.463398341612438f * kBlueLimit);
  const float exponent =
      fminf(frequency_value, gamma_value + blue_sum * 0.90590804735610064f);
  const float result = FastPow2(exponent * 1.442695041f) * multiplier + addend;
  quant_field[index] = result;
  if (!isfinite(overall) || overall <= 0.0f || !isfinite(result) ||
      result <= 0.0f) {
    atomicOr(error, 1u << 2);
  }
}

__global__ void SelectionInitializeKernel(unsigned int* state,
                                          unsigned int* histogram,
                                          unsigned int median_index) {
  const unsigned int index = threadIdx.x;
  if (index == 0) {
    state[0] = 0;
    state[1] = 0;
    state[2] = median_index;
  }
  if (index < 256) histogram[index] = 0;
}

__global__ void SelectionHistogramKernel(const float* values,
                                         const float* statistics,
                                         unsigned int* histogram,
                                         const unsigned int* state,
                                         unsigned int count, unsigned int shift,
                                         bool deviation) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float value = values[index];
  if (deviation) value = fabsf(value - statistics[0]);
  const unsigned int bits = __float_as_uint(value);
  if ((bits & state[1]) != state[0]) return;
  atomicAdd(histogram + ((bits >> shift) & 255u), 1u);
}

__global__ void SelectionBucketKernel(unsigned int* histogram,
                                      unsigned int* state, float* statistics,
                                      unsigned int shift, bool deviation) {
  const unsigned int index = threadIdx.x;
  if (index == 0) {
    const unsigned int rank = state[2];
    unsigned int prefix = 0;
    for (unsigned int bucket = 0; bucket < 256; ++bucket) {
      const unsigned int count = histogram[bucket];
      if (rank < prefix + count) {
        state[0] |= bucket << shift;
        state[1] |= 255u << shift;
        state[2] = rank - prefix;
        if (shift == 0)
          statistics[deviation ? 1 : 0] = __uint_as_float(state[0]);
        break;
      }
      prefix += count;
    }
  }
  __syncthreads();
  histogram[index] = 0;
}

__global__ void FinalizeQuantizerKernel(const float* statistics,
                                        unsigned int* quantizer,
                                        unsigned int* error,
                                        unsigned int scaled_quant_dc,
                                        float quant_dc) {
  const float median = statistics[0];
  const float deviation = statistics[1];
  const float delta = __fsub_rn(median, deviation);
  const float numerator = __fmul_rn(65536.0f, delta);
  float scale = __fdiv_rn(numerator, 5.0f);
  scale = fminf(fmaxf(scale, 1.0f), 32768.0f);
  unsigned int global_scale = static_cast<unsigned int>(scale);
  if (global_scale > scaled_quant_dc) global_scale = max(1u, scaled_quant_dc);
  const float inverse = 65536.0f / static_cast<float>(global_scale);
  const float dc = fminf(65536.0f, quant_dc * inverse + 0.5f);
  quantizer[0] = global_scale;
  quantizer[1] = static_cast<unsigned int>(dc);
  if (!isfinite(median) || median <= 0.0f || !isfinite(deviation) ||
      deviation < 0.0f || global_scale == 0 || global_scale > 32768 ||
      quantizer[1] == 0 || quantizer[1] > 65536) {
    atomicOr(error, 1u << 3);
  }
}

__global__ void RawQuantKernel(const float* quant_field,
                               const unsigned int* quantizer, int* raw_quant,
                               unsigned int* error, unsigned int count) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float value =
      quant_field[index] * (65536.0f / static_cast<float>(quantizer[0])) + 0.5f;
  const int raw = static_cast<int>(fminf(fmaxf(value, 1.0f), 256.0f));
  raw_quant[index] = raw;
  if (!isfinite(value) || raw < 1 || raw > 256) atomicOr(error, 1u << 4);
}

__device__ signed char QuantizeCfl(float value) {
  constexpr float kTowardsZero = 2.6f;
  if (value >= kTowardsZero)
    value -= kTowardsZero;
  else if (value <= -kTowardsZero)
    value += kTowardsZero;
  else
    value = 0.0f;
  return static_cast<signed char>(fminf(fmaxf(roundf(value), -128.0f), 127.0f));
}

__global__ void InitialCflKernel(const float* coding_x, const float* coding_y,
                                 const float* coding_b, signed char* y_to_x,
                                 signed char* y_to_b, unsigned int* error,
                                 CudaAqGeometry geometry) {
  const unsigned int tile = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tile_count = geometry.tile_width * geometry.tile_height;
  if (tile >= tile_count) return;
  const unsigned int tile_x = tile % geometry.tile_width;
  const unsigned int tile_y = tile / geometry.tile_width;
  const unsigned int x_begin = tile_x * 64;
  const unsigned int y_begin = tile_y * 64;
  const unsigned int x_end = min(x_begin + 64, geometry.width);
  const unsigned int y_end = min(y_begin + 64, geometry.height);
  float sum_y[4] = {};
  float sum_x[4] = {};
  float sum_b[4] = {};
  unsigned int sample = 0;
  for (unsigned int y = y_begin; y < y_end; ++y) {
    const size_t row = static_cast<size_t>(y) * geometry.width;
    for (unsigned int x = x_begin; x < x_end; ++x, ++sample) {
      const unsigned int lane = sample & 3u;
      const float vy = coding_y[row + x];
      const float vx = coding_x[row + x];
      const float vb = coding_b[row + x];
      if (!isfinite(vy) || !isfinite(vx) || !isfinite(vb)) {
        atomicOr(error, 1u << 5);
        return;
      }
      sum_y[lane] += vy;
      sum_x[lane] += vx;
      sum_b[lane] += vb;
    }
  }
  if (sample == 0) {
    atomicOr(error, 1u << 5);
    return;
  }
  const float count = static_cast<float>(sample);
  const float mean_y = ((sum_y[0] + sum_y[1]) + (sum_y[2] + sum_y[3])) / count;
  const float mean_x = ((sum_x[0] + sum_x[1]) + (sum_x[2] + sum_x[3])) / count;
  const float mean_b = ((sum_b[0] + sum_b[1]) + (sum_b[2] + sum_b[3])) / count;
  float quadratic[4] = {};
  float linear_x[4] = {};
  float linear_b[4] = {};
  sample = 0;
  for (unsigned int y = y_begin; y < y_end; ++y) {
    const size_t row = static_cast<size_t>(y) * geometry.width;
    for (unsigned int x = x_begin; x < x_end; ++x, ++sample) {
      const unsigned int lane = sample & 3u;
      const float centered_y = coding_y[row + x] - mean_y;
      const float a = centered_y * (1.0f / 84.0f);
      quadratic[lane] = fmaf(a, a, quadratic[lane]);
      linear_x[lane] = fmaf(a, -(coding_x[row + x] - mean_x), linear_x[lane]);
      linear_b[lane] =
          fmaf(a, centered_y - (coding_b[row + x] - mean_b), linear_b[lane]);
    }
  }
  const float q = (quadratic[0] + quadratic[1]) + (quadratic[2] + quadratic[3]);
  const float lx = (linear_x[0] + linear_x[1]) + (linear_x[2] + linear_x[3]);
  const float lb = (linear_b[0] + linear_b[1]) + (linear_b[2] + linear_b[3]);
  const float denominator = q + count * 5.0e-10f;
  y_to_x[tile] = QuantizeCfl(-lx / denominator);
  y_to_b[tile] = QuantizeCfl(-lb / denominator);
}

__global__ void GatherDct8Kernel(const float* x, const float* y, const float* b,
                                 float* gathered, CudaAqGeometry geometry) {
  const size_t pixel_count =
      static_cast<size_t>(geometry.width) * geometry.height;
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 3 * pixel_count) return;
  const unsigned int channel = static_cast<unsigned int>(index / pixel_count);
  const size_t channel_index =
      index - static_cast<size_t>(channel) * pixel_count;
  const unsigned int block = static_cast<unsigned int>(channel_index / 64);
  const unsigned int coefficient =
      static_cast<unsigned int>(channel_index & 63u);
  const unsigned int bx = block % geometry.block_width;
  const unsigned int by = block / geometry.block_width;
  const unsigned int px = coefficient & 7u;
  const unsigned int py = coefficient >> 3;
  const float* source = channel == 0 ? x : (channel == 1 ? y : b);
  gathered[index] =
      source[static_cast<size_t>(by * 8 + py) * geometry.width + bx * 8 + px];
}

__device__ int QuantizeCoefficient(float coefficient, float inverse_dequant,
                                   unsigned int global_scale, int raw_quant,
                                   float matrix_multiplier, float threshold,
                                   unsigned int* error) {
  const float scale = (static_cast<float>(global_scale) * (1.0f / 65536.0f)) *
                      static_cast<float>(raw_quant) * matrix_multiplier;
  const float value = inverse_dequant * scale * coefficient;
  if (!isfinite(coefficient) || !isfinite(value)) {
    atomicOr(error, 1u << 6);
    return 0;
  }
  if (fabsf(value) < threshold) return 0;
  const float rounded = rintf(value);
  if (!isfinite(rounded) || rounded < -2147483648.0f ||
      rounded >= 2147483648.0f) {
    atomicOr(error, 1u << 7);
    return 0;
  }
  return static_cast<int>(rounded);
}

__device__ float AdjustBias(int quantized, unsigned int channel) {
  constexpr float kBias[3] = {
      1.0f - 0.05465007330715401f,
      1.0f - 0.07005449891748593f,
      1.0f - 0.049935103337343655f,
  };
  const float value = static_cast<float>(quantized);
  const float absolute = fabsf(value);
  if (absolute < 0.125f) return 0.0f;
  if (absolute < 1.125f) return copysignf(kBias[channel], value);
  return value - 0.145f / value;
}

__device__ float DequantizeCoefficient(int quantized, float dequant,
                                       unsigned int global_scale, int raw_quant,
                                       float matrix_multiplier,
                                       unsigned int channel,
                                       unsigned int* error) {
  const float scale = (65536.0f / static_cast<float>(global_scale)) /
                      (static_cast<float>(raw_quant) * matrix_multiplier);
  const float coefficient = AdjustBias(quantized, channel) * dequant * scale;
  if (!isfinite(coefficient)) {
    atomicOr(error, 1u << 8);
    return 0.0f;
  }
  return coefficient;
}

__device__ int AdjustQuantForChannel(const float* coefficients,
                                     const float* quant_tables,
                                     unsigned int channel_stride,
                                     unsigned int channel,
                                     unsigned int global_scale, int initial_raw,
                                     float multiplier, float thresholds[4],
                                     unsigned int* error) {
  float border_sum = 0.0f;
  float error_sum = 0.0f;
  float value_sum = 0.0f;
  float nonzeros[4] = {};
  float max_error[4] = {};
  const float qac = (static_cast<float>(global_scale) * (1.0f / 65536.0f)) *
                    static_cast<float>(initial_raw);
  for (unsigned int y = 0; y < 8; ++y) {
    for (unsigned int x = 0; x < 8; ++x) {
      if (x == 0 && y == 0) continue;
      const unsigned int index = y * 8 + x;
      const unsigned int quadrant = (y >= 4 ? 2u : 0u) + (x >= 4 ? 1u : 0u);
      const unsigned int table = channel * 64 + index;
      const float coefficient = coefficients[channel * channel_stride + index];
      const float value =
          coefficient * (quant_tables[192 + table] * qac * multiplier);
      if (!isfinite(coefficient) || !isfinite(value)) {
        atomicOr(error, 1u << 9);
        continue;
      }
      const float quantized =
          fabsf(value) < thresholds[quadrant] ? 0.0f : rintf(value);
      const float qerror = fabsf(value - quantized);
      error_sum += qerror;
      value_sum += fabsf(quantized);
      if (channel == 1 && quantized == 0.0f) {
        max_error[quadrant] = fmaxf(max_error[quadrant], qerror);
      }
      if (quantized != 0.0f) {
        nonzeros[quadrant] += fabsf(quantized);
        const bool in_corner = y >= 7 && x >= 7;
        const bool on_border = y == 7 || x == 7;
        const bool in_larger_corner = x >= 4 && y >= 4;
        if (in_corner || (on_border && in_larger_corner))
          border_sum += fabsf(value);
      }
    }
  }
  int raw = initial_raw;
  if (channel == 1 && value_sum * 8.0f < 1.0f) {
    constexpr float kLimit = 0.46f;
    int candidate = initial_raw;
    for (unsigned int q = 1; q < 4; ++q) {
      if (nonzeros[q] == 0.0f && max_error[q] > kLimit) {
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
  if (nonzeros[0] + nonzeros[1] + nonzeros[2] + nonzeros[3] < 11.0f) {
    if (++raw >= 256) raw = 255;
  }
  error_sum *= 2.2942708343284721f;
  value_sum *= 2.2942708343284721f;
  (void)error_sum;
  (void)value_sum;
  const float minimum =
      fminf(fminf(nonzeros[0], nonzeros[1]), fminf(nonzeros[2], nonzeros[3]));
  int activity = minimum < 15.0f ? static_cast<int>(minimum) : 15;
  int adjusted = raw - activity;
  if (channel == 1) {
    for (unsigned int q = 1; q < 4; ++q) thresholds[q] += 0.01f * activity;
  }
  const int limit = max(4, raw / 2);
  return adjusted < limit ? limit : adjusted;
}

__global__ void AdjustQuantKernel(const float* coefficients,
                                  const float* quant_tables, int* raw_quant,
                                  const unsigned int* quantizer,
                                  float* y_thresholds, unsigned int* error,
                                  unsigned int block_count, float x_multiplier,
                                  float b_multiplier) {
  const unsigned int block = blockIdx.x * blockDim.x + threadIdx.x;
  if (block >= block_count) return;
  const float* transform = coefficients + static_cast<size_t>(block) * 64;
  int selected = 0;
  const unsigned int channels[3] = {1, 0, 2};
  float selected_y[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  for (unsigned int ci = 0; ci < 3; ++ci) {
    const unsigned int channel = channels[ci];
    float thresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};
    const float multiplier =
        channel == 0 ? x_multiplier : (channel == 2 ? b_multiplier : 1.0f);
    const int candidate = AdjustQuantForChannel(
        transform, quant_tables, block_count * 64, channel, quantizer[0],
        raw_quant[block], multiplier, thresholds, error);
    if (channel == 1) {
      for (unsigned int q = 0; q < 4; ++q) selected_y[q] = thresholds[q];
    }
    selected = max(selected, candidate);
  }
  raw_quant[block] = selected;
  for (unsigned int q = 0; q < 4; ++q)
    y_thresholds[block * 4 + q] = selected_y[q];
}

__device__ int RoundDc(float value, unsigned int* error) {
  const float rounded = roundf(value);
  if (!isfinite(rounded) || rounded < -2147483648.0f ||
      rounded >= 2147483648.0f) {
    atomicOr(error, 1u << 10);
    return 0;
  }
  return static_cast<int>(rounded);
}

__global__ void EncodeFrameKernel(
    const float* coefficients, const float* quant_tables, const int* raw_quant,
    const unsigned int* quantizer, const signed char* y_to_x,
    const signed char* y_to_b, const float* y_thresholds, int* quantized_ac,
    int* quantized_dc, unsigned int* error, CudaAqGeometry geometry,
    float x_multiplier, float b_multiplier) {
  const unsigned int block = blockIdx.x;
  if (block >= geometry.block_width * geometry.block_height) return;
  __shared__ float reconstructed_y[64];
  const unsigned int thread = threadIdx.x;
  const unsigned int block_count = geometry.block_width * geometry.block_height;
  const unsigned int bx = block % geometry.block_width;
  const unsigned int by = block / geometry.block_width;
  const unsigned int color = (by / 8) * geometry.tile_width + bx / 8;
  const int raw = raw_quant[block];
  const unsigned int global_scale = quantizer[0];
  const unsigned int quant_dc = quantizer[1];
  const size_t base = static_cast<size_t>(block) * 64;
  const size_t channel_stride = static_cast<size_t>(block_count) * 64;

  if (thread == 0) {
    const float quant_scale =
        (static_cast<float>(global_scale) / 65536.0f) * quant_dc;
    const float dc_x = coefficients[base];
    const float dc_y = coefficients[channel_stride + base];
    const float dc_b = coefficients[2 * channel_stride + base];
    const int qy = RoundDc(dc_y * (512.0f * quant_scale), error);
    const float reconstructed_dc_y =
        static_cast<float>(qy) / (512.0f * quant_scale);
    quantized_dc[block] = RoundDc(dc_x * (4096.0f * quant_scale), error);
    quantized_dc[block_count + block] = qy;
    quantized_dc[2 * block_count + block] =
        RoundDc((dc_b - reconstructed_dc_y) * (256.0f * quant_scale), error);
  }

  const unsigned int coefficient = thread;
  const unsigned int cx = coefficient & 7u;
  const unsigned int cy = coefficient >> 3;
  const unsigned int quadrant = (cy >= 4 ? 2u : 0u) + (cx >= 4 ? 1u : 0u);
  const unsigned int y_table = 64 + coefficient;
  const size_t y_offset = channel_stride + base + coefficient;
  const int qy = QuantizeCoefficient(
      coefficients[y_offset], quant_tables[192 + y_table], global_scale, raw,
      1.0f, y_thresholds[block * 4 + quadrant], error);
  quantized_ac[y_offset] = qy;
  reconstructed_y[coefficient] = DequantizeCoefficient(
      qy, quant_tables[y_table], global_scale, raw, 1.0f, 1, error);
  __syncthreads();

  const float cfl_x = static_cast<float>(y_to_x[color]) * (1.0f / 84.0f);
  const float cfl_b = 1.0f + static_cast<float>(y_to_b[color]) * (1.0f / 84.0f);
  const float factors[2] = {cfl_x, cfl_b};
  const float multipliers[2] = {x_multiplier, b_multiplier};
  const unsigned int channels[2] = {0, 2};
  for (unsigned int i = 0; i < 2; ++i) {
    const unsigned int channel = channels[i];
    const size_t offset =
        static_cast<size_t>(channel) * channel_stride + base + coefficient;
    const float predicted =
        coefficients[offset] - factors[i] * reconstructed_y[coefficient];
    const float threshold = quadrant == 0 ? 0.58f : 0.62f;
    quantized_ac[offset] = QuantizeCoefficient(
        predicted, quant_tables[192 + channel * 64 + coefficient], global_scale,
        raw, multipliers[i], threshold, error);
  }
}

cudaError_t CheckLaunch() { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t LaunchCudaAqInitialQuantization(
    const float* coding_x, const float* coding_y, const float* coding_b,
    float* unblurred_pixel_mask, float* pixel_mask, float* pre_erosion,
    float* quant_field, float* strategy_mask, unsigned int* selection_state,
    unsigned int* histogram, float* statistics, unsigned int* quantizer_params,
    int* raw_quant, unsigned int* error, CudaAqGeometry geometry,
    float butteraugli_target, float rescale, float quant_dc,
    cudaStream_t stream) {
  cudaError_t status = cudaMemsetAsync(error, 0, sizeof(*error), stream);
  if (status != cudaSuccess) return status;
  const dim3 threads2d(16, 16);
  const dim3 gradient_grid((geometry.width / 4 + 15) / 16,
                           (geometry.height / 4 + 15) / 16);
  InitialGradientKernel<<<gradient_grid, threads2d, 0, stream>>>(
      coding_y, unblurred_pixel_mask, pre_erosion, error, geometry);
  if ((status = CheckLaunch()) != cudaSuccess) return status;

  constexpr float kBase[4] = {0.125f, 0.1f, 0.09f, 0.06f};
  constexpr float kAdd[4] = {0.0f, -0.1f, -0.09f, -0.06f};
  constexpr float kTotal = 0.29959705784054957f;
  const float mix =
      butteraugli_target < 2.0f ? (2.0f - butteraugli_target) * 0.5f : 0.0f;
  float weights[4];
  float sum = 0.0f;
  for (unsigned int i = 0; i < 4; ++i) {
    weights[i] = kBase[i] + mix * kAdd[i];
    sum += weights[i];
  }
  for (float& weight : weights) weight *= kTotal / sum;
  const dim3 block_grid((geometry.block_width + 15) / 16,
                        (geometry.block_height + 15) / 16);
  FuzzyErosionKernel<<<block_grid, threads2d, 0, stream>>>(
      pre_erosion, quant_field, strategy_mask, error, geometry, weights[0],
      weights[1], weights[2], weights[3]);
  if ((status = CheckLaunch()) != cudaSuccess) return status;

  const float scale = 0.765f / butteraugli_target * rescale;
  const float base_level = 0.48f * scale;
  float dampen = 1.0f;
  if (butteraugli_target >= 2.0f) {
    dampen = fmaxf(1.0f - (butteraugli_target - 2.0f) / 12.0f, 0.0f);
  }
  ModulationKernel<<<block_grid, threads2d, 0, stream>>>(
      coding_x, coding_y, coding_b, quant_field, error, geometry,
      scale * dampen, (1.0f - dampen) * base_level);
  if ((status = CheckLaunch()) != cudaSuccess) return status;

  constexpr float kFilter[5] = {0.364911248f, 0.05f, 0.1688888021f,
                                0.221069183f, 0.306563504f};
  constexpr double kWeightSum =
      1.0 + 4.0 * (kFilter[0] + kFilter[1] + kFilter[2] + kFilter[4] +
                   2.0 * kFilter[3]);
  constexpr float kNormalize = static_cast<float>(1.0 / kWeightSum);
  status = LaunchCudaSymmetric5Convolution(
      unblurred_pixel_mask, pixel_mask, geometry.width, geometry.height,
      geometry.width, geometry.width, kNormalize, kNormalize * kFilter[0],
      kNormalize * kFilter[2], kNormalize * kFilter[1], kNormalize * kFilter[4],
      kNormalize * kFilter[3], stream);
  if (status != cudaSuccess) return status;

  const unsigned int block_count = geometry.block_width * geometry.block_height;
  const unsigned int grid = (block_count + kThreads - 1) / kThreads;
  const unsigned int shifts[4] = {24, 16, 8, 0};
  for (unsigned int deviation = 0; deviation < 2; ++deviation) {
    SelectionInitializeKernel<<<1, 256, 0, stream>>>(selection_state, histogram,
                                                     block_count / 2);
    if ((status = CheckLaunch()) != cudaSuccess) return status;
    for (unsigned int shift : shifts) {
      SelectionHistogramKernel<<<grid, kThreads, 0, stream>>>(
          quant_field, statistics, histogram, selection_state, block_count,
          shift, deviation != 0);
      if ((status = CheckLaunch()) != cudaSuccess) return status;
      SelectionBucketKernel<<<1, 256, 0, stream>>>(
          histogram, selection_state, statistics, shift, deviation != 0);
      if ((status = CheckLaunch()) != cudaSuccess) return status;
    }
  }
  const unsigned int scaled_quant_dc = static_cast<unsigned int>(
      static_cast<int>(static_cast<double>(quant_dc * 4096.0f) * 1.6));
  FinalizeQuantizerKernel<<<1, 1, 0, stream>>>(
      statistics, quantizer_params, error, scaled_quant_dc, quant_dc);
  if ((status = CheckLaunch()) != cudaSuccess) return status;
  RawQuantKernel<<<grid, kThreads, 0, stream>>>(quant_field, quantizer_params,
                                                raw_quant, error, block_count);
  return CheckLaunch();
}

cudaError_t LaunchCudaAqSelectResidentQuantizer(
    const float* quant_field, unsigned int block_count,
    unsigned int* selection_state, unsigned int* histogram, float* statistics,
    unsigned int* quantizer_params, int* raw_quant, unsigned int* error,
    float quant_dc, cudaStream_t stream) {
  constexpr unsigned int kShifts[4] = {24, 16, 8, 0};
  constexpr unsigned int kThreads = 256;
  const unsigned int grid = (block_count + kThreads - 1) / kThreads;
  cudaError_t status = cudaSuccess;
  for (unsigned int deviation = 0; deviation < 2; ++deviation) {
    SelectionInitializeKernel<<<1, kThreads, 0, stream>>>(
        selection_state, histogram, block_count / 2);
    if ((status = CheckLaunch()) != cudaSuccess) return status;
    for (unsigned int shift : kShifts) {
      SelectionHistogramKernel<<<grid, kThreads, 0, stream>>>(
          quant_field, statistics, histogram, selection_state, block_count,
          shift, deviation != 0);
      if ((status = CheckLaunch()) != cudaSuccess) return status;
      SelectionBucketKernel<<<1, kThreads, 0, stream>>>(
          histogram, selection_state, statistics, shift, deviation != 0);
      if ((status = CheckLaunch()) != cudaSuccess) return status;
    }
  }
  const unsigned int scaled_quant_dc = static_cast<unsigned int>(
      static_cast<int>(static_cast<double>(quant_dc * 4096.0f) * 1.6));
  FinalizeQuantizerKernel<<<1, 1, 0, stream>>>(
      statistics, quantizer_params, error, scaled_quant_dc, quant_dc);
  if ((status = CheckLaunch()) != cudaSuccess) return status;
  RawQuantKernel<<<grid, kThreads, 0, stream>>>(quant_field, quantizer_params,
                                                raw_quant, error, block_count);
  return CheckLaunch();
}

cudaError_t LaunchCudaAqEncodeFrame(
    const float* coding_x, const float* coding_y, const float* coding_b,
    const float* transform_x, const float* transform_y,
    const float* transform_b, float* gathered, float* forward_coefficients,
    const float* quant_tables, int* raw_quant,
    const unsigned int* quantizer_params, signed char* y_to_x,
    signed char* y_to_b, float* y_thresholds, int* quantized_ac,
    int* quantized_dc, unsigned int* error, CudaAqGeometry geometry,
    float x_matrix_multiplier, float b_matrix_multiplier, cudaStream_t stream) {
  cudaError_t status = cudaMemsetAsync(error, 0, sizeof(*error), stream);
  if (status != cudaSuccess) return status;
  const unsigned int tile_count = geometry.tile_width * geometry.tile_height;
  InitialCflKernel<<<(tile_count + kThreads - 1) / kThreads, kThreads, 0,
                     stream>>>(coding_x, coding_y, coding_b, y_to_x, y_to_b,
                               error, geometry);
  if ((status = CheckLaunch()) != cudaSuccess) return status;
  const size_t pixel_count =
      static_cast<size_t>(geometry.width) * geometry.height;
  const size_t gathered_count = 3 * pixel_count;
  GatherDct8Kernel<<<static_cast<unsigned int>((gathered_count + kThreads - 1) /
                                               kThreads),
                     kThreads, 0, stream>>>(transform_x, transform_y,
                                            transform_b, gathered, geometry);
  if ((status = CheckLaunch()) != cudaSuccess) return status;
  const unsigned int block_count = geometry.block_width * geometry.block_height;
  status = LaunchCudaDct(true, gathered, forward_coefficients, 3 * block_count,
                         8, 8, stream);
  if (status != cudaSuccess) return status;
  AdjustQuantKernel<<<(block_count + kThreads - 1) / kThreads, kThreads, 0,
                      stream>>>(forward_coefficients, quant_tables, raw_quant,
                                quantizer_params, y_thresholds, error,
                                block_count, x_matrix_multiplier,
                                b_matrix_multiplier);
  if ((status = CheckLaunch()) != cudaSuccess) return status;
  EncodeFrameKernel<<<block_count, 64, 0, stream>>>(
      forward_coefficients, quant_tables, raw_quant, quantizer_params, y_to_x,
      y_to_b, y_thresholds, quantized_ac, quantized_dc, error, geometry,
      x_matrix_multiplier, b_matrix_multiplier);
  return CheckLaunch();
}

}  // namespace gjxl::cuda_internal
