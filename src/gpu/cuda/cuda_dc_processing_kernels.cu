// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// CUDA implementation of the shared DC quantization/prediction/smoothing
// policy.
#include "gpu/cuda/cuda_kernel_profile.h"
#include "gpu/cuda/cuda_dc_processing_kernels.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
namespace gjxl::cuda_internal {
namespace {
__device__ int64_t DcClamp(int64_t v, int64_t lo, int64_t hi) {
  return min(max(v, lo), hi);
}
__device__ int RoundDc(float value, unsigned int *error) {
  const float rounded = roundf(value);
  if (!isfinite(rounded) || rounded < -2147483648.0f ||
      rounded >= 2147483648.0f) {
    atomicOr(error, 16u);
    return 0;
  }
  return static_cast<int>(rounded);
}
__device__ __constant__ unsigned int kDcReciprocal[64] = {
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152,
    1864135,  1677721, 1525201, 1398101, 1290555, 1198372, 1118481, 1048576,
    986895,   932067,  883011,  838860,  798915,  762600,  729444,  699050,
    671088,   645277,  621378,  599186,  578524,  559240,  541200,  524288,
    508400,   493447,  479349,  466033,  453438,  441505,  430185,  419430,
    409200,   399457,  390167,  381300,  372827,  364722,  356962,  349525,
    342392,   335544,  328965,  322638,  316551,  310689,  305040,  299593,
    294337,   289262,  284359,  279620,  275036,  270600,  266305,  262144,
};

// Four waves retain every predecessor of the x+2*y traversal. Each lane owns
// one row; north-west is the oldest dependency, three waves behind. Keeping
// these values in shared memory avoids a device-wide barrier at every wave.
constexpr unsigned int kDcWaveRows = 256u;
constexpr unsigned int kDcWaveValues = 4u * kDcWaveRows;

__device__ unsigned int DcWaveIndex(unsigned int wave, unsigned int row) {
  return (wave & 3u) * kDcWaveRows + row;
}

// Store each sample's original four predictor errors and the signed error.
// The serial predictor mutates its previous row when advancing west-to-east.
// Recover that mutation from the current row's west/two-west samples instead,
// making every dependency explicit for diagonal execution.
struct AqDcWavefrontState {
  unsigned int *errors;
  unsigned int width;
  int64_t predictions[4];
  int64_t prediction;

  __device__ int64_t Predict(unsigned int x, unsigned int y, int64_t n,
                             int64_t w, int64_t ne) {
    const unsigned int wave = x + 2u * y;
    unsigned int weights[4];
    unsigned int weight_sum = 0;
    for (unsigned int i = 0; i < 4; ++i) {
      const unsigned int *plane = errors + i * kDcWaveValues;
      const unsigned int north =
          (y ? plane[DcWaveIndex(wave - 2u, y - 1u)] : 0u) +
          (x ? plane[DcWaveIndex(wave - 1u, y)] : 0u);
      const unsigned int northeast =
          x + 1u < width ? (y ? plane[DcWaveIndex(wave - 1u, y - 1u)] : 0u)
                         : north;
      const unsigned int northwest =
          x ? (y ? plane[DcWaveIndex(wave - 3u, y - 1u)] : 0u) +
                  (x > 1u ? plane[DcWaveIndex(wave - 2u, y)] : 0u)
            : north;
      const unsigned int error = north + northeast + northwest;
      const unsigned int bits = 64u - __clzll(uint64_t(error) + 1u);
      const unsigned int shift = bits > 6u ? bits - 6u : 0u;
      weights[i] =
          4u +
          (((i == 0u ? 13u : 12u) * kDcReciprocal[error >> shift]) >> shift);
      weight_sum += weights[i];
    }
    const unsigned int log_weight = 31u - __clz(weight_sum);
    weight_sum = 0u;
    for (unsigned int i = 0; i < 4; ++i) {
      weights[i] >>= log_weight - 4u;
      weight_sum += weights[i];
    }
    const int *plane =
        reinterpret_cast<const int *>(errors + 4u * kDcWaveValues);
    const int64_t ew = x ? plane[DcWaveIndex(wave - 1u, y)] : 0;
    const int64_t en = y ? plane[DcWaveIndex(wave - 2u, y - 1u)] : 0;
    const int64_t enw = x && y ? plane[DcWaveIndex(wave - 3u, y - 1u)] : en;
    const int64_t ene =
        y && x + 1u < width ? plane[DcWaveIndex(wave - 1u, y - 1u)] : en;
    n *= 8;
    w *= 8;
    ne *= 8;
    predictions[0] = w + ne - n;
    predictions[1] = n - (((en + ew + ene) * 16) >> 5);
    predictions[2] = w - (((en + ew + enw) * 10) >> 5);
    predictions[3] = n - (((enw + en + ene) * 7) >> 5);
    int64_t sum = int64_t(weight_sum >> 1u) - 1;
    for (unsigned int i = 0; i < 4; ++i)
      sum += predictions[i] * int64_t(weights[i]);
    prediction = (sum * int64_t(kDcReciprocal[weight_sum - 1u])) >> 24;
    if (((en ^ ew) | (en ^ enw)) <= 0)
      prediction = DcClamp(prediction, min(w, min(ne, n)), max(w, max(ne, n)));
    return (prediction + 3) >> 3;
  }

  __device__ bool Update(int value, unsigned int x, unsigned int y) {
    const unsigned int index = DcWaveIndex(x + 2u * y, y);
    const int64_t scaled = int64_t(value) * 8;
    const int64_t error = prediction - scaled;
    bool valid = error >= -2147483648ll && error <= 2147483647ll;
    int *plane = reinterpret_cast<int *>(errors + 4u * kDcWaveValues);
    plane[index] = valid ? int(error) : 0;
    for (unsigned int i = 0; i < 4; ++i) {
      const uint64_t magnitude = (llabs(predictions[i] - scaled) + 3u) >> 3u;
      valid = valid && magnitude <= 4294967295ull;
      errors[i * kDcWaveValues + index] = static_cast<unsigned int>(magnitude);
    }
    return valid;
  }
};

// One owns one channel of a DC group. X and Y run independently
// in the first dispatch; B runs after Y is complete in a second dispatch.
// Weighted dependencies precede x+2*y; gradient dependencies precede x+y.
// Every lane reaches the wave barrier, including partial-group inactive rows.
__global__ void QuantizeDcKernel(float *dc, int *quantized, unsigned int *error,
                                 CudaDcProcessingParams params,
                                 const unsigned int *resident_quantizer,
                                 unsigned int channel_base) {
  const dim3 position = blockIdx;
  const unsigned int row_index = threadIdx.x;
  const unsigned int group = position.x;
  const unsigned int c = channel_base + position.y;
  const unsigned int groups_x = (params.width - 1u) / 256u + 1u;
  const unsigned int groups_y = (params.height - 1u) / 256u + 1u;
  if (group >= groups_x * groups_y)
    return;
  if (c >= 3u) {
    atomicOr(error, 16u);
    return;
  }
  const unsigned int gx = (group % groups_x) * 256u;
  const unsigned int gy = (group / groups_x) * 256u;
  const unsigned int width = min(256u, params.width - gx);
  const unsigned int height = min(256u, params.height - gy);
  const unsigned int area = params.width * params.height;
  const unsigned int global_scale = params.use_resident_quantizer
                                        ? resident_quantizer[0]
                                        : params.global_scale;
  const unsigned int quant_dc =
      params.use_resident_quantizer ? resident_quantizer[1] : params.quant_dc;
  const float scale = float(global_scale) / 65536.0f;
  const float inverse_dc = (65536.0f / float(global_scale)) / float(quant_dc);
  const float precision = float(1u << params.extra_precision);
  const float factors[3] = {4096.0f, 512.0f, 256.0f};
  float steps[3], inverse[3];
  for (unsigned int c = 0; c < 3; ++c) {
    steps[c] = (inverse_dc / factors[c]) / precision;
    inverse[c] = (factors[c] * scale * float(quant_dc)) * precision;
  }
  const bool weighted =
      params.quantization_mode == 1u && params.predictor == 1u;
  __shared__ unsigned int wave_errors[5u * kDcWaveValues];
  __shared__ int wave_quantized[kDcWaveValues];
  AqDcWavefrontState state;
  state.width = width;
  state.errors = wave_errors;
  const unsigned int row_step = weighted ? 2u : 1u;
  const unsigned int waves = width + row_step * (height - 1u);
  {
    int *plane = quantized + c * area;
    for (unsigned int wave = 0; wave < waves; ++wave) {
      const int column = int(wave) - int(row_step * row_index);
      if (row_index < height && column >= 0 &&
          static_cast<unsigned int>(column) < width) {
        const unsigned int x = static_cast<unsigned int>(column), y = row_index;
        const unsigned int index = (gy + y) * params.width + gx + x;
        float value = dc[c * area + index];
        if (c == 2u) {
          value =
              params.quantization_mode == 1u
                  ? fmaf(-float(quantized[area + index]), steps[1], value)
                  : __fsub_rn(value, __fmul_rn(float(quantized[area + index]),
                                               steps[1]));
        }
        int64_t guess = 0;
        if (params.quantization_mode == 1u) {
          const int64_t w =
              x   ? wave_quantized[DcWaveIndex(wave - 1u, y)]
              : y ? wave_quantized[DcWaveIndex(wave - row_step, y - 1u)]
                  : 0;
          const int64_t n =
              y ? wave_quantized[DcWaveIndex(wave - row_step, y - 1u)] : w;
          if (weighted) {
            const int64_t ne =
                y && x + 1u < width
                    ? wave_quantized[DcWaveIndex(wave - 1u, y - 1u)]
                    : n;
            guess = state.Predict(x, y, n, w, ne);
          } else {
            const int64_t nw =
                x && y ? wave_quantized[DcWaveIndex(wave - 2u, y - 1u)] : w;
            guess = DcClamp(n + w - nw, min(n, w), max(n, w));
          }
        }
        float residual = __fmul_rn(value, inverse[c]);
        residual = __fsub_rn(residual, float(guess));
        if (params.quantization_mode == 1u && residual > -0.62f &&
            residual < 0.62f)
          residual = 0.0f;
        float rounded = roundf(residual);
        if (params.quantization_mode == 1u &&
            (rounded > 2.0f || rounded < -2.0f))
          rounded = roundf(residual * 0.5f) * 2.0f;
        const int64_t integer = guess + int64_t(RoundDc(rounded, error));
        const bool in_range =
            integer >= -2147483648ll && integer <= 2147483647ll;
        const int q = in_range ? int(integer) : 0;
        const bool predictor_valid = !weighted || state.Update(q, x, y);
        if (!in_range || !predictor_valid)
          atomicOr(error, 16u);
        plane[index] = q;
        wave_quantized[DcWaveIndex(wave, y)] = q;
        float reconstructed = __fmul_rn(float(q), steps[c]);
        if (c == 2u)
          reconstructed =
              __fadd_rn(__fmul_rn(float(quantized[area + index]), steps[1]),
                        reconstructed);
        dc[c * area + index] = reconstructed;
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

__global__ void SmoothDcKernel(const float *dc, float *smoothed,
                               unsigned int *error,
                               CudaDcProcessingParams params,
                               const unsigned int *resident_quantizer) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int area = params.width * params.height;
  if (index >= area)
    return;
  const unsigned int x = index % params.width;
  const unsigned int y = index / params.width;
  if (x == 0u || y == 0u || x + 1u == params.width || y + 1u == params.height) {
    for (unsigned int c = 0; c < 3; ++c) {
      const float value = dc[c * area + index];
      if (!isfinite(value))
        atomicOr(error, 8u);
      smoothed[c * area + index] = value;
    }
    return;
  }
  const unsigned int global_scale = params.use_resident_quantizer
                                        ? resident_quantizer[0]
                                        : params.global_scale;
  const unsigned int quant_dc =
      params.use_resident_quantizer ? resident_quantizer[1] : params.quant_dc;
  const float inverse_dc = (65536.0f / float(global_scale)) / float(quant_dc);
  const float factors[3] = {4096.0f, 512.0f, 256.0f};
  constexpr float side_weight = 0.20345139757231578f;
  constexpr float corner_weight = 0.0334829185968739f;
  constexpr float center_weight = 1.0f - 4.0f * (side_weight + corner_weight);
  float center[3], filtered[3];
  float gap = 0.5f;
  for (unsigned int c = 0; c < 3; ++c) {
    const float *row = dc + c * area + index;
    const float *top = row - params.width;
    const float *bottom = row + params.width;
    const float corners = (top[-1] + top[1]) + (bottom[-1] + bottom[1]);
    const float sides = (row[-1] + row[1]) + (top[0] + bottom[0]);
    center[c] = row[0];
    filtered[c] = fmaf(corners, corner_weight,
                       fmaf(sides, side_weight, row[0] * center_weight));
    const float normalized_gap =
        fabsf((row[0] - filtered[c]) / (inverse_dc / factors[c]));
    if (!isfinite(normalized_gap))
      atomicOr(error, 8u);
    gap = max(gap, normalized_gap);
  }
  const float factor = max(0.0f, fmaf(-4.0f, gap, 3.0f));
  for (unsigned int c = 0; c < 3; ++c) {
    const float value = fmaf(filtered[c] - center[c], factor, center[c]);
    if (!isfinite(value))
      atomicOr(error, 8u);
    smoothed[c * area + index] = value;
  }
}

} // namespace
cudaError_t LaunchCudaDcQuantization(float *dc, int *quantized,
                                     unsigned int *error,
                                     CudaDcProcessingParams params,
                                     const unsigned int *quantizer,
                                     cudaStream_t stream) {
  if (!dc || !quantized || !error || !params.width || !params.height ||
      uint64_t(params.width) * params.height > UINT32_MAX ||
      params.quantization_mode > 1 || params.predictor > 1 ||
      params.extra_precision > 3 ||
      (params.use_resident_quantizer
           ? !quantizer
           : (!params.global_scale || !params.quant_dc)))
    return cudaErrorInvalidValue;
  const unsigned int groups =
      ((params.width - 1u) / 256u + 1u) * ((params.height - 1u) / 256u + 1u);
  const unsigned int threads = params.height < 256u ? params.height : 256u;
  if (::gjxl::cuda_internal::CudaKernelProfileScope profile_scope{"QuantizeDcKernel", dim3(groups, 2), threads, stream}; profile_scope) {
    QuantizeDcKernel<<<dim3(groups, 2), threads, 0, stream>>>(
        dc, quantized, error, params, quantizer, 0);
  }
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  if (::gjxl::cuda_internal::CudaKernelProfileScope profile_scope{"QuantizeDcKernel", groups, threads, stream}; profile_scope) {
    QuantizeDcKernel<<<groups, threads, 0, stream>>>(dc, quantized, error, params,
                                                     quantizer, 2);
  }
  return cudaGetLastError();
}
cudaError_t LaunchCudaDcSmoothing(const float *dc, float *smoothed,
                                  unsigned int *error,
                                  CudaDcProcessingParams params,
                                  const unsigned int *quantizer,
                                  cudaStream_t stream) {
  if (!dc || !smoothed || dc == smoothed || !error || !params.width ||
      !params.height || uint64_t(params.width) * params.height > UINT32_MAX ||
      (params.use_resident_quantizer
           ? !quantizer
           : (!params.global_scale || !params.quant_dc)))
    return cudaErrorInvalidValue;
  const uint64_t area = uint64_t(params.width) * params.height;
  if (::gjxl::cuda_internal::CudaKernelProfileScope profile_scope{"SmoothDcKernel", static_cast<unsigned int>((area + 255) / 256), 256, stream}; profile_scope) {
    SmoothDcKernel<<<static_cast<unsigned int>((area + 255) / 256), 256, 0,
                     stream>>>(dc, smoothed, error, params, quantizer);
  }
  return cudaGetLastError();
}
} // namespace gjxl::cuda_internal
