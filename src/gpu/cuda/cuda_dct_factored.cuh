// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

namespace gjxl::cuda_internal {

// Same radix-2 DCT-II/III factorization as the Metal implementation. Forward
// produces unscaled coefficients; inverse consumes coefficients scaled by 1/N.
// Compile-time loop bounds keep values and recursive scratch in registers.
template <unsigned N>
__device__ __forceinline__ float FactoredDctMultiplier(unsigned i) {
  if constexpr (N == 4) {
    constexpr float m[] = {0.541196100146197f, 1.3065629648763764f};
    return m[i];
  } else if constexpr (N == 8) {
    constexpr float m[] = {0.5097955791041592f, 0.6013448869350453f,
                           0.8999762231364156f, 2.5629154477415055f};
    return m[i];
  } else if constexpr (N == 16) {
    constexpr float m[] = {0.5024192861881557f, 0.5224986149396889f,
                           0.5669440348163577f, 0.6468217833599901f,
                           0.7881546234512502f, 1.060677685990347f,
                           1.7224470982383342f, 5.101148618689155f};
    return m[i];
  } else {
    static_assert(N == 32);
    constexpr float m[] = {
        0.5006029982351963f, 0.5054709598975436f, 0.5154473099226246f,
        0.5310425910897841f, 0.5531038960344445f, 0.5829349682061339f,
        0.6225041230356648f, 0.6748083414550057f, 0.7445362710022986f,
        0.8393496454155268f, 0.9725682378619608f, 1.1694399334328847f,
        1.4841646163141662f, 2.057781009953411f,  3.407608418468719f,
        10.190008123548033f};
    return m[i];
  }
}

template <unsigned N, bool Forward>
__device__ __forceinline__ void FactoredDct1D(float* values, float* scratch) {
  if constexpr (N == 2) {
    const float a = values[0], b = values[1];
    values[0] = a + b;
    values[1] = a - b;
  } else {
    constexpr unsigned half = N / 2;
    if constexpr (Forward) {
#pragma unroll
      for (unsigned i = 0; i < half; ++i)
        scratch[i] = values[i] + values[N - 1 - i];
      FactoredDct1D<half, true>(scratch, scratch + N);
#pragma unroll
      for (unsigned i = 0; i < half; ++i)
        scratch[half + i] =
            (values[i] - values[N - 1 - i]) * FactoredDctMultiplier<N>(i);
      FactoredDct1D<half, true>(scratch + half, scratch + N);
      scratch[half] = scratch[half] * 1.41421356237f + scratch[half + 1];
#pragma unroll
      for (unsigned i = 1; i + 1 < half; ++i)
        scratch[half + i] += scratch[half + i + 1];
#pragma unroll
      for (unsigned i = 0; i < half; ++i) {
        values[2 * i] = scratch[i];
        values[2 * i + 1] = scratch[half + i];
      }
    } else {
#pragma unroll
      for (unsigned i = 0; i < half; ++i) {
        scratch[i] = values[2 * i];
        scratch[half + i] = values[2 * i + 1];
      }
      FactoredDct1D<half, false>(scratch, scratch + N);
#pragma unroll
      for (unsigned i = half - 1; i > 0; --i)
        scratch[half + i] += scratch[half + i - 1];
      scratch[half] *= 1.41421356237f;
      FactoredDct1D<half, false>(scratch + half, scratch + N);
#pragma unroll
      for (unsigned i = 0; i < half; ++i) {
        const float even = scratch[i],
                    odd = scratch[half + i] * FactoredDctMultiplier<N>(i);
        values[i] = even + odd;
        values[N - 1 - i] = even - odd;
      }
    }
  }
}

}  // namespace gjxl::cuda_internal
