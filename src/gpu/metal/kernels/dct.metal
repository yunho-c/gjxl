// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "metal_stdlib"

using namespace metal;

constant float kDCT8_2D[64] = {
  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,
  0.49039264f,  0.41573481f,  0.27778512f,  0.09754516f, -0.09754516f, -0.27778512f, -0.41573481f, -0.49039264f,
  0.46193977f,  0.19134172f, -0.19134172f, -0.46193977f, -0.46193977f, -0.19134172f,  0.19134172f,  0.46193977f,
  0.41573481f, -0.09754516f, -0.49039264f, -0.27778512f,  0.27778512f,  0.49039264f,  0.09754516f, -0.41573481f,
  0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f,  0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f,
  0.27778512f, -0.49039264f,  0.09754516f,  0.41573481f, -0.41573481f, -0.09754516f,  0.49039264f, -0.27778512f,
  0.19134172f, -0.46193977f,  0.46193977f, -0.19134172f, -0.19134172f,  0.46193977f, -0.46193977f,  0.19134172f,
  0.09754516f, -0.27778512f,  0.41573481f, -0.49039264f,  0.49039264f, -0.41573481f,  0.27778512f, -0.09754516f
};

// Computes forward DCT using hardcoded 8x8 DCT-II transform matrix with two matmuls:
//   B = C A C^T, where C is the 2D 8x8 DCT transform matrix.
// This exists to perform as a simple reference (and performance baseline) against optimized variants.
kernel void gjxl_forward_dct8__scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint  tid             [[thread_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  // row-major
  int row = tid / 8; // of output matrix
  int col = tid % 8; // of output matrix

  const ulong base = static_cast<ulong>(group_position.x) * 64ul;

  threadgroup float T[8][8]; // intermediate product (C * A)

  // first matmul: C * A
  float t = 0;

  for (int i=0; i < 8; ++i) { // "inner loop" over N (i.e., width of A and/or height of T)
    t += kDCT8_2D[8*row + i] * A[base + 8*i + col];
  }

  T[row][col] = t;

  // synchronize threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // second matmul: T * C^T
  float b = 0;

  for (int i=0; i < 8; ++i) {
    b += T[row][i] * kDCT8_2D[8*col + i];
  }

  // commit results
  B[base + 8*row + col] = b;
}


// Computes inverse DCT using hardcoded 8x8 DCT-II transform matrix with two matmuls:
//   B = C^T A C.
// This exists to perform as a simple reference (and performance baseline) against optimized variants.
kernel void gjxl_inverse_dct8__scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint  tid             [[thread_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  int row = tid / 8;
  int col = tid % 8;

  const ulong base = static_cast<ulong>(group_position.x) * 64ul;

  threadgroup float T[8][8]; // intermediate product (C^T * A)

  float t = 0;

  for (int i=0; i < 8; ++i) {
    t += kDCT8_2D[8*i + row] * A[base + 8*i + col];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i < 8; ++i) {
    b += T[row][i] * kDCT8_2D[8*i + col];
  }

  B[base + 8*row + col] = b;
}
