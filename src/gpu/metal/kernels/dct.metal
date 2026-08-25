// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

#include "dct_basis.h"

using namespace metal;

constant float kForwardDct8Scale = 1.0f / 8.0f;
constant float kInverseDct8Scale = 8.0f;

constant float kForwardDct16Scale = 1.0f / 16.0f;
constant float kInverseDct16Scale = 16.0f;

// 1 / sqrt(16 * 8). The transpose pair has the same normalization.
constant float kForwardDct16x8Scale = 0.0883883461f;
constant float kInverseDct16x8Scale = 11.3137083f;

constant float kForwardDct32Scale = 1.0f / 32.0f;
constant float kInverseDct32Scale = 32.0f;

// 8x8

// Computes forward DCT using hardcoded 8x8 DCT-II transform matrix with two matmuls:
//   B = C A C^T, where C is the 2D 8x8 DCT transform matrix.
// The orthonormal result is divided by 8 and transposed to match libjxl's
// scaled coefficient layout: B[u][v], with horizontal frequency first.
// This exists to perform as a simple reference (and performance baseline) against optimized variants.
kernel void gjxl_dct8_forward_scalar_2d_matmul(
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
    t += kOrthonormalDct8[8*row + i] * A[base + 8*i + col];
  }

  T[row][col] = t;

  // synchronize threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // second matmul: T * C^T
  float b = 0;

  for (int i=0; i < 8; ++i) {
    b += T[row][i] * kOrthonormalDct8[8*col + i];
  }

  // commit results (scaled and transposed to match libjxl)
  B[base + 8*col + row] = b * kForwardDct8Scale;
}


// Computes inverse DCT using hardcoded 8x8 DCT-II transform matrix with two matmuls:
//   B = C^T A^T C
//     where A uses libjxl's transposed coefficient layout.
// Input coefficients use libjxl's transposed [u][v] layout. The orthonormal
// inverse result is multiplied by 8 to invert the forward scaling.
// This exists to perform as a simple reference (and performance baseline) against optimized variants.
kernel void gjxl_dct8_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint  tid             [[thread_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  int row = tid / 8;
  int col = tid % 8;

  const ulong base = static_cast<ulong>(group_position.x) * 64ul;

  threadgroup float T[8][8]; // intermediate product (C^T * A^T)

  float t = 0;

  for (int i=0; i < 8; ++i) {
    t += kOrthonormalDct8[8*i + row] * A[base + 8*col + i];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i < 8; ++i) {
    b += T[row][i] * kOrthonormalDct8[8*i + col];
  }

  B[base + 8*row + col] = b * kInverseDct8Scale;
}


// Computes forward DCT using hardcoded 8x8 DCT-II transform matrix with two matmuls.
//
kernel void gjxl_dct8_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  const ulong base = static_cast<ulong>(group_position.x) * 64ul;

  // lift hard-coded DCT matrix onto threadgroup memory for simdgroup_load()
  threadgroup float c_shared[64];

  for (uint i = lane; i < 64; i += simd_width) {
    c_shared[i] = kOrthonormalDct8[i];
  }

  // synchronize threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 a;
  simdgroup_float8x8 b;
  simdgroup_float8x8 c;  // 8x8 DCT-II transform
  simdgroup_float8x8 ct; // 8x8 DCT-II transform, transposed

  simdgroup_load(a, A + base);
  simdgroup_load(c,  c_shared);
  simdgroup_load(ct, c_shared, 8, ulong2(0), true);

  simdgroup_multiply(b, c, a);
  simdgroup_multiply(b, b, ct);

  b.thread_elements() *= kForwardDct8Scale;

  simdgroup_store(b, B + base, 8, ulong2(0), true);
}


// Computes inverse DCT using hardcoded 8x8 DCT-II transform matrix with two matmuls.
//
kernel void gjxl_dct8_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  const ulong base = static_cast<ulong>(group_position.x) * 64ul;

  // lift hard-coded DCT matrix onto threadgroup memory for simdgroup_load()
  threadgroup float c_shared[64];

  for (uint i = lane; i < 64; i += simd_width) {
    c_shared[i] = kOrthonormalDct8[i];
  }

  // synchronize threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 a;
  simdgroup_float8x8 b;
  simdgroup_float8x8 c;  // 8x8 DCT-II transform
  simdgroup_float8x8 ct; // 8x8 DCT-II transform, transposed

  simdgroup_load(a, A + base, 8, ulong2(0), true);
  simdgroup_load(c,  c_shared);
  simdgroup_load(ct, c_shared, 8, ulong2(0), true);

  simdgroup_multiply(b, ct, a);
  simdgroup_multiply(b, b, c);

  b.thread_elements() *= kInverseDct8Scale;

  simdgroup_store(b, B + base);
}


// 16x16

// Computes forward DCT using the precomputed 16x16 DCT-II transform matrix with two matmuls:
//   B = C A C^T.
// See `gjxl_dct8_forward_scalar_2d_matmul()` for details.
kernel void gjxl_dct16_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  int row = tid / 16;
  int col = tid % 16;

  const ulong base = static_cast<ulong>(group_position.x) * 256ul;

  threadgroup float T[16][16]; // intermediate product (C * A)

  float t = 0;

  for (int i=0; i<16; ++i) {
    t += kOrthonormalDct16[16*row + i] * A[base + 16*i + col];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i<16; ++i) {
    b += T[row][i] * kOrthonormalDct16[16*col + i];
  }

  B[base + 16*col + row] = b * kForwardDct16Scale;
}


// Computes inverse DCT using the precomputed 16x16 DCT-II transform matrix with two matmuls:
//   B = C^T A^T C
// See `gjxl_dct8_inverse_scalar_2d_matmul()` for details.
kernel void gjxl_dct16_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  int row = tid / 16;
  int col = tid % 16;

  const ulong base = static_cast<ulong>(group_position.x) * 256ul;

  threadgroup float T[16][16]; // intermediate product (C^T * A^T)

  float t = 0;

  for (int i=0; i<16; ++i) {
    t += kOrthonormalDct16[16*i + row] * A[base + 16*col + i];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i<16; ++i) {
    b += T[row][i] * kOrthonormalDct16[16*i + col];
  }

  B[base + 16*row + col] = b * kInverseDct16Scale;
}


// 16x8 and 8x16

// Computes a forward DCT over 16 rows and 8 columns. The natural [v][u]
// result is transposed to libjxl's [u][v] coefficient layout.
kernel void gjxl_dct16x8_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  int row = tid / 8;
  int col = tid % 8;

  const ulong base = static_cast<ulong>(group_position.x) * 128ul;

  threadgroup float T[16][8]; // vertical DCT in natural [v][x] order

  float t = 0;

  for (int i=0; i<16; ++i) {
    t += kOrthonormalDct16[16*row + i] * A[base + 8*i + col];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i<8; ++i) {
    b += T[row][i] * kOrthonormalDct8[8*col + i];
  }

  B[base + 16*col + row] = b * kForwardDct16x8Scale;
}


// Reads libjxl's transposed [u][v] coefficient layout and reconstructs
// row-major pixels over 16 rows and 8 columns.
kernel void gjxl_dct16x8_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  int row = tid / 8;
  int col = tid % 8;

  const ulong base = static_cast<ulong>(group_position.x) * 128ul;

  threadgroup float T[16][8]; // vertical inverse in natural [y][u] order

  float t = 0;

  for (int i=0; i<16; ++i) {
    t += kOrthonormalDct16[16*i + row] * A[base + 16*col + i];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i<8; ++i) {
    b += T[row][i] * kOrthonormalDct8[8*i + col];
  }

  B[base + 8*row + col] = b * kInverseDct16x8Scale;
}


// Computes a forward DCT over 8 rows and 16 columns. For this orientation,
// libjxl stores coefficients in the natural row-major [v][u] layout.
kernel void gjxl_dct8x16_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  int row = tid / 16;
  int col = tid % 16;

  const ulong base = static_cast<ulong>(group_position.x) * 128ul;

  threadgroup float T[8][16]; // vertical DCT in natural [v][x] order

  float t = 0;

  for (int i=0; i<8; ++i) {
    t += kOrthonormalDct8[8*row + i] * A[base + 16*i + col];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i<16; ++i) {
    b += T[row][i] * kOrthonormalDct16[16*col + i];
  }

  B[base + 16*row + col] = b * kForwardDct16x8Scale;
}


// Reads libjxl's natural [v][u] coefficient layout and reconstructs
// row-major pixels over 8 rows and 16 columns.
kernel void gjxl_dct8x16_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  int row = tid / 16;
  int col = tid % 16;

  const ulong base = static_cast<ulong>(group_position.x) * 128ul;

  threadgroup float T[8][16]; // vertical inverse in natural [y][u] order

  float t = 0;

  for (int i=0; i<8; ++i) {
    t += kOrthonormalDct8[8*i + row] * A[base + 16*i + col];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i<16; ++i) {
    b += T[row][i] * kOrthonormalDct16[16*i + col];
  }

  B[base + 16*row + col] = b * kInverseDct16x8Scale;
}



// 32x32

// Computes forward DCT using the precomputed 32x32 DCT-II transform matrix with two matmuls:
//   B = C A C^T.
// See `gjxl_dct8_forward_scalar_2d_matmul()` for details.
kernel void gjxl_dct32_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  int row = tid / 32;
  int col = tid % 32;

  const ulong base = static_cast<ulong>(group_position.x) * 1024ul;

  threadgroup float T[32][32]; // intermediate product (C * A)

  float t = 0;

  for (int i=0; i<32; ++i) {
    t += kOrthonormalDct32[32*row + i] * A[base + 32*i + col];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i<32; ++i) {
    b += T[row][i] * kOrthonormalDct32[32*col + i];
  }

  B[base + 32*col + row] = b * kForwardDct32Scale;
}


// Computes inverse DCT using the precomputed 32x32 DCT-II transform matrix with two matmuls:
//   B = C^T A^T C
// See `gjxl_dct8_inverse_scalar_2d_matmul()` for details.
kernel void gjxl_dct32_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  int row = tid / 32;
  int col = tid % 32;

  const ulong base = static_cast<ulong>(group_position.x) * 1024ul;

  threadgroup float T[32][32]; // intermediate product (C^T * A^T)

  float t = 0;

  for (int i=0; i<32; ++i) {
    t += kOrthonormalDct32[32*i + row] * A[base + 32*col + i];
  }

  T[row][col] = t;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b = 0;

  for (int i=0; i<32; ++i) {
    b += T[row][i] * kOrthonormalDct32[32*i + col];
  }

  B[base + 32*row + col] = b * kInverseDct32Scale;
}
