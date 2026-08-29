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

// 1 / sqrt(32 * 16). The transpose pair has the same normalization.
constant float kForwardDct32x16Scale = 0.0441941738f;
constant float kInverseDct32x16Scale = 22.6274170f;

// 1 / sqrt(64 * 32). The transpose pair has the same normalization.
constant float kForwardDct64x32Scale = 0.0220970869f;
constant float kInverseDct64x32Scale = 45.2548340f;

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


template <uint N>
__attribute__((always_inline)) inline void ForwardSquareDctSimdgroup(
  device const float* A,
  device       float* B,
  constant const float* basis,
  threadgroup float* shared_basis,
  float scale,
  uint lane,
  uint simd_width,
  uint simdgroup_index,
  uint3 group_position)
{
  constexpr uint kTileSize = 8;
  constexpr uint kTilesPerDimension = N / kTileSize;

  const ulong base =
    static_cast<ulong>(group_position.x) * N * N;

  for (uint i = simdgroup_index * simd_width + lane;
       i < N * N;
       i += kTilesPerDimension * simd_width) {
    shared_basis[i] = basis[i];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One SIMD group owns an eight-row stripe. Keeping its complete C*A
  // stripe in registers avoids an intermediate threadgroup-memory pass.
  simdgroup_float8x8 intermediate[kTilesPerDimension];

  for (uint column_tile = 0;
       column_tile < kTilesPerDimension;
       ++column_tile) {
    simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner_tile = 0;
         inner_tile < kTilesPerDimension;
         ++inner_tile) {
      simdgroup_float8x8 c;
      simdgroup_float8x8 a;

      simdgroup_load(
        c,
        shared_basis,
        N,
        ulong2(inner_tile * kTileSize,
               simdgroup_index * kTileSize));

      simdgroup_load(
        a,
        A + base,
        N,
        ulong2(column_tile * kTileSize,
               inner_tile * kTileSize));

      simdgroup_multiply_accumulate(
        accumulator,
        c,
        a,
        accumulator);
    }

    intermediate[column_tile] = accumulator;
  }

  for (uint column_tile = 0;
       column_tile < kTilesPerDimension;
       ++column_tile) {
    simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner_tile = 0;
         inner_tile < kTilesPerDimension;
         ++inner_tile) {
      simdgroup_float8x8 ct;

      simdgroup_load(
        ct,
        shared_basis,
        N,
        ulong2(inner_tile * kTileSize,
               column_tile * kTileSize),
        true);

      simdgroup_multiply_accumulate(
        accumulator,
        intermediate[inner_tile],
        ct,
        accumulator);
    }

    accumulator.thread_elements() *= scale;

    // Square-transform coefficients use libjxl's transposed [u][v] layout.
    simdgroup_store(
      accumulator,
      B + base,
      N,
      ulong2(simdgroup_index * kTileSize,
             column_tile * kTileSize),
      true);
  }
}


template <uint N>
__attribute__((always_inline)) inline void InverseSquareDctSimdgroup(
  device const float* A,
  device       float* B,
  constant const float* basis,
  threadgroup float* shared_basis,
  float scale,
  uint lane,
  uint simd_width,
  uint simdgroup_index,
  uint3 group_position)
{
  constexpr uint kTileSize = 8;
  constexpr uint kTilesPerDimension = N / kTileSize;

  const ulong base =
    static_cast<ulong>(group_position.x) * N * N;

  for (uint i = simdgroup_index * simd_width + lane;
       i < N * N;
       i += kTilesPerDimension * simd_width) {
    shared_basis[i] = basis[i];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reconstruct one eight-row stripe of C^T*A^T*C per SIMD group. A is
  // loaded transposed because coefficients use libjxl's [u][v] layout.
  simdgroup_float8x8 intermediate[kTilesPerDimension];

  for (uint column_tile = 0;
       column_tile < kTilesPerDimension;
       ++column_tile) {
    simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner_tile = 0;
         inner_tile < kTilesPerDimension;
         ++inner_tile) {
      simdgroup_float8x8 ct;
      simdgroup_float8x8 at;

      simdgroup_load(
        ct,
        shared_basis,
        N,
        ulong2(simdgroup_index * kTileSize,
               inner_tile * kTileSize),
        true);

      simdgroup_load(
        at,
        A + base,
        N,
        ulong2(inner_tile * kTileSize,
               column_tile * kTileSize),
        true);

      simdgroup_multiply_accumulate(
        accumulator,
        ct,
        at,
        accumulator);
    }

    intermediate[column_tile] = accumulator;
  }

  for (uint column_tile = 0;
       column_tile < kTilesPerDimension;
       ++column_tile) {
    simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner_tile = 0;
         inner_tile < kTilesPerDimension;
         ++inner_tile) {
      simdgroup_float8x8 c;

      simdgroup_load(
        c,
        shared_basis,
        N,
        ulong2(column_tile * kTileSize,
               inner_tile * kTileSize));

      simdgroup_multiply_accumulate(
        accumulator,
        intermediate[inner_tile],
        c,
        accumulator);
    }

    accumulator.thread_elements() *= scale;

    simdgroup_store(
      accumulator,
      B + base,
      N,
      ulong2(column_tile * kTileSize,
             simdgroup_index * kTileSize));
  }
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


kernel void gjxl_dct16_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_basis[16 * 16];

  ForwardSquareDctSimdgroup<16>(
    A,
    B,
    kOrthonormalDct16,
    shared_basis,
    kForwardDct16Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


kernel void gjxl_dct16_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_basis[16 * 16];

  InverseSquareDctSimdgroup<16>(
    A,
    B,
    kOrthonormalDct16,
    shared_basis,
    kInverseDct16Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


// Rectangular transforms

template <uint Rows, uint Columns>
__attribute__((always_inline)) inline ulong RectangularCoefficientIndex(
  uint vertical_frequency,
  uint horizontal_frequency)
{
  return Rows < Columns
    ? static_cast<ulong>(vertical_frequency) * Columns +
        horizontal_frequency
    : static_cast<ulong>(horizontal_frequency) * Rows +
        vertical_frequency;
}


template <
  uint Rows,
  uint Columns,
  typename VerticalBasisPointer,
  typename HorizontalBasisPointer>
__attribute__((always_inline)) inline void ForwardRectangularDctStrided(
  device const float* A,
  device       float* B,
  VerticalBasisPointer vertical_basis,
  HorizontalBasisPointer horizontal_basis,
  threadgroup float* T,
  float scale,
  uint tid,
  uint threads_per_threadgroup,
  uint3 group_position)
{
  const ulong base =
    static_cast<ulong>(group_position.x) * Rows * Columns;

  for (uint element = tid;
       element < Rows * Columns;
       element += threads_per_threadgroup) {
    const uint row = element / Columns;
    const uint col = element % Columns;
    float t = 0;

    for (uint i=0; i<Rows; ++i) {
      t += vertical_basis[Rows*row + i] *
        A[base + Columns*i + col];
    }

    T[element] = t;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint element = tid;
       element < Rows * Columns;
       element += threads_per_threadgroup) {
    const uint row = element / Columns;
    const uint col = element % Columns;
    float b = 0;

    for (uint i=0; i<Columns; ++i) {
      b += T[Columns*row + i] * horizontal_basis[Columns*col + i];
    }

    const ulong coefficient =
      RectangularCoefficientIndex<Rows, Columns>(row, col);

    B[base + coefficient] = b * scale;
  }
}


template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void ForwardRectangularDct(
  device const float* A,
  device       float* B,
  constant const float* vertical_basis,
  constant const float* horizontal_basis,
  threadgroup float* T,
  float scale,
  uint tid,
  uint3 group_position)
{
  ForwardRectangularDctStrided<Rows, Columns>(
    A,
    B,
    vertical_basis,
    horizontal_basis,
    T,
    scale,
    tid,
    Rows * Columns,
    group_position);
}


template <
  uint Rows,
  uint Columns,
  typename VerticalBasisPointer,
  typename HorizontalBasisPointer>
__attribute__((always_inline)) inline void InverseRectangularDctStrided(
  device const float* A,
  device       float* B,
  VerticalBasisPointer vertical_basis,
  HorizontalBasisPointer horizontal_basis,
  threadgroup float* T,
  float scale,
  uint tid,
  uint threads_per_threadgroup,
  uint3 group_position)
{
  const ulong base =
    static_cast<ulong>(group_position.x) * Rows * Columns;

  for (uint element = tid;
       element < Rows * Columns;
       element += threads_per_threadgroup) {
    const uint row = element / Columns;
    const uint col = element % Columns;
    float t = 0;

    for (uint i=0; i<Rows; ++i) {
      const ulong coefficient =
        RectangularCoefficientIndex<Rows, Columns>(i, col);

      t += vertical_basis[Rows*i + row] * A[base + coefficient];
    }

    T[element] = t;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint element = tid;
       element < Rows * Columns;
       element += threads_per_threadgroup) {
    const uint row = element / Columns;
    const uint col = element % Columns;
    float b = 0;

    for (uint i=0; i<Columns; ++i) {
      b += T[Columns*row + i] * horizontal_basis[Columns*i + col];
    }

    B[base + element] = b * scale;
  }
}


template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void InverseRectangularDct(
  device const float* A,
  device       float* B,
  constant const float* vertical_basis,
  constant const float* horizontal_basis,
  threadgroup float* T,
  float scale,
  uint tid,
  uint3 group_position)
{
  InverseRectangularDctStrided<Rows, Columns>(
    A,
    B,
    vertical_basis,
    horizontal_basis,
    T,
    scale,
    tid,
    Rows * Columns,
    group_position);
}


template <
  uint Rows,
  uint Columns,
  typename VerticalBasisPointer,
  typename HorizontalBasisPointer>
__attribute__((always_inline)) inline void
ForwardRectangularDctSimdgroupWithBasis(
  device const float* A,
  device       float* B,
  VerticalBasisPointer vertical_basis,
  HorizontalBasisPointer horizontal_basis,
  float scale,
  uint simdgroup_index,
  uint3 group_position)
{
  constexpr uint kTileSize = 8;
  constexpr uint kRowTiles = Rows / kTileSize;
  constexpr uint kColumnTiles = Columns / kTileSize;

  const ulong base =
    static_cast<ulong>(group_position.x) * Rows * Columns;

  // One SIMD group owns an eight-row stripe and keeps its complete vertical
  // transform in registers for the horizontal pass.
  simdgroup_float8x8 intermediate[kColumnTiles];

  for (uint column_tile = 0;
       column_tile < kColumnTiles;
       ++column_tile) {
    simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner_tile = 0;
         inner_tile < kRowTiles;
         ++inner_tile) {
      simdgroup_float8x8 c;
      simdgroup_float8x8 a;

      simdgroup_load(
        c,
        vertical_basis,
        Rows,
        ulong2(inner_tile * kTileSize,
               simdgroup_index * kTileSize));

      simdgroup_load(
        a,
        A + base,
        Columns,
        ulong2(column_tile * kTileSize,
               inner_tile * kTileSize));

      simdgroup_multiply_accumulate(
        accumulator,
        c,
        a,
        accumulator);
    }

    intermediate[column_tile] = accumulator;
  }

  for (uint column_tile = 0;
       column_tile < kColumnTiles;
       ++column_tile) {
    simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner_tile = 0;
         inner_tile < kColumnTiles;
         ++inner_tile) {
      simdgroup_float8x8 ct;

      simdgroup_load(
        ct,
        horizontal_basis,
        Columns,
        ulong2(inner_tile * kTileSize,
               column_tile * kTileSize),
        true);

      simdgroup_multiply_accumulate(
        accumulator,
        intermediate[inner_tile],
        ct,
        accumulator);
    }

    accumulator.thread_elements() *= scale;

    if (Rows < Columns) {
      simdgroup_store(
        accumulator,
        B + base,
        Columns,
        ulong2(column_tile * kTileSize,
               simdgroup_index * kTileSize));
    } else {
      simdgroup_store(
        accumulator,
        B + base,
        Rows,
        ulong2(simdgroup_index * kTileSize,
               column_tile * kTileSize),
        true);
    }
  }
}


template <
  uint Rows,
  uint Columns,
  typename VerticalBasisPointer,
  typename HorizontalBasisPointer>
__attribute__((always_inline)) inline void
InverseRectangularDctSimdgroupWithBasis(
  device const float* A,
  device       float* B,
  VerticalBasisPointer vertical_basis,
  HorizontalBasisPointer horizontal_basis,
  float scale,
  uint simdgroup_index,
  uint3 group_position)
{
  constexpr uint kTileSize = 8;
  constexpr uint kRowTiles = Rows / kTileSize;
  constexpr uint kColumnTiles = Columns / kTileSize;

  const ulong base =
    static_cast<ulong>(group_position.x) * Rows * Columns;

  simdgroup_float8x8 intermediate[kColumnTiles];

  for (uint column_tile = 0;
       column_tile < kColumnTiles;
       ++column_tile) {
    simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner_tile = 0;
         inner_tile < kRowTiles;
         ++inner_tile) {
      simdgroup_float8x8 ct;
      simdgroup_float8x8 coefficients;

      simdgroup_load(
        ct,
        vertical_basis,
        Rows,
        ulong2(simdgroup_index * kTileSize,
               inner_tile * kTileSize),
        true);

      if (Rows < Columns) {
        simdgroup_load(
          coefficients,
          A + base,
          Columns,
          ulong2(column_tile * kTileSize,
                 inner_tile * kTileSize));
      } else {
        simdgroup_load(
          coefficients,
          A + base,
          Rows,
          ulong2(inner_tile * kTileSize,
                 column_tile * kTileSize),
          true);
      }

      simdgroup_multiply_accumulate(
        accumulator,
        ct,
        coefficients,
        accumulator);
    }

    intermediate[column_tile] = accumulator;
  }

  for (uint column_tile = 0;
       column_tile < kColumnTiles;
       ++column_tile) {
    simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner_tile = 0;
         inner_tile < kColumnTiles;
         ++inner_tile) {
      simdgroup_float8x8 c;

      simdgroup_load(
        c,
        horizontal_basis,
        Columns,
        ulong2(column_tile * kTileSize,
               inner_tile * kTileSize));

      simdgroup_multiply_accumulate(
        accumulator,
        intermediate[inner_tile],
        c,
        accumulator);
    }

    accumulator.thread_elements() *= scale;

    simdgroup_store(
      accumulator,
      B + base,
      Columns,
      ulong2(column_tile * kTileSize,
             simdgroup_index * kTileSize));
  }
}


template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void StageRectangularDctBasis(
  constant const float* vertical_basis,
  constant const float* horizontal_basis,
  threadgroup float* shared_vertical_basis,
  threadgroup float* shared_horizontal_basis,
  uint lane,
  uint simd_width,
  uint simdgroup_index)
{
  constexpr uint kSimdgroupsPerThreadgroup = Rows / 8;
  const uint threadgroup_stride =
    kSimdgroupsPerThreadgroup * simd_width;

  for (uint i = simdgroup_index * simd_width + lane;
       i < Rows * Rows;
       i += threadgroup_stride) {
    shared_vertical_basis[i] = vertical_basis[i];
  }

  for (uint i = simdgroup_index * simd_width + lane;
       i < Columns * Columns;
       i += threadgroup_stride) {
    shared_horizontal_basis[i] = horizontal_basis[i];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}


template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void ForwardRectangularDctSimdgroup(
  device const float* A,
  device       float* B,
  constant const float* vertical_basis,
  constant const float* horizontal_basis,
  threadgroup float* shared_vertical_basis,
  threadgroup float* shared_horizontal_basis,
  float scale,
  uint lane,
  uint simd_width,
  uint simdgroup_index,
  uint3 group_position)
{
  StageRectangularDctBasis<Rows, Columns>(
    vertical_basis,
    horizontal_basis,
    shared_vertical_basis,
    shared_horizontal_basis,
    lane,
    simd_width,
    simdgroup_index);

  ForwardRectangularDctSimdgroupWithBasis<Rows, Columns>(
    A,
    B,
    shared_vertical_basis,
    shared_horizontal_basis,
    scale,
    simdgroup_index,
    group_position);
}


template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void InverseRectangularDctSimdgroup(
  device const float* A,
  device       float* B,
  constant const float* vertical_basis,
  constant const float* horizontal_basis,
  threadgroup float* shared_vertical_basis,
  threadgroup float* shared_horizontal_basis,
  float scale,
  uint lane,
  uint simd_width,
  uint simdgroup_index,
  uint3 group_position)
{
  StageRectangularDctBasis<Rows, Columns>(
    vertical_basis,
    horizontal_basis,
    shared_vertical_basis,
    shared_horizontal_basis,
    lane,
    simd_width,
    simdgroup_index);

  InverseRectangularDctSimdgroupWithBasis<Rows, Columns>(
    A,
    B,
    shared_vertical_basis,
    shared_horizontal_basis,
    scale,
    simdgroup_index,
    group_position);
}


// Computes a forward DCT over 16 rows and 8 columns. The natural [v][u]
// result is transposed to libjxl's [u][v] coefficient layout.
kernel void gjxl_dct16x8_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float T[16 * 8];

  ForwardRectangularDct<16, 8>(
    A,
    B,
    kOrthonormalDct16,
    kOrthonormalDct8,
    T,
    kForwardDct16x8Scale,
    tid,
    group_position);
}


// Reads libjxl's transposed [u][v] coefficient layout and reconstructs
// row-major pixels over 16 rows and 8 columns.
kernel void gjxl_dct16x8_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float T[16 * 8];

  InverseRectangularDct<16, 8>(
    A,
    B,
    kOrthonormalDct16,
    kOrthonormalDct8,
    T,
    kInverseDct16x8Scale,
    tid,
    group_position);
}


kernel void gjxl_dct16x8_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_vertical_basis[16 * 16];
  threadgroup float shared_horizontal_basis[8 * 8];

  ForwardRectangularDctSimdgroup<16, 8>(
    A,
    B,
    kOrthonormalDct16,
    kOrthonormalDct8,
    shared_vertical_basis,
    shared_horizontal_basis,
    kForwardDct16x8Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


kernel void gjxl_dct16x8_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_vertical_basis[16 * 16];
  threadgroup float shared_horizontal_basis[8 * 8];

  InverseRectangularDctSimdgroup<16, 8>(
    A,
    B,
    kOrthonormalDct16,
    kOrthonormalDct8,
    shared_vertical_basis,
    shared_horizontal_basis,
    kInverseDct16x8Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


// Computes a forward DCT over 8 rows and 16 columns. For this orientation,
// libjxl stores coefficients in the natural row-major [v][u] layout.
kernel void gjxl_dct8x16_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float T[8 * 16];

  ForwardRectangularDct<8, 16>(
    A,
    B,
    kOrthonormalDct8,
    kOrthonormalDct16,
    T,
    kForwardDct16x8Scale,
    tid,
    group_position);
}


// Reads libjxl's natural [v][u] coefficient layout and reconstructs
// row-major pixels over 8 rows and 16 columns.
kernel void gjxl_dct8x16_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float T[8 * 16];

  InverseRectangularDct<8, 16>(
    A,
    B,
    kOrthonormalDct8,
    kOrthonormalDct16,
    T,
    kInverseDct16x8Scale,
    tid,
    group_position);
}


kernel void gjxl_dct8x16_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_vertical_basis[8 * 8];
  threadgroup float shared_horizontal_basis[16 * 16];

  ForwardRectangularDctSimdgroup<8, 16>(
    A,
    B,
    kOrthonormalDct8,
    kOrthonormalDct16,
    shared_vertical_basis,
    shared_horizontal_basis,
    kForwardDct16x8Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


kernel void gjxl_dct8x16_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  InverseRectangularDctSimdgroupWithBasis<8, 16>(
    A,
    B,
    vertical_basis,
    horizontal_basis,
    kInverseDct16x8Scale,
    simdgroup_index,
    group_position);
}


// Computes a forward DCT over 32 rows and 16 columns. The natural [v][u]
// result is transposed to libjxl's [u][v] coefficient layout.
kernel void gjxl_dct32x16_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float T[32 * 16];

  ForwardRectangularDct<32, 16>(
    A,
    B,
    kOrthonormalDct32,
    kOrthonormalDct16,
    T,
    kForwardDct32x16Scale,
    tid,
    group_position);
}


// Reads libjxl's transposed [u][v] coefficient layout and reconstructs
// row-major pixels over 32 rows and 16 columns.
kernel void gjxl_dct32x16_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float T[32 * 16];

  InverseRectangularDct<32, 16>(
    A,
    B,
    kOrthonormalDct32,
    kOrthonormalDct16,
    T,
    kInverseDct32x16Scale,
    tid,
    group_position);
}


kernel void gjxl_dct32x16_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_vertical_basis[32 * 32];
  threadgroup float shared_horizontal_basis[16 * 16];

  ForwardRectangularDctSimdgroup<32, 16>(
    A,
    B,
    kOrthonormalDct32,
    kOrthonormalDct16,
    shared_vertical_basis,
    shared_horizontal_basis,
    kForwardDct32x16Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


kernel void gjxl_dct32x16_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_vertical_basis[32 * 32];
  threadgroup float shared_horizontal_basis[16 * 16];

  InverseRectangularDctSimdgroup<32, 16>(
    A,
    B,
    kOrthonormalDct32,
    kOrthonormalDct16,
    shared_vertical_basis,
    shared_horizontal_basis,
    kInverseDct32x16Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


// Computes a forward DCT over 16 rows and 32 columns. For this orientation,
// libjxl stores coefficients in the natural row-major [v][u] layout.
kernel void gjxl_dct16x32_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (pixels)
  device       float* B [[buffer(1)]], // output (coefficients)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float T[16 * 32];

  ForwardRectangularDct<16, 32>(
    A,
    B,
    kOrthonormalDct16,
    kOrthonormalDct32,
    T,
    kForwardDct32x16Scale,
    tid,
    group_position);
}


// Reads libjxl's natural [v][u] coefficient layout and reconstructs
// row-major pixels over 16 rows and 32 columns.
kernel void gjxl_dct16x32_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]], // input (coefficients)
  device       float* B [[buffer(1)]], // output (pixels)
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float T[16 * 32];

  InverseRectangularDct<16, 32>(
    A,
    B,
    kOrthonormalDct16,
    kOrthonormalDct32,
    T,
    kInverseDct32x16Scale,
    tid,
    group_position);
}


kernel void gjxl_dct16x32_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_vertical_basis[16 * 16];
  threadgroup float shared_horizontal_basis[32 * 32];

  ForwardRectangularDctSimdgroup<16, 32>(
    A,
    B,
    kOrthonormalDct16,
    kOrthonormalDct32,
    shared_vertical_basis,
    shared_horizontal_basis,
    kForwardDct32x16Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


kernel void gjxl_dct16x32_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  InverseRectangularDctSimdgroupWithBasis<16, 32>(
    A,
    B,
    vertical_basis,
    horizontal_basis,
    kInverseDct32x16Scale,
    simdgroup_index,
    group_position);
}


// 64x32 and 32x64 bases are supplied from a device buffer to keep the
// 4096-element DCT64 table out of every compiled kernel's constant data.
kernel void gjxl_dct64x32_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[64 * 32];

  ForwardRectangularDctStrided<64, 32>(
    A, B, vertical_basis, horizontal_basis, intermediate,
    kForwardDct64x32Scale, tid, tg_size.x, group_position);
}


kernel void gjxl_dct64x32_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[64 * 32];

  InverseRectangularDctStrided<64, 32>(
    A, B, vertical_basis, horizontal_basis, intermediate,
    kInverseDct64x32Scale, tid, tg_size.x, group_position);
}


kernel void gjxl_dct64x32_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  ForwardRectangularDctSimdgroupWithBasis<64, 32>(
    A, B, vertical_basis, horizontal_basis, kForwardDct64x32Scale,
    simdgroup_index, group_position);
}


kernel void gjxl_dct64x32_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  InverseRectangularDctSimdgroupWithBasis<64, 32>(
    A, B, vertical_basis, horizontal_basis, kInverseDct64x32Scale,
    simdgroup_index, group_position);
}


kernel void gjxl_dct32x64_forward_scalar_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[32 * 64];

  ForwardRectangularDctStrided<32, 64>(
    A, B, vertical_basis, horizontal_basis, intermediate,
    kForwardDct64x32Scale, tid, tg_size.x, group_position);
}


kernel void gjxl_dct32x64_inverse_scalar_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[32 * 64];

  InverseRectangularDctStrided<32, 64>(
    A, B, vertical_basis, horizontal_basis, intermediate,
    kInverseDct64x32Scale, tid, tg_size.x, group_position);
}


kernel void gjxl_dct32x64_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  ForwardRectangularDctSimdgroupWithBasis<32, 64>(
    A, B, vertical_basis, horizontal_basis, kForwardDct64x32Scale,
    simdgroup_index, group_position);
}


kernel void gjxl_dct32x64_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  device const float* vertical_basis [[buffer(2)]],
  device const float* horizontal_basis [[buffer(3)]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  InverseRectangularDctSimdgroupWithBasis<32, 64>(
    A, B, vertical_basis, horizontal_basis, kInverseDct64x32Scale,
    simdgroup_index, group_position);
}


// Factored radix-2 transforms

constant float kFactoredDctSqrt2 = 1.41421356237f;

constant float kFactoredDctMultipliers4[2] = {
  0.541196100146197f,
  1.3065629648763764f,
};

constant float kFactoredDctMultipliers8[4] = {
  0.5097955791041592f,
  0.6013448869350453f,
  0.8999762231364156f,
  2.5629154477415055f,
};

constant float kFactoredDctMultipliers16[8] = {
  0.5024192861881557f,
  0.5224986149396889f,
  0.5669440348163577f,
  0.6468217833599901f,
  0.7881546234512502f,
  1.060677685990347f,
  1.7224470982383342f,
  5.101148618689155f,
};

constant float kFactoredDctMultipliers32[16] = {
  0.5006029982351963f,
  0.5054709598975436f,
  0.5154473099226246f,
  0.5310425910897841f,
  0.5531038960344445f,
  0.5829349682061339f,
  0.6225041230356648f,
  0.6748083414550057f,
  0.7445362710022986f,
  0.8393496454155268f,
  0.9725682378619608f,
  1.1694399334328847f,
  1.4841646163141662f,
  2.057781009953411f,
  3.407608418468719f,
  10.190008123548033f,
};

constant float kFactoredDctMultipliers64[32] = {
  0.50015063602065102f,
  0.50135845244640842f,
  0.50378872568104427f,
  0.50747117207255532f,
  0.51245147940822466f,
  0.51879271310533281f,
  0.52657731515426998f,
  0.535909816907992f,
  0.54692043798550882f,
  0.55976981294708017f,
  0.57465518403266003f,
  0.59181853585741651f,
  0.61155734788250993f,
  0.63423893668840325f,
  0.66031980781370614f,
  0.69037212820021232f,
  0.72512052237719848f,
  0.76549416497308898f,
  0.81270209081449052f,
  0.86834471522334811f,
  0.93458359703640748f,
  1.0144082649970547f,
  1.1120716205797176f,
  1.233832737976571f,
  1.3892939586328277f,
  1.5939722833856311f,
  1.8746759800084078f,
  2.2820500680051619f,
  2.9246284281582162f,
  4.0846110781292477f,
  6.7967507116736332f,
  20.373878167231453f,
};


template <uint N>
struct FactoredDctMultipliers;

template <>
struct FactoredDctMultipliers<4> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kFactoredDctMultipliers4[index];
  }
};

template <>
struct FactoredDctMultipliers<8> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kFactoredDctMultipliers8[index];
  }
};

template <>
struct FactoredDctMultipliers<16> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kFactoredDctMultipliers16[index];
  }
};

template <>
struct FactoredDctMultipliers<32> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kFactoredDctMultipliers32[index];
  }
};

template <>
struct FactoredDctMultipliers<64> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kFactoredDctMultipliers64[index];
  }
};


// Lowest-complexity self-recursive radix-2 DCT-II/III, following the
// factorization used by the pinned libjxl implementation. Forward() produces
// an unscaled DCT-II; Inverse() consumes coefficients scaled by 1/N.
template <uint N>
struct FactoredDct1D {
  __attribute__((always_inline)) static void Forward(
    thread float* values,
    thread float* scratch)
  {
    constexpr uint kHalf = N / 2;

    for (uint i = 0; i < kHalf; ++i) {
      scratch[i] = values[i] + values[N - i - 1];
    }

    FactoredDct1D<kHalf>::Forward(scratch, scratch + N);

    for (uint i = 0; i < kHalf; ++i) {
      scratch[kHalf + i] =
        (values[i] - values[N - i - 1]) *
        FactoredDctMultipliers<N>::Get(i);
    }

    FactoredDct1D<kHalf>::Forward(
      scratch + kHalf,
      scratch + N);

    scratch[kHalf] =
      scratch[kHalf] * kFactoredDctSqrt2 +
      scratch[kHalf + 1];

    for (uint i = 1; i + 1 < kHalf; ++i) {
      scratch[kHalf + i] += scratch[kHalf + i + 1];
    }

    for (uint i = 0; i < kHalf; ++i) {
      values[2 * i] = scratch[i];
      values[2 * i + 1] = scratch[kHalf + i];
    }
  }

  __attribute__((always_inline)) static void Inverse(
    thread float* values,
    thread float* scratch)
  {
    constexpr uint kHalf = N / 2;

    for (uint i = 0; i < kHalf; ++i) {
      scratch[i] = values[2 * i];
      scratch[kHalf + i] = values[2 * i + 1];
    }

    FactoredDct1D<kHalf>::Inverse(scratch, scratch + N);

    for (uint i = kHalf - 1; i > 0; --i) {
      scratch[kHalf + i] += scratch[kHalf + i - 1];
    }

    scratch[kHalf] *= kFactoredDctSqrt2;

    FactoredDct1D<kHalf>::Inverse(
      scratch + kHalf,
      scratch + N);

    for (uint i = 0; i < kHalf; ++i) {
      const float even = scratch[i];
      const float odd =
        scratch[kHalf + i] *
        FactoredDctMultipliers<N>::Get(i);

      values[i] = even + odd;
      values[N - i - 1] = even - odd;
    }
  }
};


template <>
struct FactoredDct1D<2> {
  __attribute__((always_inline)) static void Forward(
    thread float* values,
    thread float*)
  {
    const float first = values[0];
    const float second = values[1];
    values[0] = first + second;
    values[1] = first - second;
  }

  __attribute__((always_inline)) static void Inverse(
    thread float* values,
    thread float*)
  {
    const float dc = values[0];
    const float ac = values[1];
    values[0] = dc + ac;
    values[1] = dc - ac;
  }
};


template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void ForwardFactoredDct2D(
  device const float* A,
  device       float* B,
  threadgroup float* intermediate,
  uint tid,
  uint threads_per_threadgroup,
  ulong transform_index,
  bool transform_is_active)
{
  constexpr uint kMaxLength = Rows > Columns ? Rows : Columns;
  constexpr float kScale = 1.0f / (Rows * Columns);

  const ulong base =
    transform_index * Rows * Columns;

  thread float values[kMaxLength];
  thread float scratch[2 * kMaxLength];

  // Transform each pixel row, retaining natural [y][u] order.
  if (transform_is_active) {
    for (uint row = tid;
         row < Rows;
         row += threads_per_threadgroup) {
      for (uint x = 0; x < Columns; ++x) {
        values[x] = A[base + row * Columns + x];
      }

      FactoredDct1D<Columns>::Forward(values, scratch);

      for (uint u = 0; u < Columns; ++u) {
        intermediate[row * Columns + u] = values[u];
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Transform each column and store libjxl's shape-dependent layout.
  if (transform_is_active) {
    for (uint u = tid;
         u < Columns;
         u += threads_per_threadgroup) {
      for (uint y = 0; y < Rows; ++y) {
        values[y] = intermediate[y * Columns + u];
      }

      FactoredDct1D<Rows>::Forward(values, scratch);

      for (uint v = 0; v < Rows; ++v) {
        const ulong coefficient =
          RectangularCoefficientIndex<Rows, Columns>(v, u);

        B[base + coefficient] = values[v] * kScale;
      }
    }
  }
}


template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void InverseFactoredDct2D(
  device const float* A,
  device       float* B,
  threadgroup float* intermediate,
  uint tid,
  uint threads_per_threadgroup,
  ulong transform_index,
  bool transform_is_active)
{
  constexpr uint kMaxLength = Rows > Columns ? Rows : Columns;

  const ulong base =
    transform_index * Rows * Columns;

  thread float values[kMaxLength];
  thread float scratch[2 * kMaxLength];

  // Undo the vertical-frequency axis while reading libjxl's layout.
  if (transform_is_active) {
    for (uint u = tid;
         u < Columns;
         u += threads_per_threadgroup) {
      for (uint v = 0; v < Rows; ++v) {
        const ulong coefficient =
          RectangularCoefficientIndex<Rows, Columns>(v, u);

        values[v] = A[base + coefficient];
      }

      FactoredDct1D<Rows>::Inverse(values, scratch);

      for (uint y = 0; y < Rows; ++y) {
        intermediate[y * Columns + u] = values[y];
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Undo the horizontal-frequency axis and return row-major pixels.
  if (transform_is_active) {
    for (uint y = tid;
         y < Rows;
         y += threads_per_threadgroup) {
      for (uint u = 0; u < Columns; ++u) {
        values[u] = intermediate[y * Columns + u];
      }

      FactoredDct1D<Columns>::Inverse(values, scratch);

      for (uint x = 0; x < Columns; ++x) {
        B[base + y * Columns + x] = values[x];
      }
    }
  }
}


kernel void gjxl_dct8_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  constant uint& transform_count [[buffer(2)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  constexpr uint kTransformsPerThreadgroup = 4;
  constexpr uint kThreadsPerTransform = 8;
  constexpr uint kElementsPerTransform = 8 * 8;

  const uint transform_in_threadgroup = tid / kThreadsPerTransform;
  const uint transform_tid = tid % kThreadsPerTransform;
  const ulong transform_index =
    static_cast<ulong>(group_position.x) *
      kTransformsPerThreadgroup +
    transform_in_threadgroup;

  threadgroup float intermediate[
    kTransformsPerThreadgroup * kElementsPerTransform];

  ForwardFactoredDct2D<8, 8>(
    A,
    B,
    intermediate +
      transform_in_threadgroup * kElementsPerTransform,
    transform_tid,
    kThreadsPerTransform,
    transform_index,
    transform_index < transform_count);
}


kernel void gjxl_dct8_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  constant uint& transform_count [[buffer(2)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  constexpr uint kTransformsPerThreadgroup = 4;
  constexpr uint kThreadsPerTransform = 8;
  constexpr uint kElementsPerTransform = 8 * 8;

  const uint transform_in_threadgroup = tid / kThreadsPerTransform;
  const uint transform_tid = tid % kThreadsPerTransform;
  const ulong transform_index =
    static_cast<ulong>(group_position.x) *
      kTransformsPerThreadgroup +
    transform_in_threadgroup;

  threadgroup float intermediate[
    kTransformsPerThreadgroup * kElementsPerTransform];

  InverseFactoredDct2D<8, 8>(
    A,
    B,
    intermediate +
      transform_in_threadgroup * kElementsPerTransform,
    transform_tid,
    kThreadsPerTransform,
    transform_index,
    transform_index < transform_count);
}


kernel void gjxl_dct16_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[16 * 16];

  ForwardFactoredDct2D<16, 16>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct16_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[16 * 16];

  InverseFactoredDct2D<16, 16>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct16x8_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[16 * 8];

  ForwardFactoredDct2D<16, 8>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct16x8_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[16 * 8];

  InverseFactoredDct2D<16, 8>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct8x16_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[8 * 16];

  ForwardFactoredDct2D<8, 16>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct8x16_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[8 * 16];

  InverseFactoredDct2D<8, 16>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct32x16_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[32 * 16];

  ForwardFactoredDct2D<32, 16>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct32x16_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[32 * 16];

  InverseFactoredDct2D<32, 16>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct16x32_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[16 * 32];

  ForwardFactoredDct2D<16, 32>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct16x32_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[16 * 32];

  InverseFactoredDct2D<16, 32>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct64x32_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[64 * 32];

  ForwardFactoredDct2D<64, 32>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct64x32_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[64 * 32];

  InverseFactoredDct2D<64, 32>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct32x64_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[32 * 64];

  ForwardFactoredDct2D<32, 64>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct32x64_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[32 * 64];

  InverseFactoredDct2D<32, 64>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct32_forward_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[32 * 32];

  ForwardFactoredDct2D<32, 32>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
}


kernel void gjxl_dct32_inverse_factored_radix2(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint              tid [[thread_index_in_threadgroup]],
  uint3        tg_size  [[threads_per_threadgroup]],
  uint3  group_position [[threadgroup_position_in_grid]])
{
  threadgroup float intermediate[32 * 32];

  InverseFactoredDct2D<32, 32>(
    A, B, intermediate, tid, tg_size.x,
    static_cast<ulong>(group_position.x), true);
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


kernel void gjxl_dct32_forward_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_basis[32 * 32];

  ForwardSquareDctSimdgroup<32>(
    A,
    B,
    kOrthonormalDct32,
    shared_basis,
    kForwardDct32Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}


kernel void gjxl_dct32_inverse_simdgroup_2d_matmul(
  device const float* A [[buffer(0)]],
  device       float* B [[buffer(1)]],
  uint  lane            [[thread_index_in_simdgroup]],
  uint  simd_width      [[threads_per_simdgroup]],
  uint  simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint3 group_position  [[threadgroup_position_in_grid]])
{
  threadgroup float shared_basis[32 * 32];

  InverseSquareDctSimdgroup<32>(
    A,
    B,
    kOrthonormalDct32,
    shared_basis,
    kInverseDct32Scale,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
}
