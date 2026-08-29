// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

#include "dct_basis.h"

using namespace metal;

struct AcStrategyCandidate {
  uint block_x;
  uint block_y;
  float quant_norm;
  float entropy_multiplier;
  float cfl_x;
  float cfl_b;
};

struct AcStrategyBatchParams {
  uint pixel_width;
  uint pixel_height;
  uint opsin_row_stride;
  uint pixel_mask_row_stride;
  uint quant_field_row_stride;
  uint candidate_count;
  uint coefficient_count;
  uint transform_width;
  uint transform_height;
  uint covered_block_width;
  uint covered_block_height;
  uint covered_block_count;
  uint use_device_quant_norm;
  float info_loss_multiplier;
  float zeros_multiplier;
  float cost_delta;
};

struct ChannelRate {
  float magnitude;
  uint nonzero_count;
};

constant float kMaskOffset[3] = {12.0f, 0.0f, 4.0f};
constant float kChannelMultiplier[3] = {
  2.0441408586549744e7f,
  1.0f,
  1.266770081387616f,
};

constant float kForwardDct8Scale = 1.0f / 8.0f;
constant float kInverseDct8Scale = 8.0f;
constant float kForwardDct16Scale = 1.0f / 16.0f;
constant float kInverseDct16Scale = 16.0f;
constant float kForwardDct16x8Scale = 0.0883883461f;
constant float kInverseDct16x8Scale = 11.3137083f;
constant float kForwardDct32x16Scale = 0.0441941738f;
constant float kInverseDct32x16Scale = 22.6274170f;
constant float kForwardDct32Scale = 1.0f / 32.0f;
constant float kInverseDct32Scale = 32.0f;

inline uint CeilLog2Nonzero(uint value) {
  return value <= 1 ? 0 : 32 - clz(value - 1);
}

inline float RoundAwayFromZero(float value) {
  return copysign(floor(abs(value) + 0.5f), value);
}

inline float FastLog2(float value) {
  const uint value_bits = as_type<uint>(value);
  const int shifted_exponent = int(value_bits - 0x3f2aaaabu) >> 23;
  const uint mantissa_bits =
    value_bits - (uint(shifted_exponent) << 23);
  const float x = as_type<float>(mantissa_bits) - 1.0f;
  float numerator = fma(0.74245873327820566f, x, 1.4287160470083755f);
  numerator = fma(numerator, x, -1.8503833400518310e-06f);
  float denominator = fma(0.17409343003366853f, x, 1.0096718572241148f);
  denominator = fma(denominator, x, 0.99032814277590719f);
  return numerator / denominator + float(shifted_exponent);
}

inline float FastPow2(float value) {
  const float floor_value = floor(value);
  const int exponent = int(floor_value) + 127;
  const float exponent_value = as_type<float>(uint(exponent) << 23);
  const float fraction = value - floor_value;
  float numerator = fraction + 1.01749063e+01f;
  numerator = fma(numerator, fraction, 4.88687798e+01f);
  numerator = fma(numerator, fraction, 9.85506591e+01f);
  numerator *= exponent_value;
  float denominator = fma(fraction, 2.10242958e-01f, -2.22328856e-02f);
  denominator = fma(denominator, fraction, -1.94414990e+01f);
  denominator = fma(denominator, fraction, 9.85506633e+01f);
  return numerator / denominator;
}

inline float ComputeQuantNorm(
  device const float* quant_field,
  AcStrategyCandidate candidate,
  constant AcStrategyBatchParams& params) {

  if (params.use_device_quant_norm == 0u) return candidate.quant_norm;
  if (params.covered_block_count == 1u) {
    return quant_field[
      candidate.block_y * params.quant_field_row_stride + candidate.block_x];
  }
  if (params.covered_block_count == 2u) {
    const float first = quant_field[
      candidate.block_y * params.quant_field_row_stride + candidate.block_x];
    const uint second_x = candidate.block_x +
      (params.covered_block_width == 2u ? 1u : 0u);
    const uint second_y = candidate.block_y +
      (params.covered_block_height == 2u ? 1u : 0u);
    const float second = quant_field[
      second_y * params.quant_field_row_stride + second_x];
    return max(first, second);
  }
  float sum = 0.0f;
  for (uint dy = 0; dy < params.covered_block_height; ++dy) {
    for (uint dx = 0; dx < params.covered_block_width; ++dx) {
      float value = quant_field[
        (candidate.block_y + dy) * params.quant_field_row_stride +
        candidate.block_x + dx];
      value *= value;
      value *= value;
      value *= value;
      sum += value * value;
    }
  }
  sum /= float(params.covered_block_count);
  return FastPow2(FastLog2(sum) * (1.0f / 16.0f));
}

inline bool AcStrategyCandidateValid(
  AcStrategyCandidate candidate,
  constant AcStrategyBatchParams& params) {

  return params.transform_width <= params.pixel_width &&
    params.transform_height <= params.pixel_height &&
    candidate.block_x <=
      (params.pixel_width - params.transform_width) / 8 &&
    candidate.block_y <=
      (params.pixel_height - params.transform_height) / 8 &&
    isfinite(candidate.quant_norm) && candidate.quant_norm > 0.0f &&
    isfinite(candidate.entropy_multiplier) &&
    candidate.entropy_multiplier > 0.0f &&
    isfinite(candidate.cfl_x) && isfinite(candidate.cfl_b);
}

kernel void gjxl_ac_strategy_gather(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device float* packed_pixels [[buffer(4)]],
  constant AcStrategyBatchParams& params [[buffer(5)]],
  uint index [[thread_position_in_grid]]) {

  const uint channel_stride = params.coefficient_count;
  const uint candidate_stride = 3 * channel_stride;
  const uint element_count = params.candidate_count * candidate_stride;
  if (index >= element_count) {
    return;
  }

  const uint candidate_index = index / candidate_stride;
  const uint candidate_element = index % candidate_stride;
  const uint channel = candidate_element / channel_stride;
  const uint element = candidate_element % channel_stride;
  const uint row = element / params.transform_width;
  const uint column = element % params.transform_width;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool candidate_valid = AcStrategyCandidateValid(candidate, params);
  if (!candidate_valid) {
    packed_pixels[index] = NAN;
    return;
  }
  const uint pixel_x = candidate.block_x * 8 + column;
  const uint pixel_y = candidate.block_y * 8 + row;

  const uint source_index = pixel_y * params.opsin_row_stride + pixel_x;
  packed_pixels[index] = channel == 0u ? opsin_x[source_index] :
    channel == 1u ? opsin_y[source_index] : opsin_b[source_index];
}

template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void GatherAcStrategyPixels(
  device const float* opsin_x,
  device const float* opsin_y,
  device const float* opsin_b,
  device const AcStrategyCandidate* candidates,
  constant AcStrategyBatchParams& params,
  threadgroup float* pixels,
  uint lane,
  uint simd_width,
  uint simdgroup_index,
  uint3 group_position) {

  constexpr uint kSimdgroupsPerThreadgroup = Rows / 8;
  const uint threadgroup_stride =
    kSimdgroupsPerThreadgroup * simd_width;
  const uint transform_index = group_position.x;
  const uint candidate_index = transform_index / 3;
  const uint channel = transform_index % 3;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool valid = AcStrategyCandidateValid(candidate, params);
  device const float* source = channel == 0u ? opsin_x :
    channel == 1u ? opsin_y : opsin_b;

  for (uint index = simdgroup_index * simd_width + lane;
       index < Rows * Columns;
       index += threadgroup_stride) {
    const uint row = index / Columns;
    const uint column = index % Columns;
    const uint pixel_x = candidate.block_x * 8 + column;
    const uint pixel_y = candidate.block_y * 8 + row;
    pixels[index] = valid
      ? source[pixel_y * params.opsin_row_stride + pixel_x]
      : NAN;
  }
}

// These forward transforms consume the gathered threadgroup tile directly.
// The standalone DCT kernels use the same matrix order but require a device
// input buffer, which would restore the scratch round trip this path removes.
template <uint N>
__attribute__((always_inline)) inline void AcStrategyForwardSquareDct(
  device const float* opsin_x,
  device const float* opsin_y,
  device const float* opsin_b,
  device const AcStrategyCandidate* candidates,
  device float* coefficients,
  constant AcStrategyBatchParams& params,
  constant const float* basis,
  float scale,
  threadgroup float* pixels,
  threadgroup float* shared_basis,
  uint lane,
  uint simd_width,
  uint simdgroup_index,
  uint3 group_position) {

  constexpr uint kTileSize = 8;
  constexpr uint kTilesPerDimension = N / kTileSize;
  const uint threadgroup_stride =
    kTilesPerDimension * simd_width;

  GatherAcStrategyPixels<N, N>(
    opsin_x,
    opsin_y,
    opsin_b,
    candidates,
    params,
    pixels,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
  for (uint index = simdgroup_index * simd_width + lane;
       index < N * N;
       index += threadgroup_stride) {
    shared_basis[index] = basis[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

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
        pixels,
        N,
        ulong2(column_tile * kTileSize,
               inner_tile * kTileSize));
      simdgroup_multiply_accumulate(accumulator, c, a, accumulator);
    }
    intermediate[column_tile] = accumulator;
  }

  const ulong output_base =
    static_cast<ulong>(group_position.x) * N * N;
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
    simdgroup_store(
      accumulator,
      coefficients + output_base,
      N,
      ulong2(simdgroup_index * kTileSize,
             column_tile * kTileSize),
      true);
  }
}

template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void AcStrategyForwardRectangularDct(
  device const float* opsin_x,
  device const float* opsin_y,
  device const float* opsin_b,
  device const AcStrategyCandidate* candidates,
  device float* coefficients,
  constant AcStrategyBatchParams& params,
  constant const float* vertical_basis,
  constant const float* horizontal_basis,
  float scale,
  threadgroup float* pixels,
  threadgroup float* shared_vertical_basis,
  threadgroup float* shared_horizontal_basis,
  uint lane,
  uint simd_width,
  uint simdgroup_index,
  uint3 group_position) {

  constexpr uint kTileSize = 8;
  constexpr uint kRowTiles = Rows / kTileSize;
  constexpr uint kColumnTiles = Columns / kTileSize;
  const uint threadgroup_stride = kRowTiles * simd_width;

  GatherAcStrategyPixels<Rows, Columns>(
    opsin_x,
    opsin_y,
    opsin_b,
    candidates,
    params,
    pixels,
    lane,
    simd_width,
    simdgroup_index,
    group_position);
  for (uint index = simdgroup_index * simd_width + lane;
       index < Rows * Rows;
       index += threadgroup_stride) {
    shared_vertical_basis[index] = vertical_basis[index];
  }
  for (uint index = simdgroup_index * simd_width + lane;
       index < Columns * Columns;
       index += threadgroup_stride) {
    shared_horizontal_basis[index] = horizontal_basis[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

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
        shared_vertical_basis,
        Rows,
        ulong2(inner_tile * kTileSize,
               simdgroup_index * kTileSize));
      simdgroup_load(
        a,
        pixels,
        Columns,
        ulong2(column_tile * kTileSize,
               inner_tile * kTileSize));
      simdgroup_multiply_accumulate(accumulator, c, a, accumulator);
    }
    intermediate[column_tile] = accumulator;
  }

  const ulong output_base =
    static_cast<ulong>(group_position.x) * Rows * Columns;
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
        shared_horizontal_basis,
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
        coefficients + output_base,
        Columns,
        ulong2(column_tile * kTileSize,
               simdgroup_index * kTileSize));
    } else {
      simdgroup_store(
        accumulator,
        coefficients + output_base,
        Rows,
        ulong2(simdgroup_index * kTileSize,
               column_tile * kTileSize),
        true);
    }
  }
}

#define GJXL_AC_SQUARE_FORWARD_KERNEL(name, size, basis, scale)             \
kernel void name(                                                           \
  device const float* opsin_x [[buffer(0)]],                                \
  device const float* opsin_y [[buffer(1)]],                                \
  device const float* opsin_b [[buffer(2)]],                                \
  device const AcStrategyCandidate* candidates [[buffer(3)]],               \
  device float* coefficients [[buffer(4)]],                                 \
  constant AcStrategyBatchParams& params [[buffer(5)]],                     \
  uint lane [[thread_index_in_simdgroup]],                                   \
  uint simd_width [[threads_per_simdgroup]],                                 \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  threadgroup float pixels[size * size];                                    \
  threadgroup float shared_basis[size * size];                              \
  AcStrategyForwardSquareDct<size>(                                         \
    opsin_x, opsin_y, opsin_b, candidates, coefficients, params, basis,     \
    scale, pixels, shared_basis, lane, simd_width, simdgroup_index,          \
    group_position);                                                        \
}

#define GJXL_AC_RECTANGULAR_FORWARD_KERNEL(                                 \
  name, rows, columns, vertical_basis, horizontal_basis, scale)             \
kernel void name(                                                           \
  device const float* opsin_x [[buffer(0)]],                                \
  device const float* opsin_y [[buffer(1)]],                                \
  device const float* opsin_b [[buffer(2)]],                                \
  device const AcStrategyCandidate* candidates [[buffer(3)]],               \
  device float* coefficients [[buffer(4)]],                                 \
  constant AcStrategyBatchParams& params [[buffer(5)]],                     \
  uint lane [[thread_index_in_simdgroup]],                                   \
  uint simd_width [[threads_per_simdgroup]],                                 \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  threadgroup float pixels[rows * columns];                                 \
  threadgroup float shared_vertical_basis[rows * rows];                     \
  threadgroup float shared_horizontal_basis[columns * columns];             \
  AcStrategyForwardRectangularDct<rows, columns>(                            \
    opsin_x, opsin_y, opsin_b, candidates, coefficients, params,            \
    vertical_basis, horizontal_basis, scale, pixels,                        \
    shared_vertical_basis, shared_horizontal_basis, lane, simd_width,        \
    simdgroup_index, group_position);                                       \
}

GJXL_AC_SQUARE_FORWARD_KERNEL(
  gjxl_ac_strategy_dct8_forward_fused,
  8,
  kOrthonormalDct8,
  kForwardDct8Scale)
GJXL_AC_SQUARE_FORWARD_KERNEL(
  gjxl_ac_strategy_dct16_forward_fused,
  16,
  kOrthonormalDct16,
  kForwardDct16Scale)
GJXL_AC_SQUARE_FORWARD_KERNEL(
  gjxl_ac_strategy_dct32_forward_fused,
  32,
  kOrthonormalDct32,
  kForwardDct32Scale)
GJXL_AC_RECTANGULAR_FORWARD_KERNEL(
  gjxl_ac_strategy_dct16x8_forward_fused,
  16,
  8,
  kOrthonormalDct16,
  kOrthonormalDct8,
  kForwardDct16x8Scale)
GJXL_AC_RECTANGULAR_FORWARD_KERNEL(
  gjxl_ac_strategy_dct8x16_forward_fused,
  8,
  16,
  kOrthonormalDct8,
  kOrthonormalDct16,
  kForwardDct16x8Scale)
GJXL_AC_RECTANGULAR_FORWARD_KERNEL(
  gjxl_ac_strategy_dct32x16_forward_fused,
  32,
  16,
  kOrthonormalDct32,
  kOrthonormalDct16,
  kForwardDct32x16Scale)
GJXL_AC_RECTANGULAR_FORWARD_KERNEL(
  gjxl_ac_strategy_dct16x32_forward_fused,
  16,
  32,
  kOrthonormalDct16,
  kOrthonormalDct32,
  kForwardDct32x16Scale)

#undef GJXL_AC_SQUARE_FORWARD_KERNEL
#undef GJXL_AC_RECTANGULAR_FORWARD_KERNEL

template <typename ResidualPointer>
__attribute__((always_inline)) inline void ComputeAcStrategyResidual(
  device const float* coefficients,
  device const float* matrices,
  device const AcStrategyCandidate* candidates,
  device const float* quant_field,
  ResidualPointer residual_coefficients,
  device ChannelRate* channel_rates,
  constant AcStrategyBatchParams& params,
  threadgroup float* magnitude_reduction,
  threadgroup uint* nonzero_reduction,
  uint residual_base,
  uint tid,
  uint3 group_position) {

  const uint transform_index = group_position.x;
  const uint candidate_index = transform_index / 3;
  const uint channel = transform_index % 3;
  const uint base = transform_index * params.coefficient_count;
  const uint y_base =
    (candidate_index * 3 + 1) * params.coefficient_count;
  const uint matrix_base = channel * params.coefficient_count;
  const uint inverse_matrix_base =
    (3 + channel) * params.coefficient_count;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const float quant_norm = ComputeQuantNorm(quant_field, candidate, params);
  const float cfl_factor =
    channel == 0 ? candidate.cfl_x :
    channel == 2 ? candidate.cfl_b : 0.0f;

  const float decorrelated =
    coefficients[base + tid] - coefficients[y_base + tid] * cfl_factor;
  const float scaled =
    decorrelated * matrices[inverse_matrix_base + tid] * quant_norm;
  const float rounded = RoundAwayFromZero(scaled);
  residual_coefficients[residual_base + tid] =
    matrices[matrix_base + tid] * (scaled - rounded);

  magnitude_reduction[tid] = sqrt(abs(rounded));
  nonzero_reduction[tid] = rounded != 0.0f ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.coefficient_count / 2;
       stride != 0;
       stride /= 2) {
    if (tid < stride) {
      magnitude_reduction[tid] += magnitude_reduction[tid + stride];
      nonzero_reduction[tid] += nonzero_reduction[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    channel_rates[transform_index] = {
      magnitude_reduction[0],
      nonzero_reduction[0],
    };
  }
}

kernel void gjxl_ac_strategy_residual(
  device const float* coefficients [[buffer(0)]],
  device const float* matrices [[buffer(1)]],
  device const AcStrategyCandidate* candidates [[buffer(2)]],
  device const float* quant_field [[buffer(3)]],
  device float* residual_coefficients [[buffer(4)]],
  device ChannelRate* channel_rates [[buffer(5)]],
  constant AcStrategyBatchParams& params [[buffer(6)]],
  threadgroup float* magnitude_reduction [[threadgroup(0)]],
  threadgroup uint* nonzero_reduction [[threadgroup(1)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 group_position [[threadgroup_position_in_grid]]) {

  ComputeAcStrategyResidual(
    coefficients,
    matrices,
    candidates,
    quant_field,
    residual_coefficients,
    channel_rates,
    params,
    magnitude_reduction,
    nonzero_reduction,
    group_position.x * params.coefficient_count,
    tid,
    group_position);
}

// The fused inverse consumes the residual coefficients before they leave
// threadgroup memory. All coefficient threads participate in basis staging;
// only the row-owning SIMD groups execute the matrix tiles afterward.
template <uint N>
__attribute__((always_inline)) inline void AcStrategyInverseSquareDct(
  threadgroup const float* coefficients,
  device float* pixels,
  constant const float* basis,
  float scale,
  threadgroup float* shared_basis,
  uint tid,
  uint simdgroup_index,
  uint3 group_position) {

  constexpr uint kTileSize = 8;
  constexpr uint kTilesPerDimension = N / kTileSize;
  for (uint index = tid; index < N * N; index += N * N) {
    shared_basis[index] = basis[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_index >= kTilesPerDimension) return;

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
        coefficients,
        N,
        ulong2(inner_tile * kTileSize,
               column_tile * kTileSize),
        true);
      simdgroup_multiply_accumulate(accumulator, ct, at, accumulator);
    }
    intermediate[column_tile] = accumulator;
  }

  const ulong output_base =
    static_cast<ulong>(group_position.x) * N * N;
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
      pixels + output_base,
      N,
      ulong2(column_tile * kTileSize,
             simdgroup_index * kTileSize));
  }
}

template <uint Rows, uint Columns>
__attribute__((always_inline)) inline void AcStrategyInverseRectangularDct(
  threadgroup const float* coefficients,
  device float* pixels,
  constant const float* vertical_basis,
  constant const float* horizontal_basis,
  float scale,
  threadgroup float* shared_vertical_basis,
  threadgroup float* shared_horizontal_basis,
  uint tid,
  uint simdgroup_index,
  uint3 group_position) {

  constexpr uint kTileSize = 8;
  constexpr uint kRowTiles = Rows / kTileSize;
  constexpr uint kColumnTiles = Columns / kTileSize;
  constexpr uint kCoefficientCount = Rows * Columns;
  for (uint index = tid;
       index < Rows * Rows;
       index += kCoefficientCount) {
    shared_vertical_basis[index] = vertical_basis[index];
  }
  for (uint index = tid;
       index < Columns * Columns;
       index += kCoefficientCount) {
    shared_horizontal_basis[index] = horizontal_basis[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_index >= kRowTiles) return;

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
      simdgroup_float8x8 coefficient_tile;
      simdgroup_load(
        ct,
        shared_vertical_basis,
        Rows,
        ulong2(simdgroup_index * kTileSize,
               inner_tile * kTileSize),
        true);
      if (Rows < Columns) {
        simdgroup_load(
          coefficient_tile,
          coefficients,
          Columns,
          ulong2(column_tile * kTileSize,
                 inner_tile * kTileSize));
      } else {
        simdgroup_load(
          coefficient_tile,
          coefficients,
          Rows,
          ulong2(inner_tile * kTileSize,
                 column_tile * kTileSize),
          true);
      }
      simdgroup_multiply_accumulate(
        accumulator,
        ct,
        coefficient_tile,
        accumulator);
    }
    intermediate[column_tile] = accumulator;
  }

  const ulong output_base =
    static_cast<ulong>(group_position.x) * Rows * Columns;
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
        shared_horizontal_basis,
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
      pixels + output_base,
      Columns,
      ulong2(column_tile * kTileSize,
             simdgroup_index * kTileSize));
  }
}

#define GJXL_AC_SQUARE_RESIDUAL_INVERSE_KERNEL(name, size, basis, scale)    \
kernel void name(                                                           \
  device const float* coefficients [[buffer(0)]],                           \
  device const float* matrices [[buffer(1)]],                               \
  device const AcStrategyCandidate* candidates [[buffer(2)]],               \
  device const float* quant_field [[buffer(3)]],                            \
  device float* pixels [[buffer(4)]],                                       \
  device ChannelRate* channel_rates [[buffer(5)]],                          \
  constant AcStrategyBatchParams& params [[buffer(6)]],                     \
  threadgroup float* residual_coefficients [[threadgroup(0)]],              \
  threadgroup float* magnitude_reduction [[threadgroup(1)]],                \
  threadgroup uint* nonzero_reduction [[threadgroup(2)]],                   \
  uint tid [[thread_index_in_threadgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  ComputeAcStrategyResidual(                                                \
    coefficients, matrices, candidates, quant_field, residual_coefficients, \
    channel_rates, params, magnitude_reduction, nonzero_reduction, 0, tid,  \
    group_position);                                                        \
  threadgroup float shared_basis[size * size];                              \
  AcStrategyInverseSquareDct<size>(                                         \
    residual_coefficients, pixels, basis, scale, shared_basis, tid,         \
    simdgroup_index, group_position);                                       \
}

#define GJXL_AC_RECTANGULAR_RESIDUAL_INVERSE_KERNEL(                        \
  name, rows, columns, vertical_basis, horizontal_basis, scale)             \
kernel void name(                                                           \
  device const float* coefficients [[buffer(0)]],                           \
  device const float* matrices [[buffer(1)]],                               \
  device const AcStrategyCandidate* candidates [[buffer(2)]],               \
  device const float* quant_field [[buffer(3)]],                            \
  device float* pixels [[buffer(4)]],                                       \
  device ChannelRate* channel_rates [[buffer(5)]],                          \
  constant AcStrategyBatchParams& params [[buffer(6)]],                     \
  threadgroup float* residual_coefficients [[threadgroup(0)]],              \
  threadgroup float* magnitude_reduction [[threadgroup(1)]],                \
  threadgroup uint* nonzero_reduction [[threadgroup(2)]],                   \
  uint tid [[thread_index_in_threadgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  ComputeAcStrategyResidual(                                                \
    coefficients, matrices, candidates, quant_field, residual_coefficients, \
    channel_rates, params, magnitude_reduction, nonzero_reduction, 0, tid,  \
    group_position);                                                        \
  threadgroup float shared_vertical_basis[rows * rows];                     \
  threadgroup float shared_horizontal_basis[columns * columns];             \
  AcStrategyInverseRectangularDct<rows, columns>(                           \
    residual_coefficients, pixels, vertical_basis, horizontal_basis, scale, \
    shared_vertical_basis, shared_horizontal_basis, tid, simdgroup_index,   \
    group_position);                                                        \
}

GJXL_AC_SQUARE_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct8_residual_inverse_fused,
  8,
  kOrthonormalDct8,
  kInverseDct8Scale)
GJXL_AC_SQUARE_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct16_residual_inverse_fused,
  16,
  kOrthonormalDct16,
  kInverseDct16Scale)
GJXL_AC_SQUARE_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct32_residual_inverse_fused,
  32,
  kOrthonormalDct32,
  kInverseDct32Scale)
GJXL_AC_RECTANGULAR_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct16x8_residual_inverse_fused,
  16,
  8,
  kOrthonormalDct16,
  kOrthonormalDct8,
  kInverseDct16x8Scale)
GJXL_AC_RECTANGULAR_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct8x16_residual_inverse_fused,
  8,
  16,
  kOrthonormalDct8,
  kOrthonormalDct16,
  kInverseDct16x8Scale)
GJXL_AC_RECTANGULAR_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct32x16_residual_inverse_fused,
  32,
  16,
  kOrthonormalDct32,
  kOrthonormalDct16,
  kInverseDct32x16Scale)
GJXL_AC_RECTANGULAR_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct16x32_residual_inverse_fused,
  16,
  32,
  kOrthonormalDct16,
  kOrthonormalDct32,
  kInverseDct32x16Scale)

#undef GJXL_AC_SQUARE_RESIDUAL_INVERSE_KERNEL
#undef GJXL_AC_RECTANGULAR_RESIDUAL_INVERSE_KERNEL

kernel void gjxl_ac_strategy_cost(
  device const float* residual_pixels [[buffer(0)]],
  device const float* pixel_mask [[buffer(1)]],
  device const AcStrategyCandidate* candidates [[buffer(2)]],
  device const ChannelRate* channel_rates [[buffer(3)]],
  device float* costs [[buffer(4)]],
  device const float* quant_field [[buffer(5)]],
  constant AcStrategyBatchParams& params [[buffer(6)]],
  threadgroup float* loss_reduction [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 group_position [[threadgroup_position_in_grid]]) {

  const uint candidate_index = group_position.x;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const uint row = tid / params.transform_width;
  const uint column = tid % params.transform_width;
  const bool candidate_fits =
    candidate.block_x <=
      (params.pixel_width - params.transform_width) / 8 &&
    candidate.block_y <=
      (params.pixel_height - params.transform_height) / 8;
  if (!candidate_fits) {
    if (tid == 0) {
      costs[candidate_index] = NAN;
    }
    return;
  }
  const uint pixel_x = candidate.block_x * 8 + column;
  const uint pixel_y = candidate.block_y * 8 + row;
  const uint mask_index =
    pixel_y * params.pixel_mask_row_stride + pixel_x;
  const float mask = pixel_mask[mask_index];

  float entropy = 0.0f;
  float loss = 0.0f;

  for (uint channel = 0; channel < 3; ++channel) {
    const uint transform_index = candidate_index * 3 + channel;
    const uint base = transform_index * params.coefficient_count;
    float weighted =
      (mask + kMaskOffset[channel]) * residual_pixels[base + tid];
    weighted *= weighted;
    weighted *= weighted;
    weighted *= weighted;
    loss_reduction[channel * params.coefficient_count + tid] =
      isfinite(mask) && mask > 0.0f ? weighted : NAN;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.coefficient_count / 2;
       stride != 0;
       stride /= 2) {
    for (uint channel = 0; channel < 3; ++channel) {
      if (tid < stride) {
        const uint base = channel * params.coefficient_count;
        loss_reduction[base + tid] += loss_reduction[base + tid + stride];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    for (uint channel = 0; channel < 3; ++channel) {
      const uint transform_index = candidate_index * 3 + channel;
      const ChannelRate rate = channel_rates[transform_index];
      entropy += params.cost_delta * rate.magnitude;
      const uint nonzero_bits = CeilLog2Nonzero(rate.nonzero_count + 1) + 1;
      entropy += params.zeros_multiplier * float(
        CeilLog2Nonzero(nonzero_bits + 17) + nonzero_bits);
      loss += loss_reduction[channel * params.coefficient_count] *
        kChannelMultiplier[channel];

      if (channel == 0 && params.covered_block_count >= 2) {
        const float weight = 1.0f + min(
          3.0f,
          float(params.covered_block_count) / 8.0f);
        entropy *= weight;
        loss *= weight;
      }
    }

    const float quant_norm = ComputeQuantNorm(
      quant_field, candidate, params);
    const float normalized_loss = loss / float(params.coefficient_count);
    const float loss_cost =
      powr(normalized_loss, 0.125f) * float(params.coefficient_count) /
      quant_norm;
    const float result =
      entropy * candidate.entropy_multiplier +
      params.info_loss_multiplier * loss_cost;
    costs[candidate_index] =
      isfinite(result) && result >= 0.0f ? result : NAN;
  }
}
