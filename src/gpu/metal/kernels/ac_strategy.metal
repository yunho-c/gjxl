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
  uint quant_norm_source;
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

constant uint kQuantNormFromCandidate = 0u;
constant uint kQuantNormFromForwardPass = 2u;

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

  if (params.quant_norm_source == kQuantNormFromCandidate) {
    return candidate.quant_norm;
  }
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

// Candidate dimensions are fixed at pipeline creation. Split fallbacks keep
// their runtime-dimension quantization norm below.
template <uint Rows, uint Columns>
inline float AcStrategyStaticQuantNorm(device const float* quant_field,
  AcStrategyCandidate candidate, constant AcStrategyBatchParams& params) {

  if (params.quant_norm_source == kQuantNormFromCandidate) {
    return candidate.quant_norm;
  }
  if ((Rows * Columns / 64) == 1u) {
    return quant_field[candidate.block_y * params.quant_field_row_stride +
                       candidate.block_x];
  }
  if ((Rows * Columns / 64) == 2u) {
    const float first =
      quant_field[candidate.block_y * params.quant_field_row_stride +
                  candidate.block_x];
    const uint second_x = candidate.block_x + ((Columns / 8) == 2u ? 1u : 0u);
    const uint second_y = candidate.block_y + ((Rows / 8) == 2u ? 1u : 0u);
    const float second =
      quant_field[second_y * params.quant_field_row_stride + second_x];
    return max(first, second);
  }
  float sum = 0.0f;
  for (uint dy = 0; dy < (Rows / 8); ++dy) {
    for (uint dx = 0; dx < (Columns / 8); ++dx) {
      float value =
        quant_field[(candidate.block_y + dy) * params.quant_field_row_stride +
                    candidate.block_x + dx];
      value *= value;
      value *= value;
      value *= value;
      sum += value * value;
    }
  }
  sum /= float((Rows * Columns / 64));
  return FastPow2(FastLog2(sum) * (1.0f / 16.0f));
}

inline float AcStrategyQuantNorm(
  device const float* quant_field,
  device const float* precomputed_quant_norm,
  AcStrategyCandidate candidate,
  uint candidate_index,
  constant AcStrategyBatchParams& params) {

  return params.quant_norm_source == kQuantNormFromForwardPass
    ? precomputed_quant_norm[candidate_index]
    : ComputeQuantNorm(quant_field, candidate, params);
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
// With StageBasis=false, the caller stages one immutable shared basis before
// this helper. Its unconditional barrier publishes both the basis and pixels.
template <uint N, bool LocalOutput = false, bool StageBasis = true,
          bool InPlace = false, typename CoefficientPointer>
__attribute__((always_inline)) inline void AcStrategyForwardSquareDct(
  device const float* opsin_x,
  device const float* opsin_y,
  device const float* opsin_b,
  device const AcStrategyCandidate* candidates,
  CoefficientPointer coefficients,
  device const float* quant_field,
  device float* precomputed_quant_norm,
  constant AcStrategyBatchParams& params,
  constant const float* basis,
  float scale,
  threadgroup float* pixels,
  threadgroup float* shared_basis,
  uint lane,
  uint simd_width,
  uint simdgroup_index,
  uint3 group_position) {
  static_assert(!InPlace || LocalOutput, "In-place DCT requires local output");

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
  if (params.quant_norm_source == kQuantNormFromForwardPass &&
      group_position.x % 3u == 0u && simdgroup_index == 0u && lane == 0u) {
    const uint candidate_index = group_position.x / 3u;
    precomputed_quant_norm[candidate_index] = InPlace
      ? AcStrategyStaticQuantNorm<N, N>(quant_field, candidates[candidate_index], params)
      : ComputeQuantNorm(quant_field, candidates[candidate_index], params);
  }
  if (StageBasis) {
    for (uint index = simdgroup_index * simd_width + lane;
         index < N * N;
         index += threadgroup_stride) {
      shared_basis[index] = basis[index];
    }
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

  // Every input tile must be consumed before an aliased output overwrites it.
  if constexpr (InPlace) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong output_base =
    static_cast<ulong>(LocalOutput ? group_position.x % 3u : group_position.x) * N * N;
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

template <uint Rows, uint Columns, bool LocalOutput = false,
          bool StageBasis = true, bool InPlace = false,
          typename CoefficientPointer>
__attribute__((always_inline)) inline void AcStrategyForwardRectangularDct(
  device const float* opsin_x,
  device const float* opsin_y,
  device const float* opsin_b,
  device const AcStrategyCandidate* candidates,
  CoefficientPointer coefficients,
  device const float* quant_field,
  device float* precomputed_quant_norm,
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
  static_assert(!InPlace || LocalOutput, "In-place DCT requires local output");

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
  if (params.quant_norm_source == kQuantNormFromForwardPass &&
      group_position.x % 3u == 0u && simdgroup_index == 0u && lane == 0u) {
    const uint candidate_index = group_position.x / 3u;
    precomputed_quant_norm[candidate_index] = InPlace
      ? AcStrategyStaticQuantNorm<Rows, Columns>(quant_field, candidates[candidate_index], params)
      : ComputeQuantNorm(quant_field, candidates[candidate_index], params);
  }
  if (StageBasis) {
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

  // Every input tile must be consumed before an aliased output overwrites it.
  if constexpr (InPlace) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong output_base =
    static_cast<ulong>(LocalOutput ? group_position.x % 3u : group_position.x) *
      Rows * Columns;
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
  device const float* quant_field [[buffer(5)]],                            \
  device float* precomputed_quant_norm [[buffer(6)]],                       \
  constant AcStrategyBatchParams& params [[buffer(7)]],                     \
  uint lane [[thread_index_in_simdgroup]],                                   \
  uint simd_width [[threads_per_simdgroup]],                                 \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  threadgroup float pixels[size * size];                                    \
  threadgroup float shared_basis[size * size];                              \
  AcStrategyForwardSquareDct<size>(                                         \
    opsin_x, opsin_y, opsin_b, candidates, coefficients, quant_field,       \
    precomputed_quant_norm, params, basis, scale, pixels, shared_basis,     \
    lane, simd_width, simdgroup_index, group_position);                     \
}

#define GJXL_AC_RECTANGULAR_FORWARD_KERNEL(                                 \
  name, rows, columns, vertical_basis, horizontal_basis, scale)             \
kernel void name(                                                           \
  device const float* opsin_x [[buffer(0)]],                                \
  device const float* opsin_y [[buffer(1)]],                                \
  device const float* opsin_b [[buffer(2)]],                                \
  device const AcStrategyCandidate* candidates [[buffer(3)]],               \
  device float* coefficients [[buffer(4)]],                                 \
  device const float* quant_field [[buffer(5)]],                            \
  device float* precomputed_quant_norm [[buffer(6)]],                       \
  constant AcStrategyBatchParams& params [[buffer(7)]],                     \
  uint lane [[thread_index_in_simdgroup]],                                   \
  uint simd_width [[threads_per_simdgroup]],                                 \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  threadgroup float pixels[rows * columns];                                 \
  threadgroup float shared_vertical_basis[rows * rows];                     \
  threadgroup float shared_horizontal_basis[columns * columns];             \
  AcStrategyForwardRectangularDct<rows, columns>(                            \
    opsin_x, opsin_y, opsin_b, candidates, coefficients, quant_field,       \
    precomputed_quant_norm, params, vertical_basis, horizontal_basis,       \
    scale, pixels,                                                          \
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

// X/Y/B share one forward launch and one immutable DCT basis. Each basis
// element has one writer; the helper's existing unconditional barrier publishes
// both the basis and channel-local pixels. Device coefficient layout and the
// separate residual/inverse/loss dispatch stay unchanged.
#define GJXL_AC_GROUPED_SQUARE_FORWARD_KERNEL(name, size, basis, scale)     \
kernel void name(                                                           \
  device const float* opsin_x [[buffer(0)]],                                \
  device const float* opsin_y [[buffer(1)]],                                \
  device const float* opsin_b [[buffer(2)]],                                \
  device const AcStrategyCandidate* candidates [[buffer(3)]],               \
  device float* coefficients [[buffer(4)]],                                 \
  device const float* quant_field [[buffer(5)]],                            \
  device float* precomputed_quant_norm [[buffer(6)]],                       \
  constant AcStrategyBatchParams& params [[buffer(7)]],                     \
  uint tid [[thread_index_in_threadgroup]],                                 \
  uint lane [[thread_index_in_simdgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                  \
  uint candidate_index [[threadgroup_position_in_grid]]) {                  \
  constexpr uint Workers = size / 8 * 32;                                   \
  const uint channel = tid / Workers;                                       \
  const uint local_simdgroup = simdgroup_index % (Workers / 32);            \
  threadgroup float pixels[3 * size * size];                                \
  threadgroup float staged_basis[size * size];                              \
  for (uint i = tid; i < size * size; i += 3 * Workers) {                   \
    staged_basis[i] = basis[i];                                             \
  }                                                                         \
  AcStrategyForwardSquareDct<size, false, false>(                           \
    opsin_x, opsin_y, opsin_b, candidates, coefficients, quant_field,       \
    precomputed_quant_norm, params, basis, scale,                           \
    pixels + channel * size * size,                                         \
    staged_basis,                                                           \
    lane, 32, local_simdgroup, uint3(candidate_index * 3 + channel, 0, 0)); \
}

#define GJXL_AC_GROUPED_RECTANGULAR_FORWARD_KERNEL(                         \
  name, rows, columns, vertical_basis, horizontal_basis, scale)             \
kernel void name(                                                           \
  device const float* opsin_x [[buffer(0)]],                                \
  device const float* opsin_y [[buffer(1)]],                                \
  device const float* opsin_b [[buffer(2)]],                                \
  device const AcStrategyCandidate* candidates [[buffer(3)]],               \
  device float* coefficients [[buffer(4)]],                                 \
  device const float* quant_field [[buffer(5)]],                            \
  device float* precomputed_quant_norm [[buffer(6)]],                       \
  constant AcStrategyBatchParams& params [[buffer(7)]],                     \
  uint tid [[thread_index_in_threadgroup]],                                 \
  uint lane [[thread_index_in_simdgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                  \
  uint candidate_index [[threadgroup_position_in_grid]]) {                  \
  constexpr uint Workers = rows / 8 * 32;                                   \
  const uint channel = tid / Workers;                                       \
  const uint local_simdgroup = simdgroup_index % (Workers / 32);            \
  threadgroup float pixels[3 * rows * columns];                             \
  threadgroup float vertical[rows * rows];                                  \
  threadgroup float horizontal[columns * columns];                          \
  for (uint i = tid; i < rows * rows; i += 3 * Workers) {                   \
    vertical[i] = vertical_basis[i];                                        \
  }                                                                         \
  for (uint i = tid; i < columns * columns; i += 3 * Workers) {             \
    horizontal[i] = horizontal_basis[i];                                    \
  }                                                                         \
  AcStrategyForwardRectangularDct<rows, columns, false, false>(             \
    opsin_x, opsin_y, opsin_b, candidates, coefficients, quant_field,       \
    precomputed_quant_norm, params, vertical_basis, horizontal_basis,       \
    scale, pixels + channel * rows * columns,                               \
    vertical,                                                               \
    horizontal,                                                             \
    lane, 32, local_simdgroup, uint3(candidate_index * 3 + channel, 0, 0)); \
}

GJXL_AC_GROUPED_SQUARE_FORWARD_KERNEL(
  gjxl_ac_strategy_dct32_forward_grouped,
  32, kOrthonormalDct32, kForwardDct32Scale)
GJXL_AC_GROUPED_RECTANGULAR_FORWARD_KERNEL(
  gjxl_ac_strategy_dct32x16_forward_grouped,
  32, 16, kOrthonormalDct32, kOrthonormalDct16, kForwardDct32x16Scale)
GJXL_AC_GROUPED_RECTANGULAR_FORWARD_KERNEL(
  gjxl_ac_strategy_dct16x32_forward_grouped,
  16, 32, kOrthonormalDct16, kOrthonormalDct32, kForwardDct32x16Scale)

#undef GJXL_AC_GROUPED_SQUARE_FORWARD_KERNEL
#undef GJXL_AC_GROUPED_RECTANGULAR_FORWARD_KERNEL

template <typename ResidualPointer>
__attribute__((always_inline)) inline void ComputeAcStrategyResidual(
  device const float* coefficients,
  device const float* matrices,
  device const AcStrategyCandidate* candidates,
  device const float* quant_field,
  device const float* precomputed_quant_norm,
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
  const float quant_norm = AcStrategyQuantNorm(
    quant_field, precomputed_quant_norm, candidate, candidate_index, params);
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

// SimdTail callers require a 32-lane SIMD width and channel-local tid.
template <uint CoefficientCount, uint WorkerCount, bool LocalInput = false,
          bool SimdTail = false, typename CoefficientPointer,
          typename ResidualPointer>
__attribute__((always_inline)) inline void ComputeAcStrategyResidualCompact(
  CoefficientPointer coefficients,
  device const float* matrices,
  device const AcStrategyCandidate* candidates,
  device const float* quant_field,
  device const float* precomputed_quant_norm,
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
  const uint base = (LocalInput ? channel : transform_index) * params.coefficient_count;
  const uint y_base =
    (LocalInput ? 1u : candidate_index * 3 + 1) * params.coefficient_count;
  const uint matrix_base = channel * params.coefficient_count;
  const uint inverse_matrix_base =
    (3 + channel) * params.coefficient_count;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const float quant_norm = AcStrategyQuantNorm(
    quant_field, precomputed_quant_norm, candidate, candidate_index, params);
  const float cfl_factor =
    channel == 0 ? candidate.cfl_x :
    channel == 2 ? candidate.cfl_b : 0.0f;

  for (uint coefficient = tid;
       coefficient < CoefficientCount;
       coefficient += WorkerCount) {
    const float decorrelated = coefficients[base + coefficient] -
      coefficients[y_base + coefficient] * cfl_factor;
    const float scaled = decorrelated *
      matrices[inverse_matrix_base + coefficient] * quant_norm;
    const float rounded = RoundAwayFromZero(scaled);
    residual_coefficients[residual_base + coefficient] =
      matrices[matrix_base + coefficient] * (scaled - rounded);

    magnitude_reduction[coefficient] = sqrt(abs(rounded));
    nonzero_reduction[coefficient] = rounded != 0.0f ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = CoefficientCount / 2;
       stride != 0 && (!SimdTail || stride >= 32); stride /= 2) {
    for (uint index = tid; index < stride; index += WorkerCount) {
      magnitude_reduction[index] += magnitude_reduction[index + stride];
      nonzero_reduction[index] += nonzero_reduction[index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Only the first SIMD group of each channel runs the tail. The shared
  // stride-32 step above publishes its 32 inputs; explicit shuffles preserve
  // the original 16/8/4/2/1 tree. Every source lane stays active.
  if (SimdTail) {
    if (tid < 32) {
      float magnitude = magnitude_reduction[tid];
      uint nonzero = nonzero_reduction[tid];
      for (uint delta = 16; delta != 0; delta /= 2) {
        magnitude += simd_shuffle_down(magnitude, delta);
        nonzero += simd_shuffle_down(nonzero, delta);
      }
      if (tid == 0) channel_rates[transform_index] = {magnitude, nonzero};
    }
  } else if (tid == 0) {
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
  device const float* precomputed_quant_norm [[buffer(7)]],
  threadgroup float* magnitude_reduction [[threadgroup(0)]],
  threadgroup uint* nonzero_reduction [[threadgroup(1)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 group_position [[threadgroup_position_in_grid]]) {

  ComputeAcStrategyResidual(
    coefficients,
    matrices,
    candidates,
    quant_field,
    precomputed_quant_norm,
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
// threadgroup memory. StageBasis=false reuses an already published basis.
// The unconditional barrier also completes residual/rate production before
// inverse pixels may overwrite magnitude scratch. Only the row-owning SIMD
// groups execute the matrix tiles afterward.
template <uint N, uint ThreadgroupWidth, bool StageBasis = true,
          bool InPlace = false, typename PixelPointer>
__attribute__((always_inline)) inline void AcStrategyInverseSquareDct(
  threadgroup const float* coefficients,
  PixelPointer pixels,
  constant const float* basis,
  float scale,
  threadgroup float* shared_basis,
  uint tid,
  uint simdgroup_index,
  uint3 group_position) {
  static_assert(!InPlace || ThreadgroupWidth == N / 8 * 32,
                "Every in-place inverse SIMD group must own a row tile");

  constexpr uint kTileSize = 8;
  constexpr uint kTilesPerDimension = N / kTileSize;
  if (StageBasis) {
    for (uint index = tid; index < N * N; index += ThreadgroupWidth) {
      shared_basis[index] = basis[index];
    }
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

  // Every input tile must be consumed before an aliased output overwrites it.
  if constexpr (InPlace) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
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

template <uint Rows, uint Columns, uint ThreadgroupWidth,
          bool StageBasis = true, bool InPlace = false, typename PixelPointer>
__attribute__((always_inline)) inline void AcStrategyInverseRectangularDct(
  threadgroup const float* coefficients,
  PixelPointer pixels,
  constant const float* vertical_basis,
  constant const float* horizontal_basis,
  float scale,
  threadgroup float* shared_vertical_basis,
  threadgroup float* shared_horizontal_basis,
  uint tid,
  uint simdgroup_index,
  uint3 group_position) {
  static_assert(!InPlace || ThreadgroupWidth == Rows / 8 * 32,
                "Every in-place inverse SIMD group must own a row tile");

  constexpr uint kTileSize = 8;
  constexpr uint kRowTiles = Rows / kTileSize;
  constexpr uint kColumnTiles = Columns / kTileSize;
  if (StageBasis) {
    for (uint index = tid;
         index < Rows * Rows;
         index += ThreadgroupWidth) {
      shared_vertical_basis[index] = vertical_basis[index];
    }
    for (uint index = tid;
         index < Columns * Columns;
         index += ThreadgroupWidth) {
      shared_horizontal_basis[index] = horizontal_basis[index];
    }
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

  // Every input tile must be consumed before an aliased output overwrites it.
  if constexpr (InPlace) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
  device const float* precomputed_quant_norm [[buffer(7)]],                 \
  threadgroup float* residual_coefficients [[threadgroup(0)]],              \
  threadgroup float* magnitude_reduction [[threadgroup(1)]],                \
  threadgroup uint* nonzero_reduction [[threadgroup(2)]],                   \
  uint tid [[thread_index_in_threadgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  ComputeAcStrategyResidual(                                                \
    coefficients, matrices, candidates, quant_field,                       \
    precomputed_quant_norm, residual_coefficients, channel_rates, params,   \
    magnitude_reduction, nonzero_reduction, 0, tid, group_position);        \
  threadgroup float shared_basis[size * size];                              \
  AcStrategyInverseSquareDct<size, size * size>(                            \
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
  device const float* precomputed_quant_norm [[buffer(7)]],                 \
  threadgroup float* residual_coefficients [[threadgroup(0)]],              \
  threadgroup float* magnitude_reduction [[threadgroup(1)]],                \
  threadgroup uint* nonzero_reduction [[threadgroup(2)]],                   \
  uint tid [[thread_index_in_threadgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  ComputeAcStrategyResidual(                                                \
    coefficients, matrices, candidates, quant_field,                       \
    precomputed_quant_norm, residual_coefficients, channel_rates, params,   \
    magnitude_reduction, nonzero_reduction, 0, tid, group_position);        \
  threadgroup float shared_vertical_basis[rows * rows];                     \
  threadgroup float shared_horizontal_basis[columns * columns];             \
  AcStrategyInverseRectangularDct<rows, columns, rows * columns>(           \
    residual_coefficients, pixels, vertical_basis, horizontal_basis, scale, \
    shared_vertical_basis, shared_horizontal_basis, tid, simdgroup_index,   \
    group_position);                                                        \
}

#define GJXL_AC_SQUARE_COMPACT_RESIDUAL_INVERSE_KERNEL(                    \
  name, size, basis, scale, worker_count)                                  \
kernel void name(                                                           \
  device const float* coefficients [[buffer(0)]],                           \
  device const float* matrices [[buffer(1)]],                               \
  device const AcStrategyCandidate* candidates [[buffer(2)]],               \
  device const float* quant_field [[buffer(3)]],                            \
  device float* pixels [[buffer(4)]],                                       \
  device ChannelRate* channel_rates [[buffer(5)]],                          \
  constant AcStrategyBatchParams& params [[buffer(6)]],                     \
  device const float* precomputed_quant_norm [[buffer(7)]],                 \
  threadgroup float* residual_coefficients [[threadgroup(0)]],              \
  threadgroup float* magnitude_reduction [[threadgroup(1)]],                \
  threadgroup uint* nonzero_reduction [[threadgroup(2)]],                   \
  uint tid [[thread_index_in_threadgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  ComputeAcStrategyResidualCompact<size * size, worker_count>(              \
    coefficients, matrices, candidates, quant_field,                       \
    precomputed_quant_norm, residual_coefficients, channel_rates, params,   \
    magnitude_reduction, nonzero_reduction, 0, tid, group_position);        \
  threadgroup float shared_basis[size * size];                              \
  AcStrategyInverseSquareDct<size, worker_count>(                           \
    residual_coefficients, pixels, basis, scale, shared_basis, tid,         \
    simdgroup_index, group_position);                                       \
}

#define GJXL_AC_RECTANGULAR_COMPACT_RESIDUAL_INVERSE_KERNEL(                \
  name, rows, columns, vertical_basis, horizontal_basis, scale,             \
  worker_count)                                                             \
kernel void name(                                                           \
  device const float* coefficients [[buffer(0)]],                           \
  device const float* matrices [[buffer(1)]],                               \
  device const AcStrategyCandidate* candidates [[buffer(2)]],               \
  device const float* quant_field [[buffer(3)]],                            \
  device float* pixels [[buffer(4)]],                                       \
  device ChannelRate* channel_rates [[buffer(5)]],                          \
  constant AcStrategyBatchParams& params [[buffer(6)]],                     \
  device const float* precomputed_quant_norm [[buffer(7)]],                 \
  threadgroup float* residual_coefficients [[threadgroup(0)]],              \
  threadgroup float* magnitude_reduction [[threadgroup(1)]],                \
  threadgroup uint* nonzero_reduction [[threadgroup(2)]],                   \
  uint tid [[thread_index_in_threadgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  ComputeAcStrategyResidualCompact<rows * columns, worker_count>(           \
    coefficients, matrices, candidates, quant_field,                       \
    precomputed_quant_norm, residual_coefficients, channel_rates, params,   \
    magnitude_reduction, nonzero_reduction, 0, tid, group_position);        \
  threadgroup float shared_vertical_basis[rows * rows];                     \
  threadgroup float shared_horizontal_basis[columns * columns];             \
  AcStrategyInverseRectangularDct<rows, columns, worker_count>(             \
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

GJXL_AC_SQUARE_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct8_residual_inverse_compact,
  8,
  kOrthonormalDct8,
  kInverseDct8Scale,
  32)
GJXL_AC_SQUARE_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct16_residual_inverse_compact,
  16,
  kOrthonormalDct16,
  kInverseDct16Scale,
  64)
GJXL_AC_SQUARE_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct32_residual_inverse_compact,
  32,
  kOrthonormalDct32,
  kInverseDct32Scale,
  128)
GJXL_AC_SQUARE_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct32_residual_inverse_tuned,
  32,
  kOrthonormalDct32,
  kInverseDct32Scale,
  512)
GJXL_AC_RECTANGULAR_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct16x8_residual_inverse_compact,
  16,
  8,
  kOrthonormalDct16,
  kOrthonormalDct8,
  kInverseDct16x8Scale,
  64)
GJXL_AC_RECTANGULAR_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct8x16_residual_inverse_compact,
  8,
  16,
  kOrthonormalDct8,
  kOrthonormalDct16,
  kInverseDct16x8Scale,
  32)
GJXL_AC_RECTANGULAR_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct32x16_residual_inverse_compact,
  32,
  16,
  kOrthonormalDct32,
  kOrthonormalDct16,
  kInverseDct32x16Scale,
  128)
GJXL_AC_RECTANGULAR_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct16x32_residual_inverse_compact,
  16,
  32,
  kOrthonormalDct16,
  kOrthonormalDct32,
  kInverseDct32x16Scale,
  64)
GJXL_AC_RECTANGULAR_COMPACT_RESIDUAL_INVERSE_KERNEL(
  gjxl_ac_strategy_dct16x32_residual_inverse_tuned,
  16,
  32,
  kOrthonormalDct16,
  kOrthonormalDct32,
  kInverseDct32x16Scale,
  256)
#undef GJXL_AC_SQUARE_RESIDUAL_INVERSE_KERNEL
#undef GJXL_AC_RECTANGULAR_RESIDUAL_INVERSE_KERNEL
#undef GJXL_AC_SQUARE_COMPACT_RESIDUAL_INVERSE_KERNEL
#undef GJXL_AC_RECTANGULAR_COMPACT_RESIDUAL_INVERSE_KERNEL

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

    const float quant_norm = AcStrategyQuantNorm(
      quant_field, costs, candidate, candidate_index, params);
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

// Reuse the completed magnitude reduction arena for inverse pixels and loss.
// The inverse helper's unconditional barrier publishes the rate before this
// arena is overwritten. Every lane rejoins here, including non-row SIMD groups.
template <uint Count, uint Workers, bool SimdTail = false>
inline void AcStrategyReduceInverseLoss(
  threadgroup float* pixels,
  device const float* pixel_mask,
  device const AcStrategyCandidate* candidates,
  device float* loss_sums,
  constant AcStrategyBatchParams& params,
  uint tid,
  uint transform_index) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint candidate_index = transform_index / 3;
  const uint channel = transform_index % 3;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool fits = candidate.block_x <=
      (params.pixel_width - params.transform_width) / 8 &&
    candidate.block_y <= (params.pixel_height - params.transform_height) / 8;
  for (uint i = tid; i < Count; i += Workers) {
    float mask = NAN;
    if (fits) {
      const uint x = candidate.block_x * 8 + i % params.transform_width;
      const uint y = candidate.block_y * 8 + i / params.transform_width;
      mask = pixel_mask[y * params.pixel_mask_row_stride + x];
    }
    float weighted = (mask + kMaskOffset[channel]) * pixels[i];
    weighted *= weighted;
    weighted *= weighted;
    weighted *= weighted;
    pixels[i] = isfinite(mask) && mask > 0.0f ? weighted : NAN;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Preserve the split cost kernel's binary tree, not a SIMD sum with a
  // different association. Compact groups cover multiple indices per lane.
  for (uint stride = Count / 2;
       stride != 0 && (!SimdTail || stride >= 32); stride /= 2) {
    for (uint i = tid; i < stride; i += Workers) pixels[i] += pixels[i + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // As in the magnitude tail, keep all 32 source lanes active and preserve
  // the original tree feeding lane zero. Callers require a 32-lane SIMD width.
  if (SimdTail) {
    if (tid < 32) {
      float loss = pixels[tid];
      for (uint delta = 16; delta != 0; delta /= 2) {
        loss += simd_shuffle_down(loss, delta);
      }
      if (tid == 0) loss_sums[transform_index] = loss;
    }
  } else if (tid == 0) {
    loss_sums[transform_index] = pixels[0];
  }
}

// Candidate dimensions are known at compile time; preserve the same loss tree.
template <uint Count, uint Workers, uint Columns, bool SimdTail = false>
inline void AcStrategyReduceSpecializedLoss(
  threadgroup float* pixels,
  device const float* pixel_mask,
  device const AcStrategyCandidate* candidates,
  device float* loss_sums,
  constant AcStrategyBatchParams& params,
  uint tid,
  uint transform_index) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint candidate_index = transform_index / 3;
  const uint channel = transform_index % 3;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool fits = candidate.block_x <=
      (params.pixel_width - Columns) / 8 &&
    candidate.block_y <= (params.pixel_height - (Count / Columns)) / 8;
  for (uint i = tid; i < Count; i += Workers) {
    float mask = NAN;
    if (fits) {
      const uint x = candidate.block_x * 8 + i % Columns;
      const uint y = candidate.block_y * 8 + i / Columns;
      mask = pixel_mask[y * params.pixel_mask_row_stride + x];
    }
    float weighted = (mask + kMaskOffset[channel]) * pixels[i];
    weighted *= weighted;
    weighted *= weighted;
    weighted *= weighted;
    pixels[i] = isfinite(mask) && mask > 0.0f ? weighted : NAN;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Preserve the split cost kernel's binary tree, not a SIMD sum with a
  // different association. Compact groups cover multiple indices per lane.
  for (uint stride = Count / 2;
       stride != 0 && (!SimdTail || stride >= 32); stride /= 2) {
    for (uint i = tid; i < stride; i += Workers) pixels[i] += pixels[i + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // As in the magnitude tail, keep all 32 source lanes active and preserve
  // the original tree feeding lane zero. Callers require a 32-lane SIMD width.
  if (SimdTail) {
    if (tid < 32) {
      float loss = pixels[tid];
      for (uint delta = 16; delta != 0; delta /= 2) {
        loss += simd_shuffle_down(loss, delta);
      }
      if (tid == 0) loss_sums[transform_index] = loss;
    }
  } else if (tid == 0) {
    loss_sums[transform_index] = pixels[0];
  }
}

// Register folding must preserve the qualified binary reduction tree.
__attribute__((always_inline)) inline float AcStrategyOrderedAdd(float a, float b) {
#pragma clang fp reassociate(off) contract(off)
  return a + b;
}

template <uint Count, uint Workers, uint Columns, bool SimdTail = false>
inline void AcStrategyReduceRegisterLoss(threadgroup float* pixels,
  device const float* pixel_mask, device const AcStrategyCandidate* candidates,
  device float* loss_sums, constant AcStrategyBatchParams& params, uint tid,
  uint transform_index) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint candidate_index = transform_index / 3;
  const uint channel = transform_index % 3;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool fits =
    candidate.block_x <= (params.pixel_width - Columns) / 8 &&
    candidate.block_y <= (params.pixel_height - (Count / Columns)) / 8;
  constexpr uint Values = Count / Workers;
  float values[Values];
#pragma unroll
  for (uint value = 0; value < Values; ++value) {
    const uint i = tid + value * Workers;
    float mask = NAN;
    if (fits) {
      const uint x = candidate.block_x * 8 + i % Columns;
      const uint y = candidate.block_y * 8 + i / Columns;
      mask = pixel_mask[y * params.pixel_mask_row_stride + x];
    }
    float weighted = (mask + kMaskOffset[channel]) * pixels[i];
    weighted *= weighted;
    weighted *= weighted;
    weighted *= weighted;
    const float rounded_weighted =
      isfinite(mask) && mask > 0.0f ? weighted : NAN;
    values[value] = rounded_weighted;
  }
#pragma unroll
  for (uint stride = Values / 2; stride != 0; stride /= 2) {
#pragma unroll
    for (uint value = 0; value < stride; ++value) {
      const float folded =
        AcStrategyOrderedAdd(values[value], values[value + stride]);
      values[value] = folded;
    }
  }
  pixels[tid] = values[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Preserve the split cost kernel's binary tree, not a SIMD sum with a
  // different association. Compact groups cover multiple indices per lane.
  for (uint stride = Workers / 2; stride != 0 && (!SimdTail || stride >= 32);
    stride /= 2) {
    for (uint i = tid; i < stride; i += Workers)
      pixels[i] += pixels[i + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // As in the magnitude tail, keep all 32 source lanes active and preserve
  // the original tree feeding lane zero. Callers require a 32-lane SIMD width.
  if (SimdTail) {
    if (tid < 32) {
      float loss = pixels[tid];
      for (uint delta = 16; delta != 0; delta /= 2) {
        loss += simd_shuffle_down(loss, delta);
      }
      if (tid == 0)
        loss_sums[transform_index] = loss;
    }
  } else if (tid == 0) {
    loss_sums[transform_index] = pixels[0];
  }
}
#define GJXL_AC_SQUARE_LOSS_KERNEL(                                          \
  name, size, basis, scale, worker_count, magnitude_tail, loss_tail)         \
kernel void name(                                                           \
  device const float* coefficients [[buffer(0)]],                           \
  device const float* matrices [[buffer(1)]],                               \
  device const AcStrategyCandidate* candidates [[buffer(2)]],               \
  device const float* quant_field [[buffer(3)]],                            \
  device float* pixels [[buffer(4)]],                                       \
  device ChannelRate* channel_rates [[buffer(5)]],                          \
  constant AcStrategyBatchParams& params [[buffer(6)]],                     \
  device const float* precomputed_quant_norm [[buffer(7)]],                 \
  device const float* pixel_mask [[buffer(8)]],                            \
  threadgroup float* residual_coefficients [[threadgroup(0)]],              \
  threadgroup float* magnitude_reduction [[threadgroup(1)]],                \
  threadgroup uint* nonzero_reduction [[threadgroup(2)]],                   \
  uint tid [[thread_index_in_threadgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  ComputeAcStrategyResidualCompact<                                          \
    size * size, worker_count, false, magnitude_tail>(                       \
    coefficients, matrices, candidates, quant_field,                       \
    precomputed_quant_norm, residual_coefficients, channel_rates, params,   \
    magnitude_reduction, nonzero_reduction, 0, tid, group_position);        \
  threadgroup float shared_basis[size * size];                              \
  AcStrategyInverseSquareDct<size, worker_count>(                           \
    residual_coefficients, magnitude_reduction, basis, scale, shared_basis, tid,         \
    simdgroup_index, uint3(0));                                             \
  AcStrategyReduceInverseLoss<size * size, worker_count, loss_tail>(         \
    magnitude_reduction, pixel_mask, candidates, pixels, params, tid,      \
    group_position.x);                                                     \
}

#define GJXL_AC_RECTANGULAR_LOSS_KERNEL(                \
  name, rows, columns, vertical_basis, horizontal_basis, scale,             \
  worker_count, magnitude_tail, loss_tail)                                   \
kernel void name(                                                           \
  device const float* coefficients [[buffer(0)]],                           \
  device const float* matrices [[buffer(1)]],                               \
  device const AcStrategyCandidate* candidates [[buffer(2)]],               \
  device const float* quant_field [[buffer(3)]],                            \
  device float* pixels [[buffer(4)]],                                       \
  device ChannelRate* channel_rates [[buffer(5)]],                          \
  constant AcStrategyBatchParams& params [[buffer(6)]],                     \
  device const float* precomputed_quant_norm [[buffer(7)]],                 \
  device const float* pixel_mask [[buffer(8)]],                            \
  threadgroup float* residual_coefficients [[threadgroup(0)]],              \
  threadgroup float* magnitude_reduction [[threadgroup(1)]],                \
  threadgroup uint* nonzero_reduction [[threadgroup(2)]],                   \
  uint tid [[thread_index_in_threadgroup]],                                  \
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],                   \
  uint3 group_position [[threadgroup_position_in_grid]]) {                   \
  ComputeAcStrategyResidualCompact<                                          \
    rows * columns, worker_count, false, magnitude_tail>(                    \
    coefficients, matrices, candidates, quant_field,                       \
    precomputed_quant_norm, residual_coefficients, channel_rates, params,   \
    magnitude_reduction, nonzero_reduction, 0, tid, group_position);        \
  threadgroup float shared_vertical_basis[rows * rows];                     \
  threadgroup float shared_horizontal_basis[columns * columns];             \
  AcStrategyInverseRectangularDct<rows, columns, worker_count>(             \
    residual_coefficients, magnitude_reduction, vertical_basis, horizontal_basis, scale, \
    shared_vertical_basis, shared_horizontal_basis, tid, simdgroup_index,   \
    uint3(0));                                                              \
  AcStrategyReduceInverseLoss<rows * columns, worker_count, loss_tail>(      \
    magnitude_reduction, pixel_mask, candidates, pixels, params, tid,      \
    group_position.x);                                                     \
}

GJXL_AC_SQUARE_LOSS_KERNEL(gjxl_ac_strategy_dct8_residual_inverse_compact_loss,
  8, kOrthonormalDct8, kInverseDct8Scale, 32, true, true)
GJXL_AC_SQUARE_LOSS_KERNEL(gjxl_ac_strategy_dct16_residual_inverse_compact_loss,
  16, kOrthonormalDct16, kInverseDct16Scale, 64, false, false)
GJXL_AC_SQUARE_LOSS_KERNEL(gjxl_ac_strategy_dct32_residual_inverse_tuned_loss,
  32, kOrthonormalDct32, kInverseDct32Scale, 512, true, true)
GJXL_AC_RECTANGULAR_LOSS_KERNEL(gjxl_ac_strategy_dct16x8_residual_inverse_compact_loss,
  16, 8, kOrthonormalDct16, kOrthonormalDct8, kInverseDct16x8Scale, 64,
  false, false)
GJXL_AC_RECTANGULAR_LOSS_KERNEL(gjxl_ac_strategy_dct8x16_residual_inverse_compact_loss,
  8, 16, kOrthonormalDct8, kOrthonormalDct16, kInverseDct16x8Scale, 32,
  false, false)
GJXL_AC_RECTANGULAR_LOSS_KERNEL(gjxl_ac_strategy_dct32x16_residual_inverse_compact_loss,
  32, 16, kOrthonormalDct32, kOrthonormalDct16, kInverseDct32x16Scale, 128,
  true, true)
GJXL_AC_RECTANGULAR_LOSS_KERNEL(gjxl_ac_strategy_dct16x32_residual_inverse_tuned_loss,
  16, 32, kOrthonormalDct16, kOrthonormalDct32, kInverseDct32x16Scale, 256,
  true, true)

#undef GJXL_AC_SQUARE_LOSS_KERNEL
#undef GJXL_AC_RECTANGULAR_LOSS_KERNEL

// Reuse one transform arena and half a magnitude arena. Preserve the
// original FP32 halving tree and publish all Y reads before residual stores.
// This safe addition replaces the earlier volatile FP32 materialization.
// Together with the factored DCT it is qualified as changed arithmetic.
__attribute__((always_inline)) inline float AcStrategyMagnitudeAdd(float a, float b) {
#pragma METAL fp math_mode(safe)
  return a + b;
}

template <uint Count, uint Workers>
__attribute__((always_inline)) inline void AcStrategyHalfMagnitudeResidual(
  threadgroup float* coefficients, threadgroup float* temporary,
  device const float* matrices, device const AcStrategyCandidate* candidates,
  device const float* quant_field, device const float* quant_norms,
  device ChannelRate* rates, constant AcStrategyBatchParams& params,
  uint tid, uint transform_index) {
  constexpr uint Values = Count / Workers;
  static_assert(Values >= 2, "Residual reductions need at least two values per lane");
  const uint channel = transform_index % 3;
  const uint candidate_index = transform_index / 3;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const float quant_norm = AcStrategyQuantNorm(
    quant_field, quant_norms, candidate, candidate_index, params);
  const float cfl_factor = channel == 0 ? candidate.cfl_x :
                           channel == 2 ? candidate.cfl_b : 0.0f;
  float residual[Values];
  float magnitudes[Values];
  threadgroup float* magnitude = temporary + channel * (Count / 2);
  uint nonzero = 0;
#pragma unroll
  for (uint value = 0; value < Values; ++value) {
    const uint coefficient = tid + value * Workers;
    const float decorrelated = coefficients[channel * Count + coefficient] -
      coefficients[Count + coefficient] * cfl_factor;
    const float scaled = decorrelated *
      matrices[(3 + channel) * Count + coefficient] * quant_norm;
    const float rounded = RoundAwayFromZero(scaled);
    residual[value] = matrices[channel * Count + coefficient] * (scaled - rounded);
    const float rounded_magnitude = sqrt(abs(rounded));
    magnitudes[value] = rounded_magnitude;
    nonzero += rounded != 0.0f ? 1u : 0u;
  }
  // Identical first tree level after explicit FP32 magnitude rounding.
  for (uint value = 0; value < Values / 2; ++value) {
    magnitude[tid + value * Workers] = AcStrategyMagnitudeAdd(
      magnitudes[value], magnitudes[value + Values / 2]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint delta = 16; delta != 0; delta /= 2) {
    nonzero += simd_shuffle_down(nonzero, delta);
  }
  threadgroup uint* counts = reinterpret_cast<threadgroup uint*>(magnitude + Count / 4);
  for (uint stride = Count / 4; stride >= 32; stride /= 2) {
    for (uint i = tid; i < stride; i += Workers) {
      magnitude[i] += magnitude[i + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stride == Count / 4) {
      // The upper half of the magnitude array is now dead. Reuse a few cells
      // for integer warp totals; later magnitude levels touch only the lower half.
      if (tid % 32 == 0) counts[tid / 32] = nonzero;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (tid < 32) {
    float sum = magnitude[tid];
    for (uint delta = 16; delta != 0; delta /= 2) {
      sum += simd_shuffle_down(sum, delta);
    }
    if (tid == 0) {
      uint count = 0;
      if constexpr (Count == 64) count = nonzero;
      else for (uint warp = 0; warp < Workers / 32; ++warp) count += counts[warp];
      rates[transform_index] = {sum, count};
    }
  }
#pragma unroll
  for (uint value = 0; value < Values; ++value) {
    coefficients[channel * Count + tid + value * Workers] = residual[value];
  }
  // The inverse helper's unconditional barrier publishes the residuals and
  // finishes all reads of the temporary reduction tile before pixel writes.
}

// FP32 radix-2 DCT-II/III for the five qualified candidate shapes.
// The operation order and constants match the retained qualification.
constant float kAcFactoredDctSqrt2 = 1.41421356237f;

constant float kAcFactoredDctMultipliers4[2] = {
  0.541196100146197f,
  1.3065629648763764f,
};

constant float kAcFactoredDctMultipliers8[4] = {
  0.5097955791041592f,
  0.6013448869350453f,
  0.8999762231364156f,
  2.5629154477415055f,
};

constant float kAcFactoredDctMultipliers16[8] = {
  0.5024192861881557f,
  0.5224986149396889f,
  0.5669440348163577f,
  0.6468217833599901f,
  0.7881546234512502f,
  1.060677685990347f,
  1.7224470982383342f,
  5.101148618689155f,
};

constant float kAcFactoredDctMultipliers32[16] = {
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



template <uint N>
struct AcFactoredDctMultipliers;

template <>
struct AcFactoredDctMultipliers<4> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kAcFactoredDctMultipliers4[index];
  }
};

template <>
struct AcFactoredDctMultipliers<8> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kAcFactoredDctMultipliers8[index];
  }
};

template <>
struct AcFactoredDctMultipliers<16> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kAcFactoredDctMultipliers16[index];
  }
};

template <>
struct AcFactoredDctMultipliers<32> {
  __attribute__((always_inline)) static float Get(uint index) {
    return kAcFactoredDctMultipliers32[index];
  }
};



// Lowest-complexity self-recursive radix-2 DCT-II/III, following the
// factorization used by the pinned libjxl implementation. Forward() produces
// an unscaled DCT-II; Inverse() consumes coefficients scaled by 1/N.
template <uint N>
struct AcFactoredDct1D {
  __attribute__((always_inline)) static void Forward(
    thread float* values,
    thread float* scratch)
  {
    constexpr uint kHalf = N / 2;

    for (uint i = 0; i < kHalf; ++i) {
      scratch[i] = values[i] + values[N - i - 1];
    }

    AcFactoredDct1D<kHalf>::Forward(scratch, scratch + N);

    for (uint i = 0; i < kHalf; ++i) {
      scratch[kHalf + i] =
        (values[i] - values[N - i - 1]) *
        AcFactoredDctMultipliers<N>::Get(i);
    }

    AcFactoredDct1D<kHalf>::Forward(
      scratch + kHalf,
      scratch + N);

    scratch[kHalf] =
      scratch[kHalf] * kAcFactoredDctSqrt2 +
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

    AcFactoredDct1D<kHalf>::Inverse(scratch, scratch + N);

    for (uint i = kHalf - 1; i > 0; --i) {
      scratch[kHalf + i] += scratch[kHalf + i - 1];
    }

    scratch[kHalf] *= kAcFactoredDctSqrt2;

    AcFactoredDct1D<kHalf>::Inverse(
      scratch + kHalf,
      scratch + N);

    for (uint i = 0; i < kHalf; ++i) {
      const float even = scratch[i];
      const float odd =
        scratch[kHalf + i] *
        AcFactoredDctMultipliers<N>::Get(i);

      values[i] = even + odd;
      values[N - i - 1] = even - odd;
    }
  }
};


template <>
struct AcFactoredDct1D<2> {
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
inline uint AcStrategyCoefficientIndex(uint v, uint u) {
  return Rows < Columns ? v * Columns + u : u * Rows + v;
}

template <uint Rows, uint Columns, bool Inverse>
inline void AcStrategyFactoredDct(threadgroup float* tile, uint lane) {
  constexpr uint Length = Rows > Columns ? Rows : Columns;
  float values[Length];
  float scratch[2 * Length];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr (!Inverse) {
    if (lane < Rows) {
      for (uint x = 0; x < Columns; ++x)
        values[x] = tile[lane * Columns + x];
      AcFactoredDct1D<Columns>::Forward(values, scratch);
      for (uint u = 0; u < Columns; ++u)
        tile[lane * Columns + u] = values[u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < Columns) {
      for (uint y = 0; y < Rows; ++y)
        values[y] = tile[y * Columns + lane];
      AcFactoredDct1D<Rows>::Forward(values, scratch);
    }
    // The packed output may overwrite another lane's input column.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < Columns)
      for (uint v = 0; v < Rows; ++v)
        tile[AcStrategyCoefficientIndex<Rows, Columns>(v, lane)] =
          values[v] * (1.0f / (Rows * Columns));
  } else {
    if (lane < Columns) {
      for (uint v = 0; v < Rows; ++v)
        values[v] = tile[AcStrategyCoefficientIndex<Rows, Columns>(v, lane)];
      AcFactoredDct1D<Rows>::Inverse(values, scratch);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < Columns)
      for (uint y = 0; y < Rows; ++y)
        tile[y * Columns + lane] = values[y];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < Rows) {
      for (uint u = 0; u < Columns; ++u)
        values[u] = tile[lane * Columns + u];
      AcFactoredDct1D<Columns>::Inverse(values, scratch);
      for (uint x = 0; x < Columns; ++x)
        tile[lane * Columns + x] = values[x];
    }
  }
}

// Compact X/Y/B groups share one transform arena and half-size magnitude scratch. Y is published
// before X/B residuals consume it; the scalar candidate finalizer is unchanged.
kernel void gjxl_ac_strategy_dct16_candidate_loss_factored(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device const float* quant_field [[buffer(4)]],
  device const float* matrices [[buffer(5)]],
  device const float* pixel_mask [[buffer(6)]],
  device float* quant_norm [[buffer(7)]], device float* loss_sums [[buffer(8)]],
  device ChannelRate* channel_rates [[buffer(9)]],
  constant AcStrategyBatchParams& params [[buffer(10)]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint candidate_index [[threadgroup_position_in_grid]]) {
  constexpr uint Rows = 16, Columns = 16, Count = Rows * Columns, Workers = 32;
  const uint channel = tid / Workers, local_tid = tid % Workers;
  const uint transform_index = candidate_index * 3 + channel;
  threadgroup float coefficients[3 * Count];
  threadgroup float temporary[3 * Count / 2];
  threadgroup float* tile = coefficients + channel * Count;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool valid = AcStrategyCandidateValid(candidate, params);
  device const float* input = channel == 0   ? opsin_x
                              : channel == 1 ? opsin_y
                                             : opsin_b;
  for (uint i = local_tid; i < Count; i += Workers) {
    const uint x = candidate.block_x * 8 + i % Columns,
               y = candidate.block_y * 8 + i / Columns;
    tile[i] = valid ? input[y * params.opsin_row_stride + x] : NAN;
  }
  if (params.quant_norm_source == kQuantNormFromForwardPass && tid == 0)
    quant_norm[candidate_index] =
      AcStrategyStaticQuantNorm<16, 16>(quant_field, candidate, params);
  AcStrategyFactoredDct<Rows, Columns, false>(tile, local_tid);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  AcStrategyHalfMagnitudeResidual<Count, Workers>(coefficients, temporary,
    matrices, candidates, quant_field, quant_norm, channel_rates, params,
    local_tid, transform_index);
  AcStrategyFactoredDct<Rows, Columns, true>(tile, local_tid);
  AcStrategyReduceRegisterLoss<Count, Workers, Columns, true>(tile, pixel_mask,
    candidates, loss_sums, params, local_tid, transform_index);
}

// Each channel keeps the split path's worker count and reduction tree. The
// shared barrier publishes Y before either chroma channel reads its coefficients.
kernel void gjxl_ac_strategy_dct16x8_candidate_loss_factored(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device const float* quant_field [[buffer(4)]],
  device const float* matrices [[buffer(5)]],
  device const float* pixel_mask [[buffer(6)]],
  device float* quant_norm [[buffer(7)]], device float* loss_sums [[buffer(8)]],
  device ChannelRate* channel_rates [[buffer(9)]],
  constant AcStrategyBatchParams& params [[buffer(10)]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint candidate_index [[threadgroup_position_in_grid]]) {
  constexpr uint Rows = 16, Columns = 8, Count = Rows * Columns, Workers = 32;
  const uint channel = tid / Workers, local_tid = tid % Workers;
  const uint transform_index = candidate_index * 3 + channel;
  threadgroup float coefficients[3 * Count];
  threadgroup float temporary[3 * Count / 2];
  threadgroup float* tile = coefficients + channel * Count;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool valid = AcStrategyCandidateValid(candidate, params);
  device const float* input = channel == 0   ? opsin_x
                              : channel == 1 ? opsin_y
                                             : opsin_b;
  for (uint i = local_tid; i < Count; i += Workers) {
    const uint x = candidate.block_x * 8 + i % Columns,
               y = candidate.block_y * 8 + i / Columns;
    tile[i] = valid ? input[y * params.opsin_row_stride + x] : NAN;
  }
  if (params.quant_norm_source == kQuantNormFromForwardPass && tid == 0)
    quant_norm[candidate_index] =
      AcStrategyStaticQuantNorm<16, 8>(quant_field, candidate, params);
  AcStrategyFactoredDct<Rows, Columns, false>(tile, local_tid);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  AcStrategyHalfMagnitudeResidual<Count, Workers>(coefficients, temporary,
    matrices, candidates, quant_field, quant_norm, channel_rates, params,
    local_tid, transform_index);
  AcStrategyFactoredDct<Rows, Columns, true>(tile, local_tid);
  AcStrategyReduceRegisterLoss<Count, Workers, Columns, true>(tile, pixel_mask,
    candidates, loss_sums, params, local_tid, transform_index);
}

kernel void gjxl_ac_strategy_dct8x16_candidate_loss_parallel(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device const float* quant_field [[buffer(4)]],
  device const float* matrices [[buffer(5)]],
  device const float* pixel_mask [[buffer(6)]],
  device float* quant_norm [[buffer(7)]],
  device float* loss_sums [[buffer(8)]],
  device ChannelRate* channel_rates [[buffer(9)]],
  constant AcStrategyBatchParams& params [[buffer(10)]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint candidate_index [[threadgroup_position_in_grid]]) {
  constexpr uint Count = 8 * 16;
  constexpr uint Workers = 8 / 8 * 32;
  const uint channel = tid / Workers;
  const uint local_tid = tid % Workers;
  const uint local_simdgroup = simdgroup_index % (Workers / 32);
  const uint transform_index = candidate_index * 3 + channel;
  threadgroup float coefficients[3 * Count];
  threadgroup float temporary[3 * Count / 2];
  threadgroup float vertical[64];
  threadgroup float horizontal[256];
  for (uint i = tid; i < 64; i += 3 * Workers) vertical[i] = kOrthonormalDct8[i];
  for (uint i = tid; i < 256; i += 3 * Workers) horizontal[i] = kOrthonormalDct16[i];
  AcStrategyForwardRectangularDct<8, 16, true, false, true>(
    opsin_x, opsin_y, opsin_b, candidates, coefficients, quant_field,
    quant_norm, params, kOrthonormalDct8, kOrthonormalDct16,
    kForwardDct16x8Scale, coefficients + channel * Count, vertical, horizontal,
    lane, 32, local_simdgroup, uint3(transform_index, 0, 0));
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  AcStrategyHalfMagnitudeResidual<Count, Workers>(
    coefficients, temporary, matrices, candidates, quant_field, quant_norm,
    channel_rates, params, local_tid, transform_index);
  AcStrategyInverseRectangularDct<8, 16, Workers, false, true>(
    coefficients + channel * Count, coefficients + channel * Count,
    kOrthonormalDct8, kOrthonormalDct16, kInverseDct16x8Scale,
    vertical, horizontal, local_tid, local_simdgroup, uint3(0));
  AcStrategyReduceSpecializedLoss<Count, Workers, 16, true>(
    coefficients + channel * Count, pixel_mask, candidates, loss_sums,
    params, local_tid, transform_index);
}


// Candidate-local residuals preserve the original FP32 halving tree. Each lane
// first reads its own and Y coefficients into registers. The first barrier
// publishes magnitudes and completes ALL Y reads before in-place stores.
// The second tile is dead forward-pixel storage, then magnitude reduction, then
// inverse pixels. Integer counts reuse dead magnitude cells; Y is not copied.
template <uint Count, uint Workers>
__attribute__((always_inline)) inline void AcStrategyLocalResidual(
  threadgroup float* coefficients, threadgroup float* temporary,
  device const float* matrices, device const AcStrategyCandidate* candidates,
  device const float* quant_field, device const float* quant_norms,
  device ChannelRate* rates, constant AcStrategyBatchParams& params,
  uint tid, uint transform_index) {
  constexpr uint Values = Count / Workers;
  static_assert(Values >= 2, "Residual reductions need at least two values per lane");
  const uint channel = transform_index % 3;
  const uint candidate_index = transform_index / 3;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const float quant_norm = AcStrategyQuantNorm(
    quant_field, quant_norms, candidate, candidate_index, params);
  const float cfl_factor = channel == 0 ? candidate.cfl_x :
                           channel == 2 ? candidate.cfl_b : 0.0f;
  float residual[Values];
  threadgroup float* magnitude = temporary + channel * Count;
  uint nonzero = 0;
#pragma unroll
  for (uint value = 0; value < Values; ++value) {
    const uint coefficient = tid + value * Workers;
    const float decorrelated = coefficients[channel * Count + coefficient] -
      coefficients[Count + coefficient] * cfl_factor;
    const float scaled = decorrelated *
      matrices[(3 + channel) * Count + coefficient] * quant_norm;
    const float rounded = RoundAwayFromZero(scaled);
    residual[value] = matrices[channel * Count + coefficient] * (scaled - rounded);
    magnitude[coefficient] = sqrt(abs(rounded));
    nonzero += rounded != 0.0f ? 1u : 0u;
  }
  // Preserve the original shared-memory magnitude reduction, including the
  // first rounded store. A register-only variant changed a large-input rate.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint delta = 16; delta != 0; delta /= 2) {
    nonzero += simd_shuffle_down(nonzero, delta);
  }
  threadgroup uint* counts = reinterpret_cast<threadgroup uint*>(magnitude + Count / 2);
  for (uint stride = Count / 2; stride >= 32; stride /= 2) {
    for (uint i = tid; i < stride; i += Workers) {
      magnitude[i] += magnitude[i + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stride == Count / 2) {
      // The upper half of the magnitude array is now dead. Reuse a few cells
      // for integer warp totals; later magnitude levels touch only the lower half.
      if (tid % 32 == 0) counts[tid / 32] = nonzero;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (tid < 32) {
    float sum = magnitude[tid];
    for (uint delta = 16; delta != 0; delta /= 2) {
      sum += simd_shuffle_down(sum, delta);
    }
    if (tid == 0) {
      uint count = 0;
      for (uint warp = 0; warp < Workers / 32; ++warp) count += counts[warp];
      rates[transform_index] = {sum, count};
    }
  }
#pragma unroll
  for (uint value = 0; value < Values; ++value) {
    coefficients[channel * Count + tid + value * Workers] = residual[value];
  }
  // The inverse helper's unconditional barrier publishes the residuals and
  // finishes all reads of the temporary reduction tile before pixel writes.
}

kernel void gjxl_ac_strategy_dct8_candidate_loss_local(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device const float* quant_field [[buffer(4)]],
  device const float* matrices [[buffer(5)]],
  device const float* pixel_mask [[buffer(6)]],
  device float* quant_norm [[buffer(7)]],
  device float* loss_sums [[buffer(8)]],
  device ChannelRate* channel_rates [[buffer(9)]],
  constant AcStrategyBatchParams& params [[buffer(10)]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint candidate_index [[threadgroup_position_in_grid]]) {
  constexpr uint Count = 8 * 8;
  constexpr uint Workers = 8 / 8 * 32;
  const uint channel = tid / Workers;
  const uint local_tid = tid % Workers;
  const uint local_simdgroup = simdgroup_index % (Workers / 32);
  const uint transform_index = candidate_index * 3 + channel;
  threadgroup float coefficients[3 * Count];
  threadgroup float temporary[3 * Count / 2];
  threadgroup float basis[Count];
  for (uint i = tid; i < Count; i += 3 * Workers) basis[i] = kOrthonormalDct8[i];
  AcStrategyForwardSquareDct<8, true, false, true>(
    opsin_x, opsin_y, opsin_b, candidates, coefficients, quant_field,
    quant_norm, params, kOrthonormalDct8, kForwardDct8Scale,
    coefficients + channel * Count, basis, lane, 32, local_simdgroup,
    uint3(transform_index, 0, 0));
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  AcStrategyHalfMagnitudeResidual<Count, Workers>(
    coefficients, temporary, matrices, candidates, quant_field, quant_norm,
    channel_rates, params, local_tid, transform_index);
  AcStrategyInverseSquareDct<8, Workers, false, true>(
    coefficients + channel * Count, coefficients + channel * Count,
    kOrthonormalDct8, kInverseDct8Scale, basis,
    local_tid, local_simdgroup, uint3(0));
  AcStrategyReduceSpecializedLoss<Count, Workers, 8, true>(
    coefficients + channel * Count, pixel_mask, candidates, loss_sums,
    params, local_tid, transform_index);
}

kernel void gjxl_ac_strategy_dct32x16_candidate_loss_factored(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device const float* quant_field [[buffer(4)]],
  device const float* matrices [[buffer(5)]],
  device const float* pixel_mask [[buffer(6)]],
  device float* quant_norm [[buffer(7)]], device float* loss_sums [[buffer(8)]],
  device ChannelRate* channel_rates [[buffer(9)]],
  constant AcStrategyBatchParams& params [[buffer(10)]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint candidate_index [[threadgroup_position_in_grid]]) {
  constexpr uint Rows = 32, Columns = 16, Count = Rows * Columns, Workers = 64;
  const uint channel = tid / Workers, local_tid = tid % Workers;
  const uint transform_index = candidate_index * 3 + channel;
  threadgroup float coefficients[3 * Count];
  threadgroup float temporary[3 * Count / 2];
  threadgroup float* tile = coefficients + channel * Count;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool valid = AcStrategyCandidateValid(candidate, params);
  device const float* input = channel == 0   ? opsin_x
                              : channel == 1 ? opsin_y
                                             : opsin_b;
  for (uint i = local_tid; i < Count; i += Workers) {
    const uint x = candidate.block_x * 8 + i % Columns,
               y = candidate.block_y * 8 + i / Columns;
    tile[i] = valid ? input[y * params.opsin_row_stride + x] : NAN;
  }
  if (params.quant_norm_source == kQuantNormFromForwardPass && tid == 0)
    quant_norm[candidate_index] =
      AcStrategyStaticQuantNorm<32, 16>(quant_field, candidate, params);
  AcStrategyFactoredDct<Rows, Columns, false>(tile, local_tid);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  AcStrategyHalfMagnitudeResidual<Count, Workers>(coefficients, temporary,
    matrices, candidates, quant_field, quant_norm, channel_rates, params,
    local_tid, transform_index);
  AcStrategyFactoredDct<Rows, Columns, true>(tile, local_tid);
  AcStrategyReduceSpecializedLoss<Count, Workers, Columns, true>(tile,
    pixel_mask, candidates, loss_sums, params, local_tid, transform_index);
}

kernel void gjxl_ac_strategy_dct16x32_candidate_loss_factored(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device const float* quant_field [[buffer(4)]],
  device const float* matrices [[buffer(5)]],
  device const float* pixel_mask [[buffer(6)]],
  device float* quant_norm [[buffer(7)]], device float* loss_sums [[buffer(8)]],
  device ChannelRate* channel_rates [[buffer(9)]],
  constant AcStrategyBatchParams& params [[buffer(10)]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint candidate_index [[threadgroup_position_in_grid]]) {
  constexpr uint Rows = 16, Columns = 32, Count = Rows * Columns, Workers = 64;
  const uint channel = tid / Workers, local_tid = tid % Workers;
  const uint transform_index = candidate_index * 3 + channel;
  threadgroup float coefficients[3 * Count];
  threadgroup float temporary[3 * Count / 2];
  threadgroup float* tile = coefficients + channel * Count;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool valid = AcStrategyCandidateValid(candidate, params);
  device const float* input = channel == 0   ? opsin_x
                              : channel == 1 ? opsin_y
                                             : opsin_b;
  for (uint i = local_tid; i < Count; i += Workers) {
    const uint x = candidate.block_x * 8 + i % Columns,
               y = candidate.block_y * 8 + i / Columns;
    tile[i] = valid ? input[y * params.opsin_row_stride + x] : NAN;
  }
  if (params.quant_norm_source == kQuantNormFromForwardPass && tid == 0)
    quant_norm[candidate_index] =
      AcStrategyStaticQuantNorm<16, 32>(quant_field, candidate, params);
  AcStrategyFactoredDct<Rows, Columns, false>(tile, local_tid);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  AcStrategyHalfMagnitudeResidual<Count, Workers>(coefficients, temporary,
    matrices, candidates, quant_field, quant_norm, channel_rates, params,
    local_tid, transform_index);
  AcStrategyFactoredDct<Rows, Columns, true>(tile, local_tid);
  AcStrategyReduceRegisterLoss<Count, Workers, Columns, true>(tile, pixel_mask,
    candidates, loss_sums, params, local_tid, transform_index);
}

kernel void gjxl_ac_strategy_dct32_candidate_loss_factored(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device const float* quant_field [[buffer(4)]],
  device const float* matrices [[buffer(5)]],
  device const float* pixel_mask [[buffer(6)]],
  device float* quant_norm [[buffer(7)]], device float* loss_sums [[buffer(8)]],
  device ChannelRate* channel_rates [[buffer(9)]],
  constant AcStrategyBatchParams& params [[buffer(10)]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint candidate_index [[threadgroup_position_in_grid]]) {
  constexpr uint Rows = 32, Columns = 32, Count = Rows * Columns, Workers = 128;
  const uint channel = tid / Workers, local_tid = tid % Workers;
  const uint transform_index = candidate_index * 3 + channel;
  threadgroup float coefficients[3 * Count];
  threadgroup float temporary[3 * Count / 2];
  threadgroup float* tile = coefficients + channel * Count;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool valid = AcStrategyCandidateValid(candidate, params);
  device const float* input = channel == 0   ? opsin_x
                              : channel == 1 ? opsin_y
                                             : opsin_b;
  for (uint i = local_tid; i < Count; i += Workers) {
    const uint x = candidate.block_x * 8 + i % Columns,
               y = candidate.block_y * 8 + i / Columns;
    tile[i] = valid ? input[y * params.opsin_row_stride + x] : NAN;
  }
  if (params.quant_norm_source == kQuantNormFromForwardPass && tid == 0)
    quant_norm[candidate_index] =
      AcStrategyStaticQuantNorm<32, 32>(quant_field, candidate, params);
  AcStrategyFactoredDct<Rows, Columns, false>(tile, local_tid);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  AcStrategyHalfMagnitudeResidual<Count, Workers>(coefficients, temporary,
    matrices, candidates, quant_field, quant_norm, channel_rates, params,
    local_tid, transform_index);
  AcStrategyFactoredDct<Rows, Columns, true>(tile, local_tid);
  AcStrategyReduceSpecializedLoss<Count, Workers, Columns, true>(tile,
    pixel_mask, candidates, loss_sums, params, local_tid, transform_index);
}

kernel void gjxl_ac_strategy_cost_from_loss(
  device const float* loss_sums [[buffer(0)]],
  device const float* pixel_mask [[buffer(1)]],
  device const AcStrategyCandidate* candidates [[buffer(2)]],
  device const ChannelRate* channel_rates [[buffer(3)]],
  device float* costs [[buffer(4)]],
  device const float* quant_field [[buffer(5)]],
  constant AcStrategyBatchParams& params [[buffer(6)]],
  uint candidate_index [[thread_position_in_grid]]) {
  if (candidate_index >= params.candidate_count) return;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  float entropy = 0.0f;
  float loss = 0.0f;
    for (uint channel = 0; channel < 3; ++channel) {
      const uint transform_index = candidate_index * 3 + channel;
      const ChannelRate rate = channel_rates[transform_index];
      entropy += params.cost_delta * rate.magnitude;
      const uint nonzero_bits = CeilLog2Nonzero(rate.nonzero_count + 1) + 1;
      entropy += params.zeros_multiplier * float(
        CeilLog2Nonzero(nonzero_bits + 17) + nonzero_bits);
      loss += loss_sums[transform_index] *
        kChannelMultiplier[channel];

      if (channel == 0 && params.covered_block_count >= 2) {
        const float weight = 1.0f + min(
          3.0f,
          float(params.covered_block_count) / 8.0f);
        entropy *= weight;
        loss *= weight;
      }
    }

    const float quant_norm = AcStrategyQuantNorm(
      quant_field, costs, candidate, candidate_index, params);
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
