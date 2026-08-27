// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <metal_stdlib>

#include "aq_quantization.h"

using namespace metal;

struct AqContractProbeParams {
  uint source_width;
  uint source_height;
  uint coding_width;
  uint coding_height;
  uint block_width;
  uint block_height;
  uint tile_width;
  uint tile_height;
  uint original_stride;
  uint coding_stride;
  uint strategy_stride;
  uint raw_quant_stride;
  uint inverse_sigma_stride;
  uint color_stride;
  uint output_stride;
  uint global_scale;
  uint quant_dc;
  float option_probe;
};

kernel void gjxl_aq_contract_probe(
  device const float* original_x [[buffer(0)]],
  device const float* original_y [[buffer(1)]],
  device const float* original_b [[buffer(2)]],
  device const float* coding_x [[buffer(3)]],
  device const float* coding_y [[buffer(4)]],
  device const float* coding_b [[buffer(5)]],
  device const int* strategies [[buffer(6)]],
  device const float* quant_tables [[buffer(7)]],
  device const int* raw_quant [[buffer(8)]],
  device const float* inverse_sigma [[buffer(9)]],
  device const char* y_to_x [[buffer(10)]],
  device const char* y_to_b [[buffer(11)]],
  device float* output [[buffer(12)]],
  constant AqContractProbeParams& params [[buffer(13)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.block_width ||
      position.y >= params.block_height) {
    return;
  }

  const uint source_x = min(position.x * 8u, params.source_width - 1u);
  const uint source_y = min(position.y * 8u, params.source_height - 1u);
  const uint coding_x_pos = position.x * 8u;
  const uint coding_y_pos = position.y * 8u;
  const uint source_index = source_y * params.original_stride + source_x;
  const uint coding_index =
    coding_y_pos * params.coding_stride + coding_x_pos;
  const uint block_index =
    position.y * params.raw_quant_stride + position.x;
  const uint strategy_index =
    position.y * params.strategy_stride + position.x * 2u;
  const uint color_x = min(position.x / 8u, params.tile_width - 1u);
  const uint color_y = min(position.y / 8u, params.tile_height - 1u);
  const uint color_index = color_y * params.color_stride + color_x;
  const uint output_index = position.y * params.output_stride + position.x;

  const uint strategy = uint(strategies[strategy_index]);
  const uint anchor = uint(strategies[strategy_index + 1u]);
  const uint2 table_offsets = aq_quant_table_offsets(strategy);

  // Keep this order in sync with the independent CPU oracle in the test.
  float value = original_x[source_index];
  value += original_y[source_index] * 0.5f;
  value += original_b[source_index] * 0.25f;
  value += coding_x[coding_index] * 0.125f;
  value += coding_y[coding_index] * 0.0625f;
  value += coding_b[coding_index] * 0.03125f;
  value += float(raw_quant[block_index]) * (1.0f / 256.0f);
  value += inverse_sigma[
    position.y * params.inverse_sigma_stride + position.x];
  value += float(strategy) * (1.0f / 32.0f);
  value += float(anchor) * 0.5f;
  value += float(y_to_x[color_index]) * (1.0f / 256.0f);
  value += float(y_to_b[color_index]) * (1.0f / 256.0f);
  value += quant_tables[table_offsets.x];
  value += quant_tables[table_offsets.y + 1u] * (1.0f / 256.0f);
  value += float(params.global_scale) * (1.0f / 65536.0f);
  value += float(params.quant_dc) * (1.0f / 65536.0f);
  value += params.option_probe;
  output[output_index] = value;
}
