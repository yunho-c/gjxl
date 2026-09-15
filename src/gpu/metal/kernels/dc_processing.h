// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// DC quantization/prediction/smoothing adapted from pinned libjxl.
// Included after the shared AQ transform and rounding helpers.

struct AqDcProcessingParams {
  uint width;
  uint height;
  uint global_scale;
  uint quant_dc;
  uint quantization_mode;
  uint predictor;
  uint extra_precision;
  uint use_resident_quantizer;
};

constant uint kDcReciprocal[64] = {
  16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152,
  1864135, 1677721, 1525201, 1398101, 1290555, 1198372, 1118481, 1048576,
  986895, 932067, 883011, 838860, 798915, 762600, 729444, 699050,
  671088, 645277, 621378, 599186, 578524, 559240, 541200, 524288,
  508400, 493447, 479349, 466033, 453438, 441505, 430185, 419430,
  409200, 399457, 390167, 381300, 372827, 364722, 356962, 349525,
  342392, 335544, 328965, 322638, 316551, 310689, 305040, 299593,
  294337, 289262, 284359, 279620, 275036, 270600, 266305, 262144,
};

// Four waves retain every predecessor of the x+2*y traversal. Each lane owns
// one row; north-west is the oldest dependency, three waves behind. Keeping
// these values in threadgroup memory avoids a device barrier at every wave.
constant uint kDcWaveRows = 256u;
constant uint kDcWaveValues = 4u * kDcWaveRows;

inline uint DcWaveIndex(uint wave, uint row) {
  return (wave & 3u) * kDcWaveRows + row;
}

// Store each sample's original four predictor errors and the signed error.
// The serial predictor mutates its previous row when advancing west-to-east.
// Recover that mutation from the current row's west/two-west samples instead,
// making every dependency explicit for diagonal execution.
struct AqDcWavefrontState {
  threadgroup uint* errors;
  uint width;
  long predictions[4];
  long prediction;

  long Predict(uint x, uint y, long n, long w, long ne) {
    const uint wave = x + 2u * y;
    uint weights[4];
    uint weight_sum = 0;
    for (uint i = 0; i < 4; ++i) {
      threadgroup const uint* plane = errors + i * kDcWaveValues;
      const uint north = (y ? plane[DcWaveIndex(wave - 2u, y - 1u)] : 0u) +
                         (x ? plane[DcWaveIndex(wave - 1u, y)] : 0u);
      const uint northeast = x + 1u < width
        ? (y ? plane[DcWaveIndex(wave - 1u, y - 1u)] : 0u) : north;
      const uint northwest = x
        ? (y ? plane[DcWaveIndex(wave - 3u, y - 1u)] : 0u) +
          (x > 1u ? plane[DcWaveIndex(wave - 2u, y)] : 0u) : north;
      const uint error = north + northeast + northwest;
      const uint bits = 64u - clz(ulong(error) + 1u);
      const uint shift = bits > 6u ? bits - 6u : 0u;
      weights[i] = 4u + (((i == 0u ? 13u : 12u) *
                          kDcReciprocal[error >> shift]) >> shift);
      weight_sum += weights[i];
    }
    const uint log_weight = 31u - clz(weight_sum);
    weight_sum = 0u;
    for (uint i = 0; i < 4; ++i) {
      weights[i] >>= log_weight - 4u;
      weight_sum += weights[i];
    }
    threadgroup const int* plane =
      reinterpret_cast<threadgroup const int*>(errors + 4u * kDcWaveValues);
    const long ew = x ? plane[DcWaveIndex(wave - 1u, y)] : 0;
    const long en = y ? plane[DcWaveIndex(wave - 2u, y - 1u)] : 0;
    const long enw = x && y ? plane[DcWaveIndex(wave - 3u, y - 1u)] : en;
    const long ene = y && x + 1u < width ? plane[DcWaveIndex(wave - 1u, y - 1u)] : en;
    n *= 8; w *= 8; ne *= 8;
    predictions[0] = w + ne - n;
    predictions[1] = n - (((en + ew + ene) * 16) >> 5);
    predictions[2] = w - (((en + ew + enw) * 10) >> 5);
    predictions[3] = n - (((enw + en + ene) * 7) >> 5);
    long sum = long(weight_sum >> 1u) - 1;
    for (uint i = 0; i < 4; ++i) sum += predictions[i] * long(weights[i]);
    prediction = (sum * long(kDcReciprocal[weight_sum - 1u])) >> 24;
    if (((en ^ ew) | (en ^ enw)) <= 0)
      prediction = clamp(prediction, min(w, min(ne, n)), max(w, max(ne, n)));
    return (prediction + 3) >> 3;
  }

  bool Update(int value, uint x, uint y) {
    const uint index = DcWaveIndex(x + 2u * y, y);
    const long scaled = long(value) * 8;
    const long error = prediction - scaled;
    bool valid = error >= -2147483648l && error <= 2147483647l;
    threadgroup int* plane =
      reinterpret_cast<threadgroup int*>(errors + 4u * kDcWaveValues);
    plane[index] = valid ? int(error) : 0;
    for (uint i = 0; i < 4; ++i) {
      const ulong magnitude = (abs(predictions[i] - scaled) + 3u) >> 3u;
      valid = valid && magnitude <= 4294967295ul;
      errors[i * kDcWaveValues + index] = uint(magnitude);
    }
    return valid;
  }
};

// One threadgroup owns one channel of a DC group. X and Y run independently
// in the first dispatch; B runs after Y is complete in a second dispatch.
// Weighted dependencies precede x+2*y; gradient dependencies precede x+y.
// Every lane reaches the wave barrier, including partial-group inactive rows.
kernel void gjxl_aq_dc_quantize(
  device float* dc [[buffer(0)]],
  device int* quantized [[buffer(1)]],
  device uint* scratch [[buffer(2)]],
  device atomic_uint* error [[buffer(3)]],
  constant AqDcProcessingParams& params [[buffer(4)]],
  device const uint* resident_quantizer [[buffer(5)]],
  constant uint& channel_base [[buffer(6)]],
  uint3 position [[threadgroup_position_in_grid]],
  uint row_index [[thread_index_in_threadgroup]]) {
  const uint group = position.x;
  const uint c = channel_base + position.y;
  const uint groups_x = (params.width + 255u) / 256u;
  const uint groups_y = (params.height + 255u) / 256u;
  if (group >= groups_x * groups_y) return;
  if (c >= 3u) {
    atomic_fetch_or_explicit(error, 16u, memory_order_relaxed);
    return;
  }
  const uint gx = (group % groups_x) * 256u;
  const uint gy = (group / groups_x) * 256u;
  const uint width = min(256u, params.width - gx);
  const uint height = min(256u, params.height - gy);
  const uint area = params.width * params.height;
  const uint global_scale = params.use_resident_quantizer ? resident_quantizer[0] : params.global_scale;
  const uint quant_dc = params.use_resident_quantizer ? resident_quantizer[1] : params.quant_dc;
  const float scale = float(global_scale) / 65536.0f;
  const float inverse_dc = (65536.0f / float(global_scale)) / float(quant_dc);
  const float precision = float(1u << params.extra_precision);
  const float factors[3] = {4096.0f, 512.0f, 256.0f};
  float steps[3], inverse[3];
  for (uint c = 0; c < 3; ++c) {
    steps[c] = (inverse_dc / factors[c]) / precision;
    inverse[c] = (factors[c] * scale * float(quant_dc)) * precision;
  }
  const bool weighted = params.quantization_mode == 1u && params.predictor == 1u;
  threadgroup uint wave_errors[5u * kDcWaveValues];
  threadgroup int wave_quantized[kDcWaveValues];
  AqDcWavefrontState state;
  state.width = width;
  state.errors = wave_errors;
  const uint row_step = weighted ? 2u : 1u;
  const uint waves = width + row_step * (height - 1u);
  {
    device int* plane = quantized + c * area;
    for (uint wave = 0; wave < waves; ++wave) {
      const int column = int(wave) - int(row_step * row_index);
      if (row_index < height && column >= 0 && uint(column) < width) {
        const uint x = uint(column), y = row_index;
        const uint index = (gy + y) * params.width + gx + x;
        float value = dc[c * area + index];
        if (c == 2u) {
          value = params.quantization_mode == 1u
            ? fma(-float(quantized[area + index]), steps[1], value)
            : value - float(quantized[area + index]) * steps[1];
        }
        long guess = 0;
        if (params.quantization_mode == 1u) {
          const long w = x ? wave_quantized[DcWaveIndex(wave - 1u, y)]
            : y ? wave_quantized[DcWaveIndex(wave - row_step, y - 1u)] : 0;
          const long n = y ? wave_quantized[DcWaveIndex(wave - row_step, y - 1u)] : w;
          if (weighted) {
            const long ne = y && x + 1u < width
              ? wave_quantized[DcWaveIndex(wave - 1u, y - 1u)] : n;
            guess = state.Predict(x, y, n, w, ne);
          } else {
            const long nw = x && y
              ? wave_quantized[DcWaveIndex(wave - 2u, y - 1u)] : w;
            guess = clamp(n + w - nw, min(n, w), max(n, w));
          }
        }
        float residual = value * inverse[c];
        residual -= float(guess);
        if (params.quantization_mode == 1u && residual > -0.62f && residual < 0.62f)
          residual = 0.0f;
        float rounded = round(residual);
        if (params.quantization_mode == 1u && (rounded > 2.0f || rounded < -2.0f))
          rounded = round(residual * 0.5f) * 2.0f;
        const long integer = guess + long(aq_round_dc(rounded, error));
        const bool in_range = integer >= -2147483648l && integer <= 2147483647l;
        const int q = in_range ? int(integer) : 0;
        const bool predictor_valid = !weighted || state.Update(q, x, y);
        if (!in_range || !predictor_valid)
          atomic_fetch_or_explicit(error, 16u, memory_order_relaxed);
        plane[index] = q;
        wave_quantized[DcWaveIndex(wave, y)] = q;
        float reconstructed = float(q) * steps[c];
        if (c == 2u)
          reconstructed = float(quantized[area + index]) * steps[1] + reconstructed;
        dc[c * area + index] = reconstructed;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}

kernel void gjxl_aq_dc_smooth(
  device const float* dc [[buffer(0)]],
  device float* smoothed [[buffer(1)]],
  device atomic_uint* error [[buffer(2)]],
  constant AqDcProcessingParams& params [[buffer(3)]],
  device const uint* resident_quantizer [[buffer(4)]],
  uint index [[thread_position_in_grid]]) {
  const uint area = params.width * params.height;
  if (index >= area) return;
  const uint x = index % params.width;
  const uint y = index / params.width;
  if (x == 0u || y == 0u || x + 1u == params.width || y + 1u == params.height) {
    for (uint c = 0; c < 3; ++c) {
      const float value = dc[c * area + index];
      if (!isfinite(value)) atomic_fetch_or_explicit(error, 8u, memory_order_relaxed);
      smoothed[c * area + index] = value;
    }
    return;
  }
  const uint global_scale = params.use_resident_quantizer ? resident_quantizer[0] : params.global_scale;
  const uint quant_dc = params.use_resident_quantizer ? resident_quantizer[1] : params.quant_dc;
  const float inverse_dc = (65536.0f / float(global_scale)) / float(quant_dc);
  const float factors[3] = {4096.0f, 512.0f, 256.0f};
  constexpr float side_weight = 0.20345139757231578f;
  constexpr float corner_weight = 0.0334829185968739f;
  constexpr float center_weight = 1.0f - 4.0f * (side_weight + corner_weight);
  float center[3], filtered[3];
  float gap = 0.5f;
  for (uint c = 0; c < 3; ++c) {
    device const float* row = dc + c * area + index;
    device const float* top = row - params.width;
    device const float* bottom = row + params.width;
    const float corners = (top[-1] + top[1]) + (bottom[-1] + bottom[1]);
    const float sides = (row[-1] + row[1]) + (top[0] + bottom[0]);
    center[c] = row[0];
    filtered[c] = fma(corners, corner_weight, fma(sides, side_weight, row[0] * center_weight));
    gap = max(gap, abs((row[0] - filtered[c]) / (inverse_dc / factors[c])));
  }
  const float factor = max(0.0f, fma(-4.0f, gap, 3.0f));
  for (uint c = 0; c < 3; ++c) {
    const float value = fma(filtered[c] - center[c], factor, center[c]);
    if (!isfinite(value)) atomic_fetch_or_explicit(error, 8u, memory_order_relaxed);
    smoothed[c * area + index] = value;
  }
}

kernel void gjxl_aq_dc_low_frequencies(
  device const uint2* anchors [[buffer(0)]],
  device const float* dc [[buffer(1)]],
  device float* coefficients [[buffer(2)]],
  constant AqReconstructionParams& params [[buffer(3)]],
  uint task [[thread_position_in_grid]]) {
  const uint covered = params.covered_width * params.covered_height;
  if (task >= params.anchor_count * 3u * covered) return;
  const uint anchor_index = task / (3u * covered);
  const uint channel = (task / covered) % 3u;
  const uint small = task % covered;
  const uint u = small % params.covered_width;
  const uint v = small / params.covered_width;
  const uint2 anchor = anchors[params.anchor_offset + anchor_index];
  const uint area = params.block_width * params.block_height;
  float value = 0.0f;
  for (uint y = 0; y < params.covered_height; ++y)
    for (uint x = 0; x < params.covered_width; ++x)
      value += dc[channel * area + (anchor.y + y) * params.block_width + anchor.x + x] *
               aq_forward_basis(params.covered_height, v, y) *
               aq_forward_basis(params.covered_width, u, x);
  value *= aq_upsample_scale(params.covered_height, v) * aq_upsample_scale(params.covered_width, u);
  coefficients[params.coefficient_offset +
    (channel * params.anchor_count + anchor_index) * params.coefficient_count +
    aq_coefficient_index(params, v, u)] = value;
}
