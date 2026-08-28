// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// The opsin, frequency, Malta, masking, and multiscale computations are
// adapted from pinned JPEG XL Butteraugli code distributed under its
// BSD-style license. See third_party/libjxl/LICENSE.

#include <metal_stdlib>

using namespace metal;

struct PlaneParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
};

struct ExpandParams {
  uint input_width;
  uint input_height;
  uint output_width;
  uint output_height;
  uint input_stride;
  uint output_stride;
  uint xborder;
  uint yborder;
};

struct SubsampleParams {
  uint input_width;
  uint input_height;
  uint output_width;
  uint output_height;
  uint input_stride;
  uint output_stride;
};

struct ConvolutionParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  uint kernel_size;
};

struct OpsinParams {
  uint width;
  uint height;
  uint input_stride0;
  uint input_stride1;
  uint input_stride2;
  uint blurred_stride;
  uint output_stride;
  float intensity_target;
};

struct FrequencyParams {
  uint width;
  uint height;
  uint xyb_stride;
  uint psycho_stride;
};

struct FrequencyChannelParams {
  uint width;
  uint height;
  uint input_stride;
  uint blurred_stride;
  uint output_stride;
  uint channel;
};

struct MaltaScaleParams {
  uint width;
  uint height;
  uint reference_stride;
  uint distorted_stride;
  uint output_stride;
  uint low_frequency;
  float norm2_0_gt_1;
  float norm2_0_lt_1;
  float norm;
};

struct MaltaResponseParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  uint accumulation_stride;
  uint low_frequency;
  uint initialize_accumulation;
};

struct DifferenceParams {
  uint width;
  uint height;
  uint reference_stride;
  uint distorted_stride;
  uint work_stride;
  float asymmetry;
};

struct FinalParams {
  uint width;
  uint height;
  uint stride;
  uint output_stride;
  float x_multiplier;
};

struct CropParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  uint xborder;
  uint yborder;
};

struct ComposeParams {
  uint width;
  uint height;
  uint main_stride;
  uint sub_stride;
  uint output_stride;
};

struct ReductionParams {
  uint width;
  uint input_stride;
  uint input_count;
};

inline float unfused_multiply_add(float multiplier, float value, float addend) {
  const float product = multiplier * value;
  return product + addend;
}

inline int mirror_coordinate(int coordinate, int size) {
  while (coordinate < 0 || coordinate >= size) {
    coordinate = coordinate < 0 ? -coordinate - 1
                                : 2 * size - 1 - coordinate;
  }
  return coordinate;
}

kernel void gjxl_butteraugli_copy_f32(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant PlaneParams& params [[buffer(2)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  output[position.y * params.output_stride + position.x] =
    input[position.y * params.input_stride + position.x];
}

kernel void gjxl_butteraugli_expand_f32(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant ExpandParams& params [[buffer(2)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.output_width || position.y >= params.output_height) {
    return;
  }
  const uint source_x = min(
    params.input_width - 1,
    position.x > params.xborder ? position.x - params.xborder : 0u);
  const uint source_y = min(
    params.input_height - 1,
    position.y > params.yborder ? position.y - params.yborder : 0u);
  output[position.y * params.output_stride + position.x] =
    input[source_y * params.input_stride + source_x];
}

kernel void gjxl_butteraugli_subsample2x_f32(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant SubsampleParams& params [[buffer(2)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.output_width || position.y >= params.output_height) {
    return;
  }
  const uint source_x = position.x * 2;
  const uint source_y = position.y * 2;
  float value = 0.0f;
  value += 0.25f * input[source_y * params.input_stride + source_x];
  if (source_x + 1 < params.input_width) {
    value += 0.25f * input[source_y * params.input_stride + source_x + 1];
  }
  if (source_y + 1 < params.input_height) {
    value += 0.25f * input[(source_y + 1) * params.input_stride + source_x];
    if (source_x + 1 < params.input_width) {
      value += 0.25f *
        input[(source_y + 1) * params.input_stride + source_x + 1];
    }
  }
  if ((params.input_width & 1u) != 0 &&
      position.x + 1 == params.output_width) {
    value *= 2.0f;
  }
  if ((params.input_height & 1u) != 0 &&
      position.y + 1 == params.output_height) {
    value *= 2.0f;
  }
  output[position.y * params.output_stride + position.x] = value;
}

inline float blur5_horizontal_value(
  device const float* row, int x, int width,
  float weight0, float weight1, float weight2) {

  float result = row[x] * weight0;
  const float pair1 = row[mirror_coordinate(x - 1, width)] +
                      row[mirror_coordinate(x + 1, width)];
  result = unfused_multiply_add(pair1, weight1, result);
  const float pair2 = row[mirror_coordinate(x - 2, width)] +
                      row[mirror_coordinate(x + 2, width)];
  return unfused_multiply_add(pair2, weight2, result);
}

kernel void gjxl_butteraugli_blur5_horizontal_f32(
  device const float* input [[buffer(0)]],
  device const float* weights [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant ConvolutionParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  float sum_weights = 0.0f;
  for (uint index = 0; index < 5; ++index) sum_weights += weights[index];
  const float scale = 1.0f / sum_weights;
  const float weight0 = weights[2] * scale;
  const float weight1 = weights[1] * scale;
  const float weight2 = weights[0] * scale;
  device const float* row = input + position.y * params.input_stride;
  output[position.y * params.output_stride + position.x] =
    blur5_horizontal_value(
      row, int(position.x), int(params.width), weight0, weight1, weight2);
}

kernel void gjxl_butteraugli_blur5_vertical_f32(
  device const float* input [[buffer(0)]],
  device const float* weights [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant ConvolutionParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  float sum_weights = 0.0f;
  for (uint index = 0; index < 5; ++index) sum_weights += weights[index];
  const float scale = 1.0f / sum_weights;
  const float weight0 = weights[2] * scale;
  const float weight1 = weights[1] * scale;
  const float weight2 = weights[0] * scale;
  const int y = int(position.y);
  const int height = int(params.height);
  const auto sample = [&](int source_y) {
    return input[uint(mirror_coordinate(source_y, height)) *
                 params.input_stride + position.x];
  };
  const float center = sample(y);
  const float pair1 = sample(y - 1) + sample(y + 1);
  const float pair2 = sample(y - 2) + sample(y + 2);
  float result = center * weight0;
  result = unfused_multiply_add(pair1, weight1, result);
  result = unfused_multiply_add(pair2, weight2, result);
  output[position.y * params.output_stride + position.x] = result;
}

kernel void gjxl_butteraugli_convolve_transpose_f32(
  device const float* input [[buffer(0)]],
  device const float* weights [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant ConvolutionParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const int radius = int(params.kernel_size / 2);
  const int center = int(position.x);
  const int first = max(0, center - radius);
  const int last = min(int(params.width) - 1, center + radius);
  float weight_sum = 0.0f;
  float sum = 0.0f;
  for (int source_x = first; source_x <= last; ++source_x) {
    const float weight = weights[source_x + radius - center];
    weight_sum += weight;
    sum += input[position.y * params.input_stride + uint(source_x)] * weight;
  }
  output[position.x * params.output_stride + position.y] = sum / weight_sum;
}

inline float butteraugli_fast_log2(float value) {
  constexpr float kP0 = -1.8503833400518310e-06f;
  constexpr float kP1 = 1.4287160470083755f;
  constexpr float kP2 = 0.74245873327820566f;
  constexpr float kQ0 = 0.99032814277590719f;
  constexpr float kQ1 = 1.0096718572241148f;
  constexpr float kQ2 = 0.17409343003366853f;
  const uint value_bits = as_type<uint>(value);
  const int shifted_exponent = int(value_bits - 0x3f2aaaabu) >> 23;
  const uint mantissa_bits = value_bits - (uint(shifted_exponent) << 23);
  const float x = as_type<float>(mantissa_bits) - 1.0f;
  float numerator = unfused_multiply_add(kP2, x, kP1);
  numerator = unfused_multiply_add(numerator, x, kP0);
  float denominator = unfused_multiply_add(kQ2, x, kQ1);
  denominator = unfused_multiply_add(denominator, x, kQ0);
  return numerator / denominator + float(shifted_exponent);
}

inline float gamma_value(float value) {
  constexpr float kRetMul = 19.245013259874995f * 0.6931471805599453f;
  constexpr float kRetAdd = -23.16046239805755f;
  constexpr float kBias = 9.9710635769299145f;
  return unfused_multiply_add(
    kRetMul, butteraugli_fast_log2(max(value, 0.0f) + kBias), kRetAdd);
}

inline float3 opsin_absorbance(float red, float green, float blue,
                               bool clamp_result) {
  float3 output;
  output.x = unfused_multiply_add(
    0.29956550340058319f, red,
    unfused_multiply_add(
      0.63373087833825936f, green,
      unfused_multiply_add(0.077705617820981968f, blue,
                           1.7557483643287353f)));
  output.y = unfused_multiply_add(
    0.22158691104574774f, red,
    unfused_multiply_add(
      0.69391388044116142f, green,
      unfused_multiply_add(0.0987313588422f, blue,
                           1.7557483643287353f)));
  output.z = unfused_multiply_add(
    0.02f, red,
    unfused_multiply_add(
      0.02f, green,
      unfused_multiply_add(0.20480129041026129f, blue,
                           12.226454707163354f)));
  if (clamp_result) {
    output = max(output, float3(
      1.7557483643287353f, 1.7557483643287353f, 12.226454707163354f));
  }
  return output;
}

kernel void gjxl_butteraugli_opsin_f32(
  device const float* input0 [[buffer(0)]],
  device const float* input1 [[buffer(1)]],
  device const float* input2 [[buffer(2)]],
  device const float* blurred0 [[buffer(3)]],
  device const float* blurred1 [[buffer(4)]],
  device const float* blurred2 [[buffer(5)]],
  device float* output0 [[buffer(6)]],
  device float* output1 [[buffer(7)]],
  device float* output2 [[buffer(8)]],
  constant OpsinParams& params [[buffer(9)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint index0 = position.y * params.input_stride0 + position.x;
  const uint index1 = position.y * params.input_stride1 + position.x;
  const uint index2 = position.y * params.input_stride2 + position.x;
  const uint blurred_index = position.y * params.blurred_stride + position.x;
  const uint output_index = position.y * params.output_stride + position.x;
  const float3 input_rgb(
    input0[index0], input1[index1], input2[index2]);
  const float3 blurred_rgb(
    blurred0[blurred_index], blurred1[blurred_index],
    blurred2[blurred_index]);
  if (!all(isfinite(input_rgb)) || !all(isfinite(blurred_rgb))) {
    output0[output_index] = NAN;
    output1[output_index] = NAN;
    output2[output_index] = NAN;
    return;
  }
  float3 pre = opsin_absorbance(
    blurred_rgb.x * params.intensity_target,
    blurred_rgb.y * params.intensity_target,
    blurred_rgb.z * params.intensity_target, true);
  pre = max(pre, float3(1.0e-4f));
  const float3 sensitivity = max(
    float3(gamma_value(pre.x), gamma_value(pre.y), gamma_value(pre.z)) / pre,
    float3(1.0e-4f));
  float3 current = opsin_absorbance(
    input_rgb.x * params.intensity_target,
    input_rgb.y * params.intensity_target,
    input_rgb.z * params.intensity_target, false);
  current = max(current * sensitivity,
                float3(1.7557483643287353f,
                       1.7557483643287353f,
                       12.226454707163354f));
  output0[output_index] = current.x - current.y;
  output1[output_index] = current.x + current.y;
  output2[output_index] = current.z;
}

kernel void gjxl_butteraugli_frequency_low_medium_f32(
  device const float* xyb0 [[buffer(0)]],
  device const float* xyb1 [[buffer(1)]],
  device const float* xyb2 [[buffer(2)]],
  device float* low0 [[buffer(3)]],
  device float* low1 [[buffer(4)]],
  device float* low2 [[buffer(5)]],
  device float* medium0 [[buffer(6)]],
  device float* medium1 [[buffer(7)]],
  device float* medium2 [[buffer(8)]],
  constant FrequencyParams& params [[buffer(9)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint xyb_index = position.y * params.xyb_stride + position.x;
  const uint psycho_index = position.y * params.psycho_stride + position.x;
  const float low_x = low0[psycho_index];
  const float low_y = low1[psycho_index];
  const float low_b = low2[psycho_index];
  medium0[psycho_index] = xyb0[xyb_index] - low_x;
  medium1[psycho_index] = xyb1[xyb_index] - low_y;
  medium2[psycho_index] = xyb2[xyb_index] - low_b;
  low0[psycho_index] = low_x * 33.832837186260f;
  low1[psycho_index] = low_y * 14.458268100570f;
  low2[psycho_index] =
    unfused_multiply_add(-0.362267051518f, low_y, low_b) *
    49.87984651440f;
}

inline float maximum_clamp(float value, float maximum) {
  if (value >= maximum) {
    return unfused_multiply_add(value - maximum, 0.724216145665f, maximum);
  }
  if (value < -maximum) {
    return unfused_multiply_add(value + maximum, 0.724216145665f, -maximum);
  }
  return value;
}

inline float remove_range(float value, float width) {
  return value > width ? value - width
       : value < -width ? value + width
                        : 0.0f;
}

inline float amplify_range(float value, float width) {
  return value > width ? value + width
       : value < -width ? value - width
                        : value + value;
}

kernel void gjxl_butteraugli_frequency_high_f32(
  device float* medium [[buffer(0)]],
  device const float* blurred [[buffer(1)]],
  device float* high [[buffer(2)]],
  constant FrequencyChannelParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint input_index = position.y * params.input_stride + position.x;
  const uint blurred_index = position.y * params.blurred_stride + position.x;
  const uint output_index = position.y * params.output_stride + position.x;
  const float original = medium[input_index];
  const float low_pass = blurred[blurred_index];
  high[output_index] = original - low_pass;
  medium[input_index] = params.channel == 0
    ? remove_range(low_pass, 0.29f)
    : amplify_range(low_pass, 0.1f);
}

kernel void gjxl_butteraugli_frequency_suppress_x_f32(
  device float* high_x [[buffer(0)]],
  device const float* high_y [[buffer(1)]],
  constant PlaneParams& params [[buffer(2)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint x_index = position.y * params.output_stride + position.x;
  const uint y_index = position.y * params.input_stride + position.x;
  const float y = high_y[y_index];
  const float denominator = unfused_multiply_add(y, y, 46.0f);
  const float scaler = unfused_multiply_add(
    46.0f / denominator, 1.0f - 0.653020556257f, 0.653020556257f);
  high_x[x_index] *= scaler;
}

kernel void gjxl_butteraugli_frequency_ultra_f32(
  device float* high [[buffer(0)]],
  device const float* blurred [[buffer(1)]],
  device float* ultra [[buffer(2)]],
  constant FrequencyChannelParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint high_index = position.y * params.input_stride + position.x;
  const uint blurred_index = position.y * params.blurred_stride + position.x;
  const uint ultra_index = position.y * params.output_stride + position.x;
  const float original = high[high_index];
  float low_pass = blurred[blurred_index];
  if (params.channel == 0) {
    ultra[ultra_index] = remove_range(original - low_pass, 0.04f);
    high[high_index] = remove_range(low_pass, 1.5f);
  } else {
    low_pass = maximum_clamp(low_pass, 28.4691806922f);
    ultra[ultra_index] =
      maximum_clamp(original - low_pass, 5.19175294647f) * 2.69313763794f;
    high[high_index] = amplify_range(low_pass * 2.155f, 0.132f);
  }
}

kernel void gjxl_butteraugli_malta_scale_f32(
  device const float* reference [[buffer(0)]],
  device const float* distorted [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant MaltaScaleParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint reference_index =
    position.y * params.reference_stride + position.x;
  const uint distorted_index =
    position.y * params.distorted_stride + position.x;
  const uint output_index = position.y * params.output_stride + position.x;
  const float value0 = reference[reference_index];
  const float value1 = distorted[distorted_index];
  const float absolute = 0.5f * (abs(value0) + abs(value1));
  const float difference = value0 - value1;
  const float scaler = params.norm2_0_gt_1 / (params.norm + absolute);
  float scaled = scaler * difference;
  const float scaler2 = params.norm2_0_lt_1 / (params.norm + absolute);
  const float magnitude = abs(value0);
  const float too_small = 0.55f * magnitude;
  const float too_big = 1.05f * magnitude;
  if (value0 < 0.0f) {
    if (value1 > -too_small) {
      scaled -= scaler2 * (value1 + too_small);
    } else if (value1 < -too_big) {
      scaled += scaler2 * (-value1 - too_big);
    }
  } else if (value1 < too_small) {
    scaled += scaler2 * (too_small - value1);
  } else if (value1 > too_big) {
    scaled -= scaler2 * (value1 - too_big);
  }
  output[output_index] = scaled;
}

inline float read_zero(device const float* input, int x, int y,
                       constant MaltaResponseParams& params) {
  if (x < 0 || y < 0 || x >= int(params.width) || y >= int(params.height)) {
    return 0.0f;
  }
  return input[uint(y) * params.input_stride + uint(x)];
}

inline void add_square(float value, thread float& total) {
  total = unfused_multiply_add(value, value, total);
}

inline float sum5(float a, float b, float c, float d, float e) {
  return (a + b) + (c + (d + e));
}

inline float malta_lf(device const float* input, int x, int y,
                      constant MaltaResponseParams& p) {
  const auto v = [&](int dx, int dy) { return read_zero(input, x + dx, y + dy, p); };
  float sum = sum5(v(-4,0), v(-2,0), v(0,0), v(2,0), v(4,0));
  float result = sum * sum;
  sum = sum5(v(0,-4),v(0,-2),v(0,0),v(0,2),v(0,4)); add_square(sum, result);
  sum = sum5(v(-3,-3),v(-2,-2),v(0,0),v(2,2),v(3,3)); add_square(sum,result);
  sum = sum5(v(3,-3),v(2,-2),v(0,0),v(-2,2),v(-3,3)); add_square(sum,result);
  sum = sum5(v(1,-4),v(1,-2),v(0,0),v(-1,2),v(-1,4)); add_square(sum,result);
  sum = sum5(v(-1,-4),v(-1,-2),v(0,0),v(1,2),v(1,4)); add_square(sum,result);
  sum = sum5(v(-4,-1),v(-2,-1),v(0,0),v(2,1),v(4,1)); add_square(sum,result);
  sum = sum5(v(-4,1),v(-2,1),v(0,0),v(2,-1),v(4,-1)); add_square(sum,result);
  sum = sum5(v(-2,-3),v(-1,-2),v(0,0),v(1,2),v(2,3)); add_square(sum,result);
  sum = sum5(v(2,-3),v(1,-2),v(0,0),v(-1,2),v(-2,3)); add_square(sum,result);
  sum = sum5(v(-3,-2),v(-2,-1),v(0,0),v(2,1),v(3,2)); add_square(sum,result);
  sum = sum5(v(3,-2),v(2,-1),v(0,0),v(-2,1),v(-3,2)); add_square(sum,result);
  sum = sum5(v(-4,2),v(-2,1),v(0,0),v(2,-1),v(4,-2)); add_square(sum,result);
  sum = sum5(v(-4,-2),v(-2,-1),v(0,0),v(2,1),v(4,2)); add_square(sum,result);
  sum = sum5(v(-2,-4),v(-1,-2),v(0,0),v(1,2),v(2,4)); add_square(sum,result);
  sum = sum5(v(2,-4),v(1,-2),v(0,0),v(-1,2),v(-2,4)); add_square(sum,result);
  return result;
}

inline float sum7(float a,float b,float c,float d,float e,float f,float g) {
  return ((a+b)+(c+((d+e)+(f+g))));
}

inline float sum9(float a,float b,float c,float d,float e,float f,float g,float h,float i) {
  return (((a+b)+(c+d))+((e+f)+(g+h)))+i;
}

inline float malta_full(device const float* input, int x, int y,
                        constant MaltaResponseParams& p) {
  const auto v = [&](int dx, int dy) { return read_zero(input, x + dx, y + dy, p); };
  float sum = sum9(v(-4,0),v(-3,0),v(-2,0),v(-1,0),v(0,0),v(1,0),v(2,0),v(3,0),v(4,0));
  float result = sum * sum;
  sum=sum9(v(0,-4),v(0,-3),v(0,-2),v(0,-1),v(0,0),v(0,1),v(0,2),v(0,3),v(0,4)); add_square(sum,result);
  sum=sum7(v(-3,-3),v(-2,-2),v(-1,-1),v(0,0),v(1,1),v(2,2),v(3,3)); add_square(sum,result);
  sum=sum7(v(3,-3),v(2,-2),v(1,-1),v(0,0),v(-1,1),v(-2,2),v(-3,3)); add_square(sum,result);
  sum=sum9(v(1,-4),v(1,-3),v(1,-2),v(0,-1),v(0,0),v(0,1),v(-1,2),v(-1,3),v(-1,4)); add_square(sum,result);
  sum=sum9(v(-1,-4),v(-1,-3),v(-1,-2),v(0,-1),v(0,0),v(0,1),v(1,2),v(1,3),v(1,4)); add_square(sum,result);
  sum=sum9(v(-4,-1),v(-3,-1),v(-2,-1),v(-1,0),v(0,0),v(1,0),v(2,1),v(3,1),v(4,1)); add_square(sum,result);
  sum=sum9(v(-4,1),v(-3,1),v(-2,1),v(-1,0),v(0,0),v(1,0),v(2,-1),v(3,-1),v(4,-1)); add_square(sum,result);
  sum=sum7(v(-2,-3),v(-1,-2),v(-1,-1),v(0,0),v(1,1),v(1,2),v(2,3)); add_square(sum,result);
  sum=sum7(v(2,-3),v(1,-2),v(1,-1),v(0,0),v(-1,1),v(-1,2),v(-2,3)); add_square(sum,result);
  sum=sum7(v(-3,-2),v(-2,-1),v(-1,-1),v(0,0),v(1,1),v(2,1),v(3,2)); add_square(sum,result);
  sum=sum7(v(3,-2),v(2,-1),v(1,-1),v(0,0),v(-1,1),v(-2,1),v(-3,2)); add_square(sum,result);
  sum=sum9(v(-4,1),v(-3,1),v(-2,1),v(-1,0),v(0,0),v(1,0),v(2,-1),v(3,-1),v(4,-1)); add_square(sum,result);
  sum=sum9(v(-4,-1),v(-3,-1),v(-2,-1),v(-1,0),v(0,0),v(1,0),v(2,1),v(3,1),v(4,1)); add_square(sum,result);
  sum=sum9(v(-1,-4),v(-1,-3),v(-1,-2),v(0,-1),v(0,0),v(0,1),v(1,2),v(1,3),v(1,4)); add_square(sum,result);
  sum=sum9(v(1,-4),v(1,-3),v(1,-2),v(0,-1),v(0,0),v(0,1),v(-1,2),v(-1,3),v(-1,4)); add_square(sum,result);
  return result;
}

kernel void gjxl_butteraugli_malta_response_f32(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  device float* accumulation [[buffer(2)]],
  constant MaltaResponseParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const float result = params.low_frequency != 0
    ? malta_lf(input, int(position.x), int(position.y), params)
    : malta_full(input, int(position.x), int(position.y), params);
  output[position.y * params.output_stride + position.x] = result;
  const uint accumulation_index =
    position.y * params.accumulation_stride + position.x;
  if (params.initialize_accumulation != 0) {
    accumulation[accumulation_index] = result;
  } else {
    accumulation[accumulation_index] += result;
  }
}

inline float l2_asymmetric(float value0, float value1, float weight_up,
                           float weight_down, float total) {
  const float difference = value0 - value1;
  total = unfused_multiply_add(difference * difference,
                               weight_up * 0.8f, total);
  const float magnitude = abs(value0);
  const float too_small = 0.4f * magnitude;
  float secondary = 0.0f;
  if (value0 < 0.0f) {
    if (value1 > -too_small) secondary = value1 + too_small;
    else if (value1 < -magnitude) secondary = -value1 - magnitude;
  } else if (value1 < too_small) secondary = too_small - value1;
  else if (value1 > magnitude) secondary = value1 - magnitude;
  return unfused_multiply_add(weight_down * 0.8f,
                              secondary * secondary, total);
}

kernel void gjxl_butteraugli_l2_f32(
  device const float* rlow0 [[buffer(0)]], device const float* rlow1 [[buffer(1)]],
  device const float* rlow2 [[buffer(2)]], device const float* rmed0 [[buffer(3)]],
  device const float* rmed1 [[buffer(4)]], device const float* rmed2 [[buffer(5)]],
  device const float* rhigh0 [[buffer(6)]], device const float* rhigh1 [[buffer(7)]],
  device const float* dlow0 [[buffer(8)]], device const float* dlow1 [[buffer(9)]],
  device const float* dlow2 [[buffer(10)]], device const float* dmed0 [[buffer(11)]],
  device const float* dmed1 [[buffer(12)]], device const float* dmed2 [[buffer(13)]],
  device const float* dhigh0 [[buffer(14)]], device const float* dhigh1 [[buffer(15)]],
  device float* ac0 [[buffer(16)]], device float* ac1 [[buffer(17)]],
  device float* ac2 [[buffer(18)]], device float* dc0 [[buffer(19)]],
  device float* dc1 [[buffer(20)]], device float* dc2 [[buffer(21)]],
  constant DifferenceParams& params [[buffer(22)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint reference_index =
    position.y * params.reference_stride + position.x;
  const uint distorted_index =
    position.y * params.distorted_stride + position.x;
  const uint work_index = position.y * params.work_stride + position.x;
  const float inv_asymmetry = 1.0f / params.asymmetry;
  float total0 = l2_asymmetric(rhigh0[reference_index],
                               dhigh0[distorted_index],
                               400.0f * params.asymmetry,
                               400.0f * inv_asymmetry, ac0[work_index]);
  float total1 = l2_asymmetric(rhigh1[reference_index],
                               dhigh1[distorted_index],
                               1.50815703118f * params.asymmetry,
                               1.50815703118f * inv_asymmetry,
                               ac1[work_index]);
  const float md0 = rmed0[reference_index] - dmed0[distorted_index];
  const float md1 = rmed1[reference_index] - dmed1[distorted_index];
  const float md2 = rmed2[reference_index] - dmed2[distorted_index];
  ac0[work_index] = unfused_multiply_add(md0 * md0, 2150.0f, total0);
  ac1[work_index] = unfused_multiply_add(md1 * md1, 10.6195433239f, total1);
  ac2[work_index] = md2 * md2 * 16.2176043152f;
  const float ld0 = rlow0[reference_index] - dlow0[distorted_index];
  const float ld1 = rlow1[reference_index] - dlow1[distorted_index];
  const float ld2 = rlow2[reference_index] - dlow2[distorted_index];
  dc0[work_index] = ld0 * ld0 * 29.2353797994f;
  dc1[work_index] = ld1 * ld1 * 0.844626970982f;
  dc2[work_index] = ld2 * ld2 * 0.703646627719f;
}

kernel void gjxl_butteraugli_mask_precompute_f32(
  device const float* high_x [[buffer(0)]],
  device const float* high_y [[buffer(1)]],
  device const float* ultra_x [[buffer(2)]],
  device const float* ultra_y [[buffer(3)]],
  device float* output [[buffer(4)]],
  constant PlaneParams& params [[buffer(5)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint input_index = position.y * params.input_stride + position.x;
  const uint output_index = position.y * params.output_stride + position.x;
  const float xdiff = (ultra_x[input_index] + high_x[input_index]) * 2.5f;
  const float ydiff = ultra_y[input_index] * 0.4f + high_y[input_index] * 0.4f;
  const float activity = sqrt(xdiff * xdiff + ydiff * ydiff);
  constexpr float kMultiplier = 6.19424080439f;
  constexpr float kBias = kMultiplier * 12.61050594197f;
  output[output_index] =
    sqrt(kMultiplier * abs(activity) + kBias) - sqrt(kBias);
}

inline void store_min3(float value, thread float& minimum0,
                       thread float& minimum1, thread float& minimum2) {
  if (value < minimum2) {
    if (value < minimum0) {
      minimum2 = minimum1; minimum1 = minimum0; minimum0 = value;
    } else if (value < minimum1) {
      minimum2 = minimum1; minimum1 = value;
    } else minimum2 = value;
  }
}

kernel void gjxl_butteraugli_fuzzy_erosion_f32(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant PlaneParams& params [[buffer(2)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  constexpr int step = 3;
  const int x = int(position.x), y = int(position.y);
  float minimum0 = input[position.y * params.input_stride + position.x];
  float minimum1 = 2.0f * minimum0, minimum2 = minimum1;
  for (int dy = -step; dy <= step; dy += step) {
    for (int dx = -step; dx <= step; dx += step) {
      if (dx == 0 && dy == 0) continue;
      const int sx = x + dx, sy = y + dy;
      if (sx >= 0 && sy >= 0 && sx < int(params.width) &&
          sy < int(params.height)) {
        store_min3(input[uint(sy) * params.input_stride + uint(sx)],
                   minimum0, minimum1, minimum2);
      }
    }
  }
  output[position.y * params.output_stride + position.x] =
    0.45f * minimum0 + 0.3f * minimum1 + 0.25f * minimum2;
}

kernel void gjxl_butteraugli_masked_ac_f32(
  device const float* reference [[buffer(0)]],
  device const float* distorted [[buffer(1)]],
  device float* ac_y [[buffer(2)]],
  constant PlaneParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint input_index = position.y * params.input_stride + position.x;
  const uint output_index = position.y * params.output_stride + position.x;
  const float difference = reference[input_index] - distorted[input_index];
  ac_y[output_index] += 10.0f * difference * difference;
}

inline float mask_y(float delta) {
  constexpr float global_scale = 1.0f / (17.83f * 0.79079917404f);
  const float value = global_scale *
    (1.0f + 2.5485944793f / (0.451936922203f * delta + 0.829591754942f));
  return value * value;
}

inline float mask_dc_y(float delta) {
  constexpr float global_scale = 1.0f / (17.83f * 0.79079917404f);
  const float value = global_scale *
    (1.0f + 0.505054525019f / (3.87449418804f * delta + 0.20025578522f));
  return value * value;
}

kernel void gjxl_butteraugli_final_f32(
  device const float* dc0 [[buffer(0)]], device const float* dc1 [[buffer(1)]],
  device const float* dc2 [[buffer(2)]], device const float* ac0 [[buffer(3)]],
  device const float* ac1 [[buffer(4)]], device const float* ac2 [[buffer(5)]],
  device const float* mask [[buffer(6)]], device float* output [[buffer(7)]],
  constant FinalParams& params [[buffer(8)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const uint index = position.y * params.stride + position.x;
  const float mask_value = mask_y(mask[index]);
  const float dc_mask_value = mask_dc_y(mask[index]);
  const float masked_dc = dc0[index] * params.x_multiplier * dc_mask_value +
                          dc1[index] * dc_mask_value + dc2[index] * dc_mask_value;
  const float masked_ac = ac0[index] * params.x_multiplier * mask_value +
                          ac1[index] * mask_value + ac2[index] * mask_value;
  const float result = sqrt(masked_dc + masked_ac);
  output[position.y * params.output_stride + position.x] =
    isfinite(result) && result >= 0.0f ? result : NAN;
}

kernel void gjxl_butteraugli_crop_f32(
  device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
  constant CropParams& params [[buffer(2)]],
  uint2 position [[thread_position_in_grid]]) {
  if (position.x >= params.width || position.y >= params.height) return;
  output[position.y * params.output_stride + position.x] =
    input[(position.y + params.yborder) * params.input_stride +
          position.x + params.xborder];
}

kernel void gjxl_butteraugli_compose_f32(
  device const float* main_map [[buffer(0)]],
  device const float* sub_map [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant ComposeParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {
  if (position.x >= params.width || position.y >= params.height) return;
  const float main_value = main_map[position.y * params.main_stride + position.x];
  const float sub_value = sub_map[(position.y / 2) * params.sub_stride + position.x / 2];
  output[position.y * params.output_stride + position.x] =
    main_value * 0.85f + 0.5f * sub_value;
}

kernel void gjxl_butteraugli_reduce_max_f32(
  device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
  constant ReductionParams& params [[buffer(2)]],
  uint thread_index [[thread_index_in_threadgroup]],
  uint3 group_position [[threadgroup_position_in_grid]]) {
  constexpr uint width = 256;
  threadgroup float values[width];
  const uint index = group_position.x * width + thread_index;
  float value = -INFINITY;
  if (index < params.input_count) {
    const uint y = index / params.width;
    const uint x = index - y * params.width;
    value = input[y * params.input_stride + x];
    if (!isfinite(value) || value < 0.0f) value = NAN;
  }
  values[thread_index] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = width / 2; step != 0; step /= 2) {
    if (thread_index < step) {
      const float other = values[thread_index + step];
      values[thread_index] = isnan(values[thread_index]) || isnan(other)
        ? NAN : max(values[thread_index], other);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_index == 0) output[group_position.x] = values[0];
}
