// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// The opsin, frequency, Malta, masking, and multiscale computations are
// adapted from pinned JPEG XL Butteraugli code distributed under its
// BSD-style license. See third_party/libjxl/LICENSE.

#include <cstddef>
#include <cstdint>

#include "gpu/cuda/cuda_butteraugli_kernels.h"

namespace gjxl::cuda_internal {
namespace {

constexpr unsigned int kPlaneThreads = 256;
constexpr unsigned int kReductionWidth = 256;
constexpr unsigned int kMaltaTileWidth = 32;
constexpr unsigned int kMaltaTileHeight = 8;
constexpr unsigned int kMaltaRadius = 4;
constexpr unsigned int kMaltaTileStride = kMaltaTileWidth + 2 * kMaltaRadius;
constexpr unsigned int kImage = 21;
constexpr unsigned int kAc = kImage;
constexpr unsigned int kPsychoWork = 24;
// After psycho construction only Malta AC[0:2] remains live in image work.
constexpr unsigned int kMaskInput = 23;
constexpr unsigned int kMaskIntermediate = 24;
constexpr unsigned int kReferenceMask = 25;
// The separable distorted-mask blur may overwrite its own input. Its
// horizontal intermediate is dead before the cropped/subscale map is written.
constexpr unsigned int kDistortedMask = kMaskInput;
constexpr unsigned int kFinalStaging = kMaskIntermediate;
static_assert(kPsychoWork + 3 == kCudaButteraugliWorkingPlaneCount);

struct PlaneParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
};

struct ExpandParams {
  uint32_t input_width;
  uint32_t input_height;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t xborder;
  uint32_t yborder;
};

struct SubsampleParams {
  uint32_t input_width;
  uint32_t input_height;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t input_stride;
  uint32_t output_stride;
};

struct OpsinParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride[3];
  uint32_t blurred_stride;
  uint32_t output_stride;
  float intensity_target;
};

struct OpsinConvolutionPlan {
  const float* input[3];
  const float* intermediate[3];
  float* output[3];
  const float* weights;
  uint32_t input_stride[3];
  uint32_t width;
  uint32_t height;
  uint32_t output_stride;
  float intensity_target;
};

struct FrequencyParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t low_stride;
  uint32_t output_stride;
  uint32_t channel;
};

struct LowMediumParams {
  uint32_t width;
  uint32_t height;
  uint32_t xyb_stride;
  uint32_t blurred_stride;
  uint32_t psycho_stride;
};

struct MaltaScaleParams {
  uint32_t width;
  uint32_t height;
  uint32_t reference_stride;
  uint32_t distorted_stride;
  uint32_t output_stride;
  uint32_t low_frequency;
  float norm2_0_gt_1;
  float norm2_0_lt_1;
  float norm;
};

struct MaltaResponseParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t accumulation_stride;
  uint32_t low_frequency;
  uint32_t initialize_accumulation;
};

struct DifferenceParams {
  uint32_t width;
  uint32_t height;
  uint32_t reference_stride;
  uint32_t distorted_stride;
  uint32_t work_stride;
  float asymmetry;
};

struct DifferencePlan {
  const float* reference[8];
  const float* distorted[8];
  float* ac[3];
  float* dc[3];
  DifferenceParams params;
};

struct FinalParams {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  uint32_t output_stride;
  float x_multiplier;
};

__device__ float UnfusedMultiplyAdd(float multiplier, float value,
                                    float addend) {
  return __fadd_rn(__fmul_rn(multiplier, value), addend);
}

__device__ int MirrorCoordinate(int coordinate, int size) {
  while (coordinate < 0 || coordinate >= size) {
    coordinate = coordinate < 0 ? -coordinate - 1 : 2 * size - 1 - coordinate;
  }
  return coordinate;
}

__device__ size_t PlaneIndex(PlaneParams params) {
  return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__global__ void ExpandKernel(const float* input, float* output,
                             ExpandParams params) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count =
      static_cast<size_t>(params.output_width) * params.output_height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.output_width);
  const uint32_t x = static_cast<uint32_t>(index - static_cast<size_t>(y) *
                                                       params.output_width);
  const uint32_t source_x =
      min(params.input_width - 1, x > params.xborder ? x - params.xborder : 0u);
  const uint32_t source_y = min(params.input_height - 1,
                                y > params.yborder ? y - params.yborder : 0u);
  output[static_cast<size_t>(y) * params.output_stride + x] =
      input[static_cast<size_t>(source_y) * params.input_stride + source_x];
}

__global__ void SubsampleKernel(const float* input, float* output,
                                SubsampleParams params) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count =
      static_cast<size_t>(params.output_width) * params.output_height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.output_width);
  const uint32_t x = static_cast<uint32_t>(index - static_cast<size_t>(y) *
                                                       params.output_width);
  const uint32_t source_x = x * 2;
  const uint32_t source_y = y * 2;
  float value =
      0.25f *
      input[static_cast<size_t>(source_y) * params.input_stride + source_x];
  if (source_x + 1 < params.input_width) {
    value += 0.25f * input[static_cast<size_t>(source_y) * params.input_stride +
                           source_x + 1];
  }
  if (source_y + 1 < params.input_height) {
    value +=
        0.25f * input[static_cast<size_t>(source_y + 1) * params.input_stride +
                      source_x];
    if (source_x + 1 < params.input_width) {
      value += 0.25f *
               input[static_cast<size_t>(source_y + 1) * params.input_stride +
                     source_x + 1];
    }
  }
  if ((params.input_width & 1u) != 0 && x + 1 == params.output_width) {
    value *= 2.0f;
  }
  if ((params.input_height & 1u) != 0 && y + 1 == params.output_height) {
    value *= 2.0f;
  }
  output[static_cast<size_t>(y) * params.output_stride + x] = value;
}

template <bool Horizontal>
__global__ void MirroredConvolution5Kernel(const float* input,
                                           const float* weights, float* output,
                                           uint32_t width, uint32_t height,
                                           uint32_t input_stride,
                                           uint32_t output_stride) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(width) * height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * width);
  constexpr int radius = 2;
  const int center = static_cast<int>(Horizontal ? x : y);
  const int limit = static_cast<int>(Horizontal ? width : height);
  float sum = 0.0f;
  float weight_sum = 0.0f;
#pragma unroll
  for (int delta = -radius; delta <= radius; ++delta) {
    const int coordinate = MirrorCoordinate(center + delta, limit);
    const float weight = weights[delta + radius];
    const uint32_t source_x =
        Horizontal ? static_cast<uint32_t>(coordinate) : x;
    const uint32_t source_y =
        Horizontal ? y : static_cast<uint32_t>(coordinate);
    sum +=
        input[static_cast<size_t>(source_y) * input_stride + source_x] * weight;
    weight_sum += weight;
  }
  output[static_cast<size_t>(y) * output_stride + x] = sum / weight_sum;
}

template <bool Horizontal>
struct ConvolutionTile {
  static constexpr unsigned int kWidth = Horizontal ? 256 : 32;
  static constexpr unsigned int kHeight = Horizontal ? 4 : 64;

  static unsigned int Blocks(uint32_t width, uint32_t height) {
    return ((width + kWidth - 1) / kWidth) * ((height + kHeight - 1) / kHeight);
  }
};

template <bool Horizontal, unsigned int KernelSize, typename Store>
__device__ __forceinline__ void ConvolutionTiledBody(
    const float* input, const float* weights, uint32_t width, uint32_t height,
    uint32_t input_stride, Store store) {
  static_assert(KernelSize == 7 || KernelSize == 13 || KernelSize == 15 ||
                KernelSize == 33);
  constexpr unsigned int kTileWidth = ConvolutionTile<Horizontal>::kWidth;
  constexpr unsigned int kTileHeight = ConvolutionTile<Horizontal>::kHeight;
  constexpr unsigned int kRadius = KernelSize / 2;
  constexpr unsigned int kInputWidth =
      kTileWidth + (Horizontal ? 2 * kRadius : 0);
  constexpr unsigned int kInputHeight =
      kTileHeight + (Horizontal ? 0 : 2 * kRadius);
  __shared__ float tile[kInputWidth * kInputHeight];
  __shared__ float kernel[KernelSize];
  __shared__ float normalization;
  const uint32_t tile_columns = (width + kTileWidth - 1) / kTileWidth;
  const uint32_t origin_x = (blockIdx.x % tile_columns) * kTileWidth;
  const uint32_t origin_y = (blockIdx.x / tile_columns) * kTileHeight;
  if (threadIdx.x < KernelSize) kernel[threadIdx.x] = weights[threadIdx.x];
  // Interior pixels share the same normalization. Sum it in the original
  // order once per block; edge pixels still sum only the included weights.
  if (threadIdx.x == 0) {
    float weight_sum = 0.0f;
    for (unsigned int tap = 0; tap < KernelSize; ++tap) {
      weight_sum += weights[tap];
    }
    normalization = weight_sum;
  }
  // Load the directional halo cooperatively. Partial-tile threads must reach
  // the barrier before each lane evaluates its four or eight output pixels.
  for (unsigned int index = threadIdx.x; index < kInputWidth * kInputHeight;
       index += blockDim.x) {
    const int x = static_cast<int>(origin_x + index % kInputWidth) -
                  static_cast<int>(Horizontal ? kRadius : 0);
    const int y = static_cast<int>(origin_y + index / kInputWidth) -
                  static_cast<int>(Horizontal ? 0 : kRadius);
    const bool valid = x >= 0 && y >= 0 && x < static_cast<int>(width) &&
                       y < static_cast<int>(height);
    tile[index] =
        valid ? input[static_cast<size_t>(y) * input_stride + x] : 0.0f;
  }
  __syncthreads();
  for (unsigned int index = threadIdx.x; index < kTileWidth * kTileHeight;
       index += blockDim.x) {
    const uint32_t local_x = index % kTileWidth;
    const uint32_t local_y = index / kTileWidth;
    const uint32_t x = origin_x + local_x;
    const uint32_t y = origin_y + local_y;
    if (x >= width || y >= height) continue;
    const int center = static_cast<int>(Horizontal ? x : y);
    const int limit = static_cast<int>(Horizontal ? width : height);
    const float* first = tile + local_y * kInputWidth + local_x;
    constexpr unsigned int kStep = Horizontal ? 1 : kInputWidth;
    float sum = 0.0f;
    float weight_sum = normalization;
    if (center >= static_cast<int>(kRadius) &&
        center + static_cast<int>(kRadius) < limit) {
#pragma unroll
      for (unsigned int tap = 0; tap < KernelSize; ++tap) {
        sum += first[tap * kStep] * kernel[tap];
      }
    } else {
      weight_sum = 0.0f;
#pragma unroll
      for (unsigned int tap = 0; tap < KernelSize; ++tap) {
        const int coordinate =
            center + static_cast<int>(tap) - static_cast<int>(kRadius);
        if (coordinate >= 0 && coordinate < limit) {
          sum += first[tap * kStep] * kernel[tap];
          weight_sum += kernel[tap];
        }
      }
    }
    store(x, y, sum / weight_sum);
  }
}

// Joint horizontal33 keeps each channel's sum and rounded division separate.
// Every lane participates in the halo load and barrier, including partial tiles.
template<unsigned Width, unsigned Height>
__global__ void ConvolutionHorizontal3Kernel(
    const float* input0, const float* input1, const float* input2,
    const float* weights, float* output0, float* output1, float* output2,
    uint32_t width, uint32_t height, uint32_t input_stride) {
  constexpr unsigned InputWidth = Width + 32;
  constexpr unsigned InputSize = InputWidth * Height;
  __shared__ float tile0[InputSize], tile1[InputSize], tile2[InputSize];
  __shared__ float kernel[33];
  __shared__ float normalization;
  const uint32_t columns = (width + Width - 1) / Width;
  const uint32_t origin_x = (blockIdx.x % columns) * Width;
  const uint32_t origin_y = (blockIdx.x / columns) * Height;
  if (threadIdx.x < 33) kernel[threadIdx.x] = weights[threadIdx.x];
  if (threadIdx.x == 0) {
    float weight_sum = 0.0f;
    for (unsigned tap = 0; tap < 33; ++tap) weight_sum += weights[tap];
    normalization = weight_sum;
  }
  for (unsigned index = threadIdx.x; index < InputSize; index += blockDim.x) {
    const int x = static_cast<int>(origin_x + index % InputWidth) - 16;
    const uint32_t y = origin_y + index / InputWidth;
    const bool valid = x >= 0 && x < static_cast<int>(width) && y < height;
    const size_t source = static_cast<size_t>(y) * input_stride + x;
    tile0[index] = valid ? input0[source] : 0.0f;
    tile1[index] = valid ? input1[source] : 0.0f;
    tile2[index] = valid ? input2[source] : 0.0f;
  }
  __syncthreads();
  for (unsigned index = threadIdx.x; index < Width * Height; index += blockDim.x) {
    const uint32_t local_x = index % Width;
    const uint32_t local_y = index / Width;
    const uint32_t x = origin_x + local_x, y = origin_y + local_y;
    if (x >= width || y >= height) continue;
    const unsigned first = local_y * InputWidth + local_x;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    float weight_sum = normalization;
    if (x >= 16 && static_cast<size_t>(x) + 16 < width) {
#pragma unroll
      for (unsigned tap = 0; tap < 33; ++tap) {
        sum0 += tile0[first + tap] * kernel[tap];
        sum1 += tile1[first + tap] * kernel[tap];
        sum2 += tile2[first + tap] * kernel[tap];
      }
    } else {
      weight_sum = 0.0f;
#pragma unroll
      for (unsigned tap = 0; tap < 33; ++tap) {
        const int coordinate = static_cast<int>(x) + static_cast<int>(tap) - 16;
        if (coordinate >= 0 && coordinate < static_cast<int>(width)) {
          sum0 += tile0[first + tap] * kernel[tap];
          sum1 += tile1[first + tap] * kernel[tap];
          sum2 += tile2[first + tap] * kernel[tap];
          weight_sum += kernel[tap];
        }
      }
    }
    const size_t destination = static_cast<size_t>(y) * width + x;
    output0[destination] = sum0 / weight_sum;
    output1[destination] = sum1 / weight_sum;
    output2[destination] = sum2 / weight_sum;
  }
}

struct StoreConvolution {
  float* output;
  uint32_t stride;
  __device__ void operator()(uint32_t x, uint32_t y, float value) const {
    output[static_cast<size_t>(y) * stride + x] = value;
  }
};

template <bool Horizontal, unsigned int KernelSize>
__global__ void ConvolutionTiledKernel(const float* input, const float* weights,
                                       float* output, uint32_t width,
                                       uint32_t height, uint32_t input_stride,
                                       uint32_t output_stride) {
  ConvolutionTiledBody<Horizontal, KernelSize>(
      input, weights, width, height, input_stride,
      StoreConvolution{output, output_stride});
}

__device__ float ButteraugliFastLog2(float value) {
  const uint32_t value_bits = __float_as_uint(value);
  const int shifted_exponent = static_cast<int>(value_bits - 0x3f2aaaabu) >> 23;
  const uint32_t mantissa_bits =
      value_bits - (static_cast<uint32_t>(shifted_exponent) << 23);
  const float x = __uint_as_float(mantissa_bits) - 1.0f;
  float numerator =
      UnfusedMultiplyAdd(0.74245873327820566f, x, 1.4287160470083755f);
  numerator = UnfusedMultiplyAdd(numerator, x, -1.8503833400518310e-06f);
  float denominator =
      UnfusedMultiplyAdd(0.17409343003366853f, x, 1.0096718572241148f);
  denominator = UnfusedMultiplyAdd(denominator, x, 0.99032814277590719f);
  return numerator / denominator + static_cast<float>(shifted_exponent);
}

__device__ float GammaValue(float value) {
  constexpr float kRetMul = 19.245013259874995f * 0.6931471805599453f;
  return UnfusedMultiplyAdd(
      kRetMul, ButteraugliFastLog2(fmaxf(value, 0.0f) + 9.9710635769299145f),
      -23.16046239805755f);
}

__device__ float3 OpsinAbsorbance(float red, float green, float blue,
                                  bool clamp_result) {
  float3 output;
  output.x = UnfusedMultiplyAdd(
      0.29956550340058319f, red,
      UnfusedMultiplyAdd(0.63373087833825936f, green,
                         UnfusedMultiplyAdd(0.077705617820981968f, blue,
                                            1.7557483643287353f)));
  output.y = UnfusedMultiplyAdd(
      0.22158691104574774f, red,
      UnfusedMultiplyAdd(
          0.69391388044116142f, green,
          UnfusedMultiplyAdd(0.0987313588422f, blue, 1.7557483643287353f)));
  output.z = UnfusedMultiplyAdd(
      0.02f, red,
      UnfusedMultiplyAdd(
          0.02f, green,
          UnfusedMultiplyAdd(0.20480129041026129f, blue, 12.226454707163354f)));
  if (clamp_result) {
    output.x = fmaxf(output.x, 1.7557483643287353f);
    output.y = fmaxf(output.y, 1.7557483643287353f);
    output.z = fmaxf(output.z, 12.226454707163354f);
  }
  return output;
}

__global__ void OpsinKernel(const float* input0, const float* input1,
                            const float* input2, const float* blurred0,
                            const float* blurred1, const float* blurred2,
                            float* output0, float* output1, float* output2,
                            OpsinParams params) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t blurred_index =
      static_cast<size_t>(y) * params.blurred_stride + x;
  const float3 blurred_rgb =
      make_float3(blurred0[blurred_index], blurred1[blurred_index],
                  blurred2[blurred_index]);
  const float3 input_rgb =
      make_float3(input0[static_cast<size_t>(y) * params.input_stride[0] + x],
                  input1[static_cast<size_t>(y) * params.input_stride[1] + x],
                  input2[static_cast<size_t>(y) * params.input_stride[2] + x]);
  if (!isfinite(input_rgb.x) || !isfinite(input_rgb.y) ||
      !isfinite(input_rgb.z) || !isfinite(blurred_rgb.x) ||
      !isfinite(blurred_rgb.y) || !isfinite(blurred_rgb.z)) {
    output0[blurred_index] = NAN;
    output1[blurred_index] = NAN;
    output2[blurred_index] = NAN;
    return;
  }
  float3 pre = OpsinAbsorbance(blurred_rgb.x * params.intensity_target,
                               blurred_rgb.y * params.intensity_target,
                               blurred_rgb.z * params.intensity_target, true);
  pre.x = fmaxf(pre.x, 1.0e-4f);
  pre.y = fmaxf(pre.y, 1.0e-4f);
  pre.z = fmaxf(pre.z, 1.0e-4f);
  const float3 sensitivity =
      make_float3(fmaxf(GammaValue(pre.x) / pre.x, 1.0e-4f),
                  fmaxf(GammaValue(pre.y) / pre.y, 1.0e-4f),
                  fmaxf(GammaValue(pre.z) / pre.z, 1.0e-4f));
  float3 current =
      OpsinAbsorbance(input_rgb.x * params.intensity_target,
                      input_rgb.y * params.intensity_target,
                      input_rgb.z * params.intensity_target, false);
  current.x = fmaxf(current.x * sensitivity.x, 1.7557483643287353f);
  current.y = fmaxf(current.y * sensitivity.y, 1.7557483643287353f);
  current.z = fmaxf(current.z * sensitivity.z, 12.226454707163354f);
  output0[blurred_index] = current.x - current.y;
  output1[blurred_index] = current.x + current.y;
  output2[blurred_index] = current.z;
}

// Keep OpsinKernel unchanged as the separate-pass arithmetic oracle.
__device__ float3 OpsinFromBlurredRgb(float3 input_rgb, float3 blurred_rgb,
                                     float intensity_target) {
  if (!isfinite(input_rgb.x) || !isfinite(input_rgb.y) ||
      !isfinite(input_rgb.z) || !isfinite(blurred_rgb.x) ||
      !isfinite(blurred_rgb.y) || !isfinite(blurred_rgb.z)) {
    return make_float3(NAN, NAN, NAN);
  }
  float3 pre = OpsinAbsorbance(blurred_rgb.x * intensity_target,
                               blurred_rgb.y * intensity_target,
                               blurred_rgb.z * intensity_target, true);
  pre.x = fmaxf(pre.x, 1.0e-4f);
  pre.y = fmaxf(pre.y, 1.0e-4f);
  pre.z = fmaxf(pre.z, 1.0e-4f);
  const float3 sensitivity =
      make_float3(fmaxf(GammaValue(pre.x) / pre.x, 1.0e-4f),
                  fmaxf(GammaValue(pre.y) / pre.y, 1.0e-4f),
                  fmaxf(GammaValue(pre.z) / pre.z, 1.0e-4f));
  float3 current =
      OpsinAbsorbance(input_rgb.x * intensity_target,
                      input_rgb.y * intensity_target,
                      input_rgb.z * intensity_target, false);
  current.x = fmaxf(current.x * sensitivity.x, 1.7557483643287353f);
  current.y = fmaxf(current.y * sensitivity.y, 1.7557483643287353f);
  current.z = fmaxf(current.z * sensitivity.z, 12.226454707163354f);
  return make_float3(current.x - current.y, current.x + current.y, current.z);
}

template <unsigned int TileHeight>
__global__ void ConvolutionOpsinKernel(OpsinConvolutionPlan plan) {
  constexpr unsigned int kWidth = 32;
  constexpr unsigned int kInputHeight = TileHeight + 4;
  __shared__ float tile[3][kWidth * kInputHeight];
  __shared__ float weights[5];
  __shared__ float normalization;
  const uint32_t columns = (plan.width + kWidth - 1) / kWidth;
  const uint32_t origin_x = (blockIdx.x % columns) * kWidth;
  const uint32_t origin_y = (blockIdx.x / columns) * TileHeight;
  if (threadIdx.x < 5) weights[threadIdx.x] = plan.weights[threadIdx.x];
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (unsigned int tap = 0; tap < 5; ++tap) sum += plan.weights[tap];
    normalization = sum;
  }
  // Reflect the halo, including repeated reflection for one-pixel extents.
  // Partial-tile lanes must participate in loading and reach the barrier.
  for (unsigned int index = threadIdx.x; index < kWidth * kInputHeight;
       index += blockDim.x) {
    const uint32_t x = origin_x + index % kWidth;
    const int y = MirrorCoordinate(static_cast<int>(origin_y + index / kWidth) - 2,
                                    static_cast<int>(plan.height));
    for (unsigned int channel = 0; channel < 3; ++channel) {
      tile[channel][index] = x < plan.width
          ? plan.intermediate[channel][static_cast<size_t>(y) * plan.width + x]
          : 0.0f;
    }
  }
  __syncthreads();
  for (unsigned int index = threadIdx.x; index < kWidth * TileHeight;
       index += blockDim.x) {
    const uint32_t x = origin_x + index % kWidth;
    const uint32_t y = origin_y + index / kWidth;
    if (x >= plan.width || y >= plan.height) continue;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
#pragma unroll
    for (unsigned int tap = 0; tap < 5; ++tap) {
      sum0 += tile[0][index + tap * kWidth] * weights[tap];
      sum1 += tile[1][index + tap * kWidth] * weights[tap];
      sum2 += tile[2][index + tap * kWidth] * weights[tap];
    }
    const float3 blurred = make_float3(sum0 / normalization, sum1 / normalization,
                                       sum2 / normalization);
    const float3 input = make_float3(
        plan.input[0][static_cast<size_t>(y) * plan.input_stride[0] + x],
        plan.input[1][static_cast<size_t>(y) * plan.input_stride[1] + x],
        plan.input[2][static_cast<size_t>(y) * plan.input_stride[2] + x]);
    const float3 output = OpsinFromBlurredRgb(input, blurred, plan.intensity_target);
    const size_t destination = static_cast<size_t>(y) * plan.output_stride + x;
    plan.output[0][destination] = output.x;
    plan.output[1][destination] = output.y;
    plan.output[2][destination] = output.z;
  }
}

__global__ void LowMediumKernel(const float* xyb0, const float* xyb1,
                                const float* xyb2, const float* blurred0,
                                const float* blurred1, const float* blurred2,
                                float* low0, float* low1, float* low2,
                                float* medium0, float* medium1, float* medium2,
                                LowMediumParams params) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t source = static_cast<size_t>(y) * params.xyb_stride + x;
  const size_t blurred = static_cast<size_t>(y) * params.blurred_stride + x;
  const size_t output = static_cast<size_t>(y) * params.psycho_stride + x;
  const float bx = blurred0[blurred];
  const float by = blurred1[blurred];
  const float bb = blurred2[blurred];
  medium0[output] = xyb0[source] - bx;
  medium1[output] = xyb1[source] - by;
  medium2[output] = xyb2[source] - bb;
  low0[output] = bx * 33.832837186260f;
  low1[output] = by * 14.458268100570f;
  low2[output] = UnfusedMultiplyAdd(-0.362267051518f, by, bb) * 49.87984651440f;
}

template <unsigned int TileHeight>
__global__ void ConvolutionLowMediumKernel(
    const float* xyb0, const float* xyb1, const float* xyb2,
    const float* horizontal0, const float* horizontal1, const float* horizontal2,
    const float* weights, float* low0, float* low1, float* low2,
    float* medium0, float* medium1, float* medium2, LowMediumParams params) {
  constexpr unsigned int kWidth = 32;
  constexpr unsigned int kRadius = 16;
  constexpr unsigned int kInputSize = kWidth * (TileHeight + 2 * kRadius);
  __shared__ float tile0[kInputSize];
  __shared__ float tile1[kInputSize];
  __shared__ float tile2[kInputSize];
  __shared__ float kernel[33];
  __shared__ float normalization;
  const uint32_t columns = (params.width + kWidth - 1) / kWidth;
  const uint32_t origin_x = (blockIdx.x % columns) * kWidth;
  const uint32_t origin_y = (blockIdx.x / columns) * TileHeight;
  if (threadIdx.x < 33) kernel[threadIdx.x] = weights[threadIdx.x];
  if (threadIdx.x == 0) {
    float weight_sum = 0.0f;
    for (unsigned int tap = 0; tap < 33; ++tap) weight_sum += weights[tap];
    normalization = weight_sum;
  }
  for (unsigned int index = threadIdx.x; index < kInputSize; index += blockDim.x) {
    const uint32_t x = origin_x + index % kWidth;
    const int y = static_cast<int>(origin_y + index / kWidth) - static_cast<int>(kRadius);
    const bool valid = x < params.width && y >= 0 && y < static_cast<int>(params.height);
    const size_t source = static_cast<size_t>(y) * params.blurred_stride + x;
    tile0[index] = valid ? horizontal0[source] : 0.0f;
    tile1[index] = valid ? horizontal1[source] : 0.0f;
    tile2[index] = valid ? horizontal2[source] : 0.0f;
  }
  // Partial-tile lanes participate in every cooperative load and barrier.
  __syncthreads();
  for (unsigned int index = threadIdx.x; index < kWidth * TileHeight; index += blockDim.x) {
    const uint32_t local_x = index % kWidth;
    const uint32_t local_y = index / kWidth;
    const uint32_t x = origin_x + local_x;
    const uint32_t y = origin_y + local_y;
    if (x >= params.width || y >= params.height) continue;
    const size_t first = local_y * kWidth + local_x;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    float weight_sum = normalization;
    if (y >= kRadius && static_cast<size_t>(y) + kRadius < params.height) {
#pragma unroll
      for (unsigned int tap = 0; tap < 33; ++tap) {
        sum0 += tile0[first + tap * kWidth] * kernel[tap];
        sum1 += tile1[first + tap * kWidth] * kernel[tap];
        sum2 += tile2[first + tap * kWidth] * kernel[tap];
      }
    } else {
      weight_sum = 0.0f;
#pragma unroll
      for (unsigned int tap = 0; tap < 33; ++tap) {
        const int coordinate = static_cast<int>(y) + static_cast<int>(tap) - static_cast<int>(kRadius);
        if (coordinate >= 0 && coordinate < static_cast<int>(params.height)) {
          sum0 += tile0[first + tap * kWidth] * kernel[tap];
          sum1 += tile1[first + tap * kWidth] * kernel[tap];
          sum2 += tile2[first + tap * kWidth] * kernel[tap];
          weight_sum += kernel[tap];
        }
      }
    }
    // Each channel retains its original tap order and rounded division.
    const float bx = sum0 / weight_sum;
    const float by = sum1 / weight_sum;
    const float bb = sum2 / weight_sum;
    const size_t source = static_cast<size_t>(y) * params.xyb_stride + x;
    const size_t output = static_cast<size_t>(y) * params.psycho_stride + x;
    medium0[output] = xyb0[source] - bx;
    medium1[output] = xyb1[source] - by;
    medium2[output] = xyb2[source] - bb;
    low0[output] = bx * 33.832837186260f;
    low1[output] = by * 14.458268100570f;
    low2[output] = UnfusedMultiplyAdd(-0.362267051518f, by, bb) * 49.87984651440f;
  }
}

template <unsigned TileHeight>
__global__ void ConvolutionLowMediumRowsKernel(
    const float* xyb0, const float* xyb1, const float* xyb2,
    const float* horizontal0, const float* horizontal1, const float* horizontal2,
    const float* weights, float* low0, float* low1, float* low2,
    float* medium0, float* medium1, float* medium2, LowMediumParams params) {
  constexpr unsigned Rows = 3;
  constexpr unsigned Threads = 256;
  constexpr unsigned int kWidth = 32;
  constexpr unsigned int kRadius = 16;
  constexpr unsigned int kInputSize = kWidth * (TileHeight + 2 * kRadius);
  __shared__ float tile0[kInputSize];
  __shared__ float tile1[kInputSize];
  __shared__ float tile2[kInputSize];
  __shared__ float kernel[33];
  __shared__ float normalization;
  const uint32_t columns = (params.width + kWidth - 1) / kWidth;
  const uint32_t origin_x = (blockIdx.x % columns) * kWidth;
  const uint32_t origin_y = (blockIdx.x / columns) * TileHeight;
  if (threadIdx.x < 33) kernel[threadIdx.x] = weights[threadIdx.x];
  if (threadIdx.x == 0) {
    float weight_sum = 0.0f;
    for (unsigned int tap = 0; tap < 33; ++tap) weight_sum += weights[tap];
    normalization = weight_sum;
  }
  for (unsigned int index = threadIdx.x; index < kInputSize; index += blockDim.x) {
    const uint32_t x = origin_x + index % kWidth;
    const int y = static_cast<int>(origin_y + index / kWidth) - static_cast<int>(kRadius);
    const bool valid = x < params.width && y >= 0 && y < static_cast<int>(params.height);
    const size_t source = static_cast<size_t>(y) * params.blurred_stride + x;
    tile0[index] = valid ? horizontal0[source] : 0.0f;
    tile1[index] = valid ? horizontal1[source] : 0.0f;
    tile2[index] = valid ? horizontal2[source] : 0.0f;
  }
  // Partial-tile lanes participate in every cooperative load and barrier.
  __syncthreads();
  static_assert(TileHeight % Rows == 0);
  static_assert(Threads % 32 == 0);
  constexpr unsigned Warps = Threads / 32;
  const uint32_t local_x = threadIdx.x % 32;
  const uint32_t x = origin_x + local_x;
  if (x >= params.width) return;
  for (unsigned group = threadIdx.x / 32; group < TileHeight / Rows; group += Warps) {
    const uint32_t local_y = group * Rows;
    const uint32_t y = origin_y + local_y;
    if (y >= params.height) continue;
    const unsigned first = local_y * kWidth + local_x;
    float sum0[Rows] = {}, sum1[Rows] = {}, sum2[Rows] = {};
    float weight_sum[Rows];
    const bool interior = y >= kRadius &&
        static_cast<size_t>(y) + Rows - 1 + kRadius < params.height;
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row)
      weight_sum[row] = interior ? normalization : 0.0f;
    // Keep nine independent FMA chains in the original tap order. A loaded
    // input row feeds up to three adjacent output rows before it is discarded.
    if (interior) {
#pragma unroll
      for (unsigned input_row = 0; input_row < 32 + Rows; ++input_row) {
        const float value0 = tile0[first + input_row * kWidth];
        const float value1 = tile1[first + input_row * kWidth];
        const float value2 = tile2[first + input_row * kWidth];
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
          if (input_row >= row && input_row < row + 33) {
            const float weight = kernel[input_row - row];
            sum0[row] += value0 * weight;
            sum1[row] += value1 * weight;
            sum2[row] += value2 * weight;
          }
        }
      }
    } else {
#pragma unroll
      for (unsigned input_row = 0; input_row < 32 + Rows; ++input_row) {
        const int coordinate = static_cast<int>(y) + static_cast<int>(input_row) - 16;
        if (coordinate >= 0 && coordinate < static_cast<int>(params.height)) {
          const float value0 = tile0[first + input_row * kWidth];
          const float value1 = tile1[first + input_row * kWidth];
          const float value2 = tile2[first + input_row * kWidth];
#pragma unroll
          for (unsigned row = 0; row < Rows; ++row) {
            if (input_row >= row && input_row < row + 33) {
              const float weight = kernel[input_row - row];
              sum0[row] += value0 * weight;
              sum1[row] += value1 * weight;
              sum2[row] += value2 * weight;
              weight_sum[row] += weight;
            }
          }
        }
      }
    }
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
      if (y + row >= params.height) continue;
      const float bx = sum0[row] / weight_sum[row];
      const float by = sum1[row] / weight_sum[row];
      const float bb = sum2[row] / weight_sum[row];
      const size_t source = static_cast<size_t>(y + row) * params.xyb_stride + x;
      const size_t output = static_cast<size_t>(y + row) * params.psycho_stride + x;
      medium0[output] = xyb0[source] - bx;
      medium1[output] = xyb1[source] - by;
      medium2[output] = xyb2[source] - bb;
      low0[output] = bx * 33.832837186260f;
      low1[output] = by * 14.458268100570f;
      low2[output] = UnfusedMultiplyAdd(-0.362267051518f, by, bb) * 49.87984651440f;
    }
  }
}

__device__ float MaximumClamp(float value, float maximum) {
  if (value >= maximum) {
    return UnfusedMultiplyAdd(value - maximum, 0.724216145665f, maximum);
  }
  if (value < -maximum) {
    return UnfusedMultiplyAdd(value + maximum, 0.724216145665f, -maximum);
  }
  return value;
}

__device__ float RemoveRange(float value, float width) {
  return value > width ? value - width : value < -width ? value + width : 0.0f;
}

__device__ float AmplifyRange(float value, float width) {
  return value > width    ? value + width
         : value < -width ? value - width
                          : value + value;
}

__global__ void FrequencySplitKernel(float* input, const float* blurred,
                                     float* output, FrequencyParams params) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t input_index = static_cast<size_t>(y) * params.input_stride + x;
  const size_t low_index = static_cast<size_t>(y) * params.low_stride + x;
  const size_t output_index = static_cast<size_t>(y) * params.output_stride + x;
  float low_pass = blurred[low_index];
  const float original = input[input_index];
  if (params.channel < 2) {
    output[output_index] = original - low_pass;
    input[input_index] = params.channel == 0 ? RemoveRange(low_pass, 0.29f)
                                             : AmplifyRange(low_pass, 0.1f);
  } else if (params.channel == 2) {
    input[input_index] = low_pass;
  } else if (params.channel == 3) {
    output[output_index] = RemoveRange(original - low_pass, 0.04f);
    input[input_index] = RemoveRange(low_pass, 1.5f);
  } else {
    low_pass = MaximumClamp(low_pass, 28.4691806922f);
    output[output_index] =
        MaximumClamp(original - low_pass, 5.19175294647f) * 2.69313763794f;
    input[input_index] = AmplifyRange(low_pass * 2.155f, 0.132f);
  }
}

template <uint32_t Channel>
struct StoreFrequencySplit {
  float* input;
  float* output;
  CudaButteraugliFrequencyParams params;

  __device__ void operator()(uint32_t x, uint32_t y, float low_pass) const {
    const size_t input_index = static_cast<size_t>(y) * params.input_stride + x;
    const size_t output_index = static_cast<size_t>(y) * params.output_stride + x;
    const float original = input[input_index];
    // Keep the separate FrequencySplitKernel's rounding and range decisions.
    // Each thread owns these two outputs; convolution reads only the completed
    // horizontal intermediate, never neighboring values of the in-place input.
    if constexpr (Channel < 2) {
      output[output_index] = original - low_pass;
      input[input_index] = Channel == 0 ? RemoveRange(low_pass, 0.29f)
                                       : AmplifyRange(low_pass, 0.1f);
    } else if constexpr (Channel == 3) {
      output[output_index] = RemoveRange(original - low_pass, 0.04f);
      input[input_index] = RemoveRange(low_pass, 1.5f);
    } else {
      low_pass = MaximumClamp(low_pass, 28.4691806922f);
      output[output_index] =
          MaximumClamp(original - low_pass, 5.19175294647f) * 2.69313763794f;
      input[input_index] = AmplifyRange(low_pass * 2.155f, 0.132f);
    }
  }
};

template <uint32_t Channel>
__global__ void ConvolutionFrequencyKernel(
    const float* horizontal, const float* weights, float* input, float* output,
    CudaButteraugliFrequencyParams params) {
  constexpr unsigned int kKernelSize = Channel < 2 ? 15 : 7;
  ConvolutionTiledBody<false, kKernelSize>(
      horizontal, weights, params.width, params.height, params.width,
      StoreFrequencySplit<Channel>{input, output, params});
}

template <uint32_t Channel>
cudaError_t LaunchBlurAndSplit(float* input, const float* weights,
                               float* intermediate, float* output,
                               CudaButteraugliFrequencyParams params,
                               cudaStream_t stream) {
  constexpr unsigned int kKernelSize = Channel < 2 ? 15 : 7;
  ConvolutionTiledKernel<true, kKernelSize>
      <<<ConvolutionTile<true>::Blocks(params.width, params.height),
         kPlaneThreads, 0, stream>>>(
          input, weights, intermediate, params.width, params.height,
          params.input_stride, params.width);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  ConvolutionFrequencyKernel<Channel>
      <<<ConvolutionTile<false>::Blocks(params.width, params.height),
         kPlaneThreads, 0, stream>>>(
          intermediate, weights, input, output, params);
  return cudaGetLastError();
}

__global__ void SuppressXKernel(float* high_x, const float* high_y,
                                PlaneParams params) {
  const size_t index = PlaneIndex(params);
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t x_index = static_cast<size_t>(y) * params.output_stride + x;
  const size_t y_index = static_cast<size_t>(y) * params.input_stride + x;
  const float value = high_y[y_index];
  const float denominator = UnfusedMultiplyAdd(value, value, 46.0f);
  const float scaler = UnfusedMultiplyAdd(
      46.0f / denominator, 1.0f - 0.653020556257f, 0.653020556257f);
  high_x[x_index] *= scaler;
}

__device__ float MaltaScaleValue(float value0, float value1,
                                 MaltaScaleParams params) {
  const float absolute = 0.5f * (fabsf(value0) + fabsf(value1));
  const float difference = value0 - value1;
  const float scaler = params.norm2_0_gt_1 / (params.norm + absolute);
  float scaled = scaler * difference;
  const float scaler2 = params.norm2_0_lt_1 / (params.norm + absolute);
  const float magnitude = fabsf(value0);
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
  return scaled;
}

__global__ void MaltaScaleKernel(const float* reference, const float* distorted,
                                 float* output, MaltaScaleParams params) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  output[static_cast<size_t>(y) * params.output_stride + x] = MaltaScaleValue(
      reference[static_cast<size_t>(y) * params.reference_stride + x],
      distorted[static_cast<size_t>(y) * params.distorted_stride + x], params);
}

__device__ float ReadZero(const float* input, int x, int y,
                          MaltaResponseParams params) {
  if (x < 0 || y < 0 || x >= static_cast<int>(params.width) ||
      y >= static_cast<int>(params.height)) {
    return 0.0f;
  }
  return input[static_cast<size_t>(y) * params.input_stride +
               static_cast<uint32_t>(x)];
}

__device__ void AddSquare(float value, float* total) {
  *total = UnfusedMultiplyAdd(value, value, *total);
}

__device__ float Sum5(float a, float b, float c, float d, float e) {
  return (a + b) + (c + (d + e));
}

__device__ float Sum7(float a, float b, float c, float d, float e, float f,
                      float g) {
  return (a + b) + (c + ((d + e) + (f + g)));
}

__device__ float Sum9(float a, float b, float c, float d, float e, float f,
                      float g, float h, float i) {
  return ((a + b) + (c + d)) + ((e + f) + (g + h)) + i;
}

__device__ float MaltaLf(const float* center) {
#define GJXL_V(dx, dy) center[(dy) * static_cast<int>(kMaltaTileStride) + (dx)]
  float sum = Sum5(GJXL_V(-4, 0), GJXL_V(-2, 0), GJXL_V(0, 0), GJXL_V(2, 0),
                   GJXL_V(4, 0));
  float result = sum * sum;
  sum = Sum5(GJXL_V(0, -4), GJXL_V(0, -2), GJXL_V(0, 0), GJXL_V(0, 2),
             GJXL_V(0, 4));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-3, -3), GJXL_V(-2, -2), GJXL_V(0, 0), GJXL_V(2, 2),
             GJXL_V(3, 3));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(3, -3), GJXL_V(2, -2), GJXL_V(0, 0), GJXL_V(-2, 2),
             GJXL_V(-3, 3));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(1, -4), GJXL_V(1, -2), GJXL_V(0, 0), GJXL_V(-1, 2),
             GJXL_V(-1, 4));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-1, -4), GJXL_V(-1, -2), GJXL_V(0, 0), GJXL_V(1, 2),
             GJXL_V(1, 4));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-4, -1), GJXL_V(-2, -1), GJXL_V(0, 0), GJXL_V(2, 1),
             GJXL_V(4, 1));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-4, 1), GJXL_V(-2, 1), GJXL_V(0, 0), GJXL_V(2, -1),
             GJXL_V(4, -1));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-2, -3), GJXL_V(-1, -2), GJXL_V(0, 0), GJXL_V(1, 2),
             GJXL_V(2, 3));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(2, -3), GJXL_V(1, -2), GJXL_V(0, 0), GJXL_V(-1, 2),
             GJXL_V(-2, 3));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-3, -2), GJXL_V(-2, -1), GJXL_V(0, 0), GJXL_V(2, 1),
             GJXL_V(3, 2));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(3, -2), GJXL_V(2, -1), GJXL_V(0, 0), GJXL_V(-2, 1),
             GJXL_V(-3, 2));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-4, 2), GJXL_V(-2, 1), GJXL_V(0, 0), GJXL_V(2, -1),
             GJXL_V(4, -2));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-4, -2), GJXL_V(-2, -1), GJXL_V(0, 0), GJXL_V(2, 1),
             GJXL_V(4, 2));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(-2, -4), GJXL_V(-1, -2), GJXL_V(0, 0), GJXL_V(1, 2),
             GJXL_V(2, 4));
  AddSquare(sum, &result);
  sum = Sum5(GJXL_V(2, -4), GJXL_V(1, -2), GJXL_V(0, 0), GJXL_V(-1, 2),
             GJXL_V(-2, 4));
  AddSquare(sum, &result);
#undef GJXL_V
  return result;
}

__device__ float MaltaFull(const float* center) {
#define GJXL_V(dx, dy) center[(dy) * static_cast<int>(kMaltaTileStride) + (dx)]
  float sum = Sum9(GJXL_V(-4, 0), GJXL_V(-3, 0), GJXL_V(-2, 0), GJXL_V(-1, 0),
                   GJXL_V(0, 0), GJXL_V(1, 0), GJXL_V(2, 0), GJXL_V(3, 0),
                   GJXL_V(4, 0));
  float result = sum * sum;
  sum = Sum9(GJXL_V(0, -4), GJXL_V(0, -3), GJXL_V(0, -2), GJXL_V(0, -1),
             GJXL_V(0, 0), GJXL_V(0, 1), GJXL_V(0, 2), GJXL_V(0, 3),
             GJXL_V(0, 4));
  AddSquare(sum, &result);
  sum = Sum7(GJXL_V(-3, -3), GJXL_V(-2, -2), GJXL_V(-1, -1), GJXL_V(0, 0),
             GJXL_V(1, 1), GJXL_V(2, 2), GJXL_V(3, 3));
  AddSquare(sum, &result);
  sum = Sum7(GJXL_V(3, -3), GJXL_V(2, -2), GJXL_V(1, -1), GJXL_V(0, 0),
             GJXL_V(-1, 1), GJXL_V(-2, 2), GJXL_V(-3, 3));
  AddSquare(sum, &result);
  sum = Sum9(GJXL_V(1, -4), GJXL_V(1, -3), GJXL_V(1, -2), GJXL_V(0, -1),
             GJXL_V(0, 0), GJXL_V(0, 1), GJXL_V(-1, 2), GJXL_V(-1, 3),
             GJXL_V(-1, 4));
  AddSquare(sum, &result);
  sum = Sum9(GJXL_V(-1, -4), GJXL_V(-1, -3), GJXL_V(-1, -2), GJXL_V(0, -1),
             GJXL_V(0, 0), GJXL_V(0, 1), GJXL_V(1, 2), GJXL_V(1, 3),
             GJXL_V(1, 4));
  AddSquare(sum, &result);
  sum = Sum9(GJXL_V(-4, -1), GJXL_V(-3, -1), GJXL_V(-2, -1), GJXL_V(-1, 0),
             GJXL_V(0, 0), GJXL_V(1, 0), GJXL_V(2, 1), GJXL_V(3, 1),
             GJXL_V(4, 1));
  AddSquare(sum, &result);
  sum = Sum9(GJXL_V(-4, 1), GJXL_V(-3, 1), GJXL_V(-2, 1), GJXL_V(-1, 0),
             GJXL_V(0, 0), GJXL_V(1, 0), GJXL_V(2, -1), GJXL_V(3, -1),
             GJXL_V(4, -1));
  AddSquare(sum, &result);
  sum = Sum7(GJXL_V(-2, -3), GJXL_V(-1, -2), GJXL_V(-1, -1), GJXL_V(0, 0),
             GJXL_V(1, 1), GJXL_V(1, 2), GJXL_V(2, 3));
  AddSquare(sum, &result);
  sum = Sum7(GJXL_V(2, -3), GJXL_V(1, -2), GJXL_V(1, -1), GJXL_V(0, 0),
             GJXL_V(-1, 1), GJXL_V(-1, 2), GJXL_V(-2, 3));
  AddSquare(sum, &result);
  sum = Sum7(GJXL_V(-3, -2), GJXL_V(-2, -1), GJXL_V(-1, -1), GJXL_V(0, 0),
             GJXL_V(1, 1), GJXL_V(2, 1), GJXL_V(3, 2));
  AddSquare(sum, &result);
  sum = Sum7(GJXL_V(3, -2), GJXL_V(2, -1), GJXL_V(1, -1), GJXL_V(0, 0),
             GJXL_V(-1, 1), GJXL_V(-2, 1), GJXL_V(-3, 2));
  AddSquare(sum, &result);
  sum = Sum9(GJXL_V(-4, 1), GJXL_V(-3, 1), GJXL_V(-2, 1), GJXL_V(-1, 0),
             GJXL_V(0, 0), GJXL_V(1, 0), GJXL_V(2, -1), GJXL_V(3, -1),
             GJXL_V(4, -1));
  AddSquare(sum, &result);
  sum = Sum9(GJXL_V(-4, -1), GJXL_V(-3, -1), GJXL_V(-2, -1), GJXL_V(-1, 0),
             GJXL_V(0, 0), GJXL_V(1, 0), GJXL_V(2, 1), GJXL_V(3, 1),
             GJXL_V(4, 1));
  AddSquare(sum, &result);
  sum = Sum9(GJXL_V(-1, -4), GJXL_V(-1, -3), GJXL_V(-1, -2), GJXL_V(0, -1),
             GJXL_V(0, 0), GJXL_V(0, 1), GJXL_V(1, 2), GJXL_V(1, 3),
             GJXL_V(1, 4));
  AddSquare(sum, &result);
  sum = Sum9(GJXL_V(1, -4), GJXL_V(1, -3), GJXL_V(1, -2), GJXL_V(0, -1),
             GJXL_V(0, 0), GJXL_V(0, 1), GJXL_V(-1, 2), GJXL_V(-1, 3),
             GJXL_V(-1, 4));
  AddSquare(sum, &result);
#undef GJXL_V
  return result;
}

__global__ void MaltaResponseKernel(const float* input, float* accumulation,
                                    MaltaResponseParams params) {
  constexpr unsigned int kTileValues =
      kMaltaTileStride * (kMaltaTileHeight + 2 * kMaltaRadius);
  __shared__ float tile[kTileValues];
  const uint32_t tile_columns =
      (params.width + kMaltaTileWidth - 1) / kMaltaTileWidth;
  const uint32_t origin_x = (blockIdx.x % tile_columns) * kMaltaTileWidth;
  const uint32_t origin_y = (blockIdx.x / tile_columns) * kMaltaTileHeight;
  // Include a zero-filled halo so each response can use fixed shared-memory
  // offsets without repeating image-boundary checks for every sample. All
  // threads load and synchronize, including those in a partial edge tile.
  for (unsigned int index = threadIdx.x; index < kTileValues;
       index += blockDim.x) {
    const int x = static_cast<int>(origin_x + index % kMaltaTileStride) -
                  static_cast<int>(kMaltaRadius);
    const int y = static_cast<int>(origin_y + index / kMaltaTileStride) -
                  static_cast<int>(kMaltaRadius);
    tile[index] = ReadZero(input, x, y, params);
  }
  __syncthreads();
  const uint32_t local_x = threadIdx.x % kMaltaTileWidth;
  const uint32_t local_y = threadIdx.x / kMaltaTileWidth;
  const uint32_t x = origin_x + local_x;
  const uint32_t y = origin_y + local_y;
  if (x >= params.width || y >= params.height) return;
  const float* center = tile +
      (local_y + kMaltaRadius) * kMaltaTileStride + local_x + kMaltaRadius;
  const float result =
      params.low_frequency != 0 ? MaltaLf(center) : MaltaFull(center);
  const size_t output = static_cast<size_t>(y) * params.accumulation_stride + x;
  if (params.initialize_accumulation != 0) {
    accumulation[output] = result;
  } else {
    accumulation[output] += result;
  }
}

template <unsigned int TileHeight, bool LowFrequency, bool FlatGrid>
__global__ void MaltaScaleResponseKernel(const float* reference,
                                         const float* distorted,
                                         float* accumulation,
                                         CudaButteraugliMaltaParams params) {
  constexpr unsigned int kTileValues =
      kMaltaTileStride * (TileHeight + 2 * kMaltaRadius);
  static_assert(TileHeight % kMaltaTileHeight == 0);
  __shared__ float tile[kTileValues];
  uint32_t origin_x = blockIdx.x * kMaltaTileWidth;
  uint32_t origin_y = blockIdx.y * TileHeight;
  if constexpr (FlatGrid) {
    const uint32_t tile_columns =
        (params.width + kMaltaTileWidth - 1) / kMaltaTileWidth;
    origin_x = (blockIdx.x % tile_columns) * kMaltaTileWidth;
    origin_y = (blockIdx.x / tile_columns) * TileHeight;
  }
  const MaltaScaleParams scale{params.width,
                               params.height,
                               params.reference_stride,
                               params.distorted_stride,
                               0,
                               static_cast<uint32_t>(LowFrequency),
                               params.norm2_0_gt_1,
                               params.norm2_0_lt_1,
                               params.norm};
  // Scale directly into the shared response tile, including its halo. This
  // repeats halo arithmetic but avoids an intermediate plane and a launch.
  // Every thread participates in the load/barrier, even in a partial tile.
  for (unsigned int index = threadIdx.x; index < kTileValues;
       index += blockDim.x) {
    const int x = static_cast<int>(origin_x + index % kMaltaTileStride) -
                  static_cast<int>(kMaltaRadius);
    const int y = static_cast<int>(origin_y + index / kMaltaTileStride) -
                  static_cast<int>(kMaltaRadius);
    float value = 0.0f;
    if (x >= 0 && y >= 0 && x < static_cast<int>(params.width) &&
        y < static_cast<int>(params.height)) {
      value = MaltaScaleValue(
          reference[static_cast<size_t>(y) * params.reference_stride + x],
          distorted[static_cast<size_t>(y) * params.distorted_stride + x],
          scale);
    }
    tile[index] = value;
  }
  __syncthreads();
  const uint32_t local_x = threadIdx.x % kMaltaTileWidth;
  if constexpr (TileHeight == kMaltaTileHeight) {
    const uint32_t local_y = threadIdx.x / kMaltaTileWidth;
    const uint32_t x = origin_x + local_x;
    const uint32_t y = origin_y + local_y;
    if (x >= params.width || y >= params.height) return;
    const float* center = tile + (local_y + kMaltaRadius) * kMaltaTileStride +
                          local_x + kMaltaRadius;
    const float result = LowFrequency ? MaltaLf(center) : MaltaFull(center);
    const size_t output = static_cast<size_t>(y) * params.accumulation_stride + x;
    if (params.initialize_accumulation != 0) {
      accumulation[output] = result;
    } else {
      accumulation[output] += result;
    }
  } else {
    // Reuse one loaded halo across multiple output rows per lane, while
    // retaining 256 threads. Keep responses rolled to bound register liveness.
    // Every lane has already completed the cooperative load and barrier.
#pragma unroll 1
    for (uint32_t local_y = threadIdx.x / kMaltaTileWidth;
         local_y < TileHeight; local_y += kMaltaTileHeight) {
      const uint32_t x = origin_x + local_x;
      const uint32_t y = origin_y + local_y;
      if (x >= params.width || y >= params.height) continue;
      const float* center = tile + (local_y + kMaltaRadius) * kMaltaTileStride +
                            local_x + kMaltaRadius;
      const float result = LowFrequency ? MaltaLf(center) : MaltaFull(center);
      const size_t output =
          static_cast<size_t>(y) * params.accumulation_stride + x;
      if (params.initialize_accumulation != 0) {
        accumulation[output] = result;
      } else {
        accumulation[output] += result;
      }
    }
  }
}

template <unsigned int TileHeight, bool FlatGrid>
cudaError_t LaunchFusedMalta(const float* reference, const float* distorted,
                             float* accumulation,
                             CudaButteraugliMaltaParams params, dim3 grid,
                             cudaStream_t stream) {
  constexpr unsigned int kThreads = kMaltaTileWidth * kMaltaTileHeight;
  if (params.low_frequency != 0) {
    MaltaScaleResponseKernel<TileHeight, true, FlatGrid>
        <<<grid, kThreads, 0, stream>>>(reference, distorted, accumulation, params);
  } else {
    MaltaScaleResponseKernel<TileHeight, false, FlatGrid>
        <<<grid, kThreads, 0, stream>>>(reference, distorted, accumulation, params);
  }
  return cudaGetLastError();
}

constexpr unsigned int MaltaTileHeightForSize(uint32_t width, uint32_t height) {
  const uint64_t columns = (uint64_t{width} + kMaltaTileWidth - 1) /
                          kMaltaTileWidth;
  const uint64_t rows8 = (uint64_t{height} + 7) / 8;
  const uint64_t rows64 = (uint64_t{height} + 63) / 64;
  // Small images need independent blocks more than halo reuse. These fixed
  // cutoffs balance the measured small/large-image tradeoff without querying
  // device properties or adding work to the submission stream.
  if (columns * rows8 < 128) return 8;
  return columns * rows64 < 768 ? 24 : 64;
}

static_assert(MaltaTileHeightForSize(32, 1016) == 8);
static_assert(MaltaTileHeightForSize(32, 1017) == 24);
static_assert(MaltaTileHeightForSize(1024, 1472) == 24);
static_assert(MaltaTileHeightForSize(1024, 1473) == 64);

template <unsigned int TileHeight>
cudaError_t LaunchTiledMalta(const float* reference,
                             const float* distorted, float* accumulation,
                             CudaButteraugliMaltaParams params,
                             cudaStream_t stream) {
  const uint32_t tile_columns =
      (params.width + kMaltaTileWidth - 1) / kMaltaTileWidth;
  const uint32_t tile_rows =
      (params.height + TileHeight - 1) / TileHeight;
  // Avoid per-thread grid division for normal images without introducing a
  // new height limit: CUDA's grid.y is limited to 65535 blocks.
  if (tile_rows > 65535) {
    return LaunchFusedMalta<TileHeight, true>(
        reference, distorted, accumulation, params,
        dim3(tile_columns * tile_rows), stream);
  }
  return LaunchFusedMalta<TileHeight, false>(
      reference, distorted, accumulation, params, dim3(tile_columns, tile_rows),
      stream);
}

__device__ float L2Asymmetric(float value0, float value1, float weight_up,
                              float weight_down, float total) {
  const float difference = value0 - value1;
  total = UnfusedMultiplyAdd(difference * difference, weight_up * 0.8f, total);
  const float magnitude = fabsf(value0);
  const float too_small = 0.4f * magnitude;
  float secondary = 0.0f;
  if (value0 < 0.0f) {
    if (value1 > -too_small)
      secondary = value1 + too_small;
    else if (value1 < -magnitude)
      secondary = -value1 - magnitude;
  } else if (value1 < too_small) {
    secondary = too_small - value1;
  } else if (value1 > magnitude) {
    secondary = value1 - magnitude;
  }
  return UnfusedMultiplyAdd(weight_down * 0.8f, secondary * secondary, total);
}

__global__ void L2Kernel(DifferencePlan plan) {
  const DifferenceParams params = plan.params;
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t r = static_cast<size_t>(y) * params.reference_stride + x;
  const size_t d = static_cast<size_t>(y) * params.distorted_stride + x;
  const size_t w = static_cast<size_t>(y) * params.work_stride + x;
  const float inverse_asymmetry = 1.0f / params.asymmetry;
  float total0 = L2Asymmetric(plan.reference[6][r], plan.distorted[6][d],
                              400.0f * params.asymmetry,
                              400.0f * inverse_asymmetry, plan.ac[0][w]);
  float total1 =
      L2Asymmetric(plan.reference[7][r], plan.distorted[7][d],
                   1.50815703118f * params.asymmetry,
                   1.50815703118f * inverse_asymmetry, plan.ac[1][w]);
  const float md0 = plan.reference[3][r] - plan.distorted[3][d];
  const float md1 = plan.reference[4][r] - plan.distorted[4][d];
  const float md2 = plan.reference[5][r] - plan.distorted[5][d];
  plan.ac[0][w] = UnfusedMultiplyAdd(md0 * md0, 2150.0f, total0);
  plan.ac[1][w] = UnfusedMultiplyAdd(md1 * md1, 10.6195433239f, total1);
  plan.ac[2][w] = md2 * md2 * 16.2176043152f;
  const float ld0 = plan.reference[0][r] - plan.distorted[0][d];
  const float ld1 = plan.reference[1][r] - plan.distorted[1][d];
  const float ld2 = plan.reference[2][r] - plan.distorted[2][d];
  plan.dc[0][w] = ld0 * ld0 * 29.2353797994f;
  plan.dc[1][w] = ld1 * ld1 * 0.844626970982f;
  plan.dc[2][w] = ld2 * ld2 * 0.703646627719f;
}

__global__ void MaskPrecomputeKernel(const float* high_x, const float* high_y,
                                     const float* ultra_x, const float* ultra_y,
                                     float* output, PlaneParams params) {
  const size_t index = PlaneIndex(params);
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t input = static_cast<size_t>(y) * params.input_stride + x;
  const size_t destination = static_cast<size_t>(y) * params.output_stride + x;
  const float xdiff = (ultra_x[input] + high_x[input]) * 2.5f;
  const float ydiff = ultra_y[input] * 0.4f + high_y[input] * 0.4f;
  const float activity = sqrtf(xdiff * xdiff + ydiff * ydiff);
  constexpr float kMultiplier = 6.19424080439f;
  constexpr float kBias = kMultiplier * 12.61050594197f;
  output[destination] =
      sqrtf(kMultiplier * fabsf(activity) + kBias) - sqrtf(kBias);
}

__device__ void StoreMin3(float value, float* minimum0, float* minimum1,
                          float* minimum2) {
  if (value < *minimum2) {
    if (value < *minimum0) {
      *minimum2 = *minimum1;
      *minimum1 = *minimum0;
      *minimum0 = value;
    } else if (value < *minimum1) {
      *minimum2 = *minimum1;
      *minimum1 = value;
    } else {
      *minimum2 = value;
    }
  }
}

__global__ void FuzzyErosionKernel(const float* input, float* output,
                                   PlaneParams params) {
  const size_t index = PlaneIndex(params);
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y0 = static_cast<uint32_t>(index / params.width);
  const uint32_t x0 =
      static_cast<uint32_t>(index - static_cast<size_t>(y0) * params.width);
  constexpr int kStep = 3;
  const int x = static_cast<int>(x0);
  const int y = static_cast<int>(y0);
  float minimum0 = input[static_cast<size_t>(y0) * params.input_stride + x0];
  float minimum1 = 2.0f * minimum0;
  float minimum2 = minimum1;
  for (int dy = -kStep; dy <= kStep; dy += kStep) {
    for (int dx = -kStep; dx <= kStep; dx += kStep) {
      if (dx == 0 && dy == 0) continue;
      const int sx = x + dx;
      const int sy = y + dy;
      if (sx >= 0 && sy >= 0 && sx < static_cast<int>(params.width) &&
          sy < static_cast<int>(params.height)) {
        StoreMin3(input[static_cast<size_t>(sy) * params.input_stride +
                        static_cast<uint32_t>(sx)],
                  &minimum0, &minimum1, &minimum2);
      }
    }
  }
  output[static_cast<size_t>(y0) * params.output_stride + x0] =
      0.45f * minimum0 + 0.3f * minimum1 + 0.25f * minimum2;
}

__device__ float FuzzyErosionValue(const float* input, PlaneParams params,
                                   uint32_t x0, uint32_t y0) {
  constexpr int kStep = 3;
  const int x = static_cast<int>(x0);
  const int y = static_cast<int>(y0);
  float minimum0 = input[static_cast<size_t>(y0) * params.input_stride + x0];
  float minimum1 = 2.0f * minimum0;
  float minimum2 = minimum1;
  for (int dy = -kStep; dy <= kStep; dy += kStep) {
    for (int dx = -kStep; dx <= kStep; dx += kStep) {
      if (dx == 0 && dy == 0) continue;
      const int sx = x + dx;
      const int sy = y + dy;
      if (sx >= 0 && sy >= 0 && sx < static_cast<int>(params.width) &&
          sy < static_cast<int>(params.height)) {
        StoreMin3(input[static_cast<size_t>(sy) * params.input_stride +
                        static_cast<uint32_t>(sx)],
                  &minimum0, &minimum1, &minimum2);
      }
    }
  }
  // Preserve the stored erosion result's contraction and rounding before
  // applying MaskY/MaskDcY. Source reassociation can round the other product.
  const float weighted1 = __fmul_rn(0.3f, minimum1);
  const float weighted01 = __fmaf_rn(0.45f, minimum0, weighted1);
  return __fmaf_rn(0.25f, minimum2, weighted01);
}

__device__ float MaskY(float delta) {
  constexpr float kGlobalScale = 1.0f / (17.83f * 0.79079917404f);
  const float value =
      kGlobalScale *
      (1.0f + 2.5485944793f / (0.451936922203f * delta + 0.829591754942f));
  return value * value;
}

__device__ float MaskDcY(float delta) {
  constexpr float kGlobalScale = 1.0f / (17.83f * 0.79079917404f);
  const float value =
      kGlobalScale *
      (1.0f + 0.505054525019f / (3.87449418804f * delta + 0.20025578522f));
  return value * value;
}

struct FinalPlan {
  const float* dc[3];
  const float* ac[3];
  const float* mask;
  const float* mask_reference;
  const float* mask_distorted;
  float* output;
  FinalParams params;
};

__global__ void FinalKernel(FinalPlan plan) {
  const FinalParams params = plan.params;
  const size_t flat =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (flat >= count) return;
  const uint32_t y = static_cast<uint32_t>(flat / params.width);
  const uint32_t x =
      static_cast<uint32_t>(flat - static_cast<size_t>(y) * params.width);
  const size_t index = static_cast<size_t>(y) * params.stride + x;
  const float difference =
      plan.mask_reference[index] - plan.mask_distorted[index];
  const float ac_y = plan.ac[1][index] + 10.0f * difference * difference;
  const float mask_value = MaskY(plan.mask[index]);
  const float dc_mask_value = MaskDcY(plan.mask[index]);
  const float masked_dc =
      plan.dc[0][index] * params.x_multiplier * dc_mask_value +
      plan.dc[1][index] * dc_mask_value + plan.dc[2][index] * dc_mask_value;
  const float masked_ac = plan.ac[0][index] * params.x_multiplier * mask_value +
                          ac_y * mask_value + plan.ac[2][index] * mask_value;
  const float result = sqrtf(masked_dc + masked_ac);
  plan.output[static_cast<size_t>(y) * params.output_stride + x] =
      isfinite(result) && result >= 0.0f ? result : NAN;
}

// L2's six intermediate values have no consumer before final masking.
// Keep the original kernels above as the independent separate-pass oracle.
__global__ void L2FinalKernel(DifferencePlan difference_plan, FinalPlan plan) {
  const FinalParams params = plan.params;
  const DifferenceParams difference_params = difference_plan.params;
  const size_t flat =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (flat >= count) return;
  const uint32_t y = static_cast<uint32_t>(flat / params.width);
  const uint32_t x =
      static_cast<uint32_t>(flat - static_cast<size_t>(y) * params.width);
  const size_t r = static_cast<size_t>(y) * difference_params.reference_stride + x;
  const size_t d = static_cast<size_t>(y) * difference_params.distorted_stride + x;
  const size_t index = static_cast<size_t>(y) * params.stride + x;
  const float inverse_asymmetry = 1.0f / difference_params.asymmetry;
  const float total0 = L2Asymmetric(
      difference_plan.reference[6][r], difference_plan.distorted[6][d],
      400.0f * difference_params.asymmetry, 400.0f * inverse_asymmetry,
      difference_plan.ac[0][index]);
  const float total1 = L2Asymmetric(
      difference_plan.reference[7][r], difference_plan.distorted[7][d],
      1.50815703118f * difference_params.asymmetry,
      1.50815703118f * inverse_asymmetry, difference_plan.ac[1][index]);
  const float md0 = difference_plan.reference[3][r] - difference_plan.distorted[3][d];
  const float md1 = difference_plan.reference[4][r] - difference_plan.distorted[4][d];
  const float md2 = difference_plan.reference[5][r] - difference_plan.distorted[5][d];
  const float ac0 = UnfusedMultiplyAdd(md0 * md0, 2150.0f, total0);
  const float ac1 = UnfusedMultiplyAdd(md1 * md1, 10.6195433239f, total1);
  const float ac2 = md2 * md2 * 16.2176043152f;
  const float ld0 = difference_plan.reference[0][r] - difference_plan.distorted[0][d];
  const float ld1 = difference_plan.reference[1][r] - difference_plan.distorted[1][d];
  const float ld2 = difference_plan.reference[2][r] - difference_plan.distorted[2][d];
  const float dc0 = ld0 * ld0 * 29.2353797994f;
  const float dc1 = ld1 * ld1 * 0.844626970982f;
  const float dc2 = ld2 * ld2 * 0.703646627719f;
  const float difference =
      plan.mask_reference[index] - plan.mask_distorted[index];
  const float ac_y = ac1 + 10.0f * difference * difference;
  const float mask_value = MaskY(plan.mask[index]);
  const float dc_mask_value = MaskDcY(plan.mask[index]);
  const float masked_dc = dc0 * params.x_multiplier * dc_mask_value +
                          dc1 * dc_mask_value + dc2 * dc_mask_value;
  const float masked_ac = ac0 * params.x_multiplier * mask_value +
                          ac_y * mask_value + ac2 * mask_value;
  const float result = sqrtf(masked_dc + masked_ac);
  plan.output[static_cast<size_t>(y) * params.output_stride + x] =
      isfinite(result) && result >= 0.0f ? result : NAN;
}

// Reference-mask erosion has no consumer other than final masking.
__global__ void ErosionL2FinalKernel(DifferencePlan difference_plan,
                                     FinalPlan plan) {
  const FinalParams params = plan.params;
  const DifferenceParams difference_params = difference_plan.params;
  const size_t flat =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (flat >= count) return;
  const uint32_t y = static_cast<uint32_t>(flat / params.width);
  const uint32_t x =
      static_cast<uint32_t>(flat - static_cast<size_t>(y) * params.width);
  const size_t r = static_cast<size_t>(y) * difference_params.reference_stride + x;
  const size_t d = static_cast<size_t>(y) * difference_params.distorted_stride + x;
  const size_t index = static_cast<size_t>(y) * params.stride + x;
  const float inverse_asymmetry = 1.0f / difference_params.asymmetry;
  const float total0 = L2Asymmetric(
      difference_plan.reference[6][r], difference_plan.distorted[6][d],
      400.0f * difference_params.asymmetry, 400.0f * inverse_asymmetry,
      difference_plan.ac[0][index]);
  const float total1 = L2Asymmetric(
      difference_plan.reference[7][r], difference_plan.distorted[7][d],
      1.50815703118f * difference_params.asymmetry,
      1.50815703118f * inverse_asymmetry, difference_plan.ac[1][index]);
  const float md0 = difference_plan.reference[3][r] - difference_plan.distorted[3][d];
  const float md1 = difference_plan.reference[4][r] - difference_plan.distorted[4][d];
  const float md2 = difference_plan.reference[5][r] - difference_plan.distorted[5][d];
  const float ac0 = UnfusedMultiplyAdd(md0 * md0, 2150.0f, total0);
  const float ac1 = UnfusedMultiplyAdd(md1 * md1, 10.6195433239f, total1);
  const float ac2 = md2 * md2 * 16.2176043152f;
  const float ld0 = difference_plan.reference[0][r] - difference_plan.distorted[0][d];
  const float ld1 = difference_plan.reference[1][r] - difference_plan.distorted[1][d];
  const float ld2 = difference_plan.reference[2][r] - difference_plan.distorted[2][d];
  const float dc0 = ld0 * ld0 * 29.2353797994f;
  const float dc1 = ld1 * ld1 * 0.844626970982f;
  const float dc2 = ld2 * ld2 * 0.703646627719f;
  const float difference =
      plan.mask_reference[index] - plan.mask_distorted[index];
  const float ac_y = ac1 + 10.0f * difference * difference;
  const PlaneParams erosion_params{params.width, params.height, params.stride,
                                    params.stride};
  const float eroded = FuzzyErosionValue(plan.mask_reference, erosion_params, x, y);
  const float mask_value = MaskY(eroded);
  const float dc_mask_value = MaskDcY(eroded);
  const float masked_dc = dc0 * params.x_multiplier * dc_mask_value +
                          dc1 * dc_mask_value + dc2 * dc_mask_value;
  const float masked_ac = ac0 * params.x_multiplier * mask_value +
                          ac_y * mask_value + ac2 * mask_value;
  const float result = sqrtf(masked_dc + masked_ac);
  plan.output[static_cast<size_t>(y) * params.output_stride + x] =
      isfinite(result) && result >= 0.0f ? result : NAN;
}

struct CropParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t xborder;
  uint32_t yborder;
};

__global__ void CropKernel(const float* input, float* output,
                           CropParams params) {
  const size_t flat =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (flat >= count) return;
  const uint32_t y = static_cast<uint32_t>(flat / params.width);
  const uint32_t x =
      static_cast<uint32_t>(flat - static_cast<size_t>(y) * params.width);
  output[static_cast<size_t>(y) * params.output_stride + x] =
      input[static_cast<size_t>(y + params.yborder) * params.input_stride + x +
            params.xborder];
}

struct ComposeParams {
  uint32_t width;
  uint32_t height;
  uint32_t main_stride;
  uint32_t sub_stride;
  uint32_t output_stride;
};

__global__ void ComposeKernel(const float* main_map, const float* sub_map,
                              float* output, ComposeParams params) {
  const size_t flat =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (flat >= count) return;
  const uint32_t y = static_cast<uint32_t>(flat / params.width);
  const uint32_t x =
      static_cast<uint32_t>(flat - static_cast<size_t>(y) * params.width);
  const float main_value =
      main_map[static_cast<size_t>(y) * params.main_stride + x];
  const float sub_value =
      sub_map[static_cast<size_t>(y / 2) * params.sub_stride + x / 2];
  output[static_cast<size_t>(y) * params.output_stride + x] =
      main_value * 0.85f + 0.5f * sub_value;
}

struct ReductionParams {
  uint32_t width;
  uint32_t input_stride;
  uint32_t input_count;
};

__global__ void ReduceMaximumKernel(const float* input, float* output,
                                    ReductionParams params) {
  __shared__ float values[kReductionWidth];
  const uint32_t index = blockIdx.x * kReductionWidth + threadIdx.x;
  float value = -INFINITY;
  if (index < params.input_count) {
    const uint32_t y = index / params.width;
    const uint32_t x = index - y * params.width;
    value = input[static_cast<size_t>(y) * params.input_stride + x];
    if (!isfinite(value) || value < 0.0f) value = NAN;
  }
  values[threadIdx.x] = value;
  __syncthreads();
  for (uint32_t step = kReductionWidth / 2; step != 0; step /= 2) {
    if (threadIdx.x < step) {
      const float other = values[threadIdx.x + step];
      values[threadIdx.x] = isnan(values[threadIdx.x]) || isnan(other)
                                ? NAN
                                : fmaxf(values[threadIdx.x], other);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = values[0];
}

[[nodiscard]] unsigned int PlaneBlocks(uint32_t width, uint32_t height) {
  const size_t count = static_cast<size_t>(width) * height;
  return static_cast<unsigned int>((count + kPlaneThreads - 1) / kPlaneThreads);
}

[[nodiscard]] cudaError_t CheckLaunch() { return cudaGetLastError(); }

[[nodiscard]] cudaError_t LaunchExpand(std::array<const float*, 3> input,
                                       std::array<uint32_t, 3> input_stride,
                                       std::array<float*, 3> output,
                                       const CudaButteraugliPlan& plan,
                                       cudaStream_t stream) {
  for (size_t channel = 0; channel < 3; ++channel) {
    const ExpandParams params{
        plan.width,          plan.height,           plan.working_width,
        plan.working_height, input_stride[channel], plan.working_width,
        plan.xborder,        plan.yborder};
    ExpandKernel<<<PlaneBlocks(plan.working_width, plan.working_height),
                   kPlaneThreads, 0, stream>>>(
        input[channel], output[channel], params);
    const cudaError_t error = CheckLaunch();
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

[[nodiscard]] cudaError_t LaunchSubsample(std::array<const float*, 3> input,
                                          std::array<uint32_t, 3> input_stride,
                                          std::array<float*, 3> output,
                                          uint32_t output_stride,
                                          const CudaButteraugliPlan& plan,
                                          cudaStream_t stream) {
  for (size_t channel = 0; channel < 3; ++channel) {
    const SubsampleParams params{
        plan.width,      plan.height,           plan.sub_width,
        plan.sub_height, input_stride[channel], output_stride};
    SubsampleKernel<<<PlaneBlocks(plan.sub_width, plan.sub_height),
                      kPlaneThreads, 0, stream>>>(
        input[channel], output[channel], params);
    const cudaError_t error = CheckLaunch();
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

template <unsigned int KernelSize>
[[nodiscard]] cudaError_t LaunchBlur(const float* input, uint32_t input_stride,
                                     const float* weights, float* intermediate,
                                     float* output, uint32_t output_stride,
                                     uint32_t width, uint32_t height,
                                     cudaStream_t stream) {
  if constexpr (KernelSize == 5) {
    const unsigned int blocks = PlaneBlocks(width, height);
    MirroredConvolution5Kernel<true><<<blocks, kPlaneThreads, 0, stream>>>(
        input, weights, intermediate, width, height, input_stride, width);
    cudaError_t error = CheckLaunch();
    if (error != cudaSuccess) return error;
    MirroredConvolution5Kernel<false><<<blocks, kPlaneThreads, 0, stream>>>(
        intermediate, weights, output, width, height, width, output_stride);
    return CheckLaunch();
  } else {
    const unsigned int horizontal_blocks =
        ConvolutionTile<true>::Blocks(width, height);
    const unsigned int vertical_blocks =
        ConvolutionTile<false>::Blocks(width, height);
    ConvolutionTiledKernel<true, KernelSize>
        <<<horizontal_blocks, kPlaneThreads, 0, stream>>>(
            input, weights, intermediate, width, height, input_stride, width);
    cudaError_t error = CheckLaunch();
    if (error != cudaSuccess) return error;
    ConvolutionTiledKernel<false, KernelSize>
        <<<vertical_blocks, kPlaneThreads, 0, stream>>>(
            intermediate, weights, output, width, height, width, output_stride);
    return CheckLaunch();
  }
}

[[nodiscard]] cudaError_t LaunchPsycho(
    const CudaButteraugliPlan& plan, std::array<const float*, 3> input,
    std::array<uint32_t, 3> input_stride,
    const std::array<float*, kCudaButteraugliPsychoPlaneCount>& psycho,
    uint32_t psycho_stride, uint32_t width, uint32_t height,
    cudaStream_t stream) {
  CudaButteraugliOpsinPlan opsin;
  opsin.input = input;
  opsin.input_stride = input_stride;
  opsin.weights = plan.kernels[0];
  opsin.width = width;
  opsin.height = height;
  opsin.output_stride = plan.working_width;
  opsin.intensity_target = plan.intensity_target;
  for (size_t channel = 0; channel < 3; ++channel) {
    opsin.intermediate[channel] = plan.planes[kPsychoWork + channel];
    opsin.output[channel] = plan.planes[kImage + channel];
  }
  // Original RGB is external or staged in the not-yet-produced low outputs.
  // Keep three packed horizontal RGB planes until joint vertical/Opsin; their
  // storage can then be reused immediately by the horizontal XYB blurs.
  cudaError_t error = LaunchCudaButteraugliOpsin(opsin, stream);
  if (error != cudaSuccess) return error;

  CudaButteraugliLowMediumPlan low_medium;
  for (size_t channel = 0; channel < 3; ++channel) {
    low_medium.input[channel] = plan.planes[kImage + channel];
    // The earlier RGB intermediates are dead; retain three distinct packed
    // horizontal XYB planes until the joint vertical/low-medium pass.
    low_medium.intermediate[channel] = plan.planes[kImage + 3 + channel];
    low_medium.low[channel] = psycho[channel];
    low_medium.medium[channel] = psycho[3 + channel];
  }
  low_medium.weights = plan.kernels[1];
  low_medium.width = width;
  low_medium.height = height;
  low_medium.input_stride = plan.working_width;
  low_medium.output_stride = psycho_stride;
  error = LaunchCudaButteraugliLowMedium(low_medium, stream);
  if (error != cudaSuccess) return error;

  for (size_t channel = 0; channel < 2; ++channel) {
    const CudaButteraugliFrequencyParams frequency{
        width, height, psycho_stride, psycho_stride,
        static_cast<uint32_t>(channel)};
    error = LaunchCudaButteraugliBlurAndSplit(
        psycho[3 + channel], plan.kernels[2], plan.planes[kPsychoWork],
        psycho[6 + channel], frequency, stream);
    if (error != cudaSuccess) return error;
  }
  error = LaunchBlur<15>(psycho[5], psycho_stride, plan.kernels[2],
                         plan.planes[kPsychoWork], psycho[5], psycho_stride,
                         width, height, stream);
  if (error != cudaSuccess) return error;

  const PlaneParams suppress{width, height, psycho_stride, psycho_stride};
  SuppressXKernel<<<PlaneBlocks(width, height), kPlaneThreads, 0, stream>>>(
      psycho[6], psycho[7], suppress);
  error = CheckLaunch();
  if (error != cudaSuccess) return error;

  for (size_t channel = 0; channel < 2; ++channel) {
    const CudaButteraugliFrequencyParams frequency{
        width, height, psycho_stride, psycho_stride,
        static_cast<uint32_t>(channel + 3)};
    error = LaunchCudaButteraugliBlurAndSplit(
        psycho[6 + channel], plan.kernels[3], plan.planes[kPsychoWork],
        psycho[8 + channel], frequency, stream);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

[[nodiscard]] cudaError_t LaunchMaskPrecompute(
    const std::array<const float*, kCudaButteraugliPsychoPlaneCount>& psycho,
    uint32_t psycho_stride, float* output, uint32_t output_stride,
    uint32_t width, uint32_t height, cudaStream_t stream) {
  const PlaneParams params{width, height, psycho_stride, output_stride};
  MaskPrecomputeKernel<<<PlaneBlocks(width, height), kPlaneThreads, 0,
                         stream>>>(psycho[6], psycho[7], psycho[8], psycho[9],
                                   output, params);
  return CheckLaunch();
}

template <typename T>
[[nodiscard]] std::array<const float*, kCudaButteraugliPsychoPlaneCount>
ConstPsycho(const std::array<T, kCudaButteraugliPsychoPlaneCount>& input) {
  std::array<const float*, kCudaButteraugliPsychoPlaneCount> result{};
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = input[index];
  }
  return result;
}

[[nodiscard]] std::array<float*, kCudaButteraugliPsychoPlaneCount> MainPsycho(
    const CudaButteraugliPlan& plan, size_t base) {
  std::array<float*, kCudaButteraugliPsychoPlaneCount> result{};
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = plan.planes[base + index];
  }
  return result;
}

[[nodiscard]] cudaError_t LaunchDifference(
    const CudaButteraugliPlan& plan,
    const std::array<const float*, kCudaButteraugliPsychoPlaneCount>& reference,
    uint32_t reference_stride,
    const std::array<const float*, kCudaButteraugliPsychoPlaneCount>& distorted,
    uint32_t distorted_stride, const float* cached_reference_mask,
    float* output, uint32_t output_stride,
    uint32_t width, uint32_t height, cudaStream_t stream) {
  constexpr double kWeights[6] = {37.0819870399, 8246.75321353, 18.7237414387,
                                  6923.99476109, 1.10039032555, 173.5};
  constexpr double kNorms[6] = {130262059.556, 1009002.70582, 4498534.45232,
                                8051.15833247, 71.7800275169, 5.0};
  constexpr size_t kOrder[6] = {4, 5, 2, 3, 0, 1};
  constexpr size_t kPsychoPlane[6] = {4, 3, 7, 6, 9, 8};
  const double asymmetry = plan.hf_asymmetry;
  const double sqrt_asymmetry = sqrt(asymmetry);
  for (size_t stage : kOrder) {
    const double weight_up = stage < 2   ? kWeights[stage]
                             : stage < 4 ? kWeights[stage] * sqrt_asymmetry
                                         : kWeights[stage] * asymmetry;
    const double weight_down = stage < 2   ? kWeights[stage]
                               : stage < 4 ? kWeights[stage] / sqrt_asymmetry
                                           : kWeights[stage] / asymmetry;
    const bool low_frequency = stage < 4;
    const double multiplier = low_frequency ? 0.611612573796 : 0.39905817637;
    const double pre_up =
        multiplier * sqrt(0.5 * weight_up) / (3.75 * 2.0 + 1.0);
    const double pre_down =
        multiplier * sqrt(0.33 * weight_down) / (3.75 * 2.0 + 1.0);
    const size_t channel = stage % 2 == 0 ? 1 : 0;
    const CudaButteraugliMaltaParams malta{
        width, height, reference_stride, distorted_stride, plan.working_width,
        static_cast<uint32_t>(low_frequency), static_cast<uint32_t>(stage >= 4),
        static_cast<float>(pre_up * kNorms[stage]),
        static_cast<float>(pre_down * kNorms[stage]),
        static_cast<float>(kNorms[stage])};
    const cudaError_t error = LaunchCudaButteraugliMalta(
        reference[kPsychoPlane[stage]], distorted[kPsychoPlane[stage]],
        plan.planes[kAc + channel], malta, stream);
    if (error != cudaSuccess) return error;
  }

  DifferencePlan difference{};
  for (size_t index = 0; index < 8; ++index) {
    difference.reference[index] = reference[index];
    difference.distorted[index] = distorted[index];
  }
  for (size_t channel = 0; channel < 2; ++channel) {
    difference.ac[channel] = plan.planes[kAc + channel];
  }
  difference.params = {width,
                       height,
                       reference_stride,
                       distorted_stride,
                       plan.working_width,
                       plan.hf_asymmetry};
  cudaError_t error = cudaSuccess;
  const float* reference_mask = cached_reference_mask;
  if (reference_mask == nullptr) {
    error =
        LaunchMaskPrecompute(reference, reference_stride, plan.planes[kMaskInput],
                             plan.working_width, width, height, stream);
    if (error != cudaSuccess) return error;
    error =
        LaunchBlur<13>(plan.planes[kMaskInput], plan.working_width, plan.kernels[4],
                       plan.planes[kMaskIntermediate], plan.planes[kReferenceMask],
                       plan.working_width, width, height, stream);
    if (error != cudaSuccess) return error;
    reference_mask = plan.planes[kReferenceMask];
  }

  error = LaunchMaskPrecompute(distorted, distorted_stride, plan.planes[kMaskInput],
                               plan.working_width, width, height, stream);
  if (error != cudaSuccess) return error;
  error =
      LaunchBlur<13>(plan.planes[kMaskInput], plan.working_width, plan.kernels[4],
                     plan.planes[kMaskIntermediate], plan.planes[kDistortedMask],
                     plan.working_width, width, height, stream);
  if (error != cudaSuccess) return error;

  FinalPlan final{};
  for (size_t channel = 0; channel < 2; ++channel) {
    final.ac[channel] = plan.planes[kAc + channel];
  }
  final.mask_reference = reference_mask;
  final.mask_distorted = plan.planes[kDistortedMask];
  final.output = output;
  final.params = {width, height, plan.working_width, output_stride,
                  plan.x_multiplier};
  // Only AC[0:2] is consumed by the fused pass. Leave unused AC/DC pointers
  // null: their former slots now hold masks. The horizontal mask intermediate
  // is no longer read, so it can also hold the cropped/subscale output map.
  ErosionL2FinalKernel<<<PlaneBlocks(width, height), kPlaneThreads, 0, stream>>>(
      difference, final);
  return CheckLaunch();
}

[[nodiscard]] cudaError_t LaunchMaximumReduction(
    const CudaButteraugliPlan& plan, const float* input, uint32_t input_stride,
    float* output, cudaStream_t stream) {
  uint32_t input_count = plan.width * plan.height;
  uint32_t width = plan.width;
  bool use_a = true;
  while (true) {
    const uint32_t output_count =
        (input_count + kReductionWidth - 1) / kReductionWidth;
    float* destination =
        output_count == 1 ? output : plan.reduction[use_a ? 0 : 1];
    const ReductionParams params{width, input_stride, input_count};
    ReduceMaximumKernel<<<output_count, kReductionWidth, 0, stream>>>(
        input, destination, params);
    const cudaError_t error = CheckLaunch();
    if (error != cudaSuccess) return error;
    if (output_count == 1) return cudaSuccess;
    input = destination;
    input_count = output_count;
    width = output_count;
    input_stride = output_count;
    use_a = !use_a;
  }
}

}  // namespace

namespace {
cudaError_t LaunchOpsinImpl(const CudaButteraugliOpsinPlan& plan,
                           bool reference, cudaStream_t stream) {
  if (plan.width == 0 || plan.height == 0) return cudaSuccess;
  if (plan.output_stride < plan.width) return cudaErrorInvalidValue;
  for (uint32_t stride : plan.input_stride)
    if (stride < plan.width) return cudaErrorInvalidValue;
  for (size_t channel = 0; channel < 3; ++channel) {
    cudaError_t error;
    if (reference) {
      error = LaunchBlur<5>(plan.input[channel], plan.input_stride[channel],
          plan.weights, plan.intermediate[channel], plan.blurred[channel],
          plan.output_stride, plan.width, plan.height, stream);
    } else {
      MirroredConvolution5Kernel<true>
          <<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0, stream>>>(
              plan.input[channel], plan.weights, plan.intermediate[channel],
              plan.width, plan.height, plan.input_stride[channel], plan.width);
      error = CheckLaunch();
    }
    if (error != cudaSuccess) return error;
  }
  if (reference) {
    const OpsinParams params{plan.width, plan.height,
        {plan.input_stride[0], plan.input_stride[1], plan.input_stride[2]},
        plan.output_stride, plan.output_stride, plan.intensity_target};
    OpsinKernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0, stream>>>(
        plan.input[0], plan.input[1], plan.input[2],
        plan.blurred[0], plan.blurred[1], plan.blurred[2],
        plan.output[0], plan.output[1], plan.output[2], params);
  } else {
    // 16 rows balances directional halo reuse and shared-memory residency.
    constexpr unsigned int kOpsinTileHeight = 16;
    const unsigned int blocks = ((plan.width + 31) / 32) *
        ((plan.height + kOpsinTileHeight - 1) / kOpsinTileHeight);
    OpsinConvolutionPlan fused{};
    for (size_t channel = 0; channel < 3; ++channel) {
      fused.input[channel] = plan.input[channel];
      fused.intermediate[channel] = plan.intermediate[channel];
      fused.output[channel] = plan.output[channel];
      fused.input_stride[channel] = plan.input_stride[channel];
    }
    fused.weights = plan.weights;
    fused.width = plan.width;
    fused.height = plan.height;
    fused.output_stride = plan.output_stride;
    fused.intensity_target = plan.intensity_target;
    ConvolutionOpsinKernel<kOpsinTileHeight><<<blocks, kPlaneThreads, 0, stream>>>(fused);
  }
  return CheckLaunch();
}

cudaError_t LaunchLowMediumImpl(const CudaButteraugliLowMediumPlan& plan,
                               bool reference, cudaStream_t stream,
                               bool sequential = false) {
  if (plan.width == 0 || plan.height == 0) return cudaSuccess;
  if (plan.input_stride < plan.width || plan.output_stride < plan.width ||
      (reference && plan.blurred_stride < plan.width)) return cudaErrorInvalidValue;
  if (reference) {
    for (size_t channel = 0; channel < 3; ++channel) {
      const cudaError_t error = LaunchBlur<33>(
          plan.input[channel], plan.input_stride, plan.weights,
          plan.intermediate[channel], plan.blurred[channel], plan.blurred_stride,
          plan.width, plan.height, stream);
      if (error != cudaSuccess) return error;
    }
  } else {
    // Share weight loads, normalization, and address work across channels.
    // The original horizontal intermediates and per-channel tap order remain.
    constexpr unsigned int kHorizontalWidth = 256;
    constexpr unsigned int kHorizontalHeight = 4;
    const unsigned int blocks =
        ((plan.width + kHorizontalWidth - 1) / kHorizontalWidth) *
        ((plan.height + kHorizontalHeight - 1) / kHorizontalHeight);
    ConvolutionHorizontal3Kernel<kHorizontalWidth, kHorizontalHeight>
        <<<blocks, kPlaneThreads, 0, stream>>>(
            plan.input[0], plan.input[1], plan.input[2], plan.weights,
            plan.intermediate[0], plan.intermediate[1], plan.intermediate[2],
            plan.width, plan.height, plan.input_stride);
    const cudaError_t error = CheckLaunch();
    if (error != cudaSuccess) return error;
  }
 const LowMediumParams params{plan.width, plan.height, plan.input_stride,
      reference ? plan.blurred_stride : plan.width, plan.output_stride};
  if (reference) {
    LowMediumKernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0, stream>>>(
        plan.input[0], plan.input[1], plan.input[2],
        plan.blurred[0], plan.blurred[1], plan.blurred[2],
        plan.low[0], plan.low[1], plan.low[2],
        plan.medium[0], plan.medium[1], plan.medium[2], params);
  } else {
    // Balance three-channel halo reuse against shared-memory residency.
    constexpr unsigned int kLowMediumHeight = 48;
    const unsigned int blocks = ((plan.width + 31) / 32) *
        ((plan.height + kLowMediumHeight - 1) / kLowMediumHeight);
    if (sequential) {
      ConvolutionLowMediumKernel<kLowMediumHeight><<<blocks, kPlaneThreads, 0, stream>>>(
          plan.input[0], plan.input[1], plan.input[2],
          plan.intermediate[0], plan.intermediate[1], plan.intermediate[2], plan.weights,
          plan.low[0], plan.low[1], plan.low[2],
          plan.medium[0], plan.medium[1], plan.medium[2], params);
    } else {
      ConvolutionLowMediumRowsKernel<kLowMediumHeight><<<blocks, kPlaneThreads, 0, stream>>>(
          plan.input[0], plan.input[1], plan.input[2],
          plan.intermediate[0], plan.intermediate[1], plan.intermediate[2], plan.weights,
          plan.low[0], plan.low[1], plan.low[2],
          plan.medium[0], plan.medium[1], plan.medium[2], params);
    }
  }
  return CheckLaunch();
}

cudaError_t LaunchL2FinalForTest(const CudaButteraugliL2FinalPlan& plan,
                               bool reference, cudaStream_t stream,
                               bool erode_reference = false) {
  if (plan.width == 0 || plan.height == 0) return cudaSuccess;
  if (plan.reference_stride < plan.width || plan.distorted_stride < plan.width ||
      plan.work_stride < plan.width || plan.output_stride < plan.width)
    return cudaErrorInvalidValue;
  DifferencePlan difference{};
  for (size_t i = 0; i < 8; ++i) {
    difference.reference[i] = plan.reference[i];
    difference.distorted[i] = plan.distorted[i];
  }
  FinalPlan final{};
  for (size_t i = 0; i < 3; ++i) {
    difference.ac[i] = plan.ac[i];
    difference.dc[i] = plan.dc[i];
    final.ac[i] = plan.ac[i];
    final.dc[i] = plan.dc[i];
  }
  difference.params = {plan.width, plan.height, plan.reference_stride,
                        plan.distorted_stride, plan.work_stride, plan.asymmetry};
  final.mask = plan.mask;
  final.mask_reference = plan.mask_reference;
  final.mask_distorted = plan.mask_distorted;
  final.output = plan.output;
  final.params = {plan.width, plan.height, plan.work_stride,
                   plan.output_stride, plan.x_multiplier};
  if (reference) {
    L2Kernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0, stream>>>(
        difference);
    const cudaError_t error = CheckLaunch();
    if (error != cudaSuccess) return error;
    FinalKernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0, stream>>>(
        final);
  } else if (erode_reference) {
    ErosionL2FinalKernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0, stream>>>(
        difference, final);
  } else {
    L2FinalKernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0, stream>>>(
        difference, final);
  }
  return CheckLaunch();
}
}  // namespace

cudaError_t LaunchCudaButteraugliOpsin(
    const CudaButteraugliOpsinPlan& plan, cudaStream_t stream) {
  return LaunchOpsinImpl(plan, false, stream);
}

cudaError_t LaunchCudaButteraugliOpsinReference(
    const CudaButteraugliOpsinPlan& plan, cudaStream_t stream) {
  return LaunchOpsinImpl(plan, true, stream);
}

cudaError_t LaunchCudaButteraugliLowMedium(
    const CudaButteraugliLowMediumPlan& plan, cudaStream_t stream) {
  return LaunchLowMediumImpl(plan, false, stream);
}

cudaError_t LaunchCudaButteraugliLowMediumReference(
    const CudaButteraugliLowMediumPlan& plan, cudaStream_t stream) {
  return LaunchLowMediumImpl(plan, true, stream);
}

cudaError_t LaunchCudaButteraugliLowMediumSequentialReference(
    const CudaButteraugliLowMediumPlan& plan, cudaStream_t stream) {
  return LaunchLowMediumImpl(plan, false, stream, true);
}

cudaError_t LaunchCudaButteraugliL2Final(
    const CudaButteraugliL2FinalPlan& plan, cudaStream_t stream) {
  return LaunchL2FinalForTest(plan, false, stream);
}

cudaError_t LaunchCudaButteraugliL2FinalReference(
    const CudaButteraugliL2FinalPlan& plan, cudaStream_t stream) {
  return LaunchL2FinalForTest(plan, true, stream);
}

cudaError_t LaunchCudaButteraugliErosionFinalForTesting(
    const CudaButteraugliL2FinalPlan& plan, float* erosion_scratch,
    bool reference, cudaStream_t stream) {
  if (plan.width == 0 || plan.height == 0) return cudaSuccess;
  if (plan.reference_stride < plan.width || plan.distorted_stride < plan.width ||
      plan.work_stride < plan.width || plan.output_stride < plan.width)
    return cudaErrorInvalidValue;
  if (!reference) return LaunchL2FinalForTest(plan, false, stream, true);
  if (erosion_scratch == nullptr) return cudaErrorInvalidValue;
  const PlaneParams params{plan.width, plan.height, plan.work_stride,
                            plan.work_stride};
  FuzzyErosionKernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0,
                       stream>>>(plan.mask_reference, erosion_scratch, params);
  const cudaError_t error = CheckLaunch();
  if (error != cudaSuccess) return error;
  auto separate = plan;
  separate.mask = erosion_scratch;
  return LaunchL2FinalForTest(separate, false, stream);
}

cudaError_t LaunchCudaButteraugliBlurAndSplit(
    float* input, const float* weights, float* intermediate, float* output,
    CudaButteraugliFrequencyParams params, cudaStream_t stream) {
  if (params.channel != 0 && params.channel != 1 && params.channel != 3 &&
      params.channel != 4) return cudaErrorInvalidValue;
  if (params.width == 0 || params.height == 0) return cudaSuccess;
  switch (params.channel) {
    case 0: return LaunchBlurAndSplit<0>(input, weights, intermediate, output, params, stream);
    case 1: return LaunchBlurAndSplit<1>(input, weights, intermediate, output, params, stream);
    case 3: return LaunchBlurAndSplit<3>(input, weights, intermediate, output, params, stream);
    default: return LaunchBlurAndSplit<4>(input, weights, intermediate, output, params, stream);
  }
}

cudaError_t LaunchCudaButteraugliBlurAndSplitReference(
    float* input, const float* weights, float* intermediate, float* blurred,
    uint32_t blurred_stride, float* output,
    CudaButteraugliFrequencyParams params, cudaStream_t stream) {
  if (params.channel != 0 && params.channel != 1 && params.channel != 3 &&
      params.channel != 4) return cudaErrorInvalidValue;
  if (params.width == 0 || params.height == 0) return cudaSuccess;
  const cudaError_t error = params.channel < 2
      ? LaunchBlur<15>(input, params.input_stride, weights, intermediate,
                       blurred, blurred_stride, params.width, params.height, stream)
      : LaunchBlur<7>(input, params.input_stride, weights, intermediate,
                      blurred, blurred_stride, params.width, params.height, stream);
  if (error != cudaSuccess) return error;
  const FrequencyParams frequency{params.width, params.height, params.input_stride,
                                   blurred_stride, params.output_stride, params.channel};
  FrequencySplitKernel<<<PlaneBlocks(params.width, params.height), kPlaneThreads,
                         0, stream>>>(input, blurred, output, frequency);
  return cudaGetLastError();
}

cudaError_t LaunchCudaButteraugliMalta(const float* reference,
                                       const float* distorted,
                                       float* accumulation,
                                       CudaButteraugliMaltaParams params,
                                       cudaStream_t stream) {
  switch (MaltaTileHeightForSize(params.width, params.height)) {
    case 8:
      return LaunchTiledMalta<8>(
          reference, distorted, accumulation, params, stream);
    case 24:
      return LaunchTiledMalta<24>(
          reference, distorted, accumulation, params, stream);
    default:
      return LaunchTiledMalta<64>(
          reference, distorted, accumulation, params, stream);
  }
}

cudaError_t LaunchCudaButteraugliMaltaForTesting(
    const float* reference, const float* distorted, float* accumulation,
    CudaButteraugliMaltaParams params, unsigned int tile_height, bool flat_grid,
    cudaStream_t stream) {
  const uint32_t columns =
      (params.width + kMaltaTileWidth - 1) / kMaltaTileWidth;
  if (tile_height != 8 && tile_height != 24 && tile_height != 64) {
    return cudaErrorInvalidValue;
  }
  const uint32_t rows = (params.height + tile_height - 1) / tile_height;
  const dim3 grid = flat_grid ? dim3(columns * rows) : dim3(columns, rows);
  switch (tile_height) {
    case 8:
      return flat_grid
        ? LaunchFusedMalta<8, true>(
            reference, distorted, accumulation, params, grid, stream)
        : LaunchFusedMalta<8, false>(
            reference, distorted, accumulation, params, grid, stream);
    case 24:
      return flat_grid
        ? LaunchFusedMalta<24, true>(
            reference, distorted, accumulation, params, grid, stream)
        : LaunchFusedMalta<24, false>(
            reference, distorted, accumulation, params, grid, stream);
    default:
      return flat_grid
        ? LaunchFusedMalta<64, true>(
            reference, distorted, accumulation, params, grid, stream)
        : LaunchFusedMalta<64, false>(
            reference, distorted, accumulation, params, grid, stream);
  }
}

cudaError_t LaunchCudaButteraugliMaltaReference(
    const float* reference, const float* distorted, float* scaled,
    uint32_t scaled_stride, float* accumulation,
    CudaButteraugliMaltaParams params, cudaStream_t stream) {
  const MaltaScaleParams scale{params.width,
                               params.height,
                               params.reference_stride,
                               params.distorted_stride,
                               scaled_stride,
                               params.low_frequency,
                               params.norm2_0_gt_1,
                               params.norm2_0_lt_1,
                               params.norm};
  MaltaScaleKernel<<<PlaneBlocks(params.width, params.height), kPlaneThreads, 0,
                     stream>>>(reference, distorted, scaled, scale);
  cudaError_t error = CheckLaunch();
  if (error != cudaSuccess) return error;
  const MaltaResponseParams response{
      params.width,         params.height,
      scaled_stride,        params.accumulation_stride,
      params.low_frequency, params.initialize_accumulation};
  const uint32_t blocks =
      ((params.width + kMaltaTileWidth - 1) / kMaltaTileWidth) *
      ((params.height + kMaltaTileHeight - 1) / kMaltaTileHeight);
  MaltaResponseKernel<<<blocks, kMaltaTileWidth * kMaltaTileHeight, 0,
                        stream>>>(scaled, accumulation, response);
  return CheckLaunch();
}

cudaError_t LaunchCudaButteraugliPrepare(const CudaButteraugliPlan& plan,
                                         cudaStream_t stream) {
  const auto reference_main = MainPsycho(plan, 0);
  cudaError_t error = cudaSuccess;
  if (plan.expanded != 0) {
    error = LaunchExpand(plan.reference, plan.reference_stride,
                         {reference_main[0], reference_main[1], reference_main[2]},
                         plan, stream);
    if (error != cudaSuccess) return error;
    const std::array<const float*, 3> expanded{
        reference_main[0], reference_main[1], reference_main[2]};
    const std::array<uint32_t, 3> stride{plan.working_width, plan.working_width,
                                         plan.working_width};
    error =
        LaunchPsycho(plan, expanded, stride, reference_main, plan.working_width,
                     plan.working_width, plan.working_height, stream);
  } else {
    error = LaunchPsycho(plan, plan.reference, plan.reference_stride,
                         reference_main, plan.working_width, plan.width,
                         plan.height, stream);
  }
  if (error != cudaSuccess) return error;
  error = LaunchMaskPrecompute(ConstPsycho(reference_main), plan.working_width,
                               plan.planes[20], plan.working_width,
                               plan.working_width, plan.working_height, stream);
  if (error != cudaSuccess) return error;
  error =
      LaunchBlur<13>(plan.planes[20], plan.working_width, plan.kernels[4],
                     plan.planes[kPsychoWork], plan.planes[20], plan.working_width,
                     plan.working_width, plan.working_height, stream);
  if (error != cudaSuccess || plan.multiscale == 0) return error;

  error = LaunchSubsample(plan.reference, plan.reference_stride,
                          {plan.reference_sub[0], plan.reference_sub[1],
                           plan.reference_sub[2]}, plan.sub_width, plan, stream);
  if (error != cudaSuccess) return error;
  const std::array<const float*, 3> subsampled{
      plan.reference_sub[0], plan.reference_sub[1], plan.reference_sub[2]};
  const std::array<uint32_t, 3> stride{plan.sub_width, plan.sub_width,
                                       plan.sub_width};
  return LaunchPsycho(plan, subsampled, stride, plan.reference_sub,
                      plan.sub_width, plan.sub_width, plan.sub_height, stream);
}

cudaError_t LaunchCudaButteraugliCompare(
    const CudaButteraugliPlan& plan, std::array<const float*, 3> distorted,
    std::array<uint32_t, 3> distorted_stride, float* distance_map,
    uint32_t distance_stride, float* score, cudaStream_t stream) {
  const auto reference_main = ConstPsycho(MainPsycho(plan, 0));
  const auto distorted_main_mutable = MainPsycho(plan, 10);
  const auto distorted_main = ConstPsycho(distorted_main_mutable);
  cudaError_t error = cudaSuccess;
  if (plan.expanded != 0) {
    error = LaunchExpand(distorted, distorted_stride,
                         {distorted_main_mutable[0], distorted_main_mutable[1],
                          distorted_main_mutable[2]}, plan, stream);
    if (error != cudaSuccess) return error;
    const std::array<const float*, 3> expanded{
        distorted_main_mutable[0], distorted_main_mutable[1],
        distorted_main_mutable[2]};
    const std::array<uint32_t, 3> stride{plan.working_width, plan.working_width,
                                         plan.working_width};
    error = LaunchPsycho(plan, expanded, stride, distorted_main_mutable,
                         plan.working_width, plan.working_width,
                         plan.working_height, stream);
    if (error != cudaSuccess) return error;
    error = LaunchDifference(plan, reference_main, plan.working_width,
                             distorted_main, plan.working_width,
                             plan.planes[20],
                             plan.planes[kFinalStaging], plan.working_width,
                             plan.working_width, plan.working_height, stream);
    if (error != cudaSuccess) return error;
    const CropParams crop{plan.width,      plan.height,  plan.working_width,
                          distance_stride, plan.xborder, plan.yborder};
    CropKernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0,
                 stream>>>(plan.planes[kFinalStaging], distance_map, crop);
    error = CheckLaunch();
    if (error != cudaSuccess) return error;
  } else {
    error =
        LaunchPsycho(plan, distorted, distorted_stride, distorted_main_mutable,
                     plan.working_width, plan.width, plan.height, stream);
    if (error != cudaSuccess) return error;
    error = LaunchDifference(plan, reference_main, plan.working_width,
                             distorted_main, plan.working_width,
                             plan.planes[20], distance_map,
                             distance_stride, plan.width, plan.height, stream);
    if (error != cudaSuccess) return error;
    if (plan.multiscale != 0) {
      error = LaunchSubsample(distorted, distorted_stride,
                              {distorted_main_mutable[0], distorted_main_mutable[1],
                               distorted_main_mutable[2]}, plan.working_width,
                              plan, stream);
      if (error != cudaSuccess) return error;
      const std::array<const float*, 3> subsampled{distorted_main_mutable[0],
                                                   distorted_main_mutable[1],
                                                   distorted_main_mutable[2]};
      const std::array<uint32_t, 3> stride{
          plan.working_width, plan.working_width, plan.working_width};
      error = LaunchPsycho(plan, subsampled, stride, distorted_main_mutable,
                           plan.working_width, plan.sub_width, plan.sub_height,
                           stream);
      if (error != cudaSuccess) return error;
      error = LaunchDifference(
          plan, ConstPsycho(plan.reference_sub), plan.sub_width, distorted_main,
          plan.working_width, nullptr, plan.planes[kFinalStaging],
          plan.working_width, plan.sub_width, plan.sub_height, stream);
      if (error != cudaSuccess) return error;
      const ComposeParams compose{plan.width, plan.height, distance_stride,
                                  plan.working_width, distance_stride};
      ComposeKernel<<<PlaneBlocks(plan.width, plan.height), kPlaneThreads, 0,
                      stream>>>(distance_map, plan.planes[kFinalStaging],
                                distance_map, compose);
      error = CheckLaunch();
      if (error != cudaSuccess) return error;
    }
  }
  return LaunchMaximumReduction(plan, distance_map, distance_stride, score,
                                stream);
}

}  // namespace gjxl::cuda_internal
