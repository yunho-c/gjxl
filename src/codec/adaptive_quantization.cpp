// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/adaptive_quantization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

#include "core/block_grid.h"
#include "core/geometry.h"
#include "util/fast_math.h"

namespace gjxl {
namespace {

constexpr float kAcQuant = 0.765f;
constexpr float kDcQuant = 1.095924047623553f;
constexpr float kDcQuantPower = 0.83f;
constexpr float kInverseLog2E = 0.6931471805599453f;

template <bool Invert>
float GammaDerivativeRatio(float value) {
  constexpr float kEpsilon = 1.0e-2f;
  constexpr float kSgMul = 226.77216153508914f;
  constexpr float kSgMul2 = 1.0f / 73.377132366608819f;
  constexpr float kSgReturnMul =
    kSgMul2 * 18.6580932135f * kInverseLog2E;
  constexpr float kSgOffset = 7.7825991679894591f;

  value = std::max(value, 0.0f);
  const float squared = value * value;
  const float numerator = std::fma(
    kSgReturnMul * 3.0f * kSgMul,
    squared,
    kEpsilon);
  const float denominator = std::fma(
    kInverseLog2E * kSgMul * value,
    squared,
    kSgOffset * kInverseLog2E + kEpsilon);

  if constexpr (Invert) {
    return numerator / denominator;
  }

  return denominator / numerator;
}

float ComputeMask(float value) {
  constexpr float kBase = -0.7647f;
  constexpr float kMul4 = 9.4708735624378946f;
  constexpr float kMul2 = 17.35036561631863f;
  constexpr float kOffset2 = 302.59587815579727f;
  constexpr float kMul3 = 6.7943250517376494f;
  constexpr float kOffset3 = 3.7179635626140772f;
  constexpr float kOffset4 = 0.25f * kOffset3;
  constexpr float kMul0 = 0.80061762862741759f;

  const float v1 = std::max(value * kMul0, 1.0e-3f);
  const float v2 = 1.0f / (v1 + kOffset2);
  const float v3 = 1.0f / std::fma(v1, v1, kOffset3);
  const float v4 = 1.0f / std::fma(v1, v1, kOffset4);
  return kBase + std::fma(
    kMul4,
    v4,
    std::fma(kMul2, v2, kMul3 * v3));
}

float MaskingSqrt(float value) {
  constexpr float kLogOffset = 27.505837037000106f;
  constexpr float kMul = 211.66567973503678f;
  const float inner_scale = std::sqrt(kMul * 1.0e8f);
  return 0.25f * std::sqrt(
    std::fma(value, inner_scale, kLogOffset));
}

void StoreMin4(
  float value,
  float& min0,
  float& min1,
  float& min2,
  float& min3) {

  if (value >= min3) {
    return;
  }

  if (value < min0) {
    min3 = min2;
    min2 = min1;
    min1 = min0;
    min0 = value;
  } else if (value < min1) {
    min3 = min2;
    min2 = min1;
    min1 = value;
  } else if (value < min2) {
    min3 = min2;
    min2 = value;
  } else {
    min3 = value;
  }
}

void Sort4(std::array<float, 4>* values) {
  if ((*values)[0] > (*values)[1]) {
    std::swap((*values)[0], (*values)[1]);
  }
  if ((*values)[0] > (*values)[2]) {
    std::swap((*values)[0], (*values)[2]);
  }
  if ((*values)[0] > (*values)[3]) {
    std::swap((*values)[0], (*values)[3]);
  }
  if ((*values)[1] > (*values)[2]) {
    std::swap((*values)[1], (*values)[2]);
  }
  if ((*values)[1] > (*values)[3]) {
    std::swap((*values)[1], (*values)[3]);
  }
  if ((*values)[2] > (*values)[3]) {
    std::swap((*values)[2], (*values)[3]);
  }
}

void FuzzyErosion(
  float butteraugli_target,
  Extent2D source_extent,
  const std::vector<float>& source,
  Extent2D destination_extent,
  std::vector<float>* destination) {

  constexpr std::array<float, 4> kMulBase = {
    0.125f,
    0.1f,
    0.09f,
    0.06f,
  };
  constexpr std::array<float, 4> kMulAdd = {
    0.0f,
    -0.1f,
    -0.09f,
    -0.06f,
  };
  constexpr float kTotal = 0.29959705784054957f;

  float target_mix = 0.0f;
  if (butteraugli_target < 2.0f) {
    target_mix = (2.0f - butteraugli_target) * 0.5f;
  }

  std::array<float, 4> weights{};
  float weight_sum = 0.0f;
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = kMulBase[i] + target_mix * kMulAdd[i];
    weight_sum += weights[i];
  }
  for (float& weight : weights) {
    weight *= kTotal / weight_sum;
  }

  destination->assign(
    destination_extent.width * destination_extent.height,
    0.0f);

  for (size_t y = 0; y < source_extent.height; ++y) {
    const size_t top = y == 0 ? y : y - 1;
    const size_t bottom = y + 1 < source_extent.height ? y + 1 : y;

    for (size_t x = 0; x < source_extent.width; ++x) {
      const size_t left = x == 0 ? x : x - 1;
      const size_t right = x + 1 < source_extent.width ? x + 1 : x;
      const auto at = [&](size_t sample_x, size_t sample_y) {
        return source[sample_y * source_extent.width + sample_x];
      };

      std::array<float, 4> minima = {
        at(x, y),
        at(left, y),
        at(right, y),
        at(left, top),
      };
      Sort4(&minima);
      StoreMin4(at(x, top), minima[0], minima[1], minima[2], minima[3]);
      StoreMin4(at(right, top), minima[0], minima[1], minima[2], minima[3]);
      StoreMin4(at(left, bottom), minima[0], minima[1], minima[2], minima[3]);
      StoreMin4(at(x, bottom), minima[0], minima[1], minima[2], minima[3]);
      StoreMin4(at(right, bottom), minima[0], minima[1], minima[2], minima[3]);

      const float value =
        weights[0] * minima[0] +
        weights[1] * minima[1] +
        weights[2] * minima[2] +
        weights[3] * minima[3];
      const size_t destination_index =
        (y / 2) * destination_extent.width + x / 2;

      if ((x & 1u) == 0 && (y & 1u) == 0) {
        (*destination)[destination_index] = value;
      } else {
        (*destination)[destination_index] += value;
      }
    }
  }
}

size_t Mirror(ptrdiff_t coordinate, size_t size) {
  ptrdiff_t mirrored = coordinate;
  const ptrdiff_t signed_size = static_cast<ptrdiff_t>(size);

  while (mirrored < 0 || mirrored >= signed_size) {
    if (mirrored < 0) {
      mirrored = -mirrored - 1;
    } else {
      mirrored = 2 * signed_size - 1 - mirrored;
    }
  }

  return static_cast<size_t>(mirrored);
}

float WeightedSum5(
  const std::vector<float>& input,
  Extent2D extent,
  ptrdiff_t x,
  ptrdiff_t y,
  float center_weight,
  float near_weight,
  float far_weight) {

  const size_t sample_y = Mirror(y, extent.height);
  const auto sample = [&](ptrdiff_t sample_x) {
    return input[
      sample_y * extent.width + Mirror(sample_x, extent.width)];
  };

  const float far = far_weight * (sample(x - 2) + sample(x + 2));
  const float near = near_weight * (sample(x - 1) + sample(x + 1));
  const float center = center_weight * sample(x);
  return far + (near + center);
}

void BlurPixelMask(
  Extent2D extent,
  const std::vector<float>& input,
  std::vector<float>* output) {

  constexpr std::array<float, 5> kFilter = {
    0.364911248f,
    0.05f,
    0.1688888021f,
    0.221069183f,
    0.306563504f,
  };
  double weight_sum = 1.0 + 4.0 * (
    kFilter[0] + kFilter[1] + kFilter[2] +
    kFilter[4] + 2.0 * kFilter[3]);
  weight_sum = std::max(weight_sum, 1.0e-5);
  const float normalize = static_cast<float>(1.0 / weight_sum);

  const float w0 = normalize;
  const float w1 = normalize * kFilter[0];
  const float w2 = normalize * kFilter[2];
  const float w4 = normalize * kFilter[1];
  const float w5 = normalize * kFilter[3];
  const float w8 = normalize * kFilter[4];

  output->resize(extent.width * extent.height);
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const ptrdiff_t sx = static_cast<ptrdiff_t>(x);
      const ptrdiff_t sy = static_cast<ptrdiff_t>(y);
      float sum0 = WeightedSum5(input, extent, sx, sy, w0, w1, w2);
      sum0 += WeightedSum5(input, extent, sx, sy - 2, w2, w5, w8);
      float sum1 = WeightedSum5(input, extent, sx, sy + 2, w2, w5, w8);
      sum0 += WeightedSum5(input, extent, sx, sy - 1, w1, w4, w5);
      sum1 += WeightedSum5(input, extent, sx, sy + 1, w1, w4, w5);
      (*output)[y * extent.width + x] = sum0 + sum1;
    }
  }
}

float GammaModulation(
  ConstImage3FView opsin,
  size_t block_x,
  size_t block_y,
  float value) {

  constexpr float kBias = 0.16f;
  std::array<float, 4> lane_sum{};

  for (size_t dy = 0; dy < kJxlBlockDimension; ++dy) {
    const float* x_row = opsin.plane[0].Row(block_y + dy);
    const float* y_row = opsin.plane[1].Row(block_y + dy);
    for (size_t dx = 0; dx < kJxlBlockDimension; ++dx) {
      const float in_y = y_row[block_x + dx] + kBias;
      const float in_x = x_row[block_x + dx];
      lane_sum[dx & 3u] += GammaDerivativeRatio<true>(in_y - in_x);
      lane_sum[dx & 3u] += GammaDerivativeRatio<true>(in_y + in_x);
    }
  }

  const float overall =
    ((lane_sum[0] + lane_sum[1]) +
     (lane_sum[2] + lane_sum[3])) *
    (0.5f / 64.0f);
  if (!std::isfinite(overall) || overall <= 0.0f) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  constexpr float kGamma = 0.1005613337192697f;
  return std::fma(
    kGamma,
    fast_math::FastLog2(overall),
    value);
}

float HighFrequencyModulation(
  ConstPlaneF32View y_plane,
  size_t block_x,
  size_t block_y,
  float value) {

  constexpr float kLimit = 0.0206f;
  std::array<float, 4> lane_sum{};

  for (size_t dy = 0; dy < kJxlBlockDimension; ++dy) {
    const float* row = y_plane.Row(block_y + dy) + block_x;
    const float* next = dy + 1 < kJxlBlockDimension
      ? y_plane.Row(block_y + dy + 1) + block_x
      : row;

    for (size_t dx = 0; dx < kJxlBlockDimension; ++dx) {
      if (dx + 1 < kJxlBlockDimension) {
        lane_sum[dx & 3u] += std::min(
          kLimit,
          std::abs(row[dx] - row[dx + 1]));
      }
      lane_sum[dx & 3u] += std::min(
        kLimit,
        std::abs(row[dx] - next[dx]));
    }
  }

  const float sum =
    (lane_sum[0] + lane_sum[1]) +
    (lane_sum[2] + lane_sum[3]);
  return value + (sum * -0.38f + 0.42f);
}

float BlueModulation(
  ConstImage3FView opsin,
  size_t block_x,
  size_t block_y,
  float value) {

  constexpr float kLimit = 0.010474084867598155f;
  constexpr float kOffset = 0.0031994768654636393f;
  std::array<float, 4> lane_sum{};

  for (size_t dy = 0; dy < kJxlBlockDimension; ++dy) {
    const float* x_row = opsin.plane[0].Row(block_y + dy);
    const float* y_row = opsin.plane[1].Row(block_y + dy);
    const float* b_row = opsin.plane[2].Row(block_y + dy);

    for (size_t dx = 0; dx < kJxlBlockDimension; ++dx) {
      const size_t x = block_x + dx;
      const float effective_y = y_row[x] + kOffset + std::abs(x_row[x]);
      if (b_row[x] > effective_y) {
        lane_sum[dx & 3u] += std::min(b_row[x] - effective_y, kLimit);
      }
    }
  }

  float sum =
    (lane_sum[0] + lane_sum[1]) +
    (lane_sum[2] + lane_sum[3]);
  if (sum >= 32.0f * kLimit) {
    sum = 64.0f * kLimit - sum;
  }
  constexpr float kMaxLimit = 15.463398341612438f;
  sum = std::min(sum, kMaxLimit * kLimit);
  return value + sum * 0.90590804735610064f;
}

void PerBlockModulations(
  ConstImage3FView opsin,
  InitialQuantizationOptions options,
  Extent2D block_extent,
  std::vector<float>* quant_field) {

  const float scale = kAcQuant / options.butteraugli_target * options.rescale;
  const float base_level = 0.48f * scale;
  float dampen = 1.0f;
  if (options.butteraugli_target >= 2.0f) {
    dampen = 1.0f -
      (options.butteraugli_target - 2.0f) / 12.0f;
    dampen = std::max(dampen, 0.0f);
  }
  const float multiplier = scale * dampen;
  const float addend = (1.0f - dampen) * base_level;

  for (size_t block_y = 0; block_y < block_extent.height; ++block_y) {
    for (size_t block_x = 0; block_x < block_extent.width; ++block_x) {
      const size_t index = block_y * block_extent.width + block_x;
      const size_t pixel_x = block_x * kJxlBlockDimension;
      const size_t pixel_y = block_y * kJxlBlockDimension;
      const float mask_value = ComputeMask((*quant_field)[index]);
      const float gamma_value = GammaModulation(
        opsin,
        pixel_x,
        pixel_y,
        mask_value);
      const float high_frequency_value = HighFrequencyModulation(
        opsin.plane[1],
        pixel_x,
        pixel_y,
        gamma_value);
      const float blue_value = BlueModulation(
        opsin,
        pixel_x,
        pixel_y,
        gamma_value);
      const float exponent = std::min(high_frequency_value, blue_value);
      (*quant_field)[index] =
        fast_math::FastPow2(exponent * 1.442695041f) * multiplier + addend;
    }
  }
}

Status ValidateInputs(
  ConstImage3FView opsin,
  InitialQuantizationOptions options,
  InitialQuantFieldOutput output,
  Extent2D* block_extent,
  size_t* pixel_count) {

  if (!opsin.valid()) {
    return Status::InvalidArgument(
      "Opsin image is invalid");
  }

  if (opsin.width() % kJxlBlockDimension != 0 ||
      opsin.height() % kJxlBlockDimension != 0) {
    return Status::InvalidArgument(
      "Opsin image must be padded to complete 8x8 blocks");
  }

  if (opsin.width() >
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
      opsin.height() >
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
    return Status::InvalidArgument(
      "Opsin image dimensions are too large");
  }

  if (!std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f ||
      !std::isfinite(options.rescale) ||
      options.rescale <= 0.0f) {
    return Status::InvalidArgument(
      "Initial quantization options must be finite and positive");
  }

  *block_extent = {
    .width = opsin.width() / kJxlBlockDimension,
    .height = opsin.height() / kJxlBlockDimension,
  };
  if (!output.quant_field.valid() ||
      !output.strategy_mask.valid() ||
      !output.pixel_mask.valid() ||
      output.quant_field.extent != *block_extent ||
      output.strategy_mask.extent != *block_extent ||
      output.pixel_mask.extent != opsin.extent()) {
    return Status::InvalidArgument(
      "Initial quantization outputs have invalid geometry");
  }

  if (!opsin.extent().try_area(pixel_count)) {
    return Status::InvalidArgument(
      "Opsin image dimensions are too large");
  }

  for (const ConstPlaneF32View plane : opsin.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.Row(y)[x])) {
          return Status::InvalidArgument(
            "Opsin samples must be finite");
        }
      }
    }
  }

  return Status::Ok();
}

void CopyPlane(
  Extent2D extent,
  const std::vector<float>& source,
  PlaneF32View destination) {

  for (size_t y = 0; y < extent.height; ++y) {
    std::copy_n(
      source.data() + y * extent.width,
      extent.width,
      destination.Row(y));
  }
}

}  // namespace

Status ComputeInitialQuantDc(
  float butteraugli_target,
  float* quant_dc) {

  if (quant_dc == nullptr) {
    return Status::InvalidArgument(
      "Initial DC quantization output is null");
  }

  if (!std::isfinite(butteraugli_target) ||
      butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "Butteraugli target must be finite and positive");
  }

  constexpr float kDcMul = 0.3f;
  const float nonlinear_target = kDcMul * std::pow(
    butteraugli_target / kDcMul,
    kDcQuantPower);
  const float target_dc = std::max(
    0.5f * butteraugli_target,
    std::min(butteraugli_target, nonlinear_target));
  *quant_dc = std::min(kDcQuant / target_dc, 50.0f);
  return Status::Ok();
}

Status ComputeInitialQuantField(
  ConstImage3FView opsin,
  InitialQuantizationOptions options,
  InitialQuantFieldOutput output) {

  Extent2D block_extent;
  size_t pixel_count = 0;
  Status status = ValidateInputs(
    opsin,
    options,
    output,
    &block_extent,
    &pixel_count);
  if (!status.ok()) {
    return status;
  }

  size_t block_count = 0;
  if (!block_extent.try_area(&block_count)) {
    return Status::InvalidArgument(
      "Block grid dimensions are too large");
  }

  try {
    std::vector<float> pixel_mask(pixel_count);
    std::vector<float> row_differences(opsin.width());
    const Extent2D pre_erosion_extent{
      .width = opsin.width() / 4,
      .height = opsin.height() / 4,
    };
    std::vector<float> pre_erosion(
      pre_erosion_extent.width * pre_erosion_extent.height);

    constexpr float kMatchGammaOffset = 0.019f;
    constexpr float kDifferenceLimit = 0.2f;

    for (size_t y = 0; y < opsin.height(); ++y) {
      const size_t top = y == 0 ? y : y - 1;
      const size_t bottom = y + 1 < opsin.height() ? y + 1 : y;
      const float* row = opsin.plane[1].Row(y);
      const float* top_row = opsin.plane[1].Row(top);
      const float* bottom_row = opsin.plane[1].Row(bottom);

      for (size_t x = 0; x < opsin.width(); ++x) {
        const size_t left = x == 0 ? x : x - 1;
        const size_t right = x + 1 < opsin.width() ? x + 1 : x;
        const float base = 0.25f * (
          bottom_row[x] + top_row[x] + row[left] + row[right]);
        const float gamma = GammaDerivativeRatio<false>(
          row[x] + kMatchGammaOffset);

        float pixel_difference = std::abs(gamma * (row[x] - base));
        pixel_difference = std::log1p(pixel_difference);
        pixel_mask[y * opsin.width() + x] =
          1.0f / (pixel_difference + 0.01f);

        float block_difference = gamma * (row[x] - base);
        block_difference *= block_difference;
        block_difference = std::min(block_difference, kDifferenceLimit);
        block_difference = MaskingSqrt(block_difference);

        if ((y & 3u) == 0) {
          row_differences[x] = block_difference;
        } else {
          row_differences[x] += block_difference;
        }
      }

      if ((y & 3u) == 3) {
        float* destination = pre_erosion.data() +
          (y / 4) * pre_erosion_extent.width;
        for (size_t x = 0; x < pre_erosion_extent.width; ++x) {
          destination[x] = (
            row_differences[x * 4] +
            row_differences[x * 4 + 1] +
            row_differences[x * 4 + 2] +
            row_differences[x * 4 + 3]) * 0.25f;
        }
      }
    }

    std::vector<float> quant_field;
    FuzzyErosion(
      options.butteraugli_target,
      pre_erosion_extent,
      pre_erosion,
      block_extent,
      &quant_field);

    std::vector<float> strategy_mask(block_count);
    for (size_t i = 0; i < block_count; ++i) {
      strategy_mask[i] = 1.0f / (quant_field[i] + 0.001f);
    }

    PerBlockModulations(
      opsin,
      options,
      block_extent,
      &quant_field);

    std::vector<float> blurred_pixel_mask;
    BlurPixelMask(opsin.extent(), pixel_mask, &blurred_pixel_mask);

    const auto valid_values = [](const std::vector<float>& values) {
      return std::ranges::all_of(
        values,
        [](float value) {
          return std::isfinite(value) && value > 0.0f;
        });
    };
    if (!valid_values(quant_field) ||
        !valid_values(strategy_mask) ||
        !valid_values(blurred_pixel_mask)) {
      return Status::InvalidArgument(
        "Initial quantization produced a non-finite result");
    }

    CopyPlane(block_extent, quant_field, output.quant_field);
    CopyPlane(block_extent, strategy_mask, output.strategy_mask);
    CopyPlane(opsin.extent(), blurred_pixel_mask, output.pixel_mask);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate initial quantization scratch storage");
  }

  return Status::Ok();
}

}  // namespace gjxl
