// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/adaptive_quantization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "codec/color_transform.h"
#include "codec/convolution.h"
#include "codec/quantization.h"
#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"
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

Status BlurPixelMask(
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

  output->resize(extent.width * extent.height);
  return ConvolveSymmetric5(
    {
      .data = input.data(),
      .extent = extent,
      .stride = extent.width,
    },
    {
      .distance0 = normalize,
      .distance1 = normalize * kFilter[0],
      .distance2 = normalize * kFilter[2],
      .distance4 = normalize * kFilter[1],
      .distance8 = normalize * kFilter[4],
      .distance5 = normalize * kFilter[3],
    },
    {
      .data = output->data(),
      .extent = extent,
      .stride = extent.width,
    });
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

  if (!BlockGrid::IsPaddedPixelExtent(opsin.extent())) {
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

  *block_extent =
    BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
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
    status = BlurPixelMask(
      opsin.extent(),
      pixel_mask,
      &blurred_pixel_mask);
    if (!status.ok()) {
      return status;
    }

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

    CopyContiguousPlane(quant_field, output.quant_field);
    CopyContiguousPlane(strategy_mask, output.strategy_mask);
    CopyContiguousPlane(blurred_pixel_mask, output.pixel_mask);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate initial quantization scratch storage");
  }

  return Status::Ok();
}

Status AdjustQuantField(
  const AcStrategyGrid& strategies,
  float butteraugli_target,
  ConstPlaneF32View input,
  PlaneF32View output) {

  if (!strategies.complete() ||
      !input.valid() ||
      !output.valid() ||
      input.extent != strategies.extent() ||
      output.extent != input.extent ||
      !std::isfinite(butteraugli_target) ||
      butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "Adjusted quant field inputs are invalid");
  }

  size_t value_count = 0;
  if (!input.extent.try_area(&value_count)) {
    return Status::InvalidArgument(
      "Adjusted quant field dimensions are too large");
  }

  try {
    std::vector<float> adjusted(value_count);
    for (size_t y = 0; y < input.extent.height; ++y) {
      for (size_t x = 0; x < input.extent.width; ++x) {
        const float value = input.Row(y)[x];
        if (!std::isfinite(value) || value <= 0.0f) {
          return Status::InvalidArgument(
            "Quant field must contain finite positive values");
        }
        adjusted[y * input.extent.width + x] = value;
      }
    }

    float mean_max_mixer = 1.0f;
    constexpr float kMixerLimit = 1.54138f;
    constexpr float kMixerSlope = 0.56391f;
    if (butteraugli_target > kMixerLimit) {
      mean_max_mixer = std::max(
        0.0f,
        mean_max_mixer -
          (butteraugli_target - kMixerLimit) * kMixerSlope);
    }

    const Status status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const Extent2D covered =
          GetAcStrategyInfo(strategy)->covered_blocks;
        float maximum = adjusted[
          block_y * input.extent.width + block_x];
        float mean = 0.0f;
        for (size_t dy = 0; dy < covered.height; ++dy) {
          for (size_t dx = 0; dx < covered.width; ++dx) {
            const float value = adjusted[
              (block_y + dy) * input.extent.width + block_x + dx];
            maximum = std::max(maximum, value);
            mean += value;
          }
        }
        const size_t block_count = covered.width * covered.height;
        mean /= static_cast<float>(block_count);
        float result = maximum;
        if (block_count >= 4) {
          result = maximum * mean_max_mixer +
            mean * (1.0f - mean_max_mixer);
        }
        for (size_t dy = 0; dy < covered.height; ++dy) {
          for (size_t dx = 0; dx < covered.width; ++dx) {
            adjusted[
              (block_y + dy) * input.extent.width + block_x + dx] = result;
          }
        }
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }

    CopyContiguousPlane(adjusted, output);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate adjusted quant field storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Adjusted quant field dimensions are too large");
  }
  return Status::Ok();
}

namespace {

struct QuantizationEvaluation {
  std::vector<float> block_distance;
  Image3FBuffer reconstructed_linear;
  VarDctEncoderFrame frame;
  double score = 0.0;
};

Status EvaluateQuantization(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View quant_field,
  ConstPlaneU8View epf_sharpness,
  float quant_dc,
  AdaptiveQuantizationOptions options,
  QuantizationEvaluation* evaluation) {

  const Extent2D block_extent = strategies.extent();
  size_t block_count = 0;
  if (evaluation == nullptr || !block_extent.try_area(&block_count)) {
    return Status::InvalidArgument(
      "Adaptive-quantization evaluation output is invalid");
  }

  QuantizationEvaluation result;
  std::vector<int32_t> raw_quant(block_count);
  Quantizer quantizer;
  Status status = CreateQuantizerFromField(
    quant_dc,
    quant_field,
    {raw_quant.data(), block_extent, block_extent.width},
    &quantizer);
  if (!status.ok()) {
    return status;
  }

  ColorCorrelationMap color_correlation;
  status = ComputeFinalColorCorrelationMap(
    opsin,
    strategies,
    {raw_quant.data(), block_extent, block_extent.width},
    quantizer,
    options.fast_color_correlation,
    &color_correlation);
  if (!status.ok()) {
    return status;
  }

  std::vector<float> inverse_sigma(block_count);
  status = ComputeEpfInverseSigma(
    strategies,
    {raw_quant.data(), block_extent, block_extent.width},
    quantizer,
    epf_sharpness,
    options.epf_sigma,
    {inverse_sigma.data(), block_extent, block_extent.width});
  if (!status.ok()) {
    return status;
  }

  FrameGeometry geometry;
  status = FrameGeometry::Create(original_linear_rgb.extent(), &geometry);
  if (!status.ok()) {
    return status;
  }
  status = ComputeQuantizedCoefficients(
    opsin,
    {
      .geometry = geometry,
      .strategies = &strategies,
      .raw_quant_field = {
        raw_quant.data(), block_extent, block_extent.width},
      .quantizer = &quantizer,
      .color_correlation = &color_correlation,
      .epf_sharpness = epf_sharpness,
    },
    options.coefficient_coding,
    &result.frame);
  if (!status.ok()) {
    return status;
  }

  Image3FBuffer reconstructed_opsin(opsin.extent());
  status = ReconstructQuantizedCoefficients(
    result.frame,
    reconstructed_opsin.view());
  if (!status.ok()) {
    return status;
  }

  Image3FBuffer filtered_opsin(opsin.extent());
  status = ApplyLoopFilters(
    reconstructed_opsin.const_view(),
    {inverse_sigma.data(), block_extent, block_extent.width},
    options.loop_filter,
    filtered_opsin.view());
  if (!status.ok()) {
    return status;
  }

  result.reconstructed_linear.resize(original_linear_rgb.extent());
  status = OpsinToLinearRgb(
    filtered_opsin.cropped_view(original_linear_rgb.extent()),
    options.opsin_intensity_target,
    result.reconstructed_linear.view());
  if (!status.ok()) {
    return status;
  }

  size_t pixel_count = 0;
  if (!original_linear_rgb.extent().try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "Adaptive-quantization reference extent is too large");
  }
  std::vector<float> distance_map(pixel_count);
  status = ComputeButteraugliDistance(
    original_linear_rgb,
    result.reconstructed_linear.const_view(),
    options.butteraugli,
    {
      distance_map.data(),
      original_linear_rgb.extent(),
      original_linear_rgb.width(),
    },
    &result.score);
  if (!status.ok()) {
    return status;
  }

  result.block_distance.resize(block_count);
  status = ReduceButteraugliDistanceMap(
    {
      distance_map.data(),
      original_linear_rgb.extent(),
      original_linear_rgb.width(),
    },
    strategies,
    {result.block_distance.data(), block_extent, block_extent.width});
  if (!status.ok()) {
    return status;
  }

  *evaluation = std::move(result);
  return Status::Ok();
}

Status ValidateAdaptiveQuantizationInputs(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  const AdaptiveQuantizationOutput& output) {

  if (!original_linear_rgb.valid() ||
      !opsin.valid() ||
      !strategies.complete() ||
      !initial_quant_field.valid() ||
      !epf_sharpness.valid() ||
      !output.quant_field.valid() ||
      !output.block_distance_map.valid() ||
      !output.reconstructed_linear_rgb.valid() ||
      output.frame == nullptr ||
      output.score_history == nullptr) {
    return Status::InvalidArgument(
      "Adaptive-quantization inputs or outputs are invalid");
  }

  const Extent2D block_extent = strategies.extent();
  Extent2D padded_pixel_extent;
  if (!BlockGrid{block_extent}.try_padded_pixel_extent(
        &padded_pixel_extent) ||
      opsin.extent() != padded_pixel_extent ||
      initial_quant_field.extent != block_extent ||
      epf_sharpness.extent != block_extent ||
      output.quant_field.extent != block_extent ||
      output.block_distance_map.extent != block_extent ||
      output.reconstructed_linear_rgb.extent() !=
        original_linear_rgb.extent()) {
    return Status::InvalidArgument(
      "Adaptive-quantization image and block geometry do not match");
  }

  if (original_linear_rgb.width() > opsin.width() ||
      original_linear_rgb.height() > opsin.height() ||
      original_linear_rgb.width() <=
        opsin.width() - kJxlBlockDimension ||
      original_linear_rgb.height() <=
        opsin.height() - kJxlBlockDimension) {
    return Status::InvalidArgument(
      "Adaptive-quantization padding exceeds one partial block");
  }

  if (!std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f ||
      !std::isfinite(options.opsin_intensity_target) ||
      options.opsin_intensity_target <= 0.0f ||
      options.iterations > 4) {
    return Status::InvalidArgument(
      "Adaptive-quantization options are invalid");
  }

  return Status::Ok();
}

}  // namespace

Status FindBestQuantization(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationOutput output) {

  Status status = ValidateAdaptiveQuantizationInputs(
    original_linear_rgb,
    opsin,
    strategies,
    initial_quant_field,
    epf_sharpness,
    options,
    output);
  if (!status.ok()) {
    return status;
  }

  const Extent2D block_extent = strategies.extent();
  size_t block_count = 0;
  if (!block_extent.try_area(&block_count)) {
    return Status::InvalidArgument(
      "Adaptive-quantization block grid is too large");
  }

  try {
    std::vector<float> quant_field(block_count);
    status = AdjustQuantField(
      strategies,
      options.butteraugli_target,
      initial_quant_field,
      {quant_field.data(), block_extent, block_extent.width});
    if (!status.ok()) {
      return status;
    }
    const std::vector<float> adjusted_initial = quant_field;

    const auto [minimum_it, maximum_it] = std::minmax_element(
      adjusted_initial.begin(),
      adjusted_initial.end());
    const float initial_minimum = *minimum_it;
    const float initial_maximum = *maximum_it;
    const float initial_ratio = initial_maximum / initial_minimum;
    const float maximum_deviation = std::sqrt(250.0f / initial_ratio);
    const float asymmetry = std::min(2.0f, maximum_deviation);
    const float lower_bound =
      initial_minimum / (asymmetry * maximum_deviation);
    const float upper_bound =
      initial_maximum * (maximum_deviation / asymmetry);
    if (!std::isfinite(lower_bound) ||
        !std::isfinite(upper_bound) ||
        lower_bound <= 0.0f ||
        upper_bound < lower_bound ||
        upper_bound / lower_bound >= 253.0f ||
        upper_bound >
          static_cast<float>(std::numeric_limits<long>::max()) /
            static_cast<float>(kQuantGlobalScaleDenominator)) {
      return Status::InvalidArgument(
        "Initial quant field cannot form libjxl AQ bounds");
    }

    float quant_dc = 0.0f;
    status = ComputeInitialQuantDc(
      options.butteraugli_target,
      &quant_dc);
    if (!status.ok()) {
      return status;
    }

    std::vector<double> score_history;
    score_history.reserve(options.iterations + 1);
    QuantizationEvaluation evaluation;
    for (size_t iteration = 0;
         iteration <= options.iterations;
         ++iteration) {
      status = EvaluateQuantization(
        original_linear_rgb,
        opsin,
        strategies,
        {
          quant_field.data(),
          block_extent,
          block_extent.width,
        },
        epf_sharpness,
        quant_dc,
        options,
        &evaluation);
      if (!status.ok()) {
        return status;
      }
      score_history.push_back(evaluation.score);
      if (iteration == options.iterations) {
        break;
      }

      // The second update is constrained toward the initial field to reduce
      // oscillation caused by DC reconstruction, matching libjxl.
      if (iteration == 1) {
        for (size_t index = 0; index < block_count; ++index) {
          const float clamp =
            0.4f * quant_field[index] +
            0.6f * adjusted_initial[index];
          if (quant_field[index] < clamp) {
            quant_field[index] = std::clamp(
              clamp,
              lower_bound,
              upper_bound);
          }
        }
      }

      const double power = iteration < 2 ? 0.2 : 0.0;
      for (size_t index = 0; index < block_count; ++index) {
        const float difference = evaluation.block_distance[index] /
          options.butteraugli_target;
        if (!std::isfinite(difference) || difference < 0.0f) {
          return Status::Internal(
            "Adaptive quantization produced an invalid block distance");
        }

        if (difference <= 1.0f) {
          if (power != 0.0) {
            quant_field[index] *= static_cast<float>(
              std::pow(difference, power));
          }
        } else {
          const float old = quant_field[index];
          quant_field[index] *= difference;
          const long old_raw = std::lround(
            old * evaluation.frame.quantizer().inverse_global_scale());
          const long new_raw = std::lround(
            quant_field[index] *
            evaluation.frame.quantizer().inverse_global_scale());
          if (old_raw == new_raw) {
            quant_field[index] = old + evaluation.frame.quantizer().scale();
          }
        }
        quant_field[index] = std::clamp(
          quant_field[index],
          lower_bound,
          upper_bound);
      }
    }

    CopyContiguousPlane(quant_field, output.quant_field);
    CopyContiguousPlane(
      evaluation.block_distance,
      output.block_distance_map);
    CopyImage(
      evaluation.reconstructed_linear.const_view(),
      output.reconstructed_linear_rgb);
    *output.frame = std::move(evaluation.frame);
    *output.score_history = std::move(score_history);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate adaptive-quantization scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Adaptive-quantization dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace gjxl
