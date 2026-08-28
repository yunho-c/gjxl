// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Direct scalar parity for pinned libjxl's AdjustQuantBlockAC policy.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#include "codec/codestream.h"
#include "codec/quantization.h"
#include "core/ac_strategy.h"
#include "core/quantizer.h"

namespace {

constexpr std::array kStrategies = {
  gjxl::AcStrategyType::kDct8,
  gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x32,
  gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16,
  gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
};

constexpr std::array<gjxl::XybChannel, 3> kChannels = {
  gjxl::XybChannel::kX,
  gjxl::XybChannel::kY,
  gjxl::XybChannel::kB,
};

enum class Pattern {
  kFlat,
  kSparse,
  kActive,
  kHighFrequencyBorder,
  kThresholdTie,
  kQuantLimit,
};

struct OracleDecision {
  int32_t raw_quant = 0;
  std::array<float, 4> y_thresholds{};
};

bool EqualBits(float left, float right) {
  return std::bit_cast<uint32_t>(left) == std::bit_cast<uint32_t>(right);
}

void PinnedAdjustOne(
  const gjxl::Quantizer& quantizer,
  size_t channel,
  float matrix_multiplier,
  gjxl::AcStrategyType strategy,
  const gjxl::QuantizationMatrixView& matrix,
  std::span<const float> coefficients,
  std::array<float, 4>* thresholds,
  int32_t* raw_quant) {

  const size_t xsize = matrix.coefficient_extent.width / 8;
  const size_t ysize = matrix.coefficient_extent.height / 8;
  const float qac = quantizer.scale() * *raw_quant;
  if (xsize > 1 || ysize > 1) {
    for (float& threshold : *thresholds) {
      threshold -= std::clamp(
        0.003f * xsize * ysize, 0.0f, 0.08f);
      if (threshold < 0.54f) threshold = 0.54f;
    }
  }

  float border_sum = 0.0f;
  float error_sum = 0.0f;
  float value_sum = 0.0f;
  float nonzeros[4] = {};
  float maximum_error[4] = {};
  for (size_t y = 0; y < ysize * 8; ++y) {
    for (size_t x = 0; x < xsize * 8; ++x) {
      const size_t index = y * 8 * xsize + x;
      if (x < xsize && y < ysize) continue;
      const size_t quadrant =
        static_cast<size_t>(y >= ysize * 4) * 2 +
        static_cast<size_t>(x >= xsize * 4);
      const float value = coefficients[index] *
        (matrix.inverse_dequant[index] * qac * matrix_multiplier);
      const float quantized = std::abs(value) < (*thresholds)[quadrant]
        ? 0.0f
        : std::rint(value);
      const float error = std::abs(value - quantized);
      error_sum += error;
      value_sum += std::abs(quantized);
      if (channel == 1 && quantized == 0.0f &&
          maximum_error[quadrant] < error) {
        maximum_error[quadrant] = error;
      }
      if (quantized != 0.0f) {
        nonzeros[quadrant] += std::abs(quantized);
        const bool in_corner = y >= 7 * ysize && x >= 7 * xsize;
        const bool on_border =
          y == ysize * 8 - 1 || x == xsize * 8 - 1;
        const bool in_larger_corner = x >= 4 * xsize && y >= 4 * ysize;
        if (in_corner || (on_border && in_larger_corner)) {
          border_sum += std::abs(value);
        }
      }
    }
  }

  if (channel == 1 && value_sum * 8 < xsize * ysize) {
    constexpr double kLimit[4] = {0.46, 0.46, 0.46, 0.46};
    constexpr double kMultiplier[4] = {
      0.9999, 0.9999, 0.9999, 0.9999};
    const int32_t original_quant = *raw_quant;
    int32_t new_quant = *raw_quant;
    for (size_t quadrant = 1; quadrant < 4; ++quadrant) {
      if (nonzeros[quadrant] == 0.0f &&
          maximum_error[quadrant] > kLimit[quadrant]) {
        new_quant = original_quant + 1;
        break;
      }
    }
    *raw_quant = new_quant;
    if (nonzeros[3] == 0.0f && maximum_error[3] > kLimit[3]) {
      (*thresholds)[3] = kMultiplier[3] * maximum_error[3] *
        new_quant / original_quant;
    } else if ((nonzeros[1] == 0.0f && maximum_error[1] > kLimit[1]) ||
               (nonzeros[2] == 0.0f && maximum_error[2] > kLimit[2])) {
      (*thresholds)[1] = kMultiplier[1] *
        std::max(maximum_error[1], maximum_error[2]) *
        new_quant / original_quant;
      (*thresholds)[2] = (*thresholds)[1];
    } else if (nonzeros[0] == 0.0f &&
               maximum_error[0] > kLimit[0]) {
      (*thresholds)[0] = kMultiplier[0] * maximum_error[0] *
        new_quant / original_quant;
    }
  }

  const float all =
    nonzeros[0] + nonzeros[1] + nonzeros[2] + nonzeros[3] + 1.0f;
  constexpr float kBorderMultiplier[3] = {70.0f, 30.0f, 60.0f};
  if (kBorderMultiplier[channel] * border_sum >= all) {
    *raw_quant += kBorderMultiplier[channel] * border_sum / all;
    if (*raw_quant >= gjxl::kMaxRawQuant) {
      *raw_quant = gjxl::kMaxRawQuant - 1;
    }
  }
  if (strategy == gjxl::AcStrategyType::kDct8 &&
      nonzeros[0] + nonzeros[1] + nonzeros[2] + nonzeros[3] < 11) {
    ++*raw_quant;
    if (*raw_quant >= gjxl::kMaxRawQuant) {
      *raw_quant = gjxl::kMaxRawQuant - 1;
    }
  }

  constexpr double kFirstMultiplier[4][3] = {
    {0.22080615753848404, 0.45797479824262011, 0.29859235095977965},
    {0.70109486510286834, 0.16185281305512639, 0.14387691730035473},
    {0.114985964456218638, 0.44656840441027695, 0.10587658215149048},
    {0.46849665264409396, 0.41239077937781954, 0.088667407767185444},
  };
  constexpr double kSecondMultiplier[4][3] = {
    {0.27450281941822197, 1.1255766549984996, 0.98950459134128388},
    {0.4652168675598285, 0.40945807983455818, 0.36581899811751367},
    {0.28034972424715715, 0.9182653201929738, 1.5581531543057416},
    {0.26873118114033728, 0.68863712390392484, 1.2082185408666786},
  };
  constexpr double kQuantNormalizer = 2.2942708343284721;
  error_sum *= kQuantNormalizer;
  value_sum *= kQuantNormalizer;
  if (strategy >= gjxl::AcStrategyType::kDct16x16) {
    size_t strategy_class = 3;
    if (strategy == gjxl::AcStrategyType::kDct32x16 ||
        strategy == gjxl::AcStrategyType::kDct16x32) {
      strategy_class = 1;
    } else if (strategy == gjxl::AcStrategyType::kDct16x16) {
      strategy_class = 0;
    } else if (strategy == gjxl::AcStrategyType::kDct32x32) {
      strategy_class = 2;
    }
    int32_t step = error_sum /
      (kFirstMultiplier[strategy_class][channel] *
         xsize * ysize * 8 * 8 +
       kSecondMultiplier[strategy_class][channel] * value_sum);
    step = std::clamp(step, 0, 2);
    if (error_sum >
        kFirstMultiplier[strategy_class][channel] *
          xsize * ysize * 8 * 8 +
        kSecondMultiplier[strategy_class][channel] * value_sum) {
      *raw_quant += step;
      if (*raw_quant >= gjxl::kMaxRawQuant) {
        *raw_quant = gjxl::kMaxRawQuant - 1;
      }
    }
  }

  const int32_t divisor = xsize * ysize;
  const float minimum_nonzeros =
    std::min({nonzeros[0], nonzeros[1], nonzeros[2], nonzeros[3]});
  int32_t activity = 15;
  if (minimum_nonzeros < 15.0f * divisor) {
    activity =
      (static_cast<int32_t>(minimum_nonzeros) + divisor / 2) / divisor;
  }
  int32_t adjusted_quant = *raw_quant - activity;
  if (channel == 1) {
    for (size_t quadrant = 1; quadrant < 4; ++quadrant) {
      (*thresholds)[quadrant] += 0.01 * activity;
    }
  }
  const int32_t original_limit = std::max(4, *raw_quant / 2);
  if (adjusted_quant < original_limit) adjusted_quant = original_limit;
  *raw_quant = adjusted_quant;
}

OracleDecision PinnedDecision(
  gjxl::AcStrategyType strategy,
  const gjxl::Quantizer& quantizer,
  int32_t initial_raw_quant,
  const std::array<float, 3>& matrix_multipliers,
  const std::array<gjxl::QuantizationMatrixView, 3>& matrices,
  const std::array<std::vector<float>, 3>& coefficients) {

  OracleDecision result;
  for (size_t channel : {size_t{1}, size_t{0}, size_t{2}}) {
    std::array<float, 4> thresholds = {0.58f, 0.64f, 0.64f, 0.64f};
    int32_t raw_quant = initial_raw_quant;
    PinnedAdjustOne(
      quantizer, channel, matrix_multipliers[channel], strategy,
      matrices[channel], coefficients[channel], &thresholds, &raw_quant);
    if (channel == 1) result.y_thresholds = thresholds;
    result.raw_quant = std::max(result.raw_quant, raw_quant);
  }
  return result;
}

void FillCoefficients(
  Pattern pattern,
  size_t channel,
  int32_t raw_quant,
  float matrix_multiplier,
  const gjxl::Quantizer& quantizer,
  const gjxl::QuantizationMatrixView& matrix,
  std::vector<float>* coefficients) {

  std::fill(coefficients->begin(), coefficients->end(), 0.0f);
  const size_t width = matrix.coefficient_extent.width;
  const size_t height = matrix.coefficient_extent.height;
  switch (pattern) {
    case Pattern::kFlat:
      return;
    case Pattern::kSparse:
      for (size_t i = channel + 3; i < coefficients->size(); i += 97) {
        (*coefficients)[i] = (i & 1) == 0 ? 0.31f : -0.27f;
      }
      return;
    case Pattern::kActive:
      for (size_t i = 0; i < coefficients->size(); ++i) {
        (*coefficients)[i] =
          static_cast<float>(static_cast<int32_t>(
            (i * 37 + channel * 13) % 101) - 50) * 0.019f;
      }
      return;
    case Pattern::kHighFrequencyBorder:
    case Pattern::kQuantLimit:
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          if (x + 1 == width || y + 1 == height) {
            (*coefficients)[y * width + x] =
              ((x + y + channel) & 1) == 0 ? 0.83f : -0.71f;
          }
        }
      }
      return;
    case Pattern::kThresholdTie: {
      const size_t x = width - 1;
      const size_t y = height - 1;
      const size_t index = y * width + x;
      const size_t block_count = width * height / 64;
      const float threshold = std::max(
        0.54f,
        0.64f - std::clamp(0.003f * block_count, 0.0f, 0.08f));
      (*coefficients)[index] = threshold /
        (matrix.inverse_dequant[index] * quantizer.scale() *
         raw_quant * matrix_multiplier);
      return;
    }
  }
}

std::vector<int32_t> PinnedQuantize(
  const gjxl::Quantizer& quantizer,
  int32_t raw_quant,
  float matrix_multiplier,
  const gjxl::QuantizationMatrixView& matrix,
  std::span<const float> coefficients,
  std::array<float, 4> thresholds) {

  std::vector<int32_t> result(coefficients.size());
  const float qac = quantizer.scale() * raw_quant;
  for (size_t y = 0; y < matrix.coefficient_extent.height; ++y) {
    for (size_t x = 0; x < matrix.coefficient_extent.width; ++x) {
      const size_t index = y * matrix.coefficient_extent.width + x;
      const size_t quadrant =
        static_cast<size_t>(y >= matrix.coefficient_extent.height / 2) * 2 +
        static_cast<size_t>(x >= matrix.coefficient_extent.width / 2);
      const float value = matrix.inverse_dequant[index] *
        (qac * matrix_multiplier) * coefficients[index];
      result[index] = std::abs(value) < thresholds[quadrant]
        ? 0
        : static_cast<int32_t>(std::rint(value));
    }
  }
  return result;
}

bool CheckPinnedParity() {
  gjxl::Quantizer quantizer;
  if (!gjxl::Quantizer::Create({3541, 10}, &quantizer).ok()) return false;
  const std::array<float, 3> matrix_multipliers = {
    gjxl::QuantizationMatrixMultiplier(1),
    1.0f,
    gjxl::QuantizationMatrixMultiplier(3),
  };
  constexpr std::array patterns = {
    Pattern::kFlat,
    Pattern::kSparse,
    Pattern::kActive,
    Pattern::kHighFrequencyBorder,
    Pattern::kThresholdTie,
    Pattern::kQuantLimit,
  };

  for (gjxl::AcStrategyType strategy : kStrategies) {
    std::array<gjxl::QuantizationMatrixView, 3> matrices;
    for (size_t channel = 0; channel < 3; ++channel) {
      if (!gjxl::GetDefaultQuantizationMatrix(
            strategy, kChannels[channel], &matrices[channel]).ok()) {
        return false;
      }
    }
    for (Pattern pattern : patterns) {
      const int32_t initial_raw_quant = pattern == Pattern::kQuantLimit
        ? gjxl::kMaxRawQuant
        : 37;
      std::array<std::vector<float>, 3> coefficients;
      std::array<std::span<const float>, 3> coefficient_views;
      for (size_t channel = 0; channel < 3; ++channel) {
        coefficients[channel].resize(matrices[channel].dequant.size());
        FillCoefficients(
          pattern, channel, initial_raw_quant, matrix_multipliers[channel],
          quantizer, matrices[channel], &coefficients[channel]);
        coefficient_views[channel] = coefficients[channel];
      }

      const OracleDecision expected = PinnedDecision(
        strategy, quantizer, initial_raw_quant, matrix_multipliers,
        matrices, coefficients);
      gjxl::AdjustedAcQuantization actual;
      const gjxl::Status status = gjxl::SelectAdjustedAcQuantization(
        strategy, quantizer, initial_raw_quant, matrix_multipliers,
        coefficient_views, &actual);
      bool thresholds_equal = true;
      for (size_t i = 0; i < 4; ++i) {
        thresholds_equal &=
          EqualBits(actual.y_thresholds[i], expected.y_thresholds[i]);
      }
      if (!status.ok() || actual.raw_quant != expected.raw_quant ||
          !thresholds_equal) {
        std::cerr << "Adjusted decision differs from pinned policy for "
                  << gjxl::GetAcStrategyInfo(strategy)->name << " pattern "
                  << static_cast<int>(pattern) << ": raw="
                  << actual.raw_quant << " expected=" << expected.raw_quant
                  << '\n';
        return false;
      }

      std::vector<int32_t> actual_y(coefficients[1].size());
      if (!gjxl::QuantizeAdjustedYAcBlock(
            strategy, quantizer, actual, coefficients[1], actual_y).ok()) {
        return false;
      }
      const std::vector<int32_t> expected_y = PinnedQuantize(
        quantizer, expected.raw_quant, 1.0f, matrices[1], coefficients[1],
        expected.y_thresholds);
      if (actual_y != expected_y) {
        std::cerr << "Adjusted Y coefficients differ from pinned policy\n";
        return false;
      }
      for (size_t channel : {size_t{0}, size_t{2}}) {
        std::array<float, 4> thresholds = {0.58f, 0.62f, 0.62f, 0.62f};
        const size_t block_count =
          matrices[channel].dequant.size() / 64;
        if (block_count >= 4) {
          for (float& threshold : thresholds) {
            threshold = std::max(
              0.5f,
              threshold - 0.00744f * block_count);
          }
        }
        const std::vector<int32_t> expected_channel = PinnedQuantize(
          quantizer, expected.raw_quant, matrix_multipliers[channel],
          matrices[channel], coefficients[channel], thresholds);
        std::vector<int32_t> actual_channel(coefficients[channel].size());
        if (!gjxl::QuantizeAcBlock(
              strategy,
              quantizer,
              actual.raw_quant,
              {
                .channel = kChannels[channel],
                .matrix_multiplier = matrix_multipliers[channel],
              },
              coefficients[channel],
              actual_channel).ok() ||
            actual_channel != expected_channel) {
          std::cerr << "Shared X/B coefficients differ from pinned policy\n";
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckInvalidInputIsAtomic() {
  gjxl::Quantizer quantizer;
  if (!gjxl::Quantizer::Create({3541, 10}, &quantizer).ok()) return false;
  std::array<std::vector<float>, 3> coefficients;
  std::array<std::span<const float>, 3> views;
  for (size_t channel = 0; channel < 3; ++channel) {
    coefficients[channel].assign(64, 0.0f);
    views[channel] = coefficients[channel];
  }
  const gjxl::AdjustedAcQuantization sentinel{
    .raw_quant = 123,
    .y_thresholds = {1.0f, 2.0f, 3.0f, 4.0f},
  };
  gjxl::AdjustedAcQuantization output = sentinel;
  coefficients[2][17] = std::numeric_limits<float>::quiet_NaN();
  views[2] = coefficients[2];
  if (gjxl::SelectAdjustedAcQuantization(
        gjxl::AcStrategyType::kDct8,
        quantizer,
        37,
        {1.0f, 1.0f, 1.0f},
        views,
        &output).code() != gjxl::StatusCode::kInvalidArgument ||
      output != sentinel ||
      gjxl::SelectAdjustedAcQuantization(
        gjxl::AcStrategyType::kDct8,
        quantizer,
        37,
        {1.0f, 1.0f, 1.0f},
        views,
        nullptr).code() != gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Invalid adjustment input was not rejected atomically\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckPinnedParity() || !CheckInvalidInputIsAtomic()) {
    return EXIT_FAILURE;
  }
  std::cout << "All pinned AC quantization-adjustment tests passed.\n";
  return EXIT_SUCCESS;
}
