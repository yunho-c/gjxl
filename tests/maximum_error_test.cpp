// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Direct parity tests for pinned maximum-error reduction and update policy.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/maximum_error.h"
#include "codec/adaptive_quantization_internal.h"
#include "codec/color_transform.h"
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

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D extent, float value = 0.0f)
      : extent(extent), stride(extent.width + 3) {
    for (auto& channel : plane) {
      channel.assign(stride * extent.height, value);
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView View() const {
    return {.plane = {{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }

  [[nodiscard]] gjxl::Image3FView MutableView() {
    return {.plane = {{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

bool CheckReductionAllStrategies() {
  constexpr std::array<float, 3> kLimits = {0.02f, 0.04f, 0.08f};
  for (gjxl::AcStrategyType strategy : kStrategies) {
    const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
    if (info == nullptr) return false;
    const gjxl::Extent2D block_extent = info->covered_blocks;
    const gjxl::Extent2D padded_extent{
      block_extent.width * gjxl::kJxlBlockDimension,
      block_extent.height * gjxl::kJxlBlockDimension,
    };
    const gjxl::Extent2D source_extent{
      padded_extent.width - 1,
      padded_extent.height - 1,
    };
    gjxl::AcStrategyGrid strategies;
    if (!gjxl::AcStrategyGrid::Create(block_extent, &strategies).ok() ||
        !strategies.Set(0, 0, strategy).ok()) {
      return false;
    }

    ImageStorage reference(padded_extent);
    ImageStorage reconstructed(padded_extent);
    // Each channel independently contributes a known normalized maximum.
    reconstructed.plane[0][1 * reconstructed.stride + 1] = 0.005f;
    reconstructed.plane[1][2 * reconstructed.stride + 2] = -0.03f;
    reconstructed.plane[2][3 * reconstructed.stride + 3] = 0.12f;
    // Padded samples are deliberately invalid and must be ignored.
    const size_t padded_index =
      (padded_extent.height - 1) * reconstructed.stride +
      padded_extent.width - 1;
    for (auto& channel : reconstructed.plane) {
      channel[padded_index] = std::numeric_limits<float>::quiet_NaN();
    }

    const size_t block_count = block_extent.width * block_extent.height;
    std::vector<float> block_error(block_count + 5, -777.0f);
    gjxl::MaximumErrorReduction reduction;
    const gjxl::Status status = gjxl::ReduceMaximumError(
      reference.View(), reconstructed.View(), source_extent, strategies,
      kLimits, {block_error.data(), block_extent, block_extent.width},
      &reduction);
    if (!status.ok() ||
        std::abs(reduction.channel_maximum[0] - 0.005f) > 1.0e-7f ||
        std::abs(reduction.channel_maximum[1] - 0.03f) > 1.0e-7f ||
        std::abs(reduction.channel_maximum[2] - 0.12f) > 1.0e-7f ||
        std::abs(reduction.normalized_maximum - 1.5f) > 1.0e-6f) {
      std::cerr << "Maximum-error reduction failed for " << info->name
                << ": " << status.message() << '\n';
      return false;
    }
    for (size_t index = 0; index < block_count; ++index) {
      if (std::abs(block_error[index] - 1.5f) > 1.0e-6f) {
        std::cerr << "Transform maximum was not propagated for "
                  << info->name << '\n';
        return false;
      }
    }
    for (size_t index = block_count; index < block_error.size(); ++index) {
      if (block_error[index] != -777.0f) return false;
    }
  }
  return true;
}

bool CheckPinnedUpdateIntervals() {
  constexpr gjxl::Extent2D kExtent{3, 1};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kExtent, &strategies).ok()) return false;
  strategies.fill_dct8();
  constexpr std::array<float, 3> kError = {0.25f, 0.75f, 1.5f};
  constexpr std::array<float, 3> kInput = {2.0f, 3.0f, 4.0f};
  constexpr std::array<float, 3> kExpected = {1.0f, 3.0f, 6.0f};
  std::array<float, 5> output = {-777.0f, -777.0f, -777.0f,
                                 -777.0f, -777.0f};
  bool limited = true;
  const gjxl::Status status = gjxl::UpdateMaximumErrorQuantField(
    strategies,
    {kError.data(), kExtent, kExtent.width},
    {kInput.data(), kExtent, kExtent.width},
    {output.data(), kExtent, kExtent.width},
    &limited);
  if (!status.ok() || limited) return false;
  for (size_t i = 0; i < kExpected.size(); ++i) {
    if (output[i] != kExpected[i]) {
      std::cerr << "Pinned maximum-error interval update differs\n";
      return false;
    }
  }
  return output[3] == -777.0f && output[4] == -777.0f &&
    gjxl::MaximumErrorQuantFieldMultiplier(0.25f) == 0.5f &&
    gjxl::MaximumErrorQuantFieldMultiplier(0.5f) == 1.0f &&
    gjxl::MaximumErrorQuantFieldMultiplier(1.0f) == 1.0f &&
    gjxl::MaximumErrorQuantFieldMultiplier(1.5f) == 1.5f;
}

bool CheckIndependentlyLimitingChannels() {
  constexpr gjxl::Extent2D kBlockExtent{1, 1};
  constexpr gjxl::Extent2D kPixelExtent{8, 8};
  constexpr std::array<float, 3> kLimits = {0.02f, 0.04f, 0.08f};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kBlockExtent, &strategies).ok()) {
    return false;
  }
  strategies.fill_dct8();
  ImageStorage reference(kPixelExtent);
  for (size_t dominant = 0; dominant < 3; ++dominant) {
    ImageStorage reconstructed(kPixelExtent);
    for (size_t channel = 0; channel < 3; ++channel) {
      reconstructed.plane[channel][channel] =
        kLimits[channel] * (channel == dominant ? 1.5f : 0.25f);
    }
    float block_error = -777.0f;
    gjxl::MaximumErrorReduction reduction;
    if (!gjxl::ReduceMaximumError(
          reference.View(), reconstructed.View(), kPixelExtent, strategies,
          kLimits, {&block_error, kBlockExtent, 1}, &reduction).ok() ||
        std::abs(block_error - 1.5f) > 1.0e-6f ||
        std::abs(reduction.normalized_maximum - 1.5f) > 1.0e-6f ||
        std::abs(reduction.channel_maximum[dominant] -
                 1.5f * kLimits[dominant]) > 1.0e-6f) {
      std::cerr << "Channel " << dominant
                << " did not independently limit maximum-error reduction\n";
      return false;
    }
  }
  return true;
}

bool CheckFootprintUpdateAndLimit() {
  constexpr gjxl::Extent2D kExtent{4, 2};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kExtent, &strategies).ok() ||
      !strategies.Set(
        0, 0, gjxl::AcStrategyType::kDct16x32).ok()) {
    return false;
  }
  constexpr std::array<float, 8> kErrors = {
    80.0f, 0.1f, 0.2f, 0.3f,
    0.4f, 0.5f, 0.6f, 0.7f,
  };
  constexpr std::array<float, 8> kInput = {
    2.0f, 3.0f, 4.0f, 5.0f,
    6.0f, 7.0f, 8.0f, 9.0f,
  };
  std::array<float, 8> output{};
  bool limited = false;
  if (!gjxl::UpdateMaximumErrorQuantField(
        strategies,
        {kErrors.data(), kExtent, kExtent.width},
        {kInput.data(), kExtent, kExtent.width},
        {output.data(), kExtent, kExtent.width},
        &limited).ok() || !limited) {
    return false;
  }
  for (float value : output) {
    if (value != 128.0f) {
      std::cerr << "Footprint update did not use the anchor multiplier\n";
      return false;
    }
  }
  return true;
}

bool CheckInvalidInputIsAtomic() {
  constexpr gjxl::Extent2D kBlockExtent{1, 1};
  constexpr gjxl::Extent2D kPixelExtent{8, 8};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kBlockExtent, &strategies).ok()) {
    return false;
  }
  strategies.fill_dct8();
  ImageStorage reference(kPixelExtent);
  ImageStorage reconstructed(kPixelExtent);
  reconstructed.plane[0][0] = std::numeric_limits<float>::quiet_NaN();
  std::array<float, 3> block = {17.0f, 18.0f, 19.0f};
  gjxl::MaximumErrorReduction reduction{
    .channel_maximum = {4.0f, 5.0f, 6.0f},
    .normalized_maximum = 7.0f,
  };
  const gjxl::MaximumErrorReduction sentinel = reduction;
  if (gjxl::ReduceMaximumError(
        reference.View(), reconstructed.View(), kPixelExtent, strategies,
        {1.0f, 1.0f, 1.0f},
        {block.data(), kBlockExtent, kBlockExtent.width},
        &reduction).code() != gjxl::StatusCode::kInvalidArgument ||
      block != std::array<float, 3>{17.0f, 18.0f, 19.0f} ||
      reduction != sentinel) {
    std::cerr << "Invalid maximum-error reduction was not atomic\n";
    return false;
  }
  return true;
}

class SequenceEvaluator final
    : public gjxl::adaptive_quantization_internal::
        AdaptiveQuantizationEvaluator {
public:
  explicit SequenceEvaluator(std::vector<float> normalized)
      : normalized_(std::move(normalized)) {}

  gjxl::Status Evaluate(
    gjxl::ConstPlaneF32View quant_field,
    float quant_dc,
    bool is_final,
    gjxl::adaptive_quantization_internal::
      AdaptiveQuantizationEvaluation* evaluation,
    gjxl::adaptive_quantization_internal::EvaluationProfile*) override {

    if (evaluation == nullptr || call_count_ >= normalized_.size()) {
      return gjxl::Status::Internal("Unexpected policy evaluation");
    }
    if (quant_dc != 0x1.43d136p+2f ||
        is_final != (call_count_ + 1 == normalized_.size())) {
      return gjxl::Status::Internal(
        "Maximum-error initialization or final-evaluation flag differs");
    }
    fields_.emplace_back(
      quant_field.data,
      quant_field.data + quant_field.extent.width);
    gjxl::adaptive_quantization_internal::
      AdaptiveQuantizationEvaluation candidate;
    candidate.block_distance.assign(
      quant_field.extent.width * quant_field.extent.height,
      normalized_[call_count_]);
    if (!gjxl::Quantizer::Create({3541, 10}, &candidate.quantizer).ok()) {
      return gjxl::Status::Internal("Unable to create test quantizer");
    }
    candidate.score = normalized_[call_count_];
    candidate.maximum_error = {
      .channel_maximum = {
        normalized_[call_count_], 0.0f, 0.0f},
      .normalized_maximum = normalized_[call_count_],
    };
    ++call_count_;
    *evaluation = std::move(candidate);
    return gjxl::Status::Ok();
  }

  [[nodiscard]] const std::vector<std::vector<float>>& fields() const {
    return fields_;
  }

private:
  std::vector<float> normalized_;
  size_t call_count_ = 0;
  std::vector<std::vector<float>> fields_;
};

bool CheckFixedPolicyAndSelection() {
  namespace aqi = gjxl::adaptive_quantization_internal;
  constexpr gjxl::Extent2D kExtent{3, 1};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kExtent, &strategies).ok()) return false;
  strategies.fill_dct8();
  constexpr std::array<float, 3> kInitial = {1.0f, 1.0f, 1.0f};
  SequenceEvaluator evaluator({1.5f, 0.75f, 1.2f, 0.25f, 0.6f, 0.75f});
  gjxl::AdaptiveQuantizationOptions options;
  options.control_mode =
    gjxl::AdaptiveQuantizationControlMode::kMaximumError;
  options.maximum_error = {1.0f, 1.0f, 1.0f};
  options.butteraugli_target =
    std::numeric_limits<float>::quiet_NaN();
  options.iterations = 999;
  aqi::AdaptiveQuantizationPolicyResult result;
  const gjxl::Status status = aqi::RunAdaptiveQuantizationPolicy(
    strategies,
    {kInitial.data(), kExtent, kExtent.width},
    options,
    evaluator,
    &result,
    nullptr);
  const std::array<float, 3> expected = {1.5f, 1.5f, 1.5f};
  if (!status.ok() || result.quant_field.size() != expected.size() ||
      !std::equal(result.quant_field.begin(), result.quant_field.end(),
                  expected.begin()) ||
      evaluator.fields().size() != 6 ||
      evaluator.fields().back() !=
        std::vector<float>(expected.begin(), expected.end()) ||
      result.score_history.size() != 6 ||
      result.maximum_error.evaluation_count != 6 ||
      result.maximum_error.outcome != gjxl::MaximumErrorOutcome::kMet) {
    std::cerr << "Fixed maximum-error policy or feasible selection differs\n";
    return false;
  }
  return true;
}

bool CheckInfeasibilityOutcomes() {
  namespace aqi = gjxl::adaptive_quantization_internal;
  constexpr gjxl::Extent2D kExtent{1, 1};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kExtent, &strategies).ok()) return false;
  strategies.fill_dct8();
  gjxl::AdaptiveQuantizationOptions options;
  options.control_mode =
    gjxl::AdaptiveQuantizationControlMode::kMaximumError;
  options.maximum_error = {1.0f, 1.0f, 1.0f};

  constexpr std::array<float, 1> kRangeInitial = {128.0f};
  SequenceEvaluator range_evaluator({2.0f, 2.0f, 2.0f,
                                     2.0f, 2.0f, 2.0f});
  aqi::AdaptiveQuantizationPolicyResult range_result;
  if (!aqi::RunAdaptiveQuantizationPolicy(
        strategies,
        {kRangeInitial.data(), kExtent, kExtent.width},
        options,
        range_evaluator,
        &range_result,
        nullptr).ok() ||
      range_result.maximum_error.outcome !=
        gjxl::MaximumErrorOutcome::kQuantizationRangeExhausted) {
    std::cerr << "Quantization-range infeasibility was not reported\n";
    return false;
  }

  constexpr std::array<float, 1> kIterationInitial = {1.0f};
  SequenceEvaluator iteration_evaluator({1.1f, 1.1f, 1.1f,
                                         1.1f, 1.1f, 1.1f});
  aqi::AdaptiveQuantizationPolicyResult iteration_result;
  if (!aqi::RunAdaptiveQuantizationPolicy(
        strategies,
        {kIterationInitial.data(), kExtent, kExtent.width},
        options,
        iteration_evaluator,
        &iteration_result,
        nullptr).ok() ||
      iteration_result.maximum_error.outcome !=
        gjxl::MaximumErrorOutcome::kIterationLimit) {
    std::cerr << "Maximum-error iteration exhaustion was not reported\n";
    return false;
  }
  return true;
}

uint64_t Mix(uint64_t hash, uint64_t value) {
  for (size_t byte = 0; byte < sizeof(value); ++byte) {
    hash ^= static_cast<uint8_t>(value >> (8 * byte));
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t FrameFingerprint(const gjxl::VarDctEncoderFrame& frame) {
  uint64_t hash = 14695981039346656037ull;
  hash = Mix(hash, frame.quantizer().params().global_scale);
  hash = Mix(hash, frame.quantizer().params().quant_dc);
  const gjxl::ConstPlaneI32View raw = frame.raw_quant_field();
  for (size_t y = 0; y < raw.extent.height; ++y) {
    for (size_t x = 0; x < raw.extent.width; ++x) {
      hash = Mix(hash, static_cast<uint32_t>(raw.Row(y)[x]));
    }
  }
  const gjxl::ConstImage3I32View dc = frame.quantized_dc();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < dc.plane[channel].extent.height; ++y) {
      for (size_t x = 0; x < dc.plane[channel].extent.width; ++x) {
        hash = Mix(
          hash, static_cast<uint32_t>(dc.plane[channel].Row(y)[x]));
      }
    }
  }
  for (size_t group_index = 0; group_index < frame.ac_group_count();
       ++group_index) {
    gjxl::VarDctAcGroupView group;
    if (!frame.GetAcGroup(group_index, &group).ok()) return 0;
    for (size_t channel = 0; channel < 3; ++channel) {
      for (int32_t coefficient : group.coefficients[channel]) {
        hash = Mix(hash, static_cast<uint32_t>(coefficient));
      }
    }
  }
  return hash;
}

struct AqRun {
  explicit AqRun(gjxl::Extent2D block_extent, gjxl::Extent2D source_extent)
      : block_extent(block_extent),
        quant(block_extent.width * block_extent.height),
        error(block_extent.width * block_extent.height),
        reconstructed(source_extent) {}

  [[nodiscard]] gjxl::AdaptiveQuantizationOutput Output() {
    return {
      .quant_field = {quant.data(), block_extent, block_extent.width},
      .block_distance_map = {
        error.data(), block_extent, block_extent.width},
      .reconstructed_linear_rgb = reconstructed.MutableView(),
      .frame = &frame,
      .score_history = &history,
      .maximum_error_result = &result,
    };
  }

  gjxl::Extent2D block_extent;
  std::vector<float> quant;
  std::vector<float> error;
  ImageStorage reconstructed;
  gjxl::VarDctEncoderFrame frame;
  std::vector<double> history;
  gjxl::MaximumErrorResult result;
};

bool CheckEndToEndAllStrategies() {
  for (gjxl::AcStrategyType strategy : kStrategies) {
    const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
    if (info == nullptr) return false;
    const gjxl::Extent2D block_extent = info->covered_blocks;
    const gjxl::Extent2D padded_extent = info->pixel_extent();
    const gjxl::Extent2D source_extent{
      padded_extent.width - 1,
      padded_extent.height - 1,
    };
    ImageStorage original(source_extent);
    ImageStorage padded(padded_extent);
    for (size_t y = 0; y < padded_extent.height; ++y) {
      const size_t source_y = std::min(y, source_extent.height - 1);
      for (size_t x = 0; x < padded_extent.width; ++x) {
        const size_t source_x = std::min(x, source_extent.width - 1);
        const float fx = static_cast<float>(source_x);
        const float fy = static_cast<float>(source_y);
        const std::array<float, 3> rgb = {
          0.05f + 0.015f * std::fmod(fx * 3.0f + fy, 29.0f),
          0.08f + 0.013f * std::fmod(fx + fy * 5.0f, 31.0f),
          ((source_x / 2 + source_y / 3) & 1u) == 0 ? 0.12f : 0.82f,
        };
        for (size_t channel = 0; channel < 3; ++channel) {
          padded.plane[channel][y * padded.stride + x] = rgb[channel];
          if (x < source_extent.width && y < source_extent.height) {
            original.plane[channel][y * original.stride + x] = rgb[channel];
          }
        }
      }
    }
    ImageStorage opsin(padded_extent);
    if (!gjxl::LinearRgbToOpsin(
          padded.View(), 255.0f, opsin.MutableView()).ok()) {
      return false;
    }
    gjxl::AcStrategyGrid strategies;
    if (!gjxl::AcStrategyGrid::Create(block_extent, &strategies).ok() ||
        !strategies.Set(0, 0, strategy).ok()) {
      return false;
    }
    const size_t block_count = block_extent.width * block_extent.height;
    std::vector<float> initial(block_count, 0.5f);
    std::vector<uint8_t> sharpness(block_count, 4);
    gjxl::AdaptiveQuantizationOptions options;
    options.control_mode =
      gjxl::AdaptiveQuantizationControlMode::kMaximumError;
    options.maximum_error = {0.2f, 0.2f, 0.2f};

    AqRun first(block_extent, source_extent);
    AqRun second(block_extent, source_extent);
    const auto run = [&](AqRun* output) {
      return gjxl::FindBestQuantization(
        original.View(),
        opsin.View(),
        strategies,
        {initial.data(), block_extent, block_extent.width},
        {sharpness.data(), block_extent, block_extent.width},
        options,
        output->Output());
    };
    const gjxl::Status first_status = run(&first);
    const gjxl::Status second_status = run(&second);
    if (!first_status.ok() || !second_status.ok() ||
        first.result.outcome != gjxl::MaximumErrorOutcome::kMet ||
        first.result.evaluation_count != 6 ||
        first.result.normalized_maximum > 1.0f ||
        first.quant != second.quant || first.error != second.error ||
        first.history != second.history || first.result != second.result ||
        FrameFingerprint(first.frame) == 0 ||
        FrameFingerprint(first.frame) != FrameFingerprint(second.frame)) {
      std::cerr << "Maximum-error AQ integration failed for " << info->name
                << ": " << first_status.message()
                << " second=" << second_status.message()
                << " outcome=" << static_cast<int>(first.result.outcome)
                << " ratio=" << first.result.normalized_maximum
                << " evaluations=" << first.result.evaluation_count
                << " history=";
      for (double score : first.history) std::cerr << score << ',';
      std::cerr << '\n';
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      if (first.result.achieved[channel] > options.maximum_error[channel]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckReductionAllStrategies() || !CheckPinnedUpdateIntervals() ||
      !CheckIndependentlyLimitingChannels() ||
      !CheckFootprintUpdateAndLimit() || !CheckInvalidInputIsAtomic() ||
      !CheckFixedPolicyAndSelection() || !CheckInfeasibilityOutcomes() ||
      !CheckEndToEndAllStrategies()) {
    return EXIT_FAILURE;
  }
  std::cout << "All maximum-error AQ tests passed.\n";
  return EXIT_SUCCESS;
}
