// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "native_fit.h"

#include <array>
#include <numeric>
#include <stdexcept>
#include <string>

namespace reference_learning {
namespace {
void Require(bool good, const char *message) {
  if (!good)
    throw std::runtime_error(message);
}
struct Random {
  uint32_t state;
  uint32_t Next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }
  int32_t Signed() { return static_cast<int32_t>(Next() % 2097153) - 1048576; }
};
void FitBoth(const Fixture &fixture) {
  auto qualified = fixture;
  auto samples = fixture.samples;
  Require(bool(jxl::ComputeBestTree(
              samples, fixture.threshold, fixture.multipliers, fixture.range,
              fixture.fast_decode_multiplier, &qualified.expected)),
          "Synthetic reference fit failed");
  (void)FitNative(qualified);
  (void)FitNative(qualified, true, true);
}
} // namespace

size_t RunNativeDifferentialTests() {
  using Mode = jxl::ModularOptions::TreeMode;
  constexpr std::array modes{Mode::kDefault, Mode::kWPOnly, Mode::kNoWP,
                             Mode::kGradientOnly};
  constexpr std::array predictors{
      jxl::Predictor::Best, jxl::Predictor::Variable, jxl::Predictor::Gradient,
      jxl::Predictor::Zero};
  constexpr std::array<float, 5> thresholds{0, 2, 25, 89, 1000};
  constexpr std::array<size_t, 4> bin_limits{8, 32, 96, 256};
  constexpr size_t cases = 256;
  for (size_t test = 0; test < cases; ++test) {
    try {
      Random random{static_cast<uint32_t>(test + 1)};
      Fixture fixture;
      fixture.threshold = thresholds[test % thresholds.size()];
      fixture.fast_decode_multiplier = test % 3 ? 1.001f : 1.0f;
      fixture.range = {{{0, 4}, {0, 3}}};
      // Default-mode cases exercise forced boundaries, including exact values
      // at cuts and boundaries with no rows on one side.
      if (test % 4 == 0)
        fixture.multipliers = {{{{{0, 2}, {0, 1}}}, 2},
                               {{{{2, 4}, {0, 1}}}, 4},
                               {{{{0, 4}, {1, 3}}}, 3}};
      auto &s = fixture.samples;
      std::vector<uint32_t> properties(16);
      std::iota(properties.begin(), properties.end(), 0);
      const auto mode = modes[test % modes.size()];
      Require(bool(s.SetPredictor(predictors[(test / 4) % predictors.size()],
                                  mode)),
              "Synthetic predictor setup failed");
      Require(bool(s.SetProperties(properties, mode)),
              "Synthetic property setup failed");
      std::vector<jxl::pixel_type> pixels, differences;
      for (size_t i = 0; i < 1024; ++i) {
        pixels.push_back(random.Signed());
        differences.push_back(random.Signed());
      }
      s.PreQuantizeProperties(fixture.range, fixture.multipliers,
                              {512, 512, 512}, {384, 384, 384, 384}, pixels,
                              differences, bin_limits[(test / 3) % 4]);
      const size_t rows = 32 + random.Next() % 480;
      s.PrepareForSamples(rows * 5);
      jxl::Properties props(16);
      std::array<jxl::pixel_type_w, 14> predictions{};
      for (size_t row = 0; row < rows; ++row) {
        props[0] = test >= 128 && test % 3 == 0 ? 0 : row % 4;
        props[1] = (row / 4) % 3;
        for (size_t p = 2; p < props.size(); ++p) {
          if (test < 128) {
            props[p] = test % 7 == 0 ? static_cast<int>(row % 3) - 1 : random.Signed();
          } else {
            // Constant first/last bins and properties that become constant
            // in only one subtree. A child's mask must not leak to siblings.
            switch (p % 4) {
            case 0: props[p] = -1048576; break;
            case 1: props[p] = 1048576; break;
            case 2: props[p] = (row % 8) * 10000 - 35000; break;
            default: props[p] = row % 8 < 4 ? 0 : random.Signed(); break;
            }
          }
        }
        const jxl::pixel_type_w pixel = random.Signed();
        for (size_t p = 0; p < predictions.size(); ++p) {
          // Equal predictor costs, zero-entropy sides, or diverse signed and
          // high-bit-depth residuals. Structured cases encourage deep splits.
          if (test >= 128)
            predictions[p] = pixel - (test % 2 ? (row % 8) * 27
                                              : (row % 8 + p % 3) * 9);
          else if (test % 7 == 0)
            predictions[p] = pixel - (row % 3 == 0 ? 0 : 27);
          else if (test % 7 == 1)
            predictions[p] = pixel - (p == row % 14 ? 0 : row % (p + 1) * 9);
          else
            predictions[p] = random.Signed();
        }
        const size_t repeats = 1 + random.Next() % 5;
        for (size_t i = 0; i < repeats; ++i)
          s.AddSample(pixel, props, predictions.data());
      }
      s.AllSamplesDone();
      Require(s.NumSamples() > s.NumDistinctSamples(),
              "Synthetic weighting was not exercised");
      FitBoth(fixture);
    } catch (const std::exception &e) {
      throw std::runtime_error("Differential fixture " + std::to_string(test) +
                               ": " + e.what());
    }
  }
  // Deduplication saturates a row's 16-bit weight. Multiple equal rows must
  // retain the full weighted population and fit to the same result.
  Fixture weighted;
  weighted.threshold = 89;
  weighted.fast_decode_multiplier = 1.001f;
  weighted.range = {{{0, 1}, {0, 1}}};
  auto &s = weighted.samples;
  Require(bool(s.SetPredictor(jxl::Predictor::Variable, Mode::kDefault)) &&
              bool(s.SetProperties({9}, Mode::kDefault)),
          "Weighted predictor/property setup failed");
  std::vector<jxl::pixel_type> pixels{-1, 0, 1}, diff = pixels;
  s.PreQuantizeProperties(weighted.range, {}, {1}, {1}, pixels, diff, 32);
  s.PrepareForSamples(131072);
  jxl::Properties props(16);
  std::array<jxl::pixel_type_w, 14> predictions{};
  for (size_t i = 0; i < 131072; ++i)
    s.AddSample(17, props, predictions.data());
  s.AllSamplesDone();
  Require(s.NumSamples() == 131072 && s.NumDistinctSamples() >= 3,
          "Weight saturation test failed to reach the boundary");
  FitBoth(weighted);
  return cases + 1;
}
} // namespace reference_learning
