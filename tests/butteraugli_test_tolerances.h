// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cmath>

namespace gjxl::butteraugli_test {

inline constexpr float kMapAbsoluteTolerance = 1.0e-5f;
inline constexpr float kMapRelativeTolerance = 5.0e-6f;
inline constexpr double kScoreTolerance = 1.0e-5;
inline constexpr float kIdentityTolerance = 1.0e-7f;
inline constexpr float kAqUpdateTolerance = 2.0e-6f;
inline constexpr float kPinnedAqTolerance = 2.0e-5f;
inline constexpr float kBlockReductionTolerance = 2.0e-6f;
inline constexpr float kPrimitiveReferenceAbsoluteTolerance = 2.0e-5f;
inline constexpr float kPrimitiveConstantTolerance = 2.0e-6f;

[[nodiscard]] inline float MapTolerance(float expected) {
  return kMapAbsoluteTolerance + kMapRelativeTolerance * std::abs(expected);
}

[[nodiscard]] inline float PrimitiveReferenceTolerance(float expected) {
  return kPrimitiveReferenceAbsoluteTolerance +
         kMapRelativeTolerance * std::abs(expected);
}

// Cross-target Highway drift is independent of strict scalar-golden and
// facade/live comparisons. These fixed caps are shared by every architecture.
struct ScalarDispatchTolerance {
  float full_map_absolute = 3.0e-4f;
  float stage_absolute = 1.5e-3f;
  double score_absolute = 1.0e-4;
};

inline constexpr ScalarDispatchTolerance kScalarDispatchTolerance;
inline constexpr float kNativeBlurDispatchTolerance =
    kScalarDispatchTolerance.stage_absolute;
inline constexpr float kNativeOpsinFrequencyDispatchTolerance =
    kScalarDispatchTolerance.stage_absolute;
inline constexpr float kNativeDifferenceDispatchTolerance =
    kScalarDispatchTolerance.stage_absolute;
// The complete native pipeline is scalar by design. Highway target selection
// changes upstream opsin values enough that expanded 1x1 impulses require the
// same fixed cross-target cap as intermediate stages. Strict scalar parity is
// enforced independently over the complete corpus.
inline constexpr float kNativeMapDispatchTolerance =
    kScalarDispatchTolerance.stage_absolute;
inline constexpr double kNativeScoreDispatchTolerance =
    kScalarDispatchTolerance.stage_absolute;

} // namespace gjxl::butteraugli_test
