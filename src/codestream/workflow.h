// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"
#include "gpu/ops/adaptive_quantization.h"

namespace gjxl {

enum class VarDctBackendPreference {
  /// Uses qualified Metal only within the validated quality interval and above
  /// the measured geometry floor. Availability failures before pipeline
  /// execution fall back to CPU; runtime errors do not.
  kAutomatic,
  /// Always uses the CPU reference pipeline.
  kCpu,
  /// Requires Metal regardless of automatic device, quality, and size gates.
  kMetal,
};

enum class VarDctExecutionBackend {
  kCpu,
  kMetal,
};

enum class VarDctRateControlMode {
  kButteraugliTarget,
  kMaximumError,
  kTargetBytes,
  kTargetBitsPerPixel,
};

enum class TargetSizeSelectionPolicy {
  /// Prefer the largest valid codestream no larger than the budget. If every
  /// valid candidate is over budget, select the smallest one.
  kLargestAtOrBelow,
  /// Select the valid codestream with the smallest absolute byte error.
  /// Equal-distance ties prefer the under-budget candidate.
  kClosestAbsolute,
};

/// Options for the public VarDCT encoding workflow.
struct VarDctEncodingOptions {
  float butteraugli_target = 1.0f;
  VarDctRateControlMode rate_control_mode =
    VarDctRateControlMode::kButteraugliTarget;
  /// Maximum normalized reconstruction error for X, Y, and B respectively.
  std::array<float, 3> maximum_error{};
  /// Serialized-byte budget for `kTargetBytes`.
  size_t target_bytes = 0;
  /// Bits per unpadded source pixel for `kTargetBitsPerPixel`. The normalized
  /// internal byte budget is rounded down.
  double target_bits_per_pixel = 0.0;
  /// Relative target-size tolerance in [0, 1]. The corresponding byte
  /// tolerance is rounded up; zero requires an exact byte count.
  double target_size_tolerance = 0.005;
  /// Complete encode attempts, including the two search endpoints. Valid
  /// values are in [1, 64].
  size_t target_size_maximum_attempts = 12;
  TargetSizeSelectionPolicy target_size_selection =
    TargetSizeSelectionPolicy::kLargestAtOrBelow;
  VarDctBackendPreference backend = VarDctBackendPreference::kAutomatic;
  /// Selects the Metal AQ implementation. Resident modes require an explicitly
  /// forced Metal backend and may change encoder decisions or policy bounds.
  /// Maximum-throughput mode omits perceptual diagnostics, so its reported
  /// score history is empty.
  GpuAdaptiveQuantizationMode metal_aq_mode =
    GpuAdaptiveQuantizationMode::kExactCoefficients;
};

/// Encoder analysis reported without exposing temporary pipeline storage.
struct VarDctEncodingSummary {
  Extent2D extent;
  size_t encoded_bytes = 0;
  VarDctRateControlMode rate_control_mode =
    VarDctRateControlMode::kButteraugliTarget;
  size_t requested_target_bytes = 0;
  size_t effective_target_bytes = 0;
  /// Absolute under-budget tolerance derived from the request.
  size_t target_size_tolerance_bytes = 0;
  double requested_target_bits_per_pixel = 0.0;
  double achieved_bits_per_pixel = 0.0;
  float selected_butteraugli_target = 0.0f;
  std::array<float, 3> requested_maximum_error{};
  std::array<float, 3> achieved_maximum_error{};
  float achieved_maximum_error_ratio = 0.0f;
  size_t maximum_error_evaluation_count = 0;
  MaximumErrorOutcome maximum_error_outcome =
    MaximumErrorOutcome::kNotApplicable;
  size_t encode_attempt_count = 0;
  size_t failed_encode_attempt_count = 0;
  TargetSizeSelectionPolicy target_size_selection =
    TargetSizeSelectionPolicy::kLargestAtOrBelow;
  /// For `kLargestAtOrBelow`, true only when the selected size is at or below
  /// the effective budget and no farther below it than the byte tolerance.
  /// For `kClosestAbsolute`, the tolerance is symmetric.
  bool target_size_met = false;
  /// True when no candidate met the tolerance before the attempt or
  /// representable-target search space was exhausted.
  bool target_size_search_exhausted = false;
  std::array<size_t, kAcStrategyCount> strategy_counts{};
  std::vector<double> score_history;
  VarDctExecutionBackend execution_backend = VarDctExecutionBackend::kCpu;
  /// Reports the requested mode when `execution_backend` is Metal.
  GpuAdaptiveQuantizationMode metal_aq_mode =
    GpuAdaptiveQuantizationMode::kExactCoefficients;

  friend bool operator==(
    const VarDctEncodingSummary&,
    const VarDctEncodingSummary&) = default;
};

/// Wall-clock time for one complete attempted encode, including
/// quantization, reconstruction, metric evaluation, and serialization.
struct VarDctEncodingAttemptTiming {
  float butteraugli_target = 0.0f;
  uint64_t encode_and_serialize_nanoseconds = 0;
  size_t encoded_bytes = 0;
  bool succeeded = false;
};

/// Non-deterministic timing diagnostics kept separate from the result summary.
struct VarDctEncodingTiming {
  /// Post-validation source preparation: geometry, edge extension, color
  /// conversion, host workspaces, and any CPU perceptual reference.
  uint64_t preparation_nanoseconds = 0;
  /// Complete target-size search wall time, including all attempted encodes,
  /// serialization, failures, and final candidate selection. Zero for a
  /// single-target or maximum-error request.
  uint64_t aggregate_search_nanoseconds = 0;
  /// Duration of the successful attempt retained as the final codestream.
  uint64_t selected_attempt_nanoseconds = 0;
  /// End-to-end workflow time, including validation and output commit.
  uint64_t total_nanoseconds = 0;
  std::vector<VarDctEncodingAttemptTiming> attempts;
};

/// Converts linear sRGB, selects the requested CPU/Metal quantization path, and
/// serializes one initial-profile raw JPEG XL codestream.
///
/// Input may be strided. Failure leaves both caller-visible outputs unchanged.
/// `summary` may be null when analysis reporting is not required.
[[nodiscard]] Status EncodeLinearRgbVarDctCodestream(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary = nullptr);

/// Encodes identically to EncodeLinearRgbVarDctCodestream and atomically
/// returns wall-clock diagnostics. Timing values are observational and are
/// intentionally excluded from deterministic summary equality.
[[nodiscard]] Status EncodeLinearRgbVarDctCodestreamProfiled(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingTiming* timing);

}  // namespace gjxl
