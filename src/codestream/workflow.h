// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
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

/// Options for the public VarDCT encoding workflow.
struct VarDctEncodingOptions {
  float butteraugli_target = 1.0f;
  VarDctBackendPreference backend = VarDctBackendPreference::kAutomatic;
  /// Selects the Metal AQ implementation. Fully resident mode requires an
  /// explicitly forced Metal backend and may change encoder decisions.
  GpuAdaptiveQuantizationMode metal_aq_mode =
    GpuAdaptiveQuantizationMode::kExactCoefficients;
};

/// Encoder analysis reported without exposing temporary pipeline storage.
struct VarDctEncodingSummary {
  Extent2D extent;
  size_t encoded_bytes = 0;
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

}  // namespace gjxl
