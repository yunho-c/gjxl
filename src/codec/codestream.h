// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstdint>

#include "codec/epf.h"
#include "codec/loop_filter.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

enum class QuantizationMatrixMode : uint8_t {
  kDefault,
  kCustom,
};

enum class DcCflMode : uint8_t {
  kDefault,
  kCustom,
};

enum class CoefficientOrderMode : uint8_t {
  kDefault,
  kCustom,
};

enum class ModularTransformMode : uint8_t {
  kNone,
  kCustom,
};

/// Serialization-critical settings retained with a completed VarDCT frame.
///
/// The profile is intentionally wider than the initial writer: a structurally
/// valid frame may retain a known but unsupported mode so that the writer can
/// reject it explicitly instead of silently signaling different settings.
struct SimpleVarDctCodestreamProfile {
  bool source_is_linear_srgb = true;
  bool source_is_floating_point = true;
  float intensity_target = 255.0f;

  uint32_t color_channel_count = 3;
  uint32_t extra_channel_count = 0;
  uint32_t pass_count = 1;
  uint32_t upsampling = 1;

  QuantizationMatrixMode quantization_matrix_mode =
    QuantizationMatrixMode::kDefault;
  uint8_t x_qm_scale = 2;
  uint8_t b_qm_scale = 2;

  uint8_t extra_dc_precision = 0;
  DcCflMode dc_cfl_mode = DcCflMode::kDefault;
  bool adaptive_dc_smoothing = false;

  CoefficientOrderMode coefficient_order_mode =
    CoefficientOrderMode::kDefault;

  std::array<float, 3> gaborish_inverse_multipliers = {
    1.0f, 1.0f, 1.0f};
  EpfSigmaOptions epf_sigma;
  LoopFilterOptions loop_filter;

  ModularTransformMode modular_transform_mode =
    ModularTransformMode::kNone;
  bool lz77 = false;

  /// Checks representable ranges and numerical settings, not initial-writer
  /// support. Use ValidateSimpleCodestreamFrame before serialization.
  [[nodiscard]] bool valid() const noexcept;

  friend bool operator==(
    const SimpleVarDctCodestreamProfile&,
    const SimpleVarDctCodestreamProfile&) = default;
};

/// Returns the coefficient-matrix multiplier encoded by a three-bit scale.
/// Callers must first ensure that `scale` is in the range 0...7.
[[nodiscard]] float QuantizationMatrixMultiplier(uint8_t scale) noexcept;

/// Rejects invalid frames and state outside the documented initial profile.
[[nodiscard]] Status ValidateSimpleCodestreamFrame(
  const VarDctEncoderFrame& frame);

}  // namespace gjxl
