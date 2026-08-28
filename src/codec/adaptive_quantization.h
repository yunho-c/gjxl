// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "codec/butteraugli.h"
#include "codec/chroma_from_luma.h"
#include "codec/reconstruction.h"
#include "codec/codestream.h"
#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

enum class AdaptiveQuantizationControlMode {
  kButteraugli,
  kMaximumError,
};

enum class MaximumErrorOutcome {
  kNotApplicable,
  kMet,
  kIterationLimit,
  kQuantizationRangeExhausted,
};

struct MaximumErrorResult {
  std::array<float, 3> achieved{};
  float normalized_maximum = 0.0f;
  size_t evaluation_count = 0;
  MaximumErrorOutcome outcome = MaximumErrorOutcome::kNotApplicable;

  [[nodiscard]] bool limit_met() const noexcept {
    return outcome == MaximumErrorOutcome::kMet;
  }

  friend bool operator==(
    const MaximumErrorResult&,
    const MaximumErrorResult&) = default;
};

struct InitialQuantizationOptions {
  float butteraugli_target = 1.0f;
  float rescale = 1.0f;
};

struct InitialQuantFieldOutput {
  PlaneF32View quant_field;
  PlaneF32View strategy_mask;
  PlaneF32View pixel_mask;
};

/// Computes the initial DC quantization amount for a perceptual target.
[[nodiscard]] Status ComputeInitialQuantDc(
  float butteraugli_target,
  float* quant_dc);

/// Computes libjxl's initial DCT8 quant field and masking maps on the CPU.
///
/// The opsin image must be padded to complete 8x8 blocks. `quant_field` and
/// `strategy_mask` have one value per block; `pixel_mask` matches the input.
[[nodiscard]] Status ComputeInitialQuantField(
  ConstImage3FView opsin,
  InitialQuantizationOptions options,
  InitialQuantFieldOutput output);

/// Makes every selected transform use one quant-field value, matching
/// libjxl's max-to-mean interpolation for transforms covering four or more
/// base blocks. Input and output may alias; failure leaves output unchanged.
[[nodiscard]] Status AdjustQuantField(
  const AcStrategyGrid& strategies,
  float butteraugli_target,
  ConstPlaneF32View input,
  PlaneF32View output);

struct AdaptiveQuantizationOptions {
  AdaptiveQuantizationControlMode control_mode =
    AdaptiveQuantizationControlMode::kButteraugli;
  float butteraugli_target = 1.0f;
  /// Absolute XYB reconstruction-error limits in X/Y/B order.
  std::array<float, 3> maximum_error{};
  size_t iterations = 2;
  bool fast_color_correlation = true;
  SimpleVarDctCodestreamProfile profile;
  ButteraugliOptions butteraugli;
};

struct AdaptiveQuantizationOutput {
  PlaneF32View quant_field;
  /// Transform-footprint Butteraugli distance or normalized maximum error,
  /// according to the active control mode.
  PlaneF32View block_distance_map;
  Image3FView reconstructed_linear_rgb;
  VarDctEncoderFrame* frame = nullptr;
  /// Aggregate Butteraugli score or normalized maximum-error history.
  std::vector<double>* score_history = nullptr;
  /// Required in maximum-error mode. Optional otherwise.
  MaximumErrorResult* maximum_error_result = nullptr;
};

/// Refines an initial quant field with libjxl's deterministic perceptual loop.
///
/// `opsin` is padded to complete blocks, while `original_linear_rgb` retains
/// the unpadded image extent. The score history contains `iterations + 1`
/// encode/reconstruct/measure evaluations. Outputs are committed atomically.
[[nodiscard]] Status FindBestQuantization(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationOutput output);

}  // namespace gjxl
