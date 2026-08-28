// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "codec/quantization.h"
#include "core/ac_strategy.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl::metal_internal {

struct MetalAqTransformSnapshotForTesting {
  size_t block_x = 0;
  size_t block_y = 0;
  AcStrategyType strategy = AcStrategyType::kCount;
  std::array<std::vector<float>, 3> forward_coefficients;
  std::array<std::vector<int32_t>, 3> quantized_coefficients;
};

struct MetalAqReconstructionSnapshotForTesting {
  Extent2D block_extent;
  Extent2D pixel_extent;
  std::vector<MetalAqTransformSnapshotForTesting> transforms;
  std::vector<int32_t> raw_quant;
  std::vector<float> epf_inverse_sigma;
  std::array<std::vector<float>, 3> dc;
  std::array<std::vector<float>, 3> reconstructed_opsin;
};

struct MetalAqQuantizationProbeForTesting {
  AcStrategyType strategy = AcStrategyType::kDct8;
  XybChannel channel = XybChannel::kY;
  int32_t raw_quant = 1;
  QuantizerParams quantizer;
  float matrix_multiplier = 1.0f;
  std::span<const float> coefficients;
};

struct MetalAqAdjustmentProbeForTesting {
  AcStrategyType strategy = AcStrategyType::kDct8;
  int32_t initial_raw_quant = 1;
  QuantizerParams quantizer;
  std::array<float, 3> matrix_multipliers{1.0f, 1.0f, 1.0f};
  std::array<std::span<const float>, 3> coefficients;
};

struct MetalAqAdjustmentResultForTesting {
  AdjustedAcQuantization decision;
  std::vector<int32_t> quantized_y;
};

/// Runs the Milestone 3 coefficient round trip as one Metal submission.
/// Caller-visible snapshot storage changes only after successful completion.
[[nodiscard]] Status RunMetalAqReconstructionForTesting(
    PreparedAqEvaluation &prepared, AqEvaluationInput input,
    MetalAqReconstructionSnapshotForTesting *snapshot);

/// Exercises the exact shader quantization/dequantization helpers without a
/// transform so threshold, tie, and numeric-failure cases are directly tested.
[[nodiscard]] Status RunMetalAqQuantizationProbeForTesting(
    PreparedAqEvaluation &prepared,
    const MetalAqQuantizationProbeForTesting &probe,
    std::vector<int32_t> *quantized, std::vector<float> *dequantized);

/// Exercises the complete shared-quant adjustment using caller-supplied
/// coefficients, independently of the resident forward transform.
[[nodiscard]] Status RunMetalAqAdjustmentProbeForTesting(
    PreparedAqEvaluation& prepared,
    const MetalAqAdjustmentProbeForTesting& probe,
    MetalAqAdjustmentResultForTesting* result);

} // namespace gjxl::metal_internal
