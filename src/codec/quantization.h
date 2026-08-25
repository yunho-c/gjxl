// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <span>

#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

/// Derives encoder quantizer state and fills one raw quant per base block.
[[nodiscard]] Status CreateUniformQuantizer(
  float quant_dc,
  float quant_ac,
  PlaneI32View raw_quant_field,
  Quantizer* out);

/// Converts a positive quant field using libjxl's median/MAD scale selection.
[[nodiscard]] Status CreateQuantizerFromField(
  float quant_dc,
  ConstPlaneF32View quant_field,
  PlaneI32View raw_quant_field,
  Quantizer* out);

struct QuantizationMatrixView {
  std::span<const float> dequant;
  std::span<const float> inverse_dequant;

  // Matrices use libjxl's canonical coefficient layout: the smaller
  // transform dimension is the row count.
  Extent2D coefficient_extent;
  Extent2D low_frequency_extent;
};

[[nodiscard]] Status GetDefaultQuantizationMatrix(
  AcStrategyType strategy,
  XybChannel channel,
  QuantizationMatrixView* out);

struct AcQuantizationOptions {
  XybChannel channel = XybChannel::kY;

  // X/B channel quantization may scale the matrix independently. The matching
  // dequantization operation applies its reciprocal.
  float matrix_multiplier = 1.0f;
};

[[nodiscard]] Status QuantizeAcBlock(
  AcStrategyType strategy,
  const Quantizer& quantizer,
  int32_t raw_quant,
  AcQuantizationOptions options,
  std::span<const float> coefficients,
  std::span<int32_t> quantized);

[[nodiscard]] Status DequantizeAcBlock(
  AcStrategyType strategy,
  const Quantizer& quantizer,
  int32_t raw_quant,
  AcQuantizationOptions options,
  std::span<const int32_t> quantized,
  std::span<float> coefficients);

}  // namespace gjxl
