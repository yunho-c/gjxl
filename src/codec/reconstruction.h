// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/chroma_from_luma.h"
#include "core/ac_strategy.h"
#include "core/coeff_store.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

struct CoefficientCodingOptions {
  float x_matrix_multiplier = 1.0f;
  float b_matrix_multiplier = 1.0f;
};

/// Transforms and quantizes a padded XYB image using the selected strategies.
/// Floating-point DC is preserved separately, matching VarDCT's AC/DC split.
[[nodiscard]] Status ComputeQuantizedCoefficients(
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  const ColorCorrelationMap& color_correlation,
  CoefficientCodingOptions options,
  QuantizedCoefficientFrame* out);

/// Dequantizes one coefficient frame, restores CfL and LLF, and applies the
/// inverse transforms. Output is committed only after every block succeeds.
[[nodiscard]] Status ReconstructQuantizedCoefficients(
  const QuantizedCoefficientFrame& frame,
  const Quantizer& quantizer,
  const ColorCorrelationMap& color_correlation,
  CoefficientCodingOptions options,
  Image3FView output);

}  // namespace gjxl
