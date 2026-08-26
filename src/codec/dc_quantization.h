// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

struct DcQuantizationOutput {
  Image3I32View quantized;
  Image3FView reconstructed;
};

/// Quantizes VarDCT DC for the simple 4:4:4 XYB codestream profile.
///
/// Input and output use X/Y/B plane order. Quantized DC is the modular-stream
/// representation; reconstructed DC mirrors decoder dequantization with the
/// default X=0 and B=1 DC chroma-from-luma factors. Input may alias the
/// reconstructed output. Failure leaves both outputs unchanged.
[[nodiscard]] Status QuantizeDcCoefficients(
  ConstImage3FView dc,
  const Quantizer& quantizer,
  DcQuantizationOutput output);

}  // namespace gjxl
