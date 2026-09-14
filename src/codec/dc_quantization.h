// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/dc_prediction.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

inline constexpr size_t kDcPredictionGroupBlockDimension = 256;

enum class DcQuantizationMode : uint8_t {
  kRound,
  kPredictionAware,
  /// Public workflow policy; must be resolved before calling DC primitives.
  kAutomatic,
};

struct DcQuantizationOptions {
  DcQuantizationMode mode = DcQuantizationMode::kRound;
  VarDctDcPrediction prediction = VarDctDcPrediction::kGradient;
  /// The encoder's prediction-aware experiment uses one extra precision bit.
  /// This low-level primitive also supports ordinary rounding on finer grids.
  uint8_t extra_dc_precision = 0;
};

[[nodiscard]] constexpr bool
IsValidDcQuantization(DcQuantizationOptions options) {
  return (options.mode == DcQuantizationMode::kRound ||
          options.mode == DcQuantizationMode::kPredictionAware) &&
         IsValidDcPrediction(options.prediction) &&
         options.extra_dc_precision <= 3;
}

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
[[nodiscard]] Status QuantizeDcCoefficients(ConstImage3FView dc,
                                            const Quantizer &quantizer,
                                            DcQuantizationOutput output);

/// Prediction-aware quantization resets at 256x256 DC-group boundaries and
/// uses reconstructed Y for default DC chroma prediction. It changes decoded
/// coefficients; it is separate from lossless residual coding. Both outputs
/// are committed only after every sample has been validated and quantized.
[[nodiscard]] Status QuantizeDcCoefficients(ConstImage3FView dc,
                                            const Quantizer &quantizer,
                                            DcQuantizationOutput output,
                                            DcQuantizationOptions options);

} // namespace gjxl
