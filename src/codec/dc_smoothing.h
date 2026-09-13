// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

/// Decoder-equivalent adaptive smoothing of dequantized DC, before converting
/// DC to transform low frequencies. The base quantizer's DC steps normalize
/// the shared X/Y/B attenuation even when extra DC precision is used. Borders
/// are unchanged. Input may alias output; failure leaves output unchanged.
[[nodiscard]] Status SmoothDcCoefficients(ConstImage3FView dc,
                                          const Quantizer &quantizer,
                                          Image3FView output);

} // namespace gjxl
