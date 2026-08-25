// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/image.h"
#include "core/status.h"

namespace gjxl {

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

}  // namespace gjxl
