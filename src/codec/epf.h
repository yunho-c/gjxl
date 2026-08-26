// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstdint>

#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

struct EpfSigmaOptions {
  float quant_multiplier = 0.46f;
  std::array<float, 8> sharpness_lut = {
    0.0f / 7.0f,
    1.0f / 7.0f,
    2.0f / 7.0f,
    3.0f / 7.0f,
    4.0f / 7.0f,
    5.0f / 7.0f,
    6.0f / 7.0f,
    7.0f / 7.0f,
  };
};

struct EpfFilterOptions {
  uint32_t iterations = 2;
  std::array<float, 3> channel_scale = {40.0f, 5.0f, 3.5f};
  float pass0_sigma_scale = 0.9f;
  float pass2_sigma_scale = 6.5f;
  float border_sad_multiplier = 2.0f / 3.0f;
};

/// Initializes the pre-AQ EPF sharpness field to libjxl's neutral value 4.
[[nodiscard]] Status FillDefaultEpfSharpness(PlaneU8View sharpness);

/// Computes libjxl's block-resolution reciprocal EPF sigma values.
///
/// The decoder later mirrors these values into its border padding. This
/// function deliberately returns only the unpadded block region.
[[nodiscard]] Status ComputeEpfInverseSigma(
  const AcStrategyGrid& strategies,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  ConstPlaneU8View sharpness,
  EpfSigmaOptions options,
  PlaneF32View inverse_sigma);

/// Applies libjxl's decoder-side EPF passes to a padded XYB image.
/// Iteration counts 1, 2 and 3 select pass sequences {1}, {1,2}, and
/// {0,1,2}. Zero copies the input. Input and output may alias.
[[nodiscard]] Status ApplyEpf(
  ConstImage3FView input,
  ConstPlaneF32View inverse_sigma,
  EpfFilterOptions options,
  Image3FView output);

}  // namespace gjxl
