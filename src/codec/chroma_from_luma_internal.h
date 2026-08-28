// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/chroma_from_luma.h"
#include "codec/prepared_coefficients_internal.h"

namespace gjxl::chroma_from_luma_internal {

/// Uses a deterministic tilewise pixel-domain regression for the initial map.
[[nodiscard]] Status ComputeInitialColorCorrelationMapFast(
  ConstImage3FView opsin,
  ColorCorrelationMap* out);

/// Copies validated signed-byte CfL maps into their owning codec form.
[[nodiscard]] Status CreateColorCorrelationMap(
  ConstPlaneI8View y_to_x,
  ConstPlaneI8View y_to_b,
  ColorCorrelationMap* out);

/// Recomputes final CfL from cached forward transforms. Quant-dependent
/// scaling and the reference multiplier search remain evaluation-local.
[[nodiscard]] Status ComputeFinalColorCorrelationMapPrepared(
  const prepared_coefficients_internal::PreparedForwardDctCoefficients&
    prepared,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  bool fast,
  ColorCorrelationMap* out);

}  // namespace gjxl::chroma_from_luma_internal
