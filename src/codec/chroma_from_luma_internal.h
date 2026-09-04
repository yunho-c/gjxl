// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/chroma_from_luma.h"
#include "codec/prepared_coefficients_internal.h"

namespace gjxl::chroma_from_luma_internal {

/// Returns whether one transform is wholly owned by its anchor's color tile.
/// Coefficient preparation and resident GPU reconstruction both index a
/// transform with a single tile-local CfL map entry.
[[nodiscard]] constexpr bool StrategyFitsColorTile(
  size_t block_x,
  size_t block_y,
  AcStrategyType strategy) noexcept {

  constexpr size_t kColorTileBlockDimension =
    kColorTileDimension / kJxlBlockDimension;
  static_assert(kColorTileDimension % kJxlBlockDimension == 0);
  const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
  return info != nullptr &&
    info->covered_blocks.width <=
      kColorTileBlockDimension - block_x % kColorTileBlockDimension &&
    info->covered_blocks.height <=
      kColorTileBlockDimension - block_y % kColorTileBlockDimension;
}

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
