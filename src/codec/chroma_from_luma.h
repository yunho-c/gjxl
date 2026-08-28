// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

class ColorCorrelationMap;

namespace prepared_coefficients_internal {
struct PreparedForwardDctCoefficients;
}  // namespace prepared_coefficients_internal

namespace chroma_from_luma_internal {
[[nodiscard]] Status CreateColorCorrelationMap(
  ConstPlaneI8View,
  ConstPlaneI8View,
  ColorCorrelationMap*);
[[nodiscard]] Status ComputeFinalColorCorrelationMapPrepared(
  const prepared_coefficients_internal::PreparedForwardDctCoefficients&,
  ConstPlaneI32View,
  const Quantizer&,
  bool,
  ColorCorrelationMap*);
[[nodiscard]] Status ComputeInitialColorCorrelationMapWithMode(
  ConstImage3FView,
  bool,
  ColorCorrelationMap*);
}  // namespace chroma_from_luma_internal

inline constexpr size_t kColorTileDimension = 64;
inline constexpr int32_t kDefaultColorFactor = 84;

[[nodiscard]] constexpr Extent2D ColorTileExtent(
  Extent2D pixel_extent) noexcept {

  return pixel_extent.ceil_div(kColorTileDimension);
}

/// Owns the per-color-tile AC chroma-from-luma correction factors.
///
/// The stored signed bytes match the JPEG XL representation. Actual factors
/// are X = x_factor / 84 and B = 1 + b_factor / 84; the Y factor is zero.
class ColorCorrelationMap {
public:
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] Extent2D tile_extent() const noexcept;
  [[nodiscard]] ConstPlaneI8View y_to_x_map() const noexcept;
  [[nodiscard]] ConstPlaneI8View y_to_b_map() const noexcept;

  [[nodiscard]] std::array<float, 3> AcFactors(
    size_t tile_x,
    size_t tile_y) const noexcept;

private:
  friend Status ComputeInitialColorCorrelationMap(
    ConstImage3FView,
    ColorCorrelationMap*);
  friend Status chroma_from_luma_internal::
    ComputeInitialColorCorrelationMapWithMode(
      ConstImage3FView,
      bool,
      ColorCorrelationMap*);
  friend Status ComputeFinalColorCorrelationMap(
    ConstImage3FView,
    const AcStrategyGrid&,
    ConstPlaneI32View,
    const Quantizer&,
    bool,
    ColorCorrelationMap*);
  friend Status chroma_from_luma_internal::CreateColorCorrelationMap(
    ConstPlaneI8View,
    ConstPlaneI8View,
    ColorCorrelationMap*);
  friend Status chroma_from_luma_internal::
    ComputeFinalColorCorrelationMapPrepared(
      const prepared_coefficients_internal::PreparedForwardDctCoefficients&,
      ConstPlaneI32View,
      const Quantizer&,
      bool,
      ColorCorrelationMap*);

  Extent2D tile_extent_;
  std::vector<int8_t> y_to_x_;
  std::vector<int8_t> y_to_b_;
};

/// Computes libjxl's first-pass DCT8-only chroma-from-luma AC map.
///
/// Input dimensions are padded pixel dimensions and must be non-zero
/// multiples of the JPEG XL 8x8 block size. The output is committed only
/// after the entire map has been computed successfully.
[[nodiscard]] Status ComputeInitialColorCorrelationMap(
  ConstImage3FView opsin,
  ColorCorrelationMap* out);

/// Recomputes CfL after strategy selection and raw-quant finalization.
[[nodiscard]] Status ComputeFinalColorCorrelationMap(
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  bool fast,
  ColorCorrelationMap* out);

}  // namespace gjxl
