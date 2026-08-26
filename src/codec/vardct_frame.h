// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/codestream.h"
#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

inline constexpr size_t kVarDctAcGroupDimension = 256;
inline constexpr size_t kVarDctAcGroupBlockDimension =
  kVarDctAcGroupDimension / kJxlBlockDimension;
inline constexpr size_t kVarDctAcGroupCoefficientCapacity =
  kVarDctAcGroupDimension * kVarDctAcGroupDimension;

/// Borrowed inputs copied into a completed encoder frame.
struct VarDctFrameInput {
  FrameGeometry geometry;
  const AcStrategyGrid* strategies = nullptr;
  ConstPlaneI32View raw_quant_field;
  const Quantizer* quantizer = nullptr;
  const ColorCorrelationMap* color_correlation = nullptr;
  ConstPlaneU8View epf_sharpness;
};

/// Read-only view of one fixed-capacity VarDCT AC group.
struct VarDctAcGroupView {
  size_t block_x = 0;
  size_t block_y = 0;
  Extent2D block_extent;
  size_t used_coefficient_count = 0;
  std::array<std::span<const int32_t>, 3> coefficients;
};

/// Owns the native GJXL handoff from VarDCT analysis to entropy coding.
///
/// AC coefficients use one fixed 65536-element row per group and channel.
/// Complete transforms are appended in row-major anchor order; unused edge-
/// group tails are zero. Quantized DC is authoritative; `dc()` is the
/// decoder-equivalent dequantized cache used by reconstruction and AQ.
class VarDctEncoderFrame {
public:
  VarDctEncoderFrame() = default;

  [[nodiscard]] bool valid() const;

  [[nodiscard]] const FrameGeometry& geometry() const noexcept {
    return geometry_;
  }

  [[nodiscard]] const AcStrategyGrid& strategies() const noexcept {
    return strategies_;
  }

  [[nodiscard]] ConstPlaneI32View raw_quant_field() const noexcept;

  [[nodiscard]] const Quantizer& quantizer() const noexcept {
    return quantizer_;
  }

  [[nodiscard]] const ColorCorrelationMap& color_correlation() const noexcept {
    return color_correlation_;
  }

  [[nodiscard]] ConstPlaneU8View epf_sharpness() const noexcept;

  [[nodiscard]] const SimpleVarDctCodestreamProfile& profile() const noexcept {
    return profile_;
  }

  /// Modular-stream DC coefficients in X/Y/B plane order.
  [[nodiscard]] ConstImage3I32View quantized_dc() const noexcept;

  /// Decoder-equivalent dequantized DC used by reconstruction and AQ.
  [[nodiscard]] ConstImage3FView dc() const noexcept;

  [[nodiscard]] Extent2D ac_group_extent() const noexcept {
    return ac_group_extent_;
  }

  [[nodiscard]] size_t ac_group_count() const noexcept {
    return group_used_coefficient_count_.size();
  }

  [[nodiscard]] Status GetAcGroup(
    size_t group_index,
    VarDctAcGroupView* out) const;

private:
  friend Status ComputeQuantizedCoefficients(
    ConstImage3FView,
    VarDctFrameInput,
    SimpleVarDctCodestreamProfile,
    VarDctEncoderFrame*);

  friend Status ReconstructQuantizedCoefficients(
    const VarDctEncoderFrame&,
    Image3FView);

  [[nodiscard]] size_t AcGroupChannelOffset(
    size_t group_index,
    size_t channel) const noexcept;

  FrameGeometry geometry_;
  AcStrategyGrid strategies_;
  std::vector<int32_t> raw_quant_field_;
  Quantizer quantizer_;
  ColorCorrelationMap color_correlation_;
  std::vector<uint8_t> epf_sharpness_;
  SimpleVarDctCodestreamProfile profile_;
  std::array<std::vector<int32_t>, 3> quantized_dc_;
  std::array<std::vector<float>, 3> dc_;
  Extent2D ac_group_extent_;
  std::vector<size_t> group_used_coefficient_count_;
  std::vector<int32_t> ac_coefficients_;
};

}  // namespace gjxl
