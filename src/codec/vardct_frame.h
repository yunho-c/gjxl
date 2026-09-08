// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/codestream.h"
#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/image.h"
#include "core/overwrite_array.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

enum class AcCoefficientDecisionMode {
  /// Applies the pinned cross-channel AdjustQuantBlockAC policy.
  kAdjustedSharedQuant,
  /// Retains the supplied raw quant as an independently testable diagnostic
  /// coefficient-coding mode.
  kFixedRawQuant,
};

namespace vardct_frame_internal {
struct CoefficientOrderPopulation;
[[nodiscard]] const CoefficientOrderPopulation* GetCoefficientOrderPopulation(
  const VarDctEncoderFrame&) noexcept;
struct AcStorageInfo {
  size_t coefficient_bytes = 4;
  size_t native_bytes = 0;
};
[[nodiscard]] AcStorageInfo
GetAcStorageInfo(const VarDctEncoderFrame &) noexcept;
template <typename T> struct QuantizedFrameAssemblyInputT;
using QuantizedFrameAssemblyInput = QuantizedFrameAssemblyInputT<int32_t>;
template <typename T>
[[nodiscard]] Status
AssembleVarDctEncoderFrameImpl(QuantizedFrameAssemblyInputT<T>,
                               VarDctEncoderFrame *);
[[nodiscard]] Status AssembleVarDctEncoderFrame(
  QuantizedFrameAssemblyInput,
  VarDctEncoderFrame*);
}  // namespace vardct_frame_internal

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

namespace prepared_coefficients_internal {
struct PreparedForwardDctCoefficients;
[[nodiscard]] Status ComputeQuantizedCoefficientsImpl(
  ConstImage3FView,
  const PreparedForwardDctCoefficients*,
  VarDctFrameInput,
  SimpleVarDctCodestreamProfile,
  VarDctEncoderFrame*,
  AcCoefficientDecisionMode);
}  // namespace prepared_coefficients_internal

/// Read-only view of one fixed-capacity VarDCT AC group.
template <typename T> struct VarDctAcGroupViewT {
  size_t block_x = 0;
  size_t block_y = 0;
  Extent2D block_extent;
  size_t used_coefficient_count = 0;
  std::array<std::span<const T>, 3> coefficients;
};
using VarDctAcGroupView = VarDctAcGroupViewT<int32_t>;
using VarDctNativeAcGroupView =
    std::variant<VarDctAcGroupViewT<int8_t>, VarDctAcGroupViewT<int16_t>,
                 VarDctAcGroupView>;

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

  /// Typed int32 query. Returns InvalidArgument for a narrow group; it never
  /// expands storage. General consumers must use GetNativeAcGroup.
  [[nodiscard]] Status GetAcGroup(
    size_t group_index,
    VarDctAcGroupView* out) const;

  /// Native signed storage; dispatch once per group, not per coefficient.
  /// Frame copies deep-copy the authoritative owner. No dense compatibility
  /// cache is maintained; const access is allocation-free.
  [[nodiscard]] Status GetNativeAcGroup(size_t group_index,
                                        VarDctNativeAcGroupView *out) const;

private:
  friend vardct_frame_internal::AcStorageInfo
  vardct_frame_internal::GetAcStorageInfo(const VarDctEncoderFrame &) noexcept;
  friend const vardct_frame_internal::CoefficientOrderPopulation*
    vardct_frame_internal::GetCoefficientOrderPopulation(
      const VarDctEncoderFrame&) noexcept;
  friend Status ComputeQuantizedCoefficients(
    ConstImage3FView,
    VarDctFrameInput,
    SimpleVarDctCodestreamProfile,
    VarDctEncoderFrame*,
    AcCoefficientDecisionMode);

  friend Status prepared_coefficients_internal::
    ComputeQuantizedCoefficientsImpl(
      ConstImage3FView,
      const prepared_coefficients_internal::PreparedForwardDctCoefficients*,
      VarDctFrameInput,
      SimpleVarDctCodestreamProfile,
      VarDctEncoderFrame*,
      AcCoefficientDecisionMode);

  friend Status ReconstructQuantizedCoefficients(
    const VarDctEncoderFrame&,
    Image3FView);

  friend Status vardct_frame_internal::AssembleVarDctEncoderFrame(
    vardct_frame_internal::QuantizedFrameAssemblyInput,
    VarDctEncoderFrame*);
  template <typename T>
  friend Status vardct_frame_internal::AssembleVarDctEncoderFrameImpl(
      vardct_frame_internal::QuantizedFrameAssemblyInputT<T>,
      VarDctEncoderFrame *);

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
  OverwriteArray<int32_t> ac_coefficients_;
  OverwriteArray<int8_t> ac_coefficients_i8_;
  OverwriteArray<int16_t> ac_coefficients_i16_;
  // Immutable and frame-owned: copies may share counts, never mutable input.
  std::shared_ptr<const vardct_frame_internal::CoefficientOrderPopulation>
    coefficient_order_population_;
};

}  // namespace gjxl
