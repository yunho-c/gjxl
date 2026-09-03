// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "codec/vardct_frame.h"

namespace gjxl::vardct_frame_internal {

struct QuantizedAcTransformLayout {
  size_t block_x = 0;
  size_t block_y = 0;
  AcStrategyType strategy = AcStrategyType::kCount;
  size_t coefficient_count = 0;
  std::array<size_t, 3> coefficient_offsets{};
};

inline constexpr int32_t kUnwrittenQuantizedCoefficient =
  static_cast<int32_t>(0x81234567u);

struct QuantizedFrameAssemblyInput {
  FrameGeometry geometry;
  const AcStrategyGrid* strategies = nullptr;
  ConstPlaneI32View raw_quant_field;
  const Quantizer* quantizer = nullptr;
  ConstPlaneI8View y_to_x;
  ConstPlaneI8View y_to_b;
  ConstPlaneU8View epf_sharpness;
  SimpleVarDctCodestreamProfile profile;
  ConstImage3I32View quantized_dc;
  std::span<const int32_t> quantized_ac;
  std::span<const QuantizedAcTransformLayout> transforms;
  bool reject_unwritten_coefficients = false;
};

[[nodiscard]] Status AssembleVarDctEncoderFrame(
  QuantizedFrameAssemblyInput input,
  VarDctEncoderFrame* out);

}  // namespace gjxl::vardct_frame_internal
