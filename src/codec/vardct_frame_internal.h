// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "codec/vardct_frame.h"

namespace gjxl::vardct_frame_internal {

struct QuantizedAcTransformView {
  size_t block_x = 0;
  size_t block_y = 0;
  AcStrategyType strategy = AcStrategyType::kCount;
  std::array<std::span<const int32_t>, 3> coefficients;
};

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
  std::span<const QuantizedAcTransformView> transforms;
};

[[nodiscard]] Status AssembleVarDctEncoderFrame(
  QuantizedFrameAssemblyInput input,
  VarDctEncoderFrame* out);

}  // namespace gjxl::vardct_frame_internal
