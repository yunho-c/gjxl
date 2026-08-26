// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's DC-group tokenization.

#pragma once

#include <cstddef>
#include <vector>

#include "codestream/bit_writer.h"
#include "codestream/entropy.h"
#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

inline constexpr size_t kSimpleDcGroupDimension = 2048;
inline constexpr size_t kSimpleDcGroupBlockDimension = 256;
inline constexpr size_t kSimpleDcContextCount = 45;

/// Logical token streams belonging to one row-major JPEG XL DC group.
struct SimpleDcGroupTokenStreams {
  size_t block_x = 0;
  size_t block_y = 0;
  Extent2D block_extent;
  size_t transform_anchor_count = 0;
  std::vector<EntropyToken> dc_tokens;
  std::vector<EntropyToken> ac_metadata_tokens;

  friend bool operator==(const SimpleDcGroupTokenStreams&,
                         const SimpleDcGroupTokenStreams&) = default;
};

/// Group-local metadata views. Predictors reset at each supplied view origin.
struct SimpleAcMetadataInput {
  ConstPlaneI8View y_to_x_map;
  ConstPlaneI8View y_to_b_map;
  const AcStrategyGrid* strategies = nullptr;
  size_t block_x = 0;
  size_t block_y = 0;
  ConstPlaneI32View raw_quant_field;
  ConstPlaneU8View epf_sharpness;
};

/// Emits Y/X/B clamped-gradient DC residuals for one group-local rectangle.
[[nodiscard]] Status TokenizeSimpleDcGroup(ConstImage3I32View quantized_dc,
                                           std::vector<EntropyToken>* tokens);

/// Emits CfL, strategy, quant-field, and EPF metadata in codestream order.
[[nodiscard]] Status TokenizeSimpleAcMetadata(
  const SimpleAcMetadataInput& input, std::vector<EntropyToken>* tokens,
  size_t* transform_anchor_count);

/// Slices and tokenizes every 2048x2048-pixel DC group in row-major order.
[[nodiscard]] Status TokenizeSimpleDcGroups(
  const VarDctEncoderFrame& frame,
  std::vector<SimpleDcGroupTokenStreams>* groups);

/// Writes extra_dc_precision=0, the global tree, default WP, and no transforms.
[[nodiscard]] Status WriteSimpleDcGroupModularHeader(BitWriter* writer);

/// Writes transform-anchor count followed by the simple modular header.
[[nodiscard]] Status WriteSimpleAcMetadataModularHeader(
  Extent2D block_extent, size_t transform_anchor_count, BitWriter* writer);

}  // namespace gjxl
