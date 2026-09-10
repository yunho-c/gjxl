// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's DC-group tokenization.

#pragma once

#include <cstddef>
#include <vector>

#include "codestream/storage.h"

#include "codestream/bit_writer.h"
#include "codestream/entropy.h"
#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;
namespace vardct_frame_internal {
class VarDctFrameView;
}

inline constexpr size_t kSimpleDcGroupDimension = 2048;
inline constexpr size_t kSimpleDcGroupBlockDimension = 256;
inline constexpr size_t kSimpleDcContextCount = 45;

/// Logical token streams belonging to one row-major JPEG XL DC group.
struct SimpleDcGroupTokenStreams {
  size_t block_x = 0;
  size_t block_y = 0;
  Extent2D block_extent;
  size_t transform_anchor_count = 0;
  codestream_internal::Storage<EntropyToken> dc_tokens;
  codestream_internal::Storage<EntropyToken> ac_metadata_tokens;

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
                                           codestream_internal::Storage<EntropyToken>* tokens);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleDcGroup(
  ConstImage3I32View quantized_dc,
  std::vector<EntropyToken, Allocator>* tokens) {
  return codestream_internal::LegacyStorageOutput(
    tokens, [&](auto* storage) { return TokenizeSimpleDcGroup(quantized_dc, storage); });
}

/// Emits CfL, strategy, quant-field, and EPF metadata in codestream order.
[[nodiscard]] Status TokenizeSimpleAcMetadata(
  const SimpleAcMetadataInput& input, codestream_internal::Storage<EntropyToken>* tokens,
  size_t* transform_anchor_count);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleAcMetadata(
  const SimpleAcMetadataInput& input,
  std::vector<EntropyToken, Allocator>* tokens,
  size_t* transform_anchor_count) {
  if (transform_anchor_count == nullptr) {
    return Status::InvalidArgument("Transform anchor count output is null");
  }
  size_t candidate_count = 0;
  Status status = codestream_internal::LegacyStorageOutput(
    tokens, [&](auto* storage) {
      return TokenizeSimpleAcMetadata(input, storage, &candidate_count);
    });
  if (status.ok()) *transform_anchor_count = candidate_count;
  return status;
}

/// Slices and tokenizes every 2048x2048-pixel DC group in row-major order.
[[nodiscard]] Status TokenizeSimpleDcGroups(
  const VarDctEncoderFrame& frame,
  codestream_internal::Storage<SimpleDcGroupTokenStreams>* groups);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleDcGroups(
  const VarDctEncoderFrame& frame,
  std::vector<SimpleDcGroupTokenStreams, Allocator>* groups) {
  return codestream_internal::LegacyStorageOutput(
    groups, [&](auto* storage) { return TokenizeSimpleDcGroups(frame, storage); });
}

namespace codestream_internal {

/// Serializer-only entry point for an already validated frame.
[[nodiscard]] Status TokenizeSimpleDcGroupsForEncoder(
  const vardct_frame_internal::VarDctFrameView& frame,
  codestream_internal::Storage<SimpleDcGroupTokenStreams>* groups);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleDcGroupsForEncoder(
  const vardct_frame_internal::VarDctFrameView& frame,
  std::vector<SimpleDcGroupTokenStreams, Allocator>* groups) {
  return codestream_internal::LegacyStorageOutput(
    groups, [&](auto* storage) { return TokenizeSimpleDcGroupsForEncoder(frame, storage); });
}

}  // namespace codestream_internal

/// Writes extra_dc_precision=0, the global tree, default WP, and no transforms.
[[nodiscard]] Status WriteSimpleDcGroupModularHeader(BitWriter* writer);

/// Writes transform-anchor count followed by the simple modular header.
[[nodiscard]] Status WriteSimpleAcMetadataModularHeader(
  Extent2D block_extent, size_t transform_anchor_count, BitWriter* writer);

}  // namespace gjxl
