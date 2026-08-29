// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's codestream and frame headers.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "codec/codestream.h"
#include "codestream/bit_writer.h"
#include "codestream/entropy.h"
#include "core/geometry.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

struct SimpleBlockContextMap;

/// Writes the raw marker, size, and initial-profile image metadata.
[[nodiscard]] Status WriteSimpleCodestreamHeader(
  Extent2D frame_extent, BitWriter* writer);

/// Writes the regular, final, one-pass VarDCT frame header.
[[nodiscard]] Status WriteSimpleFrameHeader(
  const SimpleVarDctCodestreamProfile& profile, BitWriter* writer);

/// Writes the JPEG XL selector encodings for global_scale and quant_dc.
[[nodiscard]] Status WriteSimpleQuantizer(
  QuantizerParams params, BitWriter* writer);

/// Writes the complete initial-profile DC global section.
[[nodiscard]] Status WriteSimpleDcGlobal(
  QuantizerParams params, size_t dc_group_count,
  const EntropyCode& dc_code, BitWriter* writer);

/// Writes DC global with the supplied validated AC block-context map.
[[nodiscard]] Status WriteSimpleDcGlobal(
  QuantizerParams params,
  size_t dc_group_count,
  const SimpleBlockContextMap& block_context_map,
  const EntropyCode& dc_code,
  BitWriter* writer);

/// Writes the complete initial-profile AC global section.
[[nodiscard]] Status WriteSimpleAcGlobal(
  size_t ac_group_count, const EntropyCode& ac_code, BitWriter* writer);

/// Writes AC global with one optional custom coefficient-order stream.
[[nodiscard]] Status WriteSimpleAcGlobal(
  size_t ac_group_count,
  uint16_t used_order_mask,
  std::span<const EntropyToken> order_tokens,
  const EntropyCode* order_code,
  const EntropyCode& ac_code,
  BitWriter* writer);

}  // namespace gjxl
