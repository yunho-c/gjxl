// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's BlockCtxMap and block-entropy heuristic.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/ac_strategy.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

inline constexpr size_t kSimpleNonzeroBucketCount = 37;
inline constexpr size_t kSimpleZeroDensityContextCount = 458;

/// JPEG XL AC block-context map without DC thresholds. Context-map rows are
/// stored in decoder order (Y, X, B), with coefficient-order family and raw
/// quantization segments nested inside each row.
struct SimpleBlockContextMap {
  std::vector<uint32_t> qf_thresholds;
  std::vector<uint8_t> context_map;
  uint8_t num_contexts = 0;

  [[nodiscard]] size_t ac_context_count() const noexcept {
    return static_cast<size_t>(num_contexts) *
      (kSimpleNonzeroBucketCount + kSimpleZeroDensityContextCount);
  }

  friend bool operator==(
    const SimpleBlockContextMap&,
    const SimpleBlockContextMap&) = default;
};

/// Returns the four-context map used by the initial simple profile.
[[nodiscard]] SimpleBlockContextMap DefaultSimpleBlockContextMap();

/// Returns the 15-context default map defined by the JPEG XL codestream.
[[nodiscard]] SimpleBlockContextMap JxlDefaultSimpleBlockContextMap();

/// Returns the two-context decoder-speed map (Y versus combined X/B).
[[nodiscard]] SimpleBlockContextMap TwoChannelSimpleBlockContextMap();

/// Validates thresholds, dimensions, cluster bounds, and canonical labels.
[[nodiscard]] Status ValidateSimpleBlockContextMap(
  const SimpleBlockContextMap& map);

/// Resolves one strategy/channel/raw-quant tuple to its transmitted context.
[[nodiscard]] Status SimpleBlockContext(
  const SimpleBlockContextMap& map,
  AcStrategyType strategy,
  size_t channel,
  int32_t raw_quant,
  uint32_t* context);

/// Builds deterministic block-context candidates. The compact map is always
/// first; additional maps are emitted only for sufficiently large frames.
[[nodiscard]] Status ComputeSimpleBlockContextMapCandidates(
  const VarDctEncoderFrame& frame,
  std::vector<SimpleBlockContextMap>* maps);

/// Derives the one block-context map used by balanced and high-density
/// serialization. Small frames retain the compact map; eligible frames use
/// the occurrence-derived adaptive map, including its quantization split only
/// above the established size threshold.
[[nodiscard]] Status ComputeSimpleBlockContextMap(
  const VarDctEncoderFrame& frame,
  SimpleBlockContextMap* map);

namespace codestream_internal {

/// Serializer-only entry points for an already validated frame.
[[nodiscard]] Status ComputeSimpleBlockContextMapCandidatesForEncoder(
  const VarDctEncoderFrame& frame,
  std::vector<SimpleBlockContextMap>* maps);

[[nodiscard]] Status ComputeSimpleBlockContextMapForEncoder(
  const VarDctEncoderFrame& frame,
  SimpleBlockContextMap* map);

}  // namespace codestream_internal

}  // namespace gjxl
