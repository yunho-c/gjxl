// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

namespace codestream_internal {

struct LibjxlTailOptions {
  int effort = 7;
  size_t thread_count = 1;
};

struct LibjxlTailStateDigest {
  uint64_t dimensions = 0;
  uint64_t strategies = 0;
  uint64_t quantizer = 0;
  uint64_t raw_quant = 0;
  uint64_t epf = 0;
  uint64_t cfl = 0;
  uint64_t quantized_dc = 0;
  uint64_t dc = 0;
  uint64_t ac_used_counts = 0;
  uint64_t ac_coefficients = 0;

  friend bool operator==(
    const LibjxlTailStateDigest&,
    const LibjxlTailStateDigest&) = default;
};

struct LibjxlTailStateAudit {
  LibjxlTailStateDigest source;
  LibjxlTailStateDigest copied;

  friend bool operator==(
    const LibjxlTailStateAudit&,
    const LibjxlTailStateAudit&) = default;
};

[[nodiscard]] bool LibjxlTailExperimentAvailable() noexcept;

/// Copies a completed frame into libjxl's ordinary encoder structures and
/// returns field-specific pre/post-copy digests. Failure leaves `audit`
/// unchanged.
[[nodiscard]] Status AuditVarDctStateWithLibjxl(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options,
  LibjxlTailStateAudit* audit);

/// Serializes a completed frame through the pinned internal libjxl bridge.
/// Failure leaves `output` unchanged.
[[nodiscard]] Status
EncodeVarDctCodestreamWithLibjxl(const VarDctEncoderFrame &frame,
                                 LibjxlTailOptions options,
                                 std::vector<uint8_t> *output);

} // namespace codestream_internal
} // namespace gjxl
