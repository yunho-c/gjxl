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

struct VarDctCodestreamProfile {
  uint64_t validation_nanoseconds = 0;
  uint64_t dc_tokenization_nanoseconds = 0;
  uint64_t ac_tokenization_nanoseconds = 0;
  uint64_t entropy_optimization_nanoseconds = 0;
  uint64_t entropy_model_bits = 0;
  uint64_t entropy_token_bits = 0;
  size_t dc_entropy_clusters = 0;
  size_t ac_entropy_clusters = 0;
  uint64_t section_writing_nanoseconds = 0;
  uint64_t assembly_nanoseconds = 0;
  uint64_t total_nanoseconds = 0;

  bool operator==(const VarDctCodestreamProfile&) const = default;
};

/// Diagnostic-only serializer entry point. On failure, both `output` and
/// `profile` remain unchanged.
[[nodiscard]] Status EncodeVarDctCodestreamProfiled(
  const VarDctEncoderFrame& frame,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile);

}  // namespace codestream_internal
}  // namespace gjxl
