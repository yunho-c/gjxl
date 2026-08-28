// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <vector>

#include "codec/profile_timing_internal.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

namespace codestream_internal {

struct VarDctCodestreamProfile {
  uint64_t validation_nanoseconds = 0;
  uint64_t dc_tokenization_nanoseconds = 0;
  uint64_t ac_tokenization_nanoseconds = 0;
  uint64_t entropy_optimization_nanoseconds = 0;
  uint64_t section_writing_nanoseconds = 0;
  uint64_t assembly_nanoseconds = 0;
  uint64_t total_nanoseconds = 0;

  bool operator==(const VarDctCodestreamProfile&) const = default;
};

struct VarDctCodestreamStageProfile {
  profile_internal::HostInterval total;
  profile_internal::HostInterval validation;
  profile_internal::HostInterval dc_tokenization;
  profile_internal::HostInterval ac_tokenization;
  profile_internal::HostInterval entropy_optimization;
  profile_internal::HostInterval section_writing;
  profile_internal::HostInterval assembly;
};

/// Diagnostic-only serializer entry point. On failure, both `output` and
/// `profile` remain unchanged.
[[nodiscard]] Status EncodeVarDctCodestreamProfiled(
  const VarDctEncoderFrame& frame,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile,
  VarDctCodestreamStageProfile* stage_profile = nullptr);

}  // namespace codestream_internal
}  // namespace gjxl
