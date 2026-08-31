// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "codestream/profile_internal.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

namespace codestream_internal {

struct VarDctCodestreamProfile {
  uint64_t validation_nanoseconds = 0;
  uint64_t dc_tokenization_nanoseconds = 0;
  uint64_t ac_tokenization_nanoseconds = 0;
  uint64_t block_context_map_work_nanoseconds = 0;
  uint64_t coefficient_order_work_nanoseconds = 0;
  uint64_t coefficient_tokenization_work_nanoseconds = 0;
  uint64_t entropy_optimization_nanoseconds = 0;
  EntropyWorkProfile entropy_work;
  uint64_t entropy_model_bits = 0;
  uint64_t entropy_token_bits = 0;
  size_t dc_entropy_clusters = 0;
  size_t ac_entropy_clusters = 0;
  bool dc_entropy_is_ans = false;
  bool ac_entropy_is_ans = false;
  bool coefficient_order_entropy_is_ans = false;
  /// Exact complete-codestream sizes considered by one serializer call.
  size_t natural_candidate_bytes = 0;
  size_t custom_order_candidate_bytes = 0;
  /// Zero when natural order wins; otherwise the selected on-wire family mask.
  uint16_t selected_coefficient_order_mask = 0;
  /// Number of distinct block-map candidates measured by the serializer.
  size_t block_context_candidate_count = 0;
  size_t compact_block_context_candidate_bytes = 0;
  size_t selected_block_context_candidate_index = 0;
  size_t selected_block_context_count = 0;
  size_t selected_block_context_qf_threshold_count = 0;
  uint64_t section_writing_nanoseconds = 0;
  SectionWritingWorkProfile section_writing_work;
  uint64_t assembly_nanoseconds = 0;
  AssemblyProfile assembly;
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
