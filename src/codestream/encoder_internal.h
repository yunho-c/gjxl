// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "codestream/storage.h"

#include "codestream/encoder.h"
#include "codestream/entropy.h"
#include "codestream/profile_internal.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;
namespace vardct_frame_internal {
class VarDctFrameView;
}

namespace codestream_internal {

struct CandidateSelectionKey {
  size_t complete_size = 0;
  bool custom_order = false;
  size_t block_context_candidate_index = 0;
};

/// Selects the ordinary-path coder before model construction, matching the
/// pinned libjxl tiny-stream and singleton-context policy.
[[nodiscard]] Status SelectOrdinaryEntropyCodingMode(
  std::span<const EntropyTokenStreamView> streams,
  const EntropyCodeOptions& options,
  EntropyCodingMode* mode);

/// Prefix fallback intentionally wins an equal complete-codestream size.
[[nodiscard]] constexpr bool PreferAllPrefixCandidate(
  size_t mixed_size, size_t prefix_size) noexcept {
  return prefix_size <= mixed_size;
}

/// Applies the serializer's stable cross-candidate size and tie policy.
[[nodiscard]] constexpr bool PreferEncodingCandidate(
  CandidateSelectionKey candidate,
  CandidateSelectionKey selected) noexcept {
  if (candidate.complete_size != selected.complete_size) {
    return candidate.complete_size < selected.complete_size;
  }
  if (candidate.custom_order != selected.custom_order) {
    return !candidate.custom_order;
  }
  return candidate.block_context_candidate_index <
    selected.block_context_candidate_index;
}

/// Converts exact logical section bit counts into physical byte sizes. A
/// single AC group collapses all sections before padding; multi-group frames
/// pad every section independently. The output remains unchanged on failure.
[[nodiscard]] Status PhysicalSectionSizesFromBitCounts(
  std::span<const uint64_t> common_section_bits,
  std::span<const uint64_t> ac_section_bits,
  size_t ac_group_count,
  codestream_internal::Storage<size_t>* sizes);

/// Compatibility adapter for caller-owned section-size vectors.
template <typename Allocator>
[[nodiscard]] Status PhysicalSectionSizesFromBitCounts(
  std::span<const uint64_t> common_section_bits,
  std::span<const uint64_t> ac_section_bits,
  size_t ac_group_count,
  std::vector<size_t, Allocator>* sizes) {
  return LegacyStorageOutput(sizes, [&](auto* storage) {
    return PhysicalSectionSizesFromBitCounts(
      common_section_bits, ac_section_bits, ac_group_count, storage);
  });
}

struct VarDctCodestreamProfile {
  VarDctEntropyBehavior entropy_behavior =
    VarDctEntropyBehavior::kBalanced;
  VarDctCoefficientOrderBehavior coefficient_order_behavior =
    VarDctCoefficientOrderBehavior::kFull;
  uint64_t validation_nanoseconds = 0;
  uint64_t dc_tokenization_nanoseconds = 0;
  uint64_t ac_tokenization_nanoseconds = 0;
  uint64_t block_context_map_work_nanoseconds = 0;
  uint64_t coefficient_order_work_nanoseconds = 0;
  /// Aggregate worker time spent building map-independent order templates.
  uint64_t coefficient_tokenization_work_nanoseconds = 0;
  /// Aggregate worker time spent resolving templates for block-map candidates.
  uint64_t coefficient_context_materialization_work_nanoseconds = 0;
  size_t coefficient_tokenization_pass_count = 0;
  size_t coefficient_token_count = 0;
  size_t coefficient_context_materialization_count = 0;
  size_t coefficient_materialized_token_count = 0;
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

/// Synchronously serializes a borrowed completed frame, including validation.
/// All parallel workers finish before return; neither the view nor its backing
/// is retained. Output and optional profile remain unchanged on failure.
[[nodiscard]] Status EncodeVarDctCodestreamFromView(
  const vardct_frame_internal::VarDctFrameView& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile = nullptr);

/// Diagnostic-only serializer entry point. On failure, both `output` and
/// `profile` remain unchanged.
[[nodiscard]] Status EncodeVarDctCodestreamProfiled(
  const VarDctEncoderFrame& frame,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile);

[[nodiscard]] Status EncodeVarDctCodestreamProfiled(
  const VarDctEncoderFrame& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile);

}  // namespace codestream_internal
}  // namespace gjxl
