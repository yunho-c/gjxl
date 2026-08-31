// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>

namespace gjxl::codestream_internal {

/// Aggregate worker time spent inside one or more parallel entropy tasks.
///
/// These counters are intentionally work, not wall-clock phase durations:
/// overlapping DC, coefficient-order, and AC-candidate tasks are summed. The
/// enclosing VarDctCodestreamProfile::entropy_optimization_nanoseconds remains
/// the authoritative elapsed time for the complete parallel phase.
struct EntropyWorkProfile {
  uint64_t prefix_histogram_build_nanoseconds = 0;
  uint64_t prefix_histogram_cost_nanoseconds = 0;
  uint64_t prefix_clustering_nanoseconds = 0;
  uint64_t prefix_code_build_nanoseconds = 0;
  uint64_t prefix_value_collection_nanoseconds = 0;
  uint64_t prefix_config_search_nanoseconds = 0;
  /// Final prefix-model serialization and checked cost assembly.
  uint64_t prefix_exact_measurement_nanoseconds = 0;
  uint64_t ans_prefix_validation_nanoseconds = 0;
  uint64_t ans_value_collection_nanoseconds = 0;
  uint64_t ans_value_aggregation_nanoseconds = 0;
  uint64_t ans_uint_config_nanoseconds = 0;
  uint64_t ans_histogram_build_nanoseconds = 0;
  uint64_t ans_model_build_nanoseconds = 0;
  uint64_t ans_token_cost_nanoseconds = 0;
  uint64_t selection_nanoseconds = 0;

  bool operator==(const EntropyWorkProfile&) const = default;
};

inline void AccumulateEntropyWorkProfile(
  const EntropyWorkProfile& source,
  EntropyWorkProfile* destination) noexcept {

  if (destination == nullptr) return;
  destination->prefix_histogram_build_nanoseconds +=
    source.prefix_histogram_build_nanoseconds;
  destination->prefix_histogram_cost_nanoseconds +=
    source.prefix_histogram_cost_nanoseconds;
  destination->prefix_clustering_nanoseconds +=
    source.prefix_clustering_nanoseconds;
  destination->prefix_code_build_nanoseconds +=
    source.prefix_code_build_nanoseconds;
  destination->prefix_value_collection_nanoseconds +=
    source.prefix_value_collection_nanoseconds;
  destination->prefix_config_search_nanoseconds +=
    source.prefix_config_search_nanoseconds;
  destination->prefix_exact_measurement_nanoseconds +=
    source.prefix_exact_measurement_nanoseconds;
  destination->ans_prefix_validation_nanoseconds +=
    source.ans_prefix_validation_nanoseconds;
  destination->ans_value_collection_nanoseconds +=
    source.ans_value_collection_nanoseconds;
  destination->ans_value_aggregation_nanoseconds +=
    source.ans_value_aggregation_nanoseconds;
  destination->ans_uint_config_nanoseconds +=
    source.ans_uint_config_nanoseconds;
  destination->ans_histogram_build_nanoseconds +=
    source.ans_histogram_build_nanoseconds;
  destination->ans_model_build_nanoseconds +=
    source.ans_model_build_nanoseconds;
  destination->ans_token_cost_nanoseconds += source.ans_token_cost_nanoseconds;
  destination->selection_nanoseconds += source.selection_nanoseconds;
}

/// Aggregate worker time spent materializing every codestream candidate.
struct SectionWritingWorkProfile {
  uint64_t model_and_header_nanoseconds = 0;
  uint64_t token_write_nanoseconds = 0;
  uint64_t candidate_measure_nanoseconds = 0;

  bool operator==(const SectionWritingWorkProfile&) const = default;
};

inline void AccumulateSectionWritingWorkProfile(
  const SectionWritingWorkProfile& source,
  SectionWritingWorkProfile* destination) noexcept {

  if (destination == nullptr) return;
  destination->model_and_header_nanoseconds +=
    source.model_and_header_nanoseconds;
  destination->token_write_nanoseconds += source.token_write_nanoseconds;
  destination->candidate_measure_nanoseconds +=
      source.candidate_measure_nanoseconds;
}

/// Wall-clock substages of the selected candidate's final assembly.
struct AssemblyProfile {
  uint64_t candidate_selection_nanoseconds = 0;
  uint64_t section_size_nanoseconds = 0;
  uint64_t frame_header_nanoseconds = 0;
  uint64_t toc_and_sections_nanoseconds = 0;
  uint64_t output_copy_nanoseconds = 0;

  bool operator==(const AssemblyProfile&) const = default;
};

}  // namespace gjxl::codestream_internal
