// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/histogram_tree_profile.h"
#include <ostream>

namespace gjxl::modular_internal {
inline void WriteHistogramSeconds(std::ostream &out,
                                  const HistogramFitSeconds &seconds) {
  out << '{';
  for (size_t i = 0; i < seconds.size(); ++i) {
    if (i)
      out << ',';
    out << '"' << kHistogramFitStageNames[i] << "\":" << seconds[i];
  }
  out << '}';
}
inline void WriteHistogramWork(std::ostream &out,
                               const HistogramNodeWork &work) {
  out << "{\"nodes\":" << work.nodes << ",\"splits\":" << work.splits
      << ",\"forced_splits\":" << work.forced_splits
      << ",\"rows\":" << work.rows
      << ",\"symbol_row_visits\":" << work.symbol_row_visits
      << ",\"parent_histogram_row_visits\":" << work.parent_histogram_row_visits
      << ",\"property_row_visits\":" << work.property_row_visits
      << ",\"joint_histogram_row_visits\":" << work.joint_histogram_row_visits
      << ",\"fused_property_row_visits\":" << work.fused_property_row_visits
      << ",\"skipped_constant_properties\":" << work.skipped_constant_properties
      << ",\"properties\":" << work.properties
      << ",\"constant_properties\":" << work.constant_properties
      << ",\"active_bins\":" << work.active_bins
      << ",\"bin_span\":" << work.bin_span
      << ",\"candidate_thresholds\":" << work.candidate_thresholds
      << ",\"entropy_evaluations\":" << work.entropy_evaluations
      << ",\"symbol_terms\":" << work.symbol_terms
      << ",\"partition_rows\":" << work.partition_rows
      << ",\"row_swaps\":" << work.row_swaps << '}';
}
template <typename Array>
void WriteHistogramDistribution(std::ostream &out, const Array &values,
                                const char *key) {
  out << '[';
  bool first = true;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!values[i])
      continue;
    if (!first)
      out << ',';
    first = false;
    out << "{\"" << key << "\":" << i << ",\"count\":" << values[i] << '}';
  }
  out << ']';
}
inline void WriteHistogramProfile(std::ostream &out,
                                  const HistogramFittingProfile &p,
                                  bool enabled) {
  out << "{\"schema\":3,\"enabled\":" << (enabled ? "true" : "false");
  if (!enabled) {
    out << '}';
    return;
  }
  out << ",\"timing\":\"sum_of_fit_wall_intervals\",\"calls\":" << p.calls
      << ",\"stage_timing\":\"sum_of_task_wall_intervals\""
      << ",\"peak_property_participants\":" << p.peak_property_participants
      << ",\"parallel_property_nodes\":" << p.parallel_property_nodes
      << ",\"successes\":" << p.successes
      << ",\"fit_seconds\":" << p.fit_seconds << ",\"stages\":";
  WriteHistogramSeconds(out, p.seconds);
  out << ",\"by_rows\":[";
  bool first = true;
  for (size_t i = 0; i < p.by_rows.size(); ++i) {
    const auto &bucket = p.by_rows[i];
    if (!bucket.work.nodes)
      continue;
    if (!first)
      out << ',';
    first = false;
    out << "{\"bucket\":" << i << ",\"maximum_rows\":" << (uint64_t{1} << i)
        << ",\"node_seconds\":" << bucket.node_seconds << ",\"work\":";
    WriteHistogramWork(out, bucket.work);
    out << ",\"stages\":";
    WriteHistogramSeconds(out, bucket.seconds);
    out << '}';
  }
  out << "],\"by_depth\":[";
  first = true;
  for (size_t i = 0; i < p.by_depth.size(); ++i) {
    const auto &bucket = p.by_depth[i];
    if (!bucket.nodes)
      continue;
    if (!first)
      out << ',';
    first = false;
    out << "{\"depth\":" << i << ",\"nodes\":" << bucket.nodes
        << ",\"rows\":" << bucket.rows << ",\"node_seconds\":" << bucket.seconds
        << '}';
  }
  out << "],\"nodes_by_symbol_span\":";
  WriteHistogramDistribution(out, p.nodes_by_symbol_span, "symbols");
  out << ",\"properties_by_active_bins\":";
  WriteHistogramDistribution(out, p.properties_by_active_bins, "bucket");
  out << ",\"properties_by_bin_span\":";
  WriteHistogramDistribution(out, p.properties_by_bin_span, "bucket");
  out << ",\"records_dropped\":" << p.records_dropped << ",\"fits\":[";
  for (size_t i = 0; i < p.records_written; ++i) {
    const auto &record = p.records[i];
    if (i)
      out << ',';
    out << "{\"index\":" << i << ",\"rows\":" << record.rows
        << ",\"population\":" << record.population
        << ",\"predictors\":" << record.predictors
        << ",\"properties\":" << record.properties
        << ",\"nodes\":" << record.nodes
        << ",\"maximum_depth\":" << record.maximum_depth
        << ",\"success\":" << (record.success ? "true" : "false")
        << ",\"seconds\":" << record.seconds << ",\"stages\":";
    WriteHistogramSeconds(out, record.stages);
    out << '}';
  }
  out << "]}";
}
} // namespace gjxl::modular_internal
