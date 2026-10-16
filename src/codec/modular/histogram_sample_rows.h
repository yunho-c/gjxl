// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/histogram_tree_learning.h"
#include "codec/modular/weighted_features.h"
#include <memory>

namespace gjxl::modular_internal {
// Private experiment. The default preserves raster preparation. Both selected
// routes retain native sampling, quantizer construction and ordered dedup.
enum class HistogramRowBackend { kRasterCpu, kSelectedCpu, kGpu };
struct HistogramRowProfile {
  // Sums of scope wall times; concurrent scopes overlap. gather includes all
  // other intervals below except uniqueness_seconds, which precedes gathering.
  // GPU generation includes allocation/copy/submit/wait.
  double gather_seconds = 0, selection_seconds = 0, generation_seconds = 0,
         replay_seconds = 0, allocation_copy_seconds = 0,
         submit_wait_seconds = 0;
  double table_growth_seconds = 0, materialization_seconds = 0;
  double uniqueness_seconds = 0;
  size_t uniqueness_checks = 0, unique_calls = 0, unique_rows = 0;
  size_t gpu_calls = 0, gpu_rows = 0, output_bytes = 0, dispatch_bytes = 0;
  // GPU replay counters are opt-in; disabled timing runs omit hot-loop counts.
  size_t diagnostic_calls = 0, replay_rows = 0, retained_rows = 0,
         lookup_probes = 0, occupied_probes = 0, duplicate_hits = 0,
         saturated_rows = 0, table_growths = 0, rehashed_rows = 0,
         deferred_calls = 0, contiguous_materializations = 0;
};
inline void MergeHistogramRowProfile(HistogramRowProfile &to,
                                     const HistogramRowProfile &from) {
  for (auto field : {&HistogramRowProfile::gather_seconds,
                     &HistogramRowProfile::selection_seconds,
                     &HistogramRowProfile::generation_seconds,
                     &HistogramRowProfile::replay_seconds,
                     &HistogramRowProfile::allocation_copy_seconds,
                     &HistogramRowProfile::submit_wait_seconds,
                     &HistogramRowProfile::table_growth_seconds,
                     &HistogramRowProfile::materialization_seconds,
                     &HistogramRowProfile::uniqueness_seconds})
    to.*field += from.*field;
  for (auto field :
       {&HistogramRowProfile::gpu_calls, &HistogramRowProfile::gpu_rows,
        &HistogramRowProfile::output_bytes,
        &HistogramRowProfile::dispatch_bytes,
        &HistogramRowProfile::diagnostic_calls,
        &HistogramRowProfile::replay_rows, &HistogramRowProfile::retained_rows,
        &HistogramRowProfile::lookup_probes,
        &HistogramRowProfile::occupied_probes,
        &HistogramRowProfile::duplicate_hits,
        &HistogramRowProfile::saturated_rows,
        &HistogramRowProfile::table_growths,
        &HistogramRowProfile::rehashed_rows,
        &HistogramRowProfile::deferred_calls,
        &HistogramRowProfile::contiguous_materializations,
        &HistogramRowProfile::uniqueness_checks,
        &HistogramRowProfile::unique_calls, &HistogramRowProfile::unique_rows})
    to.*field += from.*field;
}
class HistogramRowSource {
public:
  virtual ~HistogramRowSource() = default;
};
struct HistogramRowSlice {
  WeightedFeatureView features;
  uint32_t channel = 0, group = 0;
  size_t first_row = 0, row_count = 0;
};
struct HistogramRowRequest {
  const HistogramRowSource *source = nullptr;
  std::span<const HistogramRowSlice> slices;
  // Channel-local raster positions, in native selection order. Slice ranges
  // partition this list. No random decisions are made on the device.
  std::span<const uint32_t> positions;
  std::span<const Predictor> predictors;
  std::span<const uint8_t> properties;
  std::span<const std::array<uint16_t, 1023>> static_mapping;
  std::span<const std::array<uint8_t, 1023>> mapping;
};
class PreparedHistogramRows {
public:
  virtual ~PreparedHistogramRows() = default;
  // Compact SoA: S uint32 static columns, D byte bin columns, P pairs of
  // byte residual-token columns (interleaved symbol/extra_bits per column).
  // Exactly N*(4*S+D+2*P) bytes. No weights; the CPU replays native dedup.
  [[nodiscard]] virtual std::span<const std::byte> bytes() const = 0;
};
class HistogramRowGenerator {
public:
  virtual ~HistogramRowGenerator() = default;
  // All new device/dispatch backing; excludes borrowed feature owner and the
  // caller's host positions/slices. No pixel reads, allocations or dispatch.
  [[nodiscard]] virtual Status HistogramRowStorageBound(
      size_t rows, size_t slices, size_t statics, size_t properties,
      size_t predictors,
      resource_budget_internal::HostStorageBound *out) const = 0;
  // Synchronous and failure-atomic. The completed source remains immutable.
  // Request spans are borrowed only until return; the result retains no borrow
  // of positions, slices or mappings. It owns its completed row storage.
  [[nodiscard]] virtual Status
  GenerateHistogramRows(const HistogramRowRequest &,
                        std::unique_ptr<PreparedHistogramRows> *,
                        HistogramRowProfile *) = 0;
};
} // namespace gjxl::modular_internal
