// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codestream/entropy.h"
#include "core/host_storage_bound.h"

namespace gjxl::codestream_internal {

using resource_budget_internal::HostStorageBound;

// Shared with aggregation, rather than duplicated policy thresholds.
inline constexpr size_t kEntropyDenseValueCount = size_t{1} << 16;
inline constexpr size_t kEntropyMinimumCountingInput = size_t{1} << 12;
inline constexpr size_t kPrefixMaximumTokenBits = 31 + 15;
inline constexpr size_t kAnsMaximumTokenBits = 31 + 16;
inline constexpr size_t kAnsStreamStateBits = 32;

struct EntropyAggregationStoragePlan {
  HostStorageBound output;
  HostStorageBound scratch;
  bool operator==(const EntropyAggregationStoragePlan &) const = default;
};

/// One initially empty aggregation destination. Input backing (including the
/// owning overload's moved input) and previous output backing are excluded.
[[nodiscard]] Status
ComputeEntropyAggregationStoragePlan(size_t maximum_values,
                                     EntropyAggregationStoragePlan *out);

struct EntropyModelStoragePlan {
  size_t maximum_bits = 0;
  HostStorageBound owned;
  // WriteEntropyCode temporaries, excluding its destination BitWriter.
  HostStorageBound write_scratch;
  bool operator==(const EntropyModelStoragePlan &) const = default;
};

/// Bounds freshly constructed optimizer models, not arbitrary externally built
/// models with larger historical capacities. No token/model values are read.
[[nodiscard]] Status
ComputeEntropyModelStoragePlan(EntropyCodingMode mode, size_t contexts,
                               size_t clusters, EntropyModelStoragePlan *out);

struct EntropyTokenEmissionStoragePlan {
  size_t maximum_bits = 0;
  size_t reverse_chunks = 0;
  HostStorageBound scratch;
  bool operator==(const EntropyTokenEmissionStoragePlan &) const = default;
};

/// WriteTokenStream scratch, excluding input/model/destination backing. Add a
/// bound for the destination's entire history, not just the appended payload.
[[nodiscard]] Status
ComputeEntropyTokenEmissionStoragePlan(EntropyCodingMode mode, size_t tokens,
                                       EntropyTokenEmissionStoragePlan *out);
[[nodiscard]] Status ComputeAnsReverseChunkCount(size_t tokens, size_t *out);

/// BitWriter growth bound for the largest bit extent/reservation ever requested
/// of that owner. Allotments must be included even if fewer bits are emitted.
[[nodiscard]] Status ComputeEntropyWriterStorageBound(size_t maximum_bits,
                                                      HostStorageBound *out);

enum class EntropyStoragePolicy {
  kFastPrefix,
  kPrefix,
  kBalancedAns,
  kHighDensityAns,
  kAnsFromPrefix,
  kDeferredAnsFromPrefix,
};

struct EntropyOptimizationStorageOptions {
  EntropyStoragePolicy policy = EntropyStoragePolicy::kPrefix;
  size_t tokens = 0;
  size_t contexts = 0;
  size_t sections = 0;
  // Zero means contexts. Otherwise bounds the original initial map's histogram
  // count (1..256); its entries/input backing remain the caller's
  // responsibility.
  size_t initial_histograms = 0;
  bool return_cost = true;
  bool retain_prepared_clusters =
      false; // Non-fast Prefix; requires return_cost.
  bool borrow_prepared_clusters = false; // ANS-from-Prefix input only.
};

struct EntropyOptimizationStoragePlan {
  size_t clusters = 0;
  // Output at optimizer return; deferred ANS includes ALL width candidates,
  // including those marked !survives, not just the eventual winning model.
  HostStorageBound output;
  // COMPLETE operation envelope, INCLUDING output. Do not add output again.
  // Deferred ANS also includes later finalization scratch; the caller-owned
  // section/candidate measurement array is excluded.
  HostStorageBound working;
  bool operator==(const EntropyOptimizationStoragePlan &) const = default;
};

/// Checked geometry/count-only bounds for span-of-view optimizer overloads.
/// Borrowed input streams, fixed populations, prefix partitions and prepared
/// values, previous outputs, profile/control objects, and compatibility-adapter
/// view tables are excluded. An ANS input partition must have been generated
/// within the supplied original context/initial-histogram limits. Stack arrays,
/// immutable log tables and allocator headers are outside the managed boundary.
/// Success allocates nothing; all failures preserve *out.
[[nodiscard]] Status ComputeEntropyOptimizationStoragePlan(
    const EntropyOptimizationStorageOptions &options,
    EntropyOptimizationStoragePlan *out);

// Implemented beside the private histogram/search types whose sizeof matters.
[[nodiscard]] Status ComputePrefixOptimizationStoragePlan(
    const EntropyOptimizationStorageOptions &options,
    EntropyOptimizationStoragePlan *out);
[[nodiscard]] Status ComputeAnsOptimizationStoragePlan(
    const EntropyOptimizationStorageOptions &options,
    EntropyOptimizationStoragePlan *out);

} // namespace gjxl::codestream_internal
