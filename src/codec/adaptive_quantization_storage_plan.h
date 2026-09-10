// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/adaptive_quantization.h"
#include "core/host_storage_bound.h"

namespace gjxl::frontend_storage_internal {

using resource_budget_internal::HostStorageBound;

struct AqPolicyStoragePlan {
  size_t evaluations = 0;
  // Policy result (two block fields and scores); profile backing is separate.
  HostStorageBound output;
  HostStorageBound profile;
  // Complete policy-owned working bound, including output/profile, previous
  // evaluator block-distance output and update temporaries. Current evaluator
  // scratch/output and old caller-owned result/profile are separate owners.
  HostStorageBound working;
  bool operator==(const AqPolicyStoragePlan &) const = default;
};

/// Shared CPU-side policy used by CPU and compatibility GPU evaluators. Covers
/// adjusted and unadjusted input, but not the fully resident policy. Maximum
/// error has a pinned update count independent of iterations; Butteraugli
/// accepts 0..4 updates. No allocations or mutation on failure.
[[nodiscard]] Status ComputeAqPolicyStoragePlan(
    Extent2D blocks, AdaptiveQuantizationControlMode control, size_t iterations,
    bool collect_profile, AqPolicyStoragePlan *out);

struct CpuAqStorageOptions {
  AdaptiveQuantizationControlMode control =
      AdaptiveQuantizationControlMode::kButteraugli;
  size_t iterations = 2;
  size_t cpu_thread_count = 0;
  bool gaborish = true;
  size_t epf_iterations = 2;
  bool prepared_reference = true;
  bool collect_profile = false;
};

struct CpuAqStoragePlan {
  AqPolicyStoragePlan policy;
  // One independently retained evaluation: frame, reconstructed source image
  // and block distances. The previous evaluator retains only frame and RGB;
  // its distances are already counted by policy.working.
  HostStorageBound evaluation_output;
  HostStorageBound evaluation_working;
  HostStorageBound reference;
  // Includes the final owned frame, policy/profile and one native prepared
  // reference plus its preparation/comparison when requested. Caller source,
  // Opsin/grid/sharpness and destination planes are separate owners. Includes
  // overlap with the previous evaluation during atomic replacement.
  HostStorageBound working;
  bool operator==(const CpuAqStoragePlan &) const = default;
};

/// Complete native CPU FindBestQuantization scratch/returned-frame bound at
/// fixed geometry, plus one prepared reference when requested. O(1) and
/// allocation-free on success; atomic invalid/overflow handling. This is a
/// backing-capacity bound, not RSS, policy validation or public admission.
[[nodiscard]] Status ComputeCpuAqStoragePlan(Extent2D source,
                                             const CpuAqStorageOptions &options,
                                             CpuAqStoragePlan *out);

} // namespace gjxl::frontend_storage_internal
