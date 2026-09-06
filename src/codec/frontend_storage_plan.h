// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/geometry.h"
#include "core/host_storage_bound.h"

namespace gjxl::frontend_storage_internal {

using resource_budget_internal::HostStorageBound;

// Shared with the forward-transform dispatcher, not a separate tuning policy.
inline constexpr size_t kMinimumParallelForwardCoefficients = 256 * 256;
inline constexpr size_t kMaximumForwardWorkers = 8;

/// Three fresh contiguous F32 planes at this exact (not implicitly padded)
/// extent. Image3FBuffer replacement keeps any old image alive until success;
/// the caller must add that old backing separately.
[[nodiscard]] Status ComputeImage3FStorageBound(Extent2D extent,
                                                HostStorageBound *out);

struct OwnedFrameStoragePlan {
  size_t blocks = 0;
  size_t color_tiles = 0;
  size_t ac_groups = 0;
  size_t ac_coefficients = 0; // All channels, including unused group capacity.
  // A fresh CPU frame has thirteen backing allocations. Both CPU coefficient
  // construction (including prepared coefficients) and frame assembly use
  // only stack scratch beyond this output; old output/input owners are
  // separate.
  HostStorageBound output;
  bool operator==(const OwnedFrameStoragePlan &) const = default;
};

[[nodiscard]] Status ComputeOwnedFrameStoragePlan(Extent2D frame_extent,
                                                  OwnedFrameStoragePlan *out);

struct PreparedForwardStoragePlan {
  size_t maximum_transforms = 0;
  size_t maximum_participants = 0;
  HostStorageBound output;
  // COMPLETE operation, including output and the temporary per-tile lists.
  HostStorageBound working;
  bool operator==(const PreparedForwardStoragePlan &) const = default;
};

/// Geometry-only bound for PrepareForwardDctCoefficients on a valid padded
/// image/strategy grid. cpu_thread_count must match the executing EncodeScope;
/// zero covers automatic participation. It does not install or enforce a scope.
[[nodiscard]] Status
ComputePreparedForwardStoragePlan(Extent2D padded_extent,
                                  size_t cpu_thread_count,
                                  PreparedForwardStoragePlan *out);

/// Complete scratch for ReconstructQuantizedCoefficients: the atomic temporary
/// output image, group offsets and one transform's dequantization/DC/IDCT
/// arrays. Borrowed frame and caller-provided output image backing are NOT
/// included.
[[nodiscard]] Status
ComputeCoefficientReconstructionStorageBound(Extent2D frame_extent,
                                             HostStorageBound *out);

// All functions are allocation-free on success, O(1) in image size and atomic
// on invalid inputs/overflow. These are backing-capacity bounds under the
// reviewed HostStorageBound contract, not complete frontend/workflow admission.
// Thread stacks and small runtime control/allocator headers remain excluded.

} // namespace gjxl::frontend_storage_internal
