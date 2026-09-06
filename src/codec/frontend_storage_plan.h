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

// Shared with initial quantization's existing dispatcher.
inline constexpr size_t kMinimumParallelInitialQuantValues = 256 * 256;
inline constexpr size_t kMaximumInitialQuantWorkers = 12;

struct InitialQuantStoragePlan {
  size_t maximum_participants = 0;
  // Complete temporary backing; the three caller-provided destination planes
  // and borrowed Opsin image are separate owners. No backing survives return.
  HostStorageBound working;
  bool operator==(const InitialQuantStoragePlan &) const = default;
};

/// Bound ComputeInitialQuantField on padded pixels. The thread count must
/// match the executing EncodeScope; zero covers automatic participation.
[[nodiscard]] Status
ComputeInitialQuantStoragePlan(Extent2D padded_extent, size_t cpu_thread_count,
                               InitialQuantStoragePlan *out);

/// Complete temporary backing for AdjustQuantField / CreateQuantizerFromField.
/// Extents are in BLOCKS, not pixels. Caller destination storage is separate;
/// adjusting a field in place does not remove its atomic temporary.
[[nodiscard]] Status
ComputeQuantFieldAdjustmentStorageBound(Extent2D block_extent,
                                        HostStorageBound *out);
[[nodiscard]] Status
ComputeQuantizerSelectionStorageBound(Extent2D block_extent,
                                      HostStorageBound *out);

enum class ColorCorrelationStorageMode {
  kCopy,         // CreateColorCorrelationMap: two fresh output planes.
  kInitialPixel, // Fast initial map: temporary planes coexist with their copy.
  kTransform, // Initial, final, or prepared-final DCT path, fast or ordinary.
};

struct ColorCorrelationStoragePlan {
  size_t color_tiles = 0;
  HostStorageBound output;
  // Complete operation INCLUDING output, excluding old output/borrowed inputs.
  HostStorageBound working;
  bool operator==(const ColorCorrelationStoragePlan &) const = default;
};

/// Geometry-only CfL bound. Prepared input must be the unmodified result of
/// PrepareForwardDctCoefficients (its shallow valid() check alone does not
/// prove tile membership/unique indices). Each transform stays within its color
/// tile.
[[nodiscard]] Status
ComputeColorCorrelationStoragePlan(Extent2D padded_extent,
                                   ColorCorrelationStorageMode mode,
                                   ColorCorrelationStoragePlan *out);

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
