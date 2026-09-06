// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/host_storage_bound.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl::metal_internal {

struct AqHostStorageOptions {
  Extent2D source_extent;
  Extent2D coding_extent;
  bool frame_only = false;
  bool resident_initial_quant = false;
  bool resident_ac_strategy_inputs = false;
  bool resident_quantization = false;
  bool defer_final_transform_metadata = false;
  AqEvaluationMetric metric = AqEvaluationMetric::kButteraugli;
  // Bound all operations over this prepared owner's lifetime, not just the
  // next call. Reconfigure is supported only by a complete evaluator.
  bool reconfigure = false;
  bool exact_coefficients = false;
  bool reconstruct_exact_coefficients = false;
  bool reconstructed_rgb_readback = false;
  bool resident_quant_field_readback = false;
  bool initial_pixel_mask_readback = false;
};

struct AqHostStoragePlan {
  resource_budget_internal::HostStorageBound prepared;
  resource_budget_internal::HostStorageBound preparation;
  resource_budget_internal::HostStorageBound retained;
  resource_budget_internal::HostStorageBound operation;
  resource_budget_internal::HostStorageBound working;
  bool operator==(const AqHostStoragePlan &) const = default;
};

/// Allocation-free bound for a fresh production Metal AQ owner at fixed source
/// and coding geometry, through any number of the selected operations. Includes
/// host metadata, readbacks, old/new Reconfigure overlap, and Gaussian-kernel
/// upload scratch of its prepared device Butteraugli borrower. No test-only
/// reconstruction/probe/stage-capture methods may have populated the owner.
///
/// prepared: owners after Prepare; preparation: complete Prepare peak;
/// retained: owners after any selected operation; operation: retained owners
/// plus maximum serial-operation scratch; working: maximum of both phases.
/// retained_bytes is a capacity bound, not necessarily simultaneously live.
///
/// Device arenas/caches, profiling graphs AND callback-input arrays, score
/// histories, completed/owned-frame and initial-CfL output, borrowed inputs and
/// previous output owners are separate. Normal owned-frame export maps Metal
/// coefficients; it does not allocate exact-input staging. Failures leave the
/// plan unchanged.
[[nodiscard]] Status
ComputeAqHostStoragePlan(const AqHostStorageOptions &options,
                         AqHostStoragePlan *out);

struct CompletedFrameHostStoragePlan {
  resource_budget_internal::HostStorageBound output;
  resource_budget_internal::HostStorageBound working;
  bool operator==(const CompletedFrameHostStoragePlan &) const = default;
};

/// Host snapshot plus temporary destination upload array for the independent
/// completed frame. Use block count as anchor_count before final selection.
/// Device AC/destination backing is in ComputeCompletedFrameStoragePlan; no
/// duplicate host AC plane is charged here. Old output/evaluator are separate.
[[nodiscard]] Status
ComputeCompletedFrameHostStoragePlan(Extent2D source, Extent2D coding,
                                     size_t anchor_count,
                                     CompletedFrameHostStoragePlan *out);

} // namespace gjxl::metal_internal
