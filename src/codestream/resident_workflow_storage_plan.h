// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codestream/serializer_storage_plan.h"
#include "codestream/workflow.h"
#include "gpu/ops/profile_storage_plan.h"

namespace gjxl::codestream_internal {

struct ResidentWorkflowStorageOptions {
  VarDctEncodingOptions encoding;
  bool collect_timing = false;
  bool collect_profile = false;
  // The existing GPU-profiled entry point also collects the CPU workflow
  // profile, so this implies collect_profile for serializer planning.
  bool collect_gpu_profile = false;
};

struct ResidentWorkflowStoragePlan {
  Extent2D coding_extent;
  size_t blocks = 0;
  size_t maximum_attempts = 0;
  size_t score_count = 0;
  size_t device_bytes = 0;
  // Host preparation/evaluator/AC/policy and independent completed snapshot.
  HostStorageBound frontend;
  // Includes parent session and a serial child recording/resolution peak.
  HostStorageBound diagnostics;
  gpu_profile_internal::ProfileStorageShape profile_shape;
  HostStorageBound search_control;
  HostStorageBound retained_best;
  SerializerStoragePlan serializer;
  // Returned bytes, summary scores and requested diagnostics before outer
  // publication. COMPLETE working already includes output, not an extra charge.
  HostStorageBound output;
  HostStorageBound working;
  bool operator==(const ResidentWorkflowStoragePlan &) const = default;
};

/// Whole managed-backing bound for the existing production Metal resident
/// Butteraugli encode, including forced-Metal target-byte/bpp search. Automatic
/// single-target requests require an already selected Metal backend; this
/// function neither selects a backend nor promises a bound for the CPU path.
/// Prepared workflow state must start empty and remain at this geometry through
/// all attempts. Supports fully-resident/throughput, all current efforts and
/// serializer policies, and the current single-target GPU profiling interface.
///
/// Caller image backing, old/published output, idle backing from other jobs,
/// immutable backend/code, driver allocations, stacks and small runtime control
/// objects are separate. The managed reservation must span the complete job;
/// per-attempt reservation replacement is NOT supported. Existing cache rules
/// discard incompatible/oversized capacity before reuse.
///
/// Conservative sum of reviewed phase peaks, not expected usage or RSS. It
/// does not reserve, initialize a backend, or replace request validation. CPU,
/// exact-coefficient, maximum-error and maximum-throughput plans, caller input
/// adapters, retained batch results and public-domain admission remain
/// separate. Successful planning allocates no backing and is O(1) in image
/// size. Failure preserves output, including unsupported workflow shapes.
[[nodiscard]] Status ComputeResidentWorkflowStoragePlan(
    Extent2D source, const ResidentWorkflowStorageOptions &options,
    ResidentWorkflowStoragePlan *out);

} // namespace gjxl::codestream_internal
