// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl::gpu_profile_internal {

enum class GpuProfilingMode : uint8_t {
  kDisabled,
  kStage,
  kDispatch,
};

struct GpuProfilingCapabilities {
  bool timestamp_counter = false;
  bool stage_boundary = false;
  bool dispatch_boundary = false;

  bool operator==(const GpuProfilingCapabilities&) const = default;
};

enum class GpuDispatchKind : uint8_t {
  kThreads,
  kThreadgroups,
};

struct GpuExtent3D {
  uint64_t width = 0;
  uint64_t height = 0;
  uint64_t depth = 0;

  bool operator==(const GpuExtent3D&) const = default;
};

struct GpuDispatchProfile {
  std::string kernel_id;
  GpuDispatchKind kind = GpuDispatchKind::kThreads;
  GpuExtent3D grid;
  GpuExtent3D threads_per_threadgroup;
  uint32_t invocation = 0;
  uint64_t begin_timestamp = 0;
  uint64_t end_timestamp = 0;
  uint64_t gpu_nanoseconds = 0;

  bool operator==(const GpuDispatchProfile&) const = default;
};

struct GpuStageProfile {
  std::string stage_id;
  uint32_t iteration = 0;
  uint32_t invocation = 0;
  uint64_t begin_timestamp = 0;
  uint64_t end_timestamp = 0;
  uint64_t gpu_nanoseconds = 0;
  std::vector<GpuDispatchProfile> dispatches;

  bool operator==(const GpuStageProfile&) const = default;
};

struct GpuSubmissionProfile {
  uint64_t command_buffer_gpu_nanoseconds = 0;
  std::vector<GpuStageProfile> stages;

  bool operator==(const GpuSubmissionProfile&) const = default;
};

struct GpuExecutionProfile {
  GpuProfilingMode mode = GpuProfilingMode::kDisabled;
  GpuProfilingCapabilities capabilities;
  std::vector<GpuSubmissionProfile> submissions;

  bool operator==(const GpuExecutionProfile&) const = default;
};

/// Optional diagnostic interface implemented by prepared evaluators that can
/// collect GPU execution timestamps. Implementations must commit `profile`
/// only when both device execution and caller-visible output succeed.
class PreparedAqEvaluationProfiler {
public:
  virtual ~PreparedAqEvaluationProfiler() = default;

  [[nodiscard]] virtual Status EvaluateResidentButteraugliPolicyProfiled(
    AqResidentButteraugliPolicyInput input,
    AqResidentButteraugliPolicyOutput output,
    GpuProfilingMode mode,
    GpuExecutionProfile* profile) = 0;
};

}  // namespace gjxl::gpu_profile_internal
