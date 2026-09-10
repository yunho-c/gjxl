// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/status.h"
#include "core/managed_publication.h"
#include "gpu/ops/ac_strategy.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/primitives.h"

namespace gjxl::gpu_profile_internal {

template <typename T>
using ProfileStorage = resource_budget_internal::ManagedVector<
  T, resource_budget_internal::ResourceClass::kDiagnostics>;
using ProfileString = resource_budget_internal::ManagedString<
  resource_budget_internal::ResourceClass::kDiagnostics>;
using resource_budget_internal::ReleaseManagedBackingAfterPublication;

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
  ProfileString kernel_id;
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
  ProfileString stage_id;
  ProfileString group_id;
  uint32_t iteration = 0;
  uint32_t invocation = 0;
  uint64_t begin_timestamp = 0;
  uint64_t end_timestamp = 0;
  uint64_t gpu_nanoseconds = 0;
  ProfileStorage<GpuDispatchProfile> dispatches;

  bool operator==(const GpuStageProfile&) const = default;
};

struct GpuSubmissionProfile {
  ProfileString submission_id;
  uint32_t invocation = 0;
  uint64_t command_buffer_gpu_nanoseconds = 0;
  ProfileStorage<GpuStageProfile> stages;

  bool operator==(const GpuSubmissionProfile&) const = default;
};

enum class GpuWallStageKind : uint8_t {
  kOperation,
  kPreparation,
  kUpload,
  kWait,
  kReadback,
  kHost,
};

struct GpuWallStageProfile {
  ProfileString stage_id;
  GpuWallStageKind kind = GpuWallStageKind::kOperation;
  uint32_t invocation = 0;
  uint64_t wall_nanoseconds = 0;

  bool operator==(const GpuWallStageProfile&) const = default;
};

struct GpuExecutionProfile {
  GpuProfilingMode mode = GpuProfilingMode::kDisabled;
  GpuProfilingCapabilities capabilities;
  ProfileStorage<GpuWallStageProfile> wall_stages;
  ProfileStorage<GpuSubmissionProfile> submissions;

  /// Only the complete workflow's outer publication adapter calls this.
  /// Internal child profiles and submission snapshots retain their charges.
  void ReleaseResourceChargesAfterPublication() noexcept {
    for (auto& wall : wall_stages) ReleaseManagedBackingAfterPublication(wall.stage_id);
    for (auto& submission : submissions) {
      ReleaseManagedBackingAfterPublication(submission.submission_id);
      for (auto& stage : submission.stages) {
        ReleaseManagedBackingAfterPublication(stage.stage_id);
        ReleaseManagedBackingAfterPublication(stage.group_id);
        for (auto& dispatch : stage.dispatches)
          ReleaseManagedBackingAfterPublication(dispatch.kernel_id);
        ReleaseManagedBackingAfterPublication(stage.dispatches);
      }
      ReleaseManagedBackingAfterPublication(submission.stages);
    }
    ReleaseManagedBackingAfterPublication(wall_stages);
    ReleaseManagedBackingAfterPublication(submissions);
  }

  bool operator==(const GpuExecutionProfile&) const = default;
};

/// Owns one diagnostic pipeline profile until the complete caller-visible
/// operation succeeds. Ordinary execution never constructs a session.
class GpuProfilingSession {
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  GpuProfilingSession(
    GpuProfilingMode mode,
    GpuProfilingCapabilities capabilities) {
    profile_.mode = mode;
    profile_.capabilities = capabilities;
  }

  [[nodiscard]] GpuProfilingMode mode() const noexcept {
    return profile_.mode;
  }

  [[nodiscard]] static TimePoint BeginWallStage() noexcept {
    return Clock::now();
  }

  [[nodiscard]] Status EndWallStage(
    std::string_view stage_id,
    GpuWallStageKind kind,
    TimePoint begin) {

    if (stage_id.empty()) {
      return Status::InvalidArgument(
        "GPU profile wall-stage ID is empty");
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now() - begin).count();
    uint32_t invocation = 0;
    for (const GpuWallStageProfile& wall_stage : profile_.wall_stages) {
      if (wall_stage.stage_id == stage_id && wall_stage.kind == kind) {
        ++invocation;
      }
    }
    try {
      profile_.wall_stages.push_back({
        .stage_id = ProfileString(stage_id),
        .kind = kind,
        .invocation = invocation,
        .wall_nanoseconds = elapsed < 0 ? 0u : static_cast<uint64_t>(elapsed),
      });
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate GPU wall-stage profile metadata");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "GPU wall-stage profile metadata is too large");
    }
    return Status::Ok();
  }

  [[nodiscard]] Status Append(GpuExecutionProfile child) {
    if (child.mode != profile_.mode ||
        child.mode == GpuProfilingMode::kDisabled) {
      return Status::Internal("GPU child profile mode is inconsistent");
    }
    if (profile_.capabilities != child.capabilities) {
      return Status::Internal("GPU child profile capabilities changed");
    }
    for (const GpuSubmissionProfile& submission : child.submissions) {
      if (submission.submission_id.empty()) {
        return Status::Internal("GPU child submission ID is empty");
      }
    }
    for (size_t child_index = 0;
         child_index < child.submissions.size(); ++child_index) {
      GpuSubmissionProfile& submission = child.submissions[child_index];
      uint32_t invocation = 0;
      for (const GpuSubmissionProfile& existing : profile_.submissions) {
        if (existing.submission_id == submission.submission_id) {
          ++invocation;
        }
      }
      for (size_t prior_index = 0; prior_index < child_index; ++prior_index) {
        if (child.submissions[prior_index].submission_id ==
            submission.submission_id) {
          ++invocation;
        }
      }
      submission.invocation = invocation;
    }
    if (child.wall_stages.size() > profile_.wall_stages.max_size() - profile_.wall_stages.size() ||
        child.submissions.size() > profile_.submissions.max_size() - profile_.submissions.size()) {
      return Status::InvalidArgument("GPU pipeline profile metadata is too large");
    }
    try {
      profile_.wall_stages.reserve(
        profile_.wall_stages.size() + child.wall_stages.size());
      profile_.submissions.reserve(
        profile_.submissions.size() + child.submissions.size());
      profile_.wall_stages.insert(
        profile_.wall_stages.end(),
        std::make_move_iterator(child.wall_stages.begin()),
        std::make_move_iterator(child.wall_stages.end()));
      profile_.submissions.insert(
        profile_.submissions.end(),
        std::make_move_iterator(child.submissions.begin()),
        std::make_move_iterator(child.submissions.end()));
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate GPU pipeline profile metadata");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "GPU pipeline profile metadata is too large");
    }
    return Status::Ok();
  }

  [[nodiscard]] GpuExecutionProfile Finish() && {
    return std::move(profile_);
  }

private:
  GpuExecutionProfile profile_;
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

  [[nodiscard]] virtual Status ComputeInitialQuantizationProfiled(
    InitialQuantizationOptions options,
    InitialQuantFieldOutput output,
    QuantizerParams* quantizer,
    float quant_dc,
    ColorCorrelationMap* initial_color_correlation,
    GpuProfilingMode mode,
    GpuExecutionProfile* profile) = 0;

  [[nodiscard]] virtual Status AdjustQuantFieldResidentProfiled(
    float butteraugli_target,
    ConstPlaneF32View input,
    PlaneF32View output,
    GpuProfilingMode mode,
    GpuExecutionProfile* profile) = 0;
};

/// Optional diagnostic interface for prepared AQ construction. This captures
/// GPU work performed while building persistent reference state.
class GpuAqEvaluationProfiler {
public:
  virtual ~GpuAqEvaluationProfiler() = default;

  [[nodiscard]] virtual Status PrepareAqEvaluationProfiled(
    const AqEvaluationPreparation& preparation,
    GpuProfilingMode mode,
    std::unique_ptr<PreparedAqEvaluation>* prepared,
    GpuExecutionProfile* profile) = 0;
};

/// Optional diagnostic interface for coherent AC-strategy submissions.
class GpuAcStrategyEvaluationProfiler {
public:
  virtual ~GpuAcStrategyEvaluationProfiler() = default;

  [[nodiscard]] virtual Status EvaluateAcStrategyCandidateBatchesProfiled(
    std::span<const AcStrategyCandidateBatch> batches,
    GpuProfilingMode mode,
    std::unique_ptr<GpuSubmission>* submission) = 0;
};

/// Resolves timestamps retained by one completed profiled submission.
class GpuSubmissionProfiler {
public:
  virtual ~GpuSubmissionProfiler() = default;

  [[nodiscard]] virtual GpuProfilingCapabilities
  QueryGpuProfilingCapabilities() const = 0;

  [[nodiscard]] virtual Status ResolveGpuSubmissionProfile(
    GpuSubmission& submission,
    std::string_view submission_id,
    GpuProfilingMode mode,
    GpuExecutionProfile* profile) = 0;
};

/// Optional diagnostic interface for coherent image-primitive submissions.
class GpuImagePrimitivesProfiler {
public:
  virtual ~GpuImagePrimitivesProfiler() = default;

  [[nodiscard]] virtual Status SubmitImagePrimitiveSequenceProfiled(
    std::span<const ImagePrimitiveCommand> commands,
    std::string_view stage_id,
    GpuProfilingMode mode,
    std::unique_ptr<GpuSubmission>* submission) = 0;
};

}  // namespace gjxl::gpu_profile_internal
