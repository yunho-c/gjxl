// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "profile_json.h"

#include "gpu/ops/gpu_execution_profile_internal.h"

namespace gjxl::benchmark {

struct GpuProfileJsonOptions {
  std::string_view scope;
  std::string_view execution_path;
  gpu_profile_internal::GpuProfilingMode gpu_profiling_mode =
    gpu_profile_internal::GpuProfilingMode::kDisabled;
  std::string_view gpu_aq;
  std::string_view ac_residual_inverse;
  bool collect_final_butteraugli_score = false;
  float butteraugli_target = 1.2f;
  size_t warmups = 0;
  size_t samples = 0;
  int32_t effort = 7;
  size_t cpu_thread_count = 0;
  std::string_view timestamp_origin;
  std::string_view dc_quantization;
  std::string_view dc_prediction;
  std::optional<bool> adaptive_dc_smoothing;
  std::optional<bool> adaptive_epf_sharpness;
};

struct RawGpuProfileSample {
  size_t sample_index = 0;
  gjxl::gpu_profile_internal::GpuExecutionProfile profile;
};

struct RawGpuProfileWorkload {
  std::string workload;
  gjxl::Extent2D source_extent;
  std::vector<RawGpuProfileSample> samples;
};

[[nodiscard]] inline std::string_view GpuProfilingModeName(
    gjxl::gpu_profile_internal::GpuProfilingMode mode) {
  switch (mode) {
    case gjxl::gpu_profile_internal::GpuProfilingMode::kDisabled:
      return "disabled";
    case gjxl::gpu_profile_internal::GpuProfilingMode::kStage:
      return "stage";
    case gjxl::gpu_profile_internal::GpuProfilingMode::kDispatch:
      return "dispatch";
  }
  return "invalid";
}

[[nodiscard]] inline std::string_view GpuWallStageKindName(
    gjxl::gpu_profile_internal::GpuWallStageKind kind) {
  switch (kind) {
    case gjxl::gpu_profile_internal::GpuWallStageKind::kOperation:
      return "operation";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kPreparation:
      return "preparation";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kUpload:
      return "upload";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kWait:
      return "wait";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kReadback:
      return "readback";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kHost:
      return "host";
  }
  return "invalid";
}

inline void WriteGpuProfileSamples(
    const std::filesystem::path& destination,
    const GpuProfileJsonOptions& options,
    const std::vector<RawGpuProfileWorkload>& workloads) {
  WriteProfileJsonFile(destination, [&](std::ostream& output) {
    output << "{\n"
           << "  \"schema_version\": 4,\n"
           << "  \"scope\": \"" << JsonEscape(options.scope) << "\",\n"
           << "  \"mode\": \""
           << GpuProfilingModeName(options.gpu_profiling_mode) << "\",\n"
           << "  \"gpu_aq\": \"" << JsonEscape(options.gpu_aq)
           << "\",\n"
           << "  \"ac_residual_inverse\": \""
           << JsonEscape(options.ac_residual_inverse) << "\",\n"
           << "  \"collect_final_score\": "
           << (options.collect_final_butteraugli_score ? "true" : "false")
           << ",\n"
           << "  \"distance\": " << std::setprecision(9)
           << options.butteraugli_target << ",\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"sample_count\": " << options.samples << ",\n"
           << "  \"effort\": " << options.effort << ",\n"
           << "  \"cpu_threads\": " << options.cpu_thread_count << ",\n";
    if (options.adaptive_epf_sharpness.has_value()) {
      output << "  \"adaptive_epf_sharpness\": "
             << (*options.adaptive_epf_sharpness ? "true" : "false") << ",\n";
    }
    if (!options.execution_path.empty()) {
      output << "  \"execution_path\": \""
             << JsonEscape(options.execution_path) << "\",\n";
    }
    if (!options.timestamp_origin.empty()) {
      output << "  \"timestamp_origin\": \""
             << JsonEscape(options.timestamp_origin) << "\",\n";
    }
    if (!options.dc_quantization.empty()) {
      output << "  \"dc_quantization\": \"" << JsonEscape(options.dc_quantization)
             << "\",\n  \"dc_prediction\": \"" << JsonEscape(options.dc_prediction)
             << "\",\n  \"adaptive_dc_smoothing\": "
             << (options.adaptive_dc_smoothing.has_value()
                   ? (*options.adaptive_dc_smoothing ? "true" : "false") : "null")
             << ",\n";
    }
    output << "  \"workloads\": [\n";
    for (size_t workload_index = 0; workload_index < workloads.size();
         ++workload_index) {
      const RawGpuProfileWorkload& workload = workloads[workload_index];
      output << "    {\n"
             << "      \"name\": \"" << JsonEscape(workload.workload)
             << "\",\n"
             << "      \"source_width\": " << workload.source_extent.width
             << ",\n"
             << "      \"source_height\": " << workload.source_extent.height
             << ",\n"
             << "      \"samples\": [\n";
      for (size_t sample_index = 0; sample_index < workload.samples.size();
           ++sample_index) {
        const RawGpuProfileSample& sample = workload.samples[sample_index];
        const auto& profile = sample.profile;
        output << "        {\n"
               << "          \"sample_index\": " << sample.sample_index
               << ",\n"
               << "          \"capabilities\": {"
               << "\"timestamp_counter\": "
               << (profile.capabilities.timestamp_counter ? "true" : "false")
               << ", \"stage_boundary\": "
               << (profile.capabilities.stage_boundary ? "true" : "false")
               << ", \"dispatch_boundary\": "
               << (profile.capabilities.dispatch_boundary ? "true" : "false")
               << "},\n"
               << "          \"wall_stages\": [";
        for (size_t wall_index = 0;
             wall_index < profile.wall_stages.size(); ++wall_index) {
          const auto& wall = profile.wall_stages[wall_index];
          if (wall_index != 0) output << ", ";
          output << "{\"stage_id\": \"" << JsonEscape(wall.stage_id)
                 << "\", \"kind\": \""
                 << GpuWallStageKindName(wall.kind)
                 << "\", \"invocation\": " << wall.invocation
                 << ", \"wall_nanoseconds\": " << wall.wall_nanoseconds
                 << '}';
        }
        output << "],\n"
               << "          \"submissions\": [\n";
        for (size_t submission_index = 0;
             submission_index < profile.submissions.size();
             ++submission_index) {
          const auto& submission = profile.submissions[submission_index];
          output << "            {\"submission_index\": "
                 << submission_index
                 << ", \"submission_id\": \""
                 << JsonEscape(submission.submission_id)
                 << "\", \"invocation\": " << submission.invocation
                 << ", \"command_buffer_gpu_nanoseconds\": "
                 << submission.command_buffer_gpu_nanoseconds
                 << ", \"stages\": [\n";
          for (size_t stage_index = 0;
               stage_index < submission.stages.size(); ++stage_index) {
            const auto& stage = submission.stages[stage_index];
            output << "              {\"stage_id\": \""
                   << JsonEscape(stage.stage_id)
                   << "\", \"group_id\": \""
                   << JsonEscape(stage.group_id)
                   << "\", \"iteration\": " << stage.iteration
                   << ", \"invocation\": " << stage.invocation
                   << ", \"begin_timestamp\": " << stage.begin_timestamp
                   << ", \"end_timestamp\": " << stage.end_timestamp
                   << ", \"gpu_nanoseconds\": " << stage.gpu_nanoseconds
                   << ", \"timestamp_valid\": " << (stage.timestamp_valid ? "true" : "false")
                   << ", \"dispatches\": [";
            for (size_t dispatch_index = 0;
                 dispatch_index < stage.dispatches.size(); ++dispatch_index) {
              const auto& dispatch = stage.dispatches[dispatch_index];
              if (dispatch_index != 0) output << ", ";
              output << "{\"kernel_id\": \""
                     << JsonEscape(dispatch.kernel_id)
                     << "\", \"kind\": \""
                     << (dispatch.kind ==
                           gjxl::gpu_profile_internal::GpuDispatchKind::kThreads
                           ? "threads" : dispatch.kind ==
                             gjxl::gpu_profile_internal::GpuDispatchKind::kIndirectThreadgroups
                             ? "indirect_threadgroups" : "threadgroups")
                     << "\", \"invocation\": " << dispatch.invocation
                     << ", \"grid\": [" << dispatch.grid.width << ", "
                     << dispatch.grid.height << ", " << dispatch.grid.depth
                     << "], \"threads_per_threadgroup\": ["
                     << dispatch.threads_per_threadgroup.width << ", "
                     << dispatch.threads_per_threadgroup.height << ", "
                     << dispatch.threads_per_threadgroup.depth
                     << "], \"begin_timestamp\": "
                     << dispatch.begin_timestamp
                     << ", \"end_timestamp\": " << dispatch.end_timestamp
                     << ", \"gpu_nanoseconds\": "
                     << dispatch.gpu_nanoseconds
                     << ", \"timestamp_valid\": " << (dispatch.timestamp_valid ? "true" : "false") << '}';
            }
            output << "]}";
            if (stage_index + 1 != submission.stages.size()) output << ',';
            output << '\n';
          }
          output << "            ]}";
          if (submission_index + 1 != profile.submissions.size()) output << ',';
          output << '\n';
        }
        output << "          ]\n        }";
        if (sample_index + 1 != workload.samples.size()) output << ',';
        output << '\n';
      }
      output << "      ]\n    }";
      if (workload_index + 1 != workloads.size()) output << ',';
      output << '\n';
    }
    output << "  ]\n}\n";
  });
}

}  // namespace gjxl::benchmark
