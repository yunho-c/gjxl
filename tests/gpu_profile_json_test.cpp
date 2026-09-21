// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <iostream>
#include <locale>

#include "../benchmarks/gpu_profile_json.h"

namespace {
struct CommaDecimal : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '_'; }
  std::string do_grouping() const override { return "\3"; }
};
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    using namespace gjxl::benchmark;
    using namespace gjxl::gpu_profile_internal;
    std::locale::global(std::locale(std::locale::classic(), new CommaDecimal));
    GpuProfileJsonOptions options;
    options.scope = "fixture\n\"\\\b\f\r\t\x01";
    options.execution_path = "fixture-path\n";
    options.gpu_profiling_mode = GpuProfilingMode::kDispatch;
    options.gpu_aq = "fully-resident";
    options.ac_residual_inverse = "fused";
    options.butteraugli_target = 1.25f;
    options.samples = 1;
    GpuExecutionProfile profile;
    profile.mode = GpuProfilingMode::kDispatch;
    profile.capabilities = {true, true, true};
    profile.wall_stages.push_back({"host\n", GpuWallStageKind::kHost, 0, 12345});
    GpuStageProfile stage;
    stage.stage_id = "stage\"";
    stage.group_id = "group\\";
    stage.begin_timestamp = 12345;
    stage.end_timestamp = 23456;
    stage.gpu_nanoseconds = 11111;
    stage.timestamp_valid = true;
    stage.dispatches.push_back({"kernel\t", GpuDispatchKind::kThreadgroups,
                               {1, 2, 3}, {32, 4, 1}, 0, 12345, 23456, 11111, true});
    GpuSubmissionProfile submission;
    submission.submission_id = "submission\r";
    submission.command_buffer_gpu_nanoseconds = 11111;
    submission.stages.push_back(std::move(stage));
    GpuStageProfile empty_stage;
    empty_stage.stage_id = "empty-indirect";
    empty_stage.dispatches.push_back({"empty", GpuDispatchKind::kIndirectThreadgroups,
                                     {0, 1, 1}, {32, 1, 1}});
    submission.stages.push_back(std::move(empty_stage));
    profile.submissions.push_back(std::move(submission));
    RawGpuProfileWorkload workload{"workload\n", {1234, 9}, {}};
    workload.samples.push_back({0, std::move(profile)});
    std::vector<RawGpuProfileWorkload> workloads;
    workloads.push_back(std::move(workload));
    // Empty collections must remain valid JSON too.
    workloads.push_back({"empty", {1, 1}, {{0, {}}}});
    workloads.push_back({"no-samples", {1, 1}, {}});
    WriteGpuProfileSamples(std::filesystem::path(argv[1]), options, workloads);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
