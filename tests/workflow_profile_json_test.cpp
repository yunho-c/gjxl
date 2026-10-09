// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <iostream>
#include <locale>

#include "../benchmarks/workflow_profile_json.h"

namespace {
struct GroupedNumbers : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '_'; }
  std::string do_grouping() const override { return "\3"; }
};
}

// Fabricated counters only: no encoder invocation, GPU initialization or timing.
int main(int argc, char** argv) try {
  if (argc != 2) return 2;
  using namespace gjxl;
  using namespace gjxl::benchmark;
  std::locale::global(std::locale(std::locale::classic(), new GroupedNumbers));
  if (!ProfilePathsAlias("missing/../sample.json", "sample.json") ||
      ProfilePathsAlias("", "sample.json") ||
      ProfilePathsAlias("first.json", "second.json")) return 3;
#if defined(_WIN32)
  if (!ProfilePathsAlias("sample.json", "SAMPLE.JSON")) return 4;
#endif
  GpuProfileJsonOptions options;
  options.gpu_profiling_mode = gpu_profile_internal::GpuProfilingMode::kDispatch;
  options.gpu_aq = "fully-resident";
  options.butteraugli_target = 1.25f;
  options.effort = 4;
  options.cpu_thread_count = 8;
  options.samples = 1;
  options.warmups = 2;
  options.dc_quantization = "round";
  options.dc_prediction = "weighted";
  options.adaptive_dc_smoothing = false;
  codestream_internal::VarDctEncodingProfile profile;
  profile.total_nanoseconds = 9007199254740993ull;
  profile.input_resident_preparation_nanoseconds = 101;
  profile.codestream_encoding_nanoseconds = 500;
  profile.peak_cpu_participants = 8;
  profile.ac_coefficient_bytes = 222;
  profile.ac_storage_bytes = 333;
  auto& cs = profile.codestream;
  cs.validation_nanoseconds = 17;
  cs.dc_tokenization_nanoseconds = 31;
  cs.ac_tokenization_nanoseconds = 43;
  cs.coefficient_order_work_nanoseconds = 53;
  cs.entropy_optimization_nanoseconds = 71;
  // Worker durations can exceed wall time; exporting must not clamp them.
  cs.entropy_work.ans_histogram_build_nanoseconds = 1001;
  cs.entropy_work.prefix_clustering_nanoseconds = 1003;
  cs.section_writing_work.token_write_nanoseconds = 1007;
  cs.assembly.output_copy_nanoseconds = 13;
  cs.total_nanoseconds = 490;
  cs.coefficient_token_count = 12345;
  cs.entropy_work.ans_uint_config_candidate_count = 29;
  cs.selected_coefficient_order_mask = 5;
  cs.entropy_behavior = VarDctEntropyBehavior::kRateOptimized;
  cs.selected_balanced_fallback = true;
  cs.dc_entropy_is_ans = true;
  RawCudaWorkflowWorkload workload{"fixture\n\"\\\t\x01", {1234, 9}, {}};
  workload.samples.push_back({0, "cuda", "gpu-only", 777, profile});
  workload.samples.push_back({0, "cpu", "cpu-first", 888, {}});
  const std::vector<RawCudaWorkflowWorkload> workloads{
      std::move(workload), {"empty", {1, 1}, {}}};
  WriteCudaWorkflowSamples(argv[1], options, "device\n\"", workloads);
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
