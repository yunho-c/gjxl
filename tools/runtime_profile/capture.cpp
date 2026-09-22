// SPDX-License-Identifier: Apache-2.0
// Diagnostic capture harness. Reuse the benchmark's argument validation,
// profile-name mapping, and GPU JSON serializer without changing encoder code.
#define main GjxlUnusedEncodingBenchmarkMain
#include "../../benchmarks/encoding_benchmark.cpp"
#undef main

int main(int argc, char** argv) try {
  using namespace gjxl;
  using namespace gjxl::codestream_internal;
  using namespace gjxl::gpu_profile_internal;
  const auto options = ParseCommandLine(argc, argv);
  if (options.input_path.empty() || options.gpu_profile_path.empty() ||
      options.scope != BenchmarkScope::kMetalPublicWorkflow ||
      options.validation != ValidationMode::kMetalOnly ||
      options.gpu_aq_mode != GpuAdaptiveQuantizationMode::kFullyResident ||
      options.gpu_profiling_mode != GpuProfilingMode::kStage ||
      options.collect_final_butteraugli_score || !options.raw_samples_path.empty() ||
      !options.metallib_path.empty() || options.samples % 2 != 0) {
    throw std::runtime_error("Capture requires an input, fully-resident Metal stage profiling, "
        "metal-only validation, no final score, and an even number of paired samples");
  }
  const auto out = std::filesystem::path(options.gpu_profile_path).parent_path();
  std::filesystem::create_directories(out);
  auto image = LoadBenchmarkImage(options.input_path);
  std::unique_ptr<GpuBackend> gpu;
  RequireStatus("Backend creation", CreateEmbeddedMetalBackend(
      BackendOptions(options.implementation, options.ac_residual_inverse), &gpu));
  const VarDctEncodingOptions encoding_options{
    .butteraugli_target = options.butteraugli_target, .effort = options.effort,
    .backend = VarDctBackendPreference::kMetal, .cpu_thread_count = options.cpu_thread_count,
    .gpu_aq_mode = GpuAdaptiveQuantizationMode::kFullyResident};
  std::vector<uint8_t> expected;
  VarDctEncodingSummary expected_summary;
  // Prime all variants equally before pinning steady submission structure.
  // The adaptive token arena may grow on its first input without changing bytes.
  RequireStatus("Ordinary priming", EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image.ConstView(), encoding_options, gpu.get(), true, &expected, &expected_summary));
  const auto before = gpu->stats().committed_submissions;
  RequireStatus("Ordinary reference", EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image.ConstView(), encoding_options, gpu.get(), true, &expected, &expected_summary));
  const auto expected_submissions = gpu->stats().committed_submissions - before;
  struct Row {
    int sample; bool profiled; uint64_t wall_ns, submissions;
    WorkflowProfileNanoseconds phases;
  };
  std::vector<Row> rows;
  rows.reserve(2 * (options.warmups + options.samples));
  RawGpuProfileWorkload workload{.workload="paired-input", .source_extent=image.extent};
  workload.samples.reserve(options.samples);
  // Alternate AB/BA, including warmups. All serialization/equality checks and
  // destruction of returned values are outside the complete-call timer.
  for (int sample = -static_cast<int>(options.warmups);
       sample < static_cast<int>(options.samples); ++sample) {
    for (int slot = 0; slot < 2; ++slot) {
      const bool profiled = ((sample + static_cast<int>(options.warmups) +
                              options.effort + slot) % 2) != 0;
      std::vector<uint8_t> bytes;
      VarDctEncodingSummary summary;
      VarDctEncodingProfile host;
      GpuExecutionProfile profile;
      const auto submissions_before = gpu->stats().committed_submissions;
      const auto start = Clock::now();
      const Status status = profiled
          ? EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
              image.ConstView(), encoding_options, gpu.get(), true, GpuProfilingMode::kStage,
              &bytes, &summary, &host, &profile)
          : EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              image.ConstView(), encoding_options, gpu.get(), true, &bytes, &summary);
      const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - start).count();
      RequireStatus("Paired encode", status);
      const auto submissions = gpu->stats().committed_submissions - submissions_before;
      if (bytes != expected || summary != expected_summary || submissions != expected_submissions)
        throw std::runtime_error("Profiling changed output, encoding summary, or submission count");
      rows.push_back({sample, profiled, static_cast<uint64_t>(wall), submissions,
                     WorkflowProfileValues(host)});
      if (profiled && sample >= 0)
        workload.samples.push_back({static_cast<size_t>(sample), std::move(profile)});
    }
  }
  WriteGpuProfileSamples(options.gpu_profile_path, options, {workload});
  std::ofstream timings(out / "paired.json");
  timings.exceptions(std::ios::badbit | std::ios::failbit);
  timings << "{\"schema_version\":1,\"capture_kind\":\"paired-same-call-v1\","
          << "\"source_width\":" << image.extent.width << ",\"source_height\":" << image.extent.height
          << ",\"encoded_bytes\":" << expected.size() << ",\"rows\":[";
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows[i];
    if (i) timings << ',';
    timings << "{\"sample_index\":" << row.sample << ",\"mode\":\""
            << (row.profiled ? "profiled" : "ordinary") << "\",\"complete_call_nanoseconds\":"
            << row.wall_ns << ",\"committed_submissions\":" << row.submissions
            << ",\"byte_equal\":true,\"summary_equal\":true,\"phase_nanoseconds\":{";
    for (size_t j = 0; j < row.phases.size(); ++j) {
      if (j) timings << ',';
      timings << '"' << kWorkflowProfileNames[j] << "\":" << row.phases[j];
    }
    timings << "}}";
  }
  timings << "]}\n";
  timings.close();
  std::ofstream encoded(out / "reference.jxl", std::ios::binary);
  encoded.exceptions(std::ios::badbit | std::ios::failbit);
  encoded.write(reinterpret_cast<const char*>(expected.data()), expected.size());
  std::cout << "Captured " << options.samples << " pairs; bytes, summaries, and submissions match\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
