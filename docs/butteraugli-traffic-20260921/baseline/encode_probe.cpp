// SPDX-License-Identifier: Apache-2.0
// Investigation harness: ordinary complete calls and separate GPU stage runs.
#define main GjxlUnusedEncodingBenchmarkMain
#include "/Users/yunhocho/GitHub/gjxl-butteraugli-traffic-20260921/benchmarks/encoding_benchmark.cpp"
#undef main

int main(int argc, char** argv) try {
  using namespace gjxl;
  using namespace gjxl::codestream_internal;
  using namespace gjxl::gpu_profile_internal;
  const auto options=ParseCommandLine(argc,argv);
  if(options.input_path.empty() || options.raw_samples_path.empty() ||
     options.scope!=BenchmarkScope::kMetalPublicWorkflow ||
     options.validation!=ValidationMode::kMetalOnly ||
     options.gpu_aq_mode!=GpuAdaptiveQuantizationMode::kFullyResident ||
     options.collect_final_butteraugli_score || !options.metallib_path.empty())
    throw std::runtime_error("Requires input, raw samples, ordinary resident Metal scope, no final score");
  const bool profiled=options.gpu_profiling_mode==GpuProfilingMode::kStage;
  if(profiled != !options.gpu_profile_path.empty())
    throw std::runtime_error("Stage mode requires stage output; ordinary mode must not have it");
  const auto out=std::filesystem::path(options.raw_samples_path).parent_path();
  std::filesystem::create_directories(out);
  auto image=LoadBenchmarkImage(options.input_path);
  std::unique_ptr<GpuBackend> gpu;
  RequireStatus("Backend",CreateEmbeddedMetalBackend(
    BackendOptions(options.implementation,options.ac_residual_inverse),&gpu));
  const VarDctEncodingOptions encoding_options{
    .butteraugli_target=options.butteraugli_target,.effort=options.effort,
    .backend=VarDctBackendPreference::kMetal,.cpu_thread_count=options.cpu_thread_count,
    .metal_aq_mode=GpuAdaptiveQuantizationMode::kFullyResident};
  std::vector<uint8_t> expected;
  VarDctEncodingSummary expected_summary;
  const auto before=gpu->stats().committed_submissions;
  RequireStatus("Reference",EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
    image.ConstView(),encoding_options,gpu.get(),true,&expected,&expected_summary));
  const auto expected_submissions=gpu->stats().committed_submissions-before;
  RawGpuProfileWorkload workload{.workload="traffic-study",.source_extent=image.extent};
  workload.samples.reserve(options.samples);
  std::vector<uint64_t> times;
  for(int i=-static_cast<int>(options.warmups);i<static_cast<int>(options.samples);++i){
    std::vector<uint8_t> bytes;
    VarDctEncodingSummary summary;
    VarDctEncodingProfile host;
    GpuExecutionProfile profile;
    const auto submissions_before=gpu->stats().committed_submissions;
    const auto start=Clock::now();
    const auto status=profiled
      ? EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
          image.ConstView(),encoding_options,gpu.get(),true,GpuProfilingMode::kStage,
          &bytes,&summary,&host,&profile)
      : EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          image.ConstView(),encoding_options,gpu.get(),true,&bytes,&summary);
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
    RequireStatus("Encode",status);
    if(bytes!=expected || summary!=expected_summary ||
       gpu->stats().committed_submissions-submissions_before!=expected_submissions)
      throw std::runtime_error("Nondeterministic output, summary, or submission count");
    if(i>=0){
      times.push_back(ns);
      if(profiled) workload.samples.push_back({static_cast<size_t>(i),std::move(profile)});
    }
  }
  if(profiled)WriteGpuProfileSamples(options.gpu_profile_path,options,{workload});
  std::ofstream json(options.raw_samples_path);
  json.exceptions(std::ios::badbit|std::ios::failbit);
  json << "{\"schema_version\":1,\"profiled\":" << (profiled?"true":"false")
       << ",\"encoded_bytes\":" << expected.size()
       << ",\"submissions\":" << expected_submissions
       << ",\"samples\":[";
  for(size_t i=0;i<times.size();++i){if(i)json<<',';json<<times[i];}
  json<<"],\"byte_equal\":true,\"summary_equal\":true}\n";
  json.close();
  std::ofstream bytes(out/"reference.jxl",std::ios::binary);
  bytes.exceptions(std::ios::badbit|std::ios::failbit);
  bytes.write(reinterpret_cast<const char*>(expected.data()),expected.size());
  std::cout<<"Validated "<<times.size()<<" complete calls; "<<expected.size()<<" bytes\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
