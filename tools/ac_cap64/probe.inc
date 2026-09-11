#define main OriginalBenchmarkMain
#include "BENCHMARK_SOURCE"
#undef main

int main(int argc,char** argv) {
  try {
    if(argc!=5)throw std::runtime_error("input output distance cpu|metal");
    auto input=LoadBenchmarkImage(argv[1]);
    const bool metal=std::string(argv[4])=="metal";
    std::unique_ptr<gjxl::GpuBackend> backend;
    if(metal)RequireStatus("backend",gjxl::CreateEmbeddedMetalBackend(BackendOptions("simd", "fused-tuned"),&backend));
    gjxl::VarDctEncodingOptions options{
      .butteraugli_target=std::stof(argv[3]),.effort=7,
      .backend=metal?gjxl::VarDctBackendPreference::kMetal:gjxl::VarDctBackendPreference::kCpu,
      .cpu_thread_count=8,.metal_aq_mode=gjxl::GpuAdaptiveQuantizationMode::kFullyResident};
    std::vector<uint8_t> bytes,repeated;
    gjxl::VarDctEncodingSummary summary;
    gjxl::codestream_internal::VarDctEncodingProfile profile;
    RequireStatus("encode",gjxl::codestream_internal::EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
      input.ConstView(),options,backend.get(),true,&bytes,&summary,&profile));
    RequireStatus("repeat",gjxl::codestream_internal::EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      input.ConstView(),options,backend.get(),true,&repeated));
    if(bytes!=repeated)throw std::runtime_error("Repeated output changed");
    if(metal && (summary.execution_backend!=gjxl::VarDctExecutionBackend::kMetal ||
        summary.metal_aq_mode!=gjxl::GpuAdaptiveQuantizationMode::kFullyResident))throw std::runtime_error("Unexpected backend");
    std::ofstream output(argv[2],std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
    if(!output)throw std::runtime_error("Output write failed");
    std::cout<<"{\"bytes\":"<<bytes.size()
      <<",\"ac_clusters\":"<<profile.codestream.ac_entropy_clusters
      <<",\"dc_clusters\":"<<profile.codestream.dc_entropy_clusters<<"}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
