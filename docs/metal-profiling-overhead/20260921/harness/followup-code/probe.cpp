#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "codestream/workflow_internal.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/gpu_execution_profile_internal.h"
#include "io/pfm.h"
#include "synthetic_images.h"
#ifdef GJXL_OVERHEAD_ABLATION
#include "gpu/metal/overhead_control.h"
#endif
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::gpu_profile_internal;
using Clock = std::chrono::steady_clock;
void Require(Status s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
int main(int argc, char** argv) try {
  if (argc != 9) throw std::runtime_error("probe INPUT EFFORT PAIRS SAMPLES WARMUPS MODES SEED OUTPUT_JXL");
  const int effort = std::stoi(argv[2]), pairs = std::stoi(argv[3]);
  const int samples = std::stoi(argv[4]), warmups = std::stoi(argv[5]), seed = std::stoi(argv[7]);
  std::vector<std::string> modes;
  std::istringstream mode_stream(argv[6]);
  for (std::string s; std::getline(mode_stream, s, ',');) modes.push_back(s);
  Image3FBuffer image;
  if (std::string_view(argv[1]) == "synthetic128") {
    image = Image3FBuffer({128, 96}); benchmark::FillEncodingStress(image.view());
  } else Require(io::ReadPfm(argv[1], &image));
  const auto dct = MetalDctImplementation::kSimdgroupMatmul;
  const MetalBackendOptions metal{
      .forward_dct8=dct, .inverse_dct8=dct, .forward_dct16x16=dct, .inverse_dct16x16=dct,
      .forward_dct32x32=dct, .inverse_dct32x32=dct, .forward_dct16x8=dct, .inverse_dct16x8=dct,
      .forward_dct8x16=dct, .inverse_dct8x16=dct, .forward_dct32x16=dct, .inverse_dct32x16=dct,
      .forward_dct16x32=dct, .inverse_dct16x32=dct};
  std::unique_ptr<GpuBackend> gpu;
  Require(CreateEmbeddedMetalBackend(metal, &gpu));
  const VarDctEncodingOptions options{
      .butteraugli_target=1.2f, .effort=effort, .backend=VarDctBackendPreference::kMetal,
      .cpu_thread_count=8, .metal_aq_mode=GpuAdaptiveQuantizationMode::kFullyResident};
  std::vector<uint8_t> expected;
  VarDctEncodingSummary expected_summary;
  Require(EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image.const_view(), options, gpu.get(), true, &expected, &expected_summary));
  // Initial untimed burn-in, followed by per-mode warmups in each paired block.
  for (int warmup=0; warmup<8; ++warmup) {
    std::vector<uint8_t> bytes;
    VarDctEncodingSummary summary;
    Require(EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
        image.const_view(), options, gpu.get(), true, &bytes, &summary));
    if (bytes != expected || summary != expected_summary) throw std::runtime_error("Burn-in changed output");
  }
  std::ofstream encoded(argv[8], std::ios::binary);
  encoded.write(reinterpret_cast<const char*>(expected.data()), expected.size());
  if (!encoded) throw std::runtime_error("Could not write reference codestream");
  // Williams balanced schedules: positions and first-order carryover balance
  // over each six rounds. Shuffle the rows separately in each replicate.
  if (modes.size()!=6 || samples!=1 || pairs%6!=0)
    throw std::runtime_error("Follow-up requires six modes, one sample, rounds multiple of six");
  const std::vector<size_t> base{0,1,5,2,4,3};
  std::vector<int> schedule;
  std::mt19937 rng(20260921 + seed);
  for (int block=0; block<pairs/6; ++block) {
    std::vector<int> rows{0,1,2,3,4,5};
    std::shuffle(rows.begin(),rows.end(),rng);
    schedule.insert(schedule.end(),rows.begin(),rows.end());
  }
  for (int pair=-1; pair<pairs; ++pair) {
    auto order = modes;
    if (pair>=0) for (size_t i=0;i<base.size();++i)
      order[i]=modes[(base[i]+schedule[pair])%modes.size()];
    for (const auto& mode : order) {
      for (int rep=(pair<0 ? -warmups : 0); rep<(pair<0 ? 0 : samples); ++rep) {
        std::vector<uint8_t> bytes;
        VarDctEncodingSummary summary;
        VarDctEncodingProfile host;
        GpuExecutionProfile profile;
#ifdef GJXL_OVERHEAD_ABLATION
        metal_internal::SetOverheadControl(mode);
        metal_internal::ResetOverheadStats();
#endif
        const auto submissions_before = gpu->stats().committed_submissions;
        const auto start = Clock::now();
        Status status;
        if (mode == "ordinary") status = EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
            image.const_view(), options, gpu.get(), true, &bytes, &summary);
        else if (mode == "host") status = EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
            image.const_view(), options, gpu.get(), true, &bytes, &summary, &host);
        else status = EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
            image.const_view(), options, gpu.get(), true, GpuProfilingMode::kStage,
            &bytes, &summary, &host, &profile);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
        Require(status);
        if (bytes != expected || summary != expected_summary) throw std::runtime_error("Output or decisions changed");
        uint64_t command_gpu=0, stage_gpu=0, stages=0, dispatches=0, empty_stages=0;
        for (const auto& submission : profile.submissions) {
          command_gpu += submission.command_buffer_gpu_nanoseconds;
          for (const auto& stage : submission.stages) {
            stage_gpu += stage.gpu_nanoseconds; ++stages;
            dispatches += stage.dispatches.size(); empty_stages += !stage.timestamp_valid;
          }
        }
        std::cout << "{\"pair\":" << pair << ",\"rep\":" << rep << ",\"mode\":\"" << mode
          << "\",\"effort\":" << effort << ",\"width\":" << image.extent().width
          << ",\"height\":" << image.extent().height << ",\"wall_ns\":" << elapsed
          << ",\"host_total_ns\":" << host.total_nanoseconds
          << ",\"quantization_ns\":" << host.quantization_pipeline_nanoseconds
          << ",\"serializer_ns\":" << host.codestream_encoding_nanoseconds
          << ",\"profile_command_gpu_ns\":" << command_gpu << ",\"stage_gpu_ns\":" << stage_gpu
          << ",\"stages\":" << stages << ",\"dispatches\":" << dispatches
          << ",\"empty_stages\":" << empty_stages << ",\"encoded_bytes\":" << bytes.size()
          << ",\"submissions\":" << gpu->stats().committed_submissions-submissions_before
          << ",\"byte_equal\":true";
#ifdef GJXL_OVERHEAD_ABLATION
        const auto stats = metal_internal::GetOverheadStats();
        std::cout << ",\"all_command_gpu_ns\":" << stats.gpu_ns
          << ",\"submission_host_ns\":" << stats.encode_ns
          << ",\"resolution_host_ns\":" << stats.resolve_ns
          << ",\"encoders\":" << stats.encoders
          << ",\"counter_resolve_ns\":" << stats.counter_resolve_ns
          << ",\"counter_buffer_ns\":" << stats.counter_buffer_ns;
#endif
        std::cout << "}\n" << std::flush;
      }
    }
  }
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
