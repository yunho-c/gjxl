// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "codestream/cuda_tokenization_policy.h"
#include "codestream/cuda_workflow_storage_plan.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_ac_tokenization.h"
#include "gpu/cuda/cuda_backend.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::cuda_internal;
void Check(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Env(const char* name, const char* value) {
#ifdef _WIN32
  Require(_putenv_s(name, value ? value : "") == 0,
          "Environment update failed");
#else
  Require((value ? setenv(name, value, 1) : unsetenv(name)) == 0,
          "Environment update failed");
#endif
}
struct Encoded {
  std::vector<uint8_t> bytes;
  VarDctEncodingSummary summary;
  uint64_t providers = 0;
};
Encoded Encode(GpuBackend& backend, ConstImage3FView image,
               VarDctEncodingOptions options) {
  Encoded result;
  const auto before = cuda_token_provider_begin_count;
  Check(EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image, options, &backend, true, &result.bytes, &result.summary));
  result.providers = cuda_token_provider_begin_count - before;
  return result;
}
void Compare(GpuBackend& backend, ConstImage3FView image,
             VarDctEncodingOptions options, bool eligible) {
  Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", "0");
  auto cpu = Encode(backend, image, options);
  Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", "1");
  auto gpu = Encode(backend, image, options);
  Require(cpu.providers == 0 && (gpu.providers > 0) == eligible,
          "Workflow provider selection differs");
  Require(cpu.bytes == gpu.bytes && cpu.summary == gpu.summary,
          "Workflow bytes/summary differ");
  Env("GJXL_GPU_TOKENIZATION", "0");
  auto disabled = Encode(backend, image, options);
  Require(disabled.providers == 0 && disabled.bytes == cpu.bytes &&
              disabled.summary == cpu.summary,
          "Stable CPU override changed result or lost precedence");
  Env("GJXL_GPU_TOKENIZATION", nullptr);
}
void Admission(GpuBackend& backend, ConstImage3FView image) {
  CudaWorkflowStorageOptions options;
  auto& e = options.encoding;
  e.backend = VarDctBackendPreference::kCuda;
  e.effort = 1;
  e.cpu_thread_count = 4;
  e.collect_final_butteraugli_score = false;
  CudaWorkflowStoragePlan enabled, disabled;
  Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", "0");
  Check(ComputeCudaWorkflowStoragePlan(image.extent(), options, &disabled));
  Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", "1");
  Check(ComputeCudaWorkflowStoragePlan(image.extent(), options, &enabled));
  Require(enabled.completed.peak_bytes > disabled.completed.peak_bytes &&
              enabled.serializer.working.peak_bytes >
                  disabled.serializer.working.peak_bytes &&
              enabled.working.peak_bytes > disabled.working.peak_bytes,
          "Token admission envelope missing");
  std::shared_ptr<const ExecutionDomain> domain;
  Check(ExecutionDomain::Create(
      {.managed_memory_bytes = enabled.working.peak_bytes,
       .cpu_participant_limit = 4},
      &domain));
  e.execution_domain = domain;
  const auto first = Encode(backend, image, e);
  const auto second = Encode(backend, image, e);
  Require(first.providers == 1 && second.providers == 1 &&
              first.bytes == second.bytes && first.summary == second.summary,
          "Finite repeated workflow differs");
  Require(
      domain->snapshot().peak_committed_bytes <= enabled.working.peak_bytes &&
          domain->snapshot().active_reservations == 0,
      "Workflow exceeded or retained admission");
  Check(TrimVarDctPreparationCache());
  Require(domain->snapshot().live_capacity_bytes == 0 &&
              domain->snapshot().idle_capacity_bytes == 0 &&
              domain->snapshot().reserved_unbacked_bytes == 0,
          "Finite domain retained backing after trim");
  std::shared_ptr<const ExecutionDomain> small;
  Check(ExecutionDomain::Create(
      {.managed_memory_bytes = enabled.working.peak_bytes - 1,
       .cpu_participant_limit = 4},
      &small));
  e.execution_domain = small;
  auto bytes = first.bytes;
  auto summary = first.summary;
  const auto before = cuda_token_provider_begin_count;
  const auto failed = EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image, e, &backend, true, &bytes, &summary);
  Require(
      !failed.ok() && bytes == first.bytes && summary == first.summary &&
          cuda_token_provider_begin_count == before &&
          small->snapshot().active_reservations == 0,
      "Insufficient admission was not rejected atomically before token work");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool smoke = argc == 2 && std::strcmp(argv[1], "--smoke") == 0;
    Require(argc == 1 || smoke, "Unknown test argument");
    Env("GJXL_GPU_TOKENIZATION", nullptr);
    std::unique_ptr<GpuBackend> backend;
    Check(CreateCudaBackend(&backend));
    size_t cases = 0;
    for (auto extent : {Extent2D{17, 13}, Extent2D{273, 265}}) {
      if (smoke && extent.width != 17) continue;
      Image3FBuffer image(extent);
      for (size_t c = 0; c < 3; ++c)
        for (size_t i = 0; i < image.plane(c).size(); ++i)
          image.plane(c)[i] = .05f + .8f * ((i * (c + 3)) % 127) / 127.f;
      for (int effort : {1, 4, 5, 8, 9, 10}) {
        if (smoke && effort != 1 && effort != 5) continue;
        VarDctEncodingOptions options;
        options.backend = VarDctBackendPreference::kCuda;
        options.effort = effort;
        options.cpu_thread_count = 4;
        options.collect_final_butteraugli_score = false;
        Compare(*backend, image.const_view(), options, true);
        ++cases;
      }
      if (!smoke && extent.width == 17) {
        VarDctEncodingOptions options;
        options.backend = VarDctBackendPreference::kCuda;
        options.effort = 1;
        options.cpu_thread_count = 4;
        options.collect_final_butteraugli_score = false;
        options.gpu_aq_mode = GpuAdaptiveQuantizationMode::kThroughput;
        Compare(*backend, image.const_view(), options, true);
        ++cases;
        options.gpu_aq_mode = GpuAdaptiveQuantizationMode::kExactCoefficients;
        Compare(*backend, image.const_view(), options, false);
        ++cases;
        options.gpu_aq_mode = GpuAdaptiveQuantizationMode::kMaximumThroughput;
        Compare(*backend, image.const_view(), options, false);
        ++cases;
        options.gpu_aq_mode = GpuAdaptiveQuantizationMode::kFullyResident;
        options.compression_mode = VarDctCompressionMode::kMaximumCompression;
        Compare(*backend, image.const_view(), options, false);
        ++cases;
        options.compression_mode = VarDctCompressionMode::kAutomatic;
        options.rate_control_mode = VarDctRateControlMode::kMaximumError;
        options.maximum_error = {.05f, .05f, .05f};
        Compare(*backend, image.const_view(), options, false);
        ++cases;
        options.rate_control_mode = VarDctRateControlMode::kTargetBytes;
        options.target_bytes = 1;
        options.target_size_maximum_attempts = 3;
        options.target_size_tolerance = 0;
        Compare(*backend, image.const_view(), options, true);
        ++cases;
      }
      if (!smoke) Admission(*backend, image.const_view());
    }
    Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", nullptr);
    std::cout
        << "Verified " << cases
        << " CUDA token workflow policies, exact bytes/summary, CPU override"
        << (smoke ? "." : ", finite admission and rejection.") << '\n'
        << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n' << std::flush;
    return 1;
  }
}
