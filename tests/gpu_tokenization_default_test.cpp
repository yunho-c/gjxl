// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/ac_tokenization_provider_internal.h"
#include "codestream/compatibility_workflow_storage_plan.h"
#include "codestream/resident_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"
#include "gpu/metal/metal_backend.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
void Select(const char *value) {
  if (value)
    setenv("GJXL_GPU_TOKENIZATION", value, 1);
  else
    unsetenv("GJXL_GPU_TOKENIZATION");
}
struct Encoded {
  std::vector<uint8_t> bytes;
  VarDctEncodingSummary summary;
  uint64_t submissions = 0;
};
Encoded Encode(ConstImage3FView image, VarDctEncodingOptions options) {
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  Encoded result;
  Ok(EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image, options, gpu.get(), true, &result.bytes, &result.summary));
  result.submissions = gpu->stats().committed_submissions;
  return result;
}
void CheckPlans() {
  constexpr Extent2D extent{273, 265};
  ResidentWorkflowStorageOptions resident;
  resident.encoding.backend = VarDctBackendPreference::kMetal;
  resident.encoding.effort = 1;
  resident.encoding.cpu_thread_count = 4;
  CpuWorkflowStorageOptions cpu;
  cpu.encoding = resident.encoding;
  cpu.encoding.backend = VarDctBackendPreference::kCpu;
  auto compatibility = cpu;
  compatibility.encoding.backend = VarDctBackendPreference::kMetal;
  compatibility.encoding.gpu_aq_mode =
      GpuAdaptiveQuantizationMode::kExactCoefficients;
  ResidentWorkflowStoragePlan enabled, disabled;
  CpuWorkflowStoragePlan cpu_enabled, cpu_disabled;
  MetalCompatibilityWorkflowStoragePlan compatibility_enabled,
      compatibility_disabled;
  Select(nullptr);
  Ok(ComputeResidentWorkflowStoragePlan(extent, resident, &enabled));
  Ok(ComputeCpuWorkflowStoragePlan(extent, cpu, &cpu_enabled));
  Ok(ComputeMetalCompatibilityWorkflowStoragePlan(extent, compatibility,
                                                  &compatibility_enabled));
  Select("0");
  Ok(ComputeResidentWorkflowStoragePlan(extent, resident, &disabled));
  Ok(ComputeCpuWorkflowStoragePlan(extent, cpu, &cpu_disabled));
  Ok(ComputeMetalCompatibilityWorkflowStoragePlan(extent, compatibility,
                                                  &compatibility_disabled));
  Require(enabled.serializer.token_idle_pool_capacity[0] > 0 &&
              enabled.serializer.token_idle_pool_capacity[1] > 0 &&
              disabled.serializer.token_idle_pool_capacity ==
                  std::array<size_t, 2>{} &&
              enabled.working.peak_bytes > disabled.working.peak_bytes,
          "Default token arenas or CPU override admission are incorrect");
  Require(cpu_enabled == cpu_disabled &&
              compatibility_enabled == compatibility_disabled,
          "GPU default changed CPU/compatibility admission");
  resident.encoding.compression_mode =
      VarDctCompressionMode::kMaximumCompression;
  Select(nullptr);
  Ok(ComputeResidentWorkflowStoragePlan(extent, resident, &enabled));
  Select("0");
  Ok(ComputeResidentWorkflowStoragePlan(extent, resident, &disabled));
  Require(enabled == disabled, "Exhaustive CPU tokenizer reserved GPU arenas");
}
} // namespace

int main() try {
  for (const char *name :
       {"GJXL_EXPERIMENT_GPU_TOKENS", "GJXL_EXPERIMENT_TOKEN_COMPACT",
        "GJXL_EXPERIMENT_TOKEN_OVERLAP", "GJXL_EXPERIMENT_TOKEN_SHARDS",
        "GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8", "GJXL_EXPERIMENT_TOKEN_GROUP_DCT8",
        "GJXL_EXPERIMENT_TOKEN_CAPACITY", "GJXL_EXPERIMENT_TOKEN_CACHE"})
    unsetenv(name);
  Select(nullptr);
  Require(GpuTokenizationEnabled() && GpuTokenizationCompactLayout() == 2 &&
              GpuTokenizationOverlapEnabled(),
          "Qualified default configuration is disabled");
#ifndef GJXL_TOKENIZATION_EXPERIMENT
  setenv("GJXL_EXPERIMENT_TOKEN_COMPACT", "0", 1);
  setenv("GJXL_EXPERIMENT_TOKEN_OVERLAP", "0", 1);
  Require(GpuTokenizationCompactLayout() == 2 &&
              GpuTokenizationOverlapEnabled(),
          "Experimental controls affected a production build");
  unsetenv("GJXL_EXPERIMENT_TOKEN_COMPACT");
  unsetenv("GJXL_EXPERIMENT_TOKEN_OVERLAP");
#endif
  setenv("GJXL_EXPERIMENT_GPU_TOKENS", "1", 1);
  Select("0");
  Require(!GpuTokenizationEnabled(), "Stable CPU override lost precedence");
  unsetenv("GJXL_EXPERIMENT_GPU_TOKENS");
  CheckPlans();
  Image3FBuffer image({273, 265});
  for (size_t c = 0; c < 3; ++c)
    for (size_t i = 0; i < image.plane(c).size(); ++i)
      image.plane(c)[i] = 0.05f + 0.8f * ((i * (c + 3)) % 127) / 127.0f;
  size_t cases = 0;
  for (int effort : {1, 4, 7, 8, 9, 10}) {
    VarDctEncodingOptions options;
    options.backend = VarDctBackendPreference::kMetal;
    options.effort = effort;
    options.cpu_thread_count = 4;
    options.butteraugli_target = 1.9f;
    Select("0");
    auto cpu = Encode(image.const_view(), options);
    Select(nullptr);
    auto gpu = Encode(image.const_view(), options);
    Require(cpu.bytes == gpu.bytes && cpu.summary == gpu.summary,
            "Default GPU tokenization changed bytes or decisions");
    Require(gpu.submissions > cpu.submissions,
            "Default workflow did not submit GPU tokenization");
    Select("1");
    auto explicit_gpu = Encode(image.const_view(), options);
    Require(explicit_gpu.bytes == gpu.bytes &&
                explicit_gpu.summary == gpu.summary &&
                explicit_gpu.submissions == gpu.submissions,
            "Explicit GPU selection differs from default");
    ++cases;
  }
  // Exhaustive representation selection remains CPU-tokenized even when the
  // resident frontend produces a final coefficient buffer.
  Image3FBuffer small({17, 9});
  VarDctEncodingOptions options;
  options.backend = VarDctBackendPreference::kMetal;
  options.effort = 1;
  options.cpu_thread_count = 1;
  options.compression_mode = VarDctCompressionMode::kMaximumCompression;
  Select("0");
  auto cpu = Encode(small.const_view(), options);
  Select(nullptr);
  auto gpu = Encode(small.const_view(), options);
  Require(cpu.bytes == gpu.bytes && cpu.summary == gpu.summary &&
              cpu.submissions == gpu.submissions,
          "Exhaustive workflow activated the GPU tokenizer");
  std::cout << "Verified default/explicit GPU and CPU override across " << cases
            << " efforts, actual GPU submissions, scoped admission and "
               "exhaustive fallback.\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
