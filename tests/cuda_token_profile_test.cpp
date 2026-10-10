// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "codestream/cuda_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_ac_tokenization.h"
#include "gpu/cuda/cuda_backend.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::cuda_internal;
using namespace gjxl::gpu_profile_internal;
using namespace gjxl::resource_budget_internal;

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
void Empty(const ExecutionDomain& domain) {
  const auto s = domain.snapshot();
  Require(s.live_capacity_bytes == 0 && s.idle_capacity_bytes == 0 &&
              s.reserved_unbacked_bytes == 0 && s.active_reservations == 0 &&
              s.waiting_requests == 0 && s.active_cpu_participants == 0 &&
              s.reserved_cpu_workers == 0,
          "Token profile retained domain backing or participants");
}
void Trim(GpuBackend& backend, const ExecutionDomain& domain) {
  Check(TrimVarDctPreparationCache());
  Check(backend.TrimPreparationCache());
  Empty(domain);
}
struct Result {
  std::vector<uint8_t> bytes;
  VarDctEncodingSummary summary;
  VarDctEncodingProfile cpu;
  GpuExecutionProfile gpu;
  bool operator==(const Result&) const = default;
};
Status Encode(GpuBackend& backend, ConstImage3FView image,
              const VarDctEncodingOptions& options, GpuProfilingMode mode,
              Result* result) {
  if (mode == GpuProfilingMode::kDisabled)
    return EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
        image, options, &backend, true, &result->bytes, &result->summary,
        &result->cpu);
  return EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
      image, options, &backend, true, mode, &result->bytes, &result->summary,
      &result->cpu, &result->gpu);
}
void Graph(const Result& result, const CudaWorkflowStoragePlan& plan,
           GpuProfilingMode mode, bool eligible) {
  Require(result.cpu.execution_backend == VarDctExecutionBackend::kCuda &&
              result.cpu.total_nanoseconds > 0,
          "Host profile missing or selected a different backend");
  if (mode == GpuProfilingMode::kDisabled) {
    Require(result.gpu == GpuExecutionProfile{},
            "Host-only profile changed GPU output");
    return;
  }
  const auto& gpu = result.gpu;
  Require(
      gpu.mode == mode && gpu.capabilities.timestamp_counter &&
          gpu.capabilities.stage_boundary &&
          gpu.capabilities.dispatch_boundary && !gpu.submissions.empty() &&
          gpu.submissions.size() <= plan.profile_shape.submissions &&
          gpu.wall_stages.size() <= plan.profile_shape.wall_stages,
      "Token GPU profile exceeded its submission bound or lost capabilities");
  size_t copies = 0, stages = 0, dispatches = 0;
  for (const auto& submission : gpu.submissions) {
    Require(submission.stages.size() == 1 &&
                submission.submission_id.size() <=
                    plan.profile_shape.maximum_id_length,
            "Token submission shape differs");
    for (const auto& stage : submission.stages) {
      ++stages;
      Require(stage.stage_id.size() <= plan.profile_shape.maximum_id_length &&
                  stage.timestamp_valid &&
                  stage.end_timestamp >= stage.begin_timestamp &&
                  stage.gpu_nanoseconds ==
                      stage.end_timestamp - stage.begin_timestamp,
              "Token stage timeline is invalid");
      if (stage.stage_id == "aq.completed_token_copy") {
        ++copies;
        Require(stage.dispatches.empty(),
                "Coefficient copy was reported as a kernel");
      } else {
        Require(!stage.dispatches.empty(), "Unexpected empty kernel stage");
      }
      uint64_t previous_end = 0;
      for (const auto& dispatch : stage.dispatches) {
        ++dispatches;
        Require(!dispatch.kernel_id.empty() &&
                    dispatch.kernel_id.size() <=
                        plan.profile_shape.maximum_id_length &&
                    dispatch.kind == GpuDispatchKind::kThreadgroups &&
                    dispatch.grid.width && dispatch.grid.height &&
                    dispatch.grid.depth &&
                    dispatch.threads_per_threadgroup.width &&
                    dispatch.threads_per_threadgroup.height &&
                    dispatch.threads_per_threadgroup.depth,
                "Token profile dispatch metadata is invalid");
        Require(dispatch.begin_timestamp >= previous_end &&
                    dispatch.end_timestamp >= dispatch.begin_timestamp &&
                    dispatch.end_timestamp <= stage.end_timestamp &&
                    dispatch.gpu_nanoseconds ==
                        dispatch.end_timestamp - dispatch.begin_timestamp,
                "Token profile dispatch timeline is invalid");
        Require(
            dispatch.timestamp_valid == (mode == GpuProfilingMode::kDispatch),
            "Token profile dispatch timestamp validity differs");
        if (mode == GpuProfilingMode::kStage)
          Require(dispatch.begin_timestamp == 0 && dispatch.end_timestamp == 0,
                  "Stage profile recorded dispatch timestamps");
        previous_end = dispatch.end_timestamp;
      }
    }
  }
  Require(copies == size_t(eligible) && stages <= plan.profile_shape.stages &&
              dispatches > 0 && dispatches <= plan.profile_shape.dispatches,
          "Token copy count or profile graph bound differs");
  for (const auto& wall : gpu.wall_stages)
    Require(wall.stage_id.size() <= plan.profile_shape.maximum_id_length,
            "Token wall-stage label exceeded its bound");
}

void Case(GpuBackend& backend, ConstImage3FView image,
          VarDctEncodingOptions options, GpuProfilingMode mode, bool eligible) {
  const bool gpu_profile = mode != GpuProfilingMode::kDisabled;
  Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", "0");
  Result reference;
  const auto before_reference = cuda_token_provider_begin_count;
  Check(EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image, options, &backend, true, &reference.bytes, &reference.summary));
  Require(cuda_token_provider_begin_count == before_reference,
          "CPU oracle used GPU tokens");
  Check(TrimVarDctPreparationCache());
  Check(backend.TrimPreparationCache());
  CudaWorkflowStoragePlan disabled, enabled;
  Check(ComputeCudaWorkflowStoragePlan(
      image.extent(), {options, false, true, gpu_profile}, &disabled));
  Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", "1");
  Check(ComputeCudaWorkflowStoragePlan(
      image.extent(), {options, false, true, gpu_profile}, &enabled));
  Require((enabled.completed.peak_bytes > disabled.completed.peak_bytes) ==
                  eligible &&
              (enabled.serializer.working.peak_bytes >
               disabled.serializer.working.peak_bytes) == eligible,
          "Profile plan lost the token admission envelope");
  if (gpu_profile)
    Require(enabled.profile_shape.submissions ==
                    disabled.profile_shape.submissions + size_t(eligible) &&
                enabled.profile_shape.dispatches ==
                    disabled.profile_shape.dispatches,
            "Profile plan did not count the copy separately from kernels");
  std::shared_ptr<const ExecutionDomain> domain;
  Check(ExecutionDomain::Create({enabled.working.peak_bytes, 2}, &domain));
  options.execution_domain = domain;
  const auto fallback_peak =
      DefaultResourceBudget().snapshot().peak_backing_bytes;
  for (size_t repeat = 0; repeat < 2; ++repeat) {
    Result actual;
    const auto before = cuda_token_provider_begin_count;
    Check(Encode(backend, image, options, mode, &actual));
    Require((cuda_token_provider_begin_count > before) == eligible,
            "Profiling changed GPU token provider selection");
    Require(
        actual.bytes == reference.bytes && actual.summary == reference.summary,
        "Token profile changed codestream bytes or summary");
    Graph(actual, enabled, mode, eligible);
    Require(
        domain->snapshot().peak_committed_bytes <= enabled.working.peak_bytes &&
            domain->snapshot().active_reservations == 0 &&
            DefaultResourceBudget().snapshot().peak_backing_bytes ==
                fallback_peak,
        "Token profile exceeded admission or escaped to the default domain");
  }
  Trim(backend, *domain);
  Env("GJXL_GPU_TOKENIZATION", "0");
  Result overridden;
  const auto before_override = cuda_token_provider_begin_count;
  Check(Encode(backend, image, options, mode, &overridden));
  Require(cuda_token_provider_begin_count == before_override &&
              overridden.bytes == reference.bytes &&
              overridden.summary == reference.summary,
          "Stable CPU override failed during profiling");
  Graph(overridden, disabled, mode, false);
  Trim(backend, *domain);
  Env("GJXL_GPU_TOKENIZATION", "1");

  std::shared_ptr<const ExecutionDomain> small;
  Check(ExecutionDomain::Create({enabled.working.peak_bytes - 1, 2}, &small));
  options.execution_domain = small;
  Result sentinel;
  sentinel.bytes = {3, 1, 4};
  sentinel.summary.encoded_bytes = 17;
  sentinel.cpu.total_nanoseconds = 19;
  sentinel.gpu.mode = GpuProfilingMode::kDispatch;
  Result actual = sentinel;
  const auto before = cuda_token_provider_begin_count;
  const auto stats = backend.stats();
  Require(!Encode(backend, image, options, mode, &actual).ok() &&
              actual == sentinel && cuda_token_provider_begin_count == before &&
              backend.stats().committed_submissions ==
                  stats.committed_submissions &&
              backend.stats().successful_allocations ==
                  stats.successful_allocations,
          "Insufficient profile admission submitted work or published partial "
          "output");
  Empty(*small);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool smoke = argc == 2 && std::strcmp(argv[1], "--smoke") == 0;
    Require(argc == 1 || smoke, "Unknown argument");
    Env("GJXL_GPU_TOKENIZATION", "1");
    std::unique_ptr<GpuBackend> backend;
    Check(CreateCudaBackend(&backend));
    size_t cases = 0;
    for (auto extent : {Extent2D{17, 13}, Extent2D{273, 265}}) {
      if (smoke && extent.width != 17) continue;
      Image3FBuffer image(extent);
      for (size_t c = 0; c < 3; ++c)
        for (size_t i = 0; i < image.plane(c).size(); ++i)
          image.plane(c)[i] = .05f + .8f * ((i * (c + 3)) % 127) / 127.f;
      VarDctEncodingOptions options;
      options.backend = VarDctBackendPreference::kCuda;
      options.cpu_thread_count = 2;
      options.collect_final_butteraugli_score = false;
      for (int effort : {1, 5, 8}) {
        if (smoke && effort == 8) continue;
        options.effort = effort;
        for (auto aq : {GpuAdaptiveQuantizationMode::kFullyResident,
                        GpuAdaptiveQuantizationMode::kThroughput}) {
          if (smoke && aq == GpuAdaptiveQuantizationMode::kThroughput) continue;
          options.gpu_aq_mode = aq;
          for (auto mode :
               {GpuProfilingMode::kDisabled, GpuProfilingMode::kStage,
                GpuProfilingMode::kDispatch}) {
            Case(*backend, image.const_view(), options, mode, true);
            ++cases;
          }
        }
      }
      if (!smoke && extent.width == 17) {
        options.effort = 4;
        options.gpu_aq_mode = GpuAdaptiveQuantizationMode::kFullyResident;
        for (bool maximum_compression : {false, true}) {
          options.collect_final_butteraugli_score = !maximum_compression;
          options.compression_mode =
              maximum_compression ? VarDctCompressionMode::kMaximumCompression
                                  : VarDctCompressionMode::kAutomatic;
          for (auto mode :
               {GpuProfilingMode::kDisabled, GpuProfilingMode::kStage,
                GpuProfilingMode::kDispatch}) {
            Case(*backend, image.const_view(), options, mode,
                 !maximum_compression);
            ++cases;
          }
        }
        options.compression_mode = VarDctCompressionMode::kAutomatic;
        options.rate_control_mode = VarDctRateControlMode::kTargetBytes;
        options.target_bytes = 1;
        options.target_size_maximum_attempts = 3;
        options.target_size_tolerance = 0;
        Case(*backend, image.const_view(), options, GpuProfilingMode::kDisabled,
             true);
        ++cases;
      }
    }
    Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", nullptr);
    Env("GJXL_GPU_TOKENIZATION", nullptr);
    std::cout
        << "Verified " << cases
        << " CUDA token profile policies, exact bytes/summary, provider "
           "selection, copy stages, finite admission and atomic rejection.\n"
        << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n' << std::flush;
    return 1;
  }
}
