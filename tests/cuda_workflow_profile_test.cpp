// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "codestream/cuda_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "core/execution_domain.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_backend.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::gpu_profile_internal;
using namespace gjxl::resource_budget_internal;
void Check(Status s) {
  if (!s.ok()) throw std::runtime_error(std::string(s.message()));
}
void Require(bool good, const char* message) {
  if (!good) throw std::runtime_error(message);
}
void Empty(const ExecutionDomain& domain) {
  const auto s = domain.snapshot();
  Require(s.live_capacity_bytes == 0 && s.idle_capacity_bytes == 0 &&
    s.reserved_unbacked_bytes == 0 && s.active_reservations == 0 &&
    s.waiting_requests == 0 && s.active_cpu_participants == 0 &&
    s.reserved_cpu_workers == 0, "Profiled workflow leaked its domain");
}
VarDctEncodingOptions Options() {
  VarDctEncodingOptions o;
  o.backend = VarDctBackendPreference::kCuda;
  o.cpu_thread_count = 1;
  o.butteraugli_target = 1.2f;
  return o;
}
struct Result {
  std::vector<uint8_t> bytes;
  VarDctEncodingSummary summary;
  VarDctEncodingProfile cpu;
  GpuExecutionProfile gpu;
  bool operator==(const Result&) const = default;
};
Status Encode(GpuBackend& gpu, ConstImage3FView input,
              const VarDctEncodingOptions& options, Result* out,
              GpuProfilingMode mode = GpuProfilingMode::kStage) {
  return EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
    input, options, &gpu, true, mode, &out->bytes, &out->summary, &out->cpu, &out->gpu);
}

void Plans() {
  for (Extent2D extent : {Extent2D{1, 1}, {15, 15}, {65, 63}, {257, 9}, {3839, 2159}}) {
    for (int effort = 1; effort <= 10; ++effort) {
      for (size_t flags = 0; flags < 8; ++flags) {
        auto o = Options();
        o.effort = effort;
        o.collect_final_butteraugli_score = (flags & 1) != 0;
        o.gpu_aq_mode = flags & 2 ? GpuAdaptiveQuantizationMode::kThroughput
                                 : GpuAdaptiveQuantizationMode::kFullyResident;
        o.compression_mode = flags & 4 ? VarDctCompressionMode::kMaximumCompression
                                       : VarDctCompressionMode::kAutomatic;
        CudaWorkflowStoragePlan plan;
        ArmNextManagedHostAllocationFailureForTest();
        const auto status = ComputeCudaWorkflowStoragePlan(extent, {o, false, true, true}, &plan);
        const bool pending = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        Check(status);
        Require(pending && plan.profile_shape.submissions > 0 &&
          plan.profile_shape.dispatches == 0 && plan.diagnostics.peak_bytes > 0 &&
          plan.output.peak_bytes <= plan.working.peak_bytes,
          "CUDA profile planning allocated or lost its diagnostic bound");
      }
    }
  }
  for (size_t bad = 0; bad < 4; ++bad) {
    auto o = Options();
    if (bad == 0) o.gpu_aq_mode = GpuAdaptiveQuantizationMode::kExactCoefficients;
    if (bad == 1) o.gpu_aq_mode = GpuAdaptiveQuantizationMode::kMaximumThroughput;
    if (bad == 2) o.rate_control_mode = VarDctRateControlMode::kTargetBytes;
    if (bad == 3) o.rate_control_mode = VarDctRateControlMode::kMaximumError;
    CudaWorkflowStoragePlan sentinel;
    sentinel.profile_shape.submissions = 42;
    Require(!ComputeCudaWorkflowStoragePlan({65, 63}, {o, false, true, true}, &sentinel).ok() &&
      sentinel.profile_shape.submissions == 42, "Invalid CUDA profile plan changed output");
  }
}

Image3FBuffer Image(Extent2D extent) {
  Image3FBuffer image(extent);
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        image.view().plane[c].Row(y)[x] = 0.03f + 0.8f *
          static_cast<float>((x * 11 + y * 7 + c * 23 + x * y) % 251) / 251.0f;
  return image;
}

void Cases(GpuBackend& gpu) {
  size_t cases = 0;
  for (Extent2D extent : {Extent2D{15, 15}, {257, 65}}) {
    auto image = Image(extent);
    for (int effort : {1, 4, 7, 8, 10}) {
      for (bool score : {false, true}) {
        for (auto mode : {GpuAdaptiveQuantizationMode::kFullyResident,
                          GpuAdaptiveQuantizationMode::kThroughput}) {
          auto o = Options();
          o.effort = effort;
          o.collect_final_butteraugli_score = score;
          o.gpu_aq_mode = mode;
          Result reference;
          Check(EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
            image.const_view(), o, &gpu, true, &reference.bytes, &reference.summary));
          Check(gpu.TrimPreparationCache());
          CudaWorkflowStoragePlan plan;
          Check(ComputeCudaWorkflowStoragePlan(extent, {o, false, true, true}, &plan));
          std::shared_ptr<const ExecutionDomain> domain;
          Check(ExecutionDomain::Create({plan.working.peak_bytes, 1}, &domain));
          o.execution_domain = domain;
          const auto fallback_peak = DefaultResourceBudget().snapshot().peak_backing_bytes;
          Result actual;
          for (size_t repeat = 0; repeat < 2; ++repeat) {
            Check(Encode(gpu, image.const_view(), o, &actual));
            Require(actual.bytes == reference.bytes && actual.summary == reference.summary,
              "Profiled CUDA workflow changed bytes or summary");
            Require(actual.gpu.mode == GpuProfilingMode::kStage &&
              actual.gpu.submissions.size() <= plan.profile_shape.submissions &&
              actual.gpu.wall_stages.size() <= plan.profile_shape.wall_stages &&
              !actual.gpu.submissions.empty(), "CUDA profile exceeded its planned graph");
            for (const auto& s : actual.gpu.submissions) {
              Require(s.stages.size() == 1 && s.stages[0].dispatches.empty() &&
                s.submission_id.size() <= plan.profile_shape.maximum_id_length &&
                s.stages[0].stage_id.size() <= plan.profile_shape.maximum_id_length,
                "CUDA submission profile exceeded its planned shape");
            }
            for (const auto& wall : actual.gpu.wall_stages)
              Require(wall.stage_id.size() <= plan.profile_shape.maximum_id_length,
                "CUDA wall-stage label exceeded its planned shape");
            Require(domain->snapshot().peak_backing_bytes <= plan.working.peak_bytes,
              "Profiled CUDA workflow exceeded admission");
            Require(DefaultResourceBudget().snapshot().peak_backing_bytes == fallback_peak,
              "Profiled CUDA workflow escaped to the default resource domain");
            ++cases;
          }
          Check(gpu.TrimPreparationCache());
          Empty(*domain); // Published graphs and codestreams still exist.
        }
      }
    }
  }
  std::cout << cases << " CUDA workflow profiles preserved bytes/summary within finite admission\n";
}

void Failures(GpuBackend& gpu) {
  auto image = Image({65, 63});
  auto o = Options();
  CudaWorkflowStoragePlan plan;
  Check(ComputeCudaWorkflowStoragePlan(image.extent(), {o, false, true, true}, &plan));
  std::shared_ptr<const ExecutionDomain> domain;
  Check(ExecutionDomain::Create({plan.working.peak_bytes, 1}, &domain));
  o.execution_domain = domain;
  Result sentinel;
  sentinel.bytes = {3, 1, 4};
  sentinel.summary.encoded_bytes = 17;
  sentinel.cpu.total_nanoseconds = 19;
  sentinel.gpu.mode = GpuProfilingMode::kDispatch;
  size_t failures = 0;
  bool finished = false, after_submission = false;
  for (size_t skip = 0; skip < 128; ++skip) {
    Result actual = sentinel;
    const auto before = gpu.stats();
    ArmManagedHostClassAllocationFailureAfterForTest(ResourceClass::kDiagnostics, skip);
    const Status status = Encode(gpu, image.const_view(), o, &actual);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(gpu.TrimPreparationCache());
    Empty(*domain);
    if (pending) {
      Check(status);
      finished = true;
      break;
    }
    Require(!status.ok() && actual == sentinel,
      "CUDA diagnostic allocation failure published partial workflow outputs");
    after_submission |= gpu.stats().committed_submissions > before.committed_submissions;
    ++failures;
  }
  Require(finished && failures > 10 && after_submission,
    "CUDA diagnostic failure sweep did not reach every allocation boundary");
  Result actual = sentinel;
  Require(Encode(gpu, image.const_view(), o, &actual, GpuProfilingMode::kDispatch).code() ==
      StatusCode::kUnavailable && actual == sentinel,
    "Unsupported CUDA dispatch profiling changed caller outputs");
  Check(gpu.TrimPreparationCache());
  Empty(*domain);
  std::shared_ptr<const ExecutionDomain> tiny;
  Check(ExecutionDomain::Create({plan.working.peak_bytes - 1, 1}, &tiny));
  o.execution_domain = tiny;
  const auto before = gpu.stats();
  Require(!Encode(gpu, image.const_view(), o, &actual).ok() && actual == sentinel &&
    gpu.stats().committed_submissions == before.committed_submissions &&
    gpu.stats().successful_allocations == before.successful_allocations,
    "Insufficient CUDA profile admission submitted work or changed outputs");
  Empty(*tiny);
  std::cout << failures << " CUDA diagnostic allocation boundaries preserved caller outputs\n";
}
} // namespace

int main() {
  try {
    Plans();
    std::unique_ptr<GpuBackend> gpu;
    const Status status = CreateCudaBackend(&gpu);
    if (status.code() == StatusCode::kUnavailable) return 77;
    Check(status);
    Cases(*gpu);
    Failures(*gpu);
  } catch (const std::exception& e) {
    DisarmManagedHostAllocationFailureForTest();
    std::cerr << e.what() << '\n';
    return 1;
  }
}
