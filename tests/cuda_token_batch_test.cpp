// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "codestream/batch_workflow_test.h"
#include "codestream/workflow_admission.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_ac_tokenization.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::cuda_internal;
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
  Check(WorkflowAdmission::TrimIdle(domain));
  const auto s = domain.snapshot();
  Require(s.active_reservations == 0 && s.waiting_requests == 0 &&
              s.live_capacity_bytes == 0 && s.idle_capacity_bytes == 0 &&
              s.reserved_unbacked_bytes == 0 &&
              s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 &&
              s.suspended_cpu_workers == 0 && s.waiting_cpu_callers == 0 &&
              s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit &&
              s.peak_committed_bytes <= domain.options().managed_memory_bytes,
          "Token batch leaked or exceeded domain capacity");
}
constexpr size_t kImages = 4, kRequests = 5;
struct Observations {
  std::array<uint64_t, kRequests> before{}, providers{};
  std::array<bool, kRequests> entered{}, exited{};
  std::atomic<size_t> active{0}, peak{0};
  bool fault = false;
  bool fault_consumed = false;
  void Install() {
    batch_execution_observer_for_testing = {
        this, +[](void* opaque, size_t index, bool entering) noexcept {
          auto& o = *static_cast<Observations*>(opaque);
          if (entering) {
            o.entered[index] = true;
            o.before[index] = cuda_token_provider_begin_count;
            const auto now = o.active.fetch_add(1) + 1;
            auto peak = o.peak.load();
            while (peak < now && !o.peak.compare_exchange_weak(peak, now)) {
            }
            if (o.fault && index == 1)
              ArmManagedHostClassAllocationFailureAfterForTest(
                  ResourceClass::kSerializer, 0);
          } else {
            o.exited[index] = true;
            o.providers[index] =
                cuda_token_provider_begin_count - o.before[index];
            if (o.fault && index == 1) {
              o.fault_consumed = !ManagedHostAllocationFailurePendingForTest();
              DisarmManagedHostAllocationFailureForTest();
            }
            o.active.fetch_sub(1);
          }
        }};
  }
  ~Observations() { batch_execution_observer_for_testing = {}; }
};
struct Fixture {
  std::array<Image3FBuffer, kImages> images;
  std::array<VarDctBatchEncodingRequest, kRequests> requests;
  std::array<VarDctBatchEncodingResult, kImages> oracle;
  Fixture() {
    const std::array extents{Extent2D{17, 13}, Extent2D{273, 265},
                             Extent2D{65, 63}, Extent2D{521, 9}};
    const std::array efforts{1, 5, 8, 4};
    Env("GJXL_GPU_TOKENIZATION", "1");
    Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", "0");
    for (size_t i = 0; i < kImages; ++i) {
      images[i] = Image3FBuffer(extents[i]);
      for (size_t c = 0; c < 3; ++c)
        for (size_t j = 0; j < images[i].plane(c).size(); ++j)
          images[i].plane(c)[j] =
              .05f + .8f * ((j * (c + 3) + i) % 127) / 127.f;
      auto& options = requests[i].options;
      options.backend = VarDctBackendPreference::kCuda;
      options.cpu_thread_count = 2;
      options.effort = efforts[i];
      options.collect_final_butteraugli_score = i == 3;
      if (i == 1)
        options.gpu_aq_mode = GpuAdaptiveQuantizationMode::kThroughput;
      if (i == 3)
        options.compression_mode = VarDctCompressionMode::kMaximumCompression;
      requests[i].linear_rgb = images[i].const_view();
      const auto before = cuda_token_provider_begin_count;
      Check(EncodeLinearRgbVarDctCodestream(requests[i].linear_rgb, options,
                                            &oracle[i].codestream,
                                            &oracle[i].summary));
      Require(cuda_token_provider_begin_count == before,
              "Batch CPU oracle used GPU tokens");
    }
    requests.back().options = requests.front().options;
    Check(TrimVarDctPreparationCache());
    Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", "1");
  }
  BatchWorkflowStorageAccumulator Accumulator() const {
    BatchWorkflowStorageAccumulator accumulator;
    for (size_t i = 0; i < kImages; ++i) {
      WorkflowStoragePlan plan;
      Check(ComputeWorkflowStoragePlan(
          images[i].extent(),
          {requests[i].options, WorkflowStorageRoute::kCuda,
           WorkflowStorageAdapter::kBorrowedLinearRgb, true},
          &plan));
      Check(accumulator.AddRequest(&plan));
    }
    Check(accumulator.AddRequest(nullptr));
    return accumulator;
  }
  void Domain(std::shared_ptr<const ExecutionDomain> domain) {
    for (auto& request : requests) request.options.execution_domain = domain;
  }
  void Results(const std::vector<VarDctBatchEncodingResult>& results,
               const Observations& observation, size_t in_flight, bool enabled,
               bool fault) const {
    Require(results.size() == kRequests && observation.active == 0 &&
                observation.peak > 0 && observation.peak <= in_flight,
            "Token batch lost results or exceeded work-slot concurrency");
    Require(results.back().status.code() == StatusCode::kInvalidArgument &&
                results.back().codestream.empty() &&
                !observation.entered.back(),
            "Invalid batch item reached GPU work or lost error isolation");
    for (size_t i = 0; i < kImages; ++i) {
      Require(observation.entered[i] && observation.exited[i],
              "Batch worker observation missing");
      if (fault && i == 1) {
        Require(observation.fault_consumed &&
                    results[i].status.code() == StatusCode::kOutOfMemory &&
                    results[i].codestream.empty() &&
                    results[i].summary.encoded_bytes == 0,
                "Injected token-batch allocation failure was not isolated");
        continue;
      }
      Check(results[i].status);
      Require(results[i].codestream == oracle[i].codestream &&
                  results[i].summary == oracle[i].summary &&
                  results[i].summary.execution_backend ==
                      VarDctExecutionBackend::kCuda &&
                  observation.providers[i] == uint64_t(enabled && i != 3),
              "Mixed token batch changed bytes, summary or provider selection");
      Require(results[i].scheduling.cpu_admitted &&
                  results[i].scheduling.service_nanoseconds > 0 &&
                  results[i].scheduling.ready_nanoseconds ==
                      results[i].scheduling.queue_nanoseconds +
                          results[i].scheduling.service_nanoseconds,
              "Token batch lost queue/service timing");
    }
  }
};
void Case(Fixture& fixture, size_t workers, bool minimum, size_t* batches) {
  fixture.Domain({});
  const auto accumulator = fixture.Accumulator();
  BatchWorkflowStoragePlan unlimited, plan;
  Check(accumulator.Finish(workers, 0, &unlimited));
  const size_t limit =
      minimum ? unlimited.minimum_required_bytes : unlimited.working.peak_bytes;
  Check(accumulator.Finish(workers, limit, &plan));
  Require(!minimum || (plan.in_flight == 1 && plan.trim_after_each_image),
          "Minimum batch plan did not exercise serial reuse and trimming");
  std::shared_ptr<const ExecutionDomain> domain;
  Check(ExecutionDomain::Create({limit, 4}, &domain));
  fixture.Domain(domain);
  std::unique_ptr<VarDctBatchEncoder> driver;
  Check(VarDctBatchEncoder::Create(workers, &driver));
  std::vector<VarDctBatchEncodingResult> retained;
  const auto fallback_peak =
      DefaultResourceBudget().snapshot().peak_backing_bytes;
  for (size_t repeat = 0; repeat < 6; ++repeat) {
    // Three enabled calls, stable CPU override, an isolated allocation failure,
    // then recovery on the same persistent driver and production CUDA lanes.
    const bool enabled = repeat != 3, fault = repeat == 4;
    Env("GJXL_GPU_TOKENIZATION", enabled ? "1" : "0");
    Observations observation;
    observation.fault = fault;
    observation.Install();
    std::vector<VarDctBatchEncodingResult> results;
    Check(driver->Encode(fixture.requests, &results));
    fixture.Results(results, observation, plan.in_flight, enabled, fault);
    Require(
        DefaultResourceBudget().snapshot().peak_backing_bytes == fallback_peak,
        "Token batch escaped its explicit execution domain");
    const auto snapshot = domain->snapshot();
    Require(snapshot.active_reservations == 0 &&
                snapshot.peak_committed_bytes <= limit,
            "Token batch retained admission or exceeded its bound");
    retained = std::move(results);
    // Keep pools between repeated enabled calls; trim at route changes.
    if (repeat >= 2) Empty(*domain);
    ++*batches;
  }
  driver->Shutdown();
  Empty(*domain);
  Require(!retained.front().codestream.empty(),
          "Published batch was invalidated by shutdown");
  std::shared_ptr<const ExecutionDomain> tiny;
  Check(ExecutionDomain::Create({unlimited.minimum_required_bytes - 1, 4},
                                &tiny));
  fixture.Domain(tiny);
  Check(VarDctBatchEncoder::Create(workers, &driver));
  std::vector<VarDctBatchEncodingResult> sentinel(1);
  sentinel[0].codestream = {3, 1, 4};
  sentinel[0].summary.encoded_bytes = 17;
  sentinel[0].timing.total_nanoseconds = 19;
  Observations observation;
  observation.Install();
  Require(!driver->Encode(fixture.requests, &sentinel).ok() &&
              sentinel.size() == 1 &&
              sentinel[0].codestream == std::vector<uint8_t>({3, 1, 4}) &&
              sentinel[0].summary.encoded_bytes == 17 &&
              sentinel[0].timing.total_nanoseconds == 19 &&
              observation.peak == 0,
          "Infeasible token batch started workers or changed caller output");
  driver->Shutdown();
  Empty(*tiny);
  fixture.Domain({});
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool smoke = argc == 2 && std::strcmp(argv[1], "--smoke") == 0;
    Require(argc == 1 || smoke, "Unknown argument");
    Fixture fixture;
    size_t cases = 0, batches = 0;
    for (size_t workers : {size_t{1}, size_t{2}, size_t{4}}) {
      if (smoke && workers != 2) continue;
      for (bool minimum : {false, true}) {
        if (smoke && !minimum) continue;
        Case(fixture, workers, minimum, &batches);
        ++cases;
      }
    }
    Env("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", nullptr);
    Env("GJXL_GPU_TOKENIZATION", nullptr);
    std::cout
        << "Verified " << cases << " CUDA token batch policies, " << batches
        << " mixed-size batches, exact bytes/summary, provider selection, "
           "finite admission, failure isolation and recovery.\n"
        << std::flush;
  } catch (const std::exception& error) {
    batch_execution_observer_for_testing = {};
    std::cerr << error.what() << '\n' << std::flush;
    return 1;
  }
}
