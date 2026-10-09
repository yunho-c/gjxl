// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/workflow_admission.h"
#include "codestream/workflow_storage_plan.h"
#include "core/image_buffer.h"
#include "environment_test_utils.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using gjxl_test::SetEnvironment;
#ifdef GJXL_TEST_CUDA_ENTROPY_READINESS
constexpr auto kBackend = VarDctBackendPreference::kCuda;
constexpr auto kRoute = WorkflowStorageRoute::kCuda;
#else
constexpr auto kBackend = VarDctBackendPreference::kMetal;
constexpr auto kRoute = WorkflowStorageRoute::kMetal;
#endif

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

void CheckBoundedDefault() {
  SetEnvironment("GJXL_GPU_TOKENIZATION", nullptr);
  SetEnvironment("GJXL_EXPERIMENT_CUDA_GPU_TOKENS", nullptr);
  SetEnvironment("GJXL_EARLY_ENTROPY", nullptr);
  SetEnvironment("GJXL_EXPERIMENT_EAGER_ENTROPY", nullptr);
  size_t cases = 0;
  // The narrow shape crosses a DC-group boundary: CPU 2 must fall back,
  // while CPU 4 can reserve both DC participants and the AC participant.
  for (const char* setting : {static_cast<const char*>(nullptr), "0", "1"}) {
  SetEnvironment("GJXL_EARLY_DC", setting);
  for (Extent2D extent : {Extent2D{273, 265}, Extent2D{2057, 17}}) {
    Image3FBuffer image(extent);
    for (size_t c = 0; c < 3; ++c)
      for (size_t i = 0; i < image.plane(c).size(); ++i)
        image.plane(c)[i] = 0.05f + 0.8f * ((i * (c + 3)) % 127) / 127.0f;
    for (int effort : {1, 4, 7})
      for (size_t cpu : {1, 2, 4})
        for (auto aq : {GpuAdaptiveQuantizationMode::kFullyResident,
                        GpuAdaptiveQuantizationMode::kThroughput})
          for (bool timing : {false, true}) {
            VarDctEncodingOptions options;
            options.backend = kBackend;
            options.effort = effort;
            options.butteraugli_target = 1.9f;
            options.cpu_thread_count = cpu;
            options.gpu_aq_mode = aq;
            const auto encode = [&](std::vector<uint8_t>* bytes,
                                    VarDctEncodingSummary* summary) {
              VarDctEncodingTiming measured;
              if (timing) {
                Ok(EncodeLinearRgbVarDctCodestreamProfiled(
                    image.const_view(), options, bytes, summary, &measured));
                Require(measured.total_nanoseconds > 0 && !measured.attempts.empty(),
                        "Public workflow timing was not populated");
              } else {
                Ok(EncodeLinearRgbVarDctCodestream(
                    image.const_view(), options, bytes, summary));
              }
            };
            const WorkflowStorageOptions planning{
                options, kRoute,
                WorkflowStorageAdapter::kBorrowedLinearRgb, timing};
            WorkflowStoragePlan disabled_plan, enabled_plan;
            SetEnvironment("GJXL_EARLY_ENTROPY", "0");
            Ok(ComputeWorkflowStoragePlan(extent, planning, &disabled_plan));
            std::vector<uint8_t> expected;
            VarDctEncodingSummary expected_summary;
            encode(&expected, &expected_summary);
            Ok(TrimVarDctPreparationCache());
            SetEnvironment("GJXL_EARLY_ENTROPY", nullptr);
            Ok(ComputeWorkflowStoragePlan(extent, planning, &enabled_plan));
            Require(enabled_plan.working == disabled_plan.working,
                    "Readiness switch changed the conservative admission bound");
            const size_t limit = enabled_plan.working.peak_bytes;
            std::shared_ptr<const ExecutionDomain> domain;
            Ok(ExecutionDomain::Create({limit, cpu}, &domain));
            options.execution_domain = domain;
            for (size_t repeat = 0; repeat < 2; ++repeat) {
              std::vector<uint8_t> bytes{19};
              VarDctEncodingSummary summary;
              encode(&bytes, &summary);
              Require(bytes == expected && summary == expected_summary,
                      "Bounded early entropy changed bytes or summary");
              const auto s = domain->snapshot();
              Require(s.peak_committed_bytes <= limit &&
                      s.peak_cpu_protected_slots <= cpu &&
                      !s.active_reservations && !s.waiting_requests &&
                      !s.active_cpu_participants && !s.reserved_cpu_workers &&
                      !s.suspended_cpu_workers && !s.waiting_cpu_callers,
                      "Early entropy exceeded admission or leaked participants");
            }
            Ok(WorkflowAdmission::TrimIdle(*domain));
            const auto empty = domain->snapshot();
            Require(!empty.live_capacity_bytes && !empty.idle_capacity_bytes &&
                    !empty.reserved_unbacked_bytes,
                    "Readiness workflow retained managed backing after trim");
            ++cases;
          }
  }
  }
  SetEnvironment("GJXL_EARLY_DC", nullptr);
  std::cout << "Passed " << cases << " bounded default/opt-out/forced-on DC cases: "
      "ordinary/timing APIs, fully-resident/throughput, DC boundary, "
      "CPU 1/2/4, exact planned memory caps, repeated calls and byte/summary parity.\n";
}
}  // namespace

int main() try {
  CheckBoundedDefault();
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
