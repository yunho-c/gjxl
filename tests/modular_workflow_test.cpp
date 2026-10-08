// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/storage_plan.h"
#include "codestream/modular/workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "codestream/workflow_admission_test.h"
#include "core/cpu_execution.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void Empty(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(s.live_capacity_bytes == 0 && s.reserved_unbacked_bytes == 0 &&
            s.active_reservations == 0 && s.active_cpu_participants == 0,
        "Workflow leaked backing, admission or CPU participation");
}
void Validation() {
  std::array<uint8_t, 64> bytes{};
  const Rgb8View valid{std::span(bytes).subspan(3), {3, 4}, 13};
  Ok(valid.Validate());
  std::vector<uint8_t> output{7, 8, 9};
  for (const Rgb8View bad :
       {Rgb8View{{}, {3, 4}, 13}, Rgb8View{bytes, {0, 1}, 3}, Rgb8View{bytes, {3, 4}, 8},
        Rgb8View{std::span(bytes).first(47), {3, 4}, 13}, Rgb8View{bytes, {3, 4}, SIZE_MAX},
        Rgb8View{bytes, {SIZE_MAX, 1}, SIZE_MAX}}) {
    ArmNextManagedHostAllocationFailureForTest();
    Check(!EncodeRgb8Modular(bad, {}, &output).ok(), "Invalid view accepted");
    Check(ManagedHostAllocationFailurePendingForTest(), "Invalid input allocated backing");
    DisarmManagedHostAllocationFailureForTest();
    Check(output == std::vector<uint8_t>({7, 8, 9}), "Invalid input changed output");
  }
  Check(!EncodeRgb8Modular(valid, {}, nullptr).ok(), "Null output accepted");
  Check(!EncodeRgb8Modular(valid, {.entropy = static_cast<EntropyCodingMode>(255)}, &output).ok(),
        "Invalid entropy mode accepted");
  for (const auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularWorkflowStoragePlan plan;
    ArmNextManagedHostAllocationFailureForTest();
    Ok(ComputeModularWorkflowStoragePlan({257, 259}, mode, &plan));
    Check(ManagedHostAllocationFailurePendingForTest(), "Workflow planner allocated backing");
    DisarmManagedHostAllocationFailureForTest();
    const auto prior = plan;
    Check(!ComputeModularWorkflowStoragePlan({SIZE_MAX, 2}, mode, &plan).ok() && plan == prior,
          "Overflow mutated plan");
  }
}
void SharedDomain() {
  const Extent2D extent{17, 19};
  std::vector<uint8_t> pixels(17 * 19 * 3, 117);
  Rgb8View input{pixels, extent, 51};
  ModularWorkflowStoragePlan plan;
  Ok(ComputeModularWorkflowStoragePlan(extent, EntropyCodingMode::kPrefix, &plan));
  std::shared_ptr<const ExecutionDomain> domain, other;
  Ok(ExecutionDomain::Create(
      {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 1}, &domain));
  Ok(ExecutionDomain::Create({}, &other));
  std::vector<uint8_t> first, second;
  Status a, b;
  std::thread worker([&] { a = EncodeRgb8Modular(input, {domain}, &first); });
  b = EncodeRgb8Modular(input, {domain}, &second);
  worker.join();
  Ok(a);
  Ok(b);
  Check(first == second && domain->snapshot().peak_cpu_participants == 1 &&
            domain->snapshot().peak_committed_bytes <= plan.working.peak_bytes,
        "Simultaneous calls bypassed their shared domain");
  Empty(*domain);
  {
    codestream_internal::WorkflowAdmission outer;
    Ok(outer.Start(plan.working.peak_bytes, domain));
    thread_budget_internal::CpuExecutionScope participant;
    Ok(participant.Start(domain, 1));
    Ok(EncodeRgb8Modular(input, {domain}, &second));
    const auto saved = second;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = EncodeRgb8Modular(input, {other}, &second);
    Check(status.code() == StatusCode::kInvalidArgument && second == saved &&
              ManagedHostAllocationFailurePendingForTest(),
          "Foreign nested domain bypassed admission");
    DisarmManagedHostAllocationFailureForTest();
    Check(domain->snapshot().active_cpu_participants == 1,
          "Nested workflow released its outer CPU participant");
  }
  Empty(*domain);
  Empty(*other);
}
void Workflow(Extent2D extent, EntropyCodingMode mode, bool faults) {
  const size_t stride = extent.width * 3 + 7;
  std::vector<uint8_t> bytes(5 + stride * extent.height, 0xcd);
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width * 3; ++x)
      bytes[5 + y * stride + x] = static_cast<uint8_t>(x * 37 + y * 71);
  const auto unchanged = bytes;
  Rgb8View view{std::span(bytes).subspan(5), extent, stride};
  ModularWorkflowStoragePlan plan;
  Ok(ComputeModularWorkflowStoragePlan(extent, mode, &plan));
  std::shared_ptr<const ExecutionDomain> domain;
  // Allow a retained result to coexist with a fresh complete reservation.
  Ok(ExecutionDomain::Create(
      {.managed_memory_bytes = plan.working.peak_bytes + plan.output.peak_bytes,
       .cpu_participant_limit = 1},
      &domain));
  ModularEncodingOptions options{domain, mode};
  std::vector<uint8_t> expected;
  Ok(EncodeRgb8Modular(view, options, &expected));
  Empty(*domain);
  Check(domain->snapshot().peak_backing_bytes <= plan.working.peak_bytes &&
            domain->snapshot().peak_cpu_participants == 1,
        "Workflow exceeded planned envelope");
  Check(expected.size() <= plan.maximum_codestream_bytes, "Output exceeded planned size");
  codestream_internal::CodestreamBuffer retained;
  Ok(EncodeRgb8ModularOwned(view, options, &retained));
  Check(std::ranges::equal(retained.view(), expected) &&
            domain->snapshot().live_capacity_bytes == retained.capacity(),
        "Owned output lost its charge");
  Ok(EncodeRgb8ModularOwned(view, options, &retained));
  Check(std::ranges::equal(retained.view(), expected), "Owned replacement changed bytes");
  ArmNextManagedHostAllocationFailureForTest();
  Check(!EncodeRgb8ModularOwned(view, options, &retained).ok() &&
            std::ranges::equal(retained.view(), expected),
        "Failed owned replacement changed output");
  DisarmManagedHostAllocationFailureForTest();
  std::thread destroy([value = std::move(retained)]() mutable { value.Reset(); });
  destroy.join();
  Empty(*domain);
  std::vector<uint8_t> output{7, 8, 9};
  codestream_internal::ArmNextWorkflowAdmissionCapacityForTest(1);
  auto status = EncodeRgb8Modular(view, options, &output);
  codestream_internal::DisarmWorkflowAdmissionCapacityForTest();
  Check(status.resource_plan_exceeded() && output == std::vector<uint8_t>({7, 8, 9}),
        "Underplanning was not terminal and atomic");
  Empty(*domain);
  std::shared_ptr<const ExecutionDomain> small;
  Ok(ExecutionDomain::Create({.managed_memory_bytes = plan.working.peak_bytes - 1}, &small));
  ArmNextManagedHostAllocationFailureForTest();
  Check(!EncodeRgb8Modular(view, {small, mode}, &output).ok() &&
            ManagedHostAllocationFailurePendingForTest(),
        "Insufficient admission allocated backing");
  DisarmManagedHostAllocationFailureForTest();
  Empty(*small);
  if (faults) {
    size_t failure_count = 0;
    for (; failure_count < 10000; ++failure_count) {
      ArmManagedHostAllocationFailureAfterForTest(failure_count);
      status = EncodeRgb8Modular(view, options, &output);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      Empty(*domain);
      if (status.ok()) {
        Check(pending && output == expected, "Fault injection did not cover every allocation");
        break;
      }
      Check(!pending && status.code() == StatusCode::kOutOfMemory &&
                output == std::vector<uint8_t>({7, 8, 9}),
            "Allocation failure changed output or status");
    }
    Check(failure_count > 10 && failure_count < 10000, "Allocation sweep did not finish");
    std::cout << "allocation failures " << (mode == EntropyCodingMode::kPrefix ? "prefix " : "ANS ")
              << extent.width << 'x' << extent.height << ": " << failure_count << '\n';
  }
  Ok(EncodeRgb8Modular(view, options, &output));
  Check(output == expected && bytes == unchanged,
        "Recovery, determinism or input immutability failed");
  Empty(*domain);
}
} // namespace
int main() try {
  Validation();
  SharedDomain();
  for (const auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    Workflow({1, 1}, mode, true);
    Workflow({17, 19}, mode, true);
    Workflow({257, 3}, mode, true);
    Workflow({2049, 1}, mode, false);
  }
  std::cout << "Modular workflow contracts passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
