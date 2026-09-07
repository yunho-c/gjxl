// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

#include "codestream/batch_workflow_test.h"
#include "codestream/workflow_admission.h"
#include "core/cpu_execution.h"
#include "core/image_buffer.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::thread_budget_internal;
using Event = BatchLifecycleEventForTesting;
void Require(bool good, const char* message) {
  if (!good) { std::cerr << message << '\n'; std::exit(EXIT_FAILURE); }
}
void Ok(const Status& status) {
  if (!status.ok()) { std::cerr << status.message() << '\n'; std::exit(EXIT_FAILURE); }
}
template<class Predicate> void Until(Predicate predicate, const char* message) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!predicate()) {
    Require(std::chrono::steady_clock::now() < deadline, message);
    std::this_thread::yield();
  }
}
struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  std::atomic<bool> entered{false};
  bool released = false;
  void Wait() {
    std::unique_lock lock(mutex);
    entered = true;
    Require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return released; }),
            "Test gate was not released");
  }
  void Release() {
    std::lock_guard lock(mutex);
    released = true;
    changed.notify_all();
  }
};
struct Events {
  std::array<std::atomic<size_t>, 4> count{};
  void Install() {
    batch_lifecycle_observer_for_testing = {this, +[](void* context, Event event) noexcept {
      ++static_cast<Events*>(context)->count[static_cast<size_t>(event)];
    }};
  }
  size_t Get(Event event) const { return count[static_cast<size_t>(event)].load(); }
};
std::vector<VarDctBatchEncodingResult> Sentinel() {
  std::vector<VarDctBatchEncodingResult> result(1);
  result[0].codestream = {0xde, 0xad};
  result[0].summary.encoded_bytes = 123;
  result[0].timing.total_nanoseconds = 456;
  return result;
}
void Unchanged(const std::vector<VarDctBatchEncodingResult>& result, const void* address) {
  Require(result.data() == address && result.size() == 1 &&
          result[0].codestream == std::vector<uint8_t>({0xde, 0xad}) &&
          result[0].summary.encoded_bytes == 123 && result[0].timing.total_nanoseconds == 456,
          "Rejected batch changed caller-owned results");
}
void Empty(const ExecutionDomain& domain) {
  Ok(WorkflowAdmission::TrimIdle(domain));
  const auto s = domain.snapshot();
  Require(s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 &&
          s.suspended_cpu_workers == 0 && s.waiting_cpu_callers == 0 &&
          s.active_reservations == 0 && s.waiting_requests == 0 &&
          s.live_capacity_bytes == 0 && s.idle_capacity_bytes == 0 &&
          s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit &&
          (domain.options().managed_memory_bytes == 0 ||
           s.peak_committed_bytes <= domain.options().managed_memory_bytes),
          "Drained driver leaked or exceeded CPU/memory capacity");
}
struct Fixture {
  Image3FBuffer image{{33, 25}};
  VarDctEncodingOptions options;
  std::shared_ptr<const ExecutionDomain> domain;
  explicit Fixture(bool metal, size_t memory = 0) {
    for (size_t c = 0; c < 3; ++c)
      for (size_t i = 0; i < image.plane(c).size(); ++i)
        image.plane(c)[i] = 0.05f + 0.8f * ((i * (c + 3)) % 127) / 127.0f;
    Ok(ExecutionDomain::Create({memory, 1}, &domain));
    options.backend = metal ? VarDctBackendPreference::kMetal : VarDctBackendPreference::kCpu;
    options.effort = 1;
    options.execution_domain = domain;
  }
  VarDctBatchEncodingRequest Request() const { return {image.const_view(), options}; }
};

void ActiveAndQueued(bool metal, size_t workers) {
  Fixture fixture(metal);
  std::vector<uint8_t> expected;
  VarDctEncodingSummary summary;
  Ok(EncodeLinearRgbVarDctCodestream(fixture.image.const_view(), fixture.options, &expected, &summary));
  std::unique_ptr<VarDctBatchEncoder> driver;
  Ok(VarDctBatchEncoder::Create(workers, &driver));
  const std::array requests{fixture.Request(), VarDctBatchEncodingRequest{{}, fixture.options},
                            fixture.Request(), fixture.Request()};
  Gate work;
  Events queued_events, closing_events;
  std::vector<VarDctBatchEncodingResult> active_result;
  auto queued_result = Sentinel(), rejected_result = Sentinel();
  const void* queued_address = queued_result.data();
  const void* rejected_address = rejected_result.data();
  Status active_status, queued_status;
  std::atomic<size_t> stopped{0};
  std::thread active([&] {
    batch_execution_observer_for_testing = {&work, +[](void* context, size_t index, bool entering) noexcept {
      if (index == 0 && entering) static_cast<Gate*>(context)->Wait();
    }};
    active_status = driver->Encode(requests, &active_result);
  });
  Until([&] { return work.entered.load(); }, "Active image never reached work gate");
  std::thread queued([&] {
    queued_events.Install();
    queued_status = driver->Encode(requests, &queued_result);
  });
  Until([&] { return queued_events.Get(Event::kWaitingForDriver) == 1; }, "Second call did not queue");
  std::thread first_closer([&] {
    // A shutdown caller already holding the sole CPU slot must yield it to
    // the active batch and recover it only after draining/unlocking.
    CpuExecutionScope scope;
    Ok(scope.Start(fixture.domain));
    closing_events.Install();
    driver->Shutdown();
    Require(HasCpuParticipation(), "Shutdown failed to resume caller participation");
    ++stopped;
  });
  std::thread second_closer([&] {
    closing_events.Install();
    driver->Shutdown();
    ++stopped;
  });
  Until([&] { return closing_events.Get(Event::kClosing) == 2; }, "Shutdown did not close admission");
  Require(stopped == 0 && driver->Encode(requests, &rejected_result).code() == StatusCode::kUnavailable,
          "Shutdown abandoned active work or accepted a new batch");
  Unchanged(rejected_result, rejected_address);
  work.Release();
  active.join(); queued.join(); first_closer.join(); second_closer.join();
  Ok(active_status);
  Require(stopped == 2 && closing_events.Get(Event::kStopped) == 2 &&
          queued_events.Get(Event::kActive) == 0 && queued_status.code() == StatusCode::kUnavailable,
          "Queued batch activated after shutdown or a closer failed to drain");
  Unchanged(queued_result, queued_address);
  Require(active_result.size() == requests.size(), "Drained batch lost results");
  for (size_t i = 0; i < active_result.size(); ++i) {
    if (i == 1) {
      Require(active_result[i].status.code() == StatusCode::kInvalidArgument && active_result[i].codestream.empty(),
              "Draining changed an individual error");
    } else {
      Ok(active_result[i].status);
      Require(active_result[i].codestream == expected && active_result[i].summary == summary,
              "Draining changed bytes, decisions or result order");
    }
  }
  driver->Shutdown();
  Require(driver->max_in_flight() == workers && driver->Encode({}, &rejected_result).code() == StatusCode::kUnavailable,
          "Closed driver reopened on an empty batch");
  Unchanged(rejected_result, rejected_address);
  driver.reset(); // All external calls were joined before object destruction.
  Empty(*fixture.domain);
}

void WaitingForAdmission(bool memory) {
  Fixture fixture(false);
  WorkflowStoragePlan image_plan;
  Ok(PlanWorkflowAdmission(fixture.image.extent(),
      {fixture.options, WorkflowStorageRoute::kCpu, WorkflowStorageAdapter::kBorrowedLinearRgb, true},
      nullptr, false, true, &image_plan));
  BatchWorkflowStorageAccumulator accumulator;
  Ok(accumulator.AddRequest(&image_plan));
  BatchWorkflowStoragePlan plan;
  Ok(accumulator.Finish(1, 0, &plan));
  if (memory) {
    Ok(ExecutionDomain::Create({plan.working.peak_bytes, 1}, &fixture.domain));
    fixture.options.execution_domain = fixture.domain;
  }
  std::optional<WorkflowAdmission> held_memory;
  std::optional<CpuExecutionScope> held_cpu;
  if (memory) Ok(held_memory.emplace().Start(plan.working.peak_bytes, fixture.domain));
  else Ok(held_cpu.emplace().Start(fixture.domain));
  std::unique_ptr<VarDctBatchEncoder> driver;
  Ok(VarDctBatchEncoder::Create(1, &driver));
  const std::array requests{fixture.Request()};
  std::vector<VarDctBatchEncodingResult> result;
  Status status;
  Events events;
  std::atomic<bool> stopped{false};
  std::thread active([&] { status = driver->Encode(requests, &result); });
  Until([&] {
    const auto s = fixture.domain->snapshot();
    return memory ? s.waiting_requests == 1 : s.waiting_cpu_callers == 1;
  }, "Active batch did not wait for resource admission");
  std::thread closer([&] { events.Install(); driver->Shutdown(); stopped = true; });
  Until([&] { return events.Get(Event::kClosing) == 1; }, "Admission-wait shutdown did not close");
  Require(!stopped, "Shutdown returned without draining an admission-waiting call");
  held_cpu.reset(); held_memory.reset();
  active.join(); closer.join();
  Ok(status);
  Require(stopped && result.size() == 1 && result[0].status.ok(), "Admission-waiting batch failed to drain");
  driver.reset();
  Empty(*fixture.domain);
}

void FailureDuringDrain() {
  Fixture fixture(false);
  std::unique_ptr<VarDctBatchEncoder> driver;
  Ok(VarDctBatchEncoder::Create(2, &driver));
  const std::array requests{fixture.Request()};
  auto result = Sentinel();
  const void* address = result.data();
  Gate active_gate;
  Events closing_events;
  Status status;
  std::thread active([&] {
    batch_lifecycle_observer_for_testing = {&active_gate, +[](void* context, Event event) noexcept {
      if (event == Event::kActive) static_cast<Gate*>(context)->Wait();
    }};
    ArmNextWorkflowAdmissionCapacityForTest(1);
    status = driver->Encode(requests, &result);
    DisarmWorkflowAdmissionCapacityForTest();
  });
  Until([&] { return active_gate.entered.load(); }, "Failing batch never became active");
  std::thread closer([&] { closing_events.Install(); driver->Shutdown(); });
  Until([&] { return closing_events.Get(Event::kClosing) == 1; }, "Failure drain did not close admission");
  active_gate.Release();
  active.join(); closer.join();
  Require(status.resource_plan_exceeded() && closing_events.Get(Event::kStopped) == 1,
          "Exceptional active call failed to release its driver for shutdown");
  Unchanged(result, address);
  driver.reset();
  Empty(*fixture.domain);
}
} // namespace

int main(int argc, char** argv) {
  const bool cpu_only = argc == 2 && std::string_view(argv[1]) == "--cpu-only";
  for (size_t workers : {1, 4}) {
    ActiveAndQueued(false, workers);
    if (!cpu_only) ActiveAndQueued(true, workers);
  }
  WaitingForAdmission(false);
  WaitingForAdmission(true);
  FailureDuringDrain();
  std::cout << "Batch shutdown/drain checks passed\n";
}
