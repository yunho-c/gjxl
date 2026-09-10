// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#include "codestream/batch_workflow_test.h"
#include "codestream/workflow_lifetime_test.h"
#include "core/image_buffer.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using Clock = std::chrono::steady_clock;
void Require(bool good, const char* message) {
  if (!good) { std::cerr << message << '\n'; std::exit(EXIT_FAILURE); }
}
void Ok(const Status& status) {
  if (!status.ok()) { std::cerr << status.message() << '\n'; std::exit(EXIT_FAILURE); }
}
uint64_t Ns(Clock::duration value) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(value).count());
}
template<class Predicate> void Until(Predicate predicate, const char* message) {
  const auto deadline = Clock::now() + std::chrono::seconds(10);
  while (!predicate()) {
    Require(Clock::now() < deadline, message);
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
            "Timing gate was not released");
  }
  void Release() {
    std::lock_guard lock(mutex);
    released = true;
    changed.notify_all();
  }
};
struct Fixture {
  Image3FBuffer image{{33, 25}};
  VarDctEncodingOptions options;
  explicit Fixture(bool metal) {
    for (size_t c = 0; c < 3; ++c)
      for (size_t i = 0; i < image.plane(c).size(); ++i)
        image.plane(c)[i] = 0.05f + 0.8f * ((i * (c + 3)) % 127) / 127.0f;
    Ok(ExecutionDomain::Create({0, 1}, &options.execution_domain));
    options.backend = metal ? VarDctBackendPreference::kMetal : VarDctBackendPreference::kCpu;
    options.effort = 1;
  }
  VarDctBatchEncodingRequest Request() const { return {image.const_view(), options}; }
};
void CheckTiming(const VarDctBatchSchedulingTiming& timing, uint64_t wall, bool admitted) {
  Require(timing.cpu_admitted == admitted && timing.ready_nanoseconds > 0 &&
          timing.queue_nanoseconds + timing.service_nanoseconds == timing.ready_nanoseconds &&
          timing.ready_nanoseconds <= wall &&
          (admitted ? timing.service_nanoseconds > 0 : timing.service_nanoseconds == 0),
          "Batch queue/service spans do not respect their declared boundaries");
}

struct Boundaries {
  Gate service, worker_exit, publication;
  std::array<VarDctBatchSchedulingTiming, 3> observed;
  void Install() {
    batch_execution_observer_for_testing = {this, +[](void* context, size_t index, bool entering) noexcept {
      auto& boundaries = *static_cast<Boundaries*>(context);
      if (entering && index == 0) {
        (void)SetWorkflowLifetimeObserverForTest({+[](bool, bool, void* p) noexcept {
          static_cast<Boundaries*>(p)->service.Wait();
        }, context});
      } else if (!entering) {
        (void)SetWorkflowLifetimeObserverForTest({});
        if (index == 0) boundaries.worker_exit.Wait();
      }
    }};
    batch_publication_observer_for_testing = {this,
      +[](void* context, std::span<const VarDctBatchEncodingResult> results,
          std::span<const OwnedEncodingResult>) noexcept {
        auto& boundaries = *static_cast<Boundaries*>(context);
        Require(results.size() == boundaries.observed.size(), "Publication changed request count");
        for (size_t i = 0; i < results.size(); ++i) boundaries.observed[i] = results[i].scheduling;
        boundaries.publication.Wait();
      }};
  }
};

void DriverQueueAndPublication(bool metal) {
  Fixture fixture(metal);
  std::vector<uint8_t> expected;
  Ok(EncodeLinearRgbVarDctCodestream(fixture.image.const_view(), fixture.options, &expected));
  std::unique_ptr<VarDctBatchEncoder> driver;
  Ok(VarDctBatchEncoder::Create(1, &driver));
  const std::array first_requests{fixture.Request(), VarDctBatchEncodingRequest{{}, fixture.options}, fixture.Request()};
  const std::array second_requests{fixture.Request()};
  Boundaries boundaries;
  std::vector<VarDctBatchEncodingResult> first(1), second;
  first[0].codestream = {123};
  const void* first_address = first.data();
  std::atomic<bool> first_returned{false}, queued{false};
  Clock::time_point queued_at;
  uint64_t first_wall = 0, second_wall = 0;
  std::thread active([&] {
    boundaries.Install();
    const auto begin = Clock::now();
    Ok(driver->Encode(first_requests, &first));
    first_wall = Ns(Clock::now() - begin);
    first_returned = true;
  });
  Until([&] { return boundaries.service.entered.load(); }, "Image did not reach service gate");
  const auto service_held_begin = Clock::now();
  struct QueueObserver { std::atomic<bool>* entered; Clock::time_point* at; } queue_observer{&queued, &queued_at};
  std::thread waiting([&] {
    batch_lifecycle_observer_for_testing = {&queue_observer, +[](void* context, BatchLifecycleEventForTesting event) noexcept {
      if (event == BatchLifecycleEventForTesting::kWaitingForDriver) {
        auto& observer = *static_cast<QueueObserver*>(context);
        *observer.at = Clock::now();
        *observer.entered = true;
      }
    }};
    const auto begin = Clock::now();
    Ok(driver->Encode(second_requests, &second));
    second_wall = Ns(Clock::now() - begin);
  });
  Until([&] { return queued.load(); }, "Second call did not reach driver queue");
  const uint64_t held_service = Ns(Clock::now() - service_held_begin);
  boundaries.service.Release();
  Until([&] { return boundaries.worker_exit.entered.load(); }, "First result did not become ready");
  boundaries.worker_exit.Release();
  Until([&] { return boundaries.publication.entered.load(); }, "Batch did not reach publication gate");
  Require(!first_returned && first.data() == first_address && first.size() == 1 && first[0].codestream == std::vector<uint8_t>{123},
          "Internal result readiness prematurely published caller output");
  const uint64_t held_driver_queue = Ns(Clock::now() - queued_at);
  boundaries.publication.Release();
  active.join(); waiting.join();
  Require(first.size() == 3 && second.size() == 1, "Timed batch lost result order");
  for (size_t i = 0; i < first.size(); ++i) {
    CheckTiming(first[i].scheduling, first_wall, i != 1);
    Require(first[i].scheduling.ready_nanoseconds == boundaries.observed[i].ready_nanoseconds &&
            first[i].scheduling.queue_nanoseconds == boundaries.observed[i].queue_nanoseconds,
            "Publication relabeled internal readiness as public availability");
    if (i != 1) {
      Ok(first[i].status);
      Require(first[i].codestream == expected, "Timing changed encoded bytes");
    } else Require(first[i].status.code() == StatusCode::kInvalidArgument, "Preflight error changed");
  }
  CheckTiming(second[0].scheduling, second_wall, true);
  Ok(second[0].status);
  Require(second[0].codestream == expected && second[0].scheduling.queue_nanoseconds >= held_driver_queue &&
          first[0].scheduling.service_nanoseconds >= held_service &&
          first[2].scheduling.queue_nanoseconds >= first[0].scheduling.ready_nanoseconds,
          "Timing omitted driver queue, service or in-flight work-slot waiting");
  driver->Shutdown();
}

void FailureAfterAdmission() {
  Fixture fixture(false);
  std::unique_ptr<VarDctBatchEncoder> driver;
  Ok(VarDctBatchEncoder::Create(1, &driver));
  const std::array requests{fixture.Request(), fixture.Request()};
  batch_execution_observer_for_testing = {nullptr, +[](void*, size_t index, bool entering) noexcept {
    if (index == 0 && entering) resource_budget_internal::ArmNextManagedHostAllocationFailureForTest();
    if (!entering) resource_budget_internal::DisarmManagedHostAllocationFailureForTest();
  }};
  const auto begin = Clock::now();
  std::vector<VarDctBatchEncodingResult> results;
  Ok(driver->Encode(requests, &results));
  const auto wall = Ns(Clock::now() - begin);
  batch_execution_observer_for_testing = {};
  Require(results.size() == 2 && results[0].status.code() == StatusCode::kOutOfMemory &&
          results[0].codestream.empty() && results[0].timing.total_nanoseconds == 0 && results[1].status.ok(),
          "Post-admission allocation failure was not isolated");
  CheckTiming(results[0].scheduling, wall, true);
  CheckTiming(results[1].scheduling, wall, true);
}
} // namespace

int main(int argc, char** argv) {
  DriverQueueAndPublication(false);
  if (!(argc == 2 && std::string_view(argv[1]) == "--cpu-only")) DriverQueueAndPublication(true);
  FailureAfterAdmission();
  std::cout << "Batch queue/service boundary checks passed\n";
}
