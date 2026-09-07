// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "codestream/storage.h"
#include "core/parallel_work_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
constexpr auto kSite = WorkerLaunchSite::kColorRows;
constexpr ParallelWorkErrors kErrors{
  .allocation = "task allocation",
  .unexpected = "unexpected task",
  .length_code = StatusCode::kInvalidArgument,
  .length = "task length",
  .launch_allocation = "launch allocation",
  .launch_action = LaunchFailureAction::kReturnError,
  .launch = "launch failure",
};

void Require(bool good, const char* message) {
  if (!good) { std::cerr << message << '\n'; std::exit(EXIT_FAILURE); }
}
template <class Predicate> void Until(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    Require(std::chrono::steady_clock::now() < deadline, "Worker synchronization timed out");
    std::this_thread::yield();
  }
}
void Empty(const ExecutionDomain& domain) {
  const auto s = domain.snapshot();
  Require(s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 &&
          s.suspended_cpu_workers == 0 && s.waiting_cpu_callers == 0 &&
          s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit,
          "Worker group leaked or exceeded CPU participation");
}

template <template <class> class Storage>
void IndicesAndContext(bool managed, size_t limit, ResourceClass owner) {
  std::shared_ptr<const ExecutionDomain> domain;
  Require(ExecutionDomain::Create({0, 4}, &domain).ok(), "Domain creation failed");
  CpuParticipantTracker tracker;
  ResourceBudget budget;
  ResourceReservation reservation;
  Require(budget.Reserve(65536, &reservation).ok(), "Reservation failed");
  {
    CpuExecutionScope cpu;
    // A managed job deliberately admits fewer participants than requested.
    if (managed) Require(cpu.Start(domain, 2).ok(), "CPU admission failed");
    ResourceContextScope resources({&reservation, ResourceClass::kPreparation, true, domain.get()});
    EncodeScope encode(limit, &tracker);
    CpuWorkerGroup group(4);
    const size_t participants = managed ? 2 : 4;
    Require(group.participants() == participants, "Unexpected admitted participant count");
    const bool caller_participates = managed || limit != 0;
    const size_t spawned = participants - size_t(caller_participates);
    const auto caller = std::this_thread::get_id();
    std::atomic<size_t> arrived{0};
    std::array<std::atomic<size_t>, 4> worker_calls{};
    std::array<size_t, 8> task_calls{};
    const Status status = RunParallelWork<Storage>(8, group, spawned, kSite, kErrors,
      [&](size_t task, size_t worker) {
        Require(worker < participants, "Worker index exceeds scratch slots");
        Require((std::this_thread::get_id() == caller) == (caller_participates && worker == spawned),
                "Caller participation or worker index changed");
        const auto context = CurrentResourceContext();
        Require(context.reservation == &reservation && context.domain == domain.get() &&
                context.resource_class == ResourceClass::kPreparation && context.track_host_allocations &&
                CpuThreadCount() == limit && ParticipantTracker() == &tracker &&
                InExplicitParallelScope() == (limit != 0) && HasCpuParticipation() == managed,
                "Worker failed to inherit resource or CPU context");
        const auto usage = budget.snapshot();
        Require(usage.classes[static_cast<size_t>(owner)].backing_count == 2 &&
                usage.total.backing_count == 2, "Runner changed status/thread allocation ownership");
        if (worker_calls[worker].fetch_add(1) == 0) {
          ++arrived;
          Until([&] { return arrived == participants; });
        }
        ++task_calls[task];
        return Status::Ok();
      });
    Require(status.ok() && arrived == participants, "Not all reserved workers ran");
    for (size_t calls : task_calls) Require(calls == 1, "Task skipped or repeated");
    Require(budget.snapshot().total.backing_count == 0 && !InExplicitParallelScope(),
            "Worker backing or nested scope leaked");
  }
  Require(tracker.active() == 0, "Participant observer leaked");
  Empty(*domain);
}

void TaskErrors() {
  EncodeScope encode(2);
  CpuWorkerGroup group(2);
  for (int failure = 0; failure < 5; ++failure) {
    std::atomic<bool> later_ran{false};
    std::atomic<size_t> completed{0};
    const auto status = RunParallelWork<ManagedVector>(8, group, 1, kSite, kErrors,
      [&](size_t task, size_t) -> Status {
        ++completed;
        if (task == 0) {
          // The later error becomes available first. Task order must decide
          // which error is returned, and errors must not cancel other tasks.
          Until([&] { return later_ran.load(); });
          switch (failure) {
            case 0: throw ManagedAllocationFailure(Status::ResourcePlanExceeded("typed task"));
            case 1: throw std::bad_alloc();
            case 2: throw std::length_error("too large");
            case 3: throw 7;
            default: return Status::Unavailable("first task");
          }
        }
        if (task == 1) return Status::Internal("later task");
        if (task == 2) later_ran = true;
        return Status::Ok();
      });
    constexpr std::array codes{StatusCode::kOutOfMemory, StatusCode::kOutOfMemory,
      StatusCode::kInvalidArgument, StatusCode::kInternal, StatusCode::kUnavailable};
    constexpr std::array messages{"typed task", "task allocation", "task length", "unexpected task", "first task"};
    Require(status.code() == codes[failure] && status.message() == messages[failure] &&
            status.resource_plan_exceeded() == (failure == 0) && completed == 8,
            "Task error ordering, classification or completion changed");
  }
  // Setup errors must escape to the component handler with their precise type.
  ResourceBudget budget;
  ResourceReservation reservation;
  Require(budget.Reserve(1, &reservation).ok(), "Small reservation failed");
  ResourceContextScope resources({&reservation, ResourceClass::kPreparation});
  bool caught = false;
  try {
    (void)RunParallelWork<ManagedVector>(8, group, 1, kSite, kErrors,
      [](size_t, size_t) { return Status::Ok(); });
  } catch (const ManagedAllocationFailure& error) {
    caught = error.status().resource_plan_exceeded();
  }
  Require(caught && budget.snapshot().total.backing_count == 0, "Setup flattened a typed failure");
}

void PartialLaunch(LaunchFailureAction action, WorkerLaunchFailureKind kind) {
  std::shared_ptr<const ExecutionDomain> domain;
  Require(ExecutionDomain::Create({0, 3}, &domain).ok(), "Domain creation failed");
  {
    CpuExecutionScope cpu;
    Require(cpu.Start(domain, 3).ok(), "CPU admission failed");
    EncodeScope encode(3);
    CpuWorkerGroup group(3);
    const auto caller = std::this_thread::get_id();
    std::atomic<bool> entered{false}, left{false};
    WorkerLaunchFaultForTesting fault{kSite, 1, kind};
    fault.context = &entered;
    fault.before_failure = +[](void* p) noexcept {
      Until([&] { return static_cast<std::atomic<bool>*>(p)->load(); });
    };
    // Hold worker zero until join suspends the caller. This proves cleanup
    // yields its CPU slot and waits for the started worker before fallback.
    std::array<size_t, 8> retries{};
    auto errors = kErrors;
    errors.launch_action = action;
    WorkerLaunchFaultScopeForTesting inject(&fault);
    const auto status = RunParallelWork<ManagedVector>(8, group, 2, kSite, errors,
      [&](size_t task, size_t worker) {
        if (std::this_thread::get_id() != caller) {
          Require(worker == 0, "Partial launch changed worker index");
          entered = true;
          Until([&] { return domain->snapshot().active_cpu_participants == 1; });
          left = true;
          return Status::Internal("discarded before retry");
        }
        Require(left && HasCpuParticipation() && !InExplicitParallelScope() && worker == 0,
                "Fallback preceded join/resumption or changed serial context");
        ++retries[task];
        return Status::Ok();
      });
    const bool retry = action == LaunchFailureAction::kRetrySerial && kind == WorkerLaunchFailureKind::kSystemError;
    Require(fault.triggered && fault.launched_in_group == 1 && left && HasCpuParticipation(),
            "Partial launch did not complete cleanup");
    Require(retry ? status.ok() : status.code() == (kind == WorkerLaunchFailureKind::kBadAlloc
            ? StatusCode::kOutOfMemory : StatusCode::kInternal), "Launch policy changed");
    for (size_t calls : retries) Require(calls == size_t(retry), "Serial retry range changed");
  }
  Empty(*domain);
}

// Exercise exceptions not generated by the existing production launch hook.
// The test container throws at construction of the second real worker.
enum class ExtraFailure { kTyped, kUnexpected };
inline ExtraFailure extra_failure;
inline std::atomic<bool> extra_entered{false}, extra_left{false};
class ThrowingThreads : public ManagedVector<std::thread> {
public:
  template <class... Args> void emplace_back(Args&&... args) {
    if (size() == 1) {
      Until([] { return extra_entered.load(); });
      if (extra_failure == ExtraFailure::kTyped)
        throw ManagedAllocationFailure(Status::ResourcePlanExceeded("typed launch"));
      throw 42;
    }
    ManagedVector<std::thread>::emplace_back(std::forward<Args>(args)...);
  }
};
template <class T> using ThrowingStorage = std::conditional_t<
  std::is_same_v<T, std::thread>, ThrowingThreads, ManagedVector<T>>;

void ExtraLaunchErrors() {
  std::shared_ptr<const ExecutionDomain> domain;
  Require(ExecutionDomain::Create({0, 3}, &domain).ok(), "Domain creation failed");
  for (auto failure : {ExtraFailure::kTyped, ExtraFailure::kUnexpected}) {
    {
      CpuExecutionScope cpu;
      Require(cpu.Start(domain, 3).ok(), "CPU admission failed");
      EncodeScope encode(3);
      CpuWorkerGroup group(3);
      extra_failure = failure;
      extra_entered = false;
      extra_left = false;
      bool caught = false;
      try {
        const auto status = RunParallelWork<ThrowingStorage>(8, group, 2, kSite, kErrors,
          [&](size_t, size_t) {
            extra_entered = true;
            Until([&] { return domain->snapshot().active_cpu_participants == 1; });
            extra_left = true;
            return Status::Ok();
          });
        Require(failure == ExtraFailure::kTyped && status.resource_plan_exceeded() &&
                status.message() == "typed launch", "Launch flattened a typed failure");
      } catch (int value) {
        caught = value == 42;
      }
      Require(caught == (failure == ExtraFailure::kUnexpected) && extra_left && HasCpuParticipation(),
              "Unexpected launch exception escaped before cleanup");
    }
    Empty(*domain);
  }
}
}  // namespace

int main() {
  for (bool managed : {false, true}) for (size_t limit : {0ul, 4ul}) {
    IndicesAndContext<ManagedVector>(managed, limit, ResourceClass::kPreparation);
    IndicesAndContext<codestream_internal::Storage>(managed, limit, ResourceClass::kSerializer);
  }
  TaskErrors();
  for (auto action : {LaunchFailureAction::kReturnError, LaunchFailureAction::kRetrySerial})
    for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
      PartialLaunch(action, kind);
  ExtraLaunchErrors();
  std::cout << "Parallel worker orchestration passed\n";
}
