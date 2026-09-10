// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/batch_workflow.h"

#include "codestream/batch_workflow_test.h"
#include "codestream/workflow_admission.h"
#include "codestream/workflow_internal.h"
#include "core/cpu_execution.h"
#include "core/worker_launch_internal.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace gjxl {
using codestream_internal::OwnedEncodingResult;
using resource_budget_internal::PublicationVector;
using resource_budget_internal::ManagedVector;
using resource_budget_internal::ResourceAllocation;
namespace {
using Clock = std::chrono::steady_clock;

VarDctBatchSchedulingTiming SchedulingTiming(
    Clock::time_point arrival, const thread_budget_internal::CpuExecutionScope* cpu = nullptr) noexcept {
  const auto ready_at = Clock::now();
  const auto admitted_at = cpu == nullptr ? std::nullopt : cpu->admitted_at();
  const auto nanoseconds = [](Clock::duration duration) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
  };
  const uint64_t ready = nanoseconds(ready_at - arrival);
  const uint64_t queue = admitted_at ? nanoseconds(*admitted_at - arrival) : ready;
  return {queue, ready - queue, ready, admitted_at.has_value()};
}

void ObserveLifecycle(codestream_internal::BatchLifecycleEventForTesting event) noexcept {
  const auto observer = codestream_internal::batch_lifecycle_observer_for_testing;
  if (observer.observe != nullptr) observer.observe(observer.context, event);
}

Status PlanRequest(const VarDctBatchEncodingRequest &request,
                   codestream_internal::WorkflowStoragePlan *plan) {
  if (!request.linear_rgb.valid())
    return Status::InvalidArgument("VarDCT encoding input or output is invalid");
  return codestream_internal::PlanWorkflowAdmission(
      request.linear_rgb.extent(),
      {request.options, codestream_internal::WorkflowStorageRoute::kCpu,
       codestream_internal::WorkflowStorageAdapter::kBorrowedLinearRgb, true},
      nullptr, false, true, plan);
}

Status PlanBatch(std::span<const VarDctBatchEncodingRequest> requests, size_t workers,
                 std::shared_ptr<const ExecutionDomain> *domain, bool *explicit_domain,
                 codestream_internal::BatchWorkflowStoragePlan *plan) {
  try {
    thread_budget_internal::CpuExecutionScope planning_cpu;
    const Status cpu_status = planning_cpu.Start(requests.empty() ? nullptr :
                                                 requests.front().options.execution_domain);
    if (!cpu_status.ok()) return cpu_status;
    codestream_internal::BatchWorkflowStorageAccumulator accumulator;
    for (const auto &request : requests) {
      auto selected = request.options.execution_domain ? request.options.execution_domain
                                                       : ExecutionDomain::Default();
      if (*domain && domain->get() != selected.get())
        return Status::InvalidArgument("All batch requests must share an execution domain");
      *domain = std::move(selected);
      *explicit_domain |= bool(request.options.execution_domain);
      codestream_internal::WorkflowStoragePlan image_plan;
      const Status status = PlanRequest(request, &image_plan);
      // A shape whose complete plan cannot be represented has no admissible
      // work slot. Ordinary invalid/unavailable requests retain per-image errors.
      if (status.code() == StatusCode::kOutOfMemory)
        return status;
      const Status added = accumulator.AddRequest(status.ok() ? &image_plan : nullptr);
      if (!added.ok())
        return added;
    }
    if (!*domain)
      *domain = ExecutionDomain::Default();
    return accumulator.Finish(workers, (*domain)->options().managed_memory_bytes, plan);
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Unable to plan batch admission");
  }
}

void EncodeOne(
  const VarDctBatchEncodingRequest& request,
  VarDctBatchEncodingResult* result,
  OwnedEncodingResult* owned,
  thread_budget_internal::CpuExecutionScope* cpu_execution) noexcept {

  VarDctBatchEncodingResult candidate;
  OwnedEncodingResult candidate_owned;
  try {
    candidate.status = codestream_internal::EncodeLinearRgbVarDctCodestreamOwned(
      request.linear_rgb,
      request.options,
      &candidate_owned.codestream,
      &candidate_owned.summary,
      &candidate_owned.timing, cpu_execution);
    if (candidate.status.ok()) candidate.status = candidate_owned.Reclassify(
      resource_budget_internal::ResourceClass::kRetainedResult);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    candidate.status = failure.status();
  } catch (const std::bad_alloc&) {
    candidate.status = Status::OutOfMemory(
      "Unable to allocate batch image encoding storage");
  } catch (const std::length_error&) {
    candidate.status = Status::InvalidArgument(
      "Batch image encoding dimensions are too large");
  } catch (const std::exception&) {
    candidate.status = Status::Internal(
      "Batch image encoding failed unexpectedly");
  } catch (...) {
    candidate.status = Status::Internal(
      "Batch image encoding failed with an unknown exception");
  }
  if (!candidate.status.ok()) candidate_owned = {};
  *owned = std::move(candidate_owned);
  *result = std::move(candidate);
}

}  // namespace

class VarDctBatchEncoder::Impl {
public:
  explicit Impl(size_t max_in_flight)
    : max_in_flight_(max_in_flight) {}

  ~Impl() {
    Shutdown();
  }

  [[nodiscard]] Status Start() {
    try {
      workers_.reserve(max_in_flight_);
      for (size_t index = 0; index < max_in_flight_; ++index) {
        thread_budget_internal::LaunchWorker(workers_, thread_budget_internal::WorkerLaunchSite::kBatchDriver,
                                            index, [this, index] { WorkerLoop(index); });
      }
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
      Stop();
      return failure.status();
    } catch (const std::bad_alloc&) {
      Stop();
      return Status::OutOfMemory(
        "Unable to allocate image batch worker storage");
    } catch (const std::length_error&) {
      Stop();
      return Status::InvalidArgument(
        "Image batch maximum in-flight count is too large");
    } catch (const std::system_error&) {
      Stop();
      return Status::Internal(
        "Unable to start image batch workers");
    }
    return Status::Ok();
  }

  [[nodiscard]] size_t max_in_flight() const noexcept {
    return max_in_flight_;
  }

  void Shutdown() noexcept {
    closing_.store(true, std::memory_order_release);
    ObserveLifecycle(codestream_internal::BatchLifecycleEventForTesting::kClosing);
    // Yield through the complete drain, and unlock before resuming. An active
    // batch may need this caller's CPU slot to finish its work/publication.
    thread_budget_internal::CpuSuspension suspension;
    std::lock_guard encode_lock(encode_mutex_);
    Stop();
    ObserveLifecycle(codestream_internal::BatchLifecycleEventForTesting::kStopped);
  }

  [[nodiscard]] Status Encode(
    std::span<const VarDctBatchEncodingRequest> requests,
    std::vector<VarDctBatchEncodingResult>* results) {

    const auto arrival = Clock::now();
    if (results == nullptr) {
      return Status::InvalidArgument(
        "Image batch result output is null");
    }

    if (closing_.load(std::memory_order_acquire))
      return Status::Unavailable("Image batch encoder is shut down");
    ObserveLifecycle(codestream_internal::BatchLifecycleEventForTesting::kWaitingForDriver);
    std::unique_lock encode_lock(encode_mutex_, std::defer_lock);
    {
      thread_budget_internal::CpuSuspension suspension;
      encode_lock.lock();
    }
    // This check linearizes activation against shutdown. Do not set stopping_
    // until this active call has returned: workers must not abandon its images.
    if (closing_.load(std::memory_order_acquire))
      return Status::Unavailable("Image batch encoder is shut down");
    ObserveLifecycle(codestream_internal::BatchLifecycleEventForTesting::kActive);
    std::shared_ptr<const ExecutionDomain> domain;
    bool explicit_domain = false;
    codestream_internal::BatchWorkflowStoragePlan batch_plan;
    Status admission_status =
        PlanBatch(requests, max_in_flight_, &domain, &explicit_domain, &batch_plan);
    if (!admission_status.ok())
      return admission_status;
    codestream_internal::WorkflowAdmission admission;
    admission_status =
        admission.Start(batch_plan.working.peak_bytes, explicit_domain ? domain : nullptr);
    if (!admission_status.ok())
      return admission_status;
    const resource_budget_internal::ManagedHostScope managed_host(
      resource_budget_internal::ResourceClass::kRetainedResult);
    thread_budget_internal::CpuExecutionScope cpu_execution;
    admission_status = cpu_execution.Start(domain);
    if (!admission_status.ok()) return admission_status;
    // Declare escrow before the candidate: rollback frees backing first.
    ManagedVector<std::array<ResourceAllocation, 3>> publication_charges;
    PublicationVector<VarDctBatchEncodingResult> candidate;
    ManagedVector<OwnedEncodingResult> candidate_owned;
    try {
      publication_charges.resize(requests.size());
      candidate_owned.resize(requests.size());
      Status status = PublicationVector<VarDctBatchEncodingResult>::Create(requests.size(), &candidate);
      if (!status.ok()) return status;
      for (size_t i = 0; i < requests.size(); ++i) {
        codestream_internal::WorkflowStoragePlan ignored;
        candidate[i].status = PlanRequest(requests[i], &ignored);
        if (candidate[i].status.code() == StatusCode::kOutOfMemory)
          return candidate[i].status;
        if (!candidate[i].status.ok())
          candidate[i].scheduling = SchedulingTiming(arrival);
      }
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate image batch results");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Image batch contains too many requests");
    }

    if (requests.empty()) {
      candidate.PublishTo(results);
      return Status::Ok();
    }

    {
      std::lock_guard work_lock(work_mutex_);
      requests_ = requests;
      arrival_ = arrival;
      results_ = candidate.mutable_view();
      owned_results_ = candidate_owned;
      resource_context_ = resource_budget_internal::CurrentResourceContext();
      in_flight_ = batch_plan.in_flight;
      trim_after_each_image_ = batch_plan.trim_after_each_image;
      domain_ = domain;
      execution_observer_ = codestream_internal::batch_execution_observer_for_testing;
      terminal_index_.store(std::numeric_limits<size_t>::max(), std::memory_order_relaxed);
      next_index_.store(0, std::memory_order_relaxed);
      remaining_workers_ = workers_.size();
      ++generation_;
    }
    work_available_.notify_all();

    {
      // Resume only after releasing work_mutex_: a resumed caller may be
      // queued behind an image worker that still needs that mutex to finish.
      thread_budget_internal::CpuSuspension suspension;
      std::unique_lock work_lock(work_mutex_);
      work_complete_.wait(work_lock, [this] {
        return remaining_workers_ == 0;
      });
      requests_ = {};
      results_ = {};
      owned_results_ = {};
      domain_.reset();
    }

    const size_t terminal = terminal_index_.load(std::memory_order_relaxed);
    if (terminal != std::numeric_limits<size_t>::max())
      return candidate[terminal].status;

    const auto observer = codestream_internal::batch_publication_observer_for_testing;
    if (observer.observe != nullptr)
      observer.observe(observer.context, candidate.view(), candidate_owned);

    // No fallible work after staging starts. Keep every byte charge until the
    // entire public result array, not just an individual vector, is published.
    for (size_t i = 0; i < candidate.size(); ++i) {
      publication_charges[i][0] = candidate_owned[i].codestream.MoveToPublication(&candidate[i].codestream);
      publication_charges[i][1] = candidate_owned[i].summary.MoveToPublication(&candidate[i].summary);
      publication_charges[i][2] = candidate_owned[i].timing.MoveToPublication(&candidate[i].timing);
    }
    candidate.PublishTo(results);
    publication_charges.clear();
    return Status::Ok();
  }

private:
  void Stop() noexcept {
    thread_budget_internal::CpuSuspension suspension;
    {
      std::lock_guard lock(work_mutex_);
      stopping_ = true;
    }
    work_available_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    workers_.clear();
  }

  void WorkerLoop(size_t worker_index) noexcept {
    size_t observed_generation = 0;
    while (true) {
      std::span<const VarDctBatchEncodingRequest> requests;
      std::span<VarDctBatchEncodingResult> results;
      std::span<OwnedEncodingResult> owned;
      Clock::time_point arrival;
      resource_budget_internal::ResourceContext resource_context;
      size_t in_flight = 0;
      bool trim = false;
      std::shared_ptr<const ExecutionDomain> domain;
      codestream_internal::BatchExecutionObserverForTesting observer;
      {
        std::unique_lock lock(work_mutex_);
        work_available_.wait(lock, [&] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_) {
          return;
        }
        observed_generation = generation_;
        requests = requests_;
        arrival = arrival_;
        results = results_;
        owned = owned_results_;
        resource_context = resource_context_;
        in_flight = in_flight_;
        trim = trim_after_each_image_;
        domain = domain_;
        observer = execution_observer_;
      }

      while (worker_index < in_flight && terminal_index_.load(std::memory_order_relaxed) ==
                                             std::numeric_limits<size_t>::max()) {
        const size_t index =
          next_index_.fetch_add(1, std::memory_order_relaxed);
        if (index >= requests.size()) {
          break;
        }
        if (!results[index].status.ok())
          continue;
        const resource_budget_internal::ResourceContextScope resources(resource_context);
        // Started inside EncodeOne after validation/admission; retain it through
        // result ownership transfer and any post-image preparation-cache trim.
        thread_budget_internal::CpuExecutionScope cpu_execution;
        if (observer.observe != nullptr)
          observer.observe(observer.context, index, true);
        EncodeOne(requests[index], &results[index], &owned[index], &cpu_execution);
        bool terminal = results[index].status.resource_plan_exceeded();
        if (trim) {
          Status status;
          try {
            status = codestream_internal::WorkflowAdmission::TrimIdle(*domain);
          } catch (const std::bad_alloc &) {
            status = {StatusCode::kOutOfMemory, {}};
          } catch (...) {
            status = {StatusCode::kInternal, {}};
          }
          if (!status.ok()) {
            results[index].status = std::move(status);
            terminal = true;
          }
        }
        if (terminal) {
          size_t expected = std::numeric_limits<size_t>::max();
          terminal_index_.compare_exchange_strong(expected, index, std::memory_order_relaxed);
        }
        results[index].scheduling = SchedulingTiming(arrival, &cpu_execution);
        if (observer.observe != nullptr)
          observer.observe(observer.context, index, false);
      }

      {
        std::lock_guard lock(work_mutex_);
        if (--remaining_workers_ == 0) {
          work_complete_.notify_one();
        }
      }
    }
  }

  const size_t max_in_flight_;
  std::vector<std::thread> workers_;
  std::mutex encode_mutex_;
  std::mutex work_mutex_;
  std::condition_variable work_available_;
  std::condition_variable work_complete_;
  std::atomic<bool> closing_{false};
  bool stopping_ = false;
  size_t generation_ = 0;
  size_t remaining_workers_ = 0;
  size_t in_flight_ = 0;
  bool trim_after_each_image_ = false;
  std::shared_ptr<const ExecutionDomain> domain_;
  codestream_internal::BatchExecutionObserverForTesting execution_observer_;
  std::atomic<size_t> terminal_index_{std::numeric_limits<size_t>::max()};
  std::span<const VarDctBatchEncodingRequest> requests_;
  Clock::time_point arrival_;
  std::span<VarDctBatchEncodingResult> results_;
  std::span<OwnedEncodingResult> owned_results_;
  resource_budget_internal::ResourceContext resource_context_;
  std::atomic<size_t> next_index_{0};
};

VarDctBatchEncoder::VarDctBatchEncoder(std::unique_ptr<Impl> impl)
  : impl_(std::move(impl)) {}

VarDctBatchEncoder::~VarDctBatchEncoder() = default;

Status VarDctBatchEncoder::Create(
  size_t max_in_flight,
  std::unique_ptr<VarDctBatchEncoder>* encoder) {

  if (encoder == nullptr) {
    return Status::InvalidArgument(
      "Image batch encoder output is null");
  }
  encoder->reset();
  if (max_in_flight == 0) {
    return Status::InvalidArgument(
      "Image batch maximum in-flight count must be positive");
  }

  try {
    auto impl = std::make_unique<Impl>(max_in_flight);
    Status status = impl->Start();
    if (!status.ok()) {
      return status;
    }
    encoder->reset(new VarDctBatchEncoder(std::move(impl)));
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate image batch encoder");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Image batch maximum in-flight count is too large");
  }
}

size_t VarDctBatchEncoder::max_in_flight() const noexcept {
  return impl_->max_in_flight();
}

void VarDctBatchEncoder::Shutdown() noexcept {
  impl_->Shutdown();
}

Status VarDctBatchEncoder::Encode(
  std::span<const VarDctBatchEncodingRequest> requests,
  std::vector<VarDctBatchEncodingResult>* results) {

  return impl_->Encode(requests, results);
}

}  // namespace gjxl
