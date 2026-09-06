// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/batch_workflow.h"

#include "codestream/workflow_internal.h"
#include "codestream/batch_workflow_test.h"

#include <atomic>
#include <array>
#include <condition_variable>
#include <exception>
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

void EncodeOne(
  const VarDctBatchEncodingRequest& request,
  VarDctBatchEncodingResult* result,
  OwnedEncodingResult* owned) noexcept {

  VarDctBatchEncodingResult candidate;
  OwnedEncodingResult candidate_owned;
  try {
    candidate.status = codestream_internal::EncodeLinearRgbVarDctCodestreamOwned(
      request.linear_rgb,
      request.options,
      &candidate_owned.codestream,
      &candidate_owned.summary,
      &candidate_owned.timing);
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
    Stop();
  }

  [[nodiscard]] Status Start() {
    try {
      workers_.reserve(max_in_flight_);
      for (size_t index = 0; index < max_in_flight_; ++index) {
        workers_.emplace_back([this] { WorkerLoop(); });
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

  [[nodiscard]] Status Encode(
    std::span<const VarDctBatchEncodingRequest> requests,
    std::vector<VarDctBatchEncodingResult>* results) {

    if (results == nullptr) {
      return Status::InvalidArgument(
        "Image batch result output is null");
    }

    const resource_budget_internal::ManagedHostScope managed_host(
      resource_budget_internal::ResourceClass::kRetainedResult);
    // Declare escrow before the candidate: rollback frees backing first.
    ManagedVector<std::array<ResourceAllocation, 3>> publication_charges;
    PublicationVector<VarDctBatchEncodingResult> candidate;
    ManagedVector<OwnedEncodingResult> candidate_owned;
    try {
      publication_charges.resize(requests.size());
      candidate_owned.resize(requests.size());
      Status status = PublicationVector<VarDctBatchEncodingResult>::Create(requests.size(), &candidate);
      if (!status.ok()) return status;
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate image batch results");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Image batch contains too many requests");
    }

    std::unique_lock encode_lock(encode_mutex_);
    if (requests.empty()) {
      candidate.PublishTo(results);
      return Status::Ok();
    }

    {
      std::lock_guard work_lock(work_mutex_);
      requests_ = requests;
      results_ = candidate.mutable_view();
      owned_results_ = candidate_owned;
      resource_context_ = resource_budget_internal::CurrentResourceContext();
      next_index_.store(0, std::memory_order_relaxed);
      remaining_workers_ = workers_.size();
      ++generation_;
    }
    work_available_.notify_all();

    {
      std::unique_lock work_lock(work_mutex_);
      work_complete_.wait(work_lock, [this] {
        return remaining_workers_ == 0;
      });
      requests_ = {};
      results_ = {};
      owned_results_ = {};
    }

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

  void WorkerLoop() noexcept {
    size_t observed_generation = 0;
    while (true) {
      std::span<const VarDctBatchEncodingRequest> requests;
      std::span<VarDctBatchEncodingResult> results;
      std::span<OwnedEncodingResult> owned;
      resource_budget_internal::ResourceContext resource_context;
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
        results = results_;
        owned = owned_results_;
        resource_context = resource_context_;
      }

      while (true) {
        const size_t index =
          next_index_.fetch_add(1, std::memory_order_relaxed);
        if (index >= requests.size()) {
          break;
        }
        const resource_budget_internal::ResourceContextScope resources(resource_context);
        EncodeOne(requests[index], &results[index], &owned[index]);
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
  bool stopping_ = false;
  size_t generation_ = 0;
  size_t remaining_workers_ = 0;
  std::span<const VarDctBatchEncodingRequest> requests_;
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

Status VarDctBatchEncoder::Encode(
  std::span<const VarDctBatchEncodingRequest> requests,
  std::vector<VarDctBatchEncodingResult>* results) {

  return impl_->Encode(requests, results);
}

}  // namespace gjxl
