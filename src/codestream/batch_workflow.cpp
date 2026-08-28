// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/batch_workflow.h"
#include "codestream/batch_workflow_internal.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace gjxl {
namespace {

void EncodeOne(
  const VarDctBatchEncodingRequest& request,
  VarDctBatchEncodingResult* result,
  codestream_internal::VarDctBatchEncodingStageProfile* profile,
  size_t worker_index) noexcept {

  VarDctBatchEncodingResult candidate;
  try {
    if (profile == nullptr) {
      candidate.status = EncodeLinearRgbVarDctCodestreamProfiled(
        request.linear_rgb,
        request.options,
        &candidate.codestream,
        &candidate.summary,
        &candidate.timing);
    } else {
      codestream_internal::VarDctEncodingStageProfile candidate_profile;
      candidate.status =
        codestream_internal::EncodeLinearRgbVarDctCodestreamStageProfiled(
          request.linear_rgb,
          request.options,
          &candidate.codestream,
          &candidate.summary,
          &candidate.timing,
          &candidate_profile);
      if (candidate.status.ok()) {
        *profile = {
          .worker_index = worker_index,
          .encoding = candidate_profile,
        };
      }
    }
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
        workers_.emplace_back([this, index] { WorkerLoop(index); });
      }
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
    std::vector<VarDctBatchEncodingResult>* results,
    std::vector<codestream_internal::VarDctBatchEncodingStageProfile>*
      profiles) {

    if (results == nullptr) {
      return Status::InvalidArgument(
        "Image batch result output is null");
    }

    std::vector<VarDctBatchEncodingResult> candidate;
    std::vector<codestream_internal::VarDctBatchEncodingStageProfile>
      candidate_profiles;
    try {
      candidate.resize(requests.size());
      if (profiles != nullptr) {
        candidate_profiles.resize(requests.size());
      }
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate image batch results");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Image batch contains too many requests");
    }

    std::unique_lock encode_lock(encode_mutex_);
    if (requests.empty()) {
      *results = std::move(candidate);
      if (profiles != nullptr) {
        *profiles = std::move(candidate_profiles);
      }
      return Status::Ok();
    }

    {
      std::lock_guard work_lock(work_mutex_);
      requests_ = requests;
      results_ = &candidate;
      profiles_ = profiles == nullptr ? nullptr : &candidate_profiles;
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
      results_ = nullptr;
      profiles_ = nullptr;
    }

    *results = std::move(candidate);
    if (profiles != nullptr) {
      *profiles = std::move(candidate_profiles);
    }
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

  void WorkerLoop(size_t worker_index) noexcept {
    size_t observed_generation = 0;
    while (true) {
      std::span<const VarDctBatchEncodingRequest> requests;
      std::vector<VarDctBatchEncodingResult>* results = nullptr;
      std::vector<codestream_internal::VarDctBatchEncodingStageProfile>*
        profiles = nullptr;
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
        profiles = profiles_;
      }

      while (true) {
        const size_t index =
          next_index_.fetch_add(1, std::memory_order_relaxed);
        if (index >= requests.size()) {
          break;
        }
        EncodeOne(
          requests[index], &(*results)[index],
          profiles == nullptr ? nullptr : &(*profiles)[index], worker_index);
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
  std::vector<VarDctBatchEncodingResult>* results_ = nullptr;
  std::vector<codestream_internal::VarDctBatchEncodingStageProfile>*
    profiles_ = nullptr;
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

  return impl_->Encode(requests, results, nullptr);
}

Status codestream_internal::VarDctBatchProfileAccess::Encode(
  VarDctBatchEncoder& encoder,
  std::span<const VarDctBatchEncodingRequest> requests,
  std::vector<VarDctBatchEncodingResult>* results,
  std::vector<VarDctBatchEncodingStageProfile>* profiles) {

  if (profiles == nullptr) {
    return Status::InvalidArgument(
      "Image batch stage-profile output is null");
  }
  return encoder.impl_->Encode(requests, results, profiles);
}

Status codestream_internal::EncodeVarDctBatchProfiled(
  VarDctBatchEncoder& encoder,
  std::span<const VarDctBatchEncodingRequest> requests,
  std::vector<VarDctBatchEncodingResult>* results,
  std::vector<VarDctBatchEncodingStageProfile>* profiles) {

  return VarDctBatchProfileAccess::Encode(
    encoder, requests, results, profiles);
}

}  // namespace gjxl
