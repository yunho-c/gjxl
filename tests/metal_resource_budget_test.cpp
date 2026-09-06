// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"
#include "core/resource_context.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_butteraugli_test.h"
#include "gpu/ops/butteraugli.h"
#include "gpu/ops/resident_input.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;

bool Check(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}
bool Ok(const Status& status) {
  if (!status.ok()) std::cerr << status.message() << '\n';
  return status.ok();
}
ResourceUsage Usage(const ResourceBudget& budget, ResourceClass resource_class) {
  return budget.snapshot().classes[static_cast<size_t>(resource_class)];
}
bool Empty(const ResourceBudget& budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
    s.total.pending_count == 0 && s.open_reservations == 0 &&
    s.waiting_requests == 0, "Metal resource charge leaked");
}
Image3FBuffer Image(Extent2D extent, size_t seed = 0) {
  Image3FBuffer image(extent);
  for (size_t c = 0; c < 3; ++c)
    for (size_t i = 0; i < image.plane(c).size(); ++i)
      image.plane(c)[i] = 0.05f + 0.7f * ((i * (c + 3) + seed) % 127) / 127.0f;
  return image;
}
template <typename Predicate>
bool Until(Predicate&& predicate) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

bool CheckDirectOwnership() {
  auto& shared = DefaultResourceBudget();
  if (!Empty(shared)) return false;
  std::unique_ptr<GpuBackend> first, second;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &first)) ||
      !Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &second)) || !Empty(shared))
    return false;
  std::unique_ptr<DeviceBuffer> a, b;
  if (!Ok(first->Allocate(64, &a)) || !Ok(second->Allocate(128, &b))) return false;
  auto s = shared.snapshot();
  if (!Check(s.total.live_capacity_bytes == 192 && s.total.backing_count == 2 &&
      s.total.pending_count == 0 && s.reserved_unbacked_bytes == 0 &&
      Usage(shared, ResourceClass::kUnclassified).live_capacity_bytes == 192,
      "Independent backends did not share actual-buffer accounting")) return false;
  const DeviceBuffer* alias = a.get();
  if (!Check(alias->size_bytes() == 64 && shared.snapshot().total.backing_count == 2,
      "Borrowing a Metal buffer duplicated its charge")) return false;
  first.reset();
  if (!Check(shared.snapshot().committed_bytes() == 192,
      "Backend destruction uncharged retained buffers")) return false;
  a.reset();
  b.reset();
  second.reset();
  return Empty(shared);
}

bool CheckReservedAllocationFailure() {
  ResourceBudget budget(128);
  ResourceReservation job;
  std::unique_ptr<GpuBackend> gpu;
  std::unique_ptr<DeviceBuffer> a, b;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)) ||
      !Ok(budget.Reserve(128, &job))) return false;
  {
    ResourceContextScope context({&job, ResourceClass::kInput});
    if (!Ok(gpu->Allocate(64, &a))) return false;
    auto* previous = a.get();
    const auto allocations = gpu->stats().successful_allocations;
    if (!Check(gpu->Allocate(80, &a).code() == StatusCode::kOutOfMemory &&
        a.get() == previous && gpu->stats().successful_allocations == allocations,
        "Undersized plan changed output or allocated backing")) return false;
    if (!Ok(ArmNextMetalAllocationFailureForTest(*gpu))) return false;
    if (!Check(gpu->Allocate(16, &a).code() == StatusCode::kOutOfMemory &&
        a.get() == previous && budget.snapshot().total.pending_count == 0 &&
        budget.snapshot().total.live_capacity_bytes == 64 &&
        budget.snapshot().reserved_unbacked_bytes == 64,
        "Backing failure leaked pending credit or changed output")) return false;
    {
      ResourceClassScope completed(ResourceClass::kCompletedFrame);
      if (!Ok(gpu->Allocate(16, &b))) return false;
    }
    if (!Check(CurrentResourceContext().resource_class == ResourceClass::kInput &&
        Usage(budget, ResourceClass::kInput).live_capacity_bytes == 64 &&
        Usage(budget, ResourceClass::kCompletedFrame).live_capacity_bytes == 16,
        "Nested allocation class scope was not restored")) return false;
  }
  if (!Check(CurrentResourceContext().reservation == nullptr,
      "Admission context escaped its scope")) return false;
  job.Reset();
  gpu.reset();
  if (!Check(budget.snapshot().committed_bytes() == 80,
      "Real Metal output lost its independent charge")) return false;
  a.reset();
  b.reset();
  return Empty(budget) && Empty(DefaultResourceBudget());
}

bool PrepareInput(GpuBackend& gpu, const Image3FBuffer& image,
                  std::unique_ptr<PreparedResidentInput>* prepared) {
  return Ok(PrepareResidentInput(gpu,
    {image.const_view(), image.extent(), false}, prepared));
}

bool CheckInputBytes(GpuBackend& gpu, const Image3FBuffer& image,
                     const PreparedResidentInput& prepared) {
  const auto original = prepared.original_linear_rgb();
  const auto coding = prepared.coding_opsin();
  for (size_t c = 0; c < 3; ++c) {
    if (!Check(original.plane[c].buffer == original.plane[0].buffer &&
        coding.plane[c].buffer == original.plane[0].buffer &&
        original.plane[c].row_stride == image.extent().width,
        "Resident input no longer has one shared backing")) return false;
    std::vector<float> copy(image.plane(c).size());
    if (!Ok(gpu.CopyDeviceToHost(*original.plane[c].buffer, copy.data(),
          copy.size() * sizeof(float), original.plane[c].offset_bytes)) ||
        !std::equal(copy.begin(), copy.end(), image.plane(c).begin())) return false;
  }
  return true;
}

bool CheckCacheDomainsAndReclamation() {
  auto& shared = DefaultResourceBudget();
  std::unique_ptr<GpuBackend> gpu;
  std::unique_ptr<PreparedResidentInput> prepared;
  auto image = Image({32, 32});
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)) ||
      !PrepareInput(*gpu, image, &prepared) ||
      !CheckInputBytes(*gpu, image, *prepared)) return false;
  const size_t capacity = prepared->original_linear_rgb().plane[0].buffer->size_bytes();
  if (!Check(Usage(shared, ResourceClass::kInput).live_capacity_bytes == capacity &&
      shared.snapshot().total.backing_count == 1,
      "Resident input slices were counted as separate allocations")) return false;
  prepared.reset();
  if (!Check(Usage(shared, ResourceClass::kInput).idle_capacity_bytes == capacity,
      "Resident input cache did not retain its charge")) return false;
  ResourceBudget a(2 * capacity), b(2 * capacity);
  size_t seed = 1;
  const auto run = [&](ResourceBudget& budget, size_t fresh, bool trim = false) {
    ResourceReservation job;
    if (!Ok(budget.Reserve(capacity, &job))) return false;
    const auto before = gpu->stats().successful_allocations;
    // Caller-owned source is outside this device-only fixture's reservation.
    image = Image({32, 32}, seed++);
    {
      ResourceContextScope context({&job, ResourceClass::kUnclassified});
      if (!PrepareInput(*gpu, image, &prepared) ||
          !CheckInputBytes(*gpu, image, *prepared)) return false;
      if (!Check(gpu->stats().successful_allocations == before + fresh &&
          budget.snapshot().committed_bytes() == capacity &&
          Usage(budget, ResourceClass::kInput).live_capacity_bytes == capacity,
          "Cache acquisition failed to transfer exactly one backing")) return false;
      if (trim && !Ok(gpu->TrimPreparationCache())) return false;
      prepared.reset();
    }
    job.Reset();
    return Check(budget.snapshot().committed_bytes() == (trim ? 0 : capacity) &&
      budget.snapshot().total.live_capacity_bytes == 0,
      "Cache return ignored trim generation or producer lifetime");
  };
  // Default -> explicit, same-domain hit, real Empty recovery, then isolation.
  if (!run(a, 1) || !Empty(shared) || !run(a, 0) ||
      !Ok(metal_internal::EmptyMetalAqScratchArenasForTesting(*gpu)) ||
      !Check(a.snapshot().total.idle_capacity_bytes == capacity,
             "Volatile reclamation pretended to release allocation capacity") ||
      !run(a, 1) || !run(b, 1) || !Empty(a) || !run(b, 0, true) || !Empty(b))
    return false;
  if (!run(a, 1)) return false;
  ResourceBudget tiny(1);
  ResourceReservation too_small;
  if (!Ok(tiny.Reserve(1, &too_small))) return false;
  const auto before = gpu->stats().successful_allocations;
  {
    ResourceContextScope context({&too_small, ResourceClass::kInput});
    const Status status = PrepareResidentInput(*gpu,
      {image.const_view(), image.extent(), false}, &prepared);
    if (!Check(status.code() == StatusCode::kOutOfMemory && !prepared &&
        gpu->stats().successful_allocations == before,
        "Cache miss escaped an insufficient resource plan")) return false;
  }
  too_small.Reset();
  if (!Empty(a) || !Empty(tiny) || !run(b, 1)) return false;
  gpu.reset();
  return Empty(b) && Empty(shared);
}

bool CheckConcurrentDomains() {
  std::unique_ptr<GpuBackend> gpu;
  auto image = Image({32, 32});
  std::unique_ptr<PreparedResidentInput> probe;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)) ||
      !PrepareInput(*gpu, image, &probe)) return false;
  const size_t capacity = probe->original_linear_rgb().plane[0].buffer->size_bytes();
  probe.reset();
  if (!Ok(gpu->TrimPreparationCache())) return false;
  ResourceBudget budget(4 * capacity);
  std::array<ResourceReservation, 4> jobs;
  for (auto& job : jobs) if (!Ok(budget.Reserve(capacity, &job))) return false;
  std::atomic<size_t> ready{0};
  std::atomic<bool> release{false}, good{true};
  std::vector<std::jthread> workers;
  for (size_t i = 0; i < jobs.size(); ++i) {
    workers.emplace_back([&, context = ResourceContext{&jobs[i], ResourceClass::kInput}]
                         (std::stop_token stop) {
      ResourceContextScope scope(context);
      std::unique_ptr<PreparedResidentInput> prepared;
      if (!PrepareInput(*gpu, image, &prepared) ||
          !CheckInputBytes(*gpu, image, *prepared)) good = false;
      ++ready;
      while (!release && !stop.stop_requested()) std::this_thread::yield();
    });
  }
  if (!Check(Until([&] { return ready == jobs.size(); }) && good,
      "Concurrent resident preparation failed")) return false;
  if (!Check(budget.snapshot().total.live_capacity_bytes == 4 * capacity &&
      budget.snapshot().total.backing_count == 4 &&
      budget.snapshot().reserved_unbacked_bytes == 0,
      "Concurrent callers did not share one allowance")) return false;
  if (!Ok(gpu->TrimPreparationCache())) return false;
  release = true;
  for (auto& worker : workers) worker.join();
  for (auto& job : jobs) job.Reset();
  gpu.reset();
  return Empty(budget) && Empty(DefaultResourceBudget());
}

bool CheckButteraugliDomains() {
  std::unique_ptr<GpuBackend> gpu;
  const auto image = Image({32, 32});
  std::unique_ptr<PreparedResidentInput> input;
  std::unique_ptr<PreparedDeviceButteraugli> prepared;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)) ||
      !PrepareInput(*gpu, image, &input)) return false;
  const DeviceButteraugliPrepareDescriptor descriptor{input->original_linear_rgb(), {}};
  if (!Ok(PrepareDeviceButteraugli(*gpu, descriptor, &prepared))) return false;
  const size_t capacity = prepared->memory_stats().prepared_allocation_bytes;
  prepared.reset();
  ResourceBudget a(3 * capacity), b(3 * capacity);
  const auto run = [&](ResourceBudget& budget, size_t fresh, bool trim = false) {
    ResourceReservation job;
    if (!Ok(budget.Reserve(capacity, &job))) return false;
    const auto before = gpu->stats().successful_allocations;
    {
      ResourceContextScope context({&job, ResourceClass::kUnclassified});
      if (!Ok(PrepareDeviceButteraugli(*gpu, descriptor, &prepared))) return false;
      if (!Check(gpu->stats().successful_allocations == before + fresh &&
          Usage(budget, ResourceClass::kButteraugli).live_capacity_bytes == capacity &&
          budget.snapshot().committed_bytes() == capacity,
          "Butteraugli cache did not transfer its backing charge")) return false;
      if (trim && !Ok(gpu->TrimPreparationCache())) return false;
      prepared.reset();
    }
    job.Reset();
    return Check(budget.snapshot().committed_bytes() == (trim ? 0 : capacity) &&
      budget.snapshot().total.live_capacity_bytes == 0,
      "Butteraugli cache return ignored its resource domain or trim");
  };
  if (!run(a, 1) || !run(a, 0) || !run(b, 1) || !Empty(a) ||
      !Ok(EmptyMetalButteraugliCacheForTesting(*gpu)) ||
      !Check(b.snapshot().total.idle_capacity_bytes == capacity,
             "Reclaimed Butteraugli backing lost its capacity charge") ||
      !run(b, 1) || !run(b, 0, true) || !Empty(b) || !run(b, 1)) return false;
  ResourceReservation failed;
  if (!Ok(a.Reserve(capacity, &failed)) ||
      !Ok(ArmNextMetalAllocationFailureForTest(*gpu))) return false;
  {
    ResourceContextScope context({&failed, ResourceClass::kButteraugli});
    const auto status = PrepareDeviceButteraugli(*gpu, descriptor, &prepared);
    if (!Check(status.code() == StatusCode::kOutOfMemory && !prepared &&
        a.snapshot().total.pending_count == 0,
        "Butteraugli failed preparation retained a pending allocation")) return false;
  }
  failed.Reset();
  if (!Empty(a) || !Empty(b)) return false;
  input.reset();
  if (!Ok(gpu->TrimPreparationCache())) return false;
  gpu.reset();
  return Empty(DefaultResourceBudget());
}

bool CheckWorkflowAccountingAndFailure() {
  auto& budget = DefaultResourceBudget();
  std::unique_ptr<GpuBackend> gpu;
  const auto image = Image({129, 127});
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu))) return false;
  VarDctEncodingOptions options;
  options.backend = VarDctBackendPreference::kMetal;
  options.butteraugli_target = 1.2f;
  std::vector<uint8_t> expected;
  if (!Ok(codestream_internal::EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image.const_view(), options, gpu.get(), true, &expected))) return false;
  auto s = budget.snapshot();
  if (!Check(s.total.live_capacity_bytes == 0 && s.total.idle_capacity_bytes != 0 &&
      s.total.pending_count == 0 && s.reserved_unbacked_bytes == 0 &&
      Usage(budget, ResourceClass::kInput).idle_capacity_bytes != 0 &&
      Usage(budget, ResourceClass::kAqScratch).idle_capacity_bytes != 0 &&
      Usage(budget, ResourceClass::kButteraugli).idle_capacity_bytes != 0 &&
      Usage(budget, ResourceClass::kUnclassified).backing_count == 0,
      "Resident workflow left unclassified/live backing after serialization")) {
    for (size_t i = 0; i < s.classes.size(); ++i) {
      const auto& usage = s.classes[i];
      std::cerr << "Resource class " << i << ": live=" << usage.live_capacity_bytes
                << " idle=" << usage.idle_capacity_bytes
                << " pending=" << usage.pending_capacity_bytes
                << " backings=" << usage.backing_count << '\n';
    }
    return false;
  }
  if (!Ok(gpu->TrimPreparationCache()) || !Empty(budget) ||
      !Ok(ArmNextMetalAllocationFailureForTest(*gpu))) return false;
  std::vector<uint8_t> output{1, 2, 3};
  const auto before = gpu->stats();
  const auto status = codestream_internal::EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
    image.const_view(), options, gpu.get(), true, &output);
  if (!Check(status.code() == StatusCode::kOutOfMemory &&
      output == std::vector<uint8_t>({1, 2, 3}) &&
      gpu->stats().successful_allocations == before.successful_allocations &&
      gpu->stats().committed_submissions == before.committed_submissions,
      "Workflow allocation failure changed output or submitted work") || !Empty(budget))
    return false;
  if (!Ok(codestream_internal::EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image.const_view(), options, gpu.get(), true, &output)) ||
      !Check(output == expected, "Allocation-failure recovery changed codestream"))
    return false;
  gpu.reset();
  return Empty(budget);
}
}  // namespace

int main() {
  return CheckDirectOwnership() && CheckReservedAllocationFailure() &&
    CheckCacheDomainsAndReclamation() && CheckConcurrentDomains() && CheckButteraugliDomains() &&
    CheckWorkflowAccountingAndFailure() ? EXIT_SUCCESS : EXIT_FAILURE;
}
