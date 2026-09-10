// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "gpu/metal/metal_backend_internal.h"

namespace gjxl::metal_internal {
struct MetalCacheAdmissionTestAccess {
  static Status AcquireAq(MetalBackend &gpu, MetalAqScratchArena kind,
                          size_t bytes, DeviceScratchArena *arena) {
    return gpu.AcquireAqScratchArena(kind, bytes, arena);
  }
  static void ReleaseAq(MetalBackend &gpu, MetalAqScratchArena kind,
                        DeviceScratchArena arena) {
    gpu.ReleaseAqScratchArena(kind, std::move(arena), true);
  }
  static Status AcquireButteraugli(MetalBackend &gpu, size_t bytes,
                                   DeviceScratchArena *arena,
                                   uint64_t *generation) {
    return gpu.AcquireButteraugliArena(bytes, arena, generation);
  }
  static void ReleaseButteraugli(MetalBackend &gpu, DeviceScratchArena &&arena,
                                 uint64_t generation) {
    gpu.ReleaseButteraugliArena(std::move(arena), generation, true);
    arena = DeviceScratchArena{};
  }
};
} // namespace gjxl::metal_internal

namespace {
using namespace gjxl;
using namespace gjxl::metal_internal;
using namespace gjxl::resource_budget_internal;
using Access = MetalCacheAdmissionTestAccess;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Ok(const Status &status) {
  if (!status.ok())
    std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                   s.total.pending_count == 0 && s.open_reservations == 0 &&
                   s.waiting_requests == 0,
               "Cache admission leaked a charge or waiter");
}
template <class Predicate> bool Until(Predicate &&predicate) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= end)
      return false;
    std::this_thread::yield();
  }
  return true;
}
MetalBackend &Metal(std::unique_ptr<GpuBackend> &owner) {
  return *static_cast<MetalBackend *>(owner.get());
}
struct Arenas {
  std::array<DeviceScratchArena, 3> aq;
  DeviceScratchArena butter;
  uint64_t generation = 0;
  bool Acquire(MetalBackend &gpu, size_t bytes = 256) {
    for (size_t i = 0; i < aq.size(); ++i)
      if (!Ok(Access::AcquireAq(gpu, static_cast<MetalAqScratchArena>(i), bytes,
                                &aq[i])))
        return false;
    return Ok(Access::AcquireButteraugli(gpu, bytes, &butter, &generation));
  }
  void Release(MetalBackend &gpu) {
    for (size_t i = 0; i < aq.size(); ++i)
      Access::ReleaseAq(gpu, static_cast<MetalAqScratchArena>(i),
                        std::move(aq[i]));
    Access::ReleaseButteraugli(gpu, std::move(butter), generation);
  }
};
bool Cache(MetalBackend &gpu, ResourceBudget &budget) {
  ResourceReservation job;
  if (!Ok(budget.TryReserve(1024, &job)))
    return false;
  ResourceContextScope scope({&job, ResourceClass::kPreparation});
  Arenas arenas;
  if (!arenas.Acquire(gpu))
    return false;
  arenas.Release(gpu);
  return true;
}
Status Evict(void *opaque) {
  return TrimMetalPreparationCachesForDomain(
      *static_cast<ResourceBudget *>(opaque));
}

bool CheckDomainEviction() {
  ResourceBudget a(4096), b(4096);
  std::unique_ptr<GpuBackend> first, second, third;
  if (!Ok(TrimMetalPreparationCachesForDomain(a)) ||
      !Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &first)) ||
      !Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &second)) ||
      !Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &third)) ||
      !Cache(Metal(first), a) || !Cache(Metal(second), a) ||
      !Cache(Metal(third), b))
    return false;
  if (!Check(a.snapshot().total.idle_capacity_bytes == 2048 &&
                 b.snapshot().total.idle_capacity_bytes == 1024,
             "Fixture did not populate all four pools on each backend"))
    return false;
  ResourceReservation job;
  if (!Ok(a.Reserve(4096, &job, {}, Evict, &a)) ||
      !Check(Metal(first).PreparationCacheBytesForTesting() == 0 &&
                 Metal(second).PreparationCacheBytesForTesting() == 0 &&
                 Metal(third).PreparationCacheBytesForTesting() == 1024 &&
                 b.snapshot().total.idle_capacity_bytes == 1024,
             "Domain eviction missed a backend or evicted another domain"))
    return false;
  job.Reset();
  if (!Ok(TrimMetalPreparationCachesForDomain(b)))
    return false;
  return Empty(a) && Empty(b) && Empty(DefaultResourceBudget());
}

bool CheckActiveReturn() {
  ResourceBudget budget(1024);
  ResourceReservation producer;
  std::unique_ptr<GpuBackend> owner;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &owner)) ||
      !Ok(budget.Reserve(1024, &producer)))
    return false;
  Arenas arenas;
  {
    ResourceContextScope scope({&producer, ResourceClass::kPreparation});
    if (!arenas.Acquire(Metal(owner)))
      return false;
  }
  if (!Ok(TrimMetalPreparationCachesForDomain(budget)) ||
      !Check(budget.snapshot().total.live_capacity_bytes == 1024,
             "Idle eviction invalidated active arenas"))
    return false;
  std::atomic<bool> acquired{false};
  std::jthread waiter([&](std::stop_token stop) {
    ResourceReservation consumer;
    acquired = budget.Reserve(1024, &consumer, stop, Evict, &budget).ok();
  });
  if (!Check(Until([&] { return budget.snapshot().waiting_requests == 1; }),
             "Admission failed to queue behind active arenas"))
    return false;
  arenas.Release(Metal(owner));
  if (!Check(Metal(owner).PreparationCacheBytesForTesting() == 0 &&
                 budget.snapshot().total.backing_count == 0,
             "An active arena refilled idle cache ahead of a waiter"))
    return false;
  producer.Reset();
  if (!Check(Until([&] { return acquired.load(); }),
             "Released arenas stranded the queued encode"))
    return false;
  waiter.join();
  return Empty(budget);
}

bool CheckOversizedButteraugli() {
  ResourceBudget budget(2048);
  ResourceReservation producer, consumer;
  std::unique_ptr<GpuBackend> owner;
  DeviceScratchArena arena;
  uint64_t generation = 0;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &owner)) ||
      !Ok(budget.Reserve(1024, &producer)))
    return false;
  auto &gpu = Metal(owner);
  {
    ResourceContextScope scope({&producer, ResourceClass::kPreparation});
    if (!Ok(Access::AcquireButteraugli(gpu, 512, &arena, &generation)))
      return false;
    Access::ReleaseButteraugli(gpu, std::move(arena), generation);
  }
  producer.Reset();
  if (!Ok(budget.Reserve(1024, &consumer)))
    return false;
  {
    ResourceContextScope scope({&consumer, ResourceClass::kPreparation});
    if (!Ok(Access::AcquireButteraugli(gpu, 256, &arena, &generation)) ||
        !Check(arena.capacity_bytes() == 256,
               "Admitted Butteraugli reused unplanned oversized capacity"))
      return false;
    // The old oversized hit would fit the currently unused credit, but leave
    // too little for this later allocation in the very same valid plan.
    std::unique_ptr<DeviceBuffer> later;
    if (!Ok(gpu.Allocate(700, &later)))
      return false;
    Access::ReleaseButteraugli(gpu, std::move(arena), generation);
  }
  consumer.Reset();
  if (!Ok(gpu.TrimPreparationCache()))
    return false;
  // Unadmitted callers retain the existing bounded-capacity hysteresis.
  if (!Ok(Access::AcquireButteraugli(gpu, 512, &arena, &generation)))
    return false;
  Access::ReleaseButteraugli(gpu, std::move(arena), generation);
  if (!Ok(Access::AcquireButteraugli(gpu, 256, &arena, &generation)) ||
      !Check(arena.capacity_bytes() == 512 || arena.capacity_bytes() == 256,
             "Legacy cache violated bounded reuse or purged-cache recovery"))
    return false;
  Access::ReleaseButteraugli(gpu, std::move(arena), generation);
  if (!Ok(gpu.TrimPreparationCache()))
    return false;
  return Empty(budget) && Empty(DefaultResourceBudget());
}

bool CheckRegistryLifetime() {
  ResourceBudget budget(16384);
  std::atomic<bool> good{true};
  std::jthread trimmer([&](std::stop_token stop) {
    while (!stop.stop_requested()) {
      if (!TrimMetalPreparationCachesForDomain(budget).ok())
        good = false;
      std::this_thread::yield();
    }
  });
  // Registration, domain visits and unregister/destruction overlap. No active
  // prepared object outlives its backend, as required by the existing API.
  for (size_t i = 0; i < 12; ++i) {
    std::unique_ptr<GpuBackend> owner;
    if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &owner)) ||
        !Cache(Metal(owner), budget))
      return false;
  }
  trimmer.request_stop();
  trimmer.join();
  return Check(good, "Concurrent registry visit failed") && Empty(budget);
}
} // namespace

int main() {
  return CheckDomainEviction() && CheckActiveReturn() &&
                 CheckOversizedButteraugli() && CheckRegistryLifetime()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
