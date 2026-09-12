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
  struct FrameLease {
    std::unique_ptr<DeviceBuffer> buffer;
    std::weak_ptr<MetalBackendRegistry> registry;
  };
  static Status AcquireFrame(MetalBackend& gpu, size_t bytes, FrameLease* lease) {
    Status status = gpu.AcquireCompletedFrameAllocation(bytes, &lease->buffer);
    if (status.ok()) lease->registry = gpu.registry_;
    return status;
  }
  static void ReleaseFrame(FrameLease lease) {
    MetalBackend::ReturnCompletedFrameAllocation(
      std::move(lease.buffer), lease.registry);
  }
  static size_t FrameBytes(MetalBackend& gpu) {
    std::lock_guard lock(gpu.preparation_cache_mutex_);
    return gpu.idle_completed_frame_ ? gpu.idle_completed_frame_->size_bytes() : 0;
  }
  static void PurgeFrame(MetalBackend& gpu) {
    std::lock_guard lock(gpu.preparation_cache_mutex_);
    if (gpu.idle_completed_frame_)
      static_cast<MetalBuffer&>(*gpu.idle_completed_frame_).handle()->setPurgeableState(
        MTL::PurgeableStateEmpty);
  }
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

bool CheckFrameCache() {
  // Real completed frames use at least one 1 MiB capacity bucket. Sub-page
  // Metal buffers can ignore purgeability changes and cannot model reclamation.
  constexpr size_t bytes = size_t{1} << 20;
  std::unique_ptr<GpuBackend> owner;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH,
        {.completed_frame_cache_bytes = bytes}, &owner))) return false;
  auto& gpu = Metal(owner);
  Access::FrameLease first, live, reused;
  if (!Ok(Access::AcquireFrame(gpu, bytes, &first)) ||
      !Ok(Access::AcquireFrame(gpu, bytes, &live))) return false;
  auto* live_word = static_cast<uint32_t*>(static_cast<MetalBuffer&>(*live.buffer).contents());
  *live_word = 0x12345678;
  Access::ReleaseFrame(std::move(first));
  const auto before = gpu.stats().successful_allocations;
  if (!Check(Access::FrameBytes(gpu) == bytes, "Released frame was not cached") ||
      !Ok(Access::AcquireFrame(gpu, bytes, &reused)) ||
      !Check(gpu.stats().successful_allocations == before &&
                 reused.buffer.get() != live.buffer.get() && *live_word == 0x12345678,
             "Frame reuse allocated anew or altered a retained output")) return false;
  Access::ReleaseFrame(std::move(reused));
  Access::PurgeFrame(gpu);
  if (!Ok(Access::AcquireFrame(gpu, bytes, &reused)) ||
      !Check(gpu.stats().successful_allocations == before + 1,
             "Purged frame storage was reused")) return false;
  if (!Ok(gpu.TrimPreparationCache())) return false;
  Access::ReleaseFrame(std::move(reused));
  Access::ReleaseFrame(std::move(live));
  if (!Check(Access::FrameBytes(gpu) == 0, "Trimmed generation repopulated frame cache"))
    return false;
  if (!Ok(Access::AcquireFrame(gpu, bytes * 2, &reused))) return false;
  Access::ReleaseFrame(std::move(reused));
  if (!Check(Access::FrameBytes(gpu) == 0, "Oversized frame was retained")) return false;
  if (!Ok(Access::AcquireFrame(gpu, bytes, &reused))) return false;
  auto* word = static_cast<uint32_t*>(static_cast<MetalBuffer&>(*reused.buffer).contents());
  *word = 0xabcd0123;
  owner.reset();
  if (!Check(*word == 0xabcd0123, "Backend destruction invalidated a frame")) return false;
  Access::ReleaseFrame(std::move(reused));

  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH,
        {.completed_frame_cache_bytes = 0}, &owner)) ||
      !Ok(Access::AcquireFrame(Metal(owner), bytes, &reused))) return false;
  Access::ReleaseFrame(std::move(reused));
  if (!Check(Access::FrameBytes(Metal(owner)) == 0, "Disabled frame cache retained memory"))
    return false;
  owner.reset();
  return Empty(DefaultResourceBudget());
}

bool CheckFrameDomains() {
  constexpr size_t bytes = 4096;
  ResourceBudget a(2 * bytes), b(2 * bytes);
  ResourceReservation producer, next, other;
  std::unique_ptr<GpuBackend> owner;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &owner)) ||
      !Ok(a.Reserve(bytes, &producer))) return false;
  auto& gpu = Metal(owner);
  Access::FrameLease frame;
  {
    ResourceContextScope scope({&producer, ResourceClass::kCompletedFrame});
    if (!Ok(Access::AcquireFrame(gpu, bytes, &frame))) return false;
    Access::ReleaseFrame(std::move(frame));
  }
  producer.Reset();
  if (!Check(a.snapshot().total.idle_capacity_bytes == bytes,
             "Frame cache dropped its resource charge") ||
      !Ok(a.Reserve(bytes, &next))) return false;
  const auto before = gpu.stats().successful_allocations;
  {
    ResourceContextScope scope({&next, ResourceClass::kCompletedFrame});
    if (!Ok(Access::AcquireFrame(gpu, bytes, &frame)) ||
        !Check(gpu.stats().successful_allocations == before &&
                   a.snapshot().total.live_capacity_bytes == bytes &&
                   a.snapshot().total.idle_capacity_bytes == 0,
               "Frame charge was not transferred to its new reservation")) return false;
    Access::ReleaseFrame(std::move(frame));
  }
  next.Reset();
  if (!Ok(b.Reserve(bytes, &other))) return false;
  {
    ResourceContextScope scope({&other, ResourceClass::kCompletedFrame});
    if (!Ok(Access::AcquireFrame(gpu, bytes, &frame)) ||
        !Check(gpu.stats().successful_allocations == before + 1 &&
                   a.snapshot().total.backing_count == 0,
               "Frame capacity crossed accounting domains")) return false;
    Access::ReleaseFrame(std::move(frame));
  }
  other.Reset();
  if (!Ok(TrimMetalPreparationCachesForDomain(a)) ||
      !Check(Access::FrameBytes(gpu) == bytes, "Domain eviction removed another domain's frame") ||
      !Ok(TrimMetalPreparationCachesForDomain(b)) ||
      !Check(Access::FrameBytes(gpu) == 0, "Domain eviction missed the frame cache")) return false;
  return Empty(a) && Empty(b);
}

bool CheckFrameQueuedAdmission() {
  constexpr size_t bytes = 4096;
  ResourceBudget budget(bytes);
  ResourceReservation producer;
  std::unique_ptr<GpuBackend> owner;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &owner)) ||
      !Ok(budget.Reserve(bytes, &producer))) return false;
  auto& gpu = Metal(owner);
  Access::FrameLease frame;
  {
    ResourceContextScope scope({&producer, ResourceClass::kCompletedFrame});
    if (!Ok(Access::AcquireFrame(gpu, bytes, &frame))) return false;
  }
  std::atomic<bool> acquired{false};
  std::jthread waiter([&](std::stop_token stop) {
    ResourceReservation consumer;
    acquired = budget.Reserve(bytes, &consumer, stop, Evict, &budget).ok();
  });
  if (!Check(Until([&] { return budget.snapshot().waiting_requests == 1; }),
             "Frame admission did not queue")) return false;
  Access::ReleaseFrame(std::move(frame));
  producer.Reset();
  if (!Check(Until([&] { return acquired.load(); }), "Frame return stranded admission"))
    return false;
  waiter.join();
  return Check(Access::FrameBytes(gpu) == 0, "Frame cache bypassed queued admission") && Empty(budget);
}

bool CheckFrameProcessLimitAndTeardown() {
  constexpr size_t bytes = size_t{128} << 20;
  std::array<std::unique_ptr<GpuBackend>, 3> owners;
  for (auto& owner : owners) {
    Access::FrameLease frame;
    if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &owner)) ||
        !Ok(Access::AcquireFrame(Metal(owner), bytes, &frame))) return false;
    Access::ReleaseFrame(std::move(frame));
  }
  size_t retained = 0;
  for (auto& owner : owners) retained += Access::FrameBytes(Metal(owner));
  if (!Check(retained == (size_t{256} << 20), "Process frame-cache cap was not enforced"))
    return false;
  for (auto& owner : owners) owner.reset();

  // A late frame return races backend unregister/destruction. Only the
  // registry, never a raw backend pointer, is retained by the returning owner.
  for (size_t repeat = 0; repeat < 12; ++repeat) {
    std::unique_ptr<GpuBackend> owner;
    Access::FrameLease frame;
    if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &owner)) ||
        !Ok(Access::AcquireFrame(Metal(owner), 4096, &frame))) return false;
    std::jthread returning([lease = std::move(frame)]() mutable {
      Access::ReleaseFrame(std::move(lease));
    });
    owner.reset();
    returning.join();
  }
  return Empty(DefaultResourceBudget());
}
} // namespace

int main() {
  return CheckDomainEviction() && CheckActiveReturn() &&
                 CheckOversizedButteraugli() && CheckRegistryLifetime() &&
                 CheckFrameCache() && CheckFrameDomains() &&
                 CheckFrameQueuedAdmission() && CheckFrameProcessLimitAndTeardown()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
