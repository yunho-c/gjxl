// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <iostream>
#include <thread>

#include "core/frame_geometry.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_resource_internal.h"
#include "gpu/cuda/cuda_storage_plan.h"
#include "gpu/ops/resident_input.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;

void Check(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  Require(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
              s.total.pending_count == 0 && s.open_reservations == 0,
          "CUDA backing/ticket survived teardown or trim");
}
void *Pointer(DeviceBuffer &buffer) {
  return dynamic_cast<cuda_internal::CudaBuffer &>(buffer).pointer();
}

void DomainOwnership(GpuBackend &backend, bool cached) {
  ResourceBudget a(65536), b(65536);
  ResourceReservation first;
  Check(a.TryReserve(16384, &first));
  std::unique_ptr<DeviceBuffer> buffer;
  void *first_pointer = nullptr;
  {
    ResourceContextScope scope({&first, ResourceClass::kAcSearch});
    Check(backend.Allocate(8192, &buffer));
    first_pointer = Pointer(*buffer);
    const auto live = a.snapshot();
    Require(live.total.live_capacity_bytes == 8192 &&
                live.classes[static_cast<size_t>(ResourceClass::kAcSearch)]
                        .live_requested_bytes == 8192,
            "CUDA device backing was not charged to its allocation class");
    auto *before = buffer.get();
    Require(backend.Allocate(16384, &buffer).code() ==
                    StatusCode::kOutOfMemory &&
                buffer.get() == before && a.snapshot().total.pending_count == 0,
            "CUDA under-plan failure changed output or left a pending ticket");
    dynamic_cast<cuda_internal::CudaBackend &>(backend)
        .ArmNextAllocationFailureForTest();
    const auto count = backend.stats().successful_allocations;
    Require(backend.Allocate(1234, &buffer).code() ==
                    StatusCode::kOutOfMemory &&
                buffer.get() == before &&
                backend.stats().successful_allocations == count &&
                a.snapshot().total.pending_count == 0 &&
                a.snapshot().total.live_capacity_bytes == 8192,
            "Physical CUDA allocation failure leaked its authorization");
  }
  first.Reset();
  Require(a.snapshot().committed_bytes() == 8192,
          "Closing admission dropped a live CUDA backing ticket");
  buffer.reset();
  Require(a.snapshot().total.idle_capacity_bytes == (cached ? 8192 : 0),
          "CUDA cache capacity was not retained in its source domain");

  ResourceReservation second;
  Check(a.TryReserve(8192, &second));
  {
    ResourceContextScope scope({&second, ResourceClass::kAqScratch});
    Check(backend.Allocate(8192, &buffer));
    Require(!cached || Pointer(*buffer) == first_pointer,
            "Exact-size completed CUDA backing was not reused");
    const auto s = a.snapshot();
    Require(
        s.total.idle_capacity_bytes == 0 &&
            s.total.live_capacity_bytes == 8192 &&
            s.classes[static_cast<size_t>(ResourceClass::kAqScratch)]
                    .live_capacity_bytes == 8192 &&
            s.classes[static_cast<size_t>(ResourceClass::kAcSearch)]
                    .live_capacity_bytes == 0,
        "CUDA reuse failed to transfer/reclassify the original backing ticket");
  }
  second.Reset();
  first_pointer = Pointer(*buffer);
  buffer.reset();
  {
    ResourceReservation other;
    Check(b.TryReserve(8192, &other));
    ResourceContextScope scope({&other, ResourceClass::kPreparation});
    Check(backend.Allocate(8192, &buffer));
    Require(!cached || Pointer(*buffer) != first_pointer,
            "CUDA silently lent one domain's idle backing to another");
    Require(a.snapshot().total.idle_capacity_bytes == (cached ? 8192 : 0) &&
                b.snapshot().total.live_capacity_bytes == 8192,
            "Independent CUDA allocation changed another domain's charge");
  }
  buffer.reset();
  Check(cuda_internal::TrimCudaPreparationCaches(&a));
  Empty(a);
  Require(b.snapshot().total.idle_capacity_bytes == (cached ? 8192 : 0),
          "Domain-specific CUDA trim released another domain's idle backing");
  Check(cuda_internal::TrimCudaPreparationCaches(&b));
  Empty(b);

  // A live lease at the trim boundary must not refill the cache on release.
  {
    ResourceReservation live;
    Check(a.TryReserve(8192, &live));
    ResourceContextScope scope({&live, ResourceClass::kInput});
    Check(backend.Allocate(8192, &buffer));
  }
  Check(backend.TrimPreparationCache());
  Require(a.snapshot().total.live_capacity_bytes == 8192,
          "Trimming invalidated a live CUDA allocation");
  buffer.reset();
  Empty(a);
}

void OwnerLifetime() {
  ResourceBudget budget(4096);
  std::unique_ptr<GpuBackend> backend;
  Check(CreateCudaBackend({.memory_pool_release_threshold_bytes = 4097},
                          &backend));
  std::unique_ptr<DeviceBuffer> buffer;
  {
    ResourceReservation reservation;
    Check(budget.TryReserve(4096, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    Check(backend->Allocate(4096, &buffer));
  }
  backend.reset();
  Require(budget.snapshot().total.live_capacity_bytes == 4096,
          "Backend teardown discarded a surviving buffer's charge");
  buffer.reset();
  Empty(budget);
}

void BoundedCache(GpuBackend &backend) {
  ResourceBudget budget(1 << 20);
  for (size_t size = 1; size <= 160; ++size) {
    ResourceReservation reservation;
    Check(budget.TryReserve(size, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    std::unique_ptr<DeviceBuffer> buffer;
    Check(backend.Allocate(size, &buffer));
  }
  Require(budget.snapshot().total.backing_count <= 128 &&
              budget.snapshot().total.idle_capacity_bytes <= 65536,
          "CUDA cache exceeded its backing count/capacity bounds");
  Check(cuda_internal::TrimCudaPreparationCaches(&budget));
  Empty(budget);
}

void StorageRecipes(GpuBackend &backend) {
  using namespace cuda_internal;
  for (auto extent : {Extent2D{1, 1}, {17, 13}, {65, 67}, {1, 257}}) {
    size_t input_capacity = 0;
    Check(ComputeCudaInputStoragePlan(extent, &input_capacity));
    CudaButteraugliStoragePlan butter;
    Check(ComputeCudaButteraugliStoragePlan(extent, &butter));
    ResourceBudget budget(input_capacity + butter.capacity_bytes +
                          butter.host.peak_bytes);
    Image3FBuffer source(extent);
    FrameGeometry geometry;
    Check(FrameGeometry::Create(extent, &geometry));
    for (size_t channel = 0; channel < 3; ++channel)
      std::fill(source.plane(channel).begin(), source.plane(channel).end(),
                0.2f);
    std::unique_ptr<PreparedResidentInput> input;
    {
      ResourceReservation reservation;
      Check(budget.TryReserve(input_capacity - 1, &reservation));
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      Require(PrepareResidentInput(backend,
                                   {.original_linear_rgb = source.const_view(),
                                    .coding_extent = geometry.padded_frame()},
                                   &input)
                          .code() == StatusCode::kOutOfMemory &&
                  input == nullptr &&
                  budget.snapshot().total.backing_count == 0,
              "CUDA input preparation escaped its complete backing plan");
    }
    {
      ResourceReservation reservation;
      Check(budget.TryReserve(input_capacity, &reservation));
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      Check(PrepareResidentInput(backend,
                                 {.original_linear_rgb = source.const_view(),
                                  .coding_extent = geometry.padded_frame()},
                                 &input));
      Require(budget.snapshot().total.live_capacity_bytes == input_capacity,
              "CUDA input allocator and geometry recipe disagree");
    }
    std::unique_ptr<PreparedDeviceButteraugli> comparison;
    {
      ResourceReservation reservation;
      Check(budget.TryReserve(butter.capacity_bytes + butter.host.peak_bytes,
                              &reservation));
      ResourceContextScope scope({&reservation, ResourceClass::kButteraugli});
      Check(PrepareDeviceButteraugli(
          backend, {.reference_linear_rgb = input->original_linear_rgb()},
          &comparison));
      Require(comparison->memory_stats().prepared_allocation_bytes ==
                      butter.capacity_bytes &&
                  budget.snapshot().total.live_capacity_bytes ==
                      input_capacity + butter.capacity_bytes,
              "CUDA Butteraugli allocator and geometry recipe disagree");
    }
    comparison.reset();
    input.reset();
    Check(TrimCudaPreparationCaches(&budget));
    Empty(budget);
  }
  size_t unchanged = 321;
  Require(!ComputeCudaInputStoragePlan({}, &unchanged).ok() && unchanged == 321,
          "Invalid CUDA input plan changed its destination");
  CudaButteraugliStoragePlan unchanged_butter;
  unchanged_butter.capacity_bytes = 321;
  Require(!ComputeCudaButteraugliStoragePlan({}, &unchanged_butter).ok() &&
              unchanged_butter.capacity_bytes == 321,
          "Invalid CUDA metric plan changed its destination");
}
} // namespace

int main() {
  try {
    std::unique_ptr<GpuBackend> cached, uncached, legacy;
    const auto status = CreateCudaBackend(
        {.memory_pool_release_threshold_bytes = 65536}, &cached);
    if (status.code() == StatusCode::kUnavailable) {
      std::cout << status.message() << '\n';
      return 77;
    }
    Check(status);
    Check(CreateCudaBackend({.memory_pool_release_threshold_bytes = 0},
                            &uncached));
    Check(CreateCudaBackend({.use_stream_ordered_allocation = false}, &legacy));
    std::unique_ptr<DeviceBuffer> probe;
    Check(cached->Allocate(1, &probe));
    const bool supports_cache =
        dynamic_cast<cuda_internal::CudaBuffer &>(*probe)
            .state()
            ->memory_pool != nullptr;
    probe.reset();
    Check(cached->TrimPreparationCache());
    DomainOwnership(*cached, supports_cache);
    DomainOwnership(*uncached, false);
    DomainOwnership(*legacy, false);
    OwnerLifetime();
    BoundedCache(*cached);
    StorageRecipes(*cached);
    std::cout
        << "CUDA allocation/domain/cache/failure/lifetime accounting passed.\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
