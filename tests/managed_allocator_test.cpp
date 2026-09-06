// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <list>
#include <stdexcept>
#include <thread>

#include "codestream/bit_writer.h"
#include "core/image_buffer.h"
#include "core/managed_allocator.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;

bool Check(bool good, const char* message) {
  if (!good) std::cerr << message << '\n';
  return good;
}
bool Ok(const Status& status) {
  if (!status.ok()) std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget& budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
    s.total.pending_count == 0 && s.open_reservations == 0 &&
    s.waiting_requests == 0, "Managed host resource charge leaked");
}

bool CheckScopeAndCapacity() {
  auto& budget = DefaultResourceBudget();
  if (!Empty(budget)) return false;
  ManagedVector<int> caller(8, 7);
  Image3FBuffer input({8, 8});
  if (!Empty(budget)) return false;
  ManagedVector<int> retained;
  {
    ManagedHostScope scope(ResourceClass::kPreparation);
    retained.reserve(32);
    retained.resize(3, 9);
    const auto s = budget.snapshot();
    if (!Check(s.total.backing_count == 1 &&
        s.total.live_capacity_bytes == retained.capacity() * sizeof(int) &&
        s.total.live_requested_bytes == s.total.live_capacity_bytes,
        "Container backing capacity was replaced by logical size")) return false;
    retained.clear();
    if (!Check(budget.snapshot().total.live_capacity_bytes == 32 * sizeof(int),
        "clear incorrectly released retained vector capacity")) return false;
    Image3FBuffer prepared({8, 8});
    if (!Check(budget.snapshot().total.backing_count == 4 &&
        budget.snapshot().total.live_capacity_bytes == 32 * sizeof(int) + 3 * 64 * sizeof(float),
        "Prepared image plane backings were not accounted")) return false;
  }
  if (!Check(!CurrentResourceContext().track_host_allocations &&
      budget.snapshot().total.backing_count == 1, "Host scope did not restore context"))
    return false;
  // Destruction after the host scope, with caller-owned inputs still alive.
  retained = {};
  ManagedVector<int>().swap(retained);
  return Empty(budget) && Check(caller[0] == 7, "Caller-owned storage changed");
}

bool CheckMovesDomainsAndAlignment() {
  struct alignas(256) Value { int value = 5; };
  ResourceBudget a(8192), b(8192);
  ResourceReservation first, second;
  if (!Ok(a.Reserve(8192, &first)) || !Ok(b.Reserve(8192, &second))) return false;
  ManagedVector<Value> original, copy;
  {
    ResourceContextScope context({&first, ResourceClass::kPreparation});
    original.resize(4);
  }
  if (!Check(reinterpret_cast<uintptr_t>(original.data()) % alignof(Value) == 0,
      "Over-aligned host payload is misaligned")) return false;
  Value* backing = original.data();
  {
    ResourceContextScope context({&second, ResourceClass::kSerializer});
    copy = original;
    ManagedVector<Value> moved(std::move(original));
    if (!Check(moved.data() == backing && a.snapshot().total.backing_count == 1 &&
        b.snapshot().total.backing_count == 1, "Move duplicated or migrated the backing charge"))
      return false;
    original = std::move(moved);
    // Growth allocates in B while A's old capacity remains charged until freed.
    original.reserve(8);
  }
  first.Reset();
  second.Reset();
  if (!Empty(a) || !Check(b.snapshot().total.backing_count == 2 &&
      b.snapshot().committed_bytes() == (copy.capacity() + original.capacity()) * sizeof(Value),
      "Cross-domain growth or producer teardown lost backing ownership")) return false;
  ManagedVector<Value>().swap(original);
  ManagedVector<Value>().swap(copy);
  return Empty(b);
}

bool CheckAllocationFailures() {
  ResourceBudget budget(64);
  ResourceReservation job;
  if (!Ok(budget.Reserve(64, &job))) return false;
  {
    ResourceContextScope context({&job, ResourceClass::kSerializer});
    ManagedVector<uint8_t> bytes(32, 0x55);
    auto* backing = bytes.data();
    ArmNextManagedHostAllocationFailureForTest();
    try {
      bytes.reserve(64);  // Needs 32 + 64 while replacing the old backing.
      return Check(false, "Growth escaped its full live-capacity reservation");
    } catch (const ManagedAllocationFailure& error) {
      if (!Check(error.status().code() == StatusCode::kOutOfMemory,
          "Admission failure lost its status")) return false;
    }
    // The budget failure occurred before the armed backing-allocation boundary.
    try {
      ManagedVector<uint8_t> failed(16);
      return Check(false, "Expected injected backing allocation failure");
    } catch (const std::bad_alloc&) {}
    if (!Check(bytes.data() == backing && bytes.size() == 32 && bytes[0] == 0x55 &&
        budget.snapshot().total.pending_count == 0 &&
        budget.snapshot().reserved_unbacked_bytes == 32,
        "Failed backing allocation changed data or lost reservation credit")) return false;
    ManagedVector<uint8_t> recovery(16);
    ManagedAllocator<uint64_t> allocator;
    try {
      (void)allocator.allocate(std::numeric_limits<size_t>::max());
      return Check(false, "Overflowing allocation was accepted");
    } catch (const std::bad_array_new_length&) {}
    if (!Check(allocator.allocate(0) == nullptr, "Zero allocation unexpectedly has storage"))
      return false;
    allocator.deallocate(nullptr, 0);
  }
  job.Reset();
  return Empty(budget);
}

struct ThrowingValue {
  static inline int copies_before_throw = -1;
  int value = 17;
  ThrowingValue() = default;
  ThrowingValue(const ThrowingValue& other) : value(other.value) {
    if (copies_before_throw == 0) throw std::runtime_error("injected element copy");
    if (copies_before_throw > 0) --copies_before_throw;
  }
  ThrowingValue(ThrowingValue&& other) noexcept(false) : ThrowingValue(other) {}
};

bool CheckElementFailureAndRebind() {
  ResourceBudget budget(4096);
  ResourceReservation job;
  if (!Ok(budget.Reserve(4096, &job))) return false;
  {
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    ManagedVector<ThrowingValue> values(4);
    const auto bytes = values.capacity() * sizeof(ThrowingValue);
    auto* backing = values.data();
    ThrowingValue::copies_before_throw = 2;
    try {
      values.reserve(8);
      return Check(false, "Expected element construction failure");
    } catch (const std::runtime_error&) {}
    ThrowingValue::copies_before_throw = -1;
    if (!Check(values.data() == backing && values.size() == 4 && values[0].value == 17 &&
        budget.snapshot().total.backing_count == 1 &&
        budget.snapshot().total.live_capacity_bytes == bytes &&
        budget.snapshot().peak_backing_bytes >= bytes + 8 * sizeof(ThrowingValue),
        "Element exception lost the prior value or leaked new backing")) return false;
    std::list<int, ManagedAllocator<int>> nodes{1, 2, 3};
    if (!Check(budget.snapshot().total.backing_count == 4,
        "Rebound node allocations were not independently charged")) return false;
  }
  job.Reset();
  return Empty(budget);
}

bool CheckWorkersAndRetainedStorage() {
  ResourceBudget budget(4096);
  ResourceReservation job;
  if (!Ok(budget.Reserve(4096, &job))) return false;
  std::array<ManagedVector<int>, 4> retained;
  std::barrier boundary(5);
  std::atomic<bool> good = true;
  thread_budget_internal::CpuParticipantTracker tracker;
  std::array<std::jthread, 4> workers;
  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::jthread([&, i] {
      {
        thread_budget_internal::ParallelScope scope(
          4, &tracker, {&job, ResourceClass::kSerializer, true});
        try { retained[i].resize(256, static_cast<int>(i)); }
        catch (...) { good = false; }
        boundary.arrive_and_wait();
        boundary.arrive_and_wait();
      }
      if (CurrentResourceContext().reservation != nullptr ||
          CurrentResourceContext().track_host_allocations) good = false;
    });
  }
  boundary.arrive_and_wait();
  good = good && budget.snapshot().total.live_capacity_bytes == 4096 &&
    budget.snapshot().total.backing_count == 4 && tracker.peak() == 4;
  boundary.arrive_and_wait();
  for (auto& worker : workers) worker.join();
  job.Reset();
  if (!Check(good && budget.snapshot().committed_bytes() == 4096,
      "Worker context or retained host ownership was lost")) return false;
  for (auto& values : retained) ManagedVector<int>().swap(values);
  return Empty(budget);
}

bool CheckBitWriterAtomicity() {
  ResourceBudget budget(16);
  ResourceReservation job;
  if (!Ok(budget.Reserve(16, &job))) return false;
  BitWriter writer;
  {
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    if (!Ok(writer.WriteBits(8, 0x55)) || !Ok(writer.WriteBits(56, 0))) return false;
    const auto before = budget.snapshot();
    const auto status = writer.WriteBits(56, 0);
    if (!Check(status.code() == StatusCode::kOutOfMemory &&
        writer.bits_written() == 64 && writer.padded_bytes()[0] == 0x55 &&
        before.total.live_capacity_bytes == budget.snapshot().total.live_capacity_bytes &&
        budget.snapshot().classes[static_cast<size_t>(ResourceClass::kSerializer)].backing_count == 1,
        "Bit-writer growth failure lost bytes, capacity, or owner classification")) return false;
    bool called = false;
    if (!Check(writer.WithMaxBits(128, [&] { called = true; return Status::Ok(); }).code() ==
        StatusCode::kOutOfMemory && !called && writer.bits_written() == 64,
        "Rejected bit-writer allotment invoked its callback")) return false;
  }
  job.Reset();
  BitWriter moved = std::move(writer);
  if (!Check(budget.snapshot().total.backing_count == 1 && moved.bits_written() == 64,
      "Bit-writer move did not preserve its independent backing charge")) return false;
  moved = BitWriter{};
  return Empty(budget);
}

bool CheckImageReplacementFailure() {
  ResourceBudget budget(1792);  // Three old 8x8 planes plus one new 16x16 plane.
  ResourceReservation job;
  if (!Ok(budget.Reserve(1792, &job))) return false;
  Image3FBuffer image;
  {
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    image.resize({8, 8});
    image.plane(0)[0] = 0.25f;
    auto* previous = image.plane(0).data();
    try {
      image.resize({16, 16});
      return Check(false, "Expected partial image replacement failure");
    } catch (const ManagedAllocationFailure&) {}
    const auto s = budget.snapshot();
    if (!Check(image.extent().width == 8 && image.extent().height == 8 &&
        image.plane(0).data() == previous && image.plane(0)[0] == 0.25f &&
        s.total.backing_count == 3 && s.total.live_capacity_bytes == 768 &&
        s.total.pending_count == 0 && s.peak_backing_bytes == 1792,
        "Partial image replacement changed prior storage or leaked capacity")) return false;
    image.resize({4, 4});
  }
  job.Reset();
  image = Image3FBuffer{};
  return Empty(budget);
}

bool CheckDomainWrapperLifetime() {
  ResourceBudget observer;
  ManagedVector<uint8_t> retained;
  {
    ResourceBudget producer(64);
    observer = producer;
    ResourceReservation job;
    if (!Ok(producer.Reserve(64, &job))) return false;
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    retained.resize(64, 0x55);
  }
  if (!Check(observer.snapshot().committed_bytes() == 64 && retained[0] == 0x55,
      "Producer domain wrapper destruction lost its retained allocation")) return false;
  std::jthread consumer([storage = std::move(retained)]() mutable {
    ManagedVector<uint8_t>().swap(storage);
  });
  consumer.join();
  return Empty(observer);
}
}  // namespace

int main() {
  return CheckScopeAndCapacity() && CheckMovesDomainsAndAlignment() &&
    CheckAllocationFailures() && CheckElementFailureAndRebind() &&
    CheckWorkersAndRetainedStorage() && CheckBitWriterAtomicity() &&
    CheckImageReplacementFailure() && CheckDomainWrapperLifetime() &&
    Empty(DefaultResourceBudget()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
