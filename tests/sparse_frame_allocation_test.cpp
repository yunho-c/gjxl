// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include "sparse_frame_fixture.h"

namespace allocation_failure {
thread_local bool enabled = false;
thread_local size_t attempts = 0;
thread_local size_t fail_at = std::numeric_limits<size_t>::max();
bool ShouldFail() noexcept { return enabled && attempts++ == fail_at; }
struct Scope {
  explicit Scope(size_t index) { attempts = 0; fail_at = index; enabled = true; }
  ~Scope() { enabled = false; }
};
}
void* operator new(size_t size) {
  if (allocation_failure::ShouldFail()) throw std::bad_alloc();
  if (void* address = std::malloc(std::max<size_t>(size, 1))) return address;
  throw std::bad_alloc();
}
void* operator new[](size_t size) { return ::operator new(size); }
void operator delete(void* address) noexcept { std::free(address); }
void operator delete[](void* address) noexcept { ::operator delete(address); }
void operator delete(void* address, size_t) noexcept { ::operator delete(address); }
void operator delete[](void* address, size_t) noexcept { ::operator delete(address); }

namespace {
using namespace gjxl;
using gjxl_test::Check;
namespace internal = vardct_frame_internal;
size_t cases = 0, failures = 0, queries = 0;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T> void Case(bool cached, bool zero) {
  const auto oracle = gjxl_test::SparseOracle<T>(7, zero ? 0 : 1, 36);
  gjxl_test::PopulationAssembly layout(oracle);
  const auto population = gjxl_test::ReferencePopulation(oracle);
  const auto seed = gjxl_test::SparseStorage<T>(oracle);
  const auto input = [&](auto& owner) {
    return gjxl_test::SparseInput(oracle, layout.layouts, &owner, cached ? &population : nullptr);
  };
  auto owner = seed;
  const auto* masks = owner.masks.data();
  const auto* offsets = owner.offsets.data();
  const auto* values = owner.values.data();
  VarDctEncoderFrame frame;
  Status status;
  size_t allocations = 0;
  {
    allocation_failure::Scope scope(std::numeric_limits<size_t>::max());
    status = internal::AssembleVarDctEncoderFrame(input(owner), &frame);
    allocations = allocation_failure::attempts;
  }
  Check(status);
  Require(allocations != 0, "Sparse allocation counter inactive");
  Require(owner.masks.size() == 0 && owner.offsets.size() == 0 && owner.values.size() == 0,
          "Successful sparse assembly retained source arrays");
  auto copy = frame;
  {
    allocation_failure::Scope scope(0);
    Require(frame.valid() && copy.valid(), "Allocation-free sparse validation failed");
    for (size_t g = 0; g < frame.ac_group_count(); ++g) {
      VarDctNativeAcGroupView a, b;
      Check(frame.GetNativeAcGroup(g, &a)); Check(copy.GetNativeAcGroup(g, &b));
      const auto& span = std::get<VarDctSparseAcGroupViewT<T>>(a).coefficients[0];
      const auto& duplicate = std::get<VarDctSparseAcGroupViewT<T>>(b).coefficients[0];
      Require(span.begin().masks == masks && span.begin().offsets == offsets && span.begin().values == values,
              "Sparse frame copied the transferred payload");
      Require(duplicate.begin().masks != masks && duplicate.begin().offsets != offsets &&
                  (zero || duplicate.begin().values != values), "Sparse frame copy aliases mutable storage");
      ++queries;
    }
    Require(allocation_failure::attempts == 0, "Sparse const query allocated");
  }
  std::vector<uint8_t> expected;
  Check(EncodeVarDctCodestream(oracle, {}, &expected));
  for (size_t fail = 0; fail < allocations; ++fail) {
    auto fresh = seed;
    const auto* old_masks = fresh.masks.data();
    const auto* old_offsets = fresh.offsets.data();
    const auto* old_values = fresh.values.data();
    auto output = frame;
    VarDctNativeAcGroupView before;
    Check(output.GetNativeAcGroup(0, &before));
    {
      allocation_failure::Scope scope(fail);
      status = internal::AssembleVarDctEncoderFrame(input(fresh), &output);
    }
    Require(status.code() == StatusCode::kOutOfMemory, "Sparse allocation failure was not OutOfMemory");
    Require(fresh.coefficient_count == seed.coefficient_count && fresh.masks.data() == old_masks &&
                fresh.offsets.data() == old_offsets && fresh.values.data() == old_values &&
                std::ranges::equal(fresh.masks, seed.masks) && std::ranges::equal(fresh.offsets, seed.offsets) &&
                std::ranges::equal(fresh.values, seed.values), "Sparse allocation failure changed source");
    VarDctNativeAcGroupView after;
    Check(output.GetNativeAcGroup(0, &after));
    Require(std::get<VarDctSparseAcGroupViewT<T>>(before).coefficients[0].begin() ==
                std::get<VarDctSparseAcGroupViewT<T>>(after).coefficients[0].begin(),
            "Sparse allocation failure changed output identity");
    std::vector<uint8_t> actual;
    Check(EncodeVarDctCodestream(output, {}, &actual));
    Require(actual == expected, "Sparse allocation failure changed output bytes");
    ++failures;
  }
  ++cases;
}
}
int main() {
  try {
    for (bool cached : {false, true}) for (bool zero : {false, true}) {
      Case<int8_t>(cached, zero); Case<int16_t>(cached, zero); Case<int32_t>(cached, zero);
    }
    std::cout << "Sparse allocation PASS cases=" << cases << " injected_failures=" << failures
              << " allocation_free_queries=" << queries << '\n';
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
