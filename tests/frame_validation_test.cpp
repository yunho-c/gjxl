// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>

#include "sparse_frame_fixture.h"

namespace allocation_failure {
thread_local bool enabled = false;
thread_local size_t attempts = 0;
thread_local size_t fail_at = std::numeric_limits<size_t>::max();
bool ShouldFail() noexcept { return enabled && attempts++ == fail_at; }
struct Scope {
  explicit Scope(size_t index) {
    attempts = 0;
    fail_at = index;
    enabled = true;
  }
  ~Scope() { enabled = false; }
};
}  // namespace allocation_failure

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
namespace internal = vardct_frame_internal;
using gjxl_test::Check;
static_assert(std::is_nothrow_move_constructible_v<VarDctEncoderFrame>);
static_assert(std::is_nothrow_move_assignable_v<VarDctEncoderFrame>);
size_t cases = 0, injected = 0, queries = 0, tails = 0;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

const void* FirstPointer(const VarDctEncoderFrame& frame) {
  VarDctNativeAcGroupView group;
  Check(frame.GetNativeAcGroup(0, &group));
  return std::visit([](const auto& g) -> const void* {
    if constexpr (requires { g.coefficients[0].data(); }) return g.coefficients[0].data();
    else return g.coefficients[0].begin().masks;
  }, group);
}

template <typename T>
VarDctEncoderFrame Represent(const VarDctEncoderFrame& seed, bool sparse) {
  gjxl_test::PopulationAssembly assembly(seed);
  const auto population = gjxl_test::ReferencePopulation(seed);
  VarDctEncoderFrame result;
  if (sparse) {
    auto owner = gjxl_test::SparseStorage<T>(seed);
    Check(internal::AssembleVarDctEncoderFrameImpl(
      gjxl_test::SparseInput(seed, assembly.layouts, &owner, &population), &result));
  } else {
    OverwriteArray<T> owner;
    owner.ResetForOverwrite(assembly.coefficients.size());
    std::copy(assembly.coefficients.begin(), assembly.coefficients.end(), owner.begin());
    auto input = gjxl_test::SparseInput<T>(seed, assembly.layouts, nullptr, &population);
    input.quantized_ac = {owner.data(), owner.size()};
    input.ac_group_storage = &owner;
    Check(internal::AssembleVarDctEncoderFrameImpl(input, &result));
  }
  return result;
}

std::vector<VarDctEncoderFrame> Representations(size_t side, size_t pattern) {
  const auto seed = gjxl_test::MakeFrame(7, pattern, side);
  std::vector<VarDctEncoderFrame> result;
  for (bool sparse : {false, true}) {
    result.push_back(Represent<int8_t>(seed, sparse));
    result.push_back(Represent<int16_t>(seed, sparse));
    result.push_back(Represent<int32_t>(seed, sparse));
  }
  return result;
}

void CopyCase(const VarDctEncoderFrame& source, const VarDctEncoderFrame& seed) {
  const void* source_pointer = FirstPointer(source);
  auto destination = seed;
  size_t allocations;
  {
    allocation_failure::Scope scope(std::numeric_limits<size_t>::max());
    destination = source;
    allocations = allocation_failure::attempts;
  }
  Require(allocations > 0, "Copy assignment did not allocate");
  gjxl_test::EqualSparseCoefficients(source, destination);
  Require(FirstPointer(destination) != source_pointer, "Copy did not own AC storage");
  std::vector<uint8_t> expected, actual;
  Check(EncodeVarDctCodestream(source, {}, &expected));
  Check(EncodeVarDctCodestream(destination, {}, &actual));
  Require(actual == expected, "Copy changed codestream");
  for (size_t index = 0; index < allocations; ++index) {
    destination = seed;
    const void* pointer = FirstPointer(destination);
    const auto* population = internal::GetCoefficientOrderPopulation(destination);
    bool threw = false;
    {
      allocation_failure::Scope scope(index);
      try { destination = source; } catch (const std::bad_alloc&) { threw = true; }
    }
    Require(threw, "Copy allocation fault was not reached");
    Require(FirstPointer(destination) == pointer &&
      internal::GetCoefficientOrderPopulation(destination) == population,
      "Failed copy replaced destination ownership");
    gjxl_test::EqualSparseCoefficients(seed, destination);
    Require(FirstPointer(source) == source_pointer && source.valid(), "Copy changed source");
    ++injected;
  }
  {
    allocation_failure::Scope scope(0);
    destination = destination;
    Require(destination.valid() && source.valid() && allocation_failure::attempts == 0,
      "Self-copy or const validation allocated");
  }
  queries += 2;
  auto copied(source);
  auto moved(std::move(copied));
  Require(!copied.valid(), "Moved-from copy remains valid");
  destination = std::move(moved);
  Require(!moved.valid(), "Move-assigned source remains valid");
  gjxl_test::EqualSparseCoefficients(source, destination);
  VarDctEncoderFrame empty;
  destination = empty;
  Require(!destination.valid(), "Assignment of invalid frame retained validation");
  VarDctEncoderFrame empty_copy(empty);
  Require(!empty_copy.valid(), "Copy of invalid frame became valid");
  ++cases;
}

template <typename T>
void TailCases() {
  const auto seed = gjxl_test::MakeFrame(7, 1, 36);
  gjxl_test::PopulationAssembly assembly(seed);
  for (size_t g = 0; g < seed.ac_group_count(); ++g) {
    VarDctNativeAcGroupView native;
    Check(seed.GetNativeAcGroup(g, &native));
    const size_t used = std::visit([](const auto& group) { return group.used_coefficient_count; }, native);
    constexpr size_t capacity = kVarDctAcGroupCoefficientCapacity;
    if (used == capacity) continue;
    for (size_t c = 0; c < 3; ++c) {
      for (size_t position : {used, used + (capacity - used) / 2, capacity - 1}) {
        OverwriteArray<T> owner;
        owner.ResetForOverwrite(assembly.coefficients.size());
        std::copy(assembly.coefficients.begin(), assembly.coefficients.end(), owner.begin());
        owner.data()[(g * 3 + c) * capacity + position] = 1;
        const void* input_pointer = owner.data();
        auto output = seed;
        const void* output_pointer = FirstPointer(output);
        auto input = gjxl_test::SparseInput<T>(seed, assembly.layouts, nullptr);
        input.quantized_ac = {owner.data(), owner.size()};
        input.ac_group_storage = &owner;
        const auto status = internal::AssembleVarDctEncoderFrameImpl(input, &output);
        Require(!status.ok() && owner.data() == input_pointer && FirstPointer(output) == output_pointer,
          "Malformed dense tail published or consumed ownership");
        gjxl_test::EqualSparseCoefficients(seed, output);
        ++tails;
      }
    }
  }
}
}  // namespace

int main() {
  try {
    // Same group capacity/count but different active ranges: stale provenance
    // after a partial memberwise copy must never validate the old owner's tail.
    const auto small = Representations(36, 2);
    const auto large = Representations(60, 1);
    for (const auto& a : small) for (const auto& b : large) {
      CopyCase(a, b);
      CopyCase(b, a);
    }
    TailCases<int8_t>(); TailCases<int16_t>(); TailCases<int32_t>();
    std::cout << "Frame validation PASS cases=" << cases << " injected_failures=" << injected
      << " allocation_free_validations=" << queries << " rejected_tails=" << tails << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
