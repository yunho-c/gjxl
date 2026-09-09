// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>

#include "coefficient_order_population_fixture.h"
#include "quantized_frame_fixture.h"

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

void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

const void* FirstPointer(const VarDctEncoderFrame& frame) {
  VarDctNativeAcGroupView group;
  Check(frame.GetNativeAcGroup(0, &group));
  return std::visit([](const auto& g) -> const void* {
    if constexpr (requires { g.coefficients[0].data(); }) return g.coefficients[0].data();
    else throw std::runtime_error("Expected dense compact test storage");
  }, group);
}

size_t cases = 0, injected = 0, allocation_free_queries = 0;

template<typename T>
void Case(bool owned, bool cached) {
  const auto seed = gjxl_test::MakeFrame(7, 1, 36);
  gjxl_test::PopulationAssembly assembly(seed);
  for (const auto& transform : assembly.layouts) {
    for (size_t c = 0; c < 3; ++c) {
      T value = static_cast<T>(c == 0 ? -17 : 29);
      std::fill_n(assembly.coefficients.data() + transform.coefficient_offsets[c],
                  transform.coefficient_count, value);
    }
  }
  VarDctEncoderFrame oracle;
  Check(internal::AssembleVarDctEncoderFrame(assembly.Input(nullptr, false), &oracle));
  const auto population = gjxl_test::ReferencePopulation(oracle);
  std::vector<uint8_t> expected;
  Check(EncodeVarDctCodestream(oracle, {}, &expected));
  const auto make_owner = [&] {
    OverwriteArray<T> owner;
    owner.ResetForOverwrite(assembly.coefficients.size());
    std::copy(assembly.coefficients.begin(), assembly.coefficients.end(), owner.begin());
    return owner;
  };
  const auto input = [&](OverwriteArray<T>& owner) {
    return internal::QuantizedFrameAssemblyInputT<T>{
      .geometry = oracle.geometry(), .strategies = &oracle.strategies(),
      .raw_quant_field = oracle.raw_quant_field(), .quantizer = &oracle.quantizer(),
      .y_to_x = oracle.color_correlation().y_to_x_map(),
      .y_to_b = oracle.color_correlation().y_to_b_map(),
      .epf_sharpness = oracle.epf_sharpness(), .profile = oracle.profile(),
      .quantized_dc = oracle.quantized_dc(),
      .quantized_ac = {owner.data(), owner.size()}, .transforms = assembly.layouts,
      .ac_group_storage = owned ? &owner : nullptr,
      .coefficient_order_population = cached ? &population : nullptr};
  };
  size_t allocations = 0;
  {
    auto owner = make_owner();
    VarDctEncoderFrame frame;
    Status status;
    {
      allocation_failure::Scope scope(std::numeric_limits<size_t>::max());
      status = internal::AssembleVarDctEncoderFrame(input(owner), &frame);
      allocations = allocation_failure::attempts;
    }
    Check(status);
    Require(allocations > 0, "Allocation counter did not observe assembly");
    Require(internal::GetAcStorageInfo(frame).coefficient_bytes == sizeof(T),
            "Assembly changed native width");
    std::vector<uint8_t> actual;
    Check(EncodeVarDctCodestream(frame, {}, &actual));
    Require(actual == expected, "Successful assembly differs");
    {
      allocation_failure::Scope scope(0);
      VarDctNativeAcGroupView group;
      status = frame.GetNativeAcGroup(0, &group);
      Require(status.ok() && allocation_failure::attempts == 0,
              "Native query allocated");
    }
    ++allocation_free_queries;
  }
  for (size_t index = 0; index < allocations; ++index) {
    auto owner = make_owner();
    const auto* address = owner.data();
    VarDctEncoderFrame output = oracle;
    const void* original = FirstPointer(output);
    Status status;
    {
      allocation_failure::Scope scope(index);
      status = internal::AssembleVarDctEncoderFrame(input(owner), &output);
    }
    Require(status.code() == StatusCode::kOutOfMemory,
            "Injected assembly allocation did not report OutOfMemory");
    Require(owner.data() == address &&
                std::ranges::equal(owner, assembly.coefficients),
            "Allocation failure changed input owner");
    Require(output.valid() && FirstPointer(output) == original,
            "Allocation failure changed output owner");
    std::vector<uint8_t> actual;
    Check(EncodeVarDctCodestream(output, {}, &actual));
    Require(actual == expected, "Allocation failure changed output contents");
    ++injected;
  }
  ++cases;
}
}  // namespace

int main() {
  try {
    for (bool owned : {false, true}) {
      for (bool cached : {false, true}) {
        Case<int8_t>(owned, cached);
        Case<int16_t>(owned, cached);
        Case<int32_t>(owned, cached);
      }
    }
    std::cout << "Compact allocation PASS cases=" << cases
              << " injected_failures=" << injected
              << " allocation_free_queries=" << allocation_free_queries
              << '\n' << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
