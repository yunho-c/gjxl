// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>

#include "codec/quantization_pipeline_internal.h"

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
  if (void* result = std::malloc(std::max<size_t>(size, 1))) return result;
  throw std::bad_alloc();
}
void* operator new[](size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }

namespace {
using gjxl::quantization_pipeline_internal::PreparedQuantizationPipeline;
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
auto Pointers(const PreparedQuantizationPipeline& p) {
  return std::array{p.initial_quant.data(), p.strategy_mask.data(), p.pixel_mask.data()};
}
void CheckSize(const PreparedQuantizationPipeline& p) {
  const size_t blocks = p.block_extent.width * p.block_extent.height;
  Require(p.initial_quant.size() == blocks && p.strategy_mask.size() == blocks &&
      p.pixel_mask.size() == p.padded_extent.width * p.padded_extent.height,
      "Host initial storage has incorrect sizes");
}
size_t failures = 0, cases = 0;
void Case(gjxl::Extent2D extent, bool existing) {
  PreparedQuantizationPipeline p;
  p.padded_extent = extent;
  p.block_extent = {extent.width / 8, extent.height / 8};
  if (existing) {
    p.initial_quant.assign(2, -17.0f);
    p.strategy_mask.assign(3, -29.0f);
    p.pixel_mask.assign(5, -41.0f);
  }
  const auto pointers = Pointers(p);
  const auto quant = p.initial_quant, strategy = p.strategy_mask, pixel = p.pixel_mask;
  for (size_t fail = 0; fail < 3; ++fail) {
    gjxl::Status status;
    {
      allocation_failure::Scope scope(fail);
      status = p.PrepareHostInitialStorage();
    }
    Require(status.code() == gjxl::StatusCode::kOutOfMemory && Pointers(p) == pointers &&
        p.initial_quant == quant && p.strategy_mask == strategy && p.pixel_mask == pixel,
        "Failed allocation partially changed host storage");
    ++failures;
  }
  size_t attempts = 0;
  gjxl::Status status;
  {
    allocation_failure::Scope scope(std::numeric_limits<size_t>::max());
    status = p.PrepareHostInitialStorage();
    attempts = allocation_failure::attempts;
  }
  Require(status.ok() && attempts == 3, "Retry did not allocate all three arrays");
  CheckSize(p);
  for (const auto* values : {&p.initial_quant, &p.strategy_mask, &p.pixel_mask})
    Require(std::ranges::all_of(*values, [](float x) { return x == 0.0f; }),
            "New host storage is not initialized");
  p.initial_quant[0] = 11.0f; p.strategy_mask[0] = 13.0f; p.pixel_mask[0] = 17.0f;
  const auto ready = Pointers(p);
  {
    allocation_failure::Scope scope(0);
    status = p.PrepareHostInitialStorage();
    attempts = allocation_failure::attempts;
  }
  Require(status.ok() && attempts == 0 && Pointers(p) == ready &&
      p.initial_quant[0] == 11.0f && p.strategy_mask[0] == 13.0f && p.pixel_mask[0] == 17.0f,
      "Ready host storage allocated or changed contents");
  ++p.block_extent.width;
  Require(p.PrepareHostInitialStorage().code() == gjxl::StatusCode::kInvalidArgument &&
      Pointers(p) == ready && p.pixel_mask[0] == 17.0f,
      "Invalid geometry changed storage");
  p.padded_extent = {std::numeric_limits<size_t>::max() & ~size_t{7}, 16};
  p.block_extent = {p.padded_extent.width / 8, 2};
  Require(p.PrepareHostInitialStorage().code() == gjxl::StatusCode::kInvalidArgument &&
      Pointers(p) == ready, "Overflow geometry changed storage");
  ++cases;
}
}
int main() {
  try {
    for (gjxl::Extent2D extent : {gjxl::Extent2D{8, 8}, {264, 24}, {3840, 2160}})
      for (bool existing : {false, true}) Case(extent, existing);
    std::cout << "Host initial storage: " << cases << " cases, " << failures
              << " atomic allocation failures passed.\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
