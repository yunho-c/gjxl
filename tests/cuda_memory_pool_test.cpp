// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"

namespace {

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void Check(const gjxl::Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void CheckCuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
gjxl::cuda_internal::CudaBuffer& Buffer(gjxl::DeviceBuffer& buffer) {
  return dynamic_cast<gjxl::cuda_internal::CudaBuffer&>(buffer);
}

void RoundTrip(gjxl::GpuBackend& backend, uint32_t seed, size_t count) {
  std::vector<uint32_t> source(count), result(count);
  for (size_t i = 0; i < count; ++i) source[i] = seed + uint32_t(i) * 7919;
  std::unique_ptr<gjxl::DeviceBuffer> buffer;
  Check(backend.Allocate(count * sizeof(uint32_t) + 28, &buffer));
  Check(backend.CopyHostToDevice(*buffer, source.data(),
                                 count * sizeof(uint32_t), 12));
  Check(backend.CopyDeviceToHost(*buffer, result.data(),
                                 count * sizeof(uint32_t), 12));
  Require(source == result, "Reused allocation changed copied values");
  // Replacing the existing owner must not recursively acquire the lane mutex.
  Check(backend.Allocate(1024, &buffer));
}

void CheckOrderedReuse(gjxl::GpuBackend& backend) {
  using gjxl::cuda_internal::ScopedCudaDevice;
  for (size_t bytes : {1024u, 12345u, 524288u}) {
    std::unique_ptr<gjxl::DeviceBuffer> input, saved;
    Check(backend.Allocate(bytes, &input));
    Check(backend.Allocate(bytes, &saved));
    const auto* state = Buffer(*input).state();
    ScopedCudaDevice device(state->ordinal);
    CheckCuda(device.status());
    // No host synchronization separates initialization, use, release, and
    // reallocation. The lane's stream order must preserve the saved bytes.
    CheckCuda(
        cudaMemsetAsync(Buffer(*input).pointer(), 0xa5, bytes, state->stream));
    CheckCuda(cudaMemcpyAsync(Buffer(*saved).pointer(),
                              Buffer(*input).pointer(), bytes,
                              cudaMemcpyDeviceToDevice, state->stream));
    input.reset();
    Check(backend.Allocate(bytes, &input));
    CheckCuda(
        cudaMemsetAsync(Buffer(*input).pointer(), 0x37, bytes, state->stream));
    std::vector<unsigned char> values(bytes);
    Check(backend.CopyDeviceToHost(*saved, values.data(), bytes));
    for (auto value : values)
      Require(value == 0xa5, "Queued copy used recycled data");
    Check(backend.CopyDeviceToHost(*input, values.data(), bytes));
    for (auto value : values)
      Require(value == 0x37, "Reallocation was freed too early");
  }
}

void CheckOwnerLifetime() {
  std::unique_ptr<gjxl::GpuBackend> backend;
  Check(gjxl::CreateCudaBackend(&backend));
  std::unique_ptr<gjxl::DeviceBuffer> input, output;
  constexpr size_t kBytes = 64 * sizeof(float);
  Check(backend->Allocate(kBytes, &input));
  Check(backend->Allocate(kBytes, &output));
  const auto automatic_pool = Buffer(*output).state()->memory_pool;
  if (automatic_pool != nullptr) {
    cudaDeviceProp properties{};
    CheckCuda(cudaGetDeviceProperties(&properties, automatic_pool->ordinal));
    Require(automatic_pool->release_threshold_bytes ==
                std::min<uint64_t>(properties.totalGlobalMem / 2,
                                   uint64_t{4} << 30),
            "Automatic retention policy differs from device budget");
  }
  std::array<float, 64> zeros{};
  Check(backend->CopyHostToDevice(*input, zeros.data(), kBytes));
  std::unique_ptr<gjxl::GpuSubmission> submission;
  Check(backend->ForwardTransform(
      {.input = input.get(), .output = output.get(), .transform_count = 1},
      &submission));
  Require(submission != nullptr, "Missing queued transform");
  const auto* state = Buffer(*output).state();
  backend.reset();
  input.reset();
  Check(submission->Wait());
  gjxl::cuda_internal::ScopedCudaDevice device(state->ordinal);
  CheckCuda(device.status());
  std::array<float, 64> result;
  result.fill(1.0f);
  CheckCuda(cudaMemcpyAsync(result.data(), Buffer(*output).pointer(), kBytes,
                            cudaMemcpyDeviceToHost, state->stream));
  CheckCuda(cudaStreamSynchronize(state->stream));
  for (float value : result)
    Require(value == 0.0f, "Owner teardown changed pending work");
  output.reset();
  Check(submission->Wait());
  submission.reset();
}

void CheckTrimWithLiveBuffer(gjxl::GpuBackend& backend) {
  constexpr size_t kBytes = 1024 * 1024;
  std::unique_ptr<gjxl::DeviceBuffer> live, disposable;
  Check(backend.Allocate(kBytes, &live));
  Check(backend.Allocate(3 * kBytes, &disposable));
  const auto* state = Buffer(*live).state();
  gjxl::cuda_internal::ScopedCudaDevice device(state->ordinal);
  CheckCuda(device.status());
  CheckCuda(
      cudaMemsetAsync(Buffer(*live).pointer(), 0x5c, kBytes, state->stream));
  disposable.reset();
  Check(gjxl::TrimCudaDeviceMemory(state->ordinal));
  std::vector<unsigned char> values(kBytes);
  Check(backend.CopyDeviceToHost(*live, values.data(), kBytes));
  for (auto value : values)
    Require(value == 0x5c, "Trim invalidated live storage");
}

void Run() {
  constexpr uint64_t kRetention = uint64_t{64} << 20;
#if CUDART_VERSION >= 11020
  int supported = 0;
  CheckCuda(
      cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, 0));
  cudaMemPool_t default_pool = nullptr;
  uint64_t default_threshold = 0;
  if (supported != 0) {
    CheckCuda(cudaDeviceGetDefaultMemPool(&default_pool, 0));
    CheckCuda(cudaMemPoolGetAttribute(
        default_pool, cudaMemPoolAttrReleaseThreshold, &default_threshold));
  }
#endif
  std::unique_ptr<gjxl::GpuBackend> a, b, uncached, legacy;
  Check(gjxl::CreateCudaBackend(
      {.memory_pool_release_threshold_bytes = kRetention}, &a));
  Check(gjxl::CreateCudaBackend(
      {.memory_pool_release_threshold_bytes = kRetention}, &b));
  Check(gjxl::CreateCudaBackend({.memory_pool_release_threshold_bytes = 0},
                                &uncached));
  Check(gjxl::CreateCudaBackend({.use_stream_ordered_allocation = false},
                                &legacy));
  std::unique_ptr<gjxl::DeviceBuffer> ab, bb, ub, lb;
  Check(a->Allocate(4096, &ab));
  Check(b->Allocate(4096, &bb));
  Check(uncached->Allocate(4096, &ub));
  Check(legacy->Allocate(4096, &lb));
  auto pool = Buffer(*ab).state()->memory_pool;
  Require(Buffer(*lb).state()->memory_pool == nullptr,
          "Legacy override ignored");
#if CUDART_VERSION >= 11020
  Require((pool != nullptr) == (supported != 0),
          "Incorrect allocator selection");
  if (pool != nullptr) {
    Require(pool == Buffer(*bb).state()->memory_pool,
            "Matching backends did not share a pool");
    Require(pool != Buffer(*ub).state()->memory_pool,
            "Different retention policies shared a pool");
    Require(pool->pool != default_pool,
            "Modified the application default pool");
    uint64_t threshold = 0;
    int internal_dependencies = 1;
    CheckCuda(cudaMemPoolGetAttribute(
        pool->pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    CheckCuda(cudaMemPoolGetAttribute(pool->pool,
                                      cudaMemPoolReuseAllowInternalDependencies,
                                      &internal_dependencies));
    Require(threshold == kRetention && internal_dependencies == 0,
            "Incorrect pool attributes");
  }
#endif
  const auto allocations = a->stats().successful_allocations;
  auto* original = ab.get();
  const auto failure = a->Allocate(std::numeric_limits<size_t>::max(), &ab);
  Require(!failure.ok() && ab.get() == original &&
              a->stats().successful_allocations == allocations,
          "Impossible allocation changed caller output or statistics");
  // Consume the deliberate runtime allocation failure before further kernels.
  (void)cudaGetLastError();
  Require(!a->CopyHostToDevice(*bb, &allocations, sizeof(allocations)).ok(),
          "Shared pool bypassed backend ownership validation");
  ab.reset();
  bb.reset();
  ub.reset();
  lb.reset();
  for (auto* backend : {a.get(), b.get(), uncached.get(), legacy.get()}) {
    RoundTrip(*backend, 123, 4097);
    CheckOrderedReuse(*backend);
  }
  std::array<bool, 4> passed{};
  std::array<std::thread, 4> threads;
  for (size_t thread = 0; thread < threads.size(); ++thread) {
    threads[thread] = std::thread([&, thread] {
      try {
        auto& backend = thread % 2 == 0 ? *a : *b;
        for (uint32_t repeat = 0; repeat < 20; ++repeat)
          RoundTrip(backend, uint32_t(thread) * 99991 + repeat,
                    513 + repeat * 37);
        passed[thread] = true;
      } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (bool result : passed)
    Require(result, "Concurrent pool round trip failed");
  CheckOwnerLifetime();
  CheckTrimWithLiveBuffer(*a);
  CheckTrimWithLiveBuffer(*legacy);
  Check(gjxl::TrimCudaDeviceMemory());
#if CUDART_VERSION >= 11030
  if (pool != nullptr) {
    uint64_t reserved = 0, used = 0;
    CheckCuda(cudaMemPoolGetAttribute(
        pool->pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
    CheckCuda(cudaMemPoolGetAttribute(pool->pool, cudaMemPoolAttrUsedMemCurrent,
                                      &used));
    Require(reserved == 0 && used == 0,
            "Explicit trim did not release unused storage");
  }
#endif
#if CUDART_VERSION >= 11020
  if (supported != 0) {
    uint64_t unchanged = 0;
    CheckCuda(cudaMemPoolGetAttribute(
        default_pool, cudaMemPoolAttrReleaseThreshold, &unchanged));
    Require(unchanged == default_threshold, "Default pool threshold changed");
  }
#endif
  std::weak_ptr<gjxl::cuda_internal::CudaMemoryPoolState> lifetime = pool;
  pool.reset();
  a.reset();
  b.reset();
  uncached.reset();
  legacy.reset();
  Require(lifetime.expired(), "Registry retained an unused pool");
  Require(gjxl::TrimCudaDeviceMemory(-1).code() ==
              gjxl::StatusCode::kInvalidArgument,
          "Negative trim device accepted");
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    Run();
    std::cout << "CUDA memory pool ownership, reuse, concurrency, and trimming "
                 "passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
