// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gpu/cuda/cuda_aq_resident_kernels.h"

namespace {
using namespace gjxl::cuda_internal;
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
  cudaStream_t value{};
  Stream() { Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { (void)cudaStreamDestroy(value); }
};
struct Input {
  std::vector<CudaAqColorTransformRecord> records;
  std::vector<uint32_t> offsets;
  std::vector<float> tables, coefficients;
  std::vector<int> raw;
  uint32_t global_scale = 2048;
  uint32_t Tiles() const { return static_cast<uint32_t>(offsets.size() - 1); }
};
Input Synthetic(unsigned tiles, unsigned pattern) {
  Input input;
  input.tables.resize(11904);
  input.offsets.push_back(0);
  input.global_scale = pattern % 3 == 0 ? 1 : pattern % 3 == 1 ? 2048 : 32768;
  std::mt19937 rng(610731u + tiles + pattern * 129);
  for (float& value : input.tables)
    value = 0.25f + static_cast<float>(rng() % 10000) / 2000.0f;
  const std::array<unsigned, 7> strategies{0, 4, 5, 6, 7, 10, 11};
  for (unsigned tile = 0; tile < tiles; ++tile) {
    unsigned tile_value = pattern % 4;
    const unsigned transforms =
        pattern == 5 && tile % 3 == 0 ? 0 : 1 + (tile + pattern) % 3;
    for (unsigned i = 0; i < transforms; ++i) {
      unsigned strategy = strategies[(tile + i + pattern) % 7];
      unsigned count = strategy == 0   ? 64
                       : strategy == 4 ? 256
                       : strategy == 5 ? 1024
                       : strategy < 8  ? 128
                                       : 512;
      const unsigned offset = static_cast<unsigned>(input.coefficients.size());
      const unsigned stride = count + 13;
      input.coefficients.resize(offset + 3 * stride);
      for (unsigned c = 0; c < 3; ++c)
        for (unsigned j = 0; j < count; ++j) {
          float value = static_cast<int>(rng() % 100001) - 50000.0f;
          value *= 0.00001f;
          if (pattern == 1) value = 0.0f;
          if (pattern == 2) value = (j % 2) ? -0.0f : 0.0f;
          if (pattern == 3) value *= 1.0e10f;
          if (pattern == 4 && (j + tile + i) % 61 == 0)
            value = std::numeric_limits<float>::quiet_NaN();
          if (pattern == 10 && (j + tile + i) % 71 == 0)
            value = std::numeric_limits<float>::infinity();
          if (pattern == 11 && (j + tile + i) % 17 == 0)
            value = std::numeric_limits<float>::denorm_min();
          input.coefficients[offset + c * stride + j] = value;
        }
      const unsigned raw_index = static_cast<unsigned>(input.raw.size());
      input.raw.push_back(pattern == 6 && tile % 3 == 0 ? 0
                          : pattern == 7 && tile % 3 == 0
                              ? 257
                              : 1 + static_cast<int>(rng() % 256));
      if (pattern == 8 && tile % 3 == 0) strategy = 99;
      input.records.push_back(
          {offset, stride, pattern == 9 && tile % 3 == 0 ? count - 1 : count,
           strategy, raw_index, tile_value});
      tile_value += count;
    }
    input.offsets.push_back(static_cast<unsigned>(input.records.size()));
  }
  if (input.records.empty()) {
    input.records.resize(1);
    input.coefficients.resize(3 * 64);
    input.raw.resize(1);
  }
  return input;
}
struct Buffer {
  static constexpr size_t Guard = 64;
  unsigned char* data{};
  std::vector<unsigned char> initial;
  template <class T>
  explicit Buffer(const std::vector<T>& values)
      : initial(values.size() * sizeof(T) + 2 * Guard, 0xa5) {
    if (!values.empty())
      std::memcpy(initial.data() + Guard, values.data(),
                  values.size() * sizeof(T));
    Check(cudaMalloc(&data, initial.size()));
    Reset();
  }
  ~Buffer() { (void)cudaFree(data); }
  Buffer(const Buffer&) = delete;
  void Reset() {
    Check(cudaMemcpy(data, initial.data(), initial.size(),
                     cudaMemcpyHostToDevice));
  }
  template <class T>
  T* Ptr() {
    return reinterpret_cast<T*>(data + Guard);
  }
  std::vector<unsigned char> Get() const {
    std::vector<unsigned char> result(initial.size());
    Check(
        cudaMemcpy(result.data(), data, result.size(), cudaMemcpyDeviceToHost));
    return result;
  }
  void Verify() const {
    if (Get() != initial) throw std::runtime_error("Input/guard changed");
  }
  void Guards(const std::vector<unsigned char>& value) const {
    if (!std::equal(initial.begin(), initial.begin() + Guard, value.begin()) ||
        !std::equal(initial.end() - Guard, initial.end(), value.end() - Guard))
      throw std::runtime_error("Output guard changed");
  }
};
using Result = std::array<std::vector<unsigned char>, 3>;
struct Fixture {
  uint32_t tiles;
  Buffer records, offsets, tables, coefficients, raw, quantizer, x, b, error;
  Stream stream;
  explicit Fixture(const Input& input)
      : tiles(input.Tiles()),
        records(input.records),
        offsets(input.offsets),
        tables(input.tables),
        coefficients(input.coefficients),
        raw(input.raw),
        quantizer(std::vector<uint32_t>{input.global_scale}),
        x(std::vector<signed char>(tiles, -91)),
        b(std::vector<signed char>(tiles, -73)),
        error(std::vector<uint32_t>{0x10000000u}) {
    // Order all pageable default-stream fixture copies before the nonblocking
    // work stream; returning from a copy alone does not establish that edge.
    Check(cudaDeviceSynchronize());
  }
  void Reset() {
    x.Reset();
    b.Reset();
    error.Reset();
    Check(cudaDeviceSynchronize());
  }
  void Launch(bool reference) {
    const auto launch = reference ? LaunchCudaAqFinalColorCorrelationReference
                                  : LaunchCudaAqFinalColorCorrelation;
    Check(launch(
        records.Ptr<CudaAqColorTransformRecord>(), offsets.Ptr<uint32_t>(),
        tables.Ptr<float>(), coefficients.Ptr<float>(), raw.Ptr<int>(),
        quantizer.Ptr<unsigned>(), x.Ptr<signed char>(), b.Ptr<signed char>(),
        error.Ptr<unsigned>(), tiles, stream.value));
  }
  Result Get() {
    Check(cudaStreamSynchronize(stream.value));
    Result result{x.Get(), b.Get(), error.Get()};
    x.Guards(result[0]);
    b.Guards(result[1]);
    error.Guards(result[2]);
    return result;
  }
  void VerifyInputs() {
    records.Verify();
    offsets.Verify();
    tables.Verify();
    coefficients.Verify();
    raw.Verify();
    quantizer.Verify();
  }
  void Compare(const Result& expected) {
    if (Get() != expected) throw std::runtime_error("Color map/error differs");
  }
};
void Guards(bool small) {
  size_t comparisons = 0;
  for (unsigned tiles :
       {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129}) {
    if (small && tiles != 9 && tiles != 33) continue;
    for (unsigned pattern = 0; pattern < 12; ++pattern) {
      if (small && pattern != 0 && pattern != 4 && pattern != 5 && pattern != 8)
        continue;
      Input input = Synthetic(tiles, pattern);
      Fixture fixture(input);
      for (unsigned reuse = 0; reuse < 3; ++reuse) {
        if (reuse == 2) {
          const unsigned scale = pattern % 3 == 0   ? 0
                                 : pattern % 3 == 1 ? 32769
                                                    : 1;
          std::memcpy(fixture.quantizer.initial.data() + Buffer::Guard, &scale,
                      sizeof(scale));
          fixture.quantizer.Reset();
          Check(cudaDeviceSynchronize());
        }
        fixture.Reset();
        fixture.Launch(true);
        const auto expected = fixture.Get();
        fixture.VerifyInputs();
        fixture.Reset();
        fixture.Launch(false);
        fixture.Compare(expected);
        fixture.VerifyInputs();
        ++comparisons;
      }
    }
    std::cout << "Verified final-CfL tiles=" << tiles
              << " comparisons=" << comparisons << '\n'
              << std::flush;
  }
  const size_t expected = small ? 24 : 576;
  if (comparisons != expected)
    throw std::runtime_error("Guard coverage differs");
  std::cout << "Verified " << comparisons
            << " guarded final-CfL map/error comparisons across three "
               "nonblocking-stream reuse stages.\n"
            << std::flush;
}
}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    Check(cudaSetDevice(0));
    Guards(argc > 1 && std::string_view(argv[1]) == "--sanitizer");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
