// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gpu/cuda/cuda_kernels.h"

namespace {
using namespace gjxl::cuda_internal;
void Check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct Stream {
  cudaStream_t value{};
  Stream() { Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { (void)cudaStreamDestroy(value); }
};
struct Buffer {
  static constexpr size_t Guard = 64;
  unsigned char* data{};
  std::vector<unsigned char> initial;
  explicit Buffer(size_t bytes) : initial(bytes + 2 * Guard, 0xa7) {
    Check(cudaMalloc(&data, initial.size()));
    Reset();
  }
  ~Buffer() { (void)cudaFree(data); }
  Buffer(const Buffer&) = delete;
  template <class T>
  T* Get() {
    return reinterpret_cast<T*>(data + Guard);
  }
  // A pageable H2D copy may return before its DMA completes. Order the reset
  // before work on the non-blocking stream, including poison/error resets.
  void Reset() {
    Check(cudaMemcpy(data, initial.data(), initial.size(),
                     cudaMemcpyHostToDevice));
    Check(cudaStreamSynchronize(nullptr));
  }
  std::vector<unsigned char> Read() {
    std::vector<unsigned char> r(initial.size());
    Check(cudaMemcpy(r.data(), data, r.size(), cudaMemcpyDeviceToHost));
    return r;
  }
  void Guards(const std::vector<unsigned char>& values) {
    if (!std::equal(initial.begin(), initial.begin() + Guard, values.begin()) ||
        !std::equal(initial.end() - Guard, initial.end(), values.end() - Guard))
      throw std::runtime_error("Output guard changed");
  }
  void Same() {
    if (Read() != initial)
      throw std::runtime_error("Read-only input or guard changed");
  }
};
struct Fixture {
  CudaAqGeometry geometry;
  std::array<std::unique_ptr<Buffer>, 3> input;
  Buffer out_x, out_b, error;
  Stream stream;
  Fixture(unsigned width, unsigned height)
      : geometry{width,
                 height,
                 width / 8,
                 height / 8,
                 (width + 63) / 64,
                 (height + 63) / 64},
        out_x(geometry.tile_width * geometry.tile_height),
        out_b(geometry.tile_width * geometry.tile_height),
        error(sizeof(unsigned)) {
    if (!width || !height || width % 8 || height % 8)
      throw std::runtime_error("Invalid fixture geometry");
    for (auto& b : input)
      b = std::make_unique<Buffer>(size_t(width) * height * sizeof(float));
    unsigned prior = 1u << 12;
    std::memcpy(error.initial.data() + Buffer::Guard, &prior, sizeof(prior));
    error.Reset();
  }
  void Fill(unsigned pattern, unsigned reuse) {
    const size_t count = size_t(geometry.width) * geometry.height;
    uint32_t state =
        640127u + geometry.width + 7 * geometry.height + reuse * 371u;
    for (size_t i = 0; i < count; ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      const float base = (static_cast<int>(state % 65537) - 32768) * 0.00001f;
      for (unsigned c = 0; c < 3; ++c) {
        float value = base * (c + 1) + 0.03f * c;
        if (pattern == 1) value = 0.25f * c;
        if (pattern == 2) value = (i & 1) ? -0.0f : 0.0f;
        if (pattern == 3)
          value = base * (c == 0 ? 0.062f : c == 1 ? 1.0f : 1.078f);
        if (pattern == 4) value *= 1.0e10f;
        if (pattern == 5)
          value = ((i & 1) ? 1 : -1) * std::numeric_limits<float>::denorm_min();
        if (pattern == 6 && ((i + 13 * c + reuse) % 127) == 0)
          value = std::numeric_limits<float>::quiet_NaN();
        if (pattern == 7 && ((i + 13 * c + reuse) % 131) == 0)
          value = std::numeric_limits<float>::infinity();
        if (pattern == 8)
          value = (i % 4 == 0 ? 1.0e5f : i % 4 == 1 ? -1.0e5f : base);
        if (pattern == 9)
          value = 0.1f * c +
                  0.0001f * static_cast<float>((i / geometry.width) % 71) +
                  base * .2f;
        std::memcpy(
            input[c]->initial.data() + Buffer::Guard + i * sizeof(float),
            &value, sizeof(value));
      }
    }
    for (auto& b : input) b->Reset();
  }
  void Reset() {
    out_x.Reset();
    out_b.Reset();
    error.Reset();
  }
  void Launch(bool reference) {
    const auto launch =
        reference ? LaunchCudaAqInitialCflReference : LaunchCudaAqInitialCfl;
    Check(launch(input[0]->Get<float>(), input[1]->Get<float>(),
                 input[2]->Get<float>(), out_x.Get<signed char>(),
                 out_b.Get<signed char>(), error.Get<unsigned>(), geometry,
                 stream.value));
  }
  auto Result() {
    Check(cudaStreamSynchronize(stream.value));
    std::array<std::vector<unsigned char>, 3> result{out_x.Read(), out_b.Read(),
                                                     error.Read()};
    out_x.Guards(result[0]);
    out_b.Guards(result[1]);
    error.Guards(result[2]);
    return result;
  }
  unsigned ExpectedError() const {
    unsigned expected = 1u << 12;
    for (const auto& b : input)
      for (size_t i = Buffer::Guard; i < b->initial.size() - Buffer::Guard;
           i += 4) {
        uint32_t bits{};
        std::memcpy(&bits, b->initial.data() + i, 4);
        if ((bits & 0x7f800000u) == 0x7f800000u) expected |= 1u << 5;
      }
    return expected;
  }
  void Inputs() {
    for (auto& b : input) b->Same();
  }
};

void Verify(bool sanitizer) {
  std::vector<std::array<unsigned, 2>> shapes;
  if (sanitizer)
    shapes = {{8, 8},   {16, 8},  {24, 16}, {32, 24},
              {40, 32}, {64, 64}, {72, 72}, {128, 128}};
  else {
    shapes = {{8, 8},   {16, 8},  {24, 16},   {32, 24},   {40, 32},
              {48, 40}, {56, 48}, {64, 56},   {72, 64},   {8, 72},
              {64, 64}, {72, 72}, {120, 136}, {128, 128}, {136, 120}};
    for (unsigned tiles :
         {2u, 3u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u})
      shapes.push_back({tiles * 64 - 8, 72});
    shapes.push_back({32, 32768});
    shapes.push_back({16384, 64});
  }
  size_t fixtures = 0, comparisons = 0;
  for (auto shape : shapes) {
    Fixture f(shape[0], shape[1]);
    for (unsigned pattern = 0; pattern < 10; ++pattern) {
      if (sanitizer && pattern != 0 && pattern != 2 && pattern != 6 &&
          pattern != 7)
        continue;
      ++fixtures;
      for (unsigned reuse = 0; reuse < 3; ++reuse) {
        f.Fill(pattern, reuse);
        f.Reset();
        f.Launch(true);
        const auto expected = f.Result();
        unsigned bits{};
        std::memcpy(&bits, expected[2].data() + Buffer::Guard, sizeof(bits));
        if (bits != f.ExpectedError())
          throw std::runtime_error(
              "Reference error bits differ from host expectation");
        f.Reset();
        f.Launch(false);
        if (f.Result() != expected)
          throw std::runtime_error("Initial CfL map/error differs");
        f.Inputs();
        ++comparisons;
      }
    }
  }
  if (fixtures != (sanitizer ? 32u : 310u) || comparisons != 3 * fixtures)
    throw std::runtime_error("Fixture coverage differs");
  std::cout << "Verified " << fixtures
            << " initial CfL fixtures with three-stage reuse (" << comparisons
            << " comparisons)\n" << std::flush;
}
}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "Skipping CUDA initial CfL test: no CUDA device\n";
    return 77;
  }
  try {
    Check(cudaSetDevice(0));
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--sanitizer"))
      throw std::runtime_error("Expected optional --sanitizer");
    Verify(argc == 2);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
