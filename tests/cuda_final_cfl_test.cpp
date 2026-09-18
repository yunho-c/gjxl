// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
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
  void LaunchNonlinear(uint32_t iterations) {
    Check(LaunchCudaAqFinalColorCorrelationNonlinear(
        records.Ptr<CudaAqColorTransformRecord>(), offsets.Ptr<uint32_t>(),
        tables.Ptr<float>(), coefficients.Ptr<float>(), raw.Ptr<int>(),
        quantizer.Ptr<unsigned>(), x.Ptr<signed char>(), b.Ptr<signed char>(),
        error.Ptr<unsigned>(), tiles, iterations, stream.value));
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

// Scalar CPU objective, with no GPU chunks, barriers, or shared accumulation.
// Include zero low-frequency slots and tile prefixes in the regularizer and
// lane numbering, just as the codec's prepared-coefficient reference does.
signed char NonlinearReference(const Input& input, unsigned tile,
                              unsigned channel, unsigned iterations) {
  std::vector<std::array<float, 2>> samples;
  for (unsigned t = input.offsets[tile]; t < input.offsets[tile + 1]; ++t) {
    const auto& record = input.records[t];
    const unsigned width = record.strategy == 0 ? 8 :
      record.strategy == 4 || record.strategy == 6 || record.strategy == 7 ? 16 : 32;
    const unsigned height = record.coefficient_count / width;
    const unsigned table = record.strategy == 0 ? 192 : record.strategy == 4 ? 1152 :
      record.strategy == 5 ? 4992 : record.strategy < 8 ? 8448 : 10368;
    const float scale = float(input.global_scale) * (1.0f / 65536.0f) *
      128.0f * input.raw[record.raw_quant_index];
    samples.resize(record.tile_value_offset + record.coefficient_count);
    for (unsigned i = 0; i < record.coefficient_count; ++i) {
      if (i % width < width / 8 && i / width < height / 8) continue;
      const auto base = record.coefficient_offset + i;
      const float weight = input.tables[table + 2 * channel * record.coefficient_count + i];
      const float y = input.coefficients[base + record.channel_stride] * weight * scale;
      const float chroma = input.coefficients[base + 2 * channel * record.channel_stride] * weight * scale;
      samples[record.tile_value_offset + i] = {y / 84.0f, channel == 0 ? -chroma : y - chroma};
    }
  }
  float estimate = 0;
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    std::array<std::array<float, 4>, 3> lanes{};
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto [a, b] = samples[i];
      if (std::abs(std::fma(a, estimate, b)) >= 100.0f) continue;
      for (unsigned k = 0; k < 3; ++k) {
        const float offset = k == 0 ? 0 : k == 1 ? 100 : -100;
        const float residual = std::fma(a, estimate + offset, b);
        const float magnitude = (2.0f / 3.0f) * a * (std::abs(residual) + 1.0f);
        lanes[k][i % 4] += residual < 0 ? -magnitude : magnitude;
      }
    }
    std::array<float, 3> d;
    for (unsigned k = 0; k < 3; ++k) {
      const float offset = k == 0 ? 0 : k == 1 ? 100 : -100;
      d[k] = ((2.0f * 1.0e-9f) * float(samples.size())) * (estimate + offset);
      d[k] += (lanes[k][0] + lanes[k][1]) + (lanes[k][2] + lanes[k][3]);
    }
    const float step = d[0] / ((d[1] - d[2]) / 200.0f + 0.85f);
    estimate -= std::clamp(step, -20.0f, 20.0f);
    if (std::abs(step) < 3.0e-3f) break;
  }
  estimate = estimate >= 2.6f ? estimate - 2.6f : estimate <= -2.6f ? estimate + 2.6f : 0;
  return static_cast<signed char>(std::clamp(std::round(estimate), -128.0f, 127.0f));
}

void NonlinearGuards(bool small) {
  size_t comparisons = 0;
  for (unsigned tiles : {1u, 9u, 33u}) {
    if (small && tiles != 9) continue;
    for (unsigned pattern : {0u, 1u, 2u, 3u, 11u}) {
      const Input input = Synthetic(tiles, pattern);
      Fixture fixture(input);
      for (unsigned iterations : {1u, 4u, 8u, 20u}) {
        fixture.Reset();
        fixture.LaunchNonlinear(iterations);
        const auto result = fixture.Get();
        fixture.VerifyInputs();
        if (result[2] != fixture.error.initial)
          throw std::runtime_error("Valid nonlinear CfL set the error flag");
        for (unsigned tile = 0; tile < tiles; ++tile) {
          for (unsigned channel = 0; channel < 2; ++channel) {
            const auto expected = NonlinearReference(input, tile, channel, iterations);
            const auto actual = static_cast<signed char>(result[channel][Buffer::Guard + tile]);
            if (actual != expected)
              throw std::runtime_error("Nonlinear CfL differs from CPU objective: pattern=" +
                std::to_string(pattern) + " tile=" + std::to_string(tile) +
                " iterations=" + std::to_string(iterations) + " expected=" +
                std::to_string(expected) + " actual=" + std::to_string(actual));
            ++comparisons;
          }
        }
      }
      // The same buffers remain valid when switching back to the fast policy.
      fixture.Reset();
      fixture.Launch(true);
      const auto fast = fixture.Get();
      fixture.Reset();
      fixture.Launch(false);
      fixture.Compare(fast);
    }
  }
  for (unsigned pattern : {4u, 5u, 6u, 7u, 8u, 9u, 10u}) {
    Fixture fixture(Synthetic(9, pattern));
    fixture.LaunchNonlinear(8);
    const auto result = fixture.Get();
    unsigned error = 0;
    std::memcpy(&error, result[2].data() + Buffer::Guard, sizeof(error));
    if (error != (0x10000000u | 64u))
      throw std::runtime_error("Nonlinear CfL failed to flag invalid input");
    fixture.VerifyInputs();
  }
  std::cout << "Verified " << comparisons << " nonlinear CfL CPU map comparisons.\n" << std::flush;
}

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
    std::cout << "Starting nonlinear CfL validation.\n" << std::flush;
    NonlinearGuards(argc > 1 && std::string_view(argv[1]) == "--sanitizer");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
