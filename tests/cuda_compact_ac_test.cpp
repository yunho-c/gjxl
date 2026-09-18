// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/cuda/cuda_compact_ac_kernels.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void Require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Device {
  uint32_t *data = nullptr;
  explicit Device(size_t words) {
    void* allocation = nullptr;
    Check(cudaMalloc(&allocation, words * 4));
    data = static_cast<uint32_t*>(allocation);
  }
  ~Device() {
    if (data)
      cudaFree(data);
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
};

void Case(uint32_t count, size_t pattern) {
  constexpr uint32_t guard = 0xa193c7e5;
  const size_t padded = (size_t{count} + 3) / 4 * 4;
  std::vector<int32_t> source(count + 16, static_cast<int32_t>(guard));
  uint32_t expected_flags = 0;
  for (size_t i = 0; i < count; ++i) {
    int32_t value = 0;
    if (pattern == 1)
      value = i & 1 ? -128 : 127;
    if (pattern == 2)
      value = i & 1 ? -32768 : 32767;
    if (pattern == 3)
      value = i & 1 ? INT32_MIN : INT32_MAX;
    if (pattern == 4)
      value = i + 1 == count ? 128 : -128;
    if (pattern == 5)
      value = i + 1 == count ? -32769 : 32767;
    source[i + 8] = value;
    expected_flags |= value < -128 || value > 127 ? 1u : 0u;
    expected_flags |= value < -32768 || value > 32767 ? 2u : 0u;
  }
  std::vector<uint32_t> bytes(padded / 4 + 16, guard);
  std::vector<uint32_t> words(padded / 2 + 16, guard);
  std::array<uint32_t, 3> flags{guard, 0, guard};
  Device d_source(source.size()), d_bytes(bytes.size()), d_words(words.size()),
      d_flags(3);
  Check(cudaMemcpy(d_source.data, source.data(), source.size() * 4,
                   cudaMemcpyHostToDevice));
  Check(cudaMemcpy(d_bytes.data, bytes.data(), bytes.size() * 4,
                   cudaMemcpyHostToDevice));
  Check(cudaMemcpy(d_words.data, words.data(), words.size() * 4,
                   cudaMemcpyHostToDevice));
  Check(cudaMemcpy(d_flags.data, flags.data(), flags.size() * 4,
                   cudaMemcpyHostToDevice));
  Check(gjxl::LaunchCudaCompactAc(
      reinterpret_cast<int32_t *>(d_source.data + 8), count, d_bytes.data + 8,
      d_words.data + 8, d_flags.data + 1, nullptr));
  Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(bytes.data(), d_bytes.data, bytes.size() * 4,
                   cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(words.data(), d_words.data, words.size() * 4,
                   cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(flags.data(), d_flags.data, flags.size() * 4,
                   cudaMemcpyDeviceToHost));
  Require(flags == std::array<uint32_t, 3>{guard, expected_flags, guard},
          "Width flags/guards differ");
  for (size_t i = 0; i < 8; ++i) {
    Require(bytes[i] == guard && bytes[bytes.size() - 1 - i] == guard,
            "Byte guard changed");
    Require(words[i] == guard && words[words.size() - 1 - i] == guard,
            "Word guard changed");
  }
  const auto *raw_bytes = reinterpret_cast<const uint8_t *>(bytes.data() + 8);
  const auto *raw_words = reinterpret_cast<const uint8_t *>(words.data() + 8);
  for (size_t i = 0; i < padded; ++i) {
    const uint32_t expected =
        i < count ? static_cast<uint32_t>(source[i + 8]) : 0;
    uint16_t word;
    std::memcpy(&word, raw_words + i * 2, 2);
    Require(raw_bytes[i] == (expected & 255u) && word == (expected & 65535u),
            "Compact payload/padding differs");
  }
  auto returned = source;
  Check(cudaMemcpy(returned.data(), d_source.data, returned.size() * 4,
                   cudaMemcpyDeviceToHost));
  Require(returned == source, "Dense fallback source changed");
}
} // namespace

int main() {
  const cudaError_t probe = cudaFree(nullptr);
  if (probe != cudaSuccess) {
    std::cerr << "CUDA context probe: " << cudaGetErrorString(probe) << '\n';
    return 77;
  }
  try {
    size_t cases = 0;
    for (uint32_t count : {0u, 1u, 3u, 4u, 5u, 7u, 8u, 15u, 31u, 255u, 256u,
                           257u, 1023u, 1024u, 1025u, 8193u, 65537u}) {
      for (size_t pattern = 0; pattern < 6; ++pattern) {
        Case(count, pattern);
        ++cases;
      }
    }
    Require(gjxl::LaunchCudaCompactAc(nullptr, 1, nullptr, nullptr, nullptr,
                                      nullptr) == cudaErrorInvalidValue,
            "Null dispatch accepted");
    std::cout << "CUDA compact PASS cases=" << cases << '\n' << std::flush;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
