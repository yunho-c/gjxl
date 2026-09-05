// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "gpu/cuda/cuda_butteraugli_kernels.h"

namespace {
using gjxl::cuda_internal::CudaButteraugliFrequencyParams;

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceArray {
  explicit DeviceArray(const std::vector<float>& values) : count(values.size()) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    Write(values);
  }
  ~DeviceArray() { (void)cudaFree(data); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  void Write(const std::vector<float>& values) {
    CheckCuda(cudaMemcpy(data, values.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  }
  std::vector<float> Read() const {
    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
  }
  float* data = nullptr;
  size_t count;
};

void Equal(const char* name, const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::runtime_error("Array size mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      std::cerr << name << " mismatch at " << i << ": " << a[i] << " / " << b[i] << '\n';
      throw std::runtime_error("Bitwise frequency-split mismatch");
    }
  }
}

void Guards(const std::vector<float>& actual, const std::vector<float>& initial,
            size_t offset, uint32_t stride, uint32_t width, uint32_t height) {
  for (size_t i = 0; i < actual.size(); ++i) {
    const bool active = i >= offset && i < offset + static_cast<size_t>(stride) * height &&
                        (i - offset) % stride < width;
    if (!active && std::memcmp(&actual[i], &initial[i], sizeof(float)) != 0)
      throw std::runtime_error("Frequency-split guard overwritten");
  }
}

struct Case {
  static constexpr uint32_t kInputOffset = 3, kOutputOffset = 5;
  static constexpr uint32_t kWorkOffset = 7, kBlurOffset = 11, kWeightOffset = 3;
  CudaButteraugliFrequencyParams params;
  uint32_t blur_stride;
  std::vector<float> input, output, work, blurred, weights;

  Case(uint32_t width, uint32_t height, uint32_t channel, unsigned pattern)
      : params{width, height, width + 13, width + 19, channel},
        blur_stride(width + 23) {
    const float poison = std::numeric_limits<float>::quiet_NaN();
    input.assign(kInputOffset + static_cast<size_t>(params.input_stride) * height + 31, poison);
    output.assign(kOutputOffset + static_cast<size_t>(params.output_stride) * height + 31, poison);
    work.assign(kWorkOffset + static_cast<size_t>(width) * height + 31, poison);
    blurred.assign(kBlurOffset + static_cast<size_t>(blur_stride) * height + 31, poison);
    const uint32_t taps = channel < 2 ? 15 : 7;
    weights.assign(kWeightOffset + taps + 7, poison);
    for (uint32_t i = 0; i < taps; ++i) {
      const float delta = static_cast<float>(i) - taps / 2;
      weights[kWeightOffset + i] = pattern == 3 ? (i == taps / 2 ? 1.0f : 0.0f)
          : pattern == 4 ? 0.03f + (i % 3) * 0.11f : std::exp(-0.07f * delta * delta);
    }
    std::mt19937 rng(78271u + width + 31 * height + pattern);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    constexpr float boundaries[] = {0.04f, 0.1f, 0.29f, 1.5f, 5.19175294647f,
                                     28.4691806922f, 0.132f / 2.155f};
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        float value = random(rng);
        if (pattern == 0) value = (x + y) % 2 ? -0.0f : 0.0f;
        if (pattern == 1) value *= 0.15f;
        if (pattern == 2) value *= 100.0f;
        if (pattern == 3) {
          value = boundaries[(x + y) % 7];
          if ((x + y) % 3 == 1) value = std::nextafter(value, 0.0f);
          if ((x + y) % 3 == 2) value = std::nextafter(value, 100.0f);
          if ((x + y) % 2) value = -value;
        }
        if (pattern == 4) value *= (x + y) % 3 == 0 ? 1.0e-37f : 1.0e19f;
        input[kInputOffset + static_cast<size_t>(y) * params.input_stride + x] = value;
      }
    }
  }
};

struct DeviceCase {
  DeviceArray input, output, work, blurred, weights;
  explicit DeviceCase(const Case& c)
      : input(c.input), output(c.output), work(c.work), blurred(c.blurred), weights(c.weights) {}
  void Launch(const Case& c, bool reference, cudaStream_t stream = nullptr) {
    using namespace gjxl::cuda_internal;
    if (reference) {
      CheckCuda(LaunchCudaButteraugliBlurAndSplitReference(
          input.data + Case::kInputOffset, weights.data + Case::kWeightOffset,
          work.data + Case::kWorkOffset, blurred.data + Case::kBlurOffset, c.blur_stride,
          output.data + Case::kOutputOffset, c.params, stream));
    } else {
      CheckCuda(LaunchCudaButteraugliBlurAndSplit(
          input.data + Case::kInputOffset, weights.data + Case::kWeightOffset,
          work.data + Case::kWorkOffset, output.data + Case::kOutputOffset, c.params, stream));
    }
  }
};

void Verify(uint32_t width, uint32_t height, uint32_t channel, unsigned pattern) {
  Case c(width, height, channel, pattern);
  DeviceCase reference(c), candidate(c);
  for (unsigned reuse = 0; reuse < 3; ++reuse) {
    if (reuse == 2) {
      auto changed = reference.input.Read();
      for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
          float& value = changed[Case::kInputOffset + static_cast<size_t>(y) * c.params.input_stride + x];
          value = -0.75f * value + 0.013f;
        }
      reference.input.Write(changed);
      candidate.input.Write(changed);
    }
    reference.Launch(c, true);
    candidate.Launch(c, false);
    CheckCuda(cudaDeviceSynchronize());
    Equal("low input", reference.input.Read(), candidate.input.Read());
    Equal("high output", reference.output.Read(), candidate.output.Read());
    Equal("horizontal intermediate", reference.work.Read(), candidate.work.Read());
    Equal("unused candidate blur", c.blurred, candidate.blurred.Read());
    for (auto* device : {&reference, &candidate}) {
      Guards(device->input.Read(), c.input, Case::kInputOffset, c.params.input_stride, width, height);
      Guards(device->output.Read(), c.output, Case::kOutputOffset, c.params.output_stride, width, height);
      Guards(device->work.Read(), c.work, Case::kWorkOffset, width, width, height);
      Equal("immutable weights", c.weights, device->weights.Read());
    }
    Guards(reference.blurred.Read(), c.blurred, Case::kBlurOffset, c.blur_stride, width, height);
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    CheckCuda(cudaSetDevice(0));
    using namespace gjxl::cuda_internal;
    for (uint32_t channel : {0u, 1u, 3u, 4u}) {
      for (bool zero_width : {false, true}) {
        const CudaButteraugliFrequencyParams empty{
            zero_width ? 0u : 17u, zero_width ? 17u : 0u, 17, 17, channel};
        CheckCuda(LaunchCudaButteraugliBlurAndSplit(nullptr, nullptr, nullptr, nullptr, empty, nullptr));
        CheckCuda(LaunchCudaButteraugliBlurAndSplitReference(nullptr, nullptr, nullptr, nullptr, 0, nullptr, empty, nullptr));
      }
    }
    for (uint32_t channel : {2u, 5u, 99u}) {
      const CudaButteraugliFrequencyParams invalid{1, 1, 1, 1, channel};
      if (LaunchCudaButteraugliBlurAndSplit(nullptr, nullptr, nullptr, nullptr, invalid, nullptr) != cudaErrorInvalidValue ||
          LaunchCudaButteraugliBlurAndSplitReference(nullptr, nullptr, nullptr, nullptr, 0, nullptr, invalid, nullptr) != cudaErrorInvalidValue)
        throw std::runtime_error("Unsupported channel not rejected");
    }
    constexpr std::array<std::array<uint32_t, 2>, 16> shapes{{
        {1,1}, {1,19}, {19,1}, {7,11}, {15,15}, {31,63}, {32,64}, {33,65},
        {63,31}, {65,33}, {127,65}, {255,3}, {256,4}, {257,67}, {511,129}, {3,257}}};
    unsigned cases = 0;
    for (const auto& shape : shapes)
      for (uint32_t channel : {0u, 1u, 3u, 4u})
        for (unsigned pattern = 0; pattern < 5; ++pattern) {
          try { Verify(shape[0], shape[1], channel, pattern); }
          catch (...) {
            std::cerr << "Case " << shape[0] << 'x' << shape[1] << " channel=" << channel
                      << " pattern=" << pattern << '\n';
            throw;
          }
          ++cases;
          if (cases % 20 == 0)
            std::cout << "Verified frequency geometry " << shape[0] << 'x' << shape[1]
                      << " (" << cases << " cases)\n" << std::flush;
        }
    std::cout << "Verified " << cases << " guarded blur/frequency cases with three-stage reuse\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
