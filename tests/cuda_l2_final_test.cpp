// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gpu/cuda/cuda_butteraugli_kernels.h"

namespace {
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

void Equal(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::runtime_error("Array size mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      std::cerr << "Mismatch at " << i << ": " << a[i] << " / " << b[i] << '\n';
      throw std::runtime_error("Bitwise L2/final mismatch");
    }
  }
}

struct Case {
  // 8 reference, 8 distorted, 3 AC, 3 DC, 3 mask, and one output plane.
  std::array<std::vector<float>, 26> planes;
  std::array<uint32_t, 26> strides{};
  gjxl::cuda_internal::CudaButteraugliL2FinalPlan plan;
  static size_t Offset(size_t plane) { return 3 + plane % 7; }

  Case(uint32_t width, uint32_t height, bool padded, unsigned pattern, float asymmetry) {
    plan.width = width;
    plan.height = height;
    plan.reference_stride = width + (padded ? 11 : 0);
    plan.distorted_stride = width + (padded ? 17 : 0);
    plan.work_stride = width + (padded ? 23 : 0);
    plan.output_stride = width + (padded ? 29 : 0);
    plan.asymmetry = asymmetry;
    plan.x_multiplier = pattern % 3 == 0 ? 0.0f : pattern % 3 == 1 ? 1.0f : 3.7f;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 rng(3871u + width + height * 31 + pattern);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    for (size_t p = 0; p < planes.size(); ++p) {
      strides[p] = p < 8 ? plan.reference_stride : p < 16 ? plan.distorted_stride
                      : p == 25 ? plan.output_stride : plan.work_stride;
      planes[p].assign(Offset(p) + static_cast<size_t>(strides[p]) * height + 31, poison);
      if ((p >= 18 && p <= 21) || p == 25) continue;
      for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
          float value = random(rng);
          if (p >= 16) value = std::fabs(value) * (p >= 22 ? 10.0f : 0.3f);
          if (pattern == 0) value = (x + y) % 2 ? -0.0f : 0.0f;
          if (p < 16 && pattern == 2) value *= (x + y) % 3 ? 1.0e-36f : 1.0e18f;
          if (p < 16 && pattern == 3 && (x + y + p) % 7 == 0)
            value = (x + p) % 2 ? poison : std::numeric_limits<float>::infinity();
          if (p < 16 && pattern == 4) {
            const float sign = (x + y) % 2 ? -1.0f : 1.0f;
            value = sign;
            if (p >= 8) {
              const float boundary = (x + y) % 3 ? sign * 0.4f : sign;
              value = (x + y) % 3 == 0 ? boundary : std::nextafter(
                  boundary, (x + y) % 3 == 1 ? 0.0f : sign * 2.0f);
            }
          }
          planes[p][Offset(p) + static_cast<size_t>(y) * strides[p] + x] = value;
        }
      }
    }
  }
};

struct DeviceCase {
  std::array<std::unique_ptr<DeviceArray>, 26> planes;
  gjxl::cuda_internal::CudaButteraugliL2FinalPlan plan;
  explicit DeviceCase(const Case& c) : plan(c.plan) {
    for (size_t p = 0; p < planes.size(); ++p)
      planes[p] = std::make_unique<DeviceArray>(c.planes[p]);
    const auto pointer = [&](size_t p) { return planes[p]->data + Case::Offset(p); };
    for (size_t p = 0; p < 8; ++p) {
      plan.reference[p] = pointer(p);
      plan.distorted[p] = pointer(8 + p);
    }
    for (size_t p = 0; p < 3; ++p) {
      plan.ac[p] = pointer(16 + p);
      plan.dc[p] = pointer(19 + p);
    }
    plan.mask = pointer(22);
    plan.mask_reference = pointer(23);
    plan.mask_distorted = pointer(24);
    plan.output = pointer(25);
  }
  void Launch(bool reference, cudaStream_t stream = nullptr) {
    using namespace gjxl::cuda_internal;
    CheckCuda(reference ? LaunchCudaButteraugliL2FinalReference(plan, stream)
                        : LaunchCudaButteraugliL2Final(plan, stream));
  }
};

void Verify(uint32_t width, uint32_t height, bool padded, unsigned pattern, float asymmetry) {
  Case c(width, height, padded, pattern, asymmetry);
  DeviceCase reference(c), candidate(c);
  for (unsigned reuse = 0; reuse < 3; ++reuse) {
    if (reuse == 2) {
      for (size_t p = 8; p < 16; ++p)
        for (uint32_t y = 0; y < height; ++y)
          for (uint32_t x = 0; x < width; ++x) {
            auto& value = c.planes[p][Case::Offset(p) + static_cast<size_t>(y) * c.strides[p] + x];
            value = -0.75f * value + 0.013f;
          }
      // Unused fused scratch is genuinely optional, not merely poisoned.
      candidate.plan.ac[2] = nullptr;
      candidate.plan.dc.fill(nullptr);
    }
    for (size_t p = 0; p < c.planes.size(); ++p) {
      reference.planes[p]->Write(c.planes[p]);
      candidate.planes[p]->Write(c.planes[p]);
    }
    reference.Launch(true);
    candidate.Launch(false);
    CheckCuda(cudaDeviceSynchronize());
    for (size_t p = 0; p < c.planes.size(); ++p) {
      const auto old_values = reference.planes[p]->Read();
      const auto new_values = candidate.planes[p]->Read();
      if (p == 25) Equal(old_values, new_values);
      else Equal(c.planes[p], new_values);
      if (p < 16 || (p >= 22 && p < 25)) Equal(c.planes[p], old_values);
      for (size_t i = 0; i < old_values.size(); ++i) {
        const bool active = i >= Case::Offset(p) &&
            i < Case::Offset(p) + static_cast<size_t>(c.strides[p]) * height &&
            (i - Case::Offset(p)) % c.strides[p] < width;
        if (!active && (std::memcmp(&old_values[i], &c.planes[p][i], sizeof(float)) != 0 ||
                        std::memcmp(&new_values[i], &c.planes[p][i], sizeof(float)) != 0))
          throw std::runtime_error("L2/final padding guard overwritten");
      }
    }
  }
}
struct NonblockingStream {
  cudaStream_t value = nullptr;
  NonblockingStream() {
    CheckCuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }
  ~NonblockingStream() { (void)cudaStreamDestroy(value); }
};

Case ErosionCase(uint32_t width, uint32_t height, bool padded,
                 unsigned pattern, float asymmetry) {
  Case c(width, height, padded, pattern % 5, asymmetry);
  c.plan.x_multiplier = pattern % 3 == 0 ? 0.0f : pattern % 3 == 1 ? 1.0f : 3.7f;
  std::mt19937 rng(58321u + width + 31 * height + pattern);
  const std::array<float, 11> special{
      0.0f, -0.0f, std::numeric_limits<float>::min(),
      -std::numeric_limits<float>::min(), std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(), std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      float& value = c.planes[23][Case::Offset(23) +
          static_cast<size_t>(y) * c.strides[23] + x];
      if (pattern == 0) value = (x + y) % 2 ? -0.0f : 0.0f;
      if (pattern == 2)
        value = 0.1f + static_cast<float>(x) / width +
                 0.3f * static_cast<float>(y) / height;
      if (pattern == 3 && (x + y) % 2) value = -value;
      if (pattern == 4) value = special[(x + 7 * y) % special.size()];
      if (pattern == 5) {
        const uint32_t bits = rng();
        std::memcpy(&value, &bits, sizeof(value));
      }
    }
  }
  return c;
}

void VerifyErosion(uint32_t width, uint32_t height, bool padded,
                   unsigned pattern, float asymmetry) {
  using namespace gjxl::cuda_internal;
  Case c = ErosionCase(width, height, padded, pattern, asymmetry);
  DeviceCase reference(c), candidate(c);
  NonblockingStream stream;
  // The fused path does not read the eroded-mask plane or unused L2 scratch.
  reference.plan.mask = candidate.plan.mask = nullptr;
  candidate.plan.ac[2] = nullptr;
  candidate.plan.dc.fill(nullptr);
  for (unsigned reuse = 0; reuse < 3; ++reuse) {
    if (reuse == 2) {
      for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
          float& value = c.planes[23][Case::Offset(23) +
              static_cast<size_t>(y) * c.strides[23] + x];
          value = -0.75f * value + 0.013f;
        }
    }
    for (size_t p = 0; p < c.planes.size(); ++p) {
      reference.planes[p]->Write(c.planes[p]);
      candidate.planes[p]->Write(c.planes[p]);
    }
    // Pageable copies on the default stream can finish after returning to
    // the host. Explicitly order them before the nonblocking test stream.
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(LaunchCudaButteraugliErosionFinalForTesting(
        reference.plan, reference.planes[22]->data + Case::Offset(22),
        true, stream.value));
    CheckCuda(LaunchCudaButteraugliErosionFinalForTesting(
        candidate.plan, nullptr, false, stream.value));
    CheckCuda(cudaStreamSynchronize(stream.value));
    for (size_t p = 0; p < c.planes.size(); ++p) {
      const auto expected = reference.planes[p]->Read();
      const auto actual = candidate.planes[p]->Read();
      if (p == 25) Equal(expected, actual);
      else Equal(c.planes[p], actual);
      if (p != 22 && p != 25) Equal(c.planes[p], expected);
      for (size_t i = 0; i < expected.size(); ++i) {
        const bool active = i >= Case::Offset(p) &&
            i < Case::Offset(p) + static_cast<size_t>(c.strides[p]) * height &&
            (i - Case::Offset(p)) % c.strides[p] < width;
        if (!active &&
            (std::memcmp(&expected[i], &c.planes[p][i], sizeof(float)) != 0 ||
             std::memcmp(&actual[i], &c.planes[p][i], sizeof(float)) != 0))
          throw std::runtime_error("Erosion/final padding guard overwritten");
      }
    }
  }
}

void VerifyErosionCases(bool small) {
  using namespace gjxl::cuda_internal;
  for (bool zero_width : {false, true}) {
    CudaButteraugliL2FinalPlan empty;
    empty.width = zero_width ? 0 : 17;
    empty.height = zero_width ? 17 : 0;
    for (bool reference : {false, true})
      CheckCuda(LaunchCudaButteraugliErosionFinalForTesting(
          empty, nullptr, reference, nullptr));
  }
  for (size_t invalid = 0; invalid < 4; ++invalid) {
    CudaButteraugliL2FinalPlan malformed;
    malformed.width = malformed.height = 1;
    malformed.reference_stride = malformed.distorted_stride = 1;
    malformed.work_stride = malformed.output_stride = 1;
    uint32_t* const strides[] = {&malformed.reference_stride,
        &malformed.distorted_stride, &malformed.work_stride, &malformed.output_stride};
    *strides[invalid] = 0;
    for (bool reference : {false, true})
      if (LaunchCudaButteraugliErosionFinalForTesting(
              malformed, nullptr, reference, nullptr) != cudaErrorInvalidValue)
        throw std::runtime_error("Invalid erosion/final stride not rejected");
  }
  CudaButteraugliL2FinalPlan no_scratch;
  no_scratch.width = no_scratch.height = 1;
  no_scratch.reference_stride = no_scratch.distorted_stride = 1;
  no_scratch.work_stride = no_scratch.output_stride = 1;
  if (LaunchCudaButteraugliErosionFinalForTesting(
          no_scratch, nullptr, true, nullptr) != cudaErrorInvalidValue)
    throw std::runtime_error("Null erosion reference scratch not rejected");
  constexpr std::array<std::array<uint32_t, 2>, 12> shapes{{
      {1,1}, {1,19}, {19,1}, {7,11}, {31,63}, {32,64},
      {33,65}, {63,31}, {127,65}, {255,3}, {256,4}, {257,67}}};
  size_t cases = 0;
  for (const auto& shape : shapes) {
    if (small && shape != std::array<uint32_t, 2>{33,65}) continue;
    for (bool padded : {false, true}) {
      for (unsigned pattern = 0; pattern < 6; ++pattern) {
        if (small && pattern != 1 && pattern != 3 && pattern != 4 && pattern != 5)
          continue;
        for (float asymmetry : {0.6f, 1.0f, 2.5f}) {
          if (small && asymmetry != 1.0f) continue;
          VerifyErosion(shape[0], shape[1], padded, pattern, asymmetry);
          ++cases;
        }
      }
    }
    std::cout << "Verified erosion/final geometry " << shape[0] << 'x' << shape[1]
              << " (" << cases << " cases)\n" << std::flush;
  }
  if (cases != (small ? 8 : 432)) throw std::runtime_error("Erosion coverage differs");
  std::cout << "Verified " << cases
            << " guarded erosion/final cases with three-stage nondefault-stream reuse\n";
}
}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  const bool small = argc > 1 && std::string_view(argv[1]) == "--sanitizer";
  try {
    CheckCuda(cudaSetDevice(0));
    if (argc > 1 && std::string_view(argv[1]) == "--erosion-sanitizer") {
      VerifyErosionCases(true);
      return 0;
    }
    using namespace gjxl::cuda_internal;
    for (bool zero_width : {false, true}) {
      CudaButteraugliL2FinalPlan empty;
      empty.width = zero_width ? 0 : 17;
      empty.height = zero_width ? 17 : 0;
      CheckCuda(LaunchCudaButteraugliL2Final(empty, nullptr));
      CheckCuda(LaunchCudaButteraugliL2FinalReference(empty, nullptr));
    }
    for (size_t invalid = 0; invalid < 4; ++invalid) {
      CudaButteraugliL2FinalPlan malformed;
      malformed.width = malformed.height = 1;
      malformed.reference_stride = malformed.distorted_stride = 1;
      malformed.work_stride = malformed.output_stride = 1;
      uint32_t* const strides[] = {&malformed.reference_stride,
          &malformed.distorted_stride, &malformed.work_stride, &malformed.output_stride};
      *strides[invalid] = 0;
      if (LaunchCudaButteraugliL2Final(malformed, nullptr) != cudaErrorInvalidValue ||
          LaunchCudaButteraugliL2FinalReference(malformed, nullptr) != cudaErrorInvalidValue)
        throw std::runtime_error("Invalid L2/final stride not rejected");
    }
    constexpr std::array<std::array<uint32_t, 2>, 12> shapes{{
        {1,1}, {1,19}, {19,1}, {7,11}, {31,63}, {32,64},
        {33,65}, {63,31}, {127,65}, {255,3}, {256,4}, {257,67}}};
    size_t cases = 0;
    for (const auto& shape : shapes) {
      if (small && shape[0] != 1 && shape[0] != 33) continue;
      for (bool padded : {false, true})
        for (unsigned pattern = 0; pattern < 5; ++pattern)
          for (float asymmetry : {0.6f, 1.0f, 2.5f}) {
            Verify(shape[0], shape[1], padded, pattern, asymmetry);
            ++cases;
          }
      std::cout << "Verified L2/final geometry " << shape[0] << 'x' << shape[1]
                << " (" << cases << " cases)\n" << std::flush;
    }
    std::cout << "Verified " << cases << " guarded L2/final cases with three-stage reuse\n";
    VerifyErosionCases(small);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
