// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "gpu/cuda/cuda_aq_exact_kernels.h"
#include "gpu/cuda/cuda_aq_resident_kernels.h"
#include "gpu/cuda/cuda_kernels.h"

namespace {

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
struct DeviceArray {
  explicit DeviceArray(const std::vector<T>& values) : count(values.size()) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    const cudaError_t status = cudaMemcpy(data, values.data(), count * sizeof(T),
                                         cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      (void)cudaFree(data);
      CheckCuda(status);
    }
  }
  ~DeviceArray() { (void)cudaFree(data); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  std::vector<T> Read() const {
    std::vector<T> result(count);
    CheckCuda(cudaMemcpy(result.data(), data, count * sizeof(T),
                         cudaMemcpyDeviceToHost));
    return result;
  }
  T* data = nullptr;
  size_t count;
};

template <typename T>
bool Equal(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

void Verify(uint32_t width, uint32_t height, uint32_t count, unsigned pattern) {
  using namespace gjxl::cuda_internal;
  constexpr uint32_t kAnchorOffset = 3, kCoefficientOffset = 19;
  constexpr uint32_t kPixelOffset = 11;
  const uint32_t columns = std::min(count, 5u);
  const uint32_t rows = (count + columns - 1) / columns;
  const uint32_t source_stride = columns * (width + 8) + 21;
  const uint32_t output_stride = source_stride + 6;
  const uint32_t image_height = rows * (height + 8) + 16;
  const size_t source_plane = kPixelOffset +
    static_cast<size_t>(source_stride) * image_height + 23;
  const size_t output_plane = kPixelOffset +
    static_cast<size_t>(output_stride) * image_height + 23;
  const size_t coefficient_count = 3 * static_cast<size_t>(count) * width * height;
  const float poison = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> source(3 * source_plane, poison);
  std::vector<float> output(3 * output_plane, -12345.25f);
  std::vector<float> coefficients(kCoefficientOffset + coefficient_count + 29,
                                 poison);
  std::vector<CudaAqAnchor> anchors(kAnchorOffset + count + 5,
                                    {0xffffffffu, 0xffffffffu});
  std::mt19937 rng(127u + count * 31 + width * height + pattern);
  std::uniform_real_distribution<float> random(-2, 2);
  for (uint32_t i = 0; i < count; ++i) {
    // Deliberately permute anchor order and leave holes around each rectangle.
    const uint32_t position = count - 1 - i;
    const CudaAqAnchor anchor{
      1 + (position % columns) * (width / 8 + 1),
      1 + (position / columns) * (height / 8 + 1)};
    anchors[kAnchorOffset + i] = anchor;
    for (size_t c = 0; c < 3; ++c) {
      for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
          float value = random(rng);
          if (pattern == 0) value = (x + y + c) % 2 ? -0.0f : 0.0f;
          if (pattern == 1) value = static_cast<float>(c + 1) * 0.125f;
          if (pattern == 2) value = x == i % width && y == i % height ? 1 : 0;
          if (pattern == 3) value *= 0.00001f;
          if (pattern == 4) value *= 10000;
          source[c * source_plane + kPixelOffset +
            static_cast<size_t>(anchor.y * 8 + y) * source_stride +
            anchor.x * 8 + x] = value;
        }
      }
    }
  }
  const CudaAqExactBatch batch{kAnchorOffset, count, kCoefficientOffset,
    width * height, width, height, width / 8, height / 8};
  DeviceArray<float> ds(source), packed(coefficients), reference(coefficients),
    fused(coefficients), inverse(coefficients), dr(output), df(output);
  DeviceArray<CudaAqAnchor> da(anchors);
  const std::array<const float*, 3> coding{
    ds.data + kPixelOffset, ds.data + source_plane + kPixelOffset,
    ds.data + 2 * source_plane + kPixelOffset};
  const std::array<float*, 3> old_output{
    dr.data + kPixelOffset, dr.data + output_plane + kPixelOffset,
    dr.data + 2 * output_plane + kPixelOffset};
  const std::array<float*, 3> new_output{
    df.data + kPixelOffset, df.data + output_plane + kPixelOffset,
    df.data + 2 * output_plane + kPixelOffset};
  CheckCuda(LaunchCudaAqGatherTransformPixels(coding[0], coding[1], coding[2],
    da.data, packed.data, batch, source_stride, nullptr));
  CheckCuda(LaunchCudaDct(true, packed.data + kCoefficientOffset,
    reference.data + kCoefficientOffset, 3 * count, width, height, nullptr));
  CheckCuda(LaunchCudaAqForwardDct(coding, da.data, fused.data, source_stride,
    batch, nullptr));
  const auto expected = reference.Read();
  if (!Equal(expected, fused.Read()))
    throw std::runtime_error("Resident forward DCT is not bitwise identical");
  for (size_t i = 0; i < expected.size(); ++i) {
    if (i >= kCoefficientOffset && i < kCoefficientOffset + coefficient_count) {
      if (!std::isfinite(expected[i]))
        throw std::runtime_error("Forward DCT did not overwrite active poison");
    } else if (std::memcmp(&expected[i], &coefficients[i], sizeof(float)) != 0) {
      throw std::runtime_error("Forward DCT coefficient guard overwritten");
    }
  }
  // Also exercise arbitrary, quantized-like inverse inputs, not only a
  // forward/inverse round trip that might hide a coefficient-layout error.
  for (size_t i = 0; i < coefficient_count; ++i)
    coefficients[kCoefficientOffset + i] = pattern == 0 ? 0.0f :
      std::round(random(rng) * 32) * (pattern == 3 ? 0.00001f : 0.125f);
  DeviceArray<float> dc(coefficients);
  for (unsigned reuse = 0; reuse < 2; ++reuse) {
    CheckCuda(LaunchCudaDct(false, dc.data + kCoefficientOffset,
      inverse.data + kCoefficientOffset, 3 * count, width, height, nullptr));
    CheckCuda(LaunchCudaAqScatterReconstruction(da.data, inverse.data,
      old_output, output_stride, batch, nullptr));
    CheckCuda(LaunchCudaAqInverseDct(dc.data, da.data, new_output,
      output_stride, batch, nullptr));
    const auto actual = df.Read();
    if (!Equal(dr.Read(), actual))
      throw std::runtime_error("Resident inverse DCT is not bitwise identical");
    auto guarded = output;
    const auto inverse_pixels = inverse.Read();
    for (size_t c = 0; c < 3; ++c)
      for (uint32_t i = 0; i < count; ++i)
        for (uint32_t y = 0; y < height; ++y)
          for (uint32_t x = 0; x < width; ++x) {
            const auto anchor = anchors[kAnchorOffset + i];
            const size_t destination = c * output_plane + kPixelOffset +
              static_cast<size_t>(anchor.y * 8 + y) * output_stride +
              anchor.x * 8 + x;
            guarded[destination] = inverse_pixels[kCoefficientOffset +
              (c * count + i) * width * height + y * width + x];
            if (!std::isfinite(actual[destination]))
              throw std::runtime_error("Inverse DCT output is nonfinite");
          }
    if (!Equal(guarded, actual))
      throw std::runtime_error("Inverse image layout or guard mismatch");
  }
  if (!Equal(ds.Read(), source) || !Equal(da.Read(), anchors) ||
      !Equal(dc.Read(), coefficients))
    throw std::runtime_error("Resident DCT input overwritten");
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "CUDA device unavailable\n";
    return 77;
  }
  try {
    CheckCuda(cudaSetDevice(0));
    constexpr std::array<std::array<uint32_t, 2>, 7> shapes{{
      {8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 16}, {16, 32}, {32, 32}}};
    size_t cases = 0;
    for (const auto& shape : shapes) {
      for (uint32_t count : {1u, 2u, 3u, 5u, 9u, 17u})
        for (unsigned pattern = 0; pattern < 6; ++pattern) {
          Verify(shape[0], shape[1], count, pattern);
          ++cases;
        }
      gjxl::cuda_internal::CudaAqExactBatch empty;
      empty.pixel_width = shape[0];
      empty.pixel_height = shape[1];
      CheckCuda(gjxl::cuda_internal::LaunchCudaAqForwardDct(
        {}, nullptr, nullptr, 0, empty, nullptr));
      CheckCuda(gjxl::cuda_internal::LaunchCudaAqInverseDct(
        nullptr, nullptr, {}, 0, empty, nullptr));
    }
    gjxl::cuda_internal::CudaAqExactBatch unsupported;
    unsupported.pixel_width = unsupported.pixel_height = 64;
    if (gjxl::cuda_internal::LaunchCudaAqForwardDct(
          {}, nullptr, nullptr, 0, unsupported, nullptr) != cudaErrorInvalidValue ||
        gjxl::cuda_internal::LaunchCudaAqInverseDct(
          nullptr, nullptr, {}, 0, unsupported, nullptr) != cudaErrorInvalidValue)
      throw std::runtime_error("Unsupported resident DCT shape accepted");
    std::cout << "Verified " << cases << " guarded resident DCT cases\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
