// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "codec/quantization.h"
#include "core/ac_strategy.h"
#include "gpu/cuda/cuda_aq_resident_kernels.h"

namespace {

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
struct DeviceArray {
  explicit DeviceArray(const std::vector<T>& values) : count(values.size()) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    CheckCuda(cudaMemcpy(data, values.data(), count * sizeof(T),
                         cudaMemcpyHostToDevice));
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

constexpr std::array kStrategies = {
    gjxl::AcStrategyType::kDct8,     gjxl::AcStrategyType::kDct16x16,
    gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
    gjxl::AcStrategyType::kDct8x16,  gjxl::AcStrategyType::kDct32x16,
    gjxl::AcStrategyType::kDct16x32};

bool CheckCase(gjxl::AcStrategyType strategy, uint32_t count, uint32_t scale,
               const std::array<float, 3>& multipliers, unsigned invalid_mode) {
  using namespace gjxl::cuda_internal;
  const auto extent = gjxl::GetAcStrategyInfo(strategy)->pixel_extent();
  const uint32_t width =
      static_cast<uint32_t>(std::max(extent.width, extent.height));
  const uint32_t height =
      static_cast<uint32_t>(std::min(extent.width, extent.height));
  const uint32_t n = width * height;
  constexpr uint32_t kCoefficientOffset = 13, kAnchorOffset = 3, kRawOffset = 7;
  constexpr int kRawGuard = 0x12345678;
  constexpr float kGuard = -9999.0f;
  constexpr unsigned kInitialError = 0x20000000u;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const uint32_t stride = 2 * count + 5;
  std::vector<CudaAqAnchor> anchors(kAnchorOffset + count + 5,
                                    {UINT32_MAX, UINT32_MAX});
  std::vector<int> raw(kRawOffset + 3 * stride + 11, kRawGuard);
  for (uint32_t a = 0; a < count; ++a) {
    anchors[kAnchorOffset + a] = {2 * a + 1, 1};
    raw[kRawOffset + stride + 2 * a + 1] =
        a % 12 == 10   ? 256
        : a % 12 == 11 ? 255
                       : 1 + static_cast<int>(a * 37 % 256);
  }
  // Non-used table entries are poisonous so wrong shape/channel addressing
  // fails.
  std::vector<float> tables(11904, nan);
  const uint32_t table_offset = n == 64    ? 192
                                : n == 128 ? 8448
                                : n == 256 ? 1152
                                : n == 512 ? 10368
                                           : 4992;
  std::array<gjxl::QuantizationMatrixView, 3> matrices;
  for (unsigned c = 0; c < 3; ++c) {
    if (!gjxl::GetDefaultQuantizationMatrix(
             strategy, static_cast<gjxl::XybChannel>(c), &matrices[c])
             .ok())
      return false;
    std::copy(matrices[c].inverse_dequant.begin(),
              matrices[c].inverse_dequant.end(),
              tables.begin() + table_offset + c * n);
  }
  gjxl::Quantizer quantizer;
  if (!gjxl::Quantizer::Create({scale, 10}, &quantizer).ok()) return false;
  std::mt19937 random(314159u + n + count + scale);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::vector<float> coefficients(
      kCoefficientOffset + 3 * static_cast<size_t>(count) * n + 11, kGuard);
  for (unsigned c = 0; c < 3; ++c) {
    for (uint32_t a = 0; a < count; ++a) {
      const int initial = raw[kRawOffset + stride + 2 * a + 1];
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t x = i % width, y = i / width;
        float value = unit(random);
        switch (a % 12) {
          case 0:
            value = 0.0f;
            break;
          case 1:
            value = i == n - 1 ? 0.53f : 0.0f;
            break;
          case 2:
            value = i == n - 1
                        ? std::max(0.54f,
                                   0.64f - (n > 64 ? 0.003f * (n / 64) : 0.0f))
                        : 0.0f;
            break;
          case 3:
            value = x + 1 == width || y + 1 == height ? value * 3.0f : 0.0f;
            break;
          case 4:
            value *= 3.0f;
            break;
          case 5:
            value *= 0.7f;
            break;
          case 6:
            value *= 50.0f;
            break;
          case 7:
            value *= 0.501f;
            break;
          case 8:
            value = (i % 2 == 0 ? 1.5f : -1.5f);
            break;
          case 9:
            value = (i % 2 == 0 ? 0.0f : -0.0f);
            break;
          default:
            value *= 8.0f;
            break;
        }
        coefficients[kCoefficientOffset +
                     (static_cast<size_t>(c) * count + a) * n + i] =
            x < width / 8 && y < height / 8
                ? 0.0f
                : value / (matrices[c].inverse_dequant[i] *
                           (quantizer.scale() * initial) * multipliers[c]);
      }
    }
  }
  // AC non-finites report an error; low-frequency/DC non-finites are skipped.
  if (invalid_mode != 0) {
    for (unsigned c = 0; c < 3; ++c) {
      coefficients[kCoefficientOffset + static_cast<size_t>(c) * count * n +
                   (invalid_mode == 1 ? n - 1 : 0)] =
          c == 0   ? nan
          : c == 1 ? std::numeric_limits<float>::infinity()
                   : -std::numeric_limits<float>::infinity();
    }
  }
  std::vector<float> thresholds(kCoefficientOffset + 4 * count + 11, kGuard);
  DeviceArray<CudaAqAnchor> device_anchors(anchors);
  DeviceArray<float> device_tables(tables), device_coefficients(coefficients);
  DeviceArray<unsigned> device_quantizer(std::vector<unsigned>{scale});
  CudaAqExactBatch batch{};
  batch.anchor_offset = kAnchorOffset;
  batch.anchor_count = count;
  batch.coefficient_offset = kCoefficientOffset;
  batch.coefficient_count = n;
  batch.pixel_width = static_cast<uint32_t>(extent.width);
  batch.pixel_height = static_cast<uint32_t>(extent.height);
  CudaAqResidentParams params{};
  params.block_width = stride;
  params.block_height = 3;
  params.strategy = static_cast<uint32_t>(strategy);
  params.x_matrix_multiplier = multipliers[0];
  params.b_matrix_multiplier = multipliers[2];
  std::vector<int> expected_raw;
  std::vector<float> expected_thresholds;
  for (const auto launch : {LaunchCudaAqSelectAdjustedQuantizationScalar,
                            LaunchCudaAqSelectAdjustedQuantization}) {
    DeviceArray<int> device_raw(raw);
    DeviceArray<float> device_thresholds(thresholds);
    DeviceArray<unsigned> device_error(std::vector<unsigned>{kInitialError});
    CheckCuda(launch(device_anchors.data, device_tables.data,
                     device_raw.data + kRawOffset, device_coefficients.data,
                     device_thresholds.data, device_quantizer.data,
                     device_error.data, batch, params, nullptr));
    CheckCuda(cudaDeviceSynchronize());
    const auto actual_raw = device_raw.Read();
    const auto actual_thresholds = device_thresholds.Read();
    if (device_error.Read()[0] !=
        (kInitialError | (invalid_mode == 1 ? 16u : 0u)))
      return false;
    for (size_t i = 0; i < raw.size(); ++i) {
      if (raw[i] == kRawGuard && actual_raw[i] != kRawGuard) return false;
    }
    for (size_t i = 0; i < thresholds.size(); ++i) {
      if ((i < kCoefficientOffset || i >= kCoefficientOffset + 4 * count) &&
          actual_thresholds[i] != kGuard)
        return false;
    }
    if (expected_raw.empty()) {
      expected_raw = actual_raw;
      expected_thresholds = actual_thresholds;
    } else if (actual_raw != expected_raw ||
               std::memcmp(actual_thresholds.data(), expected_thresholds.data(),
                           thresholds.size() * sizeof(float)) != 0) {
      std::cerr << "Cooperative decisions differ from scalar oracle\n";
      return false;
    }
    if (invalid_mode == 0) {
      for (uint32_t a = 0; a < count; ++a) {
        std::array<std::span<const float>, 3> planes;
        for (unsigned c = 0; c < 3; ++c)
          planes[c] = {coefficients.data() + kCoefficientOffset +
                           (static_cast<size_t>(c) * count + a) * n,
                       n};
        gjxl::AdjustedAcQuantization cpu;
        const size_t raw_index = kRawOffset + stride + 2 * a + 1;
        const auto status = gjxl::SelectAdjustedAcQuantization(
            strategy, quantizer, raw[raw_index], multipliers, planes, &cpu);
        if (!status.ok() || actual_raw[raw_index] != cpu.raw_quant) {
          std::cerr << "Raw quant differs from CPU policy at anchor " << a
                    << " actual=" << actual_raw[raw_index]
                    << " expected=" << cpu.raw_quant
                    << " status=" << status.message() << '\n';
          return false;
        }
        for (unsigned q = 0; q < 4; ++q) {
          if (std::abs(actual_thresholds[kCoefficientOffset + 4 * a + q] -
                       cpu.y_thresholds[q]) > 2.0e-6f) {
            std::cerr << "Y threshold differs from CPU policy\n";
            return false;
          }
        }
      }
    }
  }
  const auto unchanged = device_coefficients.Read();
  return std::memcmp(unchanged.data(), coefficients.data(),
                     coefficients.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    CheckCuda(cudaSetDevice(0));
    size_t cases = 0;
    for (auto strategy : kStrategies) {
      for (uint32_t scale : {1u, 3541u, 32768u}) {
        for (uint32_t count : {1u, 2u, 3u, 7u, 17u, 33u, 129u, 257u}) {
          const std::array<float, 3> multipliers =
              scale == 3541 ? std::array<float, 3>{1.25f, 1.0f, 0.75f}
                            : std::array<float, 3>{0.5f, 1.0f, 2.0f};
          if (!CheckCase(strategy, count, scale, multipliers, 0)) {
            std::cerr << "strategy=" << static_cast<unsigned>(strategy)
                      << " count=" << count << " scale=" << scale << '\n';
            return 1;
          }
          ++cases;
        }
      }
      for (unsigned mode : {1u, 2u}) {
        if (!CheckCase(strategy, 17, 3541, {1.25f, 1.0f, 0.75f}, mode))
          return 1;
        ++cases;
      }
    }
    std::cout << "Checked " << cases
              << " guarded quantization-adjustment batches\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
