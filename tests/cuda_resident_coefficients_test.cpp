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
#include <stdexcept>
#include <string>
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

template <typename T>
void CheckEqual(const char* name, const std::vector<T>& expected,
                const std::vector<T>& actual) {
  if (expected.size() != actual.size())
    throw std::runtime_error("Coefficient output extent differs");
  if (Equal(expected, actual)) return;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::memcmp(&expected[i], &actual[i], sizeof(T)) != 0) {
      std::cerr << name << " mismatch at " << i << ": expected 0x" << std::hex
                << std::bit_cast<unsigned>(expected[i]) << ", actual 0x"
                << std::bit_cast<unsigned>(actual[i]) << std::dec << '\n';
      break;
    }
  }
  throw std::runtime_error("Resident coefficient fusion is not bitwise identical");
}

constexpr std::array kStrategies = {
    gjxl::AcStrategyType::kDct8, gjxl::AcStrategyType::kDct16x16,
    gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
    gjxl::AcStrategyType::kDct8x16, gjxl::AcStrategyType::kDct32x16,
    gjxl::AcStrategyType::kDct16x32};
constexpr uint32_t kBlockOffset = 7;
constexpr int kIntGuard = 0x12345678;
constexpr float kFloatGuard = -12345.25f;
constexpr unsigned kInitialError = 0x20000000u;

struct Case {
  gjxl::cuda_internal::CudaAqExactBatch batch{};
  gjxl::cuda_internal::CudaAqResidentParams params{};
  std::vector<gjxl::cuda_internal::CudaAqAnchor> anchors;
  std::vector<float> tables, forward, thresholds, reconstruction, dc, sigma;
  std::vector<int> raw, ac, quantized_dc;
  std::vector<signed char> cfl_x, cfl_b;
  std::vector<unsigned char> sharpness;
  std::vector<unsigned> quantizer;
  std::vector<bool> active_coefficients, active_blocks;

  Case(gjxl::AcStrategyType strategy, uint32_t count, unsigned pattern,
       bool adjust, unsigned invalid = 0, uint32_t columns_override = 0) {
    const auto extent = gjxl::GetAcStrategyInfo(strategy)->pixel_extent();
    const uint32_t w = static_cast<uint32_t>(extent.width);
    const uint32_t h = static_cast<uint32_t>(extent.height);
    const uint32_t n = w * h;
    batch = {3, count, 19, n, w, h, w / 8, h / 8};
    const uint32_t columns =
        std::min(count, columns_override ? columns_override : 7u);
    const uint32_t rows = (count + columns - 1) / columns;
    params.block_width = columns * (w / 8 + 1) + 5;
    params.block_height = rows * (h / 8 + 1) + 3;
    params.coding_stride = params.block_width * 8;
    params.color_stride = (params.block_width + 7) / 8 + 2;
    params.strategy = static_cast<uint32_t>(strategy);
    params.adjust_ac_quant = adjust;
    params.x_matrix_multiplier = pattern % 2 ? 1.25f : 0.5f;
    params.b_matrix_multiplier = pattern % 2 ? 0.75f : 2.0f;
    params.epf_quant_multiplier = 0.75f;
    for (unsigned i = 0; i < 8; ++i)
      params.epf_sharpness_lut[i] = 0.25f + 0.125f * i;
    const uint32_t blocks = params.block_width * params.block_height;
    const size_t color_count = params.color_stride *
                              ((params.block_height + 7) / 8 + 1);
    const size_t coefficient_size = batch.coefficient_offset +
                                    3 * static_cast<size_t>(count) * n + 23;
    anchors.resize(batch.anchor_offset + count + 5, {UINT32_MAX, UINT32_MAX});
    raw.resize(kBlockOffset + blocks + 11, kIntGuard);
    sharpness.resize(raw.size(), 255);
    cfl_x.resize(color_count);
    cfl_b.resize(color_count);
    for (size_t i = 0; i < color_count; ++i) {
      cfl_x[i] = static_cast<signed char>(static_cast<int>(i * 37 % 256) - 128);
      cfl_b[i] = static_cast<signed char>(127 - static_cast<int>(i * 53 % 256));
    }
    constexpr std::array<unsigned, 3> scales{1, 3541, 32768};
    quantizer = {scales[pattern % scales.size()], pattern % 2 ? 1u : 10u};
    tables.resize(11904, std::numeric_limits<float>::quiet_NaN());
    const uint32_t dequant = n == 64 ? 0 : n == 128 ? 8064 :
                            n == 256 ? 384 : n == 512 ? 8832 : 1920;
    const uint32_t inverse = dequant + 3 * n;
    for (unsigned c = 0; c < 3; ++c) {
      gjxl::QuantizationMatrixView matrix;
      if (!gjxl::GetDefaultQuantizationMatrix(
            strategy, static_cast<gjxl::XybChannel>(c), &matrix).ok())
        throw std::runtime_error("Default quantization matrix unavailable");
      std::copy(matrix.dequant.begin(), matrix.dequant.end(),
                tables.begin() + dequant + c * n);
      std::copy(matrix.inverse_dequant.begin(), matrix.inverse_dequant.end(),
                tables.begin() + inverse + c * n);
    }
    forward.resize(coefficient_size, kFloatGuard);
    reconstruction = forward;
    ac.resize(coefficient_size, kIntGuard);
    dc.resize(kBlockOffset + 3 * blocks + 11, kFloatGuard);
    quantized_dc.resize(dc.size(), kIntGuard);
    sigma.resize(raw.size(), kFloatGuard);
    thresholds.resize(batch.coefficient_offset + 4 * count + 11, kFloatGuard);
    active_coefficients.resize(coefficient_size);
    active_blocks.resize(raw.size());
    std::mt19937 rng(314159u + n + count + pattern);
    std::uniform_real_distribution<float> random(-1, 1);
    for (uint32_t a = 0; a < count; ++a) {
      const uint32_t position = count - 1 - a;
      const gjxl::cuda_internal::CudaAqAnchor anchor{
        1 + position % columns * (w / 8 + 1),
        1 + position / columns * (h / 8 + 1)};
      anchors[batch.anchor_offset + a] = anchor;
      const int raw_value = a % 3 == 0 ? 1 : a % 3 == 1 ? 256 : 73;
      raw[kBlockOffset + anchor.y * params.block_width + anchor.x] = raw_value;
      for (uint32_t y = 0; y < h / 8; ++y)
        for (uint32_t x = 0; x < w / 8; ++x) {
          const size_t block = kBlockOffset + (anchor.y + y) *
                              params.block_width + anchor.x + x;
          active_blocks[block] = true;
          sharpness[block] = static_cast<unsigned char>((a + x + y) % 8);
        }
      for (unsigned q = 0; q < 4; ++q)
        thresholds[batch.coefficient_offset + 4 * a + q] =
          q == 0 ? 0.54f : 0.58f + 0.02f * q;
      for (unsigned c = 0; c < 3; ++c)
        for (uint32_t i = 0; i < n; ++i) {
          const uint32_t u = h < w ? i % w : i / h;
          const uint32_t v = h < w ? i / w : i % h;
          const bool llf = u < w / 8 && v < h / 8;
          float value = random(rng);
          if (pattern == 0) value = i % 2 ? 0.0f : -0.0f;
          if (pattern == 1) value = i == a % n ? 1.0f : 0.0f;
          if (pattern == 2) value *= 0.00001f;
          if (pattern == 3) value *= 100.0f;
          if (pattern == 4 && !llf) {
            const float threshold = (i % 2 ? 0.64f : -0.58f);
            value = std::nextafter(threshold, i % 3 ? 0.0f : threshold * 2) /
              (tables[inverse + c * n + i] *
               (quantizer[0] * (1.0f / 65536.0f)) * raw_value *
               (c == 0 ? params.x_matrix_multiplier :
                c == 2 ? params.b_matrix_multiplier : 1.0f));
          }
          const size_t offset = batch.coefficient_offset +
                                (static_cast<size_t>(c) * count + a) * n + i;
          forward[offset] = value;
          active_coefficients[offset] = true;
        }
    }
    if (invalid == 1 || invalid == 2) {
      for (unsigned c = 0; c < 3; ++c)
        forward[batch.coefficient_offset + static_cast<size_t>(c) * count * n +
                (invalid == 1 ? n - 1 : 0)] = c == 0 ?
          std::numeric_limits<float>::quiet_NaN() :
          std::copysign(std::numeric_limits<float>::infinity(), c == 1 ? 1 : -1);
    }
    if (invalid == 3) {
      const auto anchor = anchors[batch.anchor_offset];
      sharpness[kBlockOffset + anchor.y * params.block_width + anchor.x] = 8;
    }
    if (invalid == 4) forward[batch.coefficient_offset] = 1.0e15f;
    if (invalid >= 5 && invalid <= 7) {
      // X/B dequantization must retain its error checks even when its float
      // output is omitted. Cover Y too, since prediction still consumes it.
      const unsigned channel = invalid == 5 ? 0 : invalid == 6 ? 2 : 1;
      tables[dequant + channel * n + n - 1] =
          std::numeric_limits<float>::infinity();
    }
  }
};

struct DeviceCase {
  explicit DeviceCase(const Case& c)
      : anchors(c.anchors), tables(c.tables), forward(c.forward),
        thresholds(c.thresholds), reconstruction(c.reconstruction), dc(c.dc),
        sigma(c.sigma), raw(c.raw), ac(c.ac), quantized_dc(c.quantized_dc),
        cfl_x(c.cfl_x), cfl_b(c.cfl_b), sharpness(c.sharpness),
        quantizer(c.quantizer), error(std::vector<unsigned>{kInitialError}) {}
  void Launch(const Case& c, bool cached, cudaStream_t stream = nullptr) {
    const auto launch = cached ?
      gjxl::cuda_internal::LaunchCudaAqEncodeResidentCoefficients :
      gjxl::cuda_internal::LaunchCudaAqEncodeResidentCoefficientsReference;
    LaunchWith(c, launch, stream);
  }
  void LaunchUnfused(const Case& c, cudaStream_t stream = nullptr) {
    LaunchWith(c, gjxl::cuda_internal::LaunchCudaAqEncodeResidentCoefficientsUnfused,
               stream);
  }
  void LaunchMaterialize(const Case& c, cudaStream_t stream = nullptr,
                         bool null_reconstruction = false) {
    LaunchWith(c, gjxl::cuda_internal::LaunchCudaAqMaterializeResidentCoefficients,
               stream, null_reconstruction);
  }
  void LaunchGeneric(const Case& c, bool materialize = false,
                     cudaStream_t stream = nullptr) {
    LaunchWith(c, materialize ?
      gjxl::cuda_internal::LaunchCudaAqMaterializeResidentCoefficientsGeneric :
      gjxl::cuda_internal::LaunchCudaAqEncodeResidentCoefficientsGeneric,
      stream, materialize);
  }
  using Launcher = decltype(
      &gjxl::cuda_internal::LaunchCudaAqEncodeResidentCoefficients);
  void LaunchWith(const Case& c, Launcher launch, cudaStream_t stream,
                  bool null_reconstruction = false) {
    CheckCuda(launch(anchors.data, tables.data, raw.data + kBlockOffset,
      cfl_x.data, cfl_b.data, forward.data, ac.data,
      null_reconstruction ? nullptr : reconstruction.data,
      dc.data + kBlockOffset, quantized_dc.data + kBlockOffset,
      sigma.data + kBlockOffset, sharpness.data + kBlockOffset, quantizer.data,
      thresholds.data, error.data, c.batch, c.params, stream));
  }
  DeviceArray<gjxl::cuda_internal::CudaAqAnchor> anchors;
  DeviceArray<float> tables, forward, thresholds, reconstruction, dc, sigma;
  DeviceArray<int> raw, ac, quantized_dc;
  DeviceArray<signed char> cfl_x, cfl_b;
  DeviceArray<unsigned char> sharpness;
  DeviceArray<unsigned> quantizer, error;
};

template <typename T>
void CheckGuards(const std::vector<T>& actual, const std::vector<T>& initial,
                 const std::vector<bool>& active, const char* name) {
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!active[i] && std::memcmp(&actual[i], &initial[i], sizeof(T)) != 0)
      throw std::runtime_error(std::string(name) + " guard overwritten");
    if (active[i] && (!std::isfinite(static_cast<double>(actual[i])) ||
        std::memcmp(&actual[i], &initial[i], sizeof(T)) == 0))
      throw std::runtime_error(std::string(name) + " active output unwritten");
  }
}

void Verify(gjxl::AcStrategyType strategy, uint32_t count, unsigned pattern,
            bool adjust, unsigned invalid = 0) {
  Case c(strategy, count, pattern, adjust, invalid);
  DeviceCase reference(c), unfused(c), cached(c), materialized(c);
  DeviceCase generic(c), generic_materialized(c);
  for (unsigned reuse = 0; reuse < 2; ++reuse) {
    if (reuse != 0) {
      // Reuse the same allocations with changed inputs, not only an identical
      // second invocation that could hide stale reconstructed coefficients.
      for (size_t i = 0; i < c.forward.size(); ++i)
        if (c.active_coefficients[i] && std::isfinite(c.forward[i]))
          c.forward[i] = -0.75f * c.forward[i] + (i % 11) * 0.00003f;
      for (auto* device : {&reference, &unfused, &cached, &materialized,
                           &generic, &generic_materialized})
        CheckCuda(cudaMemcpy(device->forward.data, c.forward.data(),
          c.forward.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    reference.Launch(c, false);
    unfused.LaunchUnfused(c);
    cached.Launch(c, true);
    // First prove no writes to a guarded output, then use a null pointer.
    materialized.LaunchMaterialize(c, nullptr, reuse != 0);
    generic.LaunchGeneric(c);
    generic_materialized.LaunchGeneric(c, true);
    CheckCuda(cudaDeviceSynchronize());
    for (const auto* expected : {&generic, &unfused, &reference}) {
      try {
        CheckEqual("AC", expected->ac.Read(), cached.ac.Read());
        CheckEqual("reconstruction", expected->reconstruction.Read(),
                   cached.reconstruction.Read());
        CheckEqual("DC", expected->dc.Read(), cached.dc.Read());
        CheckEqual("DC integer", expected->quantized_dc.Read(),
                   cached.quantized_dc.Read());
        CheckEqual("inverse sigma", expected->sigma.Read(), cached.sigma.Read());
        CheckEqual("error", expected->error.Read(), cached.error.Read());
      } catch (...) {
        std::cerr << "strategy=" << static_cast<unsigned>(strategy)
                  << " count=" << count << " pattern=" << pattern
                  << " adjust=" << adjust << " invalid=" << invalid
                  << " reuse=" << reuse << '\n';
        throw;
      }
    }
    try {
      CheckEqual("generic materialized AC", generic_materialized.ac.Read(),
                 materialized.ac.Read());
      CheckEqual("generic materialized DC", generic_materialized.dc.Read(),
                 materialized.dc.Read());
      CheckEqual("generic materialized DC integer",
                 generic_materialized.quantized_dc.Read(),
                 materialized.quantized_dc.Read());
      CheckEqual("generic materialized inverse sigma",
                 generic_materialized.sigma.Read(), materialized.sigma.Read());
      CheckEqual("generic materialized error", generic_materialized.error.Read(),
                 materialized.error.Read());
      CheckEqual("materialized AC", cached.ac.Read(), materialized.ac.Read());
      CheckEqual("materialized DC", cached.dc.Read(), materialized.dc.Read());
      CheckEqual("materialized DC integer", cached.quantized_dc.Read(),
                 materialized.quantized_dc.Read());
      CheckEqual("materialized inverse sigma", cached.sigma.Read(),
                 materialized.sigma.Read());
      CheckEqual("materialized error", cached.error.Read(),
                 materialized.error.Read());
      CheckEqual("untouched reconstruction", c.reconstruction,
                 materialized.reconstruction.Read());
    } catch (...) {
      std::cerr << "materialization strategy=" << static_cast<unsigned>(strategy)
                << " count=" << count << " pattern=" << pattern
                << " adjust=" << adjust << " invalid=" << invalid
                << " reuse=" << reuse << '\n';
      throw;
    }
    const unsigned error = cached.error.Read()[0];
    const unsigned required = invalid == 1 ? 1u : invalid == 2 ? 257u :
                              invalid == 3 ? 128u : invalid == 4 ? 8u :
                              invalid >= 5 ? 4u : 0u;
    if ((error & (kInitialError | required)) != (kInitialError | required) ||
        (invalid == 0 && error != kInitialError))
      throw std::runtime_error("Coefficient error flags incorrect");
    CheckGuards(cached.ac.Read(), c.ac, c.active_coefficients, "AC");
    CheckGuards(cached.reconstruction.Read(), c.reconstruction,
                c.active_coefficients, "reconstruction");
    const size_t blocks = c.params.block_width * c.params.block_height;
    std::vector<bool> active_dc(c.dc.size());
    for (size_t b = 0; b < blocks; ++b)
      for (size_t channel = 0; channel < 3; ++channel)
        active_dc[kBlockOffset + channel * blocks + b] =
          c.active_blocks[kBlockOffset + b];
    CheckGuards(cached.dc.Read(), c.dc, active_dc, "DC");
    CheckGuards(cached.quantized_dc.Read(), c.quantized_dc, active_dc, "DC int");
    auto active_sigma = c.active_blocks;
    for (size_t i = 0; i < active_sigma.size(); ++i)
      active_sigma[i] = active_sigma[i] && adjust && c.sharpness[i] < 8;
    CheckGuards(cached.sigma.Read(), c.sigma, active_sigma, "inverse sigma");
  }
  // A full evaluation after materialization must overwrite every active
  // reconstruction entry, without depending on the skipped float output.
  materialized.Launch(c, true);
  CheckCuda(cudaDeviceSynchronize());
  CheckEqual("reconstruction after materialization", cached.reconstruction.Read(),
             materialized.reconstruction.Read());
  for (const auto* device : {&reference, &unfused, &cached, &materialized,
                             &generic, &generic_materialized}) {
    if (!Equal(device->anchors.Read(), c.anchors) ||
        !Equal(device->tables.Read(), c.tables) ||
        !Equal(device->forward.Read(), c.forward) ||
        !Equal(device->thresholds.Read(), c.thresholds) ||
        !Equal(device->raw.Read(), c.raw) ||
        !Equal(device->cfl_x.Read(), c.cfl_x) ||
        !Equal(device->cfl_b.Read(), c.cfl_b) ||
        !Equal(device->sharpness.Read(), c.sharpness) ||
        !Equal(device->quantizer.Read(), c.quantizer))
      throw std::runtime_error("Coefficient input overwritten");
  }
}

void VerifyGenericFallback(unsigned mode) {
  Case c(mode == 1 ? gjxl::AcStrategyType::kDct16x8 :
         mode == 3 ? gjxl::AcStrategyType::kDct16x16 :
                     gjxl::AcStrategyType::kDct8, 3, 5, true);
  // All accesses remain inside the fixture allocations. These internal
  // noncanonical configurations must keep the old generic entry's behavior.
  if (mode == 0) c.params.strategy = 99;
  if (mode == 1) c.params.strategy = 7;  // Alias table, opposite physical shape.
  if (mode == 2) --c.batch.coefficient_count;
  if (mode == 3) c.batch.covered_width = 1;
  for (bool materialize : {false, true}) {
    DeviceCase generic(c), candidate(c);
    generic.LaunchGeneric(c, materialize);
    if (materialize) candidate.LaunchMaterialize(c, nullptr, true);
    else candidate.Launch(c, true);
    CheckCuda(cudaDeviceSynchronize());
    CheckEqual("fallback AC", generic.ac.Read(), candidate.ac.Read());
    CheckEqual("fallback reconstruction", generic.reconstruction.Read(),
               candidate.reconstruction.Read());
    CheckEqual("fallback DC", generic.dc.Read(), candidate.dc.Read());
    CheckEqual("fallback DC integer", generic.quantized_dc.Read(),
               candidate.quantized_dc.Read());
    CheckEqual("fallback sigma", generic.sigma.Read(), candidate.sigma.Read());
    CheckEqual("fallback error", generic.error.Read(), candidate.error.Read());
    if (candidate.error.Read()[0] != kInitialError)
      throw std::runtime_error("Fallback fixture reported a numeric error");
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    CheckCuda(cudaSetDevice(0));
    size_t cases = 0;
    for (auto strategy : kStrategies) {
      for (uint32_t count : {1u, 3u, 17u})
        for (unsigned pattern = 0; pattern < 6; ++pattern)
          for (bool adjust : {false, true}) {
            Verify(strategy, count, pattern, adjust);
            ++cases;
          }
      for (unsigned invalid = 1; invalid <= 7; ++invalid) {
        Verify(strategy, 3, 5, true, invalid);
        ++cases;
      }
    }
    for (unsigned mode = 0; mode < 4; ++mode) VerifyGenericFallback(mode);
    std::cout << "Verified " << cases
              << " guarded resident coefficient cases against original and "
                 "unfused kernels, including materialization, reuse, and "
                 "four generic-dispatch fallbacks\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
