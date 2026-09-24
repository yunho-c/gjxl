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
#include <string_view>
#include <vector>

#include "gpu/cuda/cuda_butteraugli_kernels.h"

namespace {

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceArray {
  explicit DeviceArray(const std::vector<float>& values)
      : count(values.size()) {
    CheckCuda(
        cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    CheckCuda(cudaMemcpy(data, values.data(), count * sizeof(float),
                         cudaMemcpyHostToDevice));
  }
  ~DeviceArray() { (void)cudaFree(data); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  std::vector<float> Read() const {
    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), data, count * sizeof(float),
                         cudaMemcpyDeviceToHost));
    return result;
  }
  float* data = nullptr;
  size_t count;
};

bool Equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool Guards(const std::vector<float>& actual, const std::vector<float>& initial,
            uint32_t offset, uint32_t stride, uint32_t width, uint32_t height) {
  for (size_t i = 0; i < actual.size(); ++i) {
    const bool active = i >= offset &&
                        i < offset + static_cast<size_t>(stride) * height &&
                        (i - offset) % stride < width;
    if (!active && std::memcmp(&actual[i], &initial[i], sizeof(float)) != 0)
      return false;
  }
  return true;
}

void Verify(uint32_t width, uint32_t height, bool lf, bool initialize,
            unsigned pattern, bool tall, unsigned tile_height = 0,
            bool flat_grid = false, bool original_tiles = false) {
  using namespace gjxl::cuda_internal;
  constexpr uint32_t kReferenceOffset = 5, kDistortedOffset = 7;
  constexpr uint32_t kWorkOffset = 13, kOutputOffset = 19;
  const uint32_t rs = width + (tall ? 1 : 11);
  const uint32_t ds = width + (tall ? 2 : 17);
  const uint32_t ws = width + (tall ? 3 : 29);
  const uint32_t os = width + (tall ? 4 : 23);
  const float poison = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> reference(
      kReferenceOffset + static_cast<size_t>(rs) * height + 31, poison);
  std::vector<float> distorted(
      kDistortedOffset + static_cast<size_t>(ds) * height + 31, poison);
  std::vector<float> work(kWorkOffset + static_cast<size_t>(ws) * height + 31,
                          poison);
  std::vector<float> output(
      kOutputOffset + static_cast<size_t>(os) * height + 31, -12345.0f);
  std::mt19937 rng(314159u + width + height * 31 + pattern);
  std::uniform_real_distribution<float> random(-2, 2);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      float r = random(rng), d = random(rng);
      if (pattern == 0) r = d = ((x + y) & 1) ? -0.0f : 0.0f;
      if (pattern == 1) d = r + 0.03f * d;
      if (pattern == 2) {
        r *= 35;
        d *= 35;
      }
      if (pattern == 3) {
        // Exercise both asymmetric branches at and around their thresholds.
        r = ((x + y) & 1) ? -1.0f : 1.0f;
        d = r * ((x + y) % 3 == 0 ? 0.55f : 1.05f);
        if (x % 3 != 0) d = std::nextafter(d, x % 3 == 1 ? -2.0f : 2.0f);
      }
      if (pattern == 4) {
        constexpr std::array<float, 11> values{
            0.0f, -0.0f, std::numeric_limits<float>::min(),
            -std::numeric_limits<float>::min(),
            std::numeric_limits<float>::denorm_min(),
            -std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()};
        r = values[(x + y) % values.size()];
        d = values[(x * 7 + y + 3) % values.size()];
      }
      if (pattern == 5) d = r;
      if (pattern >= 6) {
        r = d = ((x + y) & 1) ? -0.0f : 0.0f;
        if (pattern == 7) {
          r = d = ((x + y) & 1)
                      ? -std::numeric_limits<float>::denorm_min()
                      : std::numeric_limits<float>::denorm_min();
        }
        // Sparse nonzero/NaN values must prevent a shortcut in every tile
        // whose response halo includes them, including partial edge tiles.
        if (pattern == 8 && x % 64 == 63 && y % 64 == 63) d = 1.0f;
        if (pattern == 9 && x == width / 2 && y == height / 2)
          r = std::numeric_limits<float>::quiet_NaN();
        // Isolate values across paired-row boundaries and tile halos. A
        // nonzero or exceptional value may affect either row of a pair.
        if (pattern == 12 && (x % 32 == 31 || x == width - 1) && y % 2 == 1)
          d = 1.0f;
        if (pattern >= 13 && (x % 32 == 0 || x == width - 1) &&
            (y % 8 == 0 || y % 8 == 7)) {
          d = pattern == 13 ? 1.0f
              : pattern == 14 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
        }
      }
      reference[kReferenceOffset + static_cast<size_t>(y) * rs + x] = r;
      distorted[kDistortedOffset + static_cast<size_t>(y) * ds + x] = d;
      output[kOutputOffset + static_cast<size_t>(y) * os + x] =
          pattern >= 6 ? -0.0f : 0.25f + (x % 11) * 0.125f;
      if (pattern == 10) {
        // A zero response still has to execute the original addition, even
        // when its destination contains a signed NaN payload or infinity.
        constexpr std::array<uint32_t, 6> bits{
            0x80000000u, 0x7fc01234u, 0xffc05678u,
            0x7f800000u, 0xff800000u, 0x00000001u};
        const uint32_t value = bits[(x + y) % bits.size()];
        std::memcpy(&output[kOutputOffset + static_cast<size_t>(y) * os + x],
                    &value, sizeof(float));
      }
    }
  }
  const float norm = pattern == 11  ? 0.0f
                     : pattern == 0 ? 5.0f
                     : pattern == 1 ? 71.7800275f
                                    : 130262059.556f;
  CudaButteraugliMaltaParams params{width,
                                    height,
                                    rs,
                                    ds,
                                    os,
                                    static_cast<uint32_t>(lf),
                                    static_cast<uint32_t>(initialize),
                                    norm * 0.7f,
                                    norm * 0.45f,
                                    norm};
  DeviceArray dr(reference), dd(distorted), dw(work);
  DeviceArray expected_output(output), prior_output(output), actual_output(output);
  const unsigned int prior_height = tile_height == 0 ? 8 : tile_height;
  const bool prior_flat = flat_grid || (height + prior_height - 1) / prior_height > 65535;
  // Keep the same storage across three stages to catch lost accumulation.
  for (unsigned stage = 0; stage < 3; ++stage) {
    CheckCuda(LaunchCudaButteraugliMaltaReference(
        dr.data + kReferenceOffset, dd.data + kDistortedOffset,
        dw.data + kWorkOffset, ws, expected_output.data + kOutputOffset, params,
        nullptr));
    CheckCuda(LaunchCudaButteraugliMaltaZeroAwareForTesting(
        dr.data + kReferenceOffset, dd.data + kDistortedOffset,
        prior_output.data + kOutputOffset, params, prior_height, prior_flat,
        nullptr));
    if (tile_height == 0) {
      CheckCuda(LaunchCudaButteraugliMalta(
          dr.data + kReferenceOffset, dd.data + kDistortedOffset,
          actual_output.data + kOutputOffset, params, nullptr));
    } else {
      const auto launch = original_tiles
                              ? LaunchCudaButteraugliMaltaForTesting
                              : LaunchCudaButteraugliMaltaRowPairForTesting;
      CheckCuda(launch(
          dr.data + kReferenceOffset, dd.data + kDistortedOffset,
          actual_output.data + kOutputOffset, params, tile_height, flat_grid,
          nullptr));
    }
    CheckCuda(cudaDeviceSynchronize());
    const auto expected = expected_output.Read();
    const auto prior = prior_output.Read();
    const auto actual = actual_output.Read();
    if (!Equal(actual, expected) || !Equal(actual, prior)) {
      std::cerr << "Mismatch " << width << 'x' << height << " lf=" << lf
                << " init=" << initialize << " pattern=" << pattern
                << " stage=" << stage << '\n';
      std::cerr << "tile_height=" << tile_height << " flat=" << flat_grid << '\n';
      throw std::runtime_error("Bitwise Malta response mismatch against separate or prior zero-aware oracle");
    }
    if (!Guards(actual, output, kOutputOffset, os, width, height) ||
        !Guards(dw.Read(), work, kWorkOffset, ws, width, height))
      throw std::runtime_error("Malta guard overwritten");
    params.initialize_accumulation = 0;
  }
  if (!Equal(dr.Read(), reference) || !Equal(dd.Read(), distorted))
    throw std::runtime_error("Malta input overwritten");
}

}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "CUDA device unavailable\n";
    return 77;
  }
  const bool tall = argc == 2 && std::string_view(argv[1]) == "--tall-only";
  const bool sanitizer =
      argc == 2 && std::string_view(argv[1]) == "--sanitizer-only";
  if (argc > 1 && !tall && !sanitizer) return 1;
  try {
    CheckCuda(cudaSetDevice(0));
    size_t cases = 0;
    if (sanitizer) {
      // Exercise the actual compiled kernels without allocating a huge image
      // solely to trigger the production tile/grid policy.
      for (unsigned tile_height : {8u, 24u, 64u}) {
        for (bool flat : {false, true})
          for (bool lf : {false, true})
            for (bool initialize : {false, true})
              for (unsigned pattern : {1u, 4u, 6u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}) {
                Verify(65, 65, lf, initialize, pattern, false, tile_height, flat);
                ++cases;
              }
        std::cout << "Verified sanitizer tile_height=" << tile_height
                  << " cases=" << cases << '\n' << std::flush;
      }
    } else if (tall) {
      // Retain the old boundary and test the current 64-row 2D/flat boundary.
      for (uint32_t height : {524280u, 524281u, 4194240u, 4194241u}) {
        for (bool lf : {false, true}) {
          for (bool initialize : {false, true}) {
            for (unsigned pattern : {1u, 6u}) {
              Verify(1, height, lf, initialize, pattern, true);
              ++cases;
            }
          }
        }
        std::cout << "Verified grid boundary height=" << height << '\n'
                  << std::flush;
      }
    } else {
      constexpr std::array<std::array<uint32_t, 2>, 10> kShapes{{{1, 1},
                                                                 {1, 9},
                                                                 {7, 11},
                                                                 {31, 8},
                                                                 {32, 9},
                                                                 {33, 17},
                                                                 {63, 31},
                                                                 {65, 33},
                                                                 {127, 65},
                                                                 {257, 67}}};
      for (const auto& shape : kShapes)
        for (bool lf : {false, true})
          for (bool initialize : {false, true})
            for (unsigned pattern = 0; pattern < 16; ++pattern) {
              Verify(shape[0], shape[1], lf, initialize, pattern, false);
              ++cases;
            }
      // Normal-policy row/tile boundaries, including both dispatch cutoffs.
      constexpr std::array<std::array<uint32_t, 2>, 26> kPolicyShapes{{
          {2048, 23}, {2048, 24}, {2048, 25}, {2048, 47}, {2048, 48},
          {2048, 49}, {2048, 63}, {2048, 64}, {2048, 65},
          {32, 1016}, {32, 1017}, {1024, 1472}, {1024, 1473},
          {1, 65535}, {1, 65536}, {1, 65537},
          {4096, 1}, {4096, 2}, {4096, 3}, {4096, 4}, {4096, 5},
          {32, 248}, {32, 249}, {992, 8}, {1024, 8}, {65, 65}}};
      for (const auto& shape : kPolicyShapes)
        for (bool lf : {false, true})
          for (bool initialize : {false, true})
            for (unsigned pattern : {1u, 4u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}) {
              Verify(shape[0], shape[1], lf, initialize, pattern, false);
              ++cases;
            }
      for (unsigned tile_height : {8u, 24u, 64u})
        for (bool flat : {false, true})
          for (const auto& shape : {std::array<uint32_t, 2>{31, 65},
                                   std::array<uint32_t, 2>{65, 63}})
            for (bool lf : {false, true})
              for (bool initialize : {false, true})
                for (unsigned pattern = 0; pattern < 16; ++pattern) {
                  Verify(shape[0], shape[1], lf, initialize, pattern, false,
                         tile_height, flat);
                  ++cases;
                  if (pattern < 6) {
                    Verify(shape[0], shape[1], lf, initialize, pattern, false,
                           tile_height, flat, true);
                    ++cases;
                  }
                }
    }
    std::cout << "Verified " << cases
              << " guarded Malta cases, three stages each\n" << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
