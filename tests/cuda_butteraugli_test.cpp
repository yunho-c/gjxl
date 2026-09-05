// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include "codec/butteraugli.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/ops/butteraugli.h"
#include "gpu_test_utils.h"

namespace {

constexpr float kTolerance = 1.5e-3f;
constexpr float kIdentityTolerance = 1.0e-7f;
constexpr float kHostPoison = -12345.0f;

struct HostImage {
  explicit HostImage(gjxl::Extent2D image_extent)
      : extent(image_extent), stride(image_extent.width + 5) {
    for (auto& values : plane) {
      values.assign(stride * extent.height, kHostPoison);
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView View() const noexcept {
    return {{{
        {plane[0].data(), extent, stride},
        {plane[1].data(), extent, stride},
        {plane[2].data(), extent, stride},
    }}};
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

struct DeviceImage {
  [[nodiscard]] bool Prepare(gjxl::GpuBackend& backend, gjxl::Extent2D extent,
                             size_t stride_bias) {
    for (size_t channel = 0; channel < plane.size(); ++channel) {
      if (!plane[channel]
               .Prepare(backend, extent, extent.width + stride_bias + channel)
               .ok()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool Upload(const HostImage& host) {
    for (size_t channel = 0; channel < plane.size(); ++channel) {
      std::vector<float> logical;
      logical.reserve(host.extent.width * host.extent.height);
      for (size_t y = 0; y < host.extent.height; ++y) {
        logical.insert(
            logical.end(), host.plane[channel].begin() + y * host.stride,
            host.plane[channel].begin() + y * host.stride + host.extent.width);
      }
      plane[channel].SetLogical(logical);
      if (!plane[channel].Upload().ok()) return false;
    }
    return true;
  }

  [[nodiscard]] gjxl::ConstDeviceImage3View View() const noexcept {
    return {
        {{plane[0].ConstView(), plane[1].ConstView(), plane[2].ConstView()}}};
  }

  std::array<gjxl::test::GuardedDevicePlane, 3> plane;
};

void FillFixture(HostImage* reference, HostImage* distorted, bool identity) {
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < reference->extent.height; ++y) {
      for (size_t x = 0; x < reference->extent.width; ++x) {
        const float base = 0.06f + 0.12f * static_cast<float>(channel) +
                           0.004f * static_cast<float>((13 * x + 7 * y) % 41) +
                           0.0007f * static_cast<float>(x * y);
        reference->plane[channel][y * reference->stride + x] = base;
        distorted->plane[channel][y * distorted->stride + x] =
            identity
                ? base
                : std::max(0.0f,
                           base + 0.011f * std::sin(static_cast<float>(
                                               3 * x + 5 * y + 2 * channel)));
      }
    }
  }
}

[[nodiscard]] bool CheckCase(gjxl::GpuBackend& backend, gjxl::Extent2D extent,
                             gjxl::ButteraugliOptions options, bool identity,
                             float* worst_map, double* worst_score) {
  HostImage reference(extent);
  HostImage distorted(extent);
  FillFixture(&reference, &distorted, identity);
  const size_t expected_stride = extent.width + 7;
  std::vector<float> expected(expected_stride * extent.height, kHostPoison);
  double expected_score = -1.0;
  gjxl::Status status = gjxl::ComputeButteraugliDistance(
      reference.View(), distorted.View(), options,
      {expected.data(), extent, expected_stride}, &expected_score);
  if (!status.ok()) {
    std::cerr << "CPU Butteraugli failed: " << status.message() << '\n';
    return false;
  }

  DeviceImage device_reference;
  DeviceImage device_distorted;
  gjxl::test::GuardedDevicePlane map;
  gjxl::test::GuardedDevicePlane score;
  if (!device_reference.Prepare(backend, extent, 3) ||
      !device_distorted.Prepare(backend, extent, 5) ||
      !device_reference.Upload(reference) ||
      !device_distorted.Upload(distorted) ||
      !map.Prepare(backend, extent, extent.width + 9).ok() ||
      !score.Prepare(backend, {1, 1}, 3).ok()) {
    return false;
  }
  map.PoisonLogical();
  score.PoisonLogical();
  if (!map.Upload().ok() || !score.Upload().ok()) return false;

  const gjxl::GpuBackendStats before_prepare = backend.stats();
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  status = gjxl::PrepareDeviceButteraugli(
      backend, {device_reference.View(), options}, &prepared);
  if (!status.ok() || prepared == nullptr) {
    std::cerr << "CUDA Butteraugli preparation failed for " << extent.width
              << 'x' << extent.height << ": " << status.message() << '\n';
    return false;
  }
  const gjxl::GpuBackendStats after_prepare = backend.stats();
  if (after_prepare.successful_allocations !=
          before_prepare.successful_allocations + 1 ||
      after_prepare.committed_submissions !=
          before_prepare.committed_submissions + 1) {
    std::cerr << "CUDA Butteraugli preparation resource invariant failed\n";
    return false;
  }
  const gjxl::DeviceButteraugliMemoryStats memory = prepared->memory_stats();
  if (memory.cached_reference_bytes == 0 ||
      memory.gaussian_kernel_bytes != 73 * sizeof(float) ||
      memory.peak_comparison_scratch_bytes == 0 ||
      memory.prepared_allocation_bytes <
          memory.cached_reference_bytes + memory.gaussian_kernel_bytes +
              memory.peak_comparison_scratch_bytes) {
    std::cerr << "CUDA Butteraugli memory accounting is inconsistent\n";
    return false;
  }

  const gjxl::GpuBackendStats before_compare = backend.stats();
  status =
      prepared->Compare({device_distorted.View(), map.View(), score.View()});
  if (!status.ok()) {
    std::cerr << "CUDA Butteraugli comparison failed for " << extent.width
              << 'x' << extent.height << ": " << status.message() << '\n';
    return false;
  }
  const gjxl::GpuBackendStats after_compare = backend.stats();
  if (after_compare.successful_allocations !=
          before_compare.successful_allocations ||
      after_compare.committed_submissions !=
          before_compare.committed_submissions + 1) {
    std::cerr << "CUDA Butteraugli comparison resource invariant failed\n";
    return false;
  }

  const size_t actual_stride = extent.width + 11;
  std::vector<float> actual(actual_stride * extent.height, kHostPoison);
  double actual_score = -1.0;
  if (!prepared->ReadDistanceMap({actual.data(), extent, actual_stride}).ok() ||
      !prepared->ReadScore(&actual_score).ok() || !map.Download().ok() ||
      !score.Download().ok() || !map.GuardsIntact() || !score.GuardsIntact()) {
    std::cerr << "CUDA Butteraugli readback or guard validation failed\n";
    return false;
  }

  const float tolerance = identity ? kIdentityTolerance : kTolerance;
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const float wanted = expected[y * expected_stride + x];
      const float found = actual[y * actual_stride + x];
      const float error = std::abs(wanted - found);
      *worst_map = std::max(*worst_map, error);
      if (!std::isfinite(found) || error > tolerance) {
        std::cerr << "CUDA Butteraugli map mismatch " << extent.width << 'x'
                  << extent.height << " at " << x << ',' << y
                  << ": actual=" << found << " expected=" << wanted
                  << " error=" << error << '\n';
        return false;
      }
    }
    for (size_t x = extent.width; x < actual_stride; ++x) {
      if (actual[y * actual_stride + x] != kHostPoison) return false;
    }
  }
  const double score_error = std::abs(expected_score - actual_score);
  *worst_score = std::max(*worst_score, score_error);
  if (!std::isfinite(actual_score) || score_error > tolerance) {
    std::cerr << "CUDA Butteraugli score mismatch " << extent.width << 'x'
              << extent.height << ": actual=" << actual_score
              << " expected=" << expected_score << " error=" << score_error
              << '\n';
    return false;
  }

  std::vector<float> first = actual;
  const uint64_t first_score = std::bit_cast<uint64_t>(actual_score);
  if (!prepared->Compare({device_distorted.View(), map.View(), score.View()})
           .ok()) {
    return false;
  }
  std::fill(actual.begin(), actual.end(), kHostPoison);
  if (!prepared->ReadDistanceMap({actual.data(), extent, actual_stride}).ok() ||
      !prepared->ReadScore(&actual_score).ok() || actual != first ||
      std::bit_cast<uint64_t>(actual_score) != first_score) {
    std::cerr << "CUDA Butteraugli comparison is not deterministic\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckFailureInvalidation(gjxl::GpuBackend& backend) {
  const gjxl::Extent2D extent{9, 13};
  HostImage reference(extent);
  HostImage distorted(extent);
  FillFixture(&reference, &distorted, false);
  DeviceImage device_reference;
  DeviceImage device_distorted;
  gjxl::test::GuardedDevicePlane map;
  gjxl::test::GuardedDevicePlane score;
  if (!device_reference.Prepare(backend, extent, 1) ||
      !device_distorted.Prepare(backend, extent, 2) ||
      !device_reference.Upload(reference) ||
      !device_distorted.Upload(distorted) ||
      !map.Prepare(backend, extent, extent.width).ok() ||
      !score.Prepare(backend, {1, 1}, 1).ok()) {
    return false;
  }
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  if (!gjxl::PrepareDeviceButteraugli(backend, {device_reference.View(), {}},
                                      &prepared)
           .ok() ||
      prepared == nullptr ||
      !gjxl::ArmNextCudaSubmissionFailureForTest(backend, false, true).ok()) {
    return false;
  }
  const gjxl::Status failure =
      prepared->Compare({device_distorted.View(), map.View(), score.View()});
  if (failure.code() != gjxl::StatusCode::kDeviceError || prepared->valid() ||
      prepared->Compare({device_distorted.View(), map.View(), score.View()})
              .code() != gjxl::StatusCode::kInvalidArgument) {
    std::cerr
        << "CUDA Butteraugli completion failure did not invalidate state\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> backend;
  const gjxl::Status create = gjxl::CreateCudaBackend(&backend);
  if (create.code() == gjxl::StatusCode::kUnavailable) {
    std::cout << "CUDA backend unavailable: " << create.message() << '\n';
    return 77;
  }
  if (!create.ok() || backend == nullptr) {
    std::cerr << "Create CUDA backend failed: " << create.message() << '\n';
    return EXIT_FAILURE;
  }

  float worst_map = 0.0f;
  double worst_score = 0.0;
  const std::array<gjxl::Extent2D, 15> extents{{
      {1, 1},
      {3, 7},
      {7, 3},
      {8, 8},
      {9, 13},
      {15, 15},
      {17, 29},
      {33, 17},
      {31, 8},
      {32, 9},
      {65, 33},
      {127, 65},
      {255, 63},
      {257, 67},
      {33, 129},
  }};
  for (size_t index = 0; index < extents.size(); ++index) {
    gjxl::ButteraugliOptions options;
    if (index + 1 == extents.size()) {
      options = {
          .hf_asymmetry = 1.6f,
          .x_multiplier = 0.75f,
          .intensity_target = 255.0f,
      };
    }
    if (!CheckCase(*backend, extents[index], options, false, &worst_map,
                   &worst_score)) {
      return EXIT_FAILURE;
    }
  }
  if (!CheckCase(*backend, {32, 24}, {}, true, &worst_map, &worst_score) ||
      !CheckFailureInvalidation(*backend)) {
    return EXIT_FAILURE;
  }
  std::cout << "CUDA Butteraugli passed on " << backend->name()
            << "; worst map error=" << worst_map
            << ", worst score error=" << worst_score << ".\n";
  return EXIT_SUCCESS;
}
