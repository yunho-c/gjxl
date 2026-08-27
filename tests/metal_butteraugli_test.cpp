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
#include <string>
#include <vector>

#include "butteraugli_fixtures.h"
#include "butteraugli_goldens_generated.h"
#include "codec/butteraugli.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_butteraugli_test.h"
#include "gpu/ops/butteraugli.h"
#include "gpu_test_utils.h"

namespace {

constexpr float kMetalTolerance = 1.5e-3f;
constexpr float kIdentityTolerance = 1.0e-7f;
constexpr float kHostPoison = -12345.0f;

namespace bt = gjxl::butteraugli_test;
namespace golden = gjxl::butteraugli_test::golden;

struct HostImage {
  explicit HostImage(gjxl::Extent2D image_extent)
    : extent(image_extent),
      stride(image_extent.width + 5) {
    for (auto& plane : values) {
      plane.assign(stride * extent.height, kHostPoison);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{{
      {values[0].data(), extent, stride},
      {values[1].data(), extent, stride},
      {values[2].data(), extent, stride},
    }}};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{{
      {values[0].data(), extent, stride},
      {values[1].data(), extent, stride},
      {values[2].data(), extent, stride},
    }}};
  }

  gjxl::Extent2D extent;
  size_t stride = 0;
  std::array<std::vector<float>, 3> values;
};

struct DeviceImage {
  [[nodiscard]] bool Prepare(
    gjxl::GpuBackend& backend,
    gjxl::Extent2D extent,
    size_t stride_bias) {

    for (size_t channel = 0; channel < plane.size(); ++channel) {
      if (!plane[channel].Prepare(
            backend, extent, extent.width + stride_bias + channel).ok()) {
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
        for (size_t x = 0; x < host.extent.width; ++x) {
          logical.push_back(host.values[channel][y * host.stride + x]);
        }
      }
      plane[channel].SetLogical(logical);
      if (!plane[channel].Upload().ok()) return false;
    }
    return true;
  }

  [[nodiscard]] gjxl::ConstDeviceImage3View View() const noexcept {
    return {{{plane[0].ConstView(), plane[1].ConstView(),
              plane[2].ConstView()}}};
  }

  std::array<gjxl::test::GuardedDevicePlane, 3> plane;
};

void FillFixture(HostImage* reference, HostImage* distorted, bool identity) {
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < reference->extent.height; ++y) {
      for (size_t x = 0; x < reference->extent.width; ++x) {
        const float base =
          0.06f + 0.12f * static_cast<float>(channel) +
          0.004f * static_cast<float>((13 * x + 7 * y) % 41) +
          0.0007f * static_cast<float>(x * y);
        reference->values[channel][y * reference->stride + x] = base;
        distorted->values[channel][y * distorted->stride + x] = identity
          ? base
          : std::max(
              0.0f,
              base + 0.011f * std::sin(
                static_cast<float>(3 * x + 5 * y + 2 * channel)));
      }
    }
  }
}

[[nodiscard]] HostImage ConvertImage(const bt::ImageStorage& source) {
  const gjxl::Extent2D extent{source.extent().width, source.extent().height};
  HostImage result(extent);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < extent.height; ++y) {
      std::copy_n(
        source.planes()[channel].data() + y * source.stride(),
        extent.width,
        result.values[channel].data() + y * result.stride);
    }
  }
  return result;
}

[[nodiscard]] gjxl::ButteraugliOptions ConvertOptions(
  bt::OracleOptions options) {

  return {
    .hf_asymmetry = options.hf_asymmetry,
    .x_multiplier = options.x_multiplier,
    .intensity_target = options.intensity_target,
  };
}

[[nodiscard]] bool CheckImages(
  const HostImage& reference,
  const HostImage& distorted,
  gjxl::ButteraugliOptions options,
  bool identity,
  float* maximum_map_error,
  double* maximum_score_error) {

  const gjxl::Extent2D extent = reference.extent;
  if (distorted.extent != extent) return false;
  const size_t host_map_stride = extent.width + 7;
  std::vector<float> expected_map(
    host_map_stride * extent.height, kHostPoison);
  double expected_score = -1.0;
  const gjxl::Status cpu_status = gjxl::ComputeButteraugliDistance(
    reference.ConstView(), distorted.ConstView(), options,
    {expected_map.data(), extent, host_map_stride}, &expected_score);
  if (!cpu_status.ok()) {
    std::cerr << "CPU fixture failed: " << cpu_status.message() << '\n';
    return false;
  }

  std::unique_ptr<gjxl::GpuBackend> backend;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &backend).ok()) {
    return false;
  }
  DeviceImage device_reference;
  DeviceImage device_distorted;
  gjxl::test::GuardedDevicePlane device_map;
  gjxl::test::GuardedDevicePlane device_score;
  if (!device_reference.Prepare(*backend, extent, 3) ||
      !device_distorted.Prepare(*backend, extent, 5) ||
      !device_reference.Upload(reference) ||
      !device_distorted.Upload(distorted) ||
      !device_map.Prepare(*backend, extent, extent.width + 9).ok() ||
      !device_score.Prepare(*backend, {1, 1}, 3).ok()) {
    return false;
  }
  device_map.PoisonLogical();
  device_score.PoisonLogical();
  if (!device_map.Upload().ok() || !device_score.Upload().ok()) return false;

  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  const gjxl::DeviceButteraugliPrepareDescriptor preparation{
    device_reference.View(), options};
  if (!gjxl::PrepareDeviceButteraugli(
        *backend, preparation, &prepared).ok() ||
      prepared == nullptr) {
    return false;
  }
  const gjxl::GpuBackendStats before = backend->stats();
  const gjxl::DeviceButteraugliComparisonDescriptor comparison{
    device_distorted.View(), device_map.View(), device_score.View()};
  const gjxl::Status metal_status = prepared->Compare(comparison);
  if (!metal_status.ok()) {
    std::cerr << "Metal comparison failed for " << extent.width << 'x'
              << extent.height << ": " << metal_status.message() << '\n';
    return false;
  }
  const gjxl::GpuBackendStats after = backend->stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 1) {
    std::cerr << "Metal comparison allocation/submission invariant failed\n";
    return false;
  }

  std::vector<float> actual_map(
    (extent.width + 11) * extent.height, kHostPoison);
  double actual_score = -2.0;
  if (!prepared->ReadDistanceMap(
        {actual_map.data(), extent, extent.width + 11}).ok() ||
      !prepared->ReadScore(&actual_score).ok() ||
      !device_map.Download().ok() || !device_map.GuardsIntact() ||
      !device_score.Download().ok() || !device_score.GuardsIntact()) {
    return false;
  }

  const float tolerance = identity ? kIdentityTolerance : kMetalTolerance;
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const float expected = expected_map[y * host_map_stride + x];
      const float actual = actual_map[y * (extent.width + 11) + x];
      const float error = std::abs(actual - expected);
      *maximum_map_error = std::max(*maximum_map_error, error);
      if (!std::isfinite(actual) || error > tolerance) {
        std::cerr << "Metal map mismatch " << extent.width << 'x'
                  << extent.height << " x=" << x << " y=" << y
                  << " actual=" << actual << " expected=" << expected
                  << " error=" << error << " limit=" << tolerance << '\n';
        return false;
      }
    }
    for (size_t x = extent.width; x < extent.width + 11; ++x) {
      if (actual_map[y * (extent.width + 11) + x] != kHostPoison) {
        return false;
      }
    }
  }
  const double score_error = std::abs(actual_score - expected_score);
  *maximum_score_error = std::max(*maximum_score_error, score_error);
  if (!std::isfinite(actual_score) || score_error > tolerance) {
    std::cerr << "Metal score mismatch " << extent.width << 'x'
              << extent.height << " actual=" << actual_score
              << " expected=" << expected_score
              << " error=" << score_error << " limit=" << tolerance << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckGeneratedCase(
  gjxl::Extent2D extent,
  gjxl::ButteraugliOptions options,
  bool identity,
  float* maximum_map_error,
  double* maximum_score_error) {

  HostImage reference(extent);
  HostImage distorted(extent);
  FillFixture(&reference, &distorted, identity);
  return CheckImages(
    reference, distorted, options, identity,
    maximum_map_error, maximum_score_error);
}

[[nodiscard]] bool CheckFixture(
  const bt::FixturePair& fixture,
  float* maximum_map_error,
  double* maximum_score_error) {

  const HostImage reference = ConvertImage(fixture.reference);
  const HostImage distorted = ConvertImage(fixture.distorted);
  const bool identity = fixture.name.starts_with("identity_");
  if (!CheckImages(
        reference, distorted, ConvertOptions(fixture.options), identity,
        maximum_map_error, maximum_score_error)) {
    std::cerr << "Metal Butteraugli fixture failed: " << fixture.name << '\n';
    return false;
  }
  return bt::PaddingIsPoisoned(fixture.reference) &&
         bt::PaddingIsPoisoned(fixture.distorted);
}

[[nodiscard]] float Decode(uint32_t bits) {
  return std::bit_cast<float>(bits);
}

[[nodiscard]] bool CheckIntermediateStageGoldens(float* maximum_stage_error) {
  static_assert(
    golden::kIntermediateStageCount ==
    static_cast<size_t>(gjxl::MetalButteraugliStage::kCount));
  const bt::FixturePair fixture = bt::MakeFixture({
    "metal_intermediate_16x12",
    {golden::kIntermediateWidth, golden::kIntermediateHeight},
    bt::FixtureKind::kIntermediate,
    {},
  });
  const HostImage reference = ConvertImage(fixture.reference);
  const HostImage distorted = ConvertImage(fixture.distorted);
  const gjxl::Extent2D extent = reference.extent;

  std::unique_ptr<gjxl::GpuBackend> backend;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &backend).ok()) {
    return false;
  }
  DeviceImage device_reference;
  DeviceImage device_distorted;
  gjxl::test::GuardedDevicePlane device_map;
  gjxl::test::GuardedDevicePlane device_score;
  if (!device_reference.Prepare(*backend, extent, 3) ||
      !device_distorted.Prepare(*backend, extent, 5) ||
      !device_reference.Upload(reference) ||
      !device_distorted.Upload(distorted) ||
      !device_map.Prepare(*backend, extent, extent.width + 9).ok() ||
      !device_score.Prepare(*backend, {1, 1}, 3).ok()) {
    return false;
  }
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  if (!gjxl::PrepareDeviceButteraugli(
        *backend, {device_reference.View(), {}}, &prepared).ok() ||
      prepared == nullptr) {
    return false;
  }
  const gjxl::DeviceButteraugliComparisonDescriptor comparison{
    device_distorted.View(), device_map.View(), device_score.View()};

  constexpr size_t kCaptureStride = golden::kIntermediateWidth + 5;
  std::vector<float> capture(
    kCaptureStride * golden::kIntermediateHeight, kHostPoison);
  for (size_t stage_index = 0;
       stage_index < golden::kIntermediateStageCount;
       ++stage_index) {
    const auto stage =
      static_cast<gjxl::MetalButteraugliStage>(stage_index);
    if (!gjxl::ConfigureMetalButteraugliStageCaptureForTest(
          *prepared, stage).ok()) {
      return false;
    }
    const gjxl::GpuBackendStats before = backend->stats();
    if (!prepared->Compare(comparison).ok()) return false;
    const gjxl::GpuBackendStats after = backend->stats();
    if (after.successful_allocations != before.successful_allocations ||
        after.committed_submissions != before.committed_submissions + 1) {
      return false;
    }
    std::fill(capture.begin(), capture.end(), kHostPoison);
    if (!gjxl::ReadMetalButteraugliStageCaptureForTest(
          *prepared, {capture.data(), extent, kCaptureStride}).ok()) {
      return false;
    }
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        const size_t golden_index = y * extent.width + x;
        const float expected = Decode(
          golden::kIntermediateStageBits[stage_index][golden_index]);
        const float actual = capture[y * kCaptureStride + x];
        const float error = std::abs(actual - expected);
        *maximum_stage_error = std::max(*maximum_stage_error, error);
        if (!std::isfinite(actual) || error > kMetalTolerance) {
          std::cerr << "Metal intermediate-stage mismatch stage="
                    << stage_index << " x=" << x << " y=" << y
                    << " actual=" << actual << " expected=" << expected
                    << " error=" << error << '\n';
          return false;
        }
      }
      for (size_t x = extent.width; x < kCaptureStride; ++x) {
        if (capture[y * kCaptureStride + x] != kHostPoison) return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  float maximum_map_error = 0.0f;
  double maximum_score_error = 0.0;
  float maximum_stage_error = 0.0f;
  const std::array<gjxl::Extent2D, 8> extents{{
    {1, 1}, {3, 7}, {7, 3}, {8, 8},
    {9, 13}, {15, 15}, {17, 29}, {33, 17},
  }};
  for (size_t index = 0; index < extents.size(); ++index) {
    gjxl::ButteraugliOptions options;
    if (index == extents.size() - 1) {
      options = {
        .hf_asymmetry = 1.6f,
        .x_multiplier = 0.75f,
        .intensity_target = 255.0f,
      };
    }
    if (!CheckGeneratedCase(
          extents[index], options, false,
          &maximum_map_error, &maximum_score_error)) {
      return EXIT_FAILURE;
    }
  }
  if (!CheckGeneratedCase(
        {32, 24}, {}, true,
        &maximum_map_error, &maximum_score_error)) {
    return EXIT_FAILURE;
  }
  for (const bt::FixturePair& fixture :
       bt::BuildSyntheticDifferentialCorpus()) {
    if (!CheckFixture(
          fixture, &maximum_map_error, &maximum_score_error)) {
      return EXIT_FAILURE;
    }
  }
#if GJXL_METAL_BUTTERAUGLI_REFERENCE_FIXTURES
  std::vector<bt::FixturePair> flower_fixtures;
  flower_fixtures.push_back(
    bt::LoadFlowerFixture(GJXL_FLOWER_PPM_PATH, 207, 218, {96, 96}));
  flower_fixtures.push_back(
    bt::LoadFlowerFixture(GJXL_FLOWER_PPM_PATH));
  for (const bt::FixturePair& fixture : flower_fixtures) {
    if (!CheckFixture(
          fixture, &maximum_map_error, &maximum_score_error)) {
      return EXIT_FAILURE;
    }
  }
#endif
  if (!CheckIntermediateStageGoldens(&maximum_stage_error)) {
    return EXIT_FAILURE;
  }
  std::cout << "Metal Butteraugli maximum map error: "
            << maximum_map_error << '\n'
            << "Metal Butteraugli maximum score error: "
            << maximum_score_error << '\n'
            << "Metal Butteraugli maximum stage error: "
            << maximum_stage_error << '\n';
  return EXIT_SUCCESS;
}
