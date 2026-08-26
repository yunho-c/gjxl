// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the always-available native Butteraugli CPU facade.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <ranges>
#include <vector>

#include "codec/butteraugli.h"

namespace allocation_failure {

thread_local bool enabled = false;
thread_local size_t attempts = 0;
thread_local size_t fail_at = std::numeric_limits<size_t>::max();

[[nodiscard]] bool ShouldFail() noexcept {
  if (!enabled) {
    return false;
  }
  return attempts++ == fail_at;
}

} // namespace allocation_failure

void *operator new(size_t size) {
  if (allocation_failure::ShouldFail()) {
    throw std::bad_alloc();
  }
  if (void *address = std::malloc(std::max<size_t>(size, 1))) {
    return address;
  }
  throw std::bad_alloc();
}

void *operator new[](size_t size) {
  return ::operator new(size);
}

void operator delete(void *address) noexcept {
  std::free(address);
}

void operator delete[](void *address) noexcept {
  ::operator delete(address);
}

void operator delete(void *address, size_t) noexcept {
  ::operator delete(address);
}

void operator delete[](void *address, size_t) noexcept {
  ::operator delete(address);
}

namespace {

constexpr float kPoison = -991.0f;
constexpr double kScorePoison = -313.0;
constexpr float kIdentityTolerance = 1.0e-7f;

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D image_extent, size_t padding = 4)
      : extent(image_extent), stride(extent.width + padding) {
    for (std::vector<float> &values : plane) {
      values.assign(stride * extent.height, kPoison);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{{{plane[0].data(), extent, stride},
              {plane[1].data(), extent, stride},
              {plane[2].data(), extent, stride}}}};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{{{plane[0].data(), extent, stride},
              {plane[1].data(), extent, stride},
              {plane[2].data(), extent, stride}}}};
  }

  [[nodiscard]] bool PaddingIsUntouched() const {
    for (const std::vector<float> &values : plane) {
      for (size_t y = 0; y < extent.height; ++y) {
        for (size_t x = extent.width; x < stride; ++x) {
          if (values[y * stride + x] != kPoison) {
            return false;
          }
        }
      }
    }
    return true;
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

struct MapStorage {
  explicit MapStorage(gjxl::Extent2D image_extent, size_t padding = 5)
      : extent(image_extent), stride(extent.width + padding),
        values(stride * extent.height, kPoison) {}

  [[nodiscard]] gjxl::PlaneF32View View() {
    return {values.data(), extent, stride};
  }

  [[nodiscard]] bool PaddingIsUntouched() const {
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = extent.width; x < stride; ++x) {
        if (values[y * stride + x] != kPoison) {
          return false;
        }
      }
    }
    return true;
  }

  [[nodiscard]] std::vector<float> LogicalValues() const {
    std::vector<float> result(extent.width * extent.height);
    for (size_t y = 0; y < extent.height; ++y) {
      std::copy_n(values.data() + y * stride, extent.width,
                  result.data() + y * extent.width);
    }
    return result;
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::vector<float> values;
};

void FillImage(ImageStorage *image, float distortion = 0.0f) {
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < image->extent.height; ++y) {
      for (size_t x = 0; x < image->extent.width; ++x) {
        const float wave =
            std::sin(0.31f * static_cast<float>(3 * x + 5 * y + channel));
        image->plane[channel][y * image->stride + x] =
            0.08f * static_cast<float>(channel + 1) +
            0.003f * static_cast<float>(x + 7 * y) + 0.04f * wave +
            distortion * static_cast<float>(1 + ((x + 2 * y + channel) % 3));
      }
    }
  }
}

[[nodiscard]] gjxl::Status Compute(const ImageStorage &reference,
                                   const ImageStorage &distorted,
                                   gjxl::ButteraugliOptions options,
                                   MapStorage *map, double *score) {
  return gjxl::ComputeButteraugliDistance(reference.ConstView(),
                                          distorted.ConstView(), options,
                                          map->View(), score);
}

[[nodiscard]] bool ValidMapAndScore(const MapStorage &map, double score) {
  if (!std::isfinite(score) || score < 0.0 || !map.PaddingIsUntouched()) {
    return false;
  }
  for (size_t y = 0; y < map.extent.height; ++y) {
    for (size_t x = 0; x < map.extent.width; ++x) {
      const float value = map.values[y * map.stride + x];
      if (!std::isfinite(value) || value < 0.0f) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool CheckImagesOptionsAndStrides() {
  constexpr std::array<gjxl::Extent2D, 5> kExtents = {
      gjxl::Extent2D{1, 1}, gjxl::Extent2D{3, 7}, gjxl::Extent2D{8, 8},
      gjxl::Extent2D{15, 17}, gjxl::Extent2D{32, 24}};
  for (gjxl::Extent2D extent : kExtents) {
    ImageStorage reference(extent, 3);
    ImageStorage distorted(extent, 7);
    FillImage(&reference);
    FillImage(&distorted, 0.004f);
    MapStorage map(extent, 6);
    double score = kScorePoison;
    gjxl::ButteraugliOptions options;
    if (extent == gjxl::Extent2D{15, 17}) {
      options = {.hf_asymmetry = 1.35f,
                 .x_multiplier = 0.8f,
                 .intensity_target = 120.0f};
    }
    if (!Compute(reference, distorted, options, &map, &score).ok() ||
        !ValidMapAndScore(map, score) || !reference.PaddingIsUntouched() ||
        !distorted.PaddingIsUntouched() || score <= 0.0) {
      return false;
    }

    MapStorage identity(extent, 2);
    double identity_score = kScorePoison;
    if (!Compute(reference, reference, options, &identity, &identity_score)
             .ok() ||
        std::abs(identity_score) > kIdentityTolerance ||
        !std::ranges::all_of(identity.LogicalValues(), [](float value) {
          return std::abs(value) <= kIdentityTolerance;
        })) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckExactOutputAliasing() {
  constexpr gjxl::Extent2D kExtent{17, 9};
  ImageStorage reference(kExtent, 5);
  ImageStorage distorted(kExtent, 6);
  FillImage(&reference);
  FillImage(&distorted, 0.007f);
  MapStorage expected(kExtent, 4);
  double expected_score = kScorePoison;
  if (!Compute(reference, distorted, {}, &expected, &expected_score).ok()) {
    return false;
  }

  double alias_score = kScorePoison;
  const gjxl::PlaneF32View alias{reference.plane[1].data(), kExtent,
                                 reference.stride};
  if (!gjxl::ComputeButteraugliDistance(reference.ConstView(),
                                        distorted.ConstView(), {}, alias,
                                        &alias_score)
           .ok() ||
      alias_score != expected_score) {
    return false;
  }
  for (size_t y = 0; y < kExtent.height; ++y) {
    if (!std::equal(expected.values.data() + y * expected.stride,
                    expected.values.data() + y * expected.stride + kExtent.width,
                    reference.plane[1].data() + y * reference.stride)) {
      return false;
    }
  }
  return reference.PaddingIsUntouched() && distorted.PaddingIsUntouched();
}

[[nodiscard]] bool FailsAtomically(gjxl::ConstImage3FView reference,
                                   gjxl::ConstImage3FView distorted,
                                   gjxl::ButteraugliOptions options,
                                   gjxl::PlaneF32View output,
                                   double *score,
                                   gjxl::StatusCode expected_code,
                                   const std::vector<float> &expected_output,
                                   double expected_score) {
  const gjxl::Status status = gjxl::ComputeButteraugliDistance(
      reference, distorted, options, output, score);
  return status.code() == expected_code &&
         (output.data == nullptr ||
          std::equal(expected_output.begin(), expected_output.end(),
                     output.data)) &&
         (score == nullptr || *score == expected_score);
}

[[nodiscard]] bool CheckInvalidAndComputedFailures() {
  constexpr gjxl::Extent2D kExtent{9, 8};
  ImageStorage reference(kExtent);
  ImageStorage distorted(kExtent);
  FillImage(&reference);
  FillImage(&distorted, 0.005f);
  MapStorage map(kExtent);
  const std::vector<float> original = map.values;
  double score = kScorePoison;
  const auto check_invalid = [&](gjxl::ConstImage3FView requested_reference,
                                 gjxl::ConstImage3FView requested_distorted,
                                 gjxl::ButteraugliOptions options,
                                 gjxl::PlaneF32View output = {}) {
    if (output.data == nullptr && output.extent.empty()) {
      output = map.View();
    }
    return FailsAtomically(requested_reference, requested_distorted, options,
                           output, &score,
                           gjxl::StatusCode::kInvalidArgument, original,
                           kScorePoison);
  };

  gjxl::ConstImage3FView invalid = reference.ConstView();
  invalid.plane[0].data = nullptr;
  if (!check_invalid(invalid, distorted.ConstView(), {}) ||
      !check_invalid(reference.ConstView(), distorted.ConstView(),
                     {.hf_asymmetry = 0.0f}) ||
      !check_invalid(reference.ConstView(), distorted.ConstView(),
                     {.x_multiplier =
                          std::numeric_limits<float>::infinity()}) ||
      !check_invalid(reference.ConstView(), distorted.ConstView(),
                     {.intensity_target =
                          std::numeric_limits<float>::quiet_NaN()}) ||
      !check_invalid(reference.ConstView(), distorted.ConstView(), {},
                     {map.values.data(), kExtent, kExtent.width - 1})) {
    return false;
  }
  invalid = distorted.ConstView();
  invalid.plane[2].extent.width -= 1;
  if (!check_invalid(reference.ConstView(), invalid, {})) {
    return false;
  }
  ImageStorage mismatched({8, 8});
  FillImage(&mismatched);
  if (!check_invalid(reference.ConstView(), mismatched.ConstView(), {})) {
    return false;
  }
  invalid = distorted.ConstView();
  invalid.plane[1].stride = std::numeric_limits<size_t>::max();
  if (!check_invalid(reference.ConstView(), invalid, {})) {
    return false;
  }
  reference.plane[0][0] = std::numeric_limits<float>::quiet_NaN();
  if (!check_invalid(reference.ConstView(), distorted.ConstView(), {})) {
    return false;
  }
  FillImage(&reference);
  distorted.plane[2][3] = std::numeric_limits<float>::infinity();
  if (!check_invalid(reference.ConstView(), distorted.ConstView(), {})) {
    return false;
  }
  FillImage(&distorted, 0.005f);

  if (gjxl::ComputeButteraugliDistance(reference.ConstView(),
                                       distorted.ConstView(), {}, map.View(),
                                       nullptr)
          .code() != gjxl::StatusCode::kInvalidArgument ||
      map.values != original) {
    return false;
  }
  const gjxl::PlaneF32View null_output{nullptr, kExtent, kExtent.width};
  if (!FailsAtomically(reference.ConstView(), distorted.ConstView(), {},
                       null_output, &score,
                       gjxl::StatusCode::kInvalidArgument, original,
                       kScorePoison)) {
    return false;
  }

  for (std::vector<float> &plane : reference.plane) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      std::fill_n(plane.data() + y * reference.stride, kExtent.width,
                  std::numeric_limits<float>::max());
    }
  }
  if (!FailsAtomically(reference.ConstView(), distorted.ConstView(),
                       {.intensity_target =
                            std::numeric_limits<float>::max()},
                       map.View(), &score, gjxl::StatusCode::kInternal,
                       original, kScorePoison)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckAllocationFailures() {
  constexpr gjxl::Extent2D kExtent{16, 12};
  ImageStorage reference(kExtent);
  ImageStorage distorted(kExtent);
  FillImage(&reference);
  FillImage(&distorted, 0.003f);

  size_t successful_allocations = 0;
  {
    MapStorage map(kExtent);
    double score = kScorePoison;
    allocation_failure::attempts = 0;
    allocation_failure::fail_at = std::numeric_limits<size_t>::max();
    allocation_failure::enabled = true;
    const gjxl::Status status = Compute(reference, distorted, {}, &map, &score);
    allocation_failure::enabled = false;
    successful_allocations = allocation_failure::attempts;
    if (!status.ok() || successful_allocations == 0) {
      return false;
    }
  }

  for (size_t fail_at = 0; fail_at < successful_allocations; ++fail_at) {
    MapStorage map(kExtent);
    const std::vector<float> original = map.values;
    double score = kScorePoison;
    allocation_failure::attempts = 0;
    allocation_failure::fail_at = fail_at;
    allocation_failure::enabled = true;
    const gjxl::Status status = Compute(reference, distorted, {}, &map, &score);
    allocation_failure::enabled = false;
    if (status.code() != gjxl::StatusCode::kOutOfMemory ||
        map.values != original || score != kScorePoison) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  if (!CheckImagesOptionsAndStrides() || !CheckExactOutputAliasing() ||
      !CheckInvalidAndComputedFailures() || !CheckAllocationFailures()) {
    return EXIT_FAILURE;
  }
  std::cout << "All native Butteraugli facade tests passed.\n";
  return EXIT_SUCCESS;
}
