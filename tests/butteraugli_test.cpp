// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the pinned Butteraugli adapter and AQ block-map reduction.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "codec/butteraugli.h"

namespace {

constexpr gjxl::Extent2D kExtent{32, 24};
constexpr size_t kStride = 35;
constexpr std::string_view kPinnedLibjxlRevision =
  "e8ff09762481785938d8e4e01333ed3917571161";

enum class Fixture {
  kFlat,
  kTexture,
  kContrast,
  kChromatic,
};

struct FixtureGolden {
  double score;
  std::array<float, 4> samples;
};

struct ImageStorage {
  std::array<std::vector<float>, 3> plane;

  explicit ImageStorage(float fill = -777.0f) {
    for (std::vector<float>& values : plane) {
      values.assign(kStride * kExtent.height, fill);
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView View() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[1].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }
};

void FillFixture(
  Fixture fixture,
  ImageStorage* reference,
  ImageStorage* distorted) {

  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      for (size_t channel = 0; channel < 3; ++channel) {
        float value = 0.0f;
        float error = 0.0f;
        switch (fixture) {
          case Fixture::kFlat:
            value = 0.18f + 0.04f * static_cast<float>(channel);
            if (x >= 10 && x < 22 && y >= 7 && y < 17) {
              error = 0.012f * static_cast<float>(channel + 1);
            }
            break;
          case Fixture::kTexture:
            value = 0.25f +
              0.11f * std::sin(
                0.41f * static_cast<float>((channel + 1) * x + 2 * y)) +
              0.07f * std::cos(
                0.27f * static_cast<float>(3 * x - y));
            error = 0.018f * std::sin(
              0.73f * static_cast<float>(5 * x + 3 * y + channel));
            break;
          case Fixture::kContrast:
            value = ((x / 4 + y / 4) & 1u) == 0 ? 0.02f : 0.92f;
            error = (x % 4 == 0 || y % 4 == 0)
              ? (value < 0.5f ? 0.035f : -0.035f)
              : 0.0f;
            break;
          case Fixture::kChromatic:
            value = channel == 0
              ? 0.05f + 0.75f * static_cast<float>(x) / 31.0f
              : channel == 1
                ? 0.08f + 0.65f * static_cast<float>(y) / 23.0f
                : 0.72f - 0.55f * static_cast<float>(x + y) / 54.0f;
            error = channel == 0
              ? 0.02f * std::sin(0.31f * static_cast<float>(y))
              : channel == 2
                ? -0.018f * std::cos(0.29f * static_cast<float>(x))
                : 0.0f;
            break;
        }
        reference->plane[channel][y * kStride + x] = value;
        distorted->plane[channel][y * kStride + x] = value + error;
      }
    }
  }
}

bool CheckFixture(
  Fixture fixture,
  std::string_view name,
  const FixtureGolden& golden) {
  ImageStorage reference;
  ImageStorage distorted;
  FillFixture(fixture, &reference, &distorted);
  constexpr size_t kMapStride = kExtent.width + 5;
  std::vector<float> map(kMapStride * kExtent.height, -777.0f);
  double score = -1.0;
  const gjxl::Status status = gjxl::ComputeButteraugliDistance(
    reference.View(),
    distorted.View(),
    {},
    {map.data(), kExtent, kMapStride},
    &score);
  if (!status.ok() || !std::isfinite(score) || score <= 0.0) {
    std::cerr << "Butteraugli fixture failed: " << name << '\n';
    return false;
  }
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      if (!std::isfinite(map[y * kMapStride + x]) ||
          map[y * kMapStride + x] < 0.0f) {
        return false;
      }
    }
    for (size_t x = kExtent.width; x < kMapStride; ++x) {
      if (map[y * kMapStride + x] != -777.0f) {
        std::cerr << "Butteraugli overwrote map padding\n";
        return false;
      }
    }
  }

  const std::array<float, 4> samples = {
    map[0],
    map[8 * kMapStride + 8],
    map[12 * kMapStride + 16],
    map[23 * kMapStride + 31],
  };
  if (std::abs(score - golden.score) > 3.0e-6 ||
      !std::equal(
        samples.begin(),
        samples.end(),
        golden.samples.begin(),
        [](float actual, float expected) {
          return std::abs(actual - expected) <= 3.0e-6f;
        })) {
    std::cerr << "Butteraugli fixture differs from libjxl "
              << kPinnedLibjxlRevision << ": " << name << '\n';
    return false;
  }
  return true;
}

bool CheckIdentityAndValidation() {
  ImageStorage image;
  ImageStorage unused;
  FillFixture(Fixture::kTexture, &image, &unused);
  std::vector<float> map(kExtent.width * kExtent.height, -1.0f);
  double score = -1.0;
  if (!gjxl::ComputeButteraugliDistance(
        image.View(),
        image.View(),
        {},
        {map.data(), kExtent, kExtent.width},
        &score).ok() ||
      score != 0.0 ||
      !std::ranges::all_of(map, [](float value) { return value == 0.0f; })) {
    std::cerr << "Identical images have nonzero Butteraugli distance\n";
    return false;
  }

  const std::vector<float> original_map = map;
  const double original_score = score;
  auto bad_options = gjxl::ButteraugliOptions{};
  bad_options.intensity_target = 0.0f;
  if (gjxl::ComputeButteraugliDistance(
        image.View(),
        image.View(),
        bad_options,
        {map.data(), kExtent, kExtent.width},
        &score).ok() ||
      map != original_map ||
      score != original_score) {
    std::cerr << "Invalid Butteraugli request was not atomic\n";
    return false;
  }
  return true;
}

bool CheckBlockReduction() {
  constexpr gjxl::Extent2D kBlockExtent{4, 4};
  constexpr gjxl::Extent2D kPixelExtent{32, 32};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kBlockExtent, &strategies).ok() ||
      !strategies.Set(0, 0, gjxl::AcStrategyType::kDct32x32).ok()) {
    return false;
  }
  std::vector<float> distance(kPixelExtent.width * kPixelExtent.height, 2.0f);
  std::vector<float> blocks(kBlockExtent.width * kBlockExtent.height, -1.0f);
  if (!gjxl::ReduceButteraugliDistanceMap(
        {distance.data(), kPixelExtent, kPixelExtent.width},
        strategies,
        {blocks.data(), kBlockExtent, kBlockExtent.width}).ok()) {
    return false;
  }
  for (float value : blocks) {
    if (std::abs(value - 2.4f) > 2.0e-6f) {
      std::cerr << "Butteraugli 16-norm block reduction is incorrect\n";
      return false;
    }
  }

  distance.back() = -1.0f;
  const std::vector<float> original = blocks;
  if (gjxl::ReduceButteraugliDistanceMap(
        {distance.data(), kPixelExtent, kPixelExtent.width},
        strategies,
        {blocks.data(), kBlockExtent, kBlockExtent.width}).ok() ||
      blocks != original) {
    std::cerr << "Invalid Butteraugli block reduction was not atomic\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  constexpr FixtureGolden kFlatGolden{
    7.416567802,
    {1.542445898f, 3.690321922f, 7.4165411f, 1.542400122f},
  };
  constexpr FixtureGolden kTextureGolden{
    0.7470397949,
    {0.1901933849f, 0.2465418875f, 0.3287552297f, 0.2964144051f},
  };
  constexpr FixtureGolden kContrastGolden{
    2.239220142,
    {2.239220142f, 1.164466739f, 0.9660935998f, 0.9154187441f},
  };
  constexpr FixtureGolden kChromaticGolden{
    1.367423773,
    {1.202098608f, 0.6952123642f, 0.5296805501f, 0.5132343769f},
  };
  if (!CheckFixture(Fixture::kFlat, "flat", kFlatGolden) ||
      !CheckFixture(Fixture::kTexture, "texture", kTextureGolden) ||
      !CheckFixture(Fixture::kContrast, "contrast", kContrastGolden) ||
      !CheckFixture(Fixture::kChromatic, "chromatic", kChromaticGolden) ||
      !CheckIdentityAndValidation() ||
      !CheckBlockReduction()) {
    return EXIT_FAILURE;
  }
  std::cout << "All Butteraugli tests passed.\n";
  return EXIT_SUCCESS;
}
