// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the initial CPU CfL map against pinned libjxl output.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/adaptive_quantization.h"
#include "codec/chroma_from_luma.h"
#include "codec/quantization.h"

namespace {

constexpr gjxl::Extent2D kExtent{80, 72};
constexpr size_t kStride = kExtent.width + 3;

bool CheckColorTileExtent() {
  const gjxl::Extent2D maximum{
    std::numeric_limits<size_t>::max(),
    std::numeric_limits<size_t>::max(),
  };
  const gjxl::Extent2D expected{
    std::numeric_limits<size_t>::max() / gjxl::kColorTileDimension + 1,
    std::numeric_limits<size_t>::max() / gjxl::kColorTileDimension + 1,
  };
  if (gjxl::ColorTileExtent({64, 65}) != gjxl::Extent2D{1, 2} ||
      gjxl::ColorTileExtent(maximum) != expected) {
    std::cerr << "Color-tile extent calculation is incorrect\n";
    return false;
  }
  return true;
}

struct OpsinStorage {
  std::array<std::vector<float>, 3> plane;

  OpsinStorage() {
    constexpr float kRatioX[2][2] = {{0.6f, -0.35f}, {1.1f, 0.15f}};
    constexpr float kRatioB[2][2] = {{1.25f, 0.7f}, {1.6f, 0.9f}};
    for (std::vector<float>& values : plane) {
      values.assign(kStride * kExtent.height, -777.0f);
    }

    for (size_t y = 0; y < kExtent.height; ++y) {
      const size_t tile_y = y / gjxl::kColorTileDimension;
      for (size_t x = 0; x < kExtent.width; ++x) {
        const size_t tile_x = x / gjxl::kColorTileDimension;
        const float luma =
          0.45f * std::sin(0.13f * static_cast<float>(x + 2 * y)) +
          0.27f * std::cos(
            0.17f * (2.0f * static_cast<float>(x) -
                     static_cast<float>(y))) +
          0.08f * std::sin(
            0.021f * static_cast<float>(x * y + 3));
        plane[1][y * kStride + x] = luma;
        plane[0][y * kStride + x] =
          kRatioX[tile_y][tile_x] * luma +
          0.025f * std::sin(0.31f * static_cast<float>(x + y));
        plane[2][y * kStride + x] =
          kRatioB[tile_y][tile_x] * luma +
          0.02f * std::cos(0.23f * static_cast<float>(2 * x + y));
      }
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

bool CheckPinnedMap() {
  constexpr std::array<int8_t, 4> kXGolden = {48, -27, 90, 6};
  constexpr std::array<int8_t, 4> kBGolden = {18, -23, 47, -6};

  const OpsinStorage opsin;
  gjxl::ColorCorrelationMap map;
  const gjxl::Status status = gjxl::ComputeInitialColorCorrelationMap(
    opsin.View(),
    &map);
  if (!status.ok()) {
    std::cerr << "Initial CfL failed: " << status.message() << '\n';
    return false;
  }
  if (!map.valid() || map.tile_extent() != gjxl::Extent2D{2, 2}) {
    std::cerr << "Initial CfL returned invalid map geometry\n";
    return false;
  }

  const gjxl::ConstPlaneI8View x_map = map.y_to_x_map();
  const gjxl::ConstPlaneI8View b_map = map.y_to_b_map();
  for (size_t y = 0; y < 2; ++y) {
    for (size_t x = 0; x < 2; ++x) {
      const size_t index = y * 2 + x;
      if (x_map.Row(y)[x] != kXGolden[index] ||
          b_map.Row(y)[x] != kBGolden[index]) {
        std::cerr
          << "Initial CfL differs from pinned libjxl at tile ("
          << x << ", " << y << "): "
          << static_cast<int>(x_map.Row(y)[x]) << ", "
          << static_cast<int>(b_map.Row(y)[x]) << '\n';
        return false;
      }

      const std::array<float, 3> factors = map.AcFactors(x, y);
      const float expected_x = static_cast<float>(kXGolden[index]) / 84.0f;
      const float expected_b = 1.0f +
        static_cast<float>(kBGolden[index]) / 84.0f;
      if (factors[0] != expected_x ||
          factors[1] != 0.0f ||
          factors[2] != expected_b) {
        std::cerr << "Initial CfL factor conversion is incorrect\n";
        return false;
      }
    }
  }

  const std::array<float, 3> invalid = map.AcFactors(2, 0);
  if (!std::isnan(invalid[0]) ||
      !std::isnan(invalid[1]) ||
      !std::isnan(invalid[2])) {
    std::cerr << "Out-of-range CfL factor lookup was not rejected\n";
    return false;
  }
  return true;
}

bool CheckPinnedFinalMap() {
  constexpr gjxl::Extent2D kPixels{32, 32};
  constexpr gjxl::Extent2D kBlocks{4, 4};
  std::array<std::vector<float>, 3> planes;
  for (std::vector<float>& plane : planes) {
    plane.resize(kPixels.width * kPixels.height);
  }
  for (size_t y = 0; y < kPixels.height; ++y) {
    for (size_t x = 0; x < kPixels.width; ++x) {
      const float luma =
        0.14f + 0.0013f * static_cast<float>(x) +
        0.0007f * static_cast<float>(y) +
        0.012f * std::sin(0.11f * static_cast<float>(x + y));
      planes[1][y * kPixels.width + x] = luma;
      planes[0][y * kPixels.width + x] = 0.35f * luma;
      planes[2][y * kPixels.width + x] = 1.18f * luma;
    }
  }
  const gjxl::ConstImage3FView opsin{{
    gjxl::ConstPlaneF32View{planes[0].data(), kPixels, kPixels.width},
    gjxl::ConstPlaneF32View{planes[1].data(), kPixels, kPixels.width},
    gjxl::ConstPlaneF32View{planes[2].data(), kPixels, kPixels.width},
  }};

  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kBlocks, &strategies).ok() ||
      !strategies.Set(0, 0, gjxl::AcStrategyType::kDct32x32).ok()) {
    return false;
  }
  std::array<float, 16> initial_quant{};
  std::array<float, 16> adjusted_quant{};
  std::array<int32_t, 16> raw_quant{};
  for (size_t y = 0; y < kBlocks.height; ++y) {
    for (size_t x = 0; x < kBlocks.width; ++x) {
      initial_quant[y * kBlocks.width + x] =
        0.48f + 0.01f * static_cast<float>(x + y);
    }
  }
  const gjxl::ConstPlaneF32View initial_view{
    initial_quant.data(), kBlocks, kBlocks.width};
  const gjxl::PlaneF32View adjusted_view{
    adjusted_quant.data(), kBlocks, kBlocks.width};
  const gjxl::PlaneI32View raw_view{
    raw_quant.data(), kBlocks, kBlocks.width};
  if (!gjxl::AdjustQuantField(
        strategies, 3.0f, initial_view, adjusted_view).ok()) {
    return false;
  }
  float quant_dc = 0.0f;
  gjxl::Quantizer quantizer;
  if (!gjxl::ComputeInitialQuantDc(3.0f, &quant_dc).ok() ||
      !gjxl::CreateQuantizerFromField(
        quant_dc,
        gjxl::ConstPlaneF32View{
          adjusted_quant.data(), kBlocks, kBlocks.width},
        raw_view,
        &quantizer).ok() ||
      quantizer.params().global_scale != 3541 ||
      quantizer.params().quant_dc != 10) {
    std::cerr << "Final CfL quantizer differs from pinned libjxl\n";
    return false;
  }
  for (int32_t raw : raw_quant) {
    if (raw != 10) {
      std::cerr << "Final CfL raw quant field differs from pinned libjxl\n";
      return false;
    }
  }

  gjxl::ColorCorrelationMap map;
  const gjxl::ConstPlaneI32View raw_const{
    raw_quant.data(), kBlocks, kBlocks.width};
  if (!gjxl::ComputeFinalColorCorrelationMap(
        opsin, strategies, raw_const, quantizer, false, &map).ok() ||
      map.y_to_x_map().Row(0)[0] != 27 ||
      map.y_to_b_map().Row(0)[0] != 11) {
    std::cerr << "Final CfL map differs from pinned libjxl\n";
    return false;
  }
  if (!gjxl::ComputeFinalColorCorrelationMap(
        opsin, strategies, raw_const, quantizer, true, &map).ok() ||
      map.y_to_x_map().Row(0)[0] != 27 ||
      map.y_to_b_map().Row(0)[0] != 13) {
    std::cerr << "Fast final CfL map differs from pinned libjxl\n";
    return false;
  }

  raw_quant[15] = 0;
  const int8_t original_x = map.y_to_x_map().Row(0)[0];
  const int8_t original_b = map.y_to_b_map().Row(0)[0];
  if (gjxl::ComputeFinalColorCorrelationMap(
        opsin, strategies, raw_const, quantizer, false, &map).ok() ||
      map.y_to_x_map().Row(0)[0] != original_x ||
      map.y_to_b_map().Row(0)[0] != original_b) {
    std::cerr << "Invalid final CfL input changed output\n";
    return false;
  }
  return true;
}

bool CheckValidationAndAtomicCommit() {
  OpsinStorage opsin;
  gjxl::ColorCorrelationMap map;
  if (!gjxl::ComputeInitialColorCorrelationMap(opsin.View(), &map).ok()) {
    return false;
  }
  const std::array<int8_t, 4> original_x = {
    map.y_to_x_map().Row(0)[0],
    map.y_to_x_map().Row(0)[1],
    map.y_to_x_map().Row(1)[0],
    map.y_to_x_map().Row(1)[1],
  };

  opsin.plane[2][65 * kStride + 70] =
    std::numeric_limits<float>::quiet_NaN();
  if (gjxl::ComputeInitialColorCorrelationMap(opsin.View(), &map).ok()) {
    std::cerr << "Non-finite CfL input was accepted\n";
    return false;
  }
  const gjxl::ConstPlaneI8View x_map = map.y_to_x_map();
  if (x_map.Row(0)[0] != original_x[0] ||
      x_map.Row(0)[1] != original_x[1] ||
      x_map.Row(1)[0] != original_x[2] ||
      x_map.Row(1)[1] != original_x[3]) {
    std::cerr << "Failed CfL computation changed the prior output\n";
    return false;
  }

  gjxl::ConstImage3FView unaligned = opsin.View();
  for (gjxl::ConstPlaneF32View& plane : unaligned.plane) {
    plane.extent.width -= 1;
  }
  if (gjxl::ComputeInitialColorCorrelationMap(unaligned, &map).ok() ||
      gjxl::ComputeInitialColorCorrelationMap(opsin.View(), nullptr).ok()) {
    std::cerr << "Invalid CfL geometry or null output was accepted\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckColorTileExtent() ||
      !CheckPinnedMap() ||
      !CheckPinnedFinalMap() ||
      !CheckValidationAndAtomicCommit()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
