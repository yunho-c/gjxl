// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the initial CPU quantization heuristic against pinned libjxl.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/adaptive_quantization.h"
#include "codec/quantization.h"

namespace {

constexpr gjxl::Extent2D kPixelExtent{24, 16};
constexpr gjxl::Extent2D kBlockExtent{3, 2};

float OpsinSample(size_t channel, size_t x, size_t y) {
  if (channel == 0) {
    return 0.025f * std::sin(static_cast<float>(x + 1) * 0.37f) -
      0.012f * std::cos(static_cast<float>(y + 2) * 0.23f);
  }
  if (channel == 1) {
    return
      0.12f +
      0.035f * std::sin(static_cast<float>(x + 2) * 0.19f) +
      0.022f * std::cos(static_cast<float>(y + 1) * 0.31f) +
      0.002f * static_cast<float>((x * 7 + y * 11) % 9);
  }
  return
    0.145f +
    0.04f * std::sin(static_cast<float>(x + y + 1) * 0.17f) -
    0.018f * std::cos(static_cast<float>(y + 3) * 0.29f);
}

struct OpsinStorage {
  static constexpr size_t kStride = kPixelExtent.width + 3;
  std::array<std::vector<float>, 3> plane;

  OpsinStorage() {
    for (size_t channel = 0; channel < plane.size(); ++channel) {
      plane[channel].assign(kStride * kPixelExtent.height, -777.0f);
      for (size_t y = 0; y < kPixelExtent.height; ++y) {
        for (size_t x = 0; x < kPixelExtent.width; ++x) {
          plane[channel][y * kStride + x] = OpsinSample(channel, x, y);
        }
      }
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView View() const {
    return {{
      gjxl::ConstPlaneF32View{
        .data = plane[0].data(),
        .extent = kPixelExtent,
        .stride = kStride,
      },
      gjxl::ConstPlaneF32View{
        .data = plane[1].data(),
        .extent = kPixelExtent,
        .stride = kStride,
      },
      gjxl::ConstPlaneF32View{
        .data = plane[2].data(),
        .extent = kPixelExtent,
        .stride = kStride,
      },
    }};
  }
};

struct OutputStorage {
  static constexpr size_t kQuantStride = kBlockExtent.width + 2;
  static constexpr size_t kStrategyMaskStride = kBlockExtent.width + 3;
  static constexpr size_t kPixelMaskStride = kPixelExtent.width + 5;

  std::vector<float> quant_field = std::vector<float>(
    kQuantStride * kBlockExtent.height,
    -777.0f);
  std::vector<float> strategy_mask = std::vector<float>(
    kStrategyMaskStride * kBlockExtent.height,
    -777.0f);
  std::vector<float> pixel_mask = std::vector<float>(
    kPixelMaskStride * kPixelExtent.height,
    -777.0f);

  [[nodiscard]] gjxl::InitialQuantFieldOutput Views() {
    return {
      .quant_field = {
        .data = quant_field.data(),
        .extent = kBlockExtent,
        .stride = kQuantStride,
      },
      .strategy_mask = {
        .data = strategy_mask.data(),
        .extent = kBlockExtent,
        .stride = kStrategyMaskStride,
      },
      .pixel_mask = {
        .data = pixel_mask.data(),
        .extent = kPixelExtent,
        .stride = kPixelMaskStride,
      },
    };
  }
};

bool CheckInitialQuantDc() {
  struct Golden {
    float target;
    float quant_dc;
  };

  constexpr std::array kGoldens = {
    Golden{0.1f, 1.095923996e+01f},
    Golden{0.3f, 3.653079987e+00f},
    Golden{1.0f, 1.344837666e+00f},
    Golden{3.0f, 5.403301120e-01f},
    Golden{10.0f, 1.989160776e-01f},
  };

  for (const Golden& golden : kGoldens) {
    float quant_dc = 0.0f;
    if (!gjxl::ComputeInitialQuantDc(golden.target, &quant_dc).ok() ||
        std::abs(quant_dc - golden.quant_dc) > 2.0e-7f) {
      std::cerr << "Initial DC quantization differs from pinned libjxl\n";
      return false;
    }
  }

  float unused = 0.0f;
  if (gjxl::ComputeInitialQuantDc(0.0f, &unused).ok() ||
      gjxl::ComputeInitialQuantDc(
        std::numeric_limits<float>::infinity(),
        &unused).ok() ||
      gjxl::ComputeInitialQuantDc(1.0f, nullptr).ok()) {
    std::cerr << "Invalid initial DC quantization request was accepted\n";
    return false;
  }

  return true;
}

bool CheckOutputPadding(const OutputStorage& storage) {
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = kBlockExtent.width;
         x < OutputStorage::kQuantStride;
         ++x) {
      if (storage.quant_field[y * OutputStorage::kQuantStride + x] !=
          -777.0f) {
        return false;
      }
    }
    for (size_t x = kBlockExtent.width;
         x < OutputStorage::kStrategyMaskStride;
         ++x) {
      if (storage.strategy_mask[
            y * OutputStorage::kStrategyMaskStride + x] != -777.0f) {
        return false;
      }
    }
  }

  for (size_t y = 0; y < kPixelExtent.height; ++y) {
    for (size_t x = kPixelExtent.width;
         x < OutputStorage::kPixelMaskStride;
         ++x) {
      if (storage.pixel_mask[y * OutputStorage::kPixelMaskStride + x] !=
          -777.0f) {
        return false;
      }
    }
  }
  return true;
}

bool CheckPinnedInitialQuantField() {
  constexpr std::array<float, 6> kFieldGolden = {
    4.385071397e-01f, 4.170474410e-01f, 3.802016675e-01f,
    4.195064306e-01f, 3.758289814e-01f, 3.278555572e-01f,
  };
  constexpr std::array<float, 6> kStrategyMaskGolden = {
    1.366233081e-01f, 1.349961460e-01f, 1.189815924e-01f,
    1.334643960e-01f, 1.285881549e-01f, 1.042966619e-01f,
  };
  constexpr std::array<std::array<size_t, 2>, 12> kPixelSamples = {{
    {0, 0}, {1, 0}, {7, 0}, {8, 0}, {23, 0}, {0, 7},
    {12, 7}, {23, 7}, {0, 15}, {8, 15}, {16, 15}, {23, 15},
  }};
  constexpr std::array<float, 12> kPixelMaskGolden = {
    6.355996704e+01f, 6.824401855e+01f, 8.048500061e+01f,
    7.717355347e+01f, 6.027917099e+01f, 6.490002441e+01f,
    6.095648193e+01f, 5.276998901e+01f, 6.248714828e+01f,
    6.458613586e+01f, 5.048929596e+01f, 5.768811417e+01f,
  };

  const OpsinStorage opsin;
  OutputStorage output;
  const gjxl::InitialQuantFieldOutput views = output.Views();
  const gjxl::Status status = gjxl::ComputeInitialQuantField(
    opsin.View(),
    {
      .butteraugli_target = 1.35f,
      .rescale = 0.85f,
    },
    views);

  if (!status.ok()) {
    std::cerr
      << "Initial quant field failed: "
      << status.message()
      << '\n';
    return false;
  }

  constexpr float kBlockTolerance = 3.0e-6f;
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      const size_t golden_index = y * kBlockExtent.width + x;
      if (std::abs(
            views.quant_field.Row(y)[x] -
            kFieldGolden[golden_index]) > kBlockTolerance ||
          std::abs(
            views.strategy_mask.Row(y)[x] -
            kStrategyMaskGolden[golden_index]) > kBlockTolerance) {
        std::cerr << "Initial block maps differ from pinned libjxl\n";
        return false;
      }
    }
  }

  for (size_t i = 0; i < kPixelSamples.size(); ++i) {
    const size_t x = kPixelSamples[i][0];
    const size_t y = kPixelSamples[i][1];
    if (std::abs(
          views.pixel_mask.Row(y)[x] -
          kPixelMaskGolden[i]) > 3.0e-4f) {
      std::cerr << "Initial pixel mask differs from pinned libjxl\n";
      return false;
    }
  }

  if (!CheckOutputPadding(output)) {
    std::cerr << "Initial quantization overwrote output padding\n";
    return false;
  }

  std::array<int32_t, 10> raw_quant;
  raw_quant.fill(-777);
  float quant_dc = 0.0f;
  gjxl::Quantizer quantizer;
  if (!gjxl::ComputeInitialQuantDc(1.35f, &quant_dc).ok() ||
      !gjxl::CreateQuantizerFromField(
        quant_dc,
        {
          .data = views.quant_field.data,
          .extent = views.quant_field.extent,
          .stride = views.quant_field.stride,
        },
        {
          .data = raw_quant.data(),
          .extent = kBlockExtent,
          .stride = 5,
        },
        &quantizer).ok() ||
      !quantizer.valid()) {
    std::cerr << "Initial field did not integrate with the quantizer\n";
    return false;
  }

  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      if (raw_quant[y * 5 + x] < 1) {
        return false;
      }
    }
    if (raw_quant[y * 5 + 3] != -777 || raw_quant[y * 5 + 4] != -777) {
      return false;
    }
  }

  return true;
}

bool CheckRescaleContract() {
  const OpsinStorage opsin;
  OutputStorage lower;
  OutputStorage higher;
  const gjxl::InitialQuantFieldOutput lower_views = lower.Views();
  const gjxl::InitialQuantFieldOutput higher_views = higher.Views();

  if (!gjxl::ComputeInitialQuantField(
        opsin.View(),
        {.butteraugli_target = 1.35f, .rescale = 0.5f},
        lower_views).ok() ||
      !gjxl::ComputeInitialQuantField(
        opsin.View(),
        {.butteraugli_target = 1.35f, .rescale = 1.0f},
        higher_views).ok()) {
    return false;
  }

  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      if (std::abs(
            2.0f * lower_views.quant_field.Row(y)[x] -
            higher_views.quant_field.Row(y)[x]) > 2.0e-6f ||
          lower_views.strategy_mask.Row(y)[x] !=
            higher_views.strategy_mask.Row(y)[x]) {
        std::cerr << "Initial quantization rescale contract failed\n";
        return false;
      }
    }
  }

  for (size_t y = 0; y < kPixelExtent.height; ++y) {
    for (size_t x = 0; x < kPixelExtent.width; ++x) {
      if (lower_views.pixel_mask.Row(y)[x] !=
          higher_views.pixel_mask.Row(y)[x]) {
        return false;
      }
    }
  }

  return true;
}

bool CheckTileBoundaryGoldens() {
  constexpr gjxl::Extent2D kLargePixels{272, 272};
  constexpr gjxl::Extent2D kLargeBlocks{34, 34};
  constexpr std::array<std::array<size_t, 2>, 12> kBlockSamples = {{
    {0, 0}, {31, 0}, {32, 0}, {33, 0},
    {31, 31}, {32, 31}, {31, 32}, {32, 32},
    {33, 32}, {32, 33}, {33, 33}, {0, 33},
  }};
  constexpr std::array<float, 12> kFieldGolden = {
    4.385071397e-01f, 3.713508546e-01f, 3.738154769e-01f,
    4.455635548e-01f, 3.474409878e-01f, 3.763015568e-01f,
    3.554690480e-01f, 3.843790293e-01f, 4.314543903e-01f,
    3.384233415e-01f, 4.107479751e-01f, 4.373983145e-01f,
  };
  constexpr std::array<float, 12> kStrategyMaskGolden = {
    1.366233081e-01f, 1.153223887e-01f, 1.238829941e-01f,
    1.393090636e-01f, 1.064833105e-01f, 1.147263125e-01f,
    1.185827851e-01f, 1.222814620e-01f, 1.378332078e-01f,
    1.204926372e-01f, 1.359787732e-01f, 1.362827867e-01f,
  };
  constexpr std::array<std::array<size_t, 2>, 12> kPixelSamples = {{
    {0, 0}, {255, 0}, {256, 0}, {271, 0},
    {255, 255}, {256, 255}, {255, 256}, {256, 256},
    {271, 256}, {256, 271}, {271, 271}, {0, 271},
  }};
  constexpr std::array<float, 12> kPixelMaskGolden = {
    6.355996704e+01f, 6.414007568e+01f, 5.606499481e+01f,
    7.190331268e+01f, 4.526648331e+01f, 4.538715363e+01f,
    5.918914032e+01f, 4.889200211e+01f, 7.494873810e+01f,
    6.031741714e+01f, 5.968778610e+01f, 6.480493927e+01f,
  };

  std::array<std::vector<float>, 3> planes;
  for (size_t channel = 0; channel < planes.size(); ++channel) {
    planes[channel].resize(kLargePixels.width * kLargePixels.height);
    for (size_t y = 0; y < kLargePixels.height; ++y) {
      for (size_t x = 0; x < kLargePixels.width; ++x) {
        planes[channel][y * kLargePixels.width + x] =
          OpsinSample(channel, x, y);
      }
    }
  }
  const gjxl::ConstImage3FView opsin{{
    gjxl::ConstPlaneF32View{
      .data = planes[0].data(),
      .extent = kLargePixels,
      .stride = kLargePixels.width,
    },
    gjxl::ConstPlaneF32View{
      .data = planes[1].data(),
      .extent = kLargePixels,
      .stride = kLargePixels.width,
    },
    gjxl::ConstPlaneF32View{
      .data = planes[2].data(),
      .extent = kLargePixels,
      .stride = kLargePixels.width,
    },
  }};

  std::vector<float> quant_field(
    kLargeBlocks.width * kLargeBlocks.height);
  std::vector<float> strategy_mask(
    kLargeBlocks.width * kLargeBlocks.height);
  std::vector<float> pixel_mask(
    kLargePixels.width * kLargePixels.height);
  const gjxl::InitialQuantFieldOutput output{
    .quant_field = {
      .data = quant_field.data(),
      .extent = kLargeBlocks,
      .stride = kLargeBlocks.width,
    },
    .strategy_mask = {
      .data = strategy_mask.data(),
      .extent = kLargeBlocks,
      .stride = kLargeBlocks.width,
    },
    .pixel_mask = {
      .data = pixel_mask.data(),
      .extent = kLargePixels,
      .stride = kLargePixels.width,
    },
  };

  if (!gjxl::ComputeInitialQuantField(
        opsin,
        {.butteraugli_target = 1.35f, .rescale = 0.85f},
        output).ok()) {
    return false;
  }

  for (size_t i = 0; i < kBlockSamples.size(); ++i) {
    const size_t x = kBlockSamples[i][0];
    const size_t y = kBlockSamples[i][1];
    if (std::abs(output.quant_field.Row(y)[x] - kFieldGolden[i]) >
          3.0e-6f ||
        std::abs(output.strategy_mask.Row(y)[x] -
          kStrategyMaskGolden[i]) > 3.0e-6f) {
      std::cerr << "Initial block map differs across a libjxl tile edge\n";
      return false;
    }
  }

  for (size_t i = 0; i < kPixelSamples.size(); ++i) {
    const size_t x = kPixelSamples[i][0];
    const size_t y = kPixelSamples[i][1];
    if (std::abs(output.pixel_mask.Row(y)[x] - kPixelMaskGolden[i]) >
        3.0e-4f) {
      std::cerr << "Initial pixel mask differs across a libjxl tile edge\n";
      return false;
    }
  }

  return true;
}

bool CheckInvalidInputs() {
  OpsinStorage opsin;
  OutputStorage output;
  const gjxl::InitialQuantFieldOutput views = output.Views();

  if (gjxl::ComputeInitialQuantField(
        opsin.View(),
        {.butteraugli_target = 0.0f},
        views).ok() ||
      gjxl::ComputeInitialQuantField(
        opsin.View(),
        {.butteraugli_target = 1.0f, .rescale = 0.0f},
        views).ok()) {
    std::cerr << "Invalid initial quantization options were accepted\n";
    return false;
  }

  gjxl::InitialQuantFieldOutput wrong_extent = views;
  wrong_extent.quant_field.extent = {2, 2};
  if (gjxl::ComputeInitialQuantField(
        opsin.View(),
        {},
        wrong_extent).ok()) {
    std::cerr << "Invalid initial quantization output was accepted\n";
    return false;
  }

  gjxl::ConstImage3FView unpadded = opsin.View();
  for (auto& plane : unpadded.plane) {
    plane.extent.width = 23;
  }
  if (gjxl::ComputeInitialQuantField(unpadded, {}, views).ok()) {
    std::cerr << "Unpadded opsin image was accepted\n";
    return false;
  }

  opsin.plane[1][5] = std::numeric_limits<float>::quiet_NaN();
  if (gjxl::ComputeInitialQuantField(opsin.View(), {}, views).ok()) {
    std::cerr << "Non-finite opsin sample was accepted\n";
    return false;
  }

  opsin.plane[1][5] = std::numeric_limits<float>::max();
  if (gjxl::ComputeInitialQuantField(opsin.View(), {}, views).ok()) {
    std::cerr << "Overflowing initial quantization input was accepted\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!CheckInitialQuantDc() ||
      !CheckPinnedInitialQuantField() ||
      !CheckRescaleContract() ||
      !CheckTileBoundaryGoldens() ||
      !CheckInvalidInputs()) {
    return EXIT_FAILURE;
  }

  std::cout << "All initial quantization tests passed.\n";
  return EXIT_SUCCESS;
}
