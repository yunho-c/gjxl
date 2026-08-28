// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates CPU coefficient coding and VarDCT reconstruction composition.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/reconstruction.h"

namespace {

constexpr std::array kStrategies = {
  gjxl::AcStrategyType::kDct8,
  gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x32,
  gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16,
  gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
};

constexpr std::string_view kPinnedLibjxlRevision =
  "e8ff09762481785938d8e4e01333ed3917571161";

uint64_t HashQuantized(
  const std::array<std::span<const int32_t>, 3>& coefficients,
  size_t count) {

  uint64_t hash = 1469598103934665603ull;
  for (std::span<const int32_t> channel : coefficients) {
    for (int32_t value : channel.first(count)) {
      hash ^= static_cast<uint32_t>(value);
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

struct ImageStorage {
  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;

  explicit ImageStorage(gjxl::Extent2D image_extent, float fill = -777.0f)
      : extent(image_extent), stride(image_extent.width + 3) {
    for (std::vector<float>& values : plane) {
      values.assign(stride * extent.height, fill);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
      gjxl::PlaneF32View{plane[0].data(), extent, stride},
      gjxl::PlaneF32View{plane[1].data(), extent, stride},
      gjxl::PlaneF32View{plane[2].data(), extent, stride},
    }};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), extent, stride},
      gjxl::ConstPlaneF32View{plane[1].data(), extent, stride},
      gjxl::ConstPlaneF32View{plane[2].data(), extent, stride},
    }};
  }
};

void FillSignal(ImageStorage* image) {
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      const float luma =
        0.24f * std::sin(0.17f * static_cast<float>(x + 2 * y)) +
        0.13f * std::cos(
          0.11f * static_cast<float>(3 * x) -
          0.11f * static_cast<float>(y)) +
        0.0015f * static_cast<float>((17 * x + 11 * y) % 29);
      image->plane[1][y * image->stride + x] = luma;
      image->plane[0][y * image->stride + x] =
        0.55f * luma +
        0.035f * std::sin(0.29f * static_cast<float>(x + y));
      image->plane[2][y * image->stride + x] =
        1.25f * luma +
        0.027f * std::cos(0.23f * static_cast<float>(2 * x + y));
    }
  }
}

bool PaddingIs(const ImageStorage& image, float value) {
  for (const std::vector<float>& plane : image.plane) {
    for (size_t y = 0; y < image.extent.height; ++y) {
      for (size_t x = image.extent.width; x < image.stride; ++x) {
        if (plane[y * image.stride + x] != value) {
          return false;
        }
      }
    }
  }
  return true;
}

gjxl::Status ComputeFrame(
  gjxl::ConstImage3FView input,
  const gjxl::AcStrategyGrid& strategies,
  gjxl::ConstPlaneI32View raw_quant,
  const gjxl::Quantizer& quantizer,
  const gjxl::ColorCorrelationMap& color_correlation,
  gjxl::SimpleVarDctCodestreamProfile profile,
  gjxl::VarDctEncoderFrame* frame) {

  gjxl::FrameGeometry geometry;
  gjxl::Status status = gjxl::FrameGeometry::Create(
    input.extent(), &geometry);
  if (!status.ok()) {
    return status;
  }
  size_t block_count = 0;
  if (!strategies.extent().try_area(&block_count)) {
    return gjxl::Status::InvalidArgument("Test block grid is too large");
  }
  std::vector<uint8_t> epf_sharpness(block_count, 4);
  return gjxl::ComputeQuantizedCoefficients(
    input,
    {
      .geometry = geometry,
      .strategies = &strategies,
      .raw_quant_field = raw_quant,
      .quantizer = &quantizer,
      .color_correlation = &color_correlation,
      .epf_sharpness = {
        epf_sharpness.data(), strategies.extent(), strategies.extent().width},
    },
    profile,
    frame);
}

bool CheckStrategy(gjxl::AcStrategyType strategy) {
  const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
  if (info == nullptr) {
    return false;
  }

  ImageStorage input(info->pixel_extent());
  FillSignal(&input);

  gjxl::AcStrategyGrid grid;
  if (!gjxl::AcStrategyGrid::Create(info->covered_blocks, &grid).ok() ||
      !grid.Set(0, 0, strategy).ok()) {
    return false;
  }

  const size_t block_count =
    info->covered_blocks.width * info->covered_blocks.height;
  std::vector<int32_t> raw_quant(block_count, 37);
  const gjxl::ConstPlaneI32View raw_view{
    raw_quant.data(),
    info->covered_blocks,
    info->covered_blocks.width,
  };
  gjxl::Quantizer quantizer;
  if (!gjxl::Quantizer::Create({3541, 10}, &quantizer).ok()) {
    return false;
  }

  gjxl::ColorCorrelationMap color_correlation;
  if (!gjxl::ComputeInitialColorCorrelationMap(
        input.ConstView(),
        &color_correlation).ok()) {
    return false;
  }

  gjxl::VarDctEncoderFrame frame;
  if (!ComputeFrame(
        input.ConstView(),
        grid,
        raw_view,
        quantizer,
        color_correlation,
        {},
        &frame).ok() ||
      !frame.valid() ||
      frame.ac_group_count() != 1) {
    std::cerr << "Coefficient coding failed for a supported strategy\n";
    return false;
  }

  gjxl::VarDctAcGroupView group;
  gjxl::AcStrategyCell cell;
  if (!frame.GetAcGroup(0, &group).ok() ||
      !frame.strategies().Get(0, 0, &cell).ok() ||
      cell.strategy != strategy ||
      !cell.is_anchor ||
      group.block_x != 0 || group.block_y != 0 ||
      group.used_coefficient_count != info->coefficient_count()) {
    std::cerr << "Stored coefficient metadata is incorrect\n";
    return false;
  }
  if (strategy == gjxl::AcStrategyType::kDct32x32) {
    // Includes the pinned AdjustQuantBlockAC shared-quant and Y dead-zone
    // decisions before X/B color-correlation removal.
    constexpr uint64_t kPinnedQuantizedHash = 0x6c0b64bd3bffc4a7ull;
    // Default 4:4:4 DC quantization of the pinned libjxl LLF outputs for this
    // fixture, in X/Y/B plane order.
    constexpr std::array<std::array<int32_t, 4>, 3> kPinnedQuantizedDc = {{
      {{317, -102, -152, 293}},
      {{62, -28, -24, 69}},
      {{7, -3, -3, 10}},
    }};
    constexpr std::array<size_t, 4> kDcIndices = {0, 3, 10, 15};
    if (color_correlation.y_to_x_map().Row(0)[0] != 46 ||
        color_correlation.y_to_b_map().Row(0)[0] != 19 ||
        HashQuantized(
          group.coefficients,
          group.used_coefficient_count) != kPinnedQuantizedHash) {
      std::cerr << "DCT32 coefficient path differs from libjxl "
                << kPinnedLibjxlRevision << '\n';
      return false;
    }
    const gjxl::ConstImage3I32View quantized_dc = frame.quantized_dc();
    const gjxl::ConstImage3FView reconstructed_dc = frame.dc();
    const auto& dc_steps = quantizer.dc_steps();
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t i = 0; i < kDcIndices.size(); ++i) {
        const size_t index = kDcIndices[i];
        if (quantized_dc.plane[channel].Row(index / 4)[index % 4] !=
            kPinnedQuantizedDc[channel][i]) {
          std::cerr << "DCT32 quantized DC differs from pinned libjxl\n";
          return false;
        }
        float expected =
          static_cast<float>(kPinnedQuantizedDc[channel][i]) *
          dc_steps[channel];
        if (channel == 2) {
          expected +=
            static_cast<float>(kPinnedQuantizedDc[1][i]) * dc_steps[1];
        }
        if (reconstructed_dc.plane[channel].Row(index / 4)[index % 4] !=
            expected) {
          std::cerr << "DCT32 reconstructed DC is not decoder-equivalent\n";
          return false;
        }
      }
    }
  }

  const gjxl::Extent2D coefficient_extent = info->coefficient_extent();
  const gjxl::Extent2D llf_extent = info->low_frequency_extent();
  for (std::span<const int32_t> coefficients : group.coefficients) {
    for (size_t y = 0; y < llf_extent.height; ++y) {
      for (size_t x = 0; x < llf_extent.width; ++x) {
        if (coefficients[y * coefficient_extent.width + x] != 0) {
          std::cerr << "Quantized LLF entry is not zero\n";
          return false;
        }
      }
    }
  }

  ImageStorage reconstructed(info->pixel_extent());
  if (!gjxl::ReconstructQuantizedCoefficients(
        frame,
        reconstructed.View()).ok() ||
      !PaddingIs(reconstructed, -777.0f)) {
    std::cerr << "Coefficient reconstruction failed\n";
    return false;
  }
  double squared_error = 0.0;
  float max_error = 0.0f;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < input.extent.height; ++y) {
      for (size_t x = 0; x < input.extent.width; ++x) {
        const float error = std::abs(
          input.plane[channel][y * input.stride + x] -
          reconstructed.plane[channel][y * reconstructed.stride + x]);
        squared_error += static_cast<double>(error) * error;
        max_error = std::max(max_error, error);
      }
    }
  }
  const double rmse = std::sqrt(
    squared_error /
    static_cast<double>(3 * info->coefficient_count()));
  if (!(rmse < 0.04) || !(max_error < 0.16f)) {
    std::cerr << "Coefficient round trip exceeds its error bound: "
              << rmse << ", " << max_error << '\n';
    return false;
  }

  return true;
}

bool CheckFlatDcQuantization() {
  constexpr gjxl::Extent2D kBlockExtent{4, 4};
  constexpr gjxl::Extent2D kPixelExtent{32, 32};
  constexpr std::array<float, 3> kValues = {0.125f, -0.25f, 0.375f};
  ImageStorage input(kPixelExtent);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kPixelExtent.height; ++y) {
      std::fill_n(
        input.plane[channel].data() + y * input.stride,
        kPixelExtent.width,
        kValues[channel]);
    }
  }

  gjxl::AcStrategyGrid grid;
  if (!gjxl::AcStrategyGrid::Create(kBlockExtent, &grid).ok() ||
      !grid.Set(0, 0, gjxl::AcStrategyType::kDct32x32).ok()) {
    return false;
  }
  std::array<int32_t, 16> raw_quant{};
  raw_quant.fill(1);
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  if (!gjxl::Quantizer::Create({3541, 10}, &quantizer).ok() ||
      !gjxl::ComputeInitialColorCorrelationMap(
        input.ConstView(),
        &color_correlation).ok()) {
    return false;
  }

  gjxl::VarDctEncoderFrame frame;
  if (!ComputeFrame(
        input.ConstView(),
        grid,
        {raw_quant.data(), kBlockExtent, kBlockExtent.width},
        quantizer,
        color_correlation,
        {},
        &frame).ok()) {
    return false;
  }
  if (frame.raw_quant_field().Row(0)[0] != 4) {
    std::cerr << "Flat transform anchor did not retain adjusted raw quant\n";
    return false;
  }
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      if ((x != 0 || y != 0) &&
          frame.raw_quant_field().Row(y)[x] != 1) {
        std::cerr << "Flat covered non-anchor raw quant was overwritten\n";
        return false;
      }
    }
  }
  const auto& inverse_dc_steps = quantizer.inverse_dc_steps();
  const auto& dc_steps = quantizer.dc_steps();
  const int32_t quantized_y = static_cast<int32_t>(
    std::round(kValues[1] * inverse_dc_steps[1]));
  const float reconstructed_y =
    static_cast<float>(quantized_y) * dc_steps[1];
  const std::array<int32_t, 3> expected_quantized = {
    static_cast<int32_t>(std::round(kValues[0] * inverse_dc_steps[0])),
    quantized_y,
    static_cast<int32_t>(std::round(
      (kValues[2] - reconstructed_y) * inverse_dc_steps[2])),
  };
  const std::array<float, 3> expected_reconstructed = {
    static_cast<float>(expected_quantized[0]) * dc_steps[0],
    reconstructed_y,
    static_cast<float>(expected_quantized[2]) * dc_steps[2] +
      reconstructed_y,
  };
  const gjxl::ConstImage3I32View quantized_dc = frame.quantized_dc();
  const gjxl::ConstImage3FView reconstructed_dc = frame.dc();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kBlockExtent.height; ++y) {
      for (size_t x = 0; x < kBlockExtent.width; ++x) {
        if (quantized_dc.plane[channel].Row(y)[x] !=
              expected_quantized[channel] ||
            reconstructed_dc.plane[channel].Row(y)[x] !=
              expected_reconstructed[channel]) {
          std::cerr << "Flat DC was not quantized or reconstructed correctly\n";
          return false;
        }
      }
    }
  }

  ImageStorage reconstructed(kPixelExtent);
  if (!gjxl::ReconstructQuantizedCoefficients(
        frame,
        reconstructed.View()).ok()) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kPixelExtent.height; ++y) {
      for (size_t x = 0; x < kPixelExtent.width; ++x) {
        if (std::abs(
              reconstructed.plane[channel][y * reconstructed.stride + x] -
              expected_reconstructed[channel]) > 3.0e-6f) {
          std::cerr << "Flat DC-only image did not match decoded DC\n";
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckMixedGridAndInvalidInputs() {
  constexpr gjxl::Extent2D kBlockExtent{11, 9};
  constexpr gjxl::Extent2D kPixelExtent{88, 72};
  ImageStorage input(kPixelExtent);
  FillSignal(&input);

  gjxl::AcStrategyGrid grid;
  if (!gjxl::AcStrategyGrid::Create(kBlockExtent, &grid).ok() ||
      !grid.Set(0, 0, gjxl::AcStrategyType::kDct32x32).ok() ||
      !grid.Set(4, 0, gjxl::AcStrategyType::kDct32x16).ok() ||
      !grid.Set(6, 0, gjxl::AcStrategyType::kDct16x32).ok() ||
      !grid.Set(6, 2, gjxl::AcStrategyType::kDct16x16).ok()) {
    return false;
  }
  grid.fill_empty_dct8();

  std::vector<int32_t> raw_quant(
    kBlockExtent.width * kBlockExtent.height,
    23);
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  if (!gjxl::Quantizer::Create({3541, 10}, &quantizer).ok() ||
      !gjxl::ComputeInitialColorCorrelationMap(
        input.ConstView(),
        &color_correlation).ok()) {
    return false;
  }

  gjxl::VarDctEncoderFrame frame;
  gjxl::SimpleVarDctCodestreamProfile profile;
  profile.x_qm_scale = 3;
  profile.b_qm_scale = 1;
  if (!ComputeFrame(
        input.ConstView(),
        grid,
        {raw_quant.data(), kBlockExtent, kBlockExtent.width},
        quantizer,
        color_correlation,
        profile,
        &frame).ok() ||
      !frame.valid() || frame.profile() != profile ||
      gjxl::QuantizationMatrixMultiplier(frame.profile().x_qm_scale) !=
        1.25f ||
      gjxl::QuantizationMatrixMultiplier(frame.profile().b_qm_scale) !=
        0.8f) {
    std::cerr << "Mixed-grid coefficient coding failed\n";
    return false;
  }

  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      gjxl::AcStrategyCell cell;
      if (!grid.Get(x, y, &cell).ok()) return false;
      const int32_t stored = frame.raw_quant_field().Row(y)[x];
      if (!cell.is_anchor &&
          stored != raw_quant[y * kBlockExtent.width + x]) {
        std::cerr << "Covered non-anchor raw quant was overwritten\n";
        return false;
      }
    }
  }

  ImageStorage reconstructed(kPixelExtent);
  if (!gjxl::ReconstructQuantizedCoefficients(
        frame,
        reconstructed.View()).ok()) {
    std::cerr << "Mixed-grid reconstruction failed\n";
    return false;
  }

  gjxl::VarDctEncoderFrame sentinel = frame;
  raw_quant.back() = 0;
  if (ComputeFrame(
        input.ConstView(),
        grid,
        {raw_quant.data(), kBlockExtent, kBlockExtent.width},
        quantizer,
        color_correlation,
        {},
        &sentinel).ok() ||
      !sentinel.valid() ||
      sentinel.raw_quant_field().Row(kBlockExtent.height - 1)[
        kBlockExtent.width - 1] != 23) {
    std::cerr << "Invalid coefficient coding was accepted or not atomic\n";
    return false;
  }

  ImageStorage invalid_output({8, 8}, -123.0f);
  if (gjxl::ReconstructQuantizedCoefficients(
        frame,
        invalid_output.View()).ok() ||
      !std::ranges::all_of(
        invalid_output.plane[0],
        [](float value) { return value == -123.0f; })) {
    std::cerr << "Invalid reconstruction was accepted or not atomic\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  for (gjxl::AcStrategyType strategy : kStrategies) {
    if (!CheckStrategy(strategy)) {
      return EXIT_FAILURE;
    }
  }
  if (!CheckFlatDcQuantization() || !CheckMixedGridAndInvalidInputs()) {
    return EXIT_FAILURE;
  }

  std::cout << "All CPU reconstruction tests passed.\n";
  return EXIT_SUCCESS;
}
