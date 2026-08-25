// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates DC/LLF conversion and its fixed-quantization integration.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "codec/dc_conversion.h"
#include "codec/quantization.h"
#include "dct_reference.h"

namespace {

using gjxl::AcStrategyType;

struct StrategyCase {
  AcStrategyType strategy;
  std::string_view name;
};

constexpr std::array kStrategyCases = {
  StrategyCase{AcStrategyType::kDct8, "DCT8"},
  StrategyCase{AcStrategyType::kDct16x16, "DCT16x16"},
  StrategyCase{AcStrategyType::kDct32x32, "DCT32x32"},
  StrategyCase{AcStrategyType::kDct16x8, "DCT16x8"},
  StrategyCase{AcStrategyType::kDct8x16, "DCT8x16"},
  StrategyCase{AcStrategyType::kDct32x16, "DCT32x16"},
  StrategyCase{AcStrategyType::kDct16x32, "DCT16x32"},
};

struct LlfGolden {
  AcStrategyType strategy;
  size_t value_count;
  std::array<float, 16> extracted_dc;
  std::array<float, 16> restored_llf;
};

// Full active outputs from libjxl e8ff09762481785938d8e4e01333ed3917571161.
constexpr std::array kLlfGoldens = {
  LlfGolden{
    AcStrategyType::kDct8,
    1,
    {-2.300000191e-02f},
    {-5.600000173e-02f}},
  LlfGolden{
    AcStrategyType::kDct16x16,
    4,
    {
      -1.960193366e-02f, -1.557689905e-02f,
      -1.148670912e-03f, -5.567250401e-02f,
    },
    {
      1.749999821e-03f, 1.940640272e-03f,
      -2.134704404e-02f, -4.949712381e-02f,
    }},
  LlfGolden{
    AcStrategyType::kDct32x32,
    16,
    {
      -4.547302797e-02f, -1.238510758e-02f,
      -2.373458631e-02f, -6.850932539e-02f,
      -3.690365329e-02f, 2.021856233e-02f,
      5.680320412e-02f, -3.801216558e-02f,
      5.123879388e-02f, -3.171588853e-02f,
      -5.434357747e-02f, 3.628167138e-02f,
      -7.373011857e-02f, -1.658564955e-01f,
      5.371539388e-03f, 1.275015343e-02f,
    },
    {
      -8.750000969e-03f, 8.007645607e-03f,
      -5.163892092e-10f, 7.048991974e-04f,
      -5.897984374e-03f, -8.390390049e-10f,
      6.920752305e-10f, -2.392912812e-10f,
      -4.463473335e-02f, 0.0f,
      -5.726433217e-10f, 8.201288954e-11f,
      -2.344738320e-02f, 2.662590703e-10f,
      3.550804650e-10f, 2.106440022e-11f,
    }},
  LlfGolden{
    AcStrategyType::kDct16x8,
    2,
    {-1.758941635e-02f, -2.841058746e-02f},
    {3.499999642e-03f, -6.598177552e-02f}},
  LlfGolden{
    AcStrategyType::kDct8x16,
    2,
    {-1.758941635e-02f, -2.841058746e-02f},
    {3.499999642e-03f, -6.598177552e-02f}},
  LlfGolden{
    AcStrategyType::kDct32x16,
    8,
    {
      -2.914492041e-02f, -4.590610787e-02f,
      -3.717178944e-03f, 4.770154133e-03f,
      7.716278080e-03f, -6.985777989e-03f,
      -1.029247567e-01f, -7.807706948e-03f,
    },
    {
      -1.750000753e-03f, 4.003822803e-03f,
      -5.163892092e-10f, 3.524508211e-04f,
      -2.134704590e-02f, -1.752097905e-02f,
      -5.726433217e-10f, -5.239420012e-02f,
    }},
  LlfGolden{
    AcStrategyType::kDct16x32,
    8,
    {
      -2.914492041e-02f, -3.717180109e-03f,
      7.716279477e-03f, -1.029247567e-01f,
      -4.590610415e-02f, 4.770155065e-03f,
      -6.985778920e-03f, -7.807709277e-03f,
    },
    {
      -1.749999821e-03f, -5.897983443e-03f,
      -4.463473335e-02f, -2.344737947e-02f,
      3.881280776e-03f, -1.433334701e-10f,
      -2.863216608e-10f, 4.286200284e-10f,
    }},
};

float InputPixel(size_t x, size_t y) {
  return
    0.08f * std::sin(static_cast<float>(x + 1) * 0.37f) +
    0.06f * std::cos(static_cast<float>(y + 2) * 0.29f) +
    0.002f * static_cast<float>((x * 11 + y * 7) % 13);
}

double BaseBlockMean(
  std::span<const double> pixels,
  gjxl::Extent2D pixel_extent,
  size_t block_x,
  size_t block_y) {

  double sum = 0.0;
  for (size_t y = 0; y < gjxl::kJxlBlockDimension; ++y) {
    for (size_t x = 0; x < gjxl::kJxlBlockDimension; ++x) {
      sum += pixels[
        (block_y * gjxl::kJxlBlockDimension + y) * pixel_extent.width +
        block_x * gjxl::kJxlBlockDimension + x];
    }
  }

  return sum /
    static_cast<double>(
      gjxl::kJxlBlockDimension * gjxl::kJxlBlockDimension);
}

bool CheckPinnedLibjxlGoldens() {
  constexpr float kTolerance = 2.0e-7f;

  for (const LlfGolden& golden : kLlfGoldens) {
    const gjxl::AcStrategyInfo* info =
      gjxl::GetAcStrategyInfo(golden.strategy);
    if (info == nullptr ||
        golden.value_count !=
          info->covered_blocks.width * info->covered_blocks.height) {
      return false;
    }

    std::vector<float> coefficients(info->coefficient_count());
    for (size_t i = 0; i < coefficients.size(); ++i) {
      coefficients[i] =
        0.001f * static_cast<float>(
          static_cast<int32_t>((i * 29 + 7) % 61) - 30);
    }

    std::array<float, 16> dc{};
    const gjxl::PlaneF32View dc_view{
      .data = dc.data(),
      .extent = info->covered_blocks,
      .stride = info->covered_blocks.width,
    };

    if (!gjxl::ConvertLowFrequenciesToDc(
          golden.strategy,
          coefficients,
          dc_view).ok()) {
      return false;
    }

    for (size_t i = 0; i < golden.value_count; ++i) {
      if (std::abs(dc[i] - golden.extracted_dc[i]) > kTolerance) {
        std::cerr << "DC extraction differs from pinned libjxl\n";
        return false;
      }
    }

    for (size_t y = 0; y < info->covered_blocks.height; ++y) {
      for (size_t x = 0; x < info->covered_blocks.width; ++x) {
        const size_t i = y * info->covered_blocks.width + x;
        dc[i] = 0.007f * static_cast<float>(
          static_cast<int32_t>((i * 17 + 3) % 23) - 11);
      }
    }

    coefficients.assign(info->coefficient_count(), -777.0f);
    if (!gjxl::ConvertDcToLowFrequencies(
          golden.strategy,
          {
            .data = dc.data(),
            .extent = info->covered_blocks,
            .stride = info->covered_blocks.width,
          },
          coefficients).ok()) {
      return false;
    }

    size_t golden_index = 0;
    for (size_t y = 0; y < info->low_frequency_extent().height; ++y) {
      for (size_t x = 0; x < info->low_frequency_extent().width; ++x) {
        const size_t coefficient_index =
          y * info->coefficient_extent().width + x;
        if (std::abs(
              coefficients[coefficient_index] -
              golden.restored_llf[golden_index]) > kTolerance) {
          std::cerr << "LLF restoration differs from pinned libjxl\n";
          return false;
        }
        ++golden_index;
      }
    }
  }

  return true;
}

bool CheckLowFrequencyRoundTrip() {
  for (const StrategyCase& test : kStrategyCases) {
    const gjxl::AcStrategyInfo* info =
      gjxl::GetAcStrategyInfo(test.strategy);
    if (info == nullptr) {
      return false;
    }

    std::vector<float> coefficients(info->coefficient_count());
    for (size_t i = 0; i < coefficients.size(); ++i) {
      coefficients[i] =
        0.001f * static_cast<float>(
          static_cast<int32_t>((i * 29 + 7) % 61) - 30);
    }
    const std::vector<float> original = coefficients;

    const size_t dc_stride = info->covered_blocks.width + 2;
    std::vector<float> dc_storage(
      dc_stride * info->covered_blocks.height,
      -777.0f);
    const gjxl::PlaneF32View dc{
      .data = dc_storage.data(),
      .extent = info->covered_blocks,
      .stride = dc_stride,
    };

    if (!gjxl::ConvertLowFrequenciesToDc(
          test.strategy,
          coefficients,
          dc).ok()) {
      std::cerr << test.name << " DC extraction failed\n";
      return false;
    }

    for (size_t y = 0; y < info->covered_blocks.height; ++y) {
      for (size_t x = info->covered_blocks.width; x < dc_stride; ++x) {
        if (dc.Row(y)[x] != -777.0f) {
          std::cerr << test.name << " DC extraction overwrote padding\n";
          return false;
        }
      }
    }

    for (size_t v = 0; v < info->covered_blocks.height; ++v) {
      for (size_t u = 0; u < info->covered_blocks.width; ++u) {
        coefficients[info->coefficient_index(v, u)] = 123.0f;
      }
    }

    if (!gjxl::ConvertDcToLowFrequencies(
          test.strategy,
          {
            .data = dc.data,
            .extent = dc.extent,
            .stride = dc.stride,
          },
          coefficients).ok()) {
      std::cerr << test.name << " LLF restoration failed\n";
      return false;
    }

    float max_error = 0.0f;
    for (size_t v = 0; v < info->covered_blocks.height; ++v) {
      for (size_t u = 0; u < info->covered_blocks.width; ++u) {
        const size_t index = info->coefficient_index(v, u);
        max_error = std::max(
          max_error,
          std::abs(coefficients[index] - original[index]));
      }
    }

    if (max_error > 2.0e-7f) {
      std::cerr
        << test.name
        << " DC/LLF round trip failed: max error "
        << max_error
        << '\n';
      return false;
    }

    const gjxl::Extent2D coefficient_extent = info->coefficient_extent();
    const gjxl::Extent2D low_frequency_extent =
      info->low_frequency_extent();
    for (size_t y = 0; y < coefficient_extent.height; ++y) {
      for (size_t x = 0; x < coefficient_extent.width; ++x) {
        if ((x >= low_frequency_extent.width ||
             y >= low_frequency_extent.height) &&
            coefficients[y * coefficient_extent.width + x] !=
              original[y * coefficient_extent.width + x]) {
          std::cerr << test.name << " LLF restoration changed AC\n";
          return false;
        }
      }
    }
  }

  return true;
}

bool CheckFixedQuantizedRoundTrip() {
  for (const StrategyCase& test : kStrategyCases) {
    const gjxl::AcStrategyInfo* info =
      gjxl::GetAcStrategyInfo(test.strategy);
    if (info == nullptr) {
      return false;
    }

    const gjxl::Extent2D pixels_extent = info->pixel_extent();
    std::vector<float> pixels(info->coefficient_count());
    for (size_t y = 0; y < pixels_extent.height; ++y) {
      for (size_t x = 0; x < pixels_extent.width; ++x) {
        pixels[y * pixels_extent.width + x] = InputPixel(x, y);
      }
    }

    std::vector<double> reference_coefficients(info->coefficient_count());
    gjxl::test::ReferenceForwardDct(
      pixels_extent,
      pixels.data(),
      reference_coefficients.data(),
      1);

    std::vector<float> coefficients(info->coefficient_count());
    std::transform(
      reference_coefficients.begin(),
      reference_coefficients.end(),
      coefficients.begin(),
      [](double value) {
        return static_cast<float>(value);
      });
    const std::vector<float> original_coefficients = coefficients;

    std::vector<float> dc_storage(
      info->covered_blocks.width * info->covered_blocks.height);
    const gjxl::PlaneF32View dc{
      .data = dc_storage.data(),
      .extent = info->covered_blocks,
      .stride = info->covered_blocks.width,
    };

    if (!gjxl::ConvertLowFrequenciesToDc(
          test.strategy,
          coefficients,
          dc).ok()) {
      std::cerr << test.name << " pipeline DC extraction failed\n";
      return false;
    }

    std::vector<int32_t> raw_quant(
      info->covered_blocks.width * info->covered_blocks.height);
    gjxl::Quantizer quantizer;
    if (!gjxl::CreateUniformQuantizer(
          1.7f,
          3.25f,
          {
            .data = raw_quant.data(),
            .extent = info->covered_blocks,
            .stride = info->covered_blocks.width,
          },
          &quantizer).ok()) {
      std::cerr << test.name << " pipeline quantizer creation failed\n";
      return false;
    }

    std::vector<int32_t> quantized(info->coefficient_count());
    if (!gjxl::QuantizeAcBlock(
          test.strategy,
          quantizer,
          raw_quant.front(),
          {.channel = gjxl::XybChannel::kY},
          coefficients,
          quantized).ok()) {
      std::cerr << test.name << " pipeline AC quantization failed\n";
      return false;
    }

    bool has_quantized_ac = false;
    for (size_t y = 0; y < info->coefficient_extent().height; ++y) {
      for (size_t x = 0; x < info->coefficient_extent().width; ++x) {
        if ((x >= info->low_frequency_extent().width ||
             y >= info->low_frequency_extent().height) &&
            quantized[y * info->coefficient_extent().width + x] != 0) {
          has_quantized_ac = true;
        }
      }
    }
    if (!has_quantized_ac) {
      std::cerr << test.name << " pipeline did not retain any AC\n";
      return false;
    }

    std::vector<float> reconstructed_coefficients(info->coefficient_count());
    if (!gjxl::DequantizeAcBlock(
          test.strategy,
          quantizer,
          raw_quant.front(),
          {.channel = gjxl::XybChannel::kY},
          quantized,
          reconstructed_coefficients).ok() ||
        !gjxl::ConvertDcToLowFrequencies(
          test.strategy,
          {
            .data = dc.data,
            .extent = dc.extent,
            .stride = dc.stride,
          },
          reconstructed_coefficients).ok()) {
      std::cerr << test.name << " pipeline coefficient reconstruction failed\n";
      return false;
    }

    for (size_t v = 0; v < info->covered_blocks.height; ++v) {
      for (size_t u = 0; u < info->covered_blocks.width; ++u) {
        const size_t index = info->coefficient_index(v, u);
        if (quantized[index] != 0 ||
            std::abs(
              reconstructed_coefficients[index] -
              original_coefficients[index]) > 2.0e-7f) {
          std::cerr << test.name << " pipeline LLF preservation failed\n";
          return false;
        }
      }
    }

    std::vector<double> reconstructed_pixels(info->coefficient_count());
    gjxl::test::ReferenceInverseDct(
      pixels_extent,
      reconstructed_coefficients.data(),
      reconstructed_pixels.data(),
      1);

    if (!std::ranges::all_of(
          reconstructed_pixels,
          [](double value) {
            return std::isfinite(value);
          })) {
      std::cerr << test.name << " pipeline produced non-finite pixels\n";
      return false;
    }

    // AC outside the LLF region can contribute to an individual 8x8 spatial
    // mean. Isolate the restored LLF before checking its DC-grid meaning.
    std::vector<float> low_frequency_coefficients(
      info->coefficient_count(),
      0.0f);
    for (size_t v = 0; v < info->covered_blocks.height; ++v) {
      for (size_t u = 0; u < info->covered_blocks.width; ++u) {
        const size_t index = info->coefficient_index(v, u);
        low_frequency_coefficients[index] =
          reconstructed_coefficients[index];
      }
    }

    std::vector<double> low_frequency_pixels(info->coefficient_count());
    gjxl::test::ReferenceInverseDct(
      pixels_extent,
      low_frequency_coefficients.data(),
      low_frequency_pixels.data(),
      1);

    for (size_t block_y = 0;
         block_y < info->covered_blocks.height;
         ++block_y) {
      for (size_t block_x = 0;
           block_x < info->covered_blocks.width;
           ++block_x) {
        const double reconstructed_mean = BaseBlockMean(
          low_frequency_pixels,
          pixels_extent,
          block_x,
          block_y);
        if (std::abs(
              reconstructed_mean -
              static_cast<double>(dc.Row(block_y)[block_x])) > 2.0e-6) {
          std::cerr << test.name << " pipeline DC mean changed\n";
          return false;
        }
      }
    }
  }

  return true;
}

bool CheckInvalidInputs() {
  std::array<float, 64> coefficients{};
  std::array<float, 4> dc_storage{};
  const gjxl::PlaneF32View dc{
    .data = dc_storage.data(),
    .extent = {1, 1},
    .stride = 1,
  };

  if (gjxl::ConvertLowFrequenciesToDc(
        AcStrategyType::kIdentity,
        coefficients,
        dc).ok() ||
      gjxl::ConvertLowFrequenciesToDc(
        AcStrategyType::kDct8,
        std::span<const float>(coefficients).first(63),
        dc).ok() ||
      gjxl::ConvertLowFrequenciesToDc(
        AcStrategyType::kDct8,
        coefficients,
        {.data = dc_storage.data(), .extent = {2, 1}, .stride = 2}).ok() ||
      gjxl::ConvertDcToLowFrequencies(
        static_cast<AcStrategyType>(255),
        {
          .data = dc.data,
          .extent = dc.extent,
          .stride = dc.stride,
        },
        coefficients).ok()) {
    std::cerr << "Invalid DC/LLF operation was accepted\n";
    return false;
  }

  coefficients[0] = std::numeric_limits<float>::infinity();
  if (gjxl::ConvertLowFrequenciesToDc(
        AcStrategyType::kDct8,
        coefficients,
        dc).ok()) {
    std::cerr << "Non-finite LLF coefficient was accepted\n";
    return false;
  }

  coefficients[0] = 0.0f;
  dc_storage[0] = std::numeric_limits<float>::quiet_NaN();
  if (gjxl::ConvertDcToLowFrequencies(
        AcStrategyType::kDct8,
        {
          .data = dc.data,
          .extent = dc.extent,
          .stride = dc.stride,
        },
        coefficients).ok()) {
    std::cerr << "Non-finite DC value was accepted\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!CheckPinnedLibjxlGoldens() ||
      !CheckLowFrequencyRoundTrip() ||
      !CheckFixedQuantizedRoundTrip() ||
      !CheckInvalidInputs()) {
    return EXIT_FAILURE;
  }

  std::cout << "All DC conversion tests passed.\n";
  return EXIT_SUCCESS;
}
