// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates default 4:4:4 VarDCT DC quantization and dequantization.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/dc_quantization.h"

namespace {

constexpr gjxl::Extent2D kExtent{2, 2};
constexpr size_t kStride = 4;

struct FloatStorage {
  explicit FloatStorage(float fill)
      : plane{std::vector<float>(kStride * kExtent.height, fill),
              std::vector<float>(kStride * kExtent.height, fill),
              std::vector<float>(kStride * kExtent.height, fill)} {}

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
        gjxl::PlaneF32View{plane[0].data(), kExtent, kStride},
        gjxl::PlaneF32View{plane[1].data(), kExtent, kStride},
        gjxl::PlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
        gjxl::ConstPlaneF32View{plane[0].data(), kExtent, kStride},
        gjxl::ConstPlaneF32View{plane[1].data(), kExtent, kStride},
        gjxl::ConstPlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }

  std::array<std::vector<float>, 3> plane;
};

struct IntStorage {
  explicit IntStorage(int32_t fill)
      : plane{std::vector<int32_t>(kStride * kExtent.height, fill),
              std::vector<int32_t>(kStride * kExtent.height, fill),
              std::vector<int32_t>(kStride * kExtent.height, fill)} {}

  [[nodiscard]] gjxl::Image3I32View View() {
    return {{
        gjxl::PlaneI32View{plane[0].data(), kExtent, kStride},
        gjxl::PlaneI32View{plane[1].data(), kExtent, kStride},
        gjxl::PlaneI32View{plane[2].data(), kExtent, kStride},
    }};
  }

  std::array<std::vector<int32_t>, 3> plane;
};

void FillDc(FloatStorage *dc) {
  constexpr std::array<std::array<float, 4>, 3> kValues = {{
      {{0.125f, -0.125f, 1.0f / 8192.0f, -1.0f / 8192.0f}},
      {{0.25f, -0.25f, 1.0f / 1024.0f, -1.0f / 1024.0f}},
      {{0.75f, -0.75f, 1.0f / 256.0f, -1.0f / 256.0f}},
  }};
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t index = 0; index < 4; ++index) {
      dc->plane[channel][(index / kExtent.width) * kStride +
                         index % kExtent.width] = kValues[channel][index];
    }
  }
}

bool CheckExpected(const IntStorage &quantized,
                   const FloatStorage &reconstructed, float float_padding,
                   int32_t int_padding) {

  // These are the direct outputs of the default libjxl 4:4:4 equations for
  // Quantizer{1024, 64}. The last two samples pin half-away-from-zero rounding.
  constexpr std::array<std::array<int32_t, 4>, 3> kQuantized = {{
      {{512, -512, 1, -1}},
      {{128, -128, 1, -1}},
      {{128, -128, 1, -1}},
  }};
  constexpr std::array<std::array<float, 4>, 3> kReconstructed = {{
      {{0.125f, -0.125f, 1.0f / 4096.0f, -1.0f / 4096.0f}},
      {{0.25f, -0.25f, 1.0f / 512.0f, -1.0f / 512.0f}},
      {{0.75f, -0.75f, 3.0f / 512.0f, -3.0f / 512.0f}},
  }};
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        const size_t index = y * kExtent.width + x;
        const size_t stored = y * kStride + x;
        if (quantized.plane[channel][stored] != kQuantized[channel][index] ||
            reconstructed.plane[channel][stored] !=
                kReconstructed[channel][index]) {
          return false;
        }
      }
      for (size_t x = kExtent.width; x < kStride; ++x) {
        const size_t stored = y * kStride + x;
        if (quantized.plane[channel][stored] != int_padding ||
            reconstructed.plane[channel][stored] != float_padding) {
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckDefaultQuantizationAndAliasing() {
  gjxl::Quantizer quantizer;
  if (!gjxl::Quantizer::Create({1024, 64}, &quantizer).ok()) {
    return false;
  }

  FloatStorage dc(-91.0f);
  FillDc(&dc);
  IntStorage quantized(-777);
  FloatStorage reconstructed(-123.0f);
  gjxl::Status status = gjxl::QuantizeDcCoefficients(
      dc.ConstView(), quantizer,
      {.quantized = quantized.View(), .reconstructed = reconstructed.View()});
  if (!status.ok() || !CheckExpected(quantized, reconstructed, -123.0f, -777)) {
    std::cerr << "Default DC quantization differs from libjxl equations: "
              << status.message() << '\n';
    return false;
  }

  FloatStorage in_place(-91.0f);
  FillDc(&in_place);
  IntStorage alias_quantized(-555);
  status = gjxl::QuantizeDcCoefficients(
      in_place.ConstView(), quantizer,
      {.quantized = alias_quantized.View(), .reconstructed = in_place.View()});
  if (!status.ok() || !CheckExpected(alias_quantized, in_place, -91.0f, -555)) {
    std::cerr << "In-place DC reconstruction is incorrect: " << status.message()
              << '\n';
    return false;
  }
  return true;
}

bool CheckFailureIsAtomic() {
  gjxl::Quantizer quantizer;
  if (!gjxl::Quantizer::Create({1024, 64}, &quantizer).ok()) {
    return false;
  }

  FloatStorage dc(-91.0f);
  FillDc(&dc);
  IntStorage quantized(-777);
  FloatStorage reconstructed(-123.0f);
  const auto original_quantized = quantized.plane;
  const auto original_reconstructed = reconstructed.plane;

  dc.plane[2][kStride + 1] = std::numeric_limits<float>::quiet_NaN();
  if (gjxl::QuantizeDcCoefficients(dc.ConstView(), quantizer,
                                   {.quantized = quantized.View(),
                                    .reconstructed = reconstructed.View()})
          .ok() ||
      quantized.plane != original_quantized ||
      reconstructed.plane != original_reconstructed) {
    std::cerr << "Invalid DC input changed output\n";
    return false;
  }

  FillDc(&dc);
  dc.plane[0][0] = std::numeric_limits<float>::max();
  if (gjxl::QuantizeDcCoefficients(dc.ConstView(), quantizer,
                                   {.quantized = quantized.View(),
                                    .reconstructed = reconstructed.View()})
          .ok() ||
      quantized.plane != original_quantized ||
      reconstructed.plane != original_reconstructed) {
    std::cerr << "Out-of-range DC input changed output\n";
    return false;
  }

  FillDc(&dc);
  gjxl::Quantizer invalid_quantizer;
  if (gjxl::QuantizeDcCoefficients(dc.ConstView(), invalid_quantizer,
                                   {.quantized = quantized.View(),
                                    .reconstructed = reconstructed.View()})
          .ok() ||
      quantized.plane != original_quantized ||
      reconstructed.plane != original_reconstructed) {
    std::cerr << "Invalid quantizer changed DC output\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  if (!CheckDefaultQuantizationAndAliasing() || !CheckFailureIsAtomic()) {
    return EXIT_FAILURE;
  }
  std::cout << "All DC quantization tests passed.\n";
  return EXIT_SUCCESS;
}
