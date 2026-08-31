// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "c_api/image_conversion.h"

namespace {

using gjxl::Image3FBuffer;
using gjxl::Status;
using gjxl::StatusCode;
using gjxl::c_api_internal::ConvertPackedSrgbToLinearRgb;
using gjxl::c_api_internal::PackedPixelFormat;
using gjxl::c_api_internal::PackedSrgbImageView;

bool Equal(const Image3FBuffer& left, const Image3FBuffer& right) {
  if (left.extent() != right.extent())
    return false;
  for (size_t channel = 0; channel < 3; ++channel) {
    const std::span<const float> left_plane = left.plane(channel);
    const std::span<const float> right_plane = right.plane(channel);
    for (size_t i = 0; i < left_plane.size(); ++i) {
      if (std::bit_cast<uint32_t>(left_plane[i]) !=
          std::bit_cast<uint32_t>(right_plane[i])) {
        return false;
      }
    }
  }
  return true;
}

Image3FBuffer Sentinel() {
  Image3FBuffer image({2, 2});
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t i = 0; i < image.plane(channel).size(); ++i) {
      image.plane(channel)[i] =
        static_cast<float>(100 * channel + i) + 0.25f;
    }
  }
  return image;
}

bool ExpectRejected(std::string_view name, PackedSrgbImageView view,
                    StatusCode expected_code) {
  Image3FBuffer output = Sentinel();
  const Image3FBuffer before = output;
  const Status status = ConvertPackedSrgbToLinearRgb(view, &output);
  if (status.code() != expected_code) {
    std::cerr << name << ": expected status "
              << static_cast<int>(expected_code) << ", got "
              << static_cast<int>(status.code()) << " (" << status.message()
              << ")\n";
    return false;
  }
  if (!Equal(before, output)) {
    std::cerr << name << ": failure modified the output image\n";
    return false;
  }
  return true;
}

float ReferenceSrgbToLinear(uint8_t sample) {
  const double encoded = static_cast<double>(sample) / 255.0;
  return static_cast<float>(encoded <= 0.04045
    ? encoded / 12.92
    : std::pow((encoded + 0.055) / 1.055, 2.4));
}

bool CheckKnownTransferPoints() {
  constexpr std::array<uint8_t, 6> samples{0, 1, 10, 11, 128, 255};
  std::array<uint8_t, samples.size() * 3> pixels{};
  for (size_t x = 0; x < samples.size(); ++x) {
    pixels[3 * x] = samples[x];
    pixels[3 * x + 1] = samples[x];
    pixels[3 * x + 2] = samples[x];
  }

  const PackedSrgbImageView view{
    .pixels = pixels.data(),
    .pixels_size = pixels.size(),
    .width = static_cast<uint32_t>(samples.size()),
    .height = 1,
    .row_stride_bytes = pixels.size(),
    .format = PackedPixelFormat::kRgb8Srgb,
  };
  Image3FBuffer output;
  const Status status = ConvertPackedSrgbToLinearRgb(view, &output);
  if (!status.ok() || output.extent() != gjxl::Extent2D{samples.size(), 1}) {
    std::cerr << "Known transfer points failed: " << status.message() << '\n';
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t x = 0; x < samples.size(); ++x) {
      const float expected = ReferenceSrgbToLinear(samples[x]);
      if (std::fabs(output.plane(channel)[x] - expected) > 1.0e-7f) {
        std::cerr << "Incorrect sRGB transfer result at sample "
                  << static_cast<int>(samples[x]) << '\n';
        return false;
      }
    }
  }
  return true;
}

bool CheckPaddedRgbAndRgba() {
  constexpr uint32_t width = 3;
  constexpr uint32_t height = 2;
  constexpr size_t rgb_stride = 13;
  constexpr size_t rgba_stride = 17;
  constexpr std::array<std::array<uint8_t, 3>, width * height> colors{{
    {{0, 10, 255}},
    {{40, 50, 60}},
    {{70, 80, 90}},
    {{100, 110, 120}},
    {{130, 140, 150}},
    {{200, 220, 240}},
  }};
  std::vector<uint8_t> rgb(rgb_stride * height, 0xa5);
  std::vector<uint8_t> rgba(rgba_stride * height, 0);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const auto& color = colors[y * width + x];
      for (size_t channel = 0; channel < 3; ++channel) {
        rgb[y * rgb_stride + 3 * x + channel] = color[channel];
        rgba[y * rgba_stride + 4 * x + channel] = color[channel];
      }
      rgba[y * rgba_stride + 4 * x + 3] = 255;
    }
  }

  const PackedSrgbImageView rgb_view{
    .pixels = rgb.data(),
    .pixels_size = rgb.size(),
    .width = width,
    .height = height,
    .row_stride_bytes = rgb_stride,
    .format = PackedPixelFormat::kRgb8Srgb,
  };
  const PackedSrgbImageView rgba_view{
    .pixels = rgba.data(),
    .pixels_size = rgba.size(),
    .width = width,
    .height = height,
    .row_stride_bytes = rgba_stride,
    .format = PackedPixelFormat::kRgba8Srgb,
  };
  Image3FBuffer rgb_output;
  Image3FBuffer rgba_output;
  const Status rgb_status = ConvertPackedSrgbToLinearRgb(rgb_view, &rgb_output);
  const Status rgba_status =
    ConvertPackedSrgbToLinearRgb(rgba_view, &rgba_output);
  if (!rgb_status.ok() || !rgba_status.ok() ||
      !Equal(rgb_output, rgba_output)) {
    std::cerr << "Padded RGB/RGBA conversion failed: "
              << rgb_status.message() << ' ' << rgba_status.message() << '\n';
    return false;
  }
  return true;
}

bool CheckSingleRowIgnoresUnusedStride() {
  constexpr std::array<uint8_t, 3> pixels{64, 128, 255};
  const PackedSrgbImageView view{
    .pixels = pixels.data(),
    .pixels_size = pixels.size(),
    .width = 1,
    .height = 1,
    .row_stride_bytes = std::numeric_limits<size_t>::max(),
    .format = PackedPixelFormat::kRgb8Srgb,
  };
  Image3FBuffer output;
  const Status status = ConvertPackedSrgbToLinearRgb(view, &output);
  if (!status.ok() || output.extent() != gjxl::Extent2D{1, 1}) {
    std::cerr << "Single-row stride handling failed: " << status.message()
              << '\n';
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (output.plane(channel)[0] !=
        ReferenceSrgbToLinear(pixels[channel])) {
      std::cerr << "Single-pixel conversion produced the wrong value\n";
      return false;
    }
  }
  return true;
}

bool CheckNonOpaqueAlpha() {
  constexpr std::array<uint8_t, 8> pixels{
    10, 20, 30, 255, 40, 50, 60, 254,
  };
  return ExpectRejected("non-opaque alpha", PackedSrgbImageView{
    .pixels = pixels.data(),
    .pixels_size = pixels.size(),
    .width = 2,
    .height = 1,
    .row_stride_bytes = pixels.size(),
    .format = PackedPixelFormat::kRgba8Srgb,
  }, StatusCode::kUnsupported);
}

bool CheckInvalidViews() {
  constexpr std::array<uint8_t, 3> pixels{1, 2, 3};
  const PackedSrgbImageView valid{
    .pixels = pixels.data(),
    .pixels_size = pixels.size(),
    .width = 1,
    .height = 1,
    .row_stride_bytes = pixels.size(),
    .format = PackedPixelFormat::kRgb8Srgb,
  };

  PackedSrgbImageView invalid = valid;
  invalid.format = PackedPixelFormat::kInvalid;
  if (!ExpectRejected("invalid format", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }
  invalid = valid;
  invalid.format = static_cast<PackedPixelFormat>(99);
  if (!ExpectRejected("unknown format", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }
  invalid = valid;
  invalid.pixels = nullptr;
  if (!ExpectRejected("null pixels", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }
  invalid = valid;
  invalid.width = 0;
  if (!ExpectRejected("zero width", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }
  invalid = valid;
  invalid.height = 0;
  if (!ExpectRejected("zero height", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }
  invalid = valid;
  invalid.row_stride_bytes = 2;
  if (!ExpectRejected("short stride", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }
  invalid = valid;
  invalid.pixels_size = 2;
  if (!ExpectRejected("short buffer", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }

  invalid = valid;
  invalid.height = 3;
  invalid.row_stride_bytes = std::numeric_limits<size_t>::max();
  invalid.pixels_size = std::numeric_limits<size_t>::max();
  if (!ExpectRejected("row span multiplication overflow", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }
  invalid = valid;
  invalid.height = 2;
  invalid.row_stride_bytes = std::numeric_limits<size_t>::max();
  invalid.pixels_size = std::numeric_limits<size_t>::max();
  if (!ExpectRejected("row span addition overflow", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }
  invalid = valid;
  invalid.width = std::numeric_limits<uint32_t>::max();
  invalid.row_stride_bytes = std::numeric_limits<size_t>::max();
  if (!ExpectRejected("huge row", invalid,
                      StatusCode::kInvalidArgument)) {
    return false;
  }

  const Status null_output = ConvertPackedSrgbToLinearRgb(valid, nullptr);
  if (null_output.code() != StatusCode::kInvalidArgument) {
    std::cerr << "Null output was not rejected\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckKnownTransferPoints() || !CheckPaddedRgbAndRgba() ||
      !CheckSingleRowIgnoresUnusedStride() || !CheckNonOpaqueAlpha() ||
      !CheckInvalidViews()) {
    return 1;
  }
  return 0;
}
