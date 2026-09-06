// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "c_api/image_conversion.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace gjxl::c_api_internal {
namespace {

[[nodiscard]] const std::array<float, 256>& SrgbToLinearTable() {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> values{};
    for (size_t i = 0; i < values.size(); ++i) {
      const double encoded = static_cast<double>(i) / 255.0;
      const double linear = encoded <= 0.04045
        ? encoded / 12.92
        : std::pow((encoded + 0.055) / 1.055, 2.4);
      values[i] = static_cast<float>(linear);
    }
    return values;
  }();
  return table;
}

[[nodiscard]] bool TryRequiredSize(size_t row_bytes, size_t row_stride,
                                   size_t height, size_t* required_size) {
  const size_t preceding_rows = height - 1;
  if (preceding_rows != 0 &&
      row_stride > std::numeric_limits<size_t>::max() / preceding_rows) {
    return false;
  }
  const size_t preceding_size = preceding_rows * row_stride;
  if (row_bytes > std::numeric_limits<size_t>::max() - preceding_size) {
    return false;
  }
  *required_size = preceding_size + row_bytes;
  return true;
}

}  // namespace

Status ConvertPackedSrgbToLinearRgb(PackedSrgbImageView image,
                                    Image3FBuffer* linear_rgb) {
  if (linear_rgb == nullptr) {
    return Status::InvalidArgument("Output image must not be null");
  }
  if (image.width == 0 || image.height == 0) {
    return Status::InvalidArgument("Image dimensions must be nonzero");
  }

  size_t bytes_per_pixel = 0;
  switch (image.format) {
    case PackedPixelFormat::kRgb8Srgb:
      bytes_per_pixel = 3;
      break;
    case PackedPixelFormat::kRgba8Srgb:
      bytes_per_pixel = 4;
      break;
    case PackedPixelFormat::kInvalid:
      return Status::InvalidArgument("Pixel format is invalid");
    default:
      return Status::InvalidArgument("Pixel format is not recognized");
  }

  const size_t width = image.width;
  const size_t height = image.height;
  if (width > std::numeric_limits<size_t>::max() / bytes_per_pixel) {
    return Status::InvalidArgument("Image row size overflows size_t");
  }
  const size_t row_bytes = width * bytes_per_pixel;
  if (image.row_stride_bytes < row_bytes) {
    return Status::InvalidArgument("Image row stride is too small");
  }

  size_t required_size = 0;
  if (!TryRequiredSize(row_bytes, image.row_stride_bytes, height,
                       &required_size)) {
    return Status::InvalidArgument("Image buffer size overflows size_t");
  }
  if (image.pixels == nullptr) {
    return Status::InvalidArgument("Image pixels must not be null");
  }
  if (image.pixels_size < required_size) {
    return Status::InvalidArgument("Image pixel buffer is too small");
  }

  const Extent2D extent{width, height};
  size_t pixel_count = 0;
  if (!extent.try_area(&pixel_count)) {
    return Status::InvalidArgument("Image dimensions overflow size_t");
  }
  (void)pixel_count;

  if (image.format == PackedPixelFormat::kRgba8Srgb) {
    for (size_t y = 0; y < height; ++y) {
      const uint8_t* row = image.pixels + y * image.row_stride_bytes;
      for (size_t x = 0; x < width; ++x) {
        if (row[4 * x + 3] != 255) {
          return Status::Unsupported("Non-opaque alpha is not supported");
        }
      }
    }
  }

  try {
    Image3FBuffer candidate(extent);
    Image3FView output = candidate.view();
    const std::array<float, 256>& table = SrgbToLinearTable();
    for (size_t y = 0; y < height; ++y) {
      const uint8_t* input_row = image.pixels + y * image.row_stride_bytes;
      float* red = output.plane[0].Row(y);
      float* green = output.plane[1].Row(y);
      float* blue = output.plane[2].Row(y);
      for (size_t x = 0; x < width; ++x) {
        const size_t input_index = x * bytes_per_pixel;
        red[x] = table[input_row[input_index]];
        green[x] = table[input_row[input_index + 1]];
        blue[x] = table[input_row[input_index + 2]];
      }
    }
    *linear_rgb = std::move(candidate);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Unable to allocate converted image");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("Image dimensions are too large");
  }
}

}  // namespace gjxl::c_api_internal
