// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>

#include "core/image_buffer.h"
#include "core/status.h"

namespace gjxl::c_api_internal {

enum class PackedPixelFormat : uint32_t {
  kInvalid = 0,
  kRgb8Srgb = 1,
  kRgba8Srgb = 2,
};

/// A checked view of packed sRGB pixels supplied by a future C API caller.
struct PackedSrgbImageView {
  const uint8_t* pixels = nullptr;
  size_t pixels_size = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  size_t row_stride_bytes = 0;
  PackedPixelFormat format = PackedPixelFormat::kInvalid;
};

/// Converts RGB8 or opaque RGBA8 sRGB pixels to planar linear sRGB.
///
/// The output is replaced only when conversion succeeds. Non-opaque alpha is
/// well-formed input, but is outside the encoder profile implemented so far.
[[nodiscard]] Status ConvertPackedSrgbToLinearRgb(
  PackedSrgbImageView image, Image3FBuffer* linear_rgb);

}  // namespace gjxl::c_api_internal
