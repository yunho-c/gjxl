// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gjxl/modular.hpp"
#include <filesystem>
namespace gjxl::io {
struct IntegerImage {
  std::vector<uint8_t> bytes;
  Extent2D extent;
  ModularPixelFormat format = ModularPixelFormat::kRgb8;
  size_t row_stride = 0;
  ModularImageView view() const {
    return {bytes, extent, row_stride, format, ModularByteOrder::kBigEndian};
  }
};
// Binary P5/P6/P7 only; maxval 255/65535; gray, RGB or unassociated RGBA.
// File byte order is big endian. Caller must explicitly declare source sRGB.
[[nodiscard]] Status ReadPnm(const std::filesystem::path &path, IntegerImage *out);
} // namespace gjxl::io
