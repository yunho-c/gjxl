// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "io/pfm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gjxl::io {
namespace {

[[nodiscard]] uint32_t ByteSwap32(uint32_t value) noexcept {
  return ((value & 0x000000ffu) << 24) |
    ((value & 0x0000ff00u) << 8) |
    ((value & 0x00ff0000u) >> 8) |
    ((value & 0xff000000u) >> 24);
}

[[nodiscard]] bool ReadHeaderLine(
  std::istream& stream,
  std::string* line) {

  while (std::getline(stream, *line)) {
    if (!line->empty() && line->back() == '\r') {
      line->pop_back();
    }
    if (!line->empty() && line->front() != '#') {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool ParseDimensions(
  const std::string& line,
  Extent2D* extent) {

  std::istringstream input(line);
  size_t width = 0;
  size_t height = 0;
  std::string trailing;
  if (!(input >> width >> height) || (input >> trailing) ||
      width == 0 || height == 0) {
    return false;
  }
  *extent = {width, height};
  return true;
}

[[nodiscard]] bool ParseScale(
  const std::string& line,
  float* scale) {

  std::istringstream input(line);
  std::string trailing;
  if (!(input >> *scale) || (input >> trailing) ||
      !std::isfinite(*scale) || *scale == 0.0f) {
    return false;
  }
  return true;
}

}  // namespace

Status ReadPfm(
  const std::filesystem::path& path,
  Image3FBuffer* image) {

  if (image == nullptr || path.empty()) {
    return Status::InvalidArgument("PFM path or output is invalid");
  }

  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return Status::InvalidArgument(
      "Unable to open PFM input: " + path.string());
  }

  try {
    std::string line;
    if (!ReadHeaderLine(stream, &line) || line != "PF") {
      return Status::InvalidArgument(
        "PFM input must contain three floating-point channels");
    }
    Extent2D extent;
    if (!ReadHeaderLine(stream, &line) || !ParseDimensions(line, &extent)) {
      return Status::InvalidArgument("PFM dimensions are invalid");
    }
    float scale = 0.0f;
    if (!ReadHeaderLine(stream, &line) || !ParseScale(line, &scale)) {
      return Status::InvalidArgument("PFM scale is invalid");
    }

    size_t pixel_count = 0;
    if (!extent.try_area(&pixel_count) ||
        pixel_count > std::numeric_limits<size_t>::max() / 12) {
      return Status::InvalidArgument("PFM dimensions are too large");
    }
    const size_t byte_count = pixel_count * 12;
    if (byte_count > static_cast<size_t>(
          std::numeric_limits<std::streamsize>::max())) {
      return Status::InvalidArgument("PFM pixel payload is too large");
    }
    std::vector<uint8_t> bytes(byte_count);
    stream.read(
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
    if (stream.gcount() != static_cast<std::streamsize>(bytes.size())) {
      return Status::InvalidArgument("PFM pixel payload is truncated");
    }
    char trailing = 0;
    if (stream.get(trailing)) {
      return Status::InvalidArgument("PFM input has trailing bytes");
    }

    const bool file_is_little_endian = scale < 0.0f;
    const bool host_is_little_endian =
      std::endian::native == std::endian::little;
    const float multiplier = std::abs(scale);
    Image3FBuffer candidate(extent);
    Image3FView output = candidate.view();
    size_t offset = 0;
    for (size_t reverse_y = 0; reverse_y < extent.height; ++reverse_y) {
      const size_t y = extent.height - 1 - reverse_y;
      for (size_t x = 0; x < extent.width; ++x) {
        for (size_t channel = 0; channel < 3; ++channel) {
          uint32_t bits = 0;
          std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
          offset += sizeof(bits);
          if (file_is_little_endian != host_is_little_endian) {
            bits = ByteSwap32(bits);
          }
          const float value = std::bit_cast<float>(bits) * multiplier;
          if (!std::isfinite(value)) {
            return Status::InvalidArgument(
              "PFM input contains a non-finite pixel");
          }
          output.plane[channel].Row(y)[x] = value;
        }
      }
    }
    *image = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Unable to allocate PFM input storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("PFM dimensions are too large");
  }
  return Status::Ok();
}

}  // namespace gjxl::io
