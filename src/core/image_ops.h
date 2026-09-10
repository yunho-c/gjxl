// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "core/image.h"

namespace gjxl {

/// Mirrors a coordinate into [0, size), repeating each edge sample once.
///
/// `size` must be non-zero and representable by ptrdiff_t.
[[nodiscard]] inline size_t MirrorCoordinate(
  ptrdiff_t coordinate,
  size_t size) noexcept {

  assert(size != 0);
  assert(size <=
    static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()));

  if (coordinate >= 0 && static_cast<size_t>(coordinate) < size) {
    return static_cast<size_t>(coordinate);
  }

  using UnsignedCoordinate = std::make_unsigned_t<ptrdiff_t>;
  const auto unsigned_size = static_cast<UnsignedCoordinate>(size);
  const UnsignedCoordinate period = 2 * unsigned_size;

  UnsignedCoordinate phase = 0;
  if (coordinate >= 0) {
    phase = static_cast<UnsignedCoordinate>(coordinate) % period;
  } else {
    const UnsignedCoordinate magnitude =
      static_cast<UnsignedCoordinate>(-(coordinate + 1)) + 1;
    const UnsignedCoordinate remainder = magnitude % period;
    phase = remainder == 0 ? 0 : period - remainder;
  }

  const UnsignedCoordinate mirrored = phase < unsigned_size
    ? phase
    : period - 1 - phase;
  return static_cast<size_t>(mirrored);
}

template <typename T>
void CopyPlane(
  PlaneView<const T> source,
  PlaneView<T> destination) {

  assert(source.valid());
  assert(destination.valid());
  assert(source.extent == destination.extent);
  for (size_t y = 0; y < source.extent.height; ++y) {
    if (source.Row(y) == destination.Row(y)) {
      continue;
    }
    std::copy_n(
      source.Row(y),
      source.extent.width,
      destination.Row(y));
  }
}

template <typename T, typename Allocator>
void CopyContiguousPlane(
  const std::vector<T, Allocator>& source,
  PlaneView<T> destination) {

  [[maybe_unused]] size_t value_count = 0;
  assert(destination.valid());
  assert(destination.extent.try_area(&value_count));
  assert(source.size() >= value_count);
  for (size_t y = 0; y < destination.extent.height; ++y) {
    std::copy_n(
      source.data() + y * destination.extent.width,
      destination.extent.width,
      destination.Row(y));
  }
}

inline void CopyImage(
  ConstImage3FView source,
  Image3FView destination) {

  assert(source.valid());
  assert(destination.valid());
  assert(source.extent() == destination.extent());
  for (size_t channel = 0; channel < source.plane.size(); ++channel) {
    CopyPlane(source.plane[channel], destination.plane[channel]);
  }
}

}  // namespace gjxl
