// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Scalar arithmetic adapted from pinned libjxl.
#pragma once
#include <array>
#include <cstdint>

namespace gjxl::modular_internal {
// Arithmetic kernels require a validated type and 8/16-bit source-derived samples.
[[nodiscard]] inline std::array<int32_t, 3> ForwardRct(std::array<int32_t, 3> rgb, uint8_t type) {
  const unsigned permutation = type / 7, transform = type % 7;
  int64_t a = rgb[permutation % 3];
  int64_t b = rgb[(permutation + 1 + permutation / 3) % 3];
  int64_t c = rgb[(permutation + 2 - permutation / 3) % 3];
  if (transform == 6) {
    const int64_t co = a - c, tmp = c + (co >> 1), cg = b - tmp;
    a = tmp + (cg >> 1);
    b = co;
    c = cg;
  } else {
    if (transform / 2 == 1)
      b -= a;
    if (transform / 2 == 2)
      b -= (a + c) >> 1;
    if (transform & 1)
      c -= a;
  }
  return {static_cast<int32_t>(a), static_cast<int32_t>(b), static_cast<int32_t>(c)};
}
[[nodiscard]] inline std::array<int32_t, 3> InverseRct(std::array<int32_t, 3> transformed,
                                                       uint8_t type) {
  const unsigned permutation = type / 7, transform = type % 7;
  int64_t a = transformed[0], b = transformed[1], c = transformed[2];
  if (transform == 6) {
    const int64_t tmp = a - (c >> 1), g = c + tmp, blue = tmp - (b >> 1);
    a = blue + b;
    b = g;
    c = blue;
  } else {
    if (transform & 1)
      c += a;
    if (transform / 2 == 1)
      b += a;
    if (transform / 2 == 2)
      b += (a + c) >> 1;
  }
  std::array<int32_t, 3> rgb;
  rgb[permutation % 3] = static_cast<int32_t>(a);
  rgb[(permutation + 1 + permutation / 3) % 3] = static_cast<int32_t>(b);
  rgb[(permutation + 2 - permutation / 3) % 3] = static_cast<int32_t>(c);
  return rgb;
}
} // namespace gjxl::modular_internal
