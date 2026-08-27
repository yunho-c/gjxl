// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cmath>
#include <cstddef>

#include "codec/butteraugli.h"

int main() {
  constexpr gjxl::Extent2D kExtent{8, 8};
  constexpr size_t kPixels = kExtent.width * kExtent.height;

  std::array<std::array<float, kPixels>, 3> pixels{};
  gjxl::ConstImage3FView image;
  for (size_t channel = 0; channel < pixels.size(); ++channel) {
    image.plane[channel] = {pixels[channel].data(), kExtent, kExtent.width};
  }

  std::array<float, kPixels> distance_map{};
  double score = -1.0;
  const gjxl::Status status = gjxl::ComputeButteraugliDistance(
      image, image, {}, {distance_map.data(), kExtent, kExtent.width}, &score);
  if (!status.ok() || std::abs(score) > 1e-6) {
    return 1;
  }
  for (float distance : distance_map) {
    if (!std::isfinite(distance) || std::abs(distance) > 1e-6f) {
      return 1;
    }
  }
  return 0;
}
