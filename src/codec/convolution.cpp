// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/convolution.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace gjxl {
namespace {

size_t Mirror(ptrdiff_t coordinate, size_t size) {
  ptrdiff_t mirrored = coordinate;
  const ptrdiff_t signed_size = static_cast<ptrdiff_t>(size);

  while (mirrored < 0 || mirrored >= signed_size) {
    if (mirrored < 0) {
      mirrored = -mirrored - 1;
    } else {
      mirrored = 2 * signed_size - 1 - mirrored;
    }
  }

  return static_cast<size_t>(mirrored);
}

float WeightedRow(
  ConstPlaneF32View input,
  ptrdiff_t x,
  ptrdiff_t y,
  float center_weight,
  float near_weight,
  float far_weight) {

  const float* row = input.Row(Mirror(y, input.extent.height));
  const auto sample = [&](ptrdiff_t sample_x) {
    return row[Mirror(sample_x, input.extent.width)];
  };

  const float far = far_weight * (sample(x - 2) + sample(x + 2));
  const float near = near_weight * (sample(x - 1) + sample(x + 1));
  const float center = center_weight * sample(x);
  return far + (near + center);
}

}  // namespace

Status ConvolveSymmetric5(
  ConstPlaneF32View input,
  Symmetric5Weights weights,
  PlaneF32View output) {

  if (!input.valid() ||
      !output.valid() ||
      input.extent != output.extent ||
      input.data == output.data) {
    return Status::InvalidArgument(
      "Symmetric5 planes are invalid, aliased, or differently sized");
  }

  if (input.extent.width >
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
      input.extent.height >
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
    return Status::InvalidArgument(
      "Symmetric5 plane dimensions are too large");
  }

  const std::array weight_values = {
    weights.distance0,
    weights.distance1,
    weights.distance2,
    weights.distance4,
    weights.distance8,
    weights.distance5,
  };
  for (float weight : weight_values) {
    if (!std::isfinite(weight)) {
      return Status::InvalidArgument(
        "Symmetric5 weights must be finite");
    }
  }

  for (size_t y = 0; y < input.extent.height; ++y) {
    float* destination = output.Row(y);
    for (size_t x = 0; x < input.extent.width; ++x) {
      const ptrdiff_t sx = static_cast<ptrdiff_t>(x);
      const ptrdiff_t sy = static_cast<ptrdiff_t>(y);
      float sum0 = WeightedRow(
        input,
        sx,
        sy,
        weights.distance0,
        weights.distance1,
        weights.distance2);
      sum0 += WeightedRow(
        input,
        sx,
        sy - 2,
        weights.distance2,
        weights.distance5,
        weights.distance8);
      float sum1 = WeightedRow(
        input,
        sx,
        sy + 2,
        weights.distance2,
        weights.distance5,
        weights.distance8);
      sum0 += WeightedRow(
        input,
        sx,
        sy - 1,
        weights.distance1,
        weights.distance4,
        weights.distance5);
      sum1 += WeightedRow(
        input,
        sx,
        sy + 1,
        weights.distance1,
        weights.distance4,
        weights.distance5);
      destination[x] = sum0 + sum1;
    }
  }

  return Status::Ok();
}

}  // namespace gjxl
