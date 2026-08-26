// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/convolution.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "core/image_ops.h"

namespace gjxl {
namespace {

float WeightedRow(
  ConstPlaneF32View input,
  ptrdiff_t x,
  ptrdiff_t y,
  float center_weight,
  float near_weight,
  float far_weight) {

  const float* row = input.Row(
    MirrorCoordinate(y, input.extent.height));
  const auto sample = [&](ptrdiff_t sample_x) {
    return row[MirrorCoordinate(sample_x, input.extent.width)];
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

  constexpr size_t kRadius = 2;
  constexpr size_t kMaximumDimension =
    static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) - kRadius;
  if (input.extent.width > kMaximumDimension ||
      input.extent.height > kMaximumDimension) {
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
