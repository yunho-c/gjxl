// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/gaborish.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "codec/convolution.h"

namespace gjxl {
namespace {

size_t Mirror(ptrdiff_t coordinate, size_t size) {
  ptrdiff_t mirrored = coordinate;
  const ptrdiff_t signed_size = static_cast<ptrdiff_t>(size);
  while (mirrored < 0 || mirrored >= signed_size) {
    if (mirrored < 0) {
      mirrored = -mirrored - 1;
    } else {
      mirrored = 2 * signed_size - mirrored - 1;
    }
  }
  return static_cast<size_t>(mirrored);
}

Symmetric5Weights GaborishWeights(float multiplier) {
  constexpr std::array<float, 5> kGaborish = {
    -0.09495815671340026f,
    -0.041031725066768575f,
    0.013710004822696948f,
    0.006510206083837737f,
    -0.0014789063378272242f,
  };

  double sum = 1.0 + static_cast<double>(multiplier) * 4.0 * (
    kGaborish[0] + kGaborish[1] + kGaborish[2] +
    kGaborish[4] + 2.0 * kGaborish[3]);
  sum = std::max(sum, 1.0e-5);
  const float normalize = static_cast<float>(1.0 / sum);
  const float normalize_multiplier = multiplier * normalize;
  return {
    .distance0 = normalize,
    .distance1 = normalize_multiplier * kGaborish[0],
    .distance2 = normalize_multiplier * kGaborish[2],
    .distance4 = normalize_multiplier * kGaborish[1],
    .distance8 = normalize_multiplier * kGaborish[4],
    .distance5 = normalize_multiplier * kGaborish[3],
  };
}

}  // namespace

Status ApplyGaborishInverse(
  ConstImage3FView input,
  std::array<float, 3> multipliers,
  Image3FView output) {

  if (!input.valid() ||
      !output.valid() ||
      output.extent() != input.extent()) {
    return Status::InvalidArgument(
      "Gaborish input and output images are invalid or differently sized");
  }
  for (float multiplier : multipliers) {
    if (!std::isfinite(multiplier)) {
      return Status::InvalidArgument(
        "Gaborish multipliers must be finite");
    }
  }

  size_t pixel_count = 0;
  if (!input.extent().try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "Gaborish image dimensions are too large");
  }

  try {
    std::array<std::vector<float>, 3> filtered;
    for (size_t channel = 0; channel < filtered.size(); ++channel) {
      filtered[channel].resize(pixel_count);
      Status status = ConvolveSymmetric5(
        input.plane[channel],
        GaborishWeights(multipliers[channel]),
        {
          .data = filtered[channel].data(),
          .extent = input.extent(),
          .stride = input.width(),
        });
      if (!status.ok()) {
        return status;
      }

      if (!std::ranges::all_of(
            filtered[channel],
            [](float value) { return std::isfinite(value); })) {
        return Status::InvalidArgument(
          "Gaborish filtering produced a non-finite result");
      }
    }

    for (size_t channel = 0; channel < filtered.size(); ++channel) {
      for (size_t y = 0; y < input.height(); ++y) {
        std::copy_n(
          filtered[channel].data() + y * input.width(),
          input.width(),
          output.plane[channel].Row(y));
      }
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate Gaborish scratch storage");
  }

  return Status::Ok();
}

Status ApplyGaborish(
  ConstImage3FView input,
  GaborishOptions options,
  Image3FView output) {

  if (!input.valid() ||
      !output.valid() ||
      input.extent() != output.extent()) {
    return Status::InvalidArgument(
      "Gaborish input and output images are invalid or differently sized");
  }
  if (input.width() >
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
      input.height() >
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
    return Status::InvalidArgument(
      "Gaborish image dimensions are too large");
  }

  std::array<std::array<float, 3>, 3> weights{};
  for (size_t channel = 0; channel < 3; ++channel) {
    const float weight1 = options.weight1[channel];
    const float weight2 = options.weight2[channel];
    const float divisor = 1.0f + 4.0f * (weight1 + weight2);
    if (!std::isfinite(weight1) ||
        !std::isfinite(weight2) ||
        !std::isfinite(divisor) ||
        std::abs(divisor) < 1.0e-8f) {
      return Status::InvalidArgument(
        "Gaborish weights are invalid");
    }
    weights[channel] = {
      1.0f / divisor,
      weight1 / divisor,
      weight2 / divisor,
    };
  }

  try {
    size_t pixel_count = 0;
    if (!input.extent().try_area(&pixel_count)) {
      return Status::InvalidArgument(
        "Gaborish image dimensions are too large");
    }
    std::array<std::vector<float>, 3> result;
    for (std::vector<float>& plane : result) {
      plane.resize(pixel_count);
    }

    for (size_t channel = 0; channel < 3; ++channel) {
      const float center_weight = weights[channel][0];
      const float axis_weight = weights[channel][1];
      const float diagonal_weight = weights[channel][2];
      for (size_t y = 0; y < input.height(); ++y) {
        for (size_t x = 0; x < input.width(); ++x) {
          const ptrdiff_t sx = static_cast<ptrdiff_t>(x);
          const ptrdiff_t sy = static_cast<ptrdiff_t>(y);
          const auto sample = [&](ptrdiff_t dx, ptrdiff_t dy) {
            return input.plane[channel].Row(
              Mirror(sy + dy, input.height()))[
                Mirror(sx + dx, input.width())];
          };
          const float axes =
            (sample(-1, 0) + sample(1, 0)) +
            (sample(0, -1) + sample(0, 1));
          const float diagonals =
            (sample(-1, -1) + sample(1, -1)) +
            (sample(-1, 1) + sample(1, 1));
          const float value =
            center_weight * sample(0, 0) +
            axis_weight * axes +
            diagonal_weight * diagonals;
          if (!std::isfinite(value)) {
            return Status::InvalidArgument(
              "Gaborish filtering produced a non-finite result");
          }
          result[channel][y * input.width() + x] = value;
        }
      }
    }

    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < input.height(); ++y) {
        std::copy_n(
          result[channel].data() + y * input.width(),
          input.width(),
          output.plane[channel].Row(y));
      }
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate Gaborish scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Gaborish image dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace gjxl
