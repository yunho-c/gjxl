// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/butteraugli.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "codec/butteraugli_distance_internal.h"

namespace gjxl {

Status ComputeButteraugliDistance(
  ConstImage3FView reference_linear_rgb,
  ConstImage3FView distorted_linear_rgb,
  ButteraugliOptions options,
  PlaneF32View distance_map,
  double* score) {

  butteraugli_internal::NativeButteraugliScratch scratch;
  const butteraugli_internal::NativeButteraugliParams params{
    .hf_asymmetry = options.hf_asymmetry,
    .x_multiplier = options.x_multiplier,
    .intensity_target = options.intensity_target,
  };
  return butteraugli_internal::ComputeButteraugliDistanceNative(
    reference_linear_rgb,
    distorted_linear_rgb,
    params,
    &scratch,
    distance_map,
    score);
}

Status ReduceButteraugliDistanceMap(
  ConstPlaneF32View distance_map,
  const AcStrategyGrid& strategies,
  PlaneF32View block_distance_map) {

  if (!distance_map.valid() ||
      !strategies.complete() ||
      !block_distance_map.valid() ||
      block_distance_map.extent != strategies.extent() ||
      strategies.extent().width >
        std::numeric_limits<size_t>::max() / kJxlBlockDimension ||
      strategies.extent().height >
        std::numeric_limits<size_t>::max() / kJxlBlockDimension ||
      distance_map.extent.width <=
        (strategies.extent().width - 1) * kJxlBlockDimension ||
      distance_map.extent.height <=
        (strategies.extent().height - 1) * kJxlBlockDimension ||
      distance_map.extent.width >
        strategies.extent().width * kJxlBlockDimension ||
      distance_map.extent.height >
        strategies.extent().height * kJxlBlockDimension) {
    return Status::InvalidArgument(
      "Butteraugli reduction inputs are invalid or differently sized");
  }
  for (size_t y = 0; y < distance_map.extent.height; ++y) {
    for (size_t x = 0; x < distance_map.extent.width; ++x) {
      if (!std::isfinite(distance_map.Row(y)[x]) ||
          distance_map.Row(y)[x] < 0.0f) {
        return Status::InvalidArgument(
          "Butteraugli distance map values must be finite and non-negative");
      }
    }
  }

  try {
    size_t block_count = 0;
    if (!strategies.extent().try_area(&block_count)) {
      return Status::InvalidArgument(
        "Butteraugli block map dimensions are too large");
    }
    std::vector<float> result(block_count);
    Status status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr) {
          return Status::Internal(
            "Butteraugli reduction encountered an unknown strategy");
        }
        const size_t x_begin = block_x * kJxlBlockDimension;
        const size_t y_begin = block_y * kJxlBlockDimension;
        const size_t x_end = std::min(
          distance_map.extent.width,
          x_begin + info->pixel_extent().width);
        const size_t y_end = std::min(
          distance_map.extent.height,
          y_begin + info->pixel_extent().height);

        float distance_norm = 0.0f;
        size_t pixel_count = 0;
        for (size_t y = y_begin; y < y_end; ++y) {
          for (size_t x = x_begin; x < x_end; ++x) {
            float value = distance_map.Row(y)[x];
            value *= value;
            value *= value;
            value *= value;
            value *= value;
            distance_norm += value;
            ++pixel_count;
          }
        }
        if (pixel_count == 0) {
          return Status::Internal(
            "Butteraugli strategy covers no distance-map pixels");
        }
        constexpr float kTileNorm = 1.2f;
        const float block_distance = kTileNorm * std::pow(
          distance_norm / static_cast<float>(pixel_count),
          1.0f / 16.0f);
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            result[(block_y + dy) * strategies.extent().width +
                   block_x + dx] = block_distance;
          }
        }
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }

    for (size_t y = 0; y < block_distance_map.extent.height; ++y) {
      std::copy_n(
        result.data() + y * block_distance_map.extent.width,
        block_distance_map.extent.width,
        block_distance_map.Row(y));
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate Butteraugli block map");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Butteraugli block map dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace gjxl
