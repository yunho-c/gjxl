// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/butteraugli.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/block_grid.h"

#if GJXL_ENABLE_LIBJXL_REFERENCE
#include <jxl/memory_manager.h>

#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/image.h"
#include "lib/jxl/memory_manager_internal.h"
#endif

namespace gjxl {
#if GJXL_ENABLE_LIBJXL_REFERENCE
namespace {

Status CopyToLibjxl(
  ConstImage3FView source,
  JxlMemoryManager* memory_manager,
  jxl::Image3F* out) {

  auto image_or = jxl::Image3F::Create(
    memory_manager,
    source.width(),
    source.height());
  if (!image_or.ok()) {
    return Status::OutOfMemory(
      "Unable to allocate Butteraugli input image");
  }
  jxl::Image3F image = std::move(image_or).value_();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < source.height(); ++y) {
      const float* source_row = source.plane[channel].Row(y);
      float* destination_row = image.PlaneRow(channel, y);
      for (size_t x = 0; x < source.width(); ++x) {
        if (!std::isfinite(source_row[x])) {
          return Status::InvalidArgument(
            "Butteraugli input pixels must be finite");
        }
        destination_row[x] = source_row[x];
      }
    }
  }
  *out = std::move(image);
  return Status::Ok();
}

}  // namespace
#endif

Status ComputeButteraugliDistance(
  ConstImage3FView reference_linear_rgb,
  ConstImage3FView distorted_linear_rgb,
  ButteraugliOptions options,
  PlaneF32View distance_map,
  double* score) {

#if GJXL_ENABLE_LIBJXL_REFERENCE
  if (!reference_linear_rgb.valid() ||
      !distorted_linear_rgb.valid() ||
      reference_linear_rgb.extent() != distorted_linear_rgb.extent() ||
      !distance_map.valid() ||
      distance_map.extent != reference_linear_rgb.extent() ||
      score == nullptr ||
      !std::isfinite(options.hf_asymmetry) ||
      options.hf_asymmetry <= 0.0f ||
      !std::isfinite(options.x_multiplier) ||
      options.x_multiplier <= 0.0f ||
      !std::isfinite(options.intensity_target) ||
      options.intensity_target <= 0.0f) {
    return Status::InvalidArgument(
      "Butteraugli images, output, or options are invalid");
  }

  JxlMemoryManager memory_manager{};
  if (!jxl::MemoryManagerInit(&memory_manager, nullptr)) {
    return Status::Internal(
      "Unable to initialize Butteraugli memory manager");
  }

  jxl::Image3F reference;
  Status status = CopyToLibjxl(
    reference_linear_rgb,
    &memory_manager,
    &reference);
  if (!status.ok()) {
    return status;
  }
  jxl::Image3F distorted;
  status = CopyToLibjxl(
    distorted_linear_rgb,
    &memory_manager,
    &distorted);
  if (!status.ok()) {
    return status;
  }

  auto result_or = jxl::ImageF::Create(
    &memory_manager,
    reference_linear_rgb.width(),
    reference_linear_rgb.height());
  if (!result_or.ok()) {
    return Status::OutOfMemory(
      "Unable to allocate Butteraugli distance map");
  }
  jxl::ImageF result = std::move(result_or).value_();
  const jxl::ButteraugliParams params{
    .hf_asymmetry = options.hf_asymmetry,
    .xmul = options.x_multiplier,
    .intensity_target = options.intensity_target,
  };
  if (!jxl::ButteraugliDiffmap(reference, distorted, params, result)) {
    return Status::Internal(
      "Pinned libjxl Butteraugli computation failed");
  }

  const double result_score =
    jxl::ButteraugliScoreFromDiffmap(result, &params);
  if (!std::isfinite(result_score)) {
    return Status::Internal(
      "Butteraugli produced a non-finite score");
  }
  for (size_t y = 0; y < distance_map.extent.height; ++y) {
    const float* source = result.ConstRow(y);
    for (size_t x = 0; x < distance_map.extent.width; ++x) {
      if (!std::isfinite(source[x]) || source[x] < 0.0f) {
        return Status::Internal(
          "Butteraugli produced an invalid distance map");
      }
    }
  }

  for (size_t y = 0; y < distance_map.extent.height; ++y) {
    std::copy_n(
      result.ConstRow(y),
      distance_map.extent.width,
      distance_map.Row(y));
  }
  *score = result_score;
  return Status::Ok();
#else
  static_cast<void>(reference_linear_rgb);
  static_cast<void>(distorted_linear_rgb);
  static_cast<void>(options);
  static_cast<void>(distance_map);
  static_cast<void>(score);
  return Status::Unavailable(
    "Butteraugli requires GJXL_ENABLE_LIBJXL_REFERENCE");
#endif
}

Status ReduceButteraugliDistanceMap(
  ConstPlaneF32View distance_map,
  const AcStrategyGrid& strategies,
  PlaneF32View block_distance_map) {

  Extent2D padded_pixel_extent;
  if (!distance_map.valid() ||
      !strategies.complete() ||
      !block_distance_map.valid() ||
      block_distance_map.extent != strategies.extent() ||
      !BlockGrid{strategies.extent()}.try_padded_pixel_extent(
        &padded_pixel_extent) ||
      distance_map.extent.width <=
        padded_pixel_extent.width - kJxlBlockDimension ||
      distance_map.extent.height <=
        padded_pixel_extent.height - kJxlBlockDimension ||
      distance_map.extent.width > padded_pixel_extent.width ||
      distance_map.extent.height > padded_pixel_extent.height) {
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
