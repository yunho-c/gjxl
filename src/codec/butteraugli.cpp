// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/butteraugli.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <vector>

#include "core/managed_allocator.h"
#include "codec/butteraugli_distance_internal.h"

namespace gjxl {
using resource_budget_internal::ManagedVector;


struct PreparedButteraugliReference::Impl {
  butteraugli_internal::NativePreparedButteraugliReference native;
};

PreparedButteraugliReference::PreparedButteraugliReference() = default;
PreparedButteraugliReference::~PreparedButteraugliReference() = default;
PreparedButteraugliReference::PreparedButteraugliReference(
  PreparedButteraugliReference&&) noexcept = default;
PreparedButteraugliReference& PreparedButteraugliReference::operator=(
  PreparedButteraugliReference&&) noexcept = default;

Status PreparedButteraugliReference::Prepare(
  ConstImage3FView reference_linear_rgb,
  ButteraugliOptions options) {

  const butteraugli_internal::NativeButteraugliParams params{
    .hf_asymmetry = options.hf_asymmetry,
    .x_multiplier = options.x_multiplier,
    .intensity_target = options.intensity_target,
  };
  try {
    auto candidate = std::make_unique<Impl>();
    Status status = butteraugli_internal::PrepareButteraugliReferenceNative(
      reference_linear_rgb, params, &candidate->native);
    if (!status.ok()) {
      return status;
    }
    impl_ = std::move(candidate);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate prepared Butteraugli reference");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Prepared Butteraugli reference dimensions are too large");
  }
  return Status::Ok();
}

Status PreparedButteraugliReference::Compare(
  ConstImage3FView distorted_linear_rgb,
  PlaneF32View distance_map,
  double* score) {

  if (impl_ == nullptr) {
    return Status::FailedPrecondition(
      "Butteraugli reference has not been prepared");
  }
  return butteraugli_internal::CompareButteraugliReferenceNative(
    &impl_->native, distorted_linear_rgb, distance_map, score);
}

Extent2D PreparedButteraugliReference::extent() const noexcept {
  return impl_ == nullptr ? Extent2D{} : impl_->native.extent();
}

ButteraugliOptions PreparedButteraugliReference::options() const noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  const butteraugli_internal::NativeButteraugliParams params =
    impl_->native.params();
  return {
    .hf_asymmetry = params.hf_asymmetry,
    .x_multiplier = params.x_multiplier,
    .intensity_target = params.intensity_target,
  };
}

bool PreparedButteraugliReference::ready() const noexcept {
  return impl_ != nullptr && impl_->native.ready();
}

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
    ManagedVector<float> result(block_count);
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
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
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
