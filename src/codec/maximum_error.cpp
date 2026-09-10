// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/maximum_error.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "core/managed_allocator.h"
#include "core/block_grid.h"
#include "core/image_ops.h"
#include "core/quantizer.h"

namespace gjxl {
using resource_budget_internal::ManagedVector;

namespace {

constexpr float kMinimumRepresentableQuantField =
  1.0f / static_cast<float>(kQuantGlobalScaleDenominator);
constexpr float kMaximumRepresentableQuantField =
  static_cast<float>(kMaxRawQuant) *
  static_cast<float>(kMaxEncoderGlobalScale) /
  static_cast<float>(kQuantGlobalScaleDenominator);

bool ValidLimits(const std::array<float, 3>& limits) {
  return std::ranges::all_of(limits, [](float limit) {
    return std::isfinite(limit) && limit > 0.0f;
  });
}

}  // namespace

Status ReduceMaximumError(
  ConstImage3FView reference_opsin,
  ConstImage3FView reconstructed_opsin,
  Extent2D source_extent,
  const AcStrategyGrid& strategies,
  const std::array<float, 3>& limits,
  PlaneF32View block_error,
  MaximumErrorReduction* reduction) {

  Extent2D padded_extent;
  if (!reference_opsin.valid() || !reconstructed_opsin.valid() ||
      !strategies.complete() || !block_error.valid() ||
      reduction == nullptr || !ValidLimits(limits) ||
      !BlockGrid{strategies.extent()}.try_padded_pixel_extent(
        &padded_extent) ||
      reference_opsin.extent() != padded_extent ||
      source_extent.width <= padded_extent.width - kJxlBlockDimension ||
      source_extent.height <= padded_extent.height - kJxlBlockDimension ||
      reconstructed_opsin.width() < source_extent.width ||
      reconstructed_opsin.height() < source_extent.height ||
      reference_opsin.width() < source_extent.width ||
      reference_opsin.height() < source_extent.height ||
      source_extent.width == 0 || source_extent.height == 0 ||
      block_error.extent != strategies.extent()) {
    return Status::InvalidArgument(
      "Maximum-error reduction inputs are invalid");
  }

  size_t block_count = 0;
  if (!strategies.extent().try_area(&block_count)) {
    return Status::InvalidArgument(
      "Maximum-error block grid is too large");
  }

  try {
    ManagedVector<float> candidate(block_count, 0.0f);
    MaximumErrorReduction candidate_reduction;
    const Status status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr) {
          return Status::Internal(
            "Maximum-error reduction found an unknown strategy");
        }
        const size_t pixel_begin_x = block_x * kJxlBlockDimension;
        const size_t pixel_begin_y = block_y * kJxlBlockDimension;
        const size_t pixel_end_x = std::min(
          source_extent.width,
          pixel_begin_x +
            info->covered_blocks.width * kJxlBlockDimension);
        const size_t pixel_end_y = std::min(
          source_extent.height,
          pixel_begin_y +
            info->covered_blocks.height * kJxlBlockDimension);
        float transform_maximum = 0.0f;
        for (size_t channel = 0; channel < 3; ++channel) {
          for (size_t y = pixel_begin_y; y < pixel_end_y; ++y) {
            const float* reference = reference_opsin.plane[channel].Row(y);
            const float* reconstructed =
              reconstructed_opsin.plane[channel].Row(y);
            for (size_t x = pixel_begin_x; x < pixel_end_x; ++x) {
              if (!std::isfinite(reference[x]) ||
                  !std::isfinite(reconstructed[x])) {
                return Status::InvalidArgument(
                  "Maximum-error samples must be finite");
              }
              const float error = std::abs(reference[x] - reconstructed[x]);
              candidate_reduction.channel_maximum[channel] = std::max(
                candidate_reduction.channel_maximum[channel], error);
              transform_maximum = std::max(
                transform_maximum, error / limits[channel]);
            }
          }
        }
        candidate_reduction.normalized_maximum = std::max(
          candidate_reduction.normalized_maximum, transform_maximum);
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            candidate[(block_y + dy) * strategies.extent().width +
                      block_x + dx] = transform_maximum;
          }
        }
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }
    CopyContiguousPlane(candidate, block_error);
    *reduction = candidate_reduction;
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate maximum-error reduction storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Maximum-error reduction dimensions are too large");
  }
}

float MaximumErrorQuantFieldMultiplier(float normalized_error) noexcept {
  if (!std::isfinite(normalized_error) || normalized_error < 0.0f) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (normalized_error < 0.5f) {
    return normalized_error * 2.0f;
  }
  if (normalized_error > 1.0f) {
    return normalized_error;
  }
  return 1.0f;
}

Status UpdateMaximumErrorQuantField(
  const AcStrategyGrid& strategies,
  ConstPlaneF32View block_error,
  ConstPlaneF32View input,
  PlaneF32View output,
  bool* upper_bound_limited) {

  if (!strategies.complete() || !block_error.valid() || !input.valid() ||
      !output.valid() || block_error.extent != strategies.extent() ||
      input.extent != strategies.extent() || output.extent != input.extent) {
    return Status::InvalidArgument(
      "Maximum-error quant-field update inputs are invalid");
  }
  size_t block_count = 0;
  if (!strategies.extent().try_area(&block_count)) {
    return Status::InvalidArgument(
      "Maximum-error quant-field dimensions are too large");
  }

  try {
    ManagedVector<float> candidate(block_count);
    for (size_t y = 0; y < input.extent.height; ++y) {
      for (size_t x = 0; x < input.extent.width; ++x) {
        const float value = input.Row(y)[x];
        if (!std::isfinite(value) || value <= 0.0f) {
          return Status::InvalidArgument(
            "Maximum-error quant field must be finite and positive");
        }
        candidate[y * input.extent.width + x] = value;
      }
    }

    bool limited = false;
    const Status status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr) {
          return Status::Internal(
            "Maximum-error update found an unknown strategy");
        }
        const float normalized = block_error.Row(block_y)[block_x];
        const float multiplier =
          MaximumErrorQuantFieldMultiplier(normalized);
        if (!std::isfinite(multiplier)) {
          return Status::InvalidArgument(
            "Maximum-error block map must be finite and non-negative");
        }
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            const size_t index =
              (block_y + dy) * input.extent.width + block_x + dx;
            const float requested = candidate[index] * multiplier;
            if (!std::isfinite(requested)) {
              return Status::InvalidArgument(
                "Maximum-error quant-field update is not finite");
            }
            if (multiplier > 1.0f &&
                requested > kMaximumRepresentableQuantField) {
              limited = true;
            }
            candidate[index] = std::clamp(
              requested,
              kMinimumRepresentableQuantField,
              kMaximumRepresentableQuantField);
          }
        }
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }
    CopyContiguousPlane(candidate, output);
    if (upper_bound_limited != nullptr) {
      *upper_bound_limited = limited;
    }
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate maximum-error quant-field storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Maximum-error quant-field dimensions are too large");
  }
}

}  // namespace gjxl
