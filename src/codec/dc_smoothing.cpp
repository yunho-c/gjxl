// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Adaptive DC smoothing adapted from pinned libjxl compressed_dc.cc.
#include "codec/dc_smoothing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "core/image_buffer.h"
#include "core/managed_allocator.h"

namespace gjxl {

Status SmoothDcCoefficients(ConstImage3FView dc, const Quantizer &quantizer,
                            Image3FView output) {
  size_t area = 0;
  if (!dc.valid() || !output.valid() || !quantizer.valid() ||
      dc.extent() != output.extent() || !dc.extent().try_area(&area) ||
      area > std::numeric_limits<size_t>::max() / (3 * sizeof(float))) {
    return Status::InvalidArgument("DC smoothing inputs are invalid");
  }
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < dc.height(); ++y)
      for (size_t x = 0; x < dc.width(); ++x)
        if (!std::isfinite(dc.plane[c].Row(y)[x]))
          return Status::InvalidArgument("DC smoothing input is not finite");
  try {
    Image3FBuffer candidate(dc.extent());
    const auto temporary = candidate.view();
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < dc.height(); ++y)
        std::copy_n(dc.plane[c].Row(y), dc.width(), temporary.plane[c].Row(y));
    constexpr float side_weight = 0.20345139757231578f;
    constexpr float corner_weight = 0.0334829185968739f;
    constexpr float center_weight = 1.0f - 4.0f * (side_weight + corner_weight);
    for (size_t y = 1; y + 1 < dc.height(); ++y) {
      for (size_t x = 1; x + 1 < dc.width(); ++x) {
        std::array<float, 3> center{}, smoothed{};
        float gap = 0.5f;
        for (size_t c = 0; c < 3; ++c) {
          const float *top = dc.plane[c].Row(y - 1);
          const float *row = dc.plane[c].Row(y);
          const float *bottom = dc.plane[c].Row(y + 1);
          const float corners =
              (top[x - 1] + top[x + 1]) + (bottom[x - 1] + bottom[x + 1]);
          const float sides = (row[x - 1] + row[x + 1]) + (top[x] + bottom[x]);
          center[c] = row[x];
          smoothed[c] =
              std::fma(corners, corner_weight,
                       std::fma(sides, side_weight, row[x] * center_weight));
          const float normalized_gap =
              std::abs((row[x] - smoothed[c]) / quantizer.dc_steps()[c]);
          if (!std::isfinite(normalized_gap))
            return Status::InvalidArgument(
                "DC smoothing arithmetic exceeds finite range");
          gap = std::max(gap, normalized_gap);
        }
        const float factor = std::max(0.0f, std::fma(-4.0f, gap, 3.0f));
        for (size_t c = 0; c < 3; ++c) {
          const float value =
              std::fma(smoothed[c] - center[c], factor, center[c]);
          if (!std::isfinite(value))
            return Status::InvalidArgument("Smoothed DC is not finite");
          temporary.plane[c].Row(y)[x] = value;
        }
      }
    }
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < dc.height(); ++y)
        std::copy_n(temporary.plane[c].Row(y), dc.width(),
                    output.plane[c].Row(y));
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure &error) {
    return error.status();
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("DC smoothing allocation failed");
  } catch (const std::length_error &) {
    return Status::OutOfMemory("DC smoothing allocation is too large");
  }
}

} // namespace gjxl
