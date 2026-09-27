// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <cstddef>
#include <cstdint>
namespace gjxl::metal_internal {
inline constexpr size_t kHighXYThreadCount = 16 * 16;
inline constexpr size_t kHighXYScratchBytes =
    2 * 16 * (64 + 14) * sizeof(float);
struct HighXYParams {
  uint32_t width, height, input_x_stride, input_y_stride;
  uint32_t medium_x_stride, medium_y_stride, high_x_stride, high_y_stride;
};
static_assert(sizeof(HighXYParams) == 32);
// Subtraction avoids overflow when rejecting an invalid static-memory value.
[[nodiscard]] constexpr bool HighXYDispatchFits(size_t maximum_threads,
                                                size_t static_memory,
                                                size_t maximum_memory) {
  return maximum_threads >= kHighXYThreadCount &&
         static_memory <= maximum_memory &&
         kHighXYScratchBytes <= maximum_memory - static_memory;
}
} // namespace gjxl::metal_internal
