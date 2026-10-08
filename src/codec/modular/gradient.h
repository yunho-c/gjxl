// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Boundary conventions follow pinned libjxl context_predict.h.
#pragma once
#include "codec/modular/image.h"
#include <algorithm>

namespace gjxl::modular_internal {
// Coordinates are local to a validated stream slice. Wide arithmetic avoids
// overflow even when this primitive is used with full int32 sample values.
[[nodiscard]] inline int64_t GradientPrediction(ModularChannelView channel, size_t x, size_t y) {
  const int64_t left = x ? channel.Row(y)[x - 1] : (y ? channel.Row(y - 1)[x] : 0);
  const int64_t top = y ? channel.Row(y - 1)[x] : left;
  const int64_t top_left = x && y ? channel.Row(y - 1)[x - 1] : left;
  return std::clamp(left + top - top_left, std::min(left, top), std::max(left, top));
}
} // namespace gjxl::modular_internal
