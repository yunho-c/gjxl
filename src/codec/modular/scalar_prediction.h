// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Scalar arithmetic adapted from pinned libjxl.
#pragma once
#include "codec/modular/image.h"
#include "codec/modular/prediction.h"

namespace gjxl::modular_internal {
struct PredictionNeighborhood {
  int64_t left, top, top_left, top_right, left_left, top_top, top_right_right;
};
[[nodiscard]] inline PredictionNeighborhood Neighbors(ModularChannelView v, size_t x, size_t y) {
  const int64_t left = x ? v.Row(y)[x - 1] : y ? v.Row(y - 1)[x] : 0;
  const int64_t top = y ? v.Row(y - 1)[x] : left;
  const int64_t right = y && x + 1 < v.descriptor.extent.width ? v.Row(y - 1)[x + 1] : top;
  return {left,
          top,
          x && y ? v.Row(y - 1)[x - 1] : left,
          right,
          x > 1 ? v.Row(y)[x - 2] : left,
          y > 1 ? v.Row(y - 2)[x] : top,
          y && x + 2 < v.descriptor.extent.width ? v.Row(y - 1)[x + 2] : right};
}
[[nodiscard]] inline int64_t Predict(Predictor p, PredictionNeighborhood n, int64_t weighted) {
  switch (p) {
  case Predictor::kZero:
    return 0;
  case Predictor::kLeft:
    return n.left;
  case Predictor::kTop:
    return n.top;
  case Predictor::kAverage0:
    return (n.left + n.top) / 2;
  case Predictor::kSelect:
    return std::abs(n.top - n.top_left) < std::abs(n.left - n.top_left) ? n.left : n.top;
  case Predictor::kGradient:
    return std::clamp(n.left + n.top - n.top_left, std::min(n.left, n.top),
                      std::max(n.left, n.top));
  case Predictor::kWeighted:
    return weighted;
  case Predictor::kTopRight:
    return n.top_right;
  case Predictor::kTopLeft:
    return n.top_left;
  case Predictor::kLeftLeft:
    return n.left_left;
  case Predictor::kAverage1:
    return (n.left + n.top_left) / 2;
  case Predictor::kAverage2:
    return (n.top_left + n.top) / 2;
  case Predictor::kAverage3:
    return (n.top + n.top_right) / 2;
  case Predictor::kAverage4:
    return (6 * n.top - 2 * n.top_top + 7 * n.left + n.left_left + n.top_right_right +
            3 * n.top_right + 8) /
           16;
  default:
    return 0; // Validated policy excludes this case.
  }
}
} // namespace gjxl::modular_internal
