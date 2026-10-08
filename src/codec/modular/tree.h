// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Scalar arithmetic adapted from pinned libjxl.
#pragma once
#include "codec/modular/scalar_prediction.h"

namespace gjxl::modular_internal {
inline constexpr size_t kMaximumTreeNodes = 31;
inline constexpr size_t kMaximumTreeLeaves = 16;
inline constexpr size_t kPropertyCount = 16; // No previous-channel properties yet.
struct DecisionNode {
  int32_t property = -1, split = 0;
  uint8_t left = 0, right = 0; // > split and <= split, respectively.
  Predictor predictor = Predictor::kGradient;
  int32_t offset = 0;
  uint32_t multiplier = 1;
  bool operator==(const DecisionNode &) const = default;
};
struct DecisionTree {
  std::array<DecisionNode, kMaximumTreeNodes> nodes{};
  size_t size = 1;
  bool operator==(const DecisionTree &) const = default;
};
struct TreeLayout {
  std::array<uint8_t, kMaximumTreeNodes> breadth_first{}, context{};
  size_t leaves = 0;
  bool weighted = false;
};
[[nodiscard]] inline Status ValidateTree(const DecisionTree &tree, TreeLayout *out) {
  if (!out || tree.size == 0 || tree.size > kMaximumTreeNodes)
    return Status::InvalidArgument("Invalid Modular tree size");
  TreeLayout result;
  std::array<bool, kMaximumTreeNodes> seen{};
  seen[0] = true;
  using Ranges = std::array<std::pair<int32_t, int32_t>, kPropertyCount>;
  std::array<Ranges, kMaximumTreeNodes> ranges{};
  for (auto &range : ranges[0])
    range = {INT32_MIN, INT32_MAX};
  size_t queued = 1;
  for (size_t i = 0; i < queued; ++i) {
    const size_t id = result.breadth_first[i];
    const auto &n = tree.nodes[id];
    if (n.property == -1) {
      if (static_cast<unsigned>(n.predictor) >= 14 || !n.multiplier || n.multiplier > INT32_MAX)
        return Status::InvalidArgument("Invalid Modular tree leaf");
      result.context[id] = static_cast<uint8_t>(result.leaves++);
      result.weighted |= n.predictor == Predictor::kWeighted;
    } else {
      if (n.property < 0 || n.property >= kPropertyCount || queued + 2 > tree.size)
        return Status::InvalidArgument("Unsupported Modular tree property or shape");
      const auto [lower, upper] = ranges[id][n.property];
      if (n.split < lower || n.split >= upper)
        return Status::InvalidArgument("Contradictory Modular tree split");
      result.weighted |= n.property == 15;
      for (auto child : {n.left, n.right}) {
        if (child >= tree.size || seen[child])
          return Status::InvalidArgument("Cyclic or shared Modular tree child");
        seen[child] = true;
        ranges[child] = ranges[id];
        if (child == n.left)
          ranges[child][n.property].first = n.split + 1;
        else
          ranges[child][n.property].second = n.split;
        result.breadth_first[queued++] = child;
      }
    }
  }
  if (queued != tree.size || result.leaves > kMaximumTreeLeaves)
    return Status::InvalidArgument("Unreachable Modular tree node");
  *out = result;
  return Status::Ok();
}
using PredictionProperties = std::array<int64_t, kPropertyCount>;
[[nodiscard]] inline PredictionProperties Properties(PredictionNeighborhood n, size_t channel,
                                                     size_t stream, size_t x, size_t y,
                                                     int64_t previous_gradient,
                                                     int64_t weighted_property) {
  return {static_cast<int64_t>(channel),
          static_cast<int64_t>(stream),
          static_cast<int64_t>(y),
          static_cast<int64_t>(x),
          std::abs(n.top),
          std::abs(n.left),
          n.top,
          n.left,
          n.left - previous_gradient,
          n.left + n.top - n.top_left,
          n.left - n.top_left,
          n.top_left - n.top,
          n.top - n.top_right,
          n.top - n.top_top,
          n.left - n.left_left,
          weighted_property};
}
[[nodiscard]] inline size_t Lookup(const DecisionTree &tree, const PredictionProperties &p) {
  size_t id = 0;
  while (tree.nodes[id].property != -1) {
    const auto &n = tree.nodes[id];
    id = p[n.property] > n.split ? n.left : n.right;
  }
  return id;
}
} // namespace gjxl::modular_internal
