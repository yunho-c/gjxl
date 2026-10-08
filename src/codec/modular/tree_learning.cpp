// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/tree_learning.h"
namespace gjxl::modular_internal {
Status LearnDecisionTrees(std::span<const TrainingSample> samples, DecisionTree *single,
                          DecisionTree *split) {
  if (!single || !split || samples.empty() || samples.size() > kMaximumTrainingSamples)
    return Status::InvalidArgument("Invalid Modular training samples");
  const size_t used = samples.size();
  std::array<uint64_t, 14> costs{};
  for (size_t i = 0; i < used; ++i)
    for (size_t p = 0; p < 14; ++p)
      costs[p] += samples[i].costs[p];
  auto best_predictor = [](const auto &values) {
    size_t best = 5; // Gradient wins equal scores, then ascending wire ID.
    for (size_t p = 0; p < 14; ++p)
      if (values[p] < values[best])
        best = p;
    return best;
  };
  DecisionTree leaf, branch;
  leaf.nodes[0].predictor = static_cast<Predictor>(best_predictor(costs));
  branch = leaf;
  uint64_t best_cost = costs[static_cast<size_t>(leaf.nodes[0].predictor)];
  for (size_t property = 0; property < kPropertyCount; ++property) {
    int64_t sum = 0;
    for (size_t i = 0; i < used; ++i)
      sum += samples[i].properties[property];
    for (int64_t threshold : {int64_t{0}, sum / static_cast<int64_t>(used)}) {
      if (threshold < INT32_MIN || threshold > INT32_MAX)
        continue;
      std::array<std::array<uint64_t, 14>, 2> sides{};
      std::array<size_t, 2> counts{};
      for (size_t i = 0; i < used; ++i) {
        const size_t side = samples[i].properties[property] > threshold ? 0 : 1;
        ++counts[side];
        for (size_t p = 0; p < 14; ++p)
          sides[side][p] += samples[i].costs[p];
      }
      if (!counts[0] || !counts[1])
        continue;
      const size_t left = best_predictor(sides[0]), right = best_predictor(sides[1]);
      const uint64_t cost = sides[0][left] + sides[1][right] + 32;
      if (cost >= best_cost)
        continue;
      best_cost = cost;
      branch.size = 3;
      branch.nodes[0] = {.property = static_cast<int32_t>(property),
                         .split = static_cast<int32_t>(threshold),
                         .left = 1,
                         .right = 2};
      branch.nodes[1].predictor = static_cast<Predictor>(left);
      branch.nodes[2].predictor = static_cast<Predictor>(right);
    }
  }
  *single = leaf;
  *split = branch;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
