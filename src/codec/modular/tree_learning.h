// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/tree.h"

namespace gjxl::modular_internal {
inline constexpr size_t kMaximumTrainingSamples = 4096;
struct TrainingSample {
  PredictionProperties properties;
  std::array<uint8_t, 14> costs;
};
// Integer residual costs are supplied by the sampling adapter, without entropy
// or stream-plan dependencies in the learner. Outputs are replaced on success.
[[nodiscard]] Status LearnDecisionTrees(std::span<const TrainingSample> samples,
                                        DecisionTree *single, DecisionTree *split);
} // namespace gjxl::modular_internal
