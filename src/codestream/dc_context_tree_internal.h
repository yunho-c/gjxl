// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "codestream/dc_prediction.h"
#include "codestream/entropy.h"
#include "codestream/headers.h"
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl::codestream_internal {

enum class DcTreePolicy : uint8_t { kLegacy, kAdaptive };
// Qualified combined default; see docs/dc-small-trees/README.md for tradeoffs.
inline constexpr DcTreePolicy kDefaultDcTreePolicy = DcTreePolicy::kAdaptive;

struct DcContextTreeLayout {
  static constexpr size_t kMaximumTokens = 313;
  static constexpr size_t kMaximumContexts = 45;
  std::array<EntropyToken, kMaximumTokens> tokens{};
  std::array<uint8_t, kMaximumContexts> context_map{};
  size_t token_count = 0;
  uint32_t context_count = 0;
  uint32_t dc_leaf_count = 0;
  VarDctDcPrediction prediction = VarDctDcPrediction::kGradient;

  [[nodiscard]] bool full_tree() const { return dc_leaf_count == 34; }
  [[nodiscard]] std::span<const EntropyToken> tree_tokens() const {
    return {tokens.data(), token_count};
  }
};

// The count covers all three DC channels over the whole frame, not one group.
[[nodiscard]] Status ComputeDcSampleCount(Extent2D blocks, size_t *samples);
[[nodiscard]] Status
SelectDcContextTreeLayout(size_t samples, VarDctDcPrediction prediction,
                          DcTreePolicy policy,
                          const DcContextTreeLayout **layout);
[[nodiscard]] const DcContextTreeLayout &
LegacyDcContextTree(VarDctDcPrediction prediction);

// Layout-aware writer used by both exact size estimation and emission.
[[nodiscard]] Status
WriteDcGlobalWithLayout(QuantizerParams params, size_t dc_group_count,
                        const SimpleBlockContextMap &block_context_map,
                        const EntropyCode &dc_code, BitWriter *writer,
                        const DcContextTreeLayout &layout);

// Internal qualification hook only. No CLI, public option, or environment
// switch. The policy is captured by the serializer before starting workers.
class ScopedDcTreePolicyForTesting {
public:
  explicit ScopedDcTreePolicyForTesting(DcTreePolicy policy);
  ~ScopedDcTreePolicyForTesting();
  ScopedDcTreePolicyForTesting(const ScopedDcTreePolicyForTesting &) = delete;
  ScopedDcTreePolicyForTesting &
  operator=(const ScopedDcTreePolicyForTesting &) = delete;

private:
  DcTreePolicy previous_;
};
[[nodiscard]] DcTreePolicy CurrentDcTreePolicy();

} // namespace gjxl::codestream_internal
