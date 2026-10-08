// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/frame.h"
#include "codestream/entropy.h"
#include "codestream/modular/stream_plan.h"

namespace gjxl::modular_internal {
inline constexpr std::array<EntropyToken, 5> kGradientTreeTokens{
    {{1, 0}, {2, 5}, {3, 0}, {4, 0}, {5, 0}}};
struct PreparedModularTokens {
  std::array<EntropyToken, 5 * kMaximumTreeNodes> tree_tokens{};
  size_t tree_token_count = 0, context_count = 1;
  ModularCodingPolicy policy;
  codestream_internal::Storage<EntropyToken> tokens;
  codestream_internal::Storage<EntropyTokenStreamView> streams;
};
// Plan must be the canonical BuildModularStreamPlan result for this frame.
// Views refer to the moved token backing. Output replacement is atomic.
[[nodiscard]] Status TokenizeIdentity(const ModularEncoderFrame &frame,
                                      const ModularStreamPlan &plan, PreparedModularTokens *out);
[[nodiscard]] Status TokenizeModular(const ModularEncoderFrame &frame,
                                     const ModularStreamPlan &plan,
                                     const ModularCodingPolicy &policy, PreparedModularTokens *out,
                                     size_t participants = 1);
// Compatibility spelling for the original private RGB8 workflow.
[[nodiscard]] inline Status TokenizeRgb8(const ModularEncoderFrame &frame,
                                         const ModularStreamPlan &plan,
                                         PreparedModularTokens *out) {
  return TokenizeIdentity(frame, plan, out);
}
} // namespace gjxl::modular_internal
