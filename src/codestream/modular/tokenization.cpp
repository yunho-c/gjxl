// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/tokenization.h"
#include "codec/modular/gradient.h"

namespace gjxl::modular_internal {
Status TokenizeIdentity(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                        PreparedModularTokens *out) {
  return TokenizeModular(frame, plan, {}, out);
}
Status TokenizeModular(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                       const ModularCodingPolicy &policy, PreparedModularTokens *out) try {
  const auto extent = frame.metadata().extent;
  const size_t channels = frame.image().channel_count();
  size_t area;
  if (!out || (channels != 1 && channels != 3 && channels != 4) ||
      plan.geometry.source() != extent || !extent.try_area(&area) || area > SIZE_MAX / channels)
    return Status::InvalidArgument("Invalid Modular identity tokenization input");
  if (auto s = ValidateCodingPolicy(policy); !s.ok())
    return s;
  TreeLayout tree_layout;
  if (auto s = ValidateTree(policy.tree, &tree_layout); !s.ok())
    return s;
  PreparedModularTokens result;
  result.policy = policy;
  result.context_count = tree_layout.leaves;
  auto tree_token = [&](uint32_t context, uint32_t value) {
    result.tree_tokens[result.tree_token_count++] = {context, value};
  };
  for (size_t i = 0; i < policy.tree.size; ++i) {
    const auto &node = policy.tree.nodes[tree_layout.breadth_first[i]];
    tree_token(1, node.property + 1);
    if (node.property == -1) {
      tree_token(2, static_cast<uint32_t>(node.predictor));
      tree_token(3, PackSigned(node.offset));
      const unsigned shift = std::countr_zero(node.multiplier);
      tree_token(4, shift);
      tree_token(5, (node.multiplier >> shift) - 1);
    } else
      tree_token(0, PackSigned(node.split));
  }
  result.tokens.resize(channels * area);
  result.streams.resize(plan.streams.size());
  size_t next = 0;
  for (size_t i = 0; i < plan.streams.size(); ++i) {
    const auto &stream = plan.streams[i];
    if (stream.slice_begin > plan.slices.size() ||
        stream.slice_count > plan.slices.size() - stream.slice_begin)
      return Status::InvalidArgument("Invalid Modular stream slices");
    const size_t begin = next;
    for (const auto &slice :
         std::span(plan.slices).subspan(stream.slice_begin, stream.slice_count)) {
      if (slice.channel >= channels)
        return Status::InvalidArgument("Invalid Modular identity channel index");
      ModularChannelView view;
      if (auto s = BorrowChannelSlice(frame.image().view(slice.channel), slice.rect, &view);
          !s.ok())
        return s;
      const auto [w, h] = view.descriptor.extent;
      if (w * h > result.tokens.size() - next)
        return Status::InvalidArgument("Overlapping Modular token count");
      std::optional<WeightedPredictor<resource_budget_internal::ResourceClass::kPreparation>> wp;
      if (tree_layout.weighted)
        wp.emplace(w, policy.weighted);
      for (size_t y = 0; y < h; ++y) {
        int64_t previous_gradient = 0;
        for (size_t x = 0; x < w; ++x) {
          const auto neighbors = Neighbors(view, x, y);
          std::pair<int64_t, int64_t> weighted{};
          if (wp)
            weighted = wp->Predict(x, y, neighbors.top, neighbors.left, neighbors.top_right,
                                   neighbors.top_left, neighbors.top_top);
          const auto properties = Properties(neighbors, slice.channel, stream.id, x, y,
                                             previous_gradient, weighted.second);
          previous_gradient = properties[9];
          const size_t leaf = Lookup(policy.tree, properties);
          const auto &node = policy.tree.nodes[leaf];
          const int64_t residual = int64_t{view.Row(y)[x]} -
                                   Predict(node.predictor, neighbors, weighted.first) - node.offset;
          if (residual % node.multiplier != 0 || residual / node.multiplier < INT32_MIN ||
              residual / node.multiplier > INT32_MAX)
            return Status::InvalidArgument("Modular leaf cannot represent residual exactly");
          result.tokens[next++] = {tree_layout.context[leaf],
                                   PackSigned(static_cast<int32_t>(residual / node.multiplier))};
          if (wp && !wp->Update(view.Row(y)[x], x, y))
            return Status::Unsupported("Modular weighted state exceeds supported range");
        }
      }
    }
    result.streams[i] =
        EntropyTokenStreamView::Interleaved(std::span(result.tokens).subspan(begin, next - begin));
  }
  if (next != result.tokens.size())
    return Status::InvalidArgument("Incomplete Modular token coverage");
  *out = std::move(result);
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular token allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular token storage overflow");
}
} // namespace gjxl::modular_internal
