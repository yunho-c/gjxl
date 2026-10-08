// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/search.h"
namespace gjxl::modular_internal {
using namespace resource_budget_internal;
Status ComputeModularSearchStoragePlan(Extent2D extent, PackedModularFormat format,
                                       EntropyCodingMode mode, ModularWorkflowStoragePlan *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular search plan");
  ModularCodingPolicy largest;
  largest.tree.size = 3;
  largest.tree.nodes[0] = {.property = 15, .left = 1, .right = 2};
  largest.tree.nodes[1].predictor = largest.tree.nodes[2].predictor = Predictor::kWeighted;
  ModularWorkflowStoragePlan plan, baseline;
  if (auto s = ComputeModularWorkflowStoragePlan(extent, format, mode, largest, &plan); !s.ok())
    return s;
  if (auto s = ComputeModularWorkflowStoragePlan(extent, format, mode, &baseline); !s.ok())
    return s;
  // Training keeps prepared planes, layout, and sample records, with one WP
  // state at a time. Preparation's token allocation is conservatively retained
  // in this bound, although actual training does not allocate tokens.
  if (!plan.preparation.AddVector<TrainingSample>(std::min(plan.tokens, kMaximumTrainingSamples),
                                                  VectorCapacityPolicy::kFreshExact))
    return Status::InvalidArgument("Modular search storage overflow");
  plan.working.peak_bytes =
      std::max({plan.working.peak_bytes, plan.preparation.peak_bytes, baseline.working.peak_bytes});
  // A winning complete codestream coexists with each following training/encode.
  if (plan.output.peak_bytes > SIZE_MAX - plan.working.peak_bytes)
    return Status::InvalidArgument("Modular search candidate overlap overflow");
  plan.working.peak_bytes += plan.output.peak_bytes;
  *out = plan;
  return Status::Ok();
}
Status LearnModularPolicies(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                            uint8_t rct, ModularCodingPolicy *single, ModularCodingPolicy *split) {
  if (!single || !split || rct >= 42 || plan.geometry.source() != frame.metadata().extent)
    return Status::InvalidArgument("Null Modular learned policy");
  size_t area;
  if (!frame.metadata().extent.try_area(&area) || !area || !frame.image().channel_count() ||
      area > SIZE_MAX / frame.image().channel_count())
    return Status::InvalidArgument("Invalid Modular training frame");
  const size_t total = area * frame.image().channel_count();
  const size_t stride = total / kMaximumTrainingSamples + (total % kMaximumTrainingSamples != 0);
  ManagedVector<TrainingSample> samples;
  samples.resize(std::min(total, kMaximumTrainingSamples));
  size_t ordinal = 0, used = 0;
  for (const auto &stream : plan.streams) {
    if (stream.slice_begin > plan.slices.size() ||
        stream.slice_count > plan.slices.size() - stream.slice_begin)
      return Status::InvalidArgument("Invalid Modular training slices");
    for (const auto &slice :
         std::span(plan.slices).subspan(stream.slice_begin, stream.slice_count)) {
      if (slice.channel >= frame.image().channel_count())
        return Status::InvalidArgument("Invalid Modular training channel");
      ModularChannelView view;
      if (auto s = BorrowChannelSlice(frame.image().view(slice.channel), slice.rect, &view);
          !s.ok())
        return s;
      const auto [w, h] = view.descriptor.extent;
      WeightedPredictor<ResourceClass::kPreparation> wp(w);
      for (size_t y = 0; y < h; ++y) {
        int64_t previous_gradient = 0;
        for (size_t x = 0; x < w; ++x, ++ordinal) {
          const auto n = Neighbors(view, x, y);
          const auto weighted = wp.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
          const auto properties =
              Properties(n, slice.channel, stream.id, x, y, previous_gradient, weighted.second);
          previous_gradient = properties[9];
          if (ordinal % stride == 0) {
            if (used == samples.size())
              return Status::InvalidArgument("Overlapping Modular training samples");
            auto &sample = samples[used++];
            sample.properties = properties;
            for (unsigned p = 0; p < 14; ++p) {
              const int64_t residual =
                  int64_t{view.Row(y)[x]} - Predict(static_cast<Predictor>(p), n, weighted.first);
              if (residual < INT32_MIN || residual > INT32_MAX)
                return Status::Unsupported("Modular training residual exceeds supported range");
              sample.costs[p] = static_cast<uint8_t>(
                  1 + std::bit_width(PackSigned(static_cast<int32_t>(residual))));
            }
          }
          if (!wp.Update(view.Row(y)[x], x, y))
            return Status::Unsupported("Modular training weighted state exceeds supported range");
        }
      }
    }
  }
  if (ordinal != total || !used)
    return Status::InvalidArgument("Incomplete Modular training coverage");
  ModularCodingPolicy leaf, branch;
  leaf.rct = branch.rct = rct;
  if (auto status = LearnDecisionTrees(std::span(samples).first(used), &leaf.tree, &branch.tree);
      !status.ok())
    return status;
  *single = leaf;
  *split = branch;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
