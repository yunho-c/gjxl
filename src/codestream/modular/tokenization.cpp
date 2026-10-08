// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/tokenization.h"
#include "codec/modular/gradient.h"

namespace gjxl::modular_internal {
Status TokenizeIdentity(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                        PreparedModularTokens *out) try {
  const auto extent = frame.metadata().extent;
  const size_t channels = frame.image().channel_count();
  size_t area;
  if (!out || (channels != 1 && channels != 3 && channels != 4) || plan.geometry.source() != extent ||
      !extent.try_area(&area) || area > SIZE_MAX / channels)
    return Status::InvalidArgument("Invalid Modular identity tokenization input");
  PreparedModularTokens result;
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
      for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
          // Identity samples and clamped predictions are [0, 2^bits-1]. For
          // 16-bit input, residuals are [-65535,65535], packed values <=131070.
          // Wide prediction arithmetic avoids narrowing intermediate sums.
          const int64_t residual = int64_t{view.Row(y)[x]} - GradientPrediction(view, x, y);
          result.tokens[next++] = {0, PackSigned(static_cast<int32_t>(residual))};
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
