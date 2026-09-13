// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Default weighted predictor adapted from pinned libjxl context_predict.h.
// This implementation has no libjxl runtime or header dependency.
#include "codestream/weighted_dc.h"
#include "codec/weighted_dc_predictor_internal.h"
#include "codestream/dc_group.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace gjxl::codestream_internal {
namespace {
// The existing gradient tree and WP tree use identical split thresholds and
// leaf numbering. Only the property (gradient -> WP error) changes.
constexpr std::array<std::pair<int32_t, uint32_t>, 34> kContextRuns{{
    {12, 44},  {120, 43}, {257, 40},  {321, 39},  {385, 38}, {417, 37},
    {449, 36}, {465, 35}, {481, 34},  {489, 33},  {497, 32}, {501, 31},
    {505, 30}, {508, 29}, {509, 28},  {511, 27},  {512, 26}, {513, 42},
    {515, 41}, {517, 25}, {519, 24},  {523, 23},  {527, 22}, {535, 21},
    {543, 20}, {559, 19}, {575, 18},  {607, 17},  {639, 16}, {703, 15},
    {767, 14}, {904, 13}, {1012, 12}, {1023, 11},
}};
constexpr auto kContexts = [] {
  std::array<uint8_t, 1024> result{};
  size_t begin = 0;
  for (const auto [end, context] : kContextRuns) {
    for (; begin <= static_cast<size_t>(end); ++begin)
      result[begin] = context;
  }
  return result;
}();
static_assert(kContextRuns.back().first == 1023);
uint32_t Context(int64_t property) {
  return kContexts[static_cast<size_t>(
      std::clamp<int64_t>(512 + property, 0, 1023))];
}

} // namespace

Status TokenizeWeightedDcGroup(ConstImage3I32View dc,
                               Storage<EntropyToken> *tokens) {
  size_t area = 0;
  if (tokens == nullptr || !dc.valid() || dc.extent().empty() ||
      dc.extent().width > kSimpleDcGroupBlockDimension ||
      dc.extent().height > kSimpleDcGroupBlockDimension ||
      !dc.extent().try_area(&area))
    return Status::InvalidArgument("Weighted DC group is invalid");
  try {
    Storage<EntropyToken> candidate;
    candidate.reserve(area * 3);
    dc_prediction_internal::WeightedDcPredictor<
        resource_budget_internal::ResourceClass::kSerializer>
        state(dc.extent().width);
    for (const size_t channel : {size_t{1}, size_t{0}, size_t{2}}) {
      const auto plane = dc.plane[channel];
      state.Reset();
      for (size_t y = 0; y < plane.extent.height; ++y)
        for (size_t x = 0; x < plane.extent.width; ++x) {
          const int64_t w = x   ? plane.Row(y)[x - 1]
                            : y ? plane.Row(y - 1)[x]
                                : 0;
          const int64_t n = y ? plane.Row(y - 1)[x] : w;
          const int64_t nw = x && y ? plane.Row(y - 1)[x - 1] : w;
          const int64_t ne =
              y && x + 1 < plane.extent.width ? plane.Row(y - 1)[x + 1] : n;
          const int64_t nn = y > 1 ? plane.Row(y - 2)[x] : n;
          const auto [prediction, property] =
              state.Predict(x, y, n, w, ne, nw, nn);
          const int64_t residual = int64_t{plane.Row(y)[x]} - prediction;
          if (residual < std::numeric_limits<int32_t>::min() ||
              residual > std::numeric_limits<int32_t>::max() ||
              !state.Update(plane.Row(y)[x], x, y))
            return Status::InvalidArgument(
                "Weighted DC predictor range exceeded");
          candidate.push_back(
              {Context(property), PackSigned(static_cast<int32_t>(residual))});
        }
    }
    *tokens = std::move(candidate);
  } catch (const resource_budget_internal::ManagedAllocationFailure &error) {
    return error.status();
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Weighted DC allocation failed");
  } catch (const std::length_error &) {
    return Status::OutOfMemory("Weighted DC allocation overflow");
  }
  return Status::Ok();
}

Status UseWeightedDcTree(std::span<EntropyToken> tokens) {
  size_t leaves = 0, dc_splits = 0;
  for (size_t i = 0; i < tokens.size();) {
    if (tokens[i].context != 1)
      return Status::InvalidArgument("Unexpected DC tree property token");
    if (tokens[i].value == 0) {
      if (tokens.size() - i < 5)
        return Status::InvalidArgument("Truncated DC tree leaf");
      for (size_t j = 1; j <= 4; ++j)
        if (tokens[i + j].context != j + 1)
          return Status::InvalidArgument("Unexpected DC tree leaf token");
      if (leaves >= 11 && tokens[i + 1].value != 5)
        return Status::InvalidArgument("Unexpected DC tree predictor");
      ++leaves;
      i += 5;
    } else {
      if (tokens.size() - i < 2 || tokens[i + 1].context != 0)
        return Status::InvalidArgument("Truncated DC tree split");
      if (tokens[i].value == 10)
        ++dc_splits;
      i += 2;
    }
  }
  if (leaves != 45 || dc_splits != 33)
    return Status::InvalidArgument("Unexpected fixed DC tree shape");
  leaves = 0;
  for (size_t i = 0; i < tokens.size();) {
    if (tokens[i].value == 0) {
      if (leaves++ >= 11)
        tokens[i + 1].value = 6;
      i += 5;
    } else {
      if (tokens[i].value == 10)
        tokens[i].value = 16;
      i += 2;
    }
  }
  return Status::Ok();
}
} // namespace gjxl::codestream_internal
