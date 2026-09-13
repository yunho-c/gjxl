// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// The pruning rule follows pinned libjxl MakeFixedTree. No runtime dependency.
#include "codestream/dc_context_tree_internal.h"

#include <bit>
#include <limits>

namespace gjxl::codestream_internal {
namespace {
#include "codestream/dc_context_tree_legacy.inc"

struct Node {
  uint32_t property = 0;  // Wire property (zero denotes a leaf).
  uint32_t threshold = 0; // Already PackSigned encoded.
  std::array<EntropyToken, 4> leaf{};
  size_t left = 0, right = 0;
  uint8_t old_context = 0;
};
struct Tree {
  std::array<Node, 89> nodes{};
  size_t count = 0;
};

constexpr Tree ParseLegacy() {
  Tree tree;
  size_t pos = 0, next = 1, context = 0;
  for (size_t i = 0; i < next; ++i) {
    auto &node = tree.nodes[i];
    node.property = kContextTreeTokens[pos++].value;
    if (node.property == 0) {
      for (auto &token : node.leaf)
        token = kContextTreeTokens[pos++];
      node.old_context = context++;
    } else {
      node.threshold = kContextTreeTokens[pos++].value;
      node.left = next++;
      node.right = next++;
    }
  }
  tree.count = next;
  return tree;
}
constexpr auto kLegacy = ParseLegacy();
static_assert(kLegacy.count == 89 && kLegacy.nodes[0].right == 2);

constexpr void PruneDc(Tree &tree, size_t node, size_t begin, size_t end,
                       size_t min_gap, VarDctDcPrediction prediction) {
  auto &n = tree.nodes[node];
  if (begin + min_gap >= end) {
    n.property = 0;
    n.leaf = {{{2, prediction == VarDctDcPrediction::kWeighted ? 6u : 5u},
               {3, 0},
               {4, 0},
               {5, 0}}};
    return;
  }
  n.property = prediction == VarDctDcPrediction::kWeighted ? 16 : 10;
  const size_t split = (begin + end) / 2;
  PruneDc(tree, n.left, split + 1, end, min_gap, prediction);
  PruneDc(tree, n.right, begin, split, min_gap, prediction);
}

constexpr void MapDescendants(size_t old_node, uint8_t context,
                              DcContextTreeLayout &layout) {
  const auto &n = kLegacy.nodes[old_node];
  if (n.property == 0) {
    layout.context_map[n.old_context] = context;
  } else {
    MapDescendants(n.left, context, layout);
    MapDescendants(n.right, context, layout);
  }
}

constexpr DcContextTreeLayout Build(size_t min_gap,
                                    VarDctDcPrediction prediction) {
  Tree tree = kLegacy;
  PruneDc(tree, tree.nodes[0].right, 0, 33, min_gap, prediction);
  DcContextTreeLayout result;
  result.prediction = prediction;
  std::array<size_t, 89> queue{};
  size_t stop = 1;
  for (size_t start = 0; start < stop; ++start) {
    const size_t id = queue[start];
    const auto &n = tree.nodes[id];
    result.tokens[result.token_count++] = {1, n.property};
    if (n.property == 0) {
      for (auto token : n.leaf)
        result.tokens[result.token_count++] = token;
      MapDescendants(id, result.context_count++, result);
    } else {
      result.tokens[result.token_count++] = {0, n.threshold};
      queue[stop++] = n.left;
      queue[stop++] = n.right;
    }
  }
  result.dc_leaf_count = result.context_count - 11;
  return result;
}
// Only four different topologies arise from min_gap in multiples of eight.
constexpr std::array<size_t, 4> kGaps{0, 8, 16, 40};
constexpr auto kLayouts = [] {
  std::array<std::array<DcContextTreeLayout, 4>, 2> result{};
  for (size_t p = 0; p < 2; ++p)
    for (size_t i = 0; i < 4; ++i)
      result[p][i] = Build(kGaps[i], p == 0 ? VarDctDcPrediction::kGradient
                                            : VarDctDcPrediction::kWeighted);
  return result;
}();
static_assert(kLayouts[0][0].token_count == std::size(kContextTreeTokens));
static_assert([] {
  for (size_t i = 0; i < std::size(kContextTreeTokens); ++i)
    if (kLayouts[0][0].tokens[i] != kContextTreeTokens[i])
      return false;
  for (size_t i = 0; i < 45; ++i)
    if (kLayouts[0][0].context_map[i] != i)
      return false;
  return true;
}());
static_assert(kLayouts[0][1].dc_leaf_count == 4 &&
              kLayouts[0][2].dc_leaf_count == 2 &&
              kLayouts[0][3].dc_leaf_count == 1);

thread_local DcTreePolicy policy = kDefaultDcTreePolicy;
} // namespace

Status ComputeDcSampleCount(Extent2D blocks, size_t *samples) {
  size_t area;
  if (samples == nullptr || blocks.empty() || !blocks.try_area(&area) ||
      area > std::numeric_limits<size_t>::max() / 3)
    return Status::InvalidArgument("DC sample count is invalid or overflows");
  *samples = 3 * area;
  return Status::Ok();
}
Status SelectDcContextTreeLayout(size_t samples, VarDctDcPrediction prediction,
                                 DcTreePolicy choice,
                                 const DcContextTreeLayout **layout) {
  if (samples == 0 || layout == nullptr || !IsValidDcPrediction(prediction) ||
      (choice != DcTreePolicy::kLegacy && choice != DcTreePolicy::kAdaptive))
    return Status::InvalidArgument("DC tree selection is invalid");
  size_t index = 0;
  if (choice == DcTreePolicy::kAdaptive) {
    const auto log = std::bit_width(samples - 1);
    index = log >= 14 ? 0 : log == 13 ? 1 : log >= 10 ? 2 : 3;
  }
  *layout = &kLayouts[prediction == VarDctDcPrediction::kWeighted][index];
  return Status::Ok();
}
const DcContextTreeLayout &LegacyDcContextTree(VarDctDcPrediction prediction) {
  return kLayouts[prediction == VarDctDcPrediction::kWeighted][0];
}
ScopedDcTreePolicyForTesting::ScopedDcTreePolicyForTesting(DcTreePolicy value)
    : previous_(policy) {
  policy = value;
}
ScopedDcTreePolicyForTesting::~ScopedDcTreePolicyForTesting() {
  policy = previous_;
}
DcTreePolicy CurrentDcTreePolicy() { return policy; }
} // namespace gjxl::codestream_internal
