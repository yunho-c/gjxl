// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <algorithm>
#include <array>
#include <climits>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "codestream/dc_context_tree_internal.h"
#include "codestream/weighted_dc.h"
#ifdef GJXL_DC_TREE_ORACLE
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
struct Node {
  int property, value, prediction = 0;
  size_t left = 0, right = 0, context = 0;
};
std::vector<Node> Parse(const DcContextTreeLayout &layout) {
  std::vector<Node> tree(1);
  size_t pos = 0, context = 0;
  for (size_t i = 0; i < tree.size(); ++i) {
    Node node{int(layout.tokens[pos++].value) - 1, 0};
    if (node.property < 0) {
      node.prediction = layout.tokens[pos++].value;
      Check(layout.tokens[pos++].value == 0 &&
                layout.tokens[pos++].value == 0 &&
                layout.tokens[pos++].value == 0,
            "Unexpected leaf parameters");
      node.context = context++;
    } else {
      const uint32_t value = layout.tokens[pos++].value;
      node.value = int(value >> 1) ^ -int(value & 1);
      node.left = tree.size();
      node.right = tree.size() + 1;
      tree.resize(tree.size() + 2);
    }
    tree[i] = node;
  }
  Check(pos == layout.token_count && context == layout.context_count,
        "Invalid token/context count");
  return tree;
}
size_t Evaluate(const std::vector<Node> &tree,
                const std::array<int, 16> &properties) {
  size_t id = 0;
  while (tree[id].property >= 0) {
    const auto &n = tree[id];
    id = properties[n.property] > n.value ? n.left : n.right;
  }
  return tree[id].context;
}
#ifdef GJXL_DC_TREE_ORACLE
void CompareDc(const std::vector<Node> &actual, size_t a,
               const jxl::Tree &expected, size_t e) {
  const auto &x = actual[a];
  const auto &y = expected[e];
  Check(x.property == y.property, "DC property differs from pinned libjxl");
  if (x.property < 0) {
    Check(x.prediction == int(y.predictor),
          "DC predictor differs from pinned libjxl");
  } else {
    Check(x.value == y.splitval, "DC cutoff differs from pinned libjxl");
    CompareDc(actual, x.left, expected, y.lchild);
    CompareDc(actual, x.right, expected, y.rchild);
  }
}
void Oracle(const DcContextTreeLayout &layout, size_t samples) {
  auto expected =
      jxl::PredefinedTree(layout.prediction == VarDctDcPrediction::kWeighted
                              ? jxl::ModularOptions::TreeKind::kWPFixedDC
                              : jxl::ModularOptions::TreeKind::kGradientFixedDC,
                          samples, 8, 0);
  const auto actual = Parse(layout);
  CompareDc(actual, actual[0].right, expected, 0);
  // Exercise GJXL entropy serialization and the independent decoder's tree
  // parser. This establishes the wire leaf IDs, not just our token parser.
  EntropyCode code;
  const std::array streams{
      EntropyTokenStreamView::Interleaved(layout.tree_tokens())};
  Check(OptimizeEntropyCode(streams, {.context_count = 6}, &code).ok(),
        "Tree model failed");
  BitWriter writer;
  Check(writer.WriteBits(1, 0).ok() && WriteEntropyCode(code, &writer).ok() &&
            WriteTokenStream(layout.tree_tokens(), code, &writer).ok(),
        "Tree write failed");
  JxlMemoryManager manager;
  Check(bool(jxl::MemoryManagerInit(&manager, nullptr)),
        "Memory manager failed");
  auto bytes = writer.padded_bytes();
  jxl::BitReader reader(jxl::Span<const uint8_t>(bytes.data(), bytes.size()));
  jxl::Tree decoded;
  Check(bool(jxl::DecodeTree(&manager, &reader, &decoded, 1000)),
        "Tree roundtrip failed");
  Check(reader.Close() && decoded.size() == actual.size(),
        "Tree roundtrip extent failed");
  for (size_t i = 0; i < actual.size(); ++i) {
    const auto &a = actual[i];
    const auto &d = decoded[i];
    Check(a.property == d.property, "Roundtrip property mismatch");
    if (a.property < 0)
      Check(a.context == d.lchild && a.prediction == int(d.predictor),
            "Roundtrip leaf mismatch");
    else
      Check(a.value == d.splitval && a.left == d.lchild && a.right == d.rchild,
            "Roundtrip split mismatch");
  }
}
#endif
void LayoutCase(size_t samples, uint32_t leaves,
                VarDctDcPrediction prediction) {
  const DcContextTreeLayout *layout = nullptr;
  Check(SelectDcContextTreeLayout(samples, prediction, DcTreePolicy::kAdaptive,
                                  &layout)
            .ok(),
        "Select failed");
  Check(layout->dc_leaf_count == leaves && layout->context_count == leaves + 11,
        "Wrong leaf count");
  const auto &full = LegacyDcContextTree(prediction);
  const auto old_tree = Parse(full), new_tree = Parse(*layout);
  std::set<uint8_t> metadata;
  for (size_t i = 0; i < 11; ++i)
    metadata.insert(layout->context_map[i]);
  Check(metadata.size() == 11, "Metadata contexts merged");
  for (auto id : layout->context_map)
    Check(id < layout->context_count, "Invalid remapping");
  std::array<int, 16> props{};
  for (int value = -1100; value <= 1100; ++value) {
    props[9] = props[15] = value;
    Check(layout->context_map[Evaluate(old_tree, props)] ==
              Evaluate(new_tree, props),
          "DC remapping mismatch");
  }
  for (int value : {INT_MIN, INT_MAX}) {
    props[9] = props[15] = value;
    Check(layout->context_map[Evaluate(old_tree, props)] ==
              Evaluate(new_tree, props),
          "Extreme DC mismatch");
  }
  props[1] = 3; // Metadata stream > root's group threshold of two.
  for (int channel = 0; channel < 4; ++channel)
    for (int y : {0, 1})
      for (int left = -1; left < 15; ++left) {
        props[0] = channel;
        props[2] = y;
        props[7] = left;
        Check(layout->context_map[Evaluate(old_tree, props)] ==
                  Evaluate(new_tree, props),
              "Metadata remapping mismatch");
      }
  if (leaves == 34)
    Check(layout->tokens == full.tokens &&
              layout->context_map == full.context_map,
          "Full tree changed");
#ifdef GJXL_DC_TREE_ORACLE
  Oracle(*layout, samples);
#endif
}
} // namespace
int main() {
  try {
    Check(kDefaultDcTreePolicy == DcTreePolicy::kAdaptive &&
              CurrentDcTreePolicy() == DcTreePolicy::kAdaptive,
          "Size-adaptive DC tree is not the default");
    for (auto prediction :
         {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted}) {
      for (size_t n : {size_t{1}, size_t{2}, size_t{511}, size_t{512}})
        LayoutCase(n, 1, prediction);
      for (size_t n : {size_t{513}, size_t{1024}, size_t{2048}, size_t{4096}})
        LayoutCase(n, 2, prediction);
      for (size_t n : {size_t{4097}, size_t{8192}})
        LayoutCase(n, 4, prediction);
      for (size_t n :
           {size_t{8193}, size_t{16384}, std::numeric_limits<size_t>::max()})
        LayoutCase(n, 34, prediction);
      for (auto extent : {Extent2D{1, 1},
                          {257, 193},
                          {129, 97},
                          {768, 512},
                          {3840, 2160},
                          {2047, 9},
                          {2048, 9},
                          {2049, 9}}) {
        size_t n = 0;
        Check(ComputeDcSampleCount(extent.ceil_div(8), &n).ok(),
              "Geometry count failed");
        LayoutCase(n,
                   n <= 512    ? 1
                   : n <= 4096 ? 2
                   : n <= 8192 ? 4
                               : 34,
                   prediction);
      }
    }
    auto weighted = LegacyDcContextTree(VarDctDcPrediction::kGradient).tokens;
    Check(UseWeightedDcTree(weighted).ok() &&
              weighted ==
                  LegacyDcContextTree(VarDctDcPrediction::kWeighted).tokens,
          "Legacy weighted wire tokens changed");
    size_t count = 42;
    Check(!ComputeDcSampleCount({}, &count).ok() && count == 42 &&
              !ComputeDcSampleCount({std::numeric_limits<size_t>::max(), 2},
                                    &count)
                   .ok() &&
              count == 42 &&
              !ComputeDcSampleCount({std::numeric_limits<size_t>::max() / 2, 1},
                                    &count)
                   .ok() &&
              count == 42 && !ComputeDcSampleCount({1, 1}, nullptr).ok(),
          "Sample failure is not atomic");
    const auto *sentinel = &LegacyDcContextTree(VarDctDcPrediction::kGradient);
    const auto *output = sentinel;
    Check(!SelectDcContextTreeLayout(0, VarDctDcPrediction::kGradient,
                                     DcTreePolicy::kAdaptive, &output)
                  .ok() &&
              output == sentinel,
          "Selection failure is not atomic");
    Check(!SelectDcContextTreeLayout(1, static_cast<VarDctDcPrediction>(255),
                                     DcTreePolicy::kAdaptive, &output)
                  .ok() &&
              output == sentinel,
          "Invalid prediction accepted");
    Check(!SelectDcContextTreeLayout(1, VarDctDcPrediction::kGradient,
                                     static_cast<DcTreePolicy>(255), &output)
                  .ok() &&
              output == sentinel,
          "Invalid policy accepted");
    auto previous = CurrentDcTreePolicy();
    {
      ScopedDcTreePolicyForTesting outer(DcTreePolicy::kAdaptive);
      {
        ScopedDcTreePolicyForTesting inner(DcTreePolicy::kLegacy);
        Check(CurrentDcTreePolicy() == DcTreePolicy::kLegacy,
              "Inner selector failed");
      }
      Check(CurrentDcTreePolicy() == DcTreePolicy::kAdaptive,
            "Selector restoration failed");
    }
    Check(CurrentDcTreePolicy() == previous,
          "Outer selector restoration failed");
    std::cout << "DC context selection, remapping, and tree checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
