// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/modular/transform/enc_squeeze.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include "lib/jxl/pack_signed.h"
#include <map>
#include <stdexcept>

// Test-only independent checks for the exact frozen inputs. The broader policy
// and transform tests additionally derive group geometry independently.
namespace gjxl::modular_oracle_reference {
using namespace modular_internal;
inline void Require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void ReferenceStep(jxl::Image &image, const TransformDescriptor &t) {
  if (t.kind == TransformKind::kSqueeze) {
    jxl::SqueezeParams params;
    params.horizontal = t.horizontal;
    params.in_place = t.in_place;
    params.begin_c = t.begin;
    params.num_c = t.count;
    Require(bool(jxl::FwdSqueeze(image, {params}, nullptr)), "Pinned forward squeeze failed");
    return;
  }
  // Prescribed exact palette: independent ordered dictionary and pinned shape application.
  std::map<std::vector<int32_t>, int32_t> dictionary;
  std::vector<std::vector<int32_t>> pixels;
  const size_t w = image.channel[t.begin].w, h = image.channel[t.begin].h;
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      std::vector<int32_t> v;
      for (size_t c = 0; c < t.count; ++c)
        v.push_back(image.channel[t.begin + c].Row(y)[x]);
      dictionary.emplace(v, 0);
      pixels.push_back(std::move(v));
    }
  Require(dictionary.size() <= t.colors, "Bad palette fixture");
  jxl::Transform transform(jxl::TransformId::kPalette);
  transform.begin_c = t.begin;
  transform.num_c = t.count;
  transform.nb_colors = t.colors;
  transform.nb_deltas = 0;
  Require(bool(transform.MetaApply(image)), "Pinned palette shape failed");
  for (size_t c = 0; c < t.count; ++c)
    std::fill_n(image.channel[0].Row(c), t.colors, 0);
  int32_t index = 0;
  for (auto &[color, id] : dictionary) {
    id = index++;
    for (size_t c = 0; c < t.count; ++c)
      image.channel[0].Row(c)[id] = color[c];
  }
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      image.channel[t.begin + 1].Row(y)[x] = dictionary.at(pixels[y * w + x]);
}

inline void Stages(PackedModularImageView input, const ModularCodingPolicy &policy,
                   const ModularImage &native, const ModularStreamPlan &plan,
                   const PreparedModularTokens &tokens) {
  ModularInputProfile profile;
  Require(ResolveModularInput(input.extent, input.format, &profile).ok(),
          "Reference input profile");
  JxlMemoryManager memory;
  Require(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Reference memory manager");
  auto created = jxl::Image::Create(&memory, input.extent.width, input.extent.height,
                                    profile.metadata.bits, profile.channel_count);
  Require(created.ok(), "Reference image allocation");
  auto image = std::move(created).value_();
  for (size_t c = 0; c < profile.channel_count; ++c)
    for (size_t y = 0; y < input.extent.height; ++y)
      for (size_t x = 0; x < input.extent.width; ++x) {
        size_t offset =
            y * input.row_stride + (x * profile.channel_count + c) * profile.bytes_per_sample;
        uint32_t v = input.bytes[offset];
        if (profile.bytes_per_sample == 2)
          v |= uint32_t{input.bytes[offset + 1]} << 8;
        image.channel[c].Row(y)[x] = static_cast<int32_t>(v);
      }
  if (policy.rct) {
    auto created_rct = jxl::Image::Create(&memory, input.extent.width, input.extent.height,
                                          profile.metadata.bits, profile.channel_count);
    Require(created_rct.ok(), "Reference RCT allocation");
    auto rct = std::move(created_rct).value_();
    Require(
        bool(jxl::FwdRct({&image.channel[0], &image.channel[1], &image.channel[2]},
                         {&rct.channel[0], &rct.channel[1], &rct.channel[2]}, policy.rct, nullptr)),
        "Reference RCT");
    if (profile.channel_count == 4)
      rct.channel[3] = std::move(image.channel[3]);
    image = std::move(rct);
  }
  for (size_t i = 0; i < policy.transforms.size; ++i)
    ReferenceStep(image, policy.transforms.entries[i]);
  Require(native.channel_count() == image.channel.size() &&
              native.metadata_channels() == image.nb_meta_channels,
          "Reference channel shape");
  for (size_t c = 0; c < native.channel_count(); ++c) {
    auto n = native.view(c);
    const auto &r = image.channel[c];
    Require(n.descriptor.extent == Extent2D{r.w, r.h}, "Reference channel dimensions");
    if (c >= native.metadata_channels())
      Require(n.descriptor.hshift == r.hshift && n.descriptor.vshift == r.vshift,
              "Reference shifts");
    for (size_t y = 0; y < r.h; ++y)
      for (size_t x = 0; x < r.w; ++x)
        Require(n.Row(y)[x] == r.Row(y)[x], "Reference transformed sample");
  }
  jxl::Tree tree, decoded;
  for (size_t i = 0; i < policy.tree.size; ++i) {
    auto n = policy.tree.nodes[i];
    tree.emplace_back(n.property, n.split, n.left, n.right,
                      static_cast<jxl::Predictor>(n.predictor), n.offset, n.multiplier);
  }
  std::vector<jxl::Token> tree_tokens;
  Require(bool(jxl::TokenizeTree(tree, &tree_tokens, &decoded)), "Reference tree");
  Require(tree_tokens.size() == tokens.tree_token_count, "Reference tree token count");
  for (size_t i = 0; i < tree_tokens.size(); ++i)
    Require(tree_tokens[i].context == tokens.tree_tokens[i].context &&
                tree_tokens[i].value == tokens.tree_tokens[i].value,
            "Reference tree tokens");
  jxl::weighted::Header header;
  auto p = policy.weighted.coefficients;
  header.p1C = p[0];
  header.p2C = p[1];
  header.p3Ca = p[2];
  header.p3Cb = p[3];
  header.p3Cc = p[4];
  header.p3Cd = p[5];
  header.p3Ce = p[6];
  for (size_t i = 0; i < 4; ++i)
    header.w[i] = policy.weighted.weights[i];
  TreeLayout layout;
  Require(ValidateTree(policy.tree, &layout).ok(), "Native tree");
  for (size_t si = 0; si < plan.streams.size(); ++si) {
    auto stream = plan.streams[si];
    size_t token = 0;
    for (auto slice : std::span(plan.slices).subspan(stream.slice_begin, stream.slice_count)) {
      auto [w, h] = slice.rect.extent;
      auto tile_result = jxl::Channel::Create(&memory, w, h);
      auto refs_result = jxl::Channel::Create(&memory, 0, w);
      Require(tile_result.ok() && refs_result.ok(), "Reference tile");
      auto tile = std::move(tile_result).value_(), refs = std::move(refs_result).value_();
      for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x)
          tile.Row(y)[x] = image.channel[slice.channel].Row(y + slice.rect.y)[x + slice.rect.x];
      jxl::weighted::State state(header, w, h);
      WeightedPredictor<resource_budget_internal::ResourceClass::kPreparation> wp(w,
                                                                                  policy.weighted);
      ModularChannelView view;
      Require(BorrowChannelSlice(native.view(slice.channel), slice.rect, &view).ok(),
              "Native tile");
      jxl::Properties properties(16);
      for (size_t y = 0; y < h; ++y) {
        jxl::InitPropsRow(&properties,
                          {static_cast<int32_t>(slice.channel), static_cast<int32_t>(stream.id)},
                          static_cast<int>(y));
        int64_t previous = 0;
        for (size_t x = 0; x < w; ++x) {
          std::array<int64_t, 14> predictions{};
          jxl::PredictLearnAll(&properties, w, tile.Row(y) + x, tile.plane.PixelsPerRow(),
                               static_cast<int>(x), static_cast<int>(y), refs, &state,
                               predictions.data());
          auto n = Neighbors(view, x, y);
          auto weighted = wp.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
          auto actual = Properties(n, slice.channel, stream.id, x, y, previous, weighted.second);
          previous = actual[9];
          for (size_t i = 0; i < 16; ++i)
            Require(actual[i] == properties[i], "Reference properties");
          for (size_t i = 0; i < 14; ++i)
            Require(Predict(static_cast<Predictor>(i), n, weighted.first) == predictions[i],
                    "Reference predictions");
          size_t leaf = 0;
          while (decoded[leaf].property != -1) {
            const auto &node = decoded[leaf];
            leaf = properties[node.property] > node.splitval ? node.lchild : node.rchild;
          }
          const auto &node = decoded[leaf];
          auto residual = tile.Row(y)[x] - predictions[static_cast<size_t>(node.predictor)] -
                          node.predictor_offset;
          auto expected = jxl::PackSigned(static_cast<int32_t>(residual / node.multiplier));
          Require(tokens.streams[si][token].context == node.lchild &&
                      tokens.streams[si][token].value == expected,
                  "Reference residual/context");
          ++token;
          state.UpdateErrors(tile.Row(y)[x], x, y, w);
          Require(wp.Update(view.Row(y)[x], x, y), "Native weighted range");
        }
      }
    }
    Require(token == tokens.streams[si].size(), "Reference token coverage");
  }
}
} // namespace gjxl::modular_oracle_reference
