// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates seven-strategy AC-group logical tokenization.

#include "codestream/ac_group.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/reconstruction.h"

namespace {

template <typename T>
gjxl::PlaneView<const T> View(const std::vector<T>& values,
                              gjxl::Extent2D extent) {
  return {values.data(), extent, extent.width};
}

uint64_t HashValues(std::span<const uint32_t> values) {
  uint64_t hash = 1469598103934665603ull;
  for (uint32_t value : values) {
    hash ^= value;
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t HashTokens(std::span<const gjxl::EntropyToken> tokens) {
  uint64_t hash = 1469598103934665603ull;
  for (const gjxl::EntropyToken token : tokens) {
    hash ^= token.context;
    hash *= 1099511628211ull;
    hash ^= token.value;
    hash *= 1099511628211ull;
  }
  return hash;
}

struct OrderGolden {
  gjxl::AcStrategyType strategy;
  size_t size;
  uint64_t hash;
  size_t middle_index;
  uint32_t middle_value;
};

// Full-order hashes and asymmetric samples from pinned libjxl
// e8ff09762481785938d8e4e01333ed3917571161.
constexpr std::array<OrderGolden, 7> kOrderGoldens = {{
  {gjxl::AcStrategyType::kDct8, 64, 0x3429f64e9a8101d3ull, 32, 35},
  {gjxl::AcStrategyType::kDct16x16, 256, 0x08f226b939195a15ull, 128, 135},
  {gjxl::AcStrategyType::kDct32x32, 1024, 0x4e30e44c269548f1ull, 512, 527},
  {gjxl::AcStrategyType::kDct16x8, 128, 0x6100fcbe75b9416bull, 64, 15},
  {gjxl::AcStrategyType::kDct8x16, 128, 0x6100fcbe75b9416bull, 64, 15},
  {gjxl::AcStrategyType::kDct32x16, 512, 0x14407aca0189e559ull, 256, 31},
  {gjxl::AcStrategyType::kDct16x32, 512, 0x14407aca0189e559ull, 256, 31},
}};

bool CheckNaturalOrders() {
  for (const OrderGolden& golden : kOrderGoldens) {
    const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(golden.strategy);
    std::vector<uint32_t> order;
    const gjxl::Status status =
      gjxl::ComputeSimpleNaturalCoefficientOrder(golden.strategy, &order);
    if (!status.ok() || info == nullptr || order.size() != golden.size
        || HashValues(order) != golden.hash
        || order[golden.middle_index] != golden.middle_value
        || order.back() != golden.size - 1) {
      std::cerr << "Natural order differs for strategy "
                << static_cast<int>(golden.strategy) << ": " << status.message()
                << '\n';
      return false;
    }

    std::vector<uint8_t> seen(order.size(), 0);
    for (uint32_t index : order) {
      if (index >= order.size() || seen[index] != 0) {
        std::cerr << "Natural order is not a permutation\n";
        return false;
      }
      seen[index] = 1;
    }
    const gjxl::Extent2D llf = info->low_frequency_extent();
    const gjxl::Extent2D coefficients = info->coefficient_extent();
    for (size_t y = 0; y < llf.height; ++y) {
      for (size_t x = 0; x < llf.width; ++x) {
        const size_t scan = y * llf.width + x;
        if (order[scan] != y * coefficients.width + x) {
          std::cerr << "Natural order has an incorrect LLF prefix\n";
          return false;
        }
      }
    }
  }

  std::vector<uint32_t> sentinel = {7, 9};
  const gjxl::Status status = gjxl::ComputeSimpleNaturalCoefficientOrder(
    gjxl::AcStrategyType::kIdentity, &sentinel);
  if (status.code() != gjxl::StatusCode::kInvalidArgument
      || sentinel != std::vector<uint32_t>({7, 9})) {
    std::cerr << "Unsupported natural order changed its output\n";
    return false;
  }
  return true;
}

struct TokenGolden {
  gjxl::AcStrategyType strategy;
  size_t token_count;
  uint64_t hash;
};

// Full token hashes from the pinned tiny/libjxl context and scan equations.
constexpr std::array<TokenGolden, 7> kTokenGoldens = {{
  {gjxl::AcStrategyType::kDct8, 192, 0x4bc215f68f1ece9dull},
  {gjxl::AcStrategyType::kDct16x16, 759, 0xb4eb38be028e0165ull},
  {gjxl::AcStrategyType::kDct32x32, 3027, 0x7fa649fd5ea7a5d5ull},
  {gjxl::AcStrategyType::kDct16x8, 381, 0xd61b35ea9efaa9e2ull},
  {gjxl::AcStrategyType::kDct8x16, 381, 0xd61b35ea9efaa9e2ull},
  {gjxl::AcStrategyType::kDct32x16, 1515, 0x85c67d0bcd6d70bcull},
  {gjxl::AcStrategyType::kDct16x32, 1515, 0x85c67d0bcd6d70bcull},
}};

gjxl::VarDctAcGroupView MakeGroupView(
  gjxl::Extent2D block_extent, size_t used,
  const std::array<std::vector<int32_t>, 3>& storage, size_t block_x = 0,
  size_t block_y = 0) {
  gjxl::VarDctAcGroupView group{
    .block_x = block_x,
    .block_y = block_y,
    .block_extent = block_extent,
    .used_coefficient_count = used,
  };
  for (size_t channel = 0; channel < 3; ++channel) {
    group.coefficients[channel] = storage[channel];
  }
  return group;
}

void PoisonLlf(const gjxl::AcStrategyInfo& info, size_t offset, int32_t base,
               std::vector<int32_t>* coefficients) {
  const gjxl::Extent2D llf = info.low_frequency_extent();
  const gjxl::Extent2D extent = info.coefficient_extent();
  for (size_t y = 0; y < llf.height; ++y) {
    for (size_t x = 0; x < llf.width; ++x) {
      (*coefficients)[offset + y * extent.width + x] =
        base + static_cast<int32_t>(y * llf.width + x);
    }
  }
}

bool CheckPinnedStrategyTokenStreams() {
  for (const TokenGolden& golden : kTokenGoldens) {
    const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(golden.strategy);
    if (info == nullptr) {
      return false;
    }
    gjxl::AcStrategyGrid strategies;
    if (!gjxl::AcStrategyGrid::Create(info->covered_blocks, &strategies).ok()
        || !strategies.Set(0, 0, golden.strategy).ok()) {
      return false;
    }

    std::vector<uint32_t> order;
    if (!gjxl::ComputeSimpleNaturalCoefficientOrder(golden.strategy, &order)
           .ok()) {
      return false;
    }
    const size_t size = info->coefficient_count();
    const size_t covered =
      info->covered_blocks.width * info->covered_blocks.height;
    std::array<std::vector<int32_t>, 3> storage;
    for (size_t channel = 0; channel < 3; ++channel) {
      storage[channel].assign(size, 0);
      PoisonLlf(*info, 0, 1000 + static_cast<int32_t>(100 * channel),
                &storage[channel]);
      storage[channel][order[covered]] = -static_cast<int32_t>(channel + 1);
      storage[channel][order[covered + (size - covered) / 2]] =
        static_cast<int32_t>(channel + 5);
      storage[channel][order[size - 1]] = -static_cast<int32_t>(channel + 10);
    }

    const gjxl::VarDctAcGroupView group =
      MakeGroupView(info->covered_blocks, size, storage);
    std::vector<gjxl::EntropyToken> tokens;
    const gjxl::Status status =
      gjxl::TokenizeSimpleAcGroup(group, strategies, &tokens);
    const uint64_t token_hash = HashTokens(tokens);
    if (!status.ok() || tokens.size() != golden.token_count
        || token_hash != golden.hash
        || !std::ranges::all_of(tokens, [](gjxl::EntropyToken token) {
             return token.context < gjxl::kSimpleAcContextCount;
           })) {
      std::cerr << "Pinned AC tokens differ for strategy "
                << static_cast<int>(golden.strategy) << ": " << status.message()
                << " count=" << tokens.size() << " hash=0x" << std::hex
                << token_hash << std::dec << '\n';
      return false;
    }
  }
  return true;
}

bool CheckPinnedStrategyBlockContexts() {
  struct ContextGolden {
    gjxl::AcStrategyType strategy;
    std::array<uint32_t, 3> first_token_contexts;
  };
  constexpr std::array<ContextGolden, 7> kContextGoldens = {{
    {gjxl::AcStrategyType::kDct8, {80, 82, 82}},
    {gjxl::AcStrategyType::kDct16x16, {80, 82, 82}},
    {gjxl::AcStrategyType::kDct32x32, {80, 82, 82}},
    {gjxl::AcStrategyType::kDct16x8, {81, 83, 83}},
    {gjxl::AcStrategyType::kDct8x16, {81, 83, 83}},
    {gjxl::AcStrategyType::kDct32x16, {81, 83, 83}},
    {gjxl::AcStrategyType::kDct16x32, {81, 83, 83}},
  }};

  for (const ContextGolden& golden : kContextGoldens) {
    const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(golden.strategy);
    gjxl::AcStrategyGrid strategies;
    if (info == nullptr
        || !gjxl::AcStrategyGrid::Create(info->covered_blocks, &strategies).ok()
        || !strategies.Set(0, 0, golden.strategy).ok()) {
      return false;
    }

    const size_t size = info->coefficient_count();
    std::array<std::vector<int32_t>, 3> storage;
    for (std::vector<int32_t>& coefficients : storage) {
      coefficients.assign(size, 0);
    }
    const gjxl::VarDctAcGroupView group =
      MakeGroupView(info->covered_blocks, size, storage);
    std::vector<gjxl::EntropyToken> tokens;
    const gjxl::Status status =
      gjxl::TokenizeSimpleAcGroup(group, strategies, &tokens);
    if (!status.ok() || tokens.size() != golden.first_token_contexts.size()) {
      std::cerr << "Block-context fixture failed for strategy "
                << static_cast<int>(golden.strategy) << '\n';
      return false;
    }
    for (size_t channel = 0; channel < tokens.size(); ++channel) {
      if (tokens[channel].context != golden.first_token_contexts[channel]
          || tokens[channel].value != 0) {
        std::cerr << "Block context differs for strategy "
                  << static_cast<int>(golden.strategy) << " at token "
                  << channel << '\n';
        return false;
      }
    }
  }
  return true;
}

bool CheckZeroDenseAndSignedExtremes() {
  constexpr gjxl::Extent2D kOneBlock{1, 1};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kOneBlock, &strategies).ok()
      || !strategies.Set(0, 0, gjxl::AcStrategyType::kDct8).ok()) {
    return false;
  }

  std::array<std::vector<int32_t>, 3> storage;
  for (size_t channel = 0; channel < 3; ++channel) {
    storage[channel].assign(64, 0);
    storage[channel][0] = std::numeric_limits<int32_t>::max();
  }
  gjxl::VarDctAcGroupView group = MakeGroupView(kOneBlock, 64, storage);
  std::vector<gjxl::EntropyToken> tokens;
  const std::array<gjxl::EntropyToken, 3> zero_expected = {{
    {80, 0},
    {82, 0},
    {82, 0},
  }};
  gjxl::Status status = gjxl::TokenizeSimpleAcGroup(group, strategies, &tokens);
  if (!status.ok() || !std::ranges::equal(tokens, zero_expected)) {
    std::cerr << "Zero-AC or poisoned-LLF fixture failed\n";
    return false;
  }

  for (size_t channel = 0; channel < 3; ++channel) {
    std::fill(storage[channel].begin(), storage[channel].end(), 1);
    storage[channel][0] = -777;
  }
  group = MakeGroupView(kOneBlock, 64, storage);
  status = gjxl::TokenizeSimpleAcGroup(group, strategies, &tokens);
  if (!status.ok() || tokens.size() != 192
      || tokens[0] != gjxl::EntropyToken{80, 63}
      || tokens[64] != gjxl::EntropyToken{82, 63}
      || tokens[128] != gjxl::EntropyToken{82, 63}) {
    std::cerr << "Dense AC fixture failed\n";
    return false;
  }

  for (std::vector<int32_t>& channel : storage) {
    std::fill(channel.begin(), channel.end(), 0);
  }
  storage[0][1] = std::numeric_limits<int32_t>::min();
  storage[0][8] = std::numeric_limits<int32_t>::max();
  group = MakeGroupView(kOneBlock, 64, storage);
  const std::array<gjxl::EntropyToken, 5> signed_expected = {{
    {80, 0},
    {82, 2},
    {1127, std::numeric_limits<uint32_t>::max()},
    {1067, std::numeric_limits<uint32_t>::max() - 1},
    {82, 0},
  }};
  status = gjxl::TokenizeSimpleAcGroup(group, strategies, &tokens);
  if (!status.ok() || !std::ranges::equal(tokens, signed_expected)) {
    std::cerr << "Signed AC extremes were not preserved\n";
    return false;
  }
  return true;
}

bool CheckMultiblockPredictionAndOffsets() {
  constexpr gjxl::Extent2D kBlocks{4, 3};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kBlocks, &strategies).ok()
      || !strategies.Set(0, 0, gjxl::AcStrategyType::kDct8x16).ok()
      || !strategies.Set(0, 1, gjxl::AcStrategyType::kDct16x8).ok()) {
    return false;
  }
  strategies.fill_empty_dct8();

  size_t used = 0;
  size_t anchor_count = 0;
  gjxl::Status status = strategies.ForEachAnchor(
    [&](size_t, size_t, gjxl::AcStrategyType strategy) {
      const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
      if (info == nullptr) {
        return gjxl::Status::InvalidArgument("Unknown test strategy");
      }
      used += info->coefficient_count();
      ++anchor_count;
      return gjxl::Status::Ok();
    });
  if (!status.ok()) {
    return false;
  }

  std::array<std::vector<int32_t>, 3> storage;
  for (std::vector<int32_t>& values : storage) {
    values.assign(used, 0);
  }
  size_t offset = 0;
  size_t anchor_index = 0;
  status = strategies.ForEachAnchor(
    [&](size_t, size_t, gjxl::AcStrategyType strategy) {
      const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
      std::vector<uint32_t> order;
      if (info == nullptr) {
        return gjxl::Status::InvalidArgument("Unknown test strategy");
      }
      gjxl::Status order_status =
        gjxl::ComputeSimpleNaturalCoefficientOrder(strategy, &order);
      if (!order_status.ok()) {
        return order_status;
      }
      const size_t covered =
        info->covered_blocks.width * info->covered_blocks.height;
      for (size_t channel = 0; channel < 3; ++channel) {
        PoisonLlf(*info, offset, 700 + static_cast<int32_t>(channel),
                  &storage[channel]);
        storage[channel][offset + order[covered]] =
          static_cast<int32_t>(100 + 10 * anchor_index + channel);
      }
      offset += info->coefficient_count();
      ++anchor_index;
      return gjxl::Status::Ok();
    });
  if (!status.ok() || offset != used) {
    return false;
  }

  const gjxl::VarDctAcGroupView group = MakeGroupView(kBlocks, used, storage);
  std::vector<gjxl::EntropyToken> tokens;
  status = gjxl::TokenizeSimpleAcGroup(group, strategies, &tokens);
  if (!status.ok() || tokens.size() != anchor_count * 6
      || tokens[0].context != 81 || tokens[6].context != 4
      || tokens[18].context != 5) {
    std::cerr << "Multiblock nonzero prediction failed: " << status.message()
              << '\n';
    return false;
  }
  for (size_t anchor = 0; anchor < anchor_count; ++anchor) {
    if (tokens[anchor * 6 + 1].value
          != gjxl::PackSigned(static_cast<int32_t>(101 + 10 * anchor))
        || tokens[anchor * 6 + 3].value
             != gjxl::PackSigned(static_cast<int32_t>(100 + 10 * anchor))
        || tokens[anchor * 6 + 5].value
             != gjxl::PackSigned(static_cast<int32_t>(102 + 10 * anchor))) {
      std::cerr << "Transform coefficient offsets were not preserved\n";
      return false;
    }
  }
  return true;
}

bool CheckMalformedGroupsAreAtomic() {
  constexpr gjxl::Extent2D kOneBlock{1, 1};
  gjxl::AcStrategyGrid dct8;
  gjxl::AcStrategyGrid unsupported;
  gjxl::AcStrategyGrid crossing;
  if (!gjxl::AcStrategyGrid::Create(kOneBlock, &dct8).ok()
      || !dct8.Set(0, 0, gjxl::AcStrategyType::kDct8).ok()
      || !gjxl::AcStrategyGrid::Create(kOneBlock, &unsupported).ok()
      || !unsupported.Set(0, 0, gjxl::AcStrategyType::kIdentity).ok()
      || !gjxl::AcStrategyGrid::Create({2, 1}, &crossing).ok()
      || !crossing.Set(0, 0, gjxl::AcStrategyType::kDct8x16).ok()) {
    return false;
  }

  std::array<std::vector<int32_t>, 3> storage;
  for (std::vector<int32_t>& values : storage) {
    values.assign(65, 0);
  }
  const gjxl::VarDctAcGroupView valid = MakeGroupView(kOneBlock, 64, storage);
  const std::vector<gjxl::EntropyToken> sentinel = {{1999, 123}};
  std::vector<gjxl::EntropyToken> tokens = sentinel;
  const auto rejects = [&](const gjxl::VarDctAcGroupView& group,
                           const gjxl::AcStrategyGrid& strategies,
                           std::string_view name) {
    const gjxl::Status status =
      gjxl::TokenizeSimpleAcGroup(group, strategies, &tokens);
    if (status.code() != gjxl::StatusCode::kInvalidArgument
        || tokens != sentinel) {
      std::cerr << name << " rejection was not atomic\n";
      return false;
    }
    return true;
  };

  gjxl::VarDctAcGroupView short_used = valid;
  short_used.used_coefficient_count = 63;
  gjxl::VarDctAcGroupView long_used = valid;
  long_used.used_coefficient_count = 65;
  gjxl::VarDctAcGroupView short_span = valid;
  short_span.coefficients[2] = short_span.coefficients[2].first(63);
  const gjxl::VarDctAcGroupView unsupported_group =
    MakeGroupView(kOneBlock, 64, storage);
  const gjxl::VarDctAcGroupView crossing_group =
    MakeGroupView(kOneBlock, 64, storage, 1, 0);
  return rejects(short_used, dct8, "short consumption")
         && rejects(long_used, dct8, "long consumption")
         && rejects(short_span, dct8, "short coefficient span")
         && rejects(unsupported_group, unsupported, "unsupported strategy")
         && rejects(crossing_group, crossing, "cross-group strategy");
}

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D extent) : extent(extent) {
    for (std::vector<float>& values : plane) {
      values.resize(extent.width * extent.height);
    }
  }

  gjxl::ConstImage3FView ConstView() const {
    return gjxl::ConstImage3FView{{
      gjxl::ConstPlaneF32View{plane[0].data(), extent, extent.width},
      gjxl::ConstPlaneF32View{plane[1].data(), extent, extent.width},
      gjxl::ConstPlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> plane;
};

gjxl::Status MakeFrame(size_t width,
                       gjxl::SimpleVarDctCodestreamProfile profile,
                       gjxl::VarDctEncoderFrame* frame) {
  gjxl::FrameGeometry geometry;
  gjxl::Status status = gjxl::FrameGeometry::Create(width, 1, &geometry);
  if (!status.ok()) {
    return status;
  }
  ImageStorage opsin(geometry.padded_frame());
  for (size_t y = 0; y < opsin.extent.height; ++y) {
    for (size_t x = 0; x < opsin.extent.width; ++x) {
      const size_t index = y * opsin.extent.width + x;
      const float value =
        static_cast<float>((x * 17 + y * 11 + 5) % 97) * 0.002f;
      opsin.plane[0][index] = value - 0.01f;
      opsin.plane[1][index] = value + 0.02f;
      opsin.plane[2][index] = 1.2f * value - 0.03f;
    }
  }

  const gjxl::Extent2D blocks = geometry.block_grid().blocks;
  size_t block_count = 0;
  if (!blocks.try_area(&block_count)) {
    return gjxl::Status::InvalidArgument("Test block count overflow");
  }
  gjxl::AcStrategyGrid strategies;
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  if (!(status = gjxl::AcStrategyGrid::Create(blocks, &strategies)).ok()) {
    return status;
  }
  strategies.fill_dct8();
  if (!(status = gjxl::Quantizer::Create({3541, 10}, &quantizer)).ok()) {
    return status;
  }
  if (!(status = gjxl::ComputeInitialColorCorrelationMap(opsin.ConstView(),
                                                         &color_correlation))
         .ok()) {
    return status;
  }
  const std::vector<int32_t> quant(block_count, 29);
  const std::vector<uint8_t> sharpness(block_count, 4);
  return gjxl::ComputeQuantizedCoefficients(
    opsin.ConstView(),
    {
      .geometry = geometry,
      .strategies = &strategies,
      .raw_quant_field = View(quant, blocks),
      .quantizer = &quantizer,
      .color_correlation = &color_correlation,
      .epf_sharpness = View(sharpness, blocks),
    },
    profile, frame);
}

bool CheckFrameGroups(size_t width, size_t expected_group_count) {
  gjxl::VarDctEncoderFrame frame;
  gjxl::Status status = MakeFrame(width, {}, &frame);
  std::vector<gjxl::SimpleAcGroupTokenStream> groups;
  if (status.ok()) {
    status = gjxl::TokenizeSimpleAcGroups(frame, &groups);
  }
  if (!status.ok() || groups.size() != expected_group_count) {
    std::cerr << width
              << "-pixel AC-group traversal failed: " << status.message()
              << '\n';
    return false;
  }

  for (size_t index = 0; index < groups.size(); ++index) {
    gjxl::VarDctAcGroupView group;
    std::vector<gjxl::EntropyToken> isolated;
    if (!frame.GetAcGroup(index, &group).ok()
        || !gjxl::TokenizeSimpleAcGroup(group, frame.strategies(), &isolated)
              .ok()
        || groups[index].block_x != group.block_x
        || groups[index].block_y != group.block_y
        || groups[index].block_extent != group.block_extent
        || groups[index].tokens != isolated) {
      std::cerr << "Frame and isolated AC-group tokenization differ\n";
      return false;
    }
  }
  if (expected_group_count == 2
      && (groups[1].block_x != 32
          || groups[1].block_extent != gjxl::Extent2D{1, 1}
          || groups[1].tokens.empty() || groups[1].tokens[0].context != 80)) {
    std::cerr << "Edge AC group did not reset its nonzero predictor\n";
    return false;
  }
  return true;
}

bool CheckFrameTraversalAndAtomicity() {
  if (!CheckFrameGroups(256, 1) || !CheckFrameGroups(257, 2)) {
    return false;
  }

  gjxl::SimpleVarDctCodestreamProfile profile;
  profile.b_qm_scale = 7;
  gjxl::VarDctEncoderFrame invalid_profile;
  if (!MakeFrame(8, profile, &invalid_profile).ok()) {
    return false;
  }
  std::vector<gjxl::SimpleAcGroupTokenStream> groups(1);
  groups[0].block_x = 999;
  const gjxl::Status status =
    gjxl::TokenizeSimpleAcGroups(invalid_profile, &groups);
  if (status.code() != gjxl::StatusCode::kInvalidArgument || groups.size() != 1
      || groups[0].block_x != 999) {
    std::cerr << "Rejected frame changed AC-group output\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckNaturalOrders() || !CheckPinnedStrategyBlockContexts()
      || !CheckPinnedStrategyTokenStreams()
      || !CheckZeroDenseAndSignedExtremes()
      || !CheckMultiblockPredictionAndOffsets()
      || !CheckMalformedGroupsAreAtomic()
      || !CheckFrameTraversalAndAtomicity()) {
    return EXIT_FAILURE;
  }
  std::cout << "All AC-group tokenization tests passed.\n";
  return EXIT_SUCCESS;
}
