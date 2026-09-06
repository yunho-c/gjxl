// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "codec/reconstruction.h"
#include "codec/vardct_frame_view_internal.h"
#include "codestream/ac_group.h"
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/dc_group.h"
#include "codestream/encoder_internal.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using vardct_frame_internal::BorrowFrame;
using vardct_frame_internal::VarDctFrameView;
using vardct_frame_internal::VarDctFrameViewData;

template <typename T>
concept CanBorrowFrame =
    requires(T&& frame) { BorrowFrame(std::forward<T>(frame)); };
static_assert(CanBorrowFrame<VarDctEncoderFrame&>);
static_assert(CanBorrowFrame<const VarDctEncoderFrame&>);
static_assert(!CanBorrowFrame<VarDctEncoderFrame>);
static_assert(!CanBorrowFrame<const VarDctEncoderFrame>);
static_assert(
    !std::is_convertible_v<const VarDctEncoderFrame&, VarDctFrameView>);

#define CHECK(condition)                                              \
  do {                                                                \
    if (!(condition)) {                                               \
      std::cerr << "Line " << __LINE__ << ": " << #condition << '\n'; \
      return false;                                                   \
    }                                                                 \
  } while (false)

template <typename T>
struct PlaneStorage {
  std::vector<T> values;
  Extent2D extent;
  size_t stride = 0;

  void Copy(PlaneView<const T> source) {
    extent = source.extent;
    stride = extent.width + 7;
    values.assign(stride * extent.height, T{99});
    for (size_t y = 0; y < extent.height; ++y) {
      std::copy_n(source.Row(y), extent.width, values.data() + y * stride);
    }
  }
  PlaneView<const T> view() const { return {values.data(), extent, stride}; }
};

// Deliberately not a VarDctEncoderFrame. Every large plane has guarded strides,
// metadata is independently owned, and AC storage is a separate allocation.
struct ExternalFrameStorage {
  FrameGeometry geometry;
  AcStrategyGrid strategies;
  Quantizer quantizer;
  ColorCorrelationMap correlation;
  SimpleVarDctCodestreamProfile profile;
  PlaneStorage<int32_t> raw;
  PlaneStorage<uint8_t> sharpness;
  std::array<PlaneStorage<int32_t>, 3> quantized_dc;
  std::array<PlaneStorage<float>, 3> dc;
  Extent2D group_extent;
  std::vector<size_t> used;
  std::vector<int32_t> ac;

  bool Copy(const VarDctEncoderFrame& frame) {
    geometry = frame.geometry();
    strategies = frame.strategies();
    quantizer = frame.quantizer();
    correlation = frame.color_correlation();
    profile = frame.profile();
    raw.Copy(frame.raw_quant_field());
    sharpness.Copy(frame.epf_sharpness());
    for (size_t c = 0; c < 3; ++c) {
      quantized_dc[c].Copy(frame.quantized_dc().plane[c]);
      dc[c].Copy(frame.dc().plane[c]);
    }
    group_extent = frame.ac_group_extent();
    for (size_t i = 0; i < frame.ac_group_count(); ++i) {
      VarDctAcGroupView group;
      CHECK(frame.GetAcGroup(i, &group).ok());
      used.push_back(group.used_coefficient_count);
      for (auto channel : group.coefficients) {
        ac.insert(ac.end(), channel.begin(), channel.end());
      }
    }
    return true;
  }

  VarDctFrameViewData data() const {
    return {
        .input =
            {
                .geometry = geometry,
                .strategies = &strategies,
                .raw_quant_field = raw.view(),
                .quantizer = &quantizer,
                .color_correlation = &correlation,
                .epf_sharpness = sharpness.view(),
            },
        .profile = profile,
        .quantized_dc = {{quantized_dc[0].view(), quantized_dc[1].view(),
                          quantized_dc[2].view()}},
        .dc = {{dc[0].view(), dc[1].view(), dc[2].view()}},
        .ac_group_extent = group_extent,
        .group_used_coefficient_count = used,
        .ac_coefficients = ac,
    };
  }
};

bool MakeFrame(Extent2D extent, VarDctEncoderFrame* frame) {
  FrameGeometry geometry;
  CHECK(FrameGeometry::Create(extent, &geometry).ok());
  const Extent2D pixels = geometry.padded_frame();
  const Extent2D blocks = geometry.block_grid().blocks;
  std::array<std::vector<float>, 3> signal;
  ConstImage3FView opsin;
  for (size_t c = 0; c < 3; ++c) {
    signal[c].resize(pixels.width * pixels.height);
    for (size_t y = 0; y < pixels.height; ++y) {
      for (size_t x = 0; x < pixels.width; ++x) {
        signal[c][y * pixels.width + x] =
            0.1f * std::sin(static_cast<float>(x + 3 * y + c) * 0.13f) +
            static_cast<float>((x * 17 + y * 11 + c) % 97) * 0.002f;
      }
    }
    opsin.plane[c] = {signal[c].data(), pixels, pixels.width};
  }
  AcStrategyGrid strategies;
  CHECK(AcStrategyGrid::Create(blocks, &strategies).ok());
  if (blocks.width >= 8 && blocks.height >= 8) {
    CHECK(strategies.Set(0, 0, AcStrategyType::kDct32x32).ok());
    CHECK(strategies.Set(4, 0, AcStrategyType::kDct16x16).ok());
    CHECK(strategies.Set(6, 0, AcStrategyType::kDct16x8).ok());
    CHECK(strategies.Set(0, 4, AcStrategyType::kDct8x16).ok());
    CHECK(strategies.Set(2, 4, AcStrategyType::kDct32x16).ok());
    CHECK(strategies.Set(4, 4, AcStrategyType::kDct16x32).ok());
  }
  strategies.fill_empty_dct8();
  Quantizer quantizer;
  ColorCorrelationMap correlation;
  CHECK(Quantizer::Create({3541, 10}, &quantizer).ok());
  CHECK(ComputeInitialColorCorrelationMap(opsin, &correlation).ok());
  std::vector<int32_t> raw(blocks.width * blocks.height);
  std::vector<uint8_t> sharpness(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = 16 + static_cast<int32_t>(i % 40);
    sharpness[i] = static_cast<uint8_t>(i % 8);
  }
  CHECK(ComputeQuantizedCoefficients(
            opsin,
            {
                .geometry = geometry,
                .strategies = &strategies,
                .raw_quant_field = {raw.data(), blocks, blocks.width},
                .quantizer = &quantizer,
                .color_correlation = &correlation,
                .epf_sharpness = {sharpness.data(), blocks, blocks.width},
            },
            {}, frame)
            .ok());
  return true;
}

bool CheckConsumers(const VarDctEncoderFrame& owned,
                    const VarDctFrameView& external) {
  SimpleCoefficientOrders orders, view_orders;
  CHECK(ComputeSimpleCoefficientOrders(owned, &orders).ok());
  CHECK(codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(
            external, VarDctCoefficientOrderBehavior::kFull, &view_orders)
            .ok());
  CHECK(orders == view_orders);
  SimpleBlockContextMap map, view_map;
  CHECK(ComputeSimpleBlockContextMap(owned, &map).ok());
  CHECK(codestream_internal::ComputeSimpleBlockContextMapForEncoder(external,
                                                                    &view_map)
            .ok());
  CHECK(map == view_map);
  std::vector<SimpleBlockContextMap> maps, view_maps;
  CHECK(ComputeSimpleBlockContextMapCandidates(owned, &maps).ok());
  CHECK(codestream_internal::ComputeSimpleBlockContextMapCandidatesForEncoder(
            external, &view_maps)
            .ok());
  CHECK(maps == view_maps);
  std::vector<SimpleDcGroupTokenStreams> dc, view_dc;
  CHECK(TokenizeSimpleDcGroups(owned, &dc).ok());
  CHECK(
      codestream_internal::TokenizeSimpleDcGroupsForEncoder(external, &view_dc)
          .ok());
  CHECK(dc == view_dc);
  std::vector<SimpleAcGroupTokenTemplate> ac, view_ac;
  CHECK(BuildSimpleAcGroupTokenTemplates(owned, orders, &ac).ok());
  CHECK(codestream_internal::BuildSimpleAcGroupTokenTemplatesForEncoder(
            external, orders, &view_ac)
            .ok());
  CHECK(ac == view_ac);
  return true;
}

bool CheckParity(Extent2D extent) {
  VarDctEncoderFrame owned;
  CHECK(MakeFrame(extent, &owned));
  ExternalFrameStorage storage;
  CHECK(storage.Copy(owned));
  const VarDctFrameView borrowed = BorrowFrame(owned);
  const VarDctFrameView external(storage.data());
  CHECK(owned.valid() && borrowed.valid() && external.valid());
  CHECK(CheckConsumers(owned, external));
  for (size_t i = 0; i < owned.ac_group_count(); ++i) {
    VarDctAcGroupView a, b, c;
    CHECK(owned.GetAcGroup(i, &a).ok() && borrowed.GetAcGroup(i, &b).ok());
    CHECK(external.GetAcGroup(i, &c).ok());
    CHECK(a.block_x == c.block_x && a.block_y == c.block_y);
    CHECK(a.block_extent == c.block_extent);
    CHECK(a.used_coefficient_count == c.used_coefficient_count);
    for (size_t channel = 0; channel < 3; ++channel) {
      CHECK(a.coefficients[channel].data() == b.coefficients[channel].data());
      CHECK(a.coefficients[channel].data() != c.coefficients[channel].data());
      CHECK(
          std::ranges::equal(a.coefficients[channel], c.coefficients[channel]));
    }
  }
  std::vector<uint8_t> retained;
  for (auto entropy :
       {VarDctEntropyBehavior::kBalanced, VarDctEntropyBehavior::kHighDensity,
        VarDctEntropyBehavior::kMaximumCompression}) {
    for (auto order : {VarDctCoefficientOrderBehavior::kFull,
                       VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled}) {
      const VarDctCodestreamOptions options{entropy, order};
      std::vector<uint8_t> expected;
      {
        thread_budget_internal::EncodeScope scope(1);
        CHECK(EncodeVarDctCodestream(owned, options, &expected).ok());
      }
      for (size_t threads : {size_t{1}, size_t{8}}) {
        thread_budget_internal::EncodeScope scope(threads);
        std::vector<uint8_t> a, b;
        codestream_internal::VarDctCodestreamProfile profile;
        CHECK(codestream_internal::EncodeVarDctCodestreamFromView(borrowed,
                                                                  options, &a)
                  .ok());
        CHECK(codestream_internal::EncodeVarDctCodestreamFromView(
                  external, options, &b, &profile)
                  .ok());
        CHECK(a == expected && b == expected);
        CHECK(profile.entropy_behavior == entropy);
      }
      if (entropy == VarDctEntropyBehavior::kBalanced &&
          order == VarDctCoefficientOrderBehavior::kFull)
        retained = expected;
    }
  }
  // No owning encoder frame survives this point. Borrowing external storage
  // still works, including two simultaneous serializer calls and their workers.
  owned = {};
  const auto encode = [&external, &retained] {
    thread_budget_internal::EncodeScope scope(8);
    std::vector<uint8_t> result;
    return codestream_internal::EncodeVarDctCodestreamFromView(external, {},
                                                               &result)
               .ok() &&
           result == retained;
  };
  auto first = std::async(std::launch::async, encode);
  auto second = std::async(std::launch::async, encode);
  CHECK(first.get() && second.get());
  return true;
}

bool CheckRejected(const VarDctFrameView& frame,
                   bool structurally_valid = false) {
  CHECK(frame.valid() == structurally_valid);
  const std::vector<uint8_t> sentinel{1, 2, 3};
  std::vector<uint8_t> output = sentinel;
  codestream_internal::VarDctCodestreamProfile profile;
  profile.total_nanoseconds = 123;
  const auto original = profile;
  CHECK(!codestream_internal::EncodeVarDctCodestreamFromView(frame, {}, &output,
                                                             &profile)
             .ok());
  CHECK(output == sentinel && profile == original);
  return true;
}

bool CheckInvalid() {
  CHECK(CheckRejected({}));
  VarDctEncoderFrame owned;
  CHECK(CheckRejected(BorrowFrame(owned)));
  CHECK(MakeFrame({273, 265}, &owned));
  ExternalFrameStorage storage;
  CHECK(storage.Copy(owned));
  const auto good = storage.data();
  const auto reject = [](const VarDctFrameViewData& data) {
    return CheckRejected(VarDctFrameView(data));
  };
  auto bad = good;
  bad.input.strategies = nullptr;
  CHECK(reject(bad));
  bad = good;
  bad.input.quantizer = nullptr;
  CHECK(reject(bad));
  bad = good;
  bad.input.color_correlation = nullptr;
  CHECK(reject(bad));
  bad = good;
  bad.input.raw_quant_field.extent.width += 1;
  CHECK(reject(bad));
  bad = good;
  bad.input.epf_sharpness.stride = 0;
  CHECK(reject(bad));
  bad = good;
  bad.quantized_dc.plane[2].data = nullptr;
  CHECK(reject(bad));
  bad = good;
  bad.dc.plane[0].stride = std::numeric_limits<size_t>::max();
  CHECK(reject(bad));
  bad = good;
  bad.ac_group_extent = {};
  CHECK(reject(bad));
  bad = good;
  bad.ac_coefficients = bad.ac_coefficients.first(1);
  CHECK(reject(bad));
  bad = good;
  bad.group_used_coefficient_count = {};
  CHECK(reject(bad));
  const int32_t raw = storage.raw.values[0];
  storage.raw.values[0] = 0;
  CHECK(reject(good));
  storage.raw.values[0] = raw;
  storage.sharpness.values[0] = 8;
  CHECK(reject(good));
  storage.sharpness.values[0] = 0;
  const float dc = storage.dc[2].values[0];
  storage.dc[2].values[0] = std::numeric_limits<float>::quiet_NaN();
  CHECK(reject(good));
  storage.dc[2].values[0] = dc + 1.0f;
  CHECK(reject(good));
  storage.dc[2].values[0] = dc;
  --storage.used[0];
  CHECK(reject(good));
  ++storage.used[0];
  storage.ac.back() = 1;
  CHECK(reject(good));
  storage.ac.back() = 0;
  for (size_t group = 0; group < storage.used.size(); ++group) {
    if (storage.used[group] == kVarDctAcGroupCoefficientCapacity) continue;
    for (size_t channel = 0; channel < 3; ++channel) {
      const size_t base =
          (group * 3 + channel) * kVarDctAcGroupCoefficientCapacity;
      for (size_t index :
           {storage.used[group], kVarDctAcGroupCoefficientCapacity - 1}) {
        storage.ac[base + index] = std::numeric_limits<int32_t>::min();
        CHECK(reject(good));
        storage.ac[base + index] = 0;
      }
    }
  }
  bad = good;
  bad.profile.quantization_matrix_mode = QuantizationMatrixMode::kCustom;
  CHECK(CheckRejected(VarDctFrameView(bad), true));
  // A complete strategy grid still must not cross the physical AC-group seam.
  const AcStrategyGrid original_strategies = storage.strategies;
  storage.strategies.clear();
  CHECK(storage.strategies.Set(31, 0, AcStrategyType::kDct16x16).ok());
  storage.strategies.fill_empty_dct8();
  CHECK(reject(good));
  storage.strategies = original_strategies;
  CHECK(VarDctFrameView(good).valid());
  VarDctAcGroupView group;
  group.block_x = 987;
  CHECK(!VarDctFrameView(good).GetAcGroup(storage.used.size(), &group).ok());
  CHECK(group.block_x == 987);
  CHECK(!VarDctFrameView{}.GetAcGroup(0, &group).ok() && group.block_x == 987);
  CHECK(!VarDctFrameView(good).GetAcGroup(0, nullptr).ok());
  std::vector<uint8_t> output{42};
  CHECK(!codestream_internal::EncodeVarDctCodestreamFromView(
             VarDctFrameView(good), {}, nullptr)
             .ok());
  CHECK(!codestream_internal::EncodeVarDctCodestreamFromView(
             VarDctFrameView(good), {static_cast<VarDctEntropyBehavior>(255)},
             &output)
             .ok());
  CHECK(output == std::vector<uint8_t>{42});
  return true;
}
}  // namespace

int main() {
  if (!CheckInvalid()) return EXIT_FAILURE;
  // Tiny, padded edges, mixed strategies, multiple AC groups, and multiple DC
  // groups. The last case also exercises quant-field context splitting.
  for (Extent2D extent : {Extent2D{1, 1}, {19, 17}, {273, 265}, {2057, 257}}) {
    if (!CheckParity(extent)) return EXIT_FAILURE;
  }
  std::cout << "Borrowed frame validation, consumer, lifetime, and byte parity "
               "passed\n";
  return EXIT_SUCCESS;
}
