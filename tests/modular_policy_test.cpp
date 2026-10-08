// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/workflow.h"
#include "codestream/modular/search.h"
#include "codec/modular/transform/rct.h"
#include "modular_reference.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include "lib/jxl/pack_signed.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
void Check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
struct Fixture {
  ref::IntegerImage source;
  std::vector<uint8_t> bytes;
  PackedModularFormat format;
  Fixture(size_t w, size_t h, uint32_t channels, uint32_t bits, unsigned pattern)
      : source{w, h, channels, bits, {}},
        format(channels == 1
                   ? (bits == 8 ? PackedModularFormat::kGray8 : PackedModularFormat::kGray16)
               : channels == 3
                   ? (bits == 8 ? PackedModularFormat::kRgb8 : PackedModularFormat::kRgb16)
                   : (bits == 8 ? PackedModularFormat::kRgba8 : PackedModularFormat::kRgba16)) {
    uint32_t rng = 1234567, mask = bits == 8 ? 255 : 65535;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        for (size_t c = 0; c < channels; ++c) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          uint16_t sample = static_cast<uint16_t>((pattern == 0   ? rng
                                                   : pattern == 1 ? (x * 127 + y * 31 + c * 67)
                                                   : pattern == 2 ? ((x + y + c) % 2 ? mask : 0)
                                                                  : 96) &
                                                  mask);
          if (channels == 4 && c == 3 && x % 3 == 0)
            sample = 0;
          source.samples.push_back(sample);
          bytes.push_back(static_cast<uint8_t>(sample));
          if (bits == 16)
            bytes.push_back(static_cast<uint8_t>(sample >> 8));
        }
  }
  PackedModularImageView view() const {
    return {bytes,
            {source.width, source.height},
            source.width * source.channels * (source.bits / 8),
            format};
  }
};
jxl::Image ReferenceImage(const Fixture &f, uint8_t rct, JxlMemoryManager *memory) {
  auto created =
      jxl::Image::Create(memory, f.source.width, f.source.height, f.source.bits, f.source.channels);
  Check(created.ok(), "Reference allocation failed");
  auto image = std::move(created).value_();
  for (size_t c = 0; c < f.source.channels; ++c)
    for (size_t y = 0; y < f.source.height; ++y)
      for (size_t x = 0; x < f.source.width; ++x)
        image.channel[c].Row(y)[x] =
            f.source.samples[(y * f.source.width + x) * f.source.channels + c];
  if (rct) {
    auto next = jxl::Image::Create(memory, f.source.width, f.source.height, f.source.bits,
                                   f.source.channels);
    Check(next.ok(), "Reference RCT allocation failed");
    auto transformed = std::move(next).value_();
    Check(bool(jxl::FwdRct(
              {&image.channel[0], &image.channel[1], &image.channel[2]},
              {&transformed.channel[0], &transformed.channel[1], &transformed.channel[2]}, rct,
              nullptr)),
          "Reference RCT failed");
    if (f.source.channels == 4)
      transformed.channel[3] = std::move(image.channel[3]);
    image = std::move(transformed);
  }
  return image;
}
void Stages(const Fixture &f, const ModularCodingPolicy &policy) {
  JxlMemoryManager memory;
  Check(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Reference memory manager failed");
  auto reference = ReferenceImage(f, policy.rct, &memory);
  ModularEncoderFrame frame;
  Ok(ModularEncoderFrame::Prepare(f.view(), policy.rct, &frame));
  ModularInputProfile profile;
  Ok(ResolveModularInput(f.view().extent, f.format, &profile));
  ModularFrameGeometry geometry;
  Ok(ModularFrameGeometry::Create(f.view().extent, &geometry));
  ModularStreamPlan plan;
  Ok(BuildModularStreamPlan(geometry, profile.channels(), 0, &plan));
  PreparedModularTokens tokens;
  Ok(TokenizeModular(frame, plan, policy, &tokens));
  jxl::Tree tree, decoded;
  for (size_t i = 0; i < policy.tree.size; ++i) {
    const auto &n = policy.tree.nodes[i];
    tree.emplace_back(n.property, n.split, n.left, n.right,
                      static_cast<jxl::Predictor>(n.predictor), n.offset, n.multiplier);
  }
  std::vector<jxl::Token> tree_tokens;
  Check(bool(jxl::TokenizeTree(tree, &tree_tokens, &decoded)),
        "Reference tree tokenization failed");
  Check(tree_tokens.size() == tokens.tree_token_count, "Tree token count differs");
  for (size_t i = 0; i < tree_tokens.size(); ++i)
    Check(tree_tokens[i].context == tokens.tree_tokens[i].context &&
              tree_tokens[i].value == tokens.tree_tokens[i].value,
          "Tree tokens differ");
  for (size_t c = 0; c < f.source.channels; ++c)
    for (size_t y = 0; y < f.source.height; ++y)
      for (size_t x = 0; x < f.source.width; ++x)
        Check(frame.image().view(c).Row(y)[x] == reference.channel[c].Row(y)[x],
              "Forward RCT differs");
  if (policy.rct)
    for (size_t y = 0; y < f.source.height; ++y)
      for (size_t x = 0; x < f.source.width; ++x) {
        const auto rgb = InverseRct({reference.channel[0].Row(y)[x], reference.channel[1].Row(y)[x],
                                     reference.channel[2].Row(y)[x]},
                                    policy.rct);
        for (size_t c = 0; c < 3; ++c)
          Check(rgb[c] == f.source.samples[(y * f.source.width + x) * f.source.channels + c],
                "Inverse RCT differs");
      }
  jxl::weighted::Header header;
  const auto &p = policy.weighted.coefficients;
  header.p1C = p[0];
  header.p2C = p[1];
  header.p3Ca = p[2];
  header.p3Cb = p[3];
  header.p3Cc = p[4];
  header.p3Cd = p[5];
  header.p3Ce = p[6];
  for (size_t i = 0; i < 4; ++i)
    header.w[i] = policy.weighted.weights[i];
  for (size_t si = 0; si < plan.streams.size(); ++si) {
    const auto &stream = plan.streams[si];
    size_t token = 0;
    for (const auto &slice :
         std::span(plan.slices).subspan(stream.slice_begin, stream.slice_count)) {
      ModularChannelView native;
      Ok(BorrowChannelSlice(frame.image().view(slice.channel), slice.rect, &native));
      const auto [w, h] = native.descriptor.extent;
      // Derive tile origin from stream identity, independently of native rectangles.
      const size_t groups_x = (f.source.width + 255) / 256;
      const size_t x0 = stream.role == StreamRole::kGlobal ? 0 : (stream.group % groups_x) * 256;
      const size_t y0 = stream.role == StreamRole::kGlobal ? 0 : (stream.group / groups_x) * 256;
      auto tile_result = jxl::Channel::Create(&memory, w, h);
      auto refs_result = jxl::Channel::Create(&memory, 0, w);
      Check(tile_result.ok() && refs_result.ok(), "Reference tile allocation failed");
      auto tile = std::move(tile_result).value_(), refs = std::move(refs_result).value_();
      for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x)
          tile.Row(y)[x] = reference.channel[slice.channel].Row(y0 + y)[x0 + x];
      jxl::weighted::State state(header, w, h);
      WeightedPredictor<resource_budget_internal::ResourceClass::kPreparation> wp(w,
                                                                                  policy.weighted);
      jxl::Properties properties(16);
      for (size_t y = 0; y < h; ++y) {
        jxl::InitPropsRow(&properties,
                          {static_cast<int32_t>(slice.channel), static_cast<int32_t>(stream.id)},
                          static_cast<int>(y));
        int64_t last_gradient = 0;
        for (size_t x = 0; x < w; ++x) {
          std::array<int64_t, 14> predictions{};
          jxl::PredictLearnAll(&properties, w, tile.Row(y) + x, tile.plane.PixelsPerRow(),
                               static_cast<int>(x), static_cast<int>(y), refs, &state,
                               predictions.data());
          const auto n = Neighbors(native, x, y);
          const auto weighted = wp.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
          const auto actual =
              Properties(n, slice.channel, stream.id, x, y, last_gradient, weighted.second);
          last_gradient = actual[9];
          for (size_t i = 0; i < 16; ++i)
            Check(actual[i] == properties[i], "Predictor property/state differs");
          for (size_t i = 0; i < 14; ++i)
            Check(Predict(static_cast<Predictor>(i), n, weighted.first) == predictions[i],
                  "Scalar prediction differs");
          size_t leaf = 0;
          while (decoded[leaf].property != -1) {
            const auto &node = decoded[leaf];
            leaf = properties[node.property] > node.splitval ? node.lchild : node.rchild;
          }
          const auto &node = decoded[leaf];
          const int64_t residual = tile.Row(y)[x] -
                                   predictions[static_cast<size_t>(node.predictor)] -
                                   node.predictor_offset;
          Check(residual % node.multiplier == 0, "Reference leaf cannot represent input");
          const auto expected = jxl::PackSigned(static_cast<int32_t>(residual / node.multiplier));
          Check(tokens.streams[si][token].value == expected &&
                    tokens.streams[si][token].context == node.lchild,
                "Prescribed token differs");
          ++token;
          state.UpdateErrors(tile.Row(y)[x], x, y, w);
          Check(wp.Update(native.Row(y)[x], x, y), "Native weighted range failed");
        }
      }
    }
    Check(token == tokens.streams[si].size(), "Prescribed token coverage differs");
  }
}
size_t files = 0, stages = 0;
std::filesystem::path artifacts;
void Case(const Fixture &f, const ModularCodingPolicy &policy, bool search = false) {
  if (!search) {
    Stages(f, policy);
    ++stages;
  }
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularWorkflowStoragePlan plan;
    Ok(search ? ComputeModularSearchStoragePlan(f.view().extent, f.format, mode, &plan)
              : ComputeModularWorkflowStoragePlan(f.view().extent, f.format, mode, policy, &plan));
    std::shared_ptr<const ExecutionDomain> domain;
    Ok(ExecutionDomain::Create(
        {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 1}, &domain));
    std::vector<uint8_t> output, again;
    const auto start = std::chrono::steady_clock::now();
    Ok(EncodeModularImage(f.view(), {domain, mode, policy, search}, &output));
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
    try {
      Check(ref::DecodeLossless(output) == f.source,
            "Prescribed/search file reconstruction differs");
    } catch (...) {
      std::cerr << "decode rct " << unsigned(policy.rct) << " predictor "
                << unsigned(policy.tree.nodes[0].predictor) << " bits " << f.source.bits
                << " channels " << f.source.channels << " mode " << unsigned(mode) << '\n';
      throw;
    }
    Check(domain->snapshot().live_capacity_bytes == 0 &&
              domain->snapshot().reserved_unbacked_bytes == 0,
          "Policy workflow leaked managed backing");
    if (f.view().extent.width > 256 || f.view().extent.height > 256) {
      ModularWorkflowStoragePlan parallel_plan;
      Ok(search
             ? ComputeModularSearchStoragePlan(f.view().extent, f.format, mode, &parallel_plan, 4)
             : ComputeModularWorkflowStoragePlan(f.view().extent, f.format, mode, policy,
                                                 &parallel_plan, 4));
      std::shared_ptr<const ExecutionDomain> parallel_domain;
      Ok(ExecutionDomain::Create(
          {.managed_memory_bytes = parallel_plan.working.peak_bytes, .cpu_participant_limit = 4},
          &parallel_domain));
      Ok(EncodeModularImage(f.view(), {parallel_domain, mode, policy, search, 4}, &again));
      Check(output == again, "Prescribed serial/parallel policy differs");
    }
    if (search) {
      Ok(EncodeModularImage(f.view(), {domain, mode, policy, search}, &again));
      Check(output == again, "Search is nondeterministic");
      const auto baseline_start = std::chrono::steady_clock::now();
      Ok(EncodeModularImage(f.view(), {.entropy = mode}, &again));
      const auto baseline_micros = std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::steady_clock::now() - baseline_start)
                                       .count();
      Check(output.size() <= again.size(), "Search lost to the complete baseline");
      std::cout << "search " << f.source.width << 'x' << f.source.height << " channels "
                << f.source.channels << " bits " << f.source.bits << " coder "
                << static_cast<int>(mode) << " baseline " << again.size() << " selected "
                << output.size() << " baseline_us " << baseline_micros << " search_us " << micros
                << '\n';
    }
    if (!artifacts.empty()) {
      std::ofstream file(artifacts / ("case-" + std::to_string(files) + ".jxl"), std::ios::binary);
      file.write(reinterpret_cast<const char *>(output.data()),
                 static_cast<std::streamsize>(output.size()));
      Check(bool(file), "Cannot retain policy fixture");
    }
    ++files;
  }
}
} // namespace
int main(int argc, char **argv) try {
  if (argc > 1) {
    artifacts = argv[1];
    std::filesystem::create_directories(artifacts);
  }
  for (auto channels : {1u, 3u, 4u})
    for (auto bits : {8u, 16u}) {
      for (unsigned predictor = 0; predictor < 14; ++predictor) {
        ModularCodingPolicy policy;
        policy.tree.nodes[0].predictor = static_cast<Predictor>(predictor);
        for (auto extent :
             {Extent2D{1, 1}, {1, 17}, {17, 1}, {17, 19}, {257, 3}, {3, 257}, {2049, 1}})
          Case(Fixture(extent.width, extent.height, channels, bits, predictor % 3), policy);
      }
      ModularCodingPolicy custom;
      custom.tree.nodes[0].predictor = Predictor::kWeighted;
      custom.weighted.coefficients = {17, 9, 6, 8, 7, 1, 2};
      custom.weighted.weights = {15, 0, 9, 11};
      Case(Fixture(257, 3, channels, bits, 0), custom);
      custom.weighted.coefficients.fill(0);
      custom.weighted.weights.fill(0);
      Case(Fixture(17, 19, channels, bits, 0), custom);
      custom.weighted.coefficients.fill(31);
      custom.weighted.weights.fill(15);
      Case(Fixture(17, 19, channels, bits, 2), custom);
      for (int property = 0; property < 16; ++property) {
        ModularCodingPolicy policy;
        policy.tree.size = 3;
        policy.tree.nodes[0] = {.property = property, .split = 0, .left = 2, .right = 1};
        policy.tree.nodes[1].predictor = Predictor::kWeighted;
        policy.tree.nodes[2].predictor = Predictor::kAverage4;
        Case(Fixture(257, 3, channels, bits, 0), policy);
      }
      ModularCodingPolicy scaled;
      scaled.tree.nodes[0].predictor = Predictor::kZero;
      scaled.tree.nodes[0].offset = 6;
      scaled.tree.nodes[0].multiplier = 3;
      Case(Fixture(17, 19, channels, bits, 3), scaled);
      for (unsigned pattern = 0; pattern < 4; ++pattern)
        Case(Fixture(257, 19, channels, bits, pattern), {}, true);
    }
  for (uint8_t rct = 1; rct < 42; ++rct)
    for (auto bits : {8u, 16u}) {
      ModularCodingPolicy policy;
      policy.rct = rct;
      policy.tree.nodes[0].predictor = rct % 2 ? Predictor::kWeighted : Predictor::kAverage4;
      Case(Fixture(257, 3, 4, bits, rct % 3), policy);
    }
  ModularCodingPolicy maximum;
  maximum.tree.size = 31;
  for (size_t i = 0; i < 15; ++i)
    maximum.tree.nodes[i] = {.property = static_cast<int32_t>(i),
                             .split = static_cast<int32_t>(i * 7),
                             .left = static_cast<uint8_t>(2 * i + 1),
                             .right = static_cast<uint8_t>(2 * i + 2)};
  Case(Fixture(257, 3, 4, 16, 0), maximum);
  for (auto bits : {8u, 16u}) {
    ModularCodingPolicy extreme;
    extreme.tree.nodes[0].predictor = Predictor::kZero;
    extreme.tree.nodes[0].offset = INT32_MAX;
    Case(Fixture(17, 19, 3, bits, 3), extreme);
    extreme.tree.nodes[0].offset = INT32_MIN;
    extreme.tree.nodes[0].multiplier = 2;
    Case(Fixture(17, 19, 3, bits, 3), extreme);
    extreme.tree.nodes[0].offset = 96;
    extreme.tree.nodes[0].multiplier = INT32_MAX;
    Case(Fixture(17, 19, 3, bits, 3), extreme);
    extreme.tree.nodes[0].multiplier = uint32_t{1} << 30;
    Case(Fixture(17, 19, 3, bits, 3), extreme);
  }
  std::cout << stages << " prescribed stage cases; " << files << " complete policy files passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
