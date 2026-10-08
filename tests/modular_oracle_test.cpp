// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include "modular_reference.h"
#include "codec/modular/prediction.h"
#include "codestream/headers_internal.h"
#include "codestream/modular/stream_plan.h"
#include "codestream/modular/tree_codec.h"
#include "codestream/sections.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/toc.h"

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}

void HeadersAndSections() {
  using namespace codec_internal;
  using namespace codestream_internal;
  for (Extent2D extent :
       {Extent2D{1, 1}, {255, 257}, {256, 256}, {2049, 257}, {0x3fffffffu, 1}})
    for (uint8_t bits : {8, 16})
      for (unsigned channels : {1, 3, 4}) {
        ImageMetadata image{extent, SampleFormat::kUnsigned, bits,
                            channels == 1 ? SourceColor::kGraySrgb
                                          : SourceColor::kSrgb};
        if (channels == 4)
          image.alpha = AlphaMetadata{bits};
        for (size_t offset = 0; offset < 8; ++offset) {
          BitWriter writer;
          Ok(writer.WriteBits(offset, 0));
          Ok(WriteImageHeader(image, &writer));
          Ok(WriteFrameHeader(image, {}, &writer));
          const auto h = ref::InspectHeaders(writer.padded_bytes(), offset);
          Check(h.width == extent.width && h.height == extent.height &&
                    h.bits == bits && h.color_channels == (channels == 1 ? 1 : 3) &&
                    h.extra_channels == (channels == 4 ? 1 : 0) &&
                    h.alpha_bits == (channels == 4 ? bits : 0) && !h.alpha_associated &&
                    h.modular && !h.xyb && !h.floating_point && !h.gaborish &&
                    !h.epf_iterations && h.source_srgb && h.final_frame &&
                    h.passes == 1 && h.upsampling == 1 && h.group_dimension == 256 &&
                    h.bits_consumed == writer.bits_written(),
                "Native metadata differs from pinned header parser");
          if (offset || extent.width > 4096)
            continue;
          // Payload bytes need not be image data to independently check TOC
          // placement, empty sections, byte padding and single-group collapse.
          std::vector<BitWriter> sections(h.sections);
          for (size_t i = 0; i < sections.size(); ++i)
            if (i % 3 != 1)
              Ok(sections[i].WriteBits(i % 17 + 1, 1));
          Ok(WriteTocAndSections(sections, &writer));
          const auto sizes = ref::ReadSectionSizes(writer.padded_bytes());
          Check(sizes.size() == sections.size(), "TOC section count changed");
          for (size_t i = 0; i < sizes.size(); ++i)
            Check(sizes[i] == sections[i].padded_size(),
                  "TOC section size differs from pinned parser");
        }
      }
  for (Extent2D extent : {Extent2D{1, 1}, {256, 257}, {2049, 2049}}) {
    ModularFrameGeometry g;
    Ok(ModularFrameGeometry::Create(extent, &g));
    jxl::FrameDimensions dim;
    dim.Set(extent.width, extent.height, 1, 0, 0, true, 1);
    Check(dim.num_groups == g.group_count() &&
              dim.num_dc_groups == g.dc_group_count() &&
              jxl::NumTocEntries(dim.num_groups, dim.num_dc_groups, 1) ==
                  g.section_count(),
          "Geometry differs from reference");
    for (size_t i = 0; i < g.group_count(); ++i) {
      size_t id;
      Ok(ModularStreamId(g, StreamRole::kGroup, i, &id));
      Check(id == jxl::ModularStreamId::ModularAC(i, 0).ID(dim),
            "Group stream ID mismatch");
    }
    for (size_t i = 0; i < g.dc_group_count(); ++i) {
      size_t id;
      Ok(ModularStreamId(g, StreamRole::kDcGroup, i, &id));
      Check(id == jxl::ModularStreamId::ModularDC(i).ID(dim), "DC stream ID mismatch");
    }
  }
  // Shift/clipping arithmetic is checked with the pinned Rect implementation.
  ModularFrameGeometry g;
  Ok(ModularFrameGeometry::Create({4097, 4097}, &g));
  const std::array channels = {
      ChannelDescriptor{{4, 1}, 0, 0, ChannelRole::kMetadata},
      ChannelDescriptor{{129, 129}, 5, 5}, ChannelDescriptor{{513, 513}, 3, 3},
      ChannelDescriptor{{2049, 2048}, 1, 1}, ChannelDescriptor{{1, 1}}};
  ModularStreamPlan p;
  Ok(BuildModularStreamPlan(g, channels, 1, &p));
  for (const auto &stream : p.streams) {
    if (stream.role == StreamRole::kGlobal)
      continue;
    const bool dc = stream.role == StreamRole::kDcGroup;
    const auto grid = dc ? g.dc_groups() : g.groups();
    const size_t dimension = dc ? 2048 : 256;
    const jxl::Rect rect((stream.group % grid.width) * dimension,
                         (stream.group / grid.width) * dimension, dimension, dimension);
    size_t index = stream.slice_begin;
    for (size_t c = 2; c < channels.size(); ++c) {
      const auto &ch = channels[c];
      if ((std::min(ch.hshift, ch.vshift) >= 3) != dc)
        continue;
      const jxl::Rect clipped(rect.x0() >> ch.hshift, rect.y0() >> ch.vshift,
                              rect.xsize() >> ch.hshift, rect.ysize() >> ch.vshift,
                              ch.extent.width, ch.extent.height);
      if (clipped.xsize() == 0 || clipped.ysize() == 0)
        continue;
      Check(index < stream.slice_begin + stream.slice_count,
            "Missing shifted reference slice");
      const auto &actual = p.slices[index++];
      Check(actual.channel == c &&
                actual.rect == ModularRect{clipped.x0(),
                                           clipped.y0(),
                                           {clipped.xsize(), clipped.ysize()}},
            "Shifted slice differs from pinned clipping");
    }
    Check(index == stream.slice_begin + stream.slice_count,
          "Extra grouped channel slice");
  }
}

void IntegerHarness() {
  size_t cases = 0;
  for (uint32_t bits : {8, 16})
    for (uint32_t channels : {1, 3, 4})
      for (Extent2D extent : {Extent2D{1, 1}, {17, 19}, {257, 3}, {2049, 1}}) {
        ref::IntegerImage image{extent.width, extent.height, channels, bits, {}};
        const uint32_t maximum = bits == 8 ? 255 : 65535;
        image.samples.resize(extent.width * extent.height * channels);
        for (size_t i = 0; i < image.samples.size(); ++i)
          image.samples[i] =
              static_cast<uint16_t>((i * 257 + (i % 3 ? maximum : 0)) & maximum);
        if (channels == 4)
          for (size_t p = 0; p < extent.width * extent.height; ++p)
            image.samples[4 * p + 3] =
                p % 3 == 0 ? 0 : (p % 3 == 1 ? maximum : maximum / 2);
        const auto encoded = ref::EncodeLossless(image);
        Check(ref::DecodeLossless(encoded) == image,
              "Integer reference roundtrip changed source samples");
        bool rejected = false;
        try {
          (void)ref::DecodeLossless(std::span(encoded).first(encoded.size() / 2));
        } catch (const std::runtime_error &) {
          rejected = true;
        }
        Check(rejected, "Truncated reference file accepted");
        ++cases;
      }
  std::cout << cases << " exact integer reference roundtrips passed\n";
}

void StageAdapters() {
  JxlMemoryManager manager;
  Check(jxl::MemoryManagerInit(&manager, nullptr), "Reference memory manager failed");
  for (Extent2D extent : {Extent2D{1, 1}, {1, 17}, {17, 1}, {17, 19}}) {
    auto created = jxl::Image::Create(&manager, extent.width, extent.height, 16, 3);
    Check(created.ok(), "Reference channel allocation failed");
    auto image = std::move(created).value_();
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x)
          image.channel[c].Row(y)[x] =
              (x + y + c) % 5 == 0 ? 65535
              : (x + y + c) % 5 == 1
                  ? 0
                  : static_cast<int32_t>((x * 371 + y * 257 + c * 31) & 65535);
    jxl::weighted::Header header;
    jxl::weighted::PredictorMode(0, &header);
    for (const auto &ch : image.channel) {
      WeightedPredictor<resource_budget_internal::ResourceClass::kPreparation> native(
          extent.width);
      jxl::weighted::State reference(header, extent.width, extent.height);
      jxl::Properties props(1);
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x) {
          const int64_t w = x ? ch.Row(y)[x - 1] : (y ? ch.Row(y - 1)[x] : 0);
          const int64_t n = y ? ch.Row(y - 1)[x] : w;
          const int64_t nw = x && y ? ch.Row(y - 1)[x - 1] : w;
          const int64_t ne = y && x + 1 < extent.width ? ch.Row(y - 1)[x + 1] : n;
          const int64_t nn = y > 1 ? ch.Row(y - 2)[x] : n;
          const auto expected =
              reference.Predict<true>(x, y, extent.width, n, w, ne, nw, nn, &props, 0);
          const auto actual = native.Predict(x, y, n, w, ne, nw, nn);
          Check(actual.first == expected && actual.second == props[0],
                "Weighted prediction/property mismatch");
          Check(native.Update(ch.Row(y)[x], x, y), "Native weighted state overflow");
          reference.UpdateErrors(ch.Row(y)[x], x, y, extent.width);
        }
    }
    jxl::Tree tree{jxl::PropertyDecisionNode::Leaf(jxl::Predictor::Gradient)};
    // Independently decode the existing native resolved-tree emission boundary.
    constexpr std::array<EntropyToken, 5> native_tree{
        {{1, 0}, {2, 5}, {3, 0}, {4, 0}, {5, 0}}};
    BitWriter tree_writer;
    Ok(WriteGlobalTreeInTransaction(native_tree, &tree_writer));
    auto tree_bytes = tree_writer.padded_bytes();
    jxl::BitReader reader(jxl::Bytes(tree_bytes.data(), tree_bytes.size()));
    Check(reader.ReadBits(1) == 1, "Missing global tree");
    jxl::Tree decoded;
    const bool tree_ok = jxl::DecodeTree(&manager, &reader, &decoded, 8);
    const bool closed = reader.Close();
    Check(tree_ok && closed && decoded.size() == 1 && decoded[0].property == -1 &&
              decoded[0].predictor == jxl::Predictor::Gradient &&
              decoded[0].predictor_offset == 0 && decoded[0].multiplier == 1,
          "Native prescribed tree differs from reference");
    jxl::ModularOptions options;
    options.predictor = jxl::Predictor::Gradient;
    jxl::GroupHeader stream_header;
    std::vector<jxl::Token> tokens;
    size_t width;
    Check(jxl::ModularCompress(image, options, 7, tree, stream_header, tokens, &width),
          "Prescribed reference tree failed");
    Check(tokens.size() == extent.width * extent.height * 3,
          "Reference token count mismatch");
    size_t i = 0;
    for (const auto &ch : image.channel)
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x) {
          const auto guess =
              jxl::PredictNoTreeNoWP(extent.width, ch.Row(y) + x,
                                     ch.plane.PixelsPerRow(), static_cast<int>(x),
                                     static_cast<int>(y), jxl::Predictor::Gradient)
                  .guess;
          Check(tokens[i].context == 0 &&
                    tokens[i].value ==
                        PackSigned(static_cast<int32_t>(ch.Row(y)[x] - guess)),
                "Prescribed tree/predictor adapter disagrees");
          ++i;
        }
    // Force two contexts with a static channel property, independently of tree
    // learning. The left child handles property > 0; the right handles <= 0.
    tree = {jxl::PropertyDecisionNode::Split(0, 0, 1, 2),
            jxl::PropertyDecisionNode::Leaf(jxl::Predictor::Gradient),
            jxl::PropertyDecisionNode::Leaf(jxl::Predictor::Zero)};
    tree[1].lchild = 0;
    tree[2].lchild = 1;
    tokens.clear();
    Check(jxl::ModularCompress(image, options, 7, tree, stream_header, tokens, &width),
          "Split reference tree failed");
    i = 0;
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x) {
          const auto &ch = image.channel[c];
          const auto guess = c == 0 ? 0
                                    : jxl::PredictNoTreeNoWP(
                                          extent.width, ch.Row(y) + x,
                                          ch.plane.PixelsPerRow(), static_cast<int>(x),
                                          static_cast<int>(y), jxl::Predictor::Gradient)
                                          .guess;
          Check(tokens[i].context == (c == 0 ? 1 : 0) &&
                    tokens[i].value ==
                        PackSigned(static_cast<int32_t>(ch.Row(y)[x] - guess)),
                "Split tree static properties or contexts disagree");
          ++i;
        }
    auto cloned = jxl::Image::Clone(image);
    Check(cloned.ok(), "Reference clone failed");
    auto transformed = std::move(cloned).value_();
    jxl::Transform rct(jxl::TransformId::kRCT);
    rct.begin_c = 0;
    rct.rct_type = 6;
    Check(jxl::TransformForward(rct, transformed, header, nullptr),
          "Reference forward RCT failed");
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x) {
        const int64_t r = image.channel[0].Row(y)[x], g = image.channel[1].Row(y)[x],
                      b = image.channel[2].Row(y)[x];
        const int64_t co = r - b, tmp = b + (co >> 1), cg = g - tmp;
        Check(transformed.channel[0].Row(y)[x] == tmp + (cg >> 1) &&
                  transformed.channel[1].Row(y)[x] == co &&
                  transformed.channel[2].Row(y)[x] == cg,
              "Reference RCT differs from prescribed arithmetic");
      }
    Check(rct.Inverse(transformed, header), "Reference inverse RCT failed");
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < extent.height; ++y)
        Check(std::equal(image.channel[c].Row(y),
                         image.channel[c].Row(y) + extent.width,
                         transformed.channel[c].Row(y)),
              "Reference inverse RCT did not recover source");
  }
}
} // namespace
int main() {
  try {
    HeadersAndSections();
    std::cout << "Header and section parsing passed\n";
    IntegerHarness();
    StageAdapters();
    std::cout << "Pinned Modular header, geometry, integer and stage oracles passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
