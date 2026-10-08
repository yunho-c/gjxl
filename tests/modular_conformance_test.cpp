// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/tokenization.h"
#include "codestream/modular/workflow.h"
#include "modular_reference.h"
#include <algorithm>
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
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
ref::IntegerImage Fixture(Extent2D extent, unsigned pattern) {
  ref::IntegerImage image{extent.width, extent.height, 3, 8, {}};
  image.samples.resize(extent.width * extent.height * 3);
  uint32_t rng = 0xabcdef01;
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < 3; ++c) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        uint8_t value = 0;
        switch (pattern) {
        case 0:
          value = 0;
          break;
        case 1:
          value = 255;
          break;
        case 2:
          value = 37;
          break;
        case 3:
          value = static_cast<uint8_t>(x + y * 3 + c * 71);
          break;
        case 4:
          value = (x == extent.width / 2 && y == extent.height / 2) ? 255 : 0;
          break;
        case 5:
          value = (x + y + c) % 2 ? 255 : 0;
          break;
        case 6:
          value = static_cast<uint8_t>(rng);
          break;
        default:
          value = c == 0 ? 0 : c == 1 ? 255 : static_cast<uint8_t>(x ^ y);
          break;
        }
        image.samples[(y * extent.width + x) * 3 + c] = value;
      }
  return image;
}
// Independently form tiles from source pixels, without native rectangles/views.
ref::IntegerImage Tile(const ref::IntegerImage &image, size_t x0, size_t y0, size_t w, size_t h) {
  ref::IntegerImage tile{w, h, 3, 8, {}};
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      for (size_t c = 0; c < 3; ++c)
        tile.samples.push_back(image.samples[((y0 + y) * image.width + x0 + x) * 3 + c]);
  return tile;
}
void Tokens(Rgb8View view, const ref::IntegerImage &source) {
  ModularEncoderFrame frame;
  ModularFrameGeometry geometry;
  ModularStreamPlan plan;
  PreparedModularTokens native;
  Ok(ModularEncoderFrame::Prepare(view, &frame));
  Ok(ModularFrameGeometry::Create(view.extent, &geometry));
  Ok(BuildModularStreamPlan(geometry, Rgb8Channels(view.extent), 0, &plan));
  Ok(TokenizeRgb8(frame, plan, &native));
  const bool global = source.width <= 256 && source.height <= 256;
  const size_t groups_x = (source.width + 255) / 256;
  const size_t dc_groups = ((source.width + 2047) / 2048) * ((source.height + 2047) / 2048);
  size_t total = 0;
  for (size_t i = 0; i < plan.streams.size(); ++i) {
    const auto &stream = plan.streams[i];
    const auto actual = native.streams[i];
    if (stream.role == StreamRole::kDcGroup || (stream.role == StreamRole::kGlobal && !global) ||
        (stream.role == StreamRole::kGroup && global)) {
      Check(actual.size() == 0, "Empty/global stream tokenized twice");
      continue;
    }
    const size_t x = global ? 0 : (stream.group % groups_x) * 256;
    const size_t y = global ? 0 : (stream.group / groups_x) * 256;
    const size_t w = global ? source.width : std::min(size_t{256}, source.width - x);
    const size_t h = global ? source.height : std::min(size_t{256}, source.height - y);
    const size_t id = global ? 0 : 1 + 3 * dc_groups + 17 + stream.group;
    Check(stream.id == id, "Stream identity differs");
    const auto reference = ref::TokenizeGradient(Tile(source, x, y, w, h), id);
    Check(actual.size() == reference.size(), "Native/reference token count differs");
    for (size_t j = 0; j < reference.size(); ++j)
      Check(actual[j].context == reference[j].context && actual[j].value == reference[j].value,
            "Native/reference gradient token differs");
    total += actual.size();
  }
  Check(total == source.samples.size(), "Sample token coverage differs");
}
void Case(Extent2D extent, unsigned pattern, const std::filesystem::path &artifacts,
          size_t *files) {
  const auto source = Fixture(extent, pattern);
  const size_t stride = 3 * extent.width + 13;
  std::vector<uint8_t> bytes(7 + stride * extent.height, 0xa5);
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < 3 * extent.width; ++x)
      bytes[7 + y * stride + x] = static_cast<uint8_t>(source.samples[y * extent.width * 3 + x]);
  Rgb8View view{std::span(bytes).subspan(7), extent, stride};
  Tokens(view, source);
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    std::vector<uint8_t> encoded;
    Ok(EncodeRgb8Modular(view, {.entropy = mode}, &encoded));
    const auto header = ref::InspectHeaders(encoded);
    Check(header.width == extent.width && header.height == extent.height && header.bits == 8 &&
              header.color_channels == 3 && header.extra_channels == 0 && header.modular &&
              header.modular_16_bit_buffer_sufficient && !header.xyb && !header.gaborish &&
              header.epf_iterations == 0 && header.source_srgb && header.final_frame &&
              header.passes == 1 && header.upsampling == 1 && header.group_dimension == 256,
          "Native metadata differs from RGB8 profile");
    const auto sizes = ref::ReadSectionSizes(encoded);
    Check(sizes.size() == header.sections && sizes[0] > 0, "TOC differs");
    if (header.groups == 1)
      Check(sizes.size() == 1, "Single group was not combined");
    else {
      for (size_t i = 1; i <= header.dc_groups + 1; ++i)
        Check(sizes[i] == 0, "Empty DC/AC global section contains bytes");
      for (size_t i = header.dc_groups + 2; i < sizes.size(); ++i)
        Check(sizes[i] > 0, "Nonempty group has no payload");
    }
    if (!artifacts.empty()) {
      const auto name = std::to_string(extent.width) + "x" + std::to_string(extent.height) + "-p" +
                        std::to_string(pattern) +
                        (mode == EntropyCodingMode::kAns ? "-ans.jxl" : "-prefix.jxl");
      std::ofstream file(artifacts / name, std::ios::binary);
      file.write(reinterpret_cast<const char *>(encoded.data()),
                 static_cast<std::streamsize>(encoded.size()));
      Check(bool(file), "Cannot retain native fixture");
    }
    try {
      Check(ref::DecodeLossless(encoded) == source, "Native integer reconstruction differs");
    } catch (...) {
      std::cerr << extent.width << "x" << extent.height << " pattern " << pattern << " mode "
                << static_cast<int>(mode) << " bytes " << encoded.size() << '\n';
      throw;
    }
    bool rejected = false;
    try {
      (void)ref::DecodeLossless(std::span(encoded).first(encoded.size() - 1));
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    Check(rejected, "Truncated native file accepted");
    ++*files;
  }
}
} // namespace
int main(int argc, char **argv) try {
  const std::filesystem::path artifacts = argc > 1 ? argv[1] : "";
  if (!artifacts.empty())
    std::filesystem::create_directories(artifacts);
  size_t files = 0;
  for (auto extent : {Extent2D{1, 1},
                      {1, 17},
                      {17, 1},
                      {17, 19},
                      {255, 3},
                      {256, 3},
                      {257, 3},
                      {3, 255},
                      {3, 256},
                      {3, 257},
                      {255, 257},
                      {256, 256},
                      {257, 257},
                      {2047, 1},
                      {2048, 1},
                      {2049, 1},
                      {1, 2047},
                      {1, 2048},
                      {1, 2049}})
    for (unsigned pattern = 0; pattern < 8; ++pattern)
      Case(extent, pattern, artifacts, &files);
  Case({2049, 2049}, 6, artifacts, &files);
  std::cout << files << " native RGB8 files decoded exactly; " << files / 2
            << " independent token cases passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
