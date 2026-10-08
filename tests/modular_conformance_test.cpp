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
ref::IntegerImage Fixture(Extent2D extent, unsigned pattern, uint32_t channels, uint32_t bits) {
  ref::IntegerImage image{extent.width, extent.height, channels, bits, {}};
  image.samples.resize(extent.width * extent.height * channels);
  uint32_t rng = 0xabcdef01;
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < channels; ++c) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        const uint16_t maximum = bits == 8 ? 255 : 65535;
        uint16_t value = 0;
        switch (pattern) {
        case 0:
          value = 0;
          break;
        case 1:
          value = maximum;
          break;
        case 2:
          value = 37;
          break;
        case 3:
          value = static_cast<uint16_t>((x + y * 3 + c * 71) & maximum);
          break;
        case 4:
          value = (x == extent.width / 2 && y == extent.height / 2) ? maximum : 0;
          break;
        case 5:
          value = (x + y + c) % 2 ? maximum : 0;
          break;
        case 6:
          value = static_cast<uint16_t>(rng & maximum);
          break;
        case 7:
          value = c == 0 ? 0 : c == 1 ? maximum : static_cast<uint16_t>((x ^ y) & maximum);
          break;
        case 8: {
          constexpr uint16_t edges[] = {0, 1, 255, 256, 32767, 32768, 65534, 65535};
          value = edges[(x + y + c) % 8] & maximum;
          break;
        }
        case 9: // A 256x256 fixture visits every unsigned 16-bit sample.
          value = static_cast<uint16_t>((y * extent.width + x + c * 257) & maximum);
          break;
        case 10: // Identical low bytes expose lost or swapped high bytes.
          value = static_cast<uint16_t>((((x + y + c) << 8) | 37) & maximum);
          break;
        default: // Transparent colored pixels, mixed with partial and opaque alpha.
          value = c == 3 ? (x % 3 == 0 ? 0 : x % 3 == 1 ? maximum / 2 : maximum)
                         : static_cast<uint16_t>((1 + x * 71 + y * 97 + c * 257) & maximum);
          break;
        }
        image.samples[(y * extent.width + x) * channels + c] = value;
      }
  return image;
}
// Independently form tiles from source pixels, without native rectangles/views.
ref::IntegerImage Tile(const ref::IntegerImage &image, size_t x0, size_t y0, size_t w, size_t h) {
  ref::IntegerImage tile{w, h, image.channels, image.bits, {}};
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      for (size_t c = 0; c < image.channels; ++c)
        tile.samples.push_back(
            image.samples[((y0 + y) * image.width + x0 + x) * image.channels + c]);
  return tile;
}
void Tokens(PackedModularImageView view, const ref::IntegerImage &source) {
  ModularEncoderFrame frame;
  ModularFrameGeometry geometry;
  ModularStreamPlan plan;
  PreparedModularTokens native;
  Ok(ModularEncoderFrame::Prepare(view, &frame));
  Ok(ModularFrameGeometry::Create(view.extent, &geometry));
  ModularInputProfile profile;
  Ok(ResolveModularInput(view.extent, view.format, &profile));
  Ok(BuildModularStreamPlan(geometry, profile.channels(), 0, &plan));
  Check(frame.image().channel_count() == source.channels, "Prepared channel count differs");
  for (size_t c = 0; c < source.channels; ++c)
    Check(frame.image().view(c).descriptor.role ==
              (c == 3 ? ChannelRole::kAlpha : ChannelRole::kColor),
          "Prepared channel role differs");
  Ok(TokenizeIdentity(frame, plan, &native));
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
struct FormatCase {
  PackedModularFormat format;
  uint32_t channels, bits;
  const char *name;
};
constexpr FormatCase kFormats[] = {
    {PackedModularFormat::kRgb8, 3, 8, "rgb8"},
    {PackedModularFormat::kGray8, 1, 8, "gray8"},
    {PackedModularFormat::kGray16, 1, 16, "gray16"},
    {PackedModularFormat::kRgb16, 3, 16, "rgb16"},
    {PackedModularFormat::kRgba8, 4, 8, "rgba8"},
    {PackedModularFormat::kRgba16, 4, 16, "rgba16"}};
void Case(Extent2D extent, unsigned pattern, const std::filesystem::path &artifacts,
          size_t *files, FormatCase format) {
  const auto source = Fixture(extent, pattern, format.channels, format.bits);
  const size_t sample_bytes = format.bits / 8;
  const size_t row_bytes = format.channels * extent.width * sample_bytes;
  const size_t stride = row_bytes + 13;
  // Offset and odd stride exercise unaligned loads. Omit final row padding.
  std::vector<uint8_t> bytes(7 + stride * (extent.height - 1) + row_bytes, 0xa5);
  std::array<std::vector<uint8_t>, 2> little_endian_outputs;
  for (auto order : {SampleByteOrder::kLittleEndian, SampleByteOrder::kBigEndian}) {
    if (format.bits == 8 && order == SampleByteOrder::kBigEndian)
      continue;
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < format.channels * extent.width; ++x) {
        const auto sample = source.samples[y * extent.width * format.channels + x];
        const size_t i = 7 + y * stride + x * sample_bytes;
        bytes[i] = static_cast<uint8_t>(
            order == SampleByteOrder::kLittleEndian ? sample : sample >> 8);
        if (sample_bytes == 2)
          bytes[i + 1] = static_cast<uint8_t>(
              order == SampleByteOrder::kLittleEndian ? sample >> 8 : sample);
      }
    const auto unchanged = bytes;
    PackedModularImageView view{std::span(bytes).subspan(7), extent, stride, format.format, order};
    Tokens(view, source);
    size_t mode_index = 0;
    for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
      std::vector<uint8_t> encoded;
      Ok(EncodeModularImage(view, {.entropy = mode}, &encoded));
      Check(bytes == unchanged, "Encoding modified source bytes");
      if (order == SampleByteOrder::kLittleEndian)
        little_endian_outputs[mode_index] = encoded;
      else
        Check(encoded == little_endian_outputs[mode_index], "Byte order changed codestream");
      ++mode_index;
      const auto header = ref::InspectHeaders(encoded);
      Check(header.width == extent.width && header.height == extent.height &&
                header.bits == format.bits &&
                header.color_channels == (format.channels == 1 ? 1u : 3u) &&
                header.extra_channels == (format.channels == 4 ? 1u : 0u) && header.modular &&
                header.alpha_bits == (format.channels == 4 ? format.bits : 0u) &&
                !header.alpha_associated && header.alpha_dimension_shift == 0 &&
                !header.floating_point &&
                header.modular_16_bit_buffer_sufficient == (format.bits == 8) &&
                !header.xyb && !header.gaborish &&
                header.epf_iterations == 0 && header.source_srgb && header.final_frame &&
                header.passes == 1 && header.upsampling == 1 && header.group_dimension == 256,
            "Native metadata differs from source profile");
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
        const auto name =
            std::to_string(extent.width) + "x" + std::to_string(extent.height) + "-p" +
                          std::to_string(pattern) +
                          (mode == EntropyCodingMode::kAns ? "-ans.jxl" : "-prefix.jxl");
        const auto directory = artifacts / format.name /
                               (order == SampleByteOrder::kLittleEndian ? "le" : "be");
        std::filesystem::create_directories(directory);
        std::ofstream file(directory / name, std::ios::binary);
        file.write(reinterpret_cast<const char *>(encoded.data()),
                   static_cast<std::streamsize>(encoded.size()));
        Check(bool(file), "Cannot retain native fixture");
      }
      try {
        Check(ref::DecodeLossless(encoded) == source, "Native integer reconstruction differs");
      } catch (...) {
        std::cerr << extent.width << "x" << extent.height << " pattern " << pattern << " mode "
                  << static_cast<int>(mode) << " format " << format.name << " bytes "
                  << encoded.size() << '\n';
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
}
} // namespace
int main(int argc, char **argv) try {
  const std::filesystem::path artifacts = argc > 1 ? argv[1] : "";
  if (!artifacts.empty())
    std::filesystem::create_directories(artifacts);
  size_t files = 0;
  for (auto format : kFormats) {
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
        Case(extent, pattern, artifacts, &files, format);
    Case({2049, 2049}, 6, artifacts, &files, format);
    Case({17, 19}, 8, artifacts, &files, format);
    if (format.bits == 16) {
      Case({256, 256}, 9, artifacts, &files, format);
      Case({257, 3}, 10, artifacts, &files, format);
    }
    if (format.channels == 4)
      Case({257, 3}, 11, artifacts, &files, format);
  }

  std::cout << files << " native integer files decoded exactly; " << files / 2
            << " independent token cases passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
