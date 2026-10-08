// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "modular_reference.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/toc.h"

namespace gjxl::test::modular_reference {
namespace {
void Require(bool value, const char *what) {
  if (!value)
    throw std::runtime_error(what);
}
struct ReaderCloser {
  jxl::BitReader &reader;
  ~ReaderCloser() { (void)reader.Close(); }
};
size_t Count(const IntegerImage &image) {
  Require(image.width && image.height &&
              (image.channels == 1 || image.channels == 3 || image.channels == 4) &&
              (image.bits == 8 || image.bits == 16),
          "Unsupported reference input");
  Require(image.width <= UINT32_MAX && image.height <= UINT32_MAX &&
              image.width <=
                  SIZE_MAX / image.height / image.channels / sizeof(uint16_t),
          "Reference input overflow");
  return image.width * image.height * image.channels;
}
} // namespace
std::vector<uint8_t> EncodeLossless(const IntegerImage &image) {
  Require(image.samples.size() == Count(image), "Reference sample count mismatch");
  std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> enc(
      JxlEncoderCreate(nullptr), JxlEncoderDestroy);
  Require(bool(enc), "Reference encoder allocation failed");
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = static_cast<uint32_t>(image.width);
  info.ysize = static_cast<uint32_t>(image.height);
  info.bits_per_sample = image.bits;
  info.uses_original_profile = JXL_TRUE;
  info.num_color_channels = image.channels == 1 ? 1 : 3;
  info.num_extra_channels = image.channels == 4 ? 1 : 0;
  info.alpha_bits = image.channels == 4 ? image.bits : 0;
  info.alpha_premultiplied = JXL_FALSE;
  Require(JxlEncoderSetBasicInfo(enc.get(), &info) == JXL_ENC_SUCCESS,
          "Reference basic info failed");
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, image.channels == 1);
  Require(JxlEncoderSetColorEncoding(enc.get(), &color) == JXL_ENC_SUCCESS,
          "Reference color failed");
  auto *settings = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  Require(settings != nullptr, "Reference frame settings failed");
  Require(JxlEncoderSetFrameLossless(settings, JXL_TRUE) == JXL_ENC_SUCCESS &&
              JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR,
                                               1) == JXL_ENC_SUCCESS &&
              JxlEncoderFrameSettingsSetOption(settings,
                                               JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE,
                                               1) == JXL_ENC_SUCCESS &&
              JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT,
                                               1) == JXL_ENC_SUCCESS,
          "Reference lossless settings failed");
  JxlPixelFormat format{image.channels,
                        image.bits == 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16,
                        JXL_NATIVE_ENDIAN, 0};
  std::vector<uint8_t> bytes;
  const void *data = image.samples.data();
  size_t size = image.samples.size() * sizeof(uint16_t);
  if (image.bits == 8) {
    bytes.reserve(image.samples.size());
    for (uint16_t value : image.samples) {
      Require(value <= 255, "Reference 8-bit sample out of range");
      bytes.push_back(static_cast<uint8_t>(value));
    }
    data = bytes.data();
    size = bytes.size();
  }
  Require(JxlEncoderAddImageFrame(settings, &format, data, size) == JXL_ENC_SUCCESS,
          "Reference frame input failed");
  JxlEncoderCloseInput(enc.get());
  std::vector<uint8_t> output(4096);
  size_t used = 0;
  for (;;) {
    uint8_t *next = output.data() + used;
    size_t available = output.size() - used;
    auto status = JxlEncoderProcessOutput(enc.get(), &next, &available);
    used = output.size() - available;
    if (status == JXL_ENC_SUCCESS)
      break;
    Require(status == JXL_ENC_NEED_MORE_OUTPUT, "Reference encoding failed");
    output.resize(output.size() * 2);
  }
  output.resize(used);
  if (output.size() >= 2 && output[0] == 0xff && output[1] == 0x0a)
    return output;
  // Reference encoder versions may wrap their output. This fixture adapter
  // accepts ordinary ordered jxlc/jxlp boxes; production remains raw-only.
  std::vector<uint8_t> raw;
  uint32_t sequence = 0;
  for (size_t pos = 0; pos < output.size();) {
    Require(output.size() - pos >= 8, "Truncated reference container box");
    const auto be32 = [&](size_t at) {
      return uint32_t(output[at]) << 24 | uint32_t(output[at + 1]) << 16 |
             uint32_t(output[at + 2]) << 8 | output[at + 3];
    };
    uint64_t length = be32(pos);
    const uint32_t type = be32(pos + 4);
    size_t header_size = 8;
    if (length == 1) {
      Require(output.size() - pos >= 16, "Truncated reference large box");
      length = uint64_t(be32(pos + 8)) << 32 | be32(pos + 12);
      header_size = 16;
    }
    if (length == 0)
      length = output.size() - pos;
    Require(length >= header_size && length <= output.size() - pos,
            "Invalid reference box size");
    size_t start = pos + header_size;
    if (type == 0x6a786c70) { // jxlp
      Require(length - header_size >= 4 && (be32(start) & 0x7fffffffu) == sequence++,
              "Unordered reference jxlp box");
      start += 4;
    }
    if (type == 0x6a786c63 || type == 0x6a786c70)
      raw.insert(raw.end(), output.begin() + start,
                 output.begin() + pos + static_cast<size_t>(length));
    pos += static_cast<size_t>(length);
  }
  Require(raw.size() >= 2 && raw[0] == 0xff && raw[1] == 0x0a,
          "Reference container has no codestream");
  return raw;
}

ParsedHeader InspectHeaders(std::span<const uint8_t> bytes, size_t offset) {
  Require(offset < 8, "Invalid reference header offset");
  jxl::BitReader reader(jxl::Bytes(bytes.data(), bytes.size()));
  ReaderCloser closer{reader};
  reader.SkipBits(offset);
  const auto marker = reader.ReadBits(16);
  Require(marker == 0x0aff,
          ("Missing raw codestream marker: " + std::to_string(marker) + " offset " +
           std::to_string(offset))
              .c_str());
  jxl::CodecMetadata metadata;
  Require(bool(jxl::ReadSizeHeader(&reader, &metadata.size)),
          "Reference size parse failed");
  Require(bool(jxl::ReadImageMetadata(&reader, &metadata.m)),
          "Reference metadata parse failed");
  Require(!metadata.m.color_encoding.WantICC(), "ICC unsupported by reference harness");
  metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
  Require(bool(jxl::Bundle::Read(&reader, &metadata.transform_data)),
          "Reference transform metadata parse failed");
  // Native image headers are independently byte padded before being appended.
  const size_t padding = (8 - (reader.TotalBitsConsumed() - offset) % 8) % 8;
  Require(reader.ReadBits(padding) == 0, "Nonzero image header padding");
  jxl::FrameHeader frame(&metadata);
  Require(bool(jxl::ReadFrameHeader(&reader, &frame)), "Reference frame parse failed");
  Require(reader.AllReadsWithinBounds(), "Truncated image/frame headers");
  const auto dim = frame.ToFrameDimensions();
  const auto &m = metadata.m;
  const bool alpha = !m.extra_channel_info.empty();
  return {
      metadata.xsize(),
      metadata.ysize(),
      m.bit_depth.bits_per_sample,
      static_cast<uint32_t>(m.color_encoding.IsGray() ? 1 : 3),
      static_cast<uint32_t>(m.extra_channel_info.size()),
      alpha ? m.extra_channel_info[0].bit_depth.bits_per_sample : 0,
      m.bit_depth.floating_point_sample,
      m.xyb_encoded,
      frame.encoding == jxl::FrameEncoding::kModular,
      alpha && m.extra_channel_info[0].alpha_associated,
      frame.loop_filter.gab,
      frame.loop_filter.epf_iters,
      dim.group_dim,
      dim.num_groups,
      dim.num_dc_groups,
      jxl::NumTocEntries(dim.num_groups, dim.num_dc_groups, frame.passes.num_passes),
      reader.TotalBitsConsumed(),
      m.color_encoding.IsSRGB(),
      m.color_encoding.IsLinearSRGB(),
      frame.is_last,
      frame.passes.num_passes,
      frame.upsampling};
}

std::vector<uint32_t> ReadSectionSizes(std::span<const uint8_t> bytes) {
  const auto header = InspectHeaders(bytes);
  jxl::BitReader reader(jxl::Bytes(bytes.data(), bytes.size()));
  ReaderCloser closer{reader};
  reader.SkipBits(header.bits_consumed);
  std::vector<uint32_t> sizes;
  std::vector<jxl::coeff_order_t> permutation;
  JxlMemoryManager manager;
  Require(bool(jxl::MemoryManagerInit(&manager, nullptr)),
          "Reference memory manager failed");
  Require(bool(jxl::ReadToc(&manager, header.sections, &reader, &sizes, &permutation)),
          "Reference TOC parse failed");
  Require(reader.AllReadsWithinBounds() && permutation.empty(),
          "Unexpected TOC permutation or truncation");
  return sizes;
}

IntegerImage DecodeLossless(std::span<const uint8_t> bytes) {
  const auto header = InspectHeaders(bytes);
  Require(header.modular && !header.xyb && !header.floating_point && !header.gaborish &&
              !header.epf_iterations,
          "Reference decoder requires an integer lossless Modular profile");
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> dec(
      JxlDecoderCreate(nullptr), JxlDecoderDestroy);
  Require(bool(dec), "Reference decoder allocation failed");
  Require(
      JxlDecoderSetUnpremultiplyAlpha(dec.get(), JXL_FALSE) == JXL_DEC_SUCCESS &&
          JxlDecoderSubscribeEvents(
              dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME |
                             JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS &&
          JxlDecoderSetInput(dec.get(), bytes.data(), bytes.size()) == JXL_DEC_SUCCESS,
      "Reference decoder setup failed");
  JxlDecoderCloseInput(dec.get());
  IntegerImage image;
  std::vector<uint8_t> narrow;
  bool basic = false, color_seen = false, full = false;
  unsigned frames = 0;
  for (;;) {
    auto status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      Require(JxlDecoderGetBasicInfo(dec.get(), &info) == JXL_DEC_SUCCESS,
              "Reference basic info missing");
      Require(
          !info.have_animation && !info.exponent_bits_per_sample &&
              info.uses_original_profile && info.orientation == JXL_ORIENT_IDENTITY &&
              info.num_extra_channels <= 1 && !info.alpha_premultiplied &&
              !info.alpha_exponent_bits &&
              (!info.num_extra_channels || (info.alpha_bits == info.bits_per_sample &&
                                            info.num_color_channels == 3)),
          "Unsupported reference output metadata");
      if (info.num_extra_channels) {
        JxlExtraChannelInfo extra;
        Require(JxlDecoderGetExtraChannelInfo(dec.get(), 0, &extra) ==
                        JXL_DEC_SUCCESS &&
                    extra.type == JXL_CHANNEL_ALPHA && extra.dim_shift == 0 &&
                    !extra.alpha_premultiplied,
                "Unsupported extra channel");
      }
      image = {info.xsize,
               info.ysize,
               info.num_color_channels + info.num_extra_channels,
               info.bits_per_sample,
               {}};
      image.samples.resize(Count(image));
      if (image.bits == 8)
        narrow.resize(image.samples.size());
      basic = true;
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      JxlColorEncoding encoded;
      Require(JxlDecoderGetColorAsEncodedProfile(dec.get(),
                                                 JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                                 &encoded) == JXL_DEC_SUCCESS &&
                  (encoded.color_space == JXL_COLOR_SPACE_RGB ||
                   encoded.color_space == JXL_COLOR_SPACE_GRAY) &&
                  encoded.white_point == JXL_WHITE_POINT_D65 &&
                  encoded.transfer_function == JXL_TRANSFER_FUNCTION_SRGB &&
                  (encoded.color_space == JXL_COLOR_SPACE_GRAY ||
                   encoded.primaries == JXL_PRIMARIES_SRGB),
              "Unsupported reference source color");
      Require(JxlDecoderSetPreferredColorProfile(dec.get(), &encoded) ==
                  JXL_DEC_SUCCESS,
              "Cannot request source-space output");
      color_seen = true;
    } else if (status == JXL_DEC_FRAME) {
      Require(++frames == 1, "Reference harness only supports one frame");
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      Require(basic, "Reference output requested before metadata");
      JxlPixelFormat format{image.channels,
                            image.bits == 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16,
                            JXL_NATIVE_ENDIAN, 0};
      size_t bytes_needed;
      const size_t capacity = image.samples.size() * (image.bits / 8);
      Require(JxlDecoderImageOutBufferSize(dec.get(), &format, &bytes_needed) ==
                      JXL_DEC_SUCCESS &&
                  bytes_needed == capacity,
              "Reference output size mismatch");
      void *destination =
          image.bits == 8 ? static_cast<void *>(narrow.data()) : image.samples.data();
      Require(JxlDecoderSetImageOutBuffer(dec.get(), &format, destination, capacity) ==
                  JXL_DEC_SUCCESS,
              "Reference output buffer failed");
    } else if (status == JXL_DEC_FULL_IMAGE) {
      full = true;
    } else if (status == JXL_DEC_SUCCESS) {
      Require(basic && color_seen && full && frames == 1,
              "Incomplete reference decode");
      if (image.bits == 8)
        std::copy(narrow.begin(), narrow.end(), image.samples.begin());
      return image;
    } else {
      throw std::runtime_error("Reference decode failed or input is truncated");
    }
  }
}
} // namespace gjxl::test::modular_reference
