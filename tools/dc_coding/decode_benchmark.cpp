// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Standalone qualification helper; never linked into the native encoder.
#include <CommonCrypto/CommonDigest.h>
#include <jxl/decode.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<uint8_t>;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
size_t Integer(const char* text) {
  const std::string value(text);
  Require(!value.empty() && value.find_first_not_of("0123456789") ==
                               std::string::npos, "Invalid integer");
  return std::stoull(value);
}
std::string Hash(const void* data, size_t size) {
  Require(size <= std::numeric_limits<CC_LONG>::max(), "Hash input too large");
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(data, static_cast<CC_LONG>(size), digest.data());
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (auto value : digest) result << std::setw(2) << static_cast<unsigned>(value);
  return result.str();
}
Bytes Read(const char* path) {
  std::ifstream input(path, std::ios::binary);
  Require(input.good(), "Cannot open input");
  Bytes result{std::istreambuf_iterator<char>(input), {}};
  Require(!result.empty() && !input.bad(), "Cannot read input");
  return result;
}
std::vector<float> Decode(const Bytes& input, size_t width, size_t height) {
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(
      JxlDecoderCreate(nullptr), JxlDecoderDestroy);
  Require(decoder != nullptr, "Cannot create decoder");
  Require(JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_BASIC_INFO |
                  JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS,
          "Cannot subscribe to decoder events");
  Require(JxlDecoderSetInput(decoder.get(), input.data(), input.size()) ==
              JXL_DEC_SUCCESS, "Cannot set decoder input");
  JxlDecoderCloseInput(decoder.get());
  const JxlPixelFormat format{3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  bool info_seen = false, profile_seen = false, image_seen = false;
  std::vector<float> pixels;
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder.get());
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info{};
      Require(JxlDecoderGetBasicInfo(decoder.get(), &info) == JXL_DEC_SUCCESS &&
                  info.xsize == width && info.ysize == height &&
                  info.num_color_channels == 3 && info.num_extra_channels == 0,
              "Unexpected decoded geometry or channels");
      info_seen = true;
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      JxlColorEncoding linear{};
      linear.color_space = JXL_COLOR_SPACE_RGB;
      linear.white_point = JXL_WHITE_POINT_D65;
      linear.primaries = JXL_PRIMARIES_SRGB;
      linear.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
      linear.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
      Require(JxlDecoderSetPreferredColorProfile(decoder.get(), &linear) ==
                  JXL_DEC_SUCCESS, "Cannot select linear sRGB output");
      profile_seen = true;
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t size = 0;
      Require(info_seen && profile_seen &&
                  JxlDecoderImageOutBufferSize(decoder.get(), &format, &size) ==
                      JXL_DEC_SUCCESS && size == width * height * 3 * sizeof(float),
              "Unexpected decoder buffer size");
      pixels.resize(size / sizeof(float));
      Require(JxlDecoderSetImageOutBuffer(decoder.get(), &format, pixels.data(),
                                         size) == JXL_DEC_SUCCESS,
              "Cannot set decoder output");
    } else if (status == JXL_DEC_FULL_IMAGE) {
      Require(!image_seen, "Unexpected extra frame");
      image_seen = true;
    } else if (status == JXL_DEC_SUCCESS) {
      Require(info_seen && profile_seen && image_seen && !pixels.empty(),
              "Incomplete decode");
      return pixels;
    } else {
      throw std::runtime_error("Decoder failed or requested unexpected input");
    }
  }
}
void Equal(const std::vector<float>& expected, const std::vector<float>& actual) {
  Require(actual.size() == expected.size() &&
              std::memcmp(expected.data(), actual.data(),
                          actual.size() * sizeof(float)) == 0,
          "Decoded float pixels differ");
}
} // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 7 || (argc == 8 && std::string(argv[7]) == "--independent-pixels"),
        "Usage: decode_benchmark gradient.jxl weighted.jxl width height warmups pairs [--independent-pixels]");
    const size_t width = Integer(argv[3]), height = Integer(argv[4]);
    const size_t warmups = Integer(argv[5]), pairs = Integer(argv[6]);
    Require(width > 0 && height > 0 && width <= 100000 && height <= 100000 &&
                height <= std::numeric_limits<size_t>::max() / width / 12 &&
                warmups <= 100 && pairs <= 1000, "Invalid decode dimensions/counts");
    const std::array<Bytes, 2> inputs{Read(argv[1]), Read(argv[2])};
    const std::array<std::vector<float>, 2> expected{
      Decode(inputs[0], width, height), Decode(inputs[1], width, height)};
    for (const auto& pixels : expected)
      Require(std::all_of(pixels.begin(), pixels.end(),
                          [](float value) { return std::isfinite(value); }),
              "Nonfinite decoded pixels");
    const bool equal = expected[0].size() == expected[1].size() &&
      std::memcmp(expected[0].data(), expected[1].data(), expected[0].size() * sizeof(float)) == 0;
    if (argc == 7) Equal(expected[0], expected[1]);
    for (size_t i = 0; i < warmups; ++i)
      for (size_t variant = 0; variant < 2; ++variant)
        Equal(expected[variant], Decode(inputs[variant], width, height));
    struct Sample { size_t pair, position, variant; int64_t ns; };
    std::vector<Sample> samples;
    samples.reserve(pairs * 2);
    for (size_t pair = 0; pair < pairs; ++pair) {
      for (size_t position = 0; position < 2; ++position) {
        const auto variant = (pair + position) % 2;
        const auto begin = Clock::now();
        const auto actual = Decode(inputs[variant], width, height);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            Clock::now() - begin).count();
        Equal(expected[variant], actual);
        Require(ns > 0, "Invalid decode time");
        samples.push_back({pair, position, variant, ns});
      }
    }
    std::cout << "{\"schema_version\":1,\"decoder_version\":" << JxlDecoderVersion()
              << ",\"thread_count\":1,\"output_color\":\"RGB_D65_SRG_Rel_Lin\","
              << "\"timing_semantics\":\"fresh-decoder-creation-allocation-decode-destruction\","
              << "\"decoded_equal\":" << (equal ? "true" : "false")
              << ",\"decoded_finite\":true,\"independent_pixels\":" << (argc == 8 ? "true" : "false")
              << ",\"second_decoded_sha256\":\""
              << Hash(expected[1].data(), expected[1].size() * sizeof(float))
              << "\",\"decoded_sha256\":\""
              << Hash(expected[0].data(), expected[0].size() * sizeof(float))
              << "\",\"gradient_sha256\":\"" << Hash(inputs[0].data(), inputs[0].size())
              << "\",\"weighted_sha256\":\"" << Hash(inputs[1].data(), inputs[1].size())
              << "\",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto& sample = samples[i];
      if (i) std::cout << ',';
      std::cout << "{\"pair\":" << sample.pair << ",\"position\":" << sample.position
                << ",\"variant\":\"" << (sample.variant ? "weighted" : "gradient")
                << "\",\"elapsed_nanoseconds\":" << sample.ns << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
