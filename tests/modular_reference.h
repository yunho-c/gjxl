// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Test-only, independently implemented libjxl adapters. No native encoding
// algorithm is used to generate the reference files or decode their samples.
namespace gjxl::test::modular_reference {
struct IntegerImage {
  size_t width = 0, height = 0;
  uint32_t channels = 3, bits = 8;
  // Interleaved source integers, without normalization (also for 8-bit input).
  std::vector<uint16_t> samples;
  bool operator==(const IntegerImage &) const = default;
};
struct ParsedHeader {
  size_t width, height;
  uint32_t bits, color_channels, extra_channels, alpha_bits;
  bool floating_point, xyb, modular, alpha_associated, gaborish;
  uint32_t epf_iterations;
  size_t group_dimension, groups, dc_groups, sections, bits_consumed;
  bool source_srgb, source_linear_srgb, final_frame;
  uint32_t passes, upsampling;
  bool modular_16_bit_buffer_sufficient;
};
struct ReferenceToken { uint32_t context, value; };
std::vector<ReferenceToken> TokenizeGradient(const IntegerImage& image, size_t stream_id);
// Throw std::runtime_error on malformed/unsupported input. All output is local
// until success; callers can retain previous values when testing rejection.
std::vector<uint8_t> EncodeLossless(const IntegerImage &image);
IntegerImage DecodeLossless(std::span<const uint8_t> codestream);
ParsedHeader InspectHeaders(std::span<const uint8_t> codestream,
                            size_t initial_bit_offset = 0);
std::vector<uint32_t> ReadSectionSizes(std::span<const uint8_t> codestream);
} // namespace gjxl::test::modular_reference
