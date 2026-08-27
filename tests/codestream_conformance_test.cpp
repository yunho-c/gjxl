// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Independently decodes raw GJXL codestreams and compares float pixels.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "codec/chroma_from_luma.h"
#include "codec/color_transform.h"
#include "codec/epf.h"
#include "codec/gaborish.h"
#include "codec/loop_filter.h"
#include "codec/reconstruction.h"
#include "codec/vardct_frame.h"
#include "codestream/encoder.h"
#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/image.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"
#include "core/quantizer.h"

namespace {

namespace fs = std::filesystem;

constexpr float kIntensityTarget = 255.0f;
constexpr double kAbsoluteTolerance = 4.0e-5;
// Decoder and local XYB-to-RGB arithmetic use different SIMD evaluation
// orders. Keep unit-scale pixels under a tight absolute bound while allowing
// float-sized relative error for the deliberately extreme signed-token case.
constexpr double kRelativeTolerance = 1.0e-4;

enum class Pattern {
  kFlat,
  kImpulse,
  kRandom,
  kGradient,
  kTexture,
  kHardEdge,
  kSaturatedPrimary,
  kLargeOpsin,
};

struct Fixture {
  std::string_view name;
  gjxl::Extent2D extent;
  Pattern pattern;
  gjxl::AcStrategyType strategy = gjxl::AcStrategyType::kDct8;
  bool force_single_strategy = false;
  gjxl::QuantizerParams quantizer = {3541, 10};
  int32_t raw_quant = 29;
  bool require_large_tokens = false;
  uint64_t expected_hash = 0;
};

struct PreparedFixture {
  gjxl::VarDctEncoderFrame frame;
  gjxl::Image3FBuffer expected;
  int64_t maximum_abs_dc = 0;
  int64_t maximum_abs_ac = 0;
};

struct PfmImage {
  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> plane;
};

struct Comparison {
  double maximum_absolute_error = 0.0;
  double maximum_relative_error = 0.0;
  double root_mean_square_error = 0.0;
  double maximum_tolerance_ratio = 0.0;
  size_t worst_channel = 0;
  size_t worst_x = 0;
  size_t worst_y = 0;
  float expected = 0.0f;
  float actual = 0.0f;
};

struct Options {
  fs::path decoder;
  fs::path info;
  fs::path artifacts;
  bool smoke = false;
};

template <typename T>
gjxl::PlaneView<const T> View(
  const std::vector<T>& values, gjxl::Extent2D extent) {
  return {values.data(), extent, extent.width};
}

uint32_t Mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7FEB352Du;
  value ^= value >> 15;
  value *= 0x846CA68Bu;
  return value ^ (value >> 16);
}

std::array<float, 3> PatternPixel(
  Pattern pattern, size_t x, size_t y, gjxl::Extent2D extent) {

  const float fx = extent.width == 1
    ? 0.0f
    : static_cast<float>(x) / static_cast<float>(extent.width - 1);
  const float fy = extent.height == 1
    ? 0.0f
    : static_cast<float>(y) / static_cast<float>(extent.height - 1);
  switch (pattern) {
    case Pattern::kFlat:
      return {0.17f, 0.31f, 0.53f};
    case Pattern::kImpulse:
      if (x == extent.width / 3 && y == extent.height / 2) {
        return {0.93f, 0.08f, 0.71f};
      }
      return {
        0.04f + 0.03f * fx,
        0.11f + 0.02f * fy,
        0.19f + 0.01f * (fx + fy),
      };
    case Pattern::kRandom: {
      const uint32_t base = Mix(
        static_cast<uint32_t>(x) * 0x9E3779B9u ^
        static_cast<uint32_t>(y) * 0x85EBCA6Bu ^ 0xA511E9B3u);
      return {
        static_cast<float>(Mix(base + 0) & 0xFFFFu) / 65535.0f,
        static_cast<float>(Mix(base + 1) & 0xFFFFu) / 65535.0f,
        static_cast<float>(Mix(base + 2) & 0xFFFFu) / 65535.0f,
      };
    }
    case Pattern::kGradient:
      return {fx, fy, 0.15f + 0.7f * (0.65f * fx + 0.35f * fy)};
    case Pattern::kTexture:
      return {
        0.5f + 0.35f * std::sin(0.31f * static_cast<float>(x) +
                                0.17f * static_cast<float>(y)),
        0.5f + 0.30f * std::cos(0.11f * static_cast<float>(x) -
                                0.29f * static_cast<float>(y)),
        0.5f + 0.25f * std::sin(0.23f * static_cast<float>(x + y)),
      };
    case Pattern::kHardEdge:
      return x < extent.width / 2
        ? std::array<float, 3>{0.02f, 0.04f, 0.08f}
        : std::array<float, 3>{0.94f, 0.83f, 0.16f};
    case Pattern::kSaturatedPrimary:
      switch ((x / 3 + y / 2) % 3) {
        case 0:
          return {1.0f, 0.0f, 0.0f};
        case 1:
          return {0.0f, 1.0f, 0.0f};
        default:
          return {0.0f, 0.0f, 1.0f};
      }
    case Pattern::kLargeOpsin:
      break;
  }
  return {};
}

void FillPaddedLinear(const Fixture& fixture, gjxl::Image3FView image) {
  for (size_t y = 0; y < image.height(); ++y) {
    const size_t source_y = std::min(y, fixture.extent.height - 1);
    for (size_t x = 0; x < image.width(); ++x) {
      const size_t source_x = std::min(x, fixture.extent.width - 1);
      const std::array<float, 3> pixel =
        PatternPixel(fixture.pattern, source_x, source_y, fixture.extent);
      for (size_t channel = 0; channel < 3; ++channel) {
        image.plane[channel].Row(y)[x] = pixel[channel];
      }
    }
  }
}

void FillLargeOpsin(gjxl::Image3FView image) {
  for (size_t y = 0; y < image.height(); ++y) {
    for (size_t x = 0; x < image.width(); ++x) {
      const float sign = ((x + 3 * y) & 1) == 0 ? 1.0f : -1.0f;
      image.plane[0].Row(y)[x] = 48.0f + sign * (96.0f + 0.5f * x);
      image.plane[1].Row(y)[x] = 21.0f - sign * (72.0f + 0.25f * y);
      image.plane[2].Row(y)[x] = -35.0f + sign * (88.0f + 0.25f * (x + y));
    }
  }
}

int64_t AbsInt32(int32_t value) {
  return value >= 0 ? value : -static_cast<int64_t>(value);
}

gjxl::Status PrepareFixture(
  const Fixture& fixture, PreparedFixture* prepared) {

  if (prepared == nullptr || fixture.extent.empty() ||
      fixture.raw_quant < 1 || fixture.raw_quant > gjxl::kMaxRawQuant) {
    return gjxl::Status::InvalidArgument("Conformance fixture is invalid");
  }

  gjxl::FrameGeometry geometry;
  gjxl::Status status = gjxl::FrameGeometry::Create(fixture.extent, &geometry);
  if (!status.ok()) {
    return status;
  }

  gjxl::Image3FBuffer preprocessed(geometry.padded_frame());
  if (fixture.pattern == Pattern::kLargeOpsin) {
    FillLargeOpsin(preprocessed.view());
  } else {
    gjxl::Image3FBuffer linear(geometry.padded_frame());
    FillPaddedLinear(fixture, linear.view());
    gjxl::Image3FBuffer opsin(geometry.padded_frame());
    status = gjxl::LinearRgbToOpsin(
      linear.const_view(), kIntensityTarget, opsin.view());
    if (!status.ok()) {
      return status;
    }
    status = gjxl::ApplyGaborishInverse(
      opsin.const_view(), {1.0f, 1.0f, 1.0f}, preprocessed.view());
    if (!status.ok()) {
      return status;
    }
  }

  const gjxl::Extent2D blocks = geometry.block_grid().blocks;
  size_t block_count = 0;
  if (!blocks.try_area(&block_count)) {
    return gjxl::Status::InvalidArgument("Fixture block count overflow");
  }
  gjxl::AcStrategyGrid strategies;
  status = gjxl::AcStrategyGrid::Create(blocks, &strategies);
  if (!status.ok()) {
    return status;
  }
  if (fixture.force_single_strategy) {
    const gjxl::AcStrategyInfo* info =
      gjxl::GetAcStrategyInfo(fixture.strategy);
    if (info == nullptr || info->covered_blocks != blocks) {
      return gjxl::Status::InvalidArgument(
        "Forced strategy does not cover the fixture");
    }
    status = strategies.Set(0, 0, fixture.strategy);
    if (!status.ok()) {
      return status;
    }
  } else {
    strategies.fill_dct8();
  }

  gjxl::Quantizer quantizer;
  status = gjxl::Quantizer::Create(fixture.quantizer, &quantizer);
  if (!status.ok()) {
    return status;
  }
  const std::vector<int32_t> raw_quant(block_count, fixture.raw_quant);
  std::vector<uint8_t> sharpness(block_count);
  for (size_t index = 0; index < block_count; ++index) {
    sharpness[index] = static_cast<uint8_t>((index * 5 + 4) % 8);
  }

  gjxl::ColorCorrelationMap color_correlation;
  status = gjxl::ComputeFinalColorCorrelationMap(
    preprocessed.const_view(), strategies, View(raw_quant, blocks), quantizer,
    false, &color_correlation);
  if (!status.ok()) {
    return status;
  }

  PreparedFixture result;
  status = gjxl::ComputeQuantizedCoefficients(
    preprocessed.const_view(),
    {
      .geometry = geometry,
      .strategies = &strategies,
      .raw_quant_field = View(raw_quant, blocks),
      .quantizer = &quantizer,
      .color_correlation = &color_correlation,
      .epf_sharpness = View(sharpness, blocks),
    },
    {}, &result.frame);
  if (!status.ok()) {
    return status;
  }

  const gjxl::ConstImage3I32View dc = result.frame.quantized_dc();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < dc.height(); ++y) {
      for (size_t x = 0; x < dc.width(); ++x) {
        result.maximum_abs_dc = std::max(
          result.maximum_abs_dc, AbsInt32(dc.plane[channel].Row(y)[x]));
      }
    }
  }
  for (size_t group_index = 0;
       group_index < result.frame.ac_group_count(); ++group_index) {
    gjxl::VarDctAcGroupView group;
    status = result.frame.GetAcGroup(group_index, &group);
    if (!status.ok()) {
      return status;
    }
    for (std::span<const int32_t> coefficients : group.coefficients) {
      for (int32_t coefficient :
           coefficients.first(group.used_coefficient_count)) {
        result.maximum_abs_ac = std::max(
          result.maximum_abs_ac, AbsInt32(coefficient));
      }
    }
  }

  gjxl::Image3FBuffer reconstructed(geometry.padded_frame());
  status = gjxl::ReconstructQuantizedCoefficients(
    result.frame, reconstructed.view());
  if (!status.ok()) {
    return status;
  }
  std::vector<float> inverse_sigma(block_count);
  status = gjxl::ComputeEpfInverseSigma(
    result.frame.strategies(), result.frame.raw_quant_field(),
    result.frame.quantizer(), result.frame.epf_sharpness(),
    result.frame.profile().epf_sigma,
    {inverse_sigma.data(), blocks, blocks.width});
  if (!status.ok()) {
    return status;
  }
  gjxl::Image3FBuffer cropped_reconstruction(fixture.extent);
  gjxl::CopyImage(
    reconstructed.cropped_view(fixture.extent),
    cropped_reconstruction.view());
  gjxl::Image3FBuffer filtered(fixture.extent);
  status = gjxl::ApplyLoopFilters(
    cropped_reconstruction.const_view(),
    {inverse_sigma.data(), blocks, blocks.width},
    result.frame.profile().loop_filter, filtered.view());
  if (!status.ok()) {
    return status;
  }

  result.expected.resize(fixture.extent);
  status = gjxl::OpsinToLinearRgb(
    filtered.const_view(), kIntensityTarget, result.expected.view());
  if (!status.ok()) {
    return status;
  }
  *prepared = std::move(result);
  return gjxl::Status::Ok();
}

uint64_t Fnv1a64(std::span<const uint8_t> bytes) {
  uint64_t hash = 1469598103934665603ull;
  for (uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

bool WriteBytes(const fs::path& path, std::span<const uint8_t> bytes) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    return false;
  }
  stream.write(
    reinterpret_cast<const char*>(bytes.data()),
    static_cast<std::streamsize>(bytes.size()));
  return stream.good();
}

void WriteU32(std::ofstream* stream, uint32_t bits, bool little_endian) {
  std::array<uint8_t, 4> bytes = {
    static_cast<uint8_t>(bits),
    static_cast<uint8_t>(bits >> 8),
    static_cast<uint8_t>(bits >> 16),
    static_cast<uint8_t>(bits >> 24),
  };
  if (!little_endian) {
    std::reverse(bytes.begin(), bytes.end());
  }
  stream->write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

bool WritePfm(const fs::path& path, gjxl::ConstImage3FView image) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    return false;
  }
  stream << "PF\n" << image.width() << ' ' << image.height() << "\n-1.0\n";
  for (size_t reverse_y = 0; reverse_y < image.height(); ++reverse_y) {
    const size_t y = image.height() - 1 - reverse_y;
    for (size_t x = 0; x < image.width(); ++x) {
      for (size_t channel = 0; channel < 3; ++channel) {
        WriteU32(
          &stream,
          std::bit_cast<uint32_t>(image.plane[channel].Row(y)[x]), true);
      }
    }
  }
  return stream.good();
}

bool ReadU32(
  std::istream* stream, bool little_endian, uint32_t* value) {
  std::array<uint8_t, 4> bytes{};
  stream->read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!stream->good()) {
    return false;
  }
  if (!little_endian) {
    std::reverse(bytes.begin(), bytes.end());
  }
  *value = static_cast<uint32_t>(bytes[0]) |
    (static_cast<uint32_t>(bytes[1]) << 8) |
    (static_cast<uint32_t>(bytes[2]) << 16) |
    (static_cast<uint32_t>(bytes[3]) << 24);
  return true;
}

bool ReadPfm(const fs::path& path, PfmImage* image, std::string* error) {
  std::ifstream stream(path, std::ios::binary);
  std::string magic;
  size_t width = 0;
  size_t height = 0;
  double scale = 0.0;
  if (!stream || !(stream >> magic >> width >> height >> scale) ||
      magic != "PF" || width == 0 || height == 0 || scale == 0.0) {
    *error = "invalid PFM header";
    return false;
  }
  stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  if (!stream) {
    *error = "invalid PFM header terminator";
    return false;
  }
  size_t pixel_count = 0;
  if (!gjxl::Extent2D{width, height}.try_area(&pixel_count) ||
      pixel_count > std::numeric_limits<size_t>::max() / 3) {
    *error = "PFM dimensions overflow";
    return false;
  }

  PfmImage result;
  result.extent = {width, height};
  for (std::vector<float>& plane : result.plane) {
    plane.resize(pixel_count);
  }
  const bool little_endian = scale < 0.0;
  for (size_t reverse_y = 0; reverse_y < height; ++reverse_y) {
    const size_t y = height - 1 - reverse_y;
    for (size_t x = 0; x < width; ++x) {
      for (size_t channel = 0; channel < 3; ++channel) {
        uint32_t bits = 0;
        if (!ReadU32(&stream, little_endian, &bits)) {
          *error = "truncated PFM pixels";
          return false;
        }
        result.plane[channel][y * width + x] =
          std::bit_cast<float>(bits) * static_cast<float>(std::abs(scale));
      }
    }
  }
  char trailing = 0;
  if (stream.get(trailing)) {
    *error = "PFM has trailing bytes";
    return false;
  }
  *image = std::move(result);
  return true;
}

std::string ShellQuote(const fs::path& path) {
  const std::string value = path.string();
  std::string result = "'";
  for (char c : value) {
    if (c == '\'') {
      result += "'\\''";
    } else {
      result += c;
    }
  }
  return result + "'";
}

int ExitCode(int status) {
  if (status == -1) {
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return -1;
}

int RunTool(
  const fs::path& executable,
  std::span<const std::string> arguments,
  const fs::path& log) {

  std::string command = ShellQuote(executable);
  for (const std::string& argument : arguments) {
    command += ' ';
    command += ShellQuote(argument);
  }
  command += " >" + ShellQuote(log) + " 2>&1";
  return ExitCode(std::system(command.c_str()));
}

bool ReadText(const fs::path& path, std::string* text) {
  std::ifstream stream(path);
  if (!stream) {
    return false;
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  *text = contents.str();
  return stream.good() || stream.eof();
}

bool ContainsMetadata(
  std::string_view info,
  const Fixture& fixture,
  std::string* error) {

  const std::array<std::string, 10> required = {
    "JPEG XL image, " + std::to_string(fixture.extent.width) + "x" +
      std::to_string(fixture.extent.height) +
      ", lossy, 32-bit float (8 exponent bits) RGB",
    "Number of color channels: 3",
    "Number of extra channels: 0",
    "Intrinsic dimensions: " + std::to_string(fixture.extent.width) + "x" +
      std::to_string(fixture.extent.height),
    "Orientation: 1 (Normal)",
    "Color space: RGB",
    "White point: D65",
    "Primaries: sRGB",
    "Transfer function: Linear",
    "Rendering intent: Relative",
  };
  for (const std::string& expected : required) {
    if (info.find(expected) == std::string_view::npos) {
      *error = "jxlinfo is missing: " + expected;
      return false;
    }
  }
  // Pinned jxlinfo prints intensity only when it differs from the 255-nit
  // default retained by the initial profile.
  if (info.find("Intensity target:") != std::string_view::npos) {
    *error = "codestream has a non-default intensity target";
    return false;
  }
  return true;
}

bool ComparePixels(
  gjxl::ConstImage3FView expected,
  const PfmImage& actual,
  Comparison* comparison,
  std::string* error) {

  if (actual.extent != expected.extent()) {
    *error = "decoded PFM dimensions differ from the source";
    return false;
  }
  Comparison result;
  long double squared_error = 0.0;
  size_t sample_count = 0;
  bool within_tolerance = true;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < expected.height(); ++y) {
      for (size_t x = 0; x < expected.width(); ++x) {
        const float reference = expected.plane[channel].Row(y)[x];
        const float decoded =
          actual.plane[channel][y * actual.extent.width + x];
        if (!std::isfinite(reference) || !std::isfinite(decoded)) {
          *error = "comparison encountered a non-finite pixel";
          return false;
        }
        const double absolute =
          std::abs(static_cast<double>(decoded) - reference);
        const double scale = std::max(
          std::abs(static_cast<double>(decoded)),
          std::abs(static_cast<double>(reference)));
        const double relative = scale == 0.0 ? 0.0 : absolute / scale;
        const double limit = kAbsoluteTolerance + kRelativeTolerance * scale;
        within_tolerance &= absolute <= limit;
        squared_error += absolute * absolute;
        ++sample_count;
        result.maximum_absolute_error = std::max(
          result.maximum_absolute_error, absolute);
        result.maximum_relative_error = std::max(
          result.maximum_relative_error, relative);
        const double tolerance_ratio = absolute / limit;
        if (tolerance_ratio > result.maximum_tolerance_ratio) {
          result.maximum_tolerance_ratio = tolerance_ratio;
          result.worst_channel = channel;
          result.worst_x = x;
          result.worst_y = y;
          result.expected = reference;
          result.actual = decoded;
        }
      }
    }
  }
  result.root_mean_square_error = std::sqrt(
    static_cast<double>(squared_error / sample_count));
  *comparison = result;
  if (!within_tolerance) {
    std::ostringstream message;
    message << std::setprecision(9)
            << "pixel tolerance exceeded at c=" << result.worst_channel
            << " x=" << result.worst_x << " y=" << result.worst_y
            << ": expected=" << result.expected
            << " actual=" << result.actual
            << " abs="
            << std::abs(
                 static_cast<double>(result.actual) - result.expected)
            << " tolerance_ratio=" << result.maximum_tolerance_ratio;
    *error = message.str();
    return false;
  }
  return true;
}

std::vector<Fixture> SmokeFixtures() {
  return {
    {"one-pixel-flat", {1, 1}, Pattern::kFlat},
    {"odd-gradient", {13, 17}, Pattern::kGradient},
    {"ac-edge-random", {257, 9}, Pattern::kRandom},
  };
}

std::vector<Fixture> FullFixtures() {
  std::vector<Fixture> fixtures = {
    {"one-pixel-flat", {1, 1}, Pattern::kFlat},
    {"single-block-impulse", {8, 8}, Pattern::kImpulse,
     gjxl::AcStrategyType::kDct8, false, {3541, 10}, 29, false,
     2096525757826868871ull},
    {"odd-gradient", {13, 17}, Pattern::kGradient},
    {"random", {17, 11}, Pattern::kRandom},
    {"texture", {24, 19}, Pattern::kTexture},
    {"hard-edge", {33, 9}, Pattern::kHardEdge},
    {"saturated-primary", {24, 8}, Pattern::kSaturatedPrimary},
    {"ac-boundary", {256, 9}, Pattern::kHardEdge},
    {"ac-edge", {257, 9}, Pattern::kRandom},
    {"dc-boundary", {2048, 9}, Pattern::kTexture},
    {"dc-edge", {2049, 9}, Pattern::kGradient},
    {"raw-quant-one", {8, 8}, Pattern::kRandom,
     gjxl::AcStrategyType::kDct8, false, {3541, 10}, 1},
    {"raw-quant-256", {8, 8}, Pattern::kTexture,
     gjxl::AcStrategyType::kDct8, false, {3541, 10}, 256},
    {"large-signed-tokens", {8, 8}, Pattern::kLargeOpsin,
     gjxl::AcStrategyType::kDct8, false, {32768, 1}, 256, true},
  };
  constexpr std::array strategies = {
    gjxl::AcStrategyType::kDct8,
    gjxl::AcStrategyType::kDct16x16,
    gjxl::AcStrategyType::kDct32x32,
    gjxl::AcStrategyType::kDct16x8,
    gjxl::AcStrategyType::kDct8x16,
    gjxl::AcStrategyType::kDct32x16,
    gjxl::AcStrategyType::kDct16x32,
  };
  for (const gjxl::AcStrategyType strategy : strategies) {
    const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
    fixtures.push_back({
      .name = info->name,
      .extent = info->pixel_extent(),
      .pattern = Pattern::kImpulse,
      .strategy = strategy,
      .force_single_strategy = true,
    });
  }
  return fixtures;
}

bool RunFixture(
  const Fixture& fixture,
  const Options& options,
  const fs::path& directory,
  std::vector<uint8_t>* corruption_source) {

  std::error_code fs_error;
  fs::create_directories(directory, fs_error);
  if (fs_error) {
    std::cerr << fixture.name << ": cannot create artifact directory: "
              << fs_error.message() << '\n';
    return false;
  }

  PreparedFixture prepared;
  gjxl::Status status = PrepareFixture(fixture, &prepared);
  if (!status.ok()) {
    std::cerr << fixture.name << ": frame preparation failed: "
              << status.message() << '\n';
    return false;
  }
  if (fixture.require_large_tokens &&
      (prepared.maximum_abs_dc <= std::numeric_limits<uint16_t>::max() ||
       prepared.maximum_abs_ac <= std::numeric_limits<uint16_t>::max())) {
    std::cerr << fixture.name << ": large-token fixture is too small: dc="
              << prepared.maximum_abs_dc << " ac=" << prepared.maximum_abs_ac
              << '\n';
    return false;
  }

  std::vector<uint8_t> codestream;
  status = gjxl::EncodeVarDctCodestream(prepared.frame, &codestream);
  if (!status.ok()) {
    std::cerr << fixture.name << ": codestream encoding failed: "
              << status.message() << '\n';
    return false;
  }
  const uint64_t hash = Fnv1a64(codestream);
  if (fixture.expected_hash != 0 && hash != fixture.expected_hash) {
    WriteBytes(directory / "encoded.jxl", codestream);
    std::cerr << fixture.name << ": codestream hash changed: " << hash << '\n';
    return false;
  }

  const fs::path compressed = directory / "encoded.jxl";
  const fs::path expected = directory / "expected.pfm";
  const fs::path decoded = directory / "decoded.pfm";
  const fs::path decoder_log = directory / "djxl.log";
  const fs::path info_log = directory / "jxlinfo.log";
  if (!WriteBytes(compressed, codestream) ||
      !WritePfm(expected, prepared.expected.const_view())) {
    std::cerr << fixture.name << ": cannot write diagnostic inputs\n";
    return false;
  }

  const std::array<std::string, 4> decode_arguments = {
    "--quiet", "--num_threads=0", compressed.string(), decoded.string()};
  const int decoder_exit =
    RunTool(options.decoder, decode_arguments, decoder_log);
  if (decoder_exit != 0) {
    std::cerr << fixture.name << ": djxl failed with exit "
              << decoder_exit << '\n';
    return false;
  }

  const std::array<std::string, 2> info_arguments = {
    "-v", compressed.string()};
  const int info_exit = RunTool(options.info, info_arguments, info_log);
  std::string info_text;
  std::string error;
  if (info_exit != 0 || !ReadText(info_log, &info_text) ||
      !ContainsMetadata(info_text, fixture, &error)) {
    std::cerr << fixture.name << ": metadata check failed: "
              << (error.empty() ? "jxlinfo failed" : error) << '\n';
    return false;
  }

  PfmImage decoded_image;
  if (!ReadPfm(decoded, &decoded_image, &error)) {
    std::cerr << fixture.name << ": cannot read decoded pixels: "
              << error << '\n';
    return false;
  }
  Comparison comparison;
  if (!ComparePixels(
        prepared.expected.const_view(), decoded_image, &comparison, &error)) {
    std::cerr << fixture.name << ": " << error << '\n';
    return false;
  }
  std::cout << fixture.name << ": " << codestream.size()
            << " bytes, max_error=" << std::setprecision(6)
            << comparison.maximum_absolute_error
            << ", tolerance_ratio=" << comparison.maximum_tolerance_ratio
            << ", rms=" << comparison.root_mean_square_error << '\n';

  if (corruption_source != nullptr && corruption_source->empty()) {
    *corruption_source = codestream;
  }
  return true;
}

bool CheckCorruption(
  std::string_view name,
  std::vector<uint8_t> bytes,
  const Options& options,
  const fs::path& directory) {

  std::error_code fs_error;
  fs::create_directories(directory, fs_error);
  if (fs_error) {
    return false;
  }
  const fs::path compressed = directory / (std::string(name) + ".jxl");
  const fs::path decoded = directory / (std::string(name) + ".pfm");
  const fs::path log = directory / (std::string(name) + ".log");
  if (!WriteBytes(compressed, bytes)) {
    return false;
  }
  const std::array<std::string, 4> arguments = {
    "--quiet", "--num_threads=0", compressed.string(), decoded.string()};
  const int exit_code = RunTool(options.decoder, arguments, log);
  if (exit_code == 0 || fs::exists(decoded)) {
    std::cerr << "Corruption " << name
              << " masqueraded as a successful decode\n";
    return false;
  }
  return true;
}

bool RunCorruptionChecks(
  const std::vector<uint8_t>& source,
  const Options& options,
  const fs::path& directory) {

  if (source.size() < 8) {
    return false;
  }
  std::vector<uint8_t> signature = source;
  signature[0] = 0;
  std::vector<uint8_t> truncated = source;
  truncated.resize(source.size() / 2);
  return CheckCorruption(
           "bad-signature", std::move(signature), options, directory) &&
         CheckCorruption(
           "truncated", std::move(truncated), options, directory);
}

bool ParseOptions(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    const auto take_path = [&](fs::path* path) {
      if (index + 1 >= argc) {
        return false;
      }
      *path = argv[++index];
      return true;
    };
    if (argument == "--decoder") {
      if (!take_path(&options->decoder)) {
        return false;
      }
    } else if (argument == "--info") {
      if (!take_path(&options->info)) {
        return false;
      }
    } else if (argument == "--artifacts") {
      if (!take_path(&options->artifacts)) {
        return false;
      }
    } else if (argument == "--smoke") {
      options->smoke = true;
    } else {
      return false;
    }
  }
  return !options->decoder.empty() && !options->info.empty() &&
    !options->artifacts.empty();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    std::cerr << "Usage: " << argv[0]
              << " --decoder DJXL --info JXLINFO --artifacts DIR [--smoke]\n";
    return EXIT_FAILURE;
  }
  if (!fs::exists(options.decoder) || !fs::exists(options.info)) {
    std::cerr << "Decoder tools do not exist\n";
    return EXIT_FAILURE;
  }

  const fs::path run_directory =
    options.artifacts / ("run-" + std::to_string(getpid()));
  std::error_code fs_error;
  fs::create_directories(run_directory, fs_error);
  if (fs_error) {
    std::cerr << "Cannot create artifact root: " << fs_error.message() << '\n';
    return EXIT_FAILURE;
  }

  const std::vector<Fixture> fixtures =
    options.smoke ? SmokeFixtures() : FullFixtures();
  bool success = true;
  std::vector<uint8_t> corruption_source;
  for (const Fixture& fixture : fixtures) {
    success &= RunFixture(
      fixture, options, run_directory / fixture.name, &corruption_source);
  }
  success &= RunCorruptionChecks(
    corruption_source, options, run_directory / "corruption");

  if (!success) {
    std::cerr << "Conformance artifacts preserved at "
              << run_directory << '\n';
    return EXIT_FAILURE;
  }

  fs::remove_all(run_directory, fs_error);
  if (fs_error) {
    std::cerr << "Conformance passed, but scratch cleanup failed: "
              << fs_error.message() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "All " << fixtures.size()
            << (options.smoke ? " smoke" : " pinned")
            << " codestream conformance fixtures passed.\n";
  return EXIT_SUCCESS;
}
