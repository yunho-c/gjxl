// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "codestream/workflow.h"
#include "core/image.h"

namespace {

constexpr gjxl::Extent2D kExtent{17, 9};
constexpr size_t kStride = kExtent.width + 3;

struct ImageStorage {
  ImageStorage() {
    for (std::vector<float>& values : plane) {
      values.assign(kStride * kExtent.height, -777.0f);
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView View() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[1].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }

  std::array<std::vector<float>, 3> plane;
};

void FillImage(ImageStorage* image) {
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      const float fx = static_cast<float>(x) /
        static_cast<float>(kExtent.width - 1);
      const float fy = static_cast<float>(y) /
        static_cast<float>(kExtent.height - 1);
      const float texture = ((3 * x + 5 * y) % 7) * (1.0f / 60.0f);
      image->plane[0][y * kStride + x] = 0.05f + 0.75f * fx;
      image->plane[1][y * kStride + x] = 0.04f + 0.68f * fy + texture;
      image->plane[2][y * kStride + x] =
        0.03f + 0.35f * fx + 0.42f * fy;
    }
  }
}

uint64_t Fnv1a64(const std::vector<uint8_t>& bytes) {
  uint64_t hash = 14695981039346656037ull;
  for (uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

bool CheckDeterministicWorkflow() {
  ImageStorage image;
  FillImage(&image);

  std::vector<uint8_t> first;
  gjxl::VarDctEncodingSummary first_summary;
  gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), {.butteraugli_target = 1.0f}, &first, &first_summary);
  if (!status.ok() || first.size() < 2 || first[0] != 0xff ||
      first[1] != 0x0a || first_summary.extent != kExtent ||
      first_summary.encoded_bytes != first.size() ||
      first_summary.rate_control_mode !=
          gjxl::VarDctRateControlMode::kButteraugliTarget ||
      first_summary.requested_target_bytes != 0 ||
      first_summary.effective_target_bytes != 0 ||
      first_summary.target_size_tolerance_bytes != 0 ||
      first_summary.requested_target_bits_per_pixel != 0.0 ||
      first_summary.achieved_bits_per_pixel !=
          8.0 * static_cast<double>(first.size()) /
            static_cast<double>(kExtent.width * kExtent.height) ||
      first_summary.selected_butteraugli_target != 1.0f ||
      first_summary.encode_attempt_count != 1 ||
      first_summary.target_size_met ||
      first_summary.score_history.size() != 3 ||
      first_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kCpu) {
    std::cerr << "Public workflow failed: " << status.message() << '\n';
    return false;
  }
  const size_t strategy_count = std::accumulate(
    first_summary.strategy_counts.begin(),
    first_summary.strategy_counts.end(),
    size_t{0});
  if (strategy_count == 0 ||
      !std::ranges::all_of(
        first_summary.score_history,
        [](double score) { return std::isfinite(score) && score >= 0.0; })) {
    std::cerr << "Public workflow summary is invalid\n";
    return false;
  }

  std::vector<uint8_t> second;
  gjxl::VarDctEncodingSummary second_summary;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), {.butteraugli_target = 1.0f}, &second, &second_summary);
  if (!status.ok() || first != second || first_summary != second_summary) {
    std::cerr << "Public workflow is not deterministic\n";
    return false;
  }

  const uint64_t hash = Fnv1a64(first);
  constexpr uint64_t kExpectedHash = 15338072593505811851ull;
  if (hash != kExpectedHash) {
    std::cerr << "Public workflow hash changed: " << hash << '\n';
    return false;
  }
  std::cout << "Public workflow bytes=" << first.size()
            << " hash=" << hash << " strategies=" << strategy_count
            << '\n';
  return true;
}

bool CheckInvalidRequestsAreAtomic() {
  ImageStorage image;
  FillImage(&image);
  const std::vector<uint8_t> original_bytes = {3, 1, 4};
  const gjxl::VarDctEncodingSummary original_summary{
    .extent = {7, 5},
    .encoded_bytes = 19,
    .strategy_counts = {},
    .score_history = {2.0, 1.0},
  };

  const auto rejected_atomically = [&](gjxl::ConstImage3FView input,
                                       gjxl::VarDctEncodingOptions options) {
    std::vector<uint8_t> bytes = original_bytes;
    gjxl::VarDctEncodingSummary summary = original_summary;
    const gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
      input, options, &bytes, &summary);
    return !status.ok() && bytes == original_bytes &&
      summary == original_summary;
  };

  const auto unavailable_atomically = [&](gjxl::VarDctEncodingOptions options) {
    std::vector<uint8_t> bytes = original_bytes;
    gjxl::VarDctEncodingSummary summary = original_summary;
    const gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
      image.View(), options, &bytes, &summary);
    return status.code() == gjxl::StatusCode::kUnavailable &&
      bytes == original_bytes && summary == original_summary;
  };

  if (!rejected_atomically(image.View(), {.butteraugli_target = 0.0f}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target =
             std::numeric_limits<float>::quiet_NaN()}) ||
      !rejected_atomically({}, {.butteraugli_target = 1.0f}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             static_cast<gjxl::VarDctRateControlMode>(99)}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             gjxl::VarDctRateControlMode::kMaximumError}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             gjxl::VarDctRateControlMode::kMaximumError,
           .maximum_error = {1.0f, 0.0f, 1.0f}}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             gjxl::VarDctRateControlMode::kMaximumError,
           .maximum_error = {
             1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f}}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             gjxl::VarDctRateControlMode::kTargetBytes}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             gjxl::VarDctRateControlMode::kTargetBitsPerPixel}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             gjxl::VarDctRateControlMode::kTargetBitsPerPixel,
           .target_bits_per_pixel =
             std::numeric_limits<double>::quiet_NaN()}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             gjxl::VarDctRateControlMode::kTargetBitsPerPixel,
           .target_bits_per_pixel = 1.0e-9}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode =
             gjxl::VarDctRateControlMode::kTargetBitsPerPixel,
           .target_bits_per_pixel =
             std::numeric_limits<double>::max()}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
           .target_bytes = 1024,
           .target_size_tolerance =
             std::numeric_limits<double>::quiet_NaN()}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
           .target_bytes = 1024,
           .target_size_tolerance = -0.1}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
           .target_bytes = 1024,
           .target_size_tolerance = 1.1}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
           .target_bytes = 1024,
           .target_size_maximum_attempts = 0}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
           .target_bytes = 1024,
           .target_size_maximum_attempts = 65}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = static_cast<gjxl::VarDctBackendPreference>(99)}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kCpu,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kFullyResident}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kAutomatic,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kFullyResident}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kMetal,
           .metal_aq_mode =
               static_cast<gjxl::GpuAdaptiveQuantizationMode>(99)})) {
    std::cerr << "Invalid workflow request changed output\n";
    return false;
  }

  const float nan = std::numeric_limits<float>::quiet_NaN();
  if (!unavailable_atomically(
        {.butteraugli_target = nan,
         .rate_control_mode = gjxl::VarDctRateControlMode::kMaximumError,
         .maximum_error = {1.0f, 2.0f, 3.0f}})) {
    std::cerr << "Valid unavailable rate-control mode changed output\n";
    return false;
  }

  gjxl::VarDctEncodingOptions inactive_fields;
  inactive_fields.butteraugli_target = 1.0f;
  inactive_fields.maximum_error = {nan, -1.0f, 0.0f};
  inactive_fields.target_bytes = 1024;
  inactive_fields.target_bits_per_pixel =
    std::numeric_limits<double>::quiet_NaN();
  inactive_fields.target_size_tolerance =
    std::numeric_limits<double>::quiet_NaN();
  inactive_fields.target_size_maximum_attempts = 0;
  std::vector<uint8_t> baseline_bytes;
  gjxl::VarDctEncodingSummary baseline_summary;
  const gjxl::Status baseline_status =
    gjxl::EncodeLinearRgbVarDctCodestream(
      image.View(), {}, &baseline_bytes, &baseline_summary);
  std::vector<uint8_t> inactive_bytes;
  gjxl::VarDctEncodingSummary inactive_summary;
  const gjxl::Status inactive_status =
    gjxl::EncodeLinearRgbVarDctCodestream(
      image.View(), inactive_fields, &inactive_bytes, &inactive_summary);
  if (!baseline_status.ok() || !inactive_status.ok() ||
      inactive_bytes != baseline_bytes ||
      inactive_summary != baseline_summary) {
    std::cerr << "Inactive rate-control fields affected the request\n";
    return false;
  }

  image.plane[1][3 * kStride + 4] =
    std::numeric_limits<float>::quiet_NaN();
  if (!rejected_atomically(image.View(), {.butteraugli_target = 1.0f}) ||
      gjxl::EncodeLinearRgbVarDctCodestream(
        image.View(), {.butteraugli_target = 1.0f}, nullptr).ok()) {
    std::cerr << "Non-finite or null-output workflow request was accepted\n";
    return false;
  }
  return true;
}

bool CheckTargetSizeControl() {
  ImageStorage image;
  FillImage(&image);
  constexpr size_t kTargetBytes = 280;
  constexpr double kTolerance = 0.1;
  constexpr size_t kMaximumAttempts = 8;

  gjxl::VarDctEncodingOptions byte_options;
  byte_options.butteraugli_target =
    std::numeric_limits<float>::quiet_NaN();
  byte_options.rate_control_mode =
    gjxl::VarDctRateControlMode::kTargetBytes;
  byte_options.target_bytes = kTargetBytes;
  byte_options.target_bits_per_pixel =
    std::numeric_limits<double>::quiet_NaN();
  byte_options.target_size_tolerance = kTolerance;
  byte_options.target_size_maximum_attempts = kMaximumAttempts;
  byte_options.backend = gjxl::VarDctBackendPreference::kCpu;

  std::vector<uint8_t> byte_codestream;
  gjxl::VarDctEncodingSummary byte_summary;
  gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), byte_options, &byte_codestream, &byte_summary);
  if (!status.ok() || byte_codestream.empty() ||
      byte_summary.rate_control_mode !=
        gjxl::VarDctRateControlMode::kTargetBytes ||
      byte_summary.requested_target_bytes != kTargetBytes ||
      byte_summary.effective_target_bytes != kTargetBytes ||
      byte_summary.target_size_tolerance_bytes != 28 ||
      byte_summary.encoded_bytes != byte_codestream.size() ||
      byte_summary.encoded_bytes > kTargetBytes ||
      kTargetBytes - byte_summary.encoded_bytes >
        byte_summary.target_size_tolerance_bytes ||
      !byte_summary.target_size_met ||
      byte_summary.encode_attempt_count == 0 ||
      byte_summary.encode_attempt_count > kMaximumAttempts ||
      !std::isfinite(byte_summary.selected_butteraugli_target) ||
      byte_summary.selected_butteraugli_target <= 0.0f) {
    std::cerr << "Target-byte workflow failed: " << status.message() << '\n';
    return false;
  }

  std::vector<uint8_t> repeated_codestream;
  gjxl::VarDctEncodingSummary repeated_summary;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), byte_options, &repeated_codestream, &repeated_summary);
  if (!status.ok() || repeated_codestream != byte_codestream ||
      repeated_summary != byte_summary) {
    std::cerr << "Target-byte workflow is not deterministic\n";
    return false;
  }

  gjxl::VarDctEncodingOptions bpp_options = byte_options;
  bpp_options.rate_control_mode =
    gjxl::VarDctRateControlMode::kTargetBitsPerPixel;
  bpp_options.target_bytes = 0;
  bpp_options.target_bits_per_pixel =
    (static_cast<double>(kTargetBytes) + 0.5) * 8.0 /
    static_cast<double>(kExtent.width * kExtent.height);
  std::vector<uint8_t> bpp_codestream;
  gjxl::VarDctEncodingSummary bpp_summary;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), bpp_options, &bpp_codestream, &bpp_summary);
  if (!status.ok() || bpp_codestream != byte_codestream ||
      bpp_summary.rate_control_mode !=
        gjxl::VarDctRateControlMode::kTargetBitsPerPixel ||
      bpp_summary.requested_target_bytes != 0 ||
      bpp_summary.effective_target_bytes != kTargetBytes ||
      bpp_summary.requested_target_bits_per_pixel !=
        bpp_options.target_bits_per_pixel ||
      bpp_summary.selected_butteraugli_target !=
        byte_summary.selected_butteraugli_target ||
      bpp_summary.encode_attempt_count != byte_summary.encode_attempt_count ||
      !bpp_summary.target_size_met) {
    std::cerr << "Target-BPP workflow did not normalize to the byte budget\n";
    return false;
  }

  gjxl::VarDctEncodingOptions oversized = byte_options;
  oversized.target_bytes = 100000;
  oversized.target_size_tolerance = 0.0;
  std::vector<uint8_t> oversized_codestream;
  gjxl::VarDctEncodingSummary oversized_summary;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), oversized, &oversized_codestream, &oversized_summary);
  if (!status.ok() || oversized_codestream.empty() ||
      oversized_summary.encoded_bytes >= oversized.target_bytes ||
      oversized_summary.encode_attempt_count != 1 ||
      oversized_summary.target_size_met) {
    std::cerr << "Oversized byte target was not reported as infeasible\n";
    return false;
  }

  gjxl::VarDctEncodingOptions undersized = byte_options;
  undersized.target_bytes = 1;
  undersized.target_size_tolerance = 0.0;
  undersized.target_size_maximum_attempts = 2;
  std::vector<uint8_t> undersized_codestream;
  gjxl::VarDctEncodingSummary undersized_summary;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), undersized, &undersized_codestream, &undersized_summary);
  if (!status.ok() || undersized_codestream.size() <= undersized.target_bytes ||
      undersized_summary.encode_attempt_count != 2 ||
      undersized_summary.target_size_met) {
    std::cerr << "Undersized byte target was not reported as infeasible\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckDeterministicWorkflow() ||
      !CheckInvalidRequestsAreAtomic() ||
      !CheckTargetSizeControl()) {
    return EXIT_FAILURE;
  }
  std::cout << "All public codestream workflow tests passed.\n";
  return EXIT_SUCCESS;
}
