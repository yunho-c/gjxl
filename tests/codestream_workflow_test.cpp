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
#include "codestream/workflow_internal.h"
#include "core/image.h"
#include "gpu/metal/metal_butteraugli_test.h"

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

struct ArbitraryImageStorage {
  explicit ArbitraryImageStorage(gjxl::Extent2D image_extent)
    : extent(image_extent) {
    for (std::vector<float>& values : plane) {
      values.resize(extent.width * extent.height);
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView View() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), extent, extent.width},
      gjxl::ConstPlaneF32View{plane[1].data(), extent, extent.width},
      gjxl::ConstPlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  gjxl::Extent2D extent;
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

bool CheckQuantizationMatrixScaleStats() {
  ArbitraryImageStorage image({2, 2});
  image.plane[0] = {0.0f, -0.1f, 0.05f, 0.2f};
  image.plane[1] = {0.0f, 0.1f, 0.2f, 0.1f};
  image.plane[2] = {0.0f, 0.3f, 0.2f, 0.5f};
  gjxl::codestream_internal::QuantizationMatrixScaleStats stats;
  gjxl::Status status =
    gjxl::codestream_internal::ComputeQuantizationMatrixScaleStats(
      image.View(), &stats);
  if (!status.ok() || std::abs(stats.x_edge - 0.3f) > 1.0e-6f ||
      std::abs(stats.b_edge - 0.4f) > 1.0e-6f ||
      std::abs(stats.exposed_blue - 0.19f) > 1.0e-6f) {
    std::cerr << "Matrix-scale pixel statistics are incorrect\n";
    return false;
  }
  gjxl::codestream_internal::QuantizationMatrixScaleStats finite_stats;
  status = gjxl::codestream_internal::
    ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
      image.View(), &finite_stats);
  if (!status.ok() || finite_stats != stats) {
    std::cerr << "Finite matrix-scale statistics differ from checked path\n";
    return false;
  }

  ImageStorage strided;
  FillImage(&strided);
  status = gjxl::codestream_internal::ComputeQuantizationMatrixScaleStats(
    strided.View(), &stats);
  if (!status.ok() ||
      !gjxl::codestream_internal::
        ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
          strided.View(), &finite_stats).ok() ||
      finite_stats != stats) {
    std::cerr << "Strided finite matrix-scale statistics differ\n";
    return false;
  }

  ArbitraryImageStorage vector_tail({18, 7});
  for (size_t y = 0; y < vector_tail.extent.height; ++y) {
    for (size_t x = 0; x < vector_tail.extent.width; ++x) {
      const size_t index = y * vector_tail.extent.width + x;
      vector_tail.plane[0][index] = 0.01f * static_cast<float>(
        static_cast<int>((3 * x + 5 * y) % 19) - 9);
      vector_tail.plane[1][index] = 0.02f * static_cast<float>(
        static_cast<int>((7 * x + 2 * y) % 17) - 8);
      vector_tail.plane[2][index] = 0.03f * static_cast<float>(
        static_cast<int>((5 * x + 11 * y) % 23) - 11);
    }
  }
  status = gjxl::codestream_internal::ComputeQuantizationMatrixScaleStats(
    vector_tail.View(), &stats);
  if (!status.ok() ||
      !gjxl::codestream_internal::
        ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
          vector_tail.View(), &finite_stats).ok() ||
      finite_stats != stats) {
    std::cerr << "Vector-tail finite matrix-scale statistics differ\n";
    return false;
  }

  ArbitraryImageStorage single_pixel({1, 1});
  single_pixel.plane[0][0] = 0.2f;
  single_pixel.plane[1][0] = 0.1f;
  single_pixel.plane[2][0] = 0.4f;
  status = gjxl::codestream_internal::ComputeQuantizationMatrixScaleStats(
    single_pixel.View(), &stats);
  if (!status.ok() ||
      !gjxl::codestream_internal::
        ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
          single_pixel.View(), &finite_stats).ok() ||
      stats !=
        gjxl::codestream_internal::QuantizationMatrixScaleStats{} ||
      finite_stats != stats) {
    std::cerr << "Degenerate matrix-scale statistics are incorrect\n";
    return false;
  }

  const gjxl::codestream_internal::QuantizationMatrixScaleStats sentinel{
    .x_edge = 1.0f, .b_edge = 2.0f, .exposed_blue = 3.0f};
  stats = sentinel;
  finite_stats = sentinel;
  image.plane[1][1] = std::numeric_limits<float>::quiet_NaN();
  ArbitraryImageStorage overflowing({2, 2});
  overflowing.plane[0] = {
    0.0f, 0.0f, -std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max()};
  if (gjxl::codestream_internal::ComputeQuantizationMatrixScaleStats(
        image.View(), &stats).code() != gjxl::StatusCode::kInvalidArgument ||
      stats != sentinel ||
      gjxl::codestream_internal::ComputeQuantizationMatrixScaleStats(
        overflowing.View(), &stats).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      stats != sentinel ||
      gjxl::codestream_internal::ComputeQuantizationMatrixScaleStats(
        {}, &stats).code() != gjxl::StatusCode::kInvalidArgument ||
      stats != sentinel ||
      gjxl::codestream_internal::ComputeQuantizationMatrixScaleStats(
        single_pixel.View(), nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      gjxl::codestream_internal::
        ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
          {}, &finite_stats).code() != gjxl::StatusCode::kInvalidArgument ||
      finite_stats != sentinel ||
      gjxl::codestream_internal::
        ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
          single_pixel.View(), nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Invalid matrix-scale statistics changed output\n";
    return false;
  }
  return true;
}

bool CheckQuantizationMatrixScaleSelection() {
  using gjxl::VarDctRateControlMode;
  using gjxl::codestream_internal::QuantizationMatrixScaleStats;
  using gjxl::codestream_internal::QuantizationMatrixScales;

  const auto selects = [](
    QuantizationMatrixScaleStats stats,
    float target,
    QuantizationMatrixScales expected) {
    QuantizationMatrixScales scales{.x = 7, .b = 7};
    const gjxl::Status status =
      gjxl::codestream_internal::SelectQuantizationMatrixScales(
        stats, VarDctRateControlMode::kButteraugliTarget, target, &scales);
    return status.ok() && scales == expected;
  };
  const float above_2_5 = std::nextafter(2.5f, 3.0f);
  const float above_5_5 = std::nextafter(5.5f, 6.0f);
  const float above_9_5 = std::nextafter(9.5f, 10.0f);
  if (!selects({}, 2.5f, {.x = 3, .b = 2}) ||
      !selects({}, above_2_5, {.x = 4, .b = 2}) ||
      !selects({}, 5.5f, {.x = 4, .b = 2}) ||
      !selects({}, above_5_5, {.x = 5, .b = 2}) ||
      !selects({}, 9.5f, {.x = 5, .b = 2}) ||
      !selects({}, above_9_5, {.x = 6, .b = 2}) ||
      !selects(
        {.x_edge = std::nextafter(0.022f, 0.0f)}, 1.0f,
        {.x = 3, .b = 2}) ||
      !selects({.x_edge = 0.015f}, 1.0f, {.x = 3, .b = 2}) ||
      !selects({.x_edge = 0.022f}, 1.0f, {.x = 4, .b = 2}) ||
      !selects(
        {.x_edge = std::nextafter(0.026f, 0.0f)}, 1.0f,
        {.x = 4, .b = 2}) ||
      !selects({.x_edge = 0.026f}, 1.0f, {.x = 5, .b = 2}) ||
      !selects({.b_edge = 0.33f}, 1.0f, {.x = 3, .b = 2}) ||
      !selects(
        {.b_edge = std::nextafter(0.33f, 1.0f)}, 1.0f,
        {.x = 3, .b = 3}) ||
      !selects({.b_edge = 0.38f}, 1.0f, {.x = 3, .b = 3}) ||
      !selects(
        {.b_edge = std::nextafter(0.38f, 1.0f)}, 1.0f,
        {.x = 3, .b = 4}) ||
      !selects(
        {.b_edge = 0.28f, .exposed_blue = 0.13f}, 1.0f,
        {.x = 3, .b = 2}) ||
      !selects(
        {.b_edge = std::nextafter(0.28f, 1.0f),
         .exposed_blue = 0.13f},
        1.0f, {.x = 3, .b = 3}) ||
      !selects(
        {.b_edge = 0.34f,
         .exposed_blue = std::nextafter(0.13f, 0.0f)},
        1.0f, {.x = 3, .b = 3}) ||
      !selects(
        {.b_edge = 0.34f, .exposed_blue = 0.13f},
        1.0f, {.x = 3, .b = 4}) ||
      !selects(
        {.b_edge = std::nextafter(0.38f, 1.0f),
         .exposed_blue = 0.13f},
        1.0f, {.x = 3, .b = 5})) {
    std::cerr << "Matrix-scale selection thresholds are incorrect\n";
    return false;
  }

  QuantizationMatrixScales scales{.x = 7, .b = 7};
  if (!gjxl::codestream_internal::SelectQuantizationMatrixScales(
         {}, VarDctRateControlMode::kMaximumError,
         std::numeric_limits<float>::quiet_NaN(), &scales).ok() ||
      scales != QuantizationMatrixScales{}) {
    std::cerr << "Maximum-error mode did not retain 2/2 matrix scales\n";
    return false;
  }

  const QuantizationMatrixScales sentinel{.x = 6, .b = 7};
  scales = sentinel;
  if (gjxl::codestream_internal::SelectQuantizationMatrixScales(
        {.x_edge = std::numeric_limits<float>::infinity()},
        VarDctRateControlMode::kButteraugliTarget, 1.0f, &scales).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      scales != sentinel ||
      gjxl::codestream_internal::SelectQuantizationMatrixScales(
        {.b_edge = -0.1f}, VarDctRateControlMode::kButteraugliTarget,
        1.0f, &scales).code() != gjxl::StatusCode::kInvalidArgument ||
      scales != sentinel ||
      gjxl::codestream_internal::SelectQuantizationMatrixScales(
        {}, VarDctRateControlMode::kTargetBytes, 1.0f, &scales).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      scales != sentinel ||
      gjxl::codestream_internal::SelectQuantizationMatrixScales(
        {}, VarDctRateControlMode::kButteraugliTarget, 1.0f, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Invalid matrix-scale selection changed output\n";
    return false;
  }
  return true;
}

bool CheckQuantizationMatrixScaleStatsPolicy() {
  using gjxl::VarDctCompressionMode;
  using gjxl::VarDctDensityMode;
  using gjxl::VarDctEncodingOptions;
  using gjxl::VarDctRateControlMode;
  using gjxl::codestream_internal::
    ShouldComputeQuantizationMatrixScaleStats;

  for (int32_t effort = 1; effort <= 6; ++effort) {
    if (ShouldComputeQuantizationMatrixScaleStats({.effort = effort})) {
      std::cerr << "Low effort " << effort
                << " retained matrix-scale pixel statistics\n";
      return false;
    }
  }
  for (int32_t effort = 7; effort <= 10; ++effort) {
    if (!ShouldComputeQuantizationMatrixScaleStats({.effort = effort})) {
      std::cerr << "Effort " << effort
                << " skipped matrix-scale pixel statistics\n";
      return false;
    }
  }

  const VarDctEncodingOptions high_density{
    .effort = 1,
    .density_mode = VarDctDensityMode::kHighDensity,
  };
  const VarDctEncodingOptions maximum_error{
    .effort = 10,
    .rate_control_mode = VarDctRateControlMode::kMaximumError,
    .maximum_error = {0.1f, 0.1f, 0.1f},
  };
  const VarDctEncodingOptions maximum_compression_low_effort{
    .effort = 1,
    .compression_mode = VarDctCompressionMode::kMaximumCompression,
  };
  const VarDctEncodingOptions maximum_compression_effort_7{
    .effort = 7,
    .compression_mode = VarDctCompressionMode::kMaximumCompression,
  };
  if (!ShouldComputeQuantizationMatrixScaleStats(high_density) ||
      ShouldComputeQuantizationMatrixScaleStats(maximum_error) ||
      ShouldComputeQuantizationMatrixScaleStats(
        maximum_compression_low_effort) ||
      !ShouldComputeQuantizationMatrixScaleStats(
        maximum_compression_effort_7)) {
    std::cerr << "Matrix-scale statistics policy overrides are incorrect\n";
    return false;
  }
  return true;
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
      !first_summary.final_butteraugli_score_evaluated ||
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

  std::vector<uint8_t> profiled;
  gjxl::VarDctEncodingSummary profiled_summary;
  gjxl::codestream_internal::VarDctEncodingProfile profile;
  status = gjxl::codestream_internal::
    EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
      image.View(),
      {.butteraugli_target = 1.0f,
       .backend = gjxl::VarDctBackendPreference::kCpu},
      nullptr, false, &profiled, &profiled_summary, &profile);
  const uint64_t profile_stage_total =
    profile.input_preparation_nanoseconds +
    profile.backend_selection_nanoseconds +
    profile.quantization_pipeline_nanoseconds +
    profile.codestream_encoding_nanoseconds +
    profile.summary_assembly_nanoseconds;
  if (!status.ok() || profiled != first || profiled_summary != first_summary ||
      profile.execution_backend != gjxl::VarDctExecutionBackend::kCpu ||
      profile_stage_total == 0 || profile.total_nanoseconds < profile_stage_total ||
      profile.codestream.total_nanoseconds == 0 ||
      profile.codestream_encoding_nanoseconds <
        profile.codestream.total_nanoseconds) {
    std::cerr << "Profiled public workflow changed its result or profile\n";
    return false;
  }

  const uint64_t hash = Fnv1a64(first);
  constexpr uint64_t kExpectedHash = 6720271014152865219ull;
  if (hash != kExpectedHash) {
    std::cerr << "Public workflow hash changed: " << hash << '\n';
    return false;
  }
  std::cout << "Public workflow bytes=" << first.size()
            << " hash=" << hash << " strategies=" << strategy_count
            << '\n';
  return true;
}

bool CheckCpuThreadBudget() {
  // Cross both the row-parallel threshold and the 256x256 codestream-group
  // boundary so the check covers nested and multi-group scheduling.
  constexpr gjxl::Extent2D kThreadedExtent{320, 272};
  ArbitraryImageStorage image(kThreadedExtent);
  for (size_t y = 0; y < kThreadedExtent.height; ++y) {
    for (size_t x = 0; x < kThreadedExtent.width; ++x) {
      const size_t index = y * kThreadedExtent.width + x;
      const float fx = static_cast<float>(x) / kThreadedExtent.width;
      const float fy = static_cast<float>(y) / kThreadedExtent.height;
      image.plane[0][index] = 0.05f + 0.75f * fx;
      image.plane[1][index] = 0.04f + 0.68f * fy;
      image.plane[2][index] = 0.03f + 0.35f * fx + 0.42f * fy;
    }
  }

  const gjxl::VarDctEncodingOptions automatic_options{
    .backend = gjxl::VarDctBackendPreference::kCpu,
  };
  std::vector<uint8_t> automatic;
  gjxl::VarDctEncodingSummary automatic_summary;
  gjxl::codestream_internal::VarDctEncodingProfile automatic_profile;
  gjxl::Status status = gjxl::codestream_internal::
    EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
      image.View(), automatic_options, nullptr, false, &automatic,
      &automatic_summary, &automatic_profile);
  if (!status.ok() || automatic.empty() ||
      automatic_profile.peak_cpu_participants == 0) {
    std::cerr << "Automatic CPU thread policy failed: " << status.message()
              << '\n';
    return false;
  }

  for (const size_t thread_count :
       {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
    gjxl::VarDctEncodingOptions options = automatic_options;
    options.cpu_thread_count = thread_count;
    std::vector<uint8_t> bytes;
    gjxl::VarDctEncodingSummary summary;
    gjxl::codestream_internal::VarDctEncodingProfile profile;
    status = gjxl::codestream_internal::
      EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
        image.View(), options, nullptr, false, &bytes, &summary, &profile);
    if (!status.ok() || bytes != automatic || summary != automatic_summary ||
        profile.peak_cpu_participants == 0 ||
        profile.peak_cpu_participants > thread_count) {
      std::cerr << "CPU thread budget " << thread_count
                << " changed the workflow result or exceeded its cap: "
                << status.message() << " peak="
                << profile.peak_cpu_participants << '\n';
      return false;
    }
  }
  return true;
}

bool CheckHighDensityMode() {
  ImageStorage image;
  FillImage(&image);
  const gjxl::VarDctEncodingOptions options{
    .butteraugli_target = 1.0f,
    .effort = 1,
    .density_mode = gjxl::VarDctDensityMode::kHighDensity,
    .backend = gjxl::VarDctBackendPreference::kCpu,
  };
  std::vector<uint8_t> first;
  std::vector<uint8_t> second;
  gjxl::VarDctEncodingSummary first_summary;
  gjxl::VarDctEncodingSummary second_summary;
  gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), options, &first, &first_summary);
  if (status.ok()) {
    status = gjxl::EncodeLinearRgbVarDctCodestream(
      image.View(), options, &second, &second_summary);
  }
  if (!status.ok() || first.empty() || first != second ||
      first_summary != second_summary ||
      first_summary.encoded_bytes != first.size() ||
      first_summary.density_mode !=
        gjxl::VarDctDensityMode::kHighDensity ||
      first_summary.score_history.size() != 5 ||
      !first_summary.final_butteraugli_score_evaluated ||
      first_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kCpu) {
    std::cerr << "High-density workflow failed: " << status.message()
              << " history=" << first_summary.score_history.size() << '\n';
    return false;
  }

  gjxl::VarDctEncodingOptions target_options = options;
  target_options.rate_control_mode =
    gjxl::VarDctRateControlMode::kTargetBytes;
  target_options.target_bytes = first.size();
  target_options.target_size_tolerance = 1.0;
  target_options.target_size_maximum_attempts = 1;
  std::vector<uint8_t> target_bytes;
  gjxl::VarDctEncodingSummary target_summary;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), target_options, &target_bytes, &target_summary);
  if (!status.ok() || target_bytes.empty() ||
      target_summary.density_mode !=
        gjxl::VarDctDensityMode::kHighDensity ||
      target_summary.encode_attempt_count != 1 ||
      target_summary.score_history.size() != 5 ||
      !target_summary.final_butteraugli_score_evaluated) {
    std::cerr << "High-density target-size workflow failed: "
              << status.message() << '\n';
    return false;
  }
  return true;
}

bool CheckEffortPolicy() {
  ImageStorage image;
  FillImage(&image);
  struct EffortCase {
    int32_t effort;
    size_t expected_score_count;
  };
  constexpr std::array<EffortCase, 10> kCases{{
    {1, 1},
    {2, 1},
    {3, 1},
    {4, 2},
    {5, 2},
    {6, 2},
    {7, 3},
    {8, 4},
    {9, 4},
    {10, 5},
  }};
  std::vector<uint8_t> default_bytes;
  gjxl::VarDctEncodingSummary default_summary;
  gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), {.backend = gjxl::VarDctBackendPreference::kCpu},
    &default_bytes, &default_summary);
  if (!status.ok()) {
    std::cerr << "Default effort workflow failed: " << status.message()
              << '\n';
    return false;
  }

  for (size_t index = 0; index < kCases.size(); ++index) {
    const EffortCase test = kCases[index];
    std::vector<uint8_t> bytes;
    gjxl::VarDctEncodingSummary summary;
    status = gjxl::EncodeLinearRgbVarDctCodestream(
      image.View(),
      {.effort = test.effort,
       .backend = gjxl::VarDctBackendPreference::kCpu},
      &bytes, &summary);
    if (!status.ok() || bytes.empty() ||
        summary.score_history.size() != test.expected_score_count ||
        !summary.final_butteraugli_score_evaluated ||
        summary.entropy_behavior !=
          (test.effort >= 9
             ? gjxl::VarDctEntropyBehavior::kHighDensity
             : gjxl::VarDctEntropyBehavior::kBalanced)) {
      std::cerr << "Effort " << test.effort << " workflow failed: "
                << status.message() << " history="
                << summary.score_history.size() << '\n';
      return false;
    }
    if (test.effort == 7 &&
        (bytes != default_bytes || summary != default_summary)) {
      std::cerr << "Explicit effort 7 changed the default workflow\n";
      return false;
    }
  }

  for (const int32_t effort : {1, 7, 10}) {
    const size_t index = static_cast<size_t>(effort - 1);
    const size_t expected_score_count = effort <= 3
      ? 1
      : kCases[index].expected_score_count - 1;
    std::vector<uint8_t> bytes;
    gjxl::VarDctEncodingSummary summary;
    status = gjxl::EncodeLinearRgbVarDctCodestream(
      image.View(),
      {.effort = effort,
       .backend = gjxl::VarDctBackendPreference::kMetal},
      &bytes, &summary);
    if (!status.ok() || bytes.empty() ||
        summary.score_history.size() != expected_score_count ||
        summary.final_butteraugli_score_evaluated != (effort <= 3) ||
        summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
        summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kFullyResident) {
      std::cerr << "Metal effort " << effort << " workflow failed: "
                << status.message() << " history="
                << summary.score_history.size() << '\n';
      return false;
    }
  }
  return true;
}

bool CheckCompressionPolicy() {
  using gjxl::VarDctCoefficientOrderBehavior;
  using gjxl::VarDctCompressionMode;
  using gjxl::VarDctDensityMode;
  using gjxl::VarDctEntropyBehavior;
  using gjxl::codestream_internal::ResolveEntropyBehavior;
  using gjxl::codestream_internal::ResolveCoefficientOrderBehavior;

  if (ResolveCoefficientOrderBehavior({.effort = 7}) !=
        VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled ||
      ResolveCoefficientOrderBehavior({.effort = 8}) !=
        VarDctCoefficientOrderBehavior::kFull ||
      ResolveCoefficientOrderBehavior(
        {.effort = 7,
         .density_mode = VarDctDensityMode::kHighDensity}) !=
        VarDctCoefficientOrderBehavior::kFull ||
      ResolveCoefficientOrderBehavior(
        {.effort = 7,
         .compression_mode =
           VarDctCompressionMode::kMaximumCompression}) !=
        VarDctCoefficientOrderBehavior::kFull) {
    std::cerr << "Coefficient-order effort resolution failed\n";
    return false;
  }

  for (const int32_t effort : {1, 7, 8}) {
    if (ResolveEntropyBehavior({.effort = effort}) !=
        VarDctEntropyBehavior::kBalanced) {
      std::cerr << "Effort " << effort
                << " did not resolve to balanced entropy\n";
      return false;
    }
  }
  for (const int32_t effort : {9, 10}) {
    if (ResolveEntropyBehavior({.effort = effort}) !=
        VarDctEntropyBehavior::kHighDensity) {
      std::cerr << "Effort " << effort
                << " did not resolve to high-density entropy\n";
      return false;
    }
  }
  if (ResolveEntropyBehavior(
        {.effort = 7, .density_mode = VarDctDensityMode::kHighDensity}) !=
        VarDctEntropyBehavior::kHighDensity ||
      ResolveEntropyBehavior(
        {.effort = 1,
         .compression_mode =
           VarDctCompressionMode::kMaximumCompression}) !=
        VarDctEntropyBehavior::kMaximumCompression) {
    std::cerr << "Explicit entropy policy resolution failed\n";
    return false;
  }

  ImageStorage image;
  FillImage(&image);
  std::vector<uint8_t> bytes;
  gjxl::VarDctEncodingSummary summary;
  const gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(),
    {.compression_mode = VarDctCompressionMode::kMaximumCompression,
     .backend = gjxl::VarDctBackendPreference::kCpu},
    &bytes, &summary);
  if (!status.ok() || bytes.empty() ||
      summary.compression_mode !=
        VarDctCompressionMode::kMaximumCompression ||
      summary.entropy_behavior !=
        VarDctEntropyBehavior::kMaximumCompression) {
    std::cerr << "Maximum-compression workflow failed: "
              << status.message() << '\n';
    return false;
  }
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
  gjxl::codestream_internal::VarDctEncodingProfile original_profile;
  original_profile.total_nanoseconds = 321;

  const auto rejected_atomically = [&](gjxl::ConstImage3FView input,
                                       gjxl::VarDctEncodingOptions options) {
    std::vector<uint8_t> bytes = original_bytes;
    gjxl::VarDctEncodingSummary summary = original_summary;
    const gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
      input, options, &bytes, &summary);
    return !status.ok() && bytes == original_bytes &&
      summary == original_summary;
  };

  if (!rejected_atomically(image.View(), {.butteraugli_target = 0.0f}) ||
      !rejected_atomically(image.View(), {.effort = 0}) ||
      !rejected_atomically(image.View(), {.effort = 11}) ||
      !rejected_atomically(
          image.View(),
          {.cpu_thread_count = gjxl::kMaximumCpuThreadCount + 1}) ||
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
           .rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
           .target_bytes = 1024,
           .target_size_selection =
             static_cast<gjxl::TargetSizeSelectionPolicy>(99)}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = static_cast<gjxl::VarDctBackendPreference>(99)}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .density_mode = static_cast<gjxl::VarDctDensityMode>(99)}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .compression_mode =
             static_cast<gjxl::VarDctCompressionMode>(99)}) ||
      !rejected_atomically(
          image.View(),
          {.density_mode = gjxl::VarDctDensityMode::kHighDensity,
           .rate_control_mode = gjxl::VarDctRateControlMode::kMaximumError,
           .maximum_error = {0.1f, 0.1f, 0.1f}}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .density_mode = gjxl::VarDctDensityMode::kHighDensity,
           .backend = gjxl::VarDctBackendPreference::kMetal,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kThroughput}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .density_mode = gjxl::VarDctDensityMode::kHighDensity,
           .backend = gjxl::VarDctBackendPreference::kMetal,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kCpu,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kThroughput}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kAutomatic,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kThroughput}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kCpu,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kAutomatic,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput}) ||
      !rejected_atomically(
          image.View(),
          {.rate_control_mode =
             gjxl::VarDctRateControlMode::kMaximumError,
           .maximum_error = {0.01f, 0.01f, 0.01f},
           .backend = gjxl::VarDctBackendPreference::kMetal,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput}) ||
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
  gjxl::VarDctEncodingOptions inactive_fields;
  inactive_fields.butteraugli_target = 1.0f;
  inactive_fields.maximum_error = {nan, -1.0f, 0.0f};
  inactive_fields.target_bytes = 1024;
  inactive_fields.target_bits_per_pixel =
    std::numeric_limits<double>::quiet_NaN();
  inactive_fields.target_size_tolerance =
    std::numeric_limits<double>::quiet_NaN();
  inactive_fields.target_size_maximum_attempts = 0;
  inactive_fields.target_size_selection =
    static_cast<gjxl::TargetSizeSelectionPolicy>(99);
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

  std::vector<uint8_t> profiled_bytes = original_bytes;
  gjxl::VarDctEncodingSummary profiled_summary = original_summary;
  auto profile = original_profile;
  if (gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
          image.View(), {.butteraugli_target = 0.0f}, nullptr, false,
          &profiled_bytes, &profiled_summary, &profile).ok() ||
      profiled_bytes != original_bytes || profiled_summary != original_summary ||
      profile != original_profile) {
    std::cerr << "Rejected profiled workflow changed output\n";
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

bool CheckMaximumErrorControl() {
  ImageStorage image;
  FillImage(&image);
  gjxl::VarDctEncodingOptions options;
  options.butteraugli_target =
    std::numeric_limits<float>::quiet_NaN();
  options.rate_control_mode =
    gjxl::VarDctRateControlMode::kMaximumError;
  options.maximum_error = {0.1f, 0.1f, 0.1f};
  options.backend = gjxl::VarDctBackendPreference::kCpu;

  std::vector<uint8_t> first;
  gjxl::VarDctEncodingSummary first_summary;
  gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), options, &first, &first_summary);
  if (!status.ok() || first.empty() ||
      first_summary.rate_control_mode !=
        gjxl::VarDctRateControlMode::kMaximumError ||
      first_summary.requested_maximum_error != options.maximum_error ||
      first_summary.maximum_error_evaluation_count != 6 ||
      first_summary.score_history.size() != 6 ||
      first_summary.final_butteraugli_score_evaluated ||
      first_summary.selected_butteraugli_target != 0.0f ||
      first_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kCpu ||
      first_summary.maximum_error_outcome !=
        gjxl::MaximumErrorOutcome::kMet ||
      first_summary.achieved_maximum_error_ratio > 1.0f) {
    std::cerr << "Maximum-error workflow failed: " << status.message()
              << " ratio="
              << first_summary.achieved_maximum_error_ratio
              << " outcome="
              << static_cast<int>(first_summary.maximum_error_outcome)
              << " history=";
    for (double score : first_summary.score_history) {
      std::cerr << score << ',';
    }
    std::cerr << '\n';
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (first_summary.achieved_maximum_error[channel] >
        options.maximum_error[channel]) {
      std::cerr << "Maximum-error channel limit was not met\n";
      return false;
    }
  }

  std::vector<uint8_t> repeated;
  gjxl::VarDctEncodingSummary repeated_summary;
  options.butteraugli_target = 42.0f;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), options, &repeated, &repeated_summary);
  if (!status.ok() || repeated != first ||
      repeated_summary != first_summary) {
    std::cerr << "Maximum-error workflow is not deterministic or used the "
                 "inactive Butteraugli target\n";
    return false;
  }
  return true;
}

bool CheckTargetSizeControl() {
  ImageStorage image;
  FillImage(&image);
  constexpr size_t kTargetBytes = 250;
  constexpr double kTolerance = 0.12;
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
      byte_summary.target_size_tolerance_bytes != 30 ||
      byte_summary.encoded_bytes != byte_codestream.size() ||
      byte_summary.encoded_bytes > kTargetBytes ||
      kTargetBytes - byte_summary.encoded_bytes >
        byte_summary.target_size_tolerance_bytes ||
      !byte_summary.target_size_met ||
      byte_summary.encode_attempt_count == 0 ||
      byte_summary.encode_attempt_count > kMaximumAttempts ||
      byte_summary.failed_encode_attempt_count != 0 ||
      byte_summary.target_size_selection !=
        gjxl::TargetSizeSelectionPolicy::kLargestAtOrBelow ||
      byte_summary.target_size_search_exhausted ||
      !std::isfinite(byte_summary.selected_butteraugli_target) ||
      byte_summary.selected_butteraugli_target <= 0.0f) {
    std::cerr << "Target-byte workflow failed: " << status.message()
              << ", bytes=" << byte_codestream.size()
              << ", met=" << byte_summary.target_size_met
              << ", attempts=" << byte_summary.encode_attempt_count
              << ", selected="
              << byte_summary.selected_butteraugli_target << '\n';
    return false;
  }

  std::vector<uint8_t> profiled_codestream;
  gjxl::VarDctEncodingSummary profiled_summary;
  gjxl::VarDctEncodingTiming timing;
  status = gjxl::EncodeLinearRgbVarDctCodestreamProfiled(
    image.View(), byte_options, &profiled_codestream, &profiled_summary,
    &timing);
  const uint64_t attempted_nanoseconds = std::accumulate(
    timing.attempts.begin(), timing.attempts.end(), uint64_t{0},
    [](uint64_t total, const gjxl::VarDctEncodingAttemptTiming& attempt) {
      return total + attempt.encode_and_serialize_nanoseconds;
    });
  const size_t successful_attempts = static_cast<size_t>(std::count_if(
    timing.attempts.begin(), timing.attempts.end(),
    [](const gjxl::VarDctEncodingAttemptTiming& attempt) {
      return attempt.succeeded;
    }));
  const bool selected_timing_present = std::ranges::any_of(
    timing.attempts,
    [&](const gjxl::VarDctEncodingAttemptTiming& attempt) {
      return attempt.succeeded &&
        attempt.butteraugli_target ==
          profiled_summary.selected_butteraugli_target &&
        attempt.encoded_bytes == profiled_codestream.size() &&
        attempt.encode_and_serialize_nanoseconds ==
          timing.selected_attempt_nanoseconds;
    });
  if (!status.ok() || profiled_codestream != byte_codestream ||
      profiled_summary != byte_summary ||
      timing.preparation_nanoseconds == 0 ||
      timing.aggregate_search_nanoseconds == 0 ||
      timing.selected_attempt_nanoseconds == 0 ||
      timing.total_nanoseconds < timing.preparation_nanoseconds ||
      timing.total_nanoseconds < timing.aggregate_search_nanoseconds ||
      timing.aggregate_search_nanoseconds < attempted_nanoseconds ||
      timing.attempts.size() != profiled_summary.encode_attempt_count ||
      successful_attempts + profiled_summary.failed_encode_attempt_count !=
        profiled_summary.encode_attempt_count ||
      !selected_timing_present) {
    std::cerr << "Profiled target-size workflow returned invalid timing\n";
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
      bpp_summary.target_size_selection !=
        byte_summary.target_size_selection ||
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
      oversized_summary.encode_attempt_count != kMaximumAttempts ||
      oversized_summary.target_size_met ||
      !oversized_summary.target_size_search_exhausted) {
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
      undersized_summary.target_size_met ||
      !undersized_summary.target_size_search_exhausted) {
    std::cerr << "Undersized byte target was not reported as infeasible\n";
    return false;
  }
  gjxl::VarDctEncodingOptions closest = undersized;
  closest.target_size_selection =
    gjxl::TargetSizeSelectionPolicy::kClosestAbsolute;
  std::vector<uint8_t> closest_codestream;
  gjxl::VarDctEncodingSummary closest_summary;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    image.View(), closest, &closest_codestream, &closest_summary);
  if (!status.ok() || closest_codestream.empty() ||
      closest_summary.target_size_selection !=
        gjxl::TargetSizeSelectionPolicy::kClosestAbsolute ||
      closest_summary.encode_attempt_count != 2 ||
      closest_summary.target_size_met ||
      !closest_summary.target_size_search_exhausted) {
    std::cerr << "Closest-absolute workflow integration failed\n";
    return false;
  }
  return true;
}

bool CheckPreparationCacheTrim() {
  if (!gjxl::TrimVarDctPreparationCache().ok() ||
      gjxl::MetalButteraugliProcessCacheBytesForTesting() != 0) return false;
  const auto available =
    gjxl::codestream_internal::EnsureProductionMetalBackendAvailable();
  if (!available.ok()) return available.code() == gjxl::StatusCode::kUnavailable;
  ArbitraryImageStorage image({32, 32});
  for (size_t c = 0; c < 3; ++c)
    for (size_t i = 0; i < image.plane[c].size(); ++i)
      image.plane[c][i] = 0.03f + 0.001f * ((i * 13 + c * 7) % 127);
  const gjxl::VarDctEncodingOptions options{
    .butteraugli_target = 1.2f,
    .effort = 7,
    .backend = gjxl::VarDctBackendPreference::kMetal};
  std::vector<uint8_t> before, after;
  if (!gjxl::EncodeLinearRgbVarDctCodestream(image.View(), options, &before).ok() ||
      gjxl::MetalButteraugliProcessCacheBytesForTesting() == 0 ||
      !gjxl::TrimVarDctPreparationCache().ok() ||
      gjxl::MetalButteraugliProcessCacheBytesForTesting() != 0 ||
      !gjxl::EncodeLinearRgbVarDctCodestream(image.View(), options, &after).ok() ||
      before != after ||
      gjxl::MetalButteraugliProcessCacheBytesForTesting() == 0) return false;
  return gjxl::TrimVarDctPreparationCache().ok();
}

bool CheckSingleAttemptTiming() {
  ImageStorage image;
  FillImage(&image);
  std::vector<uint8_t> profiled;
  gjxl::VarDctEncodingSummary summary;
  gjxl::VarDctEncodingTiming timing;
  const gjxl::Status status =
    gjxl::EncodeLinearRgbVarDctCodestreamProfiled(
      image.View(), {.butteraugli_target = 1.0f}, &profiled, &summary,
      &timing);
  if (!status.ok() || profiled.empty() ||
      timing.preparation_nanoseconds == 0 ||
      timing.aggregate_search_nanoseconds != 0 ||
      timing.selected_attempt_nanoseconds == 0 ||
      timing.total_nanoseconds < timing.preparation_nanoseconds ||
      timing.attempts.size() != 1 || !timing.attempts[0].succeeded ||
      timing.attempts[0].encoded_bytes != profiled.size() ||
      timing.attempts[0].butteraugli_target != 1.0f ||
      timing.attempts[0].encode_and_serialize_nanoseconds !=
        timing.selected_attempt_nanoseconds) {
    std::cerr << "Single-attempt workflow timing is invalid\n";
    return false;
  }

  std::vector<uint8_t> sentinel{3, 1, 4};
  const std::vector<uint8_t> original = sentinel;
  gjxl::VarDctEncodingSummary sentinel_summary{
    .extent = {7, 5}, .encoded_bytes = 19, .score_history = {2.0}};
  const gjxl::VarDctEncodingSummary original_summary = sentinel_summary;
  if (gjxl::EncodeLinearRgbVarDctCodestreamProfiled(
        image.View(), {}, &sentinel, &sentinel_summary, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      sentinel != original || sentinel_summary != original_summary) {
    std::cerr << "Null timing output was not rejected atomically\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckPreparationCacheTrim() ||
      !CheckQuantizationMatrixScaleStats() ||
      !CheckQuantizationMatrixScaleSelection() ||
      !CheckQuantizationMatrixScaleStatsPolicy() ||
      !CheckDeterministicWorkflow() ||
      !CheckCpuThreadBudget() ||
      !CheckEffortPolicy() ||
      !CheckCompressionPolicy() ||
      !CheckHighDensityMode() ||
      !CheckInvalidRequestsAreAtomic() ||
      !CheckMaximumErrorControl() ||
      !CheckTargetSizeControl() ||
      !CheckSingleAttemptTiming()) {
    return EXIT_FAILURE;
  }
  std::cout << "All public codestream workflow tests passed.\n";
  return EXIT_SUCCESS;
}
