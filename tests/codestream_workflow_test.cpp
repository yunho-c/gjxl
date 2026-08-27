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

  if (!rejected_atomically(image.View(), {.butteraugli_target = 0.0f}) ||
      !rejected_atomically({}, {.butteraugli_target = 1.0f}) ||
      !rejected_atomically(
          image.View(),
          {.butteraugli_target = 1.0f,
           .backend = static_cast<gjxl::VarDctBackendPreference>(99)})) {
    std::cerr << "Invalid workflow request changed output\n";
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

}  // namespace

int main() {
  if (!CheckDeterministicWorkflow() || !CheckInvalidRequestsAreAtomic()) {
    return EXIT_FAILURE;
  }
  std::cout << "All public codestream workflow tests passed.\n";
  return EXIT_SUCCESS;
}
