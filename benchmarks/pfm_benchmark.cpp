// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Reports bounded-memory PFM load latency and throughput.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "io/pfm.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::filesystem::path input;
  size_t samples = 9;
  size_t warmups = 2;
};

[[nodiscard]] size_t ParseSize(std::string_view text, bool allow_zero) {
  if (text.empty() || text.front() == '-') {
    throw std::runtime_error("Benchmark count is invalid");
  }
  size_t parsed = 0;
  const unsigned long long value = std::stoull(std::string(text), &parsed);
  if (parsed != text.size() || value > std::numeric_limits<size_t>::max() ||
      (!allow_zero && value == 0)) {
    throw std::runtime_error("Benchmark count is invalid");
  }
  return static_cast<size_t>(value);
}

[[nodiscard]] Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << "usage: gjxl_pfm_benchmark --input PATH "
                   "[--samples N] [--warmups N]\n";
      std::exit(EXIT_SUCCESS);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("Benchmark option is missing its value");
    }
    const std::string_view value = argv[++index];
    if (argument == "--input") {
      options.input = value;
    } else if (argument == "--samples") {
      options.samples = ParseSize(value, false);
    } else if (argument == "--warmups") {
      options.warmups = ParseSize(value, true);
    } else {
      throw std::runtime_error(
        "Unknown PFM benchmark option: " + std::string(argument));
    }
  }
  if (options.input.empty()) {
    throw std::runtime_error("PFM benchmark input is required");
  }
  return options;
}

[[nodiscard]] uint64_t SampleChecksum(const gjxl::Image3FBuffer& image) {
  const gjxl::ConstImage3FView view = image.const_view();
  uint64_t checksum = 1469598103934665603ull;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y : {size_t{0}, view.height() / 2, view.height() - 1}) {
      for (size_t x : {size_t{0}, view.width() / 2, view.width() - 1}) {
        checksum ^= std::bit_cast<uint32_t>(view.plane[channel].Row(y)[x]);
        checksum *= 1099511628211ull;
      }
    }
  }
  return checksum;
}

void Require(gjxl::Status status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    gjxl::Image3FBuffer image;
    for (size_t warmup = 0; warmup < options.warmups; ++warmup) {
      Require(gjxl::io::ReadPfm(options.input, &image));
    }

    std::vector<double> timings_ms;
    timings_ms.reserve(options.samples);
    uint64_t checksum = 0x9e3779b97f4a7c15ull;
    for (size_t sample = 0; sample < options.samples; ++sample) {
      const Clock::time_point begin = Clock::now();
      Require(gjxl::io::ReadPfm(options.input, &image));
      const Clock::time_point end = Clock::now();
      timings_ms.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
      checksum ^= SampleChecksum(image) + 0x9e3779b97f4a7c15ull +
                  (checksum << 6) + (checksum >> 2);
    }
    std::sort(timings_ms.begin(), timings_ms.end());
    const double minimum_ms = timings_ms.front();
    const double median_ms = timings_ms[timings_ms.size() / 2];
    const double maximum_ms = timings_ms.back();
    size_t pixel_count = 0;
    if (!image.extent().try_area(&pixel_count) ||
        pixel_count > std::numeric_limits<size_t>::max() / 12) {
      throw std::runtime_error("PFM benchmark extent is too large");
    }
    const size_t payload_bytes = pixel_count * 12;
    const double payload_mib =
      static_cast<double>(payload_bytes) / (1024.0 * 1024.0);
    const double throughput_gib_s =
      static_cast<double>(payload_bytes) /
      (median_ms * 1.0e-3 * 1024.0 * 1024.0 * 1024.0);

    std::cout << image.extent().width << 'x' << image.extent().height
              << " PFM, " << std::fixed << std::setprecision(3)
              << payload_mib << " MiB payload\n"
              << "load ms min/median/max: " << minimum_ms << " / "
              << median_ms << " / " << maximum_ms << '\n'
              << "median throughput: " << throughput_gib_s << " GiB/s\n"
              << "checksum: " << checksum << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "PFM benchmark failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
