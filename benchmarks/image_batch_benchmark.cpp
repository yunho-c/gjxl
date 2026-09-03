// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Measures warm image-level throughput from linear RGB through codestreams.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "codestream/batch_workflow.h"
#include "codestream/workflow.h"
#include "core/image.h"

namespace {

using Clock = std::chrono::steady_clock;

struct CommandLineOptions {
  std::string workload = "all";
  std::vector<size_t> batch_sizes = {1, 2, 4, 8};
  size_t samples = 3;
  size_t warmups = 1;
  float butteraugli_target = 1.2f;
#if GJXL_BENCHMARK_HAS_METAL
  gjxl::VarDctBackendPreference backend =
    gjxl::VarDctBackendPreference::kMetal;
#else
  gjxl::VarDctBackendPreference backend =
    gjxl::VarDctBackendPreference::kCpu;
#endif
  gjxl::GpuAdaptiveQuantizationMode gpu_aq_mode =
    gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput;
  bool gpu_aq_explicit = false;
};

struct WorkloadSpec {
  std::string_view name;
  gjxl::Extent2D extent;
};

constexpr std::array<WorkloadSpec, 5> kWorkloads = {{
  {"thumbnail_64x64", {64, 64}},
  {"small_256x192", {256, 192}},
  {"medium_512x384", {512, 384}},
  {"1080p", {1920, 1080}},
  {"4k", {3840, 2160}},
}};

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D image_extent)
    : extent(image_extent) {
    size_t pixel_count = 0;
    if (!extent.try_area(&pixel_count) || pixel_count == 0) {
      throw std::runtime_error("Benchmark image extent is invalid");
    }
    for (std::vector<float>& values : plane) {
      values.resize(pixel_count);
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

struct Distribution {
  double minimum = 0.0;
  double median = 0.0;
  double maximum = 0.0;
};

struct BenchmarkRow {
  WorkloadSpec workload;
  size_t batch_size = 0;
  Distribution sequential_ms;
  Distribution batched_ms;
  Distribution paired_speedup;
};

[[nodiscard]] size_t ParsePositiveSize(
  std::string_view text,
  std::string_view name) {

  if (text.empty() || text.front() == '-' || text.front() == '+') {
    throw std::runtime_error(
      std::string(name) + " must be a positive integer");
  }
  std::string terminated(text);
  char* end = nullptr;
  errno = 0;
  const unsigned long long value =
    std::strtoull(terminated.c_str(), &end, 10);
  if (errno == ERANGE || end != terminated.c_str() + terminated.size() ||
      value == 0 || value > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(
      std::string(name) + " must be a positive integer");
  }
  return static_cast<size_t>(value);
}

[[nodiscard]] float ParsePositiveFloat(
  std::string_view text,
  std::string_view name) {

  if (text.empty()) {
    throw std::runtime_error(std::string(name) + " is empty");
  }
  std::string terminated(text);
  char* end = nullptr;
  errno = 0;
  const float value = std::strtof(terminated.c_str(), &end);
  if (errno == ERANGE || end != terminated.c_str() + terminated.size() ||
      !std::isfinite(value) || value <= 0.0f) {
    throw std::runtime_error(
      std::string(name) + " must be finite and positive");
  }
  return value;
}

[[nodiscard]] std::vector<size_t> ParseBatchSizes(std::string_view text) {
  std::vector<size_t> result;
  size_t begin = 0;
  while (begin <= text.size()) {
    const size_t separator = text.find(',', begin);
    const size_t end = separator == std::string_view::npos
      ? text.size()
      : separator;
    result.push_back(ParsePositiveSize(
      text.substr(begin, end - begin), "Batch size"));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1;
  }
  if (!std::ranges::is_sorted(result) ||
      std::adjacent_find(result.begin(), result.end()) != result.end()) {
    throw std::runtime_error(
      "Batch sizes must be unique and increasing");
  }
  return result;
}

[[nodiscard]] gjxl::VarDctBackendPreference ParseBackend(
  std::string_view text) {

  if (text == "auto") {
    return gjxl::VarDctBackendPreference::kAutomatic;
  }
  if (text == "cpu") {
    return gjxl::VarDctBackendPreference::kCpu;
  }
  if (text == "metal") {
    return gjxl::VarDctBackendPreference::kMetal;
  }
  if (text == "cuda") {
    return gjxl::VarDctBackendPreference::kCuda;
  }
  throw std::runtime_error("Unknown backend: " + std::string(text));
}

[[nodiscard]] gjxl::GpuAdaptiveQuantizationMode ParseGpuAqMode(
  std::string_view text) {

  if (text == "exact-coefficients") {
    return gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
  }
  if (text == "fully-resident") {
    return gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  }
  if (text == "throughput") {
    return gjxl::GpuAdaptiveQuantizationMode::kThroughput;
  }
  if (text == "maximum-throughput") {
    return gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput;
  }
  throw std::runtime_error(
    "Unknown GPU AQ mode: " + std::string(text));
}

[[nodiscard]] std::string_view BackendName(
  gjxl::VarDctBackendPreference backend) {

  switch (backend) {
    case gjxl::VarDctBackendPreference::kAutomatic:
      return "auto";
    case gjxl::VarDctBackendPreference::kCpu:
      return "cpu";
    case gjxl::VarDctBackendPreference::kMetal:
      return "metal";
    case gjxl::VarDctBackendPreference::kCuda:
      return "cuda";
  }
  return "invalid";
}

[[nodiscard]] std::string_view GpuAqModeName(
  gjxl::GpuAdaptiveQuantizationMode mode) {

  switch (mode) {
    case gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients:
      return "exact-coefficients";
    case gjxl::GpuAdaptiveQuantizationMode::kFullyResident:
      return "fully-resident";
    case gjxl::GpuAdaptiveQuantizationMode::kThroughput:
      return "throughput";
    case gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput:
      return "maximum-throughput";
  }
  return "invalid";
}

void PrintUsage(std::string_view executable) {
  std::cout
    << "Usage: " << executable
    << " [--workload all|thumbnail_64x64|small_256x192|"
       "medium_512x384|1080p|4k]"
       " [--batch-sizes 1,2,4,8] [--samples N] [--warmups N]"
       " [--distance VALUE] [--backend auto|cpu|metal|cuda]"
       " [--gpu-aq exact-coefficients|fully-resident|throughput|"
       "maximum-throughput]\n";
}

[[nodiscard]] CommandLineOptions ParseCommandLine(int argc, char** argv) {
  CommandLineOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto value = [&](std::string_view name) -> std::string_view {
      if (++index >= argc) {
        throw std::runtime_error(
          std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--workload") {
      options.workload = value(argument);
    } else if (argument == "--batch-sizes") {
      options.batch_sizes = ParseBatchSizes(value(argument));
    } else if (argument == "--samples") {
      options.samples = ParsePositiveSize(value(argument), "Samples");
    } else if (argument == "--warmups") {
      options.warmups = ParsePositiveSize(value(argument), "Warmups");
    } else if (argument == "--distance") {
      options.butteraugli_target =
        ParsePositiveFloat(value(argument), "Distance");
    } else if (argument == "--backend") {
      options.backend = ParseBackend(value(argument));
    } else if (argument == "--gpu-aq" || argument == "--metal-aq") {
      options.gpu_aq_mode = ParseGpuAqMode(value(argument));
      options.gpu_aq_explicit = true;
    } else if (argument == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(argument));
    }
  }
  if (options.backend != gjxl::VarDctBackendPreference::kMetal &&
      options.backend != gjxl::VarDctBackendPreference::kCuda) {
    if (!options.gpu_aq_explicit) {
      options.gpu_aq_mode =
        gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
    } else if (options.gpu_aq_mode !=
               gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients) {
      throw std::runtime_error(
        "Experimental GPU AQ modes require a forced GPU backend");
    }
  }
  if (options.workload != "all" &&
      std::ranges::none_of(kWorkloads, [&](const WorkloadSpec& workload) {
        return workload.name == options.workload;
      })) {
    throw std::runtime_error(
      "Unknown workload: " + options.workload);
  }
  return options;
}

void FillImage(ImageStorage* image) {
  const float width_scale = image->extent.width > 1
    ? 1.0f / static_cast<float>(image->extent.width - 1)
    : 0.0f;
  const float height_scale = image->extent.height > 1
    ? 1.0f / static_cast<float>(image->extent.height - 1)
    : 0.0f;
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      const float fx = static_cast<float>(x) * width_scale;
      const float fy = static_cast<float>(y) * height_scale;
      const float texture = static_cast<float>(
        (13 * x + 17 * y + (x * y) % 29) % 97) / 1024.0f;
      const size_t index = y * image->extent.width + x;
      image->plane[0][index] = 0.025f + 0.72f * fx + texture;
      image->plane[1][index] = 0.020f + 0.64f * fy + texture;
      image->plane[2][index] =
        0.030f + 0.30f * fx + 0.38f * fy + texture;
    }
  }
}

[[nodiscard]] Distribution Summarize(std::vector<double> values) {
  if (values.empty()) {
    throw std::runtime_error("Cannot summarize an empty sample set");
  }
  std::ranges::sort(values);
  const size_t middle = values.size() / 2;
  const double median = values.size() % 2 == 0
    ? 0.5 * (values[middle - 1] + values[middle])
    : values[middle];
  return {values.front(), median, values.back()};
}

[[nodiscard]] double RunBatch(
  gjxl::VarDctBatchEncoder& encoder,
  std::span<const gjxl::VarDctBatchEncodingRequest> requests,
  const std::vector<uint8_t>& expected_codestream,
  const gjxl::VarDctEncodingSummary& expected_summary) {

  std::vector<gjxl::VarDctBatchEncodingResult> results;
  const Clock::time_point begin = Clock::now();
  const gjxl::Status status = encoder.Encode(requests, &results);
  const double elapsed_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  if (!status.ok()) {
    throw std::runtime_error(
      "Batch scheduler failed: " + std::string(status.message()));
  }
  if (results.size() != requests.size()) {
    throw std::runtime_error("Batch scheduler changed result count");
  }
  for (size_t index = 0; index < results.size(); ++index) {
    if (!results[index].status.ok()) {
      throw std::runtime_error(
        "Image " + std::to_string(index) + " failed: " +
        std::string(results[index].status.message()));
    }
    if (results[index].codestream != expected_codestream ||
        results[index].summary != expected_summary ||
        results[index].timing.total_nanoseconds == 0) {
      throw std::runtime_error(
        "Image " + std::to_string(index) +
        " did not match the single-image reference");
    }
  }
  return elapsed_ms;
}

[[nodiscard]] BenchmarkRow BenchmarkBatchSize(
  const WorkloadSpec& workload,
  const CommandLineOptions& options,
  gjxl::ConstImage3FView image,
  const std::vector<uint8_t>& expected_codestream,
  const gjxl::VarDctEncodingSummary& expected_summary,
  size_t batch_size) {

  std::vector<gjxl::VarDctBatchEncodingRequest> requests(
    batch_size,
    {
      .linear_rgb = image,
      .options = {
        .butteraugli_target = options.butteraugli_target,
        .backend = options.backend,
        .gpu_aq_mode = options.gpu_aq_mode,
      },
    });

  std::unique_ptr<gjxl::VarDctBatchEncoder> sequential;
  std::unique_ptr<gjxl::VarDctBatchEncoder> batched;
  gjxl::Status status = gjxl::VarDctBatchEncoder::Create(1, &sequential);
  if (status.ok()) {
    status = gjxl::VarDctBatchEncoder::Create(batch_size, &batched);
  }
  if (!status.ok() || sequential == nullptr || batched == nullptr) {
    throw std::runtime_error(
      "Unable to create benchmark drivers: " +
      std::string(status.message()));
  }

  for (size_t warmup = 0; warmup < options.warmups; ++warmup) {
    if (warmup % 2 == 0) {
      (void) RunBatch(
        *sequential, requests, expected_codestream, expected_summary);
      (void) RunBatch(
        *batched, requests, expected_codestream, expected_summary);
    } else {
      (void) RunBatch(
        *batched, requests, expected_codestream, expected_summary);
      (void) RunBatch(
        *sequential, requests, expected_codestream, expected_summary);
    }
  }

  std::vector<double> sequential_samples;
  std::vector<double> batched_samples;
  std::vector<double> paired_speedups;
  sequential_samples.reserve(options.samples);
  batched_samples.reserve(options.samples);
  paired_speedups.reserve(options.samples);
  for (size_t sample = 0; sample < options.samples; ++sample) {
    double sequential_ms = 0.0;
    double batched_ms = 0.0;
    const bool batch_first = sample % 2 != 0;
    if (batch_first) {
      batched_ms = RunBatch(
        *batched, requests, expected_codestream, expected_summary);
      sequential_ms = RunBatch(
        *sequential, requests, expected_codestream, expected_summary);
    } else {
      sequential_ms = RunBatch(
        *sequential, requests, expected_codestream, expected_summary);
      batched_ms = RunBatch(
        *batched, requests, expected_codestream, expected_summary);
    }
    sequential_samples.push_back(sequential_ms);
    batched_samples.push_back(batched_ms);
    paired_speedups.push_back(sequential_ms / batched_ms);
    std::cout << "sample workload=" << workload.name
              << " batch=" << batch_size
              << " order=" << (batch_first ? "batch-first" : "serial-first")
              << " serial_ms=" << std::fixed << std::setprecision(3)
              << sequential_ms << " batch_ms=" << batched_ms
              << " speedup=" << std::setprecision(3)
              << sequential_ms / batched_ms << "x\n";
  }

  return {
    .workload = workload,
    .batch_size = batch_size,
    .sequential_ms = Summarize(std::move(sequential_samples)),
    .batched_ms = Summarize(std::move(batched_samples)),
    .paired_speedup = Summarize(std::move(paired_speedups)),
  };
}

void PrintRows(const std::vector<BenchmarkRow>& rows) {
  std::cout
    << "\nworkload,width,height,batch_size,serial_median_ms,"
       "batch_median_ms,batch_ms_per_image,batch_images_per_second,"
       "paired_speedup_median,paired_speedup_min,paired_speedup_max\n";
  for (const BenchmarkRow& row : rows) {
    const double milliseconds_per_image =
      row.batched_ms.median / static_cast<double>(row.batch_size);
    const double images_per_second =
      1000.0 * static_cast<double>(row.batch_size) /
      row.batched_ms.median;
    std::cout << row.workload.name << ','
              << row.workload.extent.width << ','
              << row.workload.extent.height << ','
              << row.batch_size << ','
              << std::fixed << std::setprecision(3)
              << row.sequential_ms.median << ','
              << row.batched_ms.median << ','
              << milliseconds_per_image << ','
              << images_per_second << ','
              << row.paired_speedup.median << ','
              << row.paired_speedup.minimum << ','
              << row.paired_speedup.maximum << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CommandLineOptions options = ParseCommandLine(argc, argv);
    std::cout << "image batch benchmark backend="
              << BackendName(options.backend)
              << " gpu_aq=" << GpuAqModeName(options.gpu_aq_mode)
              << " distance=" << options.butteraugli_target
              << " samples=" << options.samples
              << " warmups=" << options.warmups << '\n';
    std::cout << "Boundary: linear RGB input through in-memory codestream; "
                 "input generation, file I/O, and driver construction are "
                 "excluded.\n";

    std::vector<BenchmarkRow> rows;
    for (const WorkloadSpec& workload : kWorkloads) {
      if (options.workload != "all" && options.workload != workload.name) {
        continue;
      }
      ImageStorage image(workload.extent);
      FillImage(&image);
      const gjxl::VarDctEncodingOptions encode_options = {
        .butteraugli_target = options.butteraugli_target,
        .backend = options.backend,
        .gpu_aq_mode = options.gpu_aq_mode,
      };
      std::vector<uint8_t> reference_codestream;
      gjxl::VarDctEncodingSummary reference_summary;
      const gjxl::Status reference_status =
        gjxl::EncodeLinearRgbVarDctCodestream(
          image.View(), encode_options,
          &reference_codestream, &reference_summary);
      if (!reference_status.ok()) {
        throw std::runtime_error(
          "Reference encode for " + std::string(workload.name) +
          " failed: " + std::string(reference_status.message()));
      }
      std::cout << "reference workload=" << workload.name
                << " bytes=" << reference_codestream.size() << '\n';
      for (size_t batch_size : options.batch_sizes) {
        rows.push_back(BenchmarkBatchSize(
          workload, options, image.View(), reference_codestream,
          reference_summary, batch_size));
      }
    }
    PrintRows(rows);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark error: " << error.what() << '\n';
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
}
