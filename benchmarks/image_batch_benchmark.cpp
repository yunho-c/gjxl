// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Measures warm image-level throughput from linear RGB through codestreams.

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
#include "io/pfm.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr int32_t kEffort = 7;

struct CommandLineOptions {
  std::string workload = "all";
  std::vector<std::filesystem::path> inputs;
  std::filesystem::path raw_samples;
  std::vector<size_t> batch_sizes = {1, 2, 4, 8};
  size_t samples = 3;
  size_t warmups = 1;
  float butteraugli_target = 1.2f;
  gjxl::VarDctBackendPreference backend =
    gjxl::VarDctBackendPreference::kMetal;
  gjxl::GpuAdaptiveQuantizationMode metal_aq_mode =
    gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  bool metal_aq_explicit = false;
};

struct WorkloadSpec {
  std::string name;
  gjxl::Extent2D extent;
  std::filesystem::path source = {};
};

const std::array<WorkloadSpec, 5> kWorkloads = {{
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
  Distribution image_queue_ms;
  Distribution image_service_ms;
  Distribution image_ready_ms;
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
  throw std::runtime_error("Unknown backend: " + std::string(text));
}

[[nodiscard]] gjxl::GpuAdaptiveQuantizationMode ParseMetalAqMode(
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
    "Unknown Metal AQ mode: " + std::string(text));
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
  }
  return "invalid";
}

[[nodiscard]] std::string_view MetalAqModeName(
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
       " [--distance VALUE] [--backend auto|cpu|metal]"
       " [--metal-aq exact-coefficients|fully-resident|throughput|"
       "maximum-throughput] [--input FILE.pfm|DIRECTORY]..."
       " [--raw-samples NEW.csv]\n"
       "Inputs replace synthetic workloads; directories select sorted PFMs "
       "without recursion. Input dimensions are preserved.\n"
       "Defaults: Metal, fully-resident, effort 7, automatic per-image CPU "
       "threads.\n";
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
    } else if (argument == "--input") {
      options.inputs.emplace_back(value(argument));
    } else if (argument == "--raw-samples") {
      options.raw_samples = value(argument);
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
    } else if (argument == "--metal-aq") {
      options.metal_aq_mode = ParseMetalAqMode(value(argument));
      options.metal_aq_explicit = true;
    } else if (argument == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(argument));
    }
  }
  if (options.backend != gjxl::VarDctBackendPreference::kMetal) {
    if (!options.metal_aq_explicit) {
      options.metal_aq_mode =
        gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
    } else if (options.metal_aq_mode !=
               gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients) {
      throw std::runtime_error(
        "Experimental Metal AQ modes require --backend metal");
    }
  }
  if (options.inputs.empty() && options.workload != "all" &&
      std::ranges::none_of(kWorkloads, [&](const WorkloadSpec& workload) {
        return workload.name == options.workload;
      })) {
    throw std::runtime_error(
      "Unknown workload: " + options.workload);
  }
  return options;
}

[[nodiscard]] std::vector<std::filesystem::path> ResolveInputs(
  const std::vector<std::filesystem::path>& inputs) {
  std::vector<std::filesystem::path> paths;
  for (const auto& input : inputs) {
    if (std::filesystem::is_directory(input)) {
      for (const auto& entry : std::filesystem::directory_iterator(input)) {
        std::string extension = entry.path().extension().string();
        std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
          return static_cast<char>(std::tolower(c));
        });
        if (entry.is_regular_file() && extension == ".pfm") {
          paths.push_back(std::filesystem::canonical(entry.path()));
        }
      }
    } else if (std::filesystem::is_regular_file(input)) {
      paths.push_back(std::filesystem::canonical(input));
    } else {
      throw std::runtime_error(
        "Input is not a file or directory: " + input.string());
    }
  }
  std::ranges::sort(paths);
  paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
  if (!inputs.empty() && paths.empty()) {
    throw std::runtime_error("No PFM inputs found");
  }
  return paths;
}

[[nodiscard]] ImageStorage LoadImage(const std::filesystem::path& path) {
  gjxl::Image3FBuffer decoded;
  const gjxl::Status status = gjxl::io::ReadPfm(path, &decoded);
  if (!status.ok()) {
    throw std::runtime_error("Unable to read " + path.string() + ": " +
                             std::string(status.message()));
  }
  ImageStorage image(decoded.extent());
  for (size_t channel = 0; channel < 3; ++channel) {
    const auto plane = decoded.plane(channel);
    if (std::ranges::any_of(plane, [](float value) {
          return !std::isfinite(value);
        })) {
      throw std::runtime_error("Non-finite PFM input: " + path.string());
    }
    std::copy(plane.begin(), plane.end(), image.plane[channel].begin());
  }
  return image;
}

[[nodiscard]] std::string CsvField(std::string_view value) {
  if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
    return std::string(value);
  }
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"') {
      escaped += '"';
    }
    escaped += c;
  }
  return escaped + '"';
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

[[nodiscard]] int64_t RunBatch(
  gjxl::VarDctBatchEncoder& encoder,
  std::span<const gjxl::VarDctBatchEncodingRequest> requests,
  const std::vector<uint8_t>& expected_codestream,
  const gjxl::VarDctEncodingSummary& expected_summary,
  std::vector<gjxl::VarDctBatchSchedulingTiming>* image_timings = nullptr) {

  std::vector<gjxl::VarDctBatchEncodingResult> results;
  const Clock::time_point begin = Clock::now();
  const gjxl::Status status = encoder.Encode(requests, &results);
  const int64_t elapsed_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now() - begin).count();
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
    const auto timing = results[index].scheduling;
    if (!timing.cpu_admitted || timing.ready_nanoseconds != timing.queue_nanoseconds + timing.service_nanoseconds ||
        timing.ready_nanoseconds > static_cast<uint64_t>(elapsed_ns)) {
      throw std::runtime_error("Image queue/service timing exceeds complete public-call boundary");
    }
    if (image_timings != nullptr) image_timings->push_back(timing);
  }
  return elapsed_ns;
}

[[nodiscard]] BenchmarkRow BenchmarkBatchSize(
  const WorkloadSpec& workload,
  const CommandLineOptions& options,
  gjxl::ConstImage3FView image,
  const std::vector<uint8_t>& expected_codestream,
  const gjxl::VarDctEncodingSummary& expected_summary,
  size_t batch_size,
  std::ostream* raw_samples) {

  std::vector<gjxl::VarDctBatchEncodingRequest> requests(
    batch_size,
    {
      .linear_rgb = image,
      .options = {
        .butteraugli_target = options.butteraugli_target,
        .effort = kEffort,
        .backend = options.backend,
        .metal_aq_mode = options.metal_aq_mode,
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
  std::vector<gjxl::VarDctBatchSchedulingTiming> image_timings;
  sequential_samples.reserve(options.samples);
  batched_samples.reserve(options.samples);
  paired_speedups.reserve(options.samples);
  for (size_t sample = 0; sample < options.samples; ++sample) {
    int64_t sequential_ns = 0;
    int64_t batched_ns = 0;
    const bool batch_first = sample % 2 != 0;
    if (batch_first) {
      batched_ns = RunBatch(
        *batched, requests, expected_codestream, expected_summary, &image_timings);
      sequential_ns = RunBatch(
        *sequential, requests, expected_codestream, expected_summary);
    } else {
      sequential_ns = RunBatch(
        *sequential, requests, expected_codestream, expected_summary);
      batched_ns = RunBatch(
        *batched, requests, expected_codestream, expected_summary, &image_timings);
    }
    const double sequential_ms = static_cast<double>(sequential_ns) / 1e6;
    const double batched_ms = static_cast<double>(batched_ns) / 1e6;
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
    if (raw_samples != nullptr) {
      const bool metal = expected_summary.execution_backend ==
        gjxl::VarDctExecutionBackend::kMetal;
      *raw_samples << "gjxl," << CsvField(workload.name) << ','
                   << CsvField(workload.source.string()) << ','
                   << workload.extent.width << ',' << workload.extent.height << ','
                   << batch_size << ',' << sample << ','
                   << (batch_first ? "batch-first" : "serial-first") << ','
                   << BackendName(options.backend) << ','
                   << (metal ? "metal" : "cpu") << ','
                   << (metal ? MetalAqModeName(options.metal_aq_mode) : "n/a") << ','
                   << std::setprecision(std::numeric_limits<float>::max_digits10)
                   << options.butteraugli_target << ',' << kEffort
                   << ",automatic_per_image,linear_rgb_to_in_memory_codestream,"
                   << sequential_ns << ',' << batched_ns << ','
                   << expected_codestream.size() << '\n' << std::flush;
    }
  }

  std::vector<double> image_queue_ms, image_service_ms, image_ready_ms;
  for (const auto& timing : image_timings) {
    image_queue_ms.push_back(static_cast<double>(timing.queue_nanoseconds) / 1e6);
    image_service_ms.push_back(static_cast<double>(timing.service_nanoseconds) / 1e6);
    image_ready_ms.push_back(static_cast<double>(timing.ready_nanoseconds) / 1e6);
  }
  return {
    .workload = workload,
    .batch_size = batch_size,
    .sequential_ms = Summarize(std::move(sequential_samples)),
    .batched_ms = Summarize(std::move(batched_samples)),
    .paired_speedup = Summarize(std::move(paired_speedups)),
    .image_queue_ms = Summarize(std::move(image_queue_ms)),
    .image_service_ms = Summarize(std::move(image_service_ms)),
    .image_ready_ms = Summarize(std::move(image_ready_ms)),
  };
}

void PrintRows(const std::vector<BenchmarkRow>& rows) {
  std::cout
    << "\nworkload,width,height,batch_size,serial_median_ms,"
       "batch_median_ms,batch_ms_per_image,batch_images_per_second,"
       "paired_speedup_median,paired_speedup_min,paired_speedup_max,"
       "image_queue_median_ms,image_queue_max_ms,image_service_median_ms,image_service_max_ms,"
       "image_ready_median_ms,image_ready_max_ms\n";
  for (const BenchmarkRow& row : rows) {
    const double milliseconds_per_image =
      row.batched_ms.median / static_cast<double>(row.batch_size);
    const double images_per_second =
      1000.0 * static_cast<double>(row.batch_size) /
      row.batched_ms.median;
    std::cout << CsvField(row.workload.name) << ','
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
              << row.paired_speedup.maximum << ','
              << row.image_queue_ms.median << ',' << row.image_queue_ms.maximum << ','
              << row.image_service_ms.median << ',' << row.image_service_ms.maximum << ','
              << row.image_ready_ms.median << ',' << row.image_ready_ms.maximum << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CommandLineOptions options = ParseCommandLine(argc, argv);
    std::vector<WorkloadSpec> workloads;
    for (const auto& path : ResolveInputs(options.inputs)) {
      workloads.push_back({path.string(), {}, path});
    }
    if (options.inputs.empty()) {
      for (const auto& workload : kWorkloads) {
        if (options.workload == "all" || options.workload == workload.name) {
          workloads.push_back(workload);
        }
      }
    }
    std::ofstream raw_samples;
    if (!options.raw_samples.empty()) {
      if (std::filesystem::exists(options.raw_samples)) {
        throw std::runtime_error(
          "Raw sample output already exists: " + options.raw_samples.string());
      }
      raw_samples.exceptions(std::ios::failbit | std::ios::badbit);
      raw_samples.open(options.raw_samples);
      raw_samples << "codec,workload,source,width,height,batch_size,sample,order,"
                     "requested_backend,backend,aq_mode,distance,effort,thread_policy,"
                     "timing_boundary,serial_ns,batch_ns,encoded_bytes_per_image\n";
    }
    std::cout << "image batch benchmark backend="
              << BackendName(options.backend)
              << " metal_aq=" << MetalAqModeName(options.metal_aq_mode)
              << " distance=" << options.butteraugli_target
              << " samples=" << options.samples
              << " warmups=" << options.warmups << '\n';
    std::cout << "Boundary: linear RGB input through in-memory codestream; "
                 "input generation, file I/O, and driver construction are "
                 "excluded.\n";
    std::cout << "Image queue/service/ready spans are pooled across measured batched calls, "
                 "excluding warmups. Ready is internally retained, not publicly available; "
                 "batch_ms_per_image is throughput-derived, not service latency.\n";

    std::vector<BenchmarkRow> rows;
    for (WorkloadSpec& workload : workloads) {
      ImageStorage image = workload.source.empty()
        ? ImageStorage(workload.extent) : LoadImage(workload.source);
      if (workload.source.empty()) {
        FillImage(&image);
      }
      workload.extent = image.extent;
      const gjxl::VarDctEncodingOptions encode_options = {
        .butteraugli_target = options.butteraugli_target,
        .effort = kEffort,
        .backend = options.backend,
        .metal_aq_mode = options.metal_aq_mode,
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
          reference_summary, batch_size,
          raw_samples.is_open() ? &raw_samples : nullptr));
      }
    }
    PrintRows(rows);
    if (raw_samples.is_open()) {
      raw_samples.close();
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark error: " << error.what() << '\n';
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
}
