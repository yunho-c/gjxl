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
#include <cstdint>
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
#include "codestream/batch_workflow_internal.h"
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
  gjxl::VarDctBackendPreference backend =
    gjxl::VarDctBackendPreference::kMetal;
  gjxl::GpuAdaptiveQuantizationMode metal_aq_mode =
    gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput;
  bool metal_aq_explicit = false;
  bool profile_stages = false;
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

struct BatchRun {
  double elapsed_ms = 0.0;
  uint64_t begin_host_nanoseconds = 0;
  uint64_t end_host_nanoseconds = 0;
  std::vector<gjxl::codestream_internal::VarDctBatchEncodingStageProfile>
    profiles;
};

struct ProfiledRun {
  WorkloadSpec workload;
  size_t batch_size = 0;
  std::string_view path;
  size_t sample = 0;
  std::string_view order;
  BatchRun run;
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
       "maximum-throughput] [--profile-stages]\n";
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
    } else if (argument == "--metal-aq") {
      options.metal_aq_mode = ParseMetalAqMode(value(argument));
      options.metal_aq_explicit = true;
    } else if (argument == "--profile-stages") {
      options.profile_stages = true;
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

[[nodiscard]] BatchRun RunBatch(
  gjxl::VarDctBatchEncoder& encoder,
  std::span<const gjxl::VarDctBatchEncodingRequest> requests,
  const std::vector<uint8_t>& expected_codestream,
  const gjxl::VarDctEncodingSummary& expected_summary,
  bool profile_stages) {

  std::vector<gjxl::VarDctBatchEncodingResult> results;
  BatchRun run;
  run.begin_host_nanoseconds =
    gjxl::profile_internal::HostNowNanoseconds();
  const Clock::time_point begin = Clock::now();
  const gjxl::Status status = profile_stages
    ? gjxl::codestream_internal::EncodeVarDctBatchProfiled(
        encoder, requests, &results, &run.profiles)
    : encoder.Encode(requests, &results);
  run.elapsed_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  run.end_host_nanoseconds =
    gjxl::profile_internal::HostNowNanoseconds();
  if (!status.ok()) {
    throw std::runtime_error(
      "Batch scheduler failed: " + std::string(status.message()));
  }
  if (results.size() != requests.size()) {
    throw std::runtime_error("Batch scheduler changed result count");
  }
  if (profile_stages && run.profiles.size() != requests.size()) {
    throw std::runtime_error("Batch profiler changed result count");
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
    if (profile_stages) {
      const auto& profile = run.profiles[index].encoding;
      const auto contained = [&](const gjxl::profile_internal::HostInterval&
                                   interval) {
        return interval.available() &&
          interval.begin_nanoseconds >= run.begin_host_nanoseconds &&
          interval.end_nanoseconds <= run.end_host_nanoseconds;
      };
      if (!contained(profile.total) ||
          !contained(profile.input_preparation) ||
          !contained(profile.backend_selection) ||
          !contained(profile.quantization_pipeline) ||
          !contained(profile.codestream_encoding) ||
          !contained(profile.summary_assembly) ||
          !contained(profile.codestream.total) ||
          !contained(profile.codestream.validation) ||
          !contained(profile.codestream.dc_tokenization) ||
          !contained(profile.codestream.ac_tokenization) ||
          !contained(profile.codestream.entropy_optimization) ||
          !contained(profile.codestream.section_writing) ||
          !contained(profile.codestream.assembly)) {
        throw std::runtime_error(
          "Image " + std::to_string(index) +
          " returned an invalid CPU stage timeline");
      }
      if (results[index].summary.execution_backend ==
            gjxl::VarDctExecutionBackend::kMetal &&
          results[index].summary.metal_aq_mode ==
            gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput) {
        const auto& maximum = profile.maximum_throughput;
        const auto valid_frame_stage = [&](const auto& frame) {
          return contained(frame.submission) &&
            contained(frame.completion_wait) &&
            contained(frame.readback) &&
            contained(frame.frame_assembly) &&
            frame.command_buffer.available() &&
            frame.command_buffer.start_host_nanoseconds >=
              run.begin_host_nanoseconds &&
            frame.command_buffer.end_host_nanoseconds <=
              run.end_host_nanoseconds;
        };
        if (!contained(maximum.initial_field_preparation) ||
            !contained(maximum.quantization_parameter_preparation) ||
            !contained(maximum.prepared_evaluation_setup) ||
            !valid_frame_stage(maximum.initial_quantization) ||
            !contained(maximum.frame_encoding.input_upload) ||
            !valid_frame_stage(maximum.frame_encoding) ||
            !contained(maximum.output_commit)) {
          throw std::runtime_error(
            "Image " + std::to_string(index) +
            " returned an invalid maximum-throughput timeline");
        }
      }
    }
  }
  return run;
}

[[nodiscard]] BenchmarkRow BenchmarkBatchSize(
  const WorkloadSpec& workload,
  const CommandLineOptions& options,
  gjxl::ConstImage3FView image,
  const std::vector<uint8_t>& expected_codestream,
  const gjxl::VarDctEncodingSummary& expected_summary,
  size_t batch_size,
  std::vector<ProfiledRun>* profiled_runs) {

  std::vector<gjxl::VarDctBatchEncodingRequest> requests(
    batch_size,
    {
      .linear_rgb = image,
      .options = {
        .butteraugli_target = options.butteraugli_target,
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
        *sequential, requests, expected_codestream, expected_summary, false);
      (void) RunBatch(
        *batched, requests, expected_codestream, expected_summary, false);
    } else {
      (void) RunBatch(
        *batched, requests, expected_codestream, expected_summary, false);
      (void) RunBatch(
        *sequential, requests, expected_codestream, expected_summary, false);
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
    BatchRun sequential_run;
    BatchRun batched_run;
    const bool batch_first = sample % 2 != 0;
    if (batch_first) {
      batched_run = RunBatch(
        *batched, requests, expected_codestream, expected_summary,
        options.profile_stages);
      sequential_run = RunBatch(
        *sequential, requests, expected_codestream, expected_summary,
        options.profile_stages);
    } else {
      sequential_run = RunBatch(
        *sequential, requests, expected_codestream, expected_summary,
        options.profile_stages);
      batched_run = RunBatch(
        *batched, requests, expected_codestream, expected_summary,
        options.profile_stages);
    }
    sequential_ms = sequential_run.elapsed_ms;
    batched_ms = batched_run.elapsed_ms;
    if (profiled_runs != nullptr) {
      const std::string_view order = batch_first
        ? "batch-first"
        : "serial-first";
      profiled_runs->push_back({
        .workload = workload,
        .batch_size = batch_size,
        .path = "serial",
        .sample = sample + 1,
        .order = order,
        .run = std::move(sequential_run),
      });
      profiled_runs->push_back({
        .workload = workload,
        .batch_size = batch_size,
        .path = "batch",
        .sample = sample + 1,
        .order = order,
        .run = std::move(batched_run),
      });
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

void PrintProfileInterval(
  const ProfiledRun& profiled,
  size_t image_index,
  size_t worker_index,
  std::string_view processor,
  std::string_view stage,
  gjxl::profile_internal::HostInterval interval,
  bool host_envelope_aligned = true) {

  std::cout << profiled.workload.name << ','
            << profiled.workload.extent.width << ','
            << profiled.workload.extent.height << ','
            << profiled.batch_size << ','
            << profiled.path << ','
            << profiled.sample << ','
            << profiled.order << ','
            << image_index << ','
            << worker_index << ','
            << processor << ','
            << stage << ','
            << (interval.available() ? 1 : 0) << ',';
  if (interval.available()) {
    const double start_ms = static_cast<double>(
      interval.begin_nanoseconds - profiled.run.begin_host_nanoseconds) /
      1.0e6;
    const double end_ms = static_cast<double>(
      interval.end_nanoseconds - profiled.run.begin_host_nanoseconds) /
      1.0e6;
    const double duration_ms = static_cast<double>(
      interval.duration_nanoseconds()) / 1.0e6;
    std::cout << (host_envelope_aligned ? 1 : 0) << ','
              << std::fixed << std::setprecision(6)
              << start_ms << ',' << end_ms << ',' << duration_ms;
  } else {
    std::cout << ",,,";
  }
  std::cout << '\n';
}

void PrintCommandBuffer(
  const ProfiledRun& profiled,
  size_t image_index,
  size_t worker_index,
  std::string_view stage,
  const gjxl::gpu_profile_internal::CommandBufferProfile& command) {

  PrintProfileInterval(
    profiled, image_index, worker_index, "gpu", stage,
    {
      .begin_nanoseconds = command.start_host_nanoseconds,
      .end_nanoseconds = command.end_host_nanoseconds,
    },
    command.host_envelope_aligned);
}

void PrintFrameEncodingProfile(
  const ProfiledRun& profiled,
  size_t image_index,
  size_t worker_index,
  std::string_view prefix,
  const gjxl::gpu_profile_internal::FrameEncodingProfile& frame) {

  const std::string base(prefix);
  PrintProfileInterval(
    profiled, image_index, worker_index, "cpu",
    base + "_input_upload", frame.input_upload);
  PrintProfileInterval(
    profiled, image_index, worker_index, "cpu",
    base + "_submission", frame.submission);
  PrintProfileInterval(
    profiled, image_index, worker_index, "wait",
    base + "_completion_wait", frame.completion_wait);
  PrintCommandBuffer(
    profiled, image_index, worker_index,
    base + "_command_buffer", frame.command_buffer);
  PrintProfileInterval(
    profiled, image_index, worker_index, "cpu",
    base + "_readback", frame.readback);
  PrintProfileInterval(
    profiled, image_index, worker_index, "cpu",
    base + "_assembly", frame.frame_assembly);
}

void PrintProfileRows(const std::vector<ProfiledRun>& runs) {
  if (runs.empty()) return;
  std::cout
    << "\nprofile_workload,width,height,batch_size,path,sample,order,"
       "image_index,worker_index,processor,stage,available,"
       "host_envelope_aligned,start_ms,end_ms,duration_ms\n";
  for (const ProfiledRun& profiled : runs) {
    for (size_t image_index = 0;
         image_index < profiled.run.profiles.size(); ++image_index) {
      const auto& batch_profile = profiled.run.profiles[image_index];
      const auto& profile = batch_profile.encoding;
      const size_t worker_index = batch_profile.worker_index;
      const auto print = [&](std::string_view processor,
                             std::string_view stage,
                             gjxl::profile_internal::HostInterval interval) {
        PrintProfileInterval(
          profiled, image_index, worker_index, processor, stage, interval);
      };
      print("mixed", "workflow_total", profile.total);
      print("cpu", "input_preparation", profile.input_preparation);
      print("cpu", "backend_selection", profile.backend_selection);
      print("mixed", "quantization_pipeline", profile.quantization_pipeline);
      print("cpu", "codestream_encoding", profile.codestream_encoding);
      print("cpu", "codestream_validation", profile.codestream.validation);
      print("cpu", "dc_tokenization", profile.codestream.dc_tokenization);
      print("cpu", "ac_tokenization", profile.codestream.ac_tokenization);
      print(
        "cpu", "entropy_optimization",
        profile.codestream.entropy_optimization);
      print("cpu", "section_writing", profile.codestream.section_writing);
      print("cpu", "codestream_assembly", profile.codestream.assembly);
      print("cpu", "summary_assembly", profile.summary_assembly);

      const auto& maximum = profile.maximum_throughput;
      if (maximum.initial_field_preparation.available()) {
        print(
          "cpu", "maximum_initial_field_preparation",
          maximum.initial_field_preparation);
        print(
          "cpu", "maximum_quantization_parameter_preparation",
          maximum.quantization_parameter_preparation);
        print(
          "mixed", "maximum_prepared_evaluation_setup",
          maximum.prepared_evaluation_setup);
        PrintFrameEncodingProfile(
          profiled, image_index, worker_index, "initial_quantization",
          maximum.initial_quantization);
        PrintFrameEncodingProfile(
          profiled, image_index, worker_index, "frame_encoding",
          maximum.frame_encoding);
        print("cpu", "maximum_output_commit", maximum.output_commit);
      } else {
        PrintCommandBuffer(
          profiled, image_index, worker_index,
          "initial_quantization_command_buffer", {});
        PrintCommandBuffer(
          profiled, image_index, worker_index,
          "frame_encoding_command_buffer", {});
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CommandLineOptions options = ParseCommandLine(argc, argv);
    std::cout << "image batch benchmark backend="
              << BackendName(options.backend)
              << " metal_aq=" << MetalAqModeName(options.metal_aq_mode)
              << " distance=" << options.butteraugli_target
              << " samples=" << options.samples
              << " warmups=" << options.warmups
              << " profile_stages="
              << (options.profile_stages ? "true" : "false") << '\n';
    std::cout << "Boundary: linear RGB input through in-memory codestream; "
                 "input generation, file I/O, and driver construction are "
                 "excluded.\n";

    std::vector<BenchmarkRow> rows;
    std::vector<ProfiledRun> profiled_runs;
    for (const WorkloadSpec& workload : kWorkloads) {
      if (options.workload != "all" && options.workload != workload.name) {
        continue;
      }
      ImageStorage image(workload.extent);
      FillImage(&image);
      const gjxl::VarDctEncodingOptions encode_options = {
        .butteraugli_target = options.butteraugli_target,
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
          options.profile_stages ? &profiled_runs : nullptr));
      }
    }
    PrintRows(rows);
    PrintProfileRows(profiled_runs);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark error: " << error.what() << '\n';
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
}
