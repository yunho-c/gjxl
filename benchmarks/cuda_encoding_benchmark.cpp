// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Measures the profiled public CPU/CUDA encoding boundary.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include "codestream/workflow.h"
#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_backend.h"
#include "io/pfm.h"

namespace {

constexpr size_t kDefaultWarmups = 2;
constexpr size_t kDefaultSamples = 5;
constexpr float kDefaultButteraugliTarget = 1.2f;

struct WorkloadSpec {
  std::string_view name;
  gjxl::Extent2D extent;
};

constexpr std::array<WorkloadSpec, 3> kWorkloads = {{
    {"synthetic_128x96", {128, 96}},
    {"padded_1080p", {1919, 1079}},
    {"padded_4k", {3839, 2159}},
}};

struct CommandLineOptions {
  std::string workload = "padded_1080p";
  std::filesystem::path input_path;
  gjxl::GpuAdaptiveQuantizationMode gpu_aq_mode =
      gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  float butteraugli_target = kDefaultButteraugliTarget;
  int32_t effort = 7;
  size_t cpu_thread_count = 0;
  size_t warmups = kDefaultWarmups;
  size_t samples = kDefaultSamples;
  bool collect_final_butteraugli_score = false;
  bool gpu_only = false;
  bool profile_range = false;
};

enum class ProfileStage : size_t {
  kTotal,
  kInputPreparation,
  kBackendSelection,
  kQuantizationPipeline,
  kCodestreamEncoding,
  kSummaryAssembly,
  kUnattributed,
  kCount,
};

constexpr size_t kProfileStageCount = static_cast<size_t>(ProfileStage::kCount);
constexpr std::array<std::string_view, kProfileStageCount> kProfileStageNames =
    {
        "total",
        "input_preparation",
        "backend_selection",
        "quantization_pipeline",
        "codestream_encoding",
        "summary_assembly",
        "unattributed",
};

using ProfileValues = std::array<double, kProfileStageCount>;
using ProfileSamples = std::array<std::vector<double>, kProfileStageCount>;

struct TimingStats {
  double minimum_ms = 0.0;
  double median_ms = 0.0;
  double maximum_ms = 0.0;
};

struct EncodeResult {
  std::vector<uint8_t> codestream;
  gjxl::VarDctEncodingSummary summary;
  gjxl::codestream_internal::VarDctEncodingProfile profile;
};

[[nodiscard]] size_t ParseSize(std::string_view text, bool allow_zero) {
  if (text.empty() || text.front() == '-' || text.front() == '+') {
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

[[nodiscard]] float ParsePositiveFloat(std::string_view text) {
  if (text.empty()) {
    throw std::runtime_error("Benchmark distance is invalid");
  }
  size_t parsed = 0;
  const double value = std::stod(std::string(text), &parsed);
  if (parsed != text.size() || !std::isfinite(value) || value <= 0.0 ||
      value > std::numeric_limits<float>::max()) {
    throw std::runtime_error("Benchmark distance is invalid");
  }
  return static_cast<float>(value);
}

[[nodiscard]] gjxl::GpuAdaptiveQuantizationMode
ParseGpuAqMode(std::string_view text) {
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
  throw std::runtime_error("Unknown GPU AQ mode: " + std::string(text));
}

[[nodiscard]] std::string_view
GpuAqModeName(gjxl::GpuAdaptiveQuantizationMode mode) {
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
  std::cout << "Usage: " << executable
            << " [--workload synthetic_128x96|padded_1080p|padded_4k|all]"
               " [--input IMAGE.pfm]"
               " [--gpu-aq exact-coefficients|fully-resident|throughput|"
               "maximum-throughput]"
               " [--distance D] [--effort 1..10] [--cpu-threads auto|N]"
               " [--warmups N] [--samples N] [--collect-final-score]"
               " [--gpu-only] [--profile-range]\n";
}

[[nodiscard]] CommandLineOptions ParseCommandLine(int argc, char **argv) {
  CommandLineOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--collect-final-score") {
      options.collect_final_butteraugli_score = true;
      continue;
    }
    if (argument == "--gpu-only") {
      options.gpu_only = true;
      continue;
    }
    if (argument == "--profile-range") {
      options.profile_range = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("Benchmark option is missing its value");
    }
    const std::string_view value = argv[++index];
    if (argument == "--workload") {
      options.workload = value;
    } else if (argument == "--input") {
      options.input_path = value;
    } else if (argument == "--gpu-aq") {
      options.gpu_aq_mode = ParseGpuAqMode(value);
    } else if (argument == "--distance") {
      options.butteraugli_target = ParsePositiveFloat(value);
    } else if (argument == "--effort") {
      const size_t effort = ParseSize(value, false);
      if (effort > 10) {
        throw std::runtime_error("Effort must be in [1, 10]");
      }
      options.effort = static_cast<int32_t>(effort);
    } else if (argument == "--cpu-threads") {
      options.cpu_thread_count = value == "auto" ? 0 : ParseSize(value, false);
      if (options.cpu_thread_count > gjxl::kMaximumCpuThreadCount) {
        throw std::runtime_error(
            "CPU thread count must be auto or at most 256");
      }
    } else if (argument == "--warmups") {
      options.warmups = ParseSize(value, true);
    } else if (argument == "--samples") {
      options.samples = ParseSize(value, false);
    } else {
      throw std::runtime_error("Unknown benchmark option: " +
                               std::string(argument));
    }
  }
  if (!options.input_path.empty() && options.workload == "all") {
    throw std::runtime_error("An input image cannot be combined with all");
  }
  if (options.profile_range && !options.gpu_only) {
    throw std::runtime_error("--profile-range requires --gpu-only");
  }
  if (options.profile_range && options.workload == "all") {
    throw std::runtime_error("--profile-range requires one workload");
  }
  if (options.input_path.empty() && options.workload != "all" &&
      std::ranges::none_of(kWorkloads, [&](const WorkloadSpec &workload) {
        return workload.name == options.workload;
      })) {
    throw std::runtime_error("Unknown workload: " + options.workload);
  }
  return options;
}

void RequireStatus(std::string_view operation, gjxl::Status status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::string(status.message()));
  }
}

void RequireCuda(std::string_view operation, cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
  }
}

class ScopedCudaProfileRange {
public:
  explicit ScopedCudaProfileRange(bool enabled) : active_(enabled) {
    if (active_) {
      RequireCuda("cudaProfilerStart", cudaProfilerStart());
    }
  }

  ScopedCudaProfileRange(const ScopedCudaProfileRange &) = delete;
  ScopedCudaProfileRange &operator=(const ScopedCudaProfileRange &) = delete;

  ~ScopedCudaProfileRange() {
    if (active_) {
      (void)cudaProfilerStop();
    }
  }

  void Stop() {
    if (active_) {
      active_ = false;
      RequireCuda("cudaProfilerStop", cudaProfilerStop());
    }
  }

private:
  bool active_;
};

[[nodiscard]] gjxl::Image3FBuffer MakeSynthetic(gjxl::Extent2D extent) {
  gjxl::Image3FBuffer image(extent);
  const float width_scale =
      extent.width > 1 ? 1.0f / static_cast<float>(extent.width - 1) : 0.0f;
  const float height_scale =
      extent.height > 1 ? 1.0f / static_cast<float>(extent.height - 1) : 0.0f;
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const float fx = static_cast<float>(x) * width_scale;
      const float fy = static_cast<float>(y) * height_scale;
      const float texture =
          static_cast<float>((13 * x + 17 * y + (x * y) % 29) % 97) / 1024.0f;
      const size_t pixel = y * extent.width + x;
      image.plane(0)[pixel] = 0.025f + 0.72f * fx + texture;
      image.plane(1)[pixel] = 0.020f + 0.64f * fy + texture;
      image.plane(2)[pixel] = 0.030f + 0.30f * fx + 0.38f * fy + texture;
    }
  }
  return image;
}

[[nodiscard]] double NanosecondsToMilliseconds(uint64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1.0e6;
}

[[nodiscard]] ProfileValues GetProfileValues(
    const gjxl::codestream_internal::VarDctEncodingProfile &profile) {
  const uint64_t attributed = profile.input_preparation_nanoseconds +
                              profile.backend_selection_nanoseconds +
                              profile.quantization_pipeline_nanoseconds +
                              profile.codestream_encoding_nanoseconds +
                              profile.summary_assembly_nanoseconds;
  const uint64_t unattributed = profile.total_nanoseconds > attributed
                                    ? profile.total_nanoseconds - attributed
                                    : 0;
  return {
      NanosecondsToMilliseconds(profile.total_nanoseconds),
      NanosecondsToMilliseconds(profile.input_preparation_nanoseconds),
      NanosecondsToMilliseconds(profile.backend_selection_nanoseconds),
      NanosecondsToMilliseconds(profile.quantization_pipeline_nanoseconds),
      NanosecondsToMilliseconds(profile.codestream_encoding_nanoseconds),
      NanosecondsToMilliseconds(profile.summary_assembly_nanoseconds),
      NanosecondsToMilliseconds(unattributed),
  };
}

void AppendProfile(
    const gjxl::codestream_internal::VarDctEncodingProfile &profile,
    ProfileSamples *samples) {
  const ProfileValues values = GetProfileValues(profile);
  for (size_t stage = 0; stage < values.size(); ++stage) {
    (*samples)[stage].push_back(values[stage]);
  }
}

[[nodiscard]] TimingStats Summarize(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::runtime_error("Cannot summarize an empty sample set");
  }
  std::ranges::sort(samples);
  const size_t middle = samples.size() / 2;
  const double median = samples.size() % 2 == 0
                            ? 0.5 * (samples[middle - 1] + samples[middle])
                            : samples[middle];
  return {samples.front(), median, samples.back()};
}

void PrintProfile(std::string_view backend, const ProfileSamples &samples) {
  std::cout << "  " << backend << "_wall_profile\n";
  for (size_t stage = 0; stage < samples.size(); ++stage) {
    const TimingStats stats = Summarize(samples[stage]);
    std::cout << "    timing_ms " << kProfileStageNames[stage]
              << " median=" << stats.median_ms << " range=[" << stats.minimum_ms
              << ',' << stats.maximum_ms << "]\n";
  }
}

[[nodiscard]] EncodeResult Encode(gjxl::ConstImage3FView image,
                                  const CommandLineOptions &options,
                                  gjxl::VarDctBackendPreference backend,
                                  gjxl::GpuBackend *gpu) {
  EncodeResult result;
  const gjxl::GpuAdaptiveQuantizationMode mode =
      backend == gjxl::VarDctBackendPreference::kCuda
          ? options.gpu_aq_mode
          : gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
  RequireStatus(
      backend == gjxl::VarDctBackendPreference::kCuda ? "CUDA encode"
                                                      : "CPU encode",
      gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
              image,
              {.butteraugli_target = options.butteraugli_target,
               .effort = options.effort,
               .cpu_thread_count = options.cpu_thread_count,
               .backend = backend,
               .gpu_aq_mode = mode,
               .collect_final_butteraugli_score =
                   options.collect_final_butteraugli_score},
              gpu, false, &result.codestream, &result.summary,
              &result.profile));
  const gjxl::VarDctExecutionBackend expected_backend =
      backend == gjxl::VarDctBackendPreference::kCuda
          ? gjxl::VarDctExecutionBackend::kCuda
          : gjxl::VarDctExecutionBackend::kCpu;
  if (result.codestream.size() < 2 || result.codestream[0] != 0xff ||
      result.codestream[1] != 0x0a ||
      result.summary.execution_backend != expected_backend ||
      result.profile.execution_backend != expected_backend ||
      result.profile.total_nanoseconds == 0) {
    throw std::runtime_error("Profiled encode returned an invalid result");
  }
  return result;
}

void RunWorkload(std::string_view name, gjxl::Image3FBuffer image,
                 const CommandLineOptions &options, gjxl::GpuBackend &gpu) {
  for (size_t warmup = 0; warmup < options.warmups; ++warmup) {
    if (!options.gpu_only) {
      (void)Encode(image.const_view(), options,
                   gjxl::VarDctBackendPreference::kCpu, nullptr);
    }
    (void)Encode(image.const_view(), options,
                 gjxl::VarDctBackendPreference::kCuda, &gpu);
  }

  ProfileSamples cpu_samples;
  ProfileSamples cuda_samples;
  std::vector<double> total_speedups;
  std::vector<double> quantization_speedups;
  total_speedups.reserve(options.samples);
  quantization_speedups.reserve(options.samples);
  size_t cpu_bytes = 0;
  size_t cuda_bytes = 0;
  bool exact_codestream = false;
  ScopedCudaProfileRange profile_range(options.profile_range);
  for (size_t sample = 0; sample < options.samples; ++sample) {
    EncodeResult cpu;
    EncodeResult cuda;
    if (options.gpu_only) {
      cuda = Encode(image.const_view(), options,
                    gjxl::VarDctBackendPreference::kCuda, &gpu);
    } else if ((sample & 1u) == 0) {
      cpu = Encode(image.const_view(), options,
                   gjxl::VarDctBackendPreference::kCpu, nullptr);
      cuda = Encode(image.const_view(), options,
                    gjxl::VarDctBackendPreference::kCuda, &gpu);
    } else {
      cuda = Encode(image.const_view(), options,
                    gjxl::VarDctBackendPreference::kCuda, &gpu);
      cpu = Encode(image.const_view(), options,
                   gjxl::VarDctBackendPreference::kCpu, nullptr);
    }
    const ProfileValues cuda_values = GetProfileValues(cuda.profile);
    AppendProfile(cuda.profile, &cuda_samples);
    cuda_bytes = cuda.codestream.size();
    std::cout
        << "sample workload=" << name << " index=" << sample
        << " backend=cuda order="
        << (options.gpu_only
                ? "gpu-only"
                : ((sample & 1u) == 0 ? "cpu-first" : "cuda-first"))
        << " total_ms="
        << cuda_values[static_cast<size_t>(ProfileStage::kTotal)]
        << " quantization_ms="
        << cuda_values[static_cast<size_t>(ProfileStage::kQuantizationPipeline)]
        << '\n';
    if (!options.gpu_only) {
      const ProfileValues cpu_values = GetProfileValues(cpu.profile);
      AppendProfile(cpu.profile, &cpu_samples);
      cpu_bytes = cpu.codestream.size();
      exact_codestream = cpu.codestream == cuda.codestream;
      if (options.gpu_aq_mode ==
              gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients &&
          !exact_codestream) {
        throw std::runtime_error(
            "Exact CUDA benchmark output differs from CPU");
      }
      total_speedups.push_back(
          cpu_values[static_cast<size_t>(ProfileStage::kTotal)] /
          cuda_values[static_cast<size_t>(ProfileStage::kTotal)]);
      quantization_speedups.push_back(
          cpu_values[static_cast<size_t>(ProfileStage::kQuantizationPipeline)] /
          cuda_values[static_cast<size_t>(
              ProfileStage::kQuantizationPipeline)]);
      std::cout << "sample workload=" << name << " index=" << sample
                << " backend=cpu total_ms="
                << cpu_values[static_cast<size_t>(ProfileStage::kTotal)]
                << " quantization_ms="
                << cpu_values[static_cast<size_t>(
                       ProfileStage::kQuantizationPipeline)]
                << '\n';
    }
  }
  profile_range.Stop();

  std::cout << "workload " << name << " source=" << image.extent().width << 'x'
            << image.extent().height
            << " distance=" << options.butteraugli_target
            << " effort=" << options.effort
            << " gpu_aq=" << GpuAqModeName(options.gpu_aq_mode)
            << " warmups=" << options.warmups << " samples=" << options.samples
            << " comparison="
            << (options.gpu_only ? "not-run"
                                 : (exact_codestream ? "exact" : "different"));
  if (!options.gpu_only) {
    std::cout << " cpu_bytes=" << cpu_bytes;
  }
  std::cout << " cuda_bytes=" << cuda_bytes << '\n';
  if (!options.gpu_only) {
    PrintProfile("cpu", cpu_samples);
  }
  PrintProfile("cuda", cuda_samples);
  if (!options.gpu_only) {
    const TimingStats total = Summarize(std::move(total_speedups));
    const TimingStats quantization =
        Summarize(std::move(quantization_speedups));
    std::cout << "  ratio cpu_to_cuda_total median=" << total.median_ms
              << " range=[" << total.minimum_ms << ',' << total.maximum_ms
              << "]\n"
              << "  ratio cpu_to_cuda_quantization median="
              << quantization.median_ms << " range=[" << quantization.minimum_ms
              << ',' << quantization.maximum_ms << "]\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const CommandLineOptions options = ParseCommandLine(argc, argv);
    std::unique_ptr<gjxl::GpuBackend> gpu;
    const gjxl::Status factory = gjxl::CreateCudaBackend(&gpu);
    if (!factory.ok()) {
      if (factory.code() == gjxl::StatusCode::kUnavailable) {
        std::cout << "CUDA unavailable: " << factory.message() << '\n';
        return 77;
      }
      throw std::runtime_error("CUDA backend creation failed: " +
                               std::string(factory.message()));
    }
    std::cout << std::fixed << std::setprecision(3)
              << "CPU/CUDA public-workflow wall profile: device=" << gpu->name()
              << " gpu_aq=" << GpuAqModeName(options.gpu_aq_mode)
              << " final_score="
              << (options.collect_final_butteraugli_score ? "collect" : "skip")
              << " cpu_threads="
              << (options.cpu_thread_count == 0
                      ? std::string("auto")
                      : std::to_string(options.cpu_thread_count))
              << '\n';
    if (!options.input_path.empty()) {
      gjxl::Image3FBuffer image;
      RequireStatus("PFM input", gjxl::io::ReadPfm(options.input_path, &image));
      RunWorkload("external_input", std::move(image), options, *gpu);
    } else {
      for (const WorkloadSpec &workload : kWorkloads) {
        if (options.workload == "all" || options.workload == workload.name) {
          RunWorkload(workload.name, MakeSynthetic(workload.extent), options,
                      *gpu);
        }
      }
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "Benchmark error: " << error.what() << '\n';
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
}
