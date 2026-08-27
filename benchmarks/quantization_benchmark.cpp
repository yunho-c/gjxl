// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Reports rotated CPU baselines and internal AQ evaluation timings.

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/adaptive_quantization.h"
#include "codec/adaptive_quantization_internal.h"
#include "codec/chroma_from_luma.h"
#include "codec/color_transform.h"
#include "codec/epf.h"
#include "codec/gaborish.h"
#include "codec/quantization_pipeline.h"

#ifndef GJXL_FLOWER_PPM_PATH
#error "GJXL_FLOWER_PPM_PATH must identify the pinned Flower PPM"
#endif

namespace {

namespace aqi = gjxl::adaptive_quantization_internal;
using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWarmups = 3;
constexpr size_t kDefaultSamples = 5;
constexpr float kButteraugliTarget = 1.2f;

enum class Phase : size_t {
  kInitialQuantField,
  kGaborishInverse,
  kInitialColorCorrelation,
  kAcStrategySearch,
  kAqOneEvaluation,
  kAqTwoUpdates,
  kCompletePipeline,
  kCount,
};

constexpr size_t kPhaseCount = static_cast<size_t>(Phase::kCount);
constexpr std::array<std::string_view, kPhaseCount> kPhaseNames = {
    "initial_quant_field",
    "gaborish_inverse",
    "initial_cfl",
    "ac_strategy_search",
    "aq_one_evaluation",
    "iterative_aq_two_updates",
    "complete_pipeline_two_updates",
};

constexpr std::array<std::string_view, aqi::kEvaluationStageCount>
    kEvaluationStageNames = {
        "field_construction", "coefficient_coding", "reconstruction",
        "loop_filters",       "color_conversion",   "butteraugli",
        "block_reduction",
};

struct CommandLineOptions {
  std::string workload = "all";
  size_t warmups = kDefaultWarmups;
  size_t samples = kDefaultSamples;
};

struct WorkloadSpec {
  std::string_view name;
  gjxl::Extent2D source_extent;
  bool flower = false;
};

constexpr std::array<WorkloadSpec, 6> kWorkloads = {{
    {"synthetic_128x96", {128, 96}, false},
    {"odd_121x89", {121, 89}, false},
    {"flower_510x532", {510, 532}, true},
    {"padded_480p", {854, 479}, false},
    {"padded_720p", {1279, 719}, false},
    {"padded_1080p", {1919, 1079}, false},
}};

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D image_extent) : extent(image_extent) {
    size_t pixel_count = 0;
    if (extent.empty() || !extent.try_area(&pixel_count)) {
      throw std::runtime_error("Benchmark image extent is invalid");
    }
    for (std::vector<float>& values : plane) {
      values.resize(pixel_count);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
        gjxl::PlaneF32View{plane[0].data(), extent, extent.width},
        gjxl::PlaneF32View{plane[1].data(), extent, extent.width},
        gjxl::PlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
        gjxl::ConstPlaneF32View{plane[0].data(), extent, extent.width},
        gjxl::ConstPlaneF32View{plane[1].data(), extent, extent.width},
        gjxl::ConstPlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> plane;
};

struct StageOutput {
  StageOutput(gjxl::Extent2D source_extent, gjxl::Extent2D coding_extent)
      : block_extent{coding_extent.width / gjxl::kJxlBlockDimension,
                     coding_extent.height / gjxl::kJxlBlockDimension},
        initial_quant(block_extent.width * block_extent.height),
        strategy_mask(block_extent.width * block_extent.height),
        pixel_mask(coding_extent.width * coding_extent.height),
        final_quant(block_extent.width * block_extent.height),
        block_distance(block_extent.width * block_extent.height),
        reconstructed(source_extent), coding_extent(coding_extent) {}

  [[nodiscard]] gjxl::InitialQuantFieldOutput InitialOutput() {
    return {
        .quant_field = {initial_quant.data(), block_extent, block_extent.width},
        .strategy_mask = {strategy_mask.data(), block_extent,
                          block_extent.width},
        .pixel_mask = {pixel_mask.data(), coding_extent, coding_extent.width},
    };
  }

  [[nodiscard]] gjxl::AdaptiveQuantizationOutput AdaptiveOutput() {
    return {
        .quant_field = {final_quant.data(), block_extent, block_extent.width},
        .block_distance_map = {block_distance.data(), block_extent,
                               block_extent.width},
        .reconstructed_linear_rgb = reconstructed.View(),
        .frame = &frame,
        .score_history = &scores,
    };
  }

  [[nodiscard]] gjxl::CpuQuantizationPipelineOutput PipelineOutput() {
    return {
        .initial_quantization = InitialOutput(),
        .adaptive_quantization = AdaptiveOutput(),
    };
  }

  gjxl::Extent2D block_extent;
  std::vector<float> initial_quant;
  std::vector<float> strategy_mask;
  std::vector<float> pixel_mask;
  std::vector<float> final_quant;
  std::vector<float> block_distance;
  ImageStorage reconstructed;
  gjxl::Extent2D coding_extent;
  gjxl::VarDctEncoderFrame frame;
  std::vector<double> scores;
  gjxl::AcStrategyGrid strategies;
};

struct TimingStats {
  double minimum_ms = 0.0;
  double median_ms = 0.0;
  double maximum_ms = 0.0;
};

[[nodiscard]] gjxl::Extent2D PaddedExtent(gjxl::Extent2D extent) {
  return {
      (extent.width + gjxl::kJxlBlockDimension - 1) / gjxl::kJxlBlockDimension *
          gjxl::kJxlBlockDimension,
      (extent.height + gjxl::kJxlBlockDimension - 1) /
          gjxl::kJxlBlockDimension * gjxl::kJxlBlockDimension,
  };
}

[[nodiscard]] size_t ParseSize(std::string_view text, bool allow_zero) {
  if (text.empty() || text.front() == '-') {
    throw std::runtime_error("Benchmark count argument is invalid");
  }
  size_t parsed = 0;
  const unsigned long long value = std::stoull(std::string(text), &parsed);
  if (parsed != text.size() || value > std::numeric_limits<size_t>::max() ||
      (!allow_zero && value == 0)) {
    throw std::runtime_error("Benchmark count argument is invalid");
  }
  return static_cast<size_t>(value);
}

[[nodiscard]] CommandLineOptions ParseCommandLine(int argc, char** argv) {
  CommandLineOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << "usage: gjxl_quantization_benchmark "
                   "[--workload NAME|all] [--warmups N] [--samples N]\n";
      std::exit(EXIT_SUCCESS);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("Benchmark option is missing its value");
    }
    const std::string_view value = argv[++index];
    if (argument == "--workload") {
      options.workload = value;
    } else if (argument == "--warmups") {
      options.warmups = ParseSize(value, true);
    } else if (argument == "--samples") {
      options.samples = ParseSize(value, false);
    } else {
      throw std::runtime_error("Unknown quantization benchmark option: " +
                               std::string(argument));
    }
  }
  return options;
}

[[nodiscard]] float SrgbToLinear(uint8_t encoded) {
  const float value = static_cast<float>(encoded) / 255.0f;
  return value <= 0.04045f ? value / 12.92f
                           : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

[[nodiscard]] std::string ReadPpmToken(std::istream* input) {
  std::string token;
  while (true) {
    *input >> std::ws;
    if (input->peek() != '#') {
      break;
    }
    input->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  *input >> token;
  if (!*input) {
    throw std::runtime_error("Malformed Flower PPM header");
  }
  return token;
}

[[nodiscard]] ImageStorage LoadFlower() {
  std::ifstream input(GJXL_FLOWER_PPM_PATH, std::ios::binary);
  if (!input || ReadPpmToken(&input) != "P6") {
    throw std::runtime_error("Unable to open the pinned binary Flower PPM");
  }
  const size_t width = std::stoull(ReadPpmToken(&input));
  const size_t height = std::stoull(ReadPpmToken(&input));
  const unsigned long maximum = std::stoul(ReadPpmToken(&input));
  if (width != 510 || height != 532 || maximum != 255) {
    throw std::runtime_error("Pinned Flower PPM dimensions or depth changed");
  }
  char separator = 0;
  input.get(separator);
  size_t pixel_count = 0;
  const gjxl::Extent2D extent{width, height};
  if (!input || !std::isspace(static_cast<unsigned char>(separator)) ||
      !extent.try_area(&pixel_count) ||
      pixel_count > std::numeric_limits<size_t>::max() / 3) {
    throw std::runtime_error("Pinned Flower PPM is malformed or too large");
  }
  std::vector<uint8_t> bytes(pixel_count * 3);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("Pinned Flower PPM pixel data is truncated");
  }

  ImageStorage image(extent);
  for (size_t index = 0; index < pixel_count; ++index) {
    for (size_t channel = 0; channel < 3; ++channel) {
      image.plane[channel][index] = SrgbToLinear(bytes[index * 3 + channel]);
    }
  }
  return image;
}

void FillSynthetic(ImageStorage* image) {
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      const float fx =
          static_cast<float>(x) /
          static_cast<float>(std::max<size_t>(1, image->extent.width - 1));
      const float fy =
          static_cast<float>(y) /
          static_cast<float>(std::max<size_t>(1, image->extent.height - 1));
      image->plane[0][y * image->extent.width + x] =
          std::clamp(0.08f + 0.72f * fx +
                         0.13f * std::sin(0.47f * static_cast<float>(x + y)),
                     0.0f, 1.0f);
      image->plane[1][y * image->extent.width + x] = std::clamp(
          0.1f + 0.68f * fy +
              0.16f * std::cos(0.39f * (2.0f * static_cast<float>(x) -
                                        static_cast<float>(y))),
          0.0f, 1.0f);
      image->plane[2][y * image->extent.width + x] =
          ((x / 7 + y / 5) & 1u) == 0 ? 0.12f : 0.84f;
    }
  }
}

void EdgePad(const ImageStorage& source, ImageStorage* destination) {
  for (size_t y = 0; y < destination->extent.height; ++y) {
    const size_t source_y = std::min(y, source.extent.height - 1);
    for (size_t x = 0; x < destination->extent.width; ++x) {
      const size_t source_x = std::min(x, source.extent.width - 1);
      for (size_t channel = 0; channel < 3; ++channel) {
        destination->plane[channel][y * destination->extent.width + x] =
            source.plane[channel][source_y * source.extent.width + source_x];
      }
    }
  }
}

void RequireStatus(std::string_view operation, gjxl::Status status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::string(status.message()));
  }
}

[[nodiscard]] TimingStats Summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return {
      samples.front(),
      samples[samples.size() / 2],
      samples.back(),
  };
}

void PrintStats(std::string_view label, const std::vector<double>& samples) {
  const TimingStats stats = Summarize(samples);
  std::cout << "  timing_ms " << label << " median=" << stats.median_ms
            << " range=[" << stats.minimum_ms << ',' << stats.maximum_ms
            << "]\n";
}

[[nodiscard]] double NanosecondsToMilliseconds(uint64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1.0e6;
}

void RunWorkload(const WorkloadSpec& spec, size_t warmups, size_t samples,
                 double* global_sink) {

  ImageStorage original =
      spec.flower ? LoadFlower() : ImageStorage(spec.source_extent);
  if (!spec.flower) {
    FillSynthetic(&original);
  }
  const gjxl::Extent2D coding_extent = PaddedExtent(original.extent);
  ImageStorage padded_linear(coding_extent);
  ImageStorage opsin(coding_extent);
  ImageStorage preprocessed(coding_extent);
  EdgePad(original, &padded_linear);
  RequireStatus(
      "linear-to-opsin setup",
      gjxl::LinearRgbToOpsin(padded_linear.ConstView(), 255.0f, opsin.View()));
  RequireStatus("Gaborish setup", gjxl::ApplyGaborishInverse(
                                      opsin.ConstView(), {1.0f, 1.0f, 1.0f},
                                      preprocessed.View()));

  StageOutput stage(original.extent, coding_extent);
  RequireStatus(
      "initial quantization setup",
      gjxl::ComputeInitialQuantField(opsin.ConstView(),
                                     {.butteraugli_target = kButteraugliTarget},
                                     stage.InitialOutput()));
  gjxl::ColorCorrelationMap initial_color_correlation;
  RequireStatus("initial color-correlation setup",
                gjxl::ComputeInitialColorCorrelationMap(
                    preprocessed.ConstView(), &initial_color_correlation));
  RequireStatus(
      "AC-strategy setup",
      gjxl::FindAcStrategyGrid(
          preprocessed.ConstView(),
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {stage.pixel_mask.data(), coding_extent, coding_extent.width},
          initial_color_correlation, {.butteraugli_target = kButteraugliTarget},
          &stage.strategies));
  std::vector<uint8_t> sharpness(stage.block_extent.width *
                                 stage.block_extent.height);
  RequireStatus(
      "EPF-sharpness setup",
      gjxl::FillDefaultEpfSharpness(
          {sharpness.data(), stage.block_extent, stage.block_extent.width}));

  gjxl::AdaptiveQuantizationOptions one_evaluation_options;
  one_evaluation_options.butteraugli_target = kButteraugliTarget;
  one_evaluation_options.iterations = 0;
  gjxl::AdaptiveQuantizationOptions two_update_options = one_evaluation_options;
  two_update_options.iterations = 2;
  StageOutput pipeline_stage(original.extent, coding_extent);
  gjxl::CpuQuantizationPipelineOptions pipeline_options;
  pipeline_options.butteraugli_target = kButteraugliTarget;
  pipeline_options.adaptive_quantization.iterations = 2;

  auto run_phase = [&](Phase phase, aqi::AdaptiveQuantizationProfile* profile) {
    switch (phase) {
    case Phase::kInitialQuantField:
      return gjxl::ComputeInitialQuantField(
          opsin.ConstView(), {.butteraugli_target = kButteraugliTarget},
          stage.InitialOutput());
    case Phase::kGaborishInverse:
      return gjxl::ApplyGaborishInverse(opsin.ConstView(), {1.0f, 1.0f, 1.0f},
                                        preprocessed.View());
    case Phase::kInitialColorCorrelation:
      return gjxl::ComputeInitialColorCorrelationMap(
          preprocessed.ConstView(), &initial_color_correlation);
    case Phase::kAcStrategySearch:
      return gjxl::FindAcStrategyGrid(
          preprocessed.ConstView(),
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {stage.pixel_mask.data(), coding_extent, coding_extent.width},
          initial_color_correlation, {.butteraugli_target = kButteraugliTarget},
          &stage.strategies);
    case Phase::kAqOneEvaluation: {
      aqi::AdaptiveQuantizationProfile ignored;
      return aqi::FindBestQuantizationProfiled(
          original.ConstView(), preprocessed.ConstView(), stage.strategies,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          one_evaluation_options, stage.AdaptiveOutput(),
          profile == nullptr ? &ignored : profile);
    }
    case Phase::kAqTwoUpdates:
      return gjxl::FindBestQuantization(
          original.ConstView(), preprocessed.ConstView(), stage.strategies,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          two_update_options, stage.AdaptiveOutput());
    case Phase::kCompletePipeline:
      return gjxl::RunCpuQuantizationPipeline(
          original.ConstView(), opsin.ConstView(), pipeline_options,
          pipeline_stage.PipelineOutput());
    case Phase::kCount:
      break;
    }
    return gjxl::Status::Internal("Unknown quantization benchmark phase");
  };

  double sink = 0.0;
  for (size_t warmup = 0; warmup < warmups; ++warmup) {
    for (size_t offset = 0; offset < kPhaseCount; ++offset) {
      const Phase phase = static_cast<Phase>((warmup + offset) % kPhaseCount);
      RequireStatus(kPhaseNames[static_cast<size_t>(phase)],
                    run_phase(phase, nullptr));
      sink += stage.initial_quant.front();
    }
  }

  std::array<std::vector<double>, kPhaseCount> phase_samples;
  std::vector<double> setup_samples;
  std::vector<double> evaluation_total_samples;
  std::array<std::vector<double>, aqi::kEvaluationStageCount> stage_samples;
  std::vector<double> commit_samples;
  for (size_t sample = 0; sample < samples; ++sample) {
    for (size_t offset = 0; offset < kPhaseCount; ++offset) {
      const Phase phase = static_cast<Phase>((sample + offset) % kPhaseCount);
      aqi::AdaptiveQuantizationProfile profile;
      const auto begin = Clock::now();
      RequireStatus(kPhaseNames[static_cast<size_t>(phase)],
                    run_phase(phase, phase == Phase::kAqOneEvaluation
                                         ? &profile
                                         : nullptr));
      const auto end = Clock::now();
      phase_samples[static_cast<size_t>(phase)].push_back(
          std::chrono::duration<double, std::milli>(end - begin).count());
      if (phase == Phase::kAqOneEvaluation) {
        if (profile.evaluations.size() != 1) {
          throw std::runtime_error(
              "One-evaluation AQ benchmark returned an invalid profile");
        }
        setup_samples.push_back(
            NanosecondsToMilliseconds(profile.loop_setup_nanoseconds));
        evaluation_total_samples.push_back(NanosecondsToMilliseconds(
            profile.evaluations.front().total_nanoseconds));
        for (size_t stage_index = 0; stage_index < aqi::kEvaluationStageCount;
             ++stage_index) {
          stage_samples[stage_index].push_back(NanosecondsToMilliseconds(
              profile.evaluations.front().stage_nanoseconds[stage_index]));
        }
        commit_samples.push_back(
            NanosecondsToMilliseconds(profile.output_commit_nanoseconds));
      }
      sink += stage.scores.empty() ? stage.initial_quant.front()
                                   : stage.scores.back();
    }
  }

  std::cout << "workload " << spec.name << " source=" << original.extent.width
            << 'x' << original.extent.height
            << " coding=" << coding_extent.width << 'x' << coding_extent.height
            << '\n';
  for (size_t phase = 0; phase < kPhaseCount; ++phase) {
    PrintStats(kPhaseNames[phase], phase_samples[phase]);
  }
  std::cout << "  one_evaluation_profile\n";
  PrintStats("loop_setup", setup_samples);
  PrintStats("evaluation_total", evaluation_total_samples);
  for (size_t stage = 0; stage < aqi::kEvaluationStageCount; ++stage) {
    PrintStats(kEvaluationStageNames[stage], stage_samples[stage]);
  }
  PrintStats("output_commit", commit_samples);
  std::cout << "  sink=" << sink << '\n';
  *global_sink += sink;
}

[[nodiscard]] bool IsKnownWorkload(std::string_view name) {
  return std::any_of(
      kWorkloads.begin(), kWorkloads.end(),
      [&](const WorkloadSpec& spec) { return spec.name == name; });
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CommandLineOptions options = ParseCommandLine(argc, argv);
    if (options.workload != "all" && !IsKnownWorkload(options.workload)) {
      throw std::runtime_error("Unknown quantization workload: " +
                               options.workload);
    }
    std::cout << std::fixed << std::setprecision(3)
              << "CPU quantization benchmark: warmups=" << options.warmups
              << " samples=" << options.samples
              << " rotated_phases=" << kPhaseCount << '\n';
    double sink = 0.0;
    for (const WorkloadSpec& workload : kWorkloads) {
      if (options.workload == "all" || options.workload == workload.name) {
        RunWorkload(workload, options.warmups, options.samples, &sink);
      }
    }
    std::cout << "global_sink=" << sink << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
