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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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
#include "codec/loop_filter.h"
#include "codec/quantization.h"
#include "codec/quantization_pipeline.h"
#include "codec/reconstruction.h"
#include "codestream/workflow.h"
#include "codestream/workflow_internal.h"
#include "core/frame_geometry.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_aq_evaluation_profile.h"
#include "gpu/ops/ac_strategy_search.h"
#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/quantization_pipeline.h"
#include "io/pfm.h"

#ifndef GJXL_FLOWER_PPM_PATH
#error "GJXL_FLOWER_PPM_PATH must identify the pinned Flower PPM"
#endif

namespace {

namespace aqi = gjxl::adaptive_quantization_internal;
using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWarmups = 3;
constexpr size_t kDefaultSamples = 5;
constexpr float kDefaultButteraugliTarget = 1.2f;

enum class BenchmarkScope {
  kFull,
  kPublicWorkflow,
  kMetalPublicWorkflow,
  kCoefficientCoding,
};

enum class ValidationMode {
  kCpuMetal,
  kMetalOnly,
};

enum class Phase : size_t {
  kInitialQuantField,
  kGaborishInverse,
  kInitialColorCorrelation,
  kAcStrategySearch,
  kAqOneEvaluation,
  kAqTwoUpdates,
  kCompletePipeline,
  kGpuAcStrategySearch,
  kGpuPreparation,
  kGpuFullyResidentEvaluation,
  kGpuResidentEvaluation,
  kGpuExactCoefficientEvaluation,
  kGpuAqOneEvaluation,
  kGpuAqTwoUpdates,
  kGpuCompletePipeline,
  kCpuPublicWorkflow,
  kGpuWarmPublicWorkflow,
  kGpuColdPublicWorkflow,
  kCount,
};

constexpr size_t kPhaseCount = static_cast<size_t>(Phase::kCount);
constexpr std::array<std::string_view, kPhaseCount> kPhaseNames = {
    "initial_quant_field",
    "gaborish_inverse",
    "initial_cfl",
    "cpu_ac_strategy_search",
    "cpu_aq_one_evaluation",
    "cpu_iterative_aq_two_updates",
    "cpu_complete_pipeline_two_updates",
    "gpu_ac_strategy_search",
    "gpu_preparation",
    "gpu_fully_resident_evaluation",
    "gpu_resident_perceptual_tail",
    "gpu_exact_coefficient_reconstruction_tail",
    "gpu_aq_one_evaluation_e2e",
    "gpu_iterative_aq_two_updates_e2e",
    "gpu_complete_pipeline",
    "cpu_public_workflow",
    "gpu_warm_public_workflow",
    "gpu_cold_public_workflow",
};

constexpr std::array<std::string_view, aqi::kEvaluationStageCount>
    kEvaluationStageNames = {
        "field_construction", "coefficient_coding", "reconstruction",
        "loop_filters",       "color_conversion",   "butteraugli",
        "block_reduction",
};

struct CommandLineOptions {
  std::string workload = "all";
  std::string input_path;
  std::string metallib_path;
  std::string raw_samples_path;
  std::string gpu_profile_path;
  std::string implementation = "simd";
  BenchmarkScope scope = BenchmarkScope::kFull;
  ValidationMode validation = ValidationMode::kCpuMetal;
  gjxl::GpuAdaptiveQuantizationMode gpu_aq_mode =
      gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
  gjxl::VarDctDensityMode density_mode =
      gjxl::VarDctDensityMode::kDefault;
  bool collect_final_butteraugli_score = false;
  float butteraugli_target = kDefaultButteraugliTarget;
  size_t warmups = kDefaultWarmups;
  size_t samples = kDefaultSamples;
  gjxl::gpu_profile_internal::GpuProfilingMode gpu_profiling_mode =
    gjxl::gpu_profile_internal::GpuProfilingMode::kDisabled;
};

struct WorkloadSpec {
  std::string_view name;
  gjxl::Extent2D source_extent;
  bool flower = false;
  bool workflow_gradient = false;
  bool gpu_complete_aq_only = false;
};

constexpr std::array<WorkloadSpec, 15> kWorkloads = {{
    {"crossover_32x24", {32, 24}, false},
    {"crossover_64x48", {64, 48}, false},
    {"crossover_96x64", {96, 64}, false},
    {"synthetic_128x96", {128, 96}, false},
    {"workflow_gradient_128x96", {128, 96}, false, true},
    {"crossover_192x128", {192, 128}, false},
    {"crossover_256x192", {256, 192}, false},
    {"crossover_512x384", {512, 384}, false},
    {"odd_121x89", {121, 89}, false},
    {"flower_510x532", {510, 532}, true},
    {"padded_480p", {854, 479}, false},
    {"padded_720p", {1279, 719}, false},
    {"padded_1080p", {1919, 1079}, false},
    {"padded_1440p", {2559, 1439}, false, false, true},
    {"padded_4k", {3839, 2159}, false, false, true},
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

template <typename T>
[[nodiscard]] bool PlanesEqual(gjxl::PlaneView<const T> left,
                               gjxl::PlaneView<const T> right) {
  if (left.extent != right.extent) return false;
  for (size_t y = 0; y < left.extent.height; ++y) {
    if (!std::equal(left.Row(y), left.Row(y) + left.extent.width,
                    right.Row(y))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool GridsEqual(const gjxl::AcStrategyGrid& left,
                              const gjxl::AcStrategyGrid& right) {
  if (left.extent() != right.extent()) return false;
  for (size_t y = 0; y < left.extent().height; ++y) {
    for (size_t x = 0; x < left.extent().width; ++x) {
      gjxl::AcStrategyCell left_cell;
      gjxl::AcStrategyCell right_cell;
      if (!left.Get(x, y, &left_cell).ok() ||
          !right.Get(x, y, &right_cell).ok() ||
          left_cell.strategy != right_cell.strategy ||
          left_cell.is_anchor != right_cell.is_anchor) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] std::string FrameDifference(
    const gjxl::VarDctEncoderFrame& cpu,
    const gjxl::VarDctEncoderFrame& gpu) {
  if (!GridsEqual(cpu.strategies(), gpu.strategies())) return "strategy_grid";
  if (!PlanesEqual(cpu.raw_quant_field(), gpu.raw_quant_field())) {
    return "raw_quant_field";
  }
  if (!PlanesEqual(cpu.epf_sharpness(), gpu.epf_sharpness())) {
    return "epf_sharpness";
  }
  if (cpu.quantizer().params().global_scale !=
          gpu.quantizer().params().global_scale ||
      cpu.quantizer().params().quant_dc != gpu.quantizer().params().quant_dc) {
    return "quantizer";
  }
  const gjxl::ColorCorrelationMap& cpu_cfl = cpu.color_correlation();
  const gjxl::ColorCorrelationMap& gpu_cfl = gpu.color_correlation();
  if (!PlanesEqual(cpu_cfl.y_to_x_map(), gpu_cfl.y_to_x_map()) ||
      !PlanesEqual(cpu_cfl.y_to_b_map(), gpu_cfl.y_to_b_map())) {
    return "color_correlation";
  }
  const gjxl::ConstImage3I32View cpu_dc = cpu.quantized_dc();
  const gjxl::ConstImage3I32View gpu_dc = gpu.quantized_dc();
  const gjxl::ConstImage3FView cpu_reconstructed_dc = cpu.dc();
  const gjxl::ConstImage3FView gpu_reconstructed_dc = gpu.dc();
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!PlanesEqual(cpu_dc.plane[channel], gpu_dc.plane[channel])) {
      return "quantized_dc";
    }
    if (!PlanesEqual(cpu_reconstructed_dc.plane[channel],
                     gpu_reconstructed_dc.plane[channel])) {
      return "reconstructed_dc";
    }
  }
  if (cpu.ac_group_count() != gpu.ac_group_count()) return "ac_group_count";
  for (size_t group_index = 0; group_index < cpu.ac_group_count();
       ++group_index) {
    gjxl::VarDctAcGroupView cpu_group;
    gjxl::VarDctAcGroupView gpu_group;
    if (!cpu.GetAcGroup(group_index, &cpu_group).ok() ||
        !gpu.GetAcGroup(group_index, &gpu_group).ok() ||
        cpu_group.used_coefficient_count !=
            gpu_group.used_coefficient_count) {
      return "ac_group_shape";
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      if (!std::equal(cpu_group.coefficients[channel].begin(),
                      cpu_group.coefficients[channel].end(),
                      gpu_group.coefficients[channel].begin())) {
        return "quantized_ac";
      }
    }
  }
  return "none";
}

struct FrameCoefficientError {
  size_t raw_quant_count = 0;
  int64_t raw_quant_max_delta = 0;
  size_t quantized_dc_count = 0;
  int64_t quantized_dc_max_delta = 0;
  size_t quantized_ac_count = 0;
  int64_t quantized_ac_max_delta = 0;
  double reconstructed_dc_max_error = 0.0;
};

[[nodiscard]] FrameCoefficientError CompareFrameCoefficients(
    const gjxl::VarDctEncoderFrame& expected,
    const gjxl::VarDctEncoderFrame& actual) {
  FrameCoefficientError result;
  const gjxl::ConstPlaneI32View expected_raw = expected.raw_quant_field();
  const gjxl::ConstPlaneI32View actual_raw = actual.raw_quant_field();
  for (size_t y = 0; y < expected_raw.extent.height; ++y) {
    for (size_t x = 0; x < expected_raw.extent.width; ++x) {
      const int64_t delta = std::abs(
          static_cast<int64_t>(expected_raw.Row(y)[x]) -
          actual_raw.Row(y)[x]);
      result.raw_quant_count += delta != 0;
      result.raw_quant_max_delta =
          std::max(result.raw_quant_max_delta, delta);
    }
  }
  const gjxl::ConstImage3I32View expected_dc = expected.quantized_dc();
  const gjxl::ConstImage3I32View actual_dc = actual.quantized_dc();
  const gjxl::ConstImage3FView expected_reconstructed_dc = expected.dc();
  const gjxl::ConstImage3FView actual_reconstructed_dc = actual.dc();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < expected_dc.extent().height; ++y) {
      for (size_t x = 0; x < expected_dc.extent().width; ++x) {
        const int64_t delta = std::abs(
            static_cast<int64_t>(expected_dc.plane[channel].Row(y)[x]) -
            actual_dc.plane[channel].Row(y)[x]);
        result.quantized_dc_count += delta != 0;
        result.quantized_dc_max_delta =
            std::max(result.quantized_dc_max_delta, delta);
        result.reconstructed_dc_max_error = std::max(
            result.reconstructed_dc_max_error,
            std::abs(static_cast<double>(
                         expected_reconstructed_dc.plane[channel].Row(y)[x]) -
                     actual_reconstructed_dc.plane[channel].Row(y)[x]));
      }
    }
  }
  const size_t group_count =
      std::min(expected.ac_group_count(), actual.ac_group_count());
  for (size_t group_index = 0; group_index < group_count; ++group_index) {
    gjxl::VarDctAcGroupView expected_group;
    gjxl::VarDctAcGroupView actual_group;
    if (!expected.GetAcGroup(group_index, &expected_group).ok() ||
        !actual.GetAcGroup(group_index, &actual_group).ok()) {
      continue;
    }
    const size_t coefficient_count = std::min(
        expected_group.used_coefficient_count,
        actual_group.used_coefficient_count);
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t index = 0; index < coefficient_count; ++index) {
        const int64_t delta = std::abs(
            static_cast<int64_t>(expected_group.coefficients[channel][index]) -
            actual_group.coefficients[channel][index]);
        result.quantized_ac_count += delta != 0;
        result.quantized_ac_max_delta =
            std::max(result.quantized_ac_max_delta, delta);
      }
    }
  }
  return result;
}

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

[[nodiscard]] float ParsePositiveFloat(std::string_view text) {
  if (text.empty()) {
    throw std::runtime_error("Benchmark floating-point argument is invalid");
  }
  size_t parsed = 0;
  const double value = std::stod(std::string(text), &parsed);
  if (parsed != text.size() || !std::isfinite(value) || value <= 0.0 ||
      value > std::numeric_limits<float>::max()) {
    throw std::runtime_error("Benchmark floating-point argument is invalid");
  }
  return static_cast<float>(value);
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
  throw std::runtime_error("Unknown GPU AQ mode: " + std::string(text));
}

[[nodiscard]] gjxl::gpu_profile_internal::GpuProfilingMode
ParseGpuProfilingMode(std::string_view text) {
  if (text == "stage") {
    return gjxl::gpu_profile_internal::GpuProfilingMode::kStage;
  }
  if (text == "dispatch") {
    return gjxl::gpu_profile_internal::GpuProfilingMode::kDispatch;
  }
  throw std::runtime_error("Unknown GPU profiling mode: " +
                           std::string(text));
}

[[nodiscard]] BenchmarkScope ParseBenchmarkScope(std::string_view text) {
  if (text == "full") {
    return BenchmarkScope::kFull;
  }
  if (text == "public-workflow") {
    return BenchmarkScope::kPublicWorkflow;
  }
  if (text == "metal-public-workflow") {
    return BenchmarkScope::kMetalPublicWorkflow;
  }
  if (text == "coefficient-coding") {
    return BenchmarkScope::kCoefficientCoding;
  }
  throw std::runtime_error("Unknown benchmark scope: " + std::string(text));
}

[[nodiscard]] std::string_view BenchmarkScopeName(BenchmarkScope scope) {
  switch (scope) {
    case BenchmarkScope::kFull:
      return "full";
    case BenchmarkScope::kPublicWorkflow:
      return "public-workflow";
    case BenchmarkScope::kMetalPublicWorkflow:
      return "metal-public-workflow";
    case BenchmarkScope::kCoefficientCoding:
      return "coefficient-coding";
  }
  return "invalid";
}

[[nodiscard]] ValidationMode ParseValidationMode(std::string_view text) {
  if (text == "cpu-metal") {
    return ValidationMode::kCpuMetal;
  }
  if (text == "metal-only") {
    return ValidationMode::kMetalOnly;
  }
  throw std::runtime_error("Unknown benchmark validation mode: " +
                           std::string(text));
}

[[nodiscard]] std::string_view ValidationModeName(ValidationMode mode) {
  switch (mode) {
    case ValidationMode::kCpuMetal:
      return "cpu-metal";
    case ValidationMode::kMetalOnly:
      return "metal-only";
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

[[nodiscard]] CommandLineOptions ParseCommandLine(int argc, char** argv) {
  CommandLineOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << "usage: gjxl_encoding_benchmark "
                   "[--workload NAME|all] "
                   "[--input IMAGE.ppm|IMAGE.pfm] "
                   "[--scope full|public-workflow|metal-public-workflow|"
                   "coefficient-coding] "
                   "[--implementation scalar|simd|factored] "
                   "[--gpu-aq exact-coefficients|fully-resident|throughput|"
                   "maximum-throughput] "
                   "[--density default|high] "
                   "[--validation cpu-metal|metal-only] "
                   "[--collect-final-score] "
                   "[--metallib PATH] [--raw-samples PATH] "
                   "[--gpu-profile stage|dispatch] "
                   "[--gpu-profile-output PATH] "
                   "[--distance D] [--warmups N] [--samples N]\n";
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--collect-final-score") {
      options.collect_final_butteraugli_score = true;
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
    } else if (argument == "--metallib") {
      options.metallib_path = value;
    } else if (argument == "--raw-samples") {
      options.raw_samples_path = value;
    } else if (argument == "--gpu-profile") {
      options.gpu_profiling_mode = ParseGpuProfilingMode(value);
    } else if (argument == "--gpu-profile-output") {
      options.gpu_profile_path = value;
    } else if (argument == "--scope") {
      options.scope = ParseBenchmarkScope(value);
    } else if (argument == "--validation") {
      options.validation = ParseValidationMode(value);
    } else if (argument == "--implementation") {
      if (value != "scalar" && value != "simd" && value != "factored") {
        throw std::runtime_error(
            "Unknown Metal DCT implementation: " + std::string(value));
      }
      options.implementation = value;
    } else if (argument == "--gpu-aq") {
      options.gpu_aq_mode = ParseGpuAqMode(value);
    } else if (argument == "--density") {
      if (value == "default") {
        options.density_mode = gjxl::VarDctDensityMode::kDefault;
      } else if (value == "high") {
        options.density_mode = gjxl::VarDctDensityMode::kHighDensity;
      } else {
        throw std::runtime_error(
          "Unknown density mode: " + std::string(value));
      }
    } else if (argument == "--distance") {
      options.butteraugli_target = ParsePositiveFloat(value);
    } else if (argument == "--warmups") {
      options.warmups = ParseSize(value, true);
    } else if (argument == "--samples") {
      options.samples = ParseSize(value, false);
    } else {
      throw std::runtime_error("Unknown encoding benchmark option: " +
                               std::string(argument));
    }
  }
  if (options.gpu_aq_mode ==
        gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput &&
      options.scope != BenchmarkScope::kPublicWorkflow &&
      options.scope != BenchmarkScope::kMetalPublicWorkflow) {
    throw std::runtime_error(
      "Maximum-throughput mode requires a public-workflow scope");
  }
  if (options.density_mode == gjxl::VarDctDensityMode::kHighDensity &&
      (options.scope != BenchmarkScope::kPublicWorkflow &&
       options.scope != BenchmarkScope::kMetalPublicWorkflow)) {
    throw std::runtime_error(
      "High density requires a public-workflow scope");
  }
  if (options.density_mode == gjxl::VarDctDensityMode::kHighDensity &&
      (options.gpu_aq_mode ==
         gjxl::GpuAdaptiveQuantizationMode::kThroughput ||
       options.gpu_aq_mode ==
         gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput)) {
    throw std::runtime_error(
      "High density is incompatible with throughput AQ");
  }
  if (options.validation == ValidationMode::kMetalOnly &&
      options.scope != BenchmarkScope::kMetalPublicWorkflow) {
    throw std::runtime_error(
        "Metal-only validation requires metal-public-workflow scope");
  }
  if (!options.raw_samples_path.empty() &&
      options.scope != BenchmarkScope::kPublicWorkflow &&
      options.scope != BenchmarkScope::kMetalPublicWorkflow) {
    throw std::runtime_error(
        "Raw workflow samples require a public-workflow scope");
  }
  const bool gpu_profiling = options.gpu_profiling_mode !=
    gjxl::gpu_profile_internal::GpuProfilingMode::kDisabled;
  if (gpu_profiling != !options.gpu_profile_path.empty()) {
    throw std::runtime_error(
      "GPU profiling mode and output path must be specified together");
  }
  if (gpu_profiling &&
      (options.scope != BenchmarkScope::kMetalPublicWorkflow ||
       options.validation != ValidationMode::kMetalOnly ||
       (options.gpu_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kFullyResident &&
        options.gpu_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kThroughput))) {
    throw std::runtime_error(
      "GPU profiling requires a resident metal-only public workflow");
  }
  if (gpu_profiling && !options.raw_samples_path.empty()) {
    throw std::runtime_error(
      "GPU profiling output is separate from raw workflow samples");
  }
  if (gpu_profiling &&
      options.density_mode == gjxl::VarDctDensityMode::kHighDensity) {
    throw std::runtime_error(
      "High density is unavailable for GPU dispatch profiling");
  }
  return options;
}

[[nodiscard]] gjxl::MetalBackendOptions BackendOptions(
    std::string_view name) {
  using Implementation = gjxl::MetalDctImplementation;
  Implementation implementation = Implementation::kSimdgroupMatmul;
  if (name == "scalar") {
    implementation = Implementation::kScalarMatmul;
  } else if (name == "factored") {
    implementation = Implementation::kFactoredRadix2;
  }
  return {
      .forward_dct8 = implementation,
      .inverse_dct8 = implementation,
      .forward_dct16x16 = implementation,
      .inverse_dct16x16 = implementation,
      .forward_dct32x32 = implementation,
      .inverse_dct32x32 = implementation,
      .forward_dct16x8 = implementation,
      .inverse_dct16x8 = implementation,
      .forward_dct8x16 = implementation,
      .inverse_dct8x16 = implementation,
      .forward_dct32x16 = implementation,
      .inverse_dct32x16 = implementation,
      .forward_dct16x32 = implementation,
      .inverse_dct16x32 = implementation,
  };
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
    throw std::runtime_error("Malformed PPM header");
  }
  return token;
}

[[nodiscard]] ImageStorage LoadPpm(std::string_view path) {
  std::ifstream input(std::string(path), std::ios::binary);
  if (!input || ReadPpmToken(&input) != "P6") {
    throw std::runtime_error("Unable to open binary PPM: " +
                             std::string(path));
  }
  const size_t width = std::stoull(ReadPpmToken(&input));
  const size_t height = std::stoull(ReadPpmToken(&input));
  const unsigned long maximum = std::stoul(ReadPpmToken(&input));
  if (maximum != 255) {
    throw std::runtime_error("Benchmark PPM must have 8-bit samples");
  }
  char separator = 0;
  input.get(separator);
  size_t pixel_count = 0;
  const gjxl::Extent2D extent{width, height};
  if (!input || !std::isspace(static_cast<unsigned char>(separator)) ||
      !extent.try_area(&pixel_count) ||
      pixel_count > std::numeric_limits<size_t>::max() / 3) {
    throw std::runtime_error("Benchmark PPM is malformed or too large");
  }
  std::vector<uint8_t> bytes(pixel_count * 3);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("Benchmark PPM pixel data is truncated");
  }

  ImageStorage image(extent);
  for (size_t index = 0; index < pixel_count; ++index) {
    for (size_t channel = 0; channel < 3; ++channel) {
      image.plane[channel][index] = SrgbToLinear(bytes[index * 3 + channel]);
    }
  }
  return image;
}

[[nodiscard]] ImageStorage LoadPfm(std::string_view path) {
  gjxl::Image3FBuffer decoded;
  const gjxl::Status status = gjxl::io::ReadPfm(
    std::filesystem::path(path), &decoded);
  if (!status.ok()) {
    throw std::runtime_error(
      "Unable to open benchmark PFM: " + std::string(path) + ": " +
      std::string(status.message()));
  }
  ImageStorage image(decoded.extent());
  for (size_t channel = 0; channel < image.plane.size(); ++channel) {
    std::copy(
      decoded.plane(channel).begin(), decoded.plane(channel).end(),
      image.plane[channel].begin());
  }
  return image;
}

[[nodiscard]] ImageStorage LoadBenchmarkImage(std::string_view path) {
  std::ifstream input(std::string(path), std::ios::binary);
  std::array<char, 2> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input) {
    throw std::runtime_error(
      "Unable to inspect benchmark image: " + std::string(path));
  }
  return magic[0] == 'P' && (magic[1] == 'F' || magic[1] == 'f')
    ? LoadPfm(path)
    : LoadPpm(path);
}

[[nodiscard]] ImageStorage LoadFlower() {
  ImageStorage image = LoadPpm(GJXL_FLOWER_PPM_PATH);
  if (image.extent != gjxl::Extent2D{510, 532}) {
    throw std::runtime_error("Pinned Flower PPM dimensions changed");
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

void FillWorkflowGradient(ImageStorage* image) {
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      const float fx = static_cast<float>(x) /
          static_cast<float>(image->extent.width - 1);
      const float fy = static_cast<float>(y) /
          static_cast<float>(image->extent.height - 1);
      image->plane[0][y * image->extent.width + x] = 0.06f + 0.78f * fx;
      image->plane[1][y * image->extent.width + x] =
          0.08f + 0.72f * fy + 0.03f * std::sin(19.0f * fx);
      image->plane[2][y * image->extent.width + x] =
          0.04f + 0.31f * fx + 0.46f * fy;
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

void PrintRatioStats(std::string_view label,
                     const std::vector<double>& samples) {
  const TimingStats stats = Summarize(samples);
  std::cout << "  ratio " << label << " median=" << stats.median_ms
            << " range=[" << stats.minimum_ms << ',' << stats.maximum_ms
            << "]\n";
}

[[nodiscard]] double NanosecondsToMilliseconds(uint64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1.0e6;
}

constexpr std::array<std::string_view, 36> kWorkflowProfileNames = {
    "total",
    "input_preparation",
    "backend_selection",
    "quantization_pipeline",
    "codestream_encoding",
    "summary_assembly",
    "codestream_validation",
    "codestream_dc_tokenization",
    "codestream_ac_tokenization",
    "codestream_block_context_map_work",
    "codestream_coefficient_order_work",
    "codestream_coefficient_tokenization_work",
    "codestream_entropy_optimization",
    "codestream_entropy_prefix_histogram_build_work",
    "codestream_entropy_prefix_histogram_cost_work",
    "codestream_entropy_prefix_clustering_work",
    "codestream_entropy_prefix_code_build_work",
    "codestream_entropy_prefix_uint_config_work",
    "codestream_entropy_ans_prefix_validation_work",
    "codestream_entropy_ans_value_collection_work",
    "codestream_entropy_ans_value_aggregation_work",
    "codestream_entropy_ans_uint_config_work",
    "codestream_entropy_ans_histogram_build_work",
    "codestream_entropy_ans_model_build_work",
    "codestream_entropy_ans_token_cost_work",
    "codestream_entropy_selection_work",
    "codestream_section_writing",
    "codestream_section_model_and_header_work",
    "codestream_section_token_write_work",
    "codestream_section_candidate_measure_work",
    "codestream_assembly",
    "codestream_assembly_candidate_selection",
    "codestream_assembly_section_size",
    "codestream_assembly_frame_header",
    "codestream_assembly_toc_and_sections",
    "codestream_assembly_output_copy",
};

struct RawWorkflowSample {
  size_t sample_index = 0;
  std::string_view backend;
  std::array<uint64_t, kWorkflowProfileNames.size()> phase_nanoseconds{};
  size_t encoded_bytes = 0;
  uint64_t entropy_model_bits = 0;
  uint64_t entropy_token_bits = 0;
  size_t dc_entropy_clusters = 0;
  size_t ac_entropy_clusters = 0;
  bool dc_entropy_is_ans = false;
  bool ac_entropy_is_ans = false;
  bool coefficient_order_entropy_is_ans = false;
  size_t natural_candidate_bytes = 0;
  size_t custom_order_candidate_bytes = 0;
  uint16_t selected_coefficient_order_mask = 0;
  size_t block_context_candidate_count = 0;
  size_t compact_block_context_candidate_bytes = 0;
  size_t selected_block_context_candidate_index = 0;
  size_t selected_block_context_count = 0;
  size_t selected_block_context_qf_threshold_count = 0;
  bool has_final_score = false;
  double final_score = 0.0;
};

struct RawWorkflowWorkload {
  std::string workload;
  gjxl::Extent2D source_extent;
  std::string codestream_comparison;
  std::vector<RawWorkflowSample> samples;
};

struct RawGpuProfileSample {
  size_t sample_index = 0;
  gjxl::gpu_profile_internal::GpuExecutionProfile profile;
};

struct RawGpuProfileWorkload {
  std::string workload;
  gjxl::Extent2D source_extent;
  std::vector<RawGpuProfileSample> samples;
};

using WorkflowProfileSamples =
    std::array<std::vector<double>, kWorkflowProfileNames.size()>;

using WorkflowProfileNanoseconds =
    std::array<uint64_t, kWorkflowProfileNames.size()>;

[[nodiscard]] WorkflowProfileNanoseconds WorkflowProfileValues(
    const gjxl::codestream_internal::VarDctEncodingProfile& profile) {
  return {
      profile.total_nanoseconds,
      profile.input_preparation_nanoseconds,
      profile.backend_selection_nanoseconds,
      profile.quantization_pipeline_nanoseconds,
      profile.codestream_encoding_nanoseconds,
      profile.summary_assembly_nanoseconds,
      profile.codestream.validation_nanoseconds,
      profile.codestream.dc_tokenization_nanoseconds,
      profile.codestream.ac_tokenization_nanoseconds,
      profile.codestream.block_context_map_work_nanoseconds,
      profile.codestream.coefficient_order_work_nanoseconds,
      profile.codestream.coefficient_tokenization_work_nanoseconds,
      profile.codestream.entropy_optimization_nanoseconds,
      profile.codestream.entropy_work.prefix_histogram_build_nanoseconds,
      profile.codestream.entropy_work.prefix_histogram_cost_nanoseconds,
      profile.codestream.entropy_work.prefix_clustering_nanoseconds,
      profile.codestream.entropy_work.prefix_code_build_nanoseconds,
      profile.codestream.entropy_work.prefix_uint_config_nanoseconds,
      profile.codestream.entropy_work.ans_prefix_validation_nanoseconds,
      profile.codestream.entropy_work.ans_value_collection_nanoseconds,
      profile.codestream.entropy_work.ans_value_aggregation_nanoseconds,
      profile.codestream.entropy_work.ans_uint_config_nanoseconds,
      profile.codestream.entropy_work.ans_histogram_build_nanoseconds,
      profile.codestream.entropy_work.ans_model_build_nanoseconds,
      profile.codestream.entropy_work.ans_token_cost_nanoseconds,
      profile.codestream.entropy_work.selection_nanoseconds,
      profile.codestream.section_writing_nanoseconds,
      profile.codestream.section_writing_work.model_and_header_nanoseconds,
      profile.codestream.section_writing_work.token_write_nanoseconds,
      profile.codestream.section_writing_work.candidate_measure_nanoseconds,
      profile.codestream.assembly_nanoseconds,
      profile.codestream.assembly.candidate_selection_nanoseconds,
      profile.codestream.assembly.section_size_nanoseconds,
      profile.codestream.assembly.frame_header_nanoseconds,
      profile.codestream.assembly.toc_and_sections_nanoseconds,
      profile.codestream.assembly.output_copy_nanoseconds,
  };
}

void AppendWorkflowProfile(
    const gjxl::codestream_internal::VarDctEncodingProfile& profile,
    WorkflowProfileSamples* samples) {
  const WorkflowProfileNanoseconds values = WorkflowProfileValues(profile);
  for (size_t index = 0; index < values.size(); ++index) {
    (*samples)[index].push_back(NanosecondsToMilliseconds(values[index]));
  }
}

[[nodiscard]] std::string JsonEscape(std::string_view value) {
  std::ostringstream escaped;
  for (const unsigned char character : value) {
    switch (character) {
      case '\"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (character < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4)
                  << std::setfill('0') << static_cast<unsigned>(character)
                  << std::dec << std::setfill(' ');
        } else {
          escaped << static_cast<char>(character);
        }
    }
  }
  return escaped.str();
}

void WriteRawWorkflowSamples(
    const std::filesystem::path& destination,
    const CommandLineOptions& options,
    const std::vector<RawWorkflowWorkload>& workloads) {
  std::filesystem::path temporary = destination;
  const uint64_t suffix = static_cast<uint64_t>(
      Clock::now().time_since_epoch().count());
  temporary += ".tmp-" + std::to_string(suffix);

  try {
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(temporary, std::ios::out | std::ios::trunc);
    output << "{\n"
           << "  \"schema_version\": 7,\n"
           << "  \"substage_work_timing\": \"aggregate-worker-time\",\n"
           << "  \"scope\": \"" << BenchmarkScopeName(options.scope)
           << "\",\n"
           << "  \"validation\": \""
           << ValidationModeName(options.validation) << "\",\n"
           << "  \"implementation\": \""
           << JsonEscape(options.implementation) << "\",\n"
           << "  \"gpu_aq\": \"" << GpuAqModeName(options.gpu_aq_mode)
           << "\",\n"
           << "  \"collect_final_score\": "
           << (options.collect_final_butteraugli_score ? "true" : "false")
           << ",\n"
           << "  \"density\": \""
           << (options.density_mode == gjxl::VarDctDensityMode::kHighDensity
                 ? "high"
                 : "default")
           << "\",\n"
           << "  \"distance\": " << std::setprecision(9)
           << options.butteraugli_target << ",\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"sample_count\": " << options.samples << ",\n"
           << "  \"workloads\": [\n";
    for (size_t workload_index = 0; workload_index < workloads.size();
         ++workload_index) {
      const RawWorkflowWorkload& workload = workloads[workload_index];
      output << "    {\n"
             << "      \"name\": \"" << JsonEscape(workload.workload)
             << "\",\n"
             << "      \"source_width\": " << workload.source_extent.width
             << ",\n"
             << "      \"source_height\": " << workload.source_extent.height
             << ",\n"
             << "      \"codestream_comparison\": \""
             << workload.codestream_comparison << "\",\n"
             << "      \"samples\": [\n";
      for (size_t sample_index = 0; sample_index < workload.samples.size();
           ++sample_index) {
        const RawWorkflowSample& sample = workload.samples[sample_index];
        output << "        {\"sample_index\": " << sample.sample_index
               << ", \"backend\": \"" << sample.backend
               << "\", \"encoded_bytes\": " << sample.encoded_bytes
               << ", \"entropy_bits\": {\"model\": "
               << sample.entropy_model_bits << ", \"tokens\": "
               << sample.entropy_token_bits << "}"
               << ", \"entropy_clusters\": {\"dc\": "
               << sample.dc_entropy_clusters << ", \"ac\": "
               << sample.ac_entropy_clusters << "}"
               << ", \"entropy_coding\": {\"dc\": \""
               << (sample.dc_entropy_is_ans ? "ans" : "prefix")
               << "\", \"ac\": \""
               << (sample.ac_entropy_is_ans ? "ans" : "prefix")
               << "\", \"coefficient_order\": \""
               << (sample.selected_coefficient_order_mask == 0
                     ? "none"
                     : sample.coefficient_order_entropy_is_ans
                         ? "ans"
                         : "prefix")
               << "\"}"
               << ", \"coefficient_order\": {\"natural_bytes\": "
               << sample.natural_candidate_bytes << ", \"custom_bytes\": "
               << sample.custom_order_candidate_bytes
               << ", \"selected_mask\": "
               << sample.selected_coefficient_order_mask << "}"
               << ", \"block_context\": {\"candidate_count\": "
               << sample.block_context_candidate_count
               << ", \"compact_bytes\": "
               << sample.compact_block_context_candidate_bytes
               << ", \"selected_index\": "
               << sample.selected_block_context_candidate_index
               << ", \"selected_contexts\": "
               << sample.selected_block_context_count
               << ", \"qf_thresholds\": "
               << sample.selected_block_context_qf_threshold_count << "}"
               << ", \"final_score\": ";
        if (sample.has_final_score) {
          output << std::setprecision(17) << sample.final_score;
        } else {
          output << "null";
        }
        output << ", \"phase_nanoseconds\": {";
        for (size_t phase = 0; phase < kWorkflowProfileNames.size(); ++phase) {
          if (phase != 0) {
            output << ", ";
          }
          output << '\"' << kWorkflowProfileNames[phase] << "\": "
                 << sample.phase_nanoseconds[phase];
        }
        output << "}}";
        if (sample_index + 1 != workload.samples.size()) {
          output << ',';
        }
        output << '\n';
      }
      output << "      ]\n"
             << "    }";
      if (workload_index + 1 != workloads.size()) {
        output << ',';
      }
      output << '\n';
    }
    output << "  ]\n}\n";
    output.close();

    std::error_code rename_error;
    std::filesystem::rename(temporary, destination, rename_error);
    if (rename_error) {
      throw std::runtime_error(
          "Could not atomically replace raw-samples output: " +
          rename_error.message());
    }
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

[[nodiscard]] std::string_view GpuProfilingModeName(
    gjxl::gpu_profile_internal::GpuProfilingMode mode) {
  switch (mode) {
    case gjxl::gpu_profile_internal::GpuProfilingMode::kDisabled:
      return "disabled";
    case gjxl::gpu_profile_internal::GpuProfilingMode::kStage:
      return "stage";
    case gjxl::gpu_profile_internal::GpuProfilingMode::kDispatch:
      return "dispatch";
  }
  return "invalid";
}

[[nodiscard]] std::string_view GpuWallStageKindName(
    gjxl::gpu_profile_internal::GpuWallStageKind kind) {
  switch (kind) {
    case gjxl::gpu_profile_internal::GpuWallStageKind::kOperation:
      return "operation";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kPreparation:
      return "preparation";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kUpload:
      return "upload";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kWait:
      return "wait";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kReadback:
      return "readback";
    case gjxl::gpu_profile_internal::GpuWallStageKind::kHost:
      return "host";
  }
  return "invalid";
}

void WriteGpuProfileSamples(
    const std::filesystem::path& destination,
    const CommandLineOptions& options,
    const std::vector<RawGpuProfileWorkload>& workloads) {
  const std::filesystem::path temporary =
    destination.string() + ".tmp-" + std::to_string(
      static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    if (!destination.parent_path().empty()) {
      std::filesystem::create_directories(destination.parent_path());
    }
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(temporary, std::ios::out | std::ios::trunc);
    output << "{\n"
           << "  \"schema_version\": 3,\n"
           << "  \"scope\": \"metal-public-workflow\",\n"
           << "  \"mode\": \""
           << GpuProfilingModeName(options.gpu_profiling_mode) << "\",\n"
           << "  \"gpu_aq\": \"" << GpuAqModeName(options.gpu_aq_mode)
           << "\",\n"
           << "  \"collect_final_score\": "
           << (options.collect_final_butteraugli_score ? "true" : "false")
           << ",\n"
           << "  \"distance\": " << std::setprecision(9)
           << options.butteraugli_target << ",\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"sample_count\": " << options.samples << ",\n"
           << "  \"workloads\": [\n";
    for (size_t workload_index = 0; workload_index < workloads.size();
         ++workload_index) {
      const RawGpuProfileWorkload& workload = workloads[workload_index];
      output << "    {\n"
             << "      \"name\": \"" << JsonEscape(workload.workload)
             << "\",\n"
             << "      \"source_width\": " << workload.source_extent.width
             << ",\n"
             << "      \"source_height\": " << workload.source_extent.height
             << ",\n"
             << "      \"samples\": [\n";
      for (size_t sample_index = 0; sample_index < workload.samples.size();
           ++sample_index) {
        const RawGpuProfileSample& sample = workload.samples[sample_index];
        const auto& profile = sample.profile;
        output << "        {\n"
               << "          \"sample_index\": " << sample.sample_index
               << ",\n"
               << "          \"capabilities\": {"
               << "\"timestamp_counter\": "
               << (profile.capabilities.timestamp_counter ? "true" : "false")
               << ", \"stage_boundary\": "
               << (profile.capabilities.stage_boundary ? "true" : "false")
               << ", \"dispatch_boundary\": "
               << (profile.capabilities.dispatch_boundary ? "true" : "false")
               << "},\n"
               << "          \"wall_stages\": [";
        for (size_t wall_index = 0;
             wall_index < profile.wall_stages.size(); ++wall_index) {
          const auto& wall = profile.wall_stages[wall_index];
          if (wall_index != 0) output << ", ";
          output << "{\"stage_id\": \"" << JsonEscape(wall.stage_id)
                 << "\", \"kind\": \""
                 << GpuWallStageKindName(wall.kind)
                 << "\", \"invocation\": " << wall.invocation
                 << ", \"wall_nanoseconds\": " << wall.wall_nanoseconds
                 << '}';
        }
        output << "],\n"
               << "          \"submissions\": [\n";
        for (size_t submission_index = 0;
             submission_index < profile.submissions.size();
             ++submission_index) {
          const auto& submission = profile.submissions[submission_index];
          output << "            {\"submission_index\": "
                 << submission_index
                 << ", \"submission_id\": \""
                 << JsonEscape(submission.submission_id)
                 << "\", \"invocation\": " << submission.invocation
                 << ", \"command_buffer_gpu_nanoseconds\": "
                 << submission.command_buffer_gpu_nanoseconds
                 << ", \"stages\": [\n";
          for (size_t stage_index = 0;
               stage_index < submission.stages.size(); ++stage_index) {
            const auto& stage = submission.stages[stage_index];
            output << "              {\"stage_id\": \""
                   << JsonEscape(stage.stage_id)
                   << "\", \"group_id\": \""
                   << JsonEscape(stage.group_id)
                   << "\", \"iteration\": " << stage.iteration
                   << ", \"invocation\": " << stage.invocation
                   << ", \"begin_timestamp\": " << stage.begin_timestamp
                   << ", \"end_timestamp\": " << stage.end_timestamp
                   << ", \"gpu_nanoseconds\": " << stage.gpu_nanoseconds
                   << ", \"dispatches\": [";
            for (size_t dispatch_index = 0;
                 dispatch_index < stage.dispatches.size(); ++dispatch_index) {
              const auto& dispatch = stage.dispatches[dispatch_index];
              if (dispatch_index != 0) output << ", ";
              output << "{\"kernel_id\": \""
                     << JsonEscape(dispatch.kernel_id)
                     << "\", \"kind\": \""
                     << (dispatch.kind ==
                           gjxl::gpu_profile_internal::GpuDispatchKind::kThreads
                           ? "threads" : "threadgroups")
                     << "\", \"invocation\": " << dispatch.invocation
                     << ", \"grid\": [" << dispatch.grid.width << ", "
                     << dispatch.grid.height << ", " << dispatch.grid.depth
                     << "], \"threads_per_threadgroup\": ["
                     << dispatch.threads_per_threadgroup.width << ", "
                     << dispatch.threads_per_threadgroup.height << ", "
                     << dispatch.threads_per_threadgroup.depth
                     << "], \"begin_timestamp\": "
                     << dispatch.begin_timestamp
                     << ", \"end_timestamp\": " << dispatch.end_timestamp
                     << ", \"gpu_nanoseconds\": "
                     << dispatch.gpu_nanoseconds << '}';
            }
            output << "]}";
            if (stage_index + 1 != submission.stages.size()) output << ',';
            output << '\n';
          }
          output << "            ]}";
          if (submission_index + 1 != profile.submissions.size()) output << ',';
          output << '\n';
        }
        output << "          ]\n        }";
        if (sample_index + 1 != workload.samples.size()) output << ',';
        output << '\n';
      }
      output << "      ]\n    }";
      if (workload_index + 1 != workloads.size()) output << ',';
      output << '\n';
    }
    output << "  ]\n}\n";
    output.close();
    std::error_code rename_error;
    std::filesystem::rename(temporary, destination, rename_error);
    if (rename_error) {
      throw std::runtime_error(
        "Could not atomically replace GPU-profile output: " +
        rename_error.message());
    }
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

void PrintWorkflowProfile(
    std::string_view backend, const WorkflowProfileSamples& samples) {
  for (size_t index = 0; index < samples.size(); ++index) {
    PrintStats(
        std::string(backend) + "." + std::string(kWorkflowProfileNames[index]),
        samples[index]);
  }
}

void RunCoefficientCodingOnlyWorkload(
    const WorkloadSpec& spec, size_t warmups, size_t samples,
    float butteraugli_target, std::string_view input_path,
    double* global_sink) {
  ImageStorage original = !input_path.empty()
      ? LoadBenchmarkImage(input_path)
      : (spec.flower ? LoadFlower() : ImageStorage(spec.source_extent));
  if (!spec.flower && input_path.empty()) {
    if (spec.workflow_gradient) {
      FillWorkflowGradient(&original);
    } else {
      FillSynthetic(&original);
    }
  }
  const gjxl::Extent2D coding_extent = PaddedExtent(original.extent);
  ImageStorage padded_linear(coding_extent);
  ImageStorage opsin(coding_extent);
  ImageStorage preprocessed(coding_extent);
  EdgePad(original, &padded_linear);
  RequireStatus(
      "linear-to-opsin setup",
      gjxl::LinearRgbToOpsin(
          padded_linear.ConstView(), 255.0f, opsin.View()));
  RequireStatus(
      "Gaborish setup",
      gjxl::ApplyGaborishInverse(
          opsin.ConstView(), {1.0f, 1.0f, 1.0f}, preprocessed.View()));

  StageOutput stage(original.extent, coding_extent);
  RequireStatus(
      "initial quantization setup",
      gjxl::ComputeInitialQuantField(
          opsin.ConstView(), {.butteraugli_target = butteraugli_target},
          stage.InitialOutput()));
  gjxl::ColorCorrelationMap initial_color_correlation;
  RequireStatus(
      "initial color-correlation setup",
      gjxl::ComputeInitialColorCorrelationMap(
          preprocessed.ConstView(), &initial_color_correlation));
  RequireStatus(
      "AC-strategy setup",
      gjxl::FindAcStrategyGrid(
          preprocessed.ConstView(),
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {stage.pixel_mask.data(), coding_extent, coding_extent.width},
          initial_color_correlation,
          {.butteraugli_target = butteraugli_target}, &stage.strategies));
  std::vector<uint8_t> sharpness(
      stage.block_extent.width * stage.block_extent.height);
  RequireStatus(
      "EPF-sharpness setup",
      gjxl::FillDefaultEpfSharpness(
          {sharpness.data(), stage.block_extent, stage.block_extent.width}));

  float quant_dc = 0.0f;
  RequireStatus(
      "initial DC quantization setup",
      gjxl::ComputeInitialQuantDc(butteraugli_target, &quant_dc));
  std::vector<int32_t> raw_quant(
      stage.block_extent.width * stage.block_extent.height);
  gjxl::Quantizer quantizer;
  RequireStatus(
      "quantizer setup",
      gjxl::CreateQuantizerFromField(
          quant_dc,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {raw_quant.data(), stage.block_extent, stage.block_extent.width},
          &quantizer));
  gjxl::ColorCorrelationMap color_correlation;
  RequireStatus(
      "final color-correlation setup",
      gjxl::ComputeFinalColorCorrelationMap(
          preprocessed.ConstView(), stage.strategies,
          {raw_quant.data(), stage.block_extent, stage.block_extent.width},
          quantizer, false, &color_correlation));
  gjxl::FrameGeometry geometry;
  RequireStatus(
      "frame geometry setup",
      gjxl::FrameGeometry::Create(original.extent, &geometry));

  gjxl::VarDctEncoderFrame frame;
  const auto run_coefficient_coding = [&] {
    return gjxl::ComputeQuantizedCoefficients(
        preprocessed.ConstView(),
        {
            .geometry = geometry,
            .strategies = &stage.strategies,
            .raw_quant_field = {
                raw_quant.data(), stage.block_extent, stage.block_extent.width},
            .quantizer = &quantizer,
            .color_correlation = &color_correlation,
            .epf_sharpness = {
                sharpness.data(), stage.block_extent, stage.block_extent.width},
        },
        {}, &frame);
  };
  for (size_t warmup = 0; warmup < warmups; ++warmup) {
    RequireStatus("coefficient-coding warmup", run_coefficient_coding());
  }

  std::vector<double> coding_samples;
  coding_samples.reserve(samples);
  double sink = 0.0;
  for (size_t sample = 0; sample < samples; ++sample) {
    const auto begin = Clock::now();
    RequireStatus("coefficient-coding sample", run_coefficient_coding());
    const auto end = Clock::now();
    coding_samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
    sink += static_cast<double>(frame.ac_group_count()) +
            static_cast<double>(frame.raw_quant_field().Row(0)[0]);
  }
  if (!frame.valid()) {
    throw std::runtime_error("Coefficient-coding benchmark frame is invalid");
  }
  std::cout << "workload " << spec.name << " source="
            << original.extent.width << 'x' << original.extent.height
            << " coding=" << coding_extent.width << 'x'
            << coding_extent.height << " distance=" << butteraugli_target
            << " scope=coefficient-coding\n";
  PrintStats("coefficient_coding", coding_samples);
  std::cout << "  sink=" << sink << '\n';
  *global_sink += sink;
}

void RunPublicWorkflowOnlyWorkload(
    const WorkloadSpec& spec, size_t warmups, size_t samples,
    float butteraugli_target,
    gjxl::GpuAdaptiveQuantizationMode gpu_aq_mode,
    gjxl::VarDctDensityMode density_mode,
    bool collect_final_butteraugli_score,
    std::string_view input_path, bool metal_only, ValidationMode validation,
    gjxl::GpuBackend& gpu,
    std::vector<RawWorkflowWorkload>* raw_results, double* global_sink) {
  ImageStorage original = !input_path.empty()
      ? LoadBenchmarkImage(input_path)
      : (spec.flower ? LoadFlower() : ImageStorage(spec.source_extent));
  if (!spec.flower && input_path.empty()) {
    if (spec.workflow_gradient) {
      FillWorkflowGradient(&original);
    } else {
      FillSynthetic(&original);
    }
  }

  const auto encode = [&](gjxl::VarDctBackendPreference backend,
                          gjxl::GpuAdaptiveQuantizationMode mode,
                          std::vector<uint8_t>* bytes,
                          gjxl::VarDctEncodingSummary* summary,
                          gjxl::codestream_internal::VarDctEncodingProfile*
                              profile) {
    return gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
            original.ConstView(),
            {.butteraugli_target = butteraugli_target,
             .density_mode = density_mode,
             .backend = backend,
             .metal_aq_mode = mode,
             .collect_final_butteraugli_score =
               collect_final_butteraugli_score},
            backend == gjxl::VarDctBackendPreference::kMetal ? &gpu : nullptr,
            true, bytes, summary, profile);
  };

  std::vector<uint8_t> cpu_bytes;
  std::vector<uint8_t> gpu_bytes;
  gjxl::VarDctEncodingSummary cpu_summary;
  gjxl::VarDctEncodingSummary gpu_summary;
  gjxl::codestream_internal::VarDctEncodingProfile ignored_profile;
  if (validation == ValidationMode::kCpuMetal) {
    RequireStatus(
        "CPU public-workflow validation",
        encode(gjxl::VarDctBackendPreference::kCpu,
               gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
               &cpu_bytes, &cpu_summary, &ignored_profile));
  }
  RequireStatus(
      "Metal public-workflow validation",
      encode(gjxl::VarDctBackendPreference::kMetal, gpu_aq_mode, &gpu_bytes,
             &gpu_summary, &ignored_profile));
  const bool codestreams_equal =
      validation == ValidationMode::kCpuMetal && cpu_bytes == gpu_bytes;
  if (gpu_bytes.size() < 2 ||
      gpu_bytes[0] != 0xff || gpu_bytes[1] != 0x0a ||
      gpu_summary.execution_backend != gjxl::VarDctExecutionBackend::kMetal ||
      (validation == ValidationMode::kCpuMetal &&
       (cpu_bytes.size() < 2 || cpu_bytes[0] != 0xff ||
        cpu_bytes[1] != 0x0a ||
        cpu_summary.execution_backend != gjxl::VarDctExecutionBackend::kCpu ||
        (gpu_aq_mode ==
           gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients &&
         !codestreams_equal)))) {
    throw std::runtime_error(
        "Public-workflow benchmark validation failed");
  }

  for (size_t warmup = 0; warmup < warmups; ++warmup) {
    gjxl::codestream_internal::VarDctEncodingProfile warmup_profile;
    if (metal_only) {
      RequireStatus(
          "Metal public-workflow warmup",
          encode(gjxl::VarDctBackendPreference::kMetal, gpu_aq_mode,
                 &gpu_bytes, &gpu_summary, &warmup_profile));
    } else if ((warmup & 1u) == 0) {
      RequireStatus(
          "CPU public-workflow warmup",
          encode(gjxl::VarDctBackendPreference::kCpu,
                 gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
                 &cpu_bytes, &cpu_summary, &warmup_profile));
      RequireStatus(
          "Metal public-workflow warmup",
          encode(gjxl::VarDctBackendPreference::kMetal, gpu_aq_mode,
                 &gpu_bytes, &gpu_summary, &warmup_profile));
    } else {
      RequireStatus(
          "Metal public-workflow warmup",
          encode(gjxl::VarDctBackendPreference::kMetal, gpu_aq_mode,
                 &gpu_bytes, &gpu_summary, &warmup_profile));
      RequireStatus(
          "CPU public-workflow warmup",
          encode(gjxl::VarDctBackendPreference::kCpu,
                 gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
                 &cpu_bytes, &cpu_summary, &warmup_profile));
    }
  }

  WorkflowProfileSamples cpu_samples;
  WorkflowProfileSamples gpu_samples;
  std::vector<double> paired_speedups;
  RawWorkflowWorkload raw_workload{
      .workload = std::string(spec.name),
      .source_extent = original.extent,
      .codestream_comparison =
          validation == ValidationMode::kCpuMetal
              ? (codestreams_equal ? "exact" : "different")
              : "not-compared",
  };
  raw_workload.samples.reserve(
      samples * (metal_only ? size_t{1} : size_t{2}));
  const auto append_raw_sample = [&](size_t sample_index,
                                     std::string_view backend,
                                     const gjxl::codestream_internal::
                                         VarDctEncodingProfile& profile,
                                     const std::vector<uint8_t>& bytes,
                                     const gjxl::VarDctEncodingSummary& summary) {
    if (raw_results == nullptr) {
      return;
    }
    RawWorkflowSample raw_sample;
    raw_sample.sample_index = sample_index;
    raw_sample.backend = backend;
    raw_sample.phase_nanoseconds = WorkflowProfileValues(profile);
    raw_sample.encoded_bytes = bytes.size();
    raw_sample.entropy_model_bits = profile.codestream.entropy_model_bits;
    raw_sample.entropy_token_bits = profile.codestream.entropy_token_bits;
    raw_sample.dc_entropy_clusters =
      profile.codestream.dc_entropy_clusters;
    raw_sample.ac_entropy_clusters =
      profile.codestream.ac_entropy_clusters;
    raw_sample.dc_entropy_is_ans = profile.codestream.dc_entropy_is_ans;
    raw_sample.ac_entropy_is_ans = profile.codestream.ac_entropy_is_ans;
    raw_sample.coefficient_order_entropy_is_ans =
      profile.codestream.coefficient_order_entropy_is_ans;
    raw_sample.natural_candidate_bytes =
      profile.codestream.natural_candidate_bytes;
    raw_sample.custom_order_candidate_bytes =
      profile.codestream.custom_order_candidate_bytes;
    raw_sample.selected_coefficient_order_mask =
      profile.codestream.selected_coefficient_order_mask;
    raw_sample.block_context_candidate_count =
      profile.codestream.block_context_candidate_count;
    raw_sample.compact_block_context_candidate_bytes =
      profile.codestream.compact_block_context_candidate_bytes;
    raw_sample.selected_block_context_candidate_index =
      profile.codestream.selected_block_context_candidate_index;
    raw_sample.selected_block_context_count =
      profile.codestream.selected_block_context_count;
    raw_sample.selected_block_context_qf_threshold_count =
      profile.codestream.selected_block_context_qf_threshold_count;
    raw_sample.has_final_score =
      summary.final_butteraugli_score_evaluated &&
      !summary.score_history.empty();
    if (raw_sample.has_final_score) {
      raw_sample.final_score = summary.score_history.back();
    }
    raw_workload.samples.push_back(raw_sample);
  };
  double sink = 0.0;
  for (size_t sample = 0; sample < samples; ++sample) {
    gjxl::codestream_internal::VarDctEncodingProfile cpu_profile;
    gjxl::codestream_internal::VarDctEncodingProfile gpu_profile;
    if (metal_only) {
      RequireStatus(
          "Metal public-workflow sample",
          encode(gjxl::VarDctBackendPreference::kMetal, gpu_aq_mode,
                 &gpu_bytes, &gpu_summary, &gpu_profile));
    } else if ((sample & 1u) == 0) {
      RequireStatus(
          "CPU public-workflow sample",
          encode(gjxl::VarDctBackendPreference::kCpu,
                 gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
                 &cpu_bytes, &cpu_summary, &cpu_profile));
      RequireStatus(
          "Metal public-workflow sample",
          encode(gjxl::VarDctBackendPreference::kMetal, gpu_aq_mode,
                 &gpu_bytes, &gpu_summary, &gpu_profile));
    } else {
      RequireStatus(
          "Metal public-workflow sample",
          encode(gjxl::VarDctBackendPreference::kMetal, gpu_aq_mode,
                 &gpu_bytes, &gpu_summary, &gpu_profile));
      RequireStatus(
          "CPU public-workflow sample",
          encode(gjxl::VarDctBackendPreference::kCpu,
                 gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
                 &cpu_bytes, &cpu_summary, &cpu_profile));
    }
    AppendWorkflowProfile(gpu_profile, &gpu_samples);
    append_raw_sample(sample, "metal", gpu_profile, gpu_bytes, gpu_summary);
    if (metal_only) {
      sink += static_cast<double>(gpu_bytes.size()) +
              (!gpu_summary.final_butteraugli_score_evaluated ||
                   gpu_summary.score_history.empty()
                 ? 0.0
                 : gpu_summary.score_history.back());
    } else {
      AppendWorkflowProfile(cpu_profile, &cpu_samples);
      append_raw_sample(sample, "cpu", cpu_profile, cpu_bytes, cpu_summary);
      paired_speedups.push_back(
          static_cast<double>(cpu_profile.total_nanoseconds) /
          static_cast<double>(gpu_profile.total_nanoseconds));
      sink += static_cast<double>(cpu_bytes.size() + gpu_bytes.size()) +
              cpu_summary.score_history.back() +
              (!gpu_summary.final_butteraugli_score_evaluated ||
                   gpu_summary.score_history.empty()
                 ? 0.0
                 : gpu_summary.score_history.back());
    }
  }

  std::cout << "workload " << spec.name << " source="
            << original.extent.width << 'x' << original.extent.height
            << " distance=" << butteraugli_target
            << " gpu_aq=" << GpuAqModeName(gpu_aq_mode)
            << " density="
            << (density_mode == gjxl::VarDctDensityMode::kHighDensity
                  ? "high"
                  : "default")
            << " scope="
            << (metal_only ? "metal-public-workflow" : "public-workflow")
            << " codestream="
            << (validation == ValidationMode::kCpuMetal
                    ? (codestreams_equal ? "exact" : "different")
                    : "not-compared");
  if (validation == ValidationMode::kCpuMetal) {
    std::cout << " cpu_bytes=" << cpu_bytes.size();
  }
  std::cout
            << " gpu_bytes=" << gpu_bytes.size() << '\n';
  if (!metal_only) {
    PrintWorkflowProfile("cpu", cpu_samples);
  }
  PrintWorkflowProfile("metal", gpu_samples);
  if (!metal_only) {
    PrintRatioStats("paired_speedup_x", paired_speedups);
  }
  std::cout << "  sink=" << sink << '\n';
  if (raw_results != nullptr) {
    raw_results->push_back(std::move(raw_workload));
  }
  *global_sink += sink;
}

void RunGpuProfileWorkflowWorkload(
    const WorkloadSpec& spec, size_t warmups, size_t samples,
    float butteraugli_target,
    gjxl::GpuAdaptiveQuantizationMode gpu_aq_mode,
    gjxl::gpu_profile_internal::GpuProfilingMode profiling_mode,
    bool collect_final_butteraugli_score,
    std::string_view input_path, gjxl::GpuBackend& gpu,
    std::vector<RawGpuProfileWorkload>* results, double* global_sink) {
  ImageStorage original = !input_path.empty()
      ? LoadBenchmarkImage(input_path)
      : (spec.flower ? LoadFlower() : ImageStorage(spec.source_extent));
  if (!spec.flower && input_path.empty()) {
    if (spec.workflow_gradient) {
      FillWorkflowGradient(&original);
    } else {
      FillSynthetic(&original);
    }
  }

  const gjxl::VarDctEncodingOptions encoding_options{
    .butteraugli_target = butteraugli_target,
    .backend = gjxl::VarDctBackendPreference::kMetal,
    .metal_aq_mode = gpu_aq_mode,
    .collect_final_butteraugli_score = collect_final_butteraugli_score,
  };
  std::vector<uint8_t> expected;
  gjxl::VarDctEncodingSummary expected_summary;
  gjxl::codestream_internal::VarDctEncodingProfile host_profile;
  RequireStatus(
    "Unprofiled Metal public-workflow validation",
    gjxl::codestream_internal::
      EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
        original.ConstView(), encoding_options, &gpu, true, &expected,
        &expected_summary, &host_profile));
  if (expected.size() < 2 || expected[0] != 0xff || expected[1] != 0x0a ||
      expected_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kMetal) {
    throw std::runtime_error(
      "GPU-profile public-workflow validation failed");
  }

  const auto encode_profiled = [&](
      std::vector<uint8_t>* bytes,
      gjxl::VarDctEncodingSummary* summary,
      gjxl::gpu_profile_internal::GpuExecutionProfile* profile) {
    gjxl::codestream_internal::VarDctEncodingProfile ignored_host_profile;
    return gjxl::codestream_internal::
      EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
        original.ConstView(), encoding_options, &gpu, true, profiling_mode,
        bytes, summary, &ignored_host_profile, profile);
  };

  std::vector<uint8_t> bytes;
  gjxl::VarDctEncodingSummary summary;
  for (size_t warmup = 0; warmup < warmups; ++warmup) {
    gjxl::gpu_profile_internal::GpuExecutionProfile profile;
    RequireStatus(
      "GPU-profile public-workflow warmup",
      encode_profiled(&bytes, &summary, &profile));
  }

  RawGpuProfileWorkload workload{
    .workload = std::string(spec.name),
    .source_extent = original.extent,
  };
  workload.samples.reserve(samples);
  double sink = 0.0;
  for (size_t sample = 0; sample < samples; ++sample) {
    gjxl::gpu_profile_internal::GpuExecutionProfile profile;
    RequireStatus(
      "GPU-profile public-workflow sample",
      encode_profiled(&bytes, &summary, &profile));
    if (bytes != expected ||
        summary.execution_backend != gjxl::VarDctExecutionBackend::kMetal ||
        profile.mode != profiling_mode || profile.submissions.empty()) {
      throw std::runtime_error(
        "GPU-profile public-workflow changed the encoded result");
    }
    workload.samples.push_back({sample, std::move(profile)});
    sink += static_cast<double>(bytes.size()) +
      (!summary.final_butteraugli_score_evaluated ||
           summary.score_history.empty()
         ? 0.0
         : summary.score_history.back());
  }
  std::cout << "workload " << spec.name << " source="
            << original.extent.width << 'x' << original.extent.height
            << " distance=" << butteraugli_target
            << " gpu_aq=" << GpuAqModeName(gpu_aq_mode)
            << " scope=gpu-profile samples=" << samples << '\n';
  results->push_back(std::move(workload));
  *global_sink += sink;
}

void RunGpuCompleteAqOnlyWorkload(
    const WorkloadSpec& spec, size_t warmups, size_t samples,
    float butteraugli_target,
    gjxl::GpuAdaptiveQuantizationMode gpu_aq_mode, gjxl::GpuBackend& gpu,
    double* global_sink) {
  ImageStorage original(spec.source_extent);
  FillSynthetic(&original);
  const gjxl::Extent2D coding_extent = PaddedExtent(original.extent);
  ImageStorage padded_linear(coding_extent);
  ImageStorage opsin(coding_extent);
  ImageStorage preprocessed(coding_extent);
  EdgePad(original, &padded_linear);
  RequireStatus(
      "linear-to-opsin setup",
      gjxl::LinearRgbToOpsin(padded_linear.ConstView(), 255.0f, opsin.View()));
  RequireStatus("Gaborish setup",
                gjxl::ApplyGaborishInverse(
                    opsin.ConstView(), {1.0f, 1.0f, 1.0f},
                    preprocessed.View()));

  StageOutput stage(original.extent, coding_extent);
  RequireStatus(
      "initial quantization setup",
      gjxl::ComputeInitialQuantField(
          opsin.ConstView(), {.butteraugli_target = butteraugli_target},
          stage.InitialOutput()));
  gjxl::ColorCorrelationMap initial_color_correlation;
  RequireStatus(
      "initial color-correlation setup",
      gjxl::ComputeInitialColorCorrelationMap(
          preprocessed.ConstView(), &initial_color_correlation));
  RequireStatus(
      "AC-strategy setup",
      gjxl::FindAcStrategyGrid(
          preprocessed.ConstView(),
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {stage.pixel_mask.data(), coding_extent, coding_extent.width},
          initial_color_correlation,
          {.butteraugli_target = butteraugli_target}, &stage.strategies));
  std::vector<uint8_t> sharpness(stage.block_extent.width *
                                 stage.block_extent.height);
  RequireStatus(
      "EPF-sharpness setup",
      gjxl::FillDefaultEpfSharpness(
          {sharpness.data(), stage.block_extent, stage.block_extent.width}));

  gjxl::AdaptiveQuantizationOptions options;
  options.butteraugli_target = butteraugli_target;
  options.iterations = 2;
  StageOutput gpu_stage(original.extent, coding_extent);
  const auto run_complete_aq = [&] {
    return gjxl::RunGpuAdaptiveQuantization(
        gpu, original.ConstView(), preprocessed.ConstView(), stage.strategies,
        {stage.initial_quant.data(), stage.block_extent,
         stage.block_extent.width},
        {sharpness.data(), stage.block_extent, stage.block_extent.width},
        options, gpu_aq_mode, gpu_stage.AdaptiveOutput());
  };

  double sink = 0.0;
  for (size_t warmup = 0; warmup < warmups; ++warmup) {
    RequireStatus("gpu_iterative_aq_two_updates_e2e", run_complete_aq());
    sink += gpu_stage.scores.back();
  }

  std::vector<double> complete_aq_samples;
  complete_aq_samples.reserve(samples);
  for (size_t sample = 0; sample < samples; ++sample) {
    const auto begin = Clock::now();
    RequireStatus("gpu_iterative_aq_two_updates_e2e", run_complete_aq());
    const auto end = Clock::now();
    complete_aq_samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
    sink += gpu_stage.scores.back();
  }
  if (!gpu_stage.frame.valid() ||
      gpu_stage.scores.size() != options.iterations + 1) {
    throw std::runtime_error(
        "GPU-only complete AQ produced an invalid final result");
  }

  std::cout << "workload " << spec.name << " source="
            << original.extent.width << 'x' << original.extent.height
            << " coding=" << coding_extent.width << 'x'
            << coding_extent.height << " distance=" << butteraugli_target
            << " gpu_aq=" << GpuAqModeName(gpu_aq_mode)
            << " phases=gpu_complete_aq_only\n";
  PrintStats("gpu_iterative_aq_two_updates_e2e", complete_aq_samples);
  std::cout << "  sink=" << sink << '\n';
  *global_sink += sink;
}

void RunWorkload(const WorkloadSpec& spec, size_t warmups, size_t samples,
                 float butteraugli_target,
                 gjxl::GpuAdaptiveQuantizationMode gpu_aq_mode,
                 std::string_view input_path,
                 gjxl::GpuBackend& gpu,
                 const gjxl::MetalBackendOptions& backend_options,
                 double* global_sink) {

  ImageStorage original = !input_path.empty()
      ? LoadBenchmarkImage(input_path)
      : (spec.flower ? LoadFlower() : ImageStorage(spec.source_extent));
  if (!spec.flower && input_path.empty()) {
    if (spec.workflow_gradient) {
      FillWorkflowGradient(&original);
    } else {
      FillSynthetic(&original);
    }
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
                                     {.butteraugli_target = butteraugli_target},
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
          initial_color_correlation,
          {.butteraugli_target = butteraugli_target},
          &stage.strategies));
  std::vector<uint8_t> sharpness(stage.block_extent.width *
                                 stage.block_extent.height);
  RequireStatus(
      "EPF-sharpness setup",
      gjxl::FillDefaultEpfSharpness(
          {sharpness.data(), stage.block_extent, stage.block_extent.width}));

  gjxl::AdaptiveQuantizationOptions one_evaluation_options;
  one_evaluation_options.butteraugli_target = butteraugli_target;
  one_evaluation_options.iterations = 0;
  gjxl::AdaptiveQuantizationOptions two_update_options = one_evaluation_options;
  two_update_options.iterations = 2;
  StageOutput pipeline_stage(original.extent, coding_extent);
  StageOutput gpu_stage(original.extent, coding_extent);
  StageOutput resident_stage(original.extent, coding_extent);
  StageOutput gpu_pipeline_stage(original.extent, coding_extent);
  gjxl::AcStrategyGrid gpu_strategies;
  gjxl::AcStrategyGpuSearchStats gpu_search_stats;
  std::vector<uint8_t> workflow_bytes;
  gjxl::VarDctEncodingSummary workflow_summary;
  gjxl::CpuQuantizationPipelineOptions pipeline_options;
  pipeline_options.butteraugli_target = butteraugli_target;
  pipeline_options.adaptive_quantization.iterations = 2;

  RequireStatus("CPU pipeline validation",
                gjxl::RunCpuQuantizationPipeline(
                    original.ConstView(), opsin.ConstView(), pipeline_options,
                    pipeline_stage.PipelineOutput()));
  RequireStatus(
      "CPU resident validation",
      gjxl::FindBestQuantization(
          original.ConstView(), preprocessed.ConstView(), stage.strategies,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          one_evaluation_options, stage.AdaptiveOutput()));
  RequireStatus("GPU pipeline validation",
                gjxl::RunGpuQuantizationPipeline(
                    gpu, original.ConstView(), opsin.ConstView(),
                    pipeline_options, gpu_aq_mode,
                    gpu_pipeline_stage.PipelineOutput()));
  RequireStatus(
      "GPU resident validation",
      gjxl::RunGpuAdaptiveQuantization(
          gpu, original.ConstView(), preprocessed.ConstView(),
          stage.strategies,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          one_evaluation_options, gpu_aq_mode, gpu_stage.AdaptiveOutput()));

  std::vector<uint8_t> cpu_validation_bytes;
  std::vector<uint8_t> gpu_validation_bytes;
  gjxl::VarDctEncodingSummary cpu_validation_summary;
  gjxl::VarDctEncodingSummary gpu_validation_summary;
  RequireStatus(
      "CPU workflow validation",
      gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = butteraugli_target,
               .backend = gjxl::VarDctBackendPreference::kCpu},
              nullptr, false, &cpu_validation_bytes,
              &cpu_validation_summary));
  RequireStatus(
      "GPU workflow validation",
      gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = butteraugli_target,
               .backend = gjxl::VarDctBackendPreference::kMetal,
               .metal_aq_mode = gpu_aq_mode},
              &gpu, true, &gpu_validation_bytes,
              &gpu_validation_summary));
  const auto maximum_vector_error = [](const std::vector<float>& left,
                                       const std::vector<float>& right) {
    if (left.size() != right.size()) {
      return std::numeric_limits<double>::infinity();
    }
    double maximum = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
      maximum = std::max(
          maximum, std::abs(static_cast<double>(left[index]) - right[index]));
    }
    return maximum;
  };
  const auto maximum_score_error = [](const std::vector<double>& left,
                                      const std::vector<double>& right) {
    if (left.size() != right.size()) {
      return std::numeric_limits<double>::infinity();
    }
    double maximum = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
      maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
  };
  const auto maximum_image_error = [](const ImageStorage& left,
                                      const ImageStorage& right) {
    if (left.extent != right.extent) {
      return std::numeric_limits<double>::infinity();
    }
    double maximum = 0.0;
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t index = 0; index < left.plane[channel].size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(left.plane[channel][index]) -
                     right.plane[channel][index]));
      }
    }
    return maximum;
  };

  const double pipeline_quant_error = maximum_vector_error(
      pipeline_stage.final_quant, gpu_pipeline_stage.final_quant);
  const double pipeline_block_error = maximum_vector_error(
      pipeline_stage.block_distance, gpu_pipeline_stage.block_distance);
  const double pipeline_score_error = maximum_score_error(
      pipeline_stage.scores, gpu_pipeline_stage.scores);
  const double pipeline_final_score_error =
      pipeline_stage.scores.empty() || gpu_pipeline_stage.scores.empty()
        ? std::numeric_limits<double>::infinity()
        : std::abs(
            pipeline_stage.scores.back() - gpu_pipeline_stage.scores.back());
  const double pipeline_reconstruction_error = maximum_image_error(
      pipeline_stage.reconstructed, gpu_pipeline_stage.reconstructed);
  const double one_block_error = maximum_vector_error(
      stage.block_distance, gpu_stage.block_distance);
  const double one_score_error = maximum_score_error(
      stage.scores, gpu_stage.scores);
  const double one_reconstruction_error = maximum_image_error(
      stage.reconstructed, gpu_stage.reconstructed);
  const std::string pipeline_frame_difference =
      FrameDifference(pipeline_stage.frame, gpu_pipeline_stage.frame);
  const FrameCoefficientError pipeline_coefficient_error =
      CompareFrameCoefficients(pipeline_stage.frame, gpu_pipeline_stage.frame);
  const std::string one_frame_difference =
      FrameDifference(stage.frame, gpu_stage.frame);
  constexpr double kRolloutTolerance = 2.0e-3;
  const bool rollout_matches =
      cpu_validation_bytes == gpu_validation_bytes &&
      cpu_validation_summary.strategy_counts ==
          gpu_validation_summary.strategy_counts &&
      pipeline_frame_difference == "none" &&
      one_frame_difference == "none" &&
      pipeline_quant_error <= kRolloutTolerance &&
      pipeline_block_error <= kRolloutTolerance &&
      pipeline_score_error <= kRolloutTolerance &&
      pipeline_reconstruction_error <= kRolloutTolerance &&
      one_block_error <= kRolloutTolerance &&
      one_score_error <= kRolloutTolerance &&
      one_reconstruction_error <= kRolloutTolerance;
  if (!rollout_matches &&
      gpu_aq_mode ==
          gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients) {
    throw std::runtime_error(
        "CPU/Metal rollout gate failed: bytes=" +
        std::string(cpu_validation_bytes == gpu_validation_bytes
                        ? "equal"
                        : "different") +
        " frame=" + pipeline_frame_difference +
        " pipeline_quant=" + std::to_string(pipeline_quant_error) +
        " pipeline_block=" + std::to_string(pipeline_block_error) +
        " pipeline_score=" + std::to_string(pipeline_score_error) +
        " pipeline_rgb=" +
        std::to_string(pipeline_reconstruction_error) +
        " one_frame=" + one_frame_difference +
        " one_block=" + std::to_string(one_block_error) +
        " one_score=" + std::to_string(one_score_error) +
        " one_rgb=" + std::to_string(one_reconstruction_error));
  }

  const size_t block_count =
      stage.block_extent.width * stage.block_extent.height;
  std::vector<float> resident_quant_field = stage.initial_quant;
  RequireStatus(
      "Resident quant-field adjustment",
      gjxl::AdjustQuantField(
          stage.strategies, butteraugli_target,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {resident_quant_field.data(), stage.block_extent,
           stage.block_extent.width}));
  float quant_dc = 0.0f;
  RequireStatus("Resident DC quantization",
                gjxl::ComputeInitialQuantDc(butteraugli_target, &quant_dc));
  std::vector<int32_t> resident_raw_quant(block_count);
  std::vector<float> resident_inverse_sigma(block_count);
  gjxl::Quantizer resident_quantizer;
  RequireStatus(
      "Resident quantizer",
      gjxl::CreateQuantizerFromField(
          quant_dc,
          {resident_quant_field.data(), stage.block_extent,
           stage.block_extent.width},
          {resident_raw_quant.data(), stage.block_extent,
           stage.block_extent.width},
          &resident_quantizer));
  gjxl::ColorCorrelationMap resident_color;
  RequireStatus(
      "Resident final color correlation",
      gjxl::ComputeFinalColorCorrelationMap(
          preprocessed.ConstView(), stage.strategies,
          {resident_raw_quant.data(), stage.block_extent,
           stage.block_extent.width},
          resident_quantizer, true, &resident_color));
  RequireStatus(
      "Resident EPF inverse sigma",
      gjxl::ComputeEpfInverseSigma(
          stage.strategies,
          {resident_raw_quant.data(), stage.block_extent,
           stage.block_extent.width},
          resident_quantizer,
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          two_update_options.profile.epf_sigma,
          {resident_inverse_sigma.data(), stage.block_extent,
           stage.block_extent.width}));
  gjxl::FrameGeometry resident_geometry;
  RequireStatus("Resident frame geometry",
                gjxl::FrameGeometry::Create(original.extent,
                                            &resident_geometry));
  gjxl::VarDctEncoderFrame resident_exact_frame;
  RequireStatus(
      "Resident exact coefficient coding",
      gjxl::ComputeQuantizedCoefficients(
          preprocessed.ConstView(),
          {
              .geometry = resident_geometry,
              .strategies = &stage.strategies,
              .raw_quant_field = {
                  resident_raw_quant.data(), stage.block_extent,
                  stage.block_extent.width},
              .quantizer = &resident_quantizer,
              .color_correlation = &resident_color,
              .epf_sharpness = {
                  sharpness.data(), stage.block_extent,
                  stage.block_extent.width},
          },
          two_update_options.profile, &resident_exact_frame));
  std::vector<float> resident_exact_inverse_sigma(block_count);
  RequireStatus(
      "Resident exact EPF inverse sigma",
      gjxl::ComputeEpfInverseSigma(
          resident_exact_frame.strategies(),
          resident_exact_frame.raw_quant_field(),
          resident_exact_frame.quantizer(),
          resident_exact_frame.epf_sharpness(),
          two_update_options.profile.epf_sigma,
          {resident_exact_inverse_sigma.data(), stage.block_extent,
           stage.block_extent.width}));
  ImageStorage resident_reconstructed_opsin(coding_extent);
  RequireStatus(
      "Resident exact reconstruction",
      gjxl::ReconstructQuantizedCoefficients(
          resident_exact_frame, resident_reconstructed_opsin.View()));
  ImageStorage resident_cropped(original.extent);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < original.extent.height; ++y) {
      std::copy_n(
          resident_reconstructed_opsin.plane[channel].data() +
              y * coding_extent.width,
          original.extent.width,
          resident_cropped.plane[channel].data() + y * original.extent.width);
    }
  }
  ImageStorage resident_filtered(original.extent);
  RequireStatus(
      "Resident exact loop filters",
      gjxl::ApplyLoopFilters(
          resident_cropped.ConstView(),
          {resident_inverse_sigma.data(), stage.block_extent,
           stage.block_extent.width},
          two_update_options.profile.loop_filter, resident_filtered.View()));
  ImageStorage resident_exact_linear(original.extent);
  RequireStatus(
      "Resident exact color conversion",
      gjxl::OpsinToLinearRgb(
          resident_filtered.ConstView(),
          two_update_options.profile.intensity_target,
          resident_exact_linear.View()));

  std::unique_ptr<gjxl::PreparedAqEvaluation> profiled_prepared;
  RequireStatus(
      "GPU memory preparation",
      gjxl::PrepareAqEvaluation(
          gpu,
          {
              .original_linear_rgb = original.ConstView(),
              .coding_opsin = preprocessed.ConstView(),
              .strategies = &stage.strategies,
              .epf_sharpness = {sharpness.data(), stage.block_extent,
                                stage.block_extent.width},
              .options = {two_update_options.profile,
                          two_update_options.butteraugli},
          },
          &profiled_prepared));
  const gjxl::AqEvaluationMemoryStats memory_stats =
      profiled_prepared->memory_stats();
  gjxl::AqEvaluationOutput::Final resident_final{
      .reconstructed_linear_rgb = resident_stage.reconstructed.View(),
      .frame = &resident_stage.frame,
  };
  StageOutput coefficient_prototype_stage(original.extent, coding_extent);
  gjxl::AqEvaluationOutput::Final coefficient_prototype_final{
      .reconstructed_linear_rgb =
          coefficient_prototype_stage.reconstructed.View(),
      .frame = &coefficient_prototype_stage.frame,
  };
  StageOutput fully_resident_stage(original.extent, coding_extent);
  gjxl::AqEvaluationOutput::Final fully_resident_final{
      .reconstructed_linear_rgb = fully_resident_stage.reconstructed.View(),
      .frame = &fully_resident_stage.frame,
  };
  double resident_score = 0.0;
  double coefficient_prototype_score = 0.0;
  double fully_resident_score = 0.0;
  gjxl::metal_internal::MetalAqEvaluationProfile gpu_profile;
  gjxl::metal_internal::MetalAqEvaluationProfile coefficient_gpu_profile;
  gjxl::metal_internal::MetalAqEvaluationProfile fully_resident_gpu_profile;

  RequireStatus(
      "Fully resident prototype validation",
      profiled_prepared->Evaluate(
          {
              .raw_quant_field = {
                  resident_raw_quant.data(), stage.block_extent,
                  stage.block_extent.width},
              .quantizer = resident_quantizer.params(),
              .y_to_x = resident_color.y_to_x_map(),
              .y_to_b = resident_color.y_to_b_map(),
              .epf_inverse_sigma = {
                  resident_inverse_sigma.data(), stage.block_extent,
                  stage.block_extent.width},
          },
          {
              .block_distance_map = {
                  fully_resident_stage.block_distance.data(),
                  stage.block_extent, stage.block_extent.width},
              .score = &fully_resident_score,
              .final = &fully_resident_final,
          }));

  RequireStatus(
      "Exact-coefficient prototype validation",
      profiled_prepared->Evaluate(
          {
              .raw_quant_field = {
                  resident_exact_frame.raw_quant_field().data,
                  stage.block_extent,
                  resident_exact_frame.raw_quant_field().stride},
              .quantizer = resident_exact_frame.quantizer().params(),
              .y_to_x =
                  resident_exact_frame.color_correlation().y_to_x_map(),
              .y_to_b =
                  resident_exact_frame.color_correlation().y_to_b_map(),
              .epf_inverse_sigma = {
                  resident_exact_inverse_sigma.data(), stage.block_extent,
                  stage.block_extent.width},
              .exact_coefficients = &resident_exact_frame,
          },
          {
              .block_distance_map = {
                  coefficient_prototype_stage.block_distance.data(),
                  stage.block_extent, stage.block_extent.width},
              .score = &coefficient_prototype_score,
              .final = &coefficient_prototype_final,
          }));
  const double coefficient_prototype_block_error = maximum_vector_error(
      stage.block_distance, coefficient_prototype_stage.block_distance);
  const double coefficient_prototype_score_error = std::abs(
      stage.scores.back() - coefficient_prototype_score);
  const double coefficient_prototype_rgb_error = maximum_image_error(
      stage.reconstructed, coefficient_prototype_stage.reconstructed);
  const std::string coefficient_prototype_frame_difference = FrameDifference(
      stage.frame, coefficient_prototype_stage.frame);
  const double fully_resident_block_error = maximum_vector_error(
      stage.block_distance, fully_resident_stage.block_distance);
  const double fully_resident_score_error = std::abs(
      stage.scores.back() - fully_resident_score);
  const double fully_resident_rgb_error = maximum_image_error(
      stage.reconstructed, fully_resident_stage.reconstructed);
  const std::string fully_resident_frame_difference = FrameDifference(
      stage.frame, fully_resident_stage.frame);
  const FrameCoefficientError fully_resident_coefficient_error =
      CompareFrameCoefficients(stage.frame, fully_resident_stage.frame);

  auto run_phase = [&](Phase phase, aqi::AdaptiveQuantizationProfile* profile) {
    switch (phase) {
    case Phase::kInitialQuantField:
      return gjxl::ComputeInitialQuantField(
          opsin.ConstView(), {.butteraugli_target = butteraugli_target},
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
          initial_color_correlation,
          {.butteraugli_target = butteraugli_target},
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
    case Phase::kGpuAcStrategySearch:
      return gjxl::FindAcStrategyGridGpu(
          gpu, preprocessed.ConstView(),
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {stage.pixel_mask.data(), coding_extent, coding_extent.width},
          initial_color_correlation,
          {.butteraugli_target = butteraugli_target}, &gpu_strategies,
          &gpu_search_stats);
    case Phase::kGpuPreparation: {
      std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
      return gjxl::PrepareAqEvaluation(
          gpu,
          {
              .original_linear_rgb = original.ConstView(),
              .coding_opsin = preprocessed.ConstView(),
              .strategies = &stage.strategies,
              .epf_sharpness = {sharpness.data(), stage.block_extent,
                                stage.block_extent.width},
              .options = {two_update_options.profile,
                          two_update_options.butteraugli},
          },
          &prepared);
    }
    case Phase::kGpuFullyResidentEvaluation:
      return gjxl::metal_internal::EvaluateMetalAqProfiled(
          *profiled_prepared,
          {
              .raw_quant_field = {
                  resident_raw_quant.data(), stage.block_extent,
                  stage.block_extent.width},
              .quantizer = resident_quantizer.params(),
              .y_to_x = resident_color.y_to_x_map(),
              .y_to_b = resident_color.y_to_b_map(),
              .epf_inverse_sigma = {
                  resident_inverse_sigma.data(), stage.block_extent,
                  stage.block_extent.width},
          },
          {
              .block_distance_map = {
                  fully_resident_stage.block_distance.data(),
                  stage.block_extent, stage.block_extent.width},
              .score = &fully_resident_score,
              .final = &fully_resident_final,
          },
          &fully_resident_gpu_profile);
    case Phase::kGpuResidentEvaluation:
      return gjxl::metal_internal::EvaluateMetalAqProfiled(
          *profiled_prepared,
          {
              .raw_quant_field = {
                  resident_exact_frame.raw_quant_field().data,
                  stage.block_extent,
                  resident_exact_frame.raw_quant_field().stride},
              .quantizer = resident_exact_frame.quantizer().params(),
              .y_to_x =
                  resident_exact_frame.color_correlation().y_to_x_map(),
              .y_to_b =
                  resident_exact_frame.color_correlation().y_to_b_map(),
              .epf_inverse_sigma = {
                  resident_exact_inverse_sigma.data(), stage.block_extent,
                  stage.block_extent.width},
              .exact_coefficients = &resident_exact_frame,
              .exact_reconstructed_linear_rgb =
                  resident_exact_linear.ConstView(),
          },
          {
              .block_distance_map = {
                  resident_stage.block_distance.data(), stage.block_extent,
                  stage.block_extent.width},
              .score = &resident_score,
              .final = &resident_final,
          },
          &gpu_profile);
    case Phase::kGpuExactCoefficientEvaluation:
      return gjxl::metal_internal::EvaluateMetalAqProfiled(
          *profiled_prepared,
          {
              .raw_quant_field = {
                  resident_exact_frame.raw_quant_field().data,
                  stage.block_extent,
                  resident_exact_frame.raw_quant_field().stride},
              .quantizer = resident_exact_frame.quantizer().params(),
              .y_to_x =
                  resident_exact_frame.color_correlation().y_to_x_map(),
              .y_to_b =
                  resident_exact_frame.color_correlation().y_to_b_map(),
              .epf_inverse_sigma = {
                  resident_exact_inverse_sigma.data(), stage.block_extent,
                  stage.block_extent.width},
              .exact_coefficients = &resident_exact_frame,
          },
          {
              .block_distance_map = {
                  coefficient_prototype_stage.block_distance.data(),
                  stage.block_extent, stage.block_extent.width},
              .score = &coefficient_prototype_score,
              .final = &coefficient_prototype_final,
          },
          &coefficient_gpu_profile);
    case Phase::kGpuAqOneEvaluation:
      return gjxl::RunGpuAdaptiveQuantization(
          gpu, original.ConstView(), preprocessed.ConstView(), stage.strategies,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          one_evaluation_options, gpu_aq_mode, gpu_stage.AdaptiveOutput());
    case Phase::kGpuAqTwoUpdates:
      return gjxl::RunGpuAdaptiveQuantization(
          gpu, original.ConstView(), preprocessed.ConstView(), stage.strategies,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          two_update_options, gpu_aq_mode, gpu_stage.AdaptiveOutput());
    case Phase::kGpuCompletePipeline:
      return gjxl::RunGpuQuantizationPipeline(
          gpu, original.ConstView(), opsin.ConstView(), pipeline_options,
          gpu_aq_mode, gpu_pipeline_stage.PipelineOutput(),
          &gpu_search_stats);
    case Phase::kCpuPublicWorkflow:
      return gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = butteraugli_target,
               .backend = gjxl::VarDctBackendPreference::kCpu},
              nullptr, false, &workflow_bytes, &workflow_summary);
    case Phase::kGpuWarmPublicWorkflow:
      return gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = butteraugli_target,
               .backend = gjxl::VarDctBackendPreference::kMetal,
               .metal_aq_mode = gpu_aq_mode},
              &gpu, true, &workflow_bytes, &workflow_summary);
    case Phase::kGpuColdPublicWorkflow: {
      std::unique_ptr<gjxl::GpuBackend> cold_gpu;
      gjxl::Status status = gjxl::CreateEmbeddedMetalBackend(
          backend_options, &cold_gpu);
      if (!status.ok()) {
        return status;
      }
      return gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = butteraugli_target,
               .backend = gjxl::VarDctBackendPreference::kMetal,
               .metal_aq_mode = gpu_aq_mode},
              cold_gpu.get(), true, &workflow_bytes, &workflow_summary);
    }
    case Phase::kCount:
      break;
    }
    return gjxl::Status::Internal("Unknown encoding benchmark phase");
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
  std::array<std::vector<double>, 7> gpu_profile_samples;
  std::array<std::vector<double>, 7> coefficient_gpu_profile_samples;
  std::array<std::vector<double>, 7> fully_resident_gpu_profile_samples;
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
      } else if (phase == Phase::kGpuFullyResidentEvaluation) {
        const std::array<uint64_t, 7> values = {
            fully_resident_gpu_profile.input_upload_nanoseconds,
            fully_resident_gpu_profile.submission_nanoseconds,
            fully_resident_gpu_profile.completion_wait_nanoseconds,
            fully_resident_gpu_profile.command_buffer_gpu_nanoseconds,
            fully_resident_gpu_profile.bounded_readback_nanoseconds,
            fully_resident_gpu_profile.final_readback_nanoseconds,
            fully_resident_gpu_profile.output_commit_nanoseconds,
        };
        for (size_t index = 0; index < values.size(); ++index) {
          fully_resident_gpu_profile_samples[index].push_back(
              NanosecondsToMilliseconds(values[index]));
        }
      } else if (phase == Phase::kGpuResidentEvaluation) {
        const std::array<uint64_t, 7> values = {
            gpu_profile.input_upload_nanoseconds,
            gpu_profile.submission_nanoseconds,
            gpu_profile.completion_wait_nanoseconds,
            gpu_profile.command_buffer_gpu_nanoseconds,
            gpu_profile.bounded_readback_nanoseconds,
            gpu_profile.final_readback_nanoseconds,
            gpu_profile.output_commit_nanoseconds,
        };
        for (size_t index = 0; index < values.size(); ++index) {
          gpu_profile_samples[index].push_back(
              NanosecondsToMilliseconds(values[index]));
        }
      } else if (phase == Phase::kGpuExactCoefficientEvaluation) {
        const std::array<uint64_t, 7> values = {
            coefficient_gpu_profile.input_upload_nanoseconds,
            coefficient_gpu_profile.submission_nanoseconds,
            coefficient_gpu_profile.completion_wait_nanoseconds,
            coefficient_gpu_profile.command_buffer_gpu_nanoseconds,
            coefficient_gpu_profile.bounded_readback_nanoseconds,
            coefficient_gpu_profile.final_readback_nanoseconds,
            coefficient_gpu_profile.output_commit_nanoseconds,
        };
        for (size_t index = 0; index < values.size(); ++index) {
          coefficient_gpu_profile_samples[index].push_back(
              NanosecondsToMilliseconds(values[index]));
        }
      }
      sink += stage.scores.empty() ? stage.initial_quant.front()
                                   : stage.scores.back();
    }
  }

  std::cout << "workload " << spec.name << " source=" << original.extent.width
            << 'x' << original.extent.height
            << " coding=" << coding_extent.width << 'x' << coding_extent.height
            << " distance=" << butteraugli_target
            << " gpu_aq=" << GpuAqModeName(gpu_aq_mode)
            << '\n';
  std::cout << std::scientific << std::setprecision(6)
            << "  rollout_errors pipeline_quant=" << pipeline_quant_error
            << " pipeline_block=" << pipeline_block_error
            << " pipeline_score=" << pipeline_score_error
            << " pipeline_final_score=" << pipeline_final_score_error
            << " pipeline_rgb=" << pipeline_reconstruction_error
            << " one_block=" << one_block_error
            << " one_score=" << one_score_error
            << " one_rgb=" << one_reconstruction_error
            << " frame=" << pipeline_frame_difference
            << " raw_count=" << pipeline_coefficient_error.raw_quant_count
            << " raw_max_delta="
            << pipeline_coefficient_error.raw_quant_max_delta
            << " qdc_count="
            << pipeline_coefficient_error.quantized_dc_count
            << " qdc_max_delta="
            << pipeline_coefficient_error.quantized_dc_max_delta
            << " ac_count="
            << pipeline_coefficient_error.quantized_ac_count
            << " ac_max_delta="
            << pipeline_coefficient_error.quantized_ac_max_delta
            << " codestream="
            << (cpu_validation_bytes == gpu_validation_bytes ? "exact"
                                                             : "different")
            << " cpu_bytes=" << cpu_validation_bytes.size()
            << " gpu_bytes=" << gpu_validation_bytes.size()
            << " parity=" << (rollout_matches ? "pass" : "different")
            << '\n'
            << "  fully_resident_errors block="
            << fully_resident_block_error
            << " score=" << fully_resident_score_error
            << " rgb=" << fully_resident_rgb_error
            << " frame=" << fully_resident_frame_difference
            << " raw_count="
            << fully_resident_coefficient_error.raw_quant_count
            << " raw_max_delta="
            << fully_resident_coefficient_error.raw_quant_max_delta
            << " qdc_count="
            << fully_resident_coefficient_error.quantized_dc_count
            << " qdc_max_delta="
            << fully_resident_coefficient_error.quantized_dc_max_delta
            << " ac_count="
            << fully_resident_coefficient_error.quantized_ac_count
            << " ac_max_delta="
            << fully_resident_coefficient_error.quantized_ac_max_delta
            << " dc_max_error="
            << fully_resident_coefficient_error.reconstructed_dc_max_error
            << '\n'
            << "  exact_coefficient_errors block="
            << coefficient_prototype_block_error
            << " score=" << coefficient_prototype_score_error
            << " rgb=" << coefficient_prototype_rgb_error
            << " frame=" << coefficient_prototype_frame_difference << '\n'
            << std::fixed << std::setprecision(3);
  std::cout << "  memory_bytes persistent=" << memory_stats.persistent_bytes
            << " staging=" << memory_stats.staging_bytes
            << " peak_scratch=" << memory_stats.peak_scratch_bytes << '\n';
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
  constexpr std::array<std::string_view, 7> kGpuProfileNames = {
      "input_upload",       "submission",       "completion_wait",
      "command_buffer_gpu", "bounded_readback", "final_readback",
      "output_commit",
  };
  std::cout << "  gpu_fully_resident_profile\n";
  for (size_t index = 0; index < kGpuProfileNames.size(); ++index) {
    PrintStats(kGpuProfileNames[index],
               fully_resident_gpu_profile_samples[index]);
  }
  std::cout << "  gpu_resident_perceptual_profile\n";
  for (size_t index = 0; index < kGpuProfileNames.size(); ++index) {
    PrintStats(kGpuProfileNames[index], gpu_profile_samples[index]);
  }
  std::cout << "  gpu_exact_coefficient_reconstruction_profile\n";
  for (size_t index = 0; index < kGpuProfileNames.size(); ++index) {
    PrintStats(kGpuProfileNames[index],
               coefficient_gpu_profile_samples[index]);
  }
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
    if (options.input_path.empty() && options.workload != "all" &&
        !IsKnownWorkload(options.workload)) {
      throw std::runtime_error("Unknown quantization workload: " +
                               options.workload);
    }
    const gjxl::MetalBackendOptions backend_options =
        BackendOptions(options.implementation);
    std::unique_ptr<gjxl::GpuBackend> gpu;
    if (options.scope != BenchmarkScope::kCoefficientCoding) {
      if (options.metallib_path.empty()) {
        RequireStatus("Create embedded benchmark Metal backend",
                      gjxl::CreateEmbeddedMetalBackend(backend_options, &gpu));
      } else {
        RequireStatus("Create external benchmark Metal backend",
                      gjxl::CreateMetalBackend(options.metallib_path,
                                               backend_options, &gpu));
      }
    }
    std::cout << std::fixed << std::setprecision(3)
              << "CPU/Metal encoding benchmark: backend="
              << (gpu == nullptr ? "cpu-only" : gpu->name())
              << " implementation=" << options.implementation
              << " scope=" << BenchmarkScopeName(options.scope)
              << " gpu_aq=" << GpuAqModeName(options.gpu_aq_mode)
              << " final_score="
              << (options.collect_final_butteraugli_score
                    ? "collect"
                    : "skip")
              << " density="
              << (options.density_mode ==
                        gjxl::VarDctDensityMode::kHighDensity
                    ? "high"
                    : "default")
              << " distance=" << options.butteraugli_target
              << " warmups=" << options.warmups
              << " samples=" << options.samples;
    if (options.scope == BenchmarkScope::kFull) {
      std::cout << " rotated_phases=" << kPhaseCount;
    }
    std::cout << '\n';
    double sink = 0.0;
    std::vector<RawWorkflowWorkload> raw_results;
    std::vector<RawGpuProfileWorkload> gpu_profile_results;
    std::vector<RawWorkflowWorkload>* raw_results_pointer =
        options.raw_samples_path.empty() ? nullptr : &raw_results;
    if (!options.input_path.empty()) {
      if (options.scope == BenchmarkScope::kCoefficientCoding) {
        RunCoefficientCodingOnlyWorkload(
            {"external_input", {}, false}, options.warmups, options.samples,
            options.butteraugli_target, options.input_path, &sink);
      } else if (options.scope == BenchmarkScope::kPublicWorkflow ||
                 options.scope == BenchmarkScope::kMetalPublicWorkflow) {
        if (!options.gpu_profile_path.empty()) {
          RunGpuProfileWorkflowWorkload(
            {"external_input", {}, false}, options.warmups, options.samples,
            options.butteraugli_target, options.gpu_aq_mode,
            options.gpu_profiling_mode,
            options.collect_final_butteraugli_score, options.input_path, *gpu,
            &gpu_profile_results, &sink);
        } else {
          RunPublicWorkflowOnlyWorkload(
              {"external_input", {}, false}, options.warmups, options.samples,
              options.butteraugli_target, options.gpu_aq_mode,
              options.density_mode,
              options.collect_final_butteraugli_score,
              options.input_path,
              options.scope == BenchmarkScope::kMetalPublicWorkflow,
              options.validation, *gpu, raw_results_pointer, &sink);
        }
      } else {
        RunWorkload({"external_input", {}, false}, options.warmups,
                    options.samples, options.butteraugli_target,
                    options.gpu_aq_mode, options.input_path, *gpu,
                    backend_options, &sink);
      }
    } else {
      for (const WorkloadSpec& workload : kWorkloads) {
        if (options.workload == "all" || options.workload == workload.name) {
          if (options.scope == BenchmarkScope::kCoefficientCoding) {
            RunCoefficientCodingOnlyWorkload(
                workload, options.warmups, options.samples,
                options.butteraugli_target, {}, &sink);
          } else if (options.scope == BenchmarkScope::kPublicWorkflow ||
                     options.scope ==
                         BenchmarkScope::kMetalPublicWorkflow) {
            if (!options.gpu_profile_path.empty()) {
              RunGpuProfileWorkflowWorkload(
                workload, options.warmups, options.samples,
                options.butteraugli_target, options.gpu_aq_mode,
                options.gpu_profiling_mode,
                options.collect_final_butteraugli_score, {}, *gpu,
                &gpu_profile_results,
                &sink);
            } else {
              RunPublicWorkflowOnlyWorkload(
                  workload, options.warmups, options.samples,
                  options.butteraugli_target, options.gpu_aq_mode,
                  options.density_mode,
                  options.collect_final_butteraugli_score, {},
                  options.scope == BenchmarkScope::kMetalPublicWorkflow,
                  options.validation, *gpu, raw_results_pointer, &sink);
            }
          } else if (workload.gpu_complete_aq_only) {
            RunGpuCompleteAqOnlyWorkload(
                workload, options.warmups, options.samples,
                options.butteraugli_target, options.gpu_aq_mode, *gpu, &sink);
          } else {
            RunWorkload(workload, options.warmups, options.samples,
                        options.butteraugli_target, options.gpu_aq_mode, {},
                        *gpu, backend_options, &sink);
          }
        }
      }
    }
    if (!options.raw_samples_path.empty()) {
      WriteRawWorkflowSamples(options.raw_samples_path, options, raw_results);
    }
    if (!options.gpu_profile_path.empty()) {
      WriteGpuProfileSamples(
        options.gpu_profile_path, options, gpu_profile_results);
    }
    std::cout << "global_sink=" << sink << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
