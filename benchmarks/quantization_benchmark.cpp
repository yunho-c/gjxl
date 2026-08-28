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
#include "codec/loop_filter.h"
#include "codec/quantization.h"
#include "codec/quantization_pipeline.h"
#include "codec/reconstruction.h"
#include "codestream/workflow.h"
#include "codestream/workflow_internal.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_aq_evaluation_profile.h"
#include "gpu/ops/ac_strategy_search.h"
#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/quantization_pipeline.h"

#ifndef GJXL_FLOWER_PPM_PATH
#error "GJXL_FLOWER_PPM_PATH must identify the pinned Flower PPM"
#endif

namespace {

namespace aqi = gjxl::adaptive_quantization_internal;
using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWarmups = 3;
constexpr size_t kDefaultSamples = 5;
constexpr float kDefaultButteraugliTarget = 1.2f;

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
    "gpu_complete_pipeline_two_updates",
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
  std::string implementation = "simd";
  float butteraugli_target = kDefaultButteraugliTarget;
  size_t warmups = kDefaultWarmups;
  size_t samples = kDefaultSamples;
};

struct WorkloadSpec {
  std::string_view name;
  gjxl::Extent2D source_extent;
  bool flower = false;
  bool workflow_gradient = false;
};

constexpr std::array<WorkloadSpec, 13> kWorkloads = {{
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

[[nodiscard]] CommandLineOptions ParseCommandLine(int argc, char** argv) {
  CommandLineOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << "usage: gjxl_quantization_benchmark "
                   "[--workload NAME|all] "
                   "[--input IMAGE.ppm] "
                   "[--implementation scalar|simd|factored] "
                   "[--distance D] [--warmups N] [--samples N]\n";
      std::exit(EXIT_SUCCESS);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("Benchmark option is missing its value");
    }
    const std::string_view value = argv[++index];
    if (argument == "--workload") {
      options.workload = value;
    } else if (argument == "--input") {
      options.input_path = value;
    } else if (argument == "--implementation") {
      if (value != "scalar" && value != "simd" && value != "factored") {
        throw std::runtime_error(
            "Unknown Metal DCT implementation: " + std::string(value));
      }
      options.implementation = value;
    } else if (argument == "--distance") {
      options.butteraugli_target = ParsePositiveFloat(value);
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

[[nodiscard]] double NanosecondsToMilliseconds(uint64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1.0e6;
}

void RunWorkload(const WorkloadSpec& spec, size_t warmups, size_t samples,
                 float butteraugli_target,
                 std::string_view input_path,
                 gjxl::GpuBackend& gpu,
                 const gjxl::MetalBackendOptions& backend_options,
                 double* global_sink) {

  ImageStorage original = !input_path.empty()
      ? LoadPpm(input_path)
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
                    pipeline_options,
                    gpu_pipeline_stage.PipelineOutput()));
  RequireStatus(
      "GPU resident validation",
      gjxl::RunGpuAdaptiveQuantization(
          gpu, original.ConstView(), preprocessed.ConstView(),
          stage.strategies,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          one_evaluation_options, gpu_stage.AdaptiveOutput()));

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
               .backend = gjxl::VarDctBackendPreference::kMetal},
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
  if (!rollout_matches) {
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
                  resident_raw_quant.data(), stage.block_extent,
                  stage.block_extent.width},
              .quantizer = resident_quantizer.params(),
              .y_to_x = resident_color.y_to_x_map(),
              .y_to_b = resident_color.y_to_b_map(),
              .epf_inverse_sigma = {
                  resident_inverse_sigma.data(), stage.block_extent,
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
                  resident_raw_quant.data(), stage.block_extent,
                  stage.block_extent.width},
              .quantizer = resident_quantizer.params(),
              .y_to_x = resident_color.y_to_x_map(),
              .y_to_b = resident_color.y_to_b_map(),
              .epf_inverse_sigma = {
                  resident_inverse_sigma.data(), stage.block_extent,
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
                  resident_raw_quant.data(), stage.block_extent,
                  stage.block_extent.width},
              .quantizer = resident_quantizer.params(),
              .y_to_x = resident_color.y_to_x_map(),
              .y_to_b = resident_color.y_to_b_map(),
              .epf_inverse_sigma = {
                  resident_inverse_sigma.data(), stage.block_extent,
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
          one_evaluation_options, gpu_stage.AdaptiveOutput());
    case Phase::kGpuAqTwoUpdates:
      return gjxl::RunGpuAdaptiveQuantization(
          gpu, original.ConstView(), preprocessed.ConstView(), stage.strategies,
          {stage.initial_quant.data(), stage.block_extent,
           stage.block_extent.width},
          {sharpness.data(), stage.block_extent, stage.block_extent.width},
          two_update_options, gpu_stage.AdaptiveOutput());
    case Phase::kGpuCompletePipeline:
      return gjxl::RunGpuQuantizationPipeline(
          gpu, original.ConstView(), opsin.ConstView(), pipeline_options,
          gpu_pipeline_stage.PipelineOutput(), &gpu_search_stats);
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
               .backend = gjxl::VarDctBackendPreference::kMetal},
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
               .backend = gjxl::VarDctBackendPreference::kMetal},
              cold_gpu.get(), true, &workflow_bytes, &workflow_summary);
    }
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
            << '\n';
  std::cout << std::scientific << std::setprecision(6)
            << "  rollout_errors pipeline_quant=" << pipeline_quant_error
            << " pipeline_block=" << pipeline_block_error
            << " pipeline_score=" << pipeline_score_error
            << " pipeline_rgb=" << pipeline_reconstruction_error
            << " one_block=" << one_block_error
            << " one_score=" << one_score_error
            << " one_rgb=" << one_reconstruction_error
            << " frame=" << pipeline_frame_difference
            << " codestream="
            << (cpu_validation_bytes == gpu_validation_bytes ? "exact"
                                                             : "different")
            << " gate=" << (rollout_matches ? "pass" : "fail") << '\n'
            << "  fully_resident_errors block="
            << fully_resident_block_error
            << " score=" << fully_resident_score_error
            << " rgb=" << fully_resident_rgb_error
            << " frame=" << fully_resident_frame_difference
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
    RequireStatus("Create embedded benchmark Metal backend",
                  gjxl::CreateEmbeddedMetalBackend(backend_options, &gpu));
    std::cout << std::fixed << std::setprecision(3)
              << "CPU/Metal quantization benchmark: backend=" << gpu->name()
              << " implementation=" << options.implementation
              << " distance=" << options.butteraugli_target
              << " warmups=" << options.warmups
              << " samples=" << options.samples
              << " rotated_phases=" << kPhaseCount << '\n';
    double sink = 0.0;
    if (!options.input_path.empty()) {
      RunWorkload({"external_input", {}, false}, options.warmups,
                  options.samples, options.butteraugli_target,
                  options.input_path, *gpu, backend_options, &sink);
    } else {
      for (const WorkloadSpec& workload : kWorkloads) {
        if (options.workload == "all" || options.workload == workload.name) {
          RunWorkload(workload, options.warmups, options.samples,
                      options.butteraugli_target, {}, *gpu, backend_options,
                      &sink);
        }
      }
    }
    std::cout << "global_sink=" << sink << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
