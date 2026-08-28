// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "codestream/workflow.h"
#include "core/ac_strategy.h"
#include "core/image_buffer.h"
#include "io/pfm.h"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Options {
  std::vector<fs::path> inputs;
  std::vector<float> targets = {0.5f, 0.75f, 1.0f, 1.2f, 1.5f, 2.0f, 3.0f,
                                4.0f};
  gjxl::VarDctBackendPreference backend =
    gjxl::VarDctBackendPreference::kCpu;
  gjxl::GpuAdaptiveQuantizationMode metal_aq_mode =
    gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
};

struct ProbeRow {
  std::string input;
  gjxl::Extent2D extent;
  float target = 0.0f;
  size_t encoded_bytes = 0;
  double bits_per_pixel = 0.0;
  double final_score = 0.0;
  std::string backend;
  std::string metal_aq_mode;
  double elapsed_ms = 0.0;
  double total_ms = 0.0;
  bool size_monotonic = true;
  std::string strategy_counts;
};

[[nodiscard]] float ParsePositiveFloat(std::string_view text) {
  if (text.empty()) {
    throw std::runtime_error("Rate-control target is empty");
  }
  std::string terminated(text);
  char* end = nullptr;
  errno = 0;
  const float value = std::strtof(terminated.c_str(), &end);
  if (errno == ERANGE || end != terminated.c_str() + terminated.size() ||
      !std::isfinite(value) || value <= 0.0f) {
    throw std::runtime_error(
      "Rate-control targets must be finite and positive");
  }
  return value;
}

[[nodiscard]] std::vector<float> ParseTargets(std::string_view text) {
  std::vector<float> result;
  size_t begin = 0;
  while (begin <= text.size()) {
    const size_t separator = text.find(',', begin);
    const size_t end = separator == std::string_view::npos
      ? text.size()
      : separator;
    result.push_back(ParsePositiveFloat(text.substr(begin, end - begin)));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1;
  }
  if (result.empty()) {
    throw std::runtime_error("At least one rate-control target is required");
  }
  for (size_t index = 1; index < result.size(); ++index) {
    if (!(result[index] > result[index - 1])) {
      throw std::runtime_error(
        "Rate-control targets must be strictly increasing");
    }
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
  throw std::runtime_error("Unknown rate-control backend: " +
                           std::string(text));
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
  throw std::runtime_error("Unknown Metal AQ mode: " + std::string(text));
}

void PrintUsage(std::string_view executable) {
  std::cout
    << "Usage: " << executable
    << " [--targets D1,D2,...] [--backend cpu|auto|metal] "
       "[--metal-aq exact-coefficients|fully-resident|throughput] "
       "INPUT.pfm [INPUT.pfm ...]\n";
}

[[nodiscard]] Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--targets" || argument == "--backend" ||
        argument == "--metal-aq") {
      if (index + 1 >= argc) {
        throw std::runtime_error(
          "Rate-control probe option is missing its value");
      }
      const std::string_view value = argv[++index];
      if (argument == "--targets") {
        options.targets = ParseTargets(value);
      } else if (argument == "--backend") {
        options.backend = ParseBackend(value);
      } else {
        options.metal_aq_mode = ParseMetalAqMode(value);
      }
    } else if (!argument.empty() && argument.front() == '-') {
      throw std::runtime_error("Unknown rate-control probe option: " +
                               std::string(argument));
    } else {
      options.inputs.emplace_back(argument);
    }
  }
  if (options.inputs.empty()) {
    throw std::runtime_error(
      "The rate-control probe requires at least one PFM input");
  }
  if (options.metal_aq_mode !=
        gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients &&
      options.backend != gjxl::VarDctBackendPreference::kMetal) {
    throw std::runtime_error(
      "Resident AQ requires the forced Metal backend");
  }
  return options;
}

[[nodiscard]] std::string BackendName(
  gjxl::VarDctExecutionBackend backend) {

  switch (backend) {
    case gjxl::VarDctExecutionBackend::kCpu:
      return "cpu";
    case gjxl::VarDctExecutionBackend::kMetal:
      return "metal";
  }
  return "invalid";
}

[[nodiscard]] std::string MetalAqModeName(
  gjxl::VarDctExecutionBackend backend,
  gjxl::GpuAdaptiveQuantizationMode mode) {

  if (backend == gjxl::VarDctExecutionBackend::kCpu) {
    return "none";
  }
  switch (mode) {
    case gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients:
      return "exact-coefficients";
    case gjxl::GpuAdaptiveQuantizationMode::kFullyResident:
      return "fully-resident";
    case gjxl::GpuAdaptiveQuantizationMode::kThroughput:
      return "throughput";
  }
  return "invalid";
}

[[nodiscard]] std::string StrategyCounts(
  const gjxl::VarDctEncodingSummary& summary) {

  std::string result;
  for (size_t index = 0; index < summary.strategy_counts.size(); ++index) {
    if (summary.strategy_counts[index] == 0) {
      continue;
    }
    if (!result.empty()) {
      result += ';';
    }
    result += gjxl::kAcStrategyInfos[index].name;
    result += '=';
    result += std::to_string(summary.strategy_counts[index]);
  }
  return result;
}

[[nodiscard]] std::string CsvField(std::string_view value) {
  if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
    return std::string(value);
  }
  std::string result;
  result.reserve(value.size() + 2);
  result.push_back('"');
  for (char character : value) {
    if (character == '"') {
      result.push_back('"');
    }
    result.push_back(character);
  }
  result.push_back('"');
  return result;
}

[[nodiscard]] std::vector<ProbeRow> ProbeInput(
  const fs::path& input,
  const Options& options) {

  gjxl::Image3FBuffer image;
  const gjxl::Status read_status = gjxl::io::ReadPfm(input, &image);
  if (!read_status.ok()) {
    throw std::runtime_error(
      "Unable to read " + input.string() + ": " +
      std::string(read_status.message()));
  }

  std::vector<ProbeRow> rows;
  rows.reserve(options.targets.size());
  size_t previous_size = 0;
  const auto workload_start = Clock::now();
  for (float target : options.targets) {
    std::vector<uint8_t> codestream;
    gjxl::VarDctEncodingSummary summary;
    const auto attempt_start = Clock::now();
    const gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
      image.const_view(),
      {
        .butteraugli_target = target,
        .backend = options.backend,
        .metal_aq_mode = options.metal_aq_mode,
      },
      &codestream,
      &summary);
    const auto attempt_end = Clock::now();
    if (!status.ok()) {
      throw std::runtime_error(
        "Unable to encode " + input.string() + " at target " +
        std::to_string(target) + ": " + std::string(status.message()));
    }
    const size_t strategy_count = std::accumulate(
      summary.strategy_counts.begin(),
      summary.strategy_counts.end(),
      size_t{0});
    if (codestream.empty() || summary.encoded_bytes != codestream.size() ||
        summary.extent != image.extent() ||
        summary.rate_control_mode !=
          gjxl::VarDctRateControlMode::kButteraugliTarget ||
        summary.selected_butteraugli_target != target ||
        summary.encode_attempt_count != 1 ||
        !std::isfinite(summary.achieved_bits_per_pixel) ||
        summary.achieved_bits_per_pixel <= 0.0 ||
        summary.score_history.empty() ||
        !std::isfinite(summary.score_history.back()) ||
        strategy_count == 0) {
      throw std::runtime_error(
        "Rate-control workflow returned an invalid result summary");
    }

    ProbeRow row;
    row.input = input.string();
    row.extent = summary.extent;
    row.target = target;
    row.encoded_bytes = summary.encoded_bytes;
    row.bits_per_pixel = summary.achieved_bits_per_pixel;
    row.final_score = summary.score_history.back();
    row.backend = BackendName(summary.execution_backend);
    row.metal_aq_mode = MetalAqModeName(
      summary.execution_backend, summary.metal_aq_mode);
    row.elapsed_ms = std::chrono::duration<double, std::milli>(
      attempt_end - attempt_start).count();
    row.size_monotonic = rows.empty() || row.encoded_bytes <= previous_size;
    row.strategy_counts = StrategyCounts(summary);
    previous_size = row.encoded_bytes;
    rows.push_back(std::move(row));
  }
  const auto workload_end = Clock::now();
  const double total_ms = std::chrono::duration<double, std::milli>(
    workload_end - workload_start).count();
  for (ProbeRow& row : rows) {
    row.total_ms = total_ms;
  }
  return rows;
}

void PrintRows(const std::vector<ProbeRow>& rows) {
  for (const ProbeRow& row : rows) {
    std::cout << std::defaultfloat
              << CsvField(row.input) << ','
              << row.extent.width << ','
              << row.extent.height << ','
              << std::setprecision(9) << row.target << ','
              << row.encoded_bytes << ','
              << std::setprecision(17) << row.bits_per_pixel << ','
              << row.final_score << ','
              << row.backend << ','
              << row.metal_aq_mode << ','
              << std::fixed << std::setprecision(6)
              << row.elapsed_ms << ','
              << row.total_ms << ','
              << (row.size_monotonic ? 1 : 0) << ','
              << CsvField(row.strategy_counts) << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    std::cout
      << "input,width,height,target,bytes,bpp,score,backend,metal_aq,"
         "elapsed_ms,total_ms,size_monotonic,strategy_counts\n";
    for (const fs::path& input : options.inputs) {
      PrintRows(ProbeInput(input, options));
    }
  } catch (const std::exception& exception) {
    std::cerr << "Rate-control probe error: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
