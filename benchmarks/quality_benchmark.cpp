// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Uninstrumented, warm, complete-public-call timing for matched-quality
// studies.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "codestream/workflow.h"
#include "io/pfm.h"

#ifndef GJXL_QUALITY_REVISION
#error "GJXL_QUALITY_REVISION must identify the checkout used for this build"
#endif

namespace {
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
struct Options {
  fs::path input, output, raw;
  float distance = 1.0f;
  size_t effort = 7, threads = 8, warmups = 1, samples = 1;
};

void Check(gjxl::Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}

size_t Integer(const std::string &value) {
  if (value.empty() ||
      value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("Expected a nonnegative integer");
  return std::stoull(value);
}

Options Parse(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 == argc)
      throw std::runtime_error("Missing argument value: " + key);
    const std::string value = argv[++i];
    if (key == "--input")
      options.input = value;
    else if (key == "--output")
      options.output = value;
    else if (key == "--raw-samples")
      options.raw = value;
    else if (key == "--distance") {
      size_t end = 0;
      options.distance = std::stof(value, &end);
      if (end != value.size())
        throw std::runtime_error("Invalid distance");
    } else if (key == "--effort")
      options.effort = Integer(value);
    else if (key == "--num-threads")
      options.threads = Integer(value);
    else if (key == "--warmups")
      options.warmups = Integer(value);
    else if (key == "--samples")
      options.samples = Integer(value);
    else
      throw std::runtime_error("Unknown option: " + key);
  }
  if (options.input.empty() || options.output.empty() || options.raw.empty() ||
      !std::isfinite(options.distance) || options.distance <= 0 ||
      options.effort < 1 || options.effort > 10 || options.samples < 1 ||
      options.threads < 1 || options.threads > gjxl::kMaximumCpuThreadCount)
    throw std::runtime_error(
        "Invalid or missing input/output/distance/effort/thread/sample option");
  const auto input = fs::weakly_canonical(options.input);
  const auto output = fs::weakly_canonical(options.output);
  const auto raw = fs::weakly_canonical(options.raw);
  if (input == output || input == raw || output == raw)
    throw std::runtime_error(
        "Input, output and raw report must be distinct paths");
  return options;
}

void Write(const fs::path &destination, const char *data, size_t size) {
  if (!destination.parent_path().empty())
    fs::create_directories(destination.parent_path());
  fs::path temporary = destination;
  temporary +=
      ".tmp-" + std::to_string(Clock::now().time_since_epoch().count());
  try {
    std::ofstream stream(temporary, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(data, static_cast<std::streamsize>(size));
    stream.close();
    fs::rename(temporary, destination);
  } catch (...) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw;
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << "{\"encoder\":\"gjxl\",\"schema_version\":1,\"revision\":\""
                << GJXL_QUALITY_REVISION
                << "\",\"shader_source\":\"embedded-production\"}\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout
          << "gjxl_quality_benchmark --input IMAGE.pfm --output IMAGE.jxl "
             "--raw-samples REPORT.json [--distance D] [--effort 1..10] "
             "[--num-threads 1..256] [--warmups N] [--samples N]\n";
      return 0;
    }
    const Options options = Parse(argc, argv);
    gjxl::Image3FBuffer image;
    Check(gjxl::io::ReadPfm(options.input, &image));
    const gjxl::VarDctEncodingOptions settings{
        .butteraugli_target = options.distance,
        .effort = static_cast<int32_t>(options.effort),
        .cpu_thread_count = options.threads,
        .backend = gjxl::VarDctBackendPreference::kMetal,
        .metal_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
        .collect_final_butteraugli_score = false,
    };
    std::vector<uint8_t> bytes;
    gjxl::VarDctEncodingSummary summary;
    const auto encode = [&] {
      return gjxl::EncodeLinearRgbVarDctCodestream(image.const_view(), settings,
                                                   &bytes, &summary);
    };
    const auto validate = [&] {
      if (summary.execution_backend != gjxl::VarDctExecutionBackend::kMetal ||
          summary.metal_aq_mode !=
              gjxl::GpuAdaptiveQuantizationMode::kFullyResident ||
          summary.extent != image.extent() ||
          summary.encoded_bytes != bytes.size() || bytes.size() < 2 ||
          bytes[0] != 0xff || bytes[1] != 0x0a)
        throw std::runtime_error(
            "Unexpected encoder backend, dimensions or output");
    };
    // Also initializes the process-cached Metal backend and CPU execution
    // state.
    Check(encode());
    validate();
    const auto expected = bytes;
    const auto check_output = [&] {
      validate();
      if (bytes != expected)
        throw std::runtime_error("Codestream changed between encodes");
    };
    for (size_t i = 0; i < options.warmups; ++i) {
      Check(encode());
      check_output();
    }
    std::vector<uint64_t> times;
    times.reserve(options.samples);
    for (size_t i = 0; i < options.samples; ++i) {
      const auto start = Clock::now();
      const auto status = encode();
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               Clock::now() - start)
                               .count();
      Check(status);
      check_output();
      if (elapsed <= 0)
        throw std::runtime_error("Invalid elapsed time");
      times.push_back(static_cast<uint64_t>(elapsed));
    }
    std::ostringstream report;
    report << std::setprecision(12)
           << "{\"schema_version\":1,\"encoder\":\"gjxl\",\"revision\":\""
           << GJXL_QUALITY_REVISION
           << "\",\"timing_semantics\":\"complete-encode-wall-time\","
           << "\"stage_profile_enabled\":false,\"backend\":\"metal\","
           << "\"metal_aq_mode\":\"fully-resident\",\"density\":\"default\","
           << "\"compression\":\"automatic\",\"collect_final_score\":false,"
           << "\"input_layout\":\"planar-linear-srgb-f32\",\"resampling\":1,"
           << "\"thread_count\":" << options.threads
           << ",\"thread_semantics\":\"maximum-participating-cpu-threads\","
           << "\"input_width\":" << image.extent().width
           << ",\"input_height\":" << image.extent().height
           << ",\"requested_distance\":" << options.distance
           << ",\"effort\":" << options.effort
           << ",\"validation_encodes\":1,\"warmups\":" << options.warmups
           << ",\"sample_count\":" << times.size() << ",\"samples\":[";
    for (size_t i = 0; i < times.size(); ++i) {
      if (i)
        report << ',';
      report << "{\"sample_index\":" << i
             << ",\"elapsed_nanoseconds\":" << times[i]
             << ",\"encoded_bytes\":" << bytes.size() << '}';
    }
    report << "]}\n";
    Write(options.output, reinterpret_cast<const char *>(bytes.data()),
          bytes.size());
    const auto text = report.str();
    Write(options.raw, text.data(), text.size());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
