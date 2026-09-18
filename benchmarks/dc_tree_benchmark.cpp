// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Paired, uninstrumented complete calls sharing one warm Metal backend.
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "codestream/dc_context_tree_internal.h"
#include "codestream/workflow.h"
#include "io/pfm.h"

namespace {
using Clock = std::chrono::steady_clock;
void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Check(gjxl::Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
size_t Integer(const char *text) {
  const std::string value(text);
  Require(!value.empty() &&
              value.find_first_not_of("0123456789") == std::string::npos,
          "Invalid integer");
  return std::stoull(value);
}
} // namespace

int main(int argc, char **argv) {
  try {
    Require(
        argc == 11,
        "Usage: gjxl_dc_tree_benchmark input.pfm new-output-dir distance "
        "effort threads warmups pairs gradient|weighted cpu|metal flags(0..3)");
    const std::filesystem::path output(argv[2]);
    Require(!std::filesystem::exists(output), "Output directory must be new");
    size_t end = 0;
    const float distance = std::stof(argv[3], &end);
    Require(end == std::string(argv[3]).size() && std::isfinite(distance) &&
                distance > 0,
            "Invalid distance");
    const auto effort = Integer(argv[4]), threads = Integer(argv[5]);
    const auto warmups = Integer(argv[6]), pairs = Integer(argv[7]);
    Require(effort >= 1 && effort <= 10 && threads >= 1 &&
                threads <= gjxl::kMaximumCpuThreadCount && warmups <= 100 &&
                pairs >= 1 && pairs <= 1000,
            "Invalid effort/thread/count option");
    const std::string prediction(argv[8]), backend(argv[9]);
    const size_t flags = Integer(argv[10]);
    Require((prediction == "gradient" || prediction == "weighted") &&
                (backend == "cpu" || backend == "metal") && flags <= 3,
            "Invalid controls");
    gjxl::Image3FBuffer image;
    Check(gjxl::io::ReadPfm(argv[1], &image));
    gjxl::VarDctEncodingOptions options{
        .butteraugli_target = distance,
        .effort = static_cast<int32_t>(effort),
        .cpu_thread_count = threads,
        .backend = backend == "metal" ? gjxl::VarDctBackendPreference::kMetal
                                      : gjxl::VarDctBackendPreference::kCpu,
        .gpu_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
        .collect_final_butteraugli_score = false,
    };
    options.dc_prediction = prediction == "weighted"
                                ? gjxl::VarDctDcPrediction::kWeighted
                                : gjxl::VarDctDcPrediction::kGradient;
    options.dc_quantization = flags & 1
                                  ? gjxl::DcQuantizationMode::kPredictionAware
                                  : gjxl::DcQuantizationMode::kRound;
    options.adaptive_dc_smoothing = flags & 2;
    size_t dc_samples = 0;
    Check(gjxl::codestream_internal::ComputeDcSampleCount(
        image.extent().ceil_div(8), &dc_samples));
    const gjxl::codestream_internal::DcContextTreeLayout *layout = nullptr;
    Check(gjxl::codestream_internal::SelectDcContextTreeLayout(
        dc_samples, options.dc_prediction,
        gjxl::codestream_internal::DcTreePolicy::kAdaptive, &layout));
    std::vector<uint8_t> bytes;
    gjxl::VarDctEncodingSummary summary;
    std::array<std::vector<uint8_t>, 2> expected;
    size_t selected = 0;
    const auto select = [&](size_t variant) { selected = variant; };
    const auto encode = [&] {
      gjxl::codestream_internal::ScopedDcTreePolicyForTesting choice(
          selected ? gjxl::codestream_internal::DcTreePolicy::kAdaptive
                   : gjxl::codestream_internal::DcTreePolicy::kLegacy);
      return gjxl::EncodeLinearRgbVarDctCodestream(image.const_view(), options,
                                                   &bytes, &summary);
    };
    const auto validate = [&](size_t variant) {
      Require(
          summary.execution_backend ==
                  (backend == "metal" ? gjxl::VarDctExecutionBackend::kMetal
                                      : gjxl::VarDctExecutionBackend::kCpu) &&
              (backend == "cpu" ||
               summary.gpu_aq_mode ==
                   gjxl::GpuAdaptiveQuantizationMode::kFullyResident) &&
              summary.dc_quantization == options.dc_quantization &&
              summary.adaptive_dc_smoothing == options.adaptive_dc_smoothing &&
              summary.dc_prediction == options.dc_prediction &&
              summary.extent == image.extent() &&
              summary.encoded_bytes == bytes.size() && bytes.size() >= 2 &&
              bytes[0] == 0xff && bytes[1] == 0x0a,
          "Unexpected encoder result");
      if (!expected[variant].empty())
        Require(bytes == expected[variant], "Timed codestream changed");
    };
    for (size_t variant = 0; variant < 2; ++variant) {
      select(variant);
      Check(encode());
      validate(variant);
      expected[variant] = bytes;
    }
    for (size_t i = 0; i < warmups; ++i) {
      for (size_t variant = 0; variant < 2; ++variant) {
        select(variant);
        Check(encode());
        validate(variant);
      }
    }
    struct Sample {
      size_t pair, position, variant;
      int64_t ns;
    };
    std::vector<Sample> samples;
    samples.reserve(pairs * 2);
    for (size_t pair = 0; pair < pairs; ++pair) {
      for (size_t position = 0; position < 2; ++position) {
        const auto variant = (pair + position) % 2;
        select(variant);
        const auto begin = Clock::now();
        const auto status = encode();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            Clock::now() - begin)
                            .count();
        Check(status);
        validate(variant);
        Require(ns > 0, "Invalid encode time");
        samples.push_back({pair, position, variant, ns});
      }
    }
    std::filesystem::create_directories(output);
    for (size_t variant = 0; variant < 2; ++variant) {
      std::ofstream stream(output / (variant ? "adaptive.jxl" : "legacy.jxl"),
                           std::ios::binary);
      stream.exceptions(std::ios::failbit | std::ios::badbit);
      stream.write(reinterpret_cast<const char *>(expected[variant].data()),
                   static_cast<std::streamsize>(expected[variant].size()));
      stream.close();
    }
    std::cout
        << std::setprecision(12) << "{\"schema_version\":1,\"revision\":\""
        << GJXL_QUALITY_REVISION << "\",\"backend\":\"" << backend
        << "\",\"dc_prediction\":\"" << prediction << "\",\"flags\":" << flags
        << ",\"dc_samples\":" << dc_samples
        << ",\"adaptive_dc_leaves\":" << layout->dc_leaf_count
        << ",\"adaptive_contexts\":" << layout->context_count << ','
        << "\"timing_semantics\":\"complete-encode-wall-time\","
        << "\"pairing\":\"alternating-calls-in-one-process\","
        << "\"stage_profile_enabled\":false,\"collect_final_score\":false,"
        << "\"thread_count\":" << threads << ",\"effort\":" << effort
        << ",\"requested_distance\":" << distance
        << ",\"warmups_per_variant\":" << warmups
        << ",\"input_width\":" << image.extent().width
        << ",\"input_height\":" << image.extent().height << ",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto &sample = samples[i];
      if (i)
        std::cout << ',';
      std::cout << "{\"pair\":" << sample.pair
                << ",\"position\":" << sample.position << ",\"variant\":\""
                << (sample.variant ? "adaptive" : "legacy")
                << "\",\"encoded_bytes\":" << expected[sample.variant].size()
                << ",\"elapsed_nanoseconds\":" << sample.ns << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
