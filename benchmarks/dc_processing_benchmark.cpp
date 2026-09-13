// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Complete calls with independently calibrated distances and a shared backend.
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "codestream/workflow.h"
#include "io/pfm.h"

namespace {
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
struct Variant {
  std::string name, prediction, quantization;
  gjxl::VarDctEncodingOptions options;
  std::vector<uint8_t> expected;
};
} // namespace

int main(int argc, char **argv) {
  try {
    Require(argc == 9,
            "Usage: gjxl_dc_processing_benchmark input.pfm new-output-dir "
            "variants.txt effort threads warmups rounds cpu|metal");
    const std::filesystem::path output(argv[2]);
    Require(!std::filesystem::exists(output), "Output directory must be new");
    const auto effort = Integer(argv[4]), threads = Integer(argv[5]);
    const auto warmups = Integer(argv[6]), rounds = Integer(argv[7]);
    const std::string backend(argv[8]);
    Require(effort >= 1 && effort <= 10 && threads >= 1 &&
                threads <= gjxl::kMaximumCpuThreadCount && warmups <= 100 &&
                rounds >= 1 && rounds <= 1000 &&
                (backend == "cpu" || backend == "metal"),
            "Invalid benchmark options");
    std::ifstream specification(argv[3]);
    Require(specification.good(), "Cannot read variant specification");
    std::vector<Variant> variants;
    std::set<std::string> names;
    std::string line;
    while (std::getline(specification, line)) {
      if (line.empty())
        continue;
      Variant variant;
      std::string distance_text, smoothing, extra;
      std::istringstream row(line);
      Require(bool(row >> variant.name >> distance_text >> variant.prediction >>
                   variant.quantization >> smoothing) &&
                  !(row >> extra),
              "Expected: name distance gradient|weighted "
              "round|prediction-aware 0|1");
      Require(!variant.name.empty() && variant.name.size() <= 64 &&
                  variant.name.find_first_not_of(
                      "abcdefghijklmnopqrstuvwxyz0123456789_-") ==
                      std::string::npos &&
                  names.insert(variant.name).second,
              "Invalid or duplicate variant name");
      size_t end = 0;
      const float distance = std::stof(distance_text, &end);
      Require(end == distance_text.size() && std::isfinite(distance) &&
                  distance > 0,
              "Invalid distance");
      Require((variant.prediction == "gradient" ||
               variant.prediction == "weighted") &&
                  (variant.quantization == "round" ||
                   variant.quantization == "prediction-aware") &&
                  (smoothing == "0" || smoothing == "1"),
              "Invalid DC controls");
      variant.options = {
          .butteraugli_target = distance,
          .effort = static_cast<int32_t>(effort),
          .cpu_thread_count = threads,
          .backend = backend == "cpu" ? gjxl::VarDctBackendPreference::kCpu
                                      : gjxl::VarDctBackendPreference::kMetal,
          .metal_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
          .collect_final_butteraugli_score = false,
          .dc_prediction = variant.prediction == "weighted"
                               ? gjxl::VarDctDcPrediction::kWeighted
                               : gjxl::VarDctDcPrediction::kGradient,
          .dc_quantization = variant.quantization == "round"
                                 ? gjxl::DcQuantizationMode::kRound
                                 : gjxl::DcQuantizationMode::kPredictionAware,
          .adaptive_dc_smoothing = smoothing == "1",
      };
      variants.push_back(std::move(variant));
      Require(variants.size() <= 16, "Too many variants");
    }
    Require(specification.eof() && !variants.empty(),
            "Missing or unreadable variants");
    gjxl::Image3FBuffer image;
    Check(gjxl::io::ReadPfm(argv[1], &image));
    std::vector<uint8_t> bytes;
    gjxl::VarDctEncodingSummary summary;
    const auto encode = [&](Variant &variant) {
      return gjxl::EncodeLinearRgbVarDctCodestream(
          image.const_view(), variant.options, &bytes, &summary);
    };
    const auto validate = [&](const Variant &variant) {
      Require(summary.execution_backend ==
                      (backend == "cpu"
                           ? gjxl::VarDctExecutionBackend::kCpu
                           : gjxl::VarDctExecutionBackend::kMetal) &&
                  (backend == "cpu" ||
                   summary.metal_aq_mode ==
                       gjxl::GpuAdaptiveQuantizationMode::kFullyResident) &&
                  summary.dc_prediction == variant.options.dc_prediction &&
                  summary.dc_quantization == variant.options.dc_quantization &&
                  summary.adaptive_dc_smoothing ==
                      variant.options.adaptive_dc_smoothing &&
                  summary.extent == image.extent() &&
                  summary.encoded_bytes == bytes.size() && bytes.size() >= 2 &&
                  bytes[0] == 0xff && bytes[1] == 0x0a,
              "Unexpected encoder provenance");
      Require(variant.expected.empty() || bytes == variant.expected,
              "Repeated codestream changed");
    };
    for (auto &variant : variants) {
      Check(encode(variant));
      validate(variant);
      variant.expected = bytes;
    }
    for (size_t i = 0; i < warmups; ++i) {
      for (auto &variant : variants) {
        Check(encode(variant));
        validate(variant);
      }
    }
    struct Sample {
      size_t round, position, variant;
      int64_t ns;
    };
    std::vector<Sample> samples;
    samples.reserve(rounds * variants.size());
    for (size_t round = 0; round < rounds; ++round) {
      for (size_t position = 0; position < variants.size(); ++position) {
        const size_t count = variants.size();
        const size_t variant_index = (round / count) % 2 == 0
                                         ? (round + position) % count
                                         : (round + count - position) % count;
        auto &variant = variants[variant_index];
        const auto begin = std::chrono::steady_clock::now();
        const auto status = encode(variant);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
        Check(status);
        validate(variant);
        Require(ns > 0, "Invalid encode time");
        samples.push_back({round, position, variant_index, ns});
      }
    }
    std::filesystem::create_directories(output);
    for (const auto &variant : variants) {
      std::ofstream file(output / (variant.name + ".jxl"), std::ios::binary);
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.write(reinterpret_cast<const char *>(variant.expected.data()),
                 static_cast<std::streamsize>(variant.expected.size()));
      file.close();
    }
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"revision\":\""
              << GJXL_QUALITY_REVISION << "\",\"backend\":\"" << backend
              << "\",\"metal_aq_mode\":\"fully-resident\","
              << "\"timing_semantics\":\"complete-encode-wall-time\",\"stage_"
                 "profile_enabled\":false,"
              << "\"collect_final_score\":false,\"order\":\"rotated-and-"
                 "reversed-rounds\","
              << "\"thread_count\":" << threads << ",\"effort\":" << effort
              << ",\"warmups_per_variant\":" << warmups
              << ",\"input_width\":" << image.extent().width
              << ",\"input_height\":" << image.extent().height
              << ",\"variants\":[";
    for (size_t i = 0; i < variants.size(); ++i) {
      const auto &v = variants[i];
      if (i)
        std::cout << ',';
      std::cout << "{\"name\":\"" << v.name
                << "\",\"distance\":" << v.options.butteraugli_target
                << ",\"dc_prediction\":\"" << v.prediction
                << "\",\"dc_quantization\":\"" << v.quantization
                << "\",\"adaptive_dc_smoothing\":"
                << (v.options.adaptive_dc_smoothing ? "true" : "false")
                << ",\"encoded_bytes\":" << v.expected.size() << '}';
    }
    std::cout << "],\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto &s = samples[i];
      if (i)
        std::cout << ',';
      std::cout << "{\"round\":" << s.round << ",\"position\":" << s.position
                << ",\"variant\":\"" << variants[s.variant].name
                << "\",\"elapsed_nanoseconds\":" << s.ns << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
