// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "codestream/workflow.h"
#include "core/ac_strategy.h"
#include "core/image_buffer.h"
#include "core/status.h"
#include "io/pfm.h"

namespace {

namespace fs = std::filesystem;

struct Options {
  fs::path input;
  fs::path output;
  float butteraugli_target = 1.0f;
  gjxl::VarDctRateControlMode rate_control_mode =
    gjxl::VarDctRateControlMode::kButteraugliTarget;
  size_t target_bytes = 0;
  double target_bits_per_pixel = 0.0;
  double target_size_tolerance = 0.005;
  size_t target_size_maximum_attempts = 12;
  gjxl::VarDctBackendPreference backend =
    gjxl::VarDctBackendPreference::kAutomatic;
  gjxl::GpuAdaptiveQuantizationMode metal_aq_mode =
    gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
};

[[nodiscard]] bool ParseBackend(
  std::string_view text,
  gjxl::VarDctBackendPreference* backend) {

  if (backend == nullptr) {
    return false;
  }
  if (text == "auto") {
    *backend = gjxl::VarDctBackendPreference::kAutomatic;
  } else if (text == "cpu") {
    *backend = gjxl::VarDctBackendPreference::kCpu;
  } else if (text == "metal") {
    *backend = gjxl::VarDctBackendPreference::kMetal;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] bool ParseMetalAqMode(
  std::string_view text,
  gjxl::GpuAdaptiveQuantizationMode* mode) {

  if (mode == nullptr) {
    return false;
  }
  if (text == "exact-coefficients") {
    *mode = gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
  } else if (text == "fully-resident") {
    *mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] bool ParsePositiveFloat(
  std::string_view text,
  float* value) {

  if (text.empty() || value == nullptr) {
    return false;
  }
  std::string terminated(text);
  char* end = nullptr;
  errno = 0;
  const float candidate = std::strtof(terminated.c_str(), &end);
  if (errno == ERANGE || end != terminated.c_str() + terminated.size() ||
      !std::isfinite(candidate) || candidate <= 0.0f) {
    return false;
  }
  *value = candidate;
  return true;
}

[[nodiscard]] bool ParsePositiveDouble(
  std::string_view text,
  double* value) {

  if (text.empty() || value == nullptr) {
    return false;
  }
  std::string terminated(text);
  char* end = nullptr;
  errno = 0;
  const double candidate = std::strtod(terminated.c_str(), &end);
  if (errno == ERANGE || end != terminated.c_str() + terminated.size() ||
      !std::isfinite(candidate) || candidate <= 0.0) {
    return false;
  }
  *value = candidate;
  return true;
}

[[nodiscard]] bool ParseNonnegativeDouble(
  std::string_view text,
  double* value) {

  if (text.empty() || value == nullptr) {
    return false;
  }
  std::string terminated(text);
  char* end = nullptr;
  errno = 0;
  const double candidate = std::strtod(terminated.c_str(), &end);
  if (errno == ERANGE || end != terminated.c_str() + terminated.size() ||
      !std::isfinite(candidate) || candidate < 0.0) {
    return false;
  }
  *value = candidate;
  return true;
}

[[nodiscard]] bool ParsePositiveSize(
  std::string_view text,
  size_t maximum,
  size_t* value) {

  if (text.empty() || text.front() == '-' || value == nullptr) {
    return false;
  }
  std::string terminated(text);
  char* end = nullptr;
  errno = 0;
  const unsigned long long candidate =
    std::strtoull(terminated.c_str(), &end, 10);
  if (errno == ERANGE || end != terminated.c_str() + terminated.size() ||
      candidate == 0 || candidate > maximum ||
      candidate > std::numeric_limits<size_t>::max()) {
    return false;
  }
  *value = static_cast<size_t>(candidate);
  return true;
}

[[nodiscard]] bool ParseOptions(
  int argc,
  char** argv,
  Options* options) {

  if (options == nullptr) {
    return false;
  }
  Options candidate;
  bool rate_control_set = false;
  bool target_search_option_set = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--distance") {
      if (rate_control_set || index + 1 >= argc ||
          !ParsePositiveFloat(argv[++index],
                              &candidate.butteraugli_target)) {
        return false;
      }
      candidate.rate_control_mode =
        gjxl::VarDctRateControlMode::kButteraugliTarget;
      rate_control_set = true;
    } else if (argument == "--target-bytes") {
      if (rate_control_set || index + 1 >= argc ||
          !ParsePositiveSize(
            argv[++index],
            std::numeric_limits<size_t>::max(),
            &candidate.target_bytes)) {
        return false;
      }
      candidate.rate_control_mode =
        gjxl::VarDctRateControlMode::kTargetBytes;
      rate_control_set = true;
    } else if (argument == "--target-bpp") {
      if (rate_control_set || index + 1 >= argc ||
          !ParsePositiveDouble(
            argv[++index], &candidate.target_bits_per_pixel)) {
        return false;
      }
      candidate.rate_control_mode =
        gjxl::VarDctRateControlMode::kTargetBitsPerPixel;
      rate_control_set = true;
    } else if (argument == "--size-tolerance") {
      if (index + 1 >= argc ||
          !ParseNonnegativeDouble(
            argv[++index], &candidate.target_size_tolerance) ||
          candidate.target_size_tolerance > 1.0) {
        return false;
      }
      target_search_option_set = true;
    } else if (argument == "--max-attempts") {
      if (index + 1 >= argc ||
          !ParsePositiveSize(
            argv[++index],
            64,
            &candidate.target_size_maximum_attempts)) {
        return false;
      }
      target_search_option_set = true;
    } else if (argument == "--backend") {
      if (index + 1 >= argc ||
          !ParseBackend(argv[++index], &candidate.backend)) {
        return false;
      }
    } else if (argument == "--metal-aq") {
      if (index + 1 >= argc ||
          !ParseMetalAqMode(argv[++index], &candidate.metal_aq_mode)) {
        return false;
      }
    } else if (!argument.empty() && argument.front() == '-') {
      return false;
    } else if (candidate.input.empty()) {
      candidate.input = argv[index];
    } else if (candidate.output.empty()) {
      candidate.output = argv[index];
    } else {
      return false;
    }
  }
  if (candidate.input.empty() || candidate.output.empty() ||
      !rate_control_set ||
      (target_search_option_set &&
       candidate.rate_control_mode ==
         gjxl::VarDctRateControlMode::kButteraugliTarget) ||
      (candidate.metal_aq_mode ==
         gjxl::GpuAdaptiveQuantizationMode::kFullyResident &&
       candidate.backend != gjxl::VarDctBackendPreference::kMetal)) {
    return false;
  }
  *options = std::move(candidate);
  return true;
}

[[nodiscard]] gjxl::Status WriteAtomically(
  const fs::path& destination,
  std::span<const uint8_t> bytes) {

  if (destination.empty() || bytes.empty()) {
    return gjxl::Status::InvalidArgument(
      "Output path or codestream is empty");
  }

  fs::path temporary;
  int descriptor = -1;
  for (size_t attempt = 0; attempt < 100; ++attempt) {
    temporary = destination;
    temporary += ".tmp." + std::to_string(getpid()) + "." +
      std::to_string(attempt);
    descriptor = open(
      temporary.c_str(),
      O_WRONLY | O_CREAT | O_EXCL,
      S_IRUSR | S_IWUSR);
    if (descriptor >= 0) {
      break;
    }
    if (errno != EEXIST) {
      return gjxl::Status::Internal(
        "Unable to create temporary output: " +
        std::string(std::strerror(errno)));
    }
  }
  if (descriptor < 0) {
    return gjxl::Status::Internal(
      "Unable to reserve a temporary output filename");
  }

  const auto fail = [&](std::string message) {
    const int saved_errno = errno;
    close(descriptor);
    std::error_code ignored;
    fs::remove(temporary, ignored);
    if (message.empty()) {
      message = std::strerror(saved_errno);
    }
    return gjxl::Status::Internal(std::move(message));
  };

  size_t offset = 0;
  while (offset < bytes.size()) {
    const size_t remaining = bytes.size() - offset;
    const size_t request = std::min(
      remaining,
      static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t written = write(
      descriptor,
      bytes.data() + offset,
      request);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return fail(
        "Unable to write temporary output: " +
        std::string(std::strerror(errno)));
    }
    if (written == 0) {
      return fail("Temporary output write made no progress");
    }
    offset += static_cast<size_t>(written);
  }
  if (fsync(descriptor) != 0) {
    return fail(
      "Unable to synchronize temporary output: " +
      std::string(std::strerror(errno)));
  }
  if (close(descriptor) != 0) {
    descriptor = -1;
    std::error_code ignored;
    fs::remove(temporary, ignored);
    return gjxl::Status::Internal(
      "Unable to close temporary output: " +
      std::string(std::strerror(errno)));
  }
  descriptor = -1;

  if (std::rename(temporary.c_str(), destination.c_str()) != 0) {
    const std::string message = std::strerror(errno);
    std::error_code ignored;
    fs::remove(temporary, ignored);
    return gjxl::Status::Internal(
      "Unable to commit output atomically: " + message);
  }
  return gjxl::Status::Ok();
}

void PrintUsage(const char* executable) {
  std::cerr << "Usage: " << executable
            << " (--distance VALUE | --target-bytes BYTES | "
               "--target-bpp BPP) [--size-tolerance FRACTION] "
               "[--max-attempts N] [--backend auto|cpu|metal] "
               "[--metal-aq exact-coefficients|fully-resident] "
               "INPUT.pfm OUTPUT.jxl\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  gjxl::Image3FBuffer linear_rgb;
  gjxl::Status status = gjxl::io::ReadPfm(options.input, &linear_rgb);
  if (!status.ok()) {
    std::cerr << "Input error: " << status.message() << '\n';
    return EXIT_FAILURE;
  }

  std::vector<uint8_t> codestream;
  gjxl::VarDctEncodingSummary summary;
  status = gjxl::EncodeLinearRgbVarDctCodestream(
    linear_rgb.const_view(),
    {.butteraugli_target = options.butteraugli_target,
     .rate_control_mode = options.rate_control_mode,
     .target_bytes = options.target_bytes,
     .target_bits_per_pixel = options.target_bits_per_pixel,
     .target_size_tolerance = options.target_size_tolerance,
     .target_size_maximum_attempts =
       options.target_size_maximum_attempts,
     .backend = options.backend,
     .metal_aq_mode = options.metal_aq_mode},
    &codestream,
    &summary);
  if (!status.ok()) {
    std::cerr << "Encoding error: " << status.message() << '\n';
    return EXIT_FAILURE;
  }
  status = WriteAtomically(options.output, codestream);
  if (!status.ok()) {
    std::cerr << "Output error: " << status.message() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "Encoded " << summary.extent.width << 'x'
            << summary.extent.height << " linear RGB to "
            << summary.encoded_bytes << " bytes";
  switch (summary.rate_control_mode) {
    case gjxl::VarDctRateControlMode::kButteraugliTarget:
      std::cout << " at Butteraugli target "
                << summary.selected_butteraugli_target;
      break;
    case gjxl::VarDctRateControlMode::kTargetBytes:
      std::cout << " for target " << summary.effective_target_bytes
                << " bytes (" << (summary.target_size_met ? "met" : "unmet")
                << "; selected Butteraugli target "
                << summary.selected_butteraugli_target << " in "
                << summary.encode_attempt_count << " attempts)";
      break;
    case gjxl::VarDctRateControlMode::kTargetBitsPerPixel:
      std::cout << " for target " << options.target_bits_per_pixel
                << " bpp / " << summary.effective_target_bytes << " bytes ("
                << (summary.target_size_met ? "met" : "unmet")
                << "; selected Butteraugli target "
                << summary.selected_butteraugli_target << " in "
                << summary.encode_attempt_count << " attempts)";
      break;
    case gjxl::VarDctRateControlMode::kMaximumError:
      std::cout << " under maximum-error control";
      break;
  }
  std::cout << " using "
            << (summary.execution_backend ==
                    gjxl::VarDctExecutionBackend::kMetal
                  ? (summary.metal_aq_mode ==
                           gjxl::GpuAdaptiveQuantizationMode::kFullyResident
                       ? "Metal fully-resident AQ"
                       : "Metal exact-coefficient AQ")
                  : "CPU")
            << ".\nStrategies:";
  for (size_t index = 0; index < summary.strategy_counts.size(); ++index) {
    if (summary.strategy_counts[index] == 0) {
      continue;
    }
    std::cout << ' ' << gjxl::kAcStrategyInfos[index].name << '='
              << summary.strategy_counts[index];
  }
  if (!summary.score_history.empty()) {
    std::cout << "\nFinal perceptual score: "
              << summary.score_history.back();
  }
  std::cout << '\n';
  return EXIT_SUCCESS;
}
