// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "codestream/workflow.h"
#include "core/ac_strategy.h"
#include "core/image_buffer.h"
#include "core/status.h"
#include "io/pfm.h"

namespace {

namespace fs = std::filesystem;

[[nodiscard]] int ProcessId() noexcept {
#if defined(_WIN32)
  return _getpid();
#else
  return getpid();
#endif
}

[[nodiscard]] int OpenExclusive(const fs::path& path) noexcept {
#if defined(_WIN32)
  return _wopen(
    path.c_str(),
    _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
    _S_IREAD | _S_IWRITE);
#else
  return open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
#endif
}

[[nodiscard]] size_t MaximumDescriptorWrite() noexcept {
#if defined(_WIN32)
  return static_cast<size_t>(std::numeric_limits<unsigned int>::max());
#else
  return static_cast<size_t>(std::numeric_limits<ssize_t>::max());
#endif
}

[[nodiscard]] std::ptrdiff_t WriteDescriptor(
  int descriptor,
  const void* data,
  size_t size) noexcept {

#if defined(_WIN32)
  return static_cast<std::ptrdiff_t>(
    _write(descriptor, data, static_cast<unsigned int>(size)));
#else
  return static_cast<std::ptrdiff_t>(write(descriptor, data, size));
#endif
}

[[nodiscard]] int SynchronizeDescriptor(int descriptor) noexcept {
#if defined(_WIN32)
  return _commit(descriptor);
#else
  return fsync(descriptor);
#endif
}

[[nodiscard]] int CloseDescriptor(int descriptor) noexcept {
#if defined(_WIN32)
  return _close(descriptor);
#else
  return close(descriptor);
#endif
}

[[nodiscard]] bool CommitTemporaryFile(
  const fs::path& temporary,
  const fs::path& destination,
  std::string* error) {

#if defined(_WIN32)
  if (MoveFileExW(
        temporary.c_str(), destination.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0) {
    return true;
  }
  const std::error_code code(
    static_cast<int>(GetLastError()), std::system_category());
  if (error != nullptr) {
    *error = code.message();
  }
  return false;
#else
  if (std::rename(temporary.c_str(), destination.c_str()) == 0) {
    return true;
  }
  if (error != nullptr) {
    *error = std::strerror(errno);
  }
  return false;
#endif
}

struct Options {
  fs::path input;
  fs::path output;
  float butteraugli_target = 1.0f;
  gjxl::VarDctRateControlMode rate_control_mode =
    gjxl::VarDctRateControlMode::kButteraugliTarget;
  std::array<float, 3> maximum_error{};
  size_t target_bytes = 0;
  double target_bits_per_pixel = 0.0;
  double target_size_tolerance = 0.005;
  size_t target_size_maximum_attempts = 12;
  gjxl::TargetSizeSelectionPolicy target_size_selection =
    gjxl::TargetSizeSelectionPolicy::kLargestAtOrBelow;
  gjxl::VarDctBackendPreference backend =
    gjxl::VarDctBackendPreference::kAutomatic;
  int32_t effort = 7;
  gjxl::VarDctDensityMode density_mode =
    gjxl::VarDctDensityMode::kDefault;
  gjxl::VarDctCompressionMode compression_mode =
    gjxl::VarDctCompressionMode::kAutomatic;
  gjxl::GpuAdaptiveQuantizationMode gpu_aq_mode =
    gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  bool collect_final_butteraugli_score = false;
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
  } else if (text == "cuda") {
    *backend = gjxl::VarDctBackendPreference::kCuda;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] bool ParseGpuAqMode(
  std::string_view text,
  gjxl::GpuAdaptiveQuantizationMode* mode) {

  if (mode == nullptr) {
    return false;
  }
  if (text == "exact-coefficients") {
    *mode = gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
  } else if (text == "fully-resident") {
    *mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  } else if (text == "throughput") {
    *mode = gjxl::GpuAdaptiveQuantizationMode::kThroughput;
  } else if (text == "maximum-throughput") {
    *mode = gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] std::string_view ExecutionBackendName(
  gjxl::VarDctExecutionBackend backend) {
  switch (backend) {
    case gjxl::VarDctExecutionBackend::kCpu:
      return "CPU";
    case gjxl::VarDctExecutionBackend::kMetal:
      return "Metal";
    case gjxl::VarDctExecutionBackend::kCuda:
      return "CUDA";
  }
  return "unknown backend";
}

[[nodiscard]] std::string_view GpuAqModeName(
  gjxl::GpuAdaptiveQuantizationMode mode) {
  switch (mode) {
    case gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients:
      return "exact-coefficient AQ";
    case gjxl::GpuAdaptiveQuantizationMode::kFullyResident:
      return "fully-resident AQ";
    case gjxl::GpuAdaptiveQuantizationMode::kThroughput:
      return "throughput AQ";
    case gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput:
      return "maximum-throughput AQ";
  }
  return "unknown AQ mode";
}

[[nodiscard]] bool ParseTargetSizeSelection(
  std::string_view text,
  gjxl::TargetSizeSelectionPolicy* selection) {

  if (selection == nullptr) {
    return false;
  }
  if (text == "under-budget") {
    *selection =
      gjxl::TargetSizeSelectionPolicy::kLargestAtOrBelow;
  } else if (text == "closest") {
    *selection = gjxl::TargetSizeSelectionPolicy::kClosestAbsolute;
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
  bool effort_set = false;
  bool high_density_set = false;
  bool maximum_compression_set = false;
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
    } else if (argument == "--maximum-error") {
      if (rate_control_set || index + 3 >= argc ||
          !ParsePositiveFloat(
            argv[++index], &candidate.maximum_error[0]) ||
          !ParsePositiveFloat(
            argv[++index], &candidate.maximum_error[1]) ||
          !ParsePositiveFloat(
            argv[++index], &candidate.maximum_error[2])) {
        return false;
      }
      candidate.rate_control_mode =
        gjxl::VarDctRateControlMode::kMaximumError;
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
    } else if (argument == "--size-selection") {
      if (index + 1 >= argc ||
          !ParseTargetSizeSelection(
            argv[++index], &candidate.target_size_selection)) {
        return false;
      }
      target_search_option_set = true;
    } else if (argument == "--backend") {
      if (index + 1 >= argc ||
          !ParseBackend(argv[++index], &candidate.backend)) {
        return false;
      }
    } else if (argument == "--effort") {
      size_t effort = 0;
      if (effort_set || index + 1 >= argc ||
          !ParsePositiveSize(argv[++index], 10, &effort)) {
        return false;
      }
      candidate.effort = static_cast<int32_t>(effort);
      effort_set = true;
    } else if (argument == "--high-density") {
      if (high_density_set) {
        return false;
      }
      candidate.density_mode = gjxl::VarDctDensityMode::kHighDensity;
      high_density_set = true;
    } else if (argument == "--maximum-compression") {
      if (maximum_compression_set) {
        return false;
      }
      candidate.compression_mode =
        gjxl::VarDctCompressionMode::kMaximumCompression;
      maximum_compression_set = true;
    } else if (argument == "--gpu-aq" || argument == "--metal-aq") {
      if (index + 1 >= argc ||
          !ParseGpuAqMode(argv[++index], &candidate.gpu_aq_mode)) {
        return false;
      }
    } else if (argument == "--collect-final-score") {
      candidate.collect_final_butteraugli_score = true;
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
      (effort_set && high_density_set) ||
      (target_search_option_set &&
       candidate.rate_control_mode ==
         gjxl::VarDctRateControlMode::kButteraugliTarget) ||
      (candidate.gpu_aq_mode ==
         gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput &&
       candidate.rate_control_mode ==
         gjxl::VarDctRateControlMode::kMaximumError) ||
      (candidate.density_mode == gjxl::VarDctDensityMode::kHighDensity &&
       (candidate.rate_control_mode ==
          gjxl::VarDctRateControlMode::kMaximumError ||
        candidate.gpu_aq_mode ==
          gjxl::GpuAdaptiveQuantizationMode::kThroughput ||
        candidate.gpu_aq_mode ==
          gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput)) ||
      ((candidate.gpu_aq_mode ==
          gjxl::GpuAdaptiveQuantizationMode::kThroughput ||
        candidate.gpu_aq_mode ==
          gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput) &&
       candidate.backend != gjxl::VarDctBackendPreference::kMetal &&
       candidate.backend != gjxl::VarDctBackendPreference::kCuda)) {
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
    temporary += ".tmp." + std::to_string(ProcessId()) + "." +
      std::to_string(attempt);
    descriptor = OpenExclusive(temporary);
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
    (void)CloseDescriptor(descriptor);
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
      MaximumDescriptorWrite());
    const std::ptrdiff_t written = WriteDescriptor(
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
  if (SynchronizeDescriptor(descriptor) != 0) {
    return fail(
      "Unable to synchronize temporary output: " +
      std::string(std::strerror(errno)));
  }
  if (CloseDescriptor(descriptor) != 0) {
    descriptor = -1;
    std::error_code ignored;
    fs::remove(temporary, ignored);
    return gjxl::Status::Internal(
      "Unable to close temporary output: " +
      std::string(std::strerror(errno)));
  }
  descriptor = -1;

  std::string commit_error;
  if (!CommitTemporaryFile(temporary, destination, &commit_error)) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    return gjxl::Status::Internal(
      "Unable to commit output atomically: " + commit_error);
  }
  return gjxl::Status::Ok();
}

void PrintUsage(const char* executable) {
  std::cerr << "Usage: " << executable
            << " (--distance VALUE | --maximum-error X Y B | "
               "--target-bytes BYTES | "
               "--target-bpp BPP) [--size-tolerance FRACTION] "
               "[--max-attempts N] "
               "[--size-selection under-budget|closest] "
               "[--effort 1..10] "
               "[--high-density] "
               "[--maximum-compression] "
               "[--backend auto|cpu|metal|cuda] "
               "[--gpu-aq exact-coefficients|fully-resident|throughput|"
               "maximum-throughput] "
               "[--collect-final-score] "
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
  gjxl::VarDctEncodingTiming timing;
  status = gjxl::EncodeLinearRgbVarDctCodestreamProfiled(
    linear_rgb.const_view(),
    {.butteraugli_target = options.butteraugli_target,
     .effort = options.effort,
     .density_mode = options.density_mode,
     .compression_mode = options.compression_mode,
     .rate_control_mode = options.rate_control_mode,
     .maximum_error = options.maximum_error,
     .target_bytes = options.target_bytes,
     .target_bits_per_pixel = options.target_bits_per_pixel,
     .target_size_tolerance = options.target_size_tolerance,
     .target_size_maximum_attempts =
       options.target_size_maximum_attempts,
     .target_size_selection = options.target_size_selection,
     .backend = options.backend,
     .gpu_aq_mode = options.gpu_aq_mode,
     .collect_final_butteraugli_score =
       options.collect_final_butteraugli_score},
    &codestream,
    &summary,
    &timing);
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
                << summary.encode_attempt_count << " attempts, "
                << summary.failed_encode_attempt_count << " failed)";
      break;
    case gjxl::VarDctRateControlMode::kTargetBitsPerPixel:
      std::cout << " for target " << options.target_bits_per_pixel
                << " bpp / " << summary.effective_target_bytes << " bytes ("
                << (summary.target_size_met ? "met" : "unmet")
                << "; selected Butteraugli target "
                << summary.selected_butteraugli_target << " in "
                << summary.encode_attempt_count << " attempts, "
                << summary.failed_encode_attempt_count << " failed)";
      break;
    case gjxl::VarDctRateControlMode::kMaximumError:
      std::cout << " under maximum-error control ("
                << summary.requested_maximum_error[0] << ','
                << summary.requested_maximum_error[1] << ','
                << summary.requested_maximum_error[2] << "; achieved "
                << summary.achieved_maximum_error[0] << ','
                << summary.achieved_maximum_error[1] << ','
                << summary.achieved_maximum_error[2] << "; "
                << (summary.maximum_error_outcome ==
                          gjxl::MaximumErrorOutcome::kMet
                      ? "met"
                      : "unmet")
                << " in " << summary.maximum_error_evaluation_count
                << " evaluations)";
      break;
  }
  std::cout << " using " << ExecutionBackendName(summary.execution_backend);
  if (summary.execution_backend != gjxl::VarDctExecutionBackend::kCpu) {
    std::cout << ' ' << GpuAqModeName(summary.gpu_aq_mode);
  }
  std::cout
            << (summary.density_mode ==
                      gjxl::VarDctDensityMode::kHighDensity
                  ? " with high-density AQ"
                  : "")
            << " and "
            << (summary.entropy_behavior ==
                      gjxl::VarDctEntropyBehavior::kMaximumCompression
                  ? "maximum-compression entropy"
                  : (summary.entropy_behavior ==
                           gjxl::VarDctEntropyBehavior::kHighDensity
                       ? "high-density entropy"
                       : "balanced entropy"))
            << ".\nStrategies:";
  for (size_t index = 0; index < summary.strategy_counts.size(); ++index) {
    if (summary.strategy_counts[index] == 0) {
      continue;
    }
    std::cout << ' ' << gjxl::kAcStrategyInfos[index].name << '='
              << summary.strategy_counts[index];
  }
  if (summary.final_butteraugli_score_evaluated &&
      !summary.score_history.empty() &&
      summary.rate_control_mode !=
        gjxl::VarDctRateControlMode::kMaximumError) {
    std::cout << "\nFinal perceptual score: "
              << summary.score_history.back();
  } else if (!summary.score_history.empty() &&
             summary.rate_control_mode !=
               gjxl::VarDctRateControlMode::kMaximumError) {
    std::cout << "\nFinal perceptual score: not evaluated";
  }
  constexpr double kNanosecondsPerMillisecond = 1.0e6;
  std::cout << "\nTiming: preparation="
            << static_cast<double>(timing.preparation_nanoseconds) /
                 kNanosecondsPerMillisecond
            << " ms, selected-attempt="
            << static_cast<double>(timing.selected_attempt_nanoseconds) /
                 kNanosecondsPerMillisecond
            << " ms";
  if (summary.rate_control_mode ==
        gjxl::VarDctRateControlMode::kTargetBytes ||
      summary.rate_control_mode ==
        gjxl::VarDctRateControlMode::kTargetBitsPerPixel) {
    std::cout << ", aggregate-search="
              << static_cast<double>(timing.aggregate_search_nanoseconds) /
                   kNanosecondsPerMillisecond
              << " ms";
  }
  std::cout << ", total="
            << static_cast<double>(timing.total_nanoseconds) /
                 kNanosecondsPerMillisecond
            << " ms";
  std::cout << '\n';
  return EXIT_SUCCESS;
}
