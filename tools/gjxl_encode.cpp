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
  float butteraugli_target = 0.0f;
};

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

[[nodiscard]] bool ParseOptions(
  int argc,
  char** argv,
  Options* options) {

  if (options == nullptr) {
    return false;
  }
  Options candidate;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--distance") {
      if (index + 1 >= argc ||
          !ParsePositiveFloat(argv[++index],
                              &candidate.butteraugli_target)) {
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
      candidate.butteraugli_target <= 0.0f) {
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
            << " --distance VALUE INPUT.pfm OUTPUT.jxl\n";
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
    {.butteraugli_target = options.butteraugli_target},
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
            << summary.encoded_bytes << " bytes at Butteraugli target "
            << options.butteraugli_target << ".\nStrategies:";
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
