// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace gjxl::benchmark {

// Reserve a sibling directory exclusively so concurrent writers cannot share
// temporary files. The final rename stays on the destination filesystem.
class AtomicProfileOutput {
 public:
  explicit AtomicProfileOutput(const std::filesystem::path& destination)
      : destination_(destination) {
    if (!destination.parent_path().empty()) {
      std::filesystem::create_directories(destination.parent_path());
    }
    static std::atomic<uint64_t> sequence{0};
    for (size_t attempt = 0; attempt < 64; ++attempt) {
      directory_ = destination;
      directory_ += ".tmp-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
        std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
      temporary_ = directory_ / "profile.json";
      if (std::filesystem::create_directory(directory_)) return;
    }
    throw std::runtime_error("Could not reserve temporary profile output");
  }
  AtomicProfileOutput(const AtomicProfileOutput&) = delete;
  AtomicProfileOutput& operator=(const AtomicProfileOutput&) = delete;
  ~AtomicProfileOutput() {
    std::error_code ignored;
    std::filesystem::remove(temporary_, ignored);
    std::filesystem::remove(directory_, ignored);
  }
  const std::filesystem::path& temporary() const noexcept { return temporary_; }
  void Commit() {
#if defined(_WIN32)
    if (!MoveFileExW(temporary_.c_str(), destination_.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
      throw std::system_error(static_cast<int>(GetLastError()),
                              std::system_category(),
                              "Could not atomically replace profile output");
    }
#else
    std::filesystem::rename(temporary_, destination_);
#endif
  }
 private:
  std::filesystem::path destination_;
  std::filesystem::path directory_;
  std::filesystem::path temporary_;
};

[[nodiscard]] inline std::string JsonEscape(std::string_view value) {
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

// The callback writes one complete JSON document. Publish only after it and
// the stream close succeed; exceptions leave the previous destination intact.
template <typename Writer>
inline void WriteProfileJsonFile(const std::filesystem::path& destination,
                                 Writer&& write) {
  AtomicProfileOutput file(destination);
  std::ofstream output;
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output.imbue(std::locale::classic());
  output.open(file.temporary(), std::ios::out | std::ios::trunc);
  write(output);
  output.close();
  file.Commit();
}

}  // namespace gjxl::benchmark
