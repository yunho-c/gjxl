// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "core/image_buffer.h"
#include "io/pfm.h"

namespace {

namespace fs = std::filesystem;

class TempFile {
public:
  TempFile() {
    static std::atomic<uint64_t> counter{0};
    const uint64_t stamp = static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
    path_ = fs::temp_directory_path() /
      ("gjxl-pfm-test-" + std::to_string(stamp) + "-" +
       std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) +
       ".pfm");
  }

  ~TempFile() {
    std::error_code error;
    fs::remove(path_, error);
  }

  [[nodiscard]] const fs::path& path() const noexcept {
    return path_;
  }

private:
  fs::path path_;
};

struct TestImage {
  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> plane;
};

TestImage Pattern(gjxl::Extent2D extent) {
  TestImage image{.extent = extent};
  const size_t pixel_count = extent.width * extent.height;
  for (size_t channel = 0; channel < 3; ++channel) {
    image.plane[channel].resize(pixel_count);
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        const size_t index = y * extent.width + x;
        const float magnitude = static_cast<float>(
          channel * 128 + (17 * x + 31 * y) % 127) + 0.25f;
        image.plane[channel][index] =
          ((x + 3 * y + channel) % 5 == 0) ? -magnitude : magnitude;
      }
    }
  }
  return image;
}

bool WritePfm(const fs::path& path, const TestImage& image,
              bool little_endian, float scale_magnitude,
              bool comments_and_crlf = false) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    return false;
  const char* newline = comments_and_crlf ? "\r\n" : "\n";
  stream << "PF" << newline;
  if (comments_and_crlf) {
    stream << "# generated test image" << newline;
  }
  stream << image.extent.width << ' ' << image.extent.height << newline;
  if (comments_and_crlf) {
    stream << "# endian and scale" << newline;
  }
  stream << (little_endian ? -scale_magnitude : scale_magnitude) << newline;
  for (size_t reverse_y = 0; reverse_y < image.extent.height; ++reverse_y) {
    const size_t y = image.extent.height - 1 - reverse_y;
    for (size_t x = 0; x < image.extent.width; ++x) {
      for (size_t channel = 0; channel < 3; ++channel) {
        const float stored =
          image.plane[channel][y * image.extent.width + x] / scale_magnitude;
        const uint32_t bits = std::bit_cast<uint32_t>(stored);
        std::array<char, 4> bytes{};
        for (size_t byte = 0; byte < bytes.size(); ++byte) {
          const size_t shift_byte = little_endian ? byte : 3 - byte;
          bytes[byte] = static_cast<char>((bits >> (8 * shift_byte)) & 0xffu);
        }
        stream.write(bytes.data(), bytes.size());
      }
    }
  }
  return stream.good();
}

bool Equal(const TestImage& expected, const gjxl::Image3FBuffer& actual) {
  if (actual.extent() != expected.extent)
    return false;
  const gjxl::ConstImage3FView view = actual.const_view();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < expected.extent.height; ++y) {
      for (size_t x = 0; x < expected.extent.width; ++x) {
        const size_t index = y * expected.extent.width + x;
        if (std::bit_cast<uint32_t>(view.plane[channel].Row(y)[x]) !=
            std::bit_cast<uint32_t>(expected.plane[channel][index])) {
          return false;
        }
      }
    }
  }
  return true;
}

bool Equal(const gjxl::Image3FBuffer& left,
           const gjxl::Image3FBuffer& right) {
  if (left.extent() != right.extent())
    return false;
  const gjxl::ConstImage3FView left_view = left.const_view();
  const gjxl::ConstImage3FView right_view = right.const_view();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < left.extent().height; ++y) {
      for (size_t x = 0; x < left.extent().width; ++x) {
        if (std::bit_cast<uint32_t>(left_view.plane[channel].Row(y)[x]) !=
            std::bit_cast<uint32_t>(right_view.plane[channel].Row(y)[x])) {
          return false;
        }
      }
    }
  }
  return true;
}

gjxl::Image3FBuffer Sentinel() {
  gjxl::Image3FBuffer image({2, 2});
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t index = 0; index < image.plane(channel).size(); ++index) {
      image.plane(channel)[index] =
        static_cast<float>(100 * channel + index) + 0.5f;
    }
  }
  return image;
}

bool ExpectFailure(const fs::path& path, gjxl::Image3FBuffer* output) {
  const gjxl::Image3FBuffer before = *output;
  const gjxl::Status status = gjxl::io::ReadPfm(path, output);
  return status.code() == gjxl::StatusCode::kInvalidArgument &&
         Equal(before, *output);
}

bool CheckValid(gjxl::Extent2D extent, bool little_endian,
                float scale_magnitude, bool comments_and_crlf) {
  const TestImage expected = Pattern(extent);
  TempFile file;
  gjxl::Image3FBuffer actual;
  if (!WritePfm(file.path(), expected, little_endian, scale_magnitude,
                comments_and_crlf)) {
    std::cerr << "Unable to write valid PFM fixture\n";
    return false;
  }
  const gjxl::Status status = gjxl::io::ReadPfm(file.path(), &actual);
  if (!status.ok() || !Equal(expected, actual)) {
    std::cerr << "Valid PFM did not round trip exactly: "
              << status.message() << '\n';
    return false;
  }
  return true;
}

bool WriteText(const fs::path& path, const std::string& text) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
  return stream.good();
}

bool CheckMalformedHeaders() {
  const std::array<std::string, 6> invalid = {{
      "Pf\n2 2\n-1\n",
      "PF\n0 2\n-1\n",
      "PF\n2 2 trailing\n-1\n",
      "PF\n2 2\n0\n",
      "PF\n2 2\nnan\n",
      "PF\n18446744073709551615 2\n-1\n",
  }};
  for (const std::string& contents : invalid) {
    TempFile file;
    gjxl::Image3FBuffer output = Sentinel();
    if (!WriteText(file.path(), contents) ||
        !ExpectFailure(file.path(), &output)) {
      std::cerr << "Malformed PFM header was accepted or changed output\n";
      return false;
    }
  }
  return true;
}

bool CheckMalformedPayloads() {
  const TestImage valid = Pattern({9, 7});
  {
    TempFile file;
    gjxl::Image3FBuffer output = Sentinel();
    if (!WritePfm(file.path(), valid, true, 1.0f))
      return false;
    std::error_code error;
    const uintmax_t size = fs::file_size(file.path(), error);
    if (error || size == 0) return false;
    fs::resize_file(file.path(), size - 1, error);
    if (error || !ExpectFailure(file.path(), &output)) {
      std::cerr << "Truncated PFM payload was accepted or changed output\n";
      return false;
    }
  }
  {
    TempFile file;
    gjxl::Image3FBuffer output = Sentinel();
    if (!WritePfm(file.path(), valid, false, 1.0f))
      return false;
    std::ofstream stream(file.path(), std::ios::binary | std::ios::app);
    stream.put('x');
    stream.close();
    if (!ExpectFailure(file.path(), &output)) {
      std::cerr << "Trailing PFM payload was accepted or changed output\n";
      return false;
    }
  }
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    TempFile file;
    TestImage image = valid;
    image.plane[1][17] = invalid;
    gjxl::Image3FBuffer output = Sentinel();
    if (!WritePfm(file.path(), image, true, 1.0f) ||
        !ExpectFailure(file.path(), &output)) {
      std::cerr << "Non-finite PFM payload was accepted or changed output\n";
      return false;
    }
  }
  return true;
}

bool CheckArgumentFailures() {
  gjxl::Image3FBuffer output = Sentinel();
  const gjxl::Image3FBuffer before = output;
  if (gjxl::io::ReadPfm({}, &output).code() !=
          gjxl::StatusCode::kInvalidArgument ||
      !Equal(before, output)) {
    return false;
  }
  TempFile missing;
  if (!ExpectFailure(missing.path(), &output) ||
      gjxl::io::ReadPfm(missing.path(), nullptr).code() !=
          gjxl::StatusCode::kInvalidArgument) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckValid({7, 5}, true, 1.0f, true) ||
      !CheckValid({13, 7}, false, 1.0f, false) ||
      !CheckValid({11, 6}, false, 2.0f, false) ||
      !CheckValid({512, 257}, true, 0.5f, false) ||
      !CheckMalformedHeaders() || !CheckMalformedPayloads() ||
      !CheckArgumentFailures()) {
    return EXIT_FAILURE;
  }
  std::cout << "PFM loader tests passed\n";
  return EXIT_SUCCESS;
}
