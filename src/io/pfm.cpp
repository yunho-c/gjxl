// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "io/pfm.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace gjxl::io {
namespace {

[[nodiscard]] uint32_t ByteSwap32(uint32_t value) noexcept {
  return ((value & 0x000000ffu) << 24) |
    ((value & 0x0000ff00u) << 8) |
    ((value & 0x00ff0000u) >> 8) |
    ((value & 0xff000000u) >> 24);
}

[[nodiscard]] bool ReadHeaderLine(
  std::istream& stream,
  std::string* line) {

  while (std::getline(stream, *line)) {
    if (!line->empty() && line->back() == '\r') {
      line->pop_back();
    }
    if (!line->empty() && line->front() != '#') {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool ParseDimensions(
  const std::string& line,
  Extent2D* extent) {

  std::istringstream input(line);
  size_t width = 0;
  size_t height = 0;
  std::string trailing;
  if (!(input >> width >> height) || (input >> trailing) ||
      width == 0 || height == 0) {
    return false;
  }
  *extent = {width, height};
  return true;
}

[[nodiscard]] bool ParseScale(
  const std::string& line,
  float* scale) {

  std::istringstream input(line);
  std::string trailing;
  if (!(input >> *scale) || (input >> trailing) ||
      !std::isfinite(*scale) || *scale == 0.0f) {
    return false;
  }
  return true;
}

[[nodiscard]] size_t PfmWorkerCount(
  Extent2D extent,
  size_t row_bytes) noexcept {

  constexpr size_t kMinimumParallelPixels = 256 * 256;
  constexpr size_t kMaximumWorkers = 3;
  constexpr size_t kMaximumAggregateRowScratch = 8 * 1024 * 1024;
  size_t pixel_count = 0;
  if (!extent.try_area(&pixel_count) ||
      pixel_count < kMinimumParallelPixels) {
    return 1;
  }
  const size_t hardware_workers = std::max<size_t>(
    std::thread::hardware_concurrency(), 1);
  const size_t scratch_workers = row_bytes > kMaximumAggregateRowScratch
    ? 1
    : std::max<size_t>(kMaximumAggregateRowScratch / row_bytes, 1);
  return std::min(
    extent.height,
    std::min(
      kMaximumWorkers,
      std::min(hardware_workers, scratch_workers)));
}

template <bool kSwapEndian, bool kApplyScale>
[[nodiscard]] Status ConvertPfmRowImpl(
  const std::vector<uint32_t>& row,
  size_t y,
  float multiplier,
  Image3FView output) {

  for (size_t x = 0; x < output.width(); ++x) {
    for (size_t channel = 0; channel < 3; ++channel) {
      uint32_t bits = row[3 * x + channel];
      if constexpr (kSwapEndian) {
        bits = ByteSwap32(bits);
      }
      float value = std::bit_cast<float>(bits);
      if constexpr (kApplyScale) {
        value *= multiplier;
      }
      if (!std::isfinite(value)) {
        return Status::InvalidArgument(
          "PFM input contains a non-finite pixel");
      }
      output.plane[channel].Row(y)[x] = value;
    }
  }
  return Status::Ok();
}

[[nodiscard]] Status ConvertPfmRow(
  const std::vector<uint32_t>& row,
  size_t y,
  bool swap_endian,
  bool apply_scale,
  float multiplier,
  Image3FView output) {

  if (swap_endian) {
    return apply_scale
      ? ConvertPfmRowImpl<true, true>(row, y, multiplier, output)
      : ConvertPfmRowImpl<true, false>(row, y, multiplier, output);
  }
  return apply_scale
    ? ConvertPfmRowImpl<false, true>(row, y, multiplier, output)
    : ConvertPfmRowImpl<false, false>(row, y, multiplier, output);
}

[[nodiscard]] Status ReadPfmPayload(
  std::istream* stream,
  Extent2D extent,
  bool swap_endian,
  float multiplier,
  Image3FView output) {

  const size_t row_value_count = 3 * extent.width;
  const size_t row_bytes = row_value_count * sizeof(uint32_t);
  const bool apply_scale = multiplier != 1.0f;
  const size_t worker_count = !swap_endian && !apply_scale
    ? 1
    : PfmWorkerCount(extent, row_bytes);
  if (worker_count == 1) {
    std::vector<uint32_t> row(row_value_count);
    for (size_t reverse_y = 0; reverse_y < extent.height; ++reverse_y) {
      stream->read(
        reinterpret_cast<char*>(row.data()),
        static_cast<std::streamsize>(row_bytes));
      if (stream->gcount() != static_cast<std::streamsize>(row_bytes)) {
        return Status::InvalidArgument("PFM pixel payload is truncated");
      }
      const size_t y = extent.height - 1 - reverse_y;
      Status status = ConvertPfmRow(
        row, y, swap_endian, apply_scale, multiplier, output);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }
  std::mutex read_mutex;
  size_t next_reverse_y = 0;
  std::atomic<bool> cancelled{false};
  std::vector<Status> statuses(worker_count);

  const auto run_worker = [&](size_t worker_index) {
    try {
      std::vector<uint32_t> row(row_value_count);
      while (!cancelled.load(std::memory_order_relaxed)) {
        size_t reverse_y = 0;
        {
          std::lock_guard lock(read_mutex);
          if (cancelled.load(std::memory_order_relaxed) ||
              next_reverse_y >= extent.height) {
            break;
          }
          reverse_y = next_reverse_y++;
          stream->read(
            reinterpret_cast<char*>(row.data()),
            static_cast<std::streamsize>(row_bytes));
          if (stream->gcount() != static_cast<std::streamsize>(row_bytes)) {
            statuses[worker_index] = Status::InvalidArgument(
              "PFM pixel payload is truncated");
            cancelled.store(true, std::memory_order_relaxed);
            break;
          }
        }
        const size_t y = extent.height - 1 - reverse_y;
        Status status = ConvertPfmRow(
          row, y, swap_endian, apply_scale, multiplier, output);
        if (!status.ok()) {
          statuses[worker_index] = std::move(status);
          cancelled.store(true, std::memory_order_relaxed);
          break;
        }
      }
    } catch (const std::bad_alloc&) {
      statuses[worker_index] = Status::OutOfMemory(
        "Unable to allocate PFM row storage");
      cancelled.store(true, std::memory_order_relaxed);
    } catch (const std::length_error&) {
      statuses[worker_index] = Status::InvalidArgument(
        "PFM row dimensions are too large");
      cancelled.store(true, std::memory_order_relaxed);
    } catch (...) {
      statuses[worker_index] = Status::Internal(
        "PFM row worker failed unexpectedly");
      cancelled.store(true, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(worker_count - 1);
  size_t launched_workers = 0;
  try {
    for (; launched_workers + 1 < worker_count; ++launched_workers) {
      workers.emplace_back(run_worker, launched_workers);
    }
  } catch (const std::system_error&) {
    // Continue with the workers already launched and the caller thread.
  }
  run_worker(launched_workers);
  for (std::thread& worker : workers) {
    worker.join();
  }
  for (const Status& status : statuses) {
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace

Status ReadPfm(
  const std::filesystem::path& path,
  Image3FBuffer* image) {

  if (image == nullptr || path.empty()) {
    return Status::InvalidArgument("PFM path or output is invalid");
  }

  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return Status::InvalidArgument(
      "Unable to open PFM input: " + path.string());
  }

  try {
    std::string line;
    if (!ReadHeaderLine(stream, &line) || line != "PF") {
      return Status::InvalidArgument(
        "PFM input must contain three floating-point channels");
    }
    Extent2D extent;
    if (!ReadHeaderLine(stream, &line) || !ParseDimensions(line, &extent)) {
      return Status::InvalidArgument("PFM dimensions are invalid");
    }
    float scale = 0.0f;
    if (!ReadHeaderLine(stream, &line) || !ParseScale(line, &scale)) {
      return Status::InvalidArgument("PFM scale is invalid");
    }

    size_t pixel_count = 0;
    if (!extent.try_area(&pixel_count) ||
        pixel_count > std::numeric_limits<size_t>::max() / 12) {
      return Status::InvalidArgument("PFM dimensions are too large");
    }
    const size_t byte_count = pixel_count * 12;
    if (byte_count > static_cast<size_t>(
          std::numeric_limits<std::streamsize>::max())) {
      return Status::InvalidArgument("PFM pixel payload is too large");
    }
    const bool file_is_little_endian = scale < 0.0f;
    const bool host_is_little_endian =
      std::endian::native == std::endian::little;
    const float multiplier = std::abs(scale);
    Image3FBuffer candidate(extent);
    Image3FView output = candidate.view();
    Status status = ReadPfmPayload(
      &stream, extent, file_is_little_endian != host_is_little_endian,
      multiplier, output);
    if (!status.ok()) return status;
    char trailing = 0;
    if (stream.get(trailing)) {
      return Status::InvalidArgument("PFM input has trailing bytes");
    }
    *image = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Unable to allocate PFM input storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("PFM dimensions are too large");
  }
  return Status::Ok();
}

}  // namespace gjxl::io
