// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/color_transform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#include "core/image_buffer.h"
#include "core/image_ops.h"

namespace gjxl {
namespace {

constexpr float kOpsinBias = 0.0037930732552754493f;
constexpr std::array<std::array<float, 3>, 3> kOpsinMatrix = {{
  {{0.30f, 0.622f, 0.078f}},
  {{0.23f, 0.692f, 0.078f}},
  {{0.24342268924547819f, 0.20476744424496821f,
    0.55180986650955360f}},
}};
constexpr std::array<std::array<float, 3>, 3> kInverseOpsinMatrix = {{
  {{11.031566901960783f, -9.866943921568629f,
    -0.16462299647058826f}},
  {{-3.254147380392157f, 4.418770392156863f,
    -0.16462299647058826f}},
  {{-3.6588512862745097f, 2.7129230470588235f,
    1.9459282392156863f}},
}};

template <typename Function>
Status RunParallelRows(
  Extent2D extent,
  Function&& function) {

  constexpr size_t kMinimumParallelPixels = 256 * 256;
  constexpr size_t kMaximumWorkers = 12;
  size_t pixel_count = 0;
  if (!extent.try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "Color-transform image dimensions are too large");
  }
  const size_t hardware_workers = std::max<size_t>(
    std::thread::hardware_concurrency(), 1);
  const size_t worker_count = pixel_count < kMinimumParallelPixels
    ? 1
    : std::min(extent.height, std::min(kMaximumWorkers, hardware_workers));
  if (worker_count == 1) {
    for (size_t y = 0; y < extent.height; ++y) {
      Status status = function(y);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  std::vector<Status> statuses(extent.height);
  std::atomic<size_t> next_row{0};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  try {
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&] {
        while (true) {
          const size_t y =
            next_row.fetch_add(1, std::memory_order_relaxed);
          if (y >= extent.height) break;
          try {
            statuses[y] = function(y);
          } catch (...) {
            statuses[y] = Status::Internal(
              "Color-transform worker failed unexpectedly");
          }
        }
      });
    }
  } catch (const std::system_error&) {
    next_row.store(extent.height, std::memory_order_relaxed);
    for (std::thread& worker : workers) worker.join();
    for (size_t y = 0; y < extent.height; ++y) {
      Status status = function(y);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }
  for (std::thread& worker : workers) worker.join();
  for (const Status& status : statuses) {
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

template <typename Convert>
Status ConvertImage(
  ConstImage3FView input,
  float intensity_target,
  Image3FView output,
  Convert&& convert) {

  if (!input.valid() ||
      !output.valid() ||
      input.extent() != output.extent() ||
      !std::isfinite(intensity_target) ||
      intensity_target <= 0.0f) {
    return Status::InvalidArgument(
      "Color-transform images or intensity target are invalid");
  }

  try {
    Image3FBuffer result(input.extent());
    const Image3FView result_view = result.view();
    Status status = RunParallelRows(input.extent(), [&](size_t y) {
      for (size_t x = 0; x < input.width(); ++x) {
        const std::array<float, 3> value = {
          input.plane[0].Row(y)[x],
          input.plane[1].Row(y)[x],
          input.plane[2].Row(y)[x],
        };
        if (!std::ranges::all_of(
              value,
              [](float sample) { return std::isfinite(sample); })) {
          return Status::InvalidArgument(
            "Color-transform input pixels must be finite");
        }
        const std::array<float, 3> converted = convert(
          value,
          intensity_target);
        if (!std::ranges::all_of(
              converted,
              [](float sample) { return std::isfinite(sample); })) {
          return Status::InvalidArgument(
            "Color transform produced non-finite pixels");
        }
        for (size_t channel = 0; channel < 3; ++channel) {
          result_view.plane[channel].Row(y)[x] = converted[channel];
        }
      }
      return Status::Ok();
    });
    if (!status.ok()) return status;
    CopyImage(result.const_view(), output);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate color-transform scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Color-transform image dimensions are too large");
  }
  return Status::Ok();
}

}  // namespace

Status LinearRgbToOpsin(
  ConstImage3FView linear_rgb,
  float intensity_target,
  Image3FView opsin) {

  return ConvertImage(
    linear_rgb,
    intensity_target,
    opsin,
    [](const std::array<float, 3>& rgb, float intensity) {
      std::array<float, 3> gamma{};
      const float scale = intensity / 255.0f;
      const float bias_cuberoot = std::cbrt(kOpsinBias);
      for (size_t row = 0; row < 3; ++row) {
        float mixed = std::fma(
          scale * kOpsinMatrix[row][2], rgb[2], kOpsinBias);
        mixed = std::fma(
          scale * kOpsinMatrix[row][1], rgb[1], mixed);
        mixed = std::fma(
          scale * kOpsinMatrix[row][0], rgb[0], mixed);
        gamma[row] = std::cbrt(std::max(0.0f, mixed)) - bias_cuberoot;
      }
      return std::array<float, 3>{
        0.5f * (gamma[0] - gamma[1]),
        0.5f * (gamma[0] + gamma[1]),
        gamma[2],
      };
    });
}

Status OpsinToLinearRgb(
  ConstImage3FView opsin,
  float intensity_target,
  Image3FView linear_rgb) {

  return ConvertImage(
    opsin,
    intensity_target,
    linear_rgb,
    [](const std::array<float, 3>& xyb, float intensity) {
      const float bias_cuberoot = std::cbrt(kOpsinBias);
      const std::array<float, 3> gamma = {
        xyb[1] + xyb[0] + bias_cuberoot,
        xyb[1] - xyb[0] + bias_cuberoot,
        xyb[2] + bias_cuberoot,
      };
      std::array<float, 3> mixed{};
      for (size_t channel = 0; channel < 3; ++channel) {
        mixed[channel] = gamma[channel] * gamma[channel] * gamma[channel] -
          kOpsinBias;
      }
      std::array<float, 3> rgb{};
      const float scale = 255.0f / intensity;
      for (size_t row = 0; row < 3; ++row) {
        rgb[row] = scale * kInverseOpsinMatrix[row][0] * mixed[0];
        rgb[row] = std::fma(
          scale * kInverseOpsinMatrix[row][1],
          mixed[1],
          rgb[row]);
        rgb[row] = std::fma(
          scale * kInverseOpsinMatrix[row][2],
          mixed[2],
          rgb[row]);
      }
      return rgb;
    });
}

}  // namespace gjxl
