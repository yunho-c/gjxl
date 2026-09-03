// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/color_transform.h"

#include "codec/color_transform_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "core/image_buffer.h"
#include "core/image_ops.h"
#include "core/thread_budget.h"

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

// Pinned from libjxl's CubeRootAndAdd in base/fast_math-inl.h. Inputs are
// nonnegative, and the three reciprocal-cube-root Newton steps plus final
// refinement have a documented maximum error of 6 ULP.
float FastCubeRootAndAdd(float value, float add) {
  constexpr float kOneThird = 1.0f / 3.0f;
  constexpr float kFourThirds = 4.0f / 3.0f;
  constexpr uint32_t kExponentBias = 0x54800000u;
  constexpr uint32_t kExponentMultiplier = 0x002AAAAAu;

  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t estimate_bits = bits == 0
    ? 0
    : kExponentBias - (bits >> 23) * kExponentMultiplier;
  float reciprocal = std::bit_cast<float>(estimate_bits);
  const float divided = kOneThird * value;
  for (size_t iteration = 0; iteration < 3; ++iteration) {
    const float squared = reciprocal * reciprocal;
    reciprocal = std::fma(
      -divided, squared * squared, kFourThirds * reciprocal);
  }
  float squared = reciprocal * reciprocal;
  reciprocal = std::fma(
    kOneThird,
    std::fma(-value, squared * squared, reciprocal),
    reciprocal);
  squared = reciprocal * reciprocal;
  return std::fma(squared, value, add);
}

#if defined(__ARM_NEON)
float32x4_t FastCubeRootAndAdd(float32x4_t value, float32x4_t add) {
  constexpr float kOneThird = 1.0f / 3.0f;
  constexpr float kFourThirds = 4.0f / 3.0f;
  const uint32x4_t bits = vreinterpretq_u32_f32(value);
  const uint32x4_t exponent = vshrq_n_u32(bits, 23);
  uint32x4_t estimate_bits = vsubq_u32(
    vdupq_n_u32(0x54800000u),
    vmulq_u32(exponent, vdupq_n_u32(0x002AAAAAu)));
  estimate_bits = vbslq_u32(
    vceqq_u32(bits, vdupq_n_u32(0)),
    vdupq_n_u32(0), estimate_bits);
  float32x4_t reciprocal = vreinterpretq_f32_u32(estimate_bits);
  const float32x4_t divided = vmulq_n_f32(value, kOneThird);
  for (size_t iteration = 0; iteration < 3; ++iteration) {
    const float32x4_t squared = vmulq_f32(reciprocal, reciprocal);
    reciprocal = vfmsq_f32(
      vmulq_n_f32(reciprocal, kFourThirds),
      divided, vmulq_f32(squared, squared));
  }
  float32x4_t squared = vmulq_f32(reciprocal, reciprocal);
  reciprocal = vfmaq_n_f32(
    reciprocal,
    vfmsq_f32(reciprocal, value, vmulq_f32(squared, squared)),
    kOneThird);
  squared = vmulq_f32(reciprocal, reciprocal);
  return vfmaq_f32(add, squared, value);
}

bool AllFinite(float32x4_t value) {
  const uint32x4_t exponent = vandq_u32(
    vreinterpretq_u32_f32(value), vdupq_n_u32(0x7F800000u));
  const uint32x4_t non_finite =
    vceqq_u32(exponent, vdupq_n_u32(0x7F800000u));
  return (vgetq_lane_u32(non_finite, 0) |
          vgetq_lane_u32(non_finite, 1) |
          vgetq_lane_u32(non_finite, 2) |
          vgetq_lane_u32(non_finite, 3)) == 0;
}
#endif

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
  const size_t automatic_worker_count = pixel_count < kMinimumParallelPixels
    ? 1
    : std::min(extent.height, std::min(kMaximumWorkers, hardware_workers));
  const size_t cpu_thread_count =
    thread_budget_internal::CpuThreadCount();
  auto* const participant_tracker =
    thread_budget_internal::ParticipantTracker();
  if (thread_budget_internal::InExplicitParallelScope()) {
    for (size_t y = 0; y < extent.height; ++y) {
      Status status = function(y);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }
  const size_t participant_count = cpu_thread_count == 0
    ? automatic_worker_count
    : std::min(automatic_worker_count, cpu_thread_count);
  if (participant_count == 1) {
    thread_budget_internal::ParallelScope scope(
      cpu_thread_count, participant_tracker);
    for (size_t y = 0; y < extent.height; ++y) {
      Status status = function(y);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  std::vector<Status> statuses(extent.height);
  std::atomic<size_t> next_row{0};
  std::vector<std::thread> workers;
  const size_t spawned_worker_count = cpu_thread_count == 0
    ? participant_count
    : participant_count - 1;
  workers.reserve(spawned_worker_count);
  const auto run_worker = [&] {
    thread_budget_internal::ParallelScope scope(
      cpu_thread_count, participant_tracker);
    while (true) {
      const size_t y = next_row.fetch_add(1, std::memory_order_relaxed);
      if (y >= extent.height) break;
      try {
        statuses[y] = function(y);
      } catch (...) {
        statuses[y] = Status::Internal(
          "Color-transform worker failed unexpectedly");
      }
    }
  };
  try {
    for (size_t worker = 0; worker < spawned_worker_count; ++worker) {
      workers.emplace_back(run_worker);
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
  if (cpu_thread_count != 0) run_worker();
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

Status TransformLinearRgbToOpsin(
  ConstImage3FView linear_rgb,
  float intensity_target,
  Image3FView opsin) {

  const float scale = intensity_target / 255.0f;
  const float bias_cuberoot = FastCubeRootAndAdd(kOpsinBias, 0.0f);
  return RunParallelRows(
    {opsin.width(), linear_rgb.height()}, [&](size_t y) {
      const float* input_r = linear_rgb.plane[0].Row(y);
      const float* input_g = linear_rgb.plane[1].Row(y);
      const float* input_b = linear_rgb.plane[2].Row(y);
      std::array<float*, 3> output = {
        opsin.plane[0].Row(y),
        opsin.plane[1].Row(y),
        opsin.plane[2].Row(y),
      };
      size_t x = 0;
#if defined(__ARM_NEON)
      const float32x4_t bias = vdupq_n_f32(kOpsinBias);
      const float32x4_t negative_bias_cuberoot =
        vdupq_n_f32(-bias_cuberoot);
      const float32x4_t zero = vdupq_n_f32(0.0f);
      const auto transform_vector = [&](float32x4_t red,
                                        float32x4_t green,
                                        float32x4_t blue,
                                        size_t output_x) {
        if (!AllFinite(red) || !AllFinite(green) || !AllFinite(blue)) {
          return Status::InvalidArgument(
            "Color-transform input pixels must be finite");
        }
        std::array<float32x4_t, 3> gamma{};
        for (size_t row = 0; row < 3; ++row) {
          float32x4_t value = vfmaq_n_f32(
            bias, blue, scale * kOpsinMatrix[row][2]);
          value = vfmaq_n_f32(
            value, green, scale * kOpsinMatrix[row][1]);
          value = vfmaq_n_f32(
            value, red, scale * kOpsinMatrix[row][0]);
          gamma[row] = FastCubeRootAndAdd(
            vmaxq_f32(zero, value), negative_bias_cuberoot);
        }
        const float32x4_t transformed_x =
          vmulq_n_f32(vsubq_f32(gamma[0], gamma[1]), 0.5f);
        const float32x4_t transformed_y =
          vmulq_n_f32(vaddq_f32(gamma[0], gamma[1]), 0.5f);
        if (!AllFinite(transformed_x) || !AllFinite(transformed_y) ||
            !AllFinite(gamma[2])) {
          return Status::InvalidArgument(
            "Color transform produced non-finite pixels");
        }
        vst1q_f32(output[0] + output_x, transformed_x);
        vst1q_f32(output[1] + output_x, transformed_y);
        vst1q_f32(output[2] + output_x, gamma[2]);
        return Status::Ok();
      };
      for (; x + 4 <= linear_rgb.width(); x += 4) {
        Status status = transform_vector(
          vld1q_f32(input_r + x),
          vld1q_f32(input_g + x),
          vld1q_f32(input_b + x), x);
        if (!status.ok()) return status;
      }
      // A padded destination has room to keep the source tail on the same
      // vector path used by the former pad-then-transform workflow. Replicate
      // the source edge in the inactive lanes, then overwrite padded outputs
      // below with the transformed edge value.
      if (x < linear_rgb.width() && x + 4 <= opsin.width()) {
        std::array<float, 4> tail_r{};
        std::array<float, 4> tail_g{};
        std::array<float, 4> tail_b{};
        for (size_t lane = 0; lane < 4; ++lane) {
          const size_t source_x =
            std::min(x + lane, linear_rgb.width() - 1);
          tail_r[lane] = input_r[source_x];
          tail_g[lane] = input_g[source_x];
          tail_b[lane] = input_b[source_x];
        }
        Status status = transform_vector(
          vld1q_f32(tail_r.data()),
          vld1q_f32(tail_g.data()),
          vld1q_f32(tail_b.data()), x);
        if (!status.ok()) return status;
        x += 4;
      }
#endif
      for (; x < linear_rgb.width(); ++x) {
        if (!std::isfinite(input_r[x]) || !std::isfinite(input_g[x]) ||
            !std::isfinite(input_b[x])) {
          return Status::InvalidArgument(
            "Color-transform input pixels must be finite");
        }
        std::array<float, 3> gamma{};
        for (size_t row = 0; row < 3; ++row) {
          float value = std::fma(
            scale * kOpsinMatrix[row][2], input_b[x], kOpsinBias);
          value = std::fma(
            scale * kOpsinMatrix[row][1], input_g[x], value);
          value = std::fma(
            scale * kOpsinMatrix[row][0], input_r[x], value);
          gamma[row] = FastCubeRootAndAdd(
            std::max(0.0f, value), -bias_cuberoot);
        }
        const std::array<float, 3> transformed = {
          0.5f * (gamma[0] - gamma[1]),
          0.5f * (gamma[0] + gamma[1]),
          gamma[2],
        };
        if (!std::ranges::all_of(
              transformed,
              [](float sample) { return std::isfinite(sample); })) {
          return Status::InvalidArgument(
            "Color transform produced non-finite pixels");
        }
        for (size_t channel = 0; channel < 3; ++channel) {
          output[channel][x] = transformed[channel];
        }
      }
      if (linear_rgb.width() < opsin.width()) {
        for (size_t channel = 0; channel < 3; ++channel) {
          std::fill(
            output[channel] + linear_rgb.width(),
            output[channel] + opsin.width(),
            output[channel][linear_rgb.width() - 1]);
        }
      }
      return Status::Ok();
    });
}

}  // namespace

Status LinearRgbToOpsin(
  ConstImage3FView linear_rgb,
  float intensity_target,
  Image3FView opsin) {

  if (!linear_rgb.valid() || !opsin.valid() ||
      linear_rgb.extent() != opsin.extent() ||
      !std::isfinite(intensity_target) || intensity_target <= 0.0f) {
    return Status::InvalidArgument(
      "Color-transform images or intensity target are invalid");
  }

  try {
    Image3FBuffer result(linear_rgb.extent());
    Status status = TransformLinearRgbToOpsin(
      linear_rgb, intensity_target, result.view());
    if (!status.ok()) return status;
    CopyImage(result.const_view(), opsin);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate color-transform scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Color-transform image dimensions are too large");
  }
  return Status::Ok();
}

Status color_transform_internal::LinearRgbToPaddedOpsin(
  ConstImage3FView linear_rgb,
  float intensity_target,
  Image3FView padded_opsin) {

  if (!linear_rgb.valid() || !padded_opsin.valid() ||
      linear_rgb.width() > padded_opsin.width() ||
      linear_rgb.height() > padded_opsin.height() ||
      !std::isfinite(intensity_target) || intensity_target <= 0.0f) {
    return Status::InvalidArgument(
      "Color-transform images or intensity target are invalid");
  }

  try {
    Status status = TransformLinearRgbToOpsin(
      linear_rgb, intensity_target, padded_opsin);
    if (!status.ok()) return status;
    for (size_t channel = 0; channel < 3; ++channel) {
      const float* edge = padded_opsin.plane[channel].Row(
        linear_rgb.height() - 1);
      for (size_t y = linear_rgb.height(); y < padded_opsin.height(); ++y) {
        std::copy_n(
          edge, padded_opsin.width(),
          padded_opsin.plane[channel].Row(y));
      }
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate color-transform scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Color-transform image dimensions are too large");
  }
  return Status::Ok();
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
