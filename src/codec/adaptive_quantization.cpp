// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/adaptive_quantization.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "core/managed_allocator.h"
#include "codec/adaptive_quantization_internal.h"
#include "codec/color_transform.h"
#include "codec/convolution.h"
#include "codec/maximum_error.h"
#include "codec/quantization.h"
#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"
#include "core/thread_budget.h"
#include "util/fast_math.h"

namespace gjxl {
using resource_budget_internal::ManagedVector;

namespace {

namespace aqi = adaptive_quantization_internal;
using ProfileClock = std::chrono::steady_clock;

uint64_t ElapsedNanoseconds(ProfileClock::time_point begin) {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      ProfileClock::now() - begin).count());
}

template <typename Function>
Status RunParallelInitialQuantWork(
  size_t count,
  size_t value_count,
  Function&& function) {

  if (count == 0) return Status::Ok();
  constexpr size_t kMinimumParallelValues = 256 * 256;
  constexpr size_t kMaximumWorkers = 12;
  const size_t hardware_workers = std::max<size_t>(
    std::thread::hardware_concurrency(), 1);
  const size_t automatic_worker_count = value_count < kMinimumParallelValues
    ? 1
    : std::min(count, std::min(kMaximumWorkers, hardware_workers));
  const size_t cpu_thread_count =
    thread_budget_internal::CpuThreadCount();
  auto* const participant_tracker =
    thread_budget_internal::ParticipantTracker();
  const auto resource_context = resource_budget_internal::CurrentResourceContext();
  if (thread_budget_internal::InExplicitParallelScope()) {
    for (size_t index = 0; index < count; ++index) {
      Status status = function(index);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }
  const size_t participant_count = cpu_thread_count == 0
    ? automatic_worker_count
    : std::min(automatic_worker_count, cpu_thread_count);
  if (participant_count == 1) {
    thread_budget_internal::ParallelScope scope(
      cpu_thread_count, participant_tracker, resource_context);
    for (size_t index = 0; index < count; ++index) {
      Status status = function(index);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  ManagedVector<Status> statuses(count);
  std::atomic<size_t> next_index{0};
  ManagedVector<std::thread> workers;
  const size_t spawned_worker_count = cpu_thread_count == 0
    ? participant_count
    : participant_count - 1;
  workers.reserve(spawned_worker_count);
  const auto run_worker = [&] {
    thread_budget_internal::ParallelScope scope(
      cpu_thread_count, participant_tracker, resource_context);
    while (true) {
      const size_t index =
        next_index.fetch_add(1, std::memory_order_relaxed);
      if (index >= count) break;
      try {
        statuses[index] = function(index);
      } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
        statuses[index] = failure.status();
      } catch (const std::bad_alloc&) {
        statuses[index] = Status::OutOfMemory(
          "Unable to allocate initial-quantization worker storage");
      } catch (...) {
        statuses[index] = Status::Internal(
          "Initial-quantization worker failed unexpectedly");
      }
    }
  };
  try {
    for (size_t worker = 0; worker < spawned_worker_count; ++worker) {
      workers.emplace_back(run_worker);
    }
  } catch (const std::bad_alloc&) {
    next_index.store(count, std::memory_order_relaxed);
    for (std::thread& worker : workers) worker.join();
    return Status::OutOfMemory("Unable to allocate CPU worker state");
  } catch (const std::system_error&) {
    next_index.store(count, std::memory_order_relaxed);
    for (std::thread& worker : workers) worker.join();
    for (size_t index = 0; index < count; ++index) {
      Status status = function(index);
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

template <typename Function>
Status MeasureEvaluationStage(
  aqi::EvaluationProfile* profile,
  aqi::EvaluationStage stage,
  Function&& function) {

  if (profile == nullptr) {
    return std::forward<Function>(function)();
  }
  const auto begin = ProfileClock::now();
  Status status = std::forward<Function>(function)();
  profile->stage_nanoseconds[static_cast<size_t>(stage)] =
    ElapsedNanoseconds(begin);
  return status;
}

constexpr float kAcQuant = 0.765f;
constexpr float kDcQuant = 1.095924047623553f;
constexpr float kDcQuantPower = 0.83f;
constexpr float kInverseLog2E = 0.6931471805599453f;

template <bool Invert>
float GammaDerivativeRatio(float value) {
  constexpr float kEpsilon = 1.0e-2f;
  constexpr float kSgMul = 226.77216153508914f;
  constexpr float kSgMul2 = 1.0f / 73.377132366608819f;
  constexpr float kSgReturnMul =
    kSgMul2 * 18.6580932135f * kInverseLog2E;
  constexpr float kSgOffset = 7.7825991679894591f;

  value = std::max(value, 0.0f);
  const float squared = value * value;
  const float numerator = std::fma(
    kSgReturnMul * 3.0f * kSgMul,
    squared,
    kEpsilon);
  const float denominator = std::fma(
    kInverseLog2E * kSgMul * value,
    squared,
    kSgOffset * kInverseLog2E + kEpsilon);

  if constexpr (Invert) {
    return numerator / denominator;
  }

  return denominator / numerator;
}

float ComputeMask(float value) {
  constexpr float kBase = -0.7647f;
  constexpr float kMul4 = 9.4708735624378946f;
  constexpr float kMul2 = 17.35036561631863f;
  constexpr float kOffset2 = 302.59587815579727f;
  constexpr float kMul3 = 6.7943250517376494f;
  constexpr float kOffset3 = 3.7179635626140772f;
  constexpr float kOffset4 = 0.25f * kOffset3;
  constexpr float kMul0 = 0.80061762862741759f;

  const float v1 = std::max(value * kMul0, 1.0e-3f);
  const float v2 = 1.0f / (v1 + kOffset2);
  const float v3 = 1.0f / std::fma(v1, v1, kOffset3);
  const float v4 = 1.0f / std::fma(v1, v1, kOffset4);
  return kBase + std::fma(
    kMul4,
    v4,
    std::fma(kMul2, v2, kMul3 * v3));
}

float MaskingSqrt(float value) {
  constexpr float kLogOffset = 27.505837037000106f;
  constexpr float kMul = 211.66567973503678f;
  const float inner_scale = std::sqrt(kMul * 1.0e8f);
  return 0.25f * std::sqrt(
    std::fma(value, inner_scale, kLogOffset));
}

void StoreMin4(
  float value,
  float& min0,
  float& min1,
  float& min2,
  float& min3) {

  if (value >= min3) {
    return;
  }

  if (value < min0) {
    min3 = min2;
    min2 = min1;
    min1 = min0;
    min0 = value;
  } else if (value < min1) {
    min3 = min2;
    min2 = min1;
    min1 = value;
  } else if (value < min2) {
    min3 = min2;
    min2 = value;
  } else {
    min3 = value;
  }
}

void Sort4(std::array<float, 4>* values) {
  if ((*values)[0] > (*values)[1]) {
    std::swap((*values)[0], (*values)[1]);
  }
  if ((*values)[0] > (*values)[2]) {
    std::swap((*values)[0], (*values)[2]);
  }
  if ((*values)[0] > (*values)[3]) {
    std::swap((*values)[0], (*values)[3]);
  }
  if ((*values)[1] > (*values)[2]) {
    std::swap((*values)[1], (*values)[2]);
  }
  if ((*values)[1] > (*values)[3]) {
    std::swap((*values)[1], (*values)[3]);
  }
  if ((*values)[2] > (*values)[3]) {
    std::swap((*values)[2], (*values)[3]);
  }
}

void FuzzyErosion(
  float butteraugli_target,
  Extent2D source_extent,
  const ManagedVector<float>& source,
  Extent2D destination_extent,
  ManagedVector<float>* destination) {

  constexpr std::array<float, 4> kMulBase = {
    0.125f,
    0.1f,
    0.09f,
    0.06f,
  };
  constexpr std::array<float, 4> kMulAdd = {
    0.0f,
    -0.1f,
    -0.09f,
    -0.06f,
  };
  constexpr float kTotal = 0.29959705784054957f;

  float target_mix = 0.0f;
  if (butteraugli_target < 2.0f) {
    target_mix = (2.0f - butteraugli_target) * 0.5f;
  }

  std::array<float, 4> weights{};
  float weight_sum = 0.0f;
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = kMulBase[i] + target_mix * kMulAdd[i];
    weight_sum += weights[i];
  }
  for (float& weight : weights) {
    weight *= kTotal / weight_sum;
  }

  destination->assign(
    destination_extent.width * destination_extent.height,
    0.0f);

  for (size_t y = 0; y < source_extent.height; ++y) {
    const size_t top = y == 0 ? y : y - 1;
    const size_t bottom = y + 1 < source_extent.height ? y + 1 : y;

    for (size_t x = 0; x < source_extent.width; ++x) {
      const size_t left = x == 0 ? x : x - 1;
      const size_t right = x + 1 < source_extent.width ? x + 1 : x;
      const auto at = [&](size_t sample_x, size_t sample_y) {
        return source[sample_y * source_extent.width + sample_x];
      };

      std::array<float, 4> minima = {
        at(x, y),
        at(left, y),
        at(right, y),
        at(left, top),
      };
      Sort4(&minima);
      StoreMin4(at(x, top), minima[0], minima[1], minima[2], minima[3]);
      StoreMin4(at(right, top), minima[0], minima[1], minima[2], minima[3]);
      StoreMin4(at(left, bottom), minima[0], minima[1], minima[2], minima[3]);
      StoreMin4(at(x, bottom), minima[0], minima[1], minima[2], minima[3]);
      StoreMin4(at(right, bottom), minima[0], minima[1], minima[2], minima[3]);

      const float value =
        weights[0] * minima[0] +
        weights[1] * minima[1] +
        weights[2] * minima[2] +
        weights[3] * minima[3];
      const size_t destination_index =
        (y / 2) * destination_extent.width + x / 2;

      if ((x & 1u) == 0 && (y & 1u) == 0) {
        (*destination)[destination_index] = value;
      } else {
        (*destination)[destination_index] += value;
      }
    }
  }
}

Status BlurPixelMask(
  Extent2D extent,
  const ManagedVector<float>& input,
  ManagedVector<float>* output) {

  constexpr std::array<float, 5> kFilter = {
    0.364911248f,
    0.05f,
    0.1688888021f,
    0.221069183f,
    0.306563504f,
  };
  double weight_sum = 1.0 + 4.0 * (
    kFilter[0] + kFilter[1] + kFilter[2] +
    kFilter[4] + 2.0 * kFilter[3]);
  weight_sum = std::max(weight_sum, 1.0e-5);
  const float normalize = static_cast<float>(1.0 / weight_sum);

  output->resize(extent.width * extent.height);
  return ConvolveSymmetric5(
    {
      .data = input.data(),
      .extent = extent,
      .stride = extent.width,
    },
    {
      .distance0 = normalize,
      .distance1 = normalize * kFilter[0],
      .distance2 = normalize * kFilter[2],
      .distance4 = normalize * kFilter[1],
      .distance8 = normalize * kFilter[4],
      .distance5 = normalize * kFilter[3],
    },
    {
      .data = output->data(),
      .extent = extent,
      .stride = extent.width,
    });
}

float GammaModulation(
  ConstImage3FView opsin,
  size_t block_x,
  size_t block_y,
  float value) {

  constexpr float kBias = 0.16f;
  std::array<float, 4> lane_sum{};

  for (size_t dy = 0; dy < kJxlBlockDimension; ++dy) {
    const float* x_row = opsin.plane[0].Row(block_y + dy);
    const float* y_row = opsin.plane[1].Row(block_y + dy);
    for (size_t dx = 0; dx < kJxlBlockDimension; ++dx) {
      const float in_y = y_row[block_x + dx] + kBias;
      const float in_x = x_row[block_x + dx];
      lane_sum[dx & 3u] += GammaDerivativeRatio<true>(in_y - in_x);
      lane_sum[dx & 3u] += GammaDerivativeRatio<true>(in_y + in_x);
    }
  }

  const float overall =
    ((lane_sum[0] + lane_sum[1]) +
     (lane_sum[2] + lane_sum[3])) *
    (0.5f / 64.0f);
  if (!std::isfinite(overall) || overall <= 0.0f) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  constexpr float kGamma = 0.1005613337192697f;
  return std::fma(
    kGamma,
    fast_math::FastLog2(overall),
    value);
}

float HighFrequencyModulation(
  ConstPlaneF32View y_plane,
  size_t block_x,
  size_t block_y,
  float value) {

  constexpr float kLimit = 0.0206f;
  std::array<float, 4> lane_sum{};

  for (size_t dy = 0; dy < kJxlBlockDimension; ++dy) {
    const float* row = y_plane.Row(block_y + dy) + block_x;
    const float* next = dy + 1 < kJxlBlockDimension
      ? y_plane.Row(block_y + dy + 1) + block_x
      : row;

    for (size_t dx = 0; dx < kJxlBlockDimension; ++dx) {
      if (dx + 1 < kJxlBlockDimension) {
        lane_sum[dx & 3u] += std::min(
          kLimit,
          std::abs(row[dx] - row[dx + 1]));
      }
      lane_sum[dx & 3u] += std::min(
        kLimit,
        std::abs(row[dx] - next[dx]));
    }
  }

  const float sum =
    (lane_sum[0] + lane_sum[1]) +
    (lane_sum[2] + lane_sum[3]);
  return value + (sum * -0.38f + 0.42f);
}

float BlueModulation(
  ConstImage3FView opsin,
  size_t block_x,
  size_t block_y,
  float value) {

  constexpr float kLimit = 0.010474084867598155f;
  constexpr float kOffset = 0.0031994768654636393f;
  std::array<float, 4> lane_sum{};

  for (size_t dy = 0; dy < kJxlBlockDimension; ++dy) {
    const float* x_row = opsin.plane[0].Row(block_y + dy);
    const float* y_row = opsin.plane[1].Row(block_y + dy);
    const float* b_row = opsin.plane[2].Row(block_y + dy);

    for (size_t dx = 0; dx < kJxlBlockDimension; ++dx) {
      const size_t x = block_x + dx;
      const float effective_y = y_row[x] + kOffset + std::abs(x_row[x]);
      if (b_row[x] > effective_y) {
        lane_sum[dx & 3u] += std::min(b_row[x] - effective_y, kLimit);
      }
    }
  }

  float sum =
    (lane_sum[0] + lane_sum[1]) +
    (lane_sum[2] + lane_sum[3]);
  if (sum >= 32.0f * kLimit) {
    sum = 64.0f * kLimit - sum;
  }
  constexpr float kMaxLimit = 15.463398341612438f;
  sum = std::min(sum, kMaxLimit * kLimit);
  return value + sum * 0.90590804735610064f;
}

void PerBlockModulations(
  ConstImage3FView opsin,
  InitialQuantizationOptions options,
  Extent2D block_extent,
  ManagedVector<float>* quant_field) {

  const float scale = kAcQuant / options.butteraugli_target * options.rescale;
  const float base_level = 0.48f * scale;
  float dampen = 1.0f;
  if (options.butteraugli_target >= 2.0f) {
    dampen = 1.0f -
      (options.butteraugli_target - 2.0f) / 12.0f;
    dampen = std::max(dampen, 0.0f);
  }
  const float multiplier = scale * dampen;
  const float addend = (1.0f - dampen) * base_level;

  for (size_t block_y = 0; block_y < block_extent.height; ++block_y) {
    for (size_t block_x = 0; block_x < block_extent.width; ++block_x) {
      const size_t index = block_y * block_extent.width + block_x;
      const size_t pixel_x = block_x * kJxlBlockDimension;
      const size_t pixel_y = block_y * kJxlBlockDimension;
      const float mask_value = ComputeMask((*quant_field)[index]);
      const float gamma_value = GammaModulation(
        opsin,
        pixel_x,
        pixel_y,
        mask_value);
      const float high_frequency_value = HighFrequencyModulation(
        opsin.plane[1],
        pixel_x,
        pixel_y,
        gamma_value);
      const float blue_value = BlueModulation(
        opsin,
        pixel_x,
        pixel_y,
        gamma_value);
      const float exponent = std::min(high_frequency_value, blue_value);
      (*quant_field)[index] =
        fast_math::FastPow2(exponent * 1.442695041f) * multiplier + addend;
    }
  }
}

Status ValidateInputs(
  ConstImage3FView opsin,
  InitialQuantizationOptions options,
  InitialQuantFieldOutput output,
  Extent2D* block_extent,
  size_t* pixel_count) {

  if (!opsin.valid()) {
    return Status::InvalidArgument(
      "Opsin image is invalid");
  }

  if (!BlockGrid::IsPaddedPixelExtent(opsin.extent())) {
    return Status::InvalidArgument(
      "Opsin image must be padded to complete 8x8 blocks");
  }

  if (opsin.width() >
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
      opsin.height() >
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
    return Status::InvalidArgument(
      "Opsin image dimensions are too large");
  }

  if (!std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f ||
      !std::isfinite(options.rescale) ||
      options.rescale <= 0.0f) {
    return Status::InvalidArgument(
      "Initial quantization options must be finite and positive");
  }

  *block_extent =
    BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
  if (!output.quant_field.valid() ||
      !output.strategy_mask.valid() ||
      !output.pixel_mask.valid() ||
      output.quant_field.extent != *block_extent ||
      output.strategy_mask.extent != *block_extent ||
      output.pixel_mask.extent != opsin.extent()) {
    return Status::InvalidArgument(
      "Initial quantization outputs have invalid geometry");
  }

  if (!opsin.extent().try_area(pixel_count)) {
    return Status::InvalidArgument(
      "Opsin image dimensions are too large");
  }

  for (const ConstPlaneF32View plane : opsin.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.Row(y)[x])) {
          return Status::InvalidArgument(
            "Opsin samples must be finite");
        }
      }
    }
  }

  return Status::Ok();
}

}  // namespace

Status ComputeInitialQuantDc(
  float butteraugli_target,
  float* quant_dc) {

  if (quant_dc == nullptr) {
    return Status::InvalidArgument(
      "Initial DC quantization output is null");
  }

  if (!std::isfinite(butteraugli_target) ||
      butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "Butteraugli target must be finite and positive");
  }

  constexpr float kDcMul = 0.3f;
  const float nonlinear_target = kDcMul * std::pow(
    butteraugli_target / kDcMul,
    kDcQuantPower);
  const float target_dc = std::max(
    0.5f * butteraugli_target,
    std::min(butteraugli_target, nonlinear_target));
  *quant_dc = std::min(kDcQuant / target_dc, 50.0f);
  return Status::Ok();
}

Status ComputeInitialQuantField(
  ConstImage3FView opsin,
  InitialQuantizationOptions options,
  InitialQuantFieldOutput output) {

  Extent2D block_extent;
  size_t pixel_count = 0;
  Status status = ValidateInputs(
    opsin,
    options,
    output,
    &block_extent,
    &pixel_count);
  if (!status.ok()) {
    return status;
  }

  size_t block_count = 0;
  if (!block_extent.try_area(&block_count)) {
    return Status::InvalidArgument(
      "Block grid dimensions are too large");
  }

  try {
    ManagedVector<float> pixel_mask(pixel_count);
    const Extent2D pre_erosion_extent{
      .width = opsin.width() / 4,
      .height = opsin.height() / 4,
    };
    ManagedVector<float> pre_erosion(
      pre_erosion_extent.width * pre_erosion_extent.height);

    constexpr float kMatchGammaOffset = 0.019f;
    constexpr float kDifferenceLimit = 0.2f;
    status = RunParallelInitialQuantWork(
      pre_erosion_extent.height, pixel_count,
      [&](size_t group_y) {
        ManagedVector<float> row_differences(opsin.width());
        const size_t y_begin = group_y * 4;
        for (size_t row_index = 0; row_index < 4; ++row_index) {
          const size_t y = y_begin + row_index;
          const size_t top = y == 0 ? y : y - 1;
          const size_t bottom = y + 1 < opsin.height() ? y + 1 : y;
          const float* row = opsin.plane[1].Row(y);
          const float* top_row = opsin.plane[1].Row(top);
          const float* bottom_row = opsin.plane[1].Row(bottom);

          for (size_t x = 0; x < opsin.width(); ++x) {
            const size_t left = x == 0 ? x : x - 1;
            const size_t right = x + 1 < opsin.width() ? x + 1 : x;
            const float base = 0.25f * (
              bottom_row[x] + top_row[x] + row[left] + row[right]);
            const float gamma = GammaDerivativeRatio<false>(
              row[x] + kMatchGammaOffset);

            float pixel_difference = std::abs(gamma * (row[x] - base));
            pixel_difference = std::log1p(pixel_difference);
            pixel_mask[y * opsin.width() + x] =
              1.0f / (pixel_difference + 0.01f);

            float block_difference = gamma * (row[x] - base);
            block_difference *= block_difference;
            block_difference = std::min(block_difference, kDifferenceLimit);
            block_difference = MaskingSqrt(block_difference);
            if (row_index == 0) {
              row_differences[x] = block_difference;
            } else {
              row_differences[x] += block_difference;
            }
          }
        }

        float* destination =
          pre_erosion.data() + group_y * pre_erosion_extent.width;
        for (size_t x = 0; x < pre_erosion_extent.width; ++x) {
          destination[x] = (
            row_differences[x * 4] +
            row_differences[x * 4 + 1] +
            row_differences[x * 4 + 2] +
            row_differences[x * 4 + 3]) * 0.25f;
        }
        return Status::Ok();
      });
    if (!status.ok()) return status;

    ManagedVector<float> quant_field;
    FuzzyErosion(
      options.butteraugli_target,
      pre_erosion_extent,
      pre_erosion,
      block_extent,
      &quant_field);

    ManagedVector<float> strategy_mask(block_count);
    for (size_t i = 0; i < block_count; ++i) {
      strategy_mask[i] = 1.0f / (quant_field[i] + 0.001f);
    }

    PerBlockModulations(
      opsin,
      options,
      block_extent,
      &quant_field);

    ManagedVector<float> blurred_pixel_mask;
    status = BlurPixelMask(
      opsin.extent(),
      pixel_mask,
      &blurred_pixel_mask);
    if (!status.ok()) {
      return status;
    }

    const auto valid_values = [](const ManagedVector<float>& values) {
      return std::ranges::all_of(
        values,
        [](float value) {
          return std::isfinite(value) && value > 0.0f;
        });
    };
    if (!valid_values(quant_field) ||
        !valid_values(strategy_mask) ||
        !valid_values(blurred_pixel_mask)) {
      return Status::InvalidArgument(
        "Initial quantization produced a non-finite result");
    }

    CopyContiguousPlane(quant_field, output.quant_field);
    CopyContiguousPlane(strategy_mask, output.strategy_mask);
    CopyContiguousPlane(blurred_pixel_mask, output.pixel_mask);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate initial quantization scratch storage");
  }

  return Status::Ok();
}

Status AdjustQuantField(
  const AcStrategyGrid& strategies,
  float butteraugli_target,
  ConstPlaneF32View input,
  PlaneF32View output) {

  if (!strategies.complete() ||
      !input.valid() ||
      !output.valid() ||
      input.extent != strategies.extent() ||
      output.extent != input.extent ||
      !std::isfinite(butteraugli_target) ||
      butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "Adjusted quant field inputs are invalid");
  }

  size_t value_count = 0;
  if (!input.extent.try_area(&value_count)) {
    return Status::InvalidArgument(
      "Adjusted quant field dimensions are too large");
  }

  try {
    ManagedVector<float> adjusted(value_count);
    for (size_t y = 0; y < input.extent.height; ++y) {
      for (size_t x = 0; x < input.extent.width; ++x) {
        const float value = input.Row(y)[x];
        if (!std::isfinite(value) || value <= 0.0f) {
          return Status::InvalidArgument(
            "Quant field must contain finite positive values");
        }
        adjusted[y * input.extent.width + x] = value;
      }
    }

    float mean_max_mixer = 1.0f;
    constexpr float kMixerLimit = 1.54138f;
    constexpr float kMixerSlope = 0.56391f;
    if (butteraugli_target > kMixerLimit) {
      mean_max_mixer = std::max(
        0.0f,
        mean_max_mixer -
          (butteraugli_target - kMixerLimit) * kMixerSlope);
    }

    const Status status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const Extent2D covered =
          GetAcStrategyInfo(strategy)->covered_blocks;
        float maximum = adjusted[
          block_y * input.extent.width + block_x];
        float mean = 0.0f;
        for (size_t dy = 0; dy < covered.height; ++dy) {
          for (size_t dx = 0; dx < covered.width; ++dx) {
            const float value = adjusted[
              (block_y + dy) * input.extent.width + block_x + dx];
            maximum = std::max(maximum, value);
            mean += value;
          }
        }
        const size_t block_count = covered.width * covered.height;
        mean /= static_cast<float>(block_count);
        float result = maximum;
        if (block_count >= 4) {
          result = maximum * mean_max_mixer +
            mean * (1.0f - mean_max_mixer);
        }
        for (size_t dy = 0; dy < covered.height; ++dy) {
          for (size_t dx = 0; dx < covered.width; ++dx) {
            adjusted[
              (block_y + dy) * input.extent.width + block_x + dx] = result;
          }
        }
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }

    CopyContiguousPlane(adjusted, output);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate adjusted quant field storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Adjusted quant field dimensions are too large");
  }
  return Status::Ok();
}

namespace {

struct QuantizationEvaluation {
  ManagedVector<float> block_distance;
  Image3FBuffer reconstructed_linear;
  VarDctEncoderFrame frame;
  double score = 0.0;
  MaximumErrorReduction maximum_error;
};

Status EvaluateQuantization(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View quant_field,
  ConstPlaneU8View epf_sharpness,
  float quant_dc,
  AdaptiveQuantizationOptions options,
  PreparedButteraugliReference* prepared_reference,
  QuantizationEvaluation* evaluation,
  aqi::EvaluationProfile* profile) {

  const Extent2D block_extent = strategies.extent();
  size_t block_count = 0;
  if (evaluation == nullptr || !block_extent.try_area(&block_count)) {
    return Status::InvalidArgument(
      "Adaptive-quantization evaluation output is invalid");
  }

  const auto evaluation_begin = profile == nullptr
    ? ProfileClock::time_point{}
    : ProfileClock::now();
  aqi::EvaluationProfile local_profile;
  aqi::EvaluationProfile* measured = profile == nullptr
    ? nullptr
    : &local_profile;
  QuantizationEvaluation result;
  ManagedVector<int32_t> raw_quant(block_count);
  Quantizer quantizer;
  ColorCorrelationMap color_correlation;
  ManagedVector<float> inverse_sigma(block_count);
  Status status = MeasureEvaluationStage(
    measured,
    aqi::EvaluationStage::kFieldConstruction,
    [&] {
      Status field_status = CreateQuantizerFromField(
        quant_dc,
        quant_field,
        {raw_quant.data(), block_extent, block_extent.width},
        &quantizer);
      if (!field_status.ok()) {
        return field_status;
      }
      field_status = ComputeFinalColorCorrelationMap(
        opsin,
        strategies,
        {raw_quant.data(), block_extent, block_extent.width},
        quantizer,
        options.fast_color_correlation,
        &color_correlation);
      if (!field_status.ok()) {
        return field_status;
      }
      return Status::Ok();
    });
  if (!status.ok()) {
    return status;
  }

  status = MeasureEvaluationStage(
    measured,
    aqi::EvaluationStage::kCoefficientCoding,
    [&] {
      FrameGeometry geometry;
      Status coding_status = FrameGeometry::Create(
        original_linear_rgb.extent(), &geometry);
      if (!coding_status.ok()) {
        return coding_status;
      }
      coding_status = ComputeQuantizedCoefficients(
        opsin,
        {
          .geometry = geometry,
          .strategies = &strategies,
          .raw_quant_field = {
            raw_quant.data(), block_extent, block_extent.width},
          .quantizer = &quantizer,
          .color_correlation = &color_correlation,
          .epf_sharpness = epf_sharpness,
        },
        options.profile,
        &result.frame);
      if (!coding_status.ok()) {
        return coding_status;
      }
      return ComputeEpfInverseSigma(
        strategies,
        result.frame.raw_quant_field(),
        quantizer,
        epf_sharpness,
        options.profile.epf_sigma,
        {inverse_sigma.data(), block_extent, block_extent.width});
    });
  if (!status.ok()) {
    return status;
  }

  Image3FBuffer reconstructed_opsin;
  status = MeasureEvaluationStage(
    measured,
    aqi::EvaluationStage::kReconstruction,
    [&] {
      reconstructed_opsin.resize(opsin.extent());
      return ReconstructQuantizedCoefficients(
        result.frame,
        reconstructed_opsin.view());
    });
  if (!status.ok()) {
    return status;
  }

  Image3FBuffer cropped_reconstruction;
  Image3FBuffer filtered_opsin;
  status = MeasureEvaluationStage(
    measured,
    aqi::EvaluationStage::kLoopFilters,
    [&] {
      cropped_reconstruction.resize(original_linear_rgb.extent());
      CopyImage(
        reconstructed_opsin.cropped_view(original_linear_rgb.extent()),
        cropped_reconstruction.view());
      filtered_opsin.resize(original_linear_rgb.extent());
      return ApplyLoopFilters(
        cropped_reconstruction.const_view(),
        {inverse_sigma.data(), block_extent, block_extent.width},
        options.profile.loop_filter,
        filtered_opsin.view());
    });
  if (!status.ok()) {
    return status;
  }

  status = MeasureEvaluationStage(
    measured,
    aqi::EvaluationStage::kColorConversion,
    [&] {
      result.reconstructed_linear.resize(original_linear_rgb.extent());
      return OpsinToLinearRgb(
        filtered_opsin.const_view(),
        options.profile.intensity_target,
        result.reconstructed_linear.view());
    });
  if (!status.ok()) {
    return status;
  }

  result.block_distance.resize(block_count);
  if (options.control_mode ==
      AdaptiveQuantizationControlMode::kMaximumError) {
    status = MeasureEvaluationStage(
      measured,
      aqi::EvaluationStage::kBlockReduction,
      [&] {
        const Status reduction_status = ReduceMaximumError(
          opsin,
          filtered_opsin.const_view(),
          original_linear_rgb.extent(),
          strategies,
          options.maximum_error,
          {result.block_distance.data(), block_extent, block_extent.width},
          &result.maximum_error);
        result.score = result.maximum_error.normalized_maximum;
        return reduction_status;
      });
  } else {
    size_t pixel_count = 0;
    if (!original_linear_rgb.extent().try_area(&pixel_count)) {
      return Status::InvalidArgument(
        "Adaptive-quantization reference extent is too large");
    }
    ManagedVector<float> distance_map(pixel_count);
    status = MeasureEvaluationStage(
      measured,
      aqi::EvaluationStage::kButteraugli,
      [&] {
        const PlaneF32View map{
          distance_map.data(), original_linear_rgb.extent(),
          original_linear_rgb.width()};
        return prepared_reference == nullptr
          ? ComputeButteraugliDistance(
              original_linear_rgb,
              result.reconstructed_linear.const_view(),
              options.butteraugli, map, &result.score)
          : prepared_reference->Compare(
              result.reconstructed_linear.const_view(), map, &result.score);
      });
    if (status.ok()) {
      status = MeasureEvaluationStage(
        measured,
        aqi::EvaluationStage::kBlockReduction,
        [&] {
          return ReduceButteraugliDistanceMap(
            {
              distance_map.data(),
              original_linear_rgb.extent(),
              original_linear_rgb.width(),
            },
            strategies,
            {result.block_distance.data(), block_extent, block_extent.width});
        });
    }
  }
  if (!status.ok()) {
    return status;
  }

  *evaluation = std::move(result);
  if (profile != nullptr) {
    local_profile.total_nanoseconds =
      ElapsedNanoseconds(evaluation_begin);
    *profile = local_profile;
  }
  return Status::Ok();
}

class CpuAdaptiveQuantizationEvaluator final
    : public aqi::AdaptiveQuantizationEvaluator {
public:
  CpuAdaptiveQuantizationEvaluator(
    ConstImage3FView original_linear_rgb,
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    PreparedButteraugliReference* prepared_reference)
    : original_linear_rgb_(original_linear_rgb),
      opsin_(opsin),
      strategies_(strategies),
      epf_sharpness_(epf_sharpness),
      options_(options),
      prepared_reference_(prepared_reference) {}

  Status Evaluate(
    ConstPlaneF32View quant_field,
    float quant_dc,
    bool,
    aqi::AdaptiveQuantizationEvaluation* evaluation,
    aqi::EvaluationProfile* profile) override {

    if (evaluation == nullptr) {
      return Status::InvalidArgument(
        "CPU adaptive-quantization evaluation output is null");
    }
    QuantizationEvaluation detailed;
    Status status = EvaluateQuantization(
      original_linear_rgb_, opsin_, strategies_, quant_field,
      epf_sharpness_, quant_dc, options_, prepared_reference_, &detailed,
      profile);
    if (!status.ok()) {
      return status;
    }

    aqi::AdaptiveQuantizationEvaluation bounded;
    bounded.block_distance = std::move(detailed.block_distance);
    bounded.quantizer = detailed.frame.quantizer();
    bounded.score = detailed.score;
    bounded.maximum_error = detailed.maximum_error;
    final_evaluation_ = std::move(detailed);
    *evaluation = std::move(bounded);
    return Status::Ok();
  }

  [[nodiscard]] QuantizationEvaluation&& TakeFinalEvaluation() noexcept {
    return std::move(final_evaluation_);
  }

private:
  ConstImage3FView original_linear_rgb_;
  ConstImage3FView opsin_;
  const AcStrategyGrid& strategies_;
  ConstPlaneU8View epf_sharpness_;
  AdaptiveQuantizationOptions options_;
  PreparedButteraugliReference* prepared_reference_ = nullptr;
  QuantizationEvaluation final_evaluation_;
};

Status ValidateAdaptiveQuantizationOutput(
  ConstImage3FView original_linear_rgb,
  const AcStrategyGrid& strategies,
  AdaptiveQuantizationOptions options,
  const AdaptiveQuantizationOutput& output) {

  if (!output.quant_field.valid() ||
      !output.block_distance_map.valid() ||
      !output.reconstructed_linear_rgb.valid() ||
      output.frame == nullptr ||
      output.score_history == nullptr ||
      (options.control_mode ==
         AdaptiveQuantizationControlMode::kMaximumError &&
       output.maximum_error_result == nullptr)) {
    return Status::InvalidArgument(
      "Adaptive-quantization output is invalid");
  }

  const Extent2D block_extent = strategies.extent();
  if (output.quant_field.extent != block_extent ||
      output.block_distance_map.extent != block_extent ||
      output.reconstructed_linear_rgb.extent() !=
        original_linear_rgb.extent()) {
    return Status::InvalidArgument(
      "Adaptive-quantization output geometry does not match");
  }
  return Status::Ok();
}

}  // namespace

namespace adaptive_quantization_internal {

Status PrepareButteraugliPolicy(
  ConstPlaneF32View adjusted_initial_quant_field,
  float butteraugli_target,
  ButteraugliPolicySetup* setup) {

  if (setup == nullptr || !adjusted_initial_quant_field.valid() ||
      !std::isfinite(butteraugli_target) || butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "Butteraugli policy setup input is invalid");
  }
  float initial_minimum = std::numeric_limits<float>::infinity();
  float initial_maximum = 0.0f;
  for (size_t y = 0; y < adjusted_initial_quant_field.extent.height; ++y) {
    for (size_t x = 0; x < adjusted_initial_quant_field.extent.width; ++x) {
      const float value = adjusted_initial_quant_field.Row(y)[x];
      if (!std::isfinite(value) || value <= 0.0f) {
        return Status::InvalidArgument(
          "Adjusted quant field must contain finite positive values");
      }
      initial_minimum = std::min(initial_minimum, value);
      initial_maximum = std::max(initial_maximum, value);
    }
  }
  const float initial_ratio = initial_maximum / initial_minimum;
  const float maximum_deviation = std::sqrt(250.0f / initial_ratio);
  const float asymmetry = std::min(2.0f, maximum_deviation);
  ButteraugliPolicySetup candidate;
  candidate.lower_bound =
    initial_minimum / (asymmetry * maximum_deviation);
  candidate.upper_bound =
    initial_maximum * (maximum_deviation / asymmetry);
  if (!std::isfinite(candidate.lower_bound) ||
      !std::isfinite(candidate.upper_bound) ||
      candidate.lower_bound <= 0.0f ||
      candidate.upper_bound < candidate.lower_bound ||
      candidate.upper_bound / candidate.lower_bound >= 253.0f ||
      candidate.upper_bound >
        static_cast<float>(std::numeric_limits<long>::max()) /
          static_cast<float>(kQuantGlobalScaleDenominator)) {
    return Status::InvalidArgument(
      "Initial quant field cannot form libjxl AQ bounds");
  }
  Status status = ComputeInitialQuantDc(
    butteraugli_target, &candidate.quant_dc);
  if (!status.ok()) return status;
  *setup = candidate;
  return Status::Ok();
}

Status ValidateAdaptiveQuantizationPolicyInputs(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options) {

  if (!original_linear_rgb.valid() ||
      !opsin.valid() ||
      !strategies.complete() ||
      !initial_quant_field.valid() ||
      !epf_sharpness.valid()) {
    return Status::InvalidArgument(
      "Adaptive-quantization inputs are invalid");
  }

  const Extent2D block_extent = strategies.extent();
  Extent2D padded_pixel_extent;
  if (!BlockGrid{block_extent}.try_padded_pixel_extent(
        &padded_pixel_extent) ||
      opsin.extent() != padded_pixel_extent ||
      initial_quant_field.extent != block_extent ||
      epf_sharpness.extent != block_extent) {
    return Status::InvalidArgument(
      "Adaptive-quantization image and block geometry do not match");
  }

  if (original_linear_rgb.width() > opsin.width() ||
      original_linear_rgb.height() > opsin.height() ||
      original_linear_rgb.width() <=
        opsin.width() - kJxlBlockDimension ||
      original_linear_rgb.height() <=
        opsin.height() - kJxlBlockDimension) {
    return Status::InvalidArgument(
      "Adaptive-quantization padding exceeds one partial block");
  }

  if (!options.profile.valid()) {
    return Status::InvalidArgument(
      "Adaptive-quantization profile is invalid");
  }
  switch (options.control_mode) {
    case AdaptiveQuantizationControlMode::kButteraugli:
      if (!std::isfinite(options.butteraugli_target) ||
          options.butteraugli_target <= 0.0f || options.iterations > 4) {
        return Status::InvalidArgument(
          "Butteraugli adaptive-quantization options are invalid");
      }
      break;
    case AdaptiveQuantizationControlMode::kMaximumError:
      if (!std::ranges::all_of(options.maximum_error, [](float value) {
            return std::isfinite(value) && value > 0.0f;
          })) {
        return Status::InvalidArgument(
          "Maximum-error adaptive-quantization limits are invalid");
      }
      break;
    default:
      return Status::InvalidArgument(
        "Adaptive-quantization control mode is invalid");
  }

  return Status::Ok();
}

Status ValidateResidentAdaptiveQuantizationPolicyInputs(
  ConstImage3FView original_linear_rgb,
  Extent2D opsin_extent,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options) {

  if (!original_linear_rgb.valid() || opsin_extent.empty() ||
      !strategies.complete() || !initial_quant_field.valid() ||
      !epf_sharpness.valid()) {
    return Status::InvalidArgument(
      "Resident adaptive-quantization inputs are invalid");
  }
  const Extent2D block_extent = strategies.extent();
  Extent2D padded_pixel_extent;
  if (!BlockGrid{block_extent}.try_padded_pixel_extent(
        &padded_pixel_extent) ||
      opsin_extent != padded_pixel_extent ||
      initial_quant_field.extent != block_extent ||
      epf_sharpness.extent != block_extent) {
    return Status::InvalidArgument(
      "Resident adaptive-quantization geometry does not match");
  }
  if (original_linear_rgb.width() > opsin_extent.width ||
      original_linear_rgb.height() > opsin_extent.height ||
      original_linear_rgb.width() <=
        opsin_extent.width - kJxlBlockDimension ||
      original_linear_rgb.height() <=
        opsin_extent.height - kJxlBlockDimension) {
    return Status::InvalidArgument(
      "Resident adaptive-quantization padding exceeds one partial block");
  }
  if (!options.profile.valid()) {
    return Status::InvalidArgument(
      "Resident adaptive-quantization profile is invalid");
  }
  switch (options.control_mode) {
    case AdaptiveQuantizationControlMode::kButteraugli:
      if (!std::isfinite(options.butteraugli_target) ||
          options.butteraugli_target <= 0.0f || options.iterations > 4) {
        return Status::InvalidArgument(
          "Resident Butteraugli adaptive-quantization options are invalid");
      }
      break;
    case AdaptiveQuantizationControlMode::kMaximumError:
      if (!std::ranges::all_of(options.maximum_error, [](float value) {
            return std::isfinite(value) && value > 0.0f;
          })) {
        return Status::InvalidArgument(
          "Resident maximum-error limits are invalid");
      }
      break;
    default:
      return Status::InvalidArgument(
        "Resident adaptive-quantization control mode is invalid");
  }
  return Status::Ok();
}

namespace {

Status RunAdaptiveQuantizationPolicyImpl(
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationEvaluator& evaluator,
  AdaptiveQuantizationPolicyResult* result,
  AdaptiveQuantizationProfile* profile,
  bool input_adjusted) {

  const Extent2D block_extent = strategies.extent();
  size_t block_count = 0;
  if (result == nullptr || !block_extent.try_area(&block_count) ||
      initial_quant_field.extent != block_extent) {
    return Status::InvalidArgument(
      "Adaptive-quantization policy input or output is invalid");
  }

  try {
    if (options.control_mode ==
        AdaptiveQuantizationControlMode::kMaximumError) {
      constexpr float kInitializationTarget = 1.0f;
      constexpr float kInitialQuantDc = 0x1.43d136p+2f;
      constexpr size_t kUpdateCount = 5;
      constexpr size_t kEvaluationCount = kUpdateCount + 1;

      AdaptiveQuantizationProfile local_profile;
      if (profile != nullptr) {
        local_profile.evaluations.reserve(kEvaluationCount);
      }
      const auto setup_begin = profile == nullptr
        ? ProfileClock::time_point{}
        : ProfileClock::now();
      ManagedVector<float> quant_field(block_count);
      Status status = Status::Ok();
      if (input_adjusted) {
        for (size_t y = 0; y < block_extent.height; ++y) {
          for (size_t x = 0; x < block_extent.width; ++x) {
            const float value = initial_quant_field.Row(y)[x];
            if (!std::isfinite(value) || value <= 0.0f) {
              return Status::InvalidArgument(
                "Adjusted quant field must contain finite positive values");
            }
            quant_field[y * block_extent.width + x] = value;
          }
        }
      } else {
        status = AdjustQuantField(
          strategies,
          kInitializationTarget,
          initial_quant_field,
          {quant_field.data(), block_extent, block_extent.width});
      }
      if (!status.ok()) {
        return status;
      }
      if (profile != nullptr) {
        local_profile.loop_setup_nanoseconds =
          ElapsedNanoseconds(setup_begin);
      }

      std::vector<double> score_history;
      score_history.reserve(kEvaluationCount);
      AdaptiveQuantizationEvaluation evaluation;
      bool upper_bound_limited = false;
      ManagedVector<float> best_feasible_field;
      float best_feasible_error = -1.0f;
      const auto evaluate = [&](bool is_final) -> Status {
        EvaluationProfile evaluation_profile;
        Status evaluation_status = evaluator.Evaluate(
          {quant_field.data(), block_extent, block_extent.width},
          kInitialQuantDc,
          is_final,
          &evaluation,
          profile == nullptr ? nullptr : &evaluation_profile);
        if (!evaluation_status.ok()) {
          return evaluation_status;
        }
        if (evaluation.block_distance.size() != block_count ||
            !std::ranges::all_of(
              evaluation.block_distance,
              [](float value) {
                return std::isfinite(value) && value >= 0.0f;
              }) ||
            !evaluation.quantizer.valid() ||
            !std::isfinite(evaluation.score) || evaluation.score < 0.0 ||
            !std::isfinite(
              evaluation.maximum_error.normalized_maximum) ||
            evaluation.maximum_error.normalized_maximum < 0.0f ||
            !std::ranges::all_of(
              evaluation.maximum_error.channel_maximum,
              [](float value) {
                return std::isfinite(value) && value >= 0.0f;
              })) {
          return Status::Internal(
            "Maximum-error evaluator returned an invalid result");
        }
        if (profile != nullptr) {
          local_profile.evaluations.push_back(evaluation_profile);
        }
        score_history.push_back(evaluation.score);
        return Status::Ok();
      };

      // Apply all five pinned updates, but retain the closest already-valid
      // field so a later transform-local oscillation cannot discard a field
      // that satisfies the hard global maximum.
      for (size_t iteration = 0; iteration < kUpdateCount; ++iteration) {
        status = evaluate(false);
        if (!status.ok()) {
          return status;
        }
        const float normalized =
          evaluation.maximum_error.normalized_maximum;
        if (normalized <= 1.0f && normalized > best_feasible_error) {
          best_feasible_error = normalized;
          best_feasible_field = quant_field;
        }

        const auto update_begin = profile == nullptr
          ? ProfileClock::time_point{}
          : ProfileClock::now();
        ManagedVector<float> updated(block_count);
        bool iteration_limited = false;
        status = UpdateMaximumErrorQuantField(
          strategies,
          {evaluation.block_distance.data(), block_extent,
           block_extent.width},
          {quant_field.data(), block_extent, block_extent.width},
          {updated.data(), block_extent, block_extent.width},
          &iteration_limited);
        if (!status.ok()) {
          return status;
        }
        upper_bound_limited |= iteration_limited;
        quant_field = std::move(updated);
        if (profile != nullptr) {
          local_profile.quant_field_update_nanoseconds +=
            ElapsedNanoseconds(update_begin);
        }
      }
      if (!best_feasible_field.empty()) {
        quant_field = std::move(best_feasible_field);
      }
      status = evaluate(true);
      if (!status.ok()) {
        return status;
      }

      AdaptiveQuantizationPolicyResult candidate;
      candidate.quant_field = std::move(quant_field);
      candidate.block_distance = std::move(evaluation.block_distance);
      candidate.score_history = std::move(score_history);
      candidate.maximum_error = {
        .achieved = evaluation.maximum_error.channel_maximum,
        .normalized_maximum =
          evaluation.maximum_error.normalized_maximum,
        .evaluation_count = kEvaluationCount,
        .outcome = evaluation.maximum_error.normalized_maximum <= 1.0f
          ? MaximumErrorOutcome::kMet
          : (upper_bound_limited
              ? MaximumErrorOutcome::kQuantizationRangeExhausted
              : MaximumErrorOutcome::kIterationLimit),
      };
      *result = std::move(candidate);
      if (profile != nullptr) {
        *profile = std::move(local_profile);
      }
      return Status::Ok();
    }

    AdaptiveQuantizationProfile local_profile;
    if (profile != nullptr) {
      local_profile.evaluations.reserve(options.iterations + 1);
    }
    const auto setup_begin = profile == nullptr
      ? ProfileClock::time_point{}
      : ProfileClock::now();
    ManagedVector<float> quant_field(block_count);
    Status status = Status::Ok();
    if (input_adjusted) {
      for (size_t y = 0; y < block_extent.height; ++y) {
        for (size_t x = 0; x < block_extent.width; ++x) {
          const float value = initial_quant_field.Row(y)[x];
          if (!std::isfinite(value) || value <= 0.0f) {
            return Status::InvalidArgument(
              "Adjusted quant field must contain finite positive values");
          }
          quant_field[y * block_extent.width + x] = value;
        }
      }
    } else {
      status = AdjustQuantField(
        strategies,
        options.butteraugli_target,
        initial_quant_field,
        {quant_field.data(), block_extent, block_extent.width});
    }
    if (!status.ok()) {
      return status;
    }
    const ManagedVector<float> adjusted_initial = quant_field;

    ButteraugliPolicySetup setup;
    status = PrepareButteraugliPolicy(
      {adjusted_initial.data(), block_extent, block_extent.width},
      options.butteraugli_target, &setup);
    if (!status.ok()) {
      return status;
    }
    if (profile != nullptr) {
      local_profile.loop_setup_nanoseconds =
        ElapsedNanoseconds(setup_begin);
    }

    std::vector<double> score_history;
    score_history.reserve(options.iterations + 1);
    AdaptiveQuantizationEvaluation evaluation;
    for (size_t iteration = 0; iteration <= options.iterations; ++iteration) {
      EvaluationProfile evaluation_profile;
      status = evaluator.Evaluate(
        {quant_field.data(), block_extent, block_extent.width},
        setup.quant_dc,
        iteration == options.iterations,
        &evaluation,
        profile == nullptr ? nullptr : &evaluation_profile);
      if (!status.ok()) {
        return status;
      }
      if (evaluation.block_distance.size() != block_count ||
          !std::ranges::all_of(
            evaluation.block_distance,
            [](float value) {
              return std::isfinite(value) && value >= 0.0f;
            }) ||
          !evaluation.quantizer.valid() ||
          !std::isfinite(evaluation.score) || evaluation.score < 0.0) {
        return Status::Internal(
          "Adaptive-quantization evaluator returned an invalid result");
      }
      if (profile != nullptr) {
        local_profile.evaluations.push_back(evaluation_profile);
      }
      score_history.push_back(evaluation.score);
      if (iteration == options.iterations) {
        break;
      }

      const auto update_begin = profile == nullptr
        ? ProfileClock::time_point{}
        : ProfileClock::now();
      // The second update is constrained toward the initial field to reduce
      // oscillation caused by DC reconstruction, matching libjxl.
      if (iteration == 1) {
        for (size_t index = 0; index < block_count; ++index) {
          const float clamp =
            0.4f * quant_field[index] +
            0.6f * adjusted_initial[index];
          if (quant_field[index] < clamp) {
            quant_field[index] = std::clamp(
              clamp, setup.lower_bound, setup.upper_bound);
          }
        }
      }

      const double power = iteration < 2 ? 0.2 : 0.0;
      for (size_t index = 0; index < block_count; ++index) {
        const float difference = evaluation.block_distance[index] /
          options.butteraugli_target;
        if (!std::isfinite(difference) || difference < 0.0f) {
          return Status::Internal(
            "Adaptive quantization produced an invalid block distance");
        }

        if (difference <= 1.0f) {
          if (power != 0.0) {
            quant_field[index] *= static_cast<float>(
              std::pow(difference, power));
          }
        } else {
          const float old = quant_field[index];
          quant_field[index] *= difference;
          const long old_raw = std::lround(
            old * evaluation.quantizer.inverse_global_scale());
          const long new_raw = std::lround(
            quant_field[index] *
            evaluation.quantizer.inverse_global_scale());
          if (old_raw == new_raw) {
            quant_field[index] = old + evaluation.quantizer.scale();
          }
        }
        quant_field[index] = std::clamp(
          quant_field[index], setup.lower_bound, setup.upper_bound);
      }
      if (profile != nullptr) {
        local_profile.quant_field_update_nanoseconds +=
          ElapsedNanoseconds(update_begin);
      }
    }

    AdaptiveQuantizationPolicyResult candidate;
    candidate.quant_field = std::move(quant_field);
    candidate.block_distance = std::move(evaluation.block_distance);
    candidate.score_history = std::move(score_history);
    *result = std::move(candidate);
    if (profile != nullptr) {
      *profile = std::move(local_profile);
    }
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate adaptive-quantization scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Adaptive-quantization dimensions are too large");
  }
  return Status::Ok();
}

}  // namespace

Status RunAdaptiveQuantizationPolicy(
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationEvaluator& evaluator,
  AdaptiveQuantizationPolicyResult* result,
  AdaptiveQuantizationProfile* profile) {

  return RunAdaptiveQuantizationPolicyImpl(
    strategies, initial_quant_field, options, evaluator, result, profile,
    false);
}

Status RunAdaptiveQuantizationPolicyAdjusted(
  const AcStrategyGrid& strategies,
  ConstPlaneF32View adjusted_initial_quant_field,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationEvaluator& evaluator,
  AdaptiveQuantizationPolicyResult* result,
  AdaptiveQuantizationProfile* profile) {

  return RunAdaptiveQuantizationPolicyImpl(
    strategies, adjusted_initial_quant_field, options, evaluator, result,
    profile, true);
}

}  // namespace adaptive_quantization_internal

namespace {

Status FindBestQuantizationImpl(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationOutput output,
  PreparedButteraugliReference* prepared_reference,
  aqi::AdaptiveQuantizationProfile* profile) {

  Status status = aqi::ValidateAdaptiveQuantizationPolicyInputs(
    original_linear_rgb,
    opsin,
    strategies,
    initial_quant_field,
    epf_sharpness,
    options);
  if (status.ok()) {
    status = ValidateAdaptiveQuantizationOutput(
      original_linear_rgb, strategies, options, output);
  }
  if (!status.ok()) {
    return status;
  }
  if (options.control_mode == AdaptiveQuantizationControlMode::kButteraugli &&
      prepared_reference != nullptr &&
      (!prepared_reference->ready() ||
       prepared_reference->extent() != original_linear_rgb.extent() ||
       prepared_reference->options() != options.butteraugli)) {
    return Status::InvalidArgument(
      "Prepared Butteraugli reference does not match CPU AQ");
  }

  CpuAdaptiveQuantizationEvaluator evaluator(
    original_linear_rgb, opsin, strategies, epf_sharpness, options,
    prepared_reference);
  aqi::AdaptiveQuantizationPolicyResult policy_result;
  aqi::AdaptiveQuantizationProfile local_profile;
  status = aqi::RunAdaptiveQuantizationPolicy(
    strategies, initial_quant_field, options, evaluator, &policy_result,
    profile == nullptr ? nullptr : &local_profile);
  if (!status.ok()) {
    return status;
  }

  QuantizationEvaluation evaluation = evaluator.TakeFinalEvaluation();
  const auto commit_begin = profile == nullptr
    ? ProfileClock::time_point{}
    : ProfileClock::now();
  CopyContiguousPlane(policy_result.quant_field, output.quant_field);
  CopyContiguousPlane(policy_result.block_distance, output.block_distance_map);
  CopyImage(
    evaluation.reconstructed_linear.const_view(),
    output.reconstructed_linear_rgb);
  *output.frame = std::move(evaluation.frame);
  *output.score_history = std::move(policy_result.score_history);
  if (output.maximum_error_result != nullptr) {
    *output.maximum_error_result = policy_result.maximum_error;
  }
  if (profile != nullptr) {
    local_profile.output_commit_nanoseconds =
      ElapsedNanoseconds(commit_begin);
    *profile = std::move(local_profile);
  }
  return Status::Ok();

}

}  // namespace

Status FindBestQuantization(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationOutput output) {

  return FindBestQuantizationImpl(
    original_linear_rgb,
    opsin,
    strategies,
    initial_quant_field,
    epf_sharpness,
    options,
    output,
    nullptr,
    nullptr);
}

namespace adaptive_quantization_internal {

Status FindBestQuantizationProfiled(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationOutput output,
  AdaptiveQuantizationProfile* profile) {

  if (profile == nullptr) {
    return Status::InvalidArgument(
      "Adaptive-quantization profile output is null");
  }
  return FindBestQuantizationImpl(
    original_linear_rgb,
    opsin,
    strategies,
    initial_quant_field,
    epf_sharpness,
    options,
    output,
    nullptr,
    profile);
}

Status FindBestQuantizationPrepared(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  PreparedButteraugliReference* prepared_reference,
  AdaptiveQuantizationOutput output) {

  return FindBestQuantizationImpl(
    original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options, output, prepared_reference, nullptr);
}

}  // namespace adaptive_quantization_internal

}  // namespace gjxl
