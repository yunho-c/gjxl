// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/reconstruction.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "core/managed_allocator.h"
#include "codec/dc_conversion.h"
#include "codec/dc_quantization.h"
#include "codec/dct.h"
#include "codec/quantization.h"
#include "codec/reconstruction_internal.h"
#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"
#include "core/thread_budget.h"

namespace gjxl {
using resource_budget_internal::ManagedVector;

namespace {

constexpr std::array<XybChannel, 3> kChannels = {
  XybChannel::kX,
  XybChannel::kY,
  XybChannel::kB,
};

template <typename Function>
Status RunParallelForwardTransforms(
  size_t count,
  size_t coefficient_count,
  Function&& function) {

  if (count == 0) return Status::Ok();
  constexpr size_t kMinimumParallelCoefficients = 256 * 256;
  constexpr size_t kMaximumWorkers = 8;
  const size_t hardware_workers = std::max<size_t>(
    std::thread::hardware_concurrency(), 1);
  const size_t automatic_worker_count =
    coefficient_count < kMinimumParallelCoefficients
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
          "Unable to allocate forward-transform worker storage");
      } catch (const std::length_error&) {
        statuses[index] = Status::InvalidArgument(
          "Forward-transform worker storage is too large");
      } catch (...) {
        statuses[index] = Status::Internal(
          "Forward-transform worker failed unexpectedly");
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

float MatrixMultiplier(
  size_t channel,
  const SimpleVarDctCodestreamProfile& profile) {

  if (channel == 0) {
    return QuantizationMatrixMultiplier(profile.x_qm_scale);
  }
  if (channel == 2) {
    return QuantizationMatrixMultiplier(profile.b_qm_scale);
  }
  return 1.0f;
}

Status ValidateFrameContract(
  Extent2D coefficient_extent,
  VarDctFrameInput input,
  const SimpleVarDctCodestreamProfile& profile) {

  if (coefficient_extent.empty() || input.geometry.frame().empty() ||
      input.strategies == nullptr ||
      !input.strategies->complete() ||
      !input.raw_quant_field.valid() ||
      input.quantizer == nullptr ||
      !input.quantizer->valid() ||
      input.color_correlation == nullptr ||
      !input.color_correlation->valid() ||
      !input.epf_sharpness.valid() ||
      !profile.valid() ||
      input.raw_quant_field.extent != input.strategies->extent() ||
      input.epf_sharpness.extent != input.strategies->extent() ||
      input.geometry.block_grid().blocks != input.strategies->extent()) {
    return Status::InvalidArgument(
      "Coefficient coding inputs are invalid or differently sized");
  }

  const Extent2D block_extent = input.strategies->extent();
  if (coefficient_extent != input.geometry.padded_frame()) {
    return Status::InvalidArgument(
      "Coefficient image does not match its block grid");
  }

  const Extent2D expected_color_tiles = ColorTileExtent(coefficient_extent);
  if (input.color_correlation->tile_extent() != expected_color_tiles) {
    return Status::InvalidArgument(
      "Color-correlation map does not match the coefficient image");
  }

  for (size_t y = 0; y < block_extent.height; ++y) {
    for (size_t x = 0; x < block_extent.width; ++x) {
      const int32_t raw_quant = input.raw_quant_field.Row(y)[x];
      if (raw_quant < 1 || raw_quant > kMaxRawQuant ||
          input.epf_sharpness.Row(y)[x] >= 8) {
        return Status::InvalidArgument(
          "Raw quantization or EPF sharpness is out of range");
      }
    }
  }

  return Status::Ok();
}

Status ValidateImageContract(
  ConstImage3FView opsin,
  VarDctFrameInput input,
  const SimpleVarDctCodestreamProfile& profile) {

  if (!opsin.valid()) {
    return Status::InvalidArgument("Coefficient image is invalid");
  }
  return ValidateFrameContract(opsin.extent(), input, profile);
}

void CopyPixelsFromImage(
  ConstPlaneF32View plane,
  size_t block_x,
  size_t block_y,
  Extent2D pixel_extent,
  std::span<float> pixels) {

  for (size_t y = 0; y < pixel_extent.height; ++y) {
    const float* source = plane.Row(
      block_y * kJxlBlockDimension + y) +
      block_x * kJxlBlockDimension;
    std::copy_n(
      source,
      pixel_extent.width,
      pixels.data() + y * pixel_extent.width);
  }
}

void CopyPixelsToImage(
  std::span<const float> pixels,
  size_t block_x,
  size_t block_y,
  Extent2D pixel_extent,
  PlaneF32View plane) {

  for (size_t y = 0; y < pixel_extent.height; ++y) {
    float* destination = plane.Row(
      block_y * kJxlBlockDimension + y) +
      block_x * kJxlBlockDimension;
    std::copy_n(
      pixels.data() + y * pixel_extent.width,
      pixel_extent.width,
      destination);
  }
}

}  // namespace

namespace prepared_coefficients_internal {

bool PreparedForwardDctCoefficients::valid() const noexcept {
  size_t pixel_count = 0;
  size_t tile_count = 0;
  if (pixel_extent.empty() || block_extent.empty() ||
      !BlockGrid::IsPaddedPixelExtent(pixel_extent) ||
      block_extent !=
        BlockGrid::FromPaddedPixelExtent(pixel_extent).blocks ||
      color_tile_extent != ColorTileExtent(pixel_extent) ||
      !pixel_extent.try_area(&pixel_count) ||
      !color_tile_extent.try_area(&tile_count) ||
      color_tile_offsets.size() != tile_count + 1 ||
      color_tile_offsets.empty() || color_tile_offsets.front() != 0 ||
      color_tile_offsets.back() != color_tile_transform_indices.size() ||
      color_tile_transform_indices.size() != transforms.size()) {
    return false;
  }
  for (size_t index = 1; index < color_tile_offsets.size(); ++index) {
    if (color_tile_offsets[index] < color_tile_offsets[index - 1] ||
        color_tile_offsets[index] > color_tile_transform_indices.size()) {
      return false;
    }
  }
  for (const auto& channel : coefficients) {
    if (channel.size() != pixel_count) {
      return false;
    }
  }
  size_t expected_offset = 0;
  for (const PreparedTransform& transform : transforms) {
    if (transform.strategy == AcStrategyType::kCount ||
        transform.coefficient_offset != expected_offset ||
        transform.coefficient_count == 0 || expected_offset > pixel_count ||
        transform.coefficient_count > pixel_count - expected_offset) {
      return false;
    }
    expected_offset += transform.coefficient_count;
  }
  if (expected_offset != pixel_count) {
    return false;
  }
  return std::ranges::all_of(
    color_tile_transform_indices,
    [&](size_t index) { return index < transforms.size(); });
}

Status PrepareForwardDctCoefficients(
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  PreparedForwardDctCoefficients* out) {

  if (out == nullptr || !opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent()) ||
      !strategies.complete() ||
      strategies.extent() !=
        BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks) {
    return Status::InvalidArgument(
      "Forward-coefficient preparation inputs are invalid");
  }

  size_t pixel_count = 0;
  const Extent2D tile_extent = ColorTileExtent(opsin.extent());
  size_t tile_count = 0;
  if (!opsin.extent().try_area(&pixel_count) ||
      !tile_extent.try_area(&tile_count)) {
    return Status::InvalidArgument(
      "Forward-coefficient preparation dimensions are too large");
  }

  try {
    PreparedForwardDctCoefficients candidate;
    candidate.pixel_extent = opsin.extent();
    candidate.block_extent = strategies.extent();
    candidate.color_tile_extent = tile_extent;
    for (auto& channel : candidate.coefficients) {
      channel.resize(pixel_count);
    }
    ManagedVector<ManagedVector<size_t>> tile_transforms(tile_count);
    size_t coefficient_offset = 0;
    Status status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr || !SupportsCpuDct(strategy)) {
          return Status::Unavailable(
            "Forward-coefficient preparation does not support a strategy");
        }
        const size_t coefficient_count = info->coefficient_count();
        if (coefficient_offset > pixel_count ||
            coefficient_count > pixel_count - coefficient_offset) {
          return Status::Internal(
            "Forward-coefficient preparation exceeded its storage");
        }
        const size_t tile_x = block_x /
          (kColorTileDimension / kJxlBlockDimension);
        const size_t tile_y = block_y /
          (kColorTileDimension / kJxlBlockDimension);
        const size_t tile_block_end_x = std::min(
          (tile_x + 1) * (kColorTileDimension / kJxlBlockDimension),
          strategies.extent().width);
        const size_t tile_block_end_y = std::min(
          (tile_y + 1) * (kColorTileDimension / kJxlBlockDimension),
          strategies.extent().height);
        if (block_x + info->covered_blocks.width > tile_block_end_x ||
            block_y + info->covered_blocks.height > tile_block_end_y) {
          return Status::InvalidArgument(
            "Forward-coefficient strategy crosses a color tile");
        }

        const size_t transform_index = candidate.transforms.size();
        candidate.transforms.push_back({
          block_x, block_y, strategy, coefficient_offset, coefficient_count});
        tile_transforms[tile_y * tile_extent.width + tile_x].push_back(
          transform_index);
        coefficient_offset += coefficient_count;
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }
    if (coefficient_offset != pixel_count) {
      return Status::Internal(
        "Prepared transforms do not cover the coefficient image");
    }

    constexpr size_t kMaxCoefficientCount = 32 * 32;
    status = RunParallelForwardTransforms(
      candidate.transforms.size(), pixel_count,
      [&](size_t transform_index) {
        const PreparedTransform& transform =
          candidate.transforms[transform_index];
        const AcStrategyInfo* info = GetAcStrategyInfo(transform.strategy);
        if (info == nullptr ||
            info->coefficient_count() != transform.coefficient_count ||
            transform.coefficient_count > kMaxCoefficientCount) {
          return Status::Internal(
            "Prepared forward-transform strategy disappeared");
        }
        std::array<float, kMaxCoefficientCount> pixels;
        const std::span<float> pixel_span{
          pixels.data(), transform.coefficient_count};
        for (size_t channel = 0; channel < 3; ++channel) {
          CopyPixelsFromImage(
            opsin.plane[channel], transform.block_x, transform.block_y,
            info->pixel_extent(), pixel_span);
          if (!std::ranges::all_of(
                pixel_span, [](float value) { return std::isfinite(value); })) {
            return Status::InvalidArgument(
              "Forward-coefficient input must contain finite values");
          }
          Status transform_status = ForwardDctCpu(
            transform.strategy, pixel_span,
            std::span<float>(candidate.coefficients[channel]).subspan(
              transform.coefficient_offset, transform.coefficient_count));
          if (!transform_status.ok()) return transform_status;
        }
        return Status::Ok();
      });
    if (!status.ok()) return status;

    candidate.color_tile_offsets.reserve(tile_count + 1);
    candidate.color_tile_offsets.push_back(0);
    for (const auto& indices : tile_transforms) {
      candidate.color_tile_transform_indices.insert(
        candidate.color_tile_transform_indices.end(),
        indices.begin(), indices.end());
      candidate.color_tile_offsets.push_back(
        candidate.color_tile_transform_indices.size());
    }
    if (!candidate.valid()) {
      return Status::Internal(
        "Forward-coefficient preparation produced invalid storage");
    }
    *out = std::move(candidate);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate prepared forward coefficients");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Prepared forward-coefficient dimensions are too large");
  }
}

}  // namespace prepared_coefficients_internal

namespace prepared_coefficients_internal {

Status ComputeQuantizedCoefficientsImpl(
  ConstImage3FView opsin,
  const PreparedForwardDctCoefficients* prepared,
  VarDctFrameInput input,
  SimpleVarDctCodestreamProfile profile,
  VarDctEncoderFrame* out,
  AcCoefficientDecisionMode decision_mode) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Quantized coefficient output is null");
  }
  switch (decision_mode) {
    case AcCoefficientDecisionMode::kAdjustedSharedQuant:
    case AcCoefficientDecisionMode::kFixedRawQuant:
      break;
    default:
      return Status::InvalidArgument(
        "AC coefficient decision mode is invalid");
  }

  Status status = prepared == nullptr
    ? ValidateImageContract(opsin, input, profile)
    : ValidateFrameContract(prepared->pixel_extent, input, profile);
  if (status.ok() && prepared != nullptr &&
      (!prepared->valid() ||
       prepared->block_extent != input.strategies->extent())) {
    status = Status::InvalidArgument(
      "Prepared coefficients do not match the coefficient-coding grid");
  }
  if (!status.ok()) {
    return status;
  }

  try {
    const Extent2D block_extent = input.strategies->extent();
    size_t block_count = 0;
    if (!block_extent.try_area(&block_count)) {
      return Status::InvalidArgument(
        "Coefficient coding block grid is too large");
    }


    const resource_budget_internal::ResourceClassScope resource_class(
      resource_budget_internal::ResourceClass::kCompletedFrame);
    VarDctEncoderFrame result;
    result.geometry_ = input.geometry;
    result.strategies_ = *input.strategies;
    result.raw_quant_field_.resize(block_count);
    result.epf_sharpness_.resize(block_count);
    for (size_t y = 0; y < block_extent.height; ++y) {
      std::copy_n(
        input.raw_quant_field.Row(y),
        block_extent.width,
        result.raw_quant_field_.data() + y * block_extent.width);
      std::copy_n(
        input.epf_sharpness.Row(y),
        block_extent.width,
        result.epf_sharpness_.data() + y * block_extent.width);
    }
    result.quantizer_ = *input.quantizer;
    result.color_correlation_ = *input.color_correlation;
    result.profile_ = profile;
    for (size_t channel = 0; channel < 3; ++channel) {
      result.quantized_dc_[channel].resize(block_count);
      result.dc_[channel].assign(block_count, 0.0f);
    }

    result.ac_group_extent_ = {
      (block_extent.width + kVarDctAcGroupBlockDimension - 1) /
        kVarDctAcGroupBlockDimension,
      (block_extent.height + kVarDctAcGroupBlockDimension - 1) /
        kVarDctAcGroupBlockDimension,
    };
    size_t group_count = 0;
    if (!result.ac_group_extent_.try_area(&group_count) ||
        group_count > std::numeric_limits<size_t>::max() / 3 ||
        group_count * 3 > std::numeric_limits<size_t>::max() /
          kVarDctAcGroupCoefficientCapacity) {
      return Status::InvalidArgument(
        "Coefficient coding group grid is too large");
    }
    result.group_used_coefficient_count_.assign(group_count, 0);
    result.ac_coefficients_.assign(
      group_count * 3 * kVarDctAcGroupCoefficientCapacity,
      0);

    constexpr size_t kMaxCoefficientCount = 32 * 32;
    constexpr size_t kMaxDcCount = 4 * 4;
    std::array<std::array<float, kMaxCoefficientCount>, 3> coefficients{};
    std::array<std::array<int32_t, kMaxCoefficientCount>, 3> quantized{};
    std::array<float, kMaxCoefficientCount> pixels{};
    std::array<float, kMaxDcCount> dc{};

    size_t prepared_index = 0;
    status = input.strategies->ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr || !SupportsCpuDct(strategy)) {
          return Status::Unavailable(
            "Coefficient coding does not support an AC strategy");
        }

        const size_t group_x = block_x / kVarDctAcGroupBlockDimension;
        const size_t group_y = block_y / kVarDctAcGroupBlockDimension;
        if ((block_x + info->covered_blocks.width - 1) /
              kVarDctAcGroupBlockDimension != group_x ||
            (block_y + info->covered_blocks.height - 1) /
              kVarDctAcGroupBlockDimension != group_y) {
          return Status::InvalidArgument(
            "AC strategy crosses a VarDCT group boundary");
        }
        const size_t group_index =
          group_y * result.ac_group_extent_.width + group_x;
        const size_t coefficient_count = info->coefficient_count();
        if (coefficient_count > kMaxCoefficientCount) {
          return Status::Internal(
            "Coefficient coding strategy exceeds its scratch capacity");
        }
        if (prepared != nullptr) {
          if (prepared_index >= prepared->transforms.size()) {
            return Status::InvalidArgument(
              "Prepared coefficients contain too few transforms");
          }
          const PreparedTransform& transform =
            prepared->transforms[prepared_index];
          if (transform.block_x != block_x || transform.block_y != block_y ||
              transform.strategy != strategy ||
              transform.coefficient_count != coefficient_count) {
            return Status::InvalidArgument(
              "Prepared coefficients do not match the strategy grid");
          }
          for (size_t channel = 0; channel < coefficients.size(); ++channel) {
            std::copy_n(
              prepared->coefficients[channel].data() +
                transform.coefficient_offset,
              coefficient_count, coefficients[channel].begin());
          }
        } else {
          const std::span<float> pixel_span{
            pixels.data(), coefficient_count};
          for (size_t channel = 0; channel < coefficients.size(); ++channel) {
            const std::span<float> coefficient_span{
              coefficients[channel].data(), coefficient_count};
            CopyPixelsFromImage(
              opsin.plane[channel], block_x, block_y, info->pixel_extent(),
              pixel_span);
            Status transform_status = ForwardDctCpu(
              strategy, pixel_span, coefficient_span);
            if (!transform_status.ok()) {
              return transform_status;
            }
          }
        }

        const int32_t initial_raw_quant =
          result.raw_quant_field_[block_y * block_extent.width + block_x];
        AdjustedAcQuantization quantization_decision{
          .raw_quant = initial_raw_quant,
          .y_thresholds = {0.58f, 0.64f, 0.64f, 0.64f},
        };
        Status block_status = Status::Ok();
        if (decision_mode ==
            AcCoefficientDecisionMode::kAdjustedSharedQuant) {
          const std::array<float, 3> matrix_multipliers = {
            MatrixMultiplier(0, profile),
            1.0f,
            MatrixMultiplier(2, profile),
          };
          const std::array<std::span<const float>, 3> coefficient_views = {
            std::span<const float>{coefficients[0].data(), coefficient_count},
            std::span<const float>{coefficients[1].data(), coefficient_count},
            std::span<const float>{coefficients[2].data(), coefficient_count},
          };
          block_status = SelectAdjustedAcQuantization(
            strategy,
            result.quantizer_,
            initial_raw_quant,
            matrix_multipliers,
            coefficient_views,
            &quantization_decision);
          if (!block_status.ok()) {
            return block_status;
          }
          // The pinned encoder stores the shared decision at the transform
          // anchor only; covered non-anchor raw-quant cells remain unchanged.
          result.raw_quant_field_[
            block_y * block_extent.width + block_x] =
              quantization_decision.raw_quant;
        }

        // Y is round-trip dequantized before removing its prediction from
        // X/B, exactly as in libjxl's VarDCT coefficient path.
        const size_t dc_count =
          info->covered_blocks.width * info->covered_blocks.height;
        if (dc_count > dc.size()) {
          return Status::Internal(
            "Coefficient coding strategy exceeds its DC scratch capacity");
        }
        const PlaneF32View y_dc_view{
          .data = dc.data(),
          .extent = info->covered_blocks,
          .stride = info->covered_blocks.width,
        };
        block_status = ConvertLowFrequenciesToDc(
          strategy,
          {coefficients[1].data(), coefficient_count},
          y_dc_view);
        if (!block_status.ok()) {
          return block_status;
        }
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            result.dc_[1][
              (block_y + dy) * block_extent.width + block_x + dx] =
                dc[dy * info->covered_blocks.width + dx];
          }
        }

        block_status = decision_mode ==
            AcCoefficientDecisionMode::kAdjustedSharedQuant
          ? QuantizeAdjustedYAcBlock(
              strategy,
              result.quantizer_,
              quantization_decision,
              {coefficients[1].data(), coefficient_count},
              {quantized[1].data(), coefficient_count})
          : QuantizeAcBlock(
              strategy,
              result.quantizer_,
              initial_raw_quant,
              {.channel = XybChannel::kY},
              {coefficients[1].data(), coefficient_count},
              {quantized[1].data(), coefficient_count});
        if (!block_status.ok()) {
          return block_status;
        }
        block_status = DequantizeAcBlock(
          strategy,
          result.quantizer_,
          quantization_decision.raw_quant,
          {.channel = XybChannel::kY},
          {quantized[1].data(), coefficient_count},
          {coefficients[1].data(), coefficient_count});
        if (!block_status.ok()) {
          return block_status;
        }

        const std::array<float, 3> factors =
          result.color_correlation_.AcFactors(block_x / 8, block_y / 8);
        for (size_t index = 0; index < coefficient_count; ++index) {
          coefficients[0][index] -= factors[0] * coefficients[1][index];
          coefficients[2][index] -= factors[2] * coefficients[1][index];
        }

        for (size_t channel : {size_t{0}, size_t{2}}) {
          const PlaneF32View dc_view{
            .data = dc.data(),
            .extent = info->covered_blocks,
            .stride = info->covered_blocks.width,
          };
          block_status = ConvertLowFrequenciesToDc(
            strategy,
            {coefficients[channel].data(), coefficient_count},
            dc_view);
          if (!block_status.ok()) {
            return block_status;
          }
          for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
            for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
              result.dc_[channel][
                (block_y + dy) * block_extent.width + block_x + dx] =
                  dc[dy * info->covered_blocks.width + dx];
            }
          }

          block_status = QuantizeAcBlock(
            strategy,
            result.quantizer_,
            quantization_decision.raw_quant,
            {
              .channel = kChannels[channel],
              .matrix_multiplier = MatrixMultiplier(channel, profile),
            },
            {coefficients[channel].data(), coefficient_count},
            {quantized[channel].data(), coefficient_count});
          if (!block_status.ok()) {
            return block_status;
          }
        }

        const size_t group_offset =
          result.group_used_coefficient_count_[group_index];
        if (coefficient_count >
            kVarDctAcGroupCoefficientCapacity - group_offset) {
          return Status::Internal(
            "VarDCT AC group coefficient storage overflowed");
        }
        for (size_t channel = 0; channel < 3; ++channel) {
          std::copy_n(
            quantized[channel].begin(),
            coefficient_count,
            result.ac_coefficients_.begin() +
              result.AcGroupChannelOffset(group_index, channel) +
              group_offset);
        }
        result.group_used_coefficient_count_[group_index] +=
          coefficient_count;
        ++prepared_index;
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }
    if (prepared != nullptr && prepared_index != prepared->transforms.size()) {
      return Status::InvalidArgument(
        "Prepared coefficients contain extra transforms");
    }

    const Image3I32View quantized_dc{{
      PlaneI32View{
        result.quantized_dc_[0].data(), block_extent, block_extent.width},
      PlaneI32View{
        result.quantized_dc_[1].data(), block_extent, block_extent.width},
      PlaneI32View{
        result.quantized_dc_[2].data(), block_extent, block_extent.width},
    }};
    const Image3FView reconstructed_dc{{
      PlaneF32View{result.dc_[0].data(), block_extent, block_extent.width},
      PlaneF32View{result.dc_[1].data(), block_extent, block_extent.width},
      PlaneF32View{result.dc_[2].data(), block_extent, block_extent.width},
    }};
    status = QuantizeDcCoefficients(
      result.dc(),
      result.quantizer_,
      {.quantized = quantized_dc, .reconstructed = reconstructed_dc});
    if (!status.ok()) {
      return status;
    }
    if (!result.valid()) {
      return Status::Internal(
        "Coefficient coding did not produce a valid encoder frame");
    }

    *out = std::move(result);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate coefficient coding scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Coefficient coding dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace prepared_coefficients_internal

Status ComputeQuantizedCoefficients(
  ConstImage3FView opsin,
  VarDctFrameInput input,
  SimpleVarDctCodestreamProfile profile,
  VarDctEncoderFrame* out,
  AcCoefficientDecisionMode decision_mode) {

  return prepared_coefficients_internal::ComputeQuantizedCoefficientsImpl(
    opsin, nullptr, input, profile, out, decision_mode);
}

Status prepared_coefficients_internal::ComputeQuantizedCoefficientsPrepared(
  const PreparedForwardDctCoefficients& prepared,
  VarDctFrameInput input,
  SimpleVarDctCodestreamProfile profile,
  VarDctEncoderFrame* out,
  AcCoefficientDecisionMode decision_mode) {

  return ComputeQuantizedCoefficientsImpl(
    {}, &prepared, input, profile, out, decision_mode);
}

Status ReconstructQuantizedCoefficients(
  const VarDctEncoderFrame& frame,
  Image3FView output) {

  if (!frame.valid() || !output.valid()) {
    return Status::InvalidArgument(
      "Coefficient reconstruction inputs are invalid");
  }

  const Extent2D block_extent = frame.geometry_.block_grid().blocks;
  if (output.extent() != frame.geometry_.padded_frame()) {
    return Status::InvalidArgument(
      "Coefficient reconstruction output has the wrong extent");
  }

  try {
    Image3FBuffer result(output.extent());
    const Image3FView result_view = result.view();

    ManagedVector<size_t> group_offsets(frame.ac_group_count(), 0);
    const ConstImage3FView frame_dc = frame.dc();
    const Status reconstruct_status = frame.strategies_.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr || !SupportsCpuDct(strategy)) {
          return Status::InvalidArgument(
            "Stored coefficient strategy is unsupported");
        }

        const size_t group_x = block_x / kVarDctAcGroupBlockDimension;
        const size_t group_y = block_y / kVarDctAcGroupBlockDimension;
        const size_t group_index =
          group_y * frame.ac_group_extent_.width + group_x;
        const size_t group_offset = group_offsets[group_index];
        const size_t coefficient_count = info->coefficient_count();
        if (group_offset >
              frame.group_used_coefficient_count_[group_index] ||
            coefficient_count >
            frame.group_used_coefficient_count_[group_index] - group_offset) {
          return Status::Internal(
            "Stored VarDCT AC group ended inside a transform");
        }

        const int32_t raw_quant = frame.raw_quant_field_[
          block_y * block_extent.width + block_x];
        std::array<ManagedVector<float>, 3> coefficients;
        for (size_t channel = 0; channel < coefficients.size(); ++channel) {
          coefficients[channel].resize(coefficient_count);
          const size_t source =
            frame.AcGroupChannelOffset(group_index, channel) + group_offset;
          Status status = DequantizeAcBlock(
            strategy,
            frame.quantizer_,
            raw_quant,
            {
              .channel = kChannels[channel],
              .matrix_multiplier = MatrixMultiplier(
                channel,
                frame.profile_),
            },
            std::span<const int32_t>(
              frame.ac_coefficients_.data() + source,
              coefficient_count),
            coefficients[channel]);
          if (!status.ok()) {
            return status;
          }
        }

        const std::array<float, 3> factors =
          frame.color_correlation_.AcFactors(block_x / 8, block_y / 8);
        for (size_t index = 0; index < coefficient_count; ++index) {
          coefficients[0][index] += factors[0] * coefficients[1][index];
          coefficients[2][index] += factors[2] * coefficients[1][index];
        }

        for (size_t channel = 0; channel < coefficients.size(); ++channel) {
          ManagedVector<float> dc(
            info->covered_blocks.width * info->covered_blocks.height);
          for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
            for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
              dc[dy * info->covered_blocks.width + dx] =
                frame_dc.plane[channel].Row(block_y + dy)[block_x + dx];
            }
          }
          const ConstPlaneF32View dc_view{
            .data = dc.data(),
            .extent = info->covered_blocks,
            .stride = info->covered_blocks.width,
          };
          Status status = ConvertDcToLowFrequencies(
            strategy,
            dc_view,
            coefficients[channel]);
          if (!status.ok()) {
            return status;
          }

          ManagedVector<float> pixels(coefficient_count);
          status = InverseDctCpu(strategy, coefficients[channel], pixels);
          if (!status.ok()) {
            return status;
          }
          CopyPixelsToImage(
            pixels,
            block_x,
            block_y,
            info->pixel_extent(),
            result_view.plane[channel]);
        }

        group_offsets[group_index] += coefficient_count;
        return Status::Ok();
      });
    if (!reconstruct_status.ok()) {
      return reconstruct_status;
    }
    if (group_offsets != frame.group_used_coefficient_count_) {
      return Status::Internal(
        "Stored VarDCT AC groups contain unconsumed coefficients");
    }

    CopyImage(result.const_view(), output);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate coefficient reconstruction scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Coefficient reconstruction dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace gjxl
