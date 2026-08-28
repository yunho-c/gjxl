// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/ac_strategy_search.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "codec/ac_strategy_search_internal.h"
#include "codec/ac_strategy_search_policy.h"
#include "codec/quantization.h"
#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "gpu/buffer.h"
#include "gpu/ops/ac_strategy.h"

namespace gjxl {
namespace {

constexpr size_t kColorTileBlockDimension =
  kColorTileDimension / kJxlBlockDimension;
static_assert(kColorTileBlockDimension == 8);

bool TryMultiply(size_t left, size_t right, size_t* result) {
  if (result == nullptr ||
      (right != 0 && left > std::numeric_limits<size_t>::max() / right)) {
    return false;
  }
  *result = left * right;
  return true;
}

Status ValidateSearchInputs(
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  Extent2D* block_extent,
  Extent2D* tile_extent,
  size_t* pixel_count,
  size_t* block_count) {
  if (out == nullptr || block_extent == nullptr || tile_extent == nullptr ||
      pixel_count == nullptr || block_count == nullptr) {
    return Status::InvalidArgument("GPU AC-strategy search output is null");
  }
  if (!opsin.valid() || opsin.width() % kJxlBlockDimension != 0 ||
      opsin.height() % kJxlBlockDimension != 0) {
    return Status::InvalidArgument(
      "GPU AC-strategy search requires a padded opsin image");
  }
  *block_extent = {
    opsin.width() / kJxlBlockDimension,
    opsin.height() / kJxlBlockDimension,
  };
  if (!opsin.extent().try_area(pixel_count) ||
      !block_extent->try_area(block_count)) {
    return Status::InvalidArgument(
      "GPU AC-strategy search dimensions are too large");
  }
  if (!quant_field.valid() || quant_field.extent != *block_extent ||
      !pixel_mask.valid() || pixel_mask.extent != opsin.extent() ||
      !color_correlation.valid()) {
    return Status::InvalidArgument(
      "GPU AC-strategy search fields have invalid geometry");
  }
  *tile_extent = {
    opsin.width() / kColorTileDimension +
      static_cast<size_t>(opsin.width() % kColorTileDimension != 0),
    opsin.height() / kColorTileDimension +
      static_cast<size_t>(opsin.height() % kColorTileDimension != 0),
  };
  if (color_correlation.tile_extent() != *tile_extent ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "GPU AC-strategy search options or color map are invalid");
  }
  constexpr size_t kUint32Maximum = std::numeric_limits<uint32_t>::max();
  if (opsin.width() > kUint32Maximum || opsin.height() > kUint32Maximum ||
      *pixel_count > kUint32Maximum || block_extent->width > kUint32Maximum ||
      block_extent->height > kUint32Maximum) {
    return Status::InvalidArgument(
      "GPU AC-strategy search exceeds 32-bit indexing limits");
  }
  return Status::Ok();
}

std::vector<float> PackOpsin(ConstImage3FView opsin, size_t pixel_count) {
  std::vector<float> packed(3 * pixel_count);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < opsin.height(); ++y) {
      std::copy_n(opsin.plane[channel].Row(y),
        opsin.width(),
        packed.begin() + channel * pixel_count + y * opsin.width());
    }
  }
  return packed;
}

std::vector<float> PackPlane(ConstPlaneF32View plane) {
  size_t pixel_count = 0;
  (void)plane.extent.try_area(&pixel_count);
  std::vector<float> packed(pixel_count);
  for (size_t y = 0; y < plane.extent.height; ++y) {
    std::copy_n(plane.Row(y),
      plane.extent.width,
      packed.begin() + y * plane.extent.width);
  }
  return packed;
}

Status PackMatrices(AcStrategyType strategy, std::vector<float>* matrices) {
  if (matrices == nullptr) {
    return Status::Internal("GPU AC-strategy matrix output is null");
  }
  const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
  const size_t coefficient_count = info->coefficient_count();
  matrices->resize(kAcStrategyCostMatrixCount * coefficient_count);
  for (size_t channel = 0; channel < 3; ++channel) {
    QuantizationMatrixView matrix;
    Status status = GetDefaultQuantizationMatrix(
      strategy, static_cast<XybChannel>(channel), &matrix);
    if (!status.ok()) {
      return status;
    }
    std::copy(matrix.dequant.begin(),
      matrix.dequant.end(),
      matrices->begin() + channel * coefficient_count);
    std::copy(matrix.inverse_dequant.begin(),
      matrix.inverse_dequant.end(),
      matrices->begin() + (3 + channel) * coefficient_count);
  }
  return Status::Ok();
}

Status MakeCandidates(
  const ac_strategy_internal::CandidateStage& staged,
  Extent2D block_extent,
  Extent2D tile_extent,
  ConstPlaneF32View quant_field,
  const ColorCorrelationMap& color_correlation,
  bool device_quant_norm,
  std::vector<AcStrategyCandidate>* candidates) {
  if (candidates == nullptr) {
    return Status::Internal("GPU AC-strategy candidate output is null");
  }
  const Extent2D covered = GetAcStrategyInfo(staged.strategy)->covered_blocks;
  size_t candidate_count = 0;
  for (size_t tile_y = 0; tile_y < tile_extent.height; ++tile_y) {
    const size_t block_y = tile_y * kColorTileBlockDimension;
    const size_t tile_height =
      std::min(kColorTileBlockDimension, block_extent.height - block_y);
    for (size_t tile_x = 0; tile_x < tile_extent.width; ++tile_x) {
      const size_t block_x = tile_x * kColorTileBlockDimension;
      const size_t tile_width =
        std::min(kColorTileBlockDimension, block_extent.width - block_x);
      if (tile_width < covered.width || tile_height < covered.height) {
        continue;
      }
      const size_t positions_x =
        (tile_width - covered.width) / staged.anchor_step + 1;
      const size_t positions_y =
        (tile_height - covered.height) / staged.anchor_step + 1;
      size_t tile_candidate_count = 0;
      if (!TryMultiply(positions_x, positions_y, &tile_candidate_count) ||
          candidate_count > std::numeric_limits<size_t>::max() -
                              tile_candidate_count) {
        return Status::InvalidArgument(
          "GPU AC-strategy candidate count overflows");
      }
      candidate_count += tile_candidate_count;
    }
  }
  if (candidate_count > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument(
      "GPU AC-strategy candidate count exceeds Metal limits");
  }
  candidates->reserve(candidate_count);

  for (size_t tile_y = 0; tile_y < tile_extent.height; ++tile_y) {
    const size_t block_y = tile_y * kColorTileBlockDimension;
    const size_t tile_height =
      std::min(kColorTileBlockDimension, block_extent.height - block_y);
    for (size_t tile_x = 0; tile_x < tile_extent.width; ++tile_x) {
      const size_t block_x = tile_x * kColorTileBlockDimension;
      const size_t tile_width =
        std::min(kColorTileBlockDimension, block_extent.width - block_x);
      const std::array<float, 3> cfl =
        color_correlation.AcFactors(tile_x, tile_y);
      for (size_t local_y = 0; local_y + covered.height <= tile_height;
        local_y += staged.anchor_step) {
        for (size_t local_x = 0; local_x + covered.width <= tile_width;
          local_x += staged.anchor_step) {
          float quant_norm = 1.0f;
          if (!device_quant_norm) {
            Status status = ComputeAcStrategyQuantNorm(staged.strategy,
              block_x + local_x,
              block_y + local_y,
              quant_field,
              &quant_norm);
            if (!status.ok()) {
              return status;
            }
          }
          candidates->push_back({
            .block_x = static_cast<uint32_t>(block_x + local_x),
            .block_y = static_cast<uint32_t>(block_y + local_y),
            .quant_norm = quant_norm,
            .entropy_multiplier = staged.entropy_multiplier,
            .cfl_x = cfl[0],
            .cfl_b = cfl[2],
          });
        }
      }
    }
  }
  return Status::Ok();
}

Status AllocateAndUpload(
  GpuBackend& gpu,
  const void* data,
  size_t size_bytes,
  std::unique_ptr<DeviceBuffer>* buffer) {
  Status status = gpu.Allocate(size_bytes, buffer);
  if (!status.ok()) {
    return status;
  }
  return gpu.CopyHostToDevice(**buffer, data, size_bytes);
}

struct StrategyResources {
  ac_strategy_internal::CandidateStage staged;
  std::vector<AcStrategyCandidate> candidates;
  std::vector<float> matrices;
  std::vector<float> costs;
  std::unique_ptr<DeviceBuffer> device_candidates;
  std::unique_ptr<DeviceBuffer> device_matrices;
  std::unique_ptr<DeviceBuffer> device_costs;
};

}  // namespace

static Status FindAcStrategyGridGpuImpl(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  const ResidentAcStrategySearchInputs* resident,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  AcStrategyGpuSearchStats* stats) {
  Extent2D block_extent;
  Extent2D tile_extent;
  size_t pixel_count = 0;
  size_t block_count = 0;
  Status status = ValidateSearchInputs(opsin,
    quant_field,
    pixel_mask,
    color_correlation,
    options,
    out,
    &block_extent,
    &tile_extent,
    &pixel_count,
    &block_count);
  if (!status.ok()) {
    return status;
  }
  if (resident != nullptr) {
    status = ValidateDeviceImage3View(resident->opsin, gpu.id());
    if (!status.ok()) return status;
    if (std::ranges::any_of(
          resident->opsin.plane,
          [&](ConstDevicePlaneView plane) {
            return plane.element_type != DeviceElementType::kF32 ||
              plane.extent != opsin.extent();
          }) ||
        resident->quant_field.element_type != DeviceElementType::kF32 ||
        resident->quant_field.extent != block_extent ||
        resident->pixel_mask.element_type != DeviceElementType::kF32 ||
        resident->pixel_mask.extent != opsin.extent()) {
      return Status::InvalidArgument(
          "Resident GPU AC-strategy inputs have invalid geometry");
    }
    DeviceMemoryRange range;
    status = ComputeDevicePlaneRange(
        resident->quant_field, gpu.id(), &range);
    if (status.ok()) {
      status = ComputeDevicePlaneRange(
          resident->pixel_mask, gpu.id(), &range);
    }
    if (!status.ok()) return status;
  }

  try {
    const std::vector<float> packed_opsin = resident == nullptr
      ? PackOpsin(opsin, pixel_count) : std::vector<float>{};
    const std::vector<float> packed_mask = resident == nullptr
      ? PackPlane(pixel_mask) : std::vector<float>{};
    std::unique_ptr<DeviceBuffer> device_opsin;
    std::unique_ptr<DeviceBuffer> device_mask;
    if (resident == nullptr) {
      status = AllocateAndUpload(gpu,
        packed_opsin.data(),
        packed_opsin.size() * sizeof(float),
        &device_opsin);
      if (!status.ok()) {
        return status;
      }
      status = AllocateAndUpload(gpu,
        packed_mask.data(),
        packed_mask.size() * sizeof(float),
        &device_mask);
      if (!status.ok()) {
        return status;
      }
    }

    constexpr const auto& kStages =
      ac_strategy_internal::kCandidateStages;
    std::array<StrategyResources, kStages.size()> resources;
    std::array<std::vector<float>, kAcStrategyCount> cost_storage;
    AcStrategyGpuSearchStats result_stats;
    size_t maximum_packed_bytes = 0;
    size_t maximum_rate_bytes = 0;
    for (size_t i = 0; i < kStages.size(); ++i) {
      StrategyResources& resource = resources[i];
      resource.staged = kStages[i];
      status = MakeCandidates(resource.staged,
        block_extent,
        tile_extent,
        quant_field,
        color_correlation,
        resident != nullptr,
        &resource.candidates);
      if (!status.ok()) {
        return status;
      }
      status = PackMatrices(resource.staged.strategy, &resource.matrices);
      if (!status.ok()) {
        return status;
      }
      resource.costs.resize(resource.candidates.size());
      const size_t strategy_index =
        static_cast<size_t>(resource.staged.strategy);
      cost_storage[strategy_index].assign(
        block_count, std::numeric_limits<float>::quiet_NaN());
      result_stats.candidate_counts[strategy_index] =
        resource.candidates.size();
      result_stats.total_candidate_count += resource.candidates.size();

      if (resource.candidates.empty()) {
        continue;
      }
      status = AllocateAndUpload(gpu,
        resource.candidates.data(),
        resource.candidates.size() * sizeof(AcStrategyCandidate),
        &resource.device_candidates);
      if (!status.ok()) {
        return status;
      }
      status = AllocateAndUpload(gpu,
        resource.matrices.data(),
        resource.matrices.size() * sizeof(float),
        &resource.device_matrices);
      if (!status.ok()) {
        return status;
      }
      status = gpu.Allocate(
        resource.costs.size() * sizeof(float), &resource.device_costs);
      if (!status.ok()) {
        return status;
      }

      const size_t coefficient_count =
        GetAcStrategyInfo(resource.staged.strategy)->coefficient_count();
      size_t packed_elements = 0;
      size_t packed_bytes = 0;
      size_t rate_bytes = 0;
      if (!TryMultiply(resource.candidates.size(), 3, &packed_elements) ||
          !TryMultiply(packed_elements, coefficient_count, &packed_elements) ||
          !TryMultiply(packed_elements, sizeof(float), &packed_bytes) ||
          !TryMultiply(resource.candidates.size(),
            3 * kAcStrategyRateScratchBytesPerChannel,
            &rate_bytes)) {
        return Status::InvalidArgument(
          "GPU AC-strategy search scratch size overflows");
      }
      maximum_packed_bytes = std::max(maximum_packed_bytes, packed_bytes);
      maximum_rate_bytes = std::max(maximum_rate_bytes, rate_bytes);
    }

    std::unique_ptr<DeviceBuffer> scratch_a;
    std::unique_ptr<DeviceBuffer> scratch_b;
    std::unique_ptr<DeviceBuffer> rate_scratch;
    status = gpu.Allocate(maximum_packed_bytes, &scratch_a);
    if (!status.ok()) {
      return status;
    }
    status = gpu.Allocate(maximum_packed_bytes, &scratch_b);
    if (!status.ok()) {
      return status;
    }
    status = gpu.Allocate(maximum_rate_bytes, &rate_scratch);
    if (!status.ok()) {
      return status;
    }

    std::array<AcStrategyCandidateBatch, kStages.size()> batches;
    for (size_t i = 0; i < resources.size(); ++i) {
      StrategyResources& resource = resources[i];
      batches[i] = {
        .strategy = resource.staged.strategy,
        .opsin = device_opsin.get(),
        .pixel_mask = device_mask.get(),
        .matrices = resource.device_matrices.get(),
        .candidates = resource.device_candidates.get(),
        .resident_opsin = resident == nullptr
          ? ConstDeviceImage3View{} : resident->opsin,
        .resident_pixel_mask = resident == nullptr
          ? ConstDevicePlaneView{} : resident->pixel_mask,
        .resident_quant_field = resident == nullptr
          ? ConstDevicePlaneView{} : resident->quant_field,
        .scratch_a = scratch_a.get(),
        .scratch_b = scratch_b.get(),
        .rate_scratch = rate_scratch.get(),
        .costs = resource.device_costs.get(),
        .pixel_extent = opsin.extent(),
        .opsin_row_stride = opsin.width(),
        .opsin_plane_stride = pixel_count,
        .pixel_mask_row_stride = pixel_mask.extent.width,
        .candidate_count = resource.candidates.size(),
        .butteraugli_target = options.butteraugli_target,
      };
    }
    std::unique_ptr<GpuSubmission> submission;
    status = EvaluateAcStrategyCandidateBatches(
      gpu, batches, &submission);
    if (!status.ok()) {
      return status;
    }
    if (submission == nullptr) {
      return Status::Internal(
        "GPU AC-strategy search returned no submission");
    }
    status = submission->Wait();
    if (!status.ok()) {
      return status;
    }

    ac_strategy_internal::CandidateCostTableView table{
      .block_extent = block_extent,
    };
    for (StrategyResources& resource : resources) {
      if (!resource.candidates.empty()) {
        status = gpu.CopyDeviceToHost(*resource.device_costs,
          resource.costs.data(),
          resource.costs.size() * sizeof(float));
        if (!status.ok()) {
          return status;
        }
      }
      const size_t strategy_index =
        static_cast<size_t>(resource.staged.strategy);
      for (size_t i = 0; i < resource.candidates.size(); ++i) {
        const AcStrategyCandidate& candidate = resource.candidates[i];
        cost_storage[strategy_index][static_cast<size_t>(candidate.block_y) *
                                       block_extent.width +
                                     candidate.block_x] = resource.costs[i];
      }
      table.strategy_costs[strategy_index] = cost_storage[strategy_index];
    }

    status = ac_strategy_internal::FindAcStrategyGridFromCandidateCosts(
      opsin, quant_field, pixel_mask, color_correlation, options, table, out);
    if (!status.ok()) {
      return status;
    }
    if (stats != nullptr) {
      *stats = result_stats;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU AC-strategy search state");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU AC-strategy search dimensions are too large");
  }
}

Status FindAcStrategyGridGpu(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  AcStrategyGpuSearchStats* stats) {

  return FindAcStrategyGridGpuImpl(
      gpu, opsin, quant_field, pixel_mask, color_correlation, nullptr,
      options, out, stats);
}

Status FindAcStrategyGridGpuResident(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  ResidentAcStrategySearchInputs resident,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  AcStrategyGpuSearchStats* stats) {

  return FindAcStrategyGridGpuImpl(
      gpu, opsin, quant_field, pixel_mask, color_correlation, &resident,
      options, out, stats);
}

}  // namespace gjxl
