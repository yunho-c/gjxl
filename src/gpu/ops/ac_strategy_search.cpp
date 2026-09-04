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
#include "gpu/ops/ac_strategy_search_profile_internal.h"
#include "gpu/scratch.h"

namespace gjxl {
namespace {

constexpr size_t kColorTileBlockDimension =
  kColorTileDimension / kJxlBlockDimension;
constexpr size_t kArenaAlignment = 256;
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
  bool resident_fields,
  bool resident_cfl,
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
  if ((!resident_fields &&
       (!quant_field.valid() || quant_field.extent != *block_extent ||
        !pixel_mask.valid() || pixel_mask.extent != opsin.extent())) ||
      (!resident_cfl && !color_correlation.valid())) {
    return Status::InvalidArgument(
      "GPU AC-strategy search fields have invalid geometry");
  }
  *tile_extent = {
    opsin.width() / kColorTileDimension +
      static_cast<size_t>(opsin.width() % kColorTileDimension != 0),
    opsin.height() / kColorTileDimension +
      static_cast<size_t>(opsin.height() % kColorTileDimension != 0),
  };
  if ((!resident_cfl &&
       color_correlation.tile_extent() != *tile_extent) ||
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
  bool device_cfl,
  std::vector<AcStrategyCandidate>* candidates) {
  if (candidates == nullptr) {
    return Status::Internal("GPU AC-strategy candidate output is null");
  }
  candidates->clear();
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
      const std::array<float, 3> cfl = device_cfl
        ? std::array<float, 3>{}
        : color_correlation.AcFactors(tile_x, tile_y);
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

Status PlanArenaBytes(size_t size_bytes, size_t* capacity_bytes) {
  if (capacity_bytes == nullptr || size_bytes == 0 ||
      *capacity_bytes >
        std::numeric_limits<size_t>::max() - (kArenaAlignment - 1)) {
    return Status::InvalidArgument(
      "GPU AC-strategy arena plan is invalid");
  }
  const size_t aligned =
    (*capacity_bytes + kArenaAlignment - 1) & ~(kArenaAlignment - 1);
  if (aligned > std::numeric_limits<size_t>::max() - size_bytes) {
    return Status::InvalidArgument(
      "GPU AC-strategy arena size overflows");
  }
  *capacity_bytes = aligned + size_bytes;
  return Status::Ok();
}

Status AllocateArenaBytes(DeviceScratchArena& arena,
                          size_t size_bytes,
                          DevicePlaneView* view) {
  return arena.AllocatePlane(DeviceElementType::kU8, {size_bytes, 1},
                             size_bytes, kArenaAlignment, view);
}

struct StrategyResources {
  ac_strategy_internal::CandidateStage staged;
  std::vector<AcStrategyCandidate> candidates;
  std::vector<float> matrices;
  std::vector<float> costs;
  DevicePlaneView device_candidates;
  DevicePlaneView device_matrices;
  DevicePlaneView device_costs;
};

}  // namespace

namespace ac_strategy_search_internal {

struct Prepared {
  GpuBackend* backend = nullptr;
  std::array<StrategyResources,
             ac_strategy_internal::kCandidateStages.size()> resources;
  std::array<std::vector<float>, kAcStrategyCount> cost_storage;
  DeviceScratchArena input_arena;
  DeviceScratchArena resource_arena;
  DevicePlaneView device_opsin;
  DevicePlaneView device_mask;
  DevicePlaneView scratch_a;
  DevicePlaneView scratch_b;
  DevicePlaneView rate_scratch;
};

}  // namespace ac_strategy_search_internal

PreparedAcStrategySearch::PreparedAcStrategySearch() = default;
PreparedAcStrategySearch::~PreparedAcStrategySearch() = default;

static Status FindAcStrategyGridGpuImpl(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  const ResidentAcStrategySearchInputs* resident,
  ac_strategy_search_internal::Prepared* prepared,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  AcStrategyGpuSearchStats* stats,
  gpu_profile_internal::GpuProfilingSession* profiling_session) {
  Extent2D block_extent;
  Extent2D tile_extent;
  size_t pixel_count = 0;
  size_t block_count = 0;
  const bool resident_cfl = resident != nullptr &&
    resident->y_to_x.buffer != nullptr && resident->y_to_b.buffer != nullptr;
  Status status = ValidateSearchInputs(opsin,
    quant_field,
    pixel_mask,
    color_correlation,
    resident != nullptr,
    resident_cfl,
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
        resident->pixel_mask.extent != opsin.extent() ||
        ((resident->y_to_x.buffer != nullptr) !=
         (resident->y_to_b.buffer != nullptr)) ||
        (resident_cfl &&
         (resident->y_to_x.element_type != DeviceElementType::kI8 ||
          resident->y_to_b.element_type != DeviceElementType::kI8 ||
          resident->y_to_x.extent != tile_extent ||
          resident->y_to_b.extent != tile_extent))) {
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
    if (status.ok() && resident_cfl) {
      status = ComputeDevicePlaneRange(resident->y_to_x, gpu.id(), &range);
    }
    if (status.ok() && resident_cfl) {
      status = ComputeDevicePlaneRange(resident->y_to_b, gpu.id(), &range);
    }
    if (!status.ok()) return status;
  }
  auto* strategy_profiler = profiling_session == nullptr
    ? nullptr
    : dynamic_cast<
        gpu_profile_internal::GpuAcStrategyEvaluationProfiler*>(&gpu);
  auto* submission_profiler = profiling_session == nullptr
    ? nullptr
    : dynamic_cast<gpu_profile_internal::GpuSubmissionProfiler*>(&gpu);
  if (profiling_session != nullptr &&
      (resident == nullptr || strategy_profiler == nullptr ||
       submission_profiler == nullptr)) {
    return Status::Unavailable(
      "GPU AC-strategy search profiling is unavailable");
  }

  try {
    const auto preparation_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    ac_strategy_search_internal::Prepared local_prepared;
    ac_strategy_search_internal::Prepared& state =
      prepared == nullptr ? local_prepared : *prepared;
    if (state.backend != nullptr && state.backend != &gpu) {
      state = ac_strategy_search_internal::Prepared{};
    }
    state.backend = &gpu;
    const std::vector<float> packed_opsin = resident == nullptr
      ? PackOpsin(opsin, pixel_count) : std::vector<float>{};
    const std::vector<float> packed_mask = resident == nullptr
      ? PackPlane(pixel_mask) : std::vector<float>{};
    size_t input_capacity = 0;
    if (resident == nullptr) {
      status = PlanArenaBytes(
        packed_opsin.size() * sizeof(float), &input_capacity);
      if (status.ok()) {
        status = PlanArenaBytes(
          packed_mask.size() * sizeof(float), &input_capacity);
      }
      if (!status.ok()) return status;
    }

    constexpr const auto& kStages =
      ac_strategy_internal::kCandidateStages;
    auto& resources = state.resources;
    auto& cost_storage = state.cost_storage;
    AcStrategyGpuSearchStats result_stats;
    size_t maximum_packed_bytes = 0;
    size_t maximum_rate_bytes = 0;
    size_t resource_capacity = 0;
    for (size_t i = 0; i < kStages.size(); ++i) {
      StrategyResources& resource = resources[i];
      resource.staged = kStages[i];
      status = MakeCandidates(resource.staged,
        block_extent,
        tile_extent,
        quant_field,
        color_correlation,
        resident != nullptr,
        resident_cfl,
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
      status = PlanArenaBytes(
        resource.candidates.size() * sizeof(AcStrategyCandidate),
        &resource_capacity);
      if (status.ok()) {
        status = PlanArenaBytes(
          resource.matrices.size() * sizeof(float), &resource_capacity);
      }
      if (status.ok()) {
        status = PlanArenaBytes(
          resource.costs.size() * sizeof(float), &resource_capacity);
      }
      if (!status.ok()) return status;

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

    status = PlanArenaBytes(maximum_packed_bytes, &resource_capacity);
    if (status.ok()) {
      status = PlanArenaBytes(maximum_packed_bytes, &resource_capacity);
    }
    if (status.ok()) {
      status = PlanArenaBytes(maximum_rate_bytes, &resource_capacity);
    }
    if (!status.ok()) return status;

    if (resident == nullptr) {
      status = state.input_arena.Prepare(gpu, input_capacity);
      if (status.ok()) {
        status = AllocateArenaBytes(state.input_arena,
          packed_opsin.size() * sizeof(float), &state.device_opsin);
      }
      if (status.ok()) {
        status = AllocateArenaBytes(state.input_arena,
          packed_mask.size() * sizeof(float), &state.device_mask);
      }
      if (status.ok()) {
        status = gpu.CopyHostToDevice(*state.device_opsin.buffer,
          packed_opsin.data(), packed_opsin.size() * sizeof(float),
          state.device_opsin.offset_bytes);
      }
      if (status.ok()) {
        status = gpu.CopyHostToDevice(*state.device_mask.buffer,
          packed_mask.data(), packed_mask.size() * sizeof(float),
          state.device_mask.offset_bytes);
      }
      if (!status.ok()) return status;
    }

    status = state.resource_arena.Prepare(gpu, resource_capacity);
    if (!status.ok()) return status;
    for (StrategyResources& resource : resources) {
      if (resource.candidates.empty()) continue;
      status = AllocateArenaBytes(state.resource_arena,
        resource.candidates.size() * sizeof(AcStrategyCandidate),
        &resource.device_candidates);
      if (status.ok()) {
        status = AllocateArenaBytes(state.resource_arena,
          resource.matrices.size() * sizeof(float),
          &resource.device_matrices);
      }
      if (status.ok()) {
        status = AllocateArenaBytes(state.resource_arena,
          resource.costs.size() * sizeof(float), &resource.device_costs);
      }
      if (status.ok()) {
        status = gpu.CopyHostToDevice(*resource.device_candidates.buffer,
          resource.candidates.data(),
          resource.candidates.size() * sizeof(AcStrategyCandidate),
          resource.device_candidates.offset_bytes);
      }
      if (status.ok()) {
        status = gpu.CopyHostToDevice(*resource.device_matrices.buffer,
          resource.matrices.data(),
          resource.matrices.size() * sizeof(float),
          resource.device_matrices.offset_bytes);
      }
      if (!status.ok()) return status;
    }
    status = AllocateArenaBytes(
      state.resource_arena, maximum_packed_bytes, &state.scratch_a);
    if (status.ok()) {
      status = AllocateArenaBytes(
        state.resource_arena, maximum_packed_bytes, &state.scratch_b);
    }
    if (status.ok()) {
      status = AllocateArenaBytes(
        state.resource_arena, maximum_rate_bytes, &state.rate_scratch);
    }
    if (!status.ok()) return status;

    std::array<AcStrategyCandidateBatch, kStages.size()> batches;
    for (size_t i = 0; i < resources.size(); ++i) {
      StrategyResources& resource = resources[i];
      batches[i] = {
        .strategy = resource.staged.strategy,
        .opsin = resident == nullptr ? state.device_opsin.buffer : nullptr,
        .pixel_mask = resident == nullptr ? state.device_mask.buffer : nullptr,
        .matrices = resource.device_matrices.buffer,
        .candidates = resource.device_candidates.buffer,
        .resident_opsin = resident == nullptr
          ? ConstDeviceImage3View{} : resident->opsin,
        .resident_pixel_mask = resident == nullptr
          ? ConstDevicePlaneView{} : resident->pixel_mask,
        .resident_quant_field = resident == nullptr
          ? ConstDevicePlaneView{} : resident->quant_field,
        .resident_y_to_x = resident == nullptr
          ? ConstDevicePlaneView{} : resident->y_to_x,
        .resident_y_to_b = resident == nullptr
          ? ConstDevicePlaneView{} : resident->y_to_b,
        .scratch_a = state.scratch_a.buffer,
        .scratch_b = state.scratch_b.buffer,
        .rate_scratch = state.rate_scratch.buffer,
        .costs = resource.device_costs.buffer,
        .opsin_offset_bytes = resident == nullptr
          ? state.device_opsin.offset_bytes : 0,
        .pixel_mask_offset_bytes = resident == nullptr
          ? state.device_mask.offset_bytes : 0,
        .matrices_offset_bytes = resource.device_matrices.offset_bytes,
        .candidates_offset_bytes = resource.device_candidates.offset_bytes,
        .scratch_a_offset_bytes = state.scratch_a.offset_bytes,
        .scratch_b_offset_bytes = state.scratch_b.offset_bytes,
        .rate_scratch_offset_bytes = state.rate_scratch.offset_bytes,
        .costs_offset_bytes = resource.device_costs.offset_bytes,
        .pixel_extent = opsin.extent(),
        .opsin_row_stride = opsin.width(),
        .opsin_plane_stride = pixel_count,
        .pixel_mask_row_stride = resident == nullptr
          ? pixel_mask.extent.width : 0,
        .candidate_count = resource.candidates.size(),
        .butteraugli_target = options.butteraugli_target,
      };
    }
    std::unique_ptr<GpuSubmission> submission;
    if (profiling_session == nullptr) {
      status = EvaluateAcStrategyCandidateBatches(
        gpu, batches, &submission);
    } else {
      status = strategy_profiler->EvaluateAcStrategyCandidateBatchesProfiled(
        batches, profiling_session->mode(), &submission);
    }
    if (!status.ok()) {
      return status;
    }
    if (submission == nullptr) {
      return Status::Internal(
        "GPU AC-strategy search returned no submission");
    }
    if (profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.ac_strategy.prepare",
        gpu_profile_internal::GpuWallStageKind::kPreparation,
        preparation_begin);
      if (!status.ok()) return status;
    }
    const auto wait_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    status = submission->Wait();
    if (!status.ok()) {
      return status;
    }
    if (profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.ac_strategy.wait",
        gpu_profile_internal::GpuWallStageKind::kWait, wait_begin);
      if (!status.ok()) return status;
      gpu_profile_internal::GpuExecutionProfile child_profile;
      status = submission_profiler->ResolveGpuSubmissionProfile(
        *submission, "frontend.ac_strategy", profiling_session->mode(),
        &child_profile);
      if (status.ok()) {
        status = profiling_session->Append(std::move(child_profile));
      }
      if (!status.ok()) return status;
    }

    ac_strategy_internal::CandidateCostTableView table{
      .block_extent = block_extent,
    };
    const auto readback_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    for (StrategyResources& resource : resources) {
      if (!resource.candidates.empty()) {
        status = gpu.CopyDeviceToHost(*resource.device_costs.buffer,
          resource.costs.data(),
          resource.costs.size() * sizeof(float),
          resource.device_costs.offset_bytes);
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
    if (profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.ac_strategy.readback",
        gpu_profile_internal::GpuWallStageKind::kReadback,
        readback_begin);
      if (!status.ok()) return status;
    }

    const auto merge_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    status = ac_strategy_internal::FindAcStrategyGridFromCandidateCosts(
      opsin, quant_field, pixel_mask, color_correlation, options, table, out);
    if (!status.ok()) {
      return status;
    }
    if (profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.ac_strategy.merge",
        gpu_profile_internal::GpuWallStageKind::kHost, merge_begin);
      if (!status.ok()) return status;
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
      nullptr, options, out, stats, nullptr);
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
  AcStrategyGpuSearchStats* stats,
  PreparedAcStrategySearch* prepared) {

  if (prepared != nullptr && prepared->impl_ == nullptr) {
    try {
      prepared->impl_ =
        std::make_unique<ac_strategy_search_internal::Prepared>();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate prepared GPU AC-strategy search state");
    }
  }

  return FindAcStrategyGridGpuImpl(
      gpu, opsin, quant_field, pixel_mask, color_correlation, &resident,
      prepared == nullptr ? nullptr : prepared->impl_.get(),
      options, out, stats, nullptr);
}

Status gpu_profile_internal::FindAcStrategyGridGpuResidentProfiled(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  ResidentAcStrategySearchInputs resident,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  PreparedAcStrategySearch* prepared,
  GpuProfilingSession* profiling_session,
  AcStrategyGpuSearchStats* stats) {

  if (profiling_session == nullptr) {
    return Status::InvalidArgument(
      "GPU AC-strategy profiling session is null");
  }
  if (prepared != nullptr && prepared->impl_ == nullptr) {
    try {
      prepared->impl_ =
        std::make_unique<ac_strategy_search_internal::Prepared>();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate prepared GPU AC-strategy search state");
    }
  }
  return FindAcStrategyGridGpuImpl(
    gpu, opsin, quant_field, pixel_mask, color_correlation, &resident,
    prepared == nullptr ? nullptr : prepared->impl_.get(),
    options, out, stats, profiling_session);
}

}  // namespace gjxl
