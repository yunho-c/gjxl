// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/ac_strategy_search.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include "gpu/ops/ac_strategy_selection.h"
#include "gpu/ops/ac_strategy_search_test.h"
#include "gpu/ops/ac_strategy_search_profile_internal.h"
#include "gpu/scratch.h"
#include "gpu/ops/ac_strategy_storage_plan.h"
#include "core/managed_allocator.h"
#ifdef GJXL_FRONTIER_EXPERIMENT
#include "gpu/ops/ac_strategy_capture_internal.h"
#endif

namespace gjxl {
using resource_budget_internal::ManagedVector;
namespace {

thread_local bool cpu_selection_for_testing = false;

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
  Extent2D resident_opsin_extent,
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
  const Extent2D pixel_extent =
    resident_fields ? resident_opsin_extent : opsin.extent();
  if ((!resident_fields && !opsin.valid()) || pixel_extent.empty() ||
      pixel_extent.width % kJxlBlockDimension != 0 ||
      pixel_extent.height % kJxlBlockDimension != 0) {
    return Status::InvalidArgument(
      "GPU AC-strategy search requires a padded opsin image");
  }
  *block_extent = {pixel_extent.width / kJxlBlockDimension,
                   pixel_extent.height / kJxlBlockDimension};
  if (!pixel_extent.try_area(pixel_count) ||
      !block_extent->try_area(block_count)) {
    return Status::InvalidArgument(
      "GPU AC-strategy search dimensions are too large");
  }
  const auto empty_plane = [](ConstPlaneF32View plane) {
    return plane.data == nullptr && plane.extent == Extent2D{} && plane.stride == 0;
  };
  if ((!(resident_fields && empty_plane(quant_field)) &&
       (!quant_field.valid() || quant_field.extent != *block_extent)) ||
      (!(resident_fields && empty_plane(pixel_mask)) &&
       (!pixel_mask.valid() || pixel_mask.extent != pixel_extent)) ||
      (!resident_cfl && !color_correlation.valid())) {
    return Status::InvalidArgument(
      "GPU AC-strategy search fields have invalid geometry");
  }
  *tile_extent = {
    pixel_extent.width / kColorTileDimension +
      static_cast<size_t>(pixel_extent.width % kColorTileDimension != 0),
    pixel_extent.height / kColorTileDimension +
      static_cast<size_t>(pixel_extent.height % kColorTileDimension != 0),
  };
  if ((!resident_cfl &&
       color_correlation.tile_extent() != *tile_extent) ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "GPU AC-strategy search options or color map are invalid");
  }
  constexpr size_t kUint32Maximum = std::numeric_limits<uint32_t>::max();
  if (pixel_extent.width > kUint32Maximum ||
      pixel_extent.height > kUint32Maximum || *pixel_count > kUint32Maximum ||
      block_extent->width > kUint32Maximum ||
      block_extent->height > kUint32Maximum) {
    return Status::InvalidArgument(
      "GPU AC-strategy search exceeds 32-bit indexing limits");
  }
  return Status::Ok();
}

ManagedVector<float> PackOpsin(ConstImage3FView opsin, size_t pixel_count) {
  ManagedVector<float> packed(3 * pixel_count);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < opsin.height(); ++y) {
      std::copy_n(opsin.plane[channel].Row(y),
        opsin.width(),
        packed.begin() + channel * pixel_count + y * opsin.width());
    }
  }
  return packed;
}

ManagedVector<float> PackPlane(ConstPlaneF32View plane) {
  size_t pixel_count = 0;
  (void)plane.extent.try_area(&pixel_count);
  ManagedVector<float> packed(pixel_count);
  for (size_t y = 0; y < plane.extent.height; ++y) {
    std::copy_n(plane.Row(y),
      plane.extent.width,
      packed.begin() + y * plane.extent.width);
  }
  return packed;
}

Status PackMatrices(AcStrategyType strategy, ManagedVector<float>* matrices) {
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
  size_t candidate_count,
  ManagedVector<AcStrategyCandidate>* candidates) {
  if (candidates == nullptr) {
    return Status::Internal("GPU AC-strategy candidate output is null");
  }
  candidates->clear();
  const Extent2D covered = GetAcStrategyInfo(staged.strategy)->covered_blocks;
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
        ? std::array<float, 3>{} : color_correlation.AcFactors(tile_x, tile_y);
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
          if (candidates->size() == candidate_count) {
            return Status::Internal("GPU AC-strategy enumeration exceeds its storage plan");
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
  return candidates->size() == candidate_count ? Status::Ok() :
    Status::Internal("GPU AC-strategy enumeration disagrees with its storage plan");
}

// CUDA reuses its qualified packed arenas. Other backends preserve main's
// separate buffer capacities and allocation peaks, which their plans bound.
struct SearchDeviceBuffer : DevicePlaneView {
  std::unique_ptr<DeviceBuffer> separate;

  Status Prepare(GpuBackend& gpu, DeviceScratchArena& arena, size_t bytes) {
    if (gpu.kind() == BackendKind::kCuda) {
      return arena.AllocatePlane(DeviceElementType::kU8, {bytes, 1},
                                 bytes, kArenaAlignment, this);
    }
    if (bytes == 0) {
      *this = {};
      return Status::Ok();
    }
    if (separate == nullptr || separate->size_bytes() < bytes ||
        !gpu.owns(*separate)) {
      std::unique_ptr<DeviceBuffer> replacement;
      Status status = gpu.Allocate(bytes, &replacement);
      if (!status.ok()) return status;
      separate = std::move(replacement);
    }
    static_cast<DevicePlaneView&>(*this) =
      {separate.get(), 0, DeviceElementType::kU8, {bytes, 1}, bytes};
    return Status::Ok();
  }
};

struct StrategyResources {
  ac_strategy_internal::CandidateStage staged;
  ManagedVector<AcStrategyCandidate> candidates;
  ManagedVector<float> matrices;
  ManagedVector<float> costs;
  SearchDeviceBuffer device_candidates;
  SearchDeviceBuffer device_matrices;
  SearchDeviceBuffer device_costs;
};

}  // namespace

namespace ac_strategy_search_internal {

ScopedCpuSelectionForTesting::ScopedCpuSelectionForTesting() noexcept
    : previous_(cpu_selection_for_testing) {
  cpu_selection_for_testing = true;
}

ScopedCpuSelectionForTesting::~ScopedCpuSelectionForTesting() {
  cpu_selection_for_testing = previous_;
}

struct Prepared {
  GpuBackend* backend = nullptr;
  std::array<StrategyResources,
             ac_strategy_internal::kCandidateStages.size()> resources;
  std::array<ManagedVector<float>, kAcStrategyCount> cost_storage;
  DeviceScratchArena input_arena;
  DeviceScratchArena resource_arena;
  SearchDeviceBuffer device_opsin;
  SearchDeviceBuffer device_mask;
  SearchDeviceBuffer scratch_a;
  SearchDeviceBuffer scratch_b;
  SearchDeviceBuffer rate_scratch;
};

}  // namespace ac_strategy_search_internal

PreparedAcStrategySearch::PreparedAcStrategySearch() = default;
PreparedAcStrategySearch::~PreparedAcStrategySearch() = default;
void PreparedAcStrategySearch::Reset() noexcept { impl_.reset(); }

bool CanDeferAcStrategySearch(GpuBackend &gpu,
                              AcStrategySearchOptions options) {
  if (cpu_selection_for_testing || options.dense_dct32_search ||
      dynamic_cast<GpuAcStrategySelection *>(&gpu) == nullptr)
    return false;
#ifdef GJXL_FRONTIER_EXPERIMENT
  if (std::getenv("GJXL_FRONTIER_CAPTURE_DIR") ||
      std::getenv("GJXL_AC_SEARCH_EXPERIMENT"))
    return false;
#endif
  return true;
}

static Status FindAcStrategyGridGpuImpl(
    GpuBackend &gpu, ConstImage3FView opsin, ConstPlaneF32View quant_field,
    ConstPlaneF32View pixel_mask, const ColorCorrelationMap &color_correlation,
    const ResidentAcStrategySearchInputs *resident,
    ac_strategy_search_internal::Prepared *prepared,
    AcStrategySearchOptions options, AcStrategyGrid *out,
    AcStrategyGpuSearchStats *stats,
    gpu_profile_internal::GpuProfilingSession *profiling_session,
    DeferredAcStrategySearch *deferred = nullptr) {
  const resource_budget_internal::ResourceClassScope resource_class(
    resource_budget_internal::ResourceClass::kAcSearch);
  Extent2D block_extent;
  Extent2D tile_extent;
  size_t pixel_count = 0;
  size_t block_count = 0;
  const bool resident_cfl = resident != nullptr &&
    resident->y_to_x.buffer != nullptr && resident->y_to_b.buffer != nullptr;
  const Extent2D resident_opsin_extent =
    resident == nullptr ? Extent2D{} : resident->opsin.plane[0].extent;
  Status status = ValidateSearchInputs(opsin,
    resident_opsin_extent,
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
  ac_strategy_search_internal::StoragePlan storage_plan;
  status = ac_strategy_search_internal::ComputeStoragePlan(
    resident == nullptr ? opsin.extent() : resident_opsin_extent,
    resident != nullptr, &storage_plan, &gpu, options.dense_dct32_search);
  if (!status.ok()) return status;
  if (resident != nullptr) {
    status = ValidateDeviceImage3View(resident->opsin, gpu.id());
    if (!status.ok()) return status;
    if (std::ranges::any_of(resident->opsin.plane,
          [&](ConstDevicePlaneView plane) {
            return plane.element_type != DeviceElementType::kF32 ||
                   plane.extent != resident_opsin_extent;
          }) ||
        resident->quant_field.element_type != DeviceElementType::kF32 ||
        resident->quant_field.extent != block_extent ||
        resident->pixel_mask.element_type != DeviceElementType::kF32 ||
        resident->pixel_mask.extent != resident_opsin_extent ||
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
    const ManagedVector<float> packed_opsin = resident == nullptr
      ? PackOpsin(opsin, pixel_count) : ManagedVector<float>{};
    const ManagedVector<float> packed_mask = resident == nullptr
      ? PackPlane(pixel_mask) : ManagedVector<float>{};
    const size_t input_capacity = storage_plan.input_arena_bytes;

    auto *device_selector = resident != nullptr &&
                                    profiling_session == nullptr &&
                                    CanDeferAcStrategySearch(gpu, options)
                                ? dynamic_cast<GpuAcStrategySelection *>(&gpu)
                                : nullptr;
    if (deferred && (!device_selector || !prepared))
      return Status::Unavailable(
          "Deferred AC strategy selection is unavailable");
    const auto stages =
      ac_strategy_internal::CandidateStages(options.dense_dct32_search);
    auto& resources = state.resources;
    auto& cost_storage = state.cost_storage;
    AcStrategyGpuSearchStats result_stats;
    const size_t maximum_scratch_a_bytes = storage_plan.maximum_scratch_a_bytes;
    const size_t maximum_scratch_b_bytes = storage_plan.maximum_scratch_b_bytes;
    const size_t maximum_rate_bytes = storage_plan.maximum_rate_bytes;
    const size_t resource_capacity = storage_plan.resource_arena_bytes;
    for (size_t i = 0; i < ac_strategy_internal::kCandidateStages.size(); ++i) {
      StrategyResources& resource = resources[i];
      resource.staged = stages[i];
      status = MakeCandidates(resource.staged,
        block_extent,
        tile_extent,
        quant_field,
        color_correlation,
        resident != nullptr,
        resident_cfl,
        storage_plan.stages[i].candidate_count,
        &resource.candidates);
      if (!status.ok()) {
        return status;
      }
      status = PackMatrices(resource.staged.strategy, &resource.matrices);
      if (!status.ok()) {
        return status;
      }
      if (resource.matrices.size() != storage_plan.stages[i].matrix_bytes / sizeof(float)) {
        return Status::Internal("GPU AC-strategy matrices disagree with storage plan");
      }
      if (device_selector == nullptr || i == 0)
        resource.costs.resize(resource.candidates.size());
      const size_t strategy_index =
        static_cast<size_t>(resource.staged.strategy);
      if (device_selector == nullptr) {
        cost_storage[strategy_index].assign(
          block_count, std::numeric_limits<float>::quiet_NaN());
      }
      result_stats.candidate_counts[strategy_index] =
        resource.candidates.size();
      result_stats.total_candidate_count += resource.candidates.size();

    }

    if (resident == nullptr) {
      status = gpu.kind() == BackendKind::kCuda
        ? state.input_arena.Prepare(gpu, input_capacity) : Status::Ok();
      if (status.ok()) {
        status = state.device_opsin.Prepare(gpu, state.input_arena,
          packed_opsin.size() * sizeof(float));
      }
      if (status.ok()) {
        status = state.device_mask.Prepare(gpu, state.input_arena,
          packed_mask.size() * sizeof(float));
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

    status = gpu.kind() == BackendKind::kCuda
      ? state.resource_arena.Prepare(gpu, resource_capacity) : Status::Ok();
    if (!status.ok()) return status;
    result_stats.scratch = {
      maximum_scratch_a_bytes, maximum_scratch_b_bytes, maximum_rate_bytes};
    result_stats.resource_capacity_bytes = state.resource_arena.capacity_bytes();
    for (StrategyResources& resource : resources) {
      if (resource.candidates.empty()) continue;
      status = resource.device_candidates.Prepare(gpu, state.resource_arena,
          resource.candidates.size() * sizeof(AcStrategyCandidate));
      if (status.ok()) {
        status = resource.device_matrices.Prepare(gpu, state.resource_arena,
          resource.matrices.size() * sizeof(float));
      }
      if (status.ok()) {
        status = resource.device_costs.Prepare(gpu, state.resource_arena,
          resource.candidates.size() * sizeof(float));
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
    status = state.scratch_a.Prepare(gpu, state.resource_arena,
          maximum_scratch_a_bytes);
    if (maximum_scratch_b_bytes == 0) state.scratch_b = {};
    if (status.ok() && maximum_scratch_b_bytes != 0) {
      status = state.scratch_b.Prepare(gpu, state.resource_arena,
          maximum_scratch_b_bytes);
    }
    if (status.ok()) {
      status = state.rate_scratch.Prepare(gpu, state.resource_arena,
          maximum_rate_bytes);
    }
    if (!status.ok()) return status;

    std::array<AcStrategyCandidateBatch, ac_strategy_internal::kCandidateStages.size()> batches;
    for (size_t i = 0; i < resources.size(); ++i) {
      StrategyResources& resource = resources[i];
      batches[i] = {
        .strategy = resource.staged.strategy,
        .opsin = resident == nullptr ? state.device_opsin.buffer : nullptr,
        .pixel_mask = resident == nullptr ? state.device_mask.buffer : nullptr,
        .matrices = resource.device_matrices.buffer,
        .candidates = resource.device_candidates.buffer,
        .resident_opsin =
          resident == nullptr ? ConstDeviceImage3View{} : resident->opsin,
        .resident_pixel_mask =
          resident == nullptr ? ConstDevicePlaneView{} : resident->pixel_mask,
        .resident_quant_field =
          resident == nullptr ? ConstDevicePlaneView{} : resident->quant_field,
        .resident_y_to_x =
          resident == nullptr ? ConstDevicePlaneView{} : resident->y_to_x,
        .resident_y_to_b =
          resident == nullptr ? ConstDevicePlaneView{} : resident->y_to_b,
        .scratch_a = state.scratch_a.buffer,
        .scratch_b = state.scratch_b.buffer,
        .rate_scratch = state.rate_scratch.buffer,
        .costs = resource.device_costs.buffer,
        .opsin_offset_bytes =
          resident == nullptr ? state.device_opsin.offset_bytes : 0,
        .pixel_mask_offset_bytes =
          resident == nullptr ? state.device_mask.offset_bytes : 0,
        .matrices_offset_bytes = resource.device_matrices.offset_bytes,
        .candidates_offset_bytes = resource.device_candidates.offset_bytes,
        .scratch_a_offset_bytes = state.scratch_a.offset_bytes,
        .scratch_b_offset_bytes = state.scratch_b.offset_bytes,
        .rate_scratch_offset_bytes = state.rate_scratch.offset_bytes,
        .costs_offset_bytes = resource.device_costs.offset_bytes,
        .pixel_extent =
          resident == nullptr ? opsin.extent() : resident_opsin_extent,
        .opsin_row_stride = resident == nullptr
                              ? opsin.width()
                              : resident->opsin.plane[0].row_stride,
        .opsin_plane_stride = pixel_count,
        .pixel_mask_row_stride =
          resident == nullptr ? pixel_mask.extent.width : 0,
        .candidate_count = resource.candidates.size(),
        .butteraugli_target = options.butteraugli_target,
      };
    }
    if (deferred) {
      *deferred = {batches, {block_extent, state.rate_scratch.buffer, state.rate_scratch.offset_bytes}};
      result_stats.device_selection = true;
      if (stats)
        *stats = result_stats;
      return Status::Ok();
    }
    std::unique_ptr<GpuSubmission> submission;
    if (device_selector != nullptr) {
      status = device_selector->EvaluateAndSelectAcStrategyCandidateBatches(
        batches, {block_extent, state.rate_scratch.buffer, state.rate_scratch.offset_bytes}, &submission);
    } else if (profiling_session == nullptr) {
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
      if (!status.ok()) {
        (void)submission->Wait();
        return status;
      }
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
    if (device_selector != nullptr) {
      // Scoring no longer needs rate scratch. Its selected byte map and error
      // flags fit inside the existing DCT8 readback owner; no new backing is
      // needed beyond the conservative search storage plan.
      const size_t tile_count = tile_extent.width * tile_extent.height;
      const size_t bytes = block_count + tile_count;
      if (bytes > resources[0].costs.size() * sizeof(float))
        return Status::Internal("Device AC selection exceeds its readback owner");
      status = gpu.CopyDeviceToHost(*state.rate_scratch.buffer,
        resources[0].costs.data(), bytes, state.rate_scratch.offset_bytes);
      if (!status.ok()) return status;
      const auto* cells = reinterpret_cast<const uint8_t*>(resources[0].costs.data());
      for (size_t tile = 0; tile < tile_count; ++tile) {
        if (cells[block_count + tile] != 0)
          return Status::Internal("Device AC selection encountered an invalid candidate cost");
      }
      AcStrategyGrid result;
      status = AcStrategyGrid::Create(block_extent, &result);
      if (!status.ok()) return status;
      for (size_t y = 0; y < block_extent.height; ++y) {
        for (size_t x = 0; x < block_extent.width; ++x) {
          const uint8_t cell = cells[y * block_extent.width + x];
          if ((cell & 1u) != 0) {
            status = result.Set(x, y, static_cast<AcStrategyType>(cell >> 1));
            if (!status.ok()) return Status::Internal("Device AC selection produced an invalid cover");
          }
        }
      }
      if (!result.complete())
        return Status::Internal("Device AC selection produced an incomplete cover");
      for (size_t y = 0; y < block_extent.height; ++y) {
        for (size_t x = 0; x < block_extent.width; ++x) {
          AcStrategyCell cell;
          status = result.Get(x, y, &cell);
          if (!status.ok() || cells[y * block_extent.width + x] !=
               ((static_cast<uint8_t>(cell.strategy) << 1) |
                static_cast<uint8_t>(cell.is_anchor)))
            return Status::Internal("Device AC selection produced inconsistent cells");
        }
      }
      *out = std::move(result);
      result_stats.device_selection = true;
      if (stats != nullptr) *stats = result_stats;
      return Status::Ok();
    }
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
      resident == nullptr ? opsin.extent() : resident_opsin_extent,
      quant_field,
      pixel_mask,
      color_correlation,
      options,
      table,
      out);
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
#ifdef GJXL_FRONTIER_EXPERIMENT
    status = frontier_experiment::Capture(
      opsin_extent, quant_field, color_correlation, options, table, *out);
    if (!status.ok()) return status;
    status = frontier_experiment::SelectDiagnostic(options, table, out);
    if (!status.ok()) return status;
#endif
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU AC-strategy search state");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU AC-strategy search dimensions are too large");
  }
}

Status PreparedAcStrategySearch::PrepareDeferred(
    GpuBackend &gpu, ConstImage3FView opsin, ConstPlaneF32View quant,
    ConstPlaneF32View mask, const ColorCorrelationMap &cfl,
    ResidentAcStrategySearchInputs resident, AcStrategySearchOptions options,
    DeferredAcStrategySearch *out, AcStrategyGpuSearchStats *stats) {
  if (!out)
    return Status::InvalidArgument("Deferred search output is null");
  if (!CanDeferAcStrategySearch(gpu, options))
    return Status::Unavailable("Deferred search is unavailable");
  try {
    if (!impl_)
      impl_ = std::make_unique<ac_strategy_search_internal::Prepared>();
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Deferred search allocation failed");
  }
  AcStrategyGrid unused;
  return FindAcStrategyGridGpuImpl(gpu, opsin, quant, mask, cfl, &resident,
                                   impl_.get(), options, &unused, stats,
                                   nullptr, out);
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
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
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
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
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
