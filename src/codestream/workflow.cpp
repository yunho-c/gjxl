// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/workflow.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "codec/color_transform_internal.h"
#include "codec/quantization_pipeline.h"
#include "codec/quantization_pipeline_internal.h"
#include "codec/vardct_frame.h"
#include "codec/vardct_frame_view_internal.h"
#include "codestream/encoder.h"
#include "codestream/encoder_internal.h"
#include "codestream/rate_control_internal.h"
#include "codestream/workflow_internal.h"
#include "core/frame_geometry.h"
#include "core/image_buffer.h"
#include "core/thread_budget.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/ac_strategy.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/quantization_pipeline.h"
#include "gpu/ops/quantization_pipeline_profile_internal.h"
#include "gpu/ops/resident_input.h"

namespace gjxl {
namespace {

constexpr float kInitialProfileIntensityTarget = 255.0f;
using WorkflowClock = std::chrono::steady_clock;

uint64_t ElapsedNanoseconds(WorkflowClock::time_point begin) {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      WorkflowClock::now() - begin).count());
}

WorkflowClock::time_point ProfileBegin(
  const codestream_internal::VarDctEncodingProfile* profile) {
  return profile == nullptr ? WorkflowClock::time_point{} : WorkflowClock::now();
}

void ProfileEnd(
  const codestream_internal::VarDctEncodingProfile* requested,
  WorkflowClock::time_point begin,
  uint64_t* destination) {
  if (requested != nullptr) {
    *destination = ElapsedNanoseconds(begin);
  }
}

void AccumulateCodestreamProfile(
  const codestream_internal::VarDctCodestreamProfile& source,
  codestream_internal::VarDctCodestreamProfile* destination) {

  destination->entropy_behavior = source.entropy_behavior;
  destination->coefficient_order_behavior =
    source.coefficient_order_behavior;
  destination->validation_nanoseconds += source.validation_nanoseconds;
  destination->dc_tokenization_nanoseconds +=
    source.dc_tokenization_nanoseconds;
  destination->ac_tokenization_nanoseconds +=
    source.ac_tokenization_nanoseconds;
  destination->block_context_map_work_nanoseconds +=
    source.block_context_map_work_nanoseconds;
  destination->coefficient_order_work_nanoseconds +=
    source.coefficient_order_work_nanoseconds;
  destination->coefficient_tokenization_work_nanoseconds +=
    source.coefficient_tokenization_work_nanoseconds;
  destination->coefficient_context_materialization_work_nanoseconds +=
    source.coefficient_context_materialization_work_nanoseconds;
  destination->coefficient_tokenization_pass_count +=
    source.coefficient_tokenization_pass_count;
  destination->coefficient_token_count += source.coefficient_token_count;
  destination->coefficient_context_materialization_count +=
    source.coefficient_context_materialization_count;
  destination->coefficient_materialized_token_count +=
    source.coefficient_materialized_token_count;
  destination->entropy_optimization_nanoseconds +=
    source.entropy_optimization_nanoseconds;
  codestream_internal::AccumulateEntropyWorkProfile(
    source.entropy_work, &destination->entropy_work);
  destination->entropy_model_bits += source.entropy_model_bits;
  destination->entropy_token_bits += source.entropy_token_bits;
  destination->dc_entropy_clusters += source.dc_entropy_clusters;
  destination->ac_entropy_clusters += source.ac_entropy_clusters;
  destination->dc_entropy_is_ans =
    destination->dc_entropy_is_ans || source.dc_entropy_is_ans;
  destination->ac_entropy_is_ans =
    destination->ac_entropy_is_ans || source.ac_entropy_is_ans;
  destination->coefficient_order_entropy_is_ans =
    destination->coefficient_order_entropy_is_ans ||
    source.coefficient_order_entropy_is_ans;
  destination->natural_candidate_bytes += source.natural_candidate_bytes;
  destination->custom_order_candidate_bytes +=
    source.custom_order_candidate_bytes;
  destination->selected_coefficient_order_mask |=
    source.selected_coefficient_order_mask;
  destination->block_context_candidate_count +=
    source.block_context_candidate_count;
  destination->compact_block_context_candidate_bytes +=
    source.compact_block_context_candidate_bytes;
  destination->selected_block_context_candidate_index =
    source.selected_block_context_candidate_index;
  destination->selected_block_context_count +=
    source.selected_block_context_count;
  destination->selected_block_context_qf_threshold_count +=
    source.selected_block_context_qf_threshold_count;
  destination->section_writing_nanoseconds +=
    source.section_writing_nanoseconds;
  codestream_internal::AccumulateSectionWritingWorkProfile(
    source.section_writing_work, &destination->section_writing_work);
  destination->assembly_nanoseconds += source.assembly_nanoseconds;
  destination->assembly.candidate_selection_nanoseconds +=
    source.assembly.candidate_selection_nanoseconds;
  destination->assembly.section_size_nanoseconds +=
    source.assembly.section_size_nanoseconds;
  destination->assembly.frame_header_nanoseconds +=
    source.assembly.frame_header_nanoseconds;
  destination->assembly.toc_and_sections_nanoseconds +=
    source.assembly.toc_and_sections_nanoseconds;
  destination->assembly.output_copy_nanoseconds +=
    source.assembly.output_copy_nanoseconds;
  destination->total_nanoseconds += source.total_nanoseconds;
}

void AccumulateEncodingProfile(
  const codestream_internal::VarDctEncodingProfile& source,
  codestream_internal::VarDctEncodingProfile* destination) {

  destination->peak_cpu_participants = std::max(
    destination->peak_cpu_participants, source.peak_cpu_participants);
  destination->backend_selection_nanoseconds +=
    source.backend_selection_nanoseconds;
  destination->quantization_pipeline_nanoseconds +=
    source.quantization_pipeline_nanoseconds;
  destination->codestream_encoding_nanoseconds +=
    source.codestream_encoding_nanoseconds;
  destination->summary_assembly_nanoseconds +=
    source.summary_assembly_nanoseconds;
  destination->execution_backend = source.execution_backend;
  AccumulateCodestreamProfile(source.codestream, &destination->codestream);
}

// Cold 64x48 ranges overlap on the qualified M4 Pro; 96x64 is the first
// non-overlapping win. Keep one measured step of headroom, so automatic
// selection begins at 128x96.
constexpr size_t kAutomaticMetalMinimumCodingPixels = 128 * 96;
constexpr size_t kAutomaticMetalMinimumCodingDimension = 96;
// The resident automatic rollout is limited to this closed quality interval;
// threshold-sensitive and broader-quality requests remain on the CPU.
constexpr float kAutomaticMetalMinimumButteraugliTarget = 1.0f;
constexpr float kAutomaticMetalMaximumButteraugliTarget = 1.2f;
constexpr std::string_view kQualifiedMetalBackend = "Metal: Apple M4 Pro";

size_t AdaptiveQuantizationIterations(
  const VarDctEncodingOptions& options) {

  if (options.density_mode == VarDctDensityMode::kHighDensity) {
    return 4;
  }
  if (options.effort <= 3) {
    return 0;
  }
  if (options.effort <= 6) {
    return 1;
  }
  if (options.effort == 7) {
    return 2;
  }
  if (options.effort <= 9) {
    return 3;
  }
  return 4;
}

bool ValidQuantizationMatrixScaleStats(
  const codestream_internal::QuantizationMatrixScaleStats& stats) {

  return std::isfinite(stats.x_edge) && stats.x_edge >= 0.0f &&
    std::isfinite(stats.b_edge) && stats.b_edge >= 0.0f &&
    std::isfinite(stats.exposed_blue) && stats.exposed_blue >= 0.0f;
}

Status EffectiveTargetBytes(
  Extent2D source_extent,
  double bits_per_pixel,
  size_t* target_bytes) {

  if (target_bytes == nullptr || !std::isfinite(bits_per_pixel) ||
      bits_per_pixel <= 0.0) {
    return Status::InvalidArgument(
      "Target bits per pixel must be finite and positive");
  }
  size_t pixel_count = 0;
  if (!source_extent.try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "Target bits-per-pixel source dimensions are too large");
  }
  const long double bytes =
    static_cast<long double>(bits_per_pixel) *
    static_cast<long double>(pixel_count) / 8.0L;
  if (bytes < 1.0L ||
      bytes > static_cast<long double>(std::numeric_limits<size_t>::max())) {
    return Status::InvalidArgument(
      "Target bits per pixel does not produce a representable byte budget");
  }
  *target_bytes = static_cast<size_t>(bytes);
  return Status::Ok();
}

Status ValidateRateControlOptions(
  Extent2D source_extent,
  const VarDctEncodingOptions& options,
  size_t* effective_target_bytes,
  size_t* tolerance_bytes) {

  if (effective_target_bytes == nullptr || tolerance_bytes == nullptr) {
    return Status::InvalidArgument(
      "Rate-control derived-value output is null");
  }
  *effective_target_bytes = 0;
  *tolerance_bytes = 0;
  switch (options.rate_control_mode) {
    case VarDctRateControlMode::kButteraugliTarget:
      if (!std::isfinite(options.butteraugli_target) ||
          options.butteraugli_target <= 0.0f) {
        return Status::InvalidArgument(
          "Butteraugli target must be finite and positive");
      }
      return Status::Ok();

    case VarDctRateControlMode::kMaximumError:
      for (float maximum_error : options.maximum_error) {
        if (!std::isfinite(maximum_error) || maximum_error <= 0.0f) {
          return Status::InvalidArgument(
            "Maximum-error limits must be finite and positive");
        }
      }
      return Status::Ok();

    case VarDctRateControlMode::kTargetBytes:
      if (options.target_bytes == 0) {
        return Status::InvalidArgument(
          "Target byte count must be nonzero");
      }
      *effective_target_bytes = options.target_bytes;
      break;

    case VarDctRateControlMode::kTargetBitsPerPixel: {
      Status status = EffectiveTargetBytes(
        source_extent,
        options.target_bits_per_pixel,
        effective_target_bytes);
      if (!status.ok()) {
        return status;
      }
      break;
    }

    default:
      return Status::InvalidArgument(
        "VarDCT rate-control mode is invalid");
  }

  if (!std::isfinite(options.target_size_tolerance) ||
      options.target_size_tolerance < 0.0 ||
      options.target_size_tolerance > 1.0 ||
      options.target_size_maximum_attempts == 0 ||
      options.target_size_maximum_attempts >
        codestream_internal::kMaximumTargetSizeEncodeAttempts) {
    return Status::InvalidArgument(
      "Target-size tolerance or attempt limit is invalid");
  }
  switch (options.target_size_selection) {
    case TargetSizeSelectionPolicy::kLargestAtOrBelow:
    case TargetSizeSelectionPolicy::kClosestAbsolute:
      break;
    default:
      return Status::InvalidArgument(
        "Target-size selection policy is invalid");
  }
  const long double tolerance = std::ceil(
    static_cast<long double>(*effective_target_bytes) *
    static_cast<long double>(options.target_size_tolerance));
  if (tolerance >
      static_cast<long double>(std::numeric_limits<size_t>::max())) {
    return Status::InvalidArgument(
      "Target-size tolerance is not representable");
  }
  *tolerance_bytes = static_cast<size_t>(tolerance);
  return Status::Ok();
}

MetalBackendOptions ProductionMetalBackendOptions() {
  constexpr auto implementation =
    MetalDctImplementation::kSimdgroupMatmul;
  return {
    .forward_dct8 = implementation,
    .inverse_dct8 = implementation,
    .forward_dct16x16 = implementation,
    .inverse_dct16x16 = implementation,
    .forward_dct32x32 = implementation,
    .inverse_dct32x32 = implementation,
    .forward_dct16x8 = implementation,
    .inverse_dct16x8 = implementation,
    .forward_dct8x16 = implementation,
    .inverse_dct8x16 = implementation,
    .forward_dct32x16 = implementation,
    .inverse_dct32x16 = implementation,
    .forward_dct16x32 = implementation,
    .inverse_dct16x32 = implementation,
    .ac_residual_inverse = MetalAcResidualInverseMode::kFusedTuned,
  };
}

Status ResolveProductionMetalBackend(GpuBackend** out) {
  if (out == nullptr) {
    return Status::InvalidArgument(
      "Production Metal backend output pointer is null");
  }
  *out = nullptr;
  struct Cache {
    std::once_flag once;
    std::unique_ptr<GpuBackend> backend;
    Status status = Status::Unavailable(
      "Production Metal backend has not been initialized");
  };
  static Cache cache;
  std::call_once(cache.once, [&] {
    cache.status = CreateEmbeddedMetalBackend(
      ProductionMetalBackendOptions(), &cache.backend);
    if (cache.status.ok() && cache.backend == nullptr) {
      cache.status = Status::Internal(
        "Embedded Metal factory returned no backend");
    }
  });
  if (!cache.status.ok()) {
    return cache.status;
  }
  *out = cache.backend.get();
  return Status::Ok();
}

bool HasRequiredGpuQuantizationCapabilities(
  GpuBackend& backend,
  GpuAdaptiveQuantizationMode mode) {

  return QueryGpuAqEvaluation(backend) != nullptr &&
    (mode == GpuAdaptiveQuantizationMode::kMaximumThroughput ||
     QueryGpuAcStrategyEvaluation(backend) != nullptr) &&
    ((mode != GpuAdaptiveQuantizationMode::kFullyResident &&
      mode != GpuAdaptiveQuantizationMode::kThroughput) ||
     QueryGpuResidentInputPreparation(backend) != nullptr);
}

struct PipelineStorage {
  PipelineStorage(Extent2D frame_extent, Extent2D padded_extent)
      : block_extent{
          padded_extent.width / kJxlBlockDimension,
          padded_extent.height / kJxlBlockDimension},
        padded_extent(padded_extent),
        initial_quant(BlockCount()),
        strategy_mask(BlockCount()),
        pixel_mask(PixelCount(padded_extent)),
        final_quant(BlockCount()),
        block_distance(BlockCount()),
        reconstructed(frame_extent) {}

  [[nodiscard]] CpuQuantizationPipelineOutput Output() {
    return {
      .initial_quantization = {
        .quant_field = {
          initial_quant.data(), block_extent, block_extent.width},
        .strategy_mask = {
          strategy_mask.data(), block_extent, block_extent.width},
        .pixel_mask = {
          pixel_mask.data(), padded_extent, padded_extent.width},
      },
      .adaptive_quantization = {
        .quant_field = {
          final_quant.data(), block_extent, block_extent.width},
        .block_distance_map = {
          block_distance.data(), block_extent, block_extent.width},
        .reconstructed_linear_rgb = reconstructed.view(),
        .frame = &frame,
        .score_history = &score_history,
        .maximum_error_result = &maximum_error_result,
      },
    };
  }

  [[nodiscard]] size_t BlockCount() const {
    size_t count = 0;
    if (!block_extent.try_area(&count)) {
      throw std::length_error("VarDCT block extent is too large");
    }
    return count;
  }

  [[nodiscard]] static size_t PixelCount(Extent2D extent) {
    size_t count = 0;
    if (!extent.try_area(&count)) {
      throw std::length_error("VarDCT padded extent is too large");
    }
    return count;
  }

  Extent2D block_extent;
  Extent2D padded_extent;
  std::vector<float> initial_quant;
  std::vector<float> strategy_mask;
  std::vector<float> pixel_mask;
  std::vector<float> final_quant;
  std::vector<float> block_distance;
  Image3FBuffer reconstructed;
  VarDctEncoderFrame frame;
  std::vector<double> score_history;
  MaximumErrorResult maximum_error_result;
};

struct EncodingArtifacts {
  VarDctEncoderFrame frame;
  std::unique_ptr<vardct_frame_internal::CompletedVarDctFrame> completed_frame;
  std::vector<double> score_history;
  MaximumErrorResult maximum_error_result;
};

struct PreparedWorkflow {
  PreparedWorkflow(
    FrameGeometry prepared_geometry,
    ConstImage3FView source_linear_rgb,
    GpuBackend* prepared_gpu,
    bool prepared_metal,
    bool has_preselected_backend)
    : geometry(prepared_geometry),
      linear_rgb(source_linear_rgb),
      selected_gpu(prepared_gpu),
      selected_metal(prepared_metal),
      backend_preselected(has_preselected_backend) {}

  [[nodiscard]] ConstImage3FView original_linear_rgb() const noexcept {
    return linear_rgb;
  }

  FrameGeometry geometry;
  // Borrows the encode caller's immutable source for this synchronous
  // prepared workflow and all of its target-size attempts.
  ConstImage3FView linear_rgb;
  std::unique_ptr<Image3FBuffer> opsin;
  codestream_internal::QuantizationMatrixScaleStats matrix_scale_stats;
  // CPU and maximum-throughput adapters retain the public complete-output
  // shape. The default resident encoder never constructs this storage.
  std::unique_ptr<PipelineStorage> compatibility_output;
  quantization_pipeline_internal::PreparedQuantizationPipeline quantization;
  // Declared before its borrowers so it is destroyed after the resident
  // evaluator and AC-search state.
  std::unique_ptr<PreparedResidentInput> resident_input;
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    gpu_adaptive_quantization;
  GpuBackend* selected_gpu = nullptr;
  bool selected_metal = false;
  bool backend_preselected = false;
};

[[nodiscard]] Status EnsureCompatibilityOutput(
  PreparedWorkflow& prepared,
  PipelineStorage** output) {

  if (output == nullptr) {
    return Status::InvalidArgument(
      "Compatibility pipeline output pointer is null");
  }
  try {
    if (prepared.compatibility_output == nullptr) {
      prepared.compatibility_output = std::make_unique<PipelineStorage>(
        prepared.geometry.frame(), prepared.geometry.padded_frame());
    }
    *output = prepared.compatibility_output.get();
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate compatibility pipeline output");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Compatibility pipeline output dimensions are too large");
  }
}

[[nodiscard]] Status PrepareWorkflow(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* selected_gpu,
  bool selected_metal,
  bool backend_preselected,
  codestream_internal::VarDctEncodingProfile* profile,
  std::unique_ptr<PreparedWorkflow>* prepared) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Prepared workflow output pointer is null");
  }
  prepared->reset();
  try {
    const WorkflowClock::time_point geometry_begin = ProfileBegin(profile);
    FrameGeometry geometry;
    Status status = FrameGeometry::Create(linear_rgb.extent(), &geometry);
    if (!status.ok()) {
      return status;
    }
    auto candidate = std::make_unique<PreparedWorkflow>(
      geometry, linear_rgb, selected_gpu, selected_metal,
      backend_preselected);
    if (profile != nullptr) {
      profile->input_geometry_and_storage_nanoseconds =
        ElapsedNanoseconds(geometry_begin);
    }
    CpuQuantizationPipelineOptions preparation_options;
    if (options.rate_control_mode == VarDctRateControlMode::kMaximumError) {
      preparation_options.adaptive_quantization.control_mode =
        AdaptiveQuantizationControlMode::kMaximumError;
      preparation_options.adaptive_quantization.maximum_error =
        options.maximum_error;
    }
    if (backend_preselected && selected_metal) {
      const WorkflowClock::time_point resident_begin = ProfileBegin(profile);
      status = PrepareResidentInput(
        *selected_gpu,
        {
          .original_linear_rgb = candidate->original_linear_rgb(),
          .coding_extent = geometry.padded_frame(),
          .compute_matrix_scale_statistics =
            codestream_internal::ShouldComputeQuantizationMatrixScaleStats(
              options),
        },
        &candidate->resident_input);
      if (!status.ok()) return status;
      if (profile != nullptr) {
        profile->input_resident_preparation_nanoseconds =
          ElapsedNanoseconds(resident_begin);
      }
      if (codestream_internal::ShouldComputeQuantizationMatrixScaleStats(
            options)) {
        const ResidentInputStatistics stats =
          candidate->resident_input->statistics();
        candidate->matrix_scale_stats = {
          stats.x_edge, stats.b_edge, stats.exposed_blue};
      }
    } else {
      const WorkflowClock::time_point storage_begin = ProfileBegin(profile);
      candidate->opsin = std::make_unique<Image3FBuffer>(
        geometry.padded_frame());
      if (profile != nullptr) {
        profile->input_geometry_and_storage_nanoseconds +=
          ElapsedNanoseconds(storage_begin);
      }
      const WorkflowClock::time_point transform_begin = ProfileBegin(profile);
      status = color_transform_internal::LinearRgbToPaddedOpsin(
        linear_rgb, kInitialProfileIntensityTarget, candidate->opsin->view());
      if (!status.ok()) return status;
      if (profile != nullptr) {
        profile->input_color_transform_nanoseconds =
          ElapsedNanoseconds(transform_begin);
      }
      const WorkflowClock::time_point stats_begin = ProfileBegin(profile);
      if (codestream_internal::ShouldComputeQuantizationMatrixScaleStats(
            options)) {
        status = codestream_internal::
          ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
            candidate->opsin->cropped_view(geometry.frame()),
            &candidate->matrix_scale_stats);
        if (!status.ok()) return status;
      }
      if (profile != nullptr) {
        profile->input_matrix_scale_stats_nanoseconds =
          ElapsedNanoseconds(stats_begin);
      }
    }
    const WorkflowClock::time_point quantization_begin = ProfileBegin(profile);
    if (candidate->resident_input != nullptr) {
      status =
        quantization_pipeline_internal::PrepareResidentQuantizationPipeline(
          candidate->original_linear_rgb(), geometry.padded_frame(),
          candidate->resident_input->original_linear_rgb(),
          candidate->resident_input->coding_opsin(), preparation_options,
          &candidate->quantization);
    } else {
      status = quantization_pipeline_internal::PrepareQuantizationPipeline(
        candidate->original_linear_rgb(), candidate->opsin->const_view(),
        preparation_options, &candidate->quantization,
        options.backend == VarDctBackendPreference::kCpu,
        options.metal_aq_mode ==
          GpuAdaptiveQuantizationMode::kExactCoefficients,
        quantization_pipeline_internal::QuantizationPipelineInputProvenance::
          kFiniteLinearRgbAndOpsin);
    }
    if (!status.ok()) {
      return status;
    }
    if (profile != nullptr) {
      profile->input_quantization_preparation_nanoseconds =
        ElapsedNanoseconds(quantization_begin);
    }
    *prepared = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate prepared VarDCT workflow");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Prepared VarDCT workflow dimensions are too large");
  }
  return Status::Ok();
}

[[nodiscard]] Status SelectAttemptBackend(
  const FrameGeometry& geometry,
  VarDctEncodingOptions options,
  GpuBackend* supplied_backend,
  bool supplied_backend_is_qualified,
  bool resolve_production_backend,
  GpuBackend** selected_gpu,
  bool* selected_metal) {

  if (selected_gpu == nullptr || selected_metal == nullptr) {
    return Status::InvalidArgument(
      "Attempt backend output pointer is null");
  }
  *selected_gpu = nullptr;
  *selected_metal = false;
  if (options.backend == VarDctBackendPreference::kCpu ||
      (options.rate_control_mode == VarDctRateControlMode::kMaximumError &&
       options.backend != VarDctBackendPreference::kMetal)) {
    return Status::Ok();
  }

  const bool should_resolve =
    options.backend == VarDctBackendPreference::kMetal ||
    (options.backend == VarDctBackendPreference::kAutomatic &&
     codestream_internal::IsAutomaticMetalGeometryEligible(
       geometry.padded_frame()) &&
     codestream_internal::IsAutomaticMetalTargetEligible(
       options.butteraugli_target));
  if (!should_resolve) {
    return Status::Ok();
  }

  *selected_gpu = supplied_backend;
  bool qualified = supplied_backend_is_qualified;
  Status status = Status::Ok();
  if (*selected_gpu == nullptr && resolve_production_backend) {
    status = ResolveProductionMetalBackend(selected_gpu);
    if (!status.ok()) {
      if (options.backend == VarDctBackendPreference::kAutomatic &&
          status.code() == StatusCode::kUnavailable) {
        *selected_gpu = nullptr;
        return Status::Ok();
      }
      return status;
    }
    qualified = *selected_gpu != nullptr &&
      codestream_internal::IsAutomaticMetalBackendQualified(**selected_gpu);
  }
  if (*selected_gpu == nullptr) {
    return options.backend == VarDctBackendPreference::kMetal
      ? Status::Unavailable(
          "Forced Metal workflow has no available backend")
      : Status::Ok();
  }
  if (!HasRequiredGpuQuantizationCapabilities(
        **selected_gpu, options.metal_aq_mode)) {
    if (options.backend == VarDctBackendPreference::kMetal) {
      return Status::Unavailable(
        "Forced Metal workflow lacks a required GPU capability");
    }
    *selected_gpu = nullptr;
    return Status::Ok();
  }
  if (options.backend == VarDctBackendPreference::kMetal || qualified) {
    *selected_metal = true;
  } else {
    *selected_gpu = nullptr;
  }
  return Status::Ok();
}

[[nodiscard]] Status EncodePreparedAttempt(
  PreparedWorkflow& prepared,
  VarDctEncodingOptions options,
  size_t effective_target_bytes,
  size_t target_size_tolerance_bytes,
  GpuBackend* supplied_backend,
  bool supplied_backend_is_qualified,
  bool resolve_production_backend,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  codestream_internal::VarDctEncodingProfile* profile,
  gpu_profile_internal::GpuProfilingMode gpu_profiling_mode,
  gpu_profile_internal::GpuExecutionProfile* gpu_profile) {

  codestream_internal::VarDctEncodingProfile candidate_profile;
  gpu_profile_internal::GpuExecutionProfile candidate_gpu_profile;

  CpuQuantizationPipelineOptions pipeline_options;
  pipeline_options.butteraugli_target = options.butteraugli_target;
  pipeline_options.adaptive_quantization.iterations =
    AdaptiveQuantizationIterations(options);
  if (options.rate_control_mode == VarDctRateControlMode::kMaximumError) {
    pipeline_options.adaptive_quantization.control_mode =
      AdaptiveQuantizationControlMode::kMaximumError;
    pipeline_options.adaptive_quantization.maximum_error =
      options.maximum_error;
  }
  codestream_internal::QuantizationMatrixScales matrix_scales;
  Status status = codestream_internal::SelectQuantizationMatrixScales(
    prepared.matrix_scale_stats, options.rate_control_mode,
    options.butteraugli_target, &matrix_scales);
  if (!status.ok()) {
    return status;
  }
  pipeline_options.adaptive_quantization.profile.x_qm_scale = matrix_scales.x;
  pipeline_options.adaptive_quantization.profile.b_qm_scale = matrix_scales.b;
  prepared.quantization.profile.x_qm_scale = matrix_scales.x;
  prepared.quantization.profile.b_qm_scale = matrix_scales.b;

  GpuBackend* selected_gpu = nullptr;
  bool selected_metal = false;
  const WorkflowClock::time_point selection_begin = ProfileBegin(profile);
  if (prepared.backend_preselected) {
    selected_gpu = prepared.selected_gpu;
    selected_metal = prepared.selected_metal;
  } else {
    status = SelectAttemptBackend(
      prepared.geometry, options, supplied_backend,
      supplied_backend_is_qualified, resolve_production_backend,
      &selected_gpu, &selected_metal);
  }
  if (!status.ok()) {
    return status;
  }
  ProfileEnd(
    profile, selection_begin,
    &candidate_profile.backend_selection_nanoseconds);
  const WorkflowClock::time_point pipeline_begin = ProfileBegin(profile);
  EncodingArtifacts encoding;
  if (!prepared.backend_preselected && selected_metal &&
      (options.metal_aq_mode ==
         GpuAdaptiveQuantizationMode::kFullyResident ||
       options.metal_aq_mode == GpuAdaptiveQuantizationMode::kThroughput) &&
      prepared.resident_input == nullptr) {
    status = PrepareResidentInput(
      *selected_gpu,
      {
        .original_linear_rgb = prepared.original_linear_rgb(),
        .coding_extent = prepared.geometry.padded_frame(),
        .compute_matrix_scale_statistics =
          codestream_internal::ShouldComputeQuantizationMatrixScaleStats(
            options),
      },
      &prepared.resident_input);
    if (!status.ok()) return status;
    prepared.quantization.resident_original_linear_rgb =
      prepared.resident_input->original_linear_rgb();
    prepared.quantization.resident_coding_opsin =
      prepared.resident_input->coding_opsin();
    if (codestream_internal::ShouldComputeQuantizationMatrixScaleStats(
          options)) {
      const ResidentInputStatistics resident_stats =
        prepared.resident_input->statistics();
      if (resident_stats.x_edge != prepared.matrix_scale_stats.x_edge ||
          resident_stats.b_edge != prepared.matrix_scale_stats.b_edge ||
          resident_stats.exposed_blue !=
            prepared.matrix_scale_stats.exposed_blue) {
        return Status::Internal(
          "Resident and host matrix-scale statistics differ");
      }
    }
  }
  if (selected_metal && options.metal_aq_mode ==
        GpuAdaptiveQuantizationMode::kMaximumThroughput) {
    PipelineStorage* compatibility_output = nullptr;
    status = EnsureCompatibilityOutput(prepared, &compatibility_output);
    if (status.ok()) {
      const CpuQuantizationPipelineOutput pipeline_output =
        compatibility_output->Output();
      status = RunGpuFrameOnlyQuantizationPipeline(
        *selected_gpu, prepared.original_linear_rgb(),
        prepared.opsin->const_view(), pipeline_options,
        {
          .initial_quantization = pipeline_output.initial_quantization,
          .quant_field = {
            compatibility_output->final_quant.data(),
            compatibility_output->block_extent,
            compatibility_output->block_extent.width},
          .frame = &compatibility_output->frame,
        });
      if (status.ok()) {
        encoding.frame = std::move(compatibility_output->frame);
      }
    }
  } else if (selected_metal) {
    const quantization_pipeline_internal::GpuEncodingQuantizationPipelineOutput
      encoding_output{
      .frame = &encoding.frame,
      .score_history = &encoding.score_history,
      .maximum_error_result = &encoding.maximum_error_result,
      .collect_final_butteraugli_score =
        options.collect_final_butteraugli_score ||
        pipeline_options.adaptive_quantization.iterations == 0 ||
        options.metal_aq_mode ==
          GpuAdaptiveQuantizationMode::kExactCoefficients ||
        options.rate_control_mode == VarDctRateControlMode::kMaximumError,
      .completed_frame = &encoding.completed_frame,
    };
    status = gpu_profile == nullptr
      ? quantization_pipeline_internal::
          RunPreparedGpuQuantizationPipelineForEncoding(
            *selected_gpu, prepared.original_linear_rgb(),
            prepared.quantization, pipeline_options, options.metal_aq_mode,
            encoding_output, nullptr, &prepared.gpu_adaptive_quantization)
      : quantization_pipeline_internal::
          RunPreparedGpuQuantizationPipelineForEncodingProfiled(
            *selected_gpu, prepared.original_linear_rgb(),
            prepared.quantization,
            pipeline_options, options.metal_aq_mode, encoding_output,
            &prepared.gpu_adaptive_quantization, gpu_profiling_mode,
            &candidate_gpu_profile);
  } else {
    PipelineStorage* compatibility_output = nullptr;
    status = EnsureCompatibilityOutput(prepared, &compatibility_output);
    if (status.ok()) {
      status = quantization_pipeline_internal::
        RunPreparedCpuQuantizationPipeline(
          prepared.original_linear_rgb(), prepared.quantization,
          pipeline_options, compatibility_output->Output());
    }
    if (status.ok()) {
      encoding.frame = std::move(compatibility_output->frame);
      encoding.score_history = std::move(compatibility_output->score_history);
      encoding.maximum_error_result =
        compatibility_output->maximum_error_result;
    }
  }
  if (!status.ok()) {
    return status;
  }
  ProfileEnd(
    profile, pipeline_begin,
    &candidate_profile.quantization_pipeline_nanoseconds);

  std::vector<uint8_t> candidate;
  const WorkflowClock::time_point codestream_begin = ProfileBegin(profile);
  const VarDctCodestreamOptions codestream_options{
    .entropy_behavior =
      codestream_internal::ResolveEntropyBehavior(options),
    .coefficient_order_behavior =
      codestream_internal::ResolveCoefficientOrderBehavior(options),
  };
  const auto frame_view = encoding.completed_frame != nullptr
    ? encoding.completed_frame->view()
    : vardct_frame_internal::BorrowFrame(encoding.frame);
  status = codestream_internal::EncodeVarDctCodestreamFromView(
    frame_view, codestream_options, &candidate,
    profile == nullptr ? nullptr : &candidate_profile.codestream);
  ProfileEnd(
    profile, codestream_begin,
    &candidate_profile.codestream_encoding_nanoseconds);
  if (!status.ok()) {
    return status;
  }

  const WorkflowClock::time_point summary_begin = ProfileBegin(profile);
  VarDctEncodingSummary candidate_summary;
  candidate_summary.extent = prepared.geometry.frame();
  candidate_summary.encoded_bytes = candidate.size();
  candidate_summary.density_mode = options.density_mode;
  candidate_summary.compression_mode = options.compression_mode;
  candidate_summary.entropy_behavior = codestream_options.entropy_behavior;
  candidate_summary.rate_control_mode = options.rate_control_mode;
  candidate_summary.effective_target_bytes = effective_target_bytes;
  candidate_summary.target_size_tolerance_bytes =
    target_size_tolerance_bytes;
  if (options.rate_control_mode == VarDctRateControlMode::kTargetBytes) {
    candidate_summary.requested_target_bytes = options.target_bytes;
  } else if (options.rate_control_mode ==
             VarDctRateControlMode::kTargetBitsPerPixel) {
    candidate_summary.requested_target_bits_per_pixel =
      options.target_bits_per_pixel;
  }
  size_t source_pixel_count = 0;
  if (!prepared.geometry.frame().try_area(&source_pixel_count) ||
      source_pixel_count == 0) {
    return Status::Internal(
      "Validated VarDCT source has no representable pixels");
  }
  candidate_summary.achieved_bits_per_pixel =
    8.0 * static_cast<double>(candidate.size()) /
    static_cast<double>(source_pixel_count);
  candidate_summary.selected_butteraugli_target =
    options.rate_control_mode == VarDctRateControlMode::kMaximumError
      ? 0.0f
      : options.butteraugli_target;
  if (options.rate_control_mode == VarDctRateControlMode::kMaximumError) {
    candidate_summary.requested_maximum_error = options.maximum_error;
    candidate_summary.achieved_maximum_error =
      encoding.maximum_error_result.achieved;
    candidate_summary.achieved_maximum_error_ratio =
      encoding.maximum_error_result.normalized_maximum;
    candidate_summary.maximum_error_evaluation_count =
      encoding.maximum_error_result.evaluation_count;
    candidate_summary.maximum_error_outcome =
      encoding.maximum_error_result.outcome;
  }
  candidate_summary.encode_attempt_count = 1;
  candidate_summary.score_history = encoding.score_history;
  candidate_summary.final_butteraugli_score_evaluated =
    options.rate_control_mode != VarDctRateControlMode::kMaximumError &&
    !candidate_summary.score_history.empty() &&
    (!selected_metal ||
     options.metal_aq_mode ==
       GpuAdaptiveQuantizationMode::kExactCoefficients ||
     options.collect_final_butteraugli_score ||
     pipeline_options.adaptive_quantization.iterations == 0);
  candidate_summary.execution_backend = selected_metal
    ? VarDctExecutionBackend::kMetal
    : VarDctExecutionBackend::kCpu;
  candidate_summary.metal_aq_mode = options.metal_aq_mode;
  status = frame_view.strategies().ForEachAnchor(
    [&](size_t, size_t, AcStrategyType strategy) {
      const size_t index = static_cast<size_t>(strategy);
      if (index >= candidate_summary.strategy_counts.size()) {
        return Status::Internal(
          "Completed frame contains an unknown AC strategy");
      }
      ++candidate_summary.strategy_counts[index];
      return Status::Ok();
    });
  if (!status.ok()) {
    return status;
  }
  ProfileEnd(
    profile, summary_begin,
    &candidate_profile.summary_assembly_nanoseconds);
  candidate_profile.execution_backend = selected_metal
    ? VarDctExecutionBackend::kMetal
    : VarDctExecutionBackend::kCpu;

  *codestream = std::move(candidate);
  if (summary != nullptr) {
    *summary = std::move(candidate_summary);
  }
  if (profile != nullptr) {
    *profile = candidate_profile;
  }
  if (gpu_profile != nullptr) {
    *gpu_profile = std::move(candidate_gpu_profile);
  }
  return Status::Ok();
}

}  // namespace

Status codestream_internal::ComputeQuantizationMatrixScaleStats(
  ConstImage3FView opsin,
  QuantizationMatrixScaleStats* stats) {

  if (stats == nullptr || !opsin.valid()) {
    return Status::InvalidArgument(
      "Quantization-matrix scale statistics input or output is invalid");
  }

  QuantizationMatrixScaleStats candidate;
  for (size_t y = 0; y < opsin.height(); ++y) {
    const float* row_x = opsin.plane[0].Row(y);
    const float* row_y = opsin.plane[1].Row(y);
    const float* row_b = opsin.plane[2].Row(y);
    const float* previous_row_x = y == 0
      ? nullptr
      : opsin.plane[0].Row(y - 1);
    const float* previous_row_y = y == 0
      ? nullptr
      : opsin.plane[1].Row(y - 1);
    const float* previous_row_b = y == 0
      ? nullptr
      : opsin.plane[2].Row(y - 1);
    for (size_t x = 0; x < opsin.width(); ++x) {
      const float current_x = row_x[x];
      const float current_y = row_y[x];
      const float current_b = row_b[x];
      if (!std::isfinite(current_x) || !std::isfinite(current_y) ||
          !std::isfinite(current_b)) {
        return Status::InvalidArgument(
          "Opsin pixels for matrix-scale statistics must be finite");
      }
      if (x == 0 || y == 0) {
        continue;
      }

      const float previous_x = row_x[x - 1];
      const float horizontal_x_edge =
        std::abs(current_x - previous_x);
      const float vertical_x_edge =
        std::abs(current_x - previous_row_x[x]);
      if (!std::isfinite(horizontal_x_edge) ||
          !std::isfinite(vertical_x_edge)) {
        return Status::InvalidArgument(
          "Opsin X edge for matrix-scale statistics is not finite");
      }
      candidate.x_edge = std::max(
        candidate.x_edge,
        std::max(horizontal_x_edge, vertical_x_edge));

      const float previous_y = row_y[x - 1];
      const float previous_b = row_b[x - 1];
      const float current_difference = current_b - current_y;
      const float horizontal_b_edge = std::abs(
        current_difference - (previous_b - previous_y));
      const float vertical_b_edge = std::abs(
        current_difference - (previous_row_b[x] - previous_row_y[x]));
      if (!std::isfinite(horizontal_b_edge) ||
          !std::isfinite(vertical_b_edge)) {
        return Status::InvalidArgument(
          "Opsin B edge for matrix-scale statistics is not finite");
      }
      candidate.b_edge = std::max(
        candidate.b_edge,
        std::max(horizontal_b_edge, vertical_b_edge));

      float exposed_blue = current_b - 1.2f * current_y;
      if (exposed_blue >= 0.0f) {
        exposed_blue *= std::abs(current_b - previous_b) +
          std::abs(current_b - previous_row_b[x]);
        if (!std::isfinite(exposed_blue)) {
          return Status::InvalidArgument(
            "Exposed-blue matrix-scale statistic is not finite");
        }
        candidate.exposed_blue =
          std::max(candidate.exposed_blue, exposed_blue);
      } else if (!std::isfinite(exposed_blue)) {
        return Status::InvalidArgument(
          "Exposed-blue matrix-scale statistic is not finite");
      }
    }
  }
  if (!ValidQuantizationMatrixScaleStats(candidate)) {
    return Status::InvalidArgument(
      "Quantization-matrix scale statistics are not finite");
  }
  *stats = candidate;
  return Status::Ok();
}

Status codestream_internal::
ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
  ConstImage3FView opsin,
  QuantizationMatrixScaleStats* stats) {

  if (stats == nullptr || !opsin.valid()) {
    return Status::InvalidArgument(
      "Finite quantization-matrix statistics input or output is invalid");
  }

  QuantizationMatrixScaleStats candidate;
  if (opsin.width() > 1 && opsin.height() > 1) {
#if defined(__ARM_NEON)
    float32x4_t vector_x_edge = vdupq_n_f32(0.0f);
    float32x4_t vector_b_edge = vdupq_n_f32(0.0f);
    float32x4_t vector_exposed_blue = vdupq_n_f32(0.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
#endif
    for (size_t y = 1; y < opsin.height(); ++y) {
      const float* row_x = opsin.plane[0].Row(y);
      const float* row_y = opsin.plane[1].Row(y);
      const float* row_b = opsin.plane[2].Row(y);
      const float* previous_row_x = opsin.plane[0].Row(y - 1);
      const float* previous_row_y = opsin.plane[1].Row(y - 1);
      const float* previous_row_b = opsin.plane[2].Row(y - 1);
      size_t x = 1;
#if defined(__ARM_NEON)
      for (; x + 4 <= opsin.width(); x += 4) {
        const float32x4_t current_x = vld1q_f32(row_x + x);
        const float32x4_t current_y = vld1q_f32(row_y + x);
        const float32x4_t current_b = vld1q_f32(row_b + x);
        const float32x4_t horizontal_x_edge = vabsq_f32(vsubq_f32(
          current_x, vld1q_f32(row_x + x - 1)));
        const float32x4_t vertical_x_edge = vabsq_f32(vsubq_f32(
          current_x, vld1q_f32(previous_row_x + x)));
        vector_x_edge = vmaxq_f32(
          vector_x_edge,
          vmaxq_f32(horizontal_x_edge, vertical_x_edge));

        const float32x4_t current_difference =
          vsubq_f32(current_b, current_y);
        const float32x4_t previous_difference = vsubq_f32(
          vld1q_f32(row_b + x - 1),
          vld1q_f32(row_y + x - 1));
        const float32x4_t previous_row_difference = vsubq_f32(
          vld1q_f32(previous_row_b + x),
          vld1q_f32(previous_row_y + x));
        const float32x4_t horizontal_b_edge = vabsq_f32(vsubq_f32(
          current_difference, previous_difference));
        const float32x4_t vertical_b_edge = vabsq_f32(vsubq_f32(
          current_difference, previous_row_difference));
        vector_b_edge = vmaxq_f32(
          vector_b_edge,
          vmaxq_f32(horizontal_b_edge, vertical_b_edge));

        const float32x4_t exposed_blue = vsubq_f32(
          current_b, vmulq_n_f32(current_y, 1.2f));
        const float32x4_t blue_edge = vaddq_f32(
          vabsq_f32(vsubq_f32(
            current_b, vld1q_f32(row_b + x - 1))),
          vabsq_f32(vsubq_f32(
            current_b, vld1q_f32(previous_row_b + x))));
        vector_exposed_blue = vmaxq_f32(
          vector_exposed_blue,
          vmulq_f32(vmaxq_f32(exposed_blue, zero), blue_edge));
      }
#endif
      for (; x < opsin.width(); ++x) {
        const float current_x = row_x[x];
        candidate.x_edge = std::max(
          candidate.x_edge,
          std::max(
            std::abs(current_x - row_x[x - 1]),
            std::abs(current_x - previous_row_x[x])));

        const float current_y = row_y[x];
        const float current_b = row_b[x];
        const float current_difference = current_b - current_y;
        candidate.b_edge = std::max(
          candidate.b_edge,
          std::max(
            std::abs(
              current_difference - (row_b[x - 1] - row_y[x - 1])),
            std::abs(
              current_difference -
                (previous_row_b[x] - previous_row_y[x]))));

        float exposed_blue = current_b - 1.2f * current_y;
        if (exposed_blue >= 0.0f) {
          exposed_blue *= std::abs(current_b - row_b[x - 1]) +
            std::abs(current_b - previous_row_b[x]);
          candidate.exposed_blue = std::max(
            candidate.exposed_blue, exposed_blue);
        }
      }
    }
#if defined(__ARM_NEON)
    std::array<float, 4> lanes{};
    vst1q_f32(lanes.data(), vector_x_edge);
    candidate.x_edge = std::max(
      candidate.x_edge, *std::ranges::max_element(lanes));
    vst1q_f32(lanes.data(), vector_b_edge);
    candidate.b_edge = std::max(
      candidate.b_edge, *std::ranges::max_element(lanes));
    vst1q_f32(lanes.data(), vector_exposed_blue);
    candidate.exposed_blue = std::max(
      candidate.exposed_blue, *std::ranges::max_element(lanes));
#endif
  }
  if (!ValidQuantizationMatrixScaleStats(candidate)) {
    return Status::InvalidArgument(
      "Finite quantization-matrix statistics are not finite");
  }
  *stats = candidate;
  return Status::Ok();
}

Status codestream_internal::SelectQuantizationMatrixScales(
  const QuantizationMatrixScaleStats& stats,
  VarDctRateControlMode mode,
  float butteraugli_target,
  QuantizationMatrixScales* scales) {

  if (scales == nullptr || !ValidQuantizationMatrixScaleStats(stats)) {
    return Status::InvalidArgument(
      "Quantization-matrix scale selection input or output is invalid");
  }
  if (mode == VarDctRateControlMode::kMaximumError) {
    *scales = {};
    return Status::Ok();
  }
  if (mode != VarDctRateControlMode::kButteraugliTarget ||
      !std::isfinite(butteraugli_target) || butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "Matrix-scale selection requires a positive Butteraugli target");
  }

  QuantizationMatrixScales candidate = {.x = 3, .b = 2};
  for (float threshold : {2.5f, 5.5f, 9.5f}) {
    if (butteraugli_target > threshold) {
      ++candidate.x;
    }
  }
  uint8_t x_pixel_adjustment = 0;
  if (stats.x_edge >= 0.026f) {
    x_pixel_adjustment = 3;
  } else if (stats.x_edge >= 0.022f) {
    x_pixel_adjustment = 2;
  } else if (stats.x_edge >= 0.015f) {
    x_pixel_adjustment = 1;
  }
  candidate.x = std::max<uint8_t>(candidate.x, 2 + x_pixel_adjustment);

  const uint8_t exposed_blue_adjustment =
    stats.exposed_blue >= 0.13f ? 1 : 0;
  uint8_t b_pixel_adjustment = 0;
  if (stats.b_edge > 0.38f) {
    b_pixel_adjustment = 2 + exposed_blue_adjustment;
  } else if (stats.b_edge > 0.33f) {
    b_pixel_adjustment = 1 + exposed_blue_adjustment;
  } else if (stats.b_edge > 0.28f) {
    b_pixel_adjustment = exposed_blue_adjustment;
  }
  candidate.b += b_pixel_adjustment;

  *scales = candidate;
  return Status::Ok();
}

Status EncodeLinearRgbVarDctCodestreamImpl(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* supplied_backend,
  bool supplied_backend_is_qualified,
  bool resolve_production_backend,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingTiming* timing,
  codestream_internal::VarDctEncodingProfile* profile,
  gpu_profile_internal::GpuProfilingMode gpu_profiling_mode,
  gpu_profile_internal::GpuExecutionProfile* gpu_profile) {

  const bool gpu_profiling =
    gpu_profiling_mode != gpu_profile_internal::GpuProfilingMode::kDisabled;
  if (gpu_profiling != (gpu_profile != nullptr)) {
    return Status::InvalidArgument(
      "VarDCT GPU profiling request is invalid");
  }

  const bool profiling = timing != nullptr || profile != nullptr;
  const auto total_begin = !profiling
    ? WorkflowClock::time_point{}
    : WorkflowClock::now();
  VarDctEncodingTiming local_timing;
  codestream_internal::VarDctEncodingProfile local_profile;
  gpu_profile_internal::GpuExecutionProfile local_gpu_profile;
  if (codestream == nullptr || !linear_rgb.valid()) {
    return Status::InvalidArgument(
      "VarDCT encoding input or output is invalid");
  }
  size_t effective_target_bytes = 0;
  size_t target_size_tolerance_bytes = 0;
  Status status = ValidateRateControlOptions(
    linear_rgb.extent(),
    options,
    &effective_target_bytes,
    &target_size_tolerance_bytes);
  if (!status.ok()) {
    return status;
  }
  if (options.effort < 1 || options.effort > 10) {
    return Status::InvalidArgument(
      "VarDCT effort must be in [1, 10]");
  }
  if (options.cpu_thread_count > kMaximumCpuThreadCount) {
    return Status::InvalidArgument(
      "VarDCT CPU thread count must be zero or at most 256");
  }
  thread_budget_internal::CpuParticipantTracker participant_tracker;
  const thread_budget_internal::EncodeScope thread_budget(
    options.cpu_thread_count,
    profile == nullptr ? nullptr : &participant_tracker);
  switch (options.backend) {
    case VarDctBackendPreference::kAutomatic:
    case VarDctBackendPreference::kCpu:
    case VarDctBackendPreference::kMetal:
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT encoding backend preference is invalid");
  }
  switch (options.density_mode) {
    case VarDctDensityMode::kDefault:
    case VarDctDensityMode::kHighDensity:
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT density mode is invalid");
  }
  switch (options.compression_mode) {
    case VarDctCompressionMode::kAutomatic:
    case VarDctCompressionMode::kMaximumCompression:
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT compression mode is invalid");
  }
  switch (options.metal_aq_mode) {
    case GpuAdaptiveQuantizationMode::kExactCoefficients:
    case GpuAdaptiveQuantizationMode::kFullyResident:
      break;
    case GpuAdaptiveQuantizationMode::kThroughput:
    case GpuAdaptiveQuantizationMode::kMaximumThroughput:
      if (options.backend != VarDctBackendPreference::kMetal) {
        return Status::InvalidArgument(
          "Throughput AQ requires an explicitly forced Metal backend");
      }
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT Metal AQ mode is invalid");
  }
  if (options.metal_aq_mode ==
        GpuAdaptiveQuantizationMode::kMaximumThroughput &&
      options.rate_control_mode == VarDctRateControlMode::kMaximumError) {
    return Status::InvalidArgument(
      "Maximum-throughput AQ does not evaluate maximum error");
  }
  if (options.density_mode == VarDctDensityMode::kHighDensity &&
      (options.rate_control_mode == VarDctRateControlMode::kMaximumError ||
       options.metal_aq_mode == GpuAdaptiveQuantizationMode::kThroughput ||
       options.metal_aq_mode ==
         GpuAdaptiveQuantizationMode::kMaximumThroughput)) {
    return Status::InvalidArgument(
      "High-density AQ requires iterative Butteraugli control");
  }
  if (gpu_profiling &&
      (options.backend != VarDctBackendPreference::kMetal ||
       (options.metal_aq_mode !=
          GpuAdaptiveQuantizationMode::kFullyResident &&
        options.metal_aq_mode != GpuAdaptiveQuantizationMode::kThroughput) ||
       options.rate_control_mode !=
          VarDctRateControlMode::kButteraugliTarget)) {
    return Status::InvalidArgument(
      "GPU profiling requires a resident Metal Butteraugli-target workflow");
  }
  const bool target_size_control =
    options.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
    options.rate_control_mode == VarDctRateControlMode::kTargetBitsPerPixel;
  const bool resident_input_candidate =
    (options.metal_aq_mode == GpuAdaptiveQuantizationMode::kFullyResident ||
     options.metal_aq_mode == GpuAdaptiveQuantizationMode::kThroughput) &&
    !(target_size_control &&
      options.backend == VarDctBackendPreference::kAutomatic);
  GpuBackend* workflow_gpu = nullptr;
  bool workflow_metal = false;
  bool backend_preselected = false;
  if (resident_input_candidate) {
    FrameGeometry selection_geometry;
    status = FrameGeometry::Create(linear_rgb.extent(), &selection_geometry);
    if (!status.ok()) return status;
    const WorkflowClock::time_point selection_begin =
      ProfileBegin(profile);
    status = SelectAttemptBackend(
      selection_geometry, options, supplied_backend,
      supplied_backend_is_qualified, resolve_production_backend,
      &workflow_gpu, &workflow_metal);
    if (!status.ok()) return status;
    backend_preselected = true;
    if (profile != nullptr) {
      local_profile.backend_selection_nanoseconds =
        ElapsedNanoseconds(selection_begin);
    }
  }
  if (options.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
      options.rate_control_mode ==
        VarDctRateControlMode::kTargetBitsPerPixel) {
    try {
      std::unique_ptr<PreparedWorkflow> prepared;
      const auto preparation_begin = !profiling
        ? WorkflowClock::time_point{}
        : WorkflowClock::now();
      status = PrepareWorkflow(
        linear_rgb, options, workflow_gpu, workflow_metal,
        backend_preselected,
        profile == nullptr ? nullptr : &local_profile, &prepared);
      if (!status.ok()) {
        return status;
      }
      if (timing != nullptr) {
        local_timing.preparation_nanoseconds =
          ElapsedNanoseconds(preparation_begin);
      }
      if (profile != nullptr) {
        local_profile.input_preparation_nanoseconds =
          ElapsedNanoseconds(preparation_begin);
      }
      codestream_internal::TargetSizeSearchResult search_result;
      const auto search_begin = timing == nullptr
        ? WorkflowClock::time_point{}
        : WorkflowClock::now();
      status = codestream_internal::SearchTargetSize(
        {
          .target_bytes = effective_target_bytes,
          .tolerance_bytes = target_size_tolerance_bytes,
          .maximum_attempts = options.target_size_maximum_attempts,
          .selection = options.target_size_selection,
        },
        [&](float butteraugli_target,
            std::vector<uint8_t>* attempt_codestream,
            VarDctEncodingSummary* attempt_summary) {
          const auto attempt_begin = timing == nullptr
            ? WorkflowClock::time_point{}
            : WorkflowClock::now();
          VarDctEncodingOptions attempt_options = options;
          attempt_options.butteraugli_target = butteraugli_target;
          attempt_options.rate_control_mode =
            VarDctRateControlMode::kButteraugliTarget;
          // A resident size search must not switch between CPU and Metal as
          // its candidate targets cross the automatic quality interval.
          // Exact coefficients retain their decision-compatible automatic
          // search behavior.
          if (options.backend == VarDctBackendPreference::kAutomatic &&
              options.metal_aq_mode ==
                GpuAdaptiveQuantizationMode::kFullyResident) {
            attempt_options.backend = VarDctBackendPreference::kCpu;
          }
          codestream_internal::VarDctEncodingProfile attempt_profile;
          const Status attempt_status = EncodePreparedAttempt(
            *prepared, attempt_options, 0, 0, supplied_backend,
            supplied_backend_is_qualified, resolve_production_backend,
            attempt_codestream, attempt_summary,
            profile == nullptr ? nullptr : &attempt_profile,
            gpu_profile_internal::GpuProfilingMode::kDisabled, nullptr);
          if (profile != nullptr && attempt_status.ok()) {
            AccumulateEncodingProfile(attempt_profile, &local_profile);
          }
          if (timing != nullptr) {
            local_timing.attempts.push_back({
              .butteraugli_target = butteraugli_target,
              .encode_and_serialize_nanoseconds =
                ElapsedNanoseconds(attempt_begin),
              .encoded_bytes = attempt_status.ok()
                ? attempt_codestream->size()
                : 0,
              .succeeded = attempt_status.ok(),
            });
          }
          return attempt_status;
        },
        &search_result);
      if (timing != nullptr) {
        local_timing.aggregate_search_nanoseconds =
          ElapsedNanoseconds(search_begin);
      }
      if (!status.ok()) {
        return status;
      }

      search_result.summary.rate_control_mode = options.rate_control_mode;
      search_result.summary.effective_target_bytes = effective_target_bytes;
      search_result.summary.target_size_tolerance_bytes =
        target_size_tolerance_bytes;
      search_result.summary.encode_attempt_count = search_result.attempt_count;
      search_result.summary.failed_encode_attempt_count =
        search_result.failed_attempt_count;
      search_result.summary.target_size_selection =
        options.target_size_selection;
      search_result.summary.target_size_met = search_result.target_size_met;
      search_result.summary.target_size_search_exhausted =
        search_result.search_exhausted;
      if (options.rate_control_mode == VarDctRateControlMode::kTargetBytes) {
        search_result.summary.requested_target_bytes = options.target_bytes;
      } else {
        search_result.summary.requested_target_bits_per_pixel =
          options.target_bits_per_pixel;
      }
      if (timing != nullptr) {
        const auto selected = std::find_if(
          local_timing.attempts.begin(), local_timing.attempts.end(),
          [&](const VarDctEncodingAttemptTiming& attempt) {
            return attempt.succeeded &&
              attempt.butteraugli_target ==
                search_result.summary.selected_butteraugli_target &&
              attempt.encoded_bytes == search_result.codestream.size();
          });
        if (selected == local_timing.attempts.end()) {
          return Status::Internal(
            "Target-size timing cannot identify the selected attempt");
        }
        local_timing.selected_attempt_nanoseconds =
          selected->encode_and_serialize_nanoseconds;
      }
      if (profile != nullptr) {
        local_profile.execution_backend =
          search_result.summary.execution_backend;
        local_profile.total_nanoseconds = ElapsedNanoseconds(total_begin);
      }
      *codestream = std::move(search_result.codestream);
      if (summary != nullptr) {
        *summary = std::move(search_result.summary);
      }
      if (timing != nullptr) {
        local_timing.total_nanoseconds = ElapsedNanoseconds(total_begin);
        *timing = std::move(local_timing);
      }
      if (profile != nullptr) {
        local_profile.peak_cpu_participants = participant_tracker.peak();
        *profile = local_profile;
      }
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate target-size workflow storage");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Target-size workflow storage is too large");
    }
    return Status::Ok();
  }

  try {
    std::unique_ptr<PreparedWorkflow> prepared;
    const auto preparation_begin = !profiling
      ? WorkflowClock::time_point{}
      : WorkflowClock::now();
    status = PrepareWorkflow(
      linear_rgb, options, workflow_gpu, workflow_metal,
      backend_preselected,
      profile == nullptr ? nullptr : &local_profile, &prepared);
    if (!status.ok()) {
      return status;
    }
    if (timing != nullptr) {
      local_timing.preparation_nanoseconds =
        ElapsedNanoseconds(preparation_begin);
    }
    if (profile != nullptr) {
      local_profile.input_preparation_nanoseconds =
        ElapsedNanoseconds(preparation_begin);
    }
    std::vector<uint8_t> candidate;
    VarDctEncodingSummary candidate_summary;
    codestream_internal::VarDctEncodingProfile attempt_profile;
    const auto attempt_begin = timing == nullptr
      ? WorkflowClock::time_point{}
      : WorkflowClock::now();
    status = EncodePreparedAttempt(
      *prepared, options, effective_target_bytes,
      target_size_tolerance_bytes, supplied_backend,
      supplied_backend_is_qualified, resolve_production_backend,
      &candidate, &candidate_summary,
      profile == nullptr ? nullptr : &attempt_profile,
      gpu_profiling_mode, gpu_profiling ? &local_gpu_profile : nullptr);
    if (timing != nullptr) {
      local_timing.attempts.push_back({
        .butteraugli_target =
          options.rate_control_mode == VarDctRateControlMode::kMaximumError
            ? 0.0f
            : options.butteraugli_target,
        .encode_and_serialize_nanoseconds =
          ElapsedNanoseconds(attempt_begin),
        .encoded_bytes = status.ok() ? candidate.size() : 0,
        .succeeded = status.ok(),
      });
    }
    if (!status.ok()) {
      return status;
    }
    if (profile != nullptr) {
      AccumulateEncodingProfile(attempt_profile, &local_profile);
    }

    if (timing != nullptr) {
      local_timing.selected_attempt_nanoseconds =
        local_timing.attempts.front().encode_and_serialize_nanoseconds;
    }
    *codestream = std::move(candidate);
    if (summary != nullptr) {
      *summary = std::move(candidate_summary);
    }
    if (timing != nullptr) {
      local_timing.total_nanoseconds = ElapsedNanoseconds(total_begin);
      *timing = std::move(local_timing);
    }
    if (profile != nullptr) {
      local_profile.execution_backend =
        attempt_profile.execution_backend;
      local_profile.total_nanoseconds = ElapsedNanoseconds(total_begin);
      local_profile.peak_cpu_participants = participant_tracker.peak();
      *profile = local_profile;
    }
    if (gpu_profiling) *gpu_profile = std::move(local_gpu_profile);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate public VarDCT encoding storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Public VarDCT encoding dimensions are too large");
  }
  return Status::Ok();
}

Status EncodeLinearRgbVarDctCodestream(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary) {

  return EncodeLinearRgbVarDctCodestreamImpl(
    linear_rgb, options, nullptr, false, true, codestream, summary, nullptr,
    nullptr, gpu_profile_internal::GpuProfilingMode::kDisabled, nullptr);
}

Status EncodeLinearRgbVarDctCodestreamProfiled(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingTiming* timing) {

  if (timing == nullptr) {
    return Status::InvalidArgument(
      "Profiled VarDCT encoding timing output is null");
  }
  return EncodeLinearRgbVarDctCodestreamImpl(
    linear_rgb, options, nullptr, false, true, codestream, summary, timing,
    nullptr, gpu_profile_internal::GpuProfilingMode::kDisabled, nullptr);
}

namespace codestream_internal {

bool ShouldComputeQuantizationMatrixScaleStats(
  const VarDctEncodingOptions& options) noexcept {

  return options.rate_control_mode != VarDctRateControlMode::kMaximumError &&
    (options.effort >= 7 ||
     options.density_mode == VarDctDensityMode::kHighDensity);
}

VarDctEntropyBehavior ResolveEntropyBehavior(
  const VarDctEncodingOptions& options) noexcept {

  if (options.compression_mode ==
      VarDctCompressionMode::kMaximumCompression) {
    return VarDctEntropyBehavior::kMaximumCompression;
  }
  if (options.effort >= 9 ||
      options.density_mode == VarDctDensityMode::kHighDensity) {
    return VarDctEntropyBehavior::kHighDensity;
  }
  return VarDctEntropyBehavior::kBalanced;
}

VarDctCoefficientOrderBehavior ResolveCoefficientOrderBehavior(
  const VarDctEncodingOptions& options) noexcept {

  return options.effort == 7 &&
      options.density_mode == VarDctDensityMode::kDefault &&
      options.compression_mode == VarDctCompressionMode::kAutomatic
    ? VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled
    : VarDctCoefficientOrderBehavior::kFull;
}

bool IsAutomaticMetalGeometryEligible(Extent2D padded_extent) noexcept {
  size_t area = 0;
  return padded_extent.try_area(&area) &&
    area >= kAutomaticMetalMinimumCodingPixels &&
    std::min(padded_extent.width, padded_extent.height) >=
      kAutomaticMetalMinimumCodingDimension;
}

bool IsAutomaticMetalTargetEligible(float butteraugli_target) noexcept {
  return std::isfinite(butteraugli_target) &&
    butteraugli_target >= kAutomaticMetalMinimumButteraugliTarget &&
    butteraugli_target <= kAutomaticMetalMaximumButteraugliTarget;
}

bool IsAutomaticMetalBackendQualified(
  const GpuBackend& backend) noexcept {

  return backend.kind() == BackendKind::kMetal &&
    backend.name() == kQualifiedMetalBackend;
}

Status EnsureProductionMetalBackendAvailable() {
  GpuBackend* backend = nullptr;
  Status status = ResolveProductionMetalBackend(&backend);
  if (!status.ok()) {
    return status;
  }
  if (backend == nullptr || !HasRequiredGpuQuantizationCapabilities(
        *backend, GpuAdaptiveQuantizationMode::kFullyResident)) {
    return Status::Unavailable(
      "Production Metal backend lacks a required GPU capability");
  }
  return Status::Ok();
}

Status EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary) {

  return EncodeLinearRgbVarDctCodestreamImpl(
    linear_rgb, options, backend, backend_is_qualified_for_automatic, false,
    codestream, summary, nullptr, nullptr,
    gpu_profile_internal::GpuProfilingMode::kDisabled, nullptr);
}

Status EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingProfile* profile) {

  if (profile == nullptr) {
    return Status::InvalidArgument("VarDCT encoding profile output is null");
  }
  return EncodeLinearRgbVarDctCodestreamImpl(
    linear_rgb, options, backend, backend_is_qualified_for_automatic, false,
    codestream, summary, nullptr, profile,
    gpu_profile_internal::GpuProfilingMode::kDisabled, nullptr);
}

Status EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  gpu_profile_internal::GpuProfilingMode profiling_mode,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingProfile* profile,
  gpu_profile_internal::GpuExecutionProfile* gpu_profile) {

  if (profile == nullptr || gpu_profile == nullptr) {
    return Status::InvalidArgument(
      "VarDCT GPU profile outputs are null");
  }
  return EncodeLinearRgbVarDctCodestreamImpl(
    linear_rgb, options, backend, backend_is_qualified_for_automatic, false,
    codestream, summary, nullptr, profile, profiling_mode, gpu_profile);
}

}  // namespace codestream_internal

}  // namespace gjxl
