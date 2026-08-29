// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/workflow.h"

#include <algorithm>
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

#include "codec/color_transform.h"
#include "codec/quantization_pipeline.h"
#include "codec/quantization_pipeline_internal.h"
#include "codec/vardct_frame.h"
#include "codestream/encoder.h"
#include "codestream/encoder_internal.h"
#include "codestream/rate_control_internal.h"
#include "codestream/workflow_internal.h"
#include "core/frame_geometry.h"
#include "core/image_buffer.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/ac_strategy.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/quantization_pipeline.h"
#include "gpu/ops/quantization_pipeline_profile_internal.h"

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

  destination->validation_nanoseconds += source.validation_nanoseconds;
  destination->dc_tokenization_nanoseconds +=
    source.dc_tokenization_nanoseconds;
  destination->ac_tokenization_nanoseconds +=
    source.ac_tokenization_nanoseconds;
  destination->entropy_optimization_nanoseconds +=
    source.entropy_optimization_nanoseconds;
  destination->entropy_model_bits += source.entropy_model_bits;
  destination->entropy_token_bits += source.entropy_token_bits;
  destination->dc_entropy_clusters += source.dc_entropy_clusters;
  destination->ac_entropy_clusters += source.ac_entropy_clusters;
  destination->natural_candidate_bytes += source.natural_candidate_bytes;
  destination->custom_order_candidate_bytes +=
    source.custom_order_candidate_bytes;
  destination->selected_coefficient_order_mask |=
    source.selected_coefficient_order_mask;
  destination->section_writing_nanoseconds +=
    source.section_writing_nanoseconds;
  destination->assembly_nanoseconds += source.assembly_nanoseconds;
  destination->total_nanoseconds += source.total_nanoseconds;
}

void AccumulateEncodingProfile(
  const codestream_internal::VarDctEncodingProfile& source,
  codestream_internal::VarDctEncodingProfile* destination) {

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
// Policy sweeps preserve exact frame decisions in this closed interval;
// threshold-sensitive fixtures outside it remain on the CPU in auto mode.
constexpr float kAutomaticMetalMinimumButteraugliTarget = 1.0f;
constexpr float kAutomaticMetalMaximumButteraugliTarget = 1.2f;
constexpr std::string_view kQualifiedMetalBackend = "Metal: Apple M4 Pro";

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
     QueryGpuAcStrategyEvaluation(backend) != nullptr);
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

[[nodiscard]] Status EdgeExtend(
  ConstImage3FView source,
  Image3FView destination) {

  if (!source.valid() || !destination.valid() ||
      source.width() > destination.width() ||
      source.height() > destination.height()) {
    return Status::InvalidArgument(
      "Linear RGB source or padded destination is invalid");
  }
  for (size_t y = 0; y < destination.height(); ++y) {
    const size_t source_y = std::min(y, source.height() - 1);
    for (size_t x = 0; x < destination.width(); ++x) {
      const size_t source_x = std::min(x, source.width() - 1);
      for (size_t channel = 0; channel < 3; ++channel) {
        const float value = source.plane[channel].Row(source_y)[source_x];
        if (!std::isfinite(value)) {
          return Status::InvalidArgument(
            "Linear RGB input pixels must be finite");
        }
        destination.plane[channel].Row(y)[x] = value;
      }
    }
  }
  return Status::Ok();
}

struct PreparedWorkflow {
  explicit PreparedWorkflow(FrameGeometry prepared_geometry)
    : geometry(prepared_geometry),
      padded_linear(geometry.padded_frame()),
      opsin(geometry.padded_frame()),
      pipeline(geometry.frame(), geometry.padded_frame()) {}

  [[nodiscard]] ConstImage3FView original_linear_rgb() const noexcept {
    return padded_linear.cropped_view(geometry.frame());
  }

  FrameGeometry geometry;
  Image3FBuffer padded_linear;
  Image3FBuffer opsin;
  codestream_internal::QuantizationMatrixScaleStats matrix_scale_stats;
  PipelineStorage pipeline;
  quantization_pipeline_internal::PreparedQuantizationPipeline quantization;
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    gpu_adaptive_quantization;
};

[[nodiscard]] Status PrepareWorkflow(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::unique_ptr<PreparedWorkflow>* prepared) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Prepared workflow output pointer is null");
  }
  prepared->reset();
  try {
    FrameGeometry geometry;
    Status status = FrameGeometry::Create(linear_rgb.extent(), &geometry);
    if (!status.ok()) {
      return status;
    }
    auto candidate = std::make_unique<PreparedWorkflow>(geometry);
    status = EdgeExtend(linear_rgb, candidate->padded_linear.view());
    if (!status.ok()) {
      return status;
    }
    status = LinearRgbToOpsin(
      candidate->padded_linear.const_view(),
      kInitialProfileIntensityTarget,
      candidate->opsin.view());
    if (!status.ok()) {
      return status;
    }
    status = codestream_internal::ComputeQuantizationMatrixScaleStats(
      candidate->opsin.cropped_view(geometry.frame()),
      &candidate->matrix_scale_stats);
    if (!status.ok()) {
      return status;
    }
    CpuQuantizationPipelineOptions preparation_options;
    if (options.rate_control_mode == VarDctRateControlMode::kMaximumError) {
      preparation_options.adaptive_quantization.control_mode =
        AdaptiveQuantizationControlMode::kMaximumError;
      preparation_options.adaptive_quantization.maximum_error =
        options.maximum_error;
    }
    status = quantization_pipeline_internal::PrepareQuantizationPipeline(
      candidate->original_linear_rgb(), candidate->opsin.const_view(),
      preparation_options,
      &candidate->quantization,
      options.backend != VarDctBackendPreference::kMetal,
      options.metal_aq_mode ==
        GpuAdaptiveQuantizationMode::kExactCoefficients);
    if (!status.ok()) {
      return status;
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
  PreparedWorkflow& prepared,
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
       prepared.geometry.padded_frame()) &&
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
  status = SelectAttemptBackend(
    prepared, options, supplied_backend, supplied_backend_is_qualified,
    resolve_production_backend, &selected_gpu, &selected_metal);
  if (!status.ok()) {
    return status;
  }
  ProfileEnd(
    profile, selection_begin,
    &candidate_profile.backend_selection_nanoseconds);
  const WorkflowClock::time_point pipeline_begin = ProfileBegin(profile);
  prepared.pipeline.score_history.clear();
  prepared.pipeline.maximum_error_result = {};
  if (selected_metal && options.metal_aq_mode ==
        GpuAdaptiveQuantizationMode::kMaximumThroughput) {
    const CpuQuantizationPipelineOutput pipeline_output =
      prepared.pipeline.Output();
    status = RunGpuFrameOnlyQuantizationPipeline(
      *selected_gpu, prepared.original_linear_rgb(),
      prepared.opsin.const_view(), pipeline_options,
      {
        .initial_quantization = pipeline_output.initial_quantization,
        .quant_field = {
          prepared.pipeline.final_quant.data(), prepared.pipeline.block_extent,
          prepared.pipeline.block_extent.width},
        .frame = &prepared.pipeline.frame,
      });
  } else if (selected_metal) {
    const quantization_pipeline_internal::GpuEncodingQuantizationPipelineOutput
      encoding_output{
      .frame = &prepared.pipeline.frame,
      .score_history = &prepared.pipeline.score_history,
      .maximum_error_result = &prepared.pipeline.maximum_error_result,
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
    status = quantization_pipeline_internal::
      RunPreparedCpuQuantizationPipeline(
        prepared.original_linear_rgb(), prepared.quantization,
        pipeline_options, prepared.pipeline.Output());
  }
  if (!status.ok()) {
    return status;
  }
  ProfileEnd(
    profile, pipeline_begin,
    &candidate_profile.quantization_pipeline_nanoseconds);

  std::vector<uint8_t> candidate;
  const WorkflowClock::time_point codestream_begin = ProfileBegin(profile);
  status = profile == nullptr
    ? EncodeVarDctCodestream(prepared.pipeline.frame, &candidate)
    : codestream_internal::EncodeVarDctCodestreamProfiled(
        prepared.pipeline.frame, &candidate, &candidate_profile.codestream);
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
      prepared.pipeline.maximum_error_result.achieved;
    candidate_summary.achieved_maximum_error_ratio =
      prepared.pipeline.maximum_error_result.normalized_maximum;
    candidate_summary.maximum_error_evaluation_count =
      prepared.pipeline.maximum_error_result.evaluation_count;
    candidate_summary.maximum_error_outcome =
      prepared.pipeline.maximum_error_result.outcome;
  }
  candidate_summary.encode_attempt_count = 1;
  candidate_summary.score_history = prepared.pipeline.score_history;
  candidate_summary.execution_backend = selected_metal
    ? VarDctExecutionBackend::kMetal
    : VarDctExecutionBackend::kCpu;
  candidate_summary.metal_aq_mode = options.metal_aq_mode;
  status = prepared.pipeline.frame.strategies().ForEachAnchor(
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
      const float previous_row_x = opsin.plane[0].Row(y - 1)[x];
      const float horizontal_x_edge =
        std::abs(current_x - previous_x);
      const float vertical_x_edge =
        std::abs(current_x - previous_row_x);
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
      const float previous_row_y = opsin.plane[1].Row(y - 1)[x];
      const float previous_row_b = opsin.plane[2].Row(y - 1)[x];
      const float current_difference = current_b - current_y;
      const float horizontal_b_edge = std::abs(
        current_difference - (previous_b - previous_y));
      const float vertical_b_edge = std::abs(
        current_difference - (previous_row_b - previous_row_y));
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
          std::abs(current_b - previous_row_b);
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
  switch (options.backend) {
    case VarDctBackendPreference::kAutomatic:
    case VarDctBackendPreference::kCpu:
    case VarDctBackendPreference::kMetal:
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT encoding backend preference is invalid");
  }
  switch (options.metal_aq_mode) {
    case GpuAdaptiveQuantizationMode::kExactCoefficients:
      break;
    case GpuAdaptiveQuantizationMode::kFullyResident:
    case GpuAdaptiveQuantizationMode::kThroughput:
    case GpuAdaptiveQuantizationMode::kMaximumThroughput:
      if (options.backend != VarDctBackendPreference::kMetal) {
        return Status::InvalidArgument(
          "Experimental AQ requires an explicitly forced Metal backend");
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
  if (options.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
      options.rate_control_mode ==
        VarDctRateControlMode::kTargetBitsPerPixel) {
    try {
      std::unique_ptr<PreparedWorkflow> prepared;
      const auto preparation_begin = !profiling
        ? WorkflowClock::time_point{}
        : WorkflowClock::now();
      status = PrepareWorkflow(linear_rgb, options, &prepared);
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
    status = PrepareWorkflow(linear_rgb, options, &prepared);
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
