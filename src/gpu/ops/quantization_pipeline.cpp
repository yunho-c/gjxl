// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/quantization_pipeline.h"

#include <cmath>
#include <new>
#include <stdexcept>
#include <vector>

#include "codec/chroma_from_luma_internal.h"
#include "codec/gaborish.h"
#include "codec/quantization_pipeline_internal.h"
#include "core/block_grid.h"
#include "core/image_ops.h"
#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/adaptive_quantization_profile_internal.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/gaborish.h"
#include "gpu/ops/quantization_pipeline_profile_internal.h"

namespace gjxl {
namespace {

class GpuPipelineGaborishProvider final
    : public quantization_pipeline_internal::GaborishInverseProvider {
public:
  GpuPipelineGaborishProvider(GpuBackend& gpu, bool use_gpu)
    : gpu_(gpu), use_gpu_(use_gpu) {}

  Status Apply(
    ConstImage3FView input,
    std::array<float, 3> multipliers,
    Image3FView output) override {

    return use_gpu_
      ? ApplyGaborishInverseGpu(gpu_, input, multipliers, output)
      : ApplyGaborishInverse(input, multipliers, output);
  }

private:
  GpuBackend& gpu_;
  bool use_gpu_;
};

class GpuAcStrategySearchProvider final : public AcStrategySearchProvider {
public:
  explicit GpuAcStrategySearchProvider(
      GpuBackend& gpu,
      const ResidentAcStrategySearchInputs* resident = nullptr)
    : gpu_(gpu), resident_(resident) {}

  Status Find(
    ConstImage3FView opsin,
    ConstPlaneF32View quant_field,
    ConstPlaneF32View pixel_mask,
    const ColorCorrelationMap& color_correlation,
    AcStrategySearchOptions options,
    AcStrategyGrid* out) override {

    return resident_ == nullptr
      ? FindAcStrategyGridGpu(
          gpu_, opsin, quant_field, pixel_mask, color_correlation,
          options, out, &stats_)
      : FindAcStrategyGridGpuResident(
          gpu_, opsin, quant_field, pixel_mask, color_correlation,
          *resident_, options, out, &stats_);
  }

  [[nodiscard]] const AcStrategyGpuSearchStats& stats() const noexcept {
    return stats_;
  }

private:
  GpuBackend& gpu_;
  const ResidentAcStrategySearchInputs* resident_ = nullptr;
  AcStrategyGpuSearchStats stats_;
};

class GpuAdaptiveQuantizationProvider final
    : public quantization_pipeline_internal::AdaptiveQuantizationProvider {
public:
  GpuAdaptiveQuantizationProvider(
    GpuBackend& gpu,
    GpuAdaptiveQuantizationMode mode,
    adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
      prepared,
    adaptive_quantization_gpu_internal::AdaptiveQuantizationMaterialization
      materialization = {},
    gpu_profile_internal::GpuProfilingMode profiling_mode =
      gpu_profile_internal::GpuProfilingMode::kDisabled,
    gpu_profile_internal::GpuExecutionProfile* profile = nullptr)
    : gpu_(gpu), mode_(mode), prepared_(prepared),
      materialization_(materialization), profiling_mode_(profiling_mode),
      profile_(profile) {}

  Status Find(
    ConstImage3FView original_linear_rgb,
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    PreparedButteraugliReference*,
    AdaptiveQuantizationOutput output) override {

    if (prepared_ != nullptr) {
      if (profiling_mode_ !=
          gpu_profile_internal::GpuProfilingMode::kDisabled) {
        return adaptive_quantization_gpu_internal::
          RunPreparedGpuAdaptiveQuantizationProfiled(
            gpu_, original_linear_rgb, opsin, strategies,
            initial_quant_field, epf_sharpness, options, mode_, prepared_,
            output, materialization_, profiling_mode_, profile_);
      }
      return adaptive_quantization_gpu_internal::
        RunPreparedGpuAdaptiveQuantization(
          gpu_, original_linear_rgb, opsin, strategies,
          initial_quant_field, epf_sharpness, options, mode_, prepared_,
          output, materialization_);
    }
    return RunGpuAdaptiveQuantization(
      gpu_, original_linear_rgb, opsin, strategies, initial_quant_field,
      epf_sharpness, options, mode_, output);
  }

private:
  GpuBackend& gpu_;
  GpuAdaptiveQuantizationMode mode_;
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_ = nullptr;
  adaptive_quantization_gpu_internal::AdaptiveQuantizationMaterialization
    materialization_;
  gpu_profile_internal::GpuProfilingMode profiling_mode_ =
    gpu_profile_internal::GpuProfilingMode::kDisabled;
  gpu_profile_internal::GpuExecutionProfile* profile_ = nullptr;
};

bool SamePlaneIdentity(ConstPlaneF32View left, ConstPlaneF32View right) {
  return left.data == right.data && left.extent == right.extent &&
    left.stride == right.stride;
}

bool SameImageIdentity(ConstImage3FView left, ConstImage3FView right) {
  return SamePlaneIdentity(left.plane[0], right.plane[0]) &&
    SamePlaneIdentity(left.plane[1], right.plane[1]) &&
    SamePlaneIdentity(left.plane[2], right.plane[2]);
}

Status PrepareResidentAcStrategyInputs(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  quantization_pipeline_internal::PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization& state,
  ResidentAcStrategySearchInputs* resident) {

  if (resident == nullptr) {
    return Status::InvalidArgument(
        "Resident AC-strategy input output is null");
  }
  const bool compatible = state.resident_frontend != nullptr &&
    state.resident_frontend_backend == &gpu &&
    SameImageIdentity(state.resident_frontend_original_linear_rgb,
                      original_linear_rgb) &&
    SameImageIdentity(state.resident_frontend_coding_opsin,
                      prepared.coding_opsin.const_view()) &&
    state.resident_frontend_profile == options.adaptive_quantization.profile;
  if (!compatible) {
    state.resident_frontend.reset();
    AcStrategyGrid provisional_strategies;
    Status status = AcStrategyGrid::Create(
        prepared.block_extent, &provisional_strategies);
    if (!status.ok()) return status;
    provisional_strategies.fill_dct8();
    status = PrepareAqEvaluation(
      gpu,
      {
        .original_linear_rgb = original_linear_rgb,
        .coding_opsin = prepared.coding_opsin.const_view(),
        .strategies = &provisional_strategies,
        .epf_sharpness = {
          prepared.epf_sharpness.data(), prepared.block_extent,
          prepared.block_extent.width},
        .options = {
          options.adaptive_quantization.profile,
          options.adaptive_quantization.butteraugli,
        },
        .frame_only = true,
        .frame_only_resident_initial_quant = true,
        .resident_ac_strategy_inputs = true,
        .coefficient_decision_mode =
          AcCoefficientDecisionMode::kAdjustedSharedQuant,
      },
      &state.resident_frontend);
    if (!status.ok()) return status;
    state.resident_frontend_backend = &gpu;
    state.resident_frontend_original_linear_rgb = original_linear_rgb;
    state.resident_frontend_coding_opsin =
      prepared.coding_opsin.const_view();
    state.resident_frontend_profile = options.adaptive_quantization.profile;
  }

  constexpr float kMaximumErrorInitializationTarget = 1.0f;
  const float control_target =
    options.adaptive_quantization.control_mode ==
        AdaptiveQuantizationControlMode::kMaximumError
      ? kMaximumErrorInitializationTarget
      : options.butteraugli_target;
  const float initial_quant_target =
    options.adaptive_quantization.profile.loop_filter.gaborish
      ? control_target : 0.62f * control_target;
  Status status = state.resident_frontend->ComputeInitialQuantization(
    {
      .butteraugli_target = initial_quant_target,
      .rescale = options.initial_quant_rescale,
    },
    {
      .quant_field = {
        prepared.initial_quant.data(), prepared.block_extent,
        prepared.block_extent.width},
      .strategy_mask = {
        prepared.strategy_mask.data(), prepared.block_extent,
        prepared.block_extent.width},
      .pixel_mask = {
        prepared.pixel_mask.data(), prepared.padded_extent,
        prepared.padded_extent.width},
    });
  if (!status.ok()) {
    state.resident_frontend.reset();
    return status;
  }
  ResidentAcStrategyInputs views;
  status = state.resident_frontend->GetResidentAcStrategyInputs(&views);
  if (!status.ok()) {
    state.resident_frontend.reset();
    return status;
  }
  *resident = {
      .opsin = views.opsin,
      .quant_field = views.quant_field,
      .pixel_mask = views.pixel_mask,
  };
  return Status::Ok();
}

}  // namespace

Status RunGpuFrameOnlyQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuFrameOnlyPipelineOutput output) {

  if (!original_linear_rgb.valid() || !opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent()) ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f ||
      !std::isfinite(options.initial_quant_rescale) ||
      options.initial_quant_rescale <= 0.0f ||
      !options.adaptive_quantization.profile.valid() ||
      output.frame == nullptr) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline inputs or options are invalid");
  }
  const Extent2D block_extent =
    BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
  if (!output.initial_quantization.quant_field.valid() ||
      output.initial_quantization.quant_field.extent != block_extent ||
      !output.initial_quantization.strategy_mask.valid() ||
      output.initial_quantization.strategy_mask.extent != block_extent ||
      !output.initial_quantization.pixel_mask.valid() ||
      output.initial_quantization.pixel_mask.extent != opsin.extent() ||
      !output.quant_field.valid() || output.quant_field.extent != block_extent) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline output geometry is invalid");
  }

  size_t block_count = 0;
  size_t pixel_count = 0;
  if (!block_extent.try_area(&block_count) ||
      !opsin.extent().try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline dimensions are too large");
  }
  try {
    std::vector<float> initial_quant(block_count);
    std::vector<float> strategy_mask(block_count);
    std::vector<float> pixel_mask(pixel_count);
    const float initial_quant_target =
      options.adaptive_quantization.profile.loop_filter.gaborish
        ? options.butteraugli_target
        : 0.62f * options.butteraugli_target;
    AcStrategyGrid strategies;
    Status status = AcStrategyGrid::Create(block_extent, &strategies);
    if (!status.ok()) return status;
    strategies.fill_dct8();
    std::vector<uint8_t> sharpness(block_count);
    status = FillDefaultEpfSharpness(
      {sharpness.data(), block_extent, block_extent.width});
    if (!status.ok()) return status;
    std::vector<float> final_quant(block_count);
    VarDctEncoderFrame frame;
    AdaptiveQuantizationOptions adaptive_options =
      options.adaptive_quantization;
    adaptive_options.butteraugli_target = options.butteraugli_target;
    status = RunGpuFrameOnlyQuantizationResidentFrontend(
      gpu, original_linear_rgb, opsin, strategies,
      {sharpness.data(), block_extent, block_extent.width},
      {
        .butteraugli_target = initial_quant_target,
        .rescale = options.initial_quant_rescale,
      },
      adaptive_options,
      {
        .quant_field = {
          initial_quant.data(), block_extent, block_extent.width},
        .strategy_mask = {
          strategy_mask.data(), block_extent, block_extent.width},
        .pixel_mask = {
          pixel_mask.data(), opsin.extent(), opsin.width()},
      },
      {
        .quant_field = {
          final_quant.data(), block_extent, block_extent.width},
        .frame = &frame,
      });
    if (!status.ok()) return status;

    CopyContiguousPlane(
      initial_quant, output.initial_quantization.quant_field);
    CopyContiguousPlane(
      strategy_mask, output.initial_quantization.strategy_mask);
    CopyContiguousPlane(pixel_mask, output.initial_quantization.pixel_mask);
    CopyContiguousPlane(final_quant, output.quant_field);
    *output.frame = std::move(frame);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU frame-only pipeline storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline dimensions are too large");
  }
}

Status RunGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats) {

  return RunGpuQuantizationPipeline(
    gpu, original_linear_rgb, opsin, options,
    GpuAdaptiveQuantizationMode::kExactCoefficients, output, stats);
}

Status RunGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats) {

  switch (aq_mode) {
    case GpuAdaptiveQuantizationMode::kExactCoefficients:
    case GpuAdaptiveQuantizationMode::kFullyResident:
    case GpuAdaptiveQuantizationMode::kThroughput:
      break;
    default:
      return Status::InvalidArgument(
        "GPU quantization pipeline AQ mode is invalid");
  }
  if (QueryGpuAqEvaluation(gpu) == nullptr) {
    return Status::Unavailable(
      "GPU quantization pipeline requires prepared AQ support");
  }
  const bool resident =
    aq_mode != GpuAdaptiveQuantizationMode::kExactCoefficients;
  quantization_pipeline_internal::PreparedQuantizationPipeline prepared;
  Status status = quantization_pipeline_internal::PrepareQuantizationPipeline(
    original_linear_rgb, opsin, options, &prepared, false,
    !resident);
  if (!status.ok()) {
    return status;
  }
  return quantization_pipeline_internal::RunPreparedGpuQuantizationPipeline(
    gpu, original_linear_rgb, prepared, options, aq_mode, output, stats);
}

namespace quantization_pipeline_internal {
namespace {

Status RunPreparedGpuQuantizationPipelineImpl(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_aq,
  QuantizationPipelineMaterialization materialization,
  gpu_profile_internal::GpuProfilingMode profiling_mode,
  gpu_profile_internal::GpuExecutionProfile* profile) {

  switch (aq_mode) {
    case GpuAdaptiveQuantizationMode::kExactCoefficients:
    case GpuAdaptiveQuantizationMode::kFullyResident:
    case GpuAdaptiveQuantizationMode::kThroughput:
      break;
    default:
      return Status::InvalidArgument(
        "GPU quantization pipeline AQ mode is invalid");
  }
  if (QueryGpuAqEvaluation(gpu) == nullptr) {
    return Status::Unavailable(
      "GPU quantization pipeline requires prepared AQ support");
  }
  const bool resident =
    aq_mode != GpuAdaptiveQuantizationMode::kExactCoefficients;
  if (aq_mode == GpuAdaptiveQuantizationMode::kThroughput) {
    options.adaptive_quantization.iterations = 1;
  }
  if (!prepared.preprocessing_ready ||
      prepared.fast_initial_color_correlation != resident) {
    GpuPipelineGaborishProvider gaborish_inverse(gpu, resident);
    Status status = PrepareQuantizationPreprocessing(
      prepared, gaborish_inverse, resident);
    if (!status.ok()) {
      return status;
    }
  }
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    local_prepared_aq;
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    aq_state = prepared_aq;
  ResidentAcStrategySearchInputs resident_inputs;
  if (resident) {
    if (aq_state == nullptr) aq_state = &local_prepared_aq;
    Status status = PrepareResidentAcStrategyInputs(
        gpu, original_linear_rgb, prepared, options, *aq_state,
        &resident_inputs);
    if (!status.ok()) return status;
  }
  GpuAcStrategySearchProvider strategy_search(
      gpu, resident ? &resident_inputs : nullptr);
  GpuAdaptiveQuantizationProvider adaptive_quantization(
    gpu, aq_mode, aq_state,
    {
      .quant_field = materialization.adaptive_quant_field,
      .block_distance_map = materialization.block_distance_map,
      .reconstructed_linear_rgb =
        materialization.reconstructed_linear_rgb,
    }, profiling_mode, profile);
  const Status status = RunPreparedQuantizationPipelineWithProviders(
    original_linear_rgb, prepared, strategy_search, adaptive_quantization,
    options, output, resident, materialization);
  if (!status.ok()) {
    return status;
  }
  if (stats != nullptr) {
    *stats = strategy_search.stats();
  }
  return Status::Ok();
}

}  // namespace

Status RunPreparedGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_aq) {

  return RunPreparedGpuQuantizationPipelineImpl(
    gpu, original_linear_rgb, prepared, options, aq_mode, output, stats,
    prepared_aq, {}, gpu_profile_internal::GpuProfilingMode::kDisabled,
    nullptr);
}

Status RunPreparedGpuQuantizationPipelineForEncoding(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  GpuEncodingQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_aq) {

  if ((aq_mode != GpuAdaptiveQuantizationMode::kExactCoefficients &&
       aq_mode != GpuAdaptiveQuantizationMode::kFullyResident &&
       aq_mode != GpuAdaptiveQuantizationMode::kThroughput) ||
      output.frame == nullptr || output.score_history == nullptr ||
      (options.adaptive_quantization.control_mode ==
         AdaptiveQuantizationControlMode::kMaximumError &&
       output.maximum_error_result == nullptr)) {
    return Status::InvalidArgument(
      "Encoding-only GPU pipeline output is invalid");
  }
  const CpuQuantizationPipelineOutput pipeline_output{
    .initial_quantization = {
      .quant_field = {
        prepared.initial_quant.data(), prepared.block_extent,
        prepared.block_extent.width},
      .strategy_mask = {
        prepared.strategy_mask.data(), prepared.block_extent,
        prepared.block_extent.width},
      .pixel_mask = {
        prepared.pixel_mask.data(), prepared.padded_extent,
        prepared.padded_extent.width},
    },
    .adaptive_quantization = {
      .quant_field = {
        prepared.final_quant.data(), prepared.block_extent,
        prepared.block_extent.width},
      .block_distance_map = {
        prepared.block_distance.data(), prepared.block_extent,
        prepared.block_extent.width},
      .reconstructed_linear_rgb = prepared.reconstructed_linear.view(),
      .frame = output.frame,
      .score_history = output.score_history,
      .maximum_error_result = output.maximum_error_result,
    },
  };
  return RunPreparedGpuQuantizationPipelineImpl(
    gpu, original_linear_rgb, prepared, options, aq_mode, pipeline_output,
    stats, prepared_aq,
    {
      .initial_quantization = false,
      .adaptive_quant_field = false,
      .block_distance_map = false,
      .reconstructed_linear_rgb = false,
    },
    gpu_profile_internal::GpuProfilingMode::kDisabled, nullptr);
}

Status RunPreparedGpuQuantizationPipelineForEncodingProfiled(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  GpuEncodingQuantizationPipelineOutput output,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_aq,
  gpu_profile_internal::GpuProfilingMode profiling_mode,
  gpu_profile_internal::GpuExecutionProfile* profile) {

  if ((aq_mode != GpuAdaptiveQuantizationMode::kFullyResident &&
       aq_mode != GpuAdaptiveQuantizationMode::kThroughput) ||
      output.frame == nullptr || output.score_history == nullptr ||
      profile == nullptr ||
      profiling_mode == gpu_profile_internal::GpuProfilingMode::kDisabled) {
    return Status::InvalidArgument(
      "Profiled encoding-only GPU pipeline request is invalid");
  }
  const CpuQuantizationPipelineOutput pipeline_output{
    .initial_quantization = {
      .quant_field = {
        prepared.initial_quant.data(), prepared.block_extent,
        prepared.block_extent.width},
      .strategy_mask = {
        prepared.strategy_mask.data(), prepared.block_extent,
        prepared.block_extent.width},
      .pixel_mask = {
        prepared.pixel_mask.data(), prepared.padded_extent,
        prepared.padded_extent.width},
    },
    .adaptive_quantization = {
      .quant_field = {
        prepared.final_quant.data(), prepared.block_extent,
        prepared.block_extent.width},
      .block_distance_map = {
        prepared.block_distance.data(), prepared.block_extent,
        prepared.block_extent.width},
      .reconstructed_linear_rgb = prepared.reconstructed_linear.view(),
      .frame = output.frame,
      .score_history = output.score_history,
      .maximum_error_result = output.maximum_error_result,
    },
  };
  return RunPreparedGpuQuantizationPipelineImpl(
    gpu, original_linear_rgb, prepared, options, aq_mode, pipeline_output,
    nullptr, prepared_aq,
    {
      .initial_quantization = false,
      .adaptive_quant_field = false,
      .block_distance_map = false,
      .reconstructed_linear_rgb = false,
    },
    profiling_mode, profile);
}

}  // namespace quantization_pipeline_internal

}  // namespace gjxl
