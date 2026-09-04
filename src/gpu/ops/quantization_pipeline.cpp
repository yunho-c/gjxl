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
#include "gpu/ops/ac_strategy_search_profile_internal.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/aq_evaluation_internal.h"
#include "gpu/ops/gaborish.h"
#include "gpu/ops/gaborish_profile_internal.h"
#include "gpu/ops/quantization_pipeline_profile_internal.h"

namespace gjxl {
namespace {

class GpuPipelineGaborishProvider final
    : public quantization_pipeline_internal::GaborishInverseProvider {
public:
  GpuPipelineGaborishProvider(
      GpuBackend& gpu, bool use_gpu,
      gpu_profile_internal::GpuProfilingSession* profiling_session)
    : gpu_(gpu), use_gpu_(use_gpu),
      profiling_session_(profiling_session) {}

  Status Apply(
    ConstImage3FView input,
    std::array<float, 3> multipliers,
    Image3FView output) override {

    if (!use_gpu_) {
      return ApplyGaborishInverse(input, multipliers, output);
    }
    return profiling_session_ == nullptr
      ? ApplyGaborishInverseGpu(gpu_, input, multipliers, output)
      : gpu_profile_internal::ApplyGaborishInverseGpuProfiled(
          gpu_, input, multipliers, output, profiling_session_);
  }

private:
  GpuBackend& gpu_;
  bool use_gpu_;
  gpu_profile_internal::GpuProfilingSession* profiling_session_ = nullptr;
};

class GpuAcStrategySearchProvider final : public AcStrategySearchProvider {
public:
  explicit GpuAcStrategySearchProvider(
      GpuBackend& gpu,
      const ResidentAcStrategySearchInputs* resident = nullptr,
      PreparedAcStrategySearch* prepared = nullptr,
      gpu_profile_internal::GpuProfilingSession* profiling_session = nullptr)
    : gpu_(gpu), resident_(resident),
      prepared_(prepared), profiling_session_(profiling_session) {}

  Status Find(
    ConstImage3FView opsin,
    ConstPlaneF32View quant_field,
    ConstPlaneF32View pixel_mask,
    const ColorCorrelationMap& color_correlation,
    AcStrategySearchOptions options,
    AcStrategyGrid* out) override {

    if (resident_ == nullptr) {
      return FindAcStrategyGridGpu(
        gpu_, opsin, quant_field, pixel_mask, color_correlation,
        options, out, &stats_);
    }
    return profiling_session_ == nullptr
      ? FindAcStrategyGridGpuResident(
          gpu_, opsin, quant_field, pixel_mask, color_correlation,
          *resident_, options, out, &stats_, prepared_)
      : gpu_profile_internal::FindAcStrategyGridGpuResidentProfiled(
          gpu_, opsin, quant_field, pixel_mask, color_correlation,
          *resident_, options, out, prepared_, profiling_session_, &stats_);
  }

  [[nodiscard]] const AcStrategyGpuSearchStats& stats() const noexcept {
    return stats_;
  }

private:
  GpuBackend& gpu_;
  const ResidentAcStrategySearchInputs* resident_ = nullptr;
  PreparedAcStrategySearch* prepared_ = nullptr;
  gpu_profile_internal::GpuProfilingSession* profiling_session_ = nullptr;
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
    gpu_profile_internal::GpuProfilingSession* profiling_session = nullptr)
    : gpu_(gpu), mode_(mode), prepared_(prepared),
      materialization_(materialization),
      profiling_session_(profiling_session) {}

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
      if (profiling_session_ != nullptr) {
        return adaptive_quantization_gpu_internal::
          RunPreparedGpuAdaptiveQuantizationProfiled(
            gpu_, original_linear_rgb, opsin, strategies,
            initial_quant_field, epf_sharpness, options, mode_, prepared_,
            output, materialization_, profiling_session_);
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
  gpu_profile_internal::GpuProfilingSession* profiling_session_ = nullptr;
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

bool HasValidatedHostImages(
  const quantization_pipeline_internal::PreparedQuantizationPipeline&
    prepared,
  ConstImage3FView original_linear_rgb) {

  return prepared.validated_original_linear_rgb.valid() &&
    prepared.validated_coding_opsin.valid() &&
    SameImageIdentity(
      prepared.validated_original_linear_rgb, original_linear_rgb) &&
    SameImageIdentity(
      prepared.validated_coding_opsin, prepared.coding_opsin);
}

Status PrepareResidentAcStrategyInputs(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  quantization_pipeline_internal::PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization& state,
  bool encoding_only,
  ResidentAcStrategySearchInputs* resident,
  bool* resident_only_initial,
  gpu_profile_internal::GpuProfilingSession* profiling_session) {

  if (resident == nullptr || resident_only_initial == nullptr) {
    return Status::InvalidArgument(
        "Resident AC-strategy input output is null");
  }
  *resident_only_initial = false;
  const AqEvaluationOptions evaluation_options{
    .profile = options.adaptive_quantization.profile,
    .butteraugli = options.adaptive_quantization.butteraugli,
    .metric = options.adaptive_quantization.control_mode ==
          AdaptiveQuantizationControlMode::kMaximumError
      ? AqEvaluationMetric::kMaximumError
      : AqEvaluationMetric::kButteraugli,
    .maximum_error = options.adaptive_quantization.maximum_error,
  };
  const bool same_preparation = state.evaluation != nullptr &&
    state.quantization_pipeline_generation == prepared.generation &&
    state.backend == &gpu &&
    SameImageIdentity(state.original_linear_rgb, original_linear_rgb) &&
    SameImageIdentity(state.coding_opsin, prepared.coding_opsin) &&
    state.resident_quantization && !state.frame_only_resident_frontend;
  bool compatible = same_preparation &&
    state.evaluation_options == evaluation_options;
  Status status = Status::Ok();
  if (same_preparation && !compatible) {
    AqEvaluationOptions normalized_previous = state.evaluation_options;
    AqEvaluationOptions normalized_current = evaluation_options;
    normalized_previous.profile.x_qm_scale = 0;
    normalized_previous.profile.b_qm_scale = 0;
    normalized_current.profile.x_qm_scale = 0;
    normalized_current.profile.b_qm_scale = 0;
    if (normalized_previous == normalized_current) {
      auto* reconfiguration = dynamic_cast<
        aq_evaluation_internal::PreparedAqScaleReconfiguration*>(
          state.evaluation.get());
      if (reconfiguration != nullptr) {
        status = reconfiguration->ReconfigureScaleSelectors(
          evaluation_options);
        compatible = status.ok();
        if (compatible) state.evaluation_options = evaluation_options;
      }
    }
  }
  if (!status.ok()) {
    state.evaluation.reset();
    state.resident_coding_opsin = {};
    return status;
  }
  if (!compatible) {
    const auto preparation_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    state.evaluation.reset();
    state.resident_coding_opsin = {};
    AcStrategyGrid provisional_strategies;
    status = AcStrategyGrid::Create(
        prepared.block_extent, &provisional_strategies);
    if (!status.ok()) return status;
    provisional_strategies.fill_dct8();
    const AqEvaluationPreparation evaluation_preparation{
      .original_linear_rgb = original_linear_rgb,
      .coding_opsin = prepared.coding_opsin,
      .strategies = &provisional_strategies,
      .epf_sharpness = {
        prepared.epf_sharpness.data(), prepared.block_extent,
        prepared.block_extent.width},
      .options = evaluation_options,
      .resident_initial_cfl = true,
      .frame_only_resident_initial_quant = true,
      .resident_ac_strategy_inputs = true,
      .resident_quantization = true,
      .coefficient_decision_mode =
        AcCoefficientDecisionMode::kAdjustedSharedQuant,
    };
    auto* const validated_preparation = HasValidatedHostImages(
        prepared, original_linear_rgb)
      ? dynamic_cast<aq_evaluation_internal::GpuValidatedAqEvaluation*>(
          &gpu)
      : nullptr;
    if (profiling_session == nullptr) {
      status = validated_preparation == nullptr
        ? PrepareAqEvaluation(
            gpu, evaluation_preparation, &state.evaluation)
        : validated_preparation->PrepareValidatedAqEvaluation(
            evaluation_preparation, &state.evaluation);
    } else {
      auto* const preparation_profiler = dynamic_cast<
        gpu_profile_internal::GpuAqEvaluationProfiler*>(&gpu);
      if (validated_preparation == nullptr &&
          preparation_profiler == nullptr) {
        return Status::Unavailable(
          "Resident evaluator preparation cannot collect GPU diagnostics");
      }
      gpu_profile_internal::GpuExecutionProfile child_profile;
      status = validated_preparation == nullptr
        ? preparation_profiler->PrepareAqEvaluationProfiled(
            evaluation_preparation, profiling_session->mode(),
            &state.evaluation, &child_profile)
        : validated_preparation->PrepareValidatedAqEvaluationProfiled(
            evaluation_preparation, profiling_session->mode(),
            &state.evaluation, &child_profile);
      if (status.ok()) {
        status = profiling_session->Append(std::move(child_profile));
      }
    }
    if (!status.ok()) return status;
    if (profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.prepare_evaluator",
        gpu_profile_internal::GpuWallStageKind::kPreparation,
        preparation_begin);
      if (!status.ok()) return status;
    }
    state.backend = &gpu;
    state.quantization_pipeline_generation = prepared.generation;
    state.original_linear_rgb = original_linear_rgb;
    // The complete evaluator owns the unfiltered coding image and regenerates
    // the resident search-domain image during initial quantization. Track the
    // immutable coding view used by the downstream AQ provider so the same
    // allocation is recognized and reconfigured instead of replaced.
    state.coding_opsin = prepared.coding_opsin;
    state.evaluation_options = evaluation_options;
    state.resident_quantization = true;
    state.frame_only_resident_frontend = false;
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
  const InitialQuantizationOptions initial_options{
    .butteraugli_target = initial_quant_target,
    .rescale = options.initial_quant_rescale,
  };
  const InitialQuantFieldOutput initial_output{
    .quant_field = {
      prepared.initial_quant.data(), prepared.block_extent,
      prepared.block_extent.width},
    .strategy_mask = {
      prepared.strategy_mask.data(), prepared.block_extent,
      prepared.block_extent.width},
    .pixel_mask = {
      prepared.pixel_mask.data(), prepared.padded_extent,
      prepared.padded_extent.width},
  };
  const auto initial_begin = profiling_session == nullptr
    ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
    : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
  auto* encoding_initial = encoding_only &&
      options.adaptive_quantization.control_mode ==
        AdaptiveQuantizationControlMode::kButteraugli
    ? dynamic_cast<
        aq_evaluation_internal::PreparedAqEncodingInitialQuantization*>(
          state.evaluation.get())
    : nullptr;
  if (encoding_initial != nullptr) {
    status = encoding_initial->ComputeInitialQuantizationForEncoding(
      initial_options);
    *resident_only_initial = status.ok();
    if (status.ok() && profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.initial_quantization",
        gpu_profile_internal::GpuWallStageKind::kOperation,
        initial_begin);
    }
  } else if (profiling_session == nullptr) {
    status = state.evaluation->ComputeInitialQuantization(
      initial_options, initial_output, nullptr, 0.0f,
      &prepared.initial_color_correlation);
  } else {
    auto* profiler = dynamic_cast<
      gpu_profile_internal::PreparedAqEvaluationProfiler*>(
        state.evaluation.get());
    if (profiler == nullptr) {
      return Status::Unavailable(
        "Resident frontend cannot collect GPU diagnostics");
    }
    gpu_profile_internal::GpuExecutionProfile child_profile;
    status = profiler->ComputeInitialQuantizationProfiled(
      initial_options, initial_output, nullptr, 0.0f,
      &prepared.initial_color_correlation, profiling_session->mode(),
      &child_profile);
    if (status.ok()) {
      status = profiling_session->Append(std::move(child_profile));
    }
    if (status.ok()) {
      status = profiling_session->EndWallStage(
        "frontend.initial_quantization",
        gpu_profile_internal::GpuWallStageKind::kOperation,
        initial_begin);
    }
  }
  if (!status.ok()) {
    state.evaluation.reset();
    state.resident_coding_opsin = {};
    return status;
  }
  ResidentAcStrategyInputs views;
  status = state.evaluation->GetResidentAcStrategyInputs(&views);
  if (!status.ok()) {
    state.evaluation.reset();
    state.resident_coding_opsin = {};
    return status;
  }
  state.resident_coding_opsin = views.opsin;
  prepared.preprocessing_ready = true;
  prepared.fast_initial_color_correlation = true;
  *resident = {
      .opsin = views.opsin,
      .quant_field = views.quant_field,
      .pixel_mask = views.pixel_mask,
      .y_to_x = views.y_to_x,
      .y_to_b = views.y_to_b,
  };
  return Status::Ok();
}

}  // namespace

static Status RunGpuFrameOnlyQuantizationPipelineImpl(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuFrameOnlyPipelineOutput output,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization* prepared) {

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
  const bool encoding_only = prepared != nullptr;
  if (!encoding_only &&
      (!output.initial_quantization.quant_field.valid() ||
      output.initial_quantization.quant_field.extent != block_extent ||
      !output.initial_quantization.strategy_mask.valid() ||
      output.initial_quantization.strategy_mask.extent != block_extent ||
      !output.initial_quantization.pixel_mask.valid() ||
      output.initial_quantization.pixel_mask.extent != opsin.extent() ||
      !output.quant_field.valid() ||
      output.quant_field.extent != block_extent)) {
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
    std::vector<float> initial_quant(
      encoding_only ? size_t{0} : block_count);
    std::vector<float> strategy_mask(
      encoding_only ? size_t{0} : block_count);
    std::vector<float> pixel_mask(
      encoding_only ? size_t{0} : pixel_count);
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
    std::vector<float> final_quant(
      encoding_only ? size_t{0} : block_count);
    VarDctEncoderFrame frame;
    AdaptiveQuantizationOptions adaptive_options =
      options.adaptive_quantization;
    adaptive_options.butteraugli_target = options.butteraugli_target;
    const InitialQuantizationOptions initial_options{
      .butteraugli_target = initial_quant_target,
      .rescale = options.initial_quant_rescale,
    };
    const GpuFrameOnlyQuantizationOutput frame_output{
      .quant_field = encoding_only
        ? PlaneF32View{}
        : PlaneF32View{
            final_quant.data(), block_extent, block_extent.width},
      .frame = &frame,
    };
    status = encoding_only
      ? adaptive_quantization_gpu_internal::
          RunPreparedGpuFrameOnlyQuantizationResidentFrontendForEncoding(
            gpu, original_linear_rgb, opsin, strategies,
            {sharpness.data(), block_extent, block_extent.width},
            initial_options, adaptive_options, prepared, frame_output)
      : adaptive_quantization_gpu_internal::
          RunPreparedGpuFrameOnlyQuantizationResidentFrontend(
            gpu, original_linear_rgb, opsin, strategies,
            {sharpness.data(), block_extent, block_extent.width},
            initial_options, adaptive_options, prepared,
            {
              .quant_field = {
                initial_quant.data(), block_extent, block_extent.width},
              .strategy_mask = {
                strategy_mask.data(), block_extent, block_extent.width},
              .pixel_mask = {
                pixel_mask.data(), opsin.extent(), opsin.width()},
            }, frame_output);
    if (!status.ok()) return status;

    if (!encoding_only) {
      CopyContiguousPlane(
        initial_quant, output.initial_quantization.quant_field);
      CopyContiguousPlane(
        strategy_mask, output.initial_quantization.strategy_mask);
      CopyContiguousPlane(pixel_mask, output.initial_quantization.pixel_mask);
      CopyContiguousPlane(final_quant, output.quant_field);
    }
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

Status RunGpuFrameOnlyQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuFrameOnlyPipelineOutput output) {

  return RunGpuFrameOnlyQuantizationPipelineImpl(
    gpu, original_linear_rgb, opsin, options, output, nullptr);
}

Status quantization_pipeline_internal::
RunPreparedGpuFrameOnlyQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared_pipeline,
  CpuQuantizationPipelineOptions options,
  GpuFrameOnlyPipelineOutput output,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization* prepared) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Prepared GPU frame-only pipeline state is null");
  }
  if (prepared_pipeline.generation == 0) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline preparation has no source generation");
  }
  if (prepared->quantization_pipeline_generation !=
      prepared_pipeline.generation) {
    prepared->resident_coding_opsin = {};
    prepared->backend = nullptr;
    prepared->original_linear_rgb = {};
    prepared->coding_opsin = {};
    prepared->evaluation_options = {};
    prepared->resident_quantization = false;
    prepared->frame_only_resident_frontend = false;
    prepared->evaluation.reset();
    prepared->quantization_pipeline_generation = prepared_pipeline.generation;
  }
  return RunGpuFrameOnlyQuantizationPipelineImpl(
    gpu, original_linear_rgb, prepared_pipeline.coding_opsin, options, output,
    prepared);
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
  gpu_profile_internal::GpuProfilingSession* profiling_session) {

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
  if (prepared.generation == 0) {
    return Status::InvalidArgument(
      "GPU quantization pipeline preparation has no source generation");
  }
  if (prepared_aq != nullptr &&
      prepared_aq->quantization_pipeline_generation !=
        prepared.generation) {
    // AC-search scratch contains no image identity and is safe to retain.
    // The evaluator and its resident views do contain source pixels, so a new
    // borrowed source generation must invalidate them even when an allocator
    // reused the same host addresses.
    prepared_aq->resident_coding_opsin = {};
    prepared_aq->backend = nullptr;
    prepared_aq->original_linear_rgb = {};
    prepared_aq->coding_opsin = {};
    prepared_aq->evaluation_options = {};
    prepared_aq->resident_quantization = false;
    prepared_aq->frame_only_resident_frontend = false;
    prepared_aq->evaluation.reset();
    prepared_aq->quantization_pipeline_generation = prepared.generation;
  }
  const bool resident =
    aq_mode != GpuAdaptiveQuantizationMode::kExactCoefficients;
  if (aq_mode == GpuAdaptiveQuantizationMode::kThroughput &&
      materialization.apply_throughput_iteration_limit) {
    options.adaptive_quantization.iterations = 1;
  }
  if (!resident &&
      (!prepared.preprocessing_ready ||
       prepared.fast_initial_color_correlation)) {
    GpuPipelineGaborishProvider gaborish_inverse(
      gpu, resident, profiling_session);
    const auto preprocessing_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    Status status = PrepareQuantizationPreprocessing(
      prepared, gaborish_inverse, resident);
    if (status.ok() && profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.preprocessing",
        gpu_profile_internal::GpuWallStageKind::kOperation,
        preprocessing_begin);
    }
    if (!status.ok()) {
      return status;
    }
  }
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    local_prepared_aq;
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    aq_state = prepared_aq;
  ResidentAcStrategySearchInputs resident_inputs;
  bool resident_only_initial = false;
  if (resident) {
    if (aq_state == nullptr) aq_state = &local_prepared_aq;
    Status status = PrepareResidentAcStrategyInputs(
        gpu, original_linear_rgb, prepared, options, *aq_state,
        !materialization.initial_quantization, &resident_inputs,
        &resident_only_initial, profiling_session);
    if (!status.ok()) return status;
  }
  QuantizationPipelineMaterialization pipeline_materialization =
    materialization;
  pipeline_materialization.resident_initial_quantization =
    resident_only_initial;
  GpuAcStrategySearchProvider strategy_search(
      gpu, resident ? &resident_inputs : nullptr,
      resident ? &aq_state->ac_strategy_search : nullptr,
      profiling_session);
  GpuAdaptiveQuantizationProvider adaptive_quantization(
    gpu, aq_mode, aq_state,
    {
      .quant_field = materialization.adaptive_quant_field,
      .block_distance_map = materialization.block_distance_map,
      .reconstructed_linear_rgb =
        materialization.reconstructed_linear_rgb,
      .final_perceptual_evaluation =
        materialization.final_perceptual_evaluation,
      .resident_initial_quantization = resident_only_initial,
    }, profiling_session);
  const Status status = RunPreparedQuantizationPipelineWithProviders(
    original_linear_rgb, prepared, strategy_search, adaptive_quantization,
    options, output, resident, pipeline_materialization);
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
    prepared_aq, {}, nullptr);
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
    .adaptive_quantization = {
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
      .final_perceptual_evaluation =
        output.collect_final_butteraugli_score,
      .apply_throughput_iteration_limit = false,
    },
    nullptr);
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
  auto* submission_profiler =
    dynamic_cast<gpu_profile_internal::GpuSubmissionProfiler*>(&gpu);
  if (submission_profiler == nullptr) {
    return Status::Unavailable(
      "GPU backend cannot collect submission diagnostics");
  }
  const gpu_profile_internal::GpuProfilingCapabilities capabilities =
    submission_profiler->QueryGpuProfilingCapabilities();
  if (!capabilities.timestamp_counter || !capabilities.stage_boundary) {
    return Status::Unavailable(
      "GPU stage-boundary timestamp sampling is unavailable");
  }
  if (profiling_mode == gpu_profile_internal::GpuProfilingMode::kDispatch &&
      !capabilities.dispatch_boundary) {
    return Status::Unavailable(
      "GPU dispatch-boundary timestamp sampling is unavailable");
  }
  gpu_profile_internal::GpuProfilingSession profiling_session(
    profiling_mode, capabilities);
  const CpuQuantizationPipelineOutput pipeline_output{
    .adaptive_quantization = {
      .frame = output.frame,
      .score_history = output.score_history,
      .maximum_error_result = output.maximum_error_result,
    },
  };
  Status status = RunPreparedGpuQuantizationPipelineImpl(
    gpu, original_linear_rgb, prepared, options, aq_mode, pipeline_output,
    nullptr, prepared_aq,
    {
      .initial_quantization = false,
      .adaptive_quant_field = false,
      .block_distance_map = false,
      .reconstructed_linear_rgb = false,
      .final_perceptual_evaluation =
        output.collect_final_butteraugli_score,
      .apply_throughput_iteration_limit = false,
    },
    &profiling_session);
  if (!status.ok()) return status;
  *profile = std::move(profiling_session).Finish();
  return Status::Ok();
}

}  // namespace quantization_pipeline_internal

}  // namespace gjxl
