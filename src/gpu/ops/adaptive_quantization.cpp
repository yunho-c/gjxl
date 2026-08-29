// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/adaptive_quantization.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "codec/adaptive_quantization_internal.h"
#include "codec/chroma_from_luma.h"
#include "codec/chroma_from_luma_internal.h"
#include "codec/epf.h"
#include "codec/quantization.h"
#include "codec/reconstruction_internal.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/adaptive_quantization_profile_internal.h"

namespace gjxl {
namespace {

namespace aqi = adaptive_quantization_internal;

template <typename T>
[[nodiscard]] bool ValidHostPlaneLayout(PlaneView<T> plane) noexcept {
  if (!plane.valid()) {
    return false;
  }
  if (plane.extent.height - 1 >
      (std::numeric_limits<size_t>::max() - plane.extent.width) /
        plane.stride) {
    return false;
  }
  const size_t elements =
    (plane.extent.height - 1) * plane.stride + plane.extent.width;
  using Value = std::remove_const_t<T>;
  return elements <= std::numeric_limits<size_t>::max() / sizeof(Value);
}

void CopyContiguousPlane(
  const std::vector<float>& source,
  PlaneF32View destination) {

  for (size_t y = 0; y < destination.extent.height; ++y) {
    std::copy_n(
      source.data() + y * destination.extent.width,
      destination.extent.width,
      destination.Row(y));
  }
}

[[nodiscard]] Status ValidateOutput(
  Extent2D block_extent,
  const GpuAdaptiveQuantizationPolicyOutput& output) {

  if (!ValidHostPlaneLayout(output.quant_field) ||
      !ValidHostPlaneLayout(output.block_distance_map) ||
      output.quant_field.extent != block_extent ||
      output.block_distance_map.extent != block_extent ||
      output.score_history == nullptr) {
    return Status::InvalidArgument(
      "GPU adaptive-quantization policy output is invalid");
  }
  return Status::Ok();
}

[[nodiscard]] Status ValidateFullOutput(
  Extent2D source_extent,
  Extent2D block_extent,
  AdaptiveQuantizationOptions options,
  const AdaptiveQuantizationOutput& output) {

  if (!ValidHostPlaneLayout(output.quant_field) ||
      !ValidHostPlaneLayout(output.block_distance_map) ||
      !output.reconstructed_linear_rgb.valid() ||
      !std::ranges::all_of(
        output.reconstructed_linear_rgb.plane,
        [](PlaneF32View plane) { return ValidHostPlaneLayout(plane); }) ||
      output.frame == nullptr || output.score_history == nullptr ||
      (options.control_mode ==
         AdaptiveQuantizationControlMode::kMaximumError &&
       output.maximum_error_result == nullptr) ||
      output.quant_field.extent != block_extent ||
      output.block_distance_map.extent != block_extent ||
      output.reconstructed_linear_rgb.extent() != source_extent) {
    return Status::InvalidArgument(
      "GPU adaptive-quantization output is invalid");
  }
  return Status::Ok();
}

[[nodiscard]] Status ValidateMode(
  GpuAdaptiveQuantizationMode mode) noexcept {

  switch (mode) {
    case GpuAdaptiveQuantizationMode::kExactCoefficients:
    case GpuAdaptiveQuantizationMode::kFullyResident:
    case GpuAdaptiveQuantizationMode::kThroughput:
      return Status::Ok();
    case GpuAdaptiveQuantizationMode::kMaximumThroughput:
      return Status::InvalidArgument(
        "Maximum-throughput mode requires the frame-only pipeline");
  }
  return Status::InvalidArgument(
    "GPU adaptive-quantization mode is invalid");
}

Status PrepareFixedThroughputColorCorrelation(
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View adjusted_quant_field,
  float butteraugli_target,
  prepared_coefficients_internal::PreparedForwardDctCoefficients*
    forward_coefficients,
  ColorCorrelationMap* color_correlation) {

  if (forward_coefficients == nullptr || color_correlation == nullptr) {
    return Status::InvalidArgument(
      "Fixed throughput CfL output is null");
  }
  size_t block_count = 0;
  if (!strategies.extent().try_area(&block_count)) {
    return Status::InvalidArgument(
      "Fixed throughput CfL block grid is too large");
  }
  try {
    Status status =
      prepared_coefficients_internal::PrepareForwardDctCoefficients(
        opsin, strategies, forward_coefficients);
    if (!status.ok()) return status;
    float quant_dc = 0.0f;
    status = ComputeInitialQuantDc(butteraugli_target, &quant_dc);
    if (!status.ok()) return status;
    std::vector<int32_t> raw_quant(block_count);
    Quantizer quantizer;
    status = CreateQuantizerFromField(
      quant_dc,
      adjusted_quant_field,
      {raw_quant.data(), strategies.extent(), strategies.extent().width},
      &quantizer);
    if (!status.ok()) return status;
    return chroma_from_luma_internal::ComputeFinalColorCorrelationMapPrepared(
      *forward_coefficients,
      {raw_quant.data(), strategies.extent(), strategies.extent().width},
      quantizer, true, color_correlation);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate fixed throughput CfL storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Fixed throughput CfL dimensions are too large");
  }
}

class PreparedGpuAdaptiveQuantizationEvaluator final
    : public aqi::AdaptiveQuantizationEvaluator {
public:
  PreparedGpuAdaptiveQuantizationEvaluator(
    const AcStrategyGrid& strategies,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    GpuAdaptiveQuantizationMode mode,
    PreparedAqEvaluation& prepared,
    prepared_coefficients_internal::PreparedForwardDctCoefficients
      forward_coefficients,
    bool materialize_frame,
    bool materialize_reconstruction,
    Extent2D source_extent)
    : strategies_(strategies),
      epf_sharpness_(epf_sharpness),
      options_(options),
      mode_(mode),
      prepared_(&prepared),
      forward_coefficients_(std::move(forward_coefficients)),
      materialize_frame_(materialize_frame),
      materialize_reconstruction_(materialize_reconstruction),
      original_source_extent_(source_extent) {
    if (materialize_reconstruction_) {
      final_reconstructed_.resize(source_extent);
    }
  }

  Status Evaluate(
    ConstPlaneF32View quant_field,
    float quant_dc,
    bool is_final_evaluation,
    aqi::AdaptiveQuantizationEvaluation* evaluation,
    aqi::EvaluationProfile* profile) override {

    if (evaluation == nullptr) {
      return Status::InvalidArgument(
        "GPU adaptive-quantization evaluation output is null");
    }
    if (profile != nullptr) {
      return Status::Unavailable(
        "GPU adaptive-quantization stage profiling is unavailable");
    }

    const Extent2D block_extent = strategies_.extent();
    size_t block_count = 0;
    if (!block_extent.try_area(&block_count)) {
      return Status::InvalidArgument(
        "GPU adaptive-quantization block grid is too large");
    }

    try {
      if (mode_ != GpuAdaptiveQuantizationMode::kExactCoefficients) {
        aqi::AdaptiveQuantizationEvaluation candidate;
        candidate.block_distance.resize(block_count);
        QuantizerParams quantizer_params;
        AqEvaluationOutput::Final final_output;
        AqEvaluationOutput prepared_output{
          .block_distance_map = {
            candidate.block_distance.data(), block_extent,
            block_extent.width},
          .score = &candidate.score,
          .maximum_error =
            options_.control_mode ==
                AdaptiveQuantizationControlMode::kMaximumError
              ? &candidate.maximum_error
              : nullptr,
          .quantizer = &quantizer_params,
        };
        if ((materialize_frame_ || materialize_reconstruction_) &&
            is_final_evaluation) {
          if (materialize_reconstruction_) {
            final_output.reconstructed_linear_rgb =
              final_reconstructed_.view();
          }
          if (materialize_frame_) final_output.frame = &final_frame_;
          prepared_output.final = &final_output;
        }
        const Status resident_status = prepared_->Evaluate(
          {
            .quant_field = quant_field,
            .quant_dc = quant_dc,
          },
          prepared_output);
        if (!resident_status.ok()) return resident_status;
        Status status = Quantizer::Create(
          quantizer_params, &candidate.quantizer);
        if (!status.ok()) return status;
        *evaluation = std::move(candidate);
        return Status::Ok();
      }

      std::vector<int32_t> raw_quant(block_count);
      std::vector<float> inverse_sigma(block_count);
      Quantizer quantizer;
      Status status = CreateQuantizerFromField(
        quant_dc,
        quant_field,
        {raw_quant.data(), block_extent, block_extent.width},
        &quantizer);
      if (!status.ok()) {
        return status;
      }
      ColorCorrelationMap color_correlation;
      status =
        chroma_from_luma_internal::ComputeFinalColorCorrelationMapPrepared(
          forward_coefficients_,
          {raw_quant.data(), block_extent, block_extent.width},
          quantizer, options_.fast_color_correlation, &color_correlation);
      if (!status.ok()) {
        return status;
      }

      VarDctEncoderFrame exact_coefficients;
      if (mode_ == GpuAdaptiveQuantizationMode::kExactCoefficients) {
        FrameGeometry geometry;
        status = FrameGeometry::Create(original_source_extent_, &geometry);
        if (!status.ok()) {
          return status;
        }
        status = prepared_coefficients_internal::
          ComputeQuantizedCoefficientsPrepared(
            forward_coefficients_,
            {
                .geometry = geometry,
                .strategies = &strategies_,
                .raw_quant_field = {
                    raw_quant.data(), block_extent, block_extent.width},
                .quantizer = &quantizer,
                .color_correlation = &color_correlation,
                .epf_sharpness = epf_sharpness_,
            },
            options_.profile, &exact_coefficients);
        if (!status.ok()) {
          return status;
        }
      }

      const ConstPlaneI32View evaluation_raw_quant =
        mode_ == GpuAdaptiveQuantizationMode::kExactCoefficients
          ? exact_coefficients.raw_quant_field()
          : ConstPlaneI32View{
              raw_quant.data(), block_extent, block_extent.width};
      status = ComputeEpfInverseSigma(
        strategies_,
        evaluation_raw_quant,
        quantizer, epf_sharpness_, options_.profile.epf_sigma,
        {inverse_sigma.data(), block_extent, block_extent.width});
      if (!status.ok()) {
        return status;
      }

      aqi::AdaptiveQuantizationEvaluation candidate;
      candidate.block_distance.resize(block_count);
      const ConstPlaneI8View y_to_x = color_correlation.y_to_x_map();
      const ConstPlaneI8View y_to_b = color_correlation.y_to_b_map();
      AqEvaluationOutput::Final final_output;
      AqEvaluationOutput prepared_output{
        .block_distance_map = {
          candidate.block_distance.data(), block_extent, block_extent.width},
        .score = &candidate.score,
        .maximum_error =
          options_.control_mode ==
              AdaptiveQuantizationControlMode::kMaximumError
            ? &candidate.maximum_error
            : nullptr,
      };
      if ((materialize_frame_ || materialize_reconstruction_) &&
          is_final_evaluation) {
        if (materialize_reconstruction_) {
          final_output.reconstructed_linear_rgb =
            final_reconstructed_.view();
        }
        if (materialize_frame_) final_output.frame = &final_frame_;
        prepared_output.final = &final_output;
      }
      AqEvaluationInput prepared_input{
          .raw_quant_field = evaluation_raw_quant,
          .quantizer = quantizer.params(),
          .y_to_x = y_to_x,
          .y_to_b = y_to_b,
          .epf_inverse_sigma = {
            inverse_sigma.data(), block_extent, block_extent.width},
      };
      if (mode_ == GpuAdaptiveQuantizationMode::kExactCoefficients) {
        prepared_input.exact_coefficients = &exact_coefficients;
      }
      status = prepared_->Evaluate(prepared_input, prepared_output);
      if (!status.ok()) {
        return status;
      }
      candidate.quantizer = quantizer;
      *evaluation = std::move(candidate);
      return Status::Ok();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate GPU adaptive-quantization host staging");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "GPU adaptive-quantization dimensions are too large");
    }
  }

  [[nodiscard]] bool HasFinalOutput() const noexcept {
    return !materialize_frame_ || final_frame_.valid();
  }

  [[nodiscard]] Image3FBuffer&& TakeFinalReconstruction() noexcept {
    return std::move(final_reconstructed_);
  }

  [[nodiscard]] VarDctEncoderFrame&& TakeFinalFrame() noexcept {
    return std::move(final_frame_);
  }

private:
  const AcStrategyGrid& strategies_;
  ConstPlaneU8View epf_sharpness_;
  AdaptiveQuantizationOptions options_;
  GpuAdaptiveQuantizationMode mode_;
  PreparedAqEvaluation* prepared_ = nullptr;
  prepared_coefficients_internal::PreparedForwardDctCoefficients
    forward_coefficients_;
  bool materialize_frame_ = false;
  bool materialize_reconstruction_ = false;
  Image3FBuffer final_reconstructed_;
  VarDctEncoderFrame final_frame_;
  Extent2D original_source_extent_;
};

Status RunGpuAdaptiveQuantizationImpl(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    reusable,
  GpuAdaptiveQuantizationPolicyOutput* bounded_output,
  AdaptiveQuantizationOutput* full_output,
  adaptive_quantization_gpu_internal::AdaptiveQuantizationMaterialization
    materialization,
  gpu_profile_internal::GpuProfilingSession* profiling_session) {

  const bool profiling = profiling_session != nullptr;

  Status status = ValidateMode(mode);
  if (status.ok()) {
    status = aqi::ValidateAdaptiveQuantizationPolicyInputs(
      original_linear_rgb, opsin, strategies, initial_quant_field,
      epf_sharpness, options);
  }
  if (status.ok()) {
    if (full_output == nullptr) {
      if (bounded_output == nullptr) {
        return Status::InvalidArgument(
          "GPU adaptive-quantization bounded output is null");
      }
      status = ValidateOutput(strategies.extent(), *bounded_output);
    } else {
      status = ValidateFullOutput(
        original_linear_rgb.extent(), strategies.extent(), options,
        *full_output);
    }
  }
  if (!status.ok()) {
    return status;
  }

  const AqEvaluationOptions evaluation_options{
    .profile = options.profile,
    .butteraugli = options.butteraugli,
    .metric = options.control_mode ==
          AdaptiveQuantizationControlMode::kMaximumError
        ? AqEvaluationMetric::kMaximumError
        : AqEvaluationMetric::kButteraugli,
    .maximum_error = options.maximum_error,
  };
  const bool resident_quantization =
    mode != GpuAdaptiveQuantizationMode::kExactCoefficients;
  auto* aq_preparation_profiler = profiling
    ? dynamic_cast<gpu_profile_internal::GpuAqEvaluationProfiler*>(&gpu)
    : nullptr;
  if (profiling && aq_preparation_profiler == nullptr) {
    return Status::Unavailable(
      "GPU adaptive-quantization preparation cannot collect diagnostics");
  }
  const AqEvaluationPreparation evaluation_preparation{
    .original_linear_rgb = original_linear_rgb,
    .coding_opsin = opsin,
    .strategies = &strategies,
    .epf_sharpness = epf_sharpness,
    .options = evaluation_options,
    .resident_quantization = resident_quantization,
    .coefficient_decision_mode =
      AcCoefficientDecisionMode::kAdjustedSharedQuant,
  };
  const auto prepare_evaluation =
    [&](std::unique_ptr<PreparedAqEvaluation>* destination) {
      if (!profiling) {
        return PrepareAqEvaluation(
          gpu, evaluation_preparation, destination);
      }
      const auto begin =
        gpu_profile_internal::GpuProfilingSession::BeginWallStage();
      gpu_profile_internal::GpuExecutionProfile preparation_profile;
      Status prepare_status =
        aq_preparation_profiler->PrepareAqEvaluationProfiled(
          evaluation_preparation, profiling_session->mode(), destination,
          &preparation_profile);
      if (prepare_status.ok()) {
        prepare_status = profiling_session->Append(
          std::move(preparation_profile));
      }
      if (prepare_status.ok()) {
        prepare_status = profiling_session->EndWallStage(
          "frontend.prepare_aq",
          gpu_profile_internal::GpuWallStageKind::kPreparation, begin);
      }
      return prepare_status;
    };
  std::unique_ptr<PreparedAqEvaluation> local_prepared;
  PreparedAqEvaluation* prepared = nullptr;
  if (reusable == nullptr) {
    status = prepare_evaluation(&local_prepared);
    prepared = local_prepared.get();
  } else {
    const auto same_plane = [](ConstPlaneF32View left,
                               ConstPlaneF32View right) {
      return left.data == right.data && left.extent == right.extent &&
        left.stride == right.stride;
    };
    const auto same_image = [&](ConstImage3FView left,
                                ConstImage3FView right) {
      return same_plane(left.plane[0], right.plane[0]) &&
        same_plane(left.plane[1], right.plane[1]) &&
        same_plane(left.plane[2], right.plane[2]);
    };
    const ButteraugliOptions& previous_butteraugli =
      reusable->evaluation_options.butteraugli;
    const ButteraugliOptions& current_butteraugli =
      evaluation_options.butteraugli;
    const bool compatible = reusable->evaluation != nullptr &&
      reusable->backend == &gpu &&
      same_image(reusable->original_linear_rgb, original_linear_rgb) &&
      same_image(reusable->coding_opsin, opsin) &&
      reusable->evaluation_options.profile == evaluation_options.profile &&
      previous_butteraugli.hf_asymmetry ==
        current_butteraugli.hf_asymmetry &&
      previous_butteraugli.x_multiplier ==
        current_butteraugli.x_multiplier &&
      previous_butteraugli.intensity_target ==
        current_butteraugli.intensity_target &&
      reusable->evaluation_options.metric == evaluation_options.metric &&
      reusable->evaluation_options.maximum_error ==
        evaluation_options.maximum_error &&
      reusable->resident_quantization == resident_quantization;
    if (compatible) {
      const auto reconfigure_begin = profiling
        ? gpu_profile_internal::GpuProfilingSession::BeginWallStage()
        : gpu_profile_internal::GpuProfilingSession::TimePoint{};
      status = reusable->evaluation->Reconfigure(
        strategies, epf_sharpness);
      if (status.ok() && profiling) {
        status = profiling_session->EndWallStage(
          "frontend.reconfigure_aq",
          gpu_profile_internal::GpuWallStageKind::kPreparation,
          reconfigure_begin);
      }
    } else {
      reusable->evaluation.reset();
      status = prepare_evaluation(&reusable->evaluation);
      if (status.ok()) {
        reusable->backend = &gpu;
        reusable->original_linear_rgb = original_linear_rgb;
        reusable->coding_opsin = opsin;
        reusable->evaluation_options = evaluation_options;
        reusable->resident_quantization = resident_quantization;
      }
    }
    prepared = reusable->evaluation.get();
  }
  if (!status.ok()) {
    if (reusable != nullptr) {
      reusable->evaluation.reset();
    }
    return status;
  }
  if (prepared == nullptr) {
    return Status::Internal(
      "GPU adaptive quantization preparation produced no state");
  }
  auto* prepared_profiler = profiling
    ? dynamic_cast<gpu_profile_internal::PreparedAqEvaluationProfiler*>(
        prepared)
    : nullptr;
  if (profiling && prepared_profiler == nullptr) {
    return Status::Unavailable(
      "Prepared GPU adaptive quantization cannot collect diagnostics");
  }

  try {
    std::vector<float> adjusted_initial;
    ConstPlaneF32View policy_initial = initial_quant_field;
    const float adjustment_target =
      options.control_mode == AdaptiveQuantizationControlMode::kMaximumError
        ? 1.0f
        : options.butteraugli_target;
    if (resident_quantization) {
      size_t block_count = 0;
      if (!strategies.extent().try_area(&block_count)) {
        return Status::InvalidArgument(
          "Resident AQ block grid is too large");
      }
      adjusted_initial.resize(block_count);
      const auto adjustment_begin = profiling
        ? gpu_profile_internal::GpuProfilingSession::BeginWallStage()
        : gpu_profile_internal::GpuProfilingSession::TimePoint{};
      if (profiling) {
        gpu_profile_internal::GpuExecutionProfile adjustment_profile;
        status = prepared_profiler->AdjustQuantFieldResidentProfiled(
          adjustment_target, initial_quant_field,
          {adjusted_initial.data(), strategies.extent(),
           strategies.extent().width},
          profiling_session->mode(), &adjustment_profile);
        if (status.ok()) {
          status = profiling_session->Append(std::move(adjustment_profile));
        }
      } else {
        status = prepared->AdjustQuantFieldResident(
          adjustment_target, initial_quant_field,
          {adjusted_initial.data(), strategies.extent(),
           strategies.extent().width});
      }
      if (status.ok() && profiling) {
        status = profiling_session->EndWallStage(
          "frontend.quant_adjustment",
          gpu_profile_internal::GpuWallStageKind::kOperation,
          adjustment_begin);
      }
      if (!status.ok()) return status;
      policy_initial = {
        adjusted_initial.data(), strategies.extent(),
        strategies.extent().width};
    }

    prepared_coefficients_internal::PreparedForwardDctCoefficients
      forward_coefficients;
    ColorCorrelationMap fixed_color_correlation;
    const auto cfl_begin = profiling
      ? gpu_profile_internal::GpuProfilingSession::BeginWallStage()
      : gpu_profile_internal::GpuProfilingSession::TimePoint{};
    status = resident_quantization
        ? PrepareFixedThroughputColorCorrelation(
            opsin, strategies, policy_initial, adjustment_target,
            &forward_coefficients, &fixed_color_correlation)
        : prepared_coefficients_internal::PrepareForwardDctCoefficients(
            opsin, strategies, &forward_coefficients);
    if (status.ok() && profiling) {
      status = profiling_session->EndWallStage(
        "frontend.fixed_cfl",
        gpu_profile_internal::GpuWallStageKind::kHost, cfl_begin);
    }
    if (!status.ok()) return status;
    if (resident_quantization) {
      const auto upload_begin = profiling
        ? gpu_profile_internal::GpuProfilingSession::BeginWallStage()
        : gpu_profile_internal::GpuProfilingSession::TimePoint{};
      status = prepared->SetInvariantColorCorrelation(
          fixed_color_correlation.y_to_x_map(),
          fixed_color_correlation.y_to_b_map());
      if (status.ok() && profiling) {
        status = profiling_session->EndWallStage(
          "frontend.cfl_upload",
          gpu_profile_internal::GpuWallStageKind::kUpload, upload_begin);
      }
      if (!status.ok()) return status;
      forward_coefficients = {};
    }

    if (resident_quantization &&
        options.control_mode ==
          AdaptiveQuantizationControlMode::kButteraugli) {
      aqi::ButteraugliPolicySetup setup;
      status = aqi::PrepareButteraugliPolicy(
        policy_initial, options.butteraugli_target, &setup);
      if (!status.ok()) return status;

      size_t block_count = 0;
      if (!strategies.extent().try_area(&block_count)) {
        return Status::InvalidArgument(
          "Resident AQ block grid is too large");
      }
      aqi::AdaptiveQuantizationPolicyResult fused_result;
      if (materialization.quant_field) {
        fused_result.quant_field.resize(block_count);
      }
      if (materialization.block_distance_map) {
        fused_result.block_distance.resize(block_count);
      }
      Image3FBuffer fused_reconstruction;
      VarDctEncoderFrame fused_frame;
      AqResidentButteraugliPolicyOutput fused_output{
        .score_history = &fused_result.score_history,
      };
      if (materialization.quant_field) {
        fused_output.quant_field = {
          fused_result.quant_field.data(), strategies.extent(),
          strategies.extent().width};
      }
      if (materialization.block_distance_map) {
        fused_output.block_distance_map = {
          fused_result.block_distance.data(), strategies.extent(),
          strategies.extent().width};
      }
      if (full_output != nullptr) {
        if (materialization.reconstructed_linear_rgb) {
          fused_reconstruction.resize(original_linear_rgb.extent());
          fused_output.reconstructed_linear_rgb =
            fused_reconstruction.view();
        }
        fused_output.frame = &fused_frame;
      }
      const AqResidentButteraugliPolicyInput resident_input{
          .adjusted_initial_quant_field = policy_initial,
          .quant_dc = setup.quant_dc,
          .butteraugli_target = options.butteraugli_target,
          .lower_bound = setup.lower_bound,
          .upper_bound = setup.upper_bound,
          .iterations = options.iterations,
      };
      if (profiling) {
        const auto policy_begin =
          gpu_profile_internal::GpuProfilingSession::BeginWallStage();
        gpu_profile_internal::GpuExecutionProfile policy_profile;
        status = prepared_profiler->EvaluateResidentButteraugliPolicyProfiled(
          resident_input, fused_output, profiling_session->mode(),
          &policy_profile);
        if (status.ok()) {
          status = profiling_session->Append(std::move(policy_profile));
        }
        if (status.ok()) {
          status = profiling_session->EndWallStage(
            "resident.aq",
            gpu_profile_internal::GpuWallStageKind::kOperation,
            policy_begin);
        }
      } else {
        status = prepared->EvaluateResidentButteraugliPolicy(
          resident_input, fused_output);
      }
      if (status.ok()) {
        if (full_output == nullptr) {
          CopyContiguousPlane(
            fused_result.quant_field, bounded_output->quant_field);
          CopyContiguousPlane(
            fused_result.block_distance,
            bounded_output->block_distance_map);
          *bounded_output->score_history =
            std::move(fused_result.score_history);
        } else {
          if (materialization.quant_field) {
            CopyContiguousPlane(
              fused_result.quant_field, full_output->quant_field);
          }
          if (materialization.block_distance_map) {
            CopyContiguousPlane(
              fused_result.block_distance,
              full_output->block_distance_map);
          }
          if (materialization.reconstructed_linear_rgb) {
            CopyImage(
              fused_reconstruction.const_view(),
              full_output->reconstructed_linear_rgb);
          }
          *full_output->frame = std::move(fused_frame);
          *full_output->score_history =
            std::move(fused_result.score_history);
        }
        return Status::Ok();
      }
      if (status.code() != StatusCode::kUnavailable) {
        if (reusable != nullptr) reusable->evaluation.reset();
        return status;
      }
    }

    PreparedGpuAdaptiveQuantizationEvaluator evaluator(
      strategies, epf_sharpness, options, mode, *prepared,
      std::move(forward_coefficients), full_output != nullptr,
      full_output != nullptr && materialization.reconstructed_linear_rgb,
      original_linear_rgb.extent());
    aqi::AdaptiveQuantizationPolicyResult result;
    status = resident_quantization
      ? aqi::RunAdaptiveQuantizationPolicyAdjusted(
          strategies, policy_initial, options, evaluator, &result, nullptr)
      : aqi::RunAdaptiveQuantizationPolicy(
          strategies, policy_initial, options, evaluator, &result, nullptr);
    if (!status.ok()) {
      if (reusable != nullptr) {
        reusable->evaluation.reset();
      }
      return status;
    }
    if (!evaluator.HasFinalOutput()) {
      return Status::Internal(
        "GPU adaptive quantization did not materialize its final output");
    }

    if (full_output == nullptr) {
      CopyContiguousPlane(result.quant_field, bounded_output->quant_field);
      CopyContiguousPlane(
        result.block_distance, bounded_output->block_distance_map);
      *bounded_output->score_history = std::move(result.score_history);
    } else {
      Image3FBuffer reconstructed = evaluator.TakeFinalReconstruction();
      VarDctEncoderFrame frame = evaluator.TakeFinalFrame();
      if (materialization.quant_field) {
        CopyContiguousPlane(result.quant_field, full_output->quant_field);
      }
      if (materialization.block_distance_map) {
        CopyContiguousPlane(
          result.block_distance, full_output->block_distance_map);
      }
      if (materialization.reconstructed_linear_rgb) {
        CopyImage(
          reconstructed.const_view(), full_output->reconstructed_linear_rgb);
      }
      *full_output->frame = std::move(frame);
      *full_output->score_history = std::move(result.score_history);
      if (full_output->maximum_error_result != nullptr) {
        *full_output->maximum_error_result = result.maximum_error;
      }
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU adaptive-quantization final storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU adaptive-quantization final dimensions are too large");
  }
}

}  // namespace

namespace {

Status FinishGpuFrameOnlyQuantization(
  PreparedAqEvaluation& prepared,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  const ColorCorrelationMap* color_correlation,
  AdaptiveQuantizationOptions options,
  GpuFrameOnlyQuantizationOutput output) {

  size_t block_count = 0;
  if (!strategies.extent().try_area(&block_count)) {
    return Status::InvalidArgument(
      "GPU frame-only block grid is too large");
  }
  try {
    std::vector<float> adjusted_quant(block_count);
    Status status = AdjustQuantField(
      strategies, options.butteraugli_target, initial_quant_field,
      {adjusted_quant.data(), strategies.extent(), strategies.extent().width});
    if (!status.ok()) return status;
    float quant_dc = 0.0f;
    status = ComputeInitialQuantDc(options.butteraugli_target, &quant_dc);
    if (!status.ok()) return status;
    std::vector<int32_t> raw_quant(block_count);
    Quantizer quantizer;
    status = CreateQuantizerFromField(
      quant_dc,
      {adjusted_quant.data(), strategies.extent(), strategies.extent().width},
      {raw_quant.data(), strategies.extent(), strategies.extent().width},
      &quantizer);
    if (!status.ok()) return status;
    std::vector<float> inverse_sigma(block_count);
    status = ComputeEpfInverseSigma(
      strategies,
      {raw_quant.data(), strategies.extent(), strategies.extent().width},
      quantizer, epf_sharpness, options.profile.epf_sigma,
      {inverse_sigma.data(), strategies.extent(), strategies.extent().width});
    if (!status.ok()) return status;

    VarDctEncoderFrame candidate;
    status = prepared.EncodeFrame(
      {
        .raw_quant_field = {
          raw_quant.data(), strategies.extent(), strategies.extent().width},
        .quantizer = quantizer.params(),
        .y_to_x = color_correlation == nullptr
          ? ConstPlaneI8View{}
          : color_correlation->y_to_x_map(),
        .y_to_b = color_correlation == nullptr
          ? ConstPlaneI8View{}
          : color_correlation->y_to_b_map(),
        .epf_inverse_sigma = {
          inverse_sigma.data(), strategies.extent(), strategies.extent().width},
      },
      &candidate);
    if (!status.ok()) return status;
    CopyContiguousPlane(adjusted_quant, output.quant_field);
    *output.frame = std::move(candidate);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU frame-only quantization storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU frame-only quantization dimensions are too large");
  }
}

Status RunGpuFrameOnlyQuantizationImpl(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  const ColorCorrelationMap* color_correlation,
  bool resident_initial_cfl,
  AdaptiveQuantizationOptions options,
  GpuFrameOnlyQuantizationOutput output) {

  Status status = aqi::ValidateAdaptiveQuantizationPolicyInputs(
    original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options);
  if (!status.ok()) return status;
  if (!output.quant_field.valid() ||
      output.quant_field.extent != strategies.extent() ||
      output.frame == nullptr) {
    return Status::InvalidArgument(
      "GPU frame-only quantization output is invalid");
  }

  try {
    std::unique_ptr<PreparedAqEvaluation> prepared;
    status = PrepareAqEvaluation(
      gpu,
      {
        .original_linear_rgb = original_linear_rgb,
        .coding_opsin = opsin,
        .strategies = &strategies,
        .epf_sharpness = epf_sharpness,
        .options = {options.profile, options.butteraugli},
        .frame_only = true,
        .frame_only_inverse_gaborish =
          options.profile.loop_filter.gaborish,
        .frame_only_resident_initial_cfl = resident_initial_cfl,
        .coefficient_decision_mode =
          AcCoefficientDecisionMode::kAdjustedSharedQuant,
      },
      &prepared);
    if (!status.ok()) return status;
    return FinishGpuFrameOnlyQuantization(
      *prepared, strategies, initial_quant_field, epf_sharpness,
      color_correlation, options, output);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU frame-only quantization storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU frame-only quantization dimensions are too large");
  }
}

}  // namespace

Status RunGpuFrameOnlyQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  const ColorCorrelationMap& color_correlation,
  AdaptiveQuantizationOptions options,
  GpuFrameOnlyQuantizationOutput output) {

  return RunGpuFrameOnlyQuantizationImpl(
      gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
      epf_sharpness, &color_correlation, false, options, output);
}

Status RunGpuFrameOnlyQuantizationResidentInitialCfl(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuFrameOnlyQuantizationOutput output) {

  return RunGpuFrameOnlyQuantizationImpl(
      gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
      epf_sharpness, nullptr, true, options, output);
}

Status RunGpuFrameOnlyQuantizationResidentFrontend(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneU8View epf_sharpness,
  InitialQuantizationOptions initial_options,
  AdaptiveQuantizationOptions options,
  InitialQuantFieldOutput initial_output,
  GpuFrameOnlyQuantizationOutput output) {

  if (!std::isfinite(initial_options.butteraugli_target) ||
      initial_options.butteraugli_target <= 0.0f ||
      !std::isfinite(initial_options.rescale) ||
      initial_options.rescale <= 0.0f ||
      !output.quant_field.valid() ||
      output.quant_field.extent != strategies.extent() ||
      output.frame == nullptr) {
    return Status::InvalidArgument(
      "Resident frame-only frontend inputs or outputs are invalid");
  }
  size_t block_count = 0;
  size_t dct8_count = 0;
  Status strategy_status = strategies.ForEachAnchor(
      [&](size_t, size_t, AcStrategyType strategy) {
        if (strategy != AcStrategyType::kDct8) {
          return Status::InvalidArgument(
            "Resident frame-only frontend requires DCT8 strategies");
        }
        ++dct8_count;
        return Status::Ok();
      });
  if (!strategy_status.ok() ||
      !strategies.extent().try_area(&block_count) ||
      dct8_count != block_count) {
    return strategy_status.ok()
      ? Status::InvalidArgument(
          "Resident frame-only frontend requires a complete DCT8 grid")
      : strategy_status;
  }
  try {
    std::unique_ptr<PreparedAqEvaluation> prepared;
    Status status = PrepareAqEvaluation(
      gpu,
      {
        .original_linear_rgb = original_linear_rgb,
        .coding_opsin = opsin,
        .strategies = &strategies,
        .epf_sharpness = epf_sharpness,
        .options = {options.profile, options.butteraugli},
        .frame_only = true,
        .frame_only_inverse_gaborish = options.profile.loop_filter.gaborish,
        .frame_only_resident_initial_cfl = true,
        .frame_only_resident_initial_quant = true,
        .frame_only_resident_quantizer = true,
        .coefficient_decision_mode =
          AcCoefficientDecisionMode::kAdjustedSharedQuant,
      },
      &prepared);
    if (!status.ok()) return status;
    float quant_dc = 0.0f;
    status = ComputeInitialQuantDc(options.butteraugli_target, &quant_dc);
    if (!status.ok()) return status;
    QuantizerParams quantizer;
    status = prepared->ComputeInitialQuantization(
      initial_options, initial_output, &quantizer, quant_dc);
    if (!status.ok()) return status;
    const ConstPlaneF32View initial_quant{
      initial_output.quant_field.data,
      initial_output.quant_field.extent,
      initial_output.quant_field.stride,
    };
    status = aqi::ValidateAdaptiveQuantizationPolicyInputs(
      original_linear_rgb, opsin, strategies, initial_quant,
      epf_sharpness, options);
    if (!status.ok()) return status;
    VarDctEncoderFrame candidate;
    status = prepared->EncodeFrame(
      {
        .quantizer = quantizer,
      },
      &candidate);
    if (!status.ok()) return status;
    for (size_t y = 0; y < strategies.extent().height; ++y) {
      std::copy_n(initial_quant.Row(y), strategies.extent().width,
                  output.quant_field.Row(y));
    }
    *output.frame = std::move(candidate);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate resident frame-only frontend storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Resident frame-only frontend dimensions are too large");
  }
}

Status RunGpuAdaptiveQuantizationPolicy(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationPolicyOutput output) {

  return RunGpuAdaptiveQuantizationImpl(
    gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options,
    GpuAdaptiveQuantizationMode::kExactCoefficients, nullptr, &output,
    nullptr, {}, nullptr);
}

Status RunGpuAdaptiveQuantizationPolicy(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  GpuAdaptiveQuantizationPolicyOutput output) {

  return RunGpuAdaptiveQuantizationImpl(
    gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options, mode, nullptr, &output, nullptr, {},
    nullptr);
}

Status RunGpuAdaptiveQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationOutput output) {

  return RunGpuAdaptiveQuantizationImpl(
    gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options,
    GpuAdaptiveQuantizationMode::kExactCoefficients, nullptr, nullptr,
    &output, {}, nullptr);
}

Status RunGpuAdaptiveQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  AdaptiveQuantizationOutput output) {

  return RunGpuAdaptiveQuantizationImpl(
    gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options, mode, nullptr, nullptr, &output, {},
    nullptr);
}

Status adaptive_quantization_gpu_internal::
RunPreparedGpuAdaptiveQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  PreparedAdaptiveQuantization* prepared,
  AdaptiveQuantizationOutput output,
  AdaptiveQuantizationMaterialization materialization) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Reusable GPU adaptive-quantization state is null");
  }
  return RunGpuAdaptiveQuantizationImpl(
    gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options, mode, prepared, nullptr, &output,
    materialization, nullptr);
}

Status adaptive_quantization_gpu_internal::
RunPreparedGpuAdaptiveQuantizationProfiled(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  PreparedAdaptiveQuantization* prepared,
  AdaptiveQuantizationOutput output,
  AdaptiveQuantizationMaterialization materialization,
  gpu_profile_internal::GpuProfilingSession* profiling_session) {

  if (prepared == nullptr || profiling_session == nullptr) {
    return Status::InvalidArgument(
      "Profiled reusable GPU adaptive-quantization state is invalid");
  }
  if (mode != GpuAdaptiveQuantizationMode::kFullyResident &&
      mode != GpuAdaptiveQuantizationMode::kThroughput) {
    return Status::InvalidArgument(
      "GPU execution profiling requires resident adaptive quantization");
  }
  return RunGpuAdaptiveQuantizationImpl(
    gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options, mode, prepared, nullptr, &output,
    materialization, profiling_session);
}

}  // namespace gjxl
