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
      return Status::Ok();
  }
  return Status::InvalidArgument(
    "GPU adaptive-quantization mode is invalid");
}

Status PrepareFixedThroughputColorCorrelation(
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
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
    std::vector<float> adjusted_quant(block_count);
    status = AdjustQuantField(
      strategies, butteraugli_target, initial_quant_field,
      {adjusted_quant.data(), strategies.extent(), strategies.extent().width});
    if (!status.ok()) return status;
    float quant_dc = 0.0f;
    status = ComputeInitialQuantDc(butteraugli_target, &quant_dc);
    if (!status.ok()) return status;
    std::vector<int32_t> raw_quant(block_count);
    Quantizer quantizer;
    status = CreateQuantizerFromField(
      quant_dc,
      {adjusted_quant.data(), strategies.extent(), strategies.extent().width},
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
    ColorCorrelationMap fixed_color_correlation,
    bool materialize_final,
    Extent2D source_extent)
    : strategies_(strategies),
      epf_sharpness_(epf_sharpness),
      options_(options),
      mode_(mode),
      prepared_(&prepared),
      forward_coefficients_(std::move(forward_coefficients)),
      fixed_color_correlation_(std::move(fixed_color_correlation)),
      materialize_final_(materialize_final),
      original_source_extent_(source_extent) {
    if (materialize_final_) {
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
      const ColorCorrelationMap* selected_color_correlation =
        &fixed_color_correlation_;
      if (mode_ != GpuAdaptiveQuantizationMode::kFullyResident) {
        status =
          chroma_from_luma_internal::ComputeFinalColorCorrelationMapPrepared(
            forward_coefficients_,
            {raw_quant.data(), block_extent, block_extent.width},
            quantizer, options_.fast_color_correlation, &color_correlation);
        if (!status.ok()) {
          return status;
        }
        selected_color_correlation = &color_correlation;
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
      const ConstPlaneI8View y_to_x =
        selected_color_correlation->y_to_x_map();
      const ConstPlaneI8View y_to_b =
        selected_color_correlation->y_to_b_map();
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
      if (materialize_final_ && is_final_evaluation) {
        final_output = {
          .reconstructed_linear_rgb = final_reconstructed_.view(),
          .frame = &final_frame_,
        };
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
    return !materialize_final_ || final_frame_.valid();
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
  ColorCorrelationMap fixed_color_correlation_;
  bool materialize_final_ = false;
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
  AdaptiveQuantizationOutput* full_output) {

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
  std::unique_ptr<PreparedAqEvaluation> local_prepared;
  PreparedAqEvaluation* prepared = nullptr;
  if (reusable == nullptr) {
    status = PrepareAqEvaluation(
      gpu,
      {
        .original_linear_rgb = original_linear_rgb,
        .coding_opsin = opsin,
        .strategies = &strategies,
        .epf_sharpness = epf_sharpness,
        .options = evaluation_options,
      },
      &local_prepared);
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
        evaluation_options.maximum_error;
    if (compatible) {
      status = reusable->evaluation->Reconfigure(
        strategies, epf_sharpness);
    } else {
      reusable->evaluation.reset();
      status = PrepareAqEvaluation(
        gpu,
        {
          .original_linear_rgb = original_linear_rgb,
          .coding_opsin = opsin,
          .strategies = &strategies,
          .epf_sharpness = epf_sharpness,
          .options = evaluation_options,
        },
        &reusable->evaluation);
      if (status.ok()) {
        reusable->backend = &gpu;
        reusable->original_linear_rgb = original_linear_rgb;
        reusable->coding_opsin = opsin;
        reusable->evaluation_options = evaluation_options;
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

  prepared_coefficients_internal::PreparedForwardDctCoefficients
    forward_coefficients;
  ColorCorrelationMap fixed_color_correlation;
  status = mode == GpuAdaptiveQuantizationMode::kFullyResident
    ? PrepareFixedThroughputColorCorrelation(
        opsin, strategies, initial_quant_field, options.butteraugli_target,
        &forward_coefficients, &fixed_color_correlation)
    : prepared_coefficients_internal::PrepareForwardDctCoefficients(
        opsin, strategies, &forward_coefficients);
  if (!status.ok()) {
    return status;
  }
  if (mode == GpuAdaptiveQuantizationMode::kFullyResident) {
    forward_coefficients = {};
  }

  try {
    PreparedGpuAdaptiveQuantizationEvaluator evaluator(
      strategies, epf_sharpness, options, mode, *prepared,
      std::move(forward_coefficients), std::move(fixed_color_correlation),
      full_output != nullptr,
      original_linear_rgb.extent());
    aqi::AdaptiveQuantizationPolicyResult result;
    status = aqi::RunAdaptiveQuantizationPolicy(
      strategies, initial_quant_field, options, evaluator, &result, nullptr);
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
      CopyContiguousPlane(result.quant_field, full_output->quant_field);
      CopyContiguousPlane(
        result.block_distance, full_output->block_distance_map);
      CopyImage(
        reconstructed.const_view(), full_output->reconstructed_linear_rgb);
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
    nullptr);
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
    epf_sharpness, options, mode, nullptr, &output, nullptr);
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
    &output);
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
    epf_sharpness, options, mode, nullptr, nullptr, &output);
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
  AdaptiveQuantizationOutput output) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Reusable GPU adaptive-quantization state is null");
  }
  return RunGpuAdaptiveQuantizationImpl(
    gpu, original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options, mode, prepared, nullptr, &output);
}

}  // namespace gjxl
