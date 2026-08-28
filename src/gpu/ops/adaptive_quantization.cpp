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
#include "codec/epf.h"
#include "codec/quantization.h"
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
  const AdaptiveQuantizationOutput& output) {

  if (!ValidHostPlaneLayout(output.quant_field) ||
      !ValidHostPlaneLayout(output.block_distance_map) ||
      !output.reconstructed_linear_rgb.valid() ||
      !std::ranges::all_of(
        output.reconstructed_linear_rgb.plane,
        [](PlaneF32View plane) { return ValidHostPlaneLayout(plane); }) ||
      output.frame == nullptr || output.score_history == nullptr ||
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

class PreparedGpuAdaptiveQuantizationEvaluator final
    : public aqi::AdaptiveQuantizationEvaluator {
public:
  PreparedGpuAdaptiveQuantizationEvaluator(
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    GpuAdaptiveQuantizationMode mode,
    std::unique_ptr<PreparedAqEvaluation> prepared,
    bool materialize_final,
    Extent2D source_extent)
    : opsin_(opsin),
      strategies_(strategies),
      epf_sharpness_(epf_sharpness),
      options_(options),
      mode_(mode),
      prepared_(std::move(prepared)),
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
      status = ComputeFinalColorCorrelationMap(
        opsin_, strategies_,
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
        status = ComputeQuantizedCoefficients(
            opsin_,
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
  ConstImage3FView opsin_;
  const AcStrategyGrid& strategies_;
  ConstPlaneU8View epf_sharpness_;
  AdaptiveQuantizationOptions options_;
  GpuAdaptiveQuantizationMode mode_;
  std::unique_ptr<PreparedAqEvaluation> prepared_;
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
        original_linear_rgb.extent(), strategies.extent(), *full_output);
    }
  }
  if (!status.ok()) {
    return status;
  }

  std::unique_ptr<PreparedAqEvaluation> prepared;
  status = PrepareAqEvaluation(
    gpu,
    {
      .original_linear_rgb = original_linear_rgb,
      .coding_opsin = opsin,
      .strategies = &strategies,
      .epf_sharpness = epf_sharpness,
      .options = {options.profile, options.butteraugli},
    },
    &prepared);
  if (!status.ok()) {
    return status;
  }

  try {
    PreparedGpuAdaptiveQuantizationEvaluator evaluator(
      opsin, strategies, epf_sharpness, options, mode, std::move(prepared),
      full_output != nullptr, original_linear_rgb.extent());
    aqi::AdaptiveQuantizationPolicyResult result;
    status = aqi::RunAdaptiveQuantizationPolicy(
      strategies, initial_quant_field, options, evaluator, &result, nullptr);
    if (!status.ok()) {
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
    GpuAdaptiveQuantizationMode::kExactCoefficients, &output, nullptr);
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
    epf_sharpness, options, mode, &output, nullptr);
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
    GpuAdaptiveQuantizationMode::kExactCoefficients, nullptr, &output);
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
    epf_sharpness, options, mode, nullptr, &output);
}

}  // namespace gjxl
