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

class PreparedGpuAdaptiveQuantizationEvaluator final
    : public aqi::AdaptiveQuantizationEvaluator {
public:
  PreparedGpuAdaptiveQuantizationEvaluator(
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    std::unique_ptr<PreparedAqEvaluation> prepared)
    : opsin_(opsin),
      strategies_(strategies),
      epf_sharpness_(epf_sharpness),
      options_(options),
      prepared_(std::move(prepared)) {}

  Status Evaluate(
    ConstPlaneF32View quant_field,
    float quant_dc,
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
      status = ComputeEpfInverseSigma(
        strategies_,
        {raw_quant.data(), block_extent, block_extent.width},
        quantizer, epf_sharpness_, options_.profile.epf_sigma,
        {inverse_sigma.data(), block_extent, block_extent.width});
      if (!status.ok()) {
        return status;
      }

      aqi::AdaptiveQuantizationEvaluation candidate;
      candidate.block_distance.resize(block_count);
      const ConstPlaneI8View y_to_x = color_correlation.y_to_x_map();
      const ConstPlaneI8View y_to_b = color_correlation.y_to_b_map();
      status = prepared_->Evaluate(
        {
          .raw_quant_field = {
            raw_quant.data(), block_extent, block_extent.width},
          .quantizer = quantizer.params(),
          .y_to_x = y_to_x,
          .y_to_b = y_to_b,
          .epf_inverse_sigma = {
            inverse_sigma.data(), block_extent, block_extent.width},
        },
        {
          .block_distance_map = {
            candidate.block_distance.data(), block_extent,
            block_extent.width},
          .score = &candidate.score,
        });
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

private:
  ConstImage3FView opsin_;
  const AcStrategyGrid& strategies_;
  ConstPlaneU8View epf_sharpness_;
  AdaptiveQuantizationOptions options_;
  std::unique_ptr<PreparedAqEvaluation> prepared_;
};

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

  Status status = aqi::ValidateAdaptiveQuantizationPolicyInputs(
    original_linear_rgb, opsin, strategies, initial_quant_field,
    epf_sharpness, options);
  if (status.ok()) {
    status = ValidateOutput(strategies.extent(), output);
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
      .options = {options.profile, options.butteraugli},
    },
    &prepared);
  if (!status.ok()) {
    return status;
  }

  PreparedGpuAdaptiveQuantizationEvaluator evaluator(
    opsin, strategies, epf_sharpness, options, std::move(prepared));
  aqi::AdaptiveQuantizationPolicyResult result;
  status = aqi::RunAdaptiveQuantizationPolicy(
    strategies, initial_quant_field, options, evaluator, &result, nullptr);
  if (!status.ok()) {
    return status;
  }

  CopyContiguousPlane(result.quant_field, output.quant_field);
  CopyContiguousPlane(result.block_distance, output.block_distance_map);
  *output.score_history = std::move(result.score_history);
  return Status::Ok();
}

}  // namespace gjxl
