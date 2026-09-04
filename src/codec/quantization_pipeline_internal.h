// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/quantization_pipeline.h"
#include "core/image_buffer.h"

namespace gjxl::quantization_pipeline_internal {

class GaborishInverseProvider {
public:
  virtual ~GaborishInverseProvider() = default;

  [[nodiscard]] virtual Status Apply(
    ConstImage3FView input,
    std::array<float, 3> multipliers,
    Image3FView output) = 0;

protected:
  GaborishInverseProvider() = default;
};

class AdaptiveQuantizationProvider {
public:
  virtual ~AdaptiveQuantizationProvider() = default;

  [[nodiscard]] virtual Status Find(
    ConstImage3FView original_linear_rgb,
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    PreparedButteraugliReference* prepared_reference,
    AdaptiveQuantizationOutput output) = 0;

protected:
  AdaptiveQuantizationProvider() = default;
};

/// Target-invariant preparation for repeated quantization attempts over one
/// source/profile pair. `coding_opsin` borrows immutable caller storage, which
/// must remain alive until the last prepared run completes. Final encoder and
/// diagnostic results are committed directly by the selected adaptive-
/// quantization provider.
struct PreparedQuantizationPipeline {
  uint64_t generation = 0;
  Extent2D source_extent;
  Extent2D padded_extent;
  Extent2D block_extent;
  float initial_quant_rescale = 1.0f;
  SimpleVarDctCodestreamProfile profile;
  ConstImage3FView coding_opsin;
  /// Exact immutable host views proven finite by the workflow that created
  /// this preparation. Empty views mean that downstream public validation is
  /// still required. Identity checks prevent provenance from being reused
  /// with a different source passed to a prepared run.
  ConstImage3FView validated_original_linear_rgb;
  ConstImage3FView validated_coding_opsin;
  Image3FBuffer preprocessed_opsin;
  ColorCorrelationMap initial_color_correlation;
  bool preprocessing_ready = false;
  bool fast_initial_color_correlation = false;
  std::vector<uint8_t> epf_sharpness;
  std::vector<float> initial_quant;
  std::vector<float> strategy_mask;
  std::vector<float> pixel_mask;
  AcStrategyGrid strategies;
  ButteraugliOptions butteraugli_options;
  std::unique_ptr<PreparedButteraugliReference> butteraugli_reference;
};

enum class QuantizationPipelineInputProvenance {
  kUnvalidated,
  kFiniteLinearRgbAndOpsin,
};

struct QuantizationPipelineMaterialization {
  bool initial_quantization = true;
  bool adaptive_quant_field = true;
  bool block_distance_map = true;
  bool reconstructed_linear_rgb = true;
  bool final_perceptual_evaluation = true;
  /// Public throughput diagnostics retain their one-update policy. Encoding
  /// paths set this false so requesting a final score cannot change the field.
  bool apply_throughput_iteration_limit = true;
  bool resident_initial_quantization = false;
};

[[nodiscard]] Status PrepareQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  PreparedQuantizationPipeline* prepared,
  bool prepare_cpu_butteraugli = true,
  bool prepare_cpu_preprocessing = true,
  QuantizationPipelineInputProvenance input_provenance =
    QuantizationPipelineInputProvenance::kUnvalidated);

[[nodiscard]] Status PrepareQuantizationPreprocessing(
  PreparedQuantizationPipeline& prepared,
  GaborishInverseProvider& gaborish_inverse,
  bool fast_initial_color_correlation);

[[nodiscard]] Status RunPreparedQuantizationPipelineWithProviders(
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  AcStrategySearchProvider& strategy_search,
  AdaptiveQuantizationProvider& adaptive_quantization,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output,
  bool initial_quantization_ready = false,
  QuantizationPipelineMaterialization materialization = {});

[[nodiscard]] Status RunPreparedCpuQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output);

[[nodiscard]] Status RunQuantizationPipelineWithProviders(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  AcStrategySearchProvider& strategy_search,
  AdaptiveQuantizationProvider& adaptive_quantization,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output);

}  // namespace gjxl::quantization_pipeline_internal
