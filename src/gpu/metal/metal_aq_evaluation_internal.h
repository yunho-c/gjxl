// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <Metal/Metal.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "core/ac_strategy.h"
#include "codec/vardct_frame_internal.h"
#include "gpu/metal/metal_aq_butteraugli_test.h"
#include "gpu/metal/metal_aq_evaluation_profile.h"
#include "gpu/metal/metal_aq_postprocess_test.h"
#include "gpu/metal/metal_aq_reconstruction_test.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/scratch.h"

namespace gjxl::metal_internal {

struct AqReconstructionParams {
  uint32_t coding_width;
  uint32_t coding_height;
  uint32_t coding_stride;
  uint32_t block_width;
  uint32_t block_height;
  uint32_t raw_quant_stride;
  uint32_t color_width;
  uint32_t color_stride;
  uint32_t anchor_offset;
  uint32_t anchor_count;
  uint32_t coefficient_offset;
  uint32_t coefficient_count;
  uint32_t pixel_width;
  uint32_t pixel_height;
  uint32_t covered_width;
  uint32_t covered_height;
  uint32_t strategy;
  uint32_t global_scale;
  uint32_t quant_dc;
  float x_matrix_multiplier;
  float b_matrix_multiplier;
  uint32_t adjust_ac_quant;
  uint32_t inverse_sigma_stride;
  uint32_t epf_sharpness_stride;
  float epf_quant_multiplier;
  std::array<float, 8> epf_sharpness_lut;
};

struct AqResetParams {
  uint32_t coefficient_value_count;
  uint32_t dc_value_count;
  uint32_t pixel_value_count;
  uint32_t block_value_count;
  uint32_t test_error_mask;
};

struct AqInitialCflParams {
  uint32_t width;
  uint32_t height;
  uint32_t coding_stride;
  uint32_t tile_width;
  uint32_t tile_height;
  uint32_t color_stride;
};

struct AqInitialQuantGradientParams {
  uint32_t width;
  uint32_t height;
  uint32_t coding_stride;
  uint32_t pixel_mask_stride;
  uint32_t pre_erosion_width;
  uint32_t pre_erosion_stride;
  uint32_t test_error_mask;
};

struct AqInitialQuantErosionParams {
  uint32_t pre_erosion_width;
  uint32_t pre_erosion_height;
  uint32_t pre_erosion_stride;
  uint32_t block_width;
  uint32_t block_height;
  uint32_t quant_stride;
  uint32_t strategy_mask_stride;
  std::array<float, 4> weights;
};

struct AqInitialQuantModulationParams {
  uint32_t coding_stride;
  uint32_t block_width;
  uint32_t block_height;
  uint32_t quant_stride;
  float multiplier;
  float addend;
};

struct AqBlockReductionParams {
  uint32_t source_width;
  uint32_t source_height;
  uint32_t distance_stride;
  uint32_t block_stride;
  uint32_t anchor_offset;
  uint32_t anchor_count;
  uint32_t pixel_width;
  uint32_t pixel_height;
  uint32_t covered_width;
  uint32_t covered_height;
};

struct AqMaximumErrorReductionParams {
  uint32_t source_width;
  uint32_t source_height;
  uint32_t reference_stride;
  uint32_t reconstruction_stride;
  uint32_t block_stride;
  uint32_t anchor_offset;
  uint32_t anchor_count;
  uint32_t pixel_width;
  uint32_t pixel_height;
  uint32_t covered_width;
  uint32_t covered_height;
  float limit_x;
  float limit_y;
  float limit_b;
};

struct AqQuantizationProbeParams {
  uint32_t coefficient_count;
  uint32_t strategy;
  uint32_t channel;
  int32_t raw_quant;
  uint32_t global_scale;
  float matrix_multiplier;
};

struct AqAdjustmentProbeParams {
  uint32_t coefficient_count;
  uint32_t coefficient_width;
  uint32_t coefficient_height;
  uint32_t strategy;
  int32_t initial_raw_quant;
  uint32_t global_scale;
  float x_matrix_multiplier;
  float b_matrix_multiplier;
};

struct AqGaborishParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  std::array<float, 3> center_weight;
  std::array<float, 3> axis_weight;
  std::array<float, 3> diagonal_weight;
};

struct AqEpfParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t inverse_sigma_stride;
  uint32_t pass;
  float sigma_scale;
  float border_sad_multiplier;
  std::array<float, 3> channel_scale;
};

struct AqOpsinToLinearParams {
  uint32_t width;
  uint32_t height;
  uint32_t input_stride;
  uint32_t output_stride;
  float scale;
};

struct AqStrategyBatch {
  AcStrategyType strategy = AcStrategyType::kCount;
  size_t anchor_offset = 0;
  size_t anchor_count = 0;
  size_t coefficient_offset = 0;
  size_t coefficient_count = 0;
};

struct AqAnchor {
  size_t block_x = 0;
  size_t block_y = 0;
  AcStrategyType strategy = AcStrategyType::kCount;
  size_t batch_index = 0;
  size_t index_in_batch = 0;
};

class MetalPreparedAqEvaluation final : public PreparedAqEvaluation {
public:
  explicit MetalPreparedAqEvaluation(MetalBackend &backend);
  ~MetalPreparedAqEvaluation() override;

  Status Prepare(const AqEvaluationPreparation &preparation);
  Status Evaluate(AqEvaluationInput input, AqEvaluationOutput output) override;
  Status Reconfigure(const AcStrategyGrid& strategies,
                     ConstPlaneU8View epf_sharpness) override;
  Status EncodeFrame(AqEvaluationInput input,
                     VarDctEncoderFrame *frame) override;
  Status ComputeInitialQuantization(
      InitialQuantizationOptions options,
      InitialQuantFieldOutput output) override;
  Status EvaluateProfiled(AqEvaluationInput input, AqEvaluationOutput output,
                          MetalAqEvaluationProfile* profile);
  AqEvaluationMemoryStats memory_stats() const noexcept override;

  Status SubmitEvaluation(AqEvaluationInput input,
                          bool profiling_reserved = false);
  Status FinishEvaluation(AqEvaluationOutput output);
  Status FailNextUpload();
  Status FailNextNumeric();
  Status FailNextReadback();
  Status SetWaitObserver(bool *observed);
  Status RunBlockReduction(ConstPlaneF32View distance_map,
                           PlaneF32View block_distance_map);

  Status RunReconstruction(AqEvaluationInput input,
                           MetalAqReconstructionSnapshotForTesting *snapshot);
  Status RunQuantizationProbe(const MetalAqQuantizationProbeForTesting &probe,
                              std::vector<int32_t> *quantized,
                              std::vector<float> *dequantized);
  Status RunAdjustmentProbe(
      const MetalAqAdjustmentProbeForTesting& probe,
      MetalAqAdjustmentResultForTesting* result);
  Status RunPostprocess(ConstImage3FView reconstructed_opsin,
                        ConstPlaneF32View epf_inverse_sigma,
                        MetalAqPostprocessSnapshotForTesting *snapshot);
  Status RunReconstructionAndPostprocess(
      AqEvaluationInput input, MetalAqPostprocessSnapshotForTesting *snapshot);
  Status GetPostprocessPlan(MetalAqPostprocessPlanForTesting *plan) const;
  Status RunButteraugli(
      AqEvaluationInput input,
      MetalAqButteraugliSnapshotForTesting *snapshot);
  Status FailNextButteraugli(bool fail_submission, bool fail_completion,
                            bool fail_readback);

private:
  enum class State {
    kReady,
    kBusy,
    kInvalid,
  };

  Status ValidatePreparation(const AqEvaluationPreparation &preparation) const;
  Status ValidateInput(AqEvaluationInput input) const;
  Status ValidateOutput(AqEvaluationOutput output) const;
  Status BeginOperation(bool profiling_reserved = false);
  Status UploadInput(AqEvaluationInput input);
  Status PrepareExactCoefficientStaging(AqEvaluationInput input);
  Status PrepareReconstructionDiagnosticReadback();
  Status PreparePostprocessDiagnosticReadback();
  Status PrepareQuantizationProbeReadback();
  Status ReadbackRawQuant();
  Status ReadbackColorCorrelation();
  Status AssembleFrameFromReadback(VarDctEncoderFrame *frame) const;
  Status WaitForOperation();
  void CompleteOperation();
  void Invalidate();

  static void EncodeEvaluationSubmission(MetalBackend &backend,
                                         MTL::ComputeCommandEncoder *encoder,
                                         const void *context);
  static void EncodeBlockReductionSubmission(
      MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
      const void *context);
  static void
  EncodeReconstructionSubmission(MetalBackend &backend,
                                 MTL::ComputeCommandEncoder *encoder,
                                 const void *context);
  static void EncodeFrameSubmission(MetalBackend &backend,
                                    MTL::ComputeCommandEncoder *encoder,
                                    const void *context);
  static void EncodeInitialQuantizationSubmission(
      MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
      const void* context);
  static void
  EncodeQuantizationProbeSubmission(MetalBackend &backend,
                                    MTL::ComputeCommandEncoder *encoder,
                                    const void *context);
  static void EncodeAdjustmentProbeSubmission(
      MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
      const void* context);
  static void EncodePostprocessSubmission(MetalBackend &backend,
                                          MTL::ComputeCommandEncoder *encoder,
                                          const void *context);
  static void EncodeReconstructionAndPostprocessSubmission(
      MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
      const void *context);

  void EncodePostprocess(MetalBackend &backend,
                         MTL::ComputeCommandEncoder *encoder) const;
  void EncodeBlockReduction(MetalBackend &backend,
                            MTL::ComputeCommandEncoder *encoder) const;
  void EncodeMaximumErrorReduction(
      MetalBackend &backend, MTL::ComputeCommandEncoder *encoder) const;
  [[nodiscard]] std::array<DevicePlaneView, 3>
  FinalFilteredImage() const noexcept;
  Status FinishPostprocess(MetalAqPostprocessSnapshotForTesting *snapshot);

  MetalBackend *backend_ = nullptr;
  DeviceScratchArena persistent_;
  DeviceScratchArena staging_;
  std::array<DevicePlaneView, 3> original_;
  std::array<DevicePlaneView, 3> coding_;
  std::array<DevicePlaneView, 3> reconstructed_;
  std::array<std::array<DevicePlaneView, 3>, 2> filter_scratch_;
  std::array<DevicePlaneView, 3> reconstructed_linear_;
  DevicePlaneView strategies_;
  DevicePlaneView anchors_;
  DevicePlaneView epf_sharpness_;
  DevicePlaneView quant_tables_;
  DevicePlaneView raw_quant_;
  DevicePlaneView inverse_sigma_;
  DevicePlaneView y_to_x_;
  DevicePlaneView y_to_b_;
  DevicePlaneView initial_quant_pre_erosion_;
  DevicePlaneView initial_quant_unblurred_pixel_mask_;
  DevicePlaneView initial_quant_field_;
  DevicePlaneView initial_quant_strategy_mask_;
  DevicePlaneView initial_quant_pixel_mask_;
  DevicePlaneView block_distance_;
  DevicePlaneView distance_map_;
  DevicePlaneView score_;
  DevicePlaneView transform_maximum_error_;
  DevicePlaneView gathered_pixels_;
  DevicePlaneView forward_coefficients_;
  DevicePlaneView quantized_coefficients_;
  DevicePlaneView reconstruction_coefficients_;
  DevicePlaneView dc_;
  DevicePlaneView quantized_dc_;
  DevicePlaneView reconstruction_error_;
  DevicePlaneView quant_probe_input_;
  DevicePlaneView quant_probe_quantized_;
  DevicePlaneView quant_probe_dequantized_;
  Extent2D source_extent_;
  Extent2D coding_extent_;
  Extent2D block_extent_;
  Extent2D tile_extent_;
  size_t block_count_ = 0;
  size_t pixel_count_ = 0;
  size_t coefficient_value_count_ = 0;
  size_t anchor_count_ = 0;
  size_t maximum_coefficient_count_ = 0;
  size_t filter_scratch_image_count_ = 0;
  int final_filter_scratch_index_ = -1;
  AqEvaluationOptions options_;
  AcStrategyGrid strategies_host_;
  std::vector<uint8_t> epf_sharpness_host_;
  std::vector<int32_t> last_raw_quant_;
  std::vector<int8_t> last_y_to_x_;
  std::vector<int8_t> last_y_to_b_;
  std::vector<float> last_initial_quant_field_;
  std::vector<float> last_initial_strategy_mask_;
  std::vector<float> last_initial_pixel_mask_;
  Quantizer last_quantizer_;
  AqEvaluationMemoryStats memory_stats_;
  AqResetParams reset_params_{};
  AqInitialCflParams initial_cfl_params_{};
  AqInitialQuantGradientParams initial_quant_gradient_params_{};
  AqInitialQuantErosionParams initial_quant_erosion_params_{};
  AqInitialQuantModulationParams initial_quant_modulation_params_{};
  std::array<AqBlockReductionParams, 7> block_reduction_params_{};
  std::array<AqMaximumErrorReductionParams, 7>
    maximum_error_reduction_params_{};
  AqQuantizationProbeParams quant_probe_params_{};
  AqAdjustmentProbeParams adjustment_probe_params_{};
  AqGaborishParams gaborish_params_{};
  std::array<AqEpfParams, 3> epf_params_{};
  AqOpsinToLinearParams opsin_to_linear_params_{};
  std::array<AqStrategyBatch, 7> batches_{};
  std::array<AqReconstructionParams, 7> reconstruction_params_{};
  std::vector<AqAnchor> row_major_anchors_;
  std::vector<vardct_frame_internal::QuantizedAcTransformView>
    final_transform_views_;
  std::vector<float> readback_;
  std::vector<float> transform_maximum_error_readback_;
  std::vector<float> forward_readback_;
  std::vector<float> exact_reconstruction_coefficients_;
  std::vector<int32_t> quantized_readback_;
  std::vector<float> dc_readback_;
  std::vector<int32_t> quantized_dc_readback_;
  std::array<std::vector<float>, 3> reconstructed_readback_;
  std::array<std::vector<float>, 3> filtered_readback_;
  std::array<std::vector<float>, 3> linear_readback_;
  std::unique_ptr<PreparedDeviceButteraugli> butteraugli_;
  std::vector<int32_t> quant_probe_quantized_readback_;
  std::vector<float> quant_probe_dequantized_readback_;
  mutable std::mutex mutex_;
  State state_ = State::kReady;
  std::unique_ptr<GpuSubmission> submission_;
  bool fail_next_readback_ = false;
  bool fail_next_upload_ = false;
  bool fail_next_numeric_ = false;
  bool fail_next_butteraugli_submission_ = false;
  bool fail_next_butteraugli_completion_ = false;
  bool fail_next_butteraugli_readback_ = false;
  bool *wait_observer_ = nullptr;
  MetalAqEvaluationProfile* active_profile_ = nullptr;
  bool exact_coefficients_ = false;
  bool exact_coefficient_reconstruction_ = false;
  bool exact_linear_reconstruction_ = false;
  AcCoefficientDecisionMode coefficient_decision_mode_ =
    AcCoefficientDecisionMode::kFixedRawQuant;
  bool frame_only_ = false;
  bool frame_only_inverse_gaborish_ = false;
  bool frame_only_resident_initial_cfl_ = false;
  bool frame_only_resident_initial_quant_ = false;
};

} // namespace gjxl::metal_internal
