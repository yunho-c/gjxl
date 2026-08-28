// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "codec/adaptive_quantization.h"
#include "codec/butteraugli.h"
#include "codec/codestream.h"
#include "codec/maximum_error.h"
#include "codec/vardct_frame.h"
#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl {

enum class AqEvaluationMetric {
  kButteraugli,
  kMaximumError,
};

struct AqEvaluationOptions {
  SimpleVarDctCodestreamProfile profile;
  ButteraugliOptions butteraugli;
  AqEvaluationMetric metric = AqEvaluationMetric::kButteraugli;
  std::array<float, 3> maximum_error{};
};

struct AqEvaluationPreparation {
  ConstImage3FView original_linear_rgb;
  ConstImage3FView coding_opsin;
  const AcStrategyGrid* strategies = nullptr;
  ConstPlaneU8View epf_sharpness;
  AqEvaluationOptions options;
  /// Omits perceptual-reference preparation when only EncodeFrame is used.
  bool frame_only = false;
  /// Applies the profile's inverse Gaborish filter on device before frame-only
  /// coefficient coding, avoiding a host materialization boundary.
  bool frame_only_inverse_gaborish = false;
  /// Computes the fast pixel-domain initial chroma-from-luma map from the
  /// resident coding image. This is valid only for frame-only encoding and
  /// allows the input color-map views to be omitted.
  bool frame_only_resident_initial_cfl = false;
  /// Enables initial quant-field and masking-map generation from the resident
  /// coding image. A complete preparation must also select a resident
  /// consumer for these fields.
  bool frame_only_resident_initial_quant = false;
  /// Retains the initial field, blurred mask, and search-domain opsin as
  /// device views for AC-strategy candidate evaluation. Requires resident
  /// initial quantization, but is valid for a complete AQ preparation.
  bool resident_ac_strategy_inputs = false;
  /// Consumes device-generated raw quantization and permits the raw-quant and
  /// EPF input views to be omitted. Requires resident initial quantization.
  bool frame_only_resident_quantizer = false;
  /// Selects whether resident coefficient coding preserves the input raw
  /// quant or applies the encoder's shared AdjustQuantBlockAC decision.
  AcCoefficientDecisionMode coefficient_decision_mode =
    AcCoefficientDecisionMode::kFixedRawQuant;
};

struct ResidentAcStrategyInputs {
  ConstDeviceImage3View opsin;
  ConstDevicePlaneView quant_field;
  ConstDevicePlaneView pixel_mask;
};

struct AqEvaluationInput {
  ConstPlaneI32View raw_quant_field;
  QuantizerParams quantizer;
  ConstPlaneI8View y_to_x;
  ConstPlaneI8View y_to_b;
  ConstPlaneF32View epf_inverse_sigma;
  // Optional exact CPU evaluation prefix. Exact coefficients alone request
  // device reconstruction; an exact linear image advances the handoff past
  // reconstruction and filtering. Backends that do not consume these fields
  // may reject them.
  const VarDctEncoderFrame* exact_coefficients = nullptr;
  ConstImage3FView exact_reconstructed_linear_rgb;
};

struct AqEvaluationOutput {
  struct Final;

  PlaneF32View block_distance_map;
  double* score = nullptr;
  MaximumErrorReduction* maximum_error = nullptr;
  Final* final = nullptr;
};

struct AqEvaluationOutput::Final {
  Image3FView reconstructed_linear_rgb;
  VarDctEncoderFrame* frame = nullptr;
};

struct AqEvaluationMemoryStats {
  size_t persistent_bytes = 0;
  size_t staging_bytes = 0;
  size_t peak_scratch_bytes = 0;
};

/// Owns one frame's device-resident adaptive-quantization evaluation state.
/// The backend used to prepare it must outlive this object.
class PreparedAqEvaluation {
public:
  virtual ~PreparedAqEvaluation() = default;

  PreparedAqEvaluation(const PreparedAqEvaluation&) = delete;
  PreparedAqEvaluation& operator=(const PreparedAqEvaluation&) = delete;

  /// Executes one complete resident reconstruction and filtering pass, then
  /// the prepared Butteraugli or maximum-error metric and strategy-aware
  /// reduction. When `output.final` is non-null, the same submission also
  /// materializes reconstructed RGB and the encoder frame. Failure never
  /// changes caller-visible output.
  [[nodiscard]] virtual Status Evaluate(
    AqEvaluationInput input,
    AqEvaluationOutput output) = 0;

  /// Rebinds target-dependent strategy and EPF metadata without reallocating
  /// the prepared source, metric reference, or evaluation scratch.
  [[nodiscard]] virtual Status Reconfigure(
    const AcStrategyGrid& strategies,
    ConstPlaneU8View epf_sharpness) = 0;

  /// Materializes only the quantized encoder frame. Backends may use this
  /// explicit fast path to omit inverse reconstruction and perceptual scoring.
  /// Failure leaves `frame` unchanged.
  [[nodiscard]] virtual Status EncodeFrame(
    AqEvaluationInput input,
    VarDctEncoderFrame* frame) {
    (void)input;
    (void)frame;
    return Status::Unavailable(
      "Prepared AQ frame-only encoding is unavailable");
  }

  /// Computes initial quantization from the prepared coding image. Backends
  /// may expose this only for an explicitly enabled frame-only preparation.
  [[nodiscard]] virtual Status ComputeInitialQuantization(
    InitialQuantizationOptions options,
    InitialQuantFieldOutput output,
    QuantizerParams* quantizer = nullptr,
    float quant_dc = 0.0f) {
    (void)options;
    (void)output;
    (void)quantizer;
    (void)quant_dc;
    return Status::Unavailable(
      "Prepared resident initial quantization is unavailable");
  }

  /// Returns non-owning views into a successfully computed resident initial
  /// quantization. The prepared operation owns the storage and must outlive
  /// every submission that consumes these views.
  [[nodiscard]] virtual Status GetResidentAcStrategyInputs(
    ResidentAcStrategyInputs* inputs) {
    (void)inputs;
    return Status::Unavailable(
      "Prepared resident AC-strategy inputs are unavailable");
  }

  [[nodiscard]] virtual AqEvaluationMemoryStats memory_stats() const noexcept = 0;

protected:
  PreparedAqEvaluation() = default;
};

/// Optional coherent adaptive-quantization operation implemented by a backend.
class GpuAqEvaluation {
public:
  virtual ~GpuAqEvaluation() = default;

  virtual Status PrepareAqEvaluation(
    const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared) = 0;
};

[[nodiscard]] inline GpuAqEvaluation* QueryGpuAqEvaluation(
  GpuBackend& backend) noexcept {

  return dynamic_cast<GpuAqEvaluation*>(&backend);
}

[[nodiscard]] inline Status PrepareAqEvaluation(
  GpuBackend& backend,
  const AqEvaluationPreparation& preparation,
  std::unique_ptr<PreparedAqEvaluation>* prepared) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Prepared AQ evaluation output pointer is null");
  }
  prepared->reset();
  GpuAqEvaluation* capability = QueryGpuAqEvaluation(backend);
  if (capability == nullptr) {
    return Status::Unavailable(
      "GPU backend does not provide prepared AQ evaluation");
  }
  return capability->PrepareAqEvaluation(preparation, prepared);
}

}  // namespace gjxl
