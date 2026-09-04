// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

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

class ColorCorrelationMap;

enum class AqEvaluationMetric {
  kButteraugli,
  kMaximumError,
};

struct AqEvaluationOptions {
  SimpleVarDctCodestreamProfile profile;
  ButteraugliOptions butteraugli;
  AqEvaluationMetric metric = AqEvaluationMetric::kButteraugli;
  std::array<float, 3> maximum_error{};

  friend bool operator==(const AqEvaluationOptions&,
                         const AqEvaluationOptions&) = default;
};

struct AqEvaluationPreparation {
  ConstImage3FView original_linear_rgb;
  ConstImage3FView coding_opsin;
  /// Optional resident coding image owned by another prepared operation.
  /// When present, the host coding view supplies geometry and validation only;
  /// the backend borrows these device planes instead of allocating and
  /// uploading a duplicate image. The owner must outlive this evaluation.
  ConstDeviceImage3View resident_coding_opsin;
  const AcStrategyGrid* strategies = nullptr;
  ConstPlaneU8View epf_sharpness;
  AqEvaluationOptions options;
  /// Omits perceptual-reference preparation when only EncodeFrame is used.
  bool frame_only = false;
  /// Applies the profile's inverse Gaborish filter on device before frame-only
  /// coefficient coding, avoiding a host materialization boundary.
  bool frame_only_inverse_gaborish = false;
  /// Computes the fast pixel-domain initial chroma-from-luma map from the
  /// resident coding image and allows the input color-map views to be omitted.
  /// Complete evaluators may subsequently replace it with final resident CfL.
  bool resident_initial_cfl = false;
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
  /// Enables strategy-aware quant-field adjustment and per-evaluation
  /// quantizer/raw-quant construction inside the prepared operation. Backends
  /// may reject this resident capability.
  bool resident_quantization = false;
  /// Selects whether resident coefficient coding preserves the input raw
  /// quant or applies the encoder's shared AdjustQuantBlockAC decision.
  AcCoefficientDecisionMode coefficient_decision_mode =
    AcCoefficientDecisionMode::kFixedRawQuant;
};

struct ResidentAcStrategyInputs {
  ConstDeviceImage3View opsin;
  ConstDevicePlaneView quant_field;
  ConstDevicePlaneView pixel_mask;
  /// Optional resident initial CfL maps. When both are present, candidate
  /// evaluation does not require a host ColorCorrelationMap.
  ConstDevicePlaneView y_to_x;
  ConstDevicePlaneView y_to_b;
};

struct AqEvaluationInput {
  ConstPlaneI32View raw_quant_field;
  QuantizerParams quantizer;
  ConstPlaneI8View y_to_x;
  ConstPlaneI8View y_to_b;
  ConstPlaneF32View epf_inverse_sigma;
  /// Optional resident field-construction input. When valid, the prepared
  /// operation derives `quantizer`, raw quantization, and EPF sigma on device;
  /// the host raw-quant and inverse-sigma views must be omitted.
  ConstPlaneF32View quant_field;
  float quant_dc = 0.0f;
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
  /// Receives the device-constructed quantizer for a resident field input.
  /// It is committed only after the evaluation completes successfully.
  QuantizerParams* quantizer = nullptr;
  Final* final = nullptr;
};

struct AqEvaluationOutput::Final {
  /// Optional diagnostic image. An empty view keeps reconstructed RGB on the
  /// device while still allowing frame materialization.
  Image3FView reconstructed_linear_rgb;
  /// Required encoder frame when final materialization is requested.
  VarDctEncoderFrame* frame = nullptr;
};

struct AqResidentButteraugliPolicyInput {
  ConstPlaneF32View adjusted_initial_quant_field;
  float quant_dc = 0.0f;
  float butteraugli_target = 0.0f;
  float lower_bound = 0.0f;
  float upper_bound = 0.0f;
  size_t iterations = 0;
  /// When false, applies every requested policy update and then quantizes the
  /// resulting field directly into `output.frame` without reconstructing and
  /// scoring that final field. Score history then contains `iterations`
  /// entries instead of `iterations + 1`. This is an explicit encoding-only
  /// optimization; diagnostic block-map and reconstruction outputs are not
  /// available for the unevaluated final field.
  bool evaluate_final_field = true;
};

struct AqResidentButteraugliPolicyOutput {
  /// Optional diagnostic materializations. An empty view leaves that result
  /// resident instead of transferring it to the host.
  PlaneF32View quant_field;
  PlaneF32View block_distance_map;
  std::vector<double>* score_history = nullptr;
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
  /// materializes the encoder frame and, when requested, reconstructed RGB.
  /// Failure never changes caller-visible output.
  [[nodiscard]] virtual Status Evaluate(
    AqEvaluationInput input,
    AqEvaluationOutput output) = 0;

  /// Executes the complete bounded Butteraugli policy while keeping each
  /// dependent quant-field update resident. Implementations may encode every
  /// evaluation into one submission. Failure never changes caller-visible
  /// output.
  [[nodiscard]] virtual Status EvaluateResidentButteraugliPolicy(
    AqResidentButteraugliPolicyInput input,
    AqResidentButteraugliPolicyOutput output) {
    (void)input;
    (void)output;
    return Status::Unavailable(
      "Prepared resident Butteraugli policy is unavailable");
  }

  /// Uploads one color-correlation map for reuse by every later evaluation.
  /// Once configured, evaluation inputs must omit `y_to_x` and `y_to_b`.
  /// Reconfiguring strategy metadata invalidates this binding.
  [[nodiscard]] virtual Status SetInvariantColorCorrelation(
    ConstPlaneI8View y_to_x,
    ConstPlaneI8View y_to_b) {
    (void)y_to_x;
    (void)y_to_b;
    return Status::Unavailable(
      "Prepared invariant color correlation is unavailable");
  }

  /// Derives and retains the fixed final color-correlation map from resident
  /// coding pixels and the supplied initial quantization field. Backends may
  /// also retain invariant forward coefficients for later evaluations. The
  /// derivation may be fused into the next evaluation, but must still consume
  /// these supplied values independently of that evaluation's quantization.
  [[nodiscard]] virtual Status PrepareInvariantColorCorrelationResident(
    ConstPlaneF32View quant_field,
    float quant_dc) {
    (void)quant_field;
    (void)quant_dc;
    return Status::Unavailable(
      "Prepared resident color correlation is unavailable");
  }

  /// Applies the prepared strategy grid's adjustment to one host field using
  /// device execution. The adjusted field is committed atomically after the
  /// submission and readback complete.
  [[nodiscard]] virtual Status AdjustQuantFieldResident(
    float butteraugli_target,
    ConstPlaneF32View input,
    PlaneF32View output) {
    (void)butteraugli_target;
    (void)input;
    (void)output;
    return Status::Unavailable(
      "Prepared resident quant-field adjustment is unavailable");
  }

  /// Rebinds target-dependent strategy and EPF metadata without reallocating
  /// the prepared source, metric reference, or evaluation scratch. A backend
  /// may reject this for a frame-only preparation whose minimal arena omits
  /// reconfiguration storage.
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
    float quant_dc = 0.0f,
    ColorCorrelationMap* initial_color_correlation = nullptr) {
    (void)options;
    (void)output;
    (void)quantizer;
    (void)quant_dc;
    (void)initial_color_correlation;
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
