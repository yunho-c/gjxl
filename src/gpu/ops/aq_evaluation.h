// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <memory>

#include "codec/butteraugli.h"
#include "codec/codestream.h"
#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"
#include "gpu/backend.h"

namespace gjxl {

struct AqEvaluationOptions {
  SimpleVarDctCodestreamProfile profile;
  ButteraugliOptions butteraugli;
};

struct AqEvaluationPreparation {
  ConstImage3FView original_linear_rgb;
  ConstImage3FView coding_opsin;
  const AcStrategyGrid* strategies = nullptr;
  ConstPlaneU8View epf_sharpness;
  AqEvaluationOptions options;
};

struct AqEvaluationInput {
  ConstPlaneI32View raw_quant_field;
  QuantizerParams quantizer;
  ConstPlaneI8View y_to_x;
  ConstPlaneI8View y_to_b;
  ConstPlaneF32View epf_inverse_sigma;
};

struct AqEvaluationOutput {
  struct Final;

  PlaneF32View block_distance_map;
  double* score = nullptr;
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

  /// Executes one complete resident reconstruction, filtering, Butteraugli,
  /// and strategy-aware block reduction. When `output.final` is non-null, the
  /// same submission also materializes reconstructed RGB and the encoder
  /// frame. Failure never changes caller-visible output.
  [[nodiscard]] virtual Status Evaluate(
    AqEvaluationInput input,
    AqEvaluationOutput output) = 0;

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
