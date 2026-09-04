// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>

#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/gpu_execution_profile_internal.h"

namespace gjxl::aq_evaluation_internal {

struct ResidentEncodingPolicySetup {
  float quant_dc = 0.0f;
  float lower_bound = 0.0f;
  float upper_bound = 0.0f;
};

/// Optional backend entry point for host images whose complete finite-value
/// validation was already performed by the same synchronous internal
/// workflow. Public GPU preparation must use GpuAqEvaluation instead.
class GpuValidatedAqEvaluation {
public:
  virtual ~GpuValidatedAqEvaluation() = default;

  [[nodiscard]] virtual Status PrepareValidatedAqEvaluation(
    const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared) = 0;

  [[nodiscard]] virtual Status PrepareValidatedAqEvaluationProfiled(
    const AqEvaluationPreparation& preparation,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<PreparedAqEvaluation>* prepared,
    gpu_profile_internal::GpuExecutionProfile* profile) = 0;
};

/// Optional prepared-operation capability for target-size retries whose only
/// evaluation-profile changes are the quantization-matrix scale selectors.
/// The operation retains its device allocations and all source data.
class PreparedAqScaleReconfiguration {
public:
  virtual ~PreparedAqScaleReconfiguration() = default;

  [[nodiscard]] virtual Status ReconfigureScaleSelectors(
    AqEvaluationOptions options) = 0;
};

/// Optional encoding-only contract that keeps initial AQ maps resident. The
/// complete public initial-quantization contract continues to materialize all
/// requested maps.
class PreparedAqEncodingInitialQuantization {
public:
  virtual ~PreparedAqEncodingInitialQuantization() = default;

  [[nodiscard]] virtual Status ComputeInitialQuantizationForEncoding(
    InitialQuantizationOptions options,
    QuantizerParams* quantizer = nullptr,
    float quant_dc = 0.0f) = 0;

  /// Applies strategy-aware adjustment to the resident initial field and
  /// returns only the scalar bounds needed by host policy control.
  [[nodiscard]] virtual Status PrepareResidentEncodingPolicy(
    float butteraugli_target,
    ResidentEncodingPolicySetup* setup) {
    (void)butteraugli_target;
    (void)setup;
    return Status::Unavailable(
      "Prepared resident encoding policy is unavailable");
  }
};

}  // namespace gjxl::aq_evaluation_internal
