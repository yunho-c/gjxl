// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>

#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/gpu_execution_profile_internal.h"

namespace gjxl::aq_evaluation_internal {

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

}  // namespace gjxl::aq_evaluation_internal
