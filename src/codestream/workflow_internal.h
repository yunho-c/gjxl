// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "codestream/encoder_internal.h"
#include "codestream/workflow.h"
#include "gpu/backend.h"
#include "gpu/ops/gpu_execution_profile_internal.h"

namespace gjxl::codestream_internal {

struct QuantizationMatrixScaleStats {
  float x_edge = 0.0f;
  float b_edge = 0.0f;
  float exposed_blue = 0.0f;

  bool operator==(const QuantizationMatrixScaleStats&) const = default;
};

struct QuantizationMatrixScales {
  uint8_t x = 2;
  uint8_t b = 2;

  bool operator==(const QuantizationMatrixScales&) const = default;
};

/// Returns whether this encoding policy uses source-dependent matrix-scale
/// statistics. High density follows libjxl's effort-9-like behavior even when
/// its explicit effort value is lower. Maximum-error control never uses the
/// statistics because its matrix scales are fixed at 2/2.
[[nodiscard]] bool ShouldComputeQuantizationMatrixScaleStats(
  const VarDctEncodingOptions& options) noexcept;

/// Resolves the public compression request to one serializer behavior.
[[nodiscard]] VarDctEntropyBehavior ResolveEntropyBehavior(
  const VarDctEncodingOptions& options) noexcept;

/// Resolves the public effort request independently from entropy intensity.
[[nodiscard]] VarDctCoefficientOrderBehavior
ResolveCoefficientOrderBehavior(
  const VarDctEncodingOptions& options) noexcept;

/// Computes libjxl's source-dependent X/B matrix-scale statistics over the
/// unpadded opsin image. Failure leaves `stats` unchanged.
[[nodiscard]] Status ComputeQuantizationMatrixScaleStats(
  ConstImage3FView opsin,
  QuantizationMatrixScaleStats* stats);

/// Computes matrix-scale statistics without revalidating an Opsin view whose
/// complete finite-value provenance is owned by the synchronous workflow.
/// Passing any other view violates this internal contract. Failure leaves
/// `stats` unchanged.
[[nodiscard]] Status ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
  ConstImage3FView opsin,
  QuantizationMatrixScaleStats* stats);

/// Selects libjxl's X/B matrix scales for one complete encode attempt.
/// Maximum-error control always selects 2/2 and ignores the Butteraugli
/// target. Failure leaves `scales` unchanged.
[[nodiscard]] Status SelectQuantizationMatrixScales(
  const QuantizationMatrixScaleStats& stats,
  VarDctRateControlMode mode,
  float butteraugli_target,
  QuantizationMatrixScales* scales);

struct VarDctEncodingProfile {
  /// Maximum CPU threads simultaneously participating in this encode.
  size_t peak_cpu_participants = 0;
  uint64_t input_preparation_nanoseconds = 0;
  uint64_t backend_selection_nanoseconds = 0;
  uint64_t quantization_pipeline_nanoseconds = 0;
  uint64_t codestream_encoding_nanoseconds = 0;
  uint64_t summary_assembly_nanoseconds = 0;
  uint64_t total_nanoseconds = 0;
  VarDctExecutionBackend execution_backend = VarDctExecutionBackend::kCpu;
  VarDctCodestreamProfile codestream;

  bool operator==(const VarDctEncodingProfile&) const = default;
};

[[nodiscard]] bool IsAutomaticMetalGeometryEligible(
  Extent2D padded_extent) noexcept;

[[nodiscard]] bool IsAutomaticMetalTargetEligible(
  float butteraugli_target) noexcept;

[[nodiscard]] bool IsAutomaticMetalBackendQualified(
  const GpuBackend& backend) noexcept;

/// Initializes and validates the process-cached production Metal backend.
/// Used by frontends that promise eager failure for an explicitly forced
/// Metal execution policy.
[[nodiscard]] Status EnsureProductionMetalBackendAvailable();

[[nodiscard]] Status EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary = nullptr);

/// Diagnostic-only public-workflow entry point. On failure, `codestream`,
/// `summary`, and `profile` remain unchanged.
[[nodiscard]] Status
EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingProfile* profile);

/// Diagnostic-only public-workflow entry point with resident Metal GPU
/// timestamps. On failure, every caller-visible output remains unchanged.
[[nodiscard]] Status
EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  gpu_profile_internal::GpuProfilingMode profiling_mode,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingProfile* profile,
  gpu_profile_internal::GpuExecutionProfile* gpu_profile);

}  // namespace gjxl::codestream_internal
