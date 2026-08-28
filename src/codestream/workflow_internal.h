// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <vector>

#include "codestream/encoder_internal.h"
#include "codestream/workflow.h"
#include "gpu/backend.h"
#include "gpu/ops/quantization_pipeline_profile_internal.h"

namespace gjxl::codestream_internal {

struct VarDctEncodingProfile {
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

struct VarDctEncodingStageProfile {
  profile_internal::HostInterval total;
  profile_internal::HostInterval input_preparation;
  profile_internal::HostInterval backend_selection;
  profile_internal::HostInterval quantization_pipeline;
  profile_internal::HostInterval codestream_encoding;
  profile_internal::HostInterval summary_assembly;
  VarDctCodestreamStageProfile codestream;
  quantization_pipeline_internal::GpuFrameOnlyPipelineProfile
    maximum_throughput;
};

[[nodiscard]] bool IsAutomaticMetalGeometryEligible(
  Extent2D padded_extent) noexcept;

[[nodiscard]] bool IsAutomaticMetalTargetEligible(
  float butteraugli_target) noexcept;

[[nodiscard]] bool IsAutomaticMetalBackendQualified(
  const GpuBackend& backend) noexcept;

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

/// Diagnostic-only production workflow entry point used by benchmarks. On
/// failure all caller-visible outputs, including both profiles, remain
/// unchanged.
[[nodiscard]] Status EncodeLinearRgbVarDctCodestreamStageProfiled(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingTiming* timing,
  VarDctEncodingStageProfile* profile);

}  // namespace gjxl::codestream_internal
