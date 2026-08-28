// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <vector>

#include "codestream/encoder_internal.h"
#include "codestream/workflow.h"
#include "gpu/backend.h"

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

}  // namespace gjxl::codestream_internal
