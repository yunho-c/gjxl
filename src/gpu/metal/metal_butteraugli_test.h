// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include "core/image.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/ops/butteraugli.h"

namespace gjxl {

/// Test-only observable stages matching the pinned scalar golden ordering.
enum class MetalButteraugliStage : size_t {
  kBlurSigma1p2,
  kBlurSigma7p15593339443,
  kBlurSigma3p22489901262,
  kBlurSigma1p56416327805,
  kBlurSigma2p7,
  kOpsinX,
  kOpsinY,
  kOpsinB,
  kLowFrequencyX,
  kLowFrequencyY,
  kLowFrequencyB,
  kMediumFrequencyX,
  kMediumFrequencyY,
  kMediumFrequencyB,
  kHighFrequencyX,
  kHighFrequencyY,
  kUltraHighFrequencyX,
  kUltraHighFrequencyY,
  kMaltaMediumFrequencyY,
  kMaltaMediumFrequencyX,
  kMaltaHighFrequencyY,
  kMaltaHighFrequencyX,
  kMaltaUltraHighFrequencyY,
  kMaltaUltraHighFrequencyX,
  kMask,
  kMaskedAcY,
  kFinalComposition,
  kCount,
};

/// Test- and benchmark-only accounting for one prepared Metal state.
struct MetalButteraugliResourceUsage {
  size_t prepared_allocation_bytes = 0;
  size_t cached_reference_bytes = 0;
  size_t gaussian_kernel_bytes = 0;
  size_t peak_comparison_scratch_bytes = 0;
};

/// Selects one stage to copy into preallocated diagnostic storage during the
/// next comparison. Only non-expanded, single-scale prepared states support
/// capture. This API is linked only by tests.
[[nodiscard]] Status ConfigureMetalButteraugliStageCaptureForTest(
  PreparedDeviceButteraugli& prepared,
  MetalButteraugliStage stage);

/// Atomically reads the selected stage after a successful comparison.
[[nodiscard]] Status ReadMetalButteraugliStageCaptureForTest(
  PreparedDeviceButteraugli& prepared,
  PlaneF32View output);

/// Returns logical cache/scratch sizes and the actual owning allocation.
[[nodiscard]] Status QueryMetalButteraugliResourceUsageForTest(
  PreparedDeviceButteraugli& prepared,
  MetalButteraugliResourceUsage* usage);

/// Injects a failure into the next Metal compute submission only.
[[nodiscard]] Status ArmNextMetalSubmissionFailureForTest(
  GpuBackend& backend,
  bool fail_submission,
  bool fail_completion);

}  // namespace gjxl
