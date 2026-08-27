// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include "core/image.h"
#include "core/status.h"
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

}  // namespace gjxl
