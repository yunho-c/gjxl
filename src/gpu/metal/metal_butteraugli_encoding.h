// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <Metal/Metal.hpp>

#include <cstdint>

#include "core/status.h"
#include "gpu/ops/butteraugli.h"

namespace gjxl::metal_internal {

class MetalBackend;

enum class MetalButteraugliProfileStage : uint8_t {
  kDistortedPsychoMain,
  kMaltaMain,
  kL2Main,
  kMaskAndFinalMain,
  kDistortedPsychoSub,
  kMaltaSub,
  kL2Sub,
  kMaskAndFinalSub,
  kScoreReduction,
};

/// Validates an AQ-owned comparison before its enclosing command buffer is
/// created. This is intentionally Metal-only and is not a generic GPU command
/// interface.
[[nodiscard]] Status ValidatePreparedMetalButteraugliEncoding(
  PreparedDeviceButteraugli& prepared,
  const DeviceButteraugliComparisonDescriptor& descriptor);

/// Appends an already validated prepared comparison to an existing encoder.
void EncodePreparedMetalButteraugli(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const DeviceButteraugliComparisonDescriptor& descriptor);

/// Appends one dependency-ordered diagnostic comparison stage. The descriptor
/// must have been validated before the command buffer was created.
void EncodePreparedMetalButteraugliProfileStage(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const DeviceButteraugliComparisonDescriptor& descriptor,
  MetalButteraugliProfileStage stage);

}  // namespace gjxl::metal_internal
