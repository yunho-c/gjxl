// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <Metal/Metal.hpp>

#include "core/status.h"
#include "gpu/ops/butteraugli.h"

namespace gjxl::metal_internal {

class MetalBackend;

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

}  // namespace gjxl::metal_internal
