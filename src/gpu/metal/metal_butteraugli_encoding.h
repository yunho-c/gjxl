// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <Metal/Metal.hpp>

#include <array>
#include <cstdint>
#include <span>

#include "core/status.h"
#include "gpu/ops/butteraugli.h"

namespace gjxl::metal_internal {

class MetalBackend;

/// Nine disjoint mutable planes borrowed from the enclosing AQ operation.
/// They must be F32, large enough for the unpadded reference, and remain owned
/// until this prepared Butteraugli object is destroyed. AQ orders reference
/// preparation before reconstruction and each comparison after filtering and
/// gathering, so no other consumer may use these planes during either phase.
/// Immutable reference data and final outputs are never stored here.
struct MetalButteraugliScratch {
  std::array<DevicePlaneView, 9> planes;
};

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
  kResidentReduction,
};

struct MetalButteraugliResidentBatch {
  uint32_t anchor_offset = 0;
  uint32_t anchor_count = 0;
  uint32_t pixel_width = 0;
  uint32_t pixel_height = 0;
  uint32_t covered_width = 0;
  uint32_t covered_height = 0;
};

/// Resident encoder sinks for a prepared multiscale comparison. Unlike the
/// public comparison descriptor, this internal form intentionally does not
/// materialize a complete per-pixel distance map.
struct MetalButteraugliResidentComparisonDescriptor {
  ConstDeviceImage3View distorted_linear_rgb;
  ConstDevicePlaneView anchors;
  DevicePlaneView block_distance;
  DevicePlaneView score_partials;
  DevicePlaneView score;
  DevicePlaneView error;
  std::span<const MetalButteraugliResidentBatch> batches;
};

/// Prevents pooling after failure in an enclosing AQ submission/readback.
/// The owner still must wait for its submission before destroying the borrower.
void DiscardPreparedMetalButteraugliLease(
  PreparedDeviceButteraugli& prepared) noexcept;

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

/// Validates a resident-only comparison before its enclosing command buffer
/// is created. Only prepared multiscale comparisons support this sink path.
[[nodiscard]] Status ValidatePreparedMetalButteraugliResidentEncoding(
  PreparedDeviceButteraugli& prepared,
  const MetalButteraugliResidentComparisonDescriptor& descriptor);

/// Appends an already validated resident-only comparison to an existing
/// encoder. The complete-map public path remains the diagnostic oracle.
void EncodePreparedMetalButteraugliResident(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const MetalButteraugliResidentComparisonDescriptor& descriptor);

/// Appends one dependency-ordered stage of a validated resident comparison.
void EncodePreparedMetalButteraugliResidentProfileStage(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const MetalButteraugliResidentComparisonDescriptor& descriptor,
  MetalButteraugliProfileStage stage);

/// Appends one dependency-ordered diagnostic comparison stage. The descriptor
/// must have been validated before the command buffer was created.
void EncodePreparedMetalButteraugliProfileStage(
  PreparedDeviceButteraugli& prepared,
  MTL::ComputeCommandEncoder* encoder,
  const DeviceButteraugliComparisonDescriptor& descriptor,
  MetalButteraugliProfileStage stage);

}  // namespace gjxl::metal_internal
