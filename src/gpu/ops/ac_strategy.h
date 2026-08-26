// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "gpu/buffer.h"

namespace gjxl {

inline constexpr size_t kAcStrategyCandidateChannelCount = 3;
inline constexpr size_t kAcStrategyCostMatrixCount = 6;
inline constexpr size_t kAcStrategyRateScratchBytesPerChannel =
  2 * sizeof(float);

/// One candidate in a same-strategy GPU evaluation batch.
///
/// Coordinates use JPEG XL 8x8 base blocks. `quant_norm` is the strategy-aware
/// aggregate of the source quant field. CfL factors contain the X and B
/// multiples of transformed Y; the Y factor is always zero. The footprint must
/// fit inside the batch image. Quant norm and entropy multiplier must be finite
/// and positive, and both CfL factors must be finite. Invalid device-resident
/// descriptors produce a non-finite cost.
struct AcStrategyCandidate {
  uint32_t block_x = 0;
  uint32_t block_y = 0;
  float quant_norm = 1.0f;
  float entropy_multiplier = 1.0f;
  float cfl_x = 0.0f;
  float cfl_b = 0.0f;
};

static_assert(std::is_standard_layout_v<AcStrategyCandidate>);
static_assert(sizeof(AcStrategyCandidate) == 6 * sizeof(uint32_t));

/// Device-resident inputs and scratch for batched AC candidate evaluation.
///
/// `opsin` stores three planar float images. Strides are expressed in floats;
/// `opsin_plane_stride` is the distance between channel starts. The mask is a
/// single float plane. `matrices` contains dequant X/Y/B followed by inverse-
/// dequant X/Y/B, with one complete strategy-sized matrix per entry.
///
/// `scratch_a` and `scratch_b` each require
/// `candidate_count * 3 * coefficient_count` floats. `rate_scratch` requires
/// `candidate_count * 3 * kAcStrategyRateScratchBytesPerChannel` bytes.
/// Inputs are expected to remain resident across batches; only candidate
/// descriptors and scalar costs need to cross the CPU/GPU boundary.
struct AcStrategyCandidateBatch {
  AcStrategyType strategy = AcStrategyType::kDct8;

  const DeviceBuffer* opsin = nullptr;
  const DeviceBuffer* pixel_mask = nullptr;
  const DeviceBuffer* matrices = nullptr;
  const DeviceBuffer* candidates = nullptr;

  DeviceBuffer* scratch_a = nullptr;
  DeviceBuffer* scratch_b = nullptr;
  DeviceBuffer* rate_scratch = nullptr;
  DeviceBuffer* costs = nullptr;

  Extent2D pixel_extent;
  size_t opsin_row_stride = 0;
  size_t opsin_plane_stride = 0;
  size_t pixel_mask_row_stride = 0;
  size_t candidate_count = 0;
  float butteraugli_target = 1.0f;
};

}  // namespace gjxl
