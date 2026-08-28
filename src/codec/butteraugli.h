// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>

#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

struct ButteraugliOptions {
  float hf_asymmetry = 1.0f;
  float x_multiplier = 1.0f;
  float intensity_target = 80.0f;

  friend bool operator==(
    const ButteraugliOptions&,
    const ButteraugliOptions&) = default;
};

/// Caches the target-invariant perceptual representation of one reference.
///
/// Comparisons reuse the prepared reference and internal working storage.
/// One instance must not be used concurrently. Preparation and comparison are
/// atomic with respect to the prior prepared state and caller-visible output.
class PreparedButteraugliReference {
public:
  PreparedButteraugliReference();
  ~PreparedButteraugliReference();

  PreparedButteraugliReference(const PreparedButteraugliReference&) = delete;
  PreparedButteraugliReference& operator=(
    const PreparedButteraugliReference&) = delete;
  PreparedButteraugliReference(PreparedButteraugliReference&&) noexcept;
  PreparedButteraugliReference& operator=(
    PreparedButteraugliReference&&) noexcept;

  [[nodiscard]] Status Prepare(
    ConstImage3FView reference_linear_rgb,
    ButteraugliOptions options);

  [[nodiscard]] Status Compare(
    ConstImage3FView distorted_linear_rgb,
    PlaneF32View distance_map,
    double* score);

  [[nodiscard]] Extent2D extent() const noexcept;
  [[nodiscard]] ButteraugliOptions options() const noexcept;
  [[nodiscard]] bool ready() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Computes the native CPU Butteraugli map and its maximum aggregate score.
/// Inputs are linear sRGB values. Strided outputs and input/output overlap are
/// supported, and outputs are committed atomically.
[[nodiscard]] Status ComputeButteraugliDistance(
  ConstImage3FView reference_linear_rgb,
  ConstImage3FView distorted_linear_rgb,
  ButteraugliOptions options,
  PlaneF32View distance_map,
  double* score);

/// Reduces a pixel-resolution Butteraugli map to one 16-norm distance per
/// base block. Every cell covered by a multiblock strategy receives the same
/// transform-wide value, matching libjxl's AQ feedback map.
[[nodiscard]] Status ReduceButteraugliDistanceMap(
  ConstPlaneF32View distance_map,
  const AcStrategyGrid& strategies,
  PlaneF32View block_distance_map);

}  // namespace gjxl
