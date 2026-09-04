// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <memory>

#include "core/image.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl {

struct LinearRgbOpsinPreparationOptions {
  Extent2D padded_extent;
  float intensity_target = 255.0f;
  bool compute_matrix_scale_stats = false;
};

/// Owns source RGB and padded Opsin planes prepared on one GPU backend.
/// Device views remain valid until this object is destroyed.
class PreparedGpuLinearRgbOpsin {
 public:
  virtual ~PreparedGpuLinearRgbOpsin() = default;

  PreparedGpuLinearRgbOpsin(const PreparedGpuLinearRgbOpsin&) = delete;
  PreparedGpuLinearRgbOpsin& operator=(
    const PreparedGpuLinearRgbOpsin&) = delete;

  [[nodiscard]] virtual ConstDeviceImage3View original_linear_rgb()
    const noexcept = 0;
  [[nodiscard]] virtual ConstDeviceImage3View coding_opsin() const noexcept = 0;
  [[nodiscard]] virtual std::array<float, 3> matrix_scale_stats()
    const noexcept = 0;

 protected:
  PreparedGpuLinearRgbOpsin() = default;
};

/// Optional backend capability for uploading linear RGB once and producing
/// padded Opsin without crossing back through host memory.
class GpuLinearRgbOpsinPreparation {
 public:
  virtual ~GpuLinearRgbOpsinPreparation() = default;

  [[nodiscard]] virtual Status PrepareLinearRgbOpsin(
    ConstImage3FView linear_rgb,
    LinearRgbOpsinPreparationOptions options,
    std::unique_ptr<PreparedGpuLinearRgbOpsin>* prepared) = 0;
};

[[nodiscard]] inline GpuLinearRgbOpsinPreparation*
QueryGpuLinearRgbOpsinPreparation(
  GpuBackend& backend) noexcept {
  return dynamic_cast<GpuLinearRgbOpsinPreparation*>(&backend);
}

}  // namespace gjxl
