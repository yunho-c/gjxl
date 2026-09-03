// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>

#include "core/image.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl {

struct ResidentInputStatistics {
  float x_edge = 0.0f;
  float b_edge = 0.0f;
  float exposed_blue = 0.0f;
};

struct ResidentInputPreparation {
  ConstImage3FView original_linear_rgb;
  Extent2D coding_extent;
  bool compute_matrix_scale_statistics = false;
};

/// Owns one validated linear-RGB upload and the corresponding padded coding
/// Opsin image. Returned device views remain valid until this object is
/// destroyed; consumers must finish every borrowing submission first.
class PreparedResidentInput {
public:
  virtual ~PreparedResidentInput() = default;

  PreparedResidentInput(const PreparedResidentInput&) = delete;
  PreparedResidentInput& operator=(const PreparedResidentInput&) = delete;

  [[nodiscard]] virtual ConstDeviceImage3View original_linear_rgb() const
    noexcept = 0;
  [[nodiscard]] virtual ConstDeviceImage3View coding_opsin() const noexcept = 0;
  [[nodiscard]] virtual ResidentInputStatistics statistics() const noexcept = 0;

protected:
  PreparedResidentInput() = default;
};

/// Optional backend capability for preparing a shared resident encoder input.
class GpuResidentInputPreparation {
public:
  virtual ~GpuResidentInputPreparation() = default;

  [[nodiscard]] virtual Status PrepareResidentInput(
    const ResidentInputPreparation& preparation,
    std::unique_ptr<PreparedResidentInput>* prepared) = 0;
};

[[nodiscard]] inline GpuResidentInputPreparation*
QueryGpuResidentInputPreparation(GpuBackend& backend) noexcept {
  return dynamic_cast<GpuResidentInputPreparation*>(&backend);
}

[[nodiscard]] inline Status PrepareResidentInput(
  GpuBackend& backend,
  const ResidentInputPreparation& preparation,
  std::unique_ptr<PreparedResidentInput>* prepared) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Prepared resident input output pointer is null");
  }
  prepared->reset();
  GpuResidentInputPreparation* capability =
    QueryGpuResidentInputPreparation(backend);
  if (capability == nullptr) {
    return Status::Unavailable(
      "GPU backend does not provide resident input preparation");
  }
  return capability->PrepareResidentInput(preparation, prepared);
}

}  // namespace gjxl
