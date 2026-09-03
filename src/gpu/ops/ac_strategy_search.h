// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "codec/ac_strategy.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl {

namespace ac_strategy_search_internal {
struct Prepared;
}

struct AcStrategyGpuSearchStats {
  std::array<size_t, kAcStrategyCount> candidate_counts{};
  size_t total_candidate_count = 0;
};

struct ResidentAcStrategySearchInputs {
  ConstDeviceImage3View opsin;
  ConstDevicePlaneView quant_field;
  ConstDevicePlaneView pixel_mask;
};

class PreparedAcStrategySearch;

namespace gpu_profile_internal {
class GpuProfilingSession;

[[nodiscard]] Status FindAcStrategyGridGpuResidentProfiled(
  GpuBackend&, ConstImage3FView, ConstPlaneF32View, ConstPlaneF32View,
  const ColorCorrelationMap&, ResidentAcStrategySearchInputs,
  AcStrategySearchOptions, AcStrategyGrid*, PreparedAcStrategySearch*,
  GpuProfilingSession*, AcStrategyGpuSearchStats*);
}

/// Reuses host staging and device allocations across compatible AC-strategy
/// searches. The backend must outlive every search and the prepared state.
class PreparedAcStrategySearch {
public:
  PreparedAcStrategySearch();
  ~PreparedAcStrategySearch();

  PreparedAcStrategySearch(const PreparedAcStrategySearch&) = delete;
  PreparedAcStrategySearch& operator=(const PreparedAcStrategySearch&) =
    delete;

private:
  std::unique_ptr<ac_strategy_search_internal::Prepared> impl_;

  friend Status FindAcStrategyGridGpuResident(
    GpuBackend&, ConstImage3FView, ConstPlaneF32View, ConstPlaneF32View,
    const ColorCorrelationMap&, ResidentAcStrategySearchInputs,
    AcStrategySearchOptions, AcStrategyGrid*, AcStrategyGpuSearchStats*,
    PreparedAcStrategySearch*);
  friend Status gpu_profile_internal::FindAcStrategyGridGpuResidentProfiled(
    GpuBackend&, ConstImage3FView, ConstPlaneF32View, ConstPlaneF32View,
    const ColorCorrelationMap&, ResidentAcStrategySearchInputs,
    AcStrategySearchOptions, AcStrategyGrid*, PreparedAcStrategySearch*,
    gpu_profile_internal::GpuProfilingSession*, AcStrategyGpuSearchStats*);
};

/// Selects an AC-strategy grid after staging every dependency-safe candidate
/// cost through one GPU submission. Search decisions and tie-breaking remain
/// on the CPU and are identical to FindAcStrategyGrid's traversal.
[[nodiscard]] Status FindAcStrategyGridGpu(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  AcStrategyGpuSearchStats* stats = nullptr);

/// Runs the same CPU merge policy while candidate evaluation consumes the
/// prepared opsin, quant field, and pixel mask directly from device memory.
/// The host quant field and pixel mask remain the search-policy inputs. The
/// host Opsin view may be empty when only its resident geometry is needed and
/// is never uploaded by this operation.
[[nodiscard]] Status FindAcStrategyGridGpuResident(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  ResidentAcStrategySearchInputs resident,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  AcStrategyGpuSearchStats* stats = nullptr,
  PreparedAcStrategySearch* prepared = nullptr);

}  // namespace gjxl
