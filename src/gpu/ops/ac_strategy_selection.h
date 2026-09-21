// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "gpu/ops/ac_strategy.h"

namespace gjxl {

/// Ordinary-bank selection output. The first block_extent.area() bytes are
/// row-major (strategy << 1) | is_anchor cells. The following ceil(width/8) *
/// ceil(height/8) bytes are row-major tile error flags, zero on success. A
/// nonzero flag invalidates the entire result. The output may reuse the
/// candidate evaluator's rate scratch because selection executes afterward.
struct AcStrategyDeviceSelection {
  Extent2D block_extent;
  DeviceBuffer *output = nullptr;
  size_t offset_bytes = 0;
};

/// Optional policy-preserving device selector. Batches must contain the seven
/// ordinary candidate families in kCandidateStages order; candidates within
/// each family must use the shared tile-row-major, then anchor-row-major
/// enumeration. Dense DCT32 placement is not supported. The backend validates
/// geometry/counts, while the caller owns this enumeration contract.
class GpuAcStrategySelection {
public:
  virtual ~GpuAcStrategySelection() = default;

  /// Scores and selects in one submission without a host wait. Every input,
  /// scratch and output buffer must outlive the returned submission. The caller
  /// must check tile errors after completion before publishing a host result;
  /// dependent device operations must propagate those errors to their final
  /// completion boundary. Failure leaves submission empty.
  virtual Status EvaluateAndSelectAcStrategyCandidateBatches(
      std::span<const AcStrategyCandidateBatch> batches,
      AcStrategyDeviceSelection selection,
      std::unique_ptr<GpuSubmission> *submission) = 0;
};

} // namespace gjxl
