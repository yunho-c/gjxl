// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/buffer.h"
#include "gpu/image.h"
#include "gpu/submission.h"

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

  // Optional checked device views used by prepared resident frontends. When
  // the first opsin plane is non-null, these replace the legacy packed opsin
  // and pixel-mask buffers above. A non-null quant field also makes the
  // strategy-aware quant norm a device-side decision.
  ConstDeviceImage3View resident_opsin;
  ConstDevicePlaneView resident_pixel_mask;
  ConstDevicePlaneView resident_quant_field;

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

/// Optional coherent AC-strategy candidate operation implemented by a backend.
/// Candidate selection and search traversal remain on the CPU.
class GpuAcStrategyEvaluation {
public:
  virtual ~GpuAcStrategyEvaluation() = default;

  /// Enqueues several candidate batches in one submission. Each batch is
  /// internally same-strategy; batches may select different strategies,
  /// execute in span order, and reuse scratch buffers. A successful non-empty
  /// sequence returns a non-null caller-owned submission. An empty sequence
  /// succeeds with a null submission.
  virtual Status EvaluateAcStrategyCandidateBatches(
    std::span<const AcStrategyCandidateBatch> batches,
    std::unique_ptr<GpuSubmission>* submission) = 0;
};

[[nodiscard]] inline GpuAcStrategyEvaluation* QueryGpuAcStrategyEvaluation(
  GpuBackend& backend) noexcept {

  return dynamic_cast<GpuAcStrategyEvaluation*>(&backend);
}

[[nodiscard]] inline Status EvaluateAcStrategyCandidateBatches(
  GpuBackend& backend,
  std::span<const AcStrategyCandidateBatch> batches,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr) {
    return Status::InvalidArgument(
      "AC-strategy submission output pointer is null");
  }
  submission->reset();
  GpuAcStrategyEvaluation* capability =
    QueryGpuAcStrategyEvaluation(backend);
  if (capability == nullptr) {
    return Status::Unavailable(
      "GPU backend does not provide AC-strategy evaluation");
  }
  Status status = capability->EvaluateAcStrategyCandidateBatches(
    batches, submission);
  if (!status.ok()) {
    submission->reset();
    return status;
  }
  bool has_work = false;
  for (const AcStrategyCandidateBatch& batch : batches) {
    has_work = has_work || batch.candidate_count != 0;
  }
  if (has_work != (*submission != nullptr)) {
    submission->reset();
    return Status::Internal(
      "GPU AC-strategy capability returned an invalid submission");
  }
  return Status::Ok();
}

[[nodiscard]] inline Status EvaluateAcStrategyCandidates(
  GpuBackend& backend,
  const AcStrategyCandidateBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {

  return EvaluateAcStrategyCandidateBatches(
    backend,
    std::span<const AcStrategyCandidateBatch>(&batch, 1),
    submission);
}

}  // namespace gjxl
