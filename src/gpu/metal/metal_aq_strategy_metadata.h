// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "gpu/scratch.h"
#include <array>
#include <memory>

namespace MTL {
class ComputeCommandEncoder;
}
namespace gjxl::metal_internal {
class MetalBackend;

// Device binding order (selection is binding zero; these follow at one).
enum AqMetadataPlane : size_t {
  kMetadataRanks,
  kMetadataChunks,
  kMetadataTileCounts,
  kMetadataFamilies,
  kMetadataControl,
  kMetadataStrategies,
  kMetadataAnchors,
  kMetadataColorRecords,
  kMetadataColorOffsets,
  kMetadataDestinations,
  kMetadataPlaneCount
};
inline constexpr size_t kMetadataScratchPlaneCount = 5;

struct AqStrategyMetadataStoragePlan {
  size_t capacity_bytes = 0;
  std::array<DevicePlaneLayout, kMetadataScratchPlaneCount> scratch;
  // Flat uint32 element counts, including existing AQ output destinations.
  std::array<size_t, kMetadataPlaneCount> elements{};
  size_t selection_bytes = 0;
};

// Allocation-free checked plan. Existing AQ/output arenas supply planes 5–9.
[[nodiscard]] Status
ComputeAqStrategyMetadataStoragePlan(Extent2D blocks,
                                     AqStrategyMetadataStoragePlan *out);

struct AqStrategyMetadataDescriptor {
  Extent2D blocks;
  // Native selection's byte cells followed by per-tile error bytes.
  ConstDevicePlaneView selection;
  // Flat contiguous I32 views, all disjoint from each other and selection.
  std::array<DevicePlaneView, kMetadataPlaneCount> planes;
};

// Families: seven five-word records in kSupportedAqStrategies order:
// strategy, anchor offset, count, coefficient offset, coefficient count.
// Control: error, total anchors, order-population mask, coefficient values.
// A nonzero error invalidates all results. On failure counts/offsets are zero
// and the expanded strategy map is safe DCT8. Other unused outputs stay
// untouched. Every referenced buffer must outlive the returned
// submission/consumer work.
class MetalAqStrategyMetadata {
public:
  [[nodiscard]] static Status
  Validate(GpuBackend &backend, const AqStrategyMetadataDescriptor &descriptor);
  [[nodiscard]] static Status
  Submit(GpuBackend &backend, const AqStrategyMetadataDescriptor &descriptor,
         std::unique_ptr<GpuSubmission> *submission);
  // Internal recording seam; Validate must precede the enclosing submission.
  static void Encode(MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
                     const AqStrategyMetadataDescriptor &descriptor);

private:
  static void EncodeSubmission(MetalBackend &, MTL::ComputeCommandEncoder *,
                               const void *);
};
} // namespace gjxl::metal_internal
