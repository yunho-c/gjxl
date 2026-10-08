// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/geometry.h"
#include "codec/modular/image.h"

namespace gjxl::modular_internal {
enum class StreamRole : uint8_t { kGlobal, kDcGroup, kGroup };
struct ChannelSlice {
  size_t channel = 0;
  ModularRect rect;
  bool operator==(const ChannelSlice &) const = default;
};
struct PlannedStream {
  StreamRole role = StreamRole::kGlobal;
  size_t group = 0;
  size_t id = 0;
  size_t section = 0;
  size_t slice_begin = 0, slice_count = 0;
  bool operator==(const PlannedStream &) const = default;
};
struct ModularStreamPlan {
  ModularFrameGeometry geometry;
  resource_budget_internal::ManagedVector<PlannedStream> streams;
  resource_budget_internal::ManagedVector<ChannelSlice> slices;
  // Multiple contributions share section 0 for single-group frames. Otherwise
  // AC global occupies this section even though it has no Modular stream.
  size_t ac_global_section = 0;
};
struct ModularStreamStoragePlan {
  size_t streams = 0, maximum_slices = 0;
  resource_budget_internal::HostStorageBound owned;
  bool operator==(const ModularStreamStoragePlan &) const = default;
};
// Canonical IDs reserve VarDCT DC, metadata and all 17 quant-table slots.
[[nodiscard]] Status ModularStreamId(const ModularFrameGeometry &geometry,
                                     StreamRole role, size_t group, size_t *out);
// Bounds fresh plan backing only; add old output backing during replacement.
[[nodiscard]] Status ComputeModularStreamStoragePlan(
    const ModularFrameGeometry &geometry, std::span<const ChannelDescriptor> channels,
    size_t metadata_channels, ModularStreamStoragePlan *out);
// No sample ownership: slices refer to stable channel indices/rectangles.
[[nodiscard]] Status BuildModularStreamPlan(const ModularFrameGeometry &geometry,
                                            std::span<const ChannelDescriptor> channels,
                                            size_t metadata_channels,
                                            ModularStreamPlan *out);
[[nodiscard]] Status BorrowChannelSlice(const ModularChannelView &channel,
                                        ModularRect rect, ModularChannelView *out);
} // namespace gjxl::modular_internal
