// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/frame_metadata.h"
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl::modular_internal {
inline constexpr size_t kMaximumDimension = codec_internal::kMaximumImageDimension;
inline constexpr size_t kGroupDimension = 256;
inline constexpr size_t kDcGroupDimension = 2048;
struct ModularRect {
  size_t x = 0, y = 0;
  Extent2D extent;
  bool operator==(const ModularRect &) const = default;
};
class ModularFrameGeometry {
public:
  [[nodiscard]] static Status Create(Extent2D source, ModularFrameGeometry *out);
  [[nodiscard]] Extent2D source() const { return source_; }
  [[nodiscard]] Extent2D groups() const { return groups_; }
  [[nodiscard]] Extent2D dc_groups() const { return dc_groups_; }
  [[nodiscard]] size_t group_count() const { return groups_.width * groups_.height; }
  [[nodiscard]] size_t dc_group_count() const {
    return dc_groups_.width * dc_groups_.height;
  }
  [[nodiscard]] size_t section_count() const {
    return group_count() == 1 ? 1 : 2 + dc_group_count() + group_count();
  }
  // Full group rectangle; channel clipping happens after shifting, like libjxl.
  [[nodiscard]] Status GroupRect(size_t group, bool dc, ModularRect *out) const;
  bool operator==(const ModularFrameGeometry &) const = default;

private:
  Extent2D source_, groups_, dc_groups_;
};
} // namespace gjxl::modular_internal
