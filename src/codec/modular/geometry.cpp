// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/geometry.h"
#include <limits>

namespace gjxl::modular_internal {
Status ModularFrameGeometry::Create(Extent2D source, ModularFrameGeometry *out) {
  if (!out || source.empty() || source.width > kMaximumDimension ||
      source.height > kMaximumDimension)
    return Status::InvalidArgument("Invalid Modular frame extent");
  ModularFrameGeometry g;
  g.source_ = source;
  g.groups_ = source.ceil_div(kGroupDimension);
  g.dc_groups_ = source.ceil_div(kDcGroupDimension);
  size_t groups, dc_groups;
  if (!g.groups_.try_area(&groups) || !g.dc_groups_.try_area(&dc_groups) ||
      dc_groups > (std::numeric_limits<size_t>::max() - 18) / 3 ||
      groups > std::numeric_limits<size_t>::max() - 18 - 3 * dc_groups)
    return Status::InvalidArgument("Modular group counts overflow");
  *out = g;
  return Status::Ok();
}
Status ModularFrameGeometry::GroupRect(size_t group, bool dc, ModularRect *out) const {
  const auto grid = dc ? dc_groups_ : groups_;
  const size_t dim = dc ? kDcGroupDimension : kGroupDimension;
  if (!out || source_.empty() || group >= grid.width * grid.height)
    return Status::InvalidArgument("Invalid Modular group index");
  *out = {(group % grid.width) * dim, (group / grid.width) * dim, {dim, dim}};
  return Status::Ok();
}
} // namespace gjxl::modular_internal
