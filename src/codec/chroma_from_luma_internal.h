// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/chroma_from_luma.h"

namespace gjxl::chroma_from_luma_internal {

/// Copies validated signed-byte CfL maps into their owning codec form.
[[nodiscard]] Status CreateColorCorrelationMap(
  ConstPlaneI8View y_to_x,
  ConstPlaneI8View y_to_b,
  ColorCorrelationMap* out);

}  // namespace gjxl::chroma_from_luma_internal
