// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/image.h"
#include "core/status.h"

namespace gjxl::color_transform_internal {

/// Converts an unpadded linear-sRGB image directly into padded Opsin storage.
/// The transformed right and bottom edges are extended to fill `padded_opsin`.
///
/// The source and destination must not overlap. Unlike the public transform,
/// this internal preparation primitive may partially modify its destination on
/// failure; callers must keep the destination private until success.
[[nodiscard]] Status LinearRgbToPaddedOpsin(
  ConstImage3FView linear_rgb,
  float intensity_target,
  Image3FView padded_opsin);

}  // namespace gjxl::color_transform_internal
