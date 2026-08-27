// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <filesystem>

#include "core/image_buffer.h"
#include "core/status.h"

namespace gjxl::io {

/// Reads a three-channel PFM as top-down planar linear RGB.
/// Failure leaves `image` unchanged.
[[nodiscard]] Status ReadPfm(
  const std::filesystem::path& path,
  Image3FBuffer* image);

}  // namespace gjxl::io
