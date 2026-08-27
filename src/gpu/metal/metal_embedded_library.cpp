// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_embedded_library_internal.h"

#include <cstddef>

#include "gjxl_embedded_metallib.inc"

namespace gjxl::metal_internal {

std::span<const uint8_t> EmbeddedMetalLibrary() noexcept {
  return {gjxl_embedded_metallib,
          static_cast<size_t>(gjxl_embedded_metallib_len)};
}

}  // namespace gjxl::metal_internal
