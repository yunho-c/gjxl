// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <span>

namespace gjxl::metal_internal {

/// Returns the build-generated production Metal shader library.
[[nodiscard]] std::span<const uint8_t> EmbeddedMetalLibrary() noexcept;

}  // namespace gjxl::metal_internal
