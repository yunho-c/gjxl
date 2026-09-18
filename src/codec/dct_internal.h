// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once
#include <cstddef>
#include <span>

namespace gjxl::dct_internal {
// Read-only basis used by the scalar inverse. Exact GPU evaluation uploads
// these host-library cosine values; unsupported lengths return an empty span.
[[nodiscard]] std::span<const double> InverseBasis(size_t length);
}  // namespace gjxl::dct_internal
