// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cstddef>

namespace gjxl::cuda_internal {
// Enforced at every instrumented launch site, including diagnostic variants.
inline constexpr size_t kCudaKernelProfileIdLength = 78;
} // namespace gjxl::cuda_internal
