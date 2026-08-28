// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/convolution.h"

namespace gjxl::gaborish_internal {

[[nodiscard]] Symmetric5Weights GaborishInverseWeights(
  float multiplier) noexcept;

}  // namespace gjxl::gaborish_internal
