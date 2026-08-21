// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

namespace gjxl::test {

// Uses the orthonormal DCT-II convention and row-major order.
// Calculations are performed in double precision.
void ReferenceForwardDct8(
  const float* input,
  float* output,
  size_t block_count);

void ReferenceInverseDct8(
  const float* input,
  float* output,
  size_t block_count);

}  // namespace gjxl::test
