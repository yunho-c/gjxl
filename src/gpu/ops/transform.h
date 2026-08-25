// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include "core/ac_strategy.h"
#include "gpu/buffer.h"

namespace gjxl {

// A packed batch of complete transforms using one JPEG XL AC strategy.
//
// Forward transforms read row-major pixels and write coefficients in the
// strategy's libjxl-compatible layout. Inverse transforms reverse that flow.
// Each input/output item contains AcStrategyInfo::coefficient_count() floats.
struct TransformBatch {
  AcStrategyType strategy = AcStrategyType::kDct8;
  const DeviceBuffer* input = nullptr;
  DeviceBuffer* output = nullptr;
  size_t transform_count = 0;
};

}  // namespace gjxl
