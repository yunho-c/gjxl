// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>

namespace gjxl::butteraugli_test {

struct OracleExtent {
  size_t width = 0;
  size_t height = 0;

  friend constexpr bool operator==(OracleExtent, OracleExtent) = default;
};

struct ConstOraclePlane {
  const float *data = nullptr;
  OracleExtent extent;
  size_t stride = 0;
};

struct OraclePlane {
  float *data = nullptr;
  OracleExtent extent;
  size_t stride = 0;
};

struct ConstOracleImage3 {
  std::array<ConstOraclePlane, 3> plane;
};

struct OracleOptions {
  float hf_asymmetry = 1.0f;
  float x_multiplier = 1.0f;
  float intensity_target = 80.0f;
};

} // namespace gjxl::butteraugli_test
