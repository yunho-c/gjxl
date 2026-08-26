// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>

#include "core/image.h"
#include "core/status.h"

namespace gjxl {

struct GaborishOptions {
  std::array<float, 3> weight1 = {
    1.1f * 0.104699568f,
    1.1f * 0.104699568f,
    1.1f * 0.104699568f,
  };
  std::array<float, 3> weight2 = {
    1.1f * 0.055680538f,
    1.1f * 0.055680538f,
    1.1f * 0.055680538f,
  };

  friend bool operator==(
    const GaborishOptions&,
    const GaborishOptions&) = default;
};

/// Applies libjxl's encoder-side approximate inverse Gaborish filter.
/// Input and output may alias; output is committed only after all channels
/// have been computed successfully.
[[nodiscard]] Status ApplyGaborishInverse(
  ConstImage3FView input,
  std::array<float, 3> multipliers,
  Image3FView output);

/// Applies the decoder-side normalized 3x3 Gaborish convolution.
/// Input and output may alias; output is committed atomically.
[[nodiscard]] Status ApplyGaborish(
  ConstImage3FView input,
  GaborishOptions options,
  Image3FView output);

}  // namespace gjxl
