// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/epf.h"
#include "codec/gaborish.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

struct LoopFilterOptions {
  bool gaborish = true;
  GaborishOptions gaborish_options;
  EpfFilterOptions epf_options;

  friend bool operator==(
    const LoopFilterOptions&,
    const LoopFilterOptions&) = default;
};

/// Applies decoder loop filters in bitstream order: Gaborish, then EPF.
/// Input and output may alias; output is committed atomically.
[[nodiscard]] Status ApplyLoopFilters(
  ConstImage3FView input,
  ConstPlaneF32View inverse_sigma,
  LoopFilterOptions options,
  Image3FView output);

}  // namespace gjxl
