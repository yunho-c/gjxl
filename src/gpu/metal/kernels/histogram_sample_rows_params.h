// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "modular_adaptive_params.h"
namespace gjxl_histogram_rows {
using gjxl_modular_tokens::Address;
using gjxl_modular_tokens::Word;
#ifdef __METAL_VERSION__
using Byte = uchar;
using Short = ushort;
#else
using Byte = uint8_t;
using Short = uint16_t;
#endif
struct Slice {
  gjxl_modular_adaptive::Slice source;
  Word first_row, row_count, channel, group;
};
struct Mapping {
  Word properties[16], predictors[14];
  Short statics[2][1023];
  Byte dynamics[16][1023];
};
struct Params {
  Address slices_offset, positions_offset, status_offset;
  Word rows, slices, statics, dynamics, predictors, reserved;
};
} // namespace gjxl_histogram_rows
