// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#ifndef __METAL_VERSION__
#include <cstdint>
#endif
namespace gjxl_aq_dispatch {
#ifdef __METAL_VERSION__
using Word = uint;
#else
using Word = uint32_t;
#endif
// Shared host/shader ABI. Parameter payloads retain the existing kernel ABIs.
enum Dispatch : Word {
  kTransforms,
  kDct,
  kAdjustedScalar,
  kAdjustedParallel,
  kQuantField,
  kPopulation,
  kLlf,
  kDispatchCount
};
struct Record {
  Word reconstruction[37];
  Word completed_reconstruction[37];
  Word forward[4];
  Word inverse[4];
  Word adjustment[6];
  Word population[4];
  Word block_reduction[10];
  Word butteraugli[13];
  Word groups[kDispatchCount][3];
};
} // namespace gjxl_aq_dispatch
