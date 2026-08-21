// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>
#include <string_view>

#include "core/status.h"
#include "gpu/backend.h"

namespace gjxl {

// Creates a Metal backend using the system-default GPU.
//
// metallib_path must point to the precompiled shader library generated
// from src/gpu/metal/kernels/*.metal.
Status CreateMetalBackend(
  std::string_view metallib_path,
  std::unique_ptr<GpuBackend>* out);

}  // namespace gjxl
