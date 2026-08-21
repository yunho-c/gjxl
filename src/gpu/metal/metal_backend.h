// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>
#include <string_view>

#include "core/status.h"
#include "gpu/backend.h"

namespace gjxl {

// Built-in DCT8 kernel pairs selectable when the Metal backend is created.
enum class MetalDct8Implementation {
  kScalarMatmul,
  kSimdgroupMatmul,
};

struct MetalBackendOptions {
  MetalDct8Implementation forward_dct8 =
    MetalDct8Implementation::kScalarMatmul;

  MetalDct8Implementation inverse_dct8 =
    MetalDct8Implementation::kScalarMatmul;
};

// Creates a Metal backend using the system-default GPU.
//
// metallib_path must point to the precompiled shader library generated
// from src/gpu/metal/kernels/*.metal.
// The overload without options uses MetalBackendOptions defaults.
// An explicitly selected implementation is never silently replaced.
Status CreateMetalBackend(
  std::string_view metallib_path,
  std::unique_ptr<GpuBackend>* out);

Status CreateMetalBackend(
  std::string_view metallib_path,
  const MetalBackendOptions& options,
  std::unique_ptr<GpuBackend>* out);

}  // namespace gjxl
