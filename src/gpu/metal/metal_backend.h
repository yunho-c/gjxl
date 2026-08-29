// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "core/status.h"
#include "gpu/backend.h"

namespace gjxl {

// Built-in DCT kernel implementations. Availability is strategy-dependent.
enum class MetalDctImplementation {
  kScalarMatmul,
  kSimdgroupMatmul,
  kFactoredRadix2,
};

struct MetalBackendOptions {
  MetalDctImplementation forward_dct8 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct8 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation forward_dct16x16 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct16x16 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation forward_dct32x32 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct32x32 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation forward_dct16x8 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct16x8 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation forward_dct8x16 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct8x16 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation forward_dct32x16 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct32x16 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation forward_dct16x32 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct16x32 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation forward_dct64x32 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct64x32 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation forward_dct32x64 =
    MetalDctImplementation::kScalarMatmul;

  MetalDctImplementation inverse_dct32x64 =
    MetalDctImplementation::kScalarMatmul;

  // Deterministic failure injection used by real-device backend tests.
  bool test_fail_submission = false;
  bool test_fail_completion = false;
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

/// Creates a Metal backend from an in-memory precompiled shader library.
Status CreateMetalBackend(
  std::span<const uint8_t> metallib,
  std::unique_ptr<GpuBackend>* out);

Status CreateMetalBackend(
  std::span<const uint8_t> metallib,
  const MetalBackendOptions& options,
  std::unique_ptr<GpuBackend>* out);

/// Creates the production backend from the build-embedded shader library.
Status CreateEmbeddedMetalBackend(
  const MetalBackendOptions& options,
  std::unique_ptr<GpuBackend>* out);

/// Injects a failure into the next Metal compute submission only.
[[nodiscard]] Status ArmNextMetalSubmissionFailureForTest(
  GpuBackend& backend,
  bool fail_submission,
  bool fail_completion);

}  // namespace gjxl
