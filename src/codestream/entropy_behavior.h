// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

namespace gjxl {

/// User-facing control for serializer search intensity.
enum class VarDctCompressionMode {
  /// Selects a serializer behavior from the requested effort and density.
  kAutomatic,
  /// Enables the exhaustive historical GJXL serializer search.
  kMaximumCompression,
};

/// Resolved serializer behavior used for one codestream encoding.
enum class VarDctEntropyBehavior {
  /// Libjxl effort-7-like serializer behavior.
  kBalanced,
  /// Libjxl effort-9-like serializer behavior.
  kHighDensity,
  /// Exhaustive historical GJXL serializer search.
  kMaximumCompression,
};

} // namespace gjxl
