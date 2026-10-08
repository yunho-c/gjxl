// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cstddef>
#include <cstdint>

namespace gjxl::modular_internal {

enum class TreeMode : uint8_t { kGlobal, kLocal };
enum class WeightedParameters : uint8_t { kDefault, kCustom };
enum class TransformMode : uint8_t { kNone, kPresent };

// Explicit Phase 1 subset. The non-default choices name unsupported syntax;
// they carry no payload yet and are rejected before touching a destination.
// Frame roles, DC precision, anchor counts and tree policy are not header fields.
struct ModularStreamHeader {
  TreeMode tree = TreeMode::kGlobal;
  WeightedParameters weighted = WeightedParameters::kDefault;
  TransformMode transforms = TransformMode::kNone;

  [[nodiscard]] constexpr bool supported() const {
    return tree == TreeMode::kGlobal &&
           weighted == WeightedParameters::kDefault &&
           transforms == TransformMode::kNone;
  }
};

inline constexpr size_t kStreamHeaderBits = 4;

}  // namespace gjxl::modular_internal
