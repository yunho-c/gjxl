// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/transform/transform.h"
namespace gjxl::modular_internal {
// Composition-only helpers: caller validates descriptors and owns an unpublished
// destination of the planned shape. Never call on aliased images.
[[nodiscard]] Status ApplyPalette(const ModularImage &input, const TransformDescriptor &descriptor,
                                  bool inverse, ModularImage *output);
[[nodiscard]] Status ApplySqueeze(const ModularImage &input, const TransformDescriptor &descriptor,
                                  bool inverse, ModularImage *output);
} // namespace gjxl::modular_internal
