// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/image.h"
#include <array>

namespace gjxl::modular_internal {
// Deliberately bounded CPU subset. RCT remains the optional first transform.
inline constexpr size_t kMaximumTransforms = 8;
inline constexpr size_t kMaximumTransformedChannels = 64;
inline constexpr size_t kMaximumPaletteColors = 256;
enum class TransformKind : uint8_t { kPalette, kSqueeze };
struct TransformDescriptor {
  TransformKind kind = TransformKind::kPalette;
  uint8_t begin = 0, count = 1;
  uint16_t colors = 1; // Exact palette capacity; unused entries are zero.
  bool horizontal = true, in_place = true;
  bool operator==(const TransformDescriptor &) const = default;
};
struct TransformSequence {
  std::array<TransformDescriptor, kMaximumTransforms> entries{};
  size_t size = 0;
  bool operator==(const TransformSequence &) const = default;
};
struct ChannelShape {
  std::array<ChannelDescriptor, kMaximumTransformedChannels> descriptors{};
  size_t count = 0, metadata = 0;
  std::span<const ChannelDescriptor> channels() const {
    return std::span(descriptors).first(count);
  }
};
[[nodiscard]] Status ValidateTransforms(const TransformSequence &sequence);
[[nodiscard]] Status DescribeImage(const ModularImage &image, ChannelShape *out);
// Allocation-free. Each shape corresponds to the image before/after one step.
[[nodiscard]] Status PlanTransforms(std::span<const ChannelDescriptor> source, size_t metadata,
                                    const TransformSequence &sequence,
                                    std::array<ChannelShape, kMaximumTransforms + 1> *out);
// Borrow input, publish only on success. Inverse requires original descriptors
// to restore roles as well as dimensions. Neither operation mutates input.
[[nodiscard]] Status ForwardTransforms(const ModularImage &input, const TransformSequence &sequence,
                                       ModularImage *out);
[[nodiscard]] Status InverseTransforms(const ModularImage &input,
                                       std::span<const ChannelDescriptor> original, size_t metadata,
                                       const TransformSequence &sequence, ModularImage *out);
} // namespace gjxl::modular_internal
