// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <cstdint>
#include <optional>
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl::codec_internal {
inline constexpr size_t kMaximumImageDimension = 0x3fffffffu;
// Private serialization values; no writer, entropy or VarDCT frame dependency.
enum class SampleFormat : uint8_t { kUnsigned, kFloat };
enum class SourceColor : uint8_t { kSrgb, kGraySrgb, kLinearSrgb };
enum class FrameEncoding : uint8_t { kVarDct, kModular };
struct AlphaMetadata {
  uint8_t bits = 8;
  bool associated = false;
  uint8_t dimension_shift = 0;
  bool operator==(const AlphaMetadata &) const = default;
};
struct ImageMetadata {
  Extent2D extent;
  SampleFormat sample_format = SampleFormat::kUnsigned;
  uint8_t bits = 8;
  SourceColor color = SourceColor::kSrgb;
  bool xyb = false;
  std::optional<AlphaMetadata> alpha;
  bool operator==(const ImageMetadata &) const = default;
};
struct VarDctFrameFields {
  uint8_t x_qm_scale = 2, b_qm_scale = 2;
  bool adaptive_dc_smoothing = false;
  bool gaborish = true;
  bool operator==(const VarDctFrameFields &) const = default;
};
struct FrameMetadata {
  FrameEncoding encoding = FrameEncoding::kModular;
  // Ignored fields would conceal mistakes: Modular requires this default value.
  VarDctFrameFields vardct;
};
[[nodiscard]] Status ValidateImageMetadata(const ImageMetadata &metadata);
[[nodiscard]] Status ValidateFrameMetadata(const ImageMetadata &image,
                                           const FrameMetadata &frame);
} // namespace gjxl::codec_internal
