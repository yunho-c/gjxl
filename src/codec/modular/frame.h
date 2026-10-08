// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/frame_metadata.h"
#include "codec/modular/image.h"
#include <array>

namespace gjxl::modular_internal {
struct Rgb8View {
  std::span<const uint8_t> bytes;
  Extent2D extent;
  size_t row_stride = 0;
  [[nodiscard]] Status Validate() const;
};
[[nodiscard]] std::array<ChannelDescriptor, 3> Rgb8Channels(Extent2D extent);
// Completed identity frame. No writer/model/search state or mutable plane API.
class ModularEncoderFrame {
public:
  [[nodiscard]] static Status Prepare(Rgb8View input, ModularEncoderFrame *out);
  [[nodiscard]] const codec_internal::ImageMetadata &metadata() const { return metadata_; }
  [[nodiscard]] const ModularImage &image() const { return image_; }

private:
  codec_internal::ImageMetadata metadata_;
  ModularImage image_;
};
} // namespace gjxl::modular_internal
