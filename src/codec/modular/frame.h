// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/frame_metadata.h"
#include "codec/modular/image.h"
#include <array>

namespace gjxl::modular_internal {
enum class PackedModularFormat : uint8_t { kGray8, kRgb8, kRgba8, kGray16, kRgb16, kRgba16 };
enum class SampleByteOrder : uint8_t { kLittleEndian, kBigEndian };
// RGBA means unassociated alpha at the color depth. Stride is in bytes;
// offsets use a subspan and no padding is required after the last sample.
// Byte order is explicit for 16-bit samples and ignored for 8-bit samples.
struct PackedModularImageView {
  std::span<const uint8_t> bytes;
  Extent2D extent;
  size_t row_stride = 0;
  PackedModularFormat format = PackedModularFormat::kRgb8;
  SampleByteOrder byte_order = SampleByteOrder::kLittleEndian;
  [[nodiscard]] Status Validate() const;
};
struct ModularInputProfile {
  codec_internal::ImageMetadata metadata;
  std::array<ChannelDescriptor, 4> descriptors;
  size_t channel_count = 0;
  size_t bytes_per_sample = 0;
  [[nodiscard]] std::span<const ChannelDescriptor> channels() const {
    return std::span(descriptors).first(channel_count);
  }
};
// Allocation-free resolution shared by validation, preparation and planning.
[[nodiscard]] Status ResolveModularInput(Extent2D extent, PackedModularFormat format,
                                         ModularInputProfile *out);
struct Rgb8View {
  std::span<const uint8_t> bytes;
  Extent2D extent;
  size_t row_stride = 0;
  [[nodiscard]] Status Validate() const;
  [[nodiscard]] PackedModularImageView packed() const { return {bytes, extent, row_stride}; }
};
[[nodiscard]] std::array<ChannelDescriptor, 3> Rgb8Channels(Extent2D extent);
// Completed identity frame. No writer/model/search state or mutable plane API.
class ModularEncoderFrame {
public:
  [[nodiscard]] static Status Prepare(Rgb8View input, ModularEncoderFrame *out);
  [[nodiscard]] static Status Prepare(PackedModularImageView input, ModularEncoderFrame *out);
  [[nodiscard]] const codec_internal::ImageMetadata &metadata() const { return metadata_; }
  [[nodiscard]] const ModularImage &image() const { return image_; }

private:
  codec_internal::ImageMetadata metadata_;
  ModularImage image_;
};
} // namespace gjxl::modular_internal
