// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/frame.h"
#include "codec/modular/profile.h"
#include "codec/modular/geometry.h"
#include "codec/modular/transform/rct.h"
#include <limits>

namespace gjxl::modular_internal {
Status ResolveModularInput(Extent2D extent, PackedModularFormat format, ModularInputProfile *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular input profile");
  ModularInputProfile p;
  p.metadata.extent = extent;
  switch (format) {
  case PackedModularFormat::kGray8:
  case PackedModularFormat::kGray16:
    p.channel_count = 1;
    p.metadata.color = codec_internal::SourceColor::kGraySrgb;
    break;
  case PackedModularFormat::kRgb8:
  case PackedModularFormat::kRgb16:
    p.channel_count = 3;
    break;
  case PackedModularFormat::kRgba8:
  case PackedModularFormat::kRgba16:
    p.channel_count = 4;
    break;
  default:
    return Status::InvalidArgument("Invalid packed Modular format");
  }
  const bool wide = format == PackedModularFormat::kGray16 ||
                    format == PackedModularFormat::kRgb16 || format == PackedModularFormat::kRgba16;
  p.bytes_per_sample = wide ? 2 : 1;
  p.metadata.bits = static_cast<uint8_t>(8 * p.bytes_per_sample);
  p.metadata.modular_16_bit_buffer_sufficient = p.metadata.bits == 8;
  if (p.channel_count == 4)
    p.metadata.alpha = codec_internal::AlphaMetadata{p.metadata.bits};
  for (size_t c = 0; c < p.channel_count; ++c)
    p.descriptors[c] = {extent, 0, 0, c == 3 ? ChannelRole::kAlpha : ChannelRole::kColor};
  if (auto s = codec_internal::ValidateImageMetadata(p.metadata); !s.ok())
    return s;
  *out = p;
  return Status::Ok();
}
Status PackedModularImageView::Validate() const {
  ModularInputProfile profile;
  if (auto s = ResolveModularInput(extent, format, &profile); !s.ok())
    return s;
  if (byte_order != SampleByteOrder::kLittleEndian && byte_order != SampleByteOrder::kBigEndian)
    return Status::InvalidArgument("Invalid Modular sample byte order");
  ModularFrameGeometry geometry;
  if (auto s = ModularFrameGeometry::Create(extent, &geometry); !s.ok())
    return s;
  const size_t pixel_bytes = profile.channel_count * profile.bytes_per_sample;
  size_t area;
  if (!extent.try_area(&area) || uint64_t{area} > (uint64_t{1} << 40) ||
      extent.width > SIZE_MAX / pixel_bytes)
    return Status::InvalidArgument("Modular dimensions exceed the format limits");
  const size_t row = pixel_bytes * extent.width;
  if (!bytes.data() || row_stride < row || extent.height - 1 > (SIZE_MAX - row) / row_stride)
    return Status::InvalidArgument("Invalid Modular stride or backing");
  const size_t end = (extent.height - 1) * row_stride + row;
  if (end > bytes.size() || end > PTRDIFF_MAX)
    return Status::InvalidArgument("Modular backing is too short");
  return Status::Ok();
}
Status Rgb8View::Validate() const { return packed().Validate(); }
std::array<ChannelDescriptor, 3> Rgb8Channels(Extent2D extent) {
  return {ChannelDescriptor{extent}, ChannelDescriptor{extent}, ChannelDescriptor{extent}};
}
Status ModularEncoderFrame::Prepare(Rgb8View input, ModularEncoderFrame *out) {
  return Prepare(input.packed(), out);
}
Status ModularEncoderFrame::Prepare(PackedModularImageView input, ModularEncoderFrame *out) {
  return Prepare(input, 0, out);
}
Status ModularEncoderFrame::Prepare(PackedModularImageView input, uint8_t rct,
                                    ModularEncoderFrame *out) {
  ProfileScope profile_scope(ProfileStage::kInput);
  if (!out)
    return Status::InvalidArgument("Null Modular frame output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  ModularEncoderFrame frame;
  ModularInputProfile profile;
  if (auto s = ResolveModularInput(input.extent, input.format, &profile); !s.ok())
    return s;
  if (rct >= 42 || (rct && profile.channel_count < 3))
    return Status::InvalidArgument("RCT requires three color channels and a valid type");
  frame.metadata_ = profile.metadata;
  if (rct)
    frame.metadata_.modular_16_bit_buffer_sufficient = false;
  if (auto s = ModularImage::Create(profile.channels(), 0, &frame.image_); !s.ok())
    return s;
  for (size_t c = 0; c < profile.channel_count; ++c) {
    auto samples = frame.image_.samples(c);
    for (size_t y = 0; y < input.extent.height; ++y)
      for (size_t x = 0; x < input.extent.width; ++x) {
        const size_t i =
            y * input.row_stride + (profile.channel_count * x + c) * profile.bytes_per_sample;
        uint32_t value = input.bytes[i];
        if (profile.bytes_per_sample == 2)
          value = input.byte_order == SampleByteOrder::kLittleEndian
                      ? value | (uint32_t{input.bytes[i + 1]} << 8)
                      : (value << 8) | input.bytes[i + 1];
        samples[y * input.extent.width + x] = static_cast<int32_t>(value);
      }
  }
  if (rct) {
    ProfileScope transform_scope(ProfileStage::kTransforms);
    auto a = frame.image_.samples(0), b = frame.image_.samples(1), c = frame.image_.samples(2);
    for (size_t i = 0; i < a.size(); ++i) {
      const auto transformed = ForwardRct({a[i], b[i], c[i]}, rct);
      a[i] = transformed[0];
      b[i] = transformed[1];
      c[i] = transformed[2];
    }
  }
  *out = std::move(frame);
  return Status::Ok();
}
Status ModularEncoderFrame::Prepare(PackedModularImageView input, const ModularCodingPolicy &policy,
                                    ModularEncoderFrame *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular frame output");
  if (auto s = ValidateCodingPolicy(policy); !s.ok())
    return s;
  ModularInputProfile profile;
  if (auto s = ResolveModularInput(input.extent, input.format, &profile); !s.ok())
    return s;
  std::array<ChannelShape, kMaximumTransforms + 1> shapes;
  if (auto s = PlanTransforms(profile.channels(), 0, policy.transforms, &shapes); !s.ok())
    return s;
  ModularEncoderFrame frame;
  if (auto s = Prepare(input, policy.rct, &frame); !s.ok())
    return s;
  if (policy.transforms.size) {
    ProfileScope transform_scope(ProfileStage::kTransforms);
    ModularImage transformed;
    if (auto s = ForwardTransforms(frame.image_, policy.transforms, &transformed); !s.ok())
      return s;
    frame.image_ = std::move(transformed);
    frame.metadata_.modular_16_bit_buffer_sufficient = false;
  }
  *out = std::move(frame);
  return Status::Ok();
}
} // namespace gjxl::modular_internal
