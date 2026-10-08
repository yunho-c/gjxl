// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/frame.h"
#include "codec/modular/geometry.h"
#include <limits>

namespace gjxl::modular_internal {
Status Rgb8View::Validate() const {
  ModularFrameGeometry geometry;
  if (auto s = ModularFrameGeometry::Create(extent, &geometry); !s.ok())
    return s;
  size_t area;
  if (!extent.try_area(&area) || uint64_t{area} > (uint64_t{1} << 40) ||
      extent.width > SIZE_MAX / 3)
    return Status::InvalidArgument("RGB8 dimensions exceed the format limits");
  const size_t row = 3 * extent.width;
  if (!bytes.data() || row_stride < row || extent.height - 1 > (SIZE_MAX - row) / row_stride)
    return Status::InvalidArgument("Invalid RGB8 stride or backing");
  const size_t end = (extent.height - 1) * row_stride + row;
  if (end > bytes.size() || end > PTRDIFF_MAX)
    return Status::InvalidArgument("RGB8 backing is too short");
  return Status::Ok();
}
std::array<ChannelDescriptor, 3> Rgb8Channels(Extent2D extent) {
  return {ChannelDescriptor{extent}, ChannelDescriptor{extent}, ChannelDescriptor{extent}};
}
Status ModularEncoderFrame::Prepare(Rgb8View input, ModularEncoderFrame *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular frame output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  ModularEncoderFrame frame;
  frame.metadata_.extent = input.extent;
  frame.metadata_.modular_16_bit_buffer_sufficient = true;
  if (auto s = ModularImage::Create(Rgb8Channels(input.extent), 0, &frame.image_); !s.ok())
    return s;
  for (size_t c = 0; c < 3; ++c) {
    auto samples = frame.image_.samples(c);
    for (size_t y = 0; y < input.extent.height; ++y)
      for (size_t x = 0; x < input.extent.width; ++x)
        samples[y * input.extent.width + x] = input.bytes[y * input.row_stride + 3 * x + c];
  }
  *out = std::move(frame);
  return Status::Ok();
}
} // namespace gjxl::modular_internal
