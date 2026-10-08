// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/transform/operations.h"
#include <algorithm>

namespace gjxl::modular_internal {
namespace {
Status Invalid() { return Status::InvalidArgument("Invalid or unsupported Modular transform"); }
Status ShapeStep(const ChannelShape &in, const TransformDescriptor &t, ChannelShape *out) {
  if (t.begin < in.metadata || t.begin >= in.count || t.count > in.count - t.begin)
    return Invalid();
  ChannelShape p = in;
  const size_t end = t.begin + t.count;
  if (t.kind == TransformKind::kPalette) {
    const auto first = in.descriptors[t.begin];
    for (size_t c = t.begin; c < end; ++c) {
      const auto ch = in.descriptors[c];
      if (ch.extent != first.extent || ch.hshift != first.hshift || ch.vshift != first.vshift)
        return Invalid();
    }
    p.count = in.count + 2 - t.count;
    if (p.count > kMaximumTransformedChannels)
      return Invalid();
    p.metadata++;
    p.descriptors[0] = {{t.colors, t.count}, 0, 0, ChannelRole::kMetadata};
    size_t dest = 1;
    for (size_t c = 0; c < in.count; ++c)
      if (c <= t.begin || c >= end)
        p.descriptors[dest++] = in.descriptors[c];
  } else {
    if (in.count + t.count > kMaximumTransformedChannels)
      return Invalid();
    const size_t offset = t.in_place ? end : in.count;
    p.count += t.count;
    for (size_t c = in.count; c-- > offset;)
      p.descriptors[c + t.count] = in.descriptors[c];
    for (size_t c = t.begin; c < end; ++c) {
      auto avg = in.descriptors[c], residual = avg;
      auto &length = t.horizontal ? avg.extent.width : avg.extent.height;
      auto &shift = t.horizontal ? avg.hshift : avg.vshift;
      if (length < 2 || shift >= 8)
        return Invalid();
      const size_t tail = length / 2;
      length -= tail;
      ++shift;
      residual = avg;
      (t.horizontal ? residual.extent.width : residual.extent.height) = tail;
      p.descriptors[c] = avg;
      p.descriptors[offset + c - t.begin] = residual;
    }
  }
  if (auto s = ValidateChannelDescriptors(p.channels(), p.metadata); !s.ok())
    return s;
  *out = p;
  return Status::Ok();
}
void Copy(const ModularImage &in, size_t src, ModularImage *out, size_t dst) {
  auto v = in.view(src);
  auto samples = out->samples(dst);
  for (size_t y = 0; y < v.descriptor.extent.height; ++y)
    std::copy_n(v.Row(y), v.descriptor.extent.width,
                samples.data() + y * v.descriptor.extent.width);
}
Status Run(const ModularImage &in, const TransformSequence &seq,
           const std::array<ChannelShape, kMaximumTransforms + 1> &shapes, bool inverse,
           ModularImage *out) {
  if (!out)
    return Invalid();
  const auto &expected = shapes[inverse ? seq.size : 0];
  ChannelShape actual;
  if (auto s = DescribeImage(in, &actual); !s.ok())
    return s;
  if (actual.count != expected.count || actual.metadata != expected.metadata ||
      !std::equal(actual.channels().begin(), actual.channels().end(), expected.channels().begin()))
    return Invalid();
  ModularImage current;
  for (size_t step = 0; step < std::max(size_t{1}, seq.size); ++step) {
    const size_t i = inverse ? seq.size - step : step;
    const auto &target = shapes[seq.size ? (inverse ? i - 1 : i + 1) : 0];
    ModularImage next;
    if (auto s = ModularImage::Create(target.channels(), target.metadata, &next); !s.ok())
      return s;
    const auto &source = step ? current : in;
    if (!seq.size) {
      for (size_t c = 0; c < source.channel_count(); ++c)
        Copy(source, c, &next, c);
    } else {
      const auto &t = seq.entries[inverse ? i - 1 : i];
      auto s = t.kind == TransformKind::kPalette ? ApplyPalette(source, t, inverse, &next)
                                                 : ApplySqueeze(source, t, inverse, &next);
      if (!s.ok())
        return s;
    }
    current = std::move(next);
  }
  *out = std::move(current);
  return Status::Ok();
}
} // namespace
Status ValidateTransforms(const TransformSequence &s) {
  if (s.size > kMaximumTransforms)
    return Invalid();
  for (size_t i = 0; i < s.size; ++i) {
    const auto &t = s.entries[i];
    if (!t.count || t.begin >= kMaximumTransformedChannels)
      return Invalid();
    if (t.kind == TransformKind::kPalette) {
      if (t.count > 4 || !t.colors || t.colors > kMaximumPaletteColors)
        return Invalid();
    } else if (t.kind == TransformKind::kSqueeze) {
      if (t.count > 19)
        return Invalid();
    } else
      return Invalid();
  }
  return Status::Ok();
}
Status DescribeImage(const ModularImage &image, ChannelShape *out) {
  if (!out || image.channel_count() > kMaximumTransformedChannels)
    return Invalid();
  ChannelShape shape;
  shape.count = image.channel_count();
  shape.metadata = image.metadata_channels();
  for (size_t c = 0; c < shape.count; ++c)
    shape.descriptors[c] = image.view(c).descriptor;
  if (auto s = ValidateChannelDescriptors(shape.channels(), shape.metadata); !s.ok())
    return s;
  *out = shape;
  return Status::Ok();
}
Status PlanTransforms(std::span<const ChannelDescriptor> source, size_t metadata,
                      const TransformSequence &sequence,
                      std::array<ChannelShape, kMaximumTransforms + 1> *out) {
  if (!out || source.size() > kMaximumTransformedChannels)
    return Invalid();
  if (auto s = ValidateTransforms(sequence); !s.ok())
    return s;
  if (auto s = ValidateChannelDescriptors(source, metadata); !s.ok())
    return s;
  std::array<ChannelShape, kMaximumTransforms + 1> shapes{};
  shapes[0].count = source.size();
  shapes[0].metadata = metadata;
  std::copy(source.begin(), source.end(), shapes[0].descriptors.begin());
  for (size_t i = 0; i < sequence.size; ++i)
    if (auto s = ShapeStep(shapes[i], sequence.entries[i], &shapes[i + 1]); !s.ok())
      return s;
  *out = shapes;
  return Status::Ok();
}
Status ForwardTransforms(const ModularImage &input, const TransformSequence &sequence,
                         ModularImage *out) {
  ChannelShape original;
  if (auto s = DescribeImage(input, &original); !s.ok())
    return s;
  std::array<ChannelShape, kMaximumTransforms + 1> shapes;
  if (auto s = PlanTransforms(original.channels(), original.metadata, sequence, &shapes); !s.ok())
    return s;
  return Run(input, sequence, shapes, false, out);
}
Status InverseTransforms(const ModularImage &input, std::span<const ChannelDescriptor> original,
                         size_t metadata, const TransformSequence &sequence, ModularImage *out) {
  std::array<ChannelShape, kMaximumTransforms + 1> shapes;
  if (auto s = PlanTransforms(original, metadata, sequence, &shapes); !s.ok())
    return s;
  return Run(input, sequence, shapes, true, out);
}
} // namespace gjxl::modular_internal
