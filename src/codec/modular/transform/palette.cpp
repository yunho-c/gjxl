// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/transform/operations.h"
#include <algorithm>
namespace gjxl::modular_internal {
namespace {
Status Invalid() { return Status::InvalidArgument("Invalid or unsupported Modular transform"); }
void Copy(const ModularImage &in, size_t src, ModularImage *out, size_t dst) {
  auto v = in.view(src);
  auto samples = out->samples(dst);
  for (size_t y = 0; y < v.descriptor.extent.height; ++y)
    std::copy_n(v.Row(y), v.descriptor.extent.width,
                samples.data() + y * v.descriptor.extent.width);
}
} // namespace
Status ApplyPalette(const ModularImage &in, const TransformDescriptor &t, bool inverse,
                    ModularImage *out) {
  const size_t end = t.begin + t.count;
  if (!inverse) {
    using Color = std::array<int32_t, 4>;
    std::array<Color, kMaximumPaletteColors> palette{};
    size_t used = 0;
    const auto shape = in.view(t.begin).descriptor.extent;
    auto color = [&](size_t x, size_t y) {
      Color v{};
      for (size_t c = 0; c < t.count; ++c)
        v[c] = in.view(t.begin + c).Row(y)[x];
      return v;
    };
    for (size_t y = 0; y < shape.height; ++y)
      for (size_t x = 0; x < shape.width; ++x) {
        const auto v = color(x, y);
        auto pos = std::lower_bound(palette.begin(), palette.begin() + used, v);
        if (pos != palette.begin() + used && *pos == v)
          continue;
        if (used == t.colors)
          return Status::InvalidArgument("Exact palette capacity exceeded");
        std::move_backward(pos, palette.begin() + used, palette.begin() + used + 1);
        *pos = v;
        ++used;
      }
    for (size_t c = 0; c < t.count; ++c)
      for (size_t i = 0; i < used; ++i)
        out->samples(0)[c * t.colors + i] = palette[i][c];
    auto indices = out->samples(t.begin + 1);
    for (size_t y = 0; y < shape.height; ++y)
      for (size_t x = 0; x < shape.width; ++x)
        indices[y * shape.width + x] = static_cast<int32_t>(
            std::lower_bound(palette.begin(), palette.begin() + used, color(x, y)) -
            palette.begin());
    for (size_t c = 0; c < in.channel_count(); ++c)
      if (c < t.begin || c >= end)
        Copy(in, c, out, c < t.begin ? c + 1 : c + 2 - t.count);
  } else {
    const auto indices = in.view(t.begin + 1);
    const auto shape = indices.descriptor.extent;
    for (size_t y = 0; y < shape.height; ++y)
      for (size_t x = 0; x < shape.width; ++x) {
        const int32_t i = indices.Row(y)[x];
        if (i < 0 || i >= t.colors)
          return Invalid();
        for (size_t c = 0; c < t.count; ++c)
          out->samples(t.begin + c)[y * shape.width + x] = in.view(0).Row(c)[i];
      }
    for (size_t c = 0; c < out->channel_count(); ++c)
      if (c < t.begin || c >= end)
        Copy(in, c < t.begin ? c + 1 : c + 2 - t.count, out, c);
  }
  return Status::Ok();
}
} // namespace gjxl::modular_internal
