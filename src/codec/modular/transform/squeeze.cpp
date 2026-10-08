// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Scalar squeeze arithmetic adapted from pinned libjxl.
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
// Wide intermediates make signed overflow rejection explicit at plane storage.
int64_t Average(int64_t a, int64_t b) { return (a + b + (a > b)) >> 1; }
int64_t Tendency(int64_t b, int64_t a, int64_t n) {
  int64_t d = 0;
  if (b >= a && a >= n) {
    d = (4 * b - 3 * n - a + 6) / 12;
    if (d - (d & 1) > 2 * (b - a))
      d = 2 * (b - a) + 1;
    if (d + (d & 1) > 2 * (a - n))
      d = 2 * (a - n);
  } else if (b <= a && a <= n) {
    d = (4 * b - 3 * n - a - 6) / 12;
    if (d + (d & 1) < 2 * (b - a))
      d = 2 * (b - a) - 1;
    if (d - (d & 1) < 2 * (a - n))
      d = 2 * (a - n);
  }
  return d;
}
bool Store(std::span<int32_t> samples, size_t i, int64_t v) {
  if (v < INT32_MIN || v > INT32_MAX)
    return false;
  samples[i] = static_cast<int32_t>(v);
  return true;
}
} // namespace
Status ApplySqueeze(const ModularImage &in, const TransformDescriptor &t, bool inverse,
                    ModularImage *out) {
  const size_t original_count = inverse ? out->channel_count() : in.channel_count();
  const size_t end = t.begin + t.count, offset = t.in_place ? end : original_count;
  for (size_t c = 0; c < original_count; ++c) {
    if (c < t.begin || c >= end) {
      const size_t mapped = c < offset ? c : c + t.count;
      if (inverse)
        Copy(in, mapped, out, c);
      else
        Copy(in, c, out, mapped);
      continue;
    }
    const size_t rc = offset + c - t.begin;
    const auto shape = inverse ? out->view(c).descriptor.extent : in.view(c).descriptor.extent;
    const size_t length = t.horizontal ? shape.width : shape.height;
    const size_t lines = t.horizontal ? shape.height : shape.width;
    auto index = [&](Extent2D e, size_t line, size_t pos) {
      return t.horizontal ? line * e.width + pos : pos * e.width + line;
    };
    const auto avg_shape = inverse ? in.view(c).descriptor.extent : out->view(c).descriptor.extent;
    const auto res_shape =
        inverse ? in.view(rc).descriptor.extent : out->view(rc).descriptor.extent;
    for (size_t line = 0; line < lines; ++line) {
      auto read = [&](size_t ch, size_t pos) -> int64_t {
        auto v = in.view(ch);
        return v.backing[index(v.descriptor.extent, line, pos)];
      };
      int64_t previous = 0;
      for (size_t p = 0; p < length / 2; ++p) {
        if (!inverse) {
          const int64_t a = read(c, 2 * p), b = read(c, 2 * p + 1), avg = Average(a, b);
          const int64_t next = 2 * p + 3 < length ? Average(read(c, 2 * p + 2), read(c, 2 * p + 3))
                               : 2 * p + 2 < length ? read(c, 2 * p + 2)
                                                    : avg;
          const int64_t residual = a - b - Tendency(p ? previous : avg, avg, next);
          if (!Store(out->samples(c), index(avg_shape, line, p), avg) ||
              !Store(out->samples(rc), index(res_shape, line, p), residual))
            return Invalid();
          previous = b;
        } else {
          const int64_t avg = read(c, p);
          const int64_t next = p + 1 < (length + 1) / 2 ? read(c, p + 1) : avg;
          const int64_t diff = read(rc, p) + Tendency(p ? previous : avg, avg, next);
          const int64_t a = avg + diff / 2, b = a - diff;
          if (!Store(out->samples(c), index(shape, line, 2 * p), a) ||
              !Store(out->samples(c), index(shape, line, 2 * p + 1), b))
            return Invalid();
          previous = b;
        }
      }
      if (length & 1) {
        const size_t pos = inverse ? length - 1 : length / 2;
        const auto dst_shape = inverse ? shape : avg_shape;
        if (!Store(out->samples(c), index(dst_shape, line, pos),
                   read(c, inverse ? length / 2 : length - 1)))
          return Invalid();
      }
    }
  }
  return Status::Ok();
}
} // namespace gjxl::modular_internal
