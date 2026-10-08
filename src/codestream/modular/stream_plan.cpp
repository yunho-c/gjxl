// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Placement follows pinned libjxl enc_modular.cc and dec_modular.h.
#include "codestream/modular/stream_plan.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gjxl::modular_internal {
using namespace resource_budget_internal;
namespace {
size_t GlobalEnd(std::span<const ChannelDescriptor> channels, size_t meta) {
  while (meta < channels.size() && channels[meta].extent.width <= kGroupDimension &&
         channels[meta].extent.height <= kGroupDimension)
    ++meta;
  return meta;
}
Status Validate(const ModularFrameGeometry &g,
                std::span<const ChannelDescriptor> channels, size_t meta) {
  ModularFrameGeometry checked;
  if (auto s = ModularFrameGeometry::Create(g.source(), &checked); !s.ok())
    return s;
  if (channels.size() > INT32_MAX)
    return Status::Unsupported("Modular channel IDs exceed the property range");
  if (auto s = ValidateChannelDescriptors(channels, meta); !s.ok())
    return s;
  // Stream IDs become signed 32-bit static tree properties. Geometry may
  // describe a larger header, but this encoder plan cannot silently narrow IDs.
  if (1 + 3 * g.dc_group_count() + 17 + g.group_count() - 1 > INT32_MAX)
    return Status::Unsupported("Modular stream IDs exceed the property range");
  const size_t global = GlobalEnd(channels, meta);
  for (size_t c = meta; c < channels.size(); ++c) {
    const auto &ch = channels[c];
    const auto max_width = g.source().ceil_div(size_t{1} << ch.hshift).width;
    const auto max_height = g.source().ceil_div(size_t{1} << ch.vshift).height;
    if (ch.extent.width > max_width || ch.extent.height > max_height)
      return Status::InvalidArgument("Modular channel exceeds source extent");
    const auto bits = std::min(ch.hshift, ch.vshift) >= 3 ? 11 : 8;
    if (c >= global && (ch.hshift > bits || ch.vshift > bits))
      return Status::Unsupported("Grouped channel shift exceeds group dimension");
  }
  return Status::Ok();
}
} // namespace

Status ModularStreamId(const ModularFrameGeometry &g, StreamRole role, size_t group,
                       size_t *out) {
  if (!out || g.source().empty())
    return Status::InvalidArgument("Invalid Modular stream geometry");
  switch (role) {
  case StreamRole::kGlobal:
    if (group != 0)
      break;
    *out = 0;
    return Status::Ok();
  case StreamRole::kDcGroup:
    if (group >= g.dc_group_count())
      break;
    *out = 1 + g.dc_group_count() + group;
    return Status::Ok();
  case StreamRole::kGroup:
    if (group >= g.group_count())
      break;
    *out = 1 + 3 * g.dc_group_count() + 17 + group;
    return Status::Ok();
  }
  return Status::InvalidArgument("Invalid Modular stream identity");
}

Status ComputeModularStreamStoragePlan(const ModularFrameGeometry &g,
                                       std::span<const ChannelDescriptor> channels,
                                       size_t meta, ModularStreamStoragePlan *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular stream storage output");
  if (auto s = Validate(g, channels, meta); !s.ok())
    return s;
  ModularStreamStoragePlan p;
  p.streams = 1 + g.dc_group_count() + g.group_count();
  const size_t global = GlobalEnd(channels, meta);
  p.maximum_slices = global;
  // A channel contributes at most once per group of its selected kind. This
  // bound intentionally includes edge groups whose shifted rectangle is empty.
  for (size_t c = global; c < channels.size(); ++c) {
    const size_t groups = std::min(channels[c].hshift, channels[c].vshift) >= 3
                              ? g.dc_group_count()
                              : g.group_count();
    if (p.maximum_slices > std::numeric_limits<size_t>::max() - groups)
      return Status::InvalidArgument("Modular slice count overflow");
    p.maximum_slices += groups;
  }
  if (!p.owned.AddVector<PlannedStream>(p.streams, VectorCapacityPolicy::kFreshExact) ||
      !p.owned.AddVector<ChannelSlice>(p.maximum_slices,
                                       VectorCapacityPolicy::kFreshExact))
    return Status::InvalidArgument("Modular stream storage overflow");
  *out = p;
  return Status::Ok();
}

Status BuildModularStreamPlan(const ModularFrameGeometry &g,
                              std::span<const ChannelDescriptor> channels, size_t meta,
                              ModularStreamPlan *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular stream plan output");
  ModularStreamStoragePlan bound;
  if (auto s = ComputeModularStreamStoragePlan(g, channels, meta, &bound); !s.ok())
    return s;
  try {
    ModularStreamPlan p;
    p.geometry = g;
    p.ac_global_section = g.group_count() == 1 ? 0 : 1 + g.dc_group_count();
    p.streams.reserve(bound.streams);
    p.slices.reserve(bound.maximum_slices);
    const size_t global = GlobalEnd(channels, meta);
    p.streams.push_back({StreamRole::kGlobal, 0, 0, 0, 0, global});
    for (size_t c = 0; c < global; ++c)
      p.slices.push_back({c, {0, 0, channels[c].extent}});
    for (bool dc : {true, false}) {
      const size_t count = dc ? g.dc_group_count() : g.group_count();
      const auto role = dc ? StreamRole::kDcGroup : StreamRole::kGroup;
      for (size_t group = 0; group < count; ++group) {
        ModularRect rect;
        if (auto s = g.GroupRect(group, dc, &rect); !s.ok())
          return s;
        size_t id;
        if (auto s = ModularStreamId(g, role, group, &id); !s.ok())
          return s;
        const size_t section = g.group_count() == 1
                                   ? 0
                                   : (dc ? 1 + group : 2 + g.dc_group_count() + group);
        PlannedStream stream{role, group, id, section, p.slices.size(), 0};
        for (size_t c = global; c < channels.size(); ++c) {
          const auto &ch = channels[c];
          if ((std::min(ch.hshift, ch.vshift) >= 3) != dc)
            continue;
          const size_t x = rect.x >> ch.hshift, y = rect.y >> ch.vshift;
          if (x >= ch.extent.width || y >= ch.extent.height)
            continue;
          const Extent2D extent{
              std::min(rect.extent.width >> ch.hshift, ch.extent.width - x),
              std::min(rect.extent.height >> ch.vshift, ch.extent.height - y)};
          if (!extent.empty())
            p.slices.push_back({c, {x, y, extent}});
        }
        stream.slice_count = p.slices.size() - stream.slice_begin;
        p.streams.push_back(stream);
      }
    }
    *out = std::move(p);
    return Status::Ok();
  } catch (const ManagedAllocationFailure &e) {
    return e.status();
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Modular plan allocation failed");
  } catch (const std::length_error &) {
    return Status::InvalidArgument("Modular plan allocation overflow");
  }
}

Status BorrowChannelSlice(const ModularChannelView &ch, ModularRect r,
                          ModularChannelView *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular slice output");
  if (auto s = ch.Validate(); !s.ok())
    return s;
  const auto extent = ch.descriptor.extent;
  if (r.extent.empty() || r.x >= extent.width || r.y >= extent.height ||
      r.extent.width > extent.width - r.x || r.extent.height > extent.height - r.y)
    return Status::InvalidArgument("Modular slice is outside its channel");
  ModularChannelView view = ch;
  view.backing = ch.backing.subspan(r.y * ch.stride + r.x);
  view.descriptor.extent = r.extent;
  *out = view;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
