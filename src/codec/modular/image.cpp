// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/image.h"
#include <limits>
#include <stdexcept>

namespace gjxl::modular_internal {
using namespace resource_budget_internal;

Status ValidateChannelDescriptors(std::span<const ChannelDescriptor> channels,
                                  size_t metadata_channels) {
  if (channels.empty() || metadata_channels > channels.size())
    return Status::InvalidArgument("Invalid Modular channel count");
  for (size_t c = 0; c < channels.size(); ++c) {
    const auto &ch = channels[c];
    size_t area;
    if (ch.extent.empty() || !ch.extent.try_area(&area) || ch.hshift > 30 ||
        ch.vshift > 30 ||
        (ch.role != ChannelRole::kColor && ch.role != ChannelRole::kAlpha &&
         ch.role != ChannelRole::kMetadata) ||
        (ch.role == ChannelRole::kMetadata) != (c < metadata_channels))
      return Status::InvalidArgument("Invalid Modular channel descriptor");
  }
  return Status::Ok();
}

Status ModularChannelView::Validate() const {
  const size_t meta = descriptor.role == ChannelRole::kMetadata ? 1 : 0;
  if (auto s = ValidateChannelDescriptors({&descriptor, 1}, meta); !s.ok())
    return s;
  const auto [w, h] = descriptor.extent;
  if (stride < w || backing.data() == nullptr ||
      h - 1 > (std::numeric_limits<size_t>::max() - w) / stride)
    return Status::InvalidArgument("Invalid Modular channel stride");
  const size_t end = (h - 1) * stride + w;
  if (end > backing.size() || end > static_cast<size_t>(PTRDIFF_MAX) / sizeof(int32_t))
    return Status::InvalidArgument("Modular channel backing is too short");
  return Status::Ok();
}

Status ComputeModularImageStorageBound(std::span<const ChannelDescriptor> channels,
                                       size_t metadata_channels,
                                       HostStorageBound *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular image storage output");
  if (auto s = ValidateChannelDescriptors(channels, metadata_channels); !s.ok())
    return s;
  HostStorageBound bound;
  if (!bound.AddVector<ModularImage::Channel>(channels.size(),
                                              VectorCapacityPolicy::kFreshExact))
    return Status::InvalidArgument("Modular channel records overflow");
  for (const auto &ch : channels) {
    size_t area;
    if (!ch.extent.try_area(&area) ||
        !bound.AddVector<int32_t>(area, VectorCapacityPolicy::kFreshExact))
      return Status::InvalidArgument("Modular channel storage overflow");
  }
  *out = bound;
  return Status::Ok();
}

Status ModularImage::Create(std::span<const ChannelDescriptor> channels,
                            size_t metadata_channels, ModularImage *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular image output");
  HostStorageBound bound;
  if (auto s = ComputeModularImageStorageBound(channels, metadata_channels, &bound);
      !s.ok())
    return s;
  try {
    ModularImage image;
    image.channels_.resize(channels.size());
    for (size_t c = 0; c < channels.size(); ++c) {
      image.channels_[c].descriptor = channels[c];
      image.channels_[c].samples.resize(channels[c].extent.width *
                                        channels[c].extent.height);
    }
    image.metadata_channels_ = metadata_channels;
    *out = std::move(image);
    return Status::Ok();
  } catch (const ManagedAllocationFailure &e) {
    return e.status();
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Modular image allocation failed");
  } catch (const std::length_error &) {
    return Status::InvalidArgument("Modular image allocation overflow");
  }
}
} // namespace gjxl::modular_internal
