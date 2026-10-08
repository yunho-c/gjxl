// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cstdint>
#include <span>
#include "core/geometry.h"
#include "core/host_storage_bound.h"

namespace gjxl::modular_internal {

enum class ChannelRole : uint8_t { kColor, kAlpha, kMetadata };
struct ChannelDescriptor {
  Extent2D extent;
  uint8_t hshift = 0;
  uint8_t vshift = 0;
  ChannelRole role = ChannelRole::kColor;
  bool operator==(const ChannelDescriptor &) const = default;
};

// Samples start at backing.data(); callers can use subspan for a nonzero offset.
// Stride is in elements. No implicit padding, color conversion or sample copy.
struct ModularChannelView {
  std::span<const int32_t> backing;
  ChannelDescriptor descriptor;
  size_t stride = 0;
  [[nodiscard]] Status Validate() const;
  [[nodiscard]] const int32_t *Row(size_t y) const {
    return backing.data() + y * stride;
  }
};

[[nodiscard]] Status
ValidateChannelDescriptors(std::span<const ChannelDescriptor> channels,
                           size_t metadata_channels);

class ModularImage {
public:
  struct Channel {
    ChannelDescriptor descriptor;
    resource_budget_internal::ManagedVector<int32_t> samples;
  };
  ModularImage() = default;
  ModularImage(const ModularImage &) = delete;
  ModularImage &operator=(const ModularImage &) = delete;
  ModularImage(ModularImage &&) noexcept = default;
  ModularImage &operator=(ModularImage &&) noexcept = default;
  // Identity storage only. Preparation inherits the caller's allocation owner.
  // Replacing an image keeps the old backing alive until this succeeds.
  [[nodiscard]] static Status Create(std::span<const ChannelDescriptor> channels,
                                     size_t metadata_channels, ModularImage *out);
  [[nodiscard]] size_t channel_count() const { return channels_.size(); }
  [[nodiscard]] size_t metadata_channels() const { return metadata_channels_; }
  [[nodiscard]] ModularChannelView view(size_t c) const {
    const auto &ch = channels_[c];
    return {ch.samples, ch.descriptor, ch.descriptor.extent.width};
  }
  [[nodiscard]] std::span<int32_t> samples(size_t c) { return channels_[c].samples; }

private:
  resource_budget_internal::ManagedVector<Channel> channels_;
  size_t metadata_channels_ = 0;
};

// Fresh channel records and planes; add existing output backing for replacement.
// Allocation-free on success. Failure leaves the output unchanged.
[[nodiscard]] Status
ComputeModularImageStorageBound(std::span<const ChannelDescriptor> channels,
                                size_t metadata_channels,
                                resource_budget_internal::HostStorageBound *out);

} // namespace gjxl::modular_internal
