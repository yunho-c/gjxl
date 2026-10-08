// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/frame_metadata.h"
#include "codestream/bit_writer.h"
#include "core/host_storage_bound.h"

namespace gjxl::codestream_internal {
[[nodiscard]] Status WriteImageHeader(const codec_internal::ImageMetadata &metadata,
                                      BitWriter *writer);
[[nodiscard]] Status WriteFrameHeader(const codec_internal::ImageMetadata &image,
                                      const codec_internal::FrameMetadata &frame,
                                      BitWriter *writer);
struct CommonHeaderStoragePlan {
  size_t maximum_image_bits = 256;
  size_t maximum_frame_bits = 128;
  resource_budget_internal::HostStorageBound temporary;
  resource_budget_internal::HostStorageBound destination;
  bool operator==(const CommonHeaderStoragePlan &) const = default;
};
// Both consecutive headers; prior destination history is included, and the
// caller adds other owners. One temporary is live at a time. Bound temporary
// and growing destination together to cover replacement/publication overlap.
[[nodiscard]] Status ComputeCommonHeaderStoragePlan(size_t prior_bits,
                                                    CommonHeaderStoragePlan *out);
} // namespace gjxl::codestream_internal
