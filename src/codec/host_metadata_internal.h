// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cstdlib>
#include <cstring>

namespace gjxl::vardct_frame_internal {

// Qualified production default; 0 opts out to the prior host handoff. Configure
// before encoding; do not mutate the process environment while encodes are active.
inline bool HostMetadataReuseEnabled() noexcept {
  const char* value = std::getenv("GJXL_HOST_METADATA");
  return value == nullptr || std::strcmp(value, "0") != 0;
}

}  // namespace gjxl::vardct_frame_internal
