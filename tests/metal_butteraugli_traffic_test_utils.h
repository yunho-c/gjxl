// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "gpu/metal/metal_backend_internal.h"

namespace gjxl::metal_internal {
struct MetalButteraugliTrafficTestAccess {
  static bool Enabled(GpuBackend &backend) {
    return static_cast<MetalBackend &>(backend)
        .butteraugli_pipelines_.traffic.enabled;
  }
  // Only call before creating prepared states or starting worker threads.
  static void Disable(GpuBackend &backend) {
    static_cast<MetalBackend &>(backend)
        .butteraugli_pipelines_.traffic.enabled = false;
  }
};
} // namespace gjxl::metal_internal

namespace gjxl::test {
inline bool force_legacy_butteraugli = false;

inline Status
CreateButteraugliTestBackend(std::string_view path,
                             const MetalBackendOptions &options,
                             std::unique_ptr<GpuBackend> *backend) {
  Status status = CreateMetalBackend(path, options, backend);
  if (!status.ok())
    return status;
  using Access = metal_internal::MetalButteraugliTrafficTestAccess;
  if ((*backend)->name() == "Metal: Apple M4 Pro" &&
      !Access::Enabled(**backend)) {
    return Status::Internal(
        "Qualified M4 Pro Butteraugli bundle was not enabled");
  }
  if ((*backend)->name() != "Metal: Apple M4 Pro" &&
      Access::Enabled(**backend)) {
    return Status::Internal(
        "Butteraugli traffic bundle enabled on unqualified GPU");
  }
  if (force_legacy_butteraugli)
    Access::Disable(**backend);
  return Status::Ok();
}

inline Status
CreateButteraugliTestBackend(std::string_view path,
                             std::unique_ptr<GpuBackend> *backend) {
  return CreateButteraugliTestBackend(path, MetalBackendOptions{}, backend);
}
} // namespace gjxl::test
