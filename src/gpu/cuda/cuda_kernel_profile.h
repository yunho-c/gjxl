// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cuda_runtime_api.h>

#include "gpu/cuda/cuda_profile_limits.h"

namespace gjxl::cuda_internal {

// The kernel translation units compile as C++17. Keep managed containers,
// status handling and event ownership in the C++20 submission implementation.
// Installed only around one synchronous encode callback on its submitting
// thread; the ordinary path performs no profile allocation or event recording.
struct CudaKernelProfileHooks {
  void* context;
  bool (*begin)(void*, const char*, dim3, dim3, cudaStream_t) noexcept;
  void (*end)(void*, cudaStream_t) noexcept;
};
extern thread_local const CudaKernelProfileHooks* cuda_kernel_profile_hooks;

class CudaKernelProfileScope {
public:
  template <size_t N>
  CudaKernelProfileScope(const char (&id)[N], dim3 grid, dim3 block,
                         cudaStream_t stream) noexcept
    : hooks_(cuda_kernel_profile_hooks), stream_(stream),
      launch_(hooks_ == nullptr || hooks_->begin(hooks_->context, id, grid, block, stream)) {
    static_assert(N - 1 <= kCudaKernelProfileIdLength);
  }
  ~CudaKernelProfileScope() {
    if (hooks_ != nullptr && launch_) hooks_->end(hooks_->context, stream_);
  }
  explicit operator bool() const noexcept { return launch_; }
  CudaKernelProfileScope(const CudaKernelProfileScope&) = delete;
  CudaKernelProfileScope& operator=(const CudaKernelProfileScope&) = delete;
private:
  const CudaKernelProfileHooks* hooks_;
  cudaStream_t stream_;
  bool launch_;
};

} // namespace gjxl::cuda_internal
