// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "codestream/ac_tokenization_provider_internal.h"
#include "codestream/entropy_readiness_internal.h"
#include "codestream/workflow_internal.h"

namespace gjxl::codestream_internal {
inline bool UseCudaGpuTokenization(const VarDctEncodingOptions& options) {
#if defined(GJXL_ENABLE_CUDA) && defined(GJXL_CUDA_RESIDENT_TOKEN_EXPERIMENT)
  if (!GpuTokenizationEnabled()) return false;
  const char* value = std::getenv("GJXL_EXPERIMENT_CUDA_GPU_TOKENS");
  // Preserve explicit legacy selection, including CPU overrides. The stable
  // switch can enable eligible batch calls without the legacy selector.
  if (value && std::strcmp(value, "0") == 0) return false;
  const char* stable = std::getenv("GJXL_GPU_TOKENIZATION");
  const bool explicitly_enabled =
      (value && std::strcmp(value, "1") == 0) ||
      (stable && std::strcmp(stable, "1") == 0);
  return (explicitly_enabled || !entropy_readiness_in_batch) &&
         (options.gpu_aq_mode == GpuAdaptiveQuantizationMode::kFullyResident ||
          options.gpu_aq_mode == GpuAdaptiveQuantizationMode::kThroughput) &&
         options.rate_control_mode != VarDctRateControlMode::kMaximumError &&
         ResolveEntropyBehavior(options) !=
             VarDctEntropyBehavior::kMaximumCompression;
#else
  (void)options;
  return false;
#endif
}
}  // namespace gjxl::codestream_internal
