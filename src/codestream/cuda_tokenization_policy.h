// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "codestream/ac_tokenization_provider_internal.h"
#include "codestream/workflow_internal.h"

namespace gjxl::codestream_internal {
inline bool UseCudaGpuTokenization(const VarDctEncodingOptions& options) {
#ifdef GJXL_CUDA_RESIDENT_TOKEN_EXPERIMENT
  const char* value = std::getenv("GJXL_EXPERIMENT_CUDA_GPU_TOKENS");
  return value && std::strcmp(value, "1") == 0 && GpuTokenizationEnabled() &&
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
