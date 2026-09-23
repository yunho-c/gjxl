// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace gjxl::cuda_internal {
// Private opt-in handoff experiment. A scope is confined to one synchronous
// policy call; repeated prepared evaluations receive independent owners.
#ifdef GJXL_CUDA_RESIDENT_TOKEN_EXPERIMENT
inline thread_local bool retain_completed_token_coefficients = false;
class CudaTokenCoefficientScope {
 public:
  explicit CudaTokenCoefficientScope(bool enabled)
      : previous_(retain_completed_token_coefficients) {
    retain_completed_token_coefficients = enabled;
  }
  ~CudaTokenCoefficientScope() {
    retain_completed_token_coefficients = previous_;
  }
  CudaTokenCoefficientScope(const CudaTokenCoefficientScope&) = delete;
  CudaTokenCoefficientScope& operator=(const CudaTokenCoefficientScope&) =
      delete;

 private:
  bool previous_;
};
#endif
}  // namespace gjxl::cuda_internal
