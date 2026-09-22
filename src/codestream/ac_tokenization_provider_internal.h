// SPDX-License-Identifier: Apache-2.0
// Resident producer/consumer seam; model selection and entropy writing stay
// on CPU.
#pragma once
#include "codec/vardct_frame_view_internal.h"
#include "codestream/ac_group.h"
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/entropy_internal.h"
#include <cstdlib>
#include <cstring>
#include <memory>

namespace gjxl {
class GpuBackend;
class DeviceBuffer;
} // namespace gjxl
namespace gjxl::codestream_internal {
class AcTokenizationProvider {
public:
  virtual ~AcTokenizationProvider() = default;
  virtual Status Begin(const vardct_frame_internal::VarDctFrameView &frame,
                       const SimpleCoefficientOrders &orders,
                       const SimpleAcNaturalOrders &natural,
                       const SimpleBlockContextMap &contexts,
                       bool populations) = 0;
  virtual Status Finish(Storage<EntropyTokenStreamView> *streams,
                        Storage<PreparedFixedAnsCluster> *populations) = 0;
};

inline thread_local AcTokenizationProvider *active_ac_tokenization_provider =
    nullptr;
class AcTokenizationProviderScope {
public:
  explicit AcTokenizationProviderScope(AcTokenizationProvider *provider)
      : previous_(active_ac_tokenization_provider) {
    active_ac_tokenization_provider = provider;
  }
  ~AcTokenizationProviderScope() {
    active_ac_tokenization_provider = previous_;
  }

private:
  AcTokenizationProvider *previous_;
};
// Production defaults match the qualified V7 configuration. Experimental
// ablations remain build-gated; the stable CPU override takes precedence.
inline const char *TokenizationExperimentSetting(const char *name) {
#ifdef GJXL_TOKENIZATION_EXPERIMENT
  return std::getenv(name);
#else
  (void)name;
  return nullptr;
#endif
}
inline bool GpuTokenizationEnabled() {
  if (const char *value = std::getenv("GJXL_GPU_TOKENIZATION")) {
    if (std::strcmp(value, "0") == 0)
      return false;
    if (std::strcmp(value, "1") == 0)
      return true;
  }
  const char *value =
      TokenizationExperimentSetting("GJXL_EXPERIMENT_GPU_TOKENS");
  return value == nullptr || std::strcmp(value, "0") != 0;
}
inline unsigned GpuTokenizationCompactLayout() {
  const char *value =
      TokenizationExperimentSetting("GJXL_EXPERIMENT_TOKEN_COMPACT");
  if (value != nullptr && value[0] >= '0' && value[0] <= '2' &&
      value[1] == '\0')
    return static_cast<unsigned>(value[0] - '0');
  return 2;
}
inline bool GpuTokenizationOverlapEnabled() {
  const char *value =
      TokenizationExperimentSetting("GJXL_EXPERIMENT_TOKEN_OVERLAP");
  return value == nullptr || std::strcmp(value, "0") != 0;
}
Status CreateMetalAcTokenizationProvider(
    GpuBackend &backend, const DeviceBuffer &coefficients, size_t offset,
    std::unique_ptr<AcTokenizationProvider> *out);
} // namespace gjxl::codestream_internal
