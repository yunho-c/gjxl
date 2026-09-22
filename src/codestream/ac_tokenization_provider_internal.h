// SPDX-License-Identifier: Apache-2.0
// Experimental producer/consumer seam; model selection and entropy writing stay
// on CPU.
#pragma once
#include "codec/vardct_frame_view_internal.h"
#include "codestream/ac_group.h"
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/entropy_internal.h"
#include <cstdlib>
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
inline bool ExperimentalGpuTokenizationEnabled() {
#ifdef GJXL_TOKENIZATION_EXPERIMENT
  const char *value = std::getenv("GJXL_EXPERIMENT_GPU_TOKENS");
  return value && value[0] == '1';
#else
  return false;
#endif
}
Status CreateMetalAcTokenizationProvider(
    GpuBackend &backend, const DeviceBuffer &coefficients, size_t offset,
    std::unique_ptr<AcTokenizationProvider> *out);
} // namespace gjxl::codestream_internal
