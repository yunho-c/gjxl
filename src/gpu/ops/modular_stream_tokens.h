// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gpu/ops/modular_adaptive_tokens.h"

namespace gjxl::modular_gpu_internal {
// Bounded independent-channel batch, after transform/WP selection. Each source
// starts at its local origin; source stride is retained. IDs are coding
// properties, not offsets or learning-image indices. Empty channels are omitted
// by the workflow without renumbering the surviving channels/streams.
struct ModularStreamFeatureSlice {
  modular_internal::ModularChannelView input;
  modular_internal::WeightedPredictorParameters parameters;
  uint32_t stream_id = 0, channel = 0;
};
inline constexpr size_t kMaximumModularFeatureBatchSlices = 1024;

struct ModularStreamFeatureStoragePlan {
  size_t samples = 0, maximum_width = 0, maximum_slice_samples = 0;
  size_t plane_bytes = 0, feature_bytes = 0, slices_bytes = 0;
  size_t parameters_offset = 0, status_offset = 0, dispatch_bytes = 0;
  bool wavefront_shapes = false;
  // Completed native owner includes one descriptor and one feature-view array,
  // plus packed planes and immutable weighted maps. Fresh execution excludes
  // borrowed source pixels and any old output owner.
  resource_budget_internal::HostStorageBound retained, cpu_preparation, gpu_preparation;
  bool operator==(const ModularStreamFeatureStoragePlan &) const = default;
};
[[nodiscard]] Status ComputeModularStreamFeatureStoragePlan(
    std::span<const ModularStreamFeatureSlice> slices, size_t cpu_participants,
    ModularStreamFeatureStoragePlan *out, const ModularTokenDeviceLimits *limits = nullptr);

// Tree bindings partition every slice exactly once in batch order. Each binding
// uses one immutable tree; contexts remain local to that tree. The enclosing
// workflow validates decoder limits and each stream's local/shared model scope.
struct ModularStreamTokenTree {
  size_t first_slice = 0, slice_count = 0;
  const modular_internal::DecisionTree *tree = nullptr;
};
struct ModularStreamTokenStoragePlan {
  size_t nodes = 0, maximum_contexts = 0;
  size_t values_bytes = 0, contexts_offset = 0, output_bytes = 0;
  size_t nodes_offset = 0, status_offset = 0, dispatch_bytes = 0;
  // New token backing only; generation adds dispatch and excludes the borrowed
  // completed feature owner and borrowed trees, which stay separately charged.
  resource_budget_internal::HostStorageBound retained, generation;
  bool operator==(const ModularStreamTokenStoragePlan &) const = default;
};
[[nodiscard]] Status ComputeModularStreamTokenStoragePlan(
    size_t samples, size_t slices, std::span<const ModularStreamTokenTree> trees,
    ModularStreamTokenStoragePlan *out, const ModularTokenDeviceLimits *limits = nullptr);
// Protect a future token generation before the local trees exist. Uses the
// same layout arithmetic as the exact plan; maximum_nodes bounds all bindings.
[[nodiscard]] Status ComputeModularStreamTokenStorageBound(
    size_t samples, size_t slices, size_t maximum_nodes,
    resource_budget_internal::HostStorageBound *out);

class PreparedGpuModularStreamFeatures {
public:
  virtual ~PreparedGpuModularStreamFeatures() = default;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  [[nodiscard]] virtual size_t sample_count() const noexcept = 0;
  [[nodiscard]] virtual modular_internal::WeightedFeatureBackend producer() const noexcept = 0;
  // Exact original source identities, in the requested batch order. Every map
  // is complete, immutable and host-readable. The backend outlives this owner;
  // original pixels outlive CPU consumers of these borrowed views.
  [[nodiscard]] virtual std::span<const modular_internal::WeightedFeatureView>
  weighted_views() const noexcept = 0;
};
class GpuModularStreamProcessing {
public:
  virtual ~GpuModularStreamProcessing() = default;
  [[nodiscard]] virtual ModularTokenDeviceLimits modular_stream_limits() const noexcept = 0;
  // Same admitted synchronous completion contract as the legacy adaptive
  // operation: publish only after all work/status validation, drain on failure,
  // no CPU retry after execution. Shape/device routing is decided beforehand.
  [[nodiscard]] virtual Status
  PrepareModularStreamFeatures(std::span<const ModularStreamFeatureSlice> slices,
                               std::unique_ptr<PreparedGpuModularStreamFeatures> *out,
                               size_t cpu_participants = 1,
                               modular_internal::WeightedFeatureBackend producer =
                                   modular_internal::WeightedFeatureBackend::kAuto) = 0;
  [[nodiscard]] virtual Status
  GenerateModularStreamTokens(const PreparedGpuModularStreamFeatures &features,
                              std::span<const ModularStreamTokenTree> trees,
                              std::unique_ptr<PreparedGpuModularTokens> *out) = 0;
};
[[nodiscard]] inline GpuModularStreamProcessing *
QueryGpuModularStreamProcessing(GpuBackend &backend) noexcept {
  return dynamic_cast<GpuModularStreamProcessing *>(&backend);
}
} // namespace gjxl::modular_gpu_internal
