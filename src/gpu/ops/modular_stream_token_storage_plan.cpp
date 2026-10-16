// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/weighted_feature_math.h"
#include "gpu/ops/modular_stream_tokens.h"
#include <algorithm>

namespace gjxl::modular_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
namespace {
Status Invalid() { return Status::InvalidArgument("Invalid Modular stream batch storage"); }
bool Add(size_t value, size_t *out) {
  if (value > size_t(PTRDIFF_MAX) - *out)
    return false;
  *out += value;
  return true;
}
bool Align(size_t value, size_t *out) {
  if (value > size_t(PTRDIFF_MAX) - 255)
    return false;
  *out = (value + 255) & ~size_t{255};
  return true;
}
bool Valid(const ModularTokenDeviceLimits *limits) {
  return !limits ||
         (limits->maximum_tokens && limits->maximum_slices && limits->maximum_buffer_bytes);
}
bool Fits(const ModularTokenDeviceLimits *limits, size_t samples, size_t slices,
          std::initializer_list<size_t> buffers) {
  return !limits || (samples <= limits->maximum_tokens && slices <= limits->maximum_slices &&
                     std::ranges::all_of(buffers, [&](size_t bytes) {
                       return bytes <= limits->maximum_buffer_bytes;
                     }));
}
Status TokenStorage(size_t samples, size_t slices, ModularStreamTokenStoragePlan p,
                    ModularStreamTokenStoragePlan *out, const ModularTokenDeviceLimits *limits) {
  if (!out || !samples || !slices || slices > samples ||
      slices > kMaximumModularFeatureBatchSlices || !p.nodes || p.nodes > UINT32_MAX ||
      p.nodes > size_t(PTRDIFF_MAX) / sizeof(ModularTokenNode) ||
      samples > size_t(PTRDIFF_MAX) / sizeof(uint32_t) || !Valid(limits)) return Invalid();
  p.values_bytes = samples * sizeof(uint32_t);
  if (!Align(p.values_bytes, &p.contexts_offset) ||
      !Align(slices * sizeof(AdaptiveModularTokenSlice), &p.nodes_offset)) return Invalid();
  p.output_bytes = p.contexts_offset;
  size_t end = p.nodes_offset;
  if (!Add(samples * sizeof(uint16_t), &p.output_bytes) ||
      !Add(p.nodes * sizeof(ModularTokenNode), &end) || !Align(end, &p.status_offset)) return Invalid();
  p.dispatch_bytes = p.status_offset;
  if (!Add(sizeof(uint32_t), &p.dispatch_bytes)) return Invalid();
  if (!Fits(limits, samples, slices, {p.output_bytes, p.dispatch_bytes}))
    return Status::Unsupported("Modular token batch exceeds device limits");
  if (!p.retained.Add({p.output_bytes, p.output_bytes})) return Invalid();
  p.generation = p.retained;
  if (!p.generation.Add({p.dispatch_bytes, p.dispatch_bytes})) return Invalid();
  *out = p;
  return Status::Ok();
}
} // namespace
Status ComputeModularStreamFeatureStoragePlan(std::span<const ModularStreamFeatureSlice> slices,
                                              size_t cpu_participants,
                                              ModularStreamFeatureStoragePlan *out,
                                              const ModularTokenDeviceLimits *limits) {
  if (!out || slices.empty() || !cpu_participants || cpu_participants > 256 || !Valid(limits))
    return Invalid();
  if (slices.size() > kMaximumModularFeatureBatchSlices)
    return Status::Unsupported("Modular feature batch exceeds slice admission");
  ModularStreamFeatureStoragePlan p;
  p.wavefront_shapes = true;
  for (const auto &slice : slices) {
    if (!slice.parameters.valid() || slice.channel >= 65535 || slice.stream_id >= 65535)
      return Invalid();
    if (auto s = slice.input.Validate(); !s.ok())
      return s;
    const auto [w, h] = slice.input.descriptor.extent;
    size_t count;
    if (!w || !h || w > UINT32_MAX || h > UINT32_MAX ||
        !slice.input.descriptor.extent.try_area(&count) || count > UINT32_MAX ||
        !Add(count, &p.samples))
      return Invalid();
    p.maximum_width = std::max(p.maximum_width, w);
    p.maximum_slice_samples = std::max(p.maximum_slice_samples, count);
    p.wavefront_shapes &= w <= 256 && h <= 256;
  }
  if (p.samples > size_t(PTRDIFF_MAX) / sizeof(WeightedPredictorFeature))
    return Invalid();
  p.plane_bytes = p.samples * sizeof(int32_t);
  p.feature_bytes = p.samples * sizeof(WeightedPredictorFeature);
  p.slices_bytes = slices.size() * sizeof(AdaptiveModularTokenSlice);
  if (!Align(p.slices_bytes, &p.parameters_offset) ||
      !Align(p.parameters_offset + slices.size() * sizeof(gjxl_weighted_features::Parameters),
             &p.status_offset))
    return Invalid();
  p.dispatch_bytes = p.status_offset + sizeof(uint32_t);
  if (!Fits(limits, p.samples, slices.size(), {p.plane_bytes, p.feature_bytes, p.dispatch_bytes}))
    return Status::Unsupported("Modular feature batch exceeds device limits");
  if (!p.retained.Add({p.plane_bytes, p.plane_bytes}) ||
      !p.retained.Add({p.feature_bytes, p.feature_bytes}) ||
      !p.retained.AddVector<AdaptiveModularTokenSlice>(slices.size(),
                                                       VectorCapacityPolicy::kFreshExact) ||
      !p.retained.AddVector<WeightedFeatureView>(slices.size(), VectorCapacityPolicy::kFreshExact))
    return Invalid();
  p.gpu_preparation = p.retained;
  if (!p.gpu_preparation.Add({p.dispatch_bytes, p.dispatch_bytes}))
    return Invalid();
  p.cpu_preparation = p.retained;
  WeightedFeatureBatchStoragePlan cpu;
  if (auto s = ComputeWeightedFeatureBatchStoragePlan(slices.size(), p.maximum_width,
                                                      cpu_participants, &cpu);
      !s.ok())
    return s;
  if (!p.cpu_preparation.Add(cpu.jobs) || !p.cpu_preparation.Add(cpu.execution) ||
      !p.cpu_preparation.AddVector<WeightedPredictorParameters>(slices.size(),
                                                                VectorCapacityPolicy::kFreshExact))
    return Invalid();
  *out = p;
  return Status::Ok();
}

Status ComputeModularStreamTokenStoragePlan(size_t samples, size_t slices,
                                            std::span<const ModularStreamTokenTree> trees,
                                            ModularStreamTokenStoragePlan *out,
                                            const ModularTokenDeviceLimits *limits) {
  if (!out || !samples || !slices || slices > samples ||
      slices > kMaximumModularFeatureBatchSlices || trees.empty() || trees.size() > slices ||
      samples > size_t(PTRDIFF_MAX) / sizeof(uint32_t) || !Valid(limits))
    return Invalid();
  ModularStreamTokenStoragePlan p;
  size_t next = 0;
  for (const auto &binding : trees) {
    if (!binding.tree || binding.first_slice != next || !binding.slice_count ||
        binding.slice_count > slices - next)
      return Invalid();
    TreeLayout layout;
    if (auto s = ValidateTree(*binding.tree, &layout); !s.ok())
      return s;
    if (!Add(binding.tree->size, &p.nodes))
      return Invalid();
    p.maximum_contexts = std::max(p.maximum_contexts, layout.leaves);
    next += binding.slice_count;
  }
  if (next != slices) return Invalid();
  return TokenStorage(samples, slices, p, out, limits);
}
Status ComputeModularStreamTokenStorageBound(size_t samples, size_t slices, size_t maximum_nodes,
                                            HostStorageBound *out) {
  if (!out) return Invalid();
  ModularStreamTokenStoragePlan p;
  p.nodes = maximum_nodes;
  if (auto s = TokenStorage(samples, slices, p, &p, nullptr); !s.ok()) return s;
  *out = p.generation;
  return Status::Ok();
}
} // namespace gjxl::modular_gpu_internal
