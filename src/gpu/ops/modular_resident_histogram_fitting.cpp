// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_resident_histogram_fitting.h"
#include <algorithm>

namespace gjxl::histogram_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
namespace resident = gjxl_histogram_resident;
namespace {
struct Arena {
  size_t bytes = 0;
  bool Add(size_t count, size_t stride, uint32_t *offset) {
    if (bytes > UINT32_MAX - 255) return false;
    bytes = (bytes + 255) & ~size_t{255};
    if (count && stride > (UINT32_MAX - bytes) / count) return false;
    *offset = static_cast<uint32_t>(bytes);
    bytes += count * stride;
    return true;
  }
};
}
Status ComputeResidentHistogramPlan(const ResidentHistogramInput &input,
                                   ResidentHistogramPlan *out,
                                   const HistogramFittingDeviceLimits *limits) {
  if (!out || input.requests.empty())
    return Status::InvalidArgument("Empty resident histogram request/output");
  if (input.requests.size() > kMaximumResidentHistogramFits)
    return Status::Unsupported("Resident histogram cohort exceeds 64 fits");
  ResidentHistogramPlan plan;
  plan.fits = input.requests.size();
  std::array<HistogramFittingShape, kMaximumResidentHistogramFits> shapes;
  Arena arena{plan.fits * sizeof(resident::Fit)};
  for (size_t i = 0; i < plan.fits; ++i) {
    const auto &r = input.requests[i];
    if (!r.source.samples)
      return Status::InvalidArgument("Null resident histogram samples");
    if (auto s = ValidateHistogramTrainingSamples(*r.source.samples, r.learning); !s.ok()) return s;
    if (auto s = ResolveHistogramFittingSource(r.source, &shapes[i]); !s.ok()) return s;
    const auto &shape = shapes[i];
    if (!r.learning.multipliers.empty() || r.learning.execution_stats ||
        shape.rows > kMaximumResidentHistogramRowsPerFit)
      return Status::Unsupported("Resident histogram prototype eligibility exceeded");
    size_t bins = 1;
    for (size_t p = 0; p < shape.properties; ++p) bins = std::max(bins, shape.bins[p]);
    if (bins > 256)
      return Status::Unsupported("Resident histogram prototype supports at most 256 bins");
    auto &f = plan.layout[i];
    f.maximum_nodes = static_cast<uint32_t>(2 * r.learning.maximum_leaves - 1);
    f.maximum_depth = static_cast<uint32_t>(r.learning.maximum_depth);
    f.maximum_bins = static_cast<uint32_t>(bins);
    f.split_threshold = r.learning.split_threshold;
    f.fast_decode_multiplier = r.learning.fast_decode_multiplier;
    if (!arena.Add(1, sizeof(resident::State), &f.state) ||
        !arena.Add(f.maximum_nodes, sizeof(resident::Node), &f.nodes) ||
        !arena.Add(f.maximum_depth + 1, sizeof(uint32_t), &f.stack) ||
        !arena.Add(shape.rows, sizeof(uint32_t), &f.order) ||
        !arena.Add(shape.rows, sizeof(uint32_t), &f.partition) ||
        !arena.Add(shape.predictors * 256, sizeof(uint32_t), &f.parent) ||
        !arena.Add(shape.predictors, sizeof(uint32_t), &f.parent_extra) ||
        !arena.Add(bins * 256, sizeof(uint32_t), &f.histogram) ||
        !arena.Add(bins, sizeof(uint32_t), &f.extra) ||
        !arena.Add(bins, sizeof(uint32_t), &f.occupancy) ||
        !arena.Add(bins * 2, sizeof(gjxl_histogram_fitting::SideWinner), &f.sides) ||
        !arena.Add(shape.predictors * shape.predictors, sizeof(float), &f.penalties))
      return Status::OutOfMemory("Resident histogram arena overflow");
    for (size_t p = 0; p < shape.properties; ++p)
      if (!arena.Add(shape.bins[p] - 1, sizeof(int32_t), &f.thresholds[p]))
        return Status::OutOfMemory("Resident histogram threshold overflow");
    if (f.maximum_nodes > kInlineTreeNodes &&
        !plan.output.AddVector<DecisionNode>(f.maximum_nodes, VectorCapacityPolicy::kFreshExact))
      return Status::OutOfMemory("Resident histogram tree bound overflow");
    plan.rows += shape.rows;
  }
  if (auto s = ComputeHistogramFittingPreparationPlan(std::span(shapes).first(plan.fits),
                                                       &plan.prepared, limits); !s.ok()) return s;
  plan.work_bytes = arena.bytes;
  if (limits && plan.work_bytes > limits->maximum_buffer_bytes)
    return Status::Unsupported("Resident histogram arena exceeds device limit");
  if (!plan.output.AddVector<DecisionTree>(plan.fits, VectorCapacityPolicy::kFreshExact) ||
      (input.capture_row_order &&
       (!plan.output.AddVector<uint32_t>(plan.rows, VectorCapacityPolicy::kFreshExact) ||
        !plan.output.AddVector<size_t>(plan.fits + 1, VectorCapacityPolicy::kFreshExact))))
    return Status::OutOfMemory("Resident histogram output bound overflow");
  plan.working = plan.output;
  if (!plan.working.Add({plan.work_bytes, plan.work_bytes}) ||
      !plan.working.Add(plan.prepared.preparation))
    return Status::OutOfMemory("Resident histogram working bound overflow");
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::histogram_gpu_internal
