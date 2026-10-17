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
  const bool tiled = input.executor == ResidentHistogramExecutor::kTiledFrontier;
  const bool frontier = tiled || input.executor == ResidentHistogramExecutor::kFrontier;
  if ((input.executor != ResidentHistogramExecutor::kDepthFirst && !frontier) ||
      (frontier && (!input.maximum_active_nodes || input.maximum_active_nodes > 1024 ||
                    !input.rounds_per_submission || input.rounds_per_submission > 64 ||
                    !input.maximum_scratch_bytes)))
    return Status::InvalidArgument("Invalid resident frontier execution controls");
  ResidentHistogramPlan plan;
  plan.fits = input.requests.size();
  std::array<HistogramFittingShape, kMaximumResidentHistogramFits> shapes;
  Arena arena{plan.fits * sizeof(resident::Fit)};
  size_t maximum_bins = 1, maximum_predictors = 1, maximum_properties = 1,
         maximum_histogram = 0, queue_capacity = 0;
  for (size_t i = 0; i < plan.fits; ++i) {
    const auto &r = input.requests[i];
    if (!r.source.samples)
      return Status::InvalidArgument("Null resident histogram samples");
    if (auto s = ValidateHistogramTrainingSamples(*r.source.samples, r.learning); !s.ok()) return s;
    if (auto s = ResolveHistogramFittingSource(r.source, &shapes[i]); !s.ok()) return s;
    const auto &shape = shapes[i];
    if (!r.learning.multipliers.empty())
      return Status::Unsupported("Resident histogram prototype does not support multiplier ranges");
    if (r.learning.execution_stats)
      return Status::Unsupported("Resident histogram prototype does not support CPU execution statistics");
    if (shape.rows > kMaximumResidentHistogramRowsPerFit)
      return Status::Unsupported("Resident histogram prototype exceeds 131072 rows per fit");
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
    maximum_bins = std::max(maximum_bins, bins);
    maximum_predictors = std::max(maximum_predictors, shape.predictors);
    maximum_properties = std::max(maximum_properties, shape.properties);
    size_t histogram = 0;
    for (size_t p = 0; p < shape.properties; ++p)
      if (shape.bins[p] > 1) histogram += shape.bins[p] * shape.predictors * 257 * sizeof(uint32_t);
    maximum_histogram = std::max(maximum_histogram, histogram);
    plan.maximum_nodes = std::max(plan.maximum_nodes, size_t(f.maximum_nodes));
    queue_capacity += f.maximum_nodes;
    if (!arena.Add(1, sizeof(resident::State), &f.state) ||
        !arena.Add(f.maximum_nodes, sizeof(resident::Node), &f.nodes) ||
        (!frontier && !arena.Add(f.maximum_depth + 1, sizeof(uint32_t), &f.stack)) ||
        !arena.Add(shape.rows, sizeof(uint32_t), &f.order) ||
        !arena.Add(shape.rows, sizeof(uint32_t), &f.partition) ||
        (!frontier && (!arena.Add(shape.predictors * 256, sizeof(uint32_t), &f.parent) ||
                      !arena.Add(shape.predictors, sizeof(uint32_t), &f.parent_extra) ||
                      !arena.Add(bins * 256, sizeof(uint32_t), &f.histogram) ||
                      !arena.Add(bins, sizeof(uint32_t), &f.extra) ||
                      !arena.Add(bins, sizeof(uint32_t), &f.occupancy) ||
                      !arena.Add(bins * 2, sizeof(gjxl_histogram_fitting::SideWinner), &f.sides))) ||
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
  if (frontier) {
    auto &p = plan.frontier;
    p.fits = static_cast<uint32_t>(plan.fits);
    p.queue_capacity = static_cast<uint32_t>(queue_capacity);
    Arena scratch;
    auto &s = p.slot;
    if (tiled) {
      auto &t = plan.tiled;
      if (!scratch.Add(1, sizeof(resident::TiledSlot), &s.state) ||
          !scratch.Add(maximum_predictors * 256, sizeof(uint32_t), &t.parent) ||
          !scratch.Add(maximum_predictors, sizeof(uint32_t), &t.parent_extra) ||
          !scratch.Add(maximum_properties * 256, sizeof(uint32_t), &t.occupancy) ||
          !scratch.Add(maximum_properties * 256, sizeof(uint32_t), &t.bin_map) ||
          !scratch.Add(maximum_properties, sizeof(resident::TiledProperty), &t.properties))
        return Status::OutOfMemory("Tiled frontier summaries overflow");
    } else if (!scratch.Add(1, sizeof(resident::State), &s.state) ||
        !scratch.Add(maximum_predictors * 256, sizeof(uint32_t), &s.parent) ||
        !scratch.Add(maximum_predictors, sizeof(uint32_t), &s.parent_extra) ||
        !scratch.Add(maximum_bins * 256, sizeof(uint32_t), &s.histogram) ||
        !scratch.Add(maximum_bins, sizeof(uint32_t), &s.extra) ||
        !scratch.Add(maximum_bins, sizeof(uint32_t), &s.occupancy) ||
        !scratch.Add(maximum_bins * 2, sizeof(gjxl_histogram_fitting::SideWinner), &s.sides))
      return Status::OutOfMemory("Frontier scratch overflow");
    s.bytes = static_cast<uint32_t>((scratch.bytes + 255) & ~size_t{255});
    size_t available = std::min(input.maximum_scratch_bytes, size_t{UINT32_MAX});
    if (limits) available = std::min({available, limits->maximum_scratch_bytes, limits->maximum_buffer_bytes});
    // Reserve enough pool space for the worst legal single node. Additional
    // nodes retain summaries while waiting for this pool, with guaranteed
    // progress and no data-dependent host allocation or resubmission.
    const size_t task_bytes = tiled ? sizeof(resident::TiledTask) * (1 + maximum_properties) + sizeof(uint32_t) : 0;
    const size_t reserve = tiled ? maximum_histogram + 3 * 255 : 0;
    if (available < reserve) return Status::Unsupported("Tiled frontier scratch cannot hold one histogram");
    p.active_nodes = static_cast<uint32_t>(std::min({input.maximum_active_nodes, queue_capacity,
        (available - reserve) / (s.bytes + task_bytes)}));
    if (!p.active_nodes) return Status::Unsupported("Frontier scratch cannot hold one node");
    plan.scratch_bytes = p.active_nodes * size_t(s.bytes);
    if (!arena.Add(1, tiled ? sizeof(resident::TiledControl) : sizeof(resident::FrontierControl), &p.control) ||
        !arena.Add(queue_capacity, sizeof(resident::NodeRef), &p.queue) ||
        !arena.Add(p.active_nodes, s.bytes, &p.scratch) ||
        !plan.publication.AddVector<DecisionNode>(plan.maximum_nodes, VectorCapacityPolicy::kFreshExact) ||
        !plan.publication.AddVector<std::pair<TreeIndex, TreeIndex>>(kMaximumTreeDepth + 1, VectorCapacityPolicy::kFreshExact))
      return Status::OutOfMemory("Frontier storage overflow");
    if (tiled) {
      auto &t = plan.tiled;
      t.frontier = p;
      const size_t tasks_begin = arena.bytes;
      if (!arena.Add(p.active_nodes, sizeof(resident::TiledTask), &t.row_tasks) ||
          !arena.Add(p.active_nodes * maximum_properties, sizeof(resident::TiledTask), &t.jobs) ||
          !arena.Add(p.active_nodes, sizeof(uint32_t), &t.selected))
        return Status::OutOfMemory("Tiled frontier tasks overflow");
      plan.scratch_bytes += ((arena.bytes + 255) & ~size_t{255}) - tasks_begin;
      t.histogram_bytes = static_cast<uint32_t>(std::min(available - plan.scratch_bytes,
          p.active_nodes * maximum_histogram) & ~size_t{3});
      if (t.histogram_bytes < maximum_histogram)
        return Status::Unsupported("Tiled frontier scratch cannot guarantee progress");
      if (!arena.Add(t.histogram_bytes, 1, &t.histogram))
        return Status::OutOfMemory("Tiled frontier histogram overflow");
      plan.scratch_bytes += t.histogram_bytes;
    }
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
      !plan.working.Add(plan.prepared.preparation) ||
      !plan.working.Add(plan.publication))
    return Status::OutOfMemory("Resident histogram working bound overflow");
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::histogram_gpu_internal
