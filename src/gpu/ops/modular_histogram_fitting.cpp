// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_histogram_fitting.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace gjxl::histogram_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
namespace wire = gjxl_histogram_fitting;
static_assert(kHistogramPredictors == 14 && kPropertyCount == 16 &&
              kHistogramStaticProperties == 2);
static_assert(sizeof(HistogramResidualToken) == 2 &&
              offsetof(HistogramResidualToken, extra_bits) == 1);
static_assert(sizeof(wire::FitDescriptor) == 336 &&
              sizeof(wire::NodeDescriptor) == 48);
static_assert(sizeof(wire::Candidate) == 32 &&
              sizeof(wire::PropertyResult) == 144);
static_assert(sizeof(wire::CandidateCosts) == 8 &&
              sizeof(wire::SideWinner) == 12);
static_assert(offsetof(wire::Candidate, left_cost) == 16 &&
              offsetof(wire::Candidate, valid) == 24);
static_assert(offsetof(wire::PropertyResult, winners) == 16 &&
              offsetof(wire::NodeDescriptor, predictor_penalties_offset) == 40);
static_assert(std::is_trivially_copyable_v<HistogramFittingShape> &&
              std::is_trivially_copyable_v<wire::PropertyResult>);
namespace {
Status Invalid() {
  return Status::InvalidArgument("Invalid GPU histogram fitting contract");
}
Status Unsupported() {
  return Status::Unsupported("GPU histogram fitting exceeds admission limits");
}
Status Overflow() {
  return Status::InvalidArgument("GPU histogram fitting storage overflow");
}
bool ValidLimits(const HistogramFittingDeviceLimits *l) {
  return !l || (l->maximum_fits && l->maximum_rows && l->maximum_nodes &&
                l->maximum_buffer_bytes && l->maximum_scratch_bytes);
}
bool BufferFits(size_t bytes, const HistogramFittingDeviceLimits *l) {
  return bytes <= UINT32_MAX && bytes <= size_t{PTRDIFF_MAX} &&
         (!l || bytes <= l->maximum_buffer_bytes);
}
struct Arena {
  size_t bytes = 0;
  bool Align() {
    if (bytes > SIZE_MAX - (kHistogramFittingAlignment - 1))
      return false;
    bytes = (bytes + kHistogramFittingAlignment - 1) &
            ~(kHistogramFittingAlignment - 1);
    return true;
  }
  bool Add(size_t count, size_t stride, size_t *offset) {
    if (!Align() || (count && stride > (SIZE_MAX - bytes) / count))
      return false;
    *offset = bytes;
    bytes += count * stride;
    return true;
  }
};
Status ValidateShape(const HistogramFittingShape &s) {
  if (!s.predictors || s.predictors > 14 || !s.properties ||
      s.properties > 16 || s.static_properties > 2 ||
      s.static_properties > s.properties ||
      s.population > kHistogramMaximumSamples || s.rows > s.population ||
      s.population > uint64_t{s.rows} * UINT16_MAX)
    return Invalid();
  std::array<bool, 14> predictors{};
  std::array<bool, 16> properties{};
  for (size_t p = 0; p < s.predictors; ++p) {
    const auto id = size_t(s.predictor_ids[p]);
    if (id >= predictors.size() || predictors[id])
      return Invalid();
    predictors[id] = true;
  }
  for (size_t p = 0; p < s.properties; ++p) {
    const size_t id = s.property_ids[p];
    const bool stat = p < s.static_properties;
    if (id >= properties.size() || properties[id] ||
        (stat ? id != p : id < 2) || !s.bins[p] ||
        s.bins[p] > (stat ? kHistogramMaximumStaticBins : 256))
      return Invalid();
    properties[id] = true;
  }
  // Empty native fits are valid, but the workflow omits them from GPU batches.
  if (!s.rows || s.rows > kMaximumHistogramFittingRows)
    return Unsupported();
  return Status::Ok();
}
bool ParentPresent(const HistogramFittingShape &s, Predictor parent) {
  return std::find(s.predictor_ids.begin(),
                   s.predictor_ids.begin() + s.predictors,
                   parent) != s.predictor_ids.begin() + s.predictors;
}
Status Scratch(size_t bins, size_t predictors, size_t symbols, size_t shards,
               HistogramFittingScratchLayout *out) {
  HistogramFittingScratchLayout p;
  Arena a;
  // Validated dimensions bound these counts below 2^28, including static bins.
  const size_t cells = bins * predictors * symbols, extra = bins * predictors;
  if (!a.Add(cells, sizeof(uint32_t) * shards, &p.shard_histograms_offset) ||
      !a.Add(extra, sizeof(uint32_t) * shards, &p.shard_extra_offset) ||
      !a.Add(bins, sizeof(uint32_t) * shards, &p.shard_occupancy_offset) ||
      !a.Add(cells, sizeof(uint32_t), &p.histograms_offset) ||
      !a.Add(extra, sizeof(uint32_t), &p.extra_offset) ||
      !a.Add(bins, sizeof(uint32_t), &p.occupancy_offset) ||
      !a.Add(extra, sizeof(wire::CandidateCosts), &p.candidate_costs_offset) ||
      !a.Add(bins, 2 * sizeof(wire::SideWinner), &p.side_winners_offset) ||
      !a.Align())
    return Overflow();
  p.bytes = a.bytes;
  *out = p;
  return Status::Ok();
}
} // namespace

Status ResolveHistogramFittingSource(const HistogramFittingSource &source,
                                     HistogramFittingShape *out) {
  if (!out || !source.samples)
    return Invalid();
  HistogramTreeLearningOptions options;
  options.maximum_samples = kHistogramMaximumSamples;
  if (auto s = ValidateHistogramTrainingSamples(*source.samples, options);
      !s.ok())
    return s;
  const auto &samples = *source.samples;
  HistogramFittingShape p;
  p.source_id = source.source_id;
  p.rows = samples.weights.size();
  p.population = samples.total_samples;
  p.predictors = samples.predictor_count;
  p.properties = samples.property_count;
  p.static_properties = samples.static_property_count;
  p.predictor_ids = samples.predictors;
  p.property_ids = samples.properties;
  for (size_t i = 0; i < p.properties; ++i)
    p.bins[i] = samples.thresholds[i].size() + 1;
  if (auto s = ValidateShape(p); !s.ok())
    return s;
  *out = p;
  return Status::Ok();
}

Status ComputeHistogramFittingColumnLayout(const HistogramFittingShape &shape,
                                           HistogramFittingColumnLayout *out) {
  if (!out)
    return Invalid();
  if (auto s = ValidateShape(shape); !s.ok())
    return s;
  HistogramFittingColumnLayout p;
  Arena a;
  if (!a.Add(shape.rows, sizeof(uint16_t), &p.weights_offset))
    return Overflow();
  for (size_t i = 0; i < shape.predictors; ++i)
    if (!a.Add(shape.rows, sizeof(HistogramResidualToken),
               &p.residual_offsets[i]))
      return Overflow();
  for (size_t i = 0; i < shape.properties; ++i)
    if (!a.Add(shape.rows,
               i < shape.static_properties ? sizeof(uint32_t) : sizeof(uint8_t),
               &p.property_offsets[i]))
      return Overflow();
  if (!a.Align())
    return Overflow();
  if (!BufferFits(a.bytes, nullptr))
    return Unsupported();
  p.bytes = a.bytes;
  *out = p;
  return Status::Ok();
}

Status ComputeHistogramFittingPreparationPlan(
    std::span<const HistogramFittingShape> shapes,
    HistogramFittingPreparationPlan *out,
    const HistogramFittingDeviceLimits *limits) {
  if (!out || shapes.empty() || !ValidLimits(limits))
    return Invalid();
  if (shapes.size() > kMaximumHistogramFittingFits ||
      (limits && shapes.size() > limits->maximum_fits))
    return Unsupported();
  HistogramFittingPreparationPlan p;
  p.fits = shapes.size();
  p.descriptors_bytes = p.fits * sizeof(wire::FitDescriptor);
  Arena a{p.descriptors_bytes};
  if (!a.Align())
    return Overflow();
  p.columns_offset = a.bytes;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto &shape = shapes[i];
    if (i && shapes[i - 1].source_id >= shape.source_id)
      return Invalid();
    HistogramFittingColumnLayout columns;
    if (auto s = ComputeHistogramFittingColumnLayout(shape, &columns); !s.ok())
      return s;
    if (shape.rows > kMaximumHistogramFittingRows - p.rows)
      return Unsupported();
    p.rows += shape.rows;
    size_t offset;
    if (!a.Add(columns.bytes, 1, &offset))
      return Overflow();
  }
  p.storage_bytes = a.bytes;
  p.host_shapes_bytes = shapes.size() * sizeof(HistogramFittingShape);
  if (!BufferFits(p.storage_bytes, limits) ||
      (limits && p.rows > limits->maximum_rows))
    return Unsupported();
  if (!p.staging.AddVector<std::byte>(p.storage_bytes,
                                      VectorCapacityPolicy::kFreshExact) ||
      !p.row_order.AddVector<uint32_t>(p.rows,
                                       VectorCapacityPolicy::kFreshExact) ||
      !p.retained.Add({p.storage_bytes, p.storage_bytes}) ||
      !p.retained.AddVector<HistogramFittingShape>(
          p.fits, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.preparation = p.retained;
  if (!p.preparation.Add(p.staging))
    return Overflow();
  *out = p;
  return Status::Ok();
}

Status
ValidateHistogramFittingSources(std::span<const HistogramFittingSource> sources,
                                HistogramFittingPreparationPlan *out,
                                const HistogramFittingDeviceLimits *limits) {
  if (!out || sources.empty() || !ValidLimits(limits))
    return Invalid();
  if (sources.size() > kMaximumHistogramFittingFits)
    return Unsupported();
  std::array<HistogramFittingShape, kMaximumHistogramFittingFits> shapes;
  for (size_t i = 0; i < sources.size(); ++i)
    if (auto s = ResolveHistogramFittingSource(sources[i], &shapes[i]); !s.ok())
      return s;
  return ComputeHistogramFittingPreparationPlan(
      std::span(shapes).first(sources.size()), out, limits);
}

Status ComputeHistogramFittingPredictorPenalties(
    std::span<const Predictor> predictors, Predictor parent,
    float change_penalty, std::span<float> out) {
  if (predictors.empty() || predictors.size() > kHistogramPredictors ||
      out.size() != predictors.size() || !std::isfinite(change_penalty) ||
      change_penalty < 0 || change_penalty > 8 ||
      std::find(predictors.begin(), predictors.end(), parent) ==
          predictors.end())
    return Invalid();
  std::array<bool, kHistogramPredictors> seen{};
  std::array<float, kHistogramPredictors> result{};
  for (size_t i = 0; i < predictors.size(); ++i) {
    const auto pred = predictors[i];
    const size_t id = size_t(pred);
    if (id >= seen.size() || seen[id])
      return Invalid();
    seen[id] = true;
    float penalty = 0;
    if (pred != parent && parent != Predictor::kWeighted)
      penalty = change_penalty;
    // Preserve the CPU evaluator's literal types and rounding, including
    // negative Zero bias. Side selection compares cost+penalty, but exported
    // candidate costs exclude this penalty, exactly as EvaluateProperty does.
    if (pred == Predictor::kWeighted)
      penalty += 1e-8;
    if (pred == Predictor::kZero)
      penalty -= 1e-8;
    result[i] = penalty;
  }
  std::copy_n(result.begin(), predictors.size(), out.begin());
  return Status::Ok();
}

Status ComputeHistogramFittingEvaluationPlan(
    std::span<const HistogramFittingShape> shapes,
    std::span<const HistogramFittingNodeShape> nodes,
    HistogramFittingExecutionOptions options,
    HistogramFittingEvaluationPlan *out,
    const HistogramFittingDeviceLimits *limits) {
  if (!out || nodes.empty() || !options.histogram_shards ||
      options.histogram_shards > kMaximumHistogramFittingShards ||
      !options.maximum_active_properties || !options.maximum_scratch_bytes)
    return Invalid();
  HistogramFittingPreparationPlan prepared;
  if (auto s =
          ComputeHistogramFittingPreparationPlan(shapes, &prepared, limits);
      !s.ok())
    return s;
  if (nodes.size() > shapes.size())
    return Invalid();
  if (limits && nodes.size() > limits->maximum_nodes)
    return Unsupported();
  HistogramFittingEvaluationPlan p;
  p.prepared_bytes = prepared.retained.retained_bytes;
  p.row_order_bytes = prepared.row_order.retained_bytes;
  p.nodes = nodes.size();
  p.histogram_shards = options.histogram_shards;
  size_t parent_counts = 0, parent_extra = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    if (node.fit >= shapes.size() || (i && nodes[i - 1].fit >= node.fit))
      return Invalid();
    const auto &s = shapes[node.fit];
    if (!node.rows || node.rows > s.rows || !node.symbols ||
        node.symbols > 256 || node.symbols % 4 ||
        node.inherited_constants >> s.properties ||
        !ParentPresent(s, node.parent_predictor) ||
        !std::isfinite(node.change_predictor_penalty) ||
        node.change_predictor_penalty < 0 || node.change_predictor_penalty > 8)
      return Invalid();
    p.rows += node.rows; // Unique fits and prepared-row cap bound all sums.
    p.properties += s.properties;
    parent_counts += s.predictors * node.symbols;
    parent_extra += s.predictors;
    for (size_t prop = 0; prop < s.properties; ++prop) {
      if ((node.inherited_constants & (uint32_t{1} << prop)) ||
          s.bins[prop] == 1)
        continue;
      ++p.tasks;
      p.maximum_bins = std::max(p.maximum_bins, s.bins[prop]);
      p.maximum_predictors = std::max(p.maximum_predictors, s.predictors);
      p.maximum_symbols = std::max(p.maximum_symbols, node.symbols);
    }
  }
  p.descriptors_bytes = p.nodes * sizeof(wire::NodeDescriptor);
  Arena a{p.descriptors_bytes};
  if (!a.Add(p.rows, sizeof(uint32_t), &p.indices_offset) ||
      !a.Add(parent_counts, sizeof(int32_t), &p.parent_counts_offset) ||
      !a.Add(parent_extra, sizeof(uint32_t), &p.parent_extra_offset) ||
      !a.Add(parent_extra, sizeof(float), &p.predictor_penalties_offset) ||
      !a.Add(1, sizeof(uint32_t), &p.status_offset))
    return Overflow();
  p.dispatch_bytes = a.bytes;
  p.output_bytes = p.properties * sizeof(HistogramPropertyResult);
  if (p.tasks) {
    if (auto s =
            Scratch(p.maximum_bins, p.maximum_predictors, p.maximum_symbols,
                    options.histogram_shards, &p.scratch);
        !s.ok())
      return s;
    size_t available =
        std::min(options.maximum_scratch_bytes, size_t{UINT32_MAX});
    available = std::min(available, size_t{PTRDIFF_MAX});
    if (limits)
      available = std::min({available, limits->maximum_scratch_bytes,
                            limits->maximum_buffer_bytes});
    p.active_properties = std::min({p.tasks, options.maximum_active_properties,
                                    available / p.scratch.bytes});
    if (!p.active_properties)
      return Unsupported();
    p.scratch_bytes = p.active_properties * p.scratch.bytes;
  }
  if (!BufferFits(p.dispatch_bytes, limits) ||
      !BufferFits(p.output_bytes, limits) ||
      !BufferFits(p.scratch_bytes, limits))
    return Unsupported();
  if (!p.staging.AddVector<std::byte>(p.dispatch_bytes,
                                      VectorCapacityPolicy::kFreshExact) ||
      !p.retained.Add({p.output_bytes, p.output_bytes}))
    return Overflow();
  p.evaluation = p.retained;
  if (!p.evaluation.Add({p.dispatch_bytes, p.dispatch_bytes}) ||
      !p.evaluation.Add({p.scratch_bytes, p.scratch_bytes}) ||
      !p.evaluation.Add(p.staging))
    return Overflow();
  *out = p;
  return Status::Ok();
}

Status
ComputeHistogramFittingWorkingSetBound(const HistogramFittingPreparationPlan &p,
                                       const HistogramFittingEvaluationPlan &e,
                                       HostStorageBound *out) {
  if (!out || !p.fits || !e.nodes || e.nodes > p.fits || e.rows > p.rows ||
      e.prepared_bytes != p.retained.retained_bytes ||
      e.row_order_bytes != p.row_order.retained_bytes)
    return Invalid();
  HostStorageBound retained = p.retained, preparation = p.preparation,
                   evaluation = p.retained;
  if (!retained.Add(p.row_order) || !retained.Add(e.retained) ||
      !preparation.Add(p.row_order) || !evaluation.Add(p.row_order) ||
      !evaluation.Add(e.evaluation))
    return Overflow();
  const size_t peak = std::max(preparation.peak_bytes, evaluation.peak_bytes);
  if (retained.retained_bytes > peak)
    return Invalid();
  *out = {retained.retained_bytes, peak};
  return Status::Ok();
}

Status
ValidateHistogramFittingInput(const HistogramFittingInput &input,
                              BackendId backend,
                              HistogramFittingEvaluationPlan *out,
                              const HistogramFittingDeviceLimits *limits) {
  if (!out || !backend || !input.prepared ||
      input.prepared->backend_id() != backend || input.nodes.empty())
    return Invalid();
  if (input.nodes.size() > kMaximumHistogramFittingFits)
    return Unsupported();
  std::array<HistogramFittingNodeShape, kMaximumHistogramFittingFits> nodes;
  for (size_t i = 0; i < input.nodes.size(); ++i)
    nodes[i] = input.nodes[i].shape;
  HistogramFittingEvaluationPlan p;
  const auto shapes = input.prepared->shapes();
  if (auto s = ComputeHistogramFittingEvaluationPlan(
          shapes, std::span(nodes).first(input.nodes.size()), input.options, &p,
          limits);
      !s.ok())
    return s;
  for (const auto &input_node : input.nodes) {
    const auto &n = input_node.shape;
    const auto &s = shapes[n.fit];
    if (input_node.rows.size() != n.rows ||
        input_node.parent_counts.size() != s.predictors * n.symbols ||
        input_node.parent_extra_bits.size() != s.predictors)
      return Invalid();
    uint64_t population = 0;
    size_t maximum_symbol = 0;
    for (size_t pred = 0; pred < s.predictors; ++pred) {
      uint64_t total = 0;
      for (size_t sym = 0; sym < n.symbols; ++sym) {
        const int32_t count = input_node.parent_counts[pred * n.symbols + sym];
        if (count < 0)
          return Invalid();
        total += static_cast<uint32_t>(count);
        if (count)
          maximum_symbol = std::max(maximum_symbol, sym);
      }
      if (!total || total > s.population || total < n.rows ||
          (pred && population != total) ||
          input_node.parent_extra_bits[pred] > total * 63)
        return Invalid();
      population = total;
    }
    if (((maximum_symbol + 4) & ~size_t{3}) != n.symbols)
      return Invalid();
  }
  *out = p;
  return Status::Ok();
}

Status CheckHistogramFittingDeviceStatus(uint32_t status) {
  constexpr uint32_t known = wire::kRowIndexError | wire::kPopulationError |
                             wire::kParentSummaryError | wire::kArithmeticError;
  if (status & ~known)
    return Status::Internal("Unknown GPU histogram fitting status");
  if (status & wire::kRowIndexError)
    return Status::InvalidArgument("GPU histogram row index is invalid");
  if (status & wire::kPopulationError)
    return Status::InvalidArgument("GPU histogram population exceeds bounds");
  if (status & wire::kParentSummaryError)
    return Status::InvalidArgument("GPU histogram parent summary differs");
  if (status & wire::kArithmeticError)
    return Status::Internal("GPU histogram arithmetic invariant failed");
  return Status::Ok();
}
} // namespace gjxl::histogram_gpu_internal
