// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_histogram_fitting.h"
#include <algorithm>
#include <bit>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::histogram_gpu_internal;
using namespace gjxl::resource_budget_internal;
namespace wire = gjxl_histogram_fitting;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
struct NoAllocation {
  NoAllocation() { ArmNextManagedHostAllocationFailureForTest(); }
  ~NoAllocation() { DisarmManagedHostAllocationFailureForTest(); }
  void CheckPending() {
    Check(ManagedHostAllocationFailurePendingForTest(),
          "Contract allocated managed storage");
  }
};
HistogramFittingShape Simple(size_t rows = 64, uint32_t id = 7) {
  HistogramFittingShape s;
  s.source_id = id;
  s.rows = s.population = rows;
  s.predictors = s.properties = 1;
  s.predictor_ids[0] = Predictor::kGradient;
  s.property_ids[0] = 9;
  s.bins[0] = 2;
  return s;
}
HistogramFittingNodeShape Node(size_t fit, size_t rows, size_t symbols = 4) {
  HistogramFittingNodeShape n;
  n.fit = fit;
  n.rows = rows;
  n.symbols = symbols;
  n.parent_predictor = Predictor::kGradient;
  n.change_predictor_penalty = 800.0f / 189.0f;
  return n;
}
HistogramTrainingSamples Samples() {
  HistogramTrainingSamples s;
  s.predictor_count = 2;
  s.property_count = 4;
  s.static_property_count = 2;
  s.predictors[0] = Predictor::kWeighted;
  s.predictors[1] = Predictor::kGradient;
  s.properties = {0, 1, 9, 15};
  for (size_t p = 0; p < 4; ++p)
    for (int32_t i = 0; i < int32_t(2 + p); ++i)
      s.thresholds[p].push_back(i);
  for (size_t i = 0; i < 64; ++i) {
    s.weights.push_back(1 + i % 3);
    s.total_samples += s.weights.back();
    for (size_t p = 0; p < 2; ++p) {
      s.residuals[p].push_back({uint8_t((i + p) % 32), uint8_t(i % 6)});
      s.static_values[p].push_back(i % (3 + p));
      s.values[p].push_back(i % (5 + p));
    }
  }
  return s;
}
struct Rows final : PreparedGpuHistogramRows {
  BackendId id = 19;
  std::array<HistogramFittingShape, 2> values;
  BackendId backend_id() const noexcept override { return id; }
  std::span<const HistogramFittingShape> shapes() const noexcept override {
    return values;
  }
};

void ExactLayoutAndLifetimes() {
  auto shape = Simple();
  auto node = Node(0, 64);
  HistogramFittingColumnLayout columns;
  HistogramFittingPreparationPlan p;
  HistogramFittingEvaluationPlan e;
  HostStorageBound working;
  NoAllocation guard;
  Ok(ComputeHistogramFittingColumnLayout(shape, &columns));
  Check(columns.weights_offset == 0 && columns.residual_offsets[0] == 256 &&
            columns.property_offsets[0] == 512 && columns.bytes == 768,
        "Compact native columns or alignment changed");
  Ok(ComputeHistogramFittingPreparationPlan({&shape, 1}, &p));
  Check(p.descriptors_bytes == 336 && p.columns_offset == 512 &&
            p.storage_bytes == 1280 &&
            p.host_shapes_bytes == sizeof(HistogramFittingShape) &&
            p.row_order.peak_bytes == 256 &&
            p.retained.retained_bytes == 1280 + sizeof(HistogramFittingShape) &&
            p.preparation.peak_bytes == 2560 + sizeof(HistogramFittingShape),
        "Preparation omitted descriptors, staging, shapes or separate "
        "row-order storage");
  HistogramFittingExecutionOptions options{1, 1, 2048};
  Ok(ComputeHistogramFittingEvaluationPlan({&shape, 1}, {&node, 1}, options,
                                           &e));
  const auto &s = e.scratch;
  Check(e.nodes == 1 && e.rows == 64 && e.properties == 1 && e.tasks == 1 &&
            e.indices_offset == 256 && e.parent_counts_offset == 512 &&
            e.parent_extra_offset == 768 &&
            e.predictor_penalties_offset == 1024 && e.tasks_offset == 1280 &&
            e.status_offset == 1536 && e.dispatch_bytes == 1540 &&
            e.output_bytes == 144 && e.scratch_bytes == 2048 &&
            e.evaluation.peak_bytes == 5272,
        "Evaluation arena omitted live storage");
  Check(s.shard_histograms_offset == 0 && s.shard_extra_offset == 256 &&
            s.shard_occupancy_offset == 512 && s.histograms_offset == 768 &&
            s.extra_offset == 1024 && s.occupancy_offset == 1280 &&
            s.candidate_costs_offset == 1536 && s.side_winners_offset == 1792 &&
            s.bytes == 2048,
        "Histogram, prefix or ordered candidate scratch changed");
  Ok(ComputeHistogramFittingWorkingSetBound(p, e, &working));
  Check(working.retained_bytes == 1680 + sizeof(HistogramFittingShape) &&
            working.peak_bytes == 6808 + sizeof(HistogramFittingShape),
        "Combined peak lost prepared columns, row order or output overlap");
  const auto old = working;
  auto wrong = e;
  ++wrong.prepared_bytes;
  Check(!ComputeHistogramFittingWorkingSetBound(p, wrong, &working).ok() &&
            working == old,
        "Unrelated evaluation plan published a working set");
  wrong = e;
  ++wrong.row_order_bytes;
  Check(!ComputeHistogramFittingWorkingSetBound(p, wrong, &working).ok() &&
            working == old,
        "Row-order mismatch published a working set");
  auto overflow = p;
  overflow.preparation = {SIZE_MAX, SIZE_MAX};
  Check(!ComputeHistogramFittingWorkingSetBound(overflow, e, &working).ok() &&
            working == old,
        "Working set arithmetic wrapped");
  Check(!ComputeHistogramFittingWorkingSetBound(p, e, nullptr).ok(),
        "Null working set accepted");
  guard.CheckPending();
}

void ShapesAndLimits() {
  std::array shapes{Simple(64, 7), Simple(65, 12)};
  shapes[1].predictors = 2;
  shapes[1].predictor_ids[1] = Predictor::kWeighted;
  shapes[1].properties = 3;
  shapes[1].static_properties = 1;
  shapes[1].property_ids = {0, 9, 15};
  shapes[1].bins = {5, 2, 8};
  std::array nodes{Node(0, 32, 8), Node(1, 16, 32)};
  nodes[1].inherited_constants = 1;
  HistogramFittingPreparationPlan prep;
  HistogramFittingEvaluationPlan plan;
  NoAllocation guard;
  Ok(ComputeHistogramFittingPreparationPlan(shapes, &prep));
  Ok(ComputeHistogramFittingEvaluationPlan(shapes, nodes, {}, &plan));
  Check(prep.rows == 129 && plan.rows == 48 && plan.properties == 4 &&
            plan.tasks == 3 && plan.maximum_bins == 8 &&
            plan.maximum_predictors == 2 && plan.maximum_symbols == 32 &&
            plan.active_properties == 3 &&
            plan.scratch_bytes == 3 * plan.scratch.bytes,
        "Independent fit/property batch shape differs");
  HistogramFittingExecutionOptions one;
  one.maximum_scratch_bytes = plan.scratch.bytes;
  HistogramFittingEvaluationPlan limited;
  Ok(ComputeHistogramFittingEvaluationPlan(shapes, nodes, one, &limited));
  Check(limited.active_properties == 1 &&
            limited.scratch_bytes == plan.scratch.bytes &&
            limited.output_bytes == plan.output_bytes,
        "Scratch pressure changed output coverage");
  const auto saved = limited;
  --one.maximum_scratch_bytes;
  Check(ComputeHistogramFittingEvaluationPlan(shapes, nodes, one, &limited)
                    .code() == StatusCode::kUnsupported &&
            limited == saved,
        "Insufficient scratch changed the prior plan");
  HistogramFittingDeviceLimits limits{
      2, 129, 2,
      std::max({prep.storage_bytes, plan.dispatch_bytes, plan.output_bytes,
                plan.scratch_bytes}),
      plan.scratch_bytes};
  Ok(ComputeHistogramFittingPreparationPlan(shapes, &prep, &limits));
  Ok(ComputeHistogramFittingEvaluationPlan(shapes, nodes, {}, &plan, &limits));
  const auto p_saved = prep;
  const auto e_saved = plan;
  auto rejects_p = [&](auto mutate) {
    auto bad = limits;
    mutate(bad);
    Check(!ComputeHistogramFittingPreparationPlan(shapes, &prep, &bad).ok() &&
              prep == p_saved,
          "Preparation device limit was ignored or published");
  };
  rejects_p([](auto &l) { l.maximum_fits = 1; });
  rejects_p([](auto &l) { --l.maximum_rows; });
  rejects_p([&](auto &l) { l.maximum_buffer_bytes = prep.storage_bytes - 1; });
  rejects_p([](auto &l) { l.maximum_scratch_bytes = 0; });
  limits.maximum_nodes = 1;
  Check(ComputeHistogramFittingEvaluationPlan(shapes, nodes, {}, &plan, &limits)
                    .code() == StatusCode::kUnsupported &&
            plan == e_saved,
        "Node device limit was ignored");
  limits.maximum_nodes = 2;
  limits.maximum_scratch_bytes = plan.scratch.bytes - 1;
  Check(ComputeHistogramFittingEvaluationPlan(shapes, nodes, {}, &plan, &limits)
                    .code() == StatusCode::kUnsupported &&
            plan == e_saved,
        "Scratch device limit was ignored");
  auto reject_shape = [&](auto mutate) {
    auto bad = shapes;
    mutate(bad);
    Check(!ComputeHistogramFittingPreparationPlan(bad, &prep).ok() &&
              prep == p_saved,
          "Malformed prepared shape was accepted or published");
  };
  reject_shape([](auto &s) { s[0].rows = SIZE_MAX; });
  reject_shape([](auto &s) { s[0].population = 1; });
  reject_shape([](auto &s) {
    s[0].rows = 1;
    s[0].population = UINT16_MAX + 1u;
  });
  reject_shape([](auto &s) { s[0].predictors = 15; });
  reject_shape([](auto &s) { s[0].properties = 17; });
  reject_shape([](auto &s) { s[1].source_id = s[0].source_id; });
  reject_shape([](auto &s) { s[1].predictor_ids[1] = s[1].predictor_ids[0]; });
  reject_shape([](auto &s) { s[0].predictor_ids[0] = Predictor(14); });
  reject_shape([](auto &s) { s[1].property_ids[2] = s[1].property_ids[1]; });
  reject_shape([](auto &s) { s[1].property_ids[0] = 1; });
  reject_shape([](auto &s) { s[0].property_ids[0] = 0; });
  reject_shape([](auto &s) { s[0].bins[0] = 257; });
  reject_shape([](auto &s) { s[1].bins[0] = 65537; });
  reject_shape([](auto &s) { s[1].bins[0] = 0; });
  auto reject_node = [&](auto mutate) {
    auto bad = nodes;
    mutate(bad);
    Check(!ComputeHistogramFittingEvaluationPlan(shapes, bad, {}, &plan).ok() &&
              plan == e_saved,
          "Malformed node was accepted or published");
  };
  reject_node([](auto &n) { n[1].fit = 0; });
  reject_node([](auto &n) { n[1].fit = SIZE_MAX; });
  reject_node([](auto &n) { n[0].rows = 65; });
  reject_node([](auto &n) { n[0].rows = 0; });
  reject_node([](auto &n) { n[0].symbols = 7; });
  reject_node([](auto &n) { n[0].symbols = 260; });
  reject_node([](auto &n) { n[0].inherited_constants = 2; });
  reject_node([](auto &n) { n[0].parent_predictor = Predictor::kZero; });
  reject_node([](auto &n) { n[0].change_predictor_penalty = -1; });
  reject_node([](auto &n) { n[0].change_predictor_penalty = 9; });
  reject_node([](auto &n) {
    n[0].change_predictor_penalty = std::numeric_limits<float>::quiet_NaN();
  });
  for (auto options : {HistogramFittingExecutionOptions{0, 1, 4096},
                       HistogramFittingExecutionOptions{33, 1, 4096},
                       HistogramFittingExecutionOptions{1, 0, 4096},
                       HistogramFittingExecutionOptions{1, 1, 0}})
    Check(!ComputeHistogramFittingEvaluationPlan(shapes, nodes, options, &plan)
                  .ok() &&
              plan == e_saved,
          "Invalid execution options published a plan");
  Check(
      !ComputeHistogramFittingPreparationPlan({}, &prep).ok() &&
          prep == p_saved &&
          !ComputeHistogramFittingPreparationPlan(shapes, nullptr).ok() &&
          !ComputeHistogramFittingEvaluationPlan(shapes, {}, {}, &plan).ok() &&
          plan == e_saved &&
          !ComputeHistogramFittingEvaluationPlan(shapes, nodes, {}, nullptr)
               .ok(),
      "Empty batch or null plan accepted");
  guard.CheckPending();
}

void WideAndConstantProperties() {
  auto shape = Simple(65536);
  shape.predictors = 14;
  for (size_t i = 0; i < 14; ++i)
    shape.predictor_ids[i] = Predictor(i);
  shape.properties = 16;
  shape.static_properties = 2;
  std::iota(shape.property_ids.begin(), shape.property_ids.end(), uint8_t{0});
  shape.bins.fill(256);
  shape.bins[0] = 65536;
  auto node = Node(0, shape.rows, 256);
  HistogramFittingEvaluationPlan plan;
  NoAllocation guard;
  Check(
      ComputeHistogramFittingEvaluationPlan({&shape, 1}, {&node, 1}, {}, &plan)
              .code() == StatusCode::kUnsupported,
      "Wide static property silently truncated or exceeded scratch");
  node.inherited_constants = 1;
  Ok(ComputeHistogramFittingEvaluationPlan({&shape, 1}, {&node, 1}, {}, &plan));
  Check(plan.tasks == 15 && plan.maximum_bins == 256 &&
            plan.maximum_predictors == 14 &&
            plan.scratch_bytes <= 64 * 1024 * 1024,
        "Inherited static constant consumed wide scratch");
  shape.bins.fill(1);
  shape.bins[0] = 65536;
  node.inherited_constants = 0;
  HistogramFittingExecutionOptions wide{1, 1, size_t{3} << 30};
  Ok(ComputeHistogramFittingEvaluationPlan({&shape, 1}, {&node, 1}, wide,
                                           &plan));
  Check(plan.maximum_bins == 65536 && plan.active_properties == 1 &&
            plan.tasks == 1,
        "Native static bins cannot be represented under explicit admission");
  node.inherited_constants = 1;
  Ok(ComputeHistogramFittingEvaluationPlan({&shape, 1}, {&node, 1}, {}, &plan));
  Check(!plan.tasks && !plan.active_properties && !plan.scratch_bytes &&
            !plan.scratch.bytes && plan.properties == 16 &&
            plan.output_bytes == 16 * sizeof(HistogramPropertyResult),
        "All-constant evaluation lost result coverage or retained histogram "
        "scratch");
  auto max = shape;
  max.rows = kMaximumHistogramFittingRows;
  max.population = kHistogramMaximumSamples;
  HistogramFittingPreparationPlan p;
  Ok(ComputeHistogramFittingPreparationPlan({&max, 1}, &p));
  Check(p.storage_bytes < UINT32_MAX &&
            p.preparation.peak_bytes > p.retained.retained_bytes &&
            p.row_order.peak_bytes == max.rows * sizeof(uint32_t),
        "Large cohort bound wrapped");
  const auto saved = p;
  ++max.rows;
  max.population = max.rows;
  Check(ComputeHistogramFittingPreparationPlan({&max, 1}, &p).code() ==
                StatusCode::kUnsupported &&
            p == saved,
        "GPU row cap was ignored");
  guard.CheckPending();
}

void CohortMaximum() {
  std::array<HistogramFittingShape, kMaximumHistogramFittingFits + 1> shapes;
  std::array<HistogramFittingNodeShape, kMaximumHistogramFittingFits> nodes;
  for (size_t i = 0; i < shapes.size(); ++i)
    shapes[i] = Simple(32768, uint32_t(i));
  for (size_t i = 0; i < nodes.size(); ++i)
    nodes[i] = Node(i, 32768);
  HistogramFittingPreparationPlan p;
  HistogramFittingEvaluationPlan e;
  NoAllocation guard;
  auto allowed = std::span(shapes).first(nodes.size());
  Ok(ComputeHistogramFittingPreparationPlan(allowed, &p));
  Ok(ComputeHistogramFittingEvaluationPlan(allowed, nodes, {}, &e));
  Check(p.rows == kMaximumHistogramFittingRows && e.nodes == 256 &&
            e.rows == p.rows && e.active_properties == 64 && e.tasks == 256,
        "Full cohort lost ready nodes");
  const auto saved = p;
  Check(ComputeHistogramFittingPreparationPlan(shapes, &p).code() ==
                StatusCode::kUnsupported &&
            p == saved,
        "Fit limit failed atomically");
  ++shapes[0].rows;
  ++shapes[0].population;
  Check(ComputeHistogramFittingPreparationPlan(allowed, &p).code() ==
                StatusCode::kUnsupported &&
            p == saved,
        "Aggregate row limit failed atomically");
  guard.CheckPending();
}

void NativeSourcesAndSummaries() {
  auto samples = Samples();
  Rows owner;
  HistogramFittingSource source{11, &samples};
  Ok(ResolveHistogramFittingSource(source, &owner.values[0]));
  owner.values[1] = owner.values[0];
  owner.values[1].source_id = 23;
  std::array sources{source, HistogramFittingSource{23, &samples}};
  std::array<uint32_t, 64> indices;
  std::iota(indices.begin(), indices.end(), 0u);
  std::array<int32_t, 64> counts{};
  std::array<uint32_t, 2> extra{};
  for (size_t pred = 0; pred < 2; ++pred)
    for (size_t i = 0; i < 64; ++i) {
      counts[pred * 32 + samples.residuals[pred][i].symbol] +=
          samples.weights[i];
      extra[pred] += samples.residuals[pred][i].extra_bits * samples.weights[i];
    }
  std::array inputs{
      HistogramFittingNodeInput{Node(0, 64, 32), indices, counts, extra},
      HistogramFittingNodeInput{Node(1, 64, 32), indices, counts, extra}};
  HistogramFittingInput input{&owner, inputs, {}};
  HistogramFittingPreparationPlan p;
  HistogramFittingEvaluationPlan e;
  NoAllocation guard;
  Ok(ValidateHistogramFittingSources(sources, &p));
  Ok(ValidateHistogramFittingInput(input, owner.id, &e));
  Check(e.properties == 8 && p.fits == 2,
        "Native source/property order was lost");
  const auto saved = e;
  const auto saved_p = p;
  auto rejects = [&] {
    Check(ValidateHistogramFittingInput(input, owner.id, &e).code() ==
                  StatusCode::kInvalidArgument &&
              e == saved,
          "Invalid parent summary or ownership published a plan");
  };
  input.prepared = nullptr;
  rejects();
  input.prepared = &owner;
  Check(!ValidateHistogramFittingInput(input, owner.id + 1, &e).ok() &&
            e == saved && !ValidateHistogramFittingInput(input, 0, &e).ok() &&
            !ValidateHistogramFittingInput(input, owner.id, nullptr).ok(),
        "Backend/null contract ignored");
  const auto rows = inputs[0].rows;
  inputs[0].rows = rows.first(63);
  rejects();
  inputs[0].rows = rows;
  const auto c = inputs[0].parent_counts;
  inputs[0].parent_counts = c.first(63);
  rejects();
  inputs[0].parent_counts = c;
  const auto eb = inputs[0].parent_extra_bits;
  inputs[0].parent_extra_bits = eb.first(1);
  rejects();
  inputs[0].parent_extra_bits = eb;
  const int32_t old = counts[0];
  counts[0] = -1;
  rejects();
  counts[0] = old;
  ++counts[0];
  rejects();
  --counts[0];
  const auto old_counts = counts;
  counts.fill(0);
  counts[0] = counts[32] = int32_t(samples.total_samples);
  rejects(); // Padded symbol span must be the node's actual span.
  counts = old_counts;
  const auto old_extra = extra[0];
  extra[0] = UINT32_MAX;
  rejects();
  extra[0] = old_extra;
  // Descriptor validation deliberately defers index contents to checked device
  // validation; it must not scan or dereference the membership list here.
  indices[0] = UINT32_MAX;
  Ok(ValidateHistogramFittingInput(input, owner.id, &e));
  indices[0] = 0;
  const auto shape_saved = owner.values[0];
  samples.weights[0] = 0;
  Check(!ResolveHistogramFittingSource(source, &owner.values[0]).ok() &&
            owner.values[0] == shape_saved,
        "Zero native weight was accepted or published");
  samples.weights[0] = 1;
  samples.residuals[0][0].extra_bits = 64;
  Check(!ValidateHistogramFittingSources(sources, &p).ok() && p == saved_p,
        "Invalid native residual was accepted or published");
  samples.residuals[0][0].extra_bits = 0;
  samples.values[0][0] = 255;
  Check(!ValidateHistogramFittingSources(sources, &p).ok() && p == saved_p,
        "Invalid native property bin was accepted or published");
  samples.values[0][0] = 0;
  sources[1].source_id = 11;
  Check(!ValidateHistogramFittingSources(sources, &p).ok() && p == saved_p,
        "Duplicate canonical source ID was accepted");
  sources[1].source_id = 23;
  Ok(ValidateHistogramFittingSources(sources, &p));
  source.samples = nullptr;
  Check(!ResolveHistogramFittingSource(source, &owner.values[0]).ok() &&
            owner.values[0] == shape_saved,
        "Null source changed the shape");
  guard.CheckPending();
}

void PredictorPenalties() {
  std::array predictors{Predictor::kWeighted, Predictor::kGradient,
                        Predictor::kZero};
  std::array<float, 3> values;
  NoAllocation guard;
  Ok(ComputeHistogramFittingPredictorPenalties(predictors, Predictor::kWeighted,
                                               8, values));
  Check(std::bit_cast<uint32_t>(values[0]) == 0x322bcc77 && values[1] == 0 &&
            std::bit_cast<uint32_t>(values[2]) == 0xb22bcc77,
        "Weighted parent incorrectly penalized changing predictor");
  Ok(ComputeHistogramFittingPredictorPenalties(predictors, Predictor::kGradient,
                                               8, values));
  Check(values[0] == 8 && values[1] == 0 && values[2] == 8,
        "Predictor penalty was not rounded to native float");
  Ok(ComputeHistogramFittingPredictorPenalties(predictors, Predictor::kGradient,
                                               1e-8f, values));
  Check(std::bit_cast<uint32_t>(values[0]) == 0x32abcc77 && values[1] == 0 &&
            std::bit_cast<uint32_t>(values[2]) == 0xa48c230e,
        "Double-literal bias changed to float arithmetic");
  const auto saved = values;
  Check(!ComputeHistogramFittingPredictorPenalties(predictors, Predictor::kLeft,
                                                   1, values)
                .ok() &&
            values == saved,
        "Unknown parent changed penalty output");
  predictors[2] = predictors[0];
  Check(!ComputeHistogramFittingPredictorPenalties(
             predictors, Predictor::kGradient, 1, values)
                .ok() &&
            values == saved,
        "Duplicate predictor partially published penalties");
  predictors[2] = Predictor::kZero;
  for (float bad : {-1.0f, 9.0f, std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()})
    Check(!ComputeHistogramFittingPredictorPenalties(
               predictors, Predictor::kGradient, bad, values)
                  .ok() &&
              values == saved,
          "Invalid penalty published output");
  Check(
      !ComputeHistogramFittingPredictorPenalties({}, Predictor::kGradient, 1,
                                                 {})
              .ok() &&
          !ComputeHistogramFittingPredictorPenalties(
               predictors, Predictor::kGradient, 1, std::span(values).first(2))
               .ok() &&
          values == saved,
      "Invalid penalty result extent accepted");
  guard.CheckPending();
}

void StatusAndWireFormat() {
  static_assert(sizeof(wire::FitDescriptor) == 336 &&
                sizeof(wire::NodeDescriptor) == 48);
  static_assert(sizeof(wire::PropertyResult) == 144 &&
                alignof(wire::PropertyResult) == 4);
  static_assert(offsetof(wire::Candidate, left_cost) == 16 &&
                offsetof(wire::Candidate, valid) == 24);
  static_assert(wire::kStaticConstant == 0 && wire::kStatic == 1 &&
                wire::kNonstatic == 2 && wire::kNonstaticWithoutWeighted == 3);
  Ok(CheckHistogramFittingDeviceStatus(0));
  for (auto status : {wire::kRowIndexError, wire::kPopulationError,
                      wire::kParentSummaryError})
    Check(CheckHistogramFittingDeviceStatus(status).code() ==
              StatusCode::kInvalidArgument,
          "Invalid device input was treated as success");
  Check(CheckHistogramFittingDeviceStatus(wire::kArithmeticError).code() ==
                StatusCode::kInternal &&
            CheckHistogramFittingDeviceStatus(16).code() ==
                StatusCode::kInternal,
        "Device arithmetic/unknown status was ignored");
}
} // namespace

int main() try {
  ExactLayoutAndLifetimes();
  ShapesAndLimits();
  WideAndConstantProperties();
  CohortMaximum();
  NativeSourcesAndSummaries();
  PredictorPenalties();
  StatusAndWireFormat();
  std::cout << "GPU histogram fitting contract: layouts, weighted sources, "
               "summaries, "
               "cohorts, constants, device limits and failure-atomic planning "
               "passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
