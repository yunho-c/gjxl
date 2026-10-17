// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/modular_resident_histogram_fitting.h"
#include "modular_histogram_fitting_test_helpers.h"
#include <future>
#include <iostream>

namespace {
using namespace histogram_fitting_test;
size_t fits = 0, failures = 0;
ResidentHistogramExecutor executor = ResidentHistogramExecutor::kDepthFirst;
size_t active_nodes = 128, rounds = 16, active_peak = 0, same_fit_peak = 0, deferred_peak = 0;
ResidentHistogramInput Input(std::span<const ResidentHistogramRequest> requests, bool capture = false) {
  ResidentHistogramInput input{requests, capture};
  input.executor = executor;
  input.maximum_active_nodes = active_nodes;
  input.rounds_per_submission = rounds;
  return input;
}
DecisionTree Oracle(HistogramTrainingSamples samples, HistogramTreeLearningOptions options,
                    std::vector<uint32_t> *order) {
  HistogramTreeFitter fitter;
  Ok(fitter.Initialize(samples, options));
  while (!fitter.complete()) {
    Ok(fitter.Advance());
    if (!fitter.ready()) continue;
    std::array<HistogramPropertyEvaluation, 16> properties;
    auto view = std::span(properties).first(samples.property_count);
    Ok(fitter.EvaluateCpu(view)); Ok(fitter.Apply(view));
  }
  order->assign(fitter.row_order().begin(), fitter.row_order().end());
  DecisionTree result;
  Ok(fitter.Publish(&result));
  return result;
}
void Case(GpuBackend &gpu, const std::vector<HistogramTrainingSamples> &samples,
          HistogramTreeLearningOptions options = {}, bool local_histograms_only = false) {
  std::vector<ResidentHistogramRequest> requests;
  std::vector<DecisionTree> expected;
  std::vector<std::vector<uint32_t>> orders(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    requests.push_back({{uint32_t(i + 17), &samples[i]}, options});
    expected.push_back(Oracle(samples[i], options, &orders[i]));
    auto copy = samples[i]; DecisionTree ordinary;
    Ok(LearnHistogramTree(copy, &ordinary, options));
    Check(ordinary == expected.back(), "Ordinary/resumable CPU oracle differs");
    copy = samples[i];
    DecisionTree scalar_frontier;
    Ok(LearnHistogramTreeFrontierReference(copy, &scalar_frontier, options));
    Check(scalar_frontier == ordinary, "Scalar frontier/native CPU oracle differs");
  }
  const auto originals = samples;
  const auto input = Input(requests, true);
  ResidentHistogramPlan plan;
  Ok(ComputeResidentHistogramPlan(input, &plan));
  ResourceBudget budget(plan.working.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.TryReserve(plan.working.peak_bytes, &reservation));
  ResourceContextScope context({&reservation, ResourceClass::kPreparation});
  ResidentHistogramResult result;
  const auto before = gpu.stats().committed_submissions;
  Ok(QueryGpuModularHistogramFitting(gpu)->FitHistogramTrees(input, &result));
  Check(gpu.stats().committed_submissions == before + result.submissions && result.submissions,
        "Resident fitting submission count differs");
  if (executor != ResidentHistogramExecutor::kDepthFirst) {
    Check(result.maximum_active_nodes <= plan.frontier.active_nodes && result.frontier_rounds > 0,
          "Frontier exceeded its admitted active slots");
    Check(result.maximum_same_fit_nodes <= result.maximum_active_nodes &&
          result.maximum_active_nodes <= result.maximum_ready_nodes,
          "Frontier population counters are inconsistent");
    active_peak = std::max(active_peak, result.maximum_active_nodes);
    same_fit_peak = std::max(same_fit_peak, result.maximum_same_fit_nodes);
    deferred_peak = std::max(deferred_peak, result.maximum_deferred_nodes);
    if (executor == ResidentHistogramExecutor::kTiledFrontier) {
      Check(plan.scratch_bytes <= input.maximum_scratch_bytes &&
            result.maximum_histogram_bytes <= plan.tiled.histogram_bytes,
            "Tiled histogram pool exceeded its storage contract");
      if (local_histograms_only)
        Check(result.maximum_histogram_bytes == 0 && expected.front().size > 1,
              "Local-only split tree used a device histogram or did not split");
    }
  } else Check(result.submissions == 1, "Depth-first fitting added submissions");
  Check(result.trees.size() == expected.size() && result.row_offsets.size() == expected.size() + 1,
        "Resident fitting omitted results");
  size_t expected_nodes = 0;
  for (const auto &tree : expected) expected_nodes += tree.size;
  Check(result.visited_nodes == expected_nodes, "Resident fitting duplicated or omitted node visits");
  for (size_t i = 0; i < samples.size(); ++i) {
    if (result.trees[i] != expected[i]) {
      std::cerr << "fit " << i << " nodes GPU/CPU " << result.trees[i].size << '/' << expected[i].size << '\n';
      for (size_t n = 0; n < std::min(result.trees[i].size, expected[i].size); ++n)
        if (result.trees[i].nodes[n] != expected[i].nodes[n]) {
          const auto &a=result.trees[i].nodes[n], &b=expected[i].nodes[n];
          std::cerr << "node " << n << " property " << a.property << '/' << b.property << " split " << a.split << '/' << b.split << " predictor " << int(a.predictor) << '/' << int(b.predictor) << '\n'; break;
        }
      Check(false, "Resident tree differs from native CPU decisions");
    }
    const auto rows = std::span(result.row_order).subspan(result.row_offsets[i], orders[i].size());
    Check(std::equal(rows.begin(), rows.end(), orders[i].begin()), "Resident partition order differs from CPU swaps");
    Check(SameRows(samples[i], originals[i]), "Resident fit mutated source columns");
  }
  Check(budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes,
        "Resident fitting exceeded its storage plan");
  fits += samples.size();
  result = {};
  reservation.Reset();
  Check(!budget.snapshot().committed_bytes() && !budget.snapshot().total.backing_count,
        "Resident fitting leaked backing");
}
void Failures(GpuBackend &gpu) {
  auto samples = Fixture(0, 257, 2);
  ResidentHistogramRequest request{{91, &samples}, {}};
  request.learning.split_threshold = 0;
  auto *op = QueryGpuModularHistogramFitting(gpu);
  ResidentHistogramResult result;
  result.visited_nodes = 983;
  result.trees.resize(1);
  result.trees[0].nodes[0].predictor = Predictor::kZero;
  const auto intact = [&] {
    Check(result.visited_nodes == 983 && result.trees.size() == 1 &&
          result.trees[0].nodes[0].predictor == Predictor::kZero,
          "Failed resident fit changed published output");
    ++failures;
  };
  for (unsigned kind = 0; kind < 4; ++kind) {
    if (kind == 0) Ok(ArmNextMetalAllocationFailureForTest(gpu));
    if (kind == 1 || kind == 2) Ok(ArmNextMetalSubmissionFailureForTest(gpu, kind == 1, kind == 2));
    if (kind == 3) ArmNextManagedHostAllocationFailureForTest();
    const auto status = op->FitHistogramTrees(Input({&request, 1}), &result);
    DisarmManagedHostAllocationFailureForTest();
    Check(!status.ok(), "Injected resident failure was ignored"); intact();
  }
  for (unsigned limit = 0; limit < 2; ++limit) {
    request.learning.maximum_leaves = limit ? kMaximumTreeLeaves : 1;
    request.learning.maximum_depth = limit ? 1 : kMaximumTreeDepth;
    Check(op->FitHistogramTrees(Input({&request, 1}), &result).code() == StatusCode::kUnsupported,
          "Resident tree capacity/depth silently truncated"); intact();
  }
  request.learning = {};
  HistogramMultiplierRange multiplier{};
  multiplier.range = request.learning.static_range;
  request.learning.multipliers = {&multiplier, 1};
  const auto before = gpu.stats().committed_submissions;
  Check(op->FitHistogramTrees(Input({&request, 1}), &result).code() == StatusCode::kUnsupported,
        "Unsupported resident multiplier accepted"); intact();
  request.learning.multipliers = {};
  samples.static_values[0][0] = 999;
  Check(op->FitHistogramTrees(Input({&request, 1}), &result).code() == StatusCode::kInvalidArgument,
        "Invalid resident property bin accepted"); intact();
  Check(gpu.stats().committed_submissions == before, "Resident preflight submitted invalid work");
  samples.static_values[0][0] = 0;
  samples.thresholds[0].resize(256);
  std::iota(samples.thresholds[0].begin(), samples.thresholds[0].end(), 0);
  Check(op->FitHistogramTrees(Input({&request, 1}), &result).code() == StatusCode::kUnsupported,
        "Oversized resident static bins accepted"); intact();
  samples = Fixture(0, 257, 2);
  ResidentHistogramPlan plan;
  Ok(ComputeResidentHistogramPlan(Input({&request, 1}), &plan));
  {
    ResourceBudget budget(1); ResourceReservation reservation;
    Ok(budget.TryReserve(1, &reservation));
    ResourceContextScope context({&reservation, ResourceClass::kPreparation});
    Check(!op->FitHistogramTrees(Input({&request, 1}), &result).ok(), "Unadmitted resident fitting succeeded"); intact();
    Check(!budget.snapshot().total.live_capacity_bytes, "Failed resident admission leaked backing");
  }
  ResidentHistogramResult recovered;
  Ok(op->FitHistogramTrees(Input({&request, 1}), &recovered));
  std::vector<uint32_t> order;
  Check(recovered.trees[0] == Oracle(samples, request.learning, &order), "Resident failure recovery differs");
  if (executor != ResidentHistogramExecutor::kDepthFirst) {
    auto invalid = Input({&request, 1});
    invalid.maximum_active_nodes = 0;
    Check(!op->FitHistogramTrees(invalid, &result).ok(), "Zero frontier slots accepted"); intact();
    invalid = Input({&request, 1}); invalid.maximum_scratch_bytes = 1;
    Check(op->FitHistogramTrees(invalid, &result).code() == StatusCode::kUnsupported,
          "Frontier exceeded its scratch allowance"); intact();
    auto bounded = Input({&request, 1});
    bounded.maximum_scratch_bytes = plan.frontier.slot.bytes * 3;
    if (executor == ResidentHistogramExecutor::kTiledFrontier) {
      bounded.maximum_active_nodes = 3;
      // The joint pool must fit one worst-case node, in addition to summaries.
      bounded.maximum_scratch_bytes = 8 * 1024 * 1024;
    }
    ResidentHistogramPlan bounded_plan;
    Ok(ComputeResidentHistogramPlan(bounded, &bounded_plan));
    Check(bounded_plan.frontier.active_nodes == 3, "Scratch allowance did not bound active nodes");
    ResidentHistogramResult bounded_result;
    Ok(op->FitHistogramTrees(bounded, &bounded_result));
    Check(bounded_result.trees == recovered.trees && bounded_result.maximum_active_nodes <= 3,
          "Bounded frontier changed decisions");

    // Fail after a completed GPU round, rather than only at initial admission.
    auto late = Input({&request, 1});
    late.maximum_active_nodes = late.rounds_per_submission = 1;
    request.learning.maximum_depth = 2;
    const auto committed = gpu.stats().committed_submissions;
    Check(op->FitHistogramTrees(late, &result).code() == StatusCode::kUnsupported &&
          gpu.stats().committed_submissions > committed + 1, "Late frontier capacity failure was not exercised"); intact();
    request.learning = {};
    bool failed_after_gpu = false, exhausted = false;
    for (size_t after = 0; after < 32; ++after) {
      ResidentHistogramResult retained;
      retained.visited_nodes = 983;
      retained.trees.resize(1);
      retained.trees[0].nodes[0].predictor = Predictor::kZero;
      const auto before = gpu.stats().committed_submissions;
      ArmManagedHostAllocationFailureAfterForTest(after);
      const auto status = op->FitHistogramTrees(Input({&request, 1}), &retained);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (status.ok()) { Check(pending, "Injected host failure disappeared"); exhausted = true; break; }
      Check(!pending && retained.visited_nodes == 983 && retained.trees.size() == 1 &&
            retained.trees[0].nodes[0].predictor == Predictor::kZero, "Frontier publication failure changed output");
      failed_after_gpu |= gpu.stats().committed_submissions > before;
      ++failures;
    }
    Check(exhausted && failed_after_gpu, "Frontier publication allocation failure was not exercised");
  }
  samples = Fixture(0, 0, 2);
  Check(op->FitHistogramTrees(Input({&request, 1}), &result).code() == StatusCode::kUnsupported,
        "Empty resident upload was not rejected"); intact();
}
}
int main(int argc, char **argv) try {
  Check(argc == 1 || (argc == 2 && (std::string_view(argv[1]) == "--frontier" || std::string_view(argv[1]) == "--tiled")), "Unknown resident test mode");
  if (argc == 2) executor = std::string_view(argv[1]) == "--tiled" ? ResidentHistogramExecutor::kTiledFrontier : ResidentHistogramExecutor::kFrontier;
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  std::cout << "Resident histogram device: " << gpu->name() << std::endl;
  Case(*gpu, {Fixture(0, 1, 2), Fixture(0, 257, 2)});
  for (size_t predictors : {size_t{1}, size_t{2}, size_t{14}}) {
    std::vector<HistogramTrainingSamples> samples;
    for (size_t seed = 0; seed < 6; ++seed) samples.push_back(Fixture(seed, 131 + seed * 17, predictors));
    Case(*gpu, samples);
  }
  auto s = Fixture(7, 513, 14);
  std::swap(s.predictors[0], s.predictors[2]); std::swap(s.residuals[0], s.residuals[2]);
  HistogramTreeLearningOptions options; options.split_threshold = 0;
  Case(*gpu, {s}, options);
  for (auto &v : s.static_values) std::fill(v.begin(), v.end(), 0);
  for (auto &v : s.values) std::fill(v.begin(), v.end(), 0);
  Case(*gpu, {s});
  auto weighted = Fixture(1, 257, 2);
  weighted.total_samples = 0;
  for (size_t i = 0; i < weighted.weights.size(); ++i) {
    weighted.weights[i] = i == 256 ? 256 : UINT16_MAX;
    weighted.total_samples += weighted.weights[i];
    for (size_t p = 0; p < weighted.predictor_count; ++p)
      weighted.residuals[p][i].extra_bits = 63;
  }
  options.maximum_samples = kHistogramMaximumSamples;
  Case(*gpu, {weighted}, options);
  auto reordered = Fixture(0, 521, 14);
  std::swap(reordered.properties[2], reordered.properties[15]);
  std::swap(reordered.thresholds[2], reordered.thresholds[15]);
  std::swap(reordered.values[0], reordered.values[13]);
  reordered.thresholds[0].resize(255);
  std::iota(reordered.thresholds[0].begin(), reordered.thresholds[0].end(), 0);
  for (size_t i = 0; i < reordered.weights.size(); ++i)
    reordered.static_values[0][i] = (i * 13) % 256;
  Case(*gpu, {reordered});
  if (executor == ResidentHistogramExecutor::kTiledFrontier) {
    // Exercise compact histograms on both sides of the 1024-cell shared
    // collection boundary and the 512-row fused-job limit, with weighted
    // counts and extra bits. The native oracle checks ties and row partitioning.
    for (size_t symbols : {size_t{4}, size_t{16}, size_t{256}}) for (size_t rows : {size_t{511}, size_t{512}, size_t{513}}) {
      auto compact = Fixture(1, rows, 2);
      for (size_t i = 0; i < compact.weights.size(); ++i) {
        compact.static_values[0][i] = i % 3;
        compact.static_values[1][i] = i % 4;
        compact.values[0][i] = i % (symbols == 4 ? 204 : 60);
        compact.values[1][i] = i % (symbols == 4 ? 205 : 61);
        for (size_t pred = 0; pred < compact.predictor_count; ++pred)
          compact.residuals[pred][i] = {
              static_cast<uint8_t>((i + pred) % symbols),
              static_cast<uint8_t>(i % 2 ? 0 : 63)};
      }
      Case(*gpu, {compact}, options);
      if (symbols == 16 && rows == 512) {
        // Every property in every descendant fits locally. The scheduler must
        // make progress when no device histogram or collection group is needed.
        for (size_t i = 0; i < compact.weights.size(); ++i) compact.values[1][i] = i % 59;
        Case(*gpu, {compact}, options, true);
      }
    }
    // Full bin and symbol spans force pool deferral. Each 512-row hash tile
    // also exercises collisions, repeated keys and a partial final tile.
    auto dense = Fixture(2, 1025, 14);
    for (size_t p = 0; p < dense.property_count - dense.static_property_count; ++p)
      for (size_t i = 0; i < dense.weights.size(); ++i)
        dense.values[p][i] = static_cast<uint8_t>((i * (2 * p + 1)) % (p == 13 ? 16 : 256));
    Case(*gpu, {dense, dense, dense});
    Check(deferred_peak >= 2, "Tiled pool deferral was not exercised");
    // Exercise scheduler lanes owning multiple slots and the exact upper
    // bound of its shared prefix array, including mostly empty frontiers.
    for (size_t width : {size_t{129}, size_t{1024}}) {
      active_nodes = width;
      Case(*gpu, {Fixture(0, 2051, 2)});
    }
    // Partial sets of SIMD-owned properties and bin counts on both sides of
    // scan boundaries, including constant properties and the final bin.
    constexpr size_t bin_counts[] = {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 2, 16, 17, 256};
    active_nodes = 128;
    for (size_t properties : {size_t{3}, size_t{5}, size_t{7}, size_t{15}}) {
      auto partial = Fixture(1, 257, 2);
      partial.property_count = properties;
      for (size_t p = 0; p < properties; ++p) {
        const size_t bins = bin_counts[(p + properties) % std::size(bin_counts)];
        partial.thresholds[p].resize(bins - 1);
        std::iota(partial.thresholds[p].begin(), partial.thresholds[p].end(), 0);
        for (size_t i = 0; i < partial.weights.size(); ++i) {
          const uint32_t bin = i % 5 == 0 ? bins - 1 : (i * 7) % bins;
          if (p < partial.static_property_count) partial.static_values[p][i] = bin;
          else partial.values[p - partial.static_property_count][i] = bin;
        }
      }
      Case(*gpu, {partial});
      if (properties == 15) {
        HistogramTreeLearningOptions stopped; stopped.split_threshold = 1e9f;
        Case(*gpu, {partial}, stopped);
      }
    }
    // A full cohort fills more than one lane's worth of active slots. Uneven
    // trees leave holes that the next round must fill in original slot order.
    std::vector<HistogramTrainingSamples> cohort;
    for (size_t i = 0; i < 64; ++i) cohort.push_back(Fixture(i % 6, 257 + i * 3, 2));
    active_nodes = 257; active_peak = 0;
    Case(*gpu, cohort);
    Check(active_peak > 128, "Parallel node setup did not fill a multi-slot-per-lane frontier");
    active_nodes = 128;
  }
  if (executor != ResidentHistogramExecutor::kDepthFirst) {
    for (size_t width : {size_t{1}, size_t{3}, size_t{17}}) {
      active_nodes = width; rounds = width == 1 ? 1 : 3;
      Case(*gpu, {Fixture(0, 129, 2)});
    }
    active_nodes = 128; rounds = 16;
    Check(same_fit_peak > 1, "Frontier never evaluated siblings together");
  }
  Failures(*gpu);
  auto first = std::async(std::launch::async, [&] {
    ResidentHistogramResult result; auto samples = Fixture(0, 257, 2);
    ResidentHistogramRequest request{{1, &samples}, {}};
    Ok(QueryGpuModularHistogramFitting(*gpu)->FitHistogramTrees(Input({&request, 1}), &result));
    return result.trees[0];
  });
  ResidentHistogramResult result; auto samples = Fixture(0, 257, 2);
  ResidentHistogramRequest request{{2, &samples}, {}};
  Ok(QueryGpuModularHistogramFitting(*gpu)->FitHistogramTrees(Input({&request, 1}), &result));
  Check(first.get() == result.trees[0], "Concurrent resident fits shared mutable state");
  result = {}; gpu.reset();
  std::cout << fits << " complete trees and exact row permutations; " << failures
            << " atomic failure checks; concurrent fitting passed\n";
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n'; return 1;
}
