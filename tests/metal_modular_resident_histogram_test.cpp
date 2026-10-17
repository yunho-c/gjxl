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
          HistogramTreeLearningOptions options = {}) {
  std::vector<ResidentHistogramRequest> requests;
  std::vector<DecisionTree> expected;
  std::vector<std::vector<uint32_t>> orders(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    requests.push_back({{uint32_t(i + 17), &samples[i]}, options});
    expected.push_back(Oracle(samples[i], options, &orders[i]));
    auto copy = samples[i]; DecisionTree ordinary;
    Ok(LearnHistogramTree(copy, &ordinary, options));
    Check(ordinary == expected.back(), "Ordinary/resumable CPU oracle differs");
  }
  const auto originals = samples;
  const ResidentHistogramInput input{requests, true};
  ResidentHistogramPlan plan;
  Ok(ComputeResidentHistogramPlan(input, &plan));
  ResourceBudget budget(plan.working.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.TryReserve(plan.working.peak_bytes, &reservation));
  ResourceContextScope context({&reservation, ResourceClass::kPreparation});
  ResidentHistogramResult result;
  const auto before = gpu.stats().committed_submissions;
  Ok(QueryGpuModularHistogramFitting(gpu)->FitHistogramTrees(input, &result));
  Check(gpu.stats().committed_submissions == before + 1,
        "Resident fitting must use one complete-fit submission");
  Check(result.trees.size() == expected.size() && result.row_offsets.size() == expected.size() + 1,
        "Resident fitting omitted results");
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
    const auto status = op->FitHistogramTrees({{&request, 1}}, &result);
    DisarmManagedHostAllocationFailureForTest();
    Check(!status.ok(), "Injected resident failure was ignored"); intact();
  }
  for (unsigned limit = 0; limit < 2; ++limit) {
    request.learning.maximum_leaves = limit ? kMaximumTreeLeaves : 1;
    request.learning.maximum_depth = limit ? 1 : kMaximumTreeDepth;
    Check(op->FitHistogramTrees({{&request, 1}}, &result).code() == StatusCode::kUnsupported,
          "Resident tree capacity/depth silently truncated"); intact();
  }
  request.learning = {};
  HistogramMultiplierRange multiplier{};
  multiplier.range = request.learning.static_range;
  request.learning.multipliers = {&multiplier, 1};
  const auto before = gpu.stats().committed_submissions;
  Check(op->FitHistogramTrees({{&request, 1}}, &result).code() == StatusCode::kUnsupported,
        "Unsupported resident multiplier accepted"); intact();
  request.learning.multipliers = {};
  samples.static_values[0][0] = 999;
  Check(op->FitHistogramTrees({{&request, 1}}, &result).code() == StatusCode::kInvalidArgument,
        "Invalid resident property bin accepted"); intact();
  Check(gpu.stats().committed_submissions == before, "Resident preflight submitted invalid work");
  samples.static_values[0][0] = 0;
  samples.thresholds[0].resize(256);
  std::iota(samples.thresholds[0].begin(), samples.thresholds[0].end(), 0);
  Check(op->FitHistogramTrees({{&request, 1}}, &result).code() == StatusCode::kUnsupported,
        "Oversized resident static bins accepted"); intact();
  samples = Fixture(0, 257, 2);
  ResidentHistogramPlan plan;
  Ok(ComputeResidentHistogramPlan({{&request, 1}}, &plan));
  {
    ResourceBudget budget(1); ResourceReservation reservation;
    Ok(budget.TryReserve(1, &reservation));
    ResourceContextScope context({&reservation, ResourceClass::kPreparation});
    Check(!op->FitHistogramTrees({{&request, 1}}, &result).ok(), "Unadmitted resident fitting succeeded"); intact();
    Check(!budget.snapshot().total.live_capacity_bytes, "Failed resident admission leaked backing");
  }
  ResidentHistogramResult recovered;
  Ok(op->FitHistogramTrees({{&request, 1}}, &recovered));
  std::vector<uint32_t> order;
  Check(recovered.trees[0] == Oracle(samples, request.learning, &order), "Resident failure recovery differs");
  samples = Fixture(0, 0, 2);
  Check(op->FitHistogramTrees({{&request, 1}}, &result).code() == StatusCode::kUnsupported,
        "Empty resident upload was not rejected"); intact();
}
}
int main() try {
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
  Failures(*gpu);
  auto first = std::async(std::launch::async, [&] {
    ResidentHistogramResult result; auto samples = Fixture(0, 257, 2);
    ResidentHistogramRequest request{{1, &samples}, {}};
    Ok(QueryGpuModularHistogramFitting(*gpu)->FitHistogramTrees({{&request, 1}}, &result));
    return result.trees[0];
  });
  ResidentHistogramResult result; auto samples = Fixture(0, 257, 2);
  ResidentHistogramRequest request{{2, &samples}, {}};
  Ok(QueryGpuModularHistogramFitting(*gpu)->FitHistogramTrees({{&request, 1}}, &result));
  Check(first.get() == result.trees[0], "Concurrent resident fits shared mutable state");
  result = {}; gpu.reset();
  std::cout << fits << " complete trees and exact row permutations; " << failures
            << " atomic failure checks; concurrent fitting passed\n";
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n'; return 1;
}
