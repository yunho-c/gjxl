// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_tree_profile.h"
#include "modular_histogram_fitting_test_helpers.h"
#include <iostream>

namespace {
using namespace histogram_fitting_test;
HistogramTreeLearningOptions Options() {
  HistogramTreeLearningOptions o;
  o.maximum_leaves = 4096;
  o.maximum_samples = kHistogramMaximumSamples;
  return o;
}
void CheckProfile(const HistogramFittingProfile &a,
                  const HistogramFittingProfile &b) {
  Check(a.calls == b.calls && a.successes == b.successes &&
            a.nodes_by_symbol_span == b.nodes_by_symbol_span &&
            a.properties_by_active_bins == b.properties_by_active_bins &&
            a.properties_by_bin_span == b.properties_by_bin_span,
        "Resumption changed fit distributions");
  for (size_t i = 0; i < a.by_rows.size(); ++i)
    Check(a.by_rows[i].work == b.by_rows[i].work,
          "Resumption changed fitting work");
  for (size_t i = 0; i < a.by_depth.size(); ++i)
    Check(a.by_depth[i].nodes == b.by_depth[i].nodes &&
              a.by_depth[i].rows == b.by_depth[i].rows,
          "Resumption changed depth-first growth");
}
void Parity(const HistogramTrainingSamples &original,
            HistogramTreeLearningOptions options = Options()) {
  auto reference = original, resumed = original;
  HistogramFittingProfile old_profile, new_profile;
  DecisionTree expected, actual;
  {
    HistogramFittingProfileSession p(&old_profile);
    Ok(LearnHistogramTree(reference, &expected, options));
  }
  HistogramTreeFitter fitter;
  Ok(fitter.Initialize(resumed, options, &new_profile));
  Check(SameRows(original, resumed), "Initialization partitioned sample rows");
  std::array<HistogramPropertyResult, 16> values;
  const auto results = std::span(values).first(original.property_count);
  ArmNextManagedHostAllocationFailureForTest();
  while (!fitter.complete()) {
    Ok(fitter.Advance());
    if (!fitter.ready())
      continue;
    const auto n = fitter.node();
    Check(n.original_rows.size() == n.end - n.begin &&
              n.parent_counts.size() == n.symbols * original.predictor_count,
          "Ready node lost membership or parent summaries");
    CheckOriginalRows(original, resumed, fitter.row_order());
    Ok(EvaluateHistogramNodeOnCpu(fitter, 17, results));
    Ok(ApplyHistogramNodeEvaluation(fitter, 17, results));
  }
  CheckOriginalRows(original, resumed, fitter.row_order());
  Ok(fitter.Publish(&actual));
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Resumption allocated after initialization");
  DisarmManagedHostAllocationFailureForTest();
  Check(actual == expected && SameRows(reference, resumed),
        "Resumed tree or final partition order differs");
  CheckProfile(old_profile, new_profile);
  Check(!fitter.ready() && !fitter.complete() && fitter.row_order().empty(),
        "Published fitter retained its state");
}
void Fixtures() {
  for (size_t seed = 0; seed < 12; ++seed)
    for (size_t predictors : {size_t{1}, size_t{2}, size_t{14}})
      Parity(Fixture(seed, 128 + seed * 37, predictors));
  Parity(Fixture(0, 0, 1));
  Parity(Fixture(7, 1, 14));
  auto extreme = Fixture(4, 257, 2);
  std::fill(extreme.weights.begin(), extreme.weights.end(), UINT16_MAX);
  extreme.weights.back() = 256;
  extreme.total_samples = kHistogramMaximumSamples;
  for (size_t p = 0; p < extreme.predictor_count; ++p)
    for (auto &r : extreme.residuals[p])
      r.extra_bits = 63;
  Parity(extreme);
  auto wide = Fixture(3, 256, 2);
  wide.thresholds[0].resize(65535);
  std::iota(wide.thresholds[0].begin(), wide.thresholds[0].end(), 0);
  std::fill(wide.static_values[0].begin(), wide.static_values[0].end(), 65535);
  Parity(wide);
  auto forced =
      Fixture(1, 256, 2); // Constant static column produces an empty child.
  const std::array<HistogramMultiplierRange, 2> multipliers{
      {{{{{0, 2}, {0, 8}}}, 2}, {{{{2, 4}, {0, 8}}}, 4}}};
  auto options = Options();
  options.static_range = {{{0, 4}, {0, 8}}};
  options.multipliers = multipliers;
  Parity(forced, options);
}
void InterleavedFits() {
  constexpr size_t count = 6;
  std::array<HistogramTrainingSamples, count> samples, originals;
  std::array<HistogramTreeFitter, count> fitters;
  std::array<DecisionTree, count> expected, actual;
  std::array<std::array<HistogramPropertyResult, 16>, count> results;
  for (size_t i = 0; i < count; ++i) {
    originals[i] = Fixture(i, 32 + i * 103, i % 2 ? 14 : 2);
    samples[i] = originals[i];
    auto ref = originals[i];
    Ok(LearnHistogramTree(ref, &expected[i], Options()));
    Ok(fitters[i].Initialize(samples[i], Options()));
  }
  // Pause every fit before evaluating any. Reverse application order to expose
  // accidental shared scratch or dependence on another fit's progress.
  for (;;) {
    bool complete = true;
    for (size_t i = 0; i < count; ++i) {
      Ok(fitters[i].Advance());
      complete &= fitters[i].complete();
      if (fitters[i].ready())
        Ok(EvaluateHistogramNodeOnCpu(fitters[i], uint32_t(i), results[i]));
    }
    if (complete)
      break;
    for (size_t i = count; i-- > 0;)
      if (fitters[i].ready())
        Ok(ApplyHistogramNodeEvaluation(fitters[i], uint32_t(i), results[i]));
  }
  for (size_t i = 0; i < count; ++i) {
    CheckOriginalRows(originals[i], samples[i], fitters[i].row_order());
    Ok(fitters[i].Publish(&actual[i]));
    Check(actual[i] == expected[i], "Interleaved fit shared mutable state");
  }
}
void InvalidResultsAndStates() {
  auto samples = Fixture(0, 256, 2);
  HistogramTreeFitter fitter;
  DecisionTree tree, saved = tree;
  Check(!fitter.Advance().ok() && !fitter.Publish(&tree).ok() && tree == saved,
        "Uninitialized fitter advanced or published");
  Ok(fitter.Initialize(samples, Options()));
  Check(!fitter.Publish(&tree).ok() && tree == saved,
        "Incomplete fitter published");
  Ok(fitter.Advance());
  Check(fitter.ready(), "Malformed result fixture has no node");
  const auto before = samples;
  std::array<HistogramPropertyResult, 16> results;
  Ok(EvaluateHistogramNodeOnCpu(fitter, 17, results));
  const auto rejects = [&](auto modify) {
    auto bad = results;
    modify(bad);
    Check(!ApplyHistogramNodeEvaluation(fitter, 17, bad).ok() &&
              fitter.ready() && SameRows(before, samples),
          "Malformed results changed the pending fit");
  };
  rejects([](auto &r) { r.back().source_id++; });
  rejects([](auto &r) { r.back().node_id++; });
  rejects([](auto &r) { r.back().property_index = 0; });
  rejects([](auto &r) { r.back().constant = 2; });
  rejects([](auto &r) { r.back().winners[0].reserved = 1; });
  rejects([](auto &r) { r.back().winners[0].valid = 2; });
  rejects([](auto &r) { r.back().winners[0].left_predictor = 256; });
  rejects([](auto &r) {
    r.back().winners[0].left_cost = std::numeric_limits<float>::quiet_NaN();
  });
  rejects([](auto &r) {
    for (auto &p : r)
      for (auto &c : p.winners)
        if (c.valid)
          c.low_rows = 0;
  });
  rejects([](auto &r) {
    for (auto &p : r)
      for (auto &c : p.winners)
        if (c.valid)
          c.bin = UINT32_MAX;
  });
  Check(!fitter.Advance().ok() && !fitter.Publish(&tree).ok() && tree == saved,
        "Pending node was skipped or partially published");
  const auto unchanged = results;
  Check(!EvaluateHistogramNodeOnCpu(fitter, 17, std::span(results).first(15))
                .ok() &&
            SameResults(results, unchanged),
        "Wrong CPU output extent changed output");
  Ok(ApplyHistogramNodeEvaluation(fitter, 17, results));
  while (!fitter.complete()) {
    Ok(fitter.Advance());
    if (fitter.ready()) {
      Ok(EvaluateHistogramNodeOnCpu(fitter, 17, results));
      Ok(ApplyHistogramNodeEvaluation(fitter, 17, results));
    }
  }
  Check(!fitter.Publish(nullptr).ok() && fitter.complete(),
        "Null publication consumed a fit");
  Ok(fitter.Publish(&tree));
  // Capacity failure does not truncate a tree or publish partial output.
  samples = before;
  auto limited = Options();
  limited.maximum_leaves = 1;
  Ok(fitter.Initialize(samples, limited));
  Ok(fitter.Advance());
  Ok(EvaluateHistogramNodeOnCpu(fitter, 17, results));
  Check(ApplyHistogramNodeEvaluation(fitter, 17, results).code() ==
                StatusCode::kUnsupported &&
            !fitter.complete() && !fitter.ready() &&
            !fitter.Publish(&saved).ok(),
        "Capacity failure was hidden");
}
void AllocationAndStorage() {
  const auto original = Fixture(0, 128, 2);
  auto options = Options();
  HostStorageBound bound;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeHistogramTreeFitterStorageBound(original, options, &bound));
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Resumable planner allocated");
  DisarmManagedHostAllocationFailureForTest();
  bool succeeded = false;
  for (size_t failure = 0; failure < 32 && !succeeded; ++failure) {
    auto samples = original;
    HistogramTreeFitter fit;
    ArmManagedHostAllocationFailureAfterForTest(failure);
    const auto status = fit.Initialize(samples, options);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(SameRows(original, samples),
          "Initialization failure partitioned rows");
    if (status.ok()) {
      succeeded = true;
      Check(pending && failure >= 10, "Incomplete allocation sweep");
    } else
      Check(status.code() == StatusCode::kOutOfMemory && !pending &&
                !fit.ready(),
            "Initialization lost allocation failure");
  }
  Check(succeeded, "Allocation sweep did not reach success");
  ResourceBudget budget(bound.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.TryReserve(bound.peak_bytes, &reservation));
  auto samples = original;
  {
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    HistogramTreeFitter fitter;
    Ok(fitter.Initialize(samples, options));
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
          "Resumable state exceeded admitted storage");
  }
  Check(!budget.snapshot().total.backing_count,
        "Resumable state leaked backing");
  const auto saved_bound = bound;
  options.cpu_participants = 2;
  Check(ComputeHistogramTreeFitterStorageBound(original, options, &bound)
                    .code() == StatusCode::kUnsupported &&
            bound == saved_bound,
        "Resumable fitter silently admitted nested property workers");
  // Failed replacement leaves an existing ready fit usable.
  HistogramTreeFitter fitter;
  samples = original;
  Ok(fitter.Initialize(samples, Options()));
  Ok(fitter.Advance());
  auto invalid = original;
  invalid.weights[0] = 0;
  ArmNextManagedHostAllocationFailureForTest();
  Check(!fitter.Initialize(invalid, Options()).ok() && fitter.ready() &&
            ManagedHostAllocationFailurePendingForTest(),
        "Invalid initialization replaced an active state or allocated");
  DisarmManagedHostAllocationFailureForTest();
  auto moved = std::move(fitter);
  Check(moved.ready() && !fitter.ready(), "Moving a fit lost pending state");
}
} // namespace
int main() try {
  Fixtures();
  InterleavedFits();
  InvalidResultsAndStates();
  AllocationAndStorage();
  std::cout << "Resumable histogram fitter: exact trees/rows/work, "
               "interleaving, contract results, failures and storage passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
