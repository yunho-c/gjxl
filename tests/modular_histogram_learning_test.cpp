// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_tree_learning.h"
#include "codec/modular/histogram_tree_profile.h"
#include "core/thread_budget.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
HistogramTrainingSamples Separated(bool weighted = false) {
  HistogramTrainingSamples s;
  s.predictor_count = 1;
  s.predictors[0] = Predictor::kGradient;
  s.property_count = 1;
  s.properties[0] = 9;
  s.thresholds[0] = {-7};
  for (size_t i = 0; i < 128; ++i) {
    s.weights.push_back(weighted ? 65535 : 1);
    s.total_samples += s.weights.back();
    s.values[0].push_back(i % 2);
    s.residuals[0].push_back({static_cast<uint8_t>(i % 2 ? 17 : 3), 0});
  }
  return s;
}
void DecisionsAndWeights() {
  for (bool weighted : {false, true}) {
    auto s = Separated(weighted);
    HistogramTreeLearningOptions options;
    options.maximum_samples = kHistogramMaximumSamples;
    options.maximum_leaves = 16;
    DecisionTree tree;
    Ok(LearnHistogramTree(s, &tree, options));
    Check(tree.size == 3 && tree.nodes[0].property == 9 &&
              tree.nodes[0].split == -7 && tree.nodes[0].left == 1 &&
              tree.nodes[0].right == 2 &&
              tree.nodes[1].predictor == Predictor::kGradient &&
              tree.nodes[2].predictor == Predictor::kGradient,
          "Weighted signed split or predictor changed");
    for (size_t i = 0; i < s.weights.size(); ++i)
      Check(s.residuals[0][i].symbol == (s.values[0][i] ? 17 : 3),
            "Partition broke row association");
    DecisionTree repeated;
    Ok(LearnHistogramTree(s, &repeated, options));
    Check(repeated == tree, "Fitting changed after row partitioning");
  }
  auto s = Separated();
  for (auto &token : s.residuals[0])
    token = {0, 0};
  DecisionTree tree;
  Ok(LearnHistogramTree(s, &tree));
  Check(tree.size == 1, "Zero entropy made a useless split");
  s.weights.clear();
  s.values[0].clear();
  s.residuals[0].clear();
  s.total_samples = 0;
  Ok(LearnHistogramTree(s, &tree));
  Check(tree.size == 1 && tree.nodes[0].predictor == Predictor::kGradient,
        "Empty finalized samples lost root predictor");
}
void MultipliersAndLimits() {
  auto s = Separated();
  s.properties[0] = 0;
  s.static_property_count = 1;
  s.static_values[0].assign(s.values[0].begin(), s.values[0].end());
  s.values[0].clear();
  s.thresholds[0] = {0};
  for (auto &token : s.residuals[0])
    token = {0, 0};
  const std::array<HistogramMultiplierRange, 2> multipliers{
      {{{{{0, 1}, {0, 1}}}, 2}, {{{{1, 2}, {0, 1}}}, 4}}};
  HistogramTreeLearningOptions options;
  options.static_range = {{{0, 2}, {0, 1}}};
  options.multipliers = multipliers;
  DecisionTree tree;
  Ok(LearnHistogramTree(s, &tree, options));
  Check(tree.size == 3 && tree.nodes[0].property == 0 &&
            tree.nodes[0].split == 0 && tree.nodes[1].multiplier == 4 &&
            tree.nodes[2].multiplier == 2,
        "Forced multiplier boundary or child direction changed");
  const auto saved = tree;
  options.maximum_leaves = 1;
  Check(LearnHistogramTree(s, &tree, options).code() ==
                StatusCode::kUnsupported &&
            tree == saved,
        "Leaf capacity silently truncated or published");
  options.maximum_leaves = 16;
  options.maximum_depth = 1;
  Check(LearnHistogramTree(s, &tree, options).code() ==
                StatusCode::kUnsupported &&
            tree == saved,
        "Depth capacity silently truncated or published");
  options.maximum_depth = kMaximumTreeDepth;
  s.thresholds[0].clear();
  std::fill(s.static_values[0].begin(), s.static_values[0].end(), 0);
  Check(!LearnHistogramTree(s, &tree, options).ok() && tree == saved,
        "Missing multiplier cut was accepted or published");
}
void ValidationAndFailures() {
  const auto original = Separated();
  auto rejects = [&](auto modify, HistogramTreeLearningOptions options = {}) {
    auto s = original;
    modify(s);
    DecisionTree tree;
    tree.nodes[0].predictor = Predictor::kZero;
    const auto saved = tree;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = LearnHistogramTree(s, &tree, options);
    Check(status.code() == StatusCode::kInvalidArgument && tree == saved &&
              ManagedHostAllocationFailurePendingForTest(),
          "Invalid samples allocated or published");
    DisarmManagedHostAllocationFailureForTest();
  };
  rejects([](auto &s) { s.predictor_count = 15; });
  rejects([](auto &s) { s.predictors[0] = Predictor::kInvalid; });
  rejects([](auto &s) { s.property_count = 17; });
  rejects([](auto &s) { s.properties[0] = 16; });
  rejects([](auto &s) { s.properties[0] = 0; });
  rejects([](auto &s) { s.weights[0] = 0; });
  rejects([](auto &s) { s.weights[0] = 2; });
  rejects([](auto &s) { s.total_samples = kMaximumTrainingSamples + 1; });
  rejects([](auto &s) { s.residuals[0].pop_back(); });
  rejects([](auto &s) { s.values[0][0] = 2; });
  rejects([](auto &s) { s.thresholds[0] = {0, -1}; });
  rejects([](auto &s) { s.thresholds[0] = {0, 0}; });
  rejects([](auto &s) { s.thresholds[0] = {INT32_MAX}; });
  rejects([](auto &s) { s.residuals[0][0].extra_bits = 64; });
  // Reductions must retain rejection across SIMD blocks and their tails.
  rejects([](auto &s) { s.weights.back() = 0; });
  rejects([](auto &s) { s.weights.back() = UINT16_MAX; });
  rejects([](auto &s) { s.values[0].back() = UINT8_MAX; });
  rejects([](auto &s) { s.residuals[0].back().extra_bits = 128; });
  rejects([](auto &s) {
    s.static_property_count = 1;
    s.properties[0] = 0;
    s.thresholds[0] = {0};
    s.static_values[0].assign(s.weights.size(), 0);
    s.static_values[0].back() = UINT32_MAX;
    s.values[0].clear();
  });
  HistogramTreeLearningOptions options;
  options.split_threshold = std::numeric_limits<float>::quiet_NaN();
  rejects([](auto &) {}, options);
  options = {};
  options.maximum_samples = SIZE_MAX;
  rejects([](auto &) {}, options);
  options = {};
  options.maximum_depth = 0;
  rejects([](auto &) {}, options);

  options = {};
  HostStorageBound bound;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeHistogramTreeLearningStorageBound(original, options, &bound));
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Storage planner allocated");
  DisarmManagedHostAllocationFailureForTest();
  const auto saved_bound = bound;
  options.maximum_leaves = 0;
  Check(!ComputeHistogramTreeLearningStorageBound(original, options, &bound)
                .ok() &&
            bound == saved_bound,
        "Invalid storage plan changed output");
  options = {};
  size_t failures = 0;
  for (size_t fail_at = 0; fail_at < 32; ++fail_at) {
    auto s = original;
    DecisionTree tree, saved = tree;
    ArmManagedHostAllocationFailureAfterForTest(fail_at);
    const auto status = LearnHistogramTree(s, &tree, options);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (status.ok()) {
      Check(pending && failures >= 10,
            "Failure sweep did not cover all fitting allocations");
      break;
    }
    Check(status.code() == StatusCode::kOutOfMemory && tree == saved &&
              !pending,
          "Allocation failure escaped or changed output");
    Check(s.weights == original.weights &&
              s.residuals[0] == original.residuals[0] &&
              s.values[0] == original.values[0],
          "Fitting allocated after beginning to partition rows");
    ++failures;
  }
  Check(failures < 32, "Allocation sweep never succeeded");
  ResourceBudget budget(bound.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.Reserve(bound.peak_bytes, &reservation));
  auto samples = original;
  {
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    DecisionTree result;
    Ok(LearnHistogramTree(samples, &result));
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
          "Fitting exceeded storage bound");
  }
  Check(budget.snapshot().total.backing_count == 0,
        "Fitting leaked managed storage");
  ResourceBudget short_budget(bound.peak_bytes - 1);
  ResourceReservation short_reservation;
  Ok(short_budget.Reserve(bound.peak_bytes - 1, &short_reservation));
  samples = original;
  DecisionTree unchanged;
  const auto saved_output = unchanged;
  {
    ResourceContextScope scope(
        {&short_reservation, ResourceClass::kPreparation});
    Check(LearnHistogramTree(samples, &unchanged).resource_plan_exceeded() &&
              unchanged == saved_output &&
              samples.weights == original.weights &&
              samples.values[0] == original.values[0],
          "Fitting escaped an insufficient reservation or published output");
  }
  Check(short_budget.snapshot().total.backing_count == 0,
        "Rejected fitting leaked admitted storage");
}

void ProfiledParity() {
  for (bool weighted : {false, true}) {
    auto plain = Separated(weighted), measured = plain;
    HistogramTreeLearningOptions options;
    options.maximum_samples = kHistogramMaximumSamples;
    DecisionTree expected, actual;
    Ok(LearnHistogramTree(plain, &expected, options));
    HistogramFittingProfile profile;
    std::array<HistogramFitRecord, 1> records;
    profile.records = records;
    {
      HistogramFittingProfileSession session(&profile);
      {
        HistogramFittingProfileSession disabled(nullptr);
        Check(!active_histogram_fitting_profile,
              "Nested profile did not disable");
      }
      Check(active_histogram_fitting_profile == &profile,
            "Nested profile did not restore");
      Ok(LearnHistogramTree(measured, &actual, options));
    }
    Check(!active_histogram_fitting_profile && actual == expected &&
              plain.weights == measured.weights &&
              plain.residuals == measured.residuals &&
              plain.values == measured.values,
          "Profiling changed tree or partition order");
    Check(profile.calls == 1 && profile.successes == 1 &&
              profile.records_written == 1 && !profile.records_dropped &&
              records[0].nodes == 3 && records[0].rows == 128 &&
              records[0].population == measured.total_samples &&
              records[0].maximum_depth == 2,
          "Profile lost fitting envelope or record");
    const auto &root = profile.by_rows[7].work;
    const auto &leaves = profile.by_rows[6].work;
    Check(root.nodes == 1 && leaves.nodes == 2 && root.rows == 128 &&
              leaves.rows == 128 && root.splits == 1 && root.properties == 1 &&
              root.active_bins == 2 && root.joint_histogram_row_visits == 128 &&
              root.candidate_thresholds == 1 && root.entropy_evaluations == 2 &&
              root.symbol_terms == 40 && root.partition_rows == 128 &&
              root.row_swaps == 32 && leaves.properties == 0 &&
              profile.by_depth[1].nodes == 1 &&
              profile.by_depth[2].nodes == 2 &&
              profile.nodes_by_symbol_span[20] == 2 &&
              profile.nodes_by_symbol_span[4] == 1,
          "Histogram work attribution is inconsistent");
    double node_seconds = 0;
    for (const auto &bucket : profile.by_rows) {
      node_seconds += bucket.node_seconds;
      Check(std::accumulate(bucket.seconds.begin(), bucket.seconds.end(),
                            0.0) <= bucket.node_seconds + 1e-9,
            "Node phase times overlap");
    }
    Check(std::isfinite(profile.fit_seconds) && profile.fit_seconds > 0 &&
              std::accumulate(profile.seconds.begin(), profile.seconds.end(),
                              0.0) <= profile.fit_seconds &&
              node_seconds <= profile.fit_seconds,
          "Fitting attribution exceeds its envelope");
  }
  // Exercise predictor ties, heterogeneous symbols/bins, weighted rows, and
  // different growth shapes through both compiled implementations.
  for (size_t seed = 1; seed <= 8; ++seed) {
    auto plain = Separated();
    plain.predictor_count = seed % 2 ? 2 : 14;
    for (size_t p = 0; p < plain.predictor_count; ++p) {
      plain.predictors[p] = static_cast<Predictor>(p);
      plain.residuals[p].resize(128);
      for (size_t r = 0; r < 128; ++r)
        plain.residuals[p][r] = {uint8_t((r * (p + 1) + seed) % (4 + 7 * seed)),
                                 uint8_t((r + p) % 3)};
    }
    plain.properties[0] = 9;
    plain.thresholds[0] = {-7, 0, 7};
    plain.total_samples = 0;
    for (size_t r = 0; r < 128; ++r) {
      plain.values[0][r] = r % 4;
      plain.weights[r] = 1 + r % 7;
      plain.total_samples += plain.weights[r];
    }
    auto measured = plain;
    DecisionTree expected, actual;
    Ok(LearnHistogramTree(plain, &expected));
    HistogramFittingProfile profile;
    {
      HistogramFittingProfileSession session(&profile);
      Ok(LearnHistogramTree(measured, &actual));
    }
    Check(actual == expected && plain.residuals == measured.residuals &&
              profile.calls == 1 && profile.successes == 1 &&
              profile.records_dropped == 1,
          "Profile changed generated fit or hid a dropped record");
  }
  // Profiling is diagnostic on failure and does not publish an incomplete tree.
  auto invalid = Separated();
  invalid.weights.back() = 0;
  DecisionTree unchanged, saved = unchanged;
  HistogramFittingProfile failed;
  {
    HistogramFittingProfileSession session(&failed);
    Check(!LearnHistogramTree(invalid, &unchanged).ok(),
          "Invalid profiled fit succeeded");
  }
  Check(unchanged == saved && failed.calls == 1 && failed.successes == 0,
        "Failed profiled fit changed output or counters");
}

void ConstantColumnsAndSparseBins() {
  for (size_t predictors : {size_t{1}, size_t{2}, kHistogramPredictors}) {
    for (bool weighted : {false, true}) {
      auto minimal = Separated(weighted);
      minimal.predictor_count = predictors;
      // Keep the original predictor first; equal alternatives exercise ties.
      size_t next = 1;
      for (size_t p = 0; next < predictors; ++p) {
        if (static_cast<Predictor>(p) != Predictor::kGradient)
          minimal.predictors[next++] = static_cast<Predictor>(p);
      }
      for (auto &token : minimal.residuals[0]) token.extra_bits = 3;
      for (size_t p = 1; p < predictors; ++p)
        minimal.residuals[p] = minimal.residuals[0];
      // Exercise sparse endpoints, including bin 255 and symbol padding that
      // shrinks in the children. Extra bits keep leaf base costs above zero.
      minimal.thresholds[0].resize(255);
      std::iota(minimal.thresholds[0].begin(), minimal.thresholds[0].end(), -127);
      for (auto &value : minimal.values[0]) value *= 255;

      auto augmented = minimal;
      augmented.property_count = 4;
      augmented.static_property_count = 1;
      augmented.properties = {0, 10, 9, 15};
      augmented.values[1] = minimal.values[0];
      augmented.values[0].assign(minimal.weights.size(), 255);
      augmented.values[2].assign(minimal.weights.size(), 0);
      augmented.static_values[0].assign(minimal.weights.size(), 65535);
      augmented.thresholds[0].resize(65535);
      std::iota(augmented.thresholds[0].begin(), augmented.thresholds[0].end(), 0);
      augmented.thresholds[1] = augmented.thresholds[2] = minimal.thresholds[0];
      augmented.thresholds[3].clear();
      HistogramTreeLearningOptions options;
      options.maximum_samples = kHistogramMaximumSamples;
      options.maximum_leaves = 16;
      DecisionTree expected, actual;
      Ok(LearnHistogramTree(minimal, &expected, options));
      HistogramFittingProfile profile;
      {
        HistogramFittingProfileSession session(&profile);
        Ok(LearnHistogramTree(augmented, &actual, options));
      }
      Check(expected.size == 3 && actual == expected &&
                augmented.weights == minimal.weights &&
                augmented.residuals == minimal.residuals &&
                augmented.values[1] == minimal.values[0],
            "Constant or sparse properties changed the tree or row partition");
      const auto &root = profile.by_rows[7].work;
      const auto &leaves = profile.by_rows[6].work;
      Check(root.constant_properties == 3 && root.skipped_constant_properties == 1 &&
                root.property_row_visits == 3 * 128 &&
                root.fused_property_row_visits == root.property_row_visits &&
                root.joint_histogram_row_visits == (2 + predictors) * 128 &&
                leaves.constant_properties == 8 && leaves.skipped_constant_properties == 6 &&
                leaves.property_row_visits == 128 && leaves.joint_histogram_row_visits == 128,
            "Constant properties were rescanned or occupancy lost distinct-row counts");
    }
  }
}
// Duplicate properties exercise strict property-order ties. Other columns
// become constant only below a split, with sparse bins and changing symbols.
HistogramTrainingSamples ParallelSamples(size_t predictors) {
  HistogramTrainingSamples s;
  s.predictor_count = predictors;
  s.property_count = 6;
  s.static_property_count = 2;
  s.properties = {0, 1, 4, 5, 9, 15};
  for (size_t pred = 0; pred < predictors; ++pred)
    s.predictors[pred] = static_cast<Predictor>(pred);
  for (size_t prop = 0; prop < s.property_count; ++prop)
    for (size_t bin = 0; bin < 255; ++bin)
      s.thresholds[prop].push_back(static_cast<int32_t>(bin));
  for (size_t i = 0; i < 32768; ++i) {
    s.weights.push_back(1 + (i % 3 == 0));
    s.total_samples += s.weights.back();
    s.static_values[0].push_back(255);
    s.static_values[1].push_back((i & 1) ? 255 : 0);
    for (size_t prop = 0; prop < 4; ++prop)
      s.values[prop].push_back(prop == 3 ? 255 : ((i >> (prop == 2 ? 2 : 1)) & 3) * 85);
    for (size_t pred = 0; pred < predictors; ++pred)
      s.residuals[pred].push_back({static_cast<uint8_t>((i & 15) + pred * (i % 7)),
                                  static_cast<uint8_t>((i >> 4) % 3)});
  }
  return s;
}
void ParallelProperties() {
  using namespace thread_budget_internal;
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{64} << 20, 8}, &domain));
  for (bool forced : {false, true})
  for (size_t predictors : {size_t{1}, size_t{2}, size_t{14}}) {
    const std::array<HistogramMultiplierRange, 2> multipliers{
        {{{{{0, 128}, {0, 256}}}, 2}, {{{{128, 256}, {0, 256}}}, 4}}};
    HistogramTreeLearningOptions control;
    if (forced) {
      control.static_range = {{{0, 256}, {0, 256}}};
      control.multipliers = multipliers;
    }
    auto original = ParallelSamples(predictors);
    auto serial = original;
    DecisionTree expected;
    HistogramFittingProfile serial_profile;
    {
      HistogramFittingProfileSession profile(&serial_profile);
      Ok(LearnHistogramTree(serial, &expected, control));
    }
    for (size_t requested : {size_t{2}, size_t{4}, size_t{8}, size_t{0}}) {
      CpuExecutionScope cpu;
      Ok(cpu.Start(domain, 8));
      EncodeScope threads(8);
      auto actual = original;
      DecisionTree tree;
      HistogramTreeLearningStats execution;
      auto options = control;
      options.cpu_participants = requested;
      options.execution_stats = &execution;
      HostStorageBound bound;
      Ok(ComputeHistogramTreeLearningStorageBound(actual, options, &bound));
      ResourceBudget budget(bound.peak_bytes);
      ResourceReservation reservation;
      Ok(budget.TryReserve(bound.peak_bytes, &reservation));
      HistogramFittingProfile parallel_profile;
      {
        ResourceContextScope resources({&reservation, ResourceClass::kPreparation});
        HistogramFittingProfileSession profile(&parallel_profile);
        Ok(LearnHistogramTree(actual, &tree, options));
      }
      Check(execution.participants > 1 && execution.parallel_nodes > 0,
            "Property fixture did not dispatch parallel work");
      Check(tree == expected && actual.weights == serial.weights &&
                actual.residuals == serial.residuals && actual.static_values == serial.static_values &&
                actual.values == serial.values,
            "Parallel property decisions or partition order differ");
      Check(parallel_profile.properties_by_active_bins == serial_profile.properties_by_active_bins &&
                parallel_profile.properties_by_bin_span == serial_profile.properties_by_bin_span &&
                parallel_profile.nodes_by_symbol_span == serial_profile.nodes_by_symbol_span,
            "Parallel property profile distributions differ");
      for (size_t i = 0; i < serial_profile.by_rows.size(); ++i)
        Check(parallel_profile.by_rows[i].work == serial_profile.by_rows[i].work,
              "Parallel property profile lost work");
    }
  }
  Check(!domain->snapshot().active_cpu_participants && !domain->snapshot().reserved_cpu_workers &&
            !domain->snapshot().suspended_cpu_workers && domain->snapshot().peak_cpu_participants <= 8,
        "Property team leaked or exceeded CPU participation");
  // Every admitted allocation occurs before row mutation or worker launch.
  auto original = ParallelSamples(2);
  bool succeeded = false;
  for (size_t fail = 0; fail < 80 && !succeeded; ++fail) {
    auto samples = original;
    DecisionTree tree;
    tree.nodes[0].predictor = Predictor::kTop;
    const auto expected = tree;
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, 4));
    EncodeScope threads(4);
    HistogramTreeLearningOptions options;
    options.cpu_participants = 4;
    ArmManagedHostAllocationFailureAfterForTest(fail);
    auto status = LearnHistogramTree(samples, &tree, options);
    DisarmManagedHostAllocationFailureForTest();
    if (status.ok()) {
      succeeded = true;
    } else {
      Check(status.code() == StatusCode::kOutOfMemory && tree == expected &&
                samples.weights == original.weights && samples.values == original.values &&
                samples.residuals == original.residuals,
            "Parallel allocation failure changed output or sample order");
    }
  }
  Check(succeeded, "Parallel allocation sweep did not reach success");
}
} // namespace
int main() try {
  DecisionsAndWeights();
  MultipliersAndLimits();
  {
    HistogramFittingProfile profile;
    HistogramFittingProfileSession session(&profile);
    MultipliersAndLimits();
    Check(profile.by_rows[7].work.forced_splits > 0,
          "Forced multiplier splits were not profiled");
  }
  ValidationAndFailures();
  ProfiledParity();
  ConstantColumnsAndSparseBins();
  ParallelProperties();
  std::cout << "Histogram learning decisions, weights, limits, storage and "
               "failure checks passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
