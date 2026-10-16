// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "modular_histogram_fitting_test_helpers.h"
#include <future>
#include <iostream>
#include <vector>

namespace {
using namespace histogram_fitting_test;
size_t batches = 0, properties = 0, complete_trees = 0, faults = 0;
GpuModularHistogramFitting &Operation(GpuBackend &gpu) {
  auto *op = QueryGpuModularHistogramFitting(gpu);
  Check(op, "Metal did not advertise native histogram fitting");
  return *op;
}
void Compare(std::span<const HistogramPropertyResult> actual,
             std::span<const HistogramPropertyResult> expected) {
  if (!SameResults(actual, expected)) {
    for (size_t p = 0; p < std::min(actual.size(), expected.size()); ++p)
      if (!SameResults(actual.subspan(p, 1), expected.subspan(p, 1))) {
        std::cerr << "Result mismatch: source " << expected[p].source_id
                  << " node " << expected[p].node_id << " property "
                  << expected[p].property_index << " constant "
                  << actual[p].is_constant << '/' << expected[p].is_constant
                  << '\n';
        for (size_t c = 0; c < 4; ++c) {
          const auto &a = actual[p].winners[c], &b = expected[p].winners[c];
          std::cerr << c << " bin " << a.bin << '/' << b.bin << " rows "
                    << a.low_rows << '/' << b.low_rows << " predictors "
                    << a.left_predictor << ',' << a.right_predictor << '/'
                    << b.left_predictor << ',' << b.right_predictor
                    << " cost bits " << std::bit_cast<uint32_t>(a.left_cost)
                    << ',' << std::bit_cast<uint32_t>(a.right_cost) << '/'
                    << std::bit_cast<uint32_t>(b.left_cost) << ','
                    << std::bit_cast<uint32_t>(b.right_cost) << '\n';
        }
        break;
      }
    throw std::runtime_error(
        "Metal candidates differ from native CPU arithmetic");
  }
  properties += actual.size();
  ++batches;
}
void Trees(GpuBackend &gpu, std::vector<HistogramTrainingSamples> originals,
           HistogramFittingExecutionOptions execution = {}, bool full = true) {
  auto samples = originals;
  std::vector<HistogramTreeFitter> fit(samples.size());
  std::vector<HistogramFittingSource> sources;
  for (size_t i = 0; i < samples.size(); ++i)
    sources.push_back({uint32_t(7 + 31 * i), &samples[i]});
  const auto before = gpu.stats();
  std::unique_ptr<PreparedGpuHistogramRows> prepared;
  Ok(Operation(gpu).PrepareHistogramRows(sources, &prepared));
  Check(gpu.stats().committed_submissions == before.committed_submissions,
        "Preparing immutable rows submitted device work");
  HistogramTreeLearningOptions options;
  options.maximum_samples = kHistogramMaximumSamples;
  for (size_t i = 0; i < samples.size(); ++i)
    Ok(fit[i].Initialize(samples[i], options));
  for (;;) {
    std::vector<HistogramFittingNodeInput> nodes;
    std::vector<HistogramPropertyResult> expected;
    for (size_t i = 0; i < samples.size(); ++i) {
      if (fit[i].complete())
        continue;
      Ok(fit[i].Advance());
      if (!fit[i].ready())
        continue;
      nodes.push_back(HistogramFittingNodeView(i, fit[i].node()));
      const size_t offset = expected.size();
      expected.resize(offset + samples[i].property_count);
      Ok(EvaluateHistogramNodeOnCpu(fit[i], sources[i].source_id,
                                    std::span(expected).subspan(offset)));
    }
    if (nodes.empty())
      break;
    std::unique_ptr<PreparedGpuHistogramEvaluation> output;
    const auto submissions = gpu.stats().committed_submissions;
    std::vector<HistogramFittingNodeShape> shapes;
    for (const auto &node : nodes) shapes.push_back(node.shape);
    HistogramFittingEvaluationPlan plan;
    Ok(ComputeHistogramFittingEvaluationPlan(prepared->shapes(), shapes, execution, &plan));
    const auto capacities = Operation(gpu).HistogramFittingBufferCapacities(plan, execution);
    Ok(Operation(gpu).EvaluateHistogramNodes({prepared.get(), nodes, execution,
        HistogramFittingProfileMode::kDisabled, nullptr, false, capacities},
                                             &output));
    Check(gpu.stats().committed_submissions == submissions + 1,
          "A node batch did not use exactly one submission");
    Compare(output->results(), expected);
    size_t offset = expected.size();
    // Apply in reverse order to expose cross-fit scratch/identity corruption.
    for (size_t j = nodes.size(); j-- != 0;) {
      const size_t i = nodes[j].shape.fit;
      offset -= samples[i].property_count;
      Ok(ApplyHistogramNodeEvaluation(
          fit[i], sources[i].source_id,
          output->results().subspan(offset, samples[i].property_count)));
      CheckOriginalRows(originals[i], samples[i], fit[i].row_order());
    }
    if (!full)
      return;
  }
  for (size_t i = 0; i < samples.size(); ++i) {
    DecisionTree actual, expected;
    Ok(fit[i].Publish(&actual));
    Ok(LearnHistogramTree(originals[i], &expected, options));
    Check(actual == expected && SameRows(originals[i], samples[i]),
          "GPU-assisted complete tree or CPU partition order changed");
    ++complete_trees;
  }
}
void Cases(GpuBackend &gpu, bool extensive) {
  Trees(gpu, {Fixture(0, 257, 2), Fixture(1, 263, 2), Fixture(7, 131, 14)});
  if (!extensive)
    return;
  for (size_t predictors : {size_t{1}, size_t{2}, size_t{14}})
    for (size_t seed = 0; seed < 12; ++seed)
      Trees(gpu, {Fixture(seed, 71 + seed * 17, predictors)}, {}, false);
  for (uint32_t seed = 0; seed < 32; ++seed) {
    auto s = Fixture(seed, 257, seed % 2 ? 14 : 2);
    uint32_t rng = seed + 17;
    const auto random = [&] { return rng = rng * 1664525u + 1013904223u; };
    s.total_samples = 0;
    for (size_t i = 0; i < s.weights.size(); ++i) {
      s.weights[i] = 1 + (random() >> 17);
      s.total_samples += s.weights[i];
      for (size_t pred = 0; pred < s.predictor_count; ++pred)
        s.residuals[pred][i] = {uint8_t(random() >> 24),
                                uint8_t(random() >> 26)};
      for (size_t prop = 0; prop < s.static_property_count; ++prop)
        s.static_values[prop][i] =
            (random() >> 24) % (s.thresholds[prop].size() + 1);
      for (size_t prop = 2; prop < s.property_count; ++prop)
        s.values[prop - 2][i] =
            (random() >> 24) % (s.thresholds[prop].size() + 1);
    }
    // Exercise exact host penalties for a parent other than Weighted as well.
    if (seed % 3 == 0) {
      std::swap(s.predictors[0], s.predictors[s.predictor_count - 1]);
      std::swap(s.residuals[0], s.residuals[s.predictor_count - 1]);
    }
    Trees(gpu, {s}, {}, false);
  }
  // Exercise reusable slots, heterogeneous fits and different shard counts.
  for (size_t shards : {size_t{1}, size_t{4}, size_t{32}})
    Trees(gpu, {Fixture(3, 517, 1), Fixture(4, 1027, 2), Fixture(6, 139, 14)},
          {shards, 1, 256 * 1024 * 1024}, false);
  {
    auto s = Fixture(2, 257, 2);
    s.total_samples = 0;
    for (size_t i = 0; i < s.weights.size(); ++i) {
      s.weights[i] = i == 256 ? 256 : UINT16_MAX;
      s.total_samples += s.weights[i];
      for (size_t p = 0; p < s.predictor_count; ++p)
        s.residuals[p][i].extra_bits = 63;
    }
    Trees(gpu, {s}, {},
          false); // Exact native 2^24 weighted population ceiling.
  }
  {
    auto s = Fixture(1, 67, 1);
    s.property_count = s.static_property_count = 1;
    for (size_t p = 1; p < 16; ++p)
      s.thresholds[p].clear();
    for (auto &v : s.values)
      v.clear();
    s.static_values[1].clear();
    s.thresholds[0].resize(65535);
    std::iota(s.thresholds[0].begin(), s.thresholds[0].end(), 0);
    for (size_t i = 0; i < s.weights.size(); ++i) {
      s.static_values[0][i] = i % 2 ? 65535 : 1;
      s.residuals[0][i] = {uint8_t(i % 4), 0};
    }
    Trees(gpu, {s}, {}, false);
  }
  {
    auto s = Fixture(1, 97, 2);
    for (auto &v : s.static_values)
      std::fill(v.begin(), v.end(), 0);
    for (auto &v : s.values)
      std::fill(v.begin(), v.end(), 0);
    Trees(gpu, {s}); // Discovered constants.
  }
}
void Failures(GpuBackend &gpu, GpuBackend &foreign) {
  auto samples = Fixture(1, 257, 2);
  HistogramTreeFitter fitter;
  Ok(fitter.Initialize(samples));
  Ok(fitter.Advance());
  auto node = HistogramFittingNodeView(0, fitter.node());
  auto &op = Operation(gpu);
  const HistogramFittingSource source{37, &samples};
  HistogramFittingShape shape;
  Ok(ResolveHistogramFittingSource(source, &shape));
  HistogramFittingPreparationPlan pp;
  HistogramFittingEvaluationPlan ep;
  Ok(ComputeHistogramFittingPreparationPlan({&shape, 1}, &pp));
  Ok(ComputeHistogramFittingEvaluationPlan({&shape, 1}, {&node.shape, 1}, {},
                                           &ep));
  const size_t exact = pp.retained.retained_bytes + ep.dispatch_bytes +
                       ep.output_bytes + ep.scratch_bytes;
  for (size_t capacity :
       {exact - 1, exact,
        pp.retained.retained_bytes + ep.evaluation.peak_bytes}) {
    ResourceBudget budget(capacity);
    ResourceReservation reservation;
    Ok(budget.TryReserve(capacity, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    std::unique_ptr<PreparedGpuHistogramRows> prepared;
    std::unique_ptr<PreparedGpuHistogramEvaluation> output;
    Ok(op.PrepareHistogramRows({&source, 1}, &prepared));
    const auto status =
        op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, {}}, &output);
    Check(capacity < exact ? status.resource_plan_exceeded() : status.ok(),
          "Histogram operation capacity boundary differs");
    if (output)
      Check(budget.snapshot().total.live_capacity_bytes ==
                pp.retained.retained_bytes + ep.retained.retained_bytes,
            "Completed results retained transient evaluation storage");
    output.reset();
    prepared.reset();
    Check(budget.snapshot().peak_backing_bytes <= capacity &&
              budget.snapshot().total.live_capacity_bytes == 0 &&
              budget.snapshot().total.pending_count == 0,
          "Histogram evaluation exceeded admission or leaked backing");
  }
  std::unique_ptr<PreparedGpuHistogramRows> prepared;
  std::unique_ptr<PreparedGpuHistogramEvaluation> output;
  Ok(op.PrepareHistogramRows({&source, 1}, &prepared));
  Ok(op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, {}}, &output));
  const auto *old_prepared = prepared.get();
  const auto *old_output = output.get();
  const std::vector<HistogramPropertyResult> expected(output->results().begin(),
                                                      output->results().end());
  const auto live =
      DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  const auto intact = [&] {
    Check(prepared.get() == old_prepared && output.get() == old_output &&
              SameResults(output->results(), expected) &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes ==
                  live &&
              DefaultResourceBudget().snapshot().total.pending_count == 0,
          "Failed histogram operation changed output or leaked storage");
    ++faults;
  };
  Ok(ArmNextMetalAllocationFailureForTest(gpu));
  Check(op.PrepareHistogramRows({&source, 1}, &prepared).code() ==
            StatusCode::kOutOfMemory,
        "Histogram preparation allocation failure differs");
  intact();
  ArmNextManagedHostAllocationFailureForTest();
  const auto host_failure = op.PrepareHistogramRows({&source, 1}, &prepared);
  DisarmManagedHostAllocationFailureForTest();
  Check(host_failure.code() == StatusCode::kOutOfMemory,
        "Histogram metadata allocation failure differs");
  intact();
  for (unsigned kind = 0; kind < 3; ++kind) {
    if (!kind)
      Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, kind == 1, kind == 2));
    const auto status =
        op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, {}}, &output);
    Check(status.code() == (kind == 0   ? StatusCode::kOutOfMemory
                            : kind == 1 ? StatusCode::kSubmissionFailed
                                        : StatusCode::kDeviceError),
          "Injected histogram device failure differs");
    intact();
  }
  const auto before = gpu.stats();
  Check(op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, {4, 64, 1}},
                                  &output)
                .code() == StatusCode::kUnsupported,
        "Insufficient scratch was not rejected before execution");
  intact();
  Check(!Operation(foreign)
             .EvaluateHistogramNodes({prepared.get(), {&node, 1}, {}}, &output)
             .ok(),
        "Foreign histogram owner accepted");
  intact();
  struct Impostor final : PreparedGpuHistogramRows {
    BackendId id;
    HistogramFittingShape shape;
    BackendId backend_id() const noexcept override { return id; }
    std::span<const HistogramFittingShape> shapes() const noexcept override {
      return {&shape, 1};
    }
  } impostor;
  impostor.id = gpu.id();
  impostor.shape = shape;
  Check(!op.EvaluateHistogramNodes({&impostor, {&node, 1}, {}}, &output).ok(),
        "Nonnative histogram owner accepted");
  intact();
  auto bad = node;
  bad.shape.symbols = 3;
  Check(
      !op.EvaluateHistogramNodes({prepared.get(), {&bad, 1}, {}}, &output).ok(),
      "Malformed node accepted");
  intact();
  Check(gpu.stats().committed_submissions == before.committed_submissions &&
            gpu.stats().successful_allocations == before.successful_allocations,
        "Invalid preflight reached device work");
  std::vector<uint32_t> rows(node.rows.begin(), node.rows.end());
  rows[17] = uint32_t(samples.weights.size());
  bad = node;
  bad.rows = rows;
  for (bool constants : {false, true}) {
    bad.shape.inherited_constants = constants ? 0xffff : 0;
    Check(
        op.EvaluateHistogramNodes({prepared.get(), {&bad, 1}, {}}, &output)
                .code() == StatusCode::kInvalidArgument,
        "Device membership validation failed, including an all-constant node");
    intact();
  }
  std::vector<uint32_t> extras(node.parent_extra_bits.begin(),
                               node.parent_extra_bits.end());
  ++extras[0];
  bad = node;
  bad.parent_extra_bits = extras;
  Check(op.EvaluateHistogramNodes({prepared.get(), {&bad, 1}, {}}, &output)
                .code() == StatusCode::kInvalidArgument,
        "Device parent summary mismatch was accepted");
  intact();
  // Completed output owns its backing independently of all inputs.
  auto concurrent = [&] {
    for (unsigned repeat = 0; repeat < 4; ++repeat) {
      std::unique_ptr<PreparedGpuHistogramEvaluation> result;
      Ok(op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, {}}, &result));
      Check(SameResults(result->results(), expected),
            "Concurrent histogram evaluations shared mutable scratch");
    }
  };
  auto first = std::async(std::launch::async, concurrent);
  auto second = std::async(std::launch::async, concurrent);
  first.get();
  second.get();
  {
    auto constants = node;
    constants.shape.inherited_constants = 0xffff;
    std::unique_ptr<PreparedGpuHistogramEvaluation> result;
    Ok(op.EvaluateHistogramNodes({prepared.get(), {&constants, 1}, {}},
                                 &result));
    for (const auto &property : result->results()) {
      Check(property.is_constant == 1, "Inherited constant was not published");
      for (const auto &winner : property.winners)
        Check(!winner.valid &&
                  winner.left_cost == std::numeric_limits<float>::max() &&
                  winner.right_cost == std::numeric_limits<float>::max(),
              "All-constant evaluation invented a split");
    }
  }
  prepared.reset();
  Check(SameResults(output->results(), expected),
        "Completed histogram result borrows prepared lifetime");
}
void OverflowValidation(GpuBackend &gpu) {
  auto samples = Fixture(1, 65540, 1);
  std::fill(samples.weights.begin(), samples.weights.end(), 1);
  samples.weights[0] = UINT16_MAX;
  samples.total_samples = samples.weights.size() - 1 + UINT16_MAX;
  HistogramTreeFitter fit;
  Ok(fit.Initialize(samples));
  Ok(fit.Advance());
  auto node = HistogramFittingNodeView(0, fit.node());
  Check(fit.ready(), "Overflow validation fixture has no ready node");
  const HistogramFittingSource source{47, &samples};
  std::unique_ptr<PreparedGpuHistogramRows> prepared;
  Ok(Operation(gpu).PrepareHistogramRows({&source, 1}, &prepared));
  std::vector<uint32_t> repeated(node.rows.size(), 0);
  node.rows = repeated;
  node.shape.inherited_constants = 0xffff;
  std::unique_ptr<PreparedGpuHistogramEvaluation> output;
  const auto status = Operation(gpu).EvaluateHistogramNodes(
      {prepared.get(), {&node, 1}, {}}, &output);
  Check(status.code() == StatusCode::kInvalidArgument &&
            status.message().find("population") != std::string_view::npos &&
            !output,
        "Weighted multiset overflow escaped the 64-bit population guard");
  ++faults;
}
void Profiles(GpuBackend &gpu) {
  auto samples = Fixture(3, 1025, 2);
  // Exercise a nonempty final partial SIMD scan, including its last bin.
  samples.thresholds[15].resize(34);
  std::iota(samples.thresholds[15].begin(), samples.thresholds[15].end(), -17);
  for (size_t i = 0; i < samples.weights.size(); ++i)
    samples.values[13][i] = static_cast<uint8_t>(i % 35);
  HistogramTreeFitter fitter;
  Ok(fitter.Initialize(samples));
  Ok(fitter.Advance());
  auto node = HistogramFittingNodeView(0, fitter.node());
  const HistogramFittingSource source{83, &samples};
  auto &op = Operation(gpu);
  HistogramFittingShape shape;
  Ok(ResolveHistogramFittingSource(source, &shape));
  HistogramFittingPreparationPlan preparation;
  Ok(ComputeHistogramFittingPreparationPlan({&shape, 1}, &preparation));
  const HistogramFittingExecutionOptions execution{4, 1, 64 * 1024 * 1024};
  for (bool constants : {false, true}) {
    node.shape.inherited_constants =
        constants ? (1u << shape.properties) - 1 : 0;
    HistogramFittingEvaluationPlan plan;
    Ok(ComputeHistogramFittingEvaluationPlan({&shape, 1}, {&node.shape, 1},
                                             execution, &plan));
    std::unique_ptr<PreparedGpuHistogramRows> prepared;
    Ok(op.PrepareHistogramRows({&source, 1}, &prepared));
    std::unique_ptr<PreparedGpuHistogramEvaluation> expected;
    Ok(op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, execution},
                                 &expected));
    for (auto [mode, trace_wait] : {
             std::pair{HistogramFittingProfileMode::kDisabled, false},
             std::pair{HistogramFittingProfileMode::kWall, false},
             std::pair{HistogramFittingProfileMode::kStage, false},
             std::pair{HistogramFittingProfileMode::kWall, true},
             std::pair{HistogramFittingProfileMode::kStage, true}}) {
      HostStorageBound profile_bound;
      Ok(op.HistogramFittingProfileStorageBound(plan, mode, &profile_bound));
      Check((profile_bound.peak_bytes != 0) ==
                (mode == HistogramFittingProfileMode::kStage),
            "Histogram profile reserved backing outside stage mode");
      const size_t capacity =
          plan.evaluation.peak_bytes + profile_bound.peak_bytes;
      ResourceBudget budget(capacity);
      ResourceReservation reservation;
      Ok(budget.TryReserve(capacity, &reservation));
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      HistogramFittingBatchProfile profile;
      std::unique_ptr<PreparedGpuHistogramEvaluation> output;
      Ok(op.EvaluateHistogramNodes(
          {prepared.get(), {&node, 1}, execution, mode, &profile, trace_wait}, &output));
      Compare(output->results(), expected->results());
      if (mode == HistogramFittingProfileMode::kDisabled) {
        Check(profile == HistogramFittingBatchProfile{},
              "Disabled profiling wrote its sink");
      } else {
        Check(profile.nodes == 1 && profile.rows == node.rows.size() &&
                  profile.tasks == plan.tasks && profile.waves == plan.tasks &&
                  profile.wall_nanoseconds[4] > 0 &&
                  profile.command_buffer_gpu_nanoseconds > 0,
              "Histogram wall profile missed work");
        const auto &t = profile.wait_trace;
        if (trace_wait) {
          Check(t.commit_begin > 0 && t.commit_begin <= t.commit_end &&
                    t.commit_end <= t.wait_begin && t.wait_begin <= t.blocking_begin &&
                    t.blocking_begin <= t.blocking_end && t.blocking_end <= t.cpu_resumed &&
                    t.commit_begin <= t.gpu_begin && t.gpu_begin < t.gpu_end &&
                    t.gpu_end <= t.blocking_end &&
                    t.commit_begin <= t.kernel_begin && t.kernel_begin <= t.kernel_end &&
                    t.kernel_end <= t.blocking_end,
                "Histogram wait trace endpoints are inconsistent");
        } else {
          Check(t == HistogramFittingWaitTrace{}, "Disabled wait trace wrote endpoints");
        }
        for (size_t phase = 0; phase < 5; ++phase) {
          const size_t dispatches = mode == HistogramFittingProfileMode::kStage
                                        ? (phase == 0 ? 1 : plan.tasks)
                                        : 0;
          Check(profile.dispatches[phase] == dispatches &&
                    (profile.gpu_nanoseconds[phase] > 0) == (dispatches > 0),
                "Histogram timestamp coverage differs");
        }
      }
      output.reset();
      Check(budget.snapshot().peak_backing_bytes <= capacity &&
                budget.snapshot().total.live_capacity_bytes == 0 &&
                budget.snapshot().total.pending_count == 0,
            "Histogram profiling exceeded admission or leaked");
    }
    for (unsigned failure = 0; failure < 3; ++failure) {
      HistogramFittingBatchProfile profile;
      profile.rows = 913;
      const auto sentinel = profile;
      const auto *old = expected.get();
      if (failure == 0)
        ArmNextManagedHostAllocationFailureForTest();
      else
        Ok(ArmNextMetalSubmissionFailureForTest(gpu, failure == 1,
                                                failure == 2));
      const auto s =
          op.EvaluateHistogramNodes({prepared.get(),
                                     {&node, 1},
                                     execution,
                                     HistogramFittingProfileMode::kStage,
                                     &profile, true},
                                    &expected);
      DisarmManagedHostAllocationFailureForTest();
      Check(!s.ok() && old == expected.get() && profile == sentinel,
            "Failed profile published partial results");
      ++faults;
    }
  }
}
void SubmissionVariants(GpuBackend &gpu) {
  auto samples = Fixture(6, 1025, 2);
  HistogramTreeFitter fitter;
  Ok(fitter.Initialize(samples));
  Ok(fitter.Advance());
  auto node = HistogramFittingNodeView(0, fitter.node());
  const HistogramFittingSource source{113, &samples};
  auto &op = Operation(gpu);
  std::unique_ptr<PreparedGpuHistogramRows> prepared;
  Ok(op.PrepareHistogramRows({&source, 1}, &prepared));
  std::vector<HistogramPropertyResult> expected(samples.property_count);
  Ok(EvaluateHistogramNodeOnCpu(fitter, source.source_id, expected));
  for (int mode : {0, 1, 2}) for (bool multiple : {false, true}) {
    const bool reuse = mode != 0;
    Ok(gpu.TrimPreparationCache());
    HistogramFittingExecutionOptions options{4, 3, 64 * 1024 * 1024, reuse, multiple};
    options.reuse_larger_buffers = mode == 2;
    HistogramFittingBatchProfile profile;
    std::unique_ptr<PreparedGpuHistogramEvaluation> first, second;
    const auto evaluate = [&](auto *out) {
      const auto submissions = gpu.stats().committed_submissions;
      HistogramFittingEvaluationPlan plan;
      Ok(ComputeHistogramFittingEvaluationPlan(prepared->shapes(), {&node.shape, 1}, options, &plan));
      const auto capacities = op.HistogramFittingBufferCapacities(plan, options);
      Ok(op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, options,
          HistogramFittingProfileMode::kWall, &profile, true, capacities}, out));
      Check(gpu.stats().committed_submissions == submissions + 1,
            "Encoder boundaries changed submission count");
      Compare((*out)->results(), expected);
      Check(profile.compute_encoders == (multiple ? 1 + 4 * profile.waves : 1),
            "Wrong compute encoder count");
    };
    evaluate(&first);
    Check(profile.reused_buffers == 0, "Cold evaluation unexpectedly reused backing");
    evaluate(&second);
    Check(profile.reused_buffers == (reuse ? 6u : 0u),
          "Held output was reused or completed scratch was not reused");
    Compare(first->results(), expected);
    first.reset(); second.reset();
    evaluate(&first);
    Check(profile.reused_buffers == (reuse ? 7u : 0u),
          "Released result backing was not reused");
    for (bool completion : {false, true}) {
      const auto old_profile = profile;
      const auto *old_output = first.get();
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, !completion, completion));
      const auto status = op.EvaluateHistogramNodes(
          {prepared.get(), {&node, 1}, options, HistogramFittingProfileMode::kWall,
           &profile, true}, &first);
      Check(!status.ok() && first.get() == old_output && profile == old_profile,
            "Submission experiment published a failed result or trace");
      Compare(first->results(), expected);
      ++faults;
    }
    first.reset();
    Ok(gpu.TrimPreparationCache());
    if (reuse) {
      evaluate(&first);
      Ok(gpu.TrimPreparationCache());
      first.reset(); // A lease acquired before trimming cannot refill the cache.
      evaluate(&first);
      Check(profile.reused_buffers == 0, "Trimmed generation refilled histogram cache");
      first.reset();
      Ok(metal_internal::EmptyMetalAqScratchArenasForTesting(gpu));
      evaluate(&first);
      // Check large scratch recovery. Metal may retain small suballocations
      // when asked to purge; their contents are still overwritten and checked.
      Check(!(profile.reused_buffers & 4), "Purged histogram scratch was reused");
      first.reset();
      // Concurrent requests borrow immutable rows but exclusively acquire their
      // own transient/result allocations, even when sharing one backend cache.
      auto pending = std::async(std::launch::async, [&] {
        std::unique_ptr<PreparedGpuHistogramEvaluation> result;
        Ok(op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, options}, &result));
        return result;
      });
      Ok(op.EvaluateHistogramNodes({prepared.get(), {&node, 1}, options}, &first));
      second = pending.get();
      Compare(second->results(), expected);
      Compare(first->results(), expected);
      first.reset(); second.reset();
    }
    Ok(gpu.TrimPreparationCache());
    Trees(gpu, {Fixture(0, 257, 2), Fixture(3, 263, 2)}, options);
    Ok(gpu.TrimPreparationCache());
  }
}
void CapacityReuse(GpuBackend &gpu) {
  Ok(gpu.TrimPreparationCache());
  auto &op = Operation(gpu);
  std::vector<HistogramTrainingSamples> samples{
      Fixture(6, 1025, 2), Fixture(9, 1033, 2), Fixture(6, 257, 2)};
  std::vector<HistogramTreeFitter> fitters(3);
  std::vector<HistogramFittingSource> sources;
  std::vector<HistogramFittingNodeInput> nodes;
  std::vector<HistogramPropertyResult> expected(48);
  for (size_t i = 0; i < 3; ++i) {
    sources.push_back({uint32_t(i + 1), &samples[i]});
    Ok(fitters[i].Initialize(samples[i])); Ok(fitters[i].Advance());
    nodes.push_back(HistogramFittingNodeView(i, fitters[i].node()));
    Ok(EvaluateHistogramNodeOnCpu(fitters[i], sources[i].source_id,
                                  std::span(expected).subspan(16 * i, 16)));
  }
  constexpr size_t limit = 256 * 1024 * 1024;
  ResourceBudget budget(limit);
  ResourceReservation preparation;
  Ok(budget.TryReserve(4 * 1024 * 1024, &preparation));
  std::unique_ptr<PreparedGpuHistogramRows> prepared;
  {
    ResourceContextScope scope({&preparation, ResourceClass::kPreparation});
    Ok(op.PrepareHistogramRows(sources, &prepared));
  }
  preparation.Reset();
  HistogramFittingExecutionOptions options;
  const auto large = std::span(nodes).first(2), small = std::span(nodes).subspan(2);
  const auto plan_for = [&](auto view) {
    std::vector<HistogramFittingNodeShape> shapes;
    for (const auto &node : view) shapes.push_back(node.shape);
    HistogramFittingEvaluationPlan plan;
    Ok(ComputeHistogramFittingEvaluationPlan(prepared->shapes(), shapes, options, &plan));
    return plan;
  };
  const auto lp = plan_for(large), sp = plan_for(small);
  const std::array large_bytes{lp.output_bytes, lp.dispatch_bytes, lp.scratch_bytes};
  const std::array small_bytes{sp.output_bytes, sp.dispatch_bytes, sp.scratch_bytes};
  for (size_t i = 0; i < 3; ++i) Check(large_bytes[i] > small_bytes[i], "Fixture must vary every capacity");
  HistogramFittingBatchProfile profile;
  const auto evaluate = [&](auto view, const auto &plan, auto caps, auto *out) {
    HostStorageBound bound;
    Ok(ComputeHistogramFittingCapacityBound(plan, caps, &bound));
    ResourceReservation reservation;
    Ok(budget.TryReserve(bound.peak_bytes, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    Ok(op.EvaluateHistogramNodes({prepared.get(), view, options,
        HistogramFittingProfileMode::kWall, &profile, false, caps}, out));
    Compare((*out)->results(), std::span(expected).subspan(view.front().shape.fit * 16, view.size() * 16));
    Check(budget.snapshot().committed_bytes() <= limit, "Reuse exceeded domain budget");
  };
  std::unique_ptr<PreparedGpuHistogramEvaluation> result, held;
  evaluate(large, lp, large_bytes, &result);
  Check(profile.reused_buffers == 0, "Cold capacity call reused backing");
  result.reset();
  evaluate(small, sp, op.HistogramFittingBufferCapacities(sp, options), &result);
  Check(profile.reused_buffers == 7 && profile.buffer_requested_bytes == small_bytes &&
        profile.buffer_capacity_bytes == large_bytes, "Large-to-small reuse lost capacity or logical bounds");
  const auto usage = budget.snapshot().total;
  Check(usage.live_capacity_bytes - usage.live_requested_bytes == lp.output_bytes - sp.output_bytes,
        "Held larger output was charged as a smaller allocation");
  result.reset();
  evaluate(large, lp, op.HistogramFittingBufferCapacities(lp, options), &held);
  Check(profile.reused_buffers == 7, "Small request shrank reusable backing");
  evaluate(small, sp, op.HistogramFittingBufferCapacities(sp, options), &result);
  Check(profile.reused_buffers == 6, "Held large output was overwritten");
  Compare(held->results(), std::span(expected).first(32));
  held.reset(); result.reset(); // Smaller late release must not replace larger output.
  Check(op.HistogramFittingBufferCapacities(sp, options) == large_bytes,
        "Late small release discarded larger cached output");

  // A snapshot cannot authorize a larger backing that arrives after planning.
  evaluate(small, sp, small_bytes, &result);
  Check(profile.reused_buffers == 0 && profile.buffer_capacity_bytes == small_bytes,
        "Stale snapshot consumed unadmitted larger backing");
  result.reset();
  evaluate(large, lp, large_bytes, &result); result.reset();
  const auto caps = op.HistogramFittingBufferCapacities(sp, options);
  HostStorageBound expanded;
  Ok(ComputeHistogramFittingCapacityBound(sp, caps, &expanded));
  ResourceReservation pressure, attempt;
  Ok(budget.TryReserve(limit - budget.snapshot().committed_bytes() - sp.evaluation.peak_bytes,
                       &pressure));
  Check(!budget.TryReserve(expanded.peak_bytes, &attempt).ok(), "Pressure must reject larger envelope");
  evaluate(small, sp, small_bytes, &result);
  Check(profile.buffer_capacity_bytes == small_bytes, "Exact fallback retained oversized backing");
  result.reset(); pressure.Reset();
  // Cached backing from this domain cannot be silently lent to the default one.
  Ok(op.EvaluateHistogramNodes({prepared.get(), small, options,
      HistogramFittingProfileMode::kWall, &profile, false, small_bytes}, &result));
  Compare(result->results(), std::span(expected).subspan(32));
  Check(profile.reused_buffers == 0, "Cross-domain cached backing was reused");
  result.reset(); prepared.reset();
  Ok(gpu.TrimPreparationCache());
  const auto end = budget.snapshot();
  Check(!end.committed_bytes() && !end.total.backing_count && !end.open_reservations,
        "Capacity reuse leaked a charge or reservation");
  // Bounds fail atomically, including overflow and unused scratch capacity.
  auto sentinel = expanded;
  auto invalid = small_bytes; invalid[0] = sp.output_bytes - 1;
  Check(!ComputeHistogramFittingCapacityBound(sp, invalid, &expanded).ok() &&
        expanded == sentinel, "Capacity bound accepted undersized output");
  invalid = small_bytes; invalid[0] = SIZE_MAX;
  Check(!ComputeHistogramFittingCapacityBound(sp, invalid, &expanded).ok() &&
        expanded == sentinel, "Capacity bound overflowed");
  auto no_scratch = sp; no_scratch.scratch_bytes = 0;
  invalid = small_bytes; invalid[2] = 1;
  Check(!ComputeHistogramFittingCapacityBound(no_scratch, invalid, &expanded).ok() &&
        expanded == sentinel, "Capacity bound accepted unused scratch backing");
}
} // namespace
int main() try {
  {
    std::unique_ptr<GpuBackend> gpu, embedded;
    Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    Ok(CreateEmbeddedMetalBackend({}, &embedded));
    std::cout << "Histogram fitting device: " << gpu->name() << std::endl;
    Cases(*gpu, true);
    Cases(*embedded, false);
    Failures(*gpu, *embedded);
    OverflowValidation(*gpu);
    Profiles(*gpu);
    SubmissionVariants(*gpu);
    CapacityReuse(*gpu);
  }
  Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes == 0 &&
            DefaultResourceBudget().snapshot().total.pending_count == 0,
        "Histogram fitting tests leaked backing");
  std::cout << batches << " exact batches, " << properties
            << " exact property results, " << complete_trees
            << " complete trees, " << faults << " failure checks\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << " after " << batches << " batches and "
            << complete_trees << " trees\n";
  return 1;
}
