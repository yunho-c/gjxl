// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/tokenization.h"
#include "core/thread_budget.h"
#include "gpu/ops/modular_stream_tokens.h"
#if GJXL_STREAM_BATCH_METAL
#include "gpu/metal/metal_backend.h"
#endif
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Fixture {
  std::vector<std::vector<int32_t>> pixels;
  std::vector<ModularStreamFeatureSlice> slices;
  std::vector<WeightedFeatureSlice> jobs;
  std::vector<WeightedPredictorParameters> parameters;
  std::vector<WeightedPredictorFeature> expected_features;
  std::vector<EntropyToken> expected_tokens;
  std::array<DecisionTree, 4> decisions;
  std::array<ModularStreamTokenTree, 4> trees;
  Fixture(bool large_metadata = false) {
    const std::array<WeightedPredictorParameters, 4> presets{
        WeightedPredictorParameters{},
        {{8, 8, 4, 0, 3, 23, 2}, {13, 12, 12, 12}},
        {{7, 15, 11, 5, 3, 2, 1}, {15, 0, 7, 3}},
        {{31, 31, 31, 31, 31, 31, 31}, {15, 15, 15, 15}}};
    decisions[0].nodes[0].predictor = Predictor::kWeighted;
    decisions[1].size = 3;
    decisions[1].nodes[0] = {.property = 15, .split = 0, .left = 1, .right = 2};
    decisions[1].nodes[1].predictor = Predictor::kWeighted;
    decisions[1].nodes[2].predictor = Predictor::kGradient;
    decisions[2].size = 27;
    for (size_t i = 0; i < 13; ++i) {
      decisions[2].nodes[2 * i] = {.property = 3,
                                   .split = static_cast<int32_t>(i * 8),
                                   .left = static_cast<TreeIndex>(2 * i + 2),
                                   .right = static_cast<TreeIndex>(2 * i + 1)};
      decisions[2].nodes[2 * i + 1].predictor = static_cast<Predictor>(i);
    }
    decisions[2].nodes[26].predictor = Predictor::kAverage4;
    decisions[3].size = 5;
    decisions[3].nodes[0] = {.property = 0, .split = 1, .left = 1, .right = 2};
    decisions[3].nodes[1].predictor = Predictor::kWeighted;
    decisions[3].nodes[2] = {.property = 1, .split = 30, .left = 3, .right = 4};
    decisions[3].nodes[3].predictor = Predictor::kLeft;
    decisions[3].nodes[4].predictor = Predictor::kTop;
    for (size_t i = 0; i < trees.size(); ++i)
      trees[i] = {3 * i, 3, &decisions[i]};
    const std::array<Extent2D, 6> shapes{
        {{256, 256}, {17, 19}, {1, 31}, {127, 1}, {255, 128}, {3, 5}}};
    size_t total = 0;
    for (size_t i = 0; i < 12; ++i) {
      const auto extent = large_metadata && i == 0 ? Extent2D{1024, 4} : shapes[i % shapes.size()];
      const size_t stride = extent.width + 3, origin = stride + 1;
      auto &plane = pixels.emplace_back((extent.height + 2) * stride, INT32_MIN);
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x)
          plane[origin + y * stride + x] =
              int32_t((x * 337 + y * 911 + x * y * 7 + i * 331) % 131071) - 65535;
      slices.push_back({{std::span(plane).subspan(origin), {extent}, stride},
                        presets[i / 3],
                        static_cast<uint32_t>(27 + i / 3 * 3),
                        static_cast<uint32_t>(i % 3)});
      jobs.push_back({slices.back().input, total});
      parameters.push_back(slices.back().parameters);
      total += extent.width * extent.height;
    }
    expected_features.resize(total);
    expected_tokens.resize(total);
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto &s = slices[i];
      const size_t count = s.input.descriptor.extent.width * s.input.descriptor.extent.height;
      Ok(PrepareWeightedPredictorFeatures(
          s.input, s.parameters,
          std::span(expected_features).subspan(jobs[i].feature_offset, count)));
      TreeLayout layout;
      Ok(ValidateTree(decisions[i / 3], &layout));
      Ok(TokenizeValidatedModularChannel(
          s.input, s.channel, s.stream_id, decisions[i / 3], layout, s.parameters,
          std::span(expected_tokens).subspan(jobs[i].feature_offset, count)));
    }
  }
};
void PlansAndCpu() {
  Fixture f;
  ModularStreamFeatureStoragePlan features;
  ModularStreamTokenStoragePlan tokens;
  ArmNextManagedHostAllocationFailureForTest();
  const auto a = ComputeModularStreamFeatureStoragePlan(f.slices, 4, &features);
  const auto b = ComputeModularStreamTokenStoragePlan(f.expected_tokens.size(), f.slices.size(),
                                                      f.trees, &tokens);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Ok(a);
  Ok(b);
  HostStorageBound exact_bound, upper_bound;
  Ok(ComputeModularStreamTokenStorageBound(features.samples, f.slices.size(), tokens.nodes,
                                           &exact_bound));
  Ok(ComputeModularStreamTokenStorageBound(features.samples, f.slices.size(),
      f.trees.size() * kMaximumTreeNodes, &upper_bound));
  Check(exact_bound == tokens.generation && upper_bound.peak_bytes >= tokens.generation.peak_bytes,
        "Protected token completion failed to bound the exact layout");
  Check(!ComputeModularStreamTokenStorageBound(features.samples, f.slices.size(), 0, &upper_bound).ok() &&
        !ComputeModularStreamTokenStorageBound(features.samples, f.slices.size(), SIZE_MAX, &upper_bound).ok(),
        "Token completion accepted an invalid node bound");
  Check(pending && features.wavefront_shapes && features.samples == f.expected_tokens.size(),
        "Batch planner allocated or miscounted samples");
  for (size_t participants : {1, 2, 4}) {
    std::shared_ptr<const ExecutionDomain> domain;
    Ok(ExecutionDomain::Create({0, 4}, &domain));
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, participants));
    EncodeScope encode(participants);
    WeightedFeatureBatchStoragePlan bound;
    Ok(ComputeWeightedFeatureBatchStoragePlan(f.slices.size(), features.maximum_width, participants,
                                              &bound));
    ResourceBudget budget(bound.execution.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.execution.peak_bytes, &reservation));
    std::vector<WeightedPredictorFeature> output(f.expected_features.size());
    FeaturePreparationProfile profile;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation, true, domain.get()});
      Ok(PrepareWeightedPredictorFeatureBatchWithParameters(f.jobs, f.parameters, output,
                                                            participants, &profile));
    }
    Check(output == f.expected_features && profile.peak_participants == participants &&
              !budget.snapshot().total.backing_count &&
              budget.snapshot().peak_backing_bytes <= bound.execution.peak_bytes,
          "Mixed-header CPU batch lost parity, participation or admission");
  }
  for (size_t fault = 0; fault < 6; ++fault) {
    auto bad = f.slices;
    if (fault == 0)
      bad[0].input.stride = 1;
    if (fault == 1)
      bad[0].parameters.weights[0] = 16;
    if (fault == 2)
      bad[0].stream_id = 65535;
    if (fault == 3)
      bad[0].channel = 65535;
    if (fault == 4)
      bad[0].input.descriptor.extent = {};
    if (fault == 5)
      bad.resize(kMaximumModularFeatureBatchSlices + 1, bad[0]);
    const auto original = features;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = ComputeModularStreamFeatureStoragePlan(bad, 4, &features);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(!status.ok() && pending && features == original,
          "Invalid feature plan allocated/published");
  }
  for (size_t fault = 0; fault < 4; ++fault) {
    auto bad = f.trees;
    if (fault == 0)
      bad[0].first_slice = 1;
    if (fault == 1)
      bad[1].slice_count = 0;
    if (fault == 2)
      bad[2].slice_count = f.slices.size();
    if (fault == 3)
      bad[3].tree = nullptr;
    const auto original = tokens;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status =
        ComputeModularStreamTokenStoragePlan(features.samples, f.slices.size(), bad, &tokens);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(!status.ok() && pending && tokens == original,
          "Invalid tree partition allocated/published");
  }
  const ModularTokenDeviceLimits short_limit{features.samples - 1, 1024, size_t{1} << 30};
  Check(ComputeModularStreamFeatureStoragePlan(f.slices, 1, &features, &short_limit).code() ==
                StatusCode::kUnsupported &&
            ComputeModularStreamTokenStoragePlan(features.samples, f.slices.size(), f.trees,
                                                 &tokens, &short_limit)
                    .code() == StatusCode::kUnsupported,
        "Device sample limit ignored");
}
#if GJXL_STREAM_BATCH_METAL
size_t gpu_calls = 0;
void MetalCase(GpuBackend &gpu, Fixture &f, WeightedFeatureBackend producer) {
  auto *op = QueryGpuModularStreamProcessing(gpu);
  Check(op, "Missing stream processing capability");
  const auto limits = op->modular_stream_limits();
  ModularStreamFeatureStoragePlan fp;
  ModularStreamTokenStoragePlan tp;
  Ok(ComputeModularStreamFeatureStoragePlan(f.slices, 4, &fp, &limits));
  Ok(ComputeModularStreamTokenStoragePlan(fp.samples, f.slices.size(), f.trees, &tp, &limits));
  const size_t preparation = std::max(fp.cpu_preparation.peak_bytes, fp.gpu_preparation.peak_bytes);
  ResourceBudget budget(
      std::max(preparation, fp.retained.retained_bytes + tp.generation.peak_bytes));
  std::unique_ptr<PreparedGpuModularStreamFeatures> features;
  std::unique_ptr<PreparedGpuModularTokens> tokens;
  ModularProfile profile;
  {
    ResourceReservation reservation;
    Ok(budget.TryReserve(preparation, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    ProfileSession profiling(&profile);
    Ok(op->PrepareModularStreamFeatures(f.slices, &features, 4, producer));
  }
  gpu_calls += profile.adaptive_gpu.feature_preparation.gpu_calls;
  if (!fp.wavefront_shapes || producer != WeightedFeatureBackend::kGpu)
    Check(profile.adaptive_gpu.feature_preparation.gpu_calls == 0, "Unexpected GPU feature route");
  const auto views = features->weighted_views();
  Check(views.size() == f.slices.size() && features->sample_count() == fp.samples,
        "Feature owner shape changed");
  for (size_t i = 0; i < views.size(); ++i) {
    Ok(ValidateWeightedFeatureView(views[i], f.slices[i].input, f.slices[i].parameters));
    Check(std::ranges::equal(views[i].values,
                             std::span(f.expected_features)
                                 .subspan(f.jobs[i].feature_offset, views[i].values.size())),
          "Metal stream features differ from scalar recurrence");
  }
  {
    ResourceReservation reservation;
    Ok(budget.TryReserve(tp.generation.peak_bytes, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    Ok(op->GenerateModularStreamTokens(*features, f.trees, &tokens));
  }
  ModularTokenHostView host;
  Ok(tokens->BorrowHostView(&host));
  Check(host.values.size() == f.expected_tokens.size() &&
            host.contexts.size() == host.values.size(),
        "Token size changed");
  for (size_t i = 0; i < f.expected_tokens.size(); ++i)
    Check(host.values[i] == f.expected_tokens[i].value &&
              host.contexts[i] == f.expected_tokens[i].context,
          "Mixed-tree GPU tokens differ from scalar tokens");
  // Tokens own their backing independently of features and dispatch state.
  features.reset();
  Check(host.values.back() == f.expected_tokens.back().value,
        "Tokens borrowed released feature backing");
  tokens.reset();
  Check(!budget.snapshot().total.backing_count && !budget.snapshot().total.pending_count,
        "Batch operation leaked admitted backing or pending work");
}
void MetalFailures(GpuBackend &gpu) {
  Fixture f;
  auto *op = QueryGpuModularStreamProcessing(gpu);
  std::unique_ptr<PreparedGpuModularStreamFeatures> features;
  Ok(op->PrepareModularStreamFeatures(f.slices, &features, 1, WeightedFeatureBackend::kGpu));
  std::unique_ptr<PreparedGpuModularTokens> tokens;
  Ok(op->GenerateModularStreamTokens(*features, f.trees, &tokens));
  const auto *old_features = features.get();
  const auto *old_tokens = tokens.get();
  {
    ResourceBudget budget(1);
    ResourceReservation reservation;
    Ok(budget.Reserve(1, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    const auto prepared =
        op->PrepareModularStreamFeatures(f.slices, &features, 1, WeightedFeatureBackend::kGpu);
    const auto generated = op->GenerateModularStreamTokens(*features, f.trees, &tokens);
    Check(prepared.resource_plan_exceeded() && generated.resource_plan_exceeded() &&
              features.get() == old_features && tokens.get() == old_tokens &&
              !budget.snapshot().total.backing_count && !budget.snapshot().total.pending_count,
          "Unadmitted batch execution published or leaked backing");
  }
  for (size_t fault = 0; fault < 4; ++fault)
    for (bool preparation : {false, true}) {
      // Token generation writes descriptors/nodes directly into admitted Metal
      // backing; it has no ManagedVector allocation to fail.
      if (fault == 0 && !preparation)
        continue;
      if (fault == 0)
        ArmNextManagedHostAllocationFailureForTest();
      if (fault == 1)
        Ok(ArmNextMetalAllocationFailureForTest(gpu));
      if (fault >= 2)
        Ok(ArmNextMetalSubmissionFailureForTest(gpu, fault == 2, fault == 3));
      const auto status = preparation
                              ? op->PrepareModularStreamFeatures(f.slices, &features, 1,
                                                                 WeightedFeatureBackend::kGpu)
                              : op->GenerateModularStreamTokens(*features, f.trees, &tokens);
      DisarmManagedHostAllocationFailureForTest();
      if (status.ok() || features.get() != old_features || tokens.get() != old_tokens)
        throw std::runtime_error("Batch failure publication: fault=" + std::to_string(fault) +
                                 " preparation=" + std::to_string(preparation) +
                                 " status=" + std::string(status.message()));
    }
  auto bad_trees = f.trees;
  DecisionTree invalid;
  invalid.nodes[0].multiplier = 2;
  bad_trees[0].tree = &invalid;
  Check(op->GenerateModularStreamTokens(*features, bad_trees, &tokens).code() ==
                StatusCode::kInvalidArgument &&
            tokens.get() == old_tokens,
        "Unrepresentable token published an owner");
  auto bad_slices = f.slices;
  int32_t extreme = INT32_MAX;
  bad_slices[0].input = {{&extreme, 1}, {{1, 1}}, 1};
  Check(op->PrepareModularStreamFeatures(bad_slices, &features, 1, WeightedFeatureBackend::kGpu)
                    .code() == StatusCode::kUnsupported &&
            features.get() == old_features,
        "Failed recurrence published or retried");
  std::unique_ptr<GpuBackend> other;
  Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &other));
  Check(QueryGpuModularStreamProcessing(*other)
                    ->GenerateModularStreamTokens(*features, f.trees, &tokens)
                    .code() == StatusCode::kInvalidArgument &&
            tokens.get() == old_tokens,
        "Cross-backend feature owner accepted");
}
#endif
} // namespace
int main(int argc, char **argv) try {
  PlansAndCpu();
#if GJXL_STREAM_BATCH_METAL
  if (argc == 2 && std::string_view(argv[1]) == "--metal") {
    std::unique_ptr<GpuBackend> gpu;
    Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    Fixture normal, metadata(true);
    for (auto producer : {WeightedFeatureBackend::kCpu, WeightedFeatureBackend::kGpu,
                          WeightedFeatureBackend::kAuto}) {
      MetalCase(*gpu, normal, producer);
      MetalCase(*gpu, metadata, producer);
    }
    if (!gpu_calls) {
      std::cout << "GPU weighted feature pipeline unavailable\n";
      return 77;
    }
    MetalFailures(*gpu);
    std::cout << "Mixed-stream Metal feature/token parity and publication checks passed; GPU "
                 "feature batches="
              << gpu_calls << '\n';
  }
#endif
  std::cout << "Batch planning and mixed-header CPU parallel parity passed\n";
  return 0;
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n';
  return 1;
}
