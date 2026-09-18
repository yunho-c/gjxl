// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <iostream>

#include "codec/coefficient_order_population_internal.h"
#include "codec/frontend_storage_plan.h"
#include "codec/vardct_frame_view_internal.h"
#include "cuda_sparse_resident_fixture.h"
#include "gpu/cuda/cuda_resource_internal.h"
#include "gpu/cuda/cuda_storage_plan.h"

namespace {
using namespace resident_sparse_test;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::cuda_internal;

void Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  Require(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
              s.total.pending_count == 0 && s.open_reservations == 0,
          "Resident CUDA preparation leaked managed backing");
}

void Case(GpuBackend &gpu, Extent2D extent, bool evaluation_free, bool frontend,
          bool dc_policy) {
  Fixture f(extent, 0.1f);
  CudaResidentStorageOptions options{
      .source = extent,
      .evaluation = {.evaluation_free = evaluation_free},
      .resident_frontend = frontend,
      .omit_initial_search_data = frontend};
  options.evaluation.profile.adaptive_dc_smoothing = dc_policy;
  options.evaluation.profile.extra_dc_precision = dc_policy ? 1 : 0;
  options.evaluation.dc_quantization =
      dc_policy ? DcQuantizationMode::kPredictionAware
                : DcQuantizationMode::kRound;
  options.evaluation.dc_prediction = VarDctDcPrediction::kWeighted;
  CudaResidentStoragePlan plan;
  Check(ComputeCudaResidentStoragePlan(options, &plan));
  HostStorageBound complete = plan.host;
  Require(complete.Add(plan.native_ac) && complete.Add(plan.butteraugli.host),
          "Test storage sum overflowed");
  for (size_t bytes :
       {plan.persistent_bytes, plan.staging_bytes, plan.sparse_header_bytes,
        plan.butteraugli.capacity_bytes})
    Require(complete.Add({bytes, bytes}), "Test device storage sum overflowed");
  frontend_storage_internal::OwnedFrameStoragePlan frame;
  Check(
      frontend_storage_internal::ComputeOwnedFrameStoragePlan(extent, &frame));
  // Native AC was already counted above. Only metadata and cached populations
  // are newly allocated at this ownership handoff.
  const size_t ac_bytes = frame.ac_coefficients * sizeof(int32_t);
  frame.output.retained_bytes -= ac_bytes;
  frame.output.peak_bytes -= ac_bytes;
  Require(
      complete.Add(frame.output) &&
          complete.AddVector<vardct_frame_internal::CoefficientOrderPopulation>(
              1, VectorCapacityPolicy::kFreshExact),
      "Test frame storage overflowed");
  ResourceBudget budget(complete.peak_bytes);
  const auto prepare = [&](std::unique_ptr<PreparedAqEvaluation> *prepared) {
    return PrepareAqEvaluation(
        gpu,
        {.original_linear_rgb = f.source.const_view(),
         .coding_opsin = f.opsin.const_view(),
         .strategies = &f.strategies,
         .epf_sharpness = {f.sharpness.data(), f.blocks, f.blocks.width},
         .options = options.evaluation,
         .resident_initial_cfl = frontend,
         .frame_only_resident_initial_quant = frontend,
         .resident_ac_strategy_inputs = frontend,
         .omit_initial_search_data = frontend,
         .resident_quantization = true,
         .coefficient_decision_mode =
             AcCoefficientDecisionMode::kAdjustedSharedQuant},
        prepared);
  };
  std::unique_ptr<vardct_frame_internal::CompletedVarDctFrame> completed;
  {
    ResourceReservation reservation;
    Check(budget.TryReserve(complete.peak_bytes, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kAqScratch});
    std::unique_ptr<PreparedAqEvaluation> prepared;
    Check(prepare(&prepared));
    const auto stats = prepared->memory_stats();
    Require(stats.persistent_bytes == plan.persistent_bytes &&
                stats.staging_bytes ==
                    plan.staging_bytes + plan.butteraugli.capacity_bytes,
            "Resident arena allocation differs from its shared recipe");
    Check(prepared->Reconfigure(
        f.strategies, {f.sharpness.data(), f.blocks, f.blocks.width}));
    if (frontend) {
      // Populate inverse Gaborish input before final coefficient generation.
      std::vector<float> initial(f.block_count);
      Check(prepared->ComputeInitialQuantization(
          {.butteraugli_target = 1.1f, .rescale = 1.0f, .uniform = true},
          {.quant_field = {initial.data(), f.blocks, f.blocks.width}}));
    }
    Check(prepared->PrepareInvariantColorCorrelationResident(
        {f.field.data(), f.blocks, f.blocks.width}, f.quant_dc));
    std::vector<double> scores;
    Check(prepared->EvaluateResidentButteraugliPolicy(
        {.adjusted_initial_quant_field = {f.field.data(), f.blocks,
                                          f.blocks.width},
         .quant_dc = f.quant_dc,
         .butteraugli_target = 1.1f,
         .lower_bound = 0.1f,
         .upper_bound = 10.0f,
         .evaluate_final_field = !evaluation_free},
        {.score_history = &scores, .completed_frame = &completed}));
    Require(completed != nullptr && completed->view().valid(),
            "Resident planned encode did not publish a valid completed frame");
  }
  Require(budget.snapshot()
                  .classes[static_cast<size_t>(ResourceClass::kCompletedFrame)]
                  .live_capacity_bytes != 0,
          "Completed CUDA frame lost its managed ownership category");
  completed.reset();
  Check(TrimCudaPreparationCaches(&budget));
  Empty(budget);

  if (extent != Extent2D{17, 33} || evaluation_free || frontend || dc_policy)
    return;
  size_t failures = 0;
  for (size_t checkpoint = 0; checkpoint < 128; ++checkpoint) {
    bool reached_success = false;
    {
      ResourceReservation reservation;
      Check(budget.TryReserve(complete.peak_bytes, &reservation));
      ResourceContextScope scope({&reservation, ResourceClass::kAqScratch});
      ArmManagedHostAllocationFailureAfterForTest(checkpoint);
      std::unique_ptr<PreparedAqEvaluation> prepared;
      const auto status = prepare(&prepared);
      reached_success = status.ok();
      DisarmManagedHostAllocationFailureForTest();
      if (!reached_success) {
        Require(
            status.code() == StatusCode::kOutOfMemory && prepared == nullptr,
            "Resident host backing failure escaped its atomic status contract");
        ++failures;
      }
    }
    Check(TrimCudaPreparationCaches(&budget));
    Empty(budget);
    if (reached_success)
      break;
  }
  Require(failures >= 20 && failures < 128,
          "Resident host allocation failure coverage is incomplete");
  std::cout << "Resident CUDA host allocation checkpoints: " << failures
            << " passed.\n";
}
} // namespace

int main() {
  try {
    std::unique_ptr<GpuBackend> gpu;
    const auto status = CreateCudaBackend(&gpu);
    if (status.code() == StatusCode::kUnavailable)
      return 77;
    Check(status);
    for (auto extent : {Extent2D{1, 1}, {17, 33}, {65, 67}, {257, 263}})
      for (bool evaluation_free : {false, true})
        for (bool frontend : {false, true})
          for (bool dc_policy : {false, true})
            Case(*gpu, extent, evaluation_free, frontend, dc_policy);
    std::cout
        << "Resident CUDA storage plans: 32 geometry/policy cases passed.\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
