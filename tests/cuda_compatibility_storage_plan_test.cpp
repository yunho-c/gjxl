// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/frontend_storage_plan.h"
#include "cuda_sparse_resident_fixture.h"
#include "gpu/cuda/cuda_resource_internal.h"
#include "gpu/cuda/cuda_storage_plan.h"
#include <iostream>

namespace {
using namespace resident_sparse_test;
using namespace gjxl::cuda_internal;
using namespace gjxl::resource_budget_internal;
void Case(GpuBackend &gpu, Extent2D extent, bool frame_only, bool dc_policy) {
  Fixture f(extent, 0.1f);
  AqEvaluationOptions options;
  options.profile.adaptive_dc_smoothing = dc_policy;
  options.profile.extra_dc_precision = dc_policy ? 1 : 0;
  options.dc_quantization = dc_policy ? DcQuantizationMode::kPredictionAware
                                      : DcQuantizationMode::kRound;
  options.dc_prediction = VarDctDcPrediction::kWeighted;
  CudaCompatibilityStoragePlan plan;
  Check(frame_only
            ? ComputeCudaFrameOnlyStoragePlan(extent, options, false, &plan)
            : ComputeCudaExactStoragePlan(extent, options, &plan));
  HostStorageBound complete = plan.host;
  Require(complete.Add(plan.butteraugli.host),
          "Compatibility host plan overflow");
  for (size_t bytes : {plan.persistent_bytes, plan.staging_bytes,
                       plan.butteraugli.capacity_bytes})
    Require(complete.Add({bytes, bytes}), "Compatibility device plan overflow");
  frontend_storage_internal::OwnedFrameStoragePlan frame_plan;
  Check(frontend_storage_internal::ComputeOwnedFrameStoragePlan(extent,
                                                                &frame_plan));
  Require(complete.Add(frame_plan.output), "Compatibility frame plan overflow");
  ResourceBudget budget(complete.peak_bytes);
  {
    ResourceReservation reservation;
    Check(budget.TryReserve(complete.peak_bytes, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kAqScratch});
    std::unique_ptr<PreparedAqEvaluation> prepared;
    Check(PrepareAqEvaluation(
        gpu,
        {.original_linear_rgb = f.source.const_view(),
         .coding_opsin = f.opsin.const_view(),
         .strategies = &f.strategies,
         .epf_sharpness = {f.sharpness.data(), f.blocks, f.blocks.width},
         .options = options,
         .frame_only = frame_only,
         .frame_only_inverse_gaborish =
             frame_only && options.profile.loop_filter.gaborish,
         .resident_initial_cfl = frame_only,
         .frame_only_resident_initial_quant = frame_only,
         .frame_only_resident_quantizer = frame_only,
         .coefficient_decision_mode =
             AcCoefficientDecisionMode::kAdjustedSharedQuant},
        &prepared));
    const auto stats = prepared->memory_stats();
    Require(stats.persistent_bytes == plan.persistent_bytes &&
                stats.staging_bytes ==
                    plan.staging_bytes + plan.butteraugli.capacity_bytes,
            "CUDA compatibility allocation differs from shared recipe");
    Check(prepared->Reconfigure(
        f.strategies, {f.sharpness.data(), f.blocks, f.blocks.width}));
    if (frame_only) {
      std::vector<float> quant(f.block_count), mask(f.block_count);
      std::vector<float> pixels(f.padded_extent.width * f.padded_extent.height);
      for (bool uniform : {true, false, true}) {
        QuantizerParams quantizer;
        const InitialQuantizationOptions initial{
            .butteraugli_target = 1.1f, .rescale = 0.875f, .uniform = uniform};
        Check(prepared->ComputeInitialQuantization(
            initial,
            {.quant_field = {quant.data(), f.blocks, f.blocks.width},
             .strategy_mask = {mask.data(), f.blocks, f.blocks.width},
             .pixel_mask = {pixels.data(), f.padded_extent,
                            f.padded_extent.width}},
            &quantizer, f.quant_dc, nullptr));
        if (uniform) {
          const float q = 0.79f / 1.1f * 0.875f;
          const float m = 1.0f / (q + 0.001f);
          Require(
              std::ranges::all_of(quant, [q](float v) { return v == q; }) &&
                  std::ranges::all_of(mask, [m](float v) { return v == m; }) &&
                  std::ranges::all_of(pixels, [m](float v) { return v == m; }),
              "Frame-only CUDA ignored uniform initial quantization");
        }
        VarDctEncoderFrame frame;
        Check(prepared->EncodeFrame({.quantizer = quantizer}, &frame));
        Require(frame.valid() && frame.profile().extra_dc_precision ==
                                     options.profile.extra_dc_precision,
                "Frame-only CUDA produced invalid DC metadata");
      }
    }
  }
  Check(TrimCudaPreparationCaches(&budget));
  Require(budget.snapshot().committed_bytes() == 0 &&
              budget.snapshot().open_reservations == 0,
          "CUDA compatibility storage leaked its domain");
}
} // namespace
int main() {
  try {
    std::unique_ptr<GpuBackend> gpu;
    const auto status = CreateCudaBackend(&gpu);
    if (status.code() == StatusCode::kUnavailable)
      return 77;
    Check(status);
    AqEvaluationOptions invalid_dc;
    invalid_dc.dc_quantization = DcQuantizationMode::kPredictionAware;
    invalid_dc.profile.extra_dc_precision = 0;
    CudaCompatibilityStoragePlan untouched;
    untouched.persistent_bytes = 123;
    Require(ComputeCudaFrameOnlyStoragePlan({17, 33}, invalid_dc, false,
                                           &untouched).code() ==
                    StatusCode::kInvalidArgument &&
                untouched.persistent_bytes == 123,
            "Frame-only plan accepted prediction-aware DC without precision");
    for (auto extent : {Extent2D{1, 1}, {17, 33}, {65, 67}, {257, 263}})
      for (bool frame_only : {false, true})
        for (bool dc_policy : {false, true})
          Case(*gpu, extent, frame_only, dc_policy);
    std::cout
        << "CUDA exact/frame-only storage: 16 geometry/policy cases passed.\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
