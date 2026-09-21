// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <iostream>

#include "codec/vardct_frame_view_internal.h"
#include "codestream/encoder_internal.h"
#include "cuda_sparse_resident_fixture.h"
#include "gpu/ops/aq_evaluation_internal.h"

namespace {
using namespace resident_sparse_test;

std::unique_ptr<PreparedAqEvaluation> Prepare(GpuBackend &gpu, const Fixture &f,
                                              bool evaluation_free, bool omit,
                                              bool gaborish, unsigned dc_policy,
                                              bool defer_metadata = false) {
  AqEvaluationOptions options;
  options.evaluation_free = evaluation_free;
  options.profile.loop_filter.gaborish = gaborish;
  options.profile.adaptive_dc_smoothing = dc_policy != 0;
  options.profile.extra_dc_precision = dc_policy >= 2 ? 1 : 0;
  options.dc_quantization = dc_policy >= 2
                                ? DcQuantizationMode::kPredictionAware
                                : DcQuantizationMode::kRound;
  options.dc_prediction = dc_policy == 3 ? VarDctDcPrediction::kWeighted
                                         : VarDctDcPrediction::kGradient;
  std::unique_ptr<PreparedAqEvaluation> result;
  Check(PrepareAqEvaluation(
      gpu,
      {.original_linear_rgb = f.source.const_view(),
       .coding_opsin = f.opsin.const_view(),
       .strategies = &f.strategies,
       .epf_sharpness = {f.sharpness.data(), f.blocks, f.blocks.width},
       .options = options,
       .resident_initial_cfl = true,
       .frame_only_resident_initial_quant = true,
       .resident_ac_strategy_inputs = true,
       .omit_initial_search_data = omit,
       .resident_quantization = true,
       .coefficient_decision_mode =
           AcCoefficientDecisionMode::kAdjustedSharedQuant,
       .defer_final_transform_metadata = defer_metadata},
      &result));
  return result;
}

void DeferredMetadata(GpuBackend &gpu) {
  for (bool mixed : {false, true}) {
    const auto seed = gjxl_test::MakeFrame(7, 0, 8);
    Fixture f(seed.geometry().frame());
    if (mixed)
      f.strategies = seed.strategies();
    for (bool evaluation_free : {false, true}) {
      auto deferred = Prepare(gpu, f, evaluation_free, false, true, 3, true);
      auto control = Prepare(gpu, f, evaluation_free, false, true, 3);
      const auto before = gpu.stats().committed_submissions;
      auto output = f.NewOutput();
      Require(
          deferred->PrepareInvariantColorCorrelationResident(
                      {f.field.data(), f.blocks, f.blocks.width}, f.quant_dc)
                  .code() == StatusCode::kFailedPrecondition,
          "Deferred CUDA preparation accepted final CfL before "
          "reconfiguration");
      Require(f.Run(*deferred, output, 2).code() ==
                      StatusCode::kFailedPrecondition &&
                  output.score == 77.5 &&
                  output.scores == std::vector<double>{99.25} &&
                  gpu.stats().committed_submissions == before,
              "Deferred CUDA evaluation submitted work or changed outputs");
      std::vector<uint8_t> expected;
      for (auto *prepared : {control.get(), deferred.get()}) {
        auto *encoding = dynamic_cast<
            aq_evaluation_internal::PreparedAqEncodingInitialQuantization *>(
            prepared);
        Require(encoding != nullptr, "Missing deferred resident frontend");
        Check(encoding->ComputeInitialQuantizationForEncoding(
            {.butteraugli_target = 1.1f, .uniform = true}));
        ResidentAcStrategyInputs search;
        Check(prepared->GetResidentAcStrategyInputs(&search));
        aq_evaluation_internal::ResidentEncodingPolicySetup setup;
        if (prepared == deferred.get()) {
          Require(
              encoding->PrepareResidentEncodingPolicy(1.1f, &setup).code() ==
                  StatusCode::kFailedPrecondition,
              "Deferred policy accepted provisional transform metadata");
          auto invalid = f.sharpness;
          invalid[0] = 8;
          Require(prepared->Reconfigure(f.strategies, {invalid.data(), f.blocks,
                                                       f.blocks.width})
                              .code() == StatusCode::kInvalidArgument &&
                      encoding->PrepareResidentEncodingPolicy(1.1f, &setup)
                              .code() == StatusCode::kFailedPrecondition,
                  "Failed reconfiguration released deferred metadata");
        }
        Check(prepared->Reconfigure(
            f.strategies, {f.sharpness.data(), f.blocks, f.blocks.width}));
        Check(encoding->PrepareResidentEncodingPolicy(1.1f, &setup));
        auto result = f.NewOutput();
        Check(prepared->EvaluateResidentButteraugliPolicy(
            {.quant_dc = setup.quant_dc,
             .butteraugli_target = 1.1f,
             .lower_bound = setup.lower_bound,
             .upper_bound = setup.upper_bound,
             .iterations = 0,
             .evaluate_final_field = !evaluation_free},
            {.score_history = &result.scores, .frame = &result.frame}));
        if (prepared == control.get())
          expected = result.Bytes();
        else
          Require(result.Bytes() == expected,
                  "Deferred metadata changed encoded coefficients");
      }
    }
  }
  std::unique_ptr<PreparedAqEvaluation> invalid;
  Require(PrepareAqEvaluation(gpu, {.defer_final_transform_metadata = true},
                              &invalid)
                      .code() == StatusCode::kInvalidArgument &&
              invalid == nullptr,
          "Invalid deferred preparation flags were ignored");
  std::cout << "CUDA explicit deferred metadata: four frontend/reconfiguration "
               "cases passed.\n";
}

void Case(GpuBackend &gpu, Extent2D extent, bool gaborish, unsigned dc_policy) {
  Fixture f(extent);
  auto complete = Prepare(gpu, f, false, false, gaborish, dc_policy);
  auto no_evaluation = Prepare(gpu, f, true, false, gaborish, dc_policy);
  auto minimal = Prepare(gpu, f, true, true, gaborish, dc_policy);
  const auto full_stats = complete->memory_stats();
  const auto free_stats = no_evaluation->memory_stats();
  const auto minimal_stats = minimal->memory_stats();
  Require(free_stats.persistent_bytes < full_stats.persistent_bytes &&
              free_stats.staging_bytes < full_stats.staging_bytes,
          "Evaluation-free preparation retained reconstruction/metric storage");
  Require(minimal_stats.staging_bytes < free_stats.staging_bytes,
          "Omitted search masks retained their device storage");

  // Alternate the policies on the same preparations to catch stale masks,
  // forward coefficients and final CfL after repeated zero-update encoding.
  for (bool uniform : {true, false, true}) {
    const InitialQuantizationOptions initial{
        .butteraugli_target = 1.1f, .rescale = 0.875f, .uniform = uniform};
    std::vector<float> quant(f.block_count), strategy(f.block_count);
    std::vector<float> pixel(f.padded_extent.width * f.padded_extent.height);
    Check(complete->ComputeInitialQuantization(
        initial, {.quant_field = {quant.data(), f.blocks, f.blocks.width},
                  .strategy_mask = {strategy.data(), f.blocks, f.blocks.width},
                  .pixel_mask = {pixel.data(), f.padded_extent,
                                 f.padded_extent.width}}));
    if (uniform) {
      const float q = 0.79f / initial.butteraugli_target * initial.rescale;
      const float m = 1.0f / (q + 0.001f);
      Require(
          std::ranges::all_of(quant, [q](float v) { return v == q; }) &&
              std::ranges::all_of(strategy, [m](float v) { return v == m; }) &&
              std::ranges::all_of(pixel, [m](float v) { return v == m; }),
          "CUDA uniform field/masks differ from CPU policy");
    }
    std::vector<uint8_t> expected;
    for (PreparedAqEvaluation *p :
         {complete.get(), no_evaluation.get(), minimal.get()}) {
      auto *encoding = dynamic_cast<
          aq_evaluation_internal::PreparedAqEncodingInitialQuantization *>(p);
      Require(encoding != nullptr,
              "Missing resident initial policy capability");
      if (p == minimal.get()) {
        std::vector<float> actual(f.block_count);
        Check(p->ComputeInitialQuantization(
            initial,
            {.quant_field = {actual.data(), f.blocks, f.blocks.width}}));
        Require(EqualBits<float>(actual, quant),
                "Omitted masks changed initial quantization");
      }
      Check(encoding->ComputeInitialQuantizationForEncoding(initial));
      ResidentAcStrategyInputs views;
      Check(p->GetResidentAcStrategyInputs(&views));
      Require((views.pixel_mask.buffer == nullptr) == (p == minimal.get()),
              "Resident mask omission contract differs");
      aq_evaluation_internal::ResidentEncodingPolicySetup setup;
      Check(encoding->PrepareResidentEncodingPolicy(1.1f, &setup));
      AqResidentButteraugliPolicyInput input{.quant_dc = setup.quant_dc,
                                             .butteraugli_target = 1.1f,
                                             .lower_bound = setup.lower_bound,
                                             .upper_bound = setup.upper_bound,
                                             .iterations = 0,
                                             .evaluate_final_field =
                                                 p == complete.get()};
      std::vector<double> scores{123.0};
      std::unique_ptr<vardct_frame_internal::CompletedVarDctFrame> frame;
      Check(p->EvaluateResidentButteraugliPolicy(
          input, {.score_history = &scores, .completed_frame = &frame}));
      Require(frame != nullptr && frame->view().valid() &&
                  scores.size() ==
                      static_cast<size_t>(input.evaluate_final_field),
              "Zero-update CUDA publication is invalid");
      std::vector<uint8_t> bytes;
      Check(codestream_internal::EncodeVarDctCodestreamFromView(frame->view(),
                                                                {}, &bytes));
      if (p == complete.get())
        expected = std::move(bytes);
      else {
        Require(bytes == expected,
                "Evaluation-free coefficients differ from scored field");
        auto *prior_frame = frame.get();
        input.evaluate_final_field = true;
        Require(
            p->EvaluateResidentButteraugliPolicy(
                 input, {.score_history = &scores, .completed_frame = &frame})
                        .code() == StatusCode::kFailedPrecondition &&
                frame.get() == prior_frame && scores.empty(),
            "Evaluation-free policy did not reject scoring atomically");
        Require(p->Evaluate({}, {}).code() == StatusCode::kFailedPrecondition,
                "Evaluation-free preparation accepted direct evaluation");
      }
    }
  }
}
} // namespace

int main() {
  try {
    std::unique_ptr<GpuBackend> gpu;
    const Status status = CreateCudaBackend(&gpu);
    if (status.code() == StatusCode::kUnavailable) {
      std::cout << status.message() << '\n';
      return 77;
    }
    Check(status);
    DeferredMetadata(*gpu);
    for (auto extent : {Extent2D{1, 1}, {17, 33}, {65, 67}, {257, 263}})
      for (bool gaborish : {false, true})
        for (unsigned dc_policy = 0; dc_policy < 4; ++dc_policy)
          Case(*gpu, extent, gaborish, dc_policy);
    std::cout << "CUDA low-effort uniform/adaptive, DC and evaluation-free "
                 "parity: 96 cases passed.\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
