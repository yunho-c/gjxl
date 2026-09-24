// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "codec/color_transform.h"
#include "codec/adaptive_quantization_internal.h"
#include "codec/epf_search.h"
#include "codec/gaborish.h"
#include "codec/loop_filter.h"
#include "codec/reconstruction.h"
#include "codec/vardct_frame_view_internal.h"
#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/gpu_execution_profile_internal.h"

namespace {
using namespace gjxl;
namespace mi = gjxl::metal_internal;

bool Ok(Status status) {
  if (status.ok()) return true;
  std::cerr << status.message() << '\n';
  return false;
}

bool Check(bool value, const char* message) {
  if (!value) std::cerr << message << '\n';
  return value;
}

template <typename A, typename B>
bool SameCoefficients(const A& a, const B& b) {
  if (!a.valid() || !b.valid() || a.ac_group_count() != b.ac_group_count())
    return false;
  const auto blocks = a.raw_quant_field().extent;
  for (size_t y = 0; y < blocks.height; ++y) {
    if (!std::equal(a.raw_quant_field().Row(y),
                    a.raw_quant_field().Row(y) + blocks.width,
                    b.raw_quant_field().Row(y))) return false;
    for (size_t c = 0; c < 3; ++c) {
      if (!std::equal(a.quantized_dc().plane[c].Row(y),
                      a.quantized_dc().plane[c].Row(y) + blocks.width,
                      b.quantized_dc().plane[c].Row(y))) return false;
    }
  }
  for (size_t i = 0; i < a.ac_group_count(); ++i) {
    VarDctAcGroupView ga, gb;
    if (!Ok(a.GetAcGroup(i, &ga)) || !Ok(b.GetAcGroup(i, &gb)) ||
        ga.used_coefficient_count != gb.used_coefficient_count) return false;
    for (size_t c = 0; c < 3; ++c) {
      if (ga.coefficients[c].size() != gb.coefficients[c].size() ||
          !std::equal(ga.coefficients[c].begin(), ga.coefficients[c].end(),
                       gb.coefficients[c].begin())) return false;
    }
  }
  return true;
}

bool SameSharpness(ConstPlaneU8View a, ConstPlaneU8View b) {
  if (a.extent != b.extent) return false;
  for (size_t y = 0; y < a.extent.height; ++y)
    if (!std::equal(a.Row(y), a.Row(y) + a.extent.width, b.Row(y))) return false;
  return true;
}

struct Fixture {
  static constexpr Extent2D source{41, 29}, padded{48, 32}, blocks{6, 4};
  Image3FBuffer original{padded}, coding{padded}, linear{source};
  std::vector<float> mask = std::vector<float>(48 * 32);
  std::array<int32_t, 24> raw{};
  std::array<uint8_t, 24> neutral{};
  std::array<float, 24> sigma{};
  AcStrategyGrid strategies;
  Quantizer quantizer;
  ColorCorrelationMap cfl;
  FrameGeometry geometry;
  VarDctEncoderFrame exact;
  AqEvaluationOptions options;

  bool Initialize(bool gaborish, uint32_t passes) {
    for (size_t y = 0; y < padded.height; ++y) {
      for (size_t x = 0; x < padded.width; ++x) {
        const float wave = std::sin(float(3 * x + 7 * y) * 0.13f);
        original.view().plane[0].Row(y)[x] = 0.012f * wave;
        original.view().plane[1].Row(y)[x] =
            0.4f + 0.08f * wave + float(x >= 24) * 0.05f;
        original.view().plane[2].Row(y)[x] = 0.45f + 0.03f * wave;
        mask[y * 48 + x] = 0.25f + float((x + 3 * y) % 13) * 0.13f;
      }
    }
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = 3 + int32_t(i * 7 % 29);
    neutral.fill(4);
    options.profile.loop_filter.gaborish = gaborish;
    options.profile.loop_filter.epf_options.iterations = passes;
    options.search_epf_sharpness = true;
    if (!Ok(AcStrategyGrid::Create(blocks, &strategies)) ||
        !Ok(strategies.Set(0, 0, AcStrategyType::kDct16x16)) ||
        !Ok(strategies.Set(2, 0, AcStrategyType::kDct8x16)) ||
        !Ok(strategies.Set(3, 2, AcStrategyType::kDct16x8))) return false;
    strategies.fill_empty_dct8();
    if (gaborish) {
      if (!Ok(ApplyGaborishInverse(original.const_view(), {1.0f, 1.0f, 1.0f},
                                   coding.view()))) return false;
    } else {
      for (size_t c = 0; c < 3; ++c)
        std::copy(original.plane(c).begin(), original.plane(c).end(),
                  coding.plane(c).begin());
    }
    return Ok(OpsinToLinearRgb(original.cropped_view(source),
                               options.profile.intensity_target, linear.view())) &&
        Ok(Quantizer::Create({3541, 10}, &quantizer)) &&
        Ok(FrameGeometry::Create(source, &geometry)) &&
        Ok(ComputeInitialColorCorrelationMap(coding.const_view(), &cfl)) &&
        Ok(ComputeEpfInverseSigma(strategies, {raw.data(), blocks, 6}, quantizer,
              {neutral.data(), blocks, 6}, options.profile.epf_sigma,
              {sigma.data(), blocks, 6})) &&
        Ok(ComputeQuantizedCoefficients(coding.const_view(),
              {.geometry = geometry, .strategies = &strategies,
               .raw_quant_field = {raw.data(), blocks, 6}, .quantizer = &quantizer,
               .color_correlation = &cfl,
               .epf_sharpness = {neutral.data(), blocks, 6}},
              options.profile, &exact));
  }
};

bool CheckCandidates(PreparedAqEvaluation& prepared, const Fixture& f,
                     const VarDctEncoderFrame& frame, float target) {
  mi::MetalEpfSearchSnapshotForTesting gpu;
  EpfSharpnessSearchConfig config;
  if (!Ok(mi::GetMetalEpfSearchSnapshotForTesting(prepared, &gpu)) ||
      !Ok(MakeEpfSharpnessSearchConfig(target, &config)) ||
      !Check(gpu.blocks == f.blocks && gpu.candidate_count == config.count,
             "Candidate snapshot geometry differs")) return false;
  Image3FBuffer reconstructed(f.padded), filtered(f.source);
  if (!Ok(ReconstructQuantizedCoefficients(frame, reconstructed.view()))) return false;
  std::array<std::array<float, 24>, 3> cpu_errors{};
  std::array<ConstPlaneF32View, 3> gpu_views;
  std::array<float, 24> sigma;
  std::array<uint8_t, 24> uniform, from_errors;
  for (size_t i = 0; i < config.count; ++i) {
    uniform.fill(config.candidates[i]);
    if (!Ok(ComputeEpfInverseSigma(f.strategies, frame.raw_quant_field(),
              frame.quantizer(), {uniform.data(), f.blocks, 6},
              frame.profile().epf_sigma, {sigma.data(), f.blocks, 6})) ||
        !Ok(ApplyLoopFilters(reconstructed.cropped_view(f.source),
              {sigma.data(), f.blocks, 6}, frame.profile().loop_filter,
              filtered.view())) ||
        !Ok(ComputeEpfBlockErrors(f.original.cropped_view(f.source),
              filtered.const_view(), {f.mask.data(), f.source, 48},
              {cpu_errors[i].data(), f.blocks, 6}))) return false;
    gpu_views[i] = {gpu.errors[i].data(), f.blocks, 6};
    for (size_t b = 0; b < 24; ++b) {
      const float tolerance = 1.0e-8f + 3.0e-4f * cpu_errors[i][b];
      if (std::abs(gpu.errors[i][b] - cpu_errors[i][b]) > tolerance) {
        std::cerr << "Candidate " << i << " block " << b << " error "
                  << gpu.errors[i][b] << " versus " << cpu_errors[i][b] << '\n';
        return false;
      }
    }
  }
  if (!Ok(SelectEpfSharpnessFromErrors(gpu_views, target,
                                        {from_errors.data(), f.blocks, 6}))) return false;
  for (size_t b = 0; b < 24; ++b) {
    const auto actual = frame.epf_sharpness().Row(b / 6)[b % 6];
    if (!Check(actual == from_errors[b] && actual == gpu.sharpness[b],
               "Serialized sharpness differs from device argmin")) return false;
    // A different CPU decision is allowed only within the measured error
    // bound; decisions with a clear margin must agree.
    float best = std::numeric_limits<float>::infinity(), selected = best;
    for (size_t i = 0; i < config.count; ++i) {
      const float error = cpu_errors[i][b] * (i == 0 ? config.no_smoothing_bias : 1);
      best = std::min(best, error);
      if (config.candidates[i] == actual) selected = error;
    }
    if (!Check(selected <= best + 2.0e-8f + 6.0e-4f * selected,
               "CPU and Metal choices differ outside the error bound")) return false;
  }
  return true;
}

bool CheckFinalImage(const Fixture& f, const VarDctEncoderFrame& frame,
                     ConstImage3FView actual) {
  Image3FBuffer reconstructed(f.padded), filtered(f.source), expected(f.source);
  std::array<float, 24> sigma;
  if (!Ok(ReconstructQuantizedCoefficients(frame, reconstructed.view())) ||
      !Ok(ComputeEpfInverseSigma(f.strategies, frame.raw_quant_field(),
            frame.quantizer(), frame.epf_sharpness(), frame.profile().epf_sigma,
            {sigma.data(), f.blocks, 6})) ||
      !Ok(ApplyLoopFilters(reconstructed.cropped_view(f.source),
            {sigma.data(), f.blocks, 6}, frame.profile().loop_filter, filtered.view())) ||
      !Ok(OpsinToLinearRgb(filtered.const_view(), frame.profile().intensity_target,
                           expected.view()))) return false;
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < f.source.height; ++y)
      for (size_t x = 0; x < f.source.width; ++x)
        if (!Check(std::abs(actual.plane[c].Row(y)[x] -
                             expected.const_view().plane[c].Row(y)[x]) < 1.0e-4f,
                   "Final RGB does not describe the selected map")) return false;
  return true;
}

bool CheckEvaluation(GpuBackend& backend, bool gaborish, uint32_t passes,
                     bool exact) {
  Fixture f;
  if (!f.Initialize(gaborish, passes)) return false;
  std::unique_ptr<PreparedAqEvaluation> prepared;
  if (!Ok(PrepareAqEvaluation(backend,
          {.original_linear_rgb = f.linear.const_view(),
           .coding_opsin = f.coding.const_view(), .strategies = &f.strategies,
           .epf_sharpness = {f.neutral.data(), f.blocks, 6}, .options = f.options,
           .epf_search_reference = {f.original.const_view(),
                                    {f.mask.data(), f.padded, 48}}}, &prepared))) return false;
  std::array<uint8_t, 24> invalid_map;
  invalid_map.fill(2);
  if (!Check(prepared->Reconfigure(f.strategies,
                    {invalid_map.data(), f.blocks, 6}).code() == StatusCode::kInvalidArgument,
             "Reconfiguration accepted a non-neutral AQ map")) return false;
  AqEvaluationInput input{
      .raw_quant_field = exact ? f.exact.raw_quant_field()
                              : ConstPlaneI32View{f.raw.data(), f.blocks, 6},
      .quantizer = f.quantizer.params(), .y_to_x = f.cfl.y_to_x_map(),
      .y_to_b = f.cfl.y_to_b_map(), .epf_inverse_sigma = {f.sigma.data(), f.blocks, 6},
      .exact_coefficients = exact ? &f.exact : nullptr};
  if (!Ok(ComputeEpfInverseSigma(f.strategies, input.raw_quant_field, f.quantizer,
            {f.neutral.data(), f.blocks, 6}, f.options.profile.epf_sigma,
            {f.sigma.data(), f.blocks, 6}))) return false;
  VarDctEncoderFrame baseline;
  std::vector<uint8_t> saved_map;
  double saved_score = 0;
  for (float target : {0.0f, 1.0f, 0.49f, 8.0f, 0.5f, 4.5f, 4.5001f, 1.0f}) {
    input.epf_sharpness_search_target = target;
    Image3FBuffer rgb(f.source);
    VarDctEncoderFrame frame;
    std::array<float, 24> distance;
    double score = -1;
    AqEvaluationOutput::Final final{rgb.view(), &frame};
    const auto before = backend.stats();
    if (!Ok(prepared->Evaluate(input, {{distance.data(), f.blocks, 6}, &score,
                                      nullptr, nullptr, &final})) ||
        !Check(backend.stats().committed_submissions == before.committed_submissions + 1,
               "Search added a submission") ||
        !Check(std::isfinite(score) && score >= 0, "Final score is invalid")) return false;
    const bool active = target >= 0.5f && passes > 0;
    if (active) {
      if (!CheckCandidates(*prepared, f, frame, target)) return false;
    } else {
      for (size_t y = 0; y < f.blocks.height; ++y)
        for (size_t x = 0; x < f.blocks.width; ++x)
          if (!Check(frame.epf_sharpness().Row(y)[x] == 4,
                     "Disabled search retained a previous map")) return false;
    }
    if (!CheckFinalImage(f, frame, rgb.const_view())) return false;
    mi::MetalAqReadbackStatsForTesting stats;
    if (!Ok(mi::GetMetalAqReadbackStatsForTesting(*prepared, &stats)) ||
        !Check(stats.epf_sharpness_bytes == (active ? 24 : 0),
               "EPF map readback accounting differs")) return false;
    if (target == 1.0f) {
      std::vector<uint8_t> map;
      for (size_t y = 0; y < f.blocks.height; ++y)
        map.insert(map.end(), frame.epf_sharpness().Row(y),
                    frame.epf_sharpness().Row(y) + f.blocks.width);
      if (!saved_map.empty() && !Check(map == saved_map && score == saved_score,
                                      "Reused search differs from its first result")) return false;
      saved_map = std::move(map);
      saved_score = score;
    }
    if (target == 0.0f) baseline = std::move(frame);
    else if (!Check(SameCoefficients(baseline, frame), "Search changed coefficients")) return false;
  }
  return true;
}

bool CheckResidentPolicy(GpuBackend& backend) {
  Fixture f;
  if (!f.Initialize(true, 3)) return false;
  std::array<float, 24> initial;
  initial.fill(0.75f);
  if (!Ok(AdjustQuantField(f.strategies, 1.0f, {initial.data(), f.blocks, 6},
                           {initial.data(), f.blocks, 6}))) return false;
  adaptive_quantization_internal::ButteraugliPolicySetup setup;
  if (!Ok(adaptive_quantization_internal::PrepareButteraugliPolicy(
            {initial.data(), f.blocks, 6}, 1.0f, &setup))) return false;
  std::array<VarDctEncoderFrame, 3> neutral_frames;
  std::array<std::vector<double>, 3> neutral_scores;
  for (bool search : {false, true}) {
    f.options.search_epf_sharpness = search;
    std::unique_ptr<PreparedAqEvaluation> prepared;
    if (!Ok(PrepareAqEvaluation(backend,
            {.original_linear_rgb = f.linear.const_view(),
             .coding_opsin = f.coding.const_view(), .strategies = &f.strategies,
             .epf_sharpness = {f.neutral.data(), f.blocks, 6}, .options = f.options,
             .resident_quantization = true,
             .coefficient_decision_mode = AcCoefficientDecisionMode::kAdjustedSharedQuant,
             .epf_search_reference = {f.original.const_view(),
                                      {f.mask.data(), f.padded, 48}}}, &prepared)) ||
        !Ok(prepared->SetInvariantColorCorrelation(f.cfl.y_to_x_map(),
                                                   f.cfl.y_to_b_map()))) return false;
    VarDctEncoderFrame cold_unscored;
    std::vector<double> cold_scores;
    if (!Ok(prepared->EvaluateResidentButteraugliPolicy(
            {.adjusted_initial_quant_field = {initial.data(), f.blocks, 6},
             .quant_dc = setup.quant_dc, .butteraugli_target = 1.0f,
             .lower_bound = setup.lower_bound, .upper_bound = setup.upper_bound,
             .iterations = 0, .evaluate_final_field = false},
            {.score_history = &cold_scores, .frame = &cold_unscored}))) return false;
    size_t case_index = 0;
    for (size_t iterations : {0u, 1u, 3u}) {
      AqResidentButteraugliPolicyInput input{
          .adjusted_initial_quant_field = {initial.data(), f.blocks, 6},
          .quant_dc = setup.quant_dc, .butteraugli_target = 1.0f,
          .lower_bound = setup.lower_bound, .upper_bound = setup.upper_bound,
          .iterations = iterations};
      VarDctEncoderFrame scored, unscored;
      Image3FBuffer rgb(f.source);
      std::vector<double> scores, partial_scores;
      const auto before = backend.stats();
      if (!Ok(prepared->EvaluateResidentButteraugliPolicy(input,
                {.score_history = &scores, .reconstructed_linear_rgb = rgb.view(),
                 .frame = &scored})) ||
          !Check(backend.stats().committed_submissions == before.committed_submissions + 1,
                 "Resident search added a submission") ||
          !Check(scores.size() == iterations + 1, "Resident score history differs") ||
          !CheckFinalImage(f, scored, rgb.const_view())) return false;
      if (search && !CheckCandidates(*prepared, f, scored, 1.0f)) return false;
      if (iterations == 0 &&
          !Check(SameCoefficients(cold_unscored, scored) &&
                     SameSharpness(cold_unscored.epf_sharpness(), scored.epf_sharpness()),
                 "Cold zero-update search differs from scored output")) return false;
      input.evaluate_final_field = false;
      if (!Ok(prepared->EvaluateResidentButteraugliPolicy(input,
                {.score_history = &partial_scores, .frame = &unscored})) ||
          !Check(SameCoefficients(scored, unscored) &&
                     SameSharpness(scored.epf_sharpness(), unscored.epf_sharpness()),
                 "Unscored final frame differs from scored final frame") ||
          !Check(partial_scores.size() == iterations &&
                     std::equal(partial_scores.begin(), partial_scores.end(), scores.begin()),
                 "Final search altered an AQ update")) return false;
      std::unique_ptr<vardct_frame_internal::CompletedVarDctFrame> completed;
      std::vector<double> completed_scores;
      if (!Ok(prepared->EvaluateResidentButteraugliPolicy(input,
                {.score_history = &completed_scores, .completed_frame = &completed})) ||
          !Check(completed && completed_scores == partial_scores &&
                     SameCoefficients(scored, completed->view()) &&
                     SameSharpness(scored.epf_sharpness(), completed->view().epf_sharpness()),
                 "Completed frame lost the selected sharpness map")) return false;
      if (search) {
        if (!Check(SameCoefficients(neutral_frames[case_index], scored) &&
                       std::equal(partial_scores.begin(), partial_scores.end(),
                                  neutral_scores[case_index].begin()),
                   "Search changed resident coefficient or AQ decisions")) return false;
      } else {
        neutral_frames[case_index] = std::move(scored);
        neutral_scores[case_index] = std::move(scores);
      }
      if (search && iterations <= 1) {
        auto* profiler = dynamic_cast<gpu_profile_internal::PreparedAqEvaluationProfiler*>(
            prepared.get());
        if (!Check(profiler != nullptr, "Resident profiler is missing")) return false;
        for (bool evaluate_final : {false, true}) {
          input.evaluate_final_field = evaluate_final;
          VarDctEncoderFrame profiled;
          std::vector<double> profiled_scores;
          gpu_profile_internal::GpuExecutionProfile profile;
          if (!Ok(profiler->EvaluateResidentButteraugliPolicyProfiled(input,
                    {.score_history = &profiled_scores, .frame = &profiled},
                    gpu_profile_internal::GpuProfilingMode::kStage, &profile)) ||
              !Check(SameCoefficients(unscored, profiled) &&
                         SameSharpness(unscored.epf_sharpness(), profiled.epf_sharpness()) &&
                         profiled_scores == (evaluate_final ? scores : partial_scores),
                     "Profiled EPF search differs")) return false;
        }
      }
      ++case_index;
    }
    if (search) {
      std::vector<double> failed_scores{-37.0};
      VarDctEncoderFrame failed_frame;
      if (!Ok(mi::FailNextMetalAqReadbackForTesting(*prepared))) return false;
      const Status status = prepared->EvaluateResidentButteraugliPolicy(
          {.adjusted_initial_quant_field = {initial.data(), f.blocks, 6},
           .quant_dc = setup.quant_dc, .butteraugli_target = 1.0f,
           .lower_bound = setup.lower_bound, .upper_bound = setup.upper_bound,
           .iterations = 1, .evaluate_final_field = false},
          {.score_history = &failed_scores, .frame = &failed_frame});
      if (!Check(status.code() == StatusCode::kDeviceError && !failed_frame.valid() &&
                     failed_scores == std::vector<double>{-37.0},
                 "Failed EPF readback published a partial result")) return false;
    }
  }
  return true;
}

bool CheckWorkflow(GpuBackend& backend) {
  Fixture f;
  if (!f.Initialize(true, 2)) return false;
  struct Setting { int effort; float target; };
  for (int path = 0; path < 3; ++path) {
    for (const Setting setting : {Setting{5, 1.0f}, {6, 0.49f}, {6, 1.0f}, {8, 8.0f}}) {
      VarDctEncodingOptions options;
      options.effort = setting.effort;
      options.butteraugli_target = setting.target;
      options.backend = path == 0 ? VarDctBackendPreference::kCpu
                                   : VarDctBackendPreference::kMetal;
      options.cpu_thread_count = 1;
      options.gpu_aq_mode = path == 1 ? GpuAdaptiveQuantizationMode::kExactCoefficients
                                       : GpuAdaptiveQuantizationMode::kFullyResident;
      options.collect_final_butteraugli_score = true;
      std::array<std::vector<uint8_t>, 2> bytes;
      std::array<VarDctEncodingSummary, 2> summaries;
      for (size_t search = 0; search < 2; ++search) {
        options.adaptive_epf_sharpness = search != 0;
        if (!Ok(codestream_internal::EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                  f.linear.const_view(), options, path == 0 ? nullptr : &backend,
                  true, &bytes[search], &summaries[search]))) return false;
      }
      const bool active = setting.effort >= 6 && setting.target >= 0.5f;
      if (!active && !Check(bytes[0] == bytes[1], "Disabled workflow gate changed bytes"))
        return false;
      if (active && setting.target <= 4.5f &&
          !Check(bytes[0] != bytes[1], "Enabled workflow did not serialize its selected map"))
        return false;
      const auto& before = summaries[0].score_history;
      const auto& after = summaries[1].score_history;
      if (!Check(!before.empty() && before.size() == after.size() &&
                     std::equal(before.begin(), before.end() - 1, after.begin()) &&
                     summaries[0].strategy_counts == summaries[1].strategy_counts,
                 "Workflow search changed pre-final decisions")) return false;
      if (path == 2) {
        options.collect_final_butteraugli_score = false;
        std::vector<uint8_t> unscored;
        if (!Ok(codestream_internal::EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                  f.linear.const_view(), options, &backend, true, &unscored)) ||
            !Check(unscored == bytes[1], "Final-score collection changed the codestream"))
          return false;
      }
    }
  }
  return true;
}
}  // namespace

int main() {
  std::unique_ptr<GpuBackend> backend;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &backend))) return EXIT_FAILURE;
  if (!CheckResidentPolicy(*backend)) return EXIT_FAILURE;
  if (!CheckWorkflow(*backend)) return EXIT_FAILURE;
  for (bool gaborish : {false, true})
    for (uint32_t passes : {0u, 1u, 2u, 3u})
      for (bool exact : {false, true})
        if (!CheckEvaluation(*backend, gaborish, passes, exact)) {
          std::cerr << "gaborish=" << gaborish << " passes=" << passes
                    << " exact=" << exact << '\n';
          return EXIT_FAILURE;
        }
  std::cout << "Metal EPF sharpness search checks passed\n";
  return EXIT_SUCCESS;
}
