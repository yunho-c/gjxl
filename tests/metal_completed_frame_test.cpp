// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "codec/adaptive_quantization_internal.h"
#include "codec/color_transform.h"
#include "codec/vardct_frame_view_internal.h"
#include "codestream/encoder.h"
#include "codestream/encoder_internal.h"
#include "core/image_buffer.h"
#include "core/resource_context.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_butteraugli_test.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/gpu_execution_profile_internal.h"

namespace {
using namespace gjxl;
using vardct_frame_internal::CompletedVarDctFrame;
using vardct_frame_internal::VarDctFrameView;

bool Check(Status status) {
  if (status.ok())
    return true;
  std::cerr << status.message() << '\n';
  return false;
}

bool Equal(const VarDctFrameView &a, const VarDctFrameView &b) {
  if (!a.valid() || !b.valid() || a.ac_group_count() != b.ac_group_count()) {
    return false;
  }
  const auto blocks = a.strategies().extent();
  if (blocks != b.strategies().extent())
    return false;
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      if (a.raw_quant_field().Row(y)[x] != b.raw_quant_field().Row(y)[x] ||
          a.epf_sharpness().Row(y)[x] != b.epf_sharpness().Row(y)[x])
        return false;
      for (size_t c = 0; c < 3; ++c) {
        if (a.quantized_dc().plane[c].Row(y)[x] !=
                b.quantized_dc().plane[c].Row(y)[x] ||
            a.dc().plane[c].Row(y)[x] != b.dc().plane[c].Row(y)[x])
          return false;
      }
    }
  }
  for (size_t g = 0; g < a.ac_group_count(); ++g) {
    VarDctAcGroupView av, bv;
    if (!Check(a.GetAcGroup(g, &av)) || !Check(b.GetAcGroup(g, &bv)) ||
        av.used_coefficient_count != bv.used_coefficient_count)
      return false;
    for (size_t c = 0; c < 3; ++c) {
      if (av.coefficients[c].data() == bv.coefficients[c].data() ||
          !std::ranges::equal(av.coefficients[c], bv.coefficients[c]))
        return false;
    }
  }
  return true;
}

struct Retained {
  std::unique_ptr<CompletedVarDctFrame> frame;
  std::vector<uint8_t> bytes;
};

bool SerializeEqual(const Retained &retained) {
  std::vector<uint8_t> bytes;
  return Check(codestream_internal::EncodeVarDctCodestreamFromView(
             retained.frame->view(), {}, &bytes)) &&
         bytes == retained.bytes;
}

bool RunCase(Extent2D extent, bool deferred_frontend = false,
             GpuBackend* shared_gpu = nullptr,
             std::vector<Retained>* exported = nullptr, size_t image_seed = 0) {
  const size_t completed_class = static_cast<size_t>(
    resource_budget_internal::ResourceClass::kCompletedFrame);
  auto& budget = resource_budget_internal::DefaultResourceBudget();
  const size_t earlier_backings = budget.snapshot().classes[completed_class].backing_count;
  std::vector<Retained> retained;
  {
    FrameGeometry geometry;
    if (!Check(FrameGeometry::Create(extent, &geometry)))
      return false;
    const auto padded = geometry.padded_frame();
    const auto blocks = geometry.block_grid().blocks;
    const size_t block_count = blocks.width * blocks.height;
    Image3FBuffer source(extent), rgb(padded), opsin(padded);
    for (size_t y = 0; y < padded.height; ++y) {
      for (size_t x = 0; x < padded.width; ++x) {
        const size_t sx = std::min(x, extent.width - 1);
        const size_t sy = std::min(y, extent.height - 1);
        for (size_t c = 0; c < 3; ++c) {
          const float value =
              0.03f +
              0.8f * static_cast<float>(
                (sx * (c + 3) + sy * 7 + image_seed * 17) % 127) / 127.0f;
          rgb.view().plane[c].Row(y)[x] = value;
          if (x < extent.width && y < extent.height) {
            source.view().plane[c].Row(y)[x] = value;
          }
        }
      }
    }
    if (!Check(LinearRgbToOpsin(rgb.const_view(), 255.0f, opsin.view())))
      return false;
    std::unique_ptr<GpuBackend> owned_gpu;
    GpuBackend* gpu = shared_gpu;
    if (gpu == nullptr) {
      if (!Check(CreateMetalBackend(GJXL_METALLIB_PATH, &owned_gpu)))
        return false;
      gpu = owned_gpu.get();
    }
    AcStrategyGrid strategies;
    if (!Check(AcStrategyGrid::Create(blocks, &strategies)))
      return false;
    strategies.fill_dct8();
    std::vector<uint8_t> sharpness(block_count, 4);
    std::unique_ptr<PreparedAqEvaluation> prepared;
    if (!Check(PrepareAqEvaluation(
            *gpu,
            {
                .original_linear_rgb = source.const_view(),
                .coding_opsin = opsin.const_view(),
                .strategies = &strategies,
                .epf_sharpness = {sharpness.data(), blocks, blocks.width},
                .frame_only_resident_initial_quant = deferred_frontend,
                .resident_ac_strategy_inputs = deferred_frontend,
                .resident_quantization = true,
                .coefficient_decision_mode =
                    AcCoefficientDecisionMode::kAdjustedSharedQuant,
                .defer_final_transform_metadata = deferred_frontend,
            },
            &prepared)))
      return false;
    const Extent2D tiles{(blocks.width + 7) / 8, (blocks.height + 7) / 8};
    std::vector<int8_t> cfl(tiles.width * tiles.height, 0);
    std::vector<float> initial(block_count, 0.8f);
    for (size_t configuration = 0; configuration < 2; ++configuration) {
      // Force a final strategy grid different from the preparation grid.
      if (configuration == 1 && blocks.width >= 12 && blocks.height >= 8) {
        if (!Check(AcStrategyGrid::Create(blocks, &strategies)) ||
            !Check(strategies.Set(0, 0, AcStrategyType::kDct32x32)) ||
            !Check(strategies.Set(4, 0, AcStrategyType::kDct32x16)) ||
            !Check(strategies.Set(6, 0, AcStrategyType::kDct16x32)) ||
            !Check(strategies.Set(10, 0, AcStrategyType::kDct16x16)) ||
            !Check(strategies.Set(6, 2, AcStrategyType::kDct16x8)) ||
            !Check(strategies.Set(7, 2, AcStrategyType::kDct8x16)))
          return false;
        strategies.fill_empty_dct8();
      }
      std::fill(sharpness.begin(), sharpness.end(), configuration == 0 ? 4 : 5);
      if (!Check(prepared->Reconfigure(
              strategies, {sharpness.data(), blocks, blocks.width})) ||
          !Check(prepared->SetInvariantColorCorrelation(
              {cfl.data(), tiles, tiles.width},
              {cfl.data(), tiles, tiles.width}))) {
        return false;
      }
      adaptive_quantization_internal::ButteraugliPolicySetup setup;
      const float target = configuration == 0 ? 1.0f : 1.2f;
      if (!Check(adaptive_quantization_internal::PrepareButteraugliPolicy(
              {initial.data(), blocks, blocks.width}, target, &setup)))
        return false;
      for (const auto [iterations, final_score] :
           std::array<std::pair<size_t, bool>, 3>{
               {{0, true}, {2, true}, {2, false}}}) {
        AqResidentButteraugliPolicyInput input{
            .adjusted_initial_quant_field = {initial.data(), blocks,
                                             blocks.width},
            .quant_dc = setup.quant_dc,
            .butteraugli_target = target,
            .lower_bound = setup.lower_bound,
            .upper_bound = setup.upper_bound,
            .iterations = iterations,
            .evaluate_final_field = final_score,
        };
        VarDctEncoderFrame owned;
        std::vector<double> expected_scores, actual_scores;
        if (!Check(prepared->EvaluateResidentButteraugliPolicy(
                input, {.score_history = &expected_scores, .frame = &owned})))
          return false;
        Retained next;
        if (!Check(EncodeVarDctCodestream(owned, &next.bytes)))
          return false;
        // Reading an earlier result while the same evaluator writes another
        // frame must not race with output or scratch reuse.
        auto reader = std::async(std::launch::async, [&] {
          return retained.empty() || SerializeEqual(retained.front());
        });
        const auto before = gpu->stats();
        if (prepared->EvaluateResidentButteraugliPolicy(
                        input,
                        {
                            .score_history = &actual_scores,
                            .frame = &owned,
                            .completed_frame = &next.frame,
                        })
                    .code() != StatusCode::kInvalidArgument ||
            gpu->stats().committed_submissions !=
                before.committed_submissions ||
            gpu->stats().successful_allocations !=
                before.successful_allocations ||
            next.frame != nullptr || !actual_scores.empty())
          return false;
        const Status status = prepared->EvaluateResidentButteraugliPolicy(
            input,
            {.score_history = &actual_scores, .completed_frame = &next.frame});
        const bool reader_ok = reader.get();
        if (!Check(status) || !reader_ok || next.frame == nullptr ||
            gpu->stats().committed_submissions !=
                before.committed_submissions + 1 ||
            expected_scores != actual_scores ||
            !Equal(vardct_frame_internal::BorrowFrame(owned),
                   next.frame->view()) ||
            !SerializeEqual(next)) {
          std::cerr << "Completed frame parity/lifetime failure\n";
          return false;
        }
        auto* profiler = dynamic_cast<
          gpu_profile_internal::PreparedAqEvaluationProfiler*>(prepared.get());
        if (profiler == nullptr) return false;
        Retained profiled;
        profiled.bytes = next.bytes;
        std::vector<double> profiled_scores;
        gpu_profile_internal::GpuExecutionProfile profile;
        const auto before_profile = gpu->stats();
        if (!Check(profiler->EvaluateResidentButteraugliPolicyProfiled(
              input, {.score_history = &profiled_scores,
                      .completed_frame = &profiled.frame},
              gpu_profile_internal::GpuProfilingMode::kStage, &profile)) ||
            profiled.frame == nullptr || profiled_scores != expected_scores ||
            profile.submissions.size() != 1 ||
            gpu->stats().committed_submissions !=
              before_profile.committed_submissions + 1 ||
            !SerializeEqual(profiled)) return false;
        retained.push_back(std::move(profiled));
        retained.push_back(std::move(next));
      }
    }
    prepared.reset();
    if (!Check(metal_internal::EmptyMetalAqScratchArenasForTesting(*gpu)))
      return false;
    for (const auto &result : retained)
      if (!SerializeEqual(result))
        return false;
  }
  // No source, strategy grid, evaluator, or locally owned backend survives.
  // At this GPU-accounting checkpoint each completed frame has one independent
  // device backing; CPU metadata is not yet attached to the ledger.
  if (budget.snapshot().classes[completed_class].backing_count !=
      earlier_backings + retained.size()) return false;
  for (const auto &result : retained)
    if (!SerializeEqual(result))
      return false;
  if (exported != nullptr) {
    for (auto& result : retained) exported->push_back(std::move(result));
  }
  return true;
}

bool CheckPreparationIntegration() {
  std::vector<Retained> retained;
  {
    std::unique_ptr<GpuBackend> gpu;
    if (!Check(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu))) return false;
    // The same backend sees changed images and layouts while earlier completed
    // frames remain live. Deferred metadata must be finalized by Reconfigure.
    if (!RunCase({273, 265}, true, gpu.get(), &retained) ||
        MetalButteraugliCacheBytesForTesting(*gpu) == 0 ||
        !Check(EmptyMetalButteraugliCacheForTesting(*gpu))) return false;
    // Reacquire the same capacity first so this takes the Empty recovery path,
    // rather than merely discarding an oversized cache entry on downsizing.
    if (!RunCase({273, 265}, true, gpu.get(), &retained, 1) ||
        !RunCase({129, 127}, true, gpu.get(), &retained, 2)) return false;
    for (const auto& result : retained)
      if (!SerializeEqual(result)) return false;

    std::atomic<bool> trim_ok{true};
    std::jthread trimmer([&](std::stop_token stop) {
      while (!stop.stop_requested()) {
        if (!gpu->TrimPreparationCache().ok()) trim_ok.store(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    // Capture stable pointees, not vector elements that may move on append.
    auto reader = std::async(std::launch::async,
      [frame = retained.front().frame.get(), expected = retained.front().bytes] {
        std::vector<uint8_t> bytes;
        return Check(codestream_internal::EncodeVarDctCodestreamFromView(
          frame->view(), {}, &bytes)) && bytes == expected;
      });
    const bool encoded = RunCase({273, 265}, true, gpu.get(), &retained, 3);
    trimmer.request_stop();
    trimmer.join();
    if (!encoded || !reader.get() || !trim_ok.load() ||
        !Check(gpu->TrimPreparationCache()) ||
        MetalPreparationCacheBytesForTesting(*gpu) != 0) return false;
  }
  for (const auto& result : retained)
    if (!SerializeEqual(result)) return false;
  return true;
}
} // namespace

int main() {
  if (!CheckPreparationIntegration()) return EXIT_FAILURE;
  for (const auto extent :
       std::array<Extent2D, 4>{{{1, 1}, {19, 17}, {273, 265}, {2057, 17}}}) {
    if (!RunCase(extent))
      return EXIT_FAILURE;
  }
  std::cout
      << "Completed Metal frames survive producer and backend lifetime.\n";
  return EXIT_SUCCESS;
}
