// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

// Policy and provider orchestration only: no image encoding or GPU work.
#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "codec/quantization_pipeline_internal.h"
#include "codestream/cpu_workflow_storage_plan.h"
#include "codestream/resident_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::quantization_pipeline_internal;

bool Check(bool ok, const char* message) {
  if (!ok) std::cerr << message << '\n';
  return ok;
}

bool CheckPolicy() {
  constexpr std::array<size_t, 10> updates{0, 0, 0, 1, 1, 1, 2, 3, 3, 4};
  for (int effort = 1; effort <= 10; ++effort) {
    for (auto density : {VarDctDensityMode::kDefault,
                         VarDctDensityMode::kHighDensity}) {
      for (auto rate : {VarDctRateControlMode::kButteraugliTarget,
                       VarDctRateControlMode::kTargetBytes,
                       VarDctRateControlMode::kTargetBitsPerPixel,
                       VarDctRateControlMode::kMaximumError}) {
        for (auto compression : {VarDctCompressionMode::kAutomatic,
                                 VarDctCompressionMode::kMaximumCompression}) {
          VarDctEncodingOptions o;
          o.effort = effort;
          o.density_mode = density;
          o.rate_control_mode = rate;
          o.compression_mode = compression;
          const bool high = density == VarDctDensityMode::kHighDensity;
          if (!Check(UseFixedDct8Strategy(o) ==
                       (effort < 5 && !high &&
                        rate != VarDctRateControlMode::kMaximumError),
                     "Effort or override selected the wrong strategy policy") ||
              !Check(AdaptiveQuantizationIterations(o) ==
                       (high ? 4 : updates[effort - 1]),
                     "Transform policy changed the AQ update schedule"))
            return false;
        }
      }
    }
  }
  return true;
}

struct Search final : AcStrategySearchProvider {
  size_t calls = 0;
  Status Find(ConstImage3FView, ConstPlaneF32View, ConstPlaneF32View,
              const ColorCorrelationMap&, AcStrategySearchOptions,
              AcStrategyGrid* out) override {
    ++calls;
    auto status = AcStrategyGrid::Create({2, 2}, out);
    if (!status.ok()) return status;
    return out->Set(0, 0, AcStrategyType::kDct16x16);
  }
};

struct Aq final : AdaptiveQuantizationProvider {
  AcStrategyType expected = AcStrategyType::kDct8;
  size_t expected_iterations = 0;
  size_t calls = 0;
  Status Find(ConstImage3FView, ConstImage3FView,
              const AcStrategyGrid& strategies, ConstPlaneF32View initial,
              ConstPlaneU8View sharpness, AdaptiveQuantizationOptions options,
              PreparedButteraugliReference*, AdaptiveQuantizationOutput) override {
    ++calls;
    if (!strategies.complete() || strategies.extent() != Extent2D{2, 2} ||
        options.iterations != expected_iterations ||
        options.butteraugli_target != 1.7f || initial.data[0] != 0.75f ||
        sharpness.data[0] != 4)
      return Status::InvalidArgument("AQ inputs or policy changed");
    for (size_t y = 0; y < 2; ++y) {
      for (size_t x = 0; x < 2; ++x) {
        AcStrategyCell cell;
        auto status = strategies.Get(x, y, &cell);
        if (!status.ok()) return status;
        if (cell.strategy != expected ||
            cell.is_anchor != (expected == AcStrategyType::kDct8 ||
                               (x == 0 && y == 0)))
          return Status::InvalidArgument("AQ received the wrong strategy grid");
      }
    }
    return Status::Ok();
  }
};

bool CheckProviderBypass() {
  std::array<float, 256> pixels{};
  const ConstPlaneF32View plane{pixels.data(), {16, 16}, 16};
  const ConstImage3FView image{{plane, plane, plane}};
  CpuQuantizationPipelineOptions o;
  o.butteraugli_target = 1.7f;
  if (!Check(!o.fixed_dct8, "Low-level pipeline default lost AC search"))
    return false;
  PreparedQuantizationPipeline prepared;
  prepared.source_extent = prepared.padded_extent = {16, 16};
  prepared.block_extent = {2, 2};
  prepared.coding_opsin = image;
  prepared.profile = o.adaptive_quantization.profile;
  prepared.preprocessing_ready = true;
  prepared.initial_quant.assign(4, 0.75f);
  prepared.epf_sharpness.assign(4, 4);
  VarDctEncoderFrame frame;
  std::vector<double> history;
  CpuQuantizationPipelineOutput output;
  output.adaptive_quantization.frame = &frame;
  output.adaptive_quantization.score_history = &history;
  Search search;
  Aq aq;
  size_t expected_search_calls = 0;
  size_t expected_aq_calls = 0;
  // Fresh fixed grid, mixed-to-fixed reuse, repeated attempts, then search.
  for (const bool fixed : {true, false, true, true, false}) {
    for (size_t iterations : {0u, 1u}) {
      o.fixed_dct8 = fixed;
      o.adaptive_quantization.iterations = iterations;
      aq.expected_iterations = iterations;
      aq.expected = fixed ? AcStrategyType::kDct8 : AcStrategyType::kDct16x16;
      const auto status = RunPreparedQuantizationPipelineWithProviders(
        image, prepared, search, aq, o, output, true,
        {.initial_quantization = false, .adaptive_quant_field = false,
         .block_distance_map = false, .reconstructed_linear_rgb = false,
         .final_perceptual_evaluation = false});
      expected_search_calls += !fixed;
      ++expected_aq_calls;
      if (!Check(status.ok(), status.message().data()) ||
          !Check(search.calls == expected_search_calls &&
                   aq.calls == expected_aq_calls,
                 "Fixed strategy invoked search or skipped AQ"))
        return false;
    }
  }
  return true;
}

bool CheckSearchStorage() {
  CpuWorkflowStorageOptions cpu;
  cpu.encoding.backend = VarDctBackendPreference::kCpu;
  cpu.encoding.effort = 4;
  CpuWorkflowStoragePlan cpu_fixed, cpu_search;
  if (!Check(ComputeCpuWorkflowStoragePlan({257, 257}, cpu, &cpu_fixed).ok(),
             "Fixed CPU plan failed")) return false;
  cpu.encoding.effort = 5;
  if (!Check(ComputeCpuWorkflowStoragePlan({257, 257}, cpu, &cpu_search).ok() &&
               cpu_fixed.frontend.peak_bytes < cpu_search.frontend.peak_bytes &&
               cpu_fixed.aq == cpu_search.aq,
             "CPU search storage was retained or AQ storage changed"))
    return false;
  ResidentWorkflowStorageOptions metal;
  metal.encoding.backend = VarDctBackendPreference::kMetal;
  metal.encoding.effort = 4;
  ResidentWorkflowStoragePlan fixed, search;
  if (!Check(ComputeResidentWorkflowStoragePlan({257, 257}, metal, &fixed).ok(),
             "Fixed resident plan failed")) return false;
  metal.encoding.effort = 5;
  return Check(ComputeResidentWorkflowStoragePlan({257, 257}, metal, &search).ok() &&
                 fixed.device_bytes < search.device_bytes &&
                 fixed.frontend.peak_bytes < search.frontend.peak_bytes &&
                 fixed.score_count == search.score_count,
               "Resident search storage was retained or AQ count changed");
}
}  // namespace

int main() {
  return CheckPolicy() && CheckProviderBypass() && CheckSearchStorage()
    ? EXIT_SUCCESS : EXIT_FAILURE;
}
