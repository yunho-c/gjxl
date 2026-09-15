// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <iostream>
#include <stdexcept>

#include "benchmarks/synthetic_images.h"
#include "codestream/batch_workflow.h"
#include "codestream/compatibility_workflow_storage_plan.h"
#include "codestream/resident_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}

VarDctEncodingOptions Explicit(VarDctEncodingOptions options) {
  if (options.dc_quantization == DcQuantizationMode::kAutomatic)
    options.dc_quantization = options.effort >= 4
                                  ? DcQuantizationMode::kPredictionAware
                                  : DcQuantizationMode::kRound;
  if (!options.adaptive_dc_smoothing.has_value())
    options.adaptive_dc_smoothing = options.effort >= 4;
  return options;
}

void CheckUintSearchPolicy() {
  for (int effort = 1; effort <= 10; ++effort) {
    VarDctEncodingOptions options{.effort = effort};
    Check(UseDcUintSearch(options) == (effort == 4),
          "DC uint search effort scope changed");
    options.dc_quantization = DcQuantizationMode::kRound;
    Check(!UseDcUintSearch(options), "Ordinary DC enabled uint search");
  }
  VarDctEncodingOptions options{.effort = 4};
  options.density_mode = VarDctDensityMode::kHighDensity;
  Check(!UseDcUintSearch(options), "High density enabled DC uint search");
  options.density_mode = VarDctDensityMode::kDefault;
  options.compression_mode = VarDctCompressionMode::kMaximumCompression;
  Check(!UseDcUintSearch(options), "Maximum compression enabled DC uint search");
  options.compression_mode = VarDctCompressionMode::kAutomatic;
  options.rate_control_mode = VarDctRateControlMode::kMaximumError;
  Check(!UseDcUintSearch(options), "Maximum error enabled DC uint search");
}

void CheckPlans() {
  for (int effort = 1; effort <= 10; ++effort) {
    for (auto quantization :
         {DcQuantizationMode::kAutomatic, DcQuantizationMode::kRound,
          DcQuantizationMode::kPredictionAware}) {
      for (auto smoothing : {std::optional<bool>{}, std::optional<bool>{false},
                             std::optional<bool>{true}}) {
        VarDctEncodingOptions automatic{.effort = effort,
                                        .backend =
                                            VarDctBackendPreference::kCpu,
                                        .dc_quantization = quantization,
                                        .adaptive_dc_smoothing = smoothing};
        CpuWorkflowStoragePlan cpu, explicit_cpu;
        Ok(ComputeCpuWorkflowStoragePlan({257, 129}, {.encoding = automatic},
                                         &cpu));
        Ok(ComputeCpuWorkflowStoragePlan(
            {257, 129}, {.encoding = Explicit(automatic)}, &explicit_cpu));
        Check(cpu == explicit_cpu,
              "Automatic CPU admission differs from resolved policy");
        automatic.backend = VarDctBackendPreference::kMetal;
        ResidentWorkflowStoragePlan resident, explicit_resident;
        Ok(ComputeResidentWorkflowStoragePlan(
            {257, 129}, {.encoding = automatic}, &resident));
        Ok(ComputeResidentWorkflowStoragePlan(
            {257, 129}, {.encoding = Explicit(automatic)}, &explicit_resident));
        Check(resident == explicit_resident,
              "Automatic resident admission differs from resolved policy");
        for (auto mode : {GpuAdaptiveQuantizationMode::kExactCoefficients,
                          GpuAdaptiveQuantizationMode::kMaximumThroughput}) {
          automatic.metal_aq_mode = mode;
          MetalCompatibilityWorkflowStoragePlan compatibility,
              explicit_compatibility;
          Ok(ComputeMetalCompatibilityWorkflowStoragePlan(
              {257, 129}, {.encoding = automatic}, &compatibility));
          Ok(ComputeMetalCompatibilityWorkflowStoragePlan(
              {257, 129}, {.encoding = Explicit(automatic)},
              &explicit_compatibility));
          Check(
              compatibility == explicit_compatibility,
              "Automatic compatibility admission differs from resolved policy");
        }
      }
    }
  }
}

void CheckEncoding(ConstImage3FView image, VarDctBackendPreference backend) {
  std::vector<VarDctBatchEncodingRequest> requests;
  std::vector<std::vector<uint8_t>> expected_batch;
  for (int effort : {3, 4, 7}) {
    for (auto quantization :
         {DcQuantizationMode::kAutomatic, DcQuantizationMode::kRound,
          DcQuantizationMode::kPredictionAware}) {
      for (auto smoothing : {std::optional<bool>{}, std::optional<bool>{false},
                             std::optional<bool>{true}}) {
        VarDctEncodingOptions automatic{.effort = effort,
                                        .cpu_thread_count = 2,
                                        .backend = backend,
                                        .dc_quantization = quantization,
                                        .adaptive_dc_smoothing = smoothing};
        const auto explicit_options = Explicit(automatic);
        std::vector<uint8_t> actual, expected;
        VarDctEncodingSummary actual_summary, expected_summary;
        const Status status = EncodeLinearRgbVarDctCodestream(
            image, automatic, &actual, &actual_summary);
        if (backend == VarDctBackendPreference::kMetal &&
            status.code() == StatusCode::kUnavailable) {
          std::cout << "Metal DC policy encoding unavailable: "
                    << status.message() << '\n';
          return;
        }
        Ok(status);
        Ok(EncodeLinearRgbVarDctCodestream(image, explicit_options, &expected,
                                           &expected_summary));
        Check(actual == expected && actual_summary == expected_summary,
              "Automatic encode differs from explicit policy");
        Check(actual_summary.dc_quantization ==
                      explicit_options.dc_quantization &&
                  actual_summary.adaptive_dc_smoothing ==
                      *explicit_options.adaptive_dc_smoothing,
              "Summary did not report effective DC controls");
        if (quantization == DcQuantizationMode::kAutomatic &&
            !smoothing.has_value()) {
          requests.push_back({image, automatic});
          expected_batch.push_back(expected);
        }
      }
    }
  }
  std::unique_ptr<VarDctBatchEncoder> batch;
  Ok(VarDctBatchEncoder::Create(2, &batch));
  std::vector<VarDctBatchEncodingResult> results;
  Ok(batch->Encode(requests, &results));
  Check(results.size() == expected_batch.size(), "Batch result count changed");
  for (size_t i = 0; i < results.size(); ++i) {
    Ok(results[i].status);
    Check(results[i].codestream == expected_batch[i],
          "Batch ignored per-image effort DC policy");
  }
}
} // namespace

int main() {
  try {
    CheckUintSearchPolicy();
    CheckPlans();
    Image3FBuffer image({64, 48});
    benchmark::FillBatchTexture(image.view());
    CheckEncoding(image.const_view(), VarDctBackendPreference::kCpu);
    CheckEncoding(image.const_view(), VarDctBackendPreference::kMetal);
    std::cout << "DC defaults, independent overrides, admission, and batch "
                 "policy passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
