// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/workflow.h"
#include "codestream/modular/frame_encoder.h"
#include "codestream/workflow_admission_scope.h"
#include "core/publication_output.h"
#include "core/thread_budget.h"

namespace gjxl::modular_internal {
namespace {
Status Encode(Rgb8View input, ModularEncodingOptions options,
              resource_budget_internal::PublicationOutput<uint8_t> output) try {
  if (output == nullptr)
    return Status::InvalidArgument("Null Modular workflow output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  ModularWorkflowStoragePlan storage;
  if (auto s = ComputeModularWorkflowStoragePlan(input.extent, options.entropy, &storage); !s.ok())
    return s;
  codestream_internal::WorkflowAdmission admission;
  if (auto s = admission.Start(storage.working.peak_bytes, options.execution_domain); !s.ok())
    return s;
  thread_budget_internal::CpuExecutionScope execution;
  if (auto s = execution.Start(options.execution_domain, 1); !s.ok())
    return s;
  thread_budget_internal::EncodeScope serial(1);
  resource_budget_internal::ManagedHostScope managed(
      resource_budget_internal::ResourceClass::kPreparation);
  ModularStreamPlan layout;
  PreparedModularTokens tokens;
  codec_internal::ImageMetadata metadata;
  {
    ModularEncoderFrame frame;
    if (auto s = ModularEncoderFrame::Prepare(input, &frame); !s.ok())
      return s;
    metadata = frame.metadata();
    if (auto s = BuildModularStreamPlan(storage.geometry, Rgb8Channels(input.extent), 0, &layout);
        !s.ok())
      return s;
    if (auto s = TokenizeRgb8(frame, layout, &tokens); !s.ok())
      return s;
  }
  codestream_internal::CodestreamBuffer candidate;
  if (auto s = EncodeModularFrame(metadata, layout, std::move(tokens), options.entropy, storage,
                                  &candidate);
      !s.ok())
    return s;
  output.Publish(std::move(candidate));
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular workflow allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular workflow storage overflow");
}
} // namespace
Status EncodeRgb8ModularOwned(Rgb8View input, ModularEncodingOptions options,
                              codestream_internal::CodestreamBuffer *out) {
  return Encode(input, std::move(options), out);
}
Status EncodeRgb8Modular(Rgb8View input, ModularEncodingOptions options,
                         std::vector<uint8_t> *out) {
  return Encode(input, std::move(options), out);
}
} // namespace gjxl::modular_internal
