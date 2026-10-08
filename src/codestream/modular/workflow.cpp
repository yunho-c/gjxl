// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/workflow.h"
#include "codestream/modular/frame_encoder.h"
#include "codestream/modular/search.h"
#include "codestream/workflow_admission_scope.h"
#include "core/publication_output.h"
#include "core/thread_budget.h"

namespace gjxl::modular_internal {
namespace {
Status EncodeSearch(PackedModularImageView input, ModularEncodingOptions options,
                    resource_budget_internal::PublicationOutput<uint8_t> output);
Status Encode(PackedModularImageView input, ModularEncodingOptions options,
              resource_budget_internal::PublicationOutput<uint8_t> output) try {
  if (output == nullptr)
    return Status::InvalidArgument("Null Modular workflow output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  if (options.search)
    return EncodeSearch(input, options, output);
  ModularWorkflowStoragePlan storage;
  if (auto s =
          ComputeModularWorkflowStoragePlan(input.extent, input.format, options.entropy,
                                            options.coding, &storage, options.cpu_thread_count);
      !s.ok())
    return s;
  codestream_internal::WorkflowAdmission admission;
  if (auto s = admission.Start(storage.working.peak_bytes, options.execution_domain); !s.ok())
    return s;
  thread_budget_internal::CpuExecutionScope execution;
  if (auto s = execution.Start(options.execution_domain, options.cpu_thread_count); !s.ok())
    return s;
  thread_budget_internal::EncodeScope threads(options.cpu_thread_count);
  resource_budget_internal::ManagedHostScope managed(
      resource_budget_internal::ResourceClass::kPreparation);
  ModularStreamPlan layout;
  PreparedModularTokens tokens;
  codec_internal::ImageMetadata metadata;
  {
    ModularEncoderFrame frame;
    if (auto s = ModularEncoderFrame::Prepare(input, options.coding, &frame); !s.ok())
      return s;
    metadata = frame.metadata();
    if (options.coding != ModularCodingPolicy{})
      metadata.modular_16_bit_buffer_sufficient = false;
    ChannelShape shape;
    if (auto s = DescribeImage(frame.image(), &shape); !s.ok())
      return s;
    if (auto s =
            BuildModularStreamPlan(storage.geometry, shape.channels(), shape.metadata, &layout);
        !s.ok())
      return s;
    if (auto s = TokenizeModular(frame, layout, options.coding, &tokens, storage.participants);
        !s.ok())
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
Status EncodeSearch(PackedModularImageView input, ModularEncodingOptions options,
                    resource_budget_internal::PublicationOutput<uint8_t> output) {
  if (options.coding != ModularCodingPolicy{})
    return Status::InvalidArgument("Search cannot override a prescribed Modular policy");
  ModularWorkflowStoragePlan storage;
  if (auto s = ComputeModularSearchStoragePlan(input.extent, input.format, options.entropy,
                                               &storage, options.cpu_thread_count);
      !s.ok())
    return s;
  codestream_internal::WorkflowAdmission admission;
  if (auto s = admission.Start(storage.working.peak_bytes, options.execution_domain); !s.ok())
    return s;
  thread_budget_internal::CpuExecutionScope execution;
  if (auto s = execution.Start(options.execution_domain, options.cpu_thread_count); !s.ok())
    return s;
  resource_budget_internal::ManagedHostScope managed(
      resource_budget_internal::ResourceClass::kPreparation);
  thread_budget_internal::EncodeScope threads(options.cpu_thread_count);
  options.search = false;
  codestream_internal::CodestreamBuffer best;
  if (auto s = Encode(input, options, &best); !s.ok())
    return s;
  ModularInputProfile profile;
  if (auto s = ResolveModularInput(input.extent, input.format, &profile); !s.ok())
    return s;
  for (uint8_t rct : {uint8_t{0}, uint8_t{6}, uint8_t{9}}) {
    if (rct && profile.channel_count < 3)
      continue;
    ModularCodingPolicy single, split;
    {
      ModularEncoderFrame frame;
      ModularStreamPlan layout;
      if (auto s = ModularEncoderFrame::Prepare(input, rct, &frame); !s.ok())
        return s;
      if (auto s = BuildModularStreamPlan(storage.geometry, profile.channels(), 0, &layout);
          !s.ok())
        return s;
      if (auto s = LearnModularPolicies(frame, layout, rct, &single, &split); !s.ok())
        return s;
    }
    for (size_t i = 0; i < 2; ++i) {
      const auto &policy = i ? split : single;
      if (policy == ModularCodingPolicy{} || (i && split == single))
        continue;
      options.coding = policy;
      codestream_internal::CodestreamBuffer candidate;
      if (auto s = Encode(input, options, &candidate); !s.ok())
        return s;
      // Compare complete bytes, including transforms/tree/models/TOC/padding.
      // Baseline and earlier candidates win ties. Replacement releases the old winner.
      if (candidate.view().size() < best.view().size())
        best = std::move(candidate);
    }
  }
  uint16_t palette_colors;
  if (auto s = ProbePalette(input, &palette_colors); !s.ok())
    return s;
  std::array<ModularCodingPolicy, 4> candidates;
  size_t count;
  if (auto s =
          BuildTransformCandidates(input.extent, input.format, palette_colors, &candidates, &count);
      !s.ok())
    return s;
  for (size_t i = 0; i < count; ++i) {
    options.coding = candidates[i];
    codestream_internal::CodestreamBuffer candidate;
    if (auto s = Encode(input, options, &candidate); !s.ok())
      return s;
    if (candidate.view().size() < best.view().size())
      best = std::move(candidate);
  }
  output.Publish(std::move(best));
  return Status::Ok();
}
} // namespace
Status EncodeRgb8ModularOwned(Rgb8View input, ModularEncodingOptions options,
                              codestream_internal::CodestreamBuffer *out) {
  return Encode(input.packed(), std::move(options), out);
}
Status EncodeRgb8Modular(Rgb8View input, ModularEncodingOptions options,
                         std::vector<uint8_t> *out) {
  return Encode(input.packed(), std::move(options), out);
}
Status EncodeModularImageOwned(PackedModularImageView input, ModularEncodingOptions options,
                               codestream_internal::CodestreamBuffer *out) {
  return Encode(input, std::move(options), out);
}
Status EncodeModularImage(PackedModularImageView input, ModularEncodingOptions options,
                          std::vector<uint8_t> *out) {
  return Encode(input, std::move(options), out);
}
} // namespace gjxl::modular_internal
