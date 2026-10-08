// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/frame_encoder.h"
#include "codestream/headers_internal.h"
#include "codestream/modular/stream_encoder.h"
#include "codestream/modular/tree_codec.h"
#include "codestream/sections.h"
#include "codestream/modular/parallel.h"

namespace gjxl::modular_internal {
using namespace codestream_internal;
Status EncodeModularFrame(const codec_internal::ImageMetadata &metadata,
                          const ModularStreamPlan &layout, PreparedModularTokens &&input,
                          EntropyCodingMode mode, const ModularWorkflowStoragePlan &storage,
                          CodestreamBuffer *out) try {
  if (!out || input.streams.size() != layout.streams.size())
    return Status::InvalidArgument("Invalid Modular frame handoff");
  Storage<BitWriter> sections;
  {
    PreparedModularTokens tokens = std::move(input);
    EntropyCode model;
    {
      thread_budget_internal::EncodeScope serial(1);
      if (auto s = OptimizeEntropyCode(
              tokens.streams, {.context_count = static_cast<uint32_t>(tokens.context_count)},
              &model);
          !s.ok())
        return s;
      if (mode == EntropyCodingMode::kAns) {
        EntropyCode ans;
        if (auto s = OptimizeAnsEntropyCode(tokens.streams, model, &ans); !s.ok())
          return s;
        model = std::move(ans);
      }
    }
    sections.resize(layout.geometry.section_count());
    {
      thread_budget_internal::EncodeScope serial(1);
      auto &global = sections[0];
      if (auto s = global.WithMaxBits(
              storage.maximum_global_bits,
              [&]() -> Status {
                if (auto s = global.WriteBits(1, 1); !s.ok())
                  return s; // default DC matrices
                if (auto s = WriteGlobalTreeInTransaction(
                        std::span(tokens.tree_tokens).first(tokens.tree_token_count), &global);
                    !s.ok())
                  return s;
                if (auto s = WriteGlobalModelInTransaction(model, &global); !s.ok())
                  return s;
                if (auto s = WriteCodingStreamHeader(tokens.policy.weighted, tokens.policy.rct,
                                                     tokens.policy.transforms, &global);
                    !s.ok())
                  return s;
                if (tokens.streams[0].size())
                  return WriteStreamTokensWithValidatedModel(tokens.streams[0], model, &global);
                return Status::Ok();
              });
          !s.ok())
        return s;
    }
    size_t active_streams = 0;
    for (size_t i = 1; i < tokens.streams.size(); ++i)
      active_streams += tokens.streams[i].size() != 0;
    const size_t participants = std::min(storage.participants, std::max(size_t{1}, active_streams));
    if (auto status = RunModularStreams(
            layout.streams.size() - 1, participants,
            thread_budget_internal::WorkerLaunchSite::kModularEmission,
            [&](size_t index) -> Status {
              const size_t i = index + 1;
              if (tokens.streams[i].size() == 0)
                return Status::Ok();
              auto &section = sections[layout.streams[i].section];
              if (auto s = section.WithMaxBits(
                      storage.maximum_group_bits,
                      [&]() -> Status {
                        if (auto s = WriteCodingStreamHeader(tokens.policy.weighted, 0, &section);
                            !s.ok())
                          return s;
                        return WriteStreamTokensWithValidatedModel(tokens.streams[i], model,
                                                                   &section);
                      });
                  !s.ok())
                return s;
              return Status::Ok();
            });
        !status.ok())
      return status;
  } // Tokens and model are no longer live during final assembly.
  BitWriter file;
  if (auto s = WriteImageHeader(metadata, &file); !s.ok())
    return s;
  if (auto s = WriteFrameHeader(metadata, {}, &file); !s.ok())
    return s;
  if (auto s = WriteTocAndSections(sections, &file); !s.ok())
    return s;
  // Clear backing as well as elements before allocating the published copy.
  Storage<BitWriter>().swap(sections);
  return CodestreamBuffer::CopyFrom(file.padded_bytes(), out,
                                    resource_budget_internal::ResourceClass::kRetainedResult);
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular frame allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular frame storage overflow");
}
} // namespace gjxl::modular_internal
