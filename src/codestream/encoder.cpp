// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder/enc_frame.cc.

#include "codestream/encoder.h"

#include <cstddef>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/codestream.h"
#include "codec/vardct_frame.h"
#include "codestream/ac_group.h"
#include "codestream/bit_writer.h"
#include "codestream/dc_group.h"
#include "codestream/entropy.h"
#include "codestream/headers.h"
#include "codestream/sections.h"

namespace gjxl {
namespace {

Status AllocationFailure() {
  return Status::OutOfMemory("Codestream assembly allocation failed");
}

Status WriteDcGroupSection(
  const SimpleDcGroupTokenStreams& group,
  std::span<const EntropyToken> dc_tokens,
  std::span<const EntropyToken> metadata_tokens,
  const EntropyCode& code,
  BitWriter* writer) {

  if (Status status = WriteSimpleDcGroupModularHeader(writer); !status.ok()) {
    return status;
  }
  if (Status status = WriteTokenStream(dc_tokens, code, writer); !status.ok()) {
    return status;
  }
  if (Status status = WriteSimpleAcMetadataModularHeader(
        group.block_extent, group.transform_anchor_count, writer);
      !status.ok()) {
    return status;
  }
  return WriteTokenStream(metadata_tokens, code, writer);
}

Status CombineFrameSections(
  std::vector<BitWriter>* sections,
  size_t ac_group_count,
  BitWriter* writer) {

  if (sections == nullptr || writer == nullptr || sections->empty()) {
    return Status::InvalidArgument("Frame-section assembly input is invalid");
  }

  // JPEG XL collapses the four logical sections of a single-group frame into
  // one physical TOC section.
  if (ac_group_count == 1) {
    if (sections->size() != 4) {
      return Status::Internal("Single-group frame has an invalid section count");
    }
    for (size_t index = 1; index < sections->size(); ++index) {
      if (Status status = (*sections)[0].Append((*sections)[index]);
          !status.ok()) {
        return status;
      }
    }
    sections->resize(1);
  }
  return WriteTocAndSections(*sections, writer);
}

}  // namespace

Status EncodeVarDctCodestream(
  const VarDctEncoderFrame& frame, std::vector<uint8_t>* output) {

  if (output == nullptr) {
    return Status::InvalidArgument("Codestream output is null");
  }
  if (Status status = ValidateSimpleCodestreamFrame(frame); !status.ok()) {
    return status;
  }

  try {
    std::vector<SimpleDcGroupTokenStreams> dc_groups;
    if (Status status = TokenizeSimpleDcGroups(frame, &dc_groups);
        !status.ok()) {
      return status;
    }
    std::vector<SimpleAcGroupTokenStream> ac_groups;
    if (Status status = TokenizeSimpleAcGroups(frame, &ac_groups);
        !status.ok()) {
      return status;
    }
    if (dc_groups.empty() || ac_groups.empty()) {
      return Status::Internal("Validated frame produced no codestream groups");
    }

    std::vector<std::vector<EntropyToken>> dc_streams;
    if (dc_groups.size() > dc_streams.max_size() / 2) {
      return AllocationFailure();
    }
    dc_streams.reserve(2 * dc_groups.size());
    for (SimpleDcGroupTokenStreams& group : dc_groups) {
      dc_streams.push_back(std::move(group.dc_tokens));
      dc_streams.push_back(std::move(group.ac_metadata_tokens));
    }

    std::vector<std::vector<EntropyToken>> ac_streams;
    ac_streams.reserve(ac_groups.size());
    for (SimpleAcGroupTokenStream& group : ac_groups) {
      ac_streams.push_back(std::move(group.tokens));
    }

    EntropyCode dc_code;
    if (Status status = OptimizeEntropyCode(
          dc_streams, {.context_count = kSimpleDcContextCount}, &dc_code);
        !status.ok()) {
      return status;
    }
    EntropyCode ac_code;
    if (Status status = OptimizeEntropyCode(
          ac_streams, {.context_count = kSimpleAcContextCount}, &ac_code);
        !status.ok()) {
      return status;
    }

    if (dc_groups.size() > std::numeric_limits<size_t>::max() -
                           ac_groups.size() - 2) {
      return AllocationFailure();
    }
    const size_t section_count = 2 + dc_groups.size() + ac_groups.size();
    std::vector<BitWriter> sections(section_count);

    if (Status status = WriteSimpleDcGlobal(
          frame.quantizer().params(), dc_groups.size(), dc_code, &sections[0]);
        !status.ok()) {
      return status;
    }
    for (size_t index = 0; index < dc_groups.size(); ++index) {
      if (Status status = WriteDcGroupSection(
            dc_groups[index], dc_streams[2 * index],
            dc_streams[2 * index + 1], dc_code, &sections[1 + index]);
          !status.ok()) {
        return status;
      }
    }

    const size_t ac_global_index = 1 + dc_groups.size();
    if (Status status = WriteSimpleAcGlobal(
          ac_groups.size(), ac_code, &sections[ac_global_index]);
        !status.ok()) {
      return status;
    }
    const size_t ac_group_start = ac_global_index + 1;
    for (size_t index = 0; index < ac_groups.size(); ++index) {
      if (Status status = WriteTokenStream(
            ac_streams[index], ac_code, &sections[ac_group_start + index]);
          !status.ok()) {
        return status;
      }
    }

    BitWriter writer;
    if (Status status = WriteSimpleCodestreamHeader(
          frame.geometry().frame(), &writer);
        !status.ok()) {
      return status;
    }
    if (Status status = WriteSimpleFrameHeader(frame.profile(), &writer);
        !status.ok()) {
      return status;
    }
    if (Status status = CombineFrameSections(
          &sections, ac_groups.size(), &writer);
        !status.ok()) {
      return status;
    }
    if (!writer.byte_aligned()) {
      return Status::Internal("Assembled codestream is not byte-aligned");
    }

    const std::span<const uint8_t> bytes = writer.padded_bytes();
    std::vector<uint8_t> candidate(bytes.begin(), bytes.end());
    *output = std::move(candidate);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

}  // namespace gjxl
