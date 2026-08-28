// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder/enc_frame.cc.

#include "codestream/encoder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "codec/codestream.h"
#include "codec/vardct_frame.h"
#include "codestream/ac_group.h"
#include "codestream/bit_writer.h"
#include "codestream/dc_group.h"
#include "codestream/encoder_internal.h"
#include "codestream/entropy.h"
#include "codestream/headers.h"
#include "codestream/sections.h"

namespace gjxl {
namespace {

using ProfileClock = std::chrono::steady_clock;

uint64_t ElapsedNanoseconds(ProfileClock::time_point begin) {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      ProfileClock::now() - begin).count());
}

ProfileClock::time_point ProfileBegin(
  const codestream_internal::VarDctCodestreamProfile* profile) {
  return profile == nullptr ? ProfileClock::time_point{} : ProfileClock::now();
}

void ProfileEnd(
  const codestream_internal::VarDctCodestreamProfile* requested,
  ProfileClock::time_point begin,
  uint64_t* destination) {
  if (requested != nullptr) {
    *destination = ElapsedNanoseconds(begin);
  }
}

Status AllocationFailure() {
  return Status::OutOfMemory("Codestream assembly allocation failed");
}

template <typename Function>
Status RunParallelSections(size_t count, Function&& function) {
  if (count == 0) return Status::Ok();
  constexpr size_t kMaximumWorkers = 8;
  const size_t hardware_workers = std::max<size_t>(
    std::thread::hardware_concurrency(), 1);
  const size_t worker_count = std::min(
    count, std::min(kMaximumWorkers, hardware_workers));
  if (worker_count == 1) {
    for (size_t index = 0; index < count; ++index) {
      Status status = function(index);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  std::vector<Status> statuses(count);
  std::atomic<size_t> next_index{0};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  try {
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&] {
        while (true) {
          const size_t index =
            next_index.fetch_add(1, std::memory_order_relaxed);
          if (index >= count) break;
          try {
            statuses[index] = function(index);
          } catch (const std::bad_alloc&) {
            statuses[index] = AllocationFailure();
          } catch (const std::length_error&) {
            statuses[index] = AllocationFailure();
          } catch (...) {
            statuses[index] = Status::Internal(
              "Codestream section worker failed unexpectedly");
          }
        }
      });
    }
  } catch (const std::system_error&) {
    next_index.store(count, std::memory_order_relaxed);
    for (std::thread& worker : workers) worker.join();
    return Status::Internal("Unable to start codestream section workers");
  }
  for (std::thread& worker : workers) worker.join();
  for (const Status& status : statuses) {
    if (!status.ok()) return status;
  }
  return Status::Ok();
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

Status EncodeVarDctCodestreamImpl(
  const VarDctEncoderFrame& frame,
  std::vector<uint8_t>* output,
  codestream_internal::VarDctCodestreamProfile* profile) {

  if (output == nullptr) {
    return Status::InvalidArgument("Codestream output is null");
  }
  codestream_internal::VarDctCodestreamProfile candidate_profile;
  const ProfileClock::time_point total_begin = ProfileBegin(profile);
  const ProfileClock::time_point validation_begin = ProfileBegin(profile);
  const Status validation = ValidateSimpleCodestreamFrame(frame);
  ProfileEnd(
    profile, validation_begin, &candidate_profile.validation_nanoseconds);
  if (!validation.ok()) {
    return validation;
  }

  try {
    const ProfileClock::time_point dc_tokenization_begin = ProfileBegin(profile);
    std::vector<SimpleDcGroupTokenStreams> dc_groups;
    Status status = TokenizeSimpleDcGroups(frame, &dc_groups);
    ProfileEnd(
      profile, dc_tokenization_begin,
      &candidate_profile.dc_tokenization_nanoseconds);
    if (!status.ok()) {
      return status;
    }

    const ProfileClock::time_point ac_tokenization_begin = ProfileBegin(profile);
    std::vector<SimpleAcGroupTokenStream> ac_groups;
    status = TokenizeSimpleAcGroups(frame, &ac_groups);
    ProfileEnd(
      profile, ac_tokenization_begin,
      &candidate_profile.ac_tokenization_nanoseconds);
    if (!status.ok()) {
      return status;
    }
    if (dc_groups.empty() || ac_groups.empty()) {
      return Status::Internal("Validated frame produced no codestream groups");
    }

    const ProfileClock::time_point entropy_begin = ProfileBegin(profile);
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
    EntropyCodeCost dc_cost;
    if (Status status = OptimizeEntropyCode(
          dc_streams, {.context_count = kSimpleDcContextCount}, &dc_code,
          &dc_cost);
        !status.ok()) {
      return status;
    }
    EntropyCode ac_code;
    EntropyCodeCost ac_cost;
    if (Status status = OptimizeEntropyCode(
          ac_streams, {.context_count = kSimpleAcContextCount}, &ac_code,
          &ac_cost);
        !status.ok()) {
      return status;
    }
    if (dc_cost.model_bits >
          std::numeric_limits<uint64_t>::max() - ac_cost.model_bits ||
        dc_cost.token_bits >
          std::numeric_limits<uint64_t>::max() - ac_cost.token_bits) {
      return Status::InvalidArgument("Entropy profile bit count overflow");
    }
    candidate_profile.entropy_model_bits =
      dc_cost.model_bits + ac_cost.model_bits;
    candidate_profile.entropy_token_bits =
      dc_cost.token_bits + ac_cost.token_bits;
    candidate_profile.dc_entropy_clusters = dc_cost.cluster_count;
    candidate_profile.ac_entropy_clusters = ac_cost.cluster_count;
    ProfileEnd(
      profile, entropy_begin,
      &candidate_profile.entropy_optimization_nanoseconds);

    const ProfileClock::time_point sections_begin = ProfileBegin(profile);
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
    const size_t ac_global_index = 1 + dc_groups.size();
    if (Status status = WriteSimpleAcGlobal(
          ac_groups.size(), ac_code, &sections[ac_global_index]);
        !status.ok()) {
      return status;
    }
    const size_t ac_group_start = ac_global_index + 1;
    status = RunParallelSections(
      dc_groups.size() + ac_groups.size(),
      [&](size_t index) {
        if (index < dc_groups.size()) {
          return WriteDcGroupSection(
            dc_groups[index], dc_streams[2 * index],
            dc_streams[2 * index + 1], dc_code, &sections[1 + index]);
        }
        const size_t ac_index = index - dc_groups.size();
        return WriteTokenStream(
          ac_streams[ac_index], ac_code,
          &sections[ac_group_start + ac_index]);
      });
    if (!status.ok()) {
      return status;
    }
    ProfileEnd(
      profile, sections_begin, &candidate_profile.section_writing_nanoseconds);

    const ProfileClock::time_point assembly_begin = ProfileBegin(profile);
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
    ProfileEnd(
      profile, assembly_begin, &candidate_profile.assembly_nanoseconds);
    *output = std::move(candidate);
    if (profile != nullptr) {
      candidate_profile.total_nanoseconds = ElapsedNanoseconds(total_begin);
      *profile = candidate_profile;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

}  // namespace

Status EncodeVarDctCodestream(
  const VarDctEncoderFrame& frame, std::vector<uint8_t>* output) {

  return EncodeVarDctCodestreamImpl(frame, output, nullptr);
}

Status codestream_internal::EncodeVarDctCodestreamProfiled(
  const VarDctEncoderFrame& frame,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile) {

  if (profile == nullptr) {
    return Status::InvalidArgument("Codestream profile output is null");
  }
  return EncodeVarDctCodestreamImpl(frame, output, profile);
}

}  // namespace gjxl
