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
#include "codestream/coefficient_order.h"
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

struct AcEncodingCandidate {
  SimpleCoefficientOrders orders;
  std::vector<std::vector<EntropyToken>> streams;
  std::vector<EntropyToken> order_tokens;
  EntropyCode ac_code;
  EntropyCode order_code;
  EntropyCodeCost ac_cost;
  EntropyCodeCost order_cost;
};

Status MoveAcStreams(
  std::vector<SimpleAcGroupTokenStream>* groups,
  std::vector<std::vector<EntropyToken>>* streams) {

  if (groups == nullptr || streams == nullptr) {
    return Status::InvalidArgument("AC stream output is null");
  }
  try {
    std::vector<std::vector<EntropyToken>> candidate;
    candidate.reserve(groups->size());
    for (SimpleAcGroupTokenStream& group : *groups) {
      candidate.push_back(std::move(group.tokens));
    }
    *streams = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status OptimizeAcCandidate(AcEncodingCandidate* candidate) {
  if (candidate == nullptr || candidate->streams.empty()) {
    return Status::InvalidArgument("AC encoding candidate is empty");
  }
  Status status = OptimizeEntropyCode(
    candidate->streams, {.context_count = kSimpleAcContextCount},
    &candidate->ac_code, &candidate->ac_cost);
  if (!status.ok()) {
    return status;
  }
  if (candidate->orders.used_order_mask == 0) {
    if (!candidate->order_tokens.empty()) {
      return Status::Internal("Natural AC candidate retained order tokens");
    }
    return Status::Ok();
  }
  const std::span<const std::vector<EntropyToken>> order_streams(
    &candidate->order_tokens, 1);
  return OptimizeEntropyCode(
    order_streams,
    {
      .context_count = kSimplePermutationContextCount,
      .uint_config = {0, 0, 0},
    },
    &candidate->order_code, &candidate->order_cost);
}

Status WriteCommonSections(
  const VarDctEncoderFrame& frame,
  std::span<const SimpleDcGroupTokenStreams> dc_groups,
  std::span<const std::vector<EntropyToken>> dc_streams,
  const EntropyCode& dc_code,
  std::vector<BitWriter>* sections) {

  if (sections == nullptr || dc_groups.empty() ||
      dc_streams.size() != 2 * dc_groups.size()) {
    return Status::InvalidArgument("Common codestream sections are invalid");
  }
  try {
    std::vector<BitWriter> candidate(1 + dc_groups.size());
    Status status = WriteSimpleDcGlobal(
      frame.quantizer().params(), dc_groups.size(), dc_code, &candidate[0]);
    if (!status.ok()) {
      return status;
    }
    status = RunParallelSections(
      dc_groups.size(),
      [&](size_t index) {
        return WriteDcGroupSection(
          dc_groups[index], dc_streams[2 * index],
          dc_streams[2 * index + 1], dc_code, &candidate[1 + index]);
      });
    if (!status.ok()) {
      return status;
    }
    *sections = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status WriteAcSections(
  const AcEncodingCandidate& ac,
  std::vector<BitWriter>* sections) {

  if (sections == nullptr || ac.streams.empty()) {
    return Status::InvalidArgument("AC codestream sections are invalid");
  }
  try {
    std::vector<BitWriter> candidate(1 + ac.streams.size());
    Status status = WriteSimpleAcGlobal(
      ac.streams.size(), ac.orders.used_order_mask, ac.order_tokens,
      ac.orders.used_order_mask == 0 ? nullptr : &ac.order_code,
      ac.ac_code, &candidate[0]);
    if (!status.ok()) {
      return status;
    }
    status = RunParallelSections(
      ac.streams.size(),
      [&](size_t index) {
        return WriteTokenStream(
          ac.streams[index], ac.ac_code, &candidate[1 + index]);
      });
    if (!status.ok()) {
      return status;
    }
    *sections = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status PhysicalSectionSizes(
  std::span<const BitWriter> common_sections,
  std::span<const BitWriter> ac_sections,
  size_t ac_group_count,
  std::vector<size_t>* sizes) {

  if (sizes == nullptr || common_sections.empty() || ac_sections.empty() ||
      ac_group_count == std::numeric_limits<size_t>::max() ||
      ac_sections.size() != ac_group_count + 1) {
    return Status::InvalidArgument("Frame-section dimensions are invalid");
  }
  sizes->clear();
  if (ac_group_count == 1) {
    if (common_sections.size() != 2 || ac_sections.size() != 2) {
      return Status::Internal(
        "Single-group frame has an invalid section count");
    }
    size_t combined_bits = 0;
    for (const BitWriter& section : common_sections) {
      if (combined_bits >
          std::numeric_limits<size_t>::max() - section.bits_written()) {
        return AllocationFailure();
      }
      combined_bits += section.bits_written();
    }
    for (const BitWriter& section : ac_sections) {
      if (combined_bits >
          std::numeric_limits<size_t>::max() - section.bits_written()) {
        return AllocationFailure();
      }
      combined_bits += section.bits_written();
    }
    if (combined_bits > std::numeric_limits<size_t>::max() - 7) {
      return AllocationFailure();
    }
    sizes->push_back((combined_bits + 7) / 8);
    return Status::Ok();
  }

  try {
    if (common_sections.size() > sizes->max_size() - ac_sections.size()) {
      return AllocationFailure();
    }
    sizes->reserve(common_sections.size() + ac_sections.size());
    for (const BitWriter& section : common_sections) {
      sizes->push_back(section.padded_size());
    }
    for (const BitWriter& section : ac_sections) {
      sizes->push_back(section.padded_size());
    }
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status WriteFramePrefix(
  const VarDctEncoderFrame& frame,
  BitWriter* writer) {

  Status status = WriteSimpleCodestreamHeader(
    frame.geometry().frame(), writer);
  if (!status.ok()) {
    return status;
  }
  return WriteSimpleFrameHeader(frame.profile(), writer);
}

Status MeasureCandidateSize(
  const VarDctEncoderFrame& frame,
  std::span<const BitWriter> common_sections,
  std::span<const BitWriter> ac_sections,
  size_t ac_group_count,
  size_t* size) {

  if (size == nullptr) {
    return Status::InvalidArgument("Codestream candidate size output is null");
  }
  std::vector<size_t> section_sizes;
  Status status = PhysicalSectionSizes(
    common_sections, ac_sections, ac_group_count, &section_sizes);
  if (!status.ok()) {
    return status;
  }
  BitWriter prefix;
  status = WriteFramePrefix(frame, &prefix);
  if (!status.ok()) {
    return status;
  }
  status = WriteTocSizes(section_sizes, &prefix);
  if (!status.ok()) {
    return status;
  }
  size_t candidate = prefix.padded_size();
  for (size_t section_size : section_sizes) {
    if (candidate > std::numeric_limits<size_t>::max() - section_size) {
      return AllocationFailure();
    }
    candidate += section_size;
  }
  *size = candidate;
  return Status::Ok();
}

Status AssembleCandidate(
  const VarDctEncoderFrame& frame,
  std::span<const BitWriter> common_sections,
  std::span<const BitWriter> ac_sections,
  size_t ac_group_count,
  std::vector<uint8_t>* output) {

  if (output == nullptr) {
    return Status::InvalidArgument("Codestream candidate output is null");
  }
  try {
    std::vector<size_t> section_sizes;
    Status status = PhysicalSectionSizes(
      common_sections, ac_sections, ac_group_count, &section_sizes);
    if (!status.ok()) {
      return status;
    }
    BitWriter writer;
    status = WriteFramePrefix(frame, &writer);
    if (!status.ok()) {
      return status;
    }
    if (ac_group_count == 1) {
      BitWriter collapsed;
      for (const BitWriter& section : common_sections) {
        if (Status append = collapsed.Append(section); !append.ok()) {
          return append;
        }
      }
      for (const BitWriter& section : ac_sections) {
        if (Status append = collapsed.Append(section); !append.ok()) {
          return append;
        }
      }
      status = WriteTocAndSections(
        std::span<const BitWriter>(&collapsed, 1), &writer);
    } else {
      status = WriteTocSizes(section_sizes, &writer);
      if (status.ok()) {
        status = ConcatenateSections(common_sections, &writer);
      }
      if (status.ok()) {
        status = ConcatenateSections(ac_sections, &writer);
      }
    }
    if (!status.ok()) {
      return status;
    }
    if (!writer.byte_aligned()) {
      return Status::Internal("Assembled codestream is not byte-aligned");
    }
    const std::span<const uint8_t> bytes = writer.padded_bytes();
    std::vector<uint8_t> candidate(bytes.begin(), bytes.end());
    *output = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
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

    if (dc_groups.empty()) {
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

    const ProfileClock::time_point ac_tokenization_begin = ProfileBegin(profile);
    AcEncodingCandidate natural;
    std::vector<SimpleAcGroupTokenStream> natural_groups;
    AcEncodingCandidate custom;
    status = RunParallelSections(
      2,
      [&](size_t index) {
        if (index == 0) {
          Status token_status =
            TokenizeSimpleAcGroups(frame, &natural_groups);
          if (!token_status.ok()) {
            return token_status;
          }
          return MoveAcStreams(&natural_groups, &natural.streams);
        }
        return ComputeSimpleCoefficientOrders(frame, &custom.orders);
      });
    if (!status.ok()) {
      return status;
    }
    if (custom.orders.used_order_mask != 0) {
      std::vector<SimpleAcGroupTokenStream> custom_groups;
      status = TokenizeSimpleAcGroups(frame, custom.orders, &custom_groups);
      if (!status.ok()) {
        return status;
      }
      status = MoveAcStreams(&custom_groups, &custom.streams);
      if (!status.ok()) {
        return status;
      }
      status = TokenizeSimpleCoefficientOrders(
        custom.orders, &custom.order_tokens);
      if (!status.ok()) {
        return status;
      }
    }
    ProfileEnd(
      profile, ac_tokenization_begin,
      &candidate_profile.ac_tokenization_nanoseconds);
    if (natural.streams.empty()) {
      return Status::Internal("Validated frame produced no AC groups");
    }

    const ProfileClock::time_point entropy_begin = ProfileBegin(profile);
    EntropyCode dc_code;
    EntropyCodeCost dc_cost;
    const size_t entropy_task_count =
      custom.orders.used_order_mask == 0 ? 2 : 3;
    status = RunParallelSections(
      entropy_task_count,
      [&](size_t index) {
        if (index == 0) {
          return OptimizeEntropyCode(
            dc_streams, {.context_count = kSimpleDcContextCount}, &dc_code,
            &dc_cost);
        }
        return index == 1
          ? OptimizeAcCandidate(&natural)
          : OptimizeAcCandidate(&custom);
      });
    if (!status.ok()) {
      return status;
    }
    ProfileEnd(
      profile, entropy_begin,
      &candidate_profile.entropy_optimization_nanoseconds);

    const ProfileClock::time_point sections_begin = ProfileBegin(profile);
    std::vector<BitWriter> common_sections;
    status = WriteCommonSections(
      frame, dc_groups, dc_streams, dc_code, &common_sections);
    if (!status.ok()) {
      return status;
    }
    std::vector<BitWriter> natural_ac_sections;
    status = WriteAcSections(natural, &natural_ac_sections);
    if (!status.ok()) {
      return status;
    }
    std::vector<BitWriter> custom_ac_sections;
    if (custom.orders.used_order_mask != 0) {
      status = WriteAcSections(custom, &custom_ac_sections);
      if (!status.ok()) {
        return status;
      }
    }
    ProfileEnd(
      profile, sections_begin, &candidate_profile.section_writing_nanoseconds);

    const ProfileClock::time_point assembly_begin = ProfileBegin(profile);
    size_t natural_size = 0;
    status = MeasureCandidateSize(
      frame, common_sections, natural_ac_sections, natural.streams.size(),
      &natural_size);
    if (!status.ok()) {
      return status;
    }
    size_t custom_size = 0;
    if (custom.orders.used_order_mask != 0) {
      status = MeasureCandidateSize(
        frame, common_sections, custom_ac_sections, custom.streams.size(),
        &custom_size);
      if (!status.ok()) {
        return status;
      }
    }
    const bool use_custom =
      custom_size != 0 && custom_size < natural_size;
    const AcEncodingCandidate& selected = use_custom ? custom : natural;
    const std::vector<BitWriter>& selected_ac_sections =
      use_custom ? custom_ac_sections : natural_ac_sections;
    std::vector<uint8_t> candidate_output;
    status = AssembleCandidate(
      frame, common_sections, selected_ac_sections, selected.streams.size(),
      &candidate_output);
    if (!status.ok()) {
      return status;
    }
    if (candidate_output.size() != (use_custom ? custom_size : natural_size)) {
      return Status::Internal(
        "Measured codestream candidate size differs from assembly");
    }
    ProfileEnd(
      profile, assembly_begin, &candidate_profile.assembly_nanoseconds);

    candidate_profile.natural_candidate_bytes = natural_size;
    candidate_profile.custom_order_candidate_bytes = custom_size;
    candidate_profile.selected_coefficient_order_mask =
      selected.orders.used_order_mask;
    uint64_t model_bits = dc_cost.model_bits;
    uint64_t token_bits = dc_cost.token_bits;
    const auto add_cost = [&](const EntropyCodeCost& cost) {
      if (model_bits > std::numeric_limits<uint64_t>::max() - cost.model_bits ||
          token_bits > std::numeric_limits<uint64_t>::max() - cost.token_bits) {
        return false;
      }
      model_bits += cost.model_bits;
      token_bits += cost.token_bits;
      return true;
    };
    if (!add_cost(selected.ac_cost) || !add_cost(selected.order_cost)) {
      return Status::InvalidArgument("Entropy profile bit count overflow");
    }
    candidate_profile.entropy_model_bits = model_bits;
    candidate_profile.entropy_token_bits = token_bits;
    candidate_profile.dc_entropy_clusters = dc_cost.cluster_count;
    candidate_profile.ac_entropy_clusters = selected.ac_cost.cluster_count;
    *output = std::move(candidate_output);
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
