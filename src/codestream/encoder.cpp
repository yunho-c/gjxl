// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder/enc_frame.cc.

#include "codestream/encoder.h"

#include <algorithm>
#include <array>
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
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/dc_group.h"
#include "codestream/encoder_internal.h"
#include "codestream/entropy.h"
#include "codestream/entropy_internal.h"
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

ProfileClock::time_point WorkBegin(bool enabled) {
  return enabled ? ProfileClock::now() : ProfileClock::time_point{};
}

void WorkEnd(
  bool enabled,
  ProfileClock::time_point begin,
  uint64_t* destination) {

  if (enabled) {
    *destination += ElapsedNanoseconds(begin);
  }
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
  BitWriter* writer,
  codestream_internal::SectionWritingWorkProfile* profile) {

  const ProfileClock::time_point dc_header_begin =
    WorkBegin(profile != nullptr);
  if (Status status = WriteSimpleDcGroupModularHeader(writer); !status.ok()) {
    return status;
  }
  WorkEnd(
    profile != nullptr, dc_header_begin,
    profile == nullptr ? nullptr : &profile->model_and_header_nanoseconds);
  const ProfileClock::time_point dc_tokens_begin =
    WorkBegin(profile != nullptr);
  if (Status status = WriteTokenStream(dc_tokens, code, writer); !status.ok()) {
    return status;
  }
  WorkEnd(
    profile != nullptr, dc_tokens_begin,
    profile == nullptr ? nullptr : &profile->token_write_nanoseconds);
  const ProfileClock::time_point metadata_header_begin =
    WorkBegin(profile != nullptr);
  if (Status status = WriteSimpleAcMetadataModularHeader(
        group.block_extent, group.transform_anchor_count, writer);
      !status.ok()) {
    return status;
  }
  WorkEnd(
    profile != nullptr, metadata_header_begin,
    profile == nullptr ? nullptr : &profile->model_and_header_nanoseconds);
  const ProfileClock::time_point metadata_tokens_begin =
    WorkBegin(profile != nullptr);
  Status status = WriteTokenStream(metadata_tokens, code, writer);
  WorkEnd(
    profile != nullptr, metadata_tokens_begin,
    profile == nullptr ? nullptr : &profile->token_write_nanoseconds);
  return status;
}

struct AcEncodingCandidate {
  size_t block_context_candidate_index = 0;
  SimpleBlockContextMap block_context_map;
  bool custom_order = false;
  std::vector<std::vector<EntropyToken>> streams;
  EntropyCode ac_code;
  EntropyCodeCost ac_cost;
  EntropyCode prefix_ac_code;
  EntropyCodeCost prefix_ac_cost;
  size_t complete_size = 0;
  bool all_prefix_entropy = false;
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

Status OptimizeBestEntropyCode(
  std::span<const std::vector<EntropyToken>> streams,
  const EntropyCodeOptions& options,
  EntropyCode* code,
  EntropyCodeCost* cost,
  EntropyCode* prefix_fallback,
  EntropyCodeCost* prefix_fallback_cost,
  codestream_internal::EntropyWorkProfile* profile) {

  if (code == nullptr || cost == nullptr || prefix_fallback == nullptr ||
      prefix_fallback_cost == nullptr) {
    return Status::InvalidArgument("Entropy selection output is null");
  }
  EntropyCode prefix;
  EntropyCodeCost prefix_cost;
  if (Status status = OptimizeEntropyCode(
        streams, options, &prefix, &prefix_cost, profile);
      !status.ok()) {
    return status;
  }
  EntropyCode ans;
  EntropyCodeCost ans_cost;
  if (Status status = OptimizeAnsEntropyCode(
        streams, prefix, &ans, &ans_cost, profile);
      !status.ok()) {
    return status;
  }
  const ProfileClock::time_point selection_begin =
    WorkBegin(profile != nullptr);
  *prefix_fallback = prefix;
  *prefix_fallback_cost = prefix_cost;
  const auto total_bits = [](const EntropyCodeCost& candidate) {
    return candidate.model_bits >
        std::numeric_limits<uint64_t>::max() - candidate.token_bits
      ? std::numeric_limits<uint64_t>::max()
      : candidate.model_bits + candidate.token_bits;
  };
  if (total_bits(ans_cost) < total_bits(prefix_cost)) {
    *code = std::move(ans);
    *cost = ans_cost;
  } else {
    *code = std::move(prefix);
    *cost = prefix_cost;
  }
  WorkEnd(
    profile != nullptr, selection_begin,
    profile == nullptr ? nullptr : &profile->selection_nanoseconds);
  return Status::Ok();
}

Status OptimizeAcCandidate(
  AcEncodingCandidate* candidate,
  codestream_internal::EntropyWorkProfile* profile) {
  if (candidate == nullptr || candidate->streams.empty()) {
    return Status::InvalidArgument("AC encoding candidate is empty");
  }
  Status status = OptimizeBestEntropyCode(
    candidate->streams,
    {
      .context_count = static_cast<uint32_t>(
        candidate->block_context_map.ac_context_count()),
    },
    &candidate->ac_code, &candidate->ac_cost,
    &candidate->prefix_ac_code, &candidate->prefix_ac_cost, profile);
  return status;
}

Status WriteCommonSections(
  const VarDctEncoderFrame& frame,
  std::span<const SimpleDcGroupTokenStreams> dc_groups,
  std::span<const std::vector<EntropyToken>> dc_streams,
  const SimpleBlockContextMap& block_context_map,
  const EntropyCode& dc_code,
  std::vector<BitWriter>* sections,
  codestream_internal::SectionWritingWorkProfile* profile) {

  if (sections == nullptr || dc_groups.empty() ||
      dc_streams.size() != 2 * dc_groups.size()) {
    return Status::InvalidArgument("Common codestream sections are invalid");
  }
  try {
    std::vector<BitWriter> candidate(1 + dc_groups.size());
    const ProfileClock::time_point global_begin =
      WorkBegin(profile != nullptr);
    Status status = WriteSimpleDcGlobal(
      frame.quantizer().params(), dc_groups.size(), block_context_map,
      dc_code, &candidate[0]);
    if (!status.ok()) {
      return status;
    }
    WorkEnd(
      profile != nullptr, global_begin,
      profile == nullptr ? nullptr : &profile->model_and_header_nanoseconds);
    std::vector<codestream_internal::SectionWritingWorkProfile>
      group_profiles(profile == nullptr ? 0 : dc_groups.size());
    status = RunParallelSections(
      dc_groups.size(),
      [&](size_t index) {
        return WriteDcGroupSection(
          dc_groups[index], dc_streams[2 * index],
          dc_streams[2 * index + 1], dc_code, &candidate[1 + index],
          profile == nullptr ? nullptr : &group_profiles[index]);
      });
    if (!status.ok()) {
      return status;
    }
    for (const auto& group_profile : group_profiles) {
      codestream_internal::AccumulateSectionWritingWorkProfile(
        group_profile, profile);
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
  const EntropyCode& ac_code,
  const SimpleCoefficientOrders& custom_orders,
  std::span<const EntropyToken> order_tokens,
  const EntropyCode* order_code,
  std::vector<BitWriter>* sections,
  codestream_internal::SectionWritingWorkProfile* profile) {

  if (sections == nullptr || ac.streams.empty()) {
    return Status::InvalidArgument("AC codestream sections are invalid");
  }
  try {
    std::vector<BitWriter> candidate(1 + ac.streams.size());
    const uint16_t used_order_mask =
      ac.custom_order ? custom_orders.used_order_mask : 0;
    const ProfileClock::time_point global_begin =
      WorkBegin(profile != nullptr);
    Status status = WriteSimpleAcGlobal(
      ac.streams.size(), used_order_mask,
      ac.custom_order ? order_tokens : std::span<const EntropyToken>{},
      ac.custom_order ? order_code : nullptr,
      ac_code, &candidate[0]);
    if (!status.ok()) {
      return status;
    }
    WorkEnd(
      profile != nullptr, global_begin,
      profile == nullptr ? nullptr : &profile->model_and_header_nanoseconds);
    std::vector<codestream_internal::SectionWritingWorkProfile>
      group_profiles(profile == nullptr ? 0 : ac.streams.size());
    status = RunParallelSections(
      ac.streams.size(),
      [&](size_t index) {
        auto* group_profile =
          profile == nullptr ? nullptr : &group_profiles[index];
        const ProfileClock::time_point tokens_begin =
          WorkBegin(group_profile != nullptr);
        Status token_status = WriteTokenStream(
          ac.streams[index], ac_code, &candidate[1 + index]);
        WorkEnd(
          group_profile != nullptr, tokens_begin,
          group_profile == nullptr
            ? nullptr
            : &group_profile->token_write_nanoseconds);
        return token_status;
      });
    if (!status.ok()) {
      return status;
    }
    for (const auto& group_profile : group_profiles) {
      codestream_internal::AccumulateSectionWritingWorkProfile(
        group_profile, profile);
    }
    *sections = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

bool AddMeasuredBits(uint64_t value, uint64_t* total) {
  if (total == nullptr ||
      *total > std::numeric_limits<uint64_t>::max() - value) {
    return false;
  }
  *total += value;
  return true;
}

Status MeasureDcGroupSections(
  std::span<const SimpleDcGroupTokenStreams> dc_groups,
  std::span<const std::vector<EntropyToken>> dc_streams,
  const EntropyCode& dc_code,
  std::vector<uint64_t>* section_bits,
  uint64_t* measurement_work) {

  if (section_bits == nullptr || dc_groups.empty() ||
      dc_streams.size() != 2 * dc_groups.size()) {
    return Status::InvalidArgument("Common section measurement is invalid");
  }
  try {
    std::vector<uint64_t> candidate(dc_groups.size());
    std::vector<uint64_t> group_work(
      measurement_work == nullptr ? 0 : dc_groups.size());
    Status status = RunParallelSections(
      dc_groups.size(),
      [&](size_t index) {
        const ProfileClock::time_point work_begin =
          WorkBegin(measurement_work != nullptr);
        BitWriter headers;
        if (Status header_status = WriteSimpleDcGroupModularHeader(&headers);
            !header_status.ok()) {
          return header_status;
        }
        if (Status header_status = WriteSimpleAcMetadataModularHeader(
              dc_groups[index].block_extent,
              dc_groups[index].transform_anchor_count, &headers);
            !header_status.ok()) {
          return header_status;
        }
        uint64_t bits = headers.bits_written();
        uint64_t tokens = 0;
        if (Status token_status = codestream_internal::CountTokenStreamBits(
              dc_streams[2 * index], dc_code, &tokens);
            !token_status.ok()) {
          return token_status;
        }
        if (!AddMeasuredBits(tokens, &bits)) {
          return Status::InvalidArgument("DC section bit count overflow");
        }
        if (Status token_status = codestream_internal::CountTokenStreamBits(
              dc_streams[2 * index + 1], dc_code, &tokens);
            !token_status.ok()) {
          return token_status;
        }
        if (!AddMeasuredBits(tokens, &bits)) {
          return Status::InvalidArgument("DC section bit count overflow");
        }
        candidate[index] = bits;
        WorkEnd(
          measurement_work != nullptr, work_begin,
          measurement_work == nullptr ? nullptr : &group_work[index]);
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }
    uint64_t work = 0;
    for (uint64_t group_nanoseconds : group_work) {
      if (!AddMeasuredBits(group_nanoseconds, &work)) {
        return Status::Internal("DC section measurement profile overflow");
      }
    }
    *section_bits = std::move(candidate);
    if (measurement_work != nullptr) {
      *measurement_work = work;
    }
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status MeasureCommonSections(
  const VarDctEncoderFrame& frame,
  size_t dc_group_count,
  const SimpleBlockContextMap& block_context_map,
  const EntropyCode& dc_code,
  std::span<const uint64_t> dc_group_section_bits,
  std::vector<uint64_t>* section_bits) {

  if (section_bits == nullptr || dc_group_count == 0 ||
      dc_group_section_bits.size() != dc_group_count) {
    return Status::InvalidArgument("Common section measurement is invalid");
  }
  try {
    BitWriter global;
    if (Status status = WriteSimpleDcGlobal(
          frame.quantizer().params(), dc_group_count, block_context_map,
          dc_code, &global);
        !status.ok()) {
      return status;
    }
    std::vector<uint64_t> candidate;
    candidate.reserve(1 + dc_group_section_bits.size());
    candidate.push_back(global.bits_written());
    candidate.insert(
      candidate.end(), dc_group_section_bits.begin(),
      dc_group_section_bits.end());
    *section_bits = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status MeasureAcGlobalBits(
  size_t ac_group_count,
  uint16_t used_order_mask,
  const EntropyCodeCost* order_cost,
  const EntropyCodeCost& ac_cost,
  uint64_t* bit_count) {

  if (bit_count == nullptr || ac_group_count == 0 ||
      ((used_order_mask == 0) != (order_cost == nullptr))) {
    return Status::InvalidArgument("AC-global measurement is invalid");
  }
  const size_t histogram_bits = std::bit_width(ac_group_count - 1);
  if (histogram_bits > BitWriter::kMaxBitsPerWrite ||
      used_order_mask >=
        (uint16_t{1} << codestream_internal::kSimpleCoefficientOrderCount)) {
    return Status::InvalidArgument("AC-global state cannot be encoded");
  }
  const uint64_t order_mask_bits =
    used_order_mask == 0 || used_order_mask == 0x5F ||
        used_order_mask == 0x13
      ? 2
      : 15;
  uint64_t candidate = 1 + histogram_bits + order_mask_bits;
  if (order_cost != nullptr &&
      (!AddMeasuredBits(1, &candidate) ||
       !AddMeasuredBits(order_cost->model_bits, &candidate) ||
       !AddMeasuredBits(order_cost->token_bits, &candidate))) {
    return Status::InvalidArgument("AC-global bit count overflow");
  }
  if (!AddMeasuredBits(1, &candidate) ||
      !AddMeasuredBits(ac_cost.model_bits, &candidate)) {
    return Status::InvalidArgument("AC-global bit count overflow");
  }
  *bit_count = candidate;
  return Status::Ok();
}

Status MeasureAcSections(
  const AcEncodingCandidate& ac,
  const EntropyCode& ac_code,
  const EntropyCodeCost& ac_cost,
  const SimpleCoefficientOrders& custom_orders,
  const EntropyCodeCost* order_cost,
  std::vector<uint64_t>* section_bits,
  uint64_t* measurement_work) {

  if (section_bits == nullptr || ac.streams.empty() ||
      (ac.custom_order != (order_cost != nullptr))) {
    return Status::InvalidArgument("AC section measurement is invalid");
  }
  try {
    std::vector<uint64_t> candidate(1 + ac.streams.size());
    const uint16_t used_order_mask =
      ac.custom_order ? custom_orders.used_order_mask : 0;
    const ProfileClock::time_point global_begin =
      WorkBegin(measurement_work != nullptr);
    if (Status status = MeasureAcGlobalBits(
          ac.streams.size(), used_order_mask, order_cost, ac_cost,
          &candidate[0]);
        !status.ok()) {
      return status;
    }
    uint64_t global_work = 0;
    WorkEnd(
      measurement_work != nullptr, global_begin,
      measurement_work == nullptr ? nullptr : &global_work);
    std::vector<uint64_t> group_work(
      measurement_work == nullptr ? 0 : ac.streams.size());
    Status status = RunParallelSections(
      ac.streams.size(),
      [&](size_t index) {
        const ProfileClock::time_point work_begin =
          WorkBegin(measurement_work != nullptr);
        Status count_status = codestream_internal::CountTokenStreamBits(
          ac.streams[index], ac_code, &candidate[1 + index]);
        WorkEnd(
          measurement_work != nullptr, work_begin,
          measurement_work == nullptr ? nullptr : &group_work[index]);
        return count_status;
      });
    if (!status.ok()) {
      return status;
    }
    uint64_t work = global_work;
    for (uint64_t group_nanoseconds : group_work) {
      if (!AddMeasuredBits(group_nanoseconds, &work)) {
        return Status::Internal("AC section measurement profile overflow");
      }
    }
    *section_bits = std::move(candidate);
    if (measurement_work != nullptr) {
      *measurement_work = work;
    }
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
  std::span<const uint64_t> common_section_bits,
  std::span<const uint64_t> ac_section_bits,
  size_t ac_group_count,
  size_t* size) {

  if (size == nullptr) {
    return Status::InvalidArgument("Codestream candidate size output is null");
  }
  std::vector<size_t> section_sizes;
  Status status = codestream_internal::PhysicalSectionSizesFromBitCounts(
    common_section_bits, ac_section_bits, ac_group_count, &section_sizes);
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
  std::vector<uint8_t>* output,
  codestream_internal::AssemblyProfile* profile) {

  if (output == nullptr) {
    return Status::InvalidArgument("Codestream candidate output is null");
  }
  try {
    std::vector<size_t> section_sizes;
    const ProfileClock::time_point section_size_begin =
      WorkBegin(profile != nullptr);
    Status status = PhysicalSectionSizes(
      common_sections, ac_sections, ac_group_count, &section_sizes);
    if (!status.ok()) {
      return status;
    }
    WorkEnd(
      profile != nullptr, section_size_begin,
      profile == nullptr ? nullptr : &profile->section_size_nanoseconds);
    BitWriter writer;
    const ProfileClock::time_point frame_header_begin =
      WorkBegin(profile != nullptr);
    status = WriteFramePrefix(frame, &writer);
    if (!status.ok()) {
      return status;
    }
    WorkEnd(
      profile != nullptr, frame_header_begin,
      profile == nullptr ? nullptr : &profile->frame_header_nanoseconds);
    const ProfileClock::time_point toc_and_sections_begin =
      WorkBegin(profile != nullptr);
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
    WorkEnd(
      profile != nullptr, toc_and_sections_begin,
      profile == nullptr ? nullptr : &profile->toc_and_sections_nanoseconds);
    if (!writer.byte_aligned()) {
      return Status::Internal("Assembled codestream is not byte-aligned");
    }
    const ProfileClock::time_point output_copy_begin =
      WorkBegin(profile != nullptr);
    const std::span<const uint8_t> bytes = writer.padded_bytes();
    std::vector<uint8_t> candidate(bytes.begin(), bytes.end());
    *output = std::move(candidate);
    WorkEnd(
      profile != nullptr, output_copy_begin,
      profile == nullptr ? nullptr : &profile->output_copy_nanoseconds);
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
    std::vector<SimpleBlockContextMap> block_context_maps;
    SimpleCoefficientOrders custom_orders;
    std::array<uint64_t, 2> preparation_work{};
    status = RunParallelSections(
      2,
      [&](size_t index) {
        const ProfileClock::time_point work_begin =
          WorkBegin(profile != nullptr);
        Status work_status = index == 0
          ? ComputeSimpleBlockContextMapCandidates(frame, &block_context_maps)
          : ComputeSimpleCoefficientOrders(frame, &custom_orders);
        WorkEnd(
          profile != nullptr, work_begin, &preparation_work[index]);
        return work_status;
      });
    if (!status.ok()) {
      return status;
    }
    if (block_context_maps.empty()) {
      return Status::Internal(
        "Validated frame produced no block-context candidates");
    }
    candidate_profile.block_context_map_work_nanoseconds =
      preparation_work[0];
    candidate_profile.coefficient_order_work_nanoseconds =
      preparation_work[1];

    std::vector<EntropyToken> order_tokens;
    if (custom_orders.used_order_mask != 0) {
      const ProfileClock::time_point order_tokenization_begin =
        WorkBegin(profile != nullptr);
      status = TokenizeSimpleCoefficientOrders(custom_orders, &order_tokens);
      WorkEnd(
        profile != nullptr, order_tokenization_begin,
        &candidate_profile.coefficient_order_work_nanoseconds);
      if (!status.ok()) {
        return status;
      }
    }
    const bool has_custom_orders = custom_orders.used_order_mask != 0;
    const size_t candidates_per_map = has_custom_orders ? 2 : 1;
    if (block_context_maps.size() >
        std::numeric_limits<size_t>::max() / candidates_per_map) {
      return AllocationFailure();
    }
    std::vector<AcEncodingCandidate> candidates;
    candidates.reserve(block_context_maps.size() * candidates_per_map);
    for (size_t map_index = 0; map_index < block_context_maps.size();
         ++map_index) {
      candidates.push_back({
        .block_context_candidate_index = map_index,
        .block_context_map = block_context_maps[map_index],
      });
      if (has_custom_orders) {
        candidates.push_back({
          .block_context_candidate_index = map_index,
          .block_context_map = block_context_maps[map_index],
          .custom_order = true,
        });
      }
    }
    const SimpleCoefficientOrders natural_orders;
    std::vector<uint64_t> coefficient_tokenization_work(
      profile == nullptr ? 0 : candidates.size());
    status = RunParallelSections(
      candidates.size(),
      [&](size_t index) {
        const ProfileClock::time_point work_begin =
          WorkBegin(profile != nullptr);
        AcEncodingCandidate& candidate = candidates[index];
        std::vector<SimpleAcGroupTokenStream> groups;
        Status token_status = TokenizeSimpleAcGroups(
          frame, candidate.custom_order ? custom_orders : natural_orders,
          candidate.block_context_map, &groups);
        if (!token_status.ok()) {
          return token_status;
        }
        token_status = MoveAcStreams(&groups, &candidate.streams);
        WorkEnd(
          profile != nullptr, work_begin,
          profile == nullptr
            ? nullptr
            : &coefficient_tokenization_work[index]);
        return token_status;
      });
    if (!status.ok()) {
      return status;
    }
    for (uint64_t work : coefficient_tokenization_work) {
      candidate_profile.coefficient_tokenization_work_nanoseconds += work;
    }
    ProfileEnd(
      profile, ac_tokenization_begin,
      &candidate_profile.ac_tokenization_nanoseconds);
    if (candidates.empty() || candidates.front().streams.empty()) {
      return Status::Internal("Validated frame produced no AC candidates");
    }

    const ProfileClock::time_point entropy_begin = ProfileBegin(profile);
    EntropyCode dc_code;
    EntropyCodeCost dc_cost;
    EntropyCode prefix_dc_code;
    EntropyCodeCost prefix_dc_cost;
    EntropyCode order_code;
    EntropyCodeCost order_cost;
    EntropyCode prefix_order_code;
    EntropyCodeCost prefix_order_cost;
    const size_t order_task_count = has_custom_orders ? 1 : 0;
    const size_t entropy_task_count = 1 + order_task_count + candidates.size();
    std::vector<codestream_internal::EntropyWorkProfile> entropy_profiles(
      profile == nullptr ? 0 : entropy_task_count);
    status = RunParallelSections(
      entropy_task_count,
      [&](size_t index) {
        auto* entropy_profile =
          profile == nullptr ? nullptr : &entropy_profiles[index];
        if (index == 0) {
          return OptimizeBestEntropyCode(
            dc_streams, {.context_count = kSimpleDcContextCount}, &dc_code,
            &dc_cost, &prefix_dc_code, &prefix_dc_cost, entropy_profile);
        }
        if (has_custom_orders && index == 1) {
          const std::span<const std::vector<EntropyToken>> order_streams(
            &order_tokens, 1);
          return OptimizeBestEntropyCode(
            order_streams,
            {
              .context_count = kSimplePermutationContextCount,
              .uint_config = {0, 0, 0},
            },
            &order_code, &order_cost,
            &prefix_order_code, &prefix_order_cost, entropy_profile);
        }
        return OptimizeAcCandidate(
          &candidates[index - 1 - order_task_count], entropy_profile);
      });
    if (!status.ok()) {
      return status;
    }
    for (const auto& entropy_profile : entropy_profiles) {
      codestream_internal::AccumulateEntropyWorkProfile(
        entropy_profile, &candidate_profile.entropy_work);
    }
    ProfileEnd(
      profile, entropy_begin,
      &candidate_profile.entropy_optimization_nanoseconds);

    const ProfileClock::time_point section_measurement_begin =
      ProfileBegin(profile);
    constexpr size_t kMixedEntropy = 0;
    constexpr size_t kPrefixEntropy = 1;
    const std::array<const EntropyCode*, 2> dc_codes = {
      &dc_code, &prefix_dc_code};
    std::array<std::vector<uint64_t>, 2> dc_group_section_bits;
    std::array<uint64_t, 2> dc_group_measurement_work{};
    status = RunParallelSections(
      dc_codes.size(),
      [&](size_t mode) {
        Status measure_status = MeasureDcGroupSections(
          dc_groups, dc_streams, *dc_codes[mode],
          &dc_group_section_bits[mode],
          profile == nullptr ? nullptr : &dc_group_measurement_work[mode]);
        return measure_status;
      });
    if (!status.ok()) {
      return status;
    }

    std::vector<std::array<std::vector<uint64_t>, 2>> common_section_bits(
      block_context_maps.size());
    if (block_context_maps.size() >
        std::numeric_limits<size_t>::max() / dc_codes.size()) {
      return AllocationFailure();
    }
    const size_t common_measurement_count =
      block_context_maps.size() * dc_codes.size();
    std::vector<uint64_t> common_measurement_work(common_measurement_count);
    status = RunParallelSections(
      common_measurement_count,
      [&](size_t index) {
        const size_t map_index = index / dc_codes.size();
        const size_t mode = index % dc_codes.size();
        const ProfileClock::time_point work_begin =
          WorkBegin(profile != nullptr);
        Status measure_status = MeasureCommonSections(
          frame, dc_groups.size(), block_context_maps[map_index],
          *dc_codes[mode], dc_group_section_bits[mode],
          &common_section_bits[map_index][mode]);
        WorkEnd(
          profile != nullptr, work_begin,
          &common_measurement_work[index]);
        return measure_status;
      });
    if (!status.ok()) {
      return status;
    }

    std::vector<uint64_t> candidate_measurement_work(candidates.size());
    status = RunParallelSections(
      candidates.size(),
      [&](size_t index) {
        AcEncodingCandidate& candidate = candidates[index];
        const auto measure = [&](
          size_t mode, size_t* complete_size, uint64_t* measurement_work) {
          const bool all_prefix = mode == kPrefixEntropy;
          const EntropyCode& ac_code = all_prefix
            ? candidate.prefix_ac_code
            : candidate.ac_code;
          const EntropyCodeCost& ac_cost = all_prefix
            ? candidate.prefix_ac_cost
            : candidate.ac_cost;
          const EntropyCodeCost* selected_order_cost = candidate.custom_order
            ? (all_prefix ? &prefix_order_cost : &order_cost)
            : nullptr;
          std::vector<uint64_t> ac_section_bits;
          Status measure_status = MeasureAcSections(
            candidate, ac_code, ac_cost, custom_orders,
            selected_order_cost, &ac_section_bits, measurement_work);
          if (!measure_status.ok()) {
            return measure_status;
          }
          const ProfileClock::time_point size_begin =
            WorkBegin(measurement_work != nullptr);
          measure_status = MeasureCandidateSize(
            frame,
            common_section_bits[candidate.block_context_candidate_index][mode],
            ac_section_bits, candidate.streams.size(), complete_size);
          uint64_t size_work = 0;
          WorkEnd(
            measurement_work != nullptr, size_begin,
            measurement_work == nullptr ? nullptr : &size_work);
          if (measurement_work != nullptr &&
              !AddMeasuredBits(size_work, measurement_work)) {
            return Status::Internal(
              "Candidate size measurement profile overflow");
          }
          return measure_status;
        };

        size_t mixed_size = 0;
        uint64_t mixed_measurement_work = 0;
        Status measure_status = measure(
          kMixedEntropy, &mixed_size,
          profile == nullptr ? nullptr : &mixed_measurement_work);
        if (!measure_status.ok()) {
          return measure_status;
        }
        size_t prefix_size = 0;
        uint64_t prefix_measurement_work = 0;
        measure_status = measure(
          kPrefixEntropy, &prefix_size,
          profile == nullptr ? nullptr : &prefix_measurement_work);
        if (!measure_status.ok()) {
          return measure_status;
        }
        candidate.complete_size = mixed_size;
        if (codestream_internal::PreferAllPrefixCandidate(
              mixed_size, prefix_size)) {
          candidate.complete_size = prefix_size;
          candidate.all_prefix_entropy = true;
        }
        if (profile != nullptr && !AddMeasuredBits(
              prefix_measurement_work, &mixed_measurement_work)) {
          return Status::Internal(
            "Candidate measurement profile overflow");
        }
        candidate_measurement_work[index] = mixed_measurement_work;
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }
    if (profile != nullptr) {
      for (uint64_t work : dc_group_measurement_work) {
        candidate_profile.section_writing_work.candidate_measure_nanoseconds +=
          work;
      }
      for (uint64_t work : common_measurement_work) {
        candidate_profile.section_writing_work.candidate_measure_nanoseconds +=
          work;
      }
      for (uint64_t work : candidate_measurement_work) {
        candidate_profile.section_writing_work.candidate_measure_nanoseconds +=
          work;
      }
    }
    uint64_t section_measurement_nanoseconds = 0;
    ProfileEnd(
      profile, section_measurement_begin, &section_measurement_nanoseconds);

    const ProfileClock::time_point candidate_selection_begin =
      WorkBegin(profile != nullptr);
    size_t selected_index = 0;
    for (size_t index = 1; index < candidates.size(); ++index) {
      const AcEncodingCandidate& candidate = candidates[index];
      const AcEncodingCandidate& selected = candidates[selected_index];
      if (codestream_internal::PreferEncodingCandidate(
            {
              candidate.complete_size,
              candidate.custom_order,
              candidate.block_context_candidate_index,
            },
            {
              selected.complete_size,
              selected.custom_order,
              selected.block_context_candidate_index,
            })) {
        selected_index = index;
      }
    }
    WorkEnd(
      profile != nullptr, candidate_selection_begin,
      &candidate_profile.assembly.candidate_selection_nanoseconds);
    const AcEncodingCandidate& selected = candidates[selected_index];
    const EntropyCode& selected_dc_code = selected.all_prefix_entropy
      ? prefix_dc_code
      : dc_code;
    const EntropyCode& selected_ac_code = selected.all_prefix_entropy
      ? selected.prefix_ac_code
      : selected.ac_code;
    const EntropyCode* selected_order_code = selected.custom_order
      ? (selected.all_prefix_entropy ? &prefix_order_code : &order_code)
      : nullptr;
    const ProfileClock::time_point selected_write_begin = ProfileBegin(profile);
    std::vector<BitWriter> common_sections;
    codestream_internal::SectionWritingWorkProfile selected_write_profile;
    status = WriteCommonSections(
      frame, dc_groups, dc_streams, selected.block_context_map,
      selected_dc_code, &common_sections,
      profile == nullptr ? nullptr : &selected_write_profile);
    if (!status.ok()) {
      return status;
    }
    std::vector<BitWriter> ac_sections;
    status = WriteAcSections(
      selected, selected_ac_code, custom_orders, order_tokens,
      selected_order_code, &ac_sections,
      profile == nullptr ? nullptr : &selected_write_profile);
    if (!status.ok()) {
      return status;
    }
    uint64_t selected_write_nanoseconds = 0;
    ProfileEnd(profile, selected_write_begin, &selected_write_nanoseconds);
    codestream_internal::AccumulateSectionWritingWorkProfile(
      selected_write_profile, &candidate_profile.section_writing_work);
    if (section_measurement_nanoseconds >
        std::numeric_limits<uint64_t>::max() - selected_write_nanoseconds) {
      return Status::Internal("Codestream section profile overflow");
    }
    candidate_profile.section_writing_nanoseconds =
      section_measurement_nanoseconds + selected_write_nanoseconds;

    const ProfileClock::time_point assembly_begin = ProfileBegin(profile);
    std::vector<uint8_t> candidate_output;
    status = AssembleCandidate(
      frame, common_sections, ac_sections,
      selected.streams.size(), &candidate_output,
      profile == nullptr ? nullptr : &candidate_profile.assembly);
    if (!status.ok()) {
      return status;
    }
    if (candidate_output.size() != selected.complete_size) {
      return Status::Internal(
        "Measured codestream candidate size differs from assembly");
    }
    uint64_t assembly_write_nanoseconds = 0;
    ProfileEnd(profile, assembly_begin, &assembly_write_nanoseconds);
    if (candidate_profile.assembly.candidate_selection_nanoseconds >
        std::numeric_limits<uint64_t>::max() - assembly_write_nanoseconds) {
      return Status::Internal("Codestream assembly profile overflow");
    }
    candidate_profile.assembly_nanoseconds =
      candidate_profile.assembly.candidate_selection_nanoseconds +
      assembly_write_nanoseconds;

    candidate_profile.natural_candidate_bytes =
      std::numeric_limits<size_t>::max();
    for (const AcEncodingCandidate& candidate : candidates) {
      size_t& minimum = candidate.custom_order
        ? candidate_profile.custom_order_candidate_bytes
        : candidate_profile.natural_candidate_bytes;
      if (minimum == 0 || candidate.complete_size < minimum) {
        minimum = candidate.complete_size;
      }
    }
    if (candidate_profile.natural_candidate_bytes ==
        std::numeric_limits<size_t>::max()) {
      return Status::Internal("Natural codestream candidate is missing");
    }
    candidate_profile.selected_coefficient_order_mask =
      selected.custom_order ? custom_orders.used_order_mask : 0;
    candidate_profile.block_context_candidate_count = candidates_per_map == 0
      ? 0
      : candidates.size() / candidates_per_map;
    for (const AcEncodingCandidate& candidate : candidates) {
      if (candidate.block_context_candidate_index == 0 &&
          (candidate_profile.compact_block_context_candidate_bytes == 0 ||
           candidate.complete_size <
             candidate_profile.compact_block_context_candidate_bytes)) {
        candidate_profile.compact_block_context_candidate_bytes =
          candidate.complete_size;
      }
    }
    candidate_profile.selected_block_context_candidate_index =
      selected.block_context_candidate_index;
    candidate_profile.selected_block_context_count =
      selected.block_context_map.num_contexts;
    candidate_profile.selected_block_context_qf_threshold_count =
      selected.block_context_map.qf_thresholds.size();
    const EntropyCodeCost& selected_dc_cost = selected.all_prefix_entropy
      ? prefix_dc_cost
      : dc_cost;
    const EntropyCodeCost& selected_order_cost = selected.all_prefix_entropy
      ? prefix_order_cost
      : order_cost;
    const EntropyCodeCost& selected_ac_cost = selected.all_prefix_entropy
      ? selected.prefix_ac_cost
      : selected.ac_cost;
    uint64_t model_bits = selected_dc_cost.model_bits;
    uint64_t token_bits = selected_dc_cost.token_bits;
    const auto add_cost = [&](const EntropyCodeCost& cost) {
      if (model_bits > std::numeric_limits<uint64_t>::max() - cost.model_bits ||
          token_bits > std::numeric_limits<uint64_t>::max() - cost.token_bits) {
        return false;
      }
      model_bits += cost.model_bits;
      token_bits += cost.token_bits;
      return true;
    };
    if (!add_cost(selected_ac_cost) ||
        (selected.custom_order && !add_cost(selected_order_cost))) {
      return Status::InvalidArgument("Entropy profile bit count overflow");
    }
    candidate_profile.entropy_model_bits = model_bits;
    candidate_profile.entropy_token_bits = token_bits;
    candidate_profile.dc_entropy_clusters = selected_dc_cost.cluster_count;
    candidate_profile.dc_entropy_is_ans =
      !selected.all_prefix_entropy && dc_code.mode == EntropyCodingMode::kAns;
    candidate_profile.ac_entropy_clusters = selected_ac_cost.cluster_count;
    candidate_profile.ac_entropy_is_ans =
      selected_ac_code.mode == EntropyCodingMode::kAns;
    candidate_profile.coefficient_order_entropy_is_ans =
      selected.custom_order && !selected.all_prefix_entropy &&
      order_code.mode == EntropyCodingMode::kAns;
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

Status codestream_internal::PhysicalSectionSizesFromBitCounts(
  std::span<const uint64_t> common_section_bits,
  std::span<const uint64_t> ac_section_bits,
  size_t ac_group_count,
  std::vector<size_t>* sizes) {

  if (sizes == nullptr || common_section_bits.empty() ||
      ac_section_bits.empty() ||
      ac_group_count == std::numeric_limits<size_t>::max() ||
      ac_section_bits.size() != ac_group_count + 1) {
    return Status::InvalidArgument("Frame-section dimensions are invalid");
  }
  const auto padded_bytes = [](uint64_t bits, size_t* bytes) {
    const uint64_t quotient = bits / 8;
    const uint64_t remainder = bits % 8;
    if (quotient > std::numeric_limits<size_t>::max() - (remainder != 0)) {
      return false;
    }
    *bytes = static_cast<size_t>(quotient) + (remainder != 0);
    return true;
  };

  try {
    std::vector<size_t> candidate;
    if (ac_group_count == 1) {
      if (common_section_bits.size() != 2 || ac_section_bits.size() != 2) {
        return Status::Internal(
          "Single-group frame has an invalid section count");
      }
      uint64_t combined_bits = 0;
      for (uint64_t bits : common_section_bits) {
        if (!AddMeasuredBits(bits, &combined_bits)) {
          return AllocationFailure();
        }
      }
      for (uint64_t bits : ac_section_bits) {
        if (!AddMeasuredBits(bits, &combined_bits)) {
          return AllocationFailure();
        }
      }
      size_t combined_size = 0;
      if (!padded_bytes(combined_bits, &combined_size)) {
        return AllocationFailure();
      }
      candidate.push_back(combined_size);
      *sizes = std::move(candidate);
      return Status::Ok();
    }

    if (common_section_bits.size() >
        candidate.max_size() - ac_section_bits.size()) {
      return AllocationFailure();
    }
    candidate.reserve(common_section_bits.size() + ac_section_bits.size());
    for (uint64_t bits : common_section_bits) {
      size_t bytes = 0;
      if (!padded_bytes(bits, &bytes)) {
        return AllocationFailure();
      }
      candidate.push_back(bytes);
    }
    for (uint64_t bits : ac_section_bits) {
      size_t bytes = 0;
      if (!padded_bytes(bits, &bytes)) {
        return AllocationFailure();
      }
      candidate.push_back(bytes);
    }
    *sizes = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

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
