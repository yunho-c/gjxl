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
#include <type_traits>
#include <utility>
#include <vector>

#include "codec/codestream.h"
#include "codec/vardct_frame_view_internal.h"
#include "codestream/ac_group.h"
#include "codestream/ans_internal.h"
#include "codestream/bit_writer.h"
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/dc_group.h"
#include "codestream/encoder_internal.h"
#include "codestream/entropy.h"
#include "codestream/entropy_internal.h"
#include "codestream/headers.h"
#include "codestream/sections.h"
#include "core/thread_budget.h"

namespace gjxl {
using vardct_frame_internal::VarDctFrameView;
namespace {

using ProfileClock = std::chrono::steady_clock;
inline constexpr size_t kMaximumSectionWorkers = 8;

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

Status WriteValidatedTokenStream(
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  BitWriter* writer) {

  // The global section has already serialized and therefore validated this
  // model. Avoid repeating that model-wide validation in every ANS section.
  return code.mode == EntropyCodingMode::kAns
    ? codestream_internal::WriteAnsTokenStream(
        tokens, code, writer)
    : WriteTokenStream(tokens, code, writer);
}

Status WriteValidatedTokenStream(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  BitWriter* writer) {

  return WriteValidatedTokenStream(
    EntropyTokenStreamView::Interleaved(tokens), code, writer);
}

template <typename Function>
Status RunParallelSections(size_t count, Function&& function) {
  const auto invoke = [&](size_t index, size_t worker_index) -> Status {
    if constexpr (std::is_invocable_r_v<
                    Status, Function&, size_t, size_t>) {
      return function(index, worker_index);
    } else {
      return function(index);
    }
  };
  if (count == 0) return Status::Ok();
  if (thread_budget_internal::InExplicitParallelScope()) {
    for (size_t index = 0; index < count; ++index) {
      Status status = invoke(index, 0);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }
  const size_t hardware_workers = std::max<size_t>(
    std::thread::hardware_concurrency(), 1);
  const size_t automatic_worker_count = std::min(
    count, std::min(kMaximumSectionWorkers, hardware_workers));
  const size_t cpu_thread_count =
    thread_budget_internal::CpuThreadCount();
  auto* const participant_tracker =
    thread_budget_internal::ParticipantTracker();
  const auto resource_context = resource_budget_internal::CurrentResourceContext();
  const size_t participant_count = cpu_thread_count == 0
    ? automatic_worker_count
    : std::min(automatic_worker_count, cpu_thread_count);
  if (participant_count == 1) {
    thread_budget_internal::ParallelScope scope(
      cpu_thread_count, participant_tracker, resource_context);
    for (size_t index = 0; index < count; ++index) {
      Status status = invoke(index, 0);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  std::vector<Status> statuses(count);
  std::atomic<size_t> next_index{0};
  std::vector<std::thread> workers;
  const size_t spawned_worker_count = cpu_thread_count == 0
    ? participant_count
    : participant_count - 1;
  workers.reserve(spawned_worker_count);
  const auto run_worker = [&](size_t worker_index) {
    thread_budget_internal::ParallelScope scope(
      cpu_thread_count, participant_tracker, resource_context);
    while (true) {
      const size_t index =
        next_index.fetch_add(1, std::memory_order_relaxed);
      if (index >= count) break;
      try {
        statuses[index] = invoke(index, worker_index);
      } catch (const std::bad_alloc&) {
        statuses[index] = AllocationFailure();
      } catch (const std::length_error&) {
        statuses[index] = AllocationFailure();
      } catch (...) {
        statuses[index] = Status::Internal(
          "Codestream section worker failed unexpectedly");
      }
    }
  };
  try {
    for (size_t worker = 0; worker < spawned_worker_count; ++worker) {
      workers.emplace_back(run_worker, worker);
    }
  } catch (const std::system_error&) {
    next_index.store(count, std::memory_order_relaxed);
    for (std::thread& worker : workers) worker.join();
    return Status::Internal("Unable to start codestream section workers");
  }
  if (cpu_thread_count != 0) run_worker(spawned_worker_count);
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
  uint64_t* token_bits,
  codestream_internal::SectionWritingWorkProfile* profile) {

  uint64_t written_token_bits = 0;
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
  const size_t dc_tokens_start = writer->bits_written();
  if (Status status = WriteValidatedTokenStream(
        dc_tokens, code, writer); !status.ok()) {
    return status;
  }
  written_token_bits = writer->bits_written() - dc_tokens_start;
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
  const size_t metadata_tokens_start = writer->bits_written();
  Status status = WriteValidatedTokenStream(
    metadata_tokens, code, writer);
  WorkEnd(
    profile != nullptr, metadata_tokens_begin,
    profile == nullptr ? nullptr : &profile->token_write_nanoseconds);
  if (!status.ok()) {
    return status;
  }
  const size_t metadata_token_bits =
    writer->bits_written() - metadata_tokens_start;
  if (written_token_bits > std::numeric_limits<uint64_t>::max() -
      metadata_token_bits) {
    return Status::InvalidArgument("DC token bit count overflow");
  }
  written_token_bits += metadata_token_bits;
  if (token_bits != nullptr) *token_bits = written_token_bits;
  return Status::Ok();
}

struct AcEncodingCandidate {
  size_t block_context_candidate_index = 0;
  SimpleBlockContextMap block_context_map;
  bool custom_order = false;
  std::vector<std::vector<uint16_t>> contexts;
  std::vector<codestream_internal::SimpleAcGroupTokenData> direct_groups;
  std::vector<codestream_internal::PreparedFixedAnsCluster>
    fixed_context_populations;
  std::vector<EntropyTokenStreamView> streams;
  EntropyCode ac_code;
  EntropyCodeCost ac_cost;
  EntropyCode prefix_ac_code;
  EntropyCodeCost prefix_ac_cost;
  codestream_internal::PreparedAnsEntropyCode prepared_ans;
  std::vector<uint64_t> ans_section_candidate_bits;
  size_t complete_size = 0;
  bool all_prefix_entropy = false;
};

Status ReduceFixedAcPopulations(
  std::span<const codestream_internal::SimpleAcGroupTokenData> groups,
  size_t context_count,
  std::vector<codestream_internal::PreparedFixedAnsCluster>* populations) {

  if (populations == nullptr || context_count == 0) {
    return Status::InvalidArgument("AC population output is invalid");
  }
  try {
    std::vector<codestream_internal::PreparedFixedAnsCluster> candidate(
      context_count);
    for (const auto& group : groups) {
      uint64_t group_token_count = 0;
      for (const auto& context_population : group.context_populations) {
        if (context_population.context >= candidate.size() ||
            context_population.symbol_offset >
              group.symbol_populations.size() ||
            context_population.symbol_count >
              group.symbol_populations.size() -
                context_population.symbol_offset) {
          return Status::Internal("AC sparse population is invalid");
        }
        auto& destination = candidate[context_population.context];
        if (destination.token_count >
              std::numeric_limits<uint64_t>::max() -
                context_population.token_count ||
            destination.extra_bits >
              std::numeric_limits<uint64_t>::max() -
                context_population.extra_bits ||
            group_token_count >
              std::numeric_limits<uint64_t>::max() -
                context_population.token_count) {
          return Status::Internal("AC population reduction overflow");
        }
        uint64_t sparse_count = 0;
        const auto symbols = std::span(group.symbol_populations).subspan(
          context_population.symbol_offset,
          context_population.symbol_count);
        for (const auto& symbol : symbols) {
          uint64_t& count = destination.counts[symbol.symbol];
          if (count > std::numeric_limits<uint64_t>::max() - symbol.count ||
              sparse_count >
                std::numeric_limits<uint64_t>::max() - symbol.count) {
            return Status::Internal("AC symbol population overflow");
          }
          count += symbol.count;
          sparse_count += symbol.count;
        }
        if (sparse_count != context_population.token_count) {
          return Status::Internal("AC sparse population count differs");
        }
        destination.token_count += context_population.token_count;
        destination.extra_bits += context_population.extra_bits;
        destination.maximum_symbol = std::max(
          destination.maximum_symbol, context_population.maximum_symbol);
        group_token_count += context_population.token_count;
      }
      if (group_token_count != group.values.size()) {
        return Status::Internal("AC group population count differs");
      }
    }
    *populations = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status SelectOrdinaryEntropyCodingModeFromFixedPopulations(
  std::span<const codestream_internal::PreparedFixedAnsCluster> populations,
  const EntropyCodeOptions& options,
  EntropyCodingMode* mode) {

  if (mode == nullptr || populations.size() != options.context_count ||
      options.uint_config != kDefaultHybridUintConfig) {
    return Status::InvalidArgument(
      "Ordinary prepared entropy selection is invalid");
  }
  size_t token_count = 0;
  bool all_singleton = true;
  for (const auto& population : populations) {
    if (population.token_count >
        std::numeric_limits<size_t>::max() - token_count) {
      return Status::InvalidArgument("Entropy token count overflow");
    }
    token_count += static_cast<size_t>(population.token_count);
    size_t nonzero_symbols = 0;
    for (const uint64_t count : population.counts) {
      nonzero_symbols += count != 0;
    }
    all_singleton &= nonzero_symbols <= 1;
  }
  *mode = token_count < 100 || all_singleton
    ? EntropyCodingMode::kPrefix
    : EntropyCodingMode::kAns;
  return Status::Ok();
}

Status OptimizeBestEntropyCode(
  std::span<const EntropyTokenStreamView> streams,
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
  codestream_internal::PreparedEntropyClusters prepared;
  if (Status status =
        codestream_internal::OptimizeEntropyCodeAndPrepareClusters(
          streams, options, &prefix, &prefix_cost, &prepared, profile);
      !status.ok()) {
    return status;
  }
  EntropyCode ans;
  EntropyCodeCost ans_cost;
  if (Status status =
        codestream_internal::OptimizeAnsEntropyCodeWithPreparedClusters(
          streams, prefix, prepared, &ans, &ans_cost, profile);
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
    *cost = std::move(ans_cost);
  } else {
    *code = std::move(prefix);
    *cost = std::move(prefix_cost);
  }
  WorkEnd(
    profile != nullptr, selection_begin,
    profile == nullptr ? nullptr : &profile->selection_nanoseconds);
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

  try {
    std::vector<EntropyTokenStreamView> views;
    views.reserve(streams.size());
    for (const std::vector<EntropyToken>& stream : streams) {
      views.push_back(EntropyTokenStreamView::Interleaved(stream));
    }
    return OptimizeBestEntropyCode(
      views, options, code, cost, prefix_fallback, prefix_fallback_cost,
      profile);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

Status OptimizeOrdinaryEntropyCode(
  std::span<const EntropyTokenStreamView> streams,
  const EntropyCodeOptions& options,
  VarDctEntropyBehavior behavior,
  std::span<const codestream_internal::PreparedFixedAnsCluster>
    fixed_context_populations,
  bool defer_ans_token_cost,
  EntropyCode* code,
  EntropyCodeCost* cost,
  codestream_internal::EntropyWorkProfile* profile) {

  if (code == nullptr || cost == nullptr || options.context_count == 0 ||
      behavior == VarDctEntropyBehavior::kMaximumCompression) {
    return Status::InvalidArgument("Ordinary entropy selection is invalid");
  }
  EntropyCodingMode mode = EntropyCodingMode::kPrefix;
  const Status selection_status = fixed_context_populations.empty()
    ? codestream_internal::SelectOrdinaryEntropyCodingMode(
        streams, options, &mode)
    : SelectOrdinaryEntropyCodingModeFromFixedPopulations(
        fixed_context_populations, options, &mode);
  if (!selection_status.ok()) {
    return selection_status;
  }
  if (mode == EntropyCodingMode::kPrefix) {
    return codestream_internal::OptimizeFastPrefixEntropyCode(
      streams, options, code, cost, profile);
  }
  const auto direct_mode = behavior == VarDctEntropyBehavior::kHighDensity
    ? codestream_internal::DirectAnsEntropyMode::kHighDensity
    : codestream_internal::DirectAnsEntropyMode::kBalanced;
  if (!defer_ans_token_cost) {
    return fixed_context_populations.empty()
      ? codestream_internal::OptimizeDirectAnsEntropyCode(
          streams, options, direct_mode, code, cost, profile)
      : codestream_internal::OptimizeDirectAnsEntropyCodeWithFixedPopulations(
          streams, options, fixed_context_populations,
          code, cost, profile);
  }
  const Status optimization_status = fixed_context_populations.empty()
    ? codestream_internal::OptimizeDirectAnsEntropyCode(
        streams, options, direct_mode, code, nullptr, profile)
    : codestream_internal::OptimizeDirectAnsEntropyCodeWithFixedPopulations(
        streams, options, fixed_context_populations,
        code, nullptr, profile);
  if (!optimization_status.ok()) {
    return optimization_status;
  }
  BitWriter model;
  if (Status status = WriteEntropyCode(*code, &model); !status.ok()) {
    return status;
  }
  *cost = {
    .model_bits = model.bits_written(),
    .cluster_count = code->ans_histograms.size(),
  };
  return Status::Ok();
}

Status OptimizeOrdinaryEntropyCode(
  std::span<const std::vector<EntropyToken>> streams,
  const EntropyCodeOptions& options,
  VarDctEntropyBehavior behavior,
  bool defer_ans_token_cost,
  EntropyCode* code,
  EntropyCodeCost* cost,
  codestream_internal::EntropyWorkProfile* profile) {

  try {
    std::vector<EntropyTokenStreamView> views;
    views.reserve(streams.size());
    for (const std::vector<EntropyToken>& stream : streams) {
      views.push_back(EntropyTokenStreamView::Interleaved(stream));
    }
    return OptimizeOrdinaryEntropyCode(
      views, options, behavior, {}, defer_ans_token_cost,
      code, cost, profile);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

Status PrepareAcCandidate(
  AcEncodingCandidate* candidate,
  codestream_internal::EntropyWorkProfile* profile) {
  if (candidate == nullptr || candidate->streams.empty()) {
    return Status::InvalidArgument("AC encoding candidate is empty");
  }
  EntropyCode prefix;
  EntropyCodeCost prefix_cost;
  codestream_internal::PreparedEntropyClusters prepared_clusters;
  Status status =
    codestream_internal::OptimizeEntropyCodeAndPrepareClusters(
      candidate->streams,
      {
        .context_count = static_cast<uint32_t>(
          candidate->block_context_map.ac_context_count()),
      },
      &prefix, &prefix_cost, &prepared_clusters, profile);
  if (!status.ok()) {
    return status;
  }
  codestream_internal::PreparedAnsEntropyCode prepared_ans;
  status = codestream_internal::PrepareAnsEntropyCodeWithPreparedClusters(
    candidate->streams, prefix, prepared_clusters, &prepared_ans, profile);
  if (!status.ok()) {
    return status;
  }
  if (prepared_ans.candidates.empty()) {
    return Status::Internal("Prepared ANS candidate set is empty");
  }
  if (prepared_ans.section_count != candidate->streams.size()) {
    return Status::Internal("Prepared ANS section count differs");
  }
  if (candidate->streams.size() >
        std::numeric_limits<size_t>::max() /
          prepared_ans.candidates.size()) {
    return AllocationFailure();
  }
  try {
    std::vector<uint64_t> section_candidate_bits(
      candidate->streams.size() * prepared_ans.candidates.size());
    candidate->prefix_ac_code = std::move(prefix);
    candidate->prefix_ac_cost = std::move(prefix_cost);
    candidate->prepared_ans = std::move(prepared_ans);
    candidate->ans_section_candidate_bits =
      std::move(section_candidate_bits);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status FinalizeAcCandidate(
  AcEncodingCandidate* candidate,
  codestream_internal::EntropyWorkProfile* profile) {
  if (candidate == nullptr || candidate->streams.empty()) {
    return Status::InvalidArgument("AC encoding candidate is empty");
  }
  EntropyCode ans;
  EntropyCodeCost ans_cost;
  const ProfileClock::time_point selection_begin =
    WorkBegin(profile != nullptr);
  Status status = codestream_internal::FinalizePreparedAnsEntropyCode(
    &candidate->prepared_ans, candidate->ans_section_candidate_bits,
    &ans, &ans_cost);
  if (!status.ok()) {
    return status;
  }
  const auto total_bits = [](const EntropyCodeCost& entropy_cost) {
    return entropy_cost.model_bits >
        std::numeric_limits<uint64_t>::max() - entropy_cost.token_bits
      ? std::numeric_limits<uint64_t>::max()
      : entropy_cost.model_bits + entropy_cost.token_bits;
  };
  if (total_bits(ans_cost) < total_bits(candidate->prefix_ac_cost)) {
    candidate->ac_code = std::move(ans);
    candidate->ac_cost = std::move(ans_cost);
  } else {
    candidate->ac_code = candidate->prefix_ac_code;
    candidate->ac_cost = candidate->prefix_ac_cost;
  }
  candidate->ans_section_candidate_bits.clear();
  WorkEnd(
    profile != nullptr, selection_begin,
    profile == nullptr ? nullptr : &profile->selection_nanoseconds);
  return Status::Ok();
}

Status WriteCommonSections(
  const VarDctFrameView& frame,
  std::span<const SimpleDcGroupTokenStreams> dc_groups,
  std::span<const std::vector<EntropyToken>> dc_streams,
  const SimpleBlockContextMap& block_context_map,
  const EntropyCode& dc_code,
  std::vector<BitWriter>* sections,
  uint64_t* token_bits,
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
    std::vector<uint64_t> group_token_bits(
      token_bits == nullptr ? 0 : dc_groups.size());
    status = RunParallelSections(
      dc_groups.size(),
      [&](size_t index) {
        return WriteDcGroupSection(
          dc_groups[index], dc_streams[2 * index],
          dc_streams[2 * index + 1], dc_code, &candidate[1 + index],
          token_bits == nullptr ? nullptr : &group_token_bits[index],
          profile == nullptr ? nullptr : &group_profiles[index]);
      });
    if (!status.ok()) {
      return status;
    }
    for (const auto& group_profile : group_profiles) {
      codestream_internal::AccumulateSectionWritingWorkProfile(
        group_profile, profile);
    }
    uint64_t candidate_token_bits = 0;
    for (uint64_t group_bits : group_token_bits) {
      if (candidate_token_bits >
          std::numeric_limits<uint64_t>::max() - group_bits) {
        return Status::InvalidArgument("DC token bit count overflow");
      }
      candidate_token_bits += group_bits;
    }
    *sections = std::move(candidate);
    if (token_bits != nullptr) *token_bits = candidate_token_bits;
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
  uint64_t* token_bits,
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
        Status token_status = WriteValidatedTokenStream(
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
    uint64_t candidate_token_bits = 0;
    if (token_bits != nullptr) {
      for (size_t index = 1; index < candidate.size(); ++index) {
        if (candidate_token_bits > std::numeric_limits<uint64_t>::max() -
            candidate[index].bits_written()) {
          return Status::InvalidArgument("AC token bit count overflow");
        }
        candidate_token_bits += candidate[index].bits_written();
      }
    }
    *sections = std::move(candidate);
    if (token_bits != nullptr) *token_bits = candidate_token_bits;
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
  const EntropyCodeCost& dc_cost,
  std::vector<uint64_t>* section_bits,
  uint64_t* measurement_work) {

  if (section_bits == nullptr || dc_groups.empty() ||
      dc_cost.section_token_bits.size() != 2 * dc_groups.size()) {
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
        if (!AddMeasuredBits(dc_cost.section_token_bits[2 * index], &bits)) {
          return Status::InvalidArgument("DC section bit count overflow");
        }
        if (!AddMeasuredBits(
              dc_cost.section_token_bits[2 * index + 1], &bits)) {
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
  const VarDctFrameView& frame,
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
  const EntropyCodeCost& ac_cost,
  const SimpleCoefficientOrders& custom_orders,
  const EntropyCodeCost* order_cost,
  std::vector<uint64_t>* section_bits,
  uint64_t* measurement_work) {

  if (section_bits == nullptr || ac.streams.empty() ||
      ac_cost.section_token_bits.size() != ac.streams.size() ||
      (ac.custom_order != (order_cost != nullptr))) {
    return Status::InvalidArgument("AC section measurement is invalid");
  }
  try {
    std::vector<uint64_t> candidate(1 + ac.streams.size());
    const uint16_t used_order_mask =
      ac.custom_order ? custom_orders.used_order_mask : 0;
    const ProfileClock::time_point measurement_begin =
      WorkBegin(measurement_work != nullptr);
    if (Status status = MeasureAcGlobalBits(
          ac.streams.size(), used_order_mask, order_cost, ac_cost,
          &candidate[0]);
        !status.ok()) {
      return status;
    }
    std::copy(
      ac_cost.section_token_bits.begin(), ac_cost.section_token_bits.end(),
      candidate.begin() + 1);
    *section_bits = std::move(candidate);
    WorkEnd(
      measurement_work != nullptr, measurement_begin, measurement_work);
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
  const VarDctFrameView& frame,
  BitWriter* writer) {

  Status status = WriteSimpleCodestreamHeader(
    frame.geometry().frame(), writer);
  if (!status.ok()) {
    return status;
  }
  return WriteSimpleFrameHeader(frame.profile(), writer);
}

Status MeasureCandidateSize(
  const VarDctFrameView& frame,
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
  const VarDctFrameView& frame,
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

Status EncodeVarDctCodestreamWithRepresentationPolicy(
  const VarDctFrameView& frame,
  VarDctCodestreamOptions options,
  bool exhaustive_representation_search,
  std::vector<uint8_t>* output,
  codestream_internal::VarDctCodestreamProfile* profile) {

  if (output == nullptr) {
    return Status::InvalidArgument("Codestream output is null");
  }
  switch (options.entropy_behavior) {
    case VarDctEntropyBehavior::kBalanced:
    case VarDctEntropyBehavior::kHighDensity:
    case VarDctEntropyBehavior::kMaximumCompression:
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT entropy behavior is invalid");
  }
  switch (options.coefficient_order_behavior) {
    case VarDctCoefficientOrderBehavior::kFull:
    case VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled:
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT coefficient-order behavior is invalid");
  }
  if (options.entropy_behavior ==
      VarDctEntropyBehavior::kMaximumCompression) {
    options.coefficient_order_behavior =
      VarDctCoefficientOrderBehavior::kFull;
  }
  codestream_internal::VarDctCodestreamProfile candidate_profile;
  candidate_profile.entropy_behavior = options.entropy_behavior;
  candidate_profile.coefficient_order_behavior =
    options.coefficient_order_behavior;
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
    Status status = codestream_internal::TokenizeSimpleDcGroupsForEncoder(
      frame, &dc_groups);
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
    const ProfileClock::time_point block_context_begin =
      WorkBegin(profile != nullptr);
    if (exhaustive_representation_search) {
      status = codestream_internal::
        ComputeSimpleBlockContextMapCandidatesForEncoder(
          frame, &block_context_maps);
    } else {
      SimpleBlockContextMap block_context_map;
      status = codestream_internal::ComputeSimpleBlockContextMapForEncoder(
        frame, &block_context_map);
      if (status.ok()) {
        block_context_maps.push_back(std::move(block_context_map));
      }
    }
    WorkEnd(
      profile != nullptr, block_context_begin,
      &candidate_profile.block_context_map_work_nanoseconds);
    if (!status.ok()) {
      return status;
    }

    const ProfileClock::time_point coefficient_order_begin =
      WorkBegin(profile != nullptr);
    status = codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(
      frame, options.coefficient_order_behavior, &custom_orders);
    WorkEnd(
      profile != nullptr, coefficient_order_begin,
      &candidate_profile.coefficient_order_work_nanoseconds);
    if (!status.ok()) {
      return status;
    }
    if (block_context_maps.empty()) {
      return Status::Internal(
        "Validated frame produced no block-context candidates");
    }

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
    const size_t candidates_per_map =
      exhaustive_representation_search && has_custom_orders ? 2 : 1;
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
        .custom_order =
          !exhaustive_representation_search && has_custom_orders,
      });
      if (exhaustive_representation_search && has_custom_orders) {
        candidates.push_back({
          .block_context_candidate_index = map_index,
          .block_context_map = block_context_maps[map_index],
          .custom_order = true,
        });
      }
    }
    const SimpleCoefficientOrders natural_orders;
    std::array<std::vector<SimpleAcGroupTokenTemplate>, 2> order_templates;
    const size_t ac_group_count = frame.ac_group_count();
    if (ac_group_count == 0) {
      return AllocationFailure();
    }
    if (!exhaustive_representation_search) {
      if (candidates.size() != 1) {
        return Status::Internal("Ordinary AC candidate count differs");
      }
      codestream_internal::SimpleAcNaturalOrders prepared_natural_orders;
      status = codestream_internal::PrepareSimpleAcNaturalOrders(
        &prepared_natural_orders);
      if (!status.ok()) return status;
      AcEncodingCandidate& candidate = candidates.front();
      candidate.direct_groups.resize(ac_group_count);
      const bool collect_fixed_populations =
        options.entropy_behavior == VarDctEntropyBehavior::kBalanced;
      std::array<codestream_internal::SimpleAcTokenizationScratch,
                 kMaximumSectionWorkers> tokenization_scratch;
      std::vector<uint64_t> coefficient_tokenization_work(
        profile == nullptr ? 0 : ac_group_count);
      status = RunParallelSections(
        ac_group_count,
        [&](size_t group_index, size_t worker_index) {
          if (worker_index >= tokenization_scratch.size()) {
            return Status::Internal("AC tokenization worker index overflow");
          }
          const ProfileClock::time_point work_begin =
            WorkBegin(profile != nullptr);
          Status token_status =
            codestream_internal::TokenizeSimpleAcGroupForEncoder(
              frame, has_custom_orders ? custom_orders : natural_orders,
              prepared_natural_orders, candidate.block_context_map,
              group_index, collect_fixed_populations,
              &tokenization_scratch[worker_index],
              &candidate.direct_groups[group_index]);
          WorkEnd(
            profile != nullptr, work_begin,
            profile == nullptr
              ? nullptr
              : &coefficient_tokenization_work[group_index]);
          return token_status;
        });
      if (!status.ok()) return status;
      candidate.streams.reserve(ac_group_count);
      for (const auto& group : candidate.direct_groups) {
        if (group.values.size() != group.contexts.size()) {
          return Status::Internal("Direct AC token count differs");
        }
        candidate.streams.push_back(
          EntropyTokenStreamView::Split(group.values, group.contexts));
        if (profile != nullptr) {
          if (group.values.size() >
              std::numeric_limits<size_t>::max() -
                candidate_profile.coefficient_token_count) {
            return Status::Internal("Direct AC token profile overflow");
          }
          candidate_profile.coefficient_token_count += group.values.size();
        }
      }
      if (collect_fixed_populations) {
        status = ReduceFixedAcPopulations(
          candidate.direct_groups,
          candidate.block_context_map.ac_context_count(),
          &candidate.fixed_context_populations);
        if (!status.ok()) return status;
        for (auto& group : candidate.direct_groups) {
          std::vector<codestream_internal::SimpleAcContextPopulation>().swap(
            group.context_populations);
          std::vector<codestream_internal::SimpleAcSymbolPopulation>().swap(
            group.symbol_populations);
        }
      }
      if (profile != nullptr) {
        for (const uint64_t work : coefficient_tokenization_work) {
          candidate_profile.coefficient_tokenization_work_nanoseconds += work;
        }
        candidate_profile.coefficient_tokenization_pass_count = 1;
      }
    } else {
      const size_t order_template_count = has_custom_orders ? 2 : 1;
      if (order_template_count >
          std::numeric_limits<size_t>::max() / ac_group_count) {
        return AllocationFailure();
      }
      for (size_t index = 0; index < order_template_count; ++index) {
        order_templates[index].resize(ac_group_count);
      }
      const size_t coefficient_tokenization_task_count =
        order_template_count * ac_group_count;
      std::vector<uint64_t> coefficient_tokenization_work(
        profile == nullptr ? 0 : coefficient_tokenization_task_count);
      status = RunParallelSections(
        coefficient_tokenization_task_count,
        [&](size_t task_index) {
          const size_t order_index = task_index / ac_group_count;
          const size_t group_index = task_index % ac_group_count;
          const ProfileClock::time_point work_begin =
            WorkBegin(profile != nullptr);
          Status token_status = codestream_internal::
            BuildSimpleAcGroupTokenTemplateForEncoder(
              frame, order_index == 0 ? natural_orders : custom_orders,
              group_index, &order_templates[order_index][group_index]);
          WorkEnd(
            profile != nullptr, work_begin,
            profile == nullptr
              ? nullptr
              : &coefficient_tokenization_work[task_index]);
          return token_status;
        });
      if (!status.ok()) return status;
      if (profile != nullptr) {
        for (const uint64_t work : coefficient_tokenization_work) {
          candidate_profile.coefficient_tokenization_work_nanoseconds += work;
        }
        candidate_profile.coefficient_tokenization_pass_count =
          order_template_count;
        for (size_t order_index = 0; order_index < order_template_count;
             ++order_index) {
          for (const SimpleAcGroupTokenTemplate& group :
               order_templates[order_index]) {
            if (group.tokens.size() >
                std::numeric_limits<size_t>::max() -
                  candidate_profile.coefficient_token_count) {
              return Status::Internal("AC token-template profile overflow");
            }
            candidate_profile.coefficient_token_count += group.tokens.size();
          }
        }
      }

      if (candidates.size() >
          std::numeric_limits<size_t>::max() / ac_group_count) {
        return AllocationFailure();
      }
      for (AcEncodingCandidate& candidate : candidates) {
        candidate.contexts.resize(ac_group_count);
      }
      const size_t context_materialization_task_count =
        candidates.size() * ac_group_count;
      std::vector<uint64_t> context_materialization_work(
        profile == nullptr ? 0 : context_materialization_task_count);
      status = RunParallelSections(
        context_materialization_task_count,
        [&](size_t task_index) {
          const size_t candidate_index = task_index / ac_group_count;
          const size_t group_index = task_index % ac_group_count;
          const ProfileClock::time_point work_begin =
            WorkBegin(profile != nullptr);
          AcEncodingCandidate& candidate = candidates[candidate_index];
          const size_t order_index = candidate.custom_order ? 1 : 0;
          const auto& token_templates = order_templates[order_index];
          Status token_status = codestream_internal::
            MaterializeSimpleAcGroupContextsForEncoder(
              token_templates[group_index], candidate.block_context_map,
              &candidate.contexts[group_index]);
          WorkEnd(
            profile != nullptr, work_begin,
            profile == nullptr
              ? nullptr
              : &context_materialization_work[task_index]);
          return token_status;
        });
      if (!status.ok()) return status;
      for (AcEncodingCandidate& candidate : candidates) {
        const size_t order_index = candidate.custom_order ? 1 : 0;
        const auto& token_templates = order_templates[order_index];
        candidate.streams.reserve(token_templates.size());
        for (size_t group_index = 0;
             group_index < token_templates.size(); ++group_index) {
          if (candidate.contexts[group_index].size() !=
              token_templates[group_index].values.size()) {
            return Status::Internal("Materialized AC token count differs");
          }
          candidate.streams.push_back(EntropyTokenStreamView::Split(
            token_templates[group_index].values,
            candidate.contexts[group_index]));
        }
      }
      if (profile != nullptr) {
        candidate_profile.coefficient_context_materialization_count =
          candidates.size();
        for (const uint64_t work : context_materialization_work) {
          candidate_profile
            .coefficient_context_materialization_work_nanoseconds += work;
        }
        for (const AcEncodingCandidate& candidate : candidates) {
          for (const EntropyTokenStreamView stream : candidate.streams) {
            if (stream.size() >
                std::numeric_limits<size_t>::max() -
                  candidate_profile.coefficient_materialized_token_count) {
              return Status::Internal(
                "Materialized AC token profile overflow");
            }
            candidate_profile.coefficient_materialized_token_count +=
              stream.size();
          }
        }
      }
      // Values stay live through entropy coding; maximum compression no longer
      // needs descriptors after all context-map candidates are materialized.
      for (auto& templates : order_templates) {
        for (SimpleAcGroupTokenTemplate& group : templates) {
          std::vector<SimpleAcBlockContextKey>().swap(
            group.block_context_keys);
          std::vector<SimpleAcTokenTemplate>().swap(group.tokens);
        }
      }
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
          if (exhaustive_representation_search) {
            return OptimizeBestEntropyCode(
              dc_streams, {.context_count = kSimpleDcContextCount}, &dc_code,
              &dc_cost, &prefix_dc_code, &prefix_dc_cost, entropy_profile);
          }
          return OptimizeOrdinaryEntropyCode(
            dc_streams, {.context_count = kSimpleDcContextCount},
            options.entropy_behavior, true,
            &dc_code, &dc_cost, entropy_profile);
        }
        if (has_custom_orders && index == 1) {
          const std::span<const std::vector<EntropyToken>> order_streams(
            &order_tokens, 1);
          const EntropyCodeOptions order_options{
            .context_count = kSimplePermutationContextCount,
            .uint_config = {0, 0, 0},
          };
          if (exhaustive_representation_search) {
            return OptimizeBestEntropyCode(
              order_streams, order_options, &order_code, &order_cost,
              &prefix_order_code, &prefix_order_cost, entropy_profile);
          }
          return OptimizeOrdinaryEntropyCode(
            order_streams, order_options, options.entropy_behavior,
            false, &order_code, &order_cost, entropy_profile);
        }
        AcEncodingCandidate& candidate =
          candidates[index - 1 - order_task_count];
        if (exhaustive_representation_search) {
          return PrepareAcCandidate(&candidate, entropy_profile);
        }
        return OptimizeOrdinaryEntropyCode(
          candidate.streams,
          {
            .context_count = static_cast<uint32_t>(
              candidate.block_context_map.ac_context_count()),
          },
          options.entropy_behavior, candidate.fixed_context_populations, true,
          &candidate.ac_code, &candidate.ac_cost, entropy_profile);
      });
    if (!status.ok()) {
      return status;
    }

    if (exhaustive_representation_search) {
      struct AnsSectionTask {
        size_t candidate_index = 0;
        size_t section_index = 0;
      };
      size_t ans_task_count = 0;
      for (const AcEncodingCandidate& candidate : candidates) {
        const size_t candidate_count = candidate.prepared_ans.candidates.size();
        if (candidate_count == 0 ||
            candidate.prepared_ans.section_count != candidate.streams.size() ||
            candidate.streams.size() >
              std::numeric_limits<size_t>::max() / candidate_count ||
            candidate.ans_section_candidate_bits.size() !=
              candidate.streams.size() * candidate_count) {
          return Status::Internal("AC candidate section dimensions differ");
        }
        if (candidate.streams.size() >
            std::numeric_limits<size_t>::max() - ans_task_count) {
          return AllocationFailure();
        }
        ans_task_count += candidate.streams.size();
      }
      std::vector<AnsSectionTask> ans_tasks;
      ans_tasks.reserve(ans_task_count);
      for (size_t candidate_index = 0; candidate_index < candidates.size();
           ++candidate_index) {
        for (size_t section_index = 0;
             section_index < candidates[candidate_index].streams.size();
             ++section_index) {
          ans_tasks.push_back({candidate_index, section_index});
        }
      }
      std::vector<uint64_t> ans_task_work(
        profile == nullptr ? 0 : ans_tasks.size());
      status = RunParallelSections(
        ans_tasks.size(),
        [&](size_t task_index) {
          const ProfileClock::time_point task_begin =
            WorkBegin(profile != nullptr);
          const AnsSectionTask task = ans_tasks[task_index];
          AcEncodingCandidate& candidate = candidates[task.candidate_index];
          const size_t candidate_count =
            candidate.prepared_ans.candidates.size();
          Status measure_status =
            codestream_internal::MeasurePreparedAnsEntropyCodeSection(
              candidate.streams[task.section_index], candidate.prepared_ans,
              std::span<uint64_t>(candidate.ans_section_candidate_bits).subspan(
                task.section_index * candidate_count, candidate_count));
          WorkEnd(
            profile != nullptr, task_begin,
            profile == nullptr ? nullptr : &ans_task_work[task_index]);
          return measure_status;
        });
      if (!status.ok()) {
        return status;
      }
      if (profile != nullptr) {
        for (const uint64_t work : ans_task_work) {
          candidate_profile.entropy_work.ans_token_cost_nanoseconds += work;
        }
      }
      status = RunParallelSections(
        candidates.size(),
        [&](size_t candidate_index) {
          auto* entropy_profile = profile == nullptr
            ? nullptr
            : &entropy_profiles[1 + order_task_count + candidate_index];
          return FinalizeAcCandidate(
            &candidates[candidate_index], entropy_profile);
        });
      if (!status.ok()) {
        return status;
      }
    }
    for (const auto& entropy_profile : entropy_profiles) {
      codestream_internal::AccumulateEntropyWorkProfile(
        entropy_profile, &candidate_profile.entropy_work);
    }
    ProfileEnd(
      profile, entropy_begin,
      &candidate_profile.entropy_optimization_nanoseconds);

    uint64_t section_measurement_nanoseconds = 0;
    size_t selected_index = 0;
    if (exhaustive_representation_search) {
      const ProfileClock::time_point section_measurement_begin =
        ProfileBegin(profile);
    constexpr size_t kMixedEntropy = 0;
    constexpr size_t kPrefixEntropy = 1;
    const size_t entropy_mode_count =
      exhaustive_representation_search ? 2 : 1;
    const std::array<const EntropyCode*, 2> dc_codes = {
      &dc_code, &prefix_dc_code};
    const std::array<const EntropyCodeCost*, 2> dc_costs = {
      &dc_cost, &prefix_dc_cost};
    std::array<std::vector<uint64_t>, 2> dc_group_section_bits;
    std::array<uint64_t, 2> dc_group_measurement_work{};
    status = RunParallelSections(
      entropy_mode_count,
      [&](size_t mode) {
        Status measure_status = MeasureDcGroupSections(
          dc_groups, *dc_costs[mode],
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
        std::numeric_limits<size_t>::max() / entropy_mode_count) {
      return AllocationFailure();
    }
    const size_t common_measurement_count =
      block_context_maps.size() * entropy_mode_count;
    std::vector<uint64_t> common_measurement_work(common_measurement_count);
    status = RunParallelSections(
      common_measurement_count,
      [&](size_t index) {
        const size_t map_index = index / entropy_mode_count;
        const size_t mode = index % entropy_mode_count;
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
          const EntropyCodeCost& ac_cost = all_prefix
            ? candidate.prefix_ac_cost
            : candidate.ac_cost;
          const EntropyCodeCost* selected_order_cost = candidate.custom_order
            ? (all_prefix ? &prefix_order_cost : &order_cost)
            : nullptr;
          std::vector<uint64_t> ac_section_bits;
          Status measure_status = MeasureAcSections(
            candidate, ac_cost, custom_orders,
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
        candidate.complete_size = mixed_size;
        if (exhaustive_representation_search) {
          size_t prefix_size = 0;
          uint64_t prefix_measurement_work = 0;
          measure_status = measure(
            kPrefixEntropy, &prefix_size,
            profile == nullptr ? nullptr : &prefix_measurement_work);
          if (!measure_status.ok()) {
            return measure_status;
          }
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
      ProfileEnd(
        profile, section_measurement_begin, &section_measurement_nanoseconds);

      const ProfileClock::time_point candidate_selection_begin =
        WorkBegin(profile != nullptr);
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
    }
    AcEncodingCandidate& selected = candidates[selected_index];
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
    uint64_t written_dc_token_bits = 0;
    codestream_internal::SectionWritingWorkProfile selected_write_profile;
    status = WriteCommonSections(
      frame, dc_groups, dc_streams, selected.block_context_map,
      selected_dc_code, &common_sections,
      exhaustive_representation_search ? nullptr : &written_dc_token_bits,
      profile == nullptr ? nullptr : &selected_write_profile);
    if (!status.ok()) {
      return status;
    }
    std::vector<BitWriter> ac_sections;
    uint64_t written_ac_token_bits = 0;
    status = WriteAcSections(
      selected, selected_ac_code, custom_orders, order_tokens,
      selected_order_code, &ac_sections,
      exhaustive_representation_search ? nullptr : &written_ac_token_bits,
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
    if (!exhaustive_representation_search) {
      dc_cost.token_bits = written_dc_token_bits;
      selected.ac_cost.token_bits = written_ac_token_bits;
    }

    const ProfileClock::time_point assembly_begin = ProfileBegin(profile);
    std::vector<uint8_t> candidate_output;
    status = AssembleCandidate(
      frame, common_sections, ac_sections,
      selected.streams.size(), &candidate_output,
      profile == nullptr ? nullptr : &candidate_profile.assembly);
    if (!status.ok()) {
      return status;
    }
    if (exhaustive_representation_search &&
        candidate_output.size() != selected.complete_size) {
      return Status::Internal(
        "Measured codestream candidate size differs from assembly");
    }
    selected.complete_size = candidate_output.size();
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
      exhaustive_representation_search
        ? std::numeric_limits<size_t>::max()
        : 0;
    for (const AcEncodingCandidate& candidate : candidates) {
      size_t& minimum = candidate.custom_order
        ? candidate_profile.custom_order_candidate_bytes
        : candidate_profile.natural_candidate_bytes;
      if (minimum == 0 || candidate.complete_size < minimum) {
        minimum = candidate.complete_size;
      }
    }
    if (exhaustive_representation_search &&
        candidate_profile.natural_candidate_bytes ==
        std::numeric_limits<size_t>::max()) {
      return Status::Internal("Natural codestream candidate is missing");
    }
    candidate_profile.selected_coefficient_order_mask =
      selected.custom_order ? custom_orders.used_order_mask : 0;
    candidate_profile.block_context_candidate_count = candidates_per_map == 0
      ? 0
      : candidates.size() / candidates_per_map;
    const SimpleBlockContextMap compact_block_context_map =
      DefaultSimpleBlockContextMap();
    for (const AcEncodingCandidate& candidate : candidates) {
      if (candidate.block_context_map == compact_block_context_map &&
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

Status EncodeVarDctCodestreamMaximumCompression(
  const VarDctFrameView& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output,
  codestream_internal::VarDctCodestreamProfile* profile) {

  return EncodeVarDctCodestreamWithRepresentationPolicy(
    frame, options, true, output, profile);
}

Status EncodeVarDctCodestreamSingleRepresentation(
  const VarDctFrameView& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output,
  codestream_internal::VarDctCodestreamProfile* profile) {

  return EncodeVarDctCodestreamWithRepresentationPolicy(
    frame, options, false, output, profile);
}

Status EncodeVarDctCodestreamImpl(
  const VarDctFrameView& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output,
  codestream_internal::VarDctCodestreamProfile* profile) {

  const resource_budget_internal::ManagedHostScope managed_host(
    resource_budget_internal::ResourceClass::kSerializer);
  return options.entropy_behavior ==
      VarDctEntropyBehavior::kMaximumCompression
    ? EncodeVarDctCodestreamMaximumCompression(
        frame, options, output, profile)
    : EncodeVarDctCodestreamSingleRepresentation(
        frame, options, output, profile);
}

}  // namespace

Status codestream_internal::SelectOrdinaryEntropyCodingMode(
  std::span<const EntropyTokenStreamView> streams,
  const EntropyCodeOptions& options,
  EntropyCodingMode* mode) {

  if (mode == nullptr || options.context_count == 0) {
    return Status::InvalidArgument("Ordinary entropy selection is invalid");
  }
  size_t token_count = 0;
  bool all_singleton = true;
  std::vector<uint32_t> first_symbols(
    options.context_count, std::numeric_limits<uint32_t>::max());
  for (const EntropyTokenStreamView stream : streams) {
    if (!stream.valid() ||
        stream.size() > std::numeric_limits<size_t>::max() - token_count) {
      return Status::InvalidArgument("Entropy token stream is invalid");
    }
    token_count += stream.size();
    for (size_t index = 0; index < stream.size(); ++index) {
      const EntropyToken token = stream[index];
      if (token.context >= options.context_count) {
        return Status::InvalidArgument("Entropy token context is out of range");
      }
      HybridUintToken encoded;
      if (Status status = EncodeHybridUint(
            token.value, options.uint_config, &encoded); !status.ok()) {
        return status;
      }
      uint32_t& first = first_symbols[token.context];
      if (first == std::numeric_limits<uint32_t>::max()) {
        first = encoded.symbol;
      } else if (first != encoded.symbol) {
        all_singleton = false;
      }
    }
  }
  *mode = token_count < 100 || all_singleton
    ? EntropyCodingMode::kPrefix
    : EntropyCodingMode::kAns;
  return Status::Ok();
}

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

  return EncodeVarDctCodestream(frame, {}, output);
}

Status EncodeVarDctCodestream(
  const VarDctEncoderFrame& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output) {

  return codestream_internal::EncodeVarDctCodestreamFromView(
    vardct_frame_internal::BorrowFrame(frame), options, output);
}

Status codestream_internal::EncodeVarDctCodestreamProfiled(
  const VarDctEncoderFrame& frame,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile) {

  return EncodeVarDctCodestreamProfiled(frame, {}, output, profile);
}

Status codestream_internal::EncodeVarDctCodestreamProfiled(
  const VarDctEncoderFrame& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile) {

  if (profile == nullptr) {
    return Status::InvalidArgument("Codestream profile output is null");
  }
  return EncodeVarDctCodestreamFromView(
    vardct_frame_internal::BorrowFrame(frame), options, output, profile);
}

Status codestream_internal::EncodeVarDctCodestreamFromView(
  const VarDctFrameView& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output,
  VarDctCodestreamProfile* profile) {
  return EncodeVarDctCodestreamImpl(frame, options, output, profile);
}

}  // namespace gjxl
