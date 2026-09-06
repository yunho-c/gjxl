// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/serializer_storage_plan.h"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>

#include "codec/vardct_frame.h"
#include "codestream/ans_internal.h"
#include "codestream/coefficient_order.h"
#include "codestream/dc_group.h"

namespace gjxl::codestream_internal {
namespace {
using enum resource_budget_internal::VectorCapacityPolicy;

Status Overflow() {
  return Status::OutOfMemory("Whole-serializer storage bound overflows");
}

bool Multiply(size_t a, size_t b, size_t *out) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
    return false;
  *out = a * b;
  return true;
}

bool AddScaled(size_t count, size_t factor, size_t *total) {
  size_t bytes = 0;
  if (!Multiply(count, factor, &bytes) ||
      bytes > std::numeric_limits<size_t>::max() - *total)
    return false;
  *total += bytes;
  return true;
}

HostStorageBound Either(HostStorageBound a, HostStorageBound b) {
  return {std::max(a.retained_bytes, b.retained_bytes),
          std::max(a.peak_bytes, b.peak_bytes)};
}

bool AddWriter(size_t bits, HostStorageBound *out, size_t count = 1) {
  HostStorageBound bound;
  return ComputeEntropyWriterStorageBound(bits, &bound).ok() &&
         out->Add(bound, count);
}

struct TaskStoragePlan {
  HostStorageBound retained, working;
};

Status ComputeTaskStoragePlan(size_t tokens, size_t contexts, size_t sections,
                              VarDctEntropyBehavior behavior, bool deferred,
                              const EntropyModelStoragePlan &either_model,
                              TaskStoragePlan *out) {
  const bool exhaustive =
      behavior == VarDctEntropyBehavior::kMaximumCompression;
  EntropyOptimizationStoragePlan prefix, ans;
  Status status = ComputeEntropyOptimizationStoragePlan(
      {.policy = exhaustive ? EntropyStoragePolicy::kPrefix
                            : EntropyStoragePolicy::kFastPrefix,
       .tokens = tokens,
       .contexts = contexts,
       .sections = sections,
       .retain_prepared_clusters = exhaustive},
      &prefix);
  if (!status.ok())
    return status;
  status = ComputeEntropyOptimizationStoragePlan(
      {.policy = exhaustive
                     ? (deferred ? EntropyStoragePolicy::kDeferredAnsFromPrefix
                                 : EntropyStoragePolicy::kAnsFromPrefix)
                     : (behavior == VarDctEntropyBehavior::kHighDensity
                            ? EntropyStoragePolicy::kHighDensityAns
                            : EntropyStoragePolicy::kBalancedAns),
       .tokens = tokens,
       .contexts = contexts,
       .sections = sections,
       .borrow_prepared_clusters = exhaustive},
      &ans);
  if (!status.ok())
    return status;
  TaskStoragePlan plan;
  if (!exhaustive) {
    // The ordinary selector may choose FastPrefix even when ANS is requested.
    // Cost-enabled ANS bounds also cover the deferred-cost ordinary path.
    plan.retained = Either(prefix.output, ans.output);
    plan.working = Either(prefix.working, ans.working);
    if (!plan.working.AddVector<uint32_t>(contexts, kFreshExact) ||
        !plan.working.Add(either_model.write_scratch) ||
        !AddWriter(either_model.maximum_bits, &plan.working))
      return Overflow();
  } else {
    EntropyModelStoragePlan prefix_model;
    status = ComputeEntropyModelStoragePlan(
        EntropyCodingMode::kPrefix, contexts,
        std::min(contexts, kMaximumPrefixClusters), &prefix_model);
    if (!status.ok())
      return status;
    // Prepared weighted populations from Prefix must survive the ANS builder,
    // but are released before this task returns. Sum complete operations to
    // include that overlap, plus the fallback copy made by selection.
    if (!plan.working.Add(prefix.working) || !plan.working.Add(ans.working) ||
        !plan.working.Add(prefix_model.owned) ||
        !plan.working.AddVector<uint64_t>(sections, kFreshExact, 3))
      return Overflow();
    if (deferred) {
      // Four width models can coexist with Prefix, then finalization can copy
      // Prefix into ac_code. Include costs; section-by-width bits are control
      // storage and remain allocated even after clear().
      if (!plan.retained.Add(ans.output) ||
          !plan.retained.Add(prefix_model.owned, 2) ||
          !plan.retained.AddVector<uint64_t>(sections, kFreshExact, 3))
        return Overflow();
    } else {
      if (!plan.retained.Add(either_model.owned) ||
          !plan.retained.Add(prefix_model.owned) ||
          !plan.retained.AddVector<uint64_t>(sections, kFreshExact, 2))
        return Overflow();
    }
  }
  *out = plan;
  return Status::Ok();
}
} // namespace

Status ComputeSerializerStoragePlan(Extent2D frame_extent,
                                    const SerializerStorageOptions &options,
                                    SerializerStoragePlan *out) {
  constexpr size_t maximum_dimension = 0x3FFFFFFFu;
  if (out == nullptr || frame_extent.empty() ||
      frame_extent.width > maximum_dimension ||
      frame_extent.height > maximum_dimension)
    return Status::InvalidArgument(
        "Serializer plan frame dimensions are invalid");
  const auto behavior = options.coding.entropy_behavior;
  if (behavior != VarDctEntropyBehavior::kBalanced &&
      behavior != VarDctEntropyBehavior::kHighDensity &&
      behavior != VarDctEntropyBehavior::kMaximumCompression)
    return Status::InvalidArgument(
        "Serializer plan entropy behavior is invalid");
  auto order_behavior = options.coding.coefficient_order_behavior;
  if (order_behavior != VarDctCoefficientOrderBehavior::kFull &&
      order_behavior != VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled)
    return Status::InvalidArgument(
        "Serializer plan coefficient order is invalid");
  const bool exhaustive =
      behavior == VarDctEntropyBehavior::kMaximumCompression;
  if (exhaustive)
    order_behavior = VarDctCoefficientOrderBehavior::kFull;
  const size_t workers = options.cpu_thread_count == 0
                             ? kSerializerMaximumSectionWorkers
                             : std::min(options.cpu_thread_count,
                                        kSerializerMaximumSectionWorkers);
  static_assert(kJxlBlockDimension == 8);
  const Extent2D blocks = frame_extent.ceil_div(kJxlBlockDimension);
  size_t block_count = 0, color_tiles = 0;
  if (!blocks.try_area(&block_count) ||
      !blocks.ceil_div(8).try_area(&color_tiles))
    return Overflow();
  CoefficientOrderStoragePlan orders;
  Status status = ComputeCoefficientOrderStoragePlan(blocks, order_behavior,
                                                     workers, &orders);
  if (!status.ok())
    return status;
  BlockContextMapStoragePlan maps;
  status = ComputeBlockContextMapStoragePlan(blocks, exhaustive, &maps);
  if (!status.ok())
    return status;
  SerializerStoragePlan plan;
  const bool has_orders = orders.maximum_tokens != 0;
  plan.maximum_order_variants = exhaustive && has_orders ? 2 : 1;
  if (!Multiply(maps.maximum_maps, plan.maximum_order_variants,
                &plan.maximum_ac_candidates))
    return Overflow();
  TokenizationStoragePlan tokenization;
  status = ComputeTokenizationStoragePlan(
      blocks,
      {.exhaustive = exhaustive,
       .collect_fixed_populations =
           behavior == VarDctEntropyBehavior::kBalanced,
       .context_count = maps.maximum_ac_contexts,
       .order_count = plan.maximum_order_variants,
       .map_count = maps.maximum_maps,
       .workers = workers},
      &tokenization);
  if (!status.ok())
    return status;
  plan.ac_group_count = tokenization.ac_group_count;
  plan.dc_group_count = tokenization.dc_group_count;
  // AC reserve: 3*(64B + anchors), anchors <=B. DC: 3B residuals plus
  // 2*color_tiles +2*anchors+B metadata. DC boundaries align to color tiles.
  if (!Multiply(block_count, 195, &plan.maximum_ac_tokens) ||
      !Multiply(block_count, 6, &plan.maximum_dc_tokens) ||
      !AddScaled(color_tiles, 2, &plan.maximum_dc_tokens))
    return Overflow();
  const size_t g = plan.ac_group_count, d = plan.dc_group_count;
  if (d > std::numeric_limits<size_t>::max() / 2)
    return Overflow();
  SerializerHeaderStoragePlan headers;
  status = ComputeSerializerHeaderStoragePlan(g, d, maps, orders.maximum_tokens,
                                              &headers);
  if (!status.ok())
    return status;
  HostStorageBound control;
  status = ComputeSerializerControlStorageBound(
      plan, options, maps.maximum_maps, has_orders, &control);
  if (!status.ok())
    return status;
  auto &work = plan.working;
  if (!work.Add(orders.working) || !work.Add(maps.working) ||
      !work.Add(maps.map, plan.maximum_ac_candidates) ||
      !work.Add(tokenization.dc) || !work.Add(tokenization.ac) ||
      !work.Add(control))
    return Overflow();

  TaskStoragePlan dc_task, order_task, ac_task;
  status = ComputeTaskStoragePlan(plan.maximum_dc_tokens, kSimpleDcContextCount,
                                  2 * d, behavior, false, headers.dc_model,
                                  &dc_task);
  if (!status.ok())
    return status;
  if (has_orders) {
    status = ComputeTaskStoragePlan(orders.maximum_tokens,
                                    kSimplePermutationContextCount, 1, behavior,
                                    false, headers.order_model, &order_task);
    if (!status.ok())
      return status;
  }
  status = ComputeTaskStoragePlan(plan.maximum_ac_tokens,
                                  maps.maximum_ac_contexts, g, behavior,
                                  exhaustive, headers.ac_model, &ac_task);
  if (!status.ok())
    return status;
  if (!work.Add(dc_task.retained) || !work.Add(order_task.retained) ||
      !work.Add(ac_task.retained, plan.maximum_ac_candidates))
    return Overflow();
  // Retain completed tasks plus the largest simultaneously executing task
  // envelopes. Do not multiply AC-sized scratch by every map/order candidate,
  // or pretend the much smaller DC/order tasks also have the AC token count.
  std::array<size_t, 14> task_peaks{};
  if (plan.maximum_ac_candidates > task_peaks.size() - 2)
    return Status::Internal(
        "Serializer candidate bound requires a policy audit");
  size_t task_count = 0;
  task_peaks[task_count++] = dc_task.working.peak_bytes;
  if (has_orders)
    task_peaks[task_count++] = order_task.working.peak_bytes;
  for (size_t i = 0; i < plan.maximum_ac_candidates; ++i)
    task_peaks[task_count++] = ac_task.working.peak_bytes;
  std::sort(task_peaks.begin(), task_peaks.begin() + task_count,
            std::greater<size_t>{});
  for (size_t i = 0; i < std::min(workers, task_count); ++i)
    if (!work.Add({task_peaks[i], task_peaks[i]}))
      return Overflow();

  const Extent2D ac_extent{
      std::min(blocks.width, kVarDctAcGroupBlockDimension),
      std::min(blocks.height, kVarDctAcGroupBlockDimension)};
  const Extent2D dc_extent{
      std::min(blocks.width, kSimpleDcGroupBlockDimension),
      std::min(blocks.height, kSimpleDcGroupBlockDimension)};
  AcGroupTokenCounts ac_counts;
  DcGroupTokenCounts dc_counts;
  status = ComputeAcGroupTokenCounts(
      ac_extent, ac_extent.width * ac_extent.height, &ac_counts);
  if (!status.ok())
    return status;
  status = ComputeDcGroupTokenCounts(
      dc_extent, dc_extent.width * dc_extent.height, &dc_counts);
  if (!status.ok())
    return status;
  EntropyTokenEmissionStoragePlan ac_emission, dc_emission;
  // ANS's 47N+32 bits and reverse chunks dominate Prefix's 46N scratch.
  status = ComputeEntropyTokenEmissionStoragePlan(
      EntropyCodingMode::kAns, ac_counts.token_capacity, &ac_emission);
  if (!status.ok())
    return status;
  status = ComputeEntropyTokenEmissionStoragePlan(
      EntropyCodingMode::kAns,
      std::max(dc_counts.dc_tokens, dc_counts.metadata_tokens), &dc_emission);
  if (!status.ok())
    return status;
  HostStorageBound ac_writers, dc_writers;
  if (!ac_writers.Add(ac_emission.scratch, std::min(workers, g)) ||
      !dc_writers.Add(dc_emission.scratch, std::min(workers, d)))
    return Overflow();
  HostStorageBound write_scratch = Either(ac_writers, dc_writers);
  HostStorageBound dc_header = headers.dc_global_scratch;
  // Measurement owns a destination global writer besides the header's own
  // temporary. Selected section destinations are also bounded below.
  if (!AddWriter(headers.dc_global_bits, &dc_header))
    return Overflow();
  HostStorageBound common_measurement;
  if (!common_measurement.Add(
          dc_header, exhaustive ? std::min(workers, 2 * maps.maximum_maps) : 1))
    return Overflow();
  write_scratch = Either(write_scratch, common_measurement);
  write_scratch = Either(write_scratch, headers.ac_global_scratch);

  size_t logical_sections = 2;
  if (!AddScaled(g, 1, &logical_sections) ||
      !AddScaled(d, 1, &logical_sections))
    return Overflow();
  const size_t physical_sections = g == 1 ? 1 : logical_sections;
  // Two ANS states and at most 26 modular-header bits per DC group; one state
  // per AC group. Each logical writer may need up to seven padding bits.
  static_assert(kSimpleDcGroupBlockDimension == 256);
  size_t payload_bits = headers.dc_global_bits;
  if (!AddScaled(headers.ac_global_bits, 1, &payload_bits) ||
      !AddScaled(plan.maximum_ac_tokens, kAnsMaximumTokenBits, &payload_bits) ||
      !AddScaled(plan.maximum_dc_tokens, kAnsMaximumTokenBits, &payload_bits) ||
      !AddScaled(d, 2 * kAnsStreamStateBits + 26, &payload_bits) ||
      !AddScaled(g, kAnsStreamStateBits, &payload_bits) ||
      !AddScaled(logical_sections, 7, &payload_bits) ||
      payload_bits > std::numeric_limits<size_t>::max() - 7)
    return Overflow();
  const size_t payload_bytes = (payload_bits + 7) / 8;
  size_t padded_payload_bits = 0, prefix_bits = headers.frame_prefix_bits;
  // Match TOC's WithMaxBits(8 + 7 + 32*section_count), not its shorter
  // eventual bit count. Include its reservation in the final writer history.
  if (!Multiply(payload_bytes, 8, &padded_payload_bits) ||
      !AddScaled(1, 15, &prefix_bits) ||
      !AddScaled(physical_sections, 32, &prefix_bits))
    return Overflow();
  HostStorageBound prefix_measurement = headers.frame_scratch;
  if (!AddWriter(prefix_bits, &prefix_measurement))
    return Overflow();
  HostStorageBound all_prefix_measurements;
  if (exhaustive &&
      !all_prefix_measurements.Add(
          prefix_measurement, std::min(workers, plan.maximum_ac_candidates)))
    return Overflow();
  write_scratch = Either(write_scratch, all_prefix_measurements);
  HostStorageBound dc_measurement_headers;
  const size_t dc_measurement_workers = options.cpu_thread_count == 0
                                            ? 2 * std::min(workers, d)
                                            : std::min(workers, size_t{2});
  if (exhaustive &&
      !AddWriter(26, &dc_measurement_headers, dc_measurement_workers))
    return Overflow();
  write_scratch = Either(write_scratch, dc_measurement_headers);
  size_t assembly_bits = prefix_bits;
  if (!AddScaled(padded_payload_bits, 1, &assembly_bits) ||
      assembly_bits > std::numeric_limits<size_t>::max() - 7)
    return Overflow();
  plan.maximum_output_bytes = (assembly_bits + 7) / 8;
  if (!plan.output.AddVector<uint8_t>(plan.maximum_output_bytes, kFreshExact) ||
      // Sum of independent section writer growth peaks <= 3*sum(padded sizes).
      !work.AddVector<uint8_t>(payload_bytes, kGrowing) ||
      !work.Add(write_scratch) || !work.Add(headers.frame_scratch) ||
      !AddWriter(assembly_bits, &work) ||
      (g == 1 && !AddWriter(padded_payload_bits, &work)) ||
      !work.Add(plan.output))
    return Overflow();
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::codestream_internal
