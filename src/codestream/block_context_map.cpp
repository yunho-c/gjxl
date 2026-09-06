// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's ac_context.h and enc_heuristics.cc.

#include "codestream/block_context_map.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/codestream.h"
#include "codec/vardct_frame_view_internal.h"
#include "codestream/simple_ac_context.h"

namespace gjxl {
using codestream_internal::Storage;
using vardct_frame_internal::VarDctFrameView;
namespace {

constexpr size_t kMinimumAdaptiveBlockCount = 1024;
constexpr size_t kMinimumQuantSplitBlockCount = 8192;
constexpr size_t kMaximumBlockContexts = 16;

constexpr std::array<uint8_t, 3 * codestream_internal::kSimpleCoefficientOrderCount>
kJxlDefaultBlockContextMap = {
  0, 1, 2, 2, 3,  3,  4,  5,  6,  6,  6,  6,  6,
  7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
  7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
};

Status AllocationFailure() {
  return Status::OutOfMemory("Block-context map allocation failed");
}

uint8_t ContextCount(std::span<const uint8_t> map) {
  return map.empty()
    ? 0
    : static_cast<uint8_t>(*std::max_element(map.begin(), map.end()) + 1);
}

void AddCandidate(
  SimpleBlockContextMap candidate,
  Storage<SimpleBlockContextMap>* maps) {

  if (std::ranges::find(*maps, candidate) == maps->end()) {
    maps->push_back(std::move(candidate));
  }
}

SimpleBlockContextMap MakeSingleContextMap() {
  SimpleBlockContextMap map;
  map.context_map.resize(
    3 * codestream_internal::kSimpleCoefficientOrderCount, 0);
  map.num_contexts = 1;
  return map;
}

Status CountRawQuantAndOrders(
  const VarDctFrameView& frame,
  std::array<std::array<size_t, 256>,
             codestream_internal::kSimpleCoefficientOrderCount>* counts,
  std::array<size_t, 256>* qf_counts,
  size_t* block_count) {

  if (counts == nullptr || qf_counts == nullptr || block_count == nullptr) {
    return Status::InvalidArgument(
      "Block-context occurrence output is null");
  }
  const Extent2D blocks = frame.geometry().block_grid().blocks;
  if (!blocks.try_area(block_count)) {
    return Status::InvalidArgument("Block-context dimensions overflow");
  }
  const ConstPlaneI32View raw_quant = frame.raw_quant_field();
  for (size_t y = 0; y < blocks.height; ++y) {
    const int32_t* qf_row = raw_quant.Row(y);
    for (size_t x = 0; x < blocks.width; ++x) {
      AcStrategyCell cell;
      Status status = frame.strategies().Get(x, y, &cell);
      if (!status.ok()) {
        return status;
      }
      if (qf_row[x] < 1 || qf_row[x] > 256) {
        return Status::InvalidArgument(
          "Block-context raw quantization is out of range");
      }
      const size_t order = codestream_internal::kSimpleStrategyOrder[
        static_cast<size_t>(cell.strategy)];
      const size_t qf = static_cast<size_t>(qf_row[x] - 1);
      ++(*counts)[order][qf];
      ++(*qf_counts)[qf];
    }
  }
  return Status::Ok();
}

Storage<uint32_t> MedianQuantThreshold(
  const std::array<size_t, 256>& qf_counts,
  size_t block_count) {

  if (block_count < kMinimumQuantSplitBlockCount) {
    return {};
  }
  size_t cumulative = 0;
  for (size_t qf = 0; qf < qf_counts.size(); ++qf) {
    cumulative += qf_counts[qf];
    if (cumulative > block_count / 2) {
      if (qf == 0 || qf > 255) {
        return {};
      }
      return {static_cast<uint32_t>(qf)};
    }
  }
  return {};
}

SimpleBlockContextMap BuildAdaptiveMap(
  const std::array<std::array<size_t, 256>,
                   codestream_internal::kSimpleCoefficientOrderCount>& counts,
  std::span<const uint32_t> qf_thresholds,
  size_t block_count) {

  const size_t segment_count = qf_thresholds.size() + 1;
  const size_t cell_count =
    codestream_internal::kSimpleCoefficientOrderCount * segment_count;
  Storage<size_t> cell_counts(cell_count, 0);
  for (size_t order = 0;
       order < codestream_internal::kSimpleCoefficientOrderCount; ++order) {
    for (size_t qf = 0; qf < 256; ++qf) {
      size_t segment = 0;
      while (segment < qf_thresholds.size() &&
             static_cast<uint32_t>(qf + 1) > qf_thresholds[segment]) {
        ++segment;
      }
      cell_counts[order * segment_count + segment] += counts[order][qf];
    }
  }

  Storage<uint8_t> remap(cell_count);
  std::iota(remap.begin(), remap.end(), uint8_t{0});
  Storage<uint8_t> clusters = remap;
  const size_t desired_luma = std::clamp<size_t>(block_count / 2048, 2, 9);
  std::array<uint8_t, 256> previous_rank{};
  while (clusters.size() > desired_luma) {
    // Preserve this iteration's incoming order on ties, without an untracked
    // stable_sort temporary. Numeric cluster order is not equivalent after a merge.
    for (size_t index = 0; index < clusters.size(); ++index) {
      previous_rank[clusters[index]] = static_cast<uint8_t>(index);
    }
    std::sort(
      clusters.begin(), clusters.end(),
      [&](uint8_t left, uint8_t right) {
        return cell_counts[left] != cell_counts[right]
          ? cell_counts[left] > cell_counts[right]
          : previous_rank[left] < previous_rank[right];
      });
    const uint8_t destination = clusters[clusters.size() - 2];
    const uint8_t source = clusters.back();
    cell_counts[destination] += cell_counts[source];
    cell_counts[source] = 0;
    remap[source] = destination;
    clusters.pop_back();
  }
  for (size_t index = 0; index < remap.size(); ++index) {
    while (remap[remap[index]] != remap[index]) {
      remap[index] = remap[remap[index]];
    }
  }
  Storage<uint8_t> labels(remap.size(), 0xFF);
  uint8_t next_label = 0;
  for (size_t index = 0; index < remap.size(); ++index) {
    if (labels[remap[index]] == 0xFF) {
      labels[remap[index]] = next_label++;
    }
    remap[index] = labels[remap[index]];
  }

  SimpleBlockContextMap map;
  map.qf_thresholds.assign(qf_thresholds.begin(), qf_thresholds.end());
  map.context_map = remap;
  map.context_map.resize(3 * remap.size());
  const uint8_t chroma_contexts = static_cast<uint8_t>(
    std::clamp<size_t>(block_count / 3072, 1, 5));
  for (size_t index = remap.size(); index < map.context_map.size(); ++index) {
    map.context_map[index] = static_cast<uint8_t>(
      next_label + std::min<uint8_t>(
        remap[index % remap.size()], chroma_contexts - 1));
  }
  map.num_contexts = ContextCount(map.context_map);
  return map;
}

}  // namespace

SimpleBlockContextMap DefaultSimpleBlockContextMap() {
  SimpleBlockContextMap map;
  map.context_map.assign(
    codestream_internal::kSimpleBlockContextMap.begin(),
    codestream_internal::kSimpleBlockContextMap.end());
  map.num_contexts = ContextCount(map.context_map);
  return map;
}

SimpleBlockContextMap JxlDefaultSimpleBlockContextMap() {
  SimpleBlockContextMap map;
  map.context_map.assign(
    kJxlDefaultBlockContextMap.begin(), kJxlDefaultBlockContextMap.end());
  map.num_contexts = ContextCount(map.context_map);
  return map;
}

bool codestream_internal::IsJxlDefaultBlockContextMap(
  const SimpleBlockContextMap& map) noexcept {
  return map.qf_thresholds.empty() &&
    map.num_contexts == ContextCount(kJxlDefaultBlockContextMap) &&
    std::ranges::equal(map.context_map, kJxlDefaultBlockContextMap);
}

SimpleBlockContextMap TwoChannelSimpleBlockContextMap() {
  SimpleBlockContextMap map;
  map.context_map.resize(
    3 * codestream_internal::kSimpleCoefficientOrderCount, 1);
  std::fill_n(
    map.context_map.begin(),
    codestream_internal::kSimpleCoefficientOrderCount,
    uint8_t{0});
  map.num_contexts = 2;
  return map;
}

Status ValidateSimpleBlockContextMap(const SimpleBlockContextMap& map) {
  if (map.qf_thresholds.size() > 15 || map.num_contexts == 0 ||
      map.num_contexts > kMaximumBlockContexts) {
    return Status::InvalidArgument("Block-context map dimensions are invalid");
  }
  for (size_t index = 0; index < map.qf_thresholds.size(); ++index) {
    if (map.qf_thresholds[index] == 0 || map.qf_thresholds[index] > 255 ||
        (index != 0 &&
         map.qf_thresholds[index - 1] >= map.qf_thresholds[index])) {
      return Status::InvalidArgument(
        "Block-context quantization thresholds are invalid");
    }
  }
  const size_t expected =
    3 * codestream_internal::kSimpleCoefficientOrderCount *
    (map.qf_thresholds.size() + 1);
  if (map.context_map.size() != expected ||
      ContextCount(map.context_map) != map.num_contexts) {
    return Status::InvalidArgument("Block-context map labels are invalid");
  }
  std::array<bool, kMaximumBlockContexts> seen{};
  for (uint8_t context : map.context_map) {
    if (context >= map.num_contexts) {
      return Status::InvalidArgument(
        "Block-context map references an absent context");
    }
    seen[context] = true;
  }
  for (size_t context = 0; context < map.num_contexts; ++context) {
    if (!seen[context]) {
      return Status::InvalidArgument(
        "Block-context map labels are not canonical");
    }
  }
  return Status::Ok();
}

Status SimpleBlockContext(
  const SimpleBlockContextMap& map,
  AcStrategyType strategy,
  size_t channel,
  int32_t raw_quant,
  uint32_t* context) {

  if (context == nullptr || channel >= 3 ||
      static_cast<size_t>(strategy) >= kAcStrategyCount ||
      raw_quant < 1 || raw_quant > 256) {
    return Status::InvalidArgument("Block-context input is invalid");
  }
  Status status = ValidateSimpleBlockContextMap(map);
  if (!status.ok()) {
    return status;
  }
  size_t qf_segment = 0;
  while (qf_segment < map.qf_thresholds.size() &&
         static_cast<uint32_t>(raw_quant) >
           map.qf_thresholds[qf_segment]) {
    ++qf_segment;
  }
  const size_t channel_row = channel < 2 ? channel ^ 1u : 2;
  const size_t order = codestream_internal::kSimpleStrategyOrder[
    static_cast<size_t>(strategy)];
  const size_t segment_count = map.qf_thresholds.size() + 1;
  *context = map.context_map[
    (channel_row * codestream_internal::kSimpleCoefficientOrderCount + order) *
      segment_count + qf_segment];
  return Status::Ok();
}

Status ComputeSimpleBlockContextMapCandidates(
  const VarDctEncoderFrame& frame,
  Storage<SimpleBlockContextMap>* maps) {

  if (maps == nullptr) {
    return Status::InvalidArgument("Block-context candidate output is null");
  }
  Status status = ValidateSimpleCodestreamFrame(frame);
  if (!status.ok()) {
    return status;
  }
  return codestream_internal::ComputeSimpleBlockContextMapCandidatesForEncoder(
    vardct_frame_internal::BorrowFrame(frame), maps);
}

Status codestream_internal::ComputeSimpleBlockContextMapCandidatesForEncoder(
  const VarDctFrameView& frame,
  Storage<SimpleBlockContextMap>* maps) {

  if (maps == nullptr) {
    return Status::InvalidArgument("Block-context candidate output is null");
  }
  Status status;
  try {
    Storage<SimpleBlockContextMap> candidate;
    AddCandidate(DefaultSimpleBlockContextMap(), &candidate);
    size_t block_count = 0;
    const Extent2D blocks = frame.geometry().block_grid().blocks;
    if (!blocks.try_area(&block_count)) {
      return Status::InvalidArgument("Block-context dimensions overflow");
    }
    if (block_count >= kMinimumAdaptiveBlockCount) {
      AddCandidate(JxlDefaultSimpleBlockContextMap(), &candidate);
      AddCandidate(TwoChannelSimpleBlockContextMap(), &candidate);
      AddCandidate(MakeSingleContextMap(), &candidate);
      std::array<std::array<size_t, 256>,
                 codestream_internal::kSimpleCoefficientOrderCount> counts{};
      std::array<size_t, 256> qf_counts{};
      status = CountRawQuantAndOrders(
        frame, &counts, &qf_counts, &block_count);
      if (!status.ok()) {
        return status;
      }
      AddCandidate(BuildAdaptiveMap(counts, {}, block_count), &candidate);
      const Storage<uint32_t> thresholds =
        MedianQuantThreshold(qf_counts, block_count);
      if (!thresholds.empty()) {
        AddCandidate(
          BuildAdaptiveMap(counts, thresholds, block_count), &candidate);
      }
    }
    for (const SimpleBlockContextMap& map : candidate) {
      status = ValidateSimpleBlockContextMap(map);
      if (!status.ok()) {
        return status;
      }
    }
    *maps = std::move(candidate);
  } catch (const resource_budget_internal::ManagedAllocationFailure& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status ComputeSimpleBlockContextMap(
  const VarDctEncoderFrame& frame,
  SimpleBlockContextMap* map) {

  if (map == nullptr) {
    return Status::InvalidArgument("Block-context map output is null");
  }
  Status status = ValidateSimpleCodestreamFrame(frame);
  if (!status.ok()) {
    return status;
  }
  return codestream_internal::ComputeSimpleBlockContextMapForEncoder(
    vardct_frame_internal::BorrowFrame(frame), map);
}

Status codestream_internal::ComputeSimpleBlockContextMapForEncoder(
  const VarDctFrameView& frame,
  SimpleBlockContextMap* map) {

  if (map == nullptr) {
    return Status::InvalidArgument("Block-context map output is null");
  }
  Status status;
  try {
    size_t block_count = 0;
    const Extent2D blocks = frame.geometry().block_grid().blocks;
    if (!blocks.try_area(&block_count)) {
      return Status::InvalidArgument("Block-context dimensions overflow");
    }
    SimpleBlockContextMap candidate;
    if (block_count < kMinimumAdaptiveBlockCount) {
      candidate = DefaultSimpleBlockContextMap();
    } else {
      std::array<std::array<size_t, 256>,
                 codestream_internal::kSimpleCoefficientOrderCount> counts{};
      std::array<size_t, 256> qf_counts{};
      status = CountRawQuantAndOrders(
        frame, &counts, &qf_counts, &block_count);
      if (!status.ok()) {
        return status;
      }
      const Storage<uint32_t> thresholds =
        MedianQuantThreshold(qf_counts, block_count);
      candidate = BuildAdaptiveMap(counts, thresholds, block_count);
    }
    status = ValidateSimpleBlockContextMap(candidate);
    if (!status.ok()) {
      return status;
    }
    *map = std::move(candidate);
  } catch (const resource_budget_internal::ManagedAllocationFailure& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

}  // namespace gjxl
