// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/entropy_storage_plan.h"

#include <algorithm>
#include <limits>
#include <unordered_map>

#include "codestream/entropy_internal.h"

namespace gjxl::codestream_internal {
namespace {
using enum resource_budget_internal::VectorCapacityPolicy;
// Bit and tree bounds below describe this coding profile. A profile expansion
// requires a new proof, not just larger array allocations.
static_assert(kAnsTableSize == 4096 && kMaximumAnsAlphabetSize == 256 &&
              kPrefixAlphabetSize == 128 && kMaximumPrefixClusters == 32);

Status Overflow() {
  return Status::OutOfMemory("Entropy storage bound overflows");
}

bool Bits(size_t count, size_t per_item, size_t fixed, size_t *out) {
  if (count > (std::numeric_limits<size_t>::max() - fixed) / per_item)
    return false;
  *out = count * per_item + fixed;
  return true;
}
} // namespace

Status ComputeEntropyWriterStorageBound(size_t maximum_bits,
                                        HostStorageBound *out) {
  if (out == nullptr)
    return Status::InvalidArgument("Entropy writer bound output is null");
  // Match BitWriter's checked padded-byte extent, including its +7 limit.
  HostStorageBound bound;
  if (maximum_bits > std::numeric_limits<size_t>::max() - 7 ||
      !bound.AddVector<uint8_t>((maximum_bits + 7) / 8, kGrowing))
    return Overflow();
  *out = bound;
  return Status::Ok();
}

Status
ComputeEntropyAggregationStoragePlan(size_t maximum_values,
                                     EntropyAggregationStoragePlan *out) {
  if (out == nullptr)
    return Status::InvalidArgument("Entropy aggregation plan output is null");
  EntropyAggregationStoragePlan plan;
  if (maximum_values < kEntropyMinimumCountingInput) {
    if (!plan.output.AddVector<WeightedValue>(maximum_values, kGrowing))
      return Overflow();
  } else {
    constexpr uint64_t value_domain = uint64_t{1} << 32;
    const size_t unique =
        static_cast<size_t>(std::min<uint64_t>(maximum_values, value_domain));
    const size_t sparse = static_cast<size_t>(std::min<uint64_t>(
        maximum_values, value_domain - kEntropyDenseValueCount));
    // The concrete libc++ rebound node, not sizeof(value_type) or an estimated
    // allocator overhead. HostStorageBound gates the reviewed libc++ C++20 ABI.
    using Node =
        std::__hash_node<std::__hash_value_type<uint32_t, uint64_t>, void *>;
    // operator[] starts empty, inserts only new keys, and keeps load factor 1.
    // At rehash old buckets < new unique count U; requested buckets <= 2U.
    // Next-prime rounding is <2x (first allocation is 2). Thus current <=4U
    // and old+replacement <=5U pointer slots. See the documented proof/audit.
    if (!plan.output.AddVector<WeightedValue>(unique, kFreshExact) ||
        !plan.scratch.AddVector<uint64_t>(kEntropyDenseValueCount,
                                          kFreshExact) ||
        !plan.scratch.Add({sizeof(Node), sizeof(Node)}, sparse) ||
        !plan.scratch.Add({4 * sizeof(void *), 5 * sizeof(void *)}, sparse))
      return Overflow();
  }
  *out = plan;
  return Status::Ok();
}

Status ComputeEntropyModelStoragePlan(EntropyCodingMode mode, size_t contexts,
                                      size_t clusters,
                                      EntropyModelStoragePlan *out) {
  if (out == nullptr || contexts == 0 || contexts > UINT32_MAX ||
      clusters == 0 || clusters > kMaximumPrefixClusters ||
      (mode != EntropyCodingMode::kPrefix && mode != EntropyCodingMode::kAns))
    return Status::InvalidArgument("Entropy model plan arguments are invalid");
  EntropyModelStoragePlan plan;
  // Config <=12 bits, alphabet <=20, tree header <=2+18*4, and at most
  // 2*128 RLE entries of depth <=5 plus <=3 extra bits. Simple trees are
  // smaller.
  constexpr size_t prefix_cluster_bits = 12 + 20 + 2 + 18 * 4 + 2 * 128 * 8;
  size_t context_bits = 0;
  if (!Bits(contexts, 15 + 31, 3 + 1 + prefix_cluster_bits, &context_bits))
    return Overflow();
  // ANS histogram: header <=20, each of 256 symbols <=7 depth +7 repeat
  // marker +11 repeat count +12 population bits. Flat/small forms are smaller.
  const size_t payload =
      mode == EntropyCodingMode::kPrefix
          ? 1 + clusters * prefix_cluster_bits
          : 3 + clusters * (12 + 20 + 256 * (7 + 7 + 11 + 12));
  if (context_bits > std::numeric_limits<size_t>::max() - payload)
    return Overflow();
  plan.maximum_bits = context_bits + payload;
  if (!plan.owned.AddVector<uint8_t>(contexts, kFreshExact) ||
      !plan.owned.AddVector<HybridUintConfig>(clusters, kFreshExact))
    return Overflow();
  if (mode == EntropyCodingMode::kPrefix) {
    if (!plan.owned.AddVector<PrefixCode>(clusters, kFreshExact))
      return Overflow();
  } else {
    HostStorageBound histogram;
    if (!histogram.AddVector<uint16_t>(kMaximumAnsAlphabetSize, kFreshExact) ||
        !histogram.AddVector<Storage<uint16_t>>(kMaximumAnsAlphabetSize,
                                                kFreshExact) ||
        !histogram.AddVector<uint16_t>(kAnsTableSize, kFreshExact) ||
        !histogram.AddVector<uint64_t>(kMaximumAnsAlphabetSize, kFreshExact) ||
        !plan.owned.AddVector<AnsHistogram>(clusters, kFreshExact) ||
        !plan.owned.Add(histogram, clusters))
      return Overflow();
  }
  HostStorageBound context_writer, model_writer;
  Status status =
      ComputeEntropyWriterStorageBound(context_bits, &context_writer);
  if (!status.ok())
    return status;
  status = ComputeEntropyWriterStorageBound(plan.maximum_bits, &model_writer);
  if (!status.ok())
    return status;
  // Prefix: outer temporary, best and candidate context writers. ANS adds
  // WriteContextMap's public-wrapper temporary. Huffman RLE is serial.
  if (!plan.write_scratch.Add(model_writer) ||
      !plan.write_scratch.Add(context_writer,
                              mode == EntropyCodingMode::kPrefix ? 2 : 3) ||
      !plan.write_scratch.AddVector<uint8_t>(2 * kPrefixAlphabetSize,
                                             kFreshExact, 2))
    return Overflow();
  *out = plan;
  return Status::Ok();
}

Status ComputeEntropyOptimizationStoragePlan(
    const EntropyOptimizationStorageOptions &options,
    EntropyOptimizationStoragePlan *out) {
  using enum EntropyStoragePolicy;
  if (out == nullptr || options.contexts == 0 ||
      options.contexts > UINT32_MAX || options.initial_histograms > 256)
    return Status::InvalidArgument("Entropy optimization plan is invalid");
  switch (options.policy) {
  case kFastPrefix:
  case kPrefix:
    if (options.borrow_prepared_clusters ||
        (options.retain_prepared_clusters &&
         (options.policy == kFastPrefix || !options.return_cost)))
      return Status::InvalidArgument("Prefix storage options are invalid");
    return ComputePrefixOptimizationStoragePlan(options, out);
  case kBalancedAns:
  case kHighDensityAns:
  case kAnsFromPrefix:
  case kDeferredAnsFromPrefix:
    if (options.retain_prepared_clusters ||
        (options.borrow_prepared_clusters && options.policy != kAnsFromPrefix &&
         options.policy != kDeferredAnsFromPrefix) ||
        (options.policy == kDeferredAnsFromPrefix &&
         !options.borrow_prepared_clusters))
      return Status::InvalidArgument("ANS storage options are invalid");
    return ComputeAnsOptimizationStoragePlan(options, out);
  }
  return Status::InvalidArgument("Entropy storage policy is invalid");
}

} // namespace gjxl::codestream_internal
