// SPDX-License-Identifier: Apache-2.0
// CPU reduction of the Metal tokenizer's fixed 128-bin histograms.
#pragma once
#include "codestream/entropy_internal.h"
#include <algorithm>
#include <span>

namespace gjxl::metal_internal {
// Output clusters must be value-initialized, including the unused upper 128
// bins. Accumulate in 64 bits before summing shards; GPU bins are uint32_t.
template <size_t Shards>
uint64_t ReduceAcPopulations(const uint32_t* hist,
    std::span<codestream_internal::PreparedFixedAnsCluster> output) {
  uint64_t total = 0;
  const size_t stride = output.size() * 128;
  for (size_t context = 0; context < output.size(); ++context) {
    auto& dst = output[context];
    uint64_t tokens = 0, extra = 0;
    uint32_t maximum = 0;
    for (size_t symbol = 0; symbol < 128; ++symbol) {
      uint64_t count = 0;
      for (size_t shard = 0; shard < Shards; ++shard)
        count += hist[shard * stride + context * 128 + symbol];
      dst.counts[symbol] = count;
      tokens += count;
      maximum = std::max(maximum, count ? static_cast<uint32_t>(symbol) : 0u);
      if (symbol >= 16) extra += count * (2 + ((symbol - 16) >> 2));
    }
    dst.token_count = tokens;
    dst.extra_bits = extra;
    dst.maximum_symbol = maximum;
    total += tokens;
  }
  return total;
}
}  // namespace gjxl::metal_internal
