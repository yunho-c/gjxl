// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/ac_population_reduction_internal.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using gjxl::codestream_internal::PreparedFixedAnsCluster;

// Independently tabulated HybridUint extra-bit widths, including transitions
// at 15/16 and every subsequent group of four symbols.
constexpr std::array<uint32_t, 128> kExtraBits = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,
  6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,
  10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,
  14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,
  18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,
  22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,
  26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,
};

template <size_t Shards>
void Check(const std::vector<uint32_t>& hist, size_t contexts) {
  std::vector<PreparedFixedAnsCluster> expected(contexts);
  // Traverse shard-major input rather than reproducing the reduction loop.
  for (size_t index = 0; index < hist.size(); ++index) {
    const size_t context = (index / 128) % contexts;
    const size_t symbol = index % 128;
    const uint64_t value = hist[index];
    expected[context].counts[symbol] += value;
    expected[context].token_count += value;
    expected[context].extra_bits += value * kExtraBits[symbol];
  }
  uint64_t total = 0;
  for (auto& cluster : expected) {
    // Descending first-nonzero search is independent of the max reduction.
    for (size_t symbol = 128; symbol-- != 0;) {
      if (cluster.counts[symbol] != 0) {
        cluster.maximum_symbol = static_cast<uint32_t>(symbol);
        break;
      }
    }
    total += cluster.token_count;
  }
  std::vector<PreparedFixedAnsCluster> actual(contexts);
  const auto reduced = gjxl::metal_internal::ReduceAcPopulations<Shards>(
      hist.data(), actual);
  if (reduced != total || actual != expected)
    throw std::runtime_error("AC population reduction differs from oracle");
}

template <size_t Shards>
size_t CheckAll() {
  std::mt19937 random(93847);
  size_t cases = 0;
  for (size_t contexts : {size_t{0}, size_t{1}, size_t{7}, size_t{495},
                          size_t{1485}, size_t{6930}}) {
    for (unsigned pattern = 0; pattern < 9; ++pattern) {
      std::vector<uint32_t> hist(Shards * contexts * 128);
      for (size_t i = 0; i < hist.size(); ++i) {
        const size_t symbol = i % 128;
        switch (pattern) {
          case 0: break;
          case 1: hist[i] = std::numeric_limits<uint32_t>::max(); break;
          case 2: hist[i] = symbol == 0; break;
          case 3: hist[i] = symbol == 15 ? 7 : 0; break;
          case 4: hist[i] = symbol == 16 ? 9 : 0; break;
          case 5: hist[i] = symbol == 127
                    ? std::numeric_limits<uint32_t>::max() : 0; break;
          case 6: hist[i] = random() % 100 == 0 ? random() : 0; break;
          case 7: hist[i] = random() % 4 == 0 ? random() : 0; break;
          case 8: hist[i] = random(); break;
        }
      }
      Check<Shards>(hist, contexts);
      ++cases;
    }
  }
  // Isolate every symbol and every shard, including all extra-bit boundaries.
  for (size_t shard = 0; shard < Shards; ++shard) {
    for (size_t symbol = 0; symbol < 128; ++symbol) {
      std::vector<uint32_t> hist(Shards * 3 * 128);
      hist[(shard * 3 + 1) * 128 + symbol] = 0xffffffffu;
      Check<Shards>(hist, 3);
      ++cases;
    }
  }
  return cases;
}
}  // namespace

int main() try {
  const size_t cases = CheckAll<1>() + CheckAll<2>() +
                       CheckAll<4>() + CheckAll<8>();
  std::cout << "Verified " << cases << " population reductions: all symbols, "
               "1/2/4/8 shards, sparse/dense inputs, wide sums and zero tails.\n";
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
