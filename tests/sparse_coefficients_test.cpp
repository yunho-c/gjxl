// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include "codec/sparse_coefficients.h"

namespace {
using namespace gjxl;
namespace internal = vardct_frame_internal;
size_t cases = 0, ranges = 0, rejected = 0, large_shapes = 0;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T> void Case(size_t count, unsigned pattern) {
  static_assert(std::forward_iterator<typename SparseCoefficientSpan<T>::Iterator>);
  std::mt19937 rng(static_cast<unsigned>(count) * 71 + pattern);
  std::vector<T> logical(count);
  const size_t words = count / 64 + (count % 64 != 0);
  internal::SparseAcStorage<T> owner;
  owner.coefficient_count = count;
  owner.masks.assign(words, 0);
  owner.offsets.ResetForOverwrite(words);
  std::vector<size_t> permutation(words);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::shuffle(permutation.begin(), permutation.end(), rng);
  std::vector<T> values;
  for (size_t w : permutation) {
    owner.offsets.data()[w] = static_cast<uint32_t>(values.size());
    for (size_t bit = 0; bit < 64 && 64 * w + bit < count; ++bit) {
      const size_t i = 64 * w + bit;
      const bool active = pattern == 0 ? false : pattern == 1 ? true :
          pattern == 2 ? bit == 63 : pattern == 3 ? bit == 0 :
          pattern == 4 ? i == count - 1 : (rng() % 100 < (pattern - 4) * 20);
      if (!active) continue;
      const T value = i % 2 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
      logical[i] = value;
      owner.masks.data()[w] |= uint64_t{1} << bit;
      values.push_back(value);
    }
  }
  owner.values.ResetForOverwrite(values.size());
  std::copy(values.begin(), values.end(), owner.values.begin());
  Require(owner.Validate(false, 0).ok(), "Valid sparse storage rejected");
  const auto view = owner.view();
  Require(std::ranges::equal(view, logical), "Sparse iterator differs");
  for (size_t start = 0; start <= count; ++start) {
    for (size_t n : {size_t{0}, std::min<size_t>(1, count - start),
                     std::min<size_t>(63, count - start), std::min<size_t>(64, count - start), count - start}) {
      const auto part = view.subspan(start, n);
      const auto expected = std::span<const T>(logical).subspan(start, n);
      Require(std::ranges::equal(part, expected), "Sparse subspan differs");
      Require(part.NonzeroCount() == static_cast<size_t>(std::ranges::count_if(expected, [](T x) { return x != 0; })),
              "Sparse mask count differs");
      ++ranges;
    }
  }
  for (bool huge : {false, true}) {
    bool threw = false;
    try { (void)view.subspan(huge ? std::numeric_limits<size_t>::max() : count + 1, 0); }
    catch (const std::out_of_range&) { threw = true; }
    Require(threw, "Out-of-bounds sparse subspan accepted"); ++rejected;
  }
  const auto bad = [&](auto mutate) {
    auto broken = owner;
    mutate(broken);
    Require(!broken.Validate(false, 0).ok(), "Malformed sparse owner accepted");
    ++rejected;
  };
  bad([](auto& x) { x.coefficient_count = 0; });
  bad([](auto& x) { x.offsets.ResetForOverwrite(0); });
  bad([](auto& x) { x.offsets.data()[0] = UINT32_MAX; });
  if (count % 64) bad([&](auto& x) { x.masks.data()[words - 1] |= uint64_t{1} << (count % 64); });
  if (owner.values.size()) {
    bad([](auto& x) { x.values.data()[0] = 0; });
    bad([](auto& x) { x.values.ResetForOverwrite(0); });
    bad([](auto& x) { std::fill(x.masks.begin(), x.masks.end(), 0); });
    Require(!owner.Validate(true, static_cast<int32_t>(owner.values.data()[0])).ok(), "Sparse sentinel accepted");
    ++rejected;
  }
  std::vector<size_t> nonempty;
  for (size_t w = 0; w < words; ++w) if (owner.masks.data()[w]) nonempty.push_back(w);
  if (nonempty.size() >= 2) bad([&](auto& x) {
    x.offsets.data()[nonempty[1]] = x.offsets.data()[nonempty[0]];
  });
  ++cases;
}
}
int main() {
  try {
    const auto reject_large_empty = []<typename T>() {
      for(size_t count : {size_t{UINT32_MAX}, size_t{UINT32_MAX - 1}, size_t{UINT32_MAX - 62}}) {
        internal::SparseAcStorage<T> owner;owner.coefficient_count=count;
        Require(!owner.ValidShape(), "Large empty sparse shape accepted");
        Require(!owner.Validate(false, 0).ok(), "Large empty sparse validation succeeded");
        ++rejected;++large_shapes;
      }
    };
    reject_large_empty.template operator()<int8_t>();
    reject_large_empty.template operator()<int16_t>();
    reject_large_empty.template operator()<int32_t>();
    for (size_t count : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u, 1025u})
      for (unsigned pattern = 0; pattern < 9; ++pattern) {
        Case<int8_t>(count, pattern); Case<int16_t>(count, pattern); Case<int32_t>(count, pattern);
      }
    std::cout << "Sparse coefficients PASS cases=" << cases << " subranges=" << ranges
              << " rejected=" << rejected << " large_empty_shapes=" << large_shapes << '\n';
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
