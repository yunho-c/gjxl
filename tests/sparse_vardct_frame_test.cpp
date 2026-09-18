// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>

#include "codec/reconstruction.h"
#include "core/image_buffer.h"
#include "sparse_frame_fixture.h"

namespace {
using namespace gjxl;
using gjxl_test::Check;
namespace internal = vardct_frame_internal;
size_t cases = 0, encodes = 0, rejections = 0, reconstructions = 0;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T> void Case(size_t strategy, size_t pattern, size_t side) {
  const auto oracle = gjxl_test::SparseOracle<T>(strategy, pattern, side);
  gjxl_test::PopulationAssembly layout(oracle);
  auto owner = gjxl_test::SparseStorage<T>(oracle);
  const auto* original_masks = owner.masks.data();
  const auto* original_values = owner.values.data();
  const size_t original_bytes = owner.bytes();
  auto input = gjxl_test::SparseInput(oracle, layout.layouts, &owner);
  VarDctEncoderFrame actual = oracle;
  const auto reject = [&](auto invalid) {
    Require(!internal::AssembleVarDctEncoderFrame(invalid, &actual).ok(), "Invalid sparse assembly accepted");
    Require(owner.masks.data() == original_masks && owner.values.data() == original_values &&
              owner.bytes() == original_bytes, "Failed sparse assembly consumed owner");
    gjxl_test::EqualSparseCoefficients(oracle, actual);
    ++rejections;
  };
  auto bad = input; bad.quantizer = nullptr; reject(bad);
  auto bad_layouts = layout.layouts; ++bad_layouts.front().coefficient_offsets[0];
  bad = input; bad.transforms = bad_layouts; reject(bad);
  const uint32_t offset = owner.offsets.data()[0];
  owner.offsets.data()[0] = static_cast<uint32_t>(owner.values.size() + 1);
  reject(input); owner.offsets.data()[0] = offset;
  if (owner.values.size() != 0) {
    const T value = owner.values.data()[0]; owner.values.data()[0] = 0;
    reject(input); owner.values.data()[0] = value;
  }
  Check(internal::AssembleVarDctEncoderFrame(input, &actual));
  Require(owner.masks.size() == 0 && owner.offsets.size() == 0 && owner.values.size() == 0, "Sparse owner not consumed");
  const auto info = internal::GetAcStorageInfo(actual);
  Require(info.sparse && info.coefficient_bytes == sizeof(T) && info.native_bytes == original_bytes,
          "Sparse ownership profile differs");
  gjxl_test::EqualSparseCoefficients(oracle, actual);
  auto copy = actual;
  auto moved = std::move(copy);
  Require(!copy.valid(), "Moved sparse frame remains valid");
  gjxl_test::EqualSparseCoefficients(oracle, moved);
  VarDctAcGroupView unsupported{.block_x = 123};
  Require(!actual.GetAcGroup(0, &unsupported).ok() && unsupported.block_x == 123,
          "Sparse query expanded dense storage");

  auto population = gjxl_test::ReferencePopulation(oracle);
  const auto expected_population = population;
  auto cached_owner = gjxl_test::SparseStorage<T>(oracle);
  auto cached_input = gjxl_test::SparseInput(oracle, layout.layouts, &cached_owner, &population);
  VarDctEncoderFrame cached;
  Check(internal::AssembleVarDctEncoderFrame(cached_input, &cached));
  population.counts.fill(UINT32_MAX);
  population.present_mask = 0;
  const auto* stored_population = internal::GetCoefficientOrderPopulation(cached);
  Require(stored_population && stored_population->counts == expected_population.counts &&
              stored_population->present_mask == expected_population.present_mask,
          "Sparse population cache differs");
  for (auto behavior : {VarDctCoefficientOrderBehavior::kFull,
                        VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled}) {
    VarDctCodestreamOptions options{.coefficient_order_behavior = behavior};
    std::vector<uint8_t> expected, a, b;
    Check(EncodeVarDctCodestream(oracle, options, &expected));
    Check(EncodeVarDctCodestream(actual, options, &a));
    Check(EncodeVarDctCodestream(cached, options, &b));
    Require(a == expected && b == expected, "Sparse codestream differs");
    encodes += 2;
  }
  Image3FBuffer expected(oracle.geometry().padded_frame()), decoded(oracle.geometry().padded_frame());
  Check(ReconstructQuantizedCoefficients(oracle, expected.view()));
  Check(ReconstructQuantizedCoefficients(actual, decoded.view()));
  const auto extent = oracle.geometry().padded_frame();
  for (size_t c = 0; c < 3; ++c) for (size_t y = 0; y < extent.height; ++y)
    Require(std::memcmp(expected.view().plane[c].Row(y), decoded.view().plane[c].Row(y), extent.width * sizeof(float)) == 0,
            "Sparse reconstruction differs bitwise");
  ++reconstructions;
  std::atomic<bool> okay{true};
  std::array<std::thread, 3> readers;
  for (auto& reader : readers) reader = std::thread([&] {
    try { gjxl_test::EqualSparseCoefficients(oracle, actual); gjxl_test::EqualSparseCoefficients(oracle, moved); }
    catch (...) { okay = false; }
  });
  for (auto& reader : readers) reader.join();
  Require(okay, "Concurrent sparse readers differ");
  ++cases;
}
}

int main() {
  try {
    for (size_t side : {4u, 36u}) for (size_t strategy = 0; strategy < 8; ++strategy)
      for (size_t pattern = 0; pattern < 8; ++pattern) {
        Case<int8_t>(strategy, pattern, side);
        Case<int16_t>(strategy, pattern, side);
        Case<int32_t>(strategy, pattern, side);
      }
    std::cout << "Sparse frame PASS cases=" << cases << " exact_streams=" << encodes
              << " atomic_failures=" << rejections << " bitwise_reconstructions=" << reconstructions << '\n';
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
