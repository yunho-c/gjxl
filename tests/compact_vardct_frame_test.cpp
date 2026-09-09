// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

#include "codec/reconstruction.h"
#include "codestream/ac_group.h"
#include "coefficient_order_population_fixture.h"
#include "core/image_buffer.h"
#include "quantized_frame_fixture.h"

namespace {
using namespace gjxl;
namespace internal = vardct_frame_internal;
using gjxl_test::Check;

void Require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

template <typename T>
internal::QuantizedFrameAssemblyInputT<T>
Input(const VarDctEncoderFrame &frame, OverwriteArray<T> &owner,
      std::span<const internal::QuantizedAcTransformLayout> layouts, bool owned,
      const internal::CoefficientOrderPopulation *population = nullptr) {
  return {.geometry = frame.geometry(),
          .strategies = &frame.strategies(),
          .raw_quant_field = frame.raw_quant_field(),
          .quantizer = &frame.quantizer(),
          .y_to_x = frame.color_correlation().y_to_x_map(),
          .y_to_b = frame.color_correlation().y_to_b_map(),
          .epf_sharpness = frame.epf_sharpness(),
          .profile = frame.profile(),
          .quantized_dc = frame.quantized_dc(),
          .quantized_ac = {owner.data(), owner.size()},
          .transforms = layouts,
          .ac_group_storage = owned ? &owner : nullptr,
          .coefficient_order_population = population};
}

template <typename T>
VarDctEncoderFrame Oracle(size_t strategy, size_t pattern, size_t side) {
  auto seed = gjxl_test::MakeFrame(strategy, 0, side);
  gjxl_test::PopulationAssembly assembly(seed);
  for (const auto &transform : assembly.layouts) {
    for (size_t channel = 0; channel < 3; ++channel) {
      auto *data =
          assembly.coefficients.data() + transform.coefficient_offsets[channel];
      for (size_t i = 0; i < transform.coefficient_count; ++i) {
        const size_t selector =
            (i * 13 + channel * 7 + transform.block_x + transform.block_y) % 19;
        if (pattern == 0)
          data[i] = 0;
        if (pattern == 1)
          data[i] = selector == 0 ? -17 : selector == 1 ? 29 : 0;
        if (pattern == 2)
          data[i] = selector == 0  ? 0
                    : selector & 1 ? std::numeric_limits<T>::min()
                                   : std::numeric_limits<T>::max();
      }
    }
  }
  VarDctEncoderFrame result;
  Check(internal::AssembleVarDctEncoderFrame(assembly.Input(nullptr), &result));
  return result;
}

const void *FirstPointer(const VarDctEncoderFrame &frame) {
  VarDctNativeAcGroupView group;
  Check(frame.GetNativeAcGroup(0, &group));
  return std::visit(
      [](const auto &g) -> const void * {
        if constexpr (requires { g.coefficients[0].data(); }) return g.coefficients[0].data();
        else throw std::runtime_error("Expected dense compact test storage");
      },
      group);
}

void EqualCoefficients(const VarDctEncoderFrame &expected,
                       const VarDctEncoderFrame &actual) {
  Require(actual.valid(), "Compact frame invalid");
  Require(actual.ac_group_count() == expected.ac_group_count(),
          "Group count differs");
  for (size_t i = 0; i < expected.ac_group_count(); ++i) {
    VarDctAcGroupView dense;
    VarDctNativeAcGroupView native;
    Check(expected.GetAcGroup(i, &dense));
    Check(actual.GetNativeAcGroup(i, &native));
    std::visit(
        [&](const auto &group) {
          Require(group.block_x == dense.block_x &&
                      group.block_y == dense.block_y &&
                      group.block_extent == dense.block_extent &&
                      group.used_coefficient_count ==
                          dense.used_coefficient_count,
                  "Group metadata differs");
          for (size_t c = 0; c < 3; ++c)
            Require(std::ranges::equal(group.coefficients[c],
                                       dense.coefficients[c]),
                    "Group values/tails differ");
        },
        native);
  }
}

size_t cases = 0, streams = 0, failures = 0, reconstructions = 0;

template <typename T> void Case(size_t strategy, size_t pattern, size_t side) {
  const auto oracle = Oracle<T>(strategy, pattern, side);
  const auto population = gjxl_test::ReferencePopulation(oracle);
  gjxl_test::PopulationAssembly assembly(oracle);
  OverwriteArray<T> owner;
  owner.ResetForOverwrite(assembly.coefficients.size());
  std::copy(assembly.coefficients.begin(), assembly.coefficients.end(),
            owner.begin());
  const T *transferred = owner.data();
  auto input = Input(oracle, owner, assembly.layouts, true);
  VarDctEncoderFrame actual = oracle;
  const void *original = FirstPointer(actual);
  const auto reject = [&](auto invalid) {
    Require(!internal::AssembleVarDctEncoderFrame(invalid, &actual).ok(),
            "Invalid assembly accepted");
    Require(owner.data() == transferred &&
                owner.size() == assembly.coefficients.size() &&
                FirstPointer(actual) == original,
            "Failed assembly consumed owner/output");
    ++failures;
  };
  auto invalid = input;
  invalid.quantized_ac = input.quantized_ac.subspan(1);
  reject(invalid);
  invalid = input;
  invalid.quantizer = nullptr;
  reject(invalid);
  invalid = input;
  auto layouts = assembly.layouts;
  ++layouts.front().coefficient_offsets[0];
  invalid.transforms = layouts;
  reject(invalid);
  invalid = input;
  auto bad_population = population;
  bad_population.present_mask ^= 0x80;
  invalid.coefficient_order_population = &bad_population;
  reject(invalid);
  if (side == 36 || side == 4) {
    const T tail = owner.data()[owner.size() - 1];
    Require(tail == 0, "Fixture has no final tail");
    owner.data()[owner.size() - 1] = 1;
    reject(input);
    owner.data()[owner.size() - 1] = tail;
  }
  Check(internal::AssembleVarDctEncoderFrame(input, &actual));
  Require(owner.size() == 0 && FirstPointer(actual) == transferred,
          "Assembly did not transfer exact owner");
  const auto bytes = internal::GetAcStorageInfo(actual);
  Require(bytes.coefficient_bytes == sizeof(T) &&
              bytes.native_bytes == assembly.coefficients.size() * sizeof(T),
          "Authoritative storage width/bytes differ");
  EqualCoefficients(oracle, actual);

  auto copied = actual;
  Require(FirstPointer(copied) != transferred,
          "Copied frame shares mutable storage");
  auto moved = std::move(copied);
  Require(!copied.valid(), "Moved-from frame remains valid");
  EqualCoefficients(oracle, moved);
  VarDctAcGroupView wrong{.block_x = 123};
  Require(!actual.GetAcGroup(0, &wrong).ok() && wrong.block_x == 123,
          "Narrow typed query silently expanded or changed output");
  VarDctNativeAcGroupView invalid_group = wrong;
  Require(
      !actual.GetNativeAcGroup(actual.ac_group_count(), &invalid_group).ok() &&
          std::get<VarDctAcGroupView>(invalid_group).block_x == 123,
      "Invalid native query changed output");

  // Borrowed narrow input is copied; cached population is also copied.
  OverwriteArray<T> borrowed;
  borrowed.ResetForOverwrite(assembly.coefficients.size());
  std::copy(assembly.coefficients.begin(), assembly.coefficients.end(),
            borrowed.begin());
  auto mutable_population = population;
  VarDctEncoderFrame with_population;
  Check(internal::AssembleVarDctEncoderFrame(
      Input(oracle, borrowed, assembly.layouts, false, &mutable_population),
      &with_population));
  mutable_population.counts.fill(UINT32_MAX);
  std::fill(borrowed.begin(), borrowed.end(), T{3});
  EqualCoefficients(oracle, with_population);

  for (auto behavior : {VarDctCoefficientOrderBehavior::kFull,
                        VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled}) {
    VarDctCodestreamOptions options{.coefficient_order_behavior = behavior};
    std::vector<uint8_t> expected, encoded, cached;
    Check(EncodeVarDctCodestream(oracle, options, &expected));
    Check(EncodeVarDctCodestream(actual, options, &encoded));
    Check(EncodeVarDctCodestream(with_population, options, &cached));
    Require(expected == encoded && expected == cached,
            "Compact codestream differs");
    streams += 2;
  }
  Image3FBuffer expected(oracle.geometry().padded_frame());
  Image3FBuffer reconstructed(oracle.geometry().padded_frame());
  Check(ReconstructQuantizedCoefficients(oracle, expected.view()));
  Check(ReconstructQuantizedCoefficients(actual, reconstructed.view()));
  const auto extent = oracle.geometry().padded_frame();
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      Require(std::memcmp(expected.view().plane[c].Row(y),
                          reconstructed.view().plane[c].Row(y),
                          extent.width * sizeof(float)) == 0,
              "Compact reconstruction differs bitwise");
  ++reconstructions;

  // Readers of the same frame and its independent copy need no mutable cache.
  std::atomic<bool> okay{true};
  std::array<std::thread, 3> readers;
  for (auto &thread : readers)
    thread = std::thread([&] {
      try {
        for (size_t i = 0; i < 3; ++i) {
          EqualCoefficients(oracle, actual);
          EqualCoefficients(oracle, moved);
        }
      } catch (...) {
        okay = false;
      }
    });
  for (auto &thread : readers)
    thread.join();
  Require(okay, "Concurrent native reader mismatch");
  ++cases;
}
} // namespace

int main() {
  try {
    for (size_t side : {size_t{4}, size_t{36}})
      for (size_t strategy = 0; strategy <= 7; ++strategy)
        for (size_t pattern = 0; pattern < 3; ++pattern) {
          Case<int8_t>(strategy, pattern, side);
          Case<int16_t>(strategy, pattern, side);
        }
    std::cout << "Compact frame PASS cases=" << cases
              << " exact_streams=" << streams << " atomic_failures=" << failures
              << " bitwise_reconstructions=" << reconstructions << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
