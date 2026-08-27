// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Compares native scalar Gaussian blur with pinned libjxl over the corpus.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <jxl/memory_manager.h>

#include "butteraugli_fixtures.h"
#include "butteraugli_goldens_generated.h"
#include "butteraugli_oracle.h"
#include "butteraugli_test_tolerances.h"
#include "codec/butteraugli_internal.h"

namespace {

namespace bi = gjxl::butteraugli_internal;
namespace bt = gjxl::butteraugli_test;
namespace golden = gjxl::butteraugli_test::golden;

constexpr float kOutputPoison = -991.0f;

struct PlaneStorage {
  explicit PlaneStorage(bt::OracleExtent image_extent, size_t padding = 5)
      : extent(image_extent), stride(image_extent.width + padding),
        values(stride * extent.height, kOutputPoison) {}

  [[nodiscard]] gjxl::PlaneF32View GjxlView() {
    return {values.data(), {extent.width, extent.height}, stride};
  }

  [[nodiscard]] bt::OraclePlane OracleView() {
    return {values.data(), extent, stride};
  }

  [[nodiscard]] bool PaddingIsUntouched() const {
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = extent.width; x < stride; ++x) {
        if (values[y * stride + x] != kOutputPoison) {
          return false;
        }
      }
    }
    return true;
  }

  bt::OracleExtent extent;
  size_t stride;
  std::vector<float> values;
};

struct BlurErrorStats {
  float golden_maximum_absolute = 0.0f;
  float dispatched_maximum_absolute = 0.0f;
};

[[nodiscard]] gjxl::ConstPlaneF32View ToGjxl(bt::ConstOraclePlane plane) {
  return {
      plane.data,
      {plane.extent.width, plane.extent.height},
      plane.stride,
  };
}

[[nodiscard]] bool CompareNativeWithOracle(bt::ConstOraclePlane input,
                                           float sigma,
                                           const std::string &label,
                                           bi::BlurScratch *scratch,
                                           BlurErrorStats *errors) {

  PlaneStorage native(input.extent, 7);
  PlaneStorage oracle(input.extent, 3);
  const gjxl::Status status =
      bi::GaussianBlur(ToGjxl(input), sigma, scratch, native.GjxlView());
  if (!status.ok() ||
      !bt::ComputeLiveGaussianBlur(input, sigma, oracle.OracleView()) ||
      !native.PaddingIsUntouched() || !oracle.PaddingIsUntouched()) {
    std::cerr << "Blur execution or padding check failed: " << label
              << " sigma=" << sigma << '\n';
    return false;
  }

  for (size_t y = 0; y < input.extent.height; ++y) {
    for (size_t x = 0; x < input.extent.width; ++x) {
      const float actual = native.values[y * native.stride + x];
      const float expected = oracle.values[y * oracle.stride + x];
      const float absolute = std::abs(actual - expected);
      errors->dispatched_maximum_absolute =
          std::max(errors->dispatched_maximum_absolute, absolute);
      if (!std::isfinite(actual) || !std::isfinite(expected) ||
          absolute > bt::kNativeBlurDispatchTolerance) {
        std::cerr << "Native/dispatched blur mismatch: " << label
                  << " sigma=" << sigma << " x=" << x << " y=" << y
                  << " actual=" << actual << " expected=" << expected
                  << " abs=" << absolute << '\n';
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool CheckScalarBlurGoldens(BlurErrorStats *errors) {
  static_assert(golden::kIntermediateStageCount == bt::kIntermediateStageCount);
  const bt::FixturePair fixture = bt::MakeFixture({
      "intermediate",
      {golden::kIntermediateWidth, golden::kIntermediateHeight},
      bt::FixtureKind::kIntermediate,
  });
  const bt::ConstOraclePlane input = fixture.reference.ConstView().plane[0];
  bi::BlurScratch scratch;
  for (size_t sigma_index = 0; sigma_index < bi::kPinnedBlurSigmas.size();
       ++sigma_index) {
    PlaneStorage output(input.extent, 6);
    const float sigma = bi::kPinnedBlurSigmas[sigma_index];
    if (!bi::GaussianBlur(ToGjxl(input), sigma, &scratch, output.GjxlView())
             .ok() ||
        !output.PaddingIsUntouched()) {
      return false;
    }
    const auto &golden_bits = golden::kIntermediateStageBits[sigma_index];
    for (size_t y = 0; y < input.extent.height; ++y) {
      for (size_t x = 0; x < input.extent.width; ++x) {
        const float actual = output.values[y * output.stride + x];
        const float expected =
            std::bit_cast<float>(golden_bits[y * input.extent.width + x]);
        const float absolute = std::abs(actual - expected);
        errors->golden_maximum_absolute =
            std::max(errors->golden_maximum_absolute, absolute);
        if (!std::isfinite(actual) || absolute > bt::MapTolerance(expected)) {
          std::cerr << "Native/scalar blur golden mismatch: sigma=" << sigma
                    << " x=" << x << " y=" << y << " actual=" << actual
                    << " expected=" << expected << " abs=" << absolute
                    << " limit=" << bt::MapTolerance(expected) << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool CheckDifferentialCorpus(BlurErrorStats *errors) {
  const std::vector<bt::FixturePair> corpus =
      bt::BuildDifferentialCorpus(GJXL_FLOWER_PPM_PATH);
  std::array<bool, 4> saw_small_extent{};
  constexpr std::array<bt::OracleExtent, 4> kSmallExtents = {
      bt::OracleExtent{1, 1},
      bt::OracleExtent{3, 7},
      bt::OracleExtent{7, 3},
      bt::OracleExtent{8, 8},
  };
  bi::BlurScratch scratch;
  for (const bt::FixturePair &fixture : corpus) {
    if (!bt::PaddingIsPoisoned(fixture.reference) ||
        !bt::PaddingIsPoisoned(fixture.distorted)) {
      return false;
    }
    for (size_t index = 0; index < kSmallExtents.size(); ++index) {
      if (fixture.reference.extent() == kSmallExtents[index]) {
        saw_small_extent[index] = true;
      }
    }
    const std::array<bt::ConstOracleImage3, 2> images = {
        fixture.reference.ConstView(),
        fixture.distorted.ConstView(),
    };
    for (size_t image_index = 0; image_index < images.size(); ++image_index) {
      for (size_t channel = 0; channel < 3; ++channel) {
        for (float sigma : bi::kPinnedBlurSigmas) {
          const std::string label =
              fixture.name +
              (image_index == 0 ? "/reference/c" : "/distorted/c") +
              std::to_string(channel);
          if (!CompareNativeWithOracle(images[image_index].plane[channel],
                                       sigma, label, &scratch, errors)) {
            return false;
          }
        }
      }
    }
  }
  if (!std::ranges::all_of(saw_small_extent,
                           [](bool value) { return value; })) {
    std::cerr << "Differential corpus omitted a required small extent\n";
    return false;
  }
  return true;
}

struct TrackingAllocator {
  struct Header {
    size_t size;
  };

  static void *Allocate(void *opaque, size_t size) {
    auto *self = static_cast<TrackingAllocator *>(opaque);
    const size_t call = self->allocation_calls++;
    if (call == self->fail_at ||
        size > std::numeric_limits<size_t>::max() - sizeof(Header)) {
      return nullptr;
    }
    auto *header = static_cast<Header *>(std::malloc(sizeof(Header) + size));
    if (header == nullptr) {
      return nullptr;
    }
    header->size = size;
    self->live_bytes += size;
    return header + 1;
  }

  static void Free(void *opaque, void *address) {
    if (address == nullptr) {
      return;
    }
    auto *self = static_cast<TrackingAllocator *>(opaque);
    auto *header = static_cast<Header *>(address) - 1;
    self->live_bytes -= header->size;
    std::free(header);
  }

  [[nodiscard]] JxlMemoryManager Manager() { return {this, Allocate, Free}; }

  size_t allocation_calls = 0;
  size_t live_bytes = 0;
  size_t fail_at = std::numeric_limits<size_t>::max();
};

[[nodiscard]] bool CheckOracleAllocationFailures() {
  const bt::FixturePair fixture = bt::MakeFixture({
      "blur_allocation",
      {16, 12},
      bt::FixtureKind::kIntermediate,
  });
  const bt::ConstOraclePlane input = fixture.reference.ConstView().plane[0];
  for (float sigma : bi::kPinnedBlurSigmas) {
    size_t successful_calls = 0;
    {
      TrackingAllocator tracker;
      JxlMemoryManager manager = tracker.Manager();
      PlaneStorage output(input.extent);
      if (!bt::ComputeLiveGaussianBlur(input, sigma, output.OracleView(),
                                       &manager) ||
          !output.PaddingIsUntouched() || tracker.live_bytes != 0) {
        return false;
      }
      successful_calls = tracker.allocation_calls;
    }
    if (successful_calls == 0) {
      return false;
    }
    for (size_t fail_at = 0; fail_at < successful_calls; ++fail_at) {
      TrackingAllocator tracker;
      tracker.fail_at = fail_at;
      JxlMemoryManager manager = tracker.Manager();
      PlaneStorage output(input.extent);
      const std::vector<float> original = output.values;
      if (bt::ComputeLiveGaussianBlur(input, sigma, output.OracleView(),
                                      &manager) ||
          output.values != original || tracker.live_bytes != 0) {
        std::cerr << "Direct blur oracle allocation failure was not atomic: "
                  << "sigma=" << sigma << " allocation=" << fail_at << '\n';
        return false;
      }
    }
  }
  return true;
}

} // namespace

int main() {
  BlurErrorStats errors;
  if (!CheckScalarBlurGoldens(&errors) || !CheckDifferentialCorpus(&errors) ||
      !CheckOracleAllocationFailures()) {
    return EXIT_FAILURE;
  }
  std::cout << std::setprecision(9)
            << "Maximum native-vs-scalar blur error: abs="
            << errors.golden_maximum_absolute << '\n'
            << "Maximum native-vs-dispatched blur error: abs="
            << errors.dispatched_maximum_absolute << '\n'
            << "All Butteraugli blur differential tests passed.\n";
  return EXIT_SUCCESS;
}
