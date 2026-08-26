// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Compares native scalar opsin/frequency stages with pinned libjxl.

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

struct StageErrorStats {
  float golden_maximum_absolute = 0.0f;
  float dispatched_maximum_absolute = 0.0f;
};

[[nodiscard]] gjxl::ConstImage3FView ToGjxl(bt::ConstOracleImage3 image) {
  gjxl::ConstImage3FView result;
  for (size_t channel = 0; channel < 3; ++channel) {
    result.plane[channel] = {
        image.plane[channel].data,
        {image.plane[channel].extent.width, image.plane[channel].extent.height},
        image.plane[channel].stride,
    };
  }
  return result;
}

[[nodiscard]] std::array<gjxl::ConstPlaneF32View, bt::kOpsinFrequencyStageCount>
NativeStageViews(const bi::OwnedImage3F &xyb,
                 const bi::OwnedPsychoImage &psycho) {
  return {
      xyb.ConstView().plane[0],
      xyb.ConstView().plane[1],
      xyb.ConstView().plane[2],
      psycho.LowFrequencyView().plane[0],
      psycho.LowFrequencyView().plane[1],
      psycho.LowFrequencyView().plane[2],
      psycho.MediumFrequencyView().plane[0],
      psycho.MediumFrequencyView().plane[1],
      psycho.MediumFrequencyView().plane[2],
      psycho.HighFrequencyView(0),
      psycho.HighFrequencyView(1),
      psycho.UltraHighFrequencyView(0),
      psycho.UltraHighFrequencyView(1),
  };
}

[[nodiscard]] bool ComputeNative(bt::ConstOracleImage3 input,
                                 float intensity_target,
                                 bi::OpsinScratch *opsin_scratch,
                                 bi::FrequencyScratch *frequency_scratch,
                                 bi::OwnedImage3F *xyb,
                                 bi::OwnedPsychoImage *psycho) {
  const gjxl::Extent2D extent{input.plane[0].extent.width,
                              input.plane[0].extent.height};
  return xyb->Resize(extent).ok() &&
         bi::OpsinDynamicsImage(ToGjxl(input), intensity_target, opsin_scratch,
                                xyb->View())
             .ok() &&
         bi::SeparateFrequencies(xyb->ConstView(), frequency_scratch, psycho)
             .ok();
}

[[nodiscard]] bool CheckScalarGoldens(StageErrorStats *errors) {
  static_assert(golden::kIntermediateStageCount == bt::kIntermediateStageCount);
  static_assert(bt::kOpsinFrequencyStageCount == 13);
  const bt::FixturePair fixture = bt::MakeFixture({
      "intermediate",
      {golden::kIntermediateWidth, golden::kIntermediateHeight},
      bt::FixtureKind::kIntermediate,
  });
  bi::OpsinScratch opsin_scratch;
  bi::FrequencyScratch frequency_scratch;
  bi::OwnedImage3F xyb;
  bi::OwnedPsychoImage psycho;
  if (!ComputeNative(fixture.reference.ConstView(),
                     fixture.options.intensity_target, &opsin_scratch,
                     &frequency_scratch, &xyb, &psycho)) {
    std::cerr << "Native scalar-golden stage execution failed\n";
    return false;
  }

  const auto native = NativeStageViews(xyb, psycho);
  constexpr size_t kGoldenFirst =
      static_cast<size_t>(bt::IntermediateStage::kOpsinX);
  for (size_t stage = 0; stage < native.size(); ++stage) {
    const auto &bits = golden::kIntermediateStageBits[kGoldenFirst + stage];
    for (size_t y = 0; y < native[stage].extent.height; ++y) {
      for (size_t x = 0; x < native[stage].extent.width; ++x) {
        const float actual = native[stage].Row(y)[x];
        const float expected =
            std::bit_cast<float>(bits[y * native[stage].extent.width + x]);
        const float absolute = std::abs(actual - expected);
        errors->golden_maximum_absolute =
            std::max(errors->golden_maximum_absolute, absolute);
        if (!std::isfinite(actual) || absolute > bt::MapTolerance(expected)) {
          std::cerr << "Native/scalar stage mismatch: "
                    << bt::OpsinFrequencyStageName(
                           static_cast<bt::OpsinFrequencyStage>(stage))
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

[[nodiscard]] bool CompareNativeWithOracle(
    bt::ConstOracleImage3 input, float intensity_target,
    const std::string &label, bi::OpsinScratch *opsin_scratch,
    bi::FrequencyScratch *frequency_scratch, StageErrorStats *errors) {
  bi::OwnedImage3F xyb;
  bi::OwnedPsychoImage psycho;
  bt::OpsinFrequencyStageOutput oracle;
  if (!ComputeNative(input, intensity_target, opsin_scratch, frequency_scratch,
                     &xyb, &psycho) ||
      !bt::ComputeLiveOpsinAndFrequencies(input, intensity_target, &oracle)) {
    std::cerr << "Opsin/frequency execution failed: " << label << '\n';
    return false;
  }

  const auto native = NativeStageViews(xyb, psycho);
  for (size_t stage = 0; stage < native.size(); ++stage) {
    if (oracle.plane[stage].size() !=
        native[stage].extent.width * native[stage].extent.height) {
      return false;
    }
    for (size_t y = 0; y < native[stage].extent.height; ++y) {
      for (size_t x = 0; x < native[stage].extent.width; ++x) {
        const float actual = native[stage].Row(y)[x];
        const float expected =
            oracle.plane[stage][y * native[stage].extent.width + x];
        const float absolute = std::abs(actual - expected);
        errors->dispatched_maximum_absolute =
            std::max(errors->dispatched_maximum_absolute, absolute);
        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            absolute > bt::kNativeOpsinFrequencyDispatchTolerance) {
          std::cerr << "Native/dispatched stage mismatch: " << label
                    << " stage="
                    << bt::OpsinFrequencyStageName(
                           static_cast<bt::OpsinFrequencyStage>(stage))
                    << " x=" << x << " y=" << y << " actual=" << actual
                    << " expected=" << expected << " abs=" << absolute << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool CheckDifferentialCorpus(StageErrorStats *errors) {
  const std::vector<bt::FixturePair> corpus =
      bt::BuildDifferentialCorpus(GJXL_FLOWER_PPM_PATH);
  constexpr std::array<bt::OracleExtent, 4> kSmallExtents = {
      bt::OracleExtent{1, 1},
      bt::OracleExtent{3, 7},
      bt::OracleExtent{7, 3},
      bt::OracleExtent{8, 8},
  };
  std::array<bool, kSmallExtents.size()> saw_small_extent{};
  bi::OpsinScratch opsin_scratch;
  bi::FrequencyScratch frequency_scratch;
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
        fixture.reference.ConstView(), fixture.distorted.ConstView()};
    for (size_t image = 0; image < images.size(); ++image) {
      const std::string label =
          fixture.name + (image == 0 ? "/reference" : "/distorted");
      if (!CompareNativeWithOracle(
              images[image], fixture.options.intensity_target, label,
              &opsin_scratch, &frequency_scratch, errors) ||
          !bt::PaddingIsPoisoned(fixture.reference) ||
          !bt::PaddingIsPoisoned(fixture.distorted)) {
        return false;
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

[[nodiscard]] bt::OpsinFrequencyStageOutput PoisonedStageOutput() {
  bt::OpsinFrequencyStageOutput output;
  output.extent = {2, 3};
  for (std::vector<float> &plane : output.plane) {
    plane = {-91.0f, -92.0f};
  }
  return output;
}

[[nodiscard]] bool SameOutput(const bt::OpsinFrequencyStageOutput &a,
                              const bt::OpsinFrequencyStageOutput &b) {
  return a.extent == b.extent && a.plane == b.plane;
}

[[nodiscard]] bool CheckOracleAllocationFailures() {
  const bt::FixturePair fixture = bt::MakeFixture({
      "opsin_frequency_allocation",
      {16, 12},
      bt::FixtureKind::kIntermediate,
  });
  size_t successful_calls = 0;
  {
    TrackingAllocator tracker;
    JxlMemoryManager manager = tracker.Manager();
    bt::OpsinFrequencyStageOutput output = PoisonedStageOutput();
    if (!bt::ComputeLiveOpsinAndFrequencies(fixture.reference.ConstView(),
                                            fixture.options.intensity_target,
                                            &output, &manager) ||
        tracker.live_bytes != 0) {
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
    bt::OpsinFrequencyStageOutput output = PoisonedStageOutput();
    const bt::OpsinFrequencyStageOutput original = output;
    if (bt::ComputeLiveOpsinAndFrequencies(fixture.reference.ConstView(),
                                           fixture.options.intensity_target,
                                           &output, &manager) ||
        !SameOutput(output, original) || tracker.live_bytes != 0) {
      std::cerr << "Direct stage oracle allocation failure was not atomic: "
                << fail_at << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckOracleValidation() {
  bt::FixturePair fixture = bt::MakeFixture({
      "opsin_frequency_validation",
      {8, 6},
      bt::FixtureKind::kIntermediate,
  });
  const auto check = [&](bt::ConstOracleImage3 input, float intensity) {
    bt::OpsinFrequencyStageOutput output = PoisonedStageOutput();
    const bt::OpsinFrequencyStageOutput original = output;
    return !bt::ComputeLiveOpsinAndFrequencies(input, intensity, &output) &&
           SameOutput(output, original);
  };

  bt::ConstOracleImage3 invalid = fixture.reference.ConstView();
  invalid.plane[0].data = nullptr;
  if (!check(invalid, 80.0f) || !check(fixture.reference.ConstView(), 0.0f) ||
      !check(fixture.reference.ConstView(),
             std::numeric_limits<float>::quiet_NaN()) ||
      bt::ComputeLiveOpsinAndFrequencies(fixture.reference.ConstView(), 80.0f,
                                         nullptr)) {
    return false;
  }
  fixture.reference.planes()[2][3] = std::numeric_limits<float>::infinity();
  return check(fixture.reference.ConstView(), 80.0f);
}

[[nodiscard]] bool CheckPerturbationDetection() {
  std::vector<float> expected = {0.0f, 1.0f, -2.0f};
  std::vector<float> perturbed = expected;
  perturbed[1] += 0.01f;
  const auto matches = [](const std::vector<float> &a,
                          const std::vector<float> &b) {
    for (size_t index = 0; index < a.size(); ++index) {
      if (std::abs(a[index] - b[index]) >
          bt::kNativeOpsinFrequencyDispatchTolerance) {
        return false;
      }
    }
    return true;
  };
  return matches(expected, expected) && !matches(expected, perturbed);
}

} // namespace

int main() {
  StageErrorStats errors;
  if (!CheckScalarGoldens(&errors) || !CheckDifferentialCorpus(&errors) ||
      !CheckOracleAllocationFailures() || !CheckOracleValidation() ||
      !CheckPerturbationDetection()) {
    return EXIT_FAILURE;
  }
  std::cout << std::setprecision(9)
            << "Maximum native-vs-scalar opsin/frequency error: abs="
            << errors.golden_maximum_absolute << '\n'
            << "Maximum native-vs-dispatched opsin/frequency error: abs="
            << errors.dispatched_maximum_absolute << '\n'
            << "All Butteraugli opsin/frequency differential tests passed.\n";
  return EXIT_SUCCESS;
}
