// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the pinned scalar goldens, live oracle, and AQ block reduction.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <jxl/memory_manager.h>

#include "butteraugli_fixtures.h"
#include "butteraugli_goldens_generated.h"
#include "butteraugli_oracle.h"
#include "butteraugli_test_tolerances.h"
#include "codec/butteraugli.h"

namespace {

namespace bt = gjxl::butteraugli_test;
namespace golden = gjxl::butteraugli_test::golden;

constexpr float kOutputPoison = -991.0f;
constexpr double kScorePoison = -313.0;

struct ErrorStats {
  float maximum_absolute = 0.0f;
  float maximum_relative = 0.0f;
  float maximum_tolerance_ratio = 0.0f;

  void Add(float actual, float expected) {
    const float absolute = std::abs(actual - expected);
    maximum_absolute = std::max(maximum_absolute, absolute);
    maximum_tolerance_ratio =
        std::max(maximum_tolerance_ratio,
                 absolute / bt::MapTolerance(expected));
    if (expected != 0.0f) {
      maximum_relative =
          std::max(maximum_relative, absolute / std::abs(expected));
    }
  }
};

[[nodiscard]] gjxl::Extent2D ToGjxlExtent(bt::OracleExtent extent) {
  return {extent.width, extent.height};
}

[[nodiscard]] gjxl::ConstImage3FView ToGjxlImage(bt::ConstOracleImage3 image) {
  return gjxl::ConstImage3FView{{{
      {image.plane[0].data, ToGjxlExtent(image.plane[0].extent),
       image.plane[0].stride},
      {image.plane[1].data, ToGjxlExtent(image.plane[1].extent),
       image.plane[1].stride},
      {image.plane[2].data, ToGjxlExtent(image.plane[2].extent),
       image.plane[2].stride},
  }}};
}

[[nodiscard]] gjxl::ButteraugliOptions
ToGjxlOptions(bt::OracleOptions options) {
  return {
      .hf_asymmetry = options.hf_asymmetry,
      .x_multiplier = options.x_multiplier,
      .intensity_target = options.intensity_target,
  };
}

struct MapStorage {
  explicit MapStorage(bt::OracleExtent image_extent)
      : extent(image_extent), stride(image_extent.width + 5),
        values(stride * extent.height, kOutputPoison) {}

  [[nodiscard]] bt::OraclePlane OracleView() {
    return {values.data(), extent, stride};
  }
  [[nodiscard]] gjxl::PlaneF32View GjxlView() {
    return {values.data(), ToGjxlExtent(extent), stride};
  }
  [[nodiscard]] bool PaddingIsUntouched() const {
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = extent.width; x < stride; ++x) {
        if (values[y * stride + x] != kOutputPoison)
          return false;
      }
    }
    return true;
  }
  [[nodiscard]] std::vector<float> LogicalValues() const {
    std::vector<float> logical(extent.width * extent.height);
    for (size_t y = 0; y < extent.height; ++y) {
      std::copy_n(values.data() + y * stride, extent.width,
                  logical.data() + y * extent.width);
    }
    return logical;
  }

  bt::OracleExtent extent;
  size_t stride;
  std::vector<float> values;
};

[[nodiscard]] bool ValuesMatch(const std::vector<float> &actual,
                               const std::vector<float> &expected,
                               ErrorStats *errors = nullptr) {
  if (actual.size() != expected.size())
    return false;
  for (size_t index = 0; index < actual.size(); ++index) {
    if (errors != nullptr)
      errors->Add(actual[index], expected[index]);
    const float tolerance = bt::MapTolerance(expected[index]);
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ScalarDispatchValuesMatch(
    const std::vector<float> &actual, const std::vector<float> &expected,
    float absolute_tolerance, ErrorStats *errors = nullptr) {
  if (actual.size() != expected.size())
    return false;
  bool matches = true;
  for (size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      return false;
    }
    if (errors != nullptr)
      errors->Add(actual[index], expected[index]);
    if (std::abs(actual[index] - expected[index]) > absolute_tolerance)
      matches = false;
  }
  return matches;
}

template <size_t Size>
[[nodiscard]] std::vector<float>
DecodeBits(const std::array<uint32_t, Size> &bits) {
  std::vector<float> values(Size);
  for (size_t index = 0; index < Size; ++index) {
    values[index] = std::bit_cast<float>(bits[index]);
  }
  return values;
}

[[nodiscard]] bool ComputeFacade(const bt::FixturePair &fixture,
                                 MapStorage *map, double *score) {
  return gjxl::ComputeButteraugliDistance(
             ToGjxlImage(fixture.reference.ConstView()),
             ToGjxlImage(fixture.distorted.ConstView()),
             ToGjxlOptions(fixture.options), map->GjxlView(), score)
      .ok();
}

[[nodiscard]] bool CheckScalarGoldens(ErrorStats *full_map_errors,
                                      double *maximum_score_error) {
  constexpr std::array<bt::FixtureKind, golden::kFullMapCount> kKinds = {
      bt::FixtureKind::kFlat,
      bt::FixtureKind::kTexture,
      bt::FixtureKind::kContrast,
      bt::FixtureKind::kChromatic,
  };
  constexpr std::array<std::string_view, golden::kFullMapCount> kNames = {
      "flat",
      "texture",
      "contrast",
      "chromatic",
  };
  for (size_t index = 0; index < kKinds.size(); ++index) {
    const bt::FixturePair fixture = bt::MakeFixture({
        std::string(kNames[index]),
        {golden::kFullMapWidth, golden::kFullMapHeight},
        kKinds[index],
    });
    MapStorage actual(fixture.reference.extent());
    double score = kScorePoison;
    const std::vector<float> expected = DecodeBits(golden::kFullMapBits[index]);
    const double expected_score =
        std::bit_cast<double>(golden::kFullMapScoreBits[index]);
    if (!ComputeFacade(fixture, &actual, &score) ||
        !actual.PaddingIsUntouched() ||
        !bt::PaddingIsPoisoned(fixture.reference) ||
        !bt::PaddingIsPoisoned(fixture.distorted) ||
        !ScalarDispatchValuesMatch(
            actual.LogicalValues(), expected,
            bt::kScalarDispatchTolerance.full_map_absolute, full_map_errors) ||
        !std::isfinite(score) ||
        std::abs(score - expected_score) >
            bt::kScalarDispatchTolerance.score_absolute) {
      std::cerr << "Scalar full-map golden differs: " << kNames[index]
                << "\n  actual score: " << std::setprecision(12) << score
                << "\n  scalar score: " << expected_score
                << "\n  maximum map error: "
                << full_map_errors->maximum_absolute << '\n';
      return false;
    }
    *maximum_score_error =
        std::max(*maximum_score_error, std::abs(score - expected_score));
  }
  std::vector<float> perturbed = DecodeBits(golden::kFullMapBits[0]);
  const std::vector<float> unperturbed = perturbed;
  perturbed[perturbed.size() / 2] += 0.01f;
  if (!ValuesMatch(unperturbed, unperturbed) ||
      ValuesMatch(unperturbed, perturbed) ||
      ScalarDispatchValuesMatch(
          unperturbed, perturbed,
          bt::kScalarDispatchTolerance.full_map_absolute)) {
    std::cerr << "Full-map comparator did not detect a perturbation\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckIntermediateGoldens(ErrorStats *stage_errors) {
  static_assert(golden::kIntermediateStageCount == bt::kIntermediateStageCount);
  const bt::FixturePair fixture = bt::MakeFixture({
      "intermediate",
      {golden::kIntermediateWidth, golden::kIntermediateHeight},
      bt::FixtureKind::kIntermediate,
  });
  bt::IntermediateStageOutput output;
  if (!bt::ComputePinnedIntermediateStages(fixture.reference.ConstView(),
                                           fixture.distorted.ConstView(),
                                           fixture.options, &output)) {
    std::cerr << "Live intermediate-stage oracle failed\n";
    return false;
  }
  for (size_t index = 0; index < bt::kIntermediateStageCount; ++index) {
    const std::vector<float> expected =
        DecodeBits(golden::kIntermediateStageBits[index]);
    if (!ScalarDispatchValuesMatch(
            output.plane[index], expected,
            bt::kScalarDispatchTolerance.stage_absolute, stage_errors)) {
      std::cerr << "Scalar intermediate golden differs: "
                << bt::IntermediateStageName(
                       static_cast<bt::IntermediateStage>(index))
                << "\n  maximum stage error: "
                << stage_errors->maximum_absolute << '\n';
      return false;
    }
  }
  MapStorage full_map(fixture.reference.extent());
  double full_score = kScorePoison;
  if (!ComputeFacade(fixture, &full_map, &full_score) ||
      !ValuesMatch(output.plane[static_cast<size_t>(
                       bt::IntermediateStage::kFinalComposition)],
                   full_map.LogicalValues())) {
    std::cerr << "Intermediate final composition differs from the public map\n";
    return false;
  }
  std::vector<float> perturbed =
      DecodeBits(golden::kIntermediateStageBits[static_cast<size_t>(
          bt::IntermediateStage::kMask)]);
  const std::vector<float> unperturbed = perturbed;
  perturbed[3] += 0.01f;
  if (ValuesMatch(unperturbed, perturbed) ||
      ScalarDispatchValuesMatch(unperturbed, perturbed,
                                bt::kScalarDispatchTolerance.stage_absolute)) {
    std::cerr << "Stage comparator did not detect a perturbation\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckDifferentialCorpus() {
  const std::vector<bt::FixturePair> corpus =
      bt::BuildDifferentialCorpus(GJXL_FLOWER_PPM_PATH);
  for (const bt::FixturePair &fixture : corpus) {
    MapStorage facade_map(fixture.reference.extent());
    MapStorage oracle_map(fixture.reference.extent());
    double facade_score = kScorePoison;
    double oracle_score = kScorePoison;
    if (!ComputeFacade(fixture, &facade_map, &facade_score) ||
        !bt::ComputeLiveButteraugli(
            fixture.reference.ConstView(), fixture.distorted.ConstView(),
            fixture.options, oracle_map.OracleView(), &oracle_score) ||
        !facade_map.PaddingIsUntouched() || !oracle_map.PaddingIsUntouched() ||
        !bt::PaddingIsPoisoned(fixture.reference) ||
        !bt::PaddingIsPoisoned(fixture.distorted) ||
        !ValuesMatch(facade_map.LogicalValues(), oracle_map.LogicalValues()) ||
        std::abs(facade_score - oracle_score) > bt::kScoreTolerance) {
      std::cerr << "Live differential fixture failed: " << fixture.name << '\n';
      return false;
    }
    if (fixture.name == "identity_texture_32x24" &&
        (std::abs(facade_score) > bt::kIdentityTolerance ||
         !std::ranges::all_of(facade_map.LogicalValues(), [](float value) {
           return std::abs(value) <= bt::kIdentityTolerance;
         }))) {
      std::cerr << "Identity map exceeds the 1e-7 tolerance\n";
      return false;
    }
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
    if (header == nullptr)
      return nullptr;
    header->size = size;
    self->live_bytes += size;
    self->peak_bytes = std::max(self->peak_bytes, self->live_bytes);
    return header + 1;
  }
  static void Free(void *opaque, void *address) {
    if (address == nullptr)
      return;
    auto *self = static_cast<TrackingAllocator *>(opaque);
    auto *header = static_cast<Header *>(address) - 1;
    self->live_bytes -= header->size;
    std::free(header);
  }
  [[nodiscard]] JxlMemoryManager Manager() { return {this, Allocate, Free}; }
  void Reset(size_t failure_index = std::numeric_limits<size_t>::max()) {
    allocation_calls = 0;
    peak_bytes = live_bytes;
    fail_at = failure_index;
  }

  size_t allocation_calls = 0;
  size_t live_bytes = 0;
  size_t peak_bytes = 0;
  size_t fail_at = std::numeric_limits<size_t>::max();
};

[[nodiscard]] bool MapAndScoreAreUntouched(const MapStorage &map,
                                           const std::vector<float> &original,
                                           double score) {
  return map.values == original && score == kScorePoison;
}

[[nodiscard]] bool
CheckOneShotAllocationFailures(const bt::FixturePair &fixture) {
  size_t successful_calls = 0;
  {
    TrackingAllocator tracker;
    JxlMemoryManager manager = tracker.Manager();
    MapStorage map(fixture.reference.extent());
    double score = kScorePoison;
    if (!bt::ComputeLiveButteraugli(
            fixture.reference.ConstView(), fixture.distorted.ConstView(),
            fixture.options, map.OracleView(), &score, &manager) ||
        tracker.live_bytes != 0)
      return false;
    successful_calls = tracker.allocation_calls;
  }
  for (size_t fail_at = 0; fail_at < successful_calls; ++fail_at) {
    TrackingAllocator tracker;
    tracker.fail_at = fail_at;
    JxlMemoryManager manager = tracker.Manager();
    MapStorage map(fixture.reference.extent());
    const std::vector<float> original = map.values;
    double score = kScorePoison;
    if (bt::ComputeLiveButteraugli(
            fixture.reference.ConstView(), fixture.distorted.ConstView(),
            fixture.options, map.OracleView(), &score, &manager) ||
        !MapAndScoreAreUntouched(map, original, score) ||
        tracker.live_bytes != 0) {
      std::cerr << "One-shot allocation failure was not atomic at " << fail_at
                << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
CheckPrepareAllocationFailures(const bt::FixturePair &fixture) {
  size_t successful_calls = 0;
  {
    TrackingAllocator tracker;
    JxlMemoryManager manager = tracker.Manager();
    {
      bt::PreparedReference prepared;
      if (!bt::PrepareLiveButteraugliReference(fixture.reference.ConstView(),
                                               fixture.options, &prepared,
                                               &manager))
        return false;
      successful_calls = tracker.allocation_calls;
    }
    if (tracker.live_bytes != 0)
      return false;
  }
  for (size_t fail_at = 0; fail_at < successful_calls; ++fail_at) {
    TrackingAllocator tracker;
    tracker.fail_at = fail_at;
    JxlMemoryManager manager = tracker.Manager();
    bt::PreparedReference prepared;
    if (bt::PrepareLiveButteraugliReference(fixture.reference.ConstView(),
                                            fixture.options, &prepared,
                                            &manager) ||
        prepared.valid() || tracker.live_bytes != 0) {
      std::cerr << "Prepare allocation failure leaked at " << fail_at << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
CheckPreparedCompareAllocationFailures(const bt::FixturePair &fixture) {
  size_t successful_compare_calls = 0;
  {
    TrackingAllocator tracker;
    JxlMemoryManager manager = tracker.Manager();
    bt::PreparedReference prepared;
    if (!bt::PrepareLiveButteraugliReference(fixture.reference.ConstView(),
                                             fixture.options, &prepared,
                                             &manager))
      return false;
    tracker.Reset();
    MapStorage map(fixture.reference.extent());
    double score = kScorePoison;
    if (!bt::CompareLiveButteraugliPrepared(
            prepared, fixture.distorted.ConstView(), map.OracleView(), &score))
      return false;
    successful_compare_calls = tracker.allocation_calls;
  }
  for (size_t fail_at = 0; fail_at < successful_compare_calls; ++fail_at) {
    TrackingAllocator tracker;
    JxlMemoryManager manager = tracker.Manager();
    {
      bt::PreparedReference prepared;
      if (!bt::PrepareLiveButteraugliReference(fixture.reference.ConstView(),
                                               fixture.options, &prepared,
                                               &manager))
        return false;
      const size_t retained_bytes = tracker.live_bytes;
      tracker.Reset(fail_at);
      MapStorage map(fixture.reference.extent());
      const std::vector<float> original = map.values;
      double score = kScorePoison;
      if (bt::CompareLiveButteraugliPrepared(prepared,
                                             fixture.distorted.ConstView(),
                                             map.OracleView(), &score) ||
          !MapAndScoreAreUntouched(map, original, score) ||
          tracker.live_bytes != retained_bytes) {
        std::cerr << "Prepared comparison failure was not atomic at " << fail_at
                  << '\n';
        return false;
      }
    }
    if (tracker.live_bytes != 0)
      return false;
  }
  return true;
}

[[nodiscard]] bool
CheckStageAllocationFailures(const bt::FixturePair &fixture) {
  size_t successful_calls = 0;
  {
    TrackingAllocator tracker;
    JxlMemoryManager manager = tracker.Manager();
    bt::IntermediateStageOutput output;
    if (!bt::ComputePinnedIntermediateStages(
            fixture.reference.ConstView(), fixture.distorted.ConstView(),
            fixture.options, &output, &manager) ||
        tracker.live_bytes != 0) {
      return false;
    }
    successful_calls = tracker.allocation_calls;
  }
  for (size_t fail_at = 0; fail_at < successful_calls; ++fail_at) {
    TrackingAllocator tracker;
    tracker.fail_at = fail_at;
    JxlMemoryManager manager = tracker.Manager();
    bt::IntermediateStageOutput output;
    output.extent = {1, 1};
    output.plane[0] = {42.0f};
    if (bt::ComputePinnedIntermediateStages(
            fixture.reference.ConstView(), fixture.distorted.ConstView(),
            fixture.options, &output, &manager) ||
        output.extent != bt::OracleExtent{1, 1} ||
        output.plane[0] != std::vector<float>{42.0f} ||
        tracker.live_bytes != 0) {
      std::cerr << "Stage allocation failure was not atomic at " << fail_at
                << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckAllocationFailures() {
  const bt::FixturePair fixture = bt::MakeFixture({
      "allocation",
      {32, 24},
      bt::FixtureKind::kTexture,
  });
  const bt::FixturePair stage_fixture = bt::MakeFixture({
      "stage_allocation",
      {16, 12},
      bt::FixtureKind::kIntermediate,
  });
  return CheckOneShotAllocationFailures(fixture) &&
         CheckPrepareAllocationFailures(fixture) &&
         CheckPreparedCompareAllocationFailures(fixture) &&
         CheckStageAllocationFailures(stage_fixture);
}

[[nodiscard]] bool ExpectFacadeFailureAtomic(
    bt::ConstOracleImage3 reference, bt::ConstOracleImage3 distorted,
    bt::OracleOptions options, gjxl::PlaneF32View output, double *score,
    const std::vector<float> &expected_map, double expected_score) {
  const gjxl::Status status = gjxl::ComputeButteraugliDistance(
      ToGjxlImage(reference), ToGjxlImage(distorted), ToGjxlOptions(options),
      output, score);
  return !status.ok() &&
         (output.data == nullptr ||
          std::equal(expected_map.begin(), expected_map.end(), output.data)) &&
         (score == nullptr || *score == expected_score);
}

[[nodiscard]] bool CheckInvalidFacadeRequests() {
  bt::FixturePair fixture = bt::MakeFixture({
      "invalid",
      {32, 24},
      bt::FixtureKind::kTexture,
  });
  const bt::ConstOracleImage3 good_reference = fixture.reference.ConstView();
  const bt::ConstOracleImage3 good_distorted = fixture.distorted.ConstView();
  MapStorage map(fixture.reference.extent());
  const std::vector<float> original = map.values;
  double score = kScorePoison;
  auto check = [&](bt::ConstOracleImage3 reference,
                   bt::ConstOracleImage3 distorted, bt::OracleOptions options,
                   gjxl::PlaneF32View output) {
    if (!ExpectFacadeFailureAtomic(reference, distorted, options, output,
                                   &score, original, kScorePoison)) {
      std::cerr << "Invalid facade request was not rejected atomically\n";
      return false;
    }
    return true;
  };

  bt::ConstOracleImage3 bad = good_distorted;
  bad.plane[2].extent.width -= 1;
  if (!check(good_reference, bad, {}, map.GjxlView()))
    return false;
  const bt::FixturePair mismatched = bt::MakeFixture({
      "mismatched",
      {31, 24},
      bt::FixtureKind::kTexture,
  });
  if (!check(good_reference, mismatched.distorted.ConstView(), {},
             map.GjxlView())) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    bad = good_distorted;
    bad.plane[channel].data = nullptr;
    if (!check(good_reference, bad, {}, map.GjxlView()))
      return false;
    bad = good_reference;
    bad.plane[channel].data = nullptr;
    if (!check(bad, good_distorted, {}, map.GjxlView()))
      return false;
  }
  bad = good_distorted;
  bad.plane[1].stride = bad.plane[1].extent.width - 1;
  if (!check(good_reference, bad, {}, map.GjxlView()))
    return false;
  if (!check(good_reference, good_distorted, {},
             {nullptr, ToGjxlExtent(fixture.reference.extent()), 32}) ||
      !check(
          good_reference, good_distorted, {},
          {map.values.data(), ToGjxlExtent(fixture.reference.extent()), 31})) {
    return false;
  }
  if (gjxl::ComputeButteraugliDistance(ToGjxlImage(good_reference),
                                       ToGjxlImage(good_distorted), {},
                                       map.GjxlView(), nullptr)
          .ok() ||
      map.values != original) {
    return false;
  }

  constexpr std::array<float, 3> kBadValues = {
      0.0f,
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
  };
  for (size_t option = 0; option < 3; ++option) {
    for (float bad_value : kBadValues) {
      bt::OracleOptions bad_options;
      if (option == 0)
        bad_options.hf_asymmetry = bad_value;
      if (option == 1)
        bad_options.x_multiplier = bad_value;
      if (option == 2)
        bad_options.intensity_target = bad_value;
      if (!check(good_reference, good_distorted, bad_options, map.GjxlView())) {
        return false;
      }
    }
  }
  fixture.reference.planes()[0][0] = std::numeric_limits<float>::quiet_NaN();
  if (!check(fixture.reference.ConstView(), good_distorted, {},
             map.GjxlView())) {
    return false;
  }
  fixture.reference.planes()[0][0] = 0.25f;
  fixture.distorted.planes()[2][3] = std::numeric_limits<float>::infinity();
  if (!check(fixture.reference.ConstView(), fixture.distorted.ConstView(), {},
             map.GjxlView())) {
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckBlockReduction() {
  constexpr gjxl::Extent2D kBlockExtent{4, 4};
  constexpr gjxl::Extent2D kPixelExtent{32, 32};
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kBlockExtent, &strategies).ok() ||
      !strategies.Set(0, 0, gjxl::AcStrategyType::kDct32x32).ok())
    return false;
  std::vector<float> distance(kPixelExtent.width * kPixelExtent.height, 2.0f);
  std::vector<float> blocks(kBlockExtent.width * kBlockExtent.height, -1.0f);
  if (!gjxl::ReduceButteraugliDistanceMap(
           {distance.data(), kPixelExtent, kPixelExtent.width}, strategies,
           {blocks.data(), kBlockExtent, kBlockExtent.width})
           .ok())
    return false;
  for (float value : blocks) {
    if (std::abs(value - 2.4f) > bt::kBlockReductionTolerance)
      return false;
  }
  distance.back() = -1.0f;
  const std::vector<float> original = blocks;
  if (gjxl::ReduceButteraugliDistanceMap(
          {distance.data(), kPixelExtent, kPixelExtent.width}, strategies,
          {blocks.data(), kBlockExtent, kBlockExtent.width})
          .ok() ||
      blocks != original) {
    std::cerr << "Invalid Butteraugli reduction was not atomic\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  ErrorStats full_map_errors;
  ErrorStats stage_errors;
  double maximum_score_error = 0.0;
  if (!CheckScalarGoldens(&full_map_errors, &maximum_score_error) ||
      !CheckIntermediateGoldens(&stage_errors) || !CheckDifferentialCorpus() ||
      !CheckInvalidFacadeRequests() || !CheckAllocationFailures() ||
      !CheckBlockReduction()) {
    return EXIT_FAILURE;
  }
  std::cout << std::setprecision(9)
            << "Maximum scalar-vs-dispatched full-map error: abs="
            << full_map_errors.maximum_absolute
            << " rel=" << full_map_errors.maximum_relative
            << " tolerance_ratio=" << full_map_errors.maximum_tolerance_ratio
            << " score_abs=" << maximum_score_error << '\n'
            << "Maximum scalar-vs-dispatched stage error: abs="
            << stage_errors.maximum_absolute
            << " rel=" << stage_errors.maximum_relative
            << " tolerance_ratio=" << stage_errors.maximum_tolerance_ratio
            << '\n'
            << "All Butteraugli tests passed.\n";
  return EXIT_SUCCESS;
}
