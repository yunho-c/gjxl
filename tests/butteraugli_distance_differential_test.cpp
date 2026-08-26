// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Compares native scalar difference maps and stages with pinned libjxl.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "butteraugli_fixtures.h"
#include "butteraugli_goldens_generated.h"
#include "butteraugli_oracle.h"
#include "butteraugli_test_tolerances.h"
#include "codec/butteraugli_distance_internal.h"

namespace {

namespace bi = gjxl::butteraugli_internal;
namespace bt = gjxl::butteraugli_test;
namespace golden = gjxl::butteraugli_test::golden;

#ifndef GJXL_NATIVE_DISTANCE_ORACLE_SCALAR
#error "GJXL_NATIVE_DISTANCE_ORACLE_SCALAR must select the oracle target"
#endif

inline constexpr bool kScalarOracle = GJXL_NATIVE_DISTANCE_ORACLE_SCALAR != 0;
inline constexpr std::string_view kOracleName =
    kScalarOracle ? "scalar" : "dispatched";

constexpr float kPoison = -991.0f;
constexpr double kScorePoison = -313.0;

struct ErrorStats {
  float scalar_stage = 0.0f;
  float scalar_map = 0.0f;
  float oracle_stage = 0.0f;
  float oracle_map = 0.0f;
  double scalar_score = 0.0;
  double oracle_score = 0.0;
};

struct MapStorage {
  explicit MapStorage(bt::OracleExtent image_extent, size_t padding = 5)
      : extent(image_extent), stride(extent.width + padding),
        values(stride * extent.height, kPoison) {}

  [[nodiscard]] gjxl::PlaneF32View GjxlView() {
    return {values.data(), {extent.width, extent.height}, stride};
  }

  [[nodiscard]] bt::OraclePlane OracleView() {
    return {values.data(), extent, stride};
  }

  [[nodiscard]] std::vector<float> LogicalValues() const {
    std::vector<float> result(extent.width * extent.height);
    for (size_t y = 0; y < extent.height; ++y) {
      std::copy_n(values.data() + y * stride, extent.width,
                  result.data() + y * extent.width);
    }
    return result;
  }

  [[nodiscard]] bool PaddingIsUntouched() const {
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = extent.width; x < stride; ++x) {
        if (values[y * stride + x] != kPoison) {
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

[[nodiscard]] bi::NativeButteraugliParams ToNative(bt::OracleOptions options) {
  return {
      .hf_asymmetry = options.hf_asymmetry,
      .x_multiplier = options.x_multiplier,
      .intensity_target = options.intensity_target,
  };
}

[[nodiscard]] bool
ComputeNativeStages(const bt::FixturePair &fixture, bi::OpsinScratch *opsin,
                    bi::FrequencyScratch *frequency,
                    bi::DifferenceScratch *difference, bi::OwnedImage3F *xyb0,
                    bi::OwnedImage3F *xyb1, bi::OwnedPsychoImage *psycho0,
                    bi::OwnedPsychoImage *psycho1,
                    bi::OwnedDifferenceStages *stages) {
  const gjxl::Extent2D extent{fixture.reference.extent().width,
                              fixture.reference.extent().height};
  return xyb0->Resize(extent).ok() && xyb1->Resize(extent).ok() &&
         bi::OpsinDynamicsImage(ToGjxl(fixture.reference.ConstView()),
                                fixture.options.intensity_target, opsin,
                                xyb0->View())
             .ok() &&
         bi::OpsinDynamicsImage(ToGjxl(fixture.distorted.ConstView()),
                                fixture.options.intensity_target, opsin,
                                xyb1->View())
             .ok() &&
         bi::SeparateFrequencies(xyb0->ConstView(), frequency, psycho0).ok() &&
         bi::SeparateFrequencies(xyb1->ConstView(), frequency, psycho1).ok() &&
         bi::ComputeDifferenceStages(
             *psycho0, *psycho1, ToNative(fixture.options), difference, stages)
             .ok();
}

[[nodiscard]] bool ComputeNativeMap(const bt::FixturePair &fixture,
                                    bi::NativeButteraugliScratch *scratch,
                                    MapStorage *map, double *score) {
  return bi::ComputeButteraugliDistanceNative(
             ToGjxl(fixture.reference.ConstView()),
             ToGjxl(fixture.distorted.ConstView()), ToNative(fixture.options),
             scratch, map->GjxlView(), score)
      .ok();
}

template <size_t Size>
[[nodiscard]] std::vector<float>
DecodeBits(const std::array<uint32_t, Size> &bits) {
  std::vector<float> result(Size);
  for (size_t index = 0; index < Size; ++index) {
    result[index] = std::bit_cast<float>(bits[index]);
  }
  return result;
}

[[nodiscard]] bool CompareStrict(gjxl::ConstPlaneF32View actual,
                                 const std::vector<float> &expected,
                                 float *maximum, std::string_view label) {
  if (expected.size() != actual.extent.width * actual.extent.height) {
    return false;
  }
  for (size_t y = 0; y < actual.extent.height; ++y) {
    for (size_t x = 0; x < actual.extent.width; ++x) {
      const float actual_value = actual.Row(y)[x];
      const float expected_value = expected[y * actual.extent.width + x];
      const float error = std::abs(actual_value - expected_value);
      *maximum = std::max(*maximum, error);
      if (!std::isfinite(actual_value) ||
          error > bt::MapTolerance(expected_value)) {
        std::cerr << "Strict native/scalar mismatch: " << label << " x=" << x
                  << " y=" << y << " actual=" << actual_value
                  << " expected=" << expected_value << " abs=" << error
                  << " limit=" << bt::MapTolerance(expected_value) << '\n';
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool CompareAbsolute(gjxl::ConstPlaneF32View actual,
                                   const std::vector<float> &expected,
                                   float tolerance, float *maximum,
                                   std::string_view label) {
  if (expected.size() != actual.extent.width * actual.extent.height) {
    return false;
  }
  bool matches = true;
  for (size_t y = 0; y < actual.extent.height; ++y) {
    for (size_t x = 0; x < actual.extent.width; ++x) {
      const float actual_value = actual.Row(y)[x];
      const float expected_value = expected[y * actual.extent.width + x];
      const float error = std::abs(actual_value - expected_value);
      *maximum = std::max(*maximum, error);
      if (!std::isfinite(actual_value) || !std::isfinite(expected_value) ||
          error > tolerance) {
        if (matches) {
          std::cerr << "Native/dispatched mismatch: " << label << " x=" << x
                    << " y=" << y << " actual=" << actual_value
                    << " expected=" << expected_value << " abs=" << error
                    << " limit=" << tolerance << '\n';
        }
        matches = false;
      }
    }
  }
  return matches;
}

[[nodiscard]] bool CheckScalarStageGoldens(ErrorStats *errors) {
  const bt::FixturePair fixture = bt::MakeFixture({
      "intermediate",
      {golden::kIntermediateWidth, golden::kIntermediateHeight},
      bt::FixtureKind::kIntermediate,
  });
  bi::OpsinScratch opsin;
  bi::FrequencyScratch frequency;
  bi::DifferenceScratch difference;
  bi::OwnedImage3F xyb0;
  bi::OwnedImage3F xyb1;
  bi::OwnedPsychoImage psycho0;
  bi::OwnedPsychoImage psycho1;
  bi::OwnedDifferenceStages stages;
  if (!ComputeNativeStages(fixture, &opsin, &frequency, &difference, &xyb0,
                           &xyb1, &psycho0, &psycho1, &stages)) {
    std::cerr << "Native scalar-stage computation failed\n";
    return false;
  }
  constexpr size_t kGoldenFirst =
      static_cast<size_t>(bt::IntermediateStage::kMaltaMediumFrequencyY);
  for (size_t stage = 0; stage < bi::kDifferenceStageCount; ++stage) {
    const std::vector<float> expected =
        DecodeBits(golden::kIntermediateStageBits[kGoldenFirst + stage]);
    if (!CompareStrict(
            std::as_const(stages).StageView(
                static_cast<bi::DifferenceStage>(stage)),
            expected, &errors->scalar_stage,
            bt::IntermediateStageName(
                static_cast<bt::IntermediateStage>(kGoldenFirst + stage)))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckScalarFullMapGoldens(ErrorStats *errors) {
  constexpr std::array<bt::FixtureKind, golden::kFullMapCount> kKinds = {
      bt::FixtureKind::kFlat,
      bt::FixtureKind::kTexture,
      bt::FixtureKind::kContrast,
      bt::FixtureKind::kChromatic,
  };
  constexpr std::array<std::string_view, golden::kFullMapCount> kNames = {
      "flat", "texture", "contrast", "chromatic"};
  bi::NativeButteraugliScratch scratch;
  for (size_t index = 0; index < kKinds.size(); ++index) {
    const bt::FixturePair fixture = bt::MakeFixture({
        std::string(kNames[index]),
        {golden::kFullMapWidth, golden::kFullMapHeight},
        kKinds[index],
    });
    MapStorage map(fixture.reference.extent());
    double score = kScorePoison;
    const std::vector<float> expected = DecodeBits(golden::kFullMapBits[index]);
    const double expected_score =
        std::bit_cast<double>(golden::kFullMapScoreBits[index]);
    if (!ComputeNativeMap(fixture, &scratch, &map, &score) ||
        !map.PaddingIsUntouched() ||
        !CompareStrict({map.values.data(),
                        {map.extent.width, map.extent.height},
                        map.stride},
                       expected, &errors->scalar_map, kNames[index]) ||
        !std::isfinite(score) ||
        std::abs(score - expected_score) > bt::kScoreTolerance) {
      std::cerr << "Native scalar full-map mismatch: " << kNames[index]
                << " score=" << std::setprecision(12) << score
                << " expected_score=" << expected_score << '\n';
      return false;
    }
    errors->scalar_score =
        std::max(errors->scalar_score, std::abs(score - expected_score));
  }
  return true;
}

[[nodiscard]] bool CheckDifferentialCorpus(ErrorStats *errors) {
  const std::vector<bt::FixturePair> corpus =
      bt::BuildDifferentialCorpus(GJXL_FLOWER_PPM_PATH);
  bi::NativeButteraugliScratch native_scratch;
  bi::OpsinScratch opsin;
  bi::FrequencyScratch frequency;
  bi::DifferenceScratch difference;
  bi::OwnedImage3F xyb0;
  bi::OwnedImage3F xyb1;
  bi::OwnedPsychoImage psycho0;
  bi::OwnedPsychoImage psycho1;
  bi::OwnedDifferenceStages stages;
  bool limits_ok = true;
  for (const bt::FixturePair &fixture : corpus) {
    MapStorage native_map(fixture.reference.extent());
    MapStorage oracle_map(fixture.reference.extent());
    double native_score = kScorePoison;
    double oracle_score = kScorePoison;
    if (!ComputeNativeMap(fixture, &native_scratch, &native_map,
                          &native_score) ||
        !bt::ComputeLiveButteraugli(
            fixture.reference.ConstView(), fixture.distorted.ConstView(),
            fixture.options, oracle_map.OracleView(), &oracle_score) ||
        !native_map.PaddingIsUntouched() || !oracle_map.PaddingIsUntouched() ||
        !bt::PaddingIsPoisoned(fixture.reference) ||
        !bt::PaddingIsPoisoned(fixture.distorted)) {
      std::cerr << "Native full-map differential execution failed: "
                << fixture.name << '\n';
      return false;
    }
    const gjxl::ConstPlaneF32View native_map_view{
        native_map.values.data(),
        {native_map.extent.width, native_map.extent.height},
        native_map.stride};
    const std::vector<float> oracle_values = oracle_map.LogicalValues();
    if constexpr (kScalarOracle) {
      if (!CompareStrict(native_map_view, oracle_values, &errors->oracle_map,
                         fixture.name)) {
        limits_ok = false;
      }
    } else if (!CompareAbsolute(native_map_view, oracle_values,
                                bt::kNativeMapDispatchTolerance,
                                &errors->oracle_map, fixture.name)) {
      limits_ok = false;
    }
    const double score_tolerance =
        kScalarOracle ? bt::kScoreTolerance : bt::kNativeScoreDispatchTolerance;
    if (!std::isfinite(native_score) || !std::isfinite(oracle_score) ||
        std::abs(native_score - oracle_score) > score_tolerance) {
      std::cerr << "Native/" << kOracleName
                << " score mismatch: " << fixture.name
                << " native=" << native_score << " expected=" << oracle_score
                << " abs=" << std::abs(native_score - oracle_score)
                << " limit=" << score_tolerance << '\n';
      limits_ok = false;
    }
    errors->oracle_score =
        std::max(errors->oracle_score, std::abs(native_score - oracle_score));
    if (fixture.name.starts_with("identity_")) {
      for (float value : native_map.LogicalValues()) {
        if (std::abs(value) > bt::kIdentityTolerance) {
          limits_ok = false;
        }
      }
      if (std::abs(native_score) > bt::kIdentityTolerance) {
        limits_ok = false;
      }
    }

    const bt::OracleExtent extent = fixture.reference.extent();
    if (extent.width >= 8 && extent.height >= 8) {
      bt::IntermediateStageOutput oracle;
      if (!ComputeNativeStages(fixture, &opsin, &frequency, &difference, &xyb0,
                               &xyb1, &psycho0, &psycho1, &stages) ||
          !bt::ComputePinnedIntermediateStages(fixture.reference.ConstView(),
                                               fixture.distorted.ConstView(),
                                               fixture.options, &oracle)) {
        std::cerr << "Native stage differential execution failed: "
                  << fixture.name << '\n';
        return false;
      }
      constexpr size_t kOracleFirst =
          static_cast<size_t>(bt::IntermediateStage::kMaltaMediumFrequencyY);
      for (size_t stage = 0; stage < bi::kDifferenceStageCount; ++stage) {
        const gjxl::ConstPlaneF32View native_stage =
            std::as_const(stages).StageView(
                static_cast<bi::DifferenceStage>(stage));
        const std::string label =
            fixture.name + "/" +
            bt::IntermediateStageName(
                static_cast<bt::IntermediateStage>(kOracleFirst + stage));
        if constexpr (kScalarOracle) {
          if (!CompareStrict(native_stage, oracle.plane[kOracleFirst + stage],
                             &errors->oracle_stage, label)) {
            limits_ok = false;
          }
        } else if (!CompareAbsolute(native_stage,
                                    oracle.plane[kOracleFirst + stage],
                                    bt::kNativeDifferenceDispatchTolerance,
                                    &errors->oracle_stage, label)) {
          limits_ok = false;
        }
      }
    }
  }
  return limits_ok;
}

[[nodiscard]] bool CheckPerturbationDetection() {
  std::vector<float> expected = {0.0f, 1.0f, -2.0f};
  std::vector<float> perturbed = expected;
  perturbed[1] += 0.01f;
  const auto matches = [](const std::vector<float> &actual,
                          const std::vector<float> &reference,
                          float tolerance) {
    if (actual.size() != reference.size()) {
      return false;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
      if (std::abs(actual[index] - reference[index]) > tolerance) {
        return false;
      }
    }
    return true;
  };
  return matches(expected, expected, 1.0e-6f) &&
         !matches(expected, perturbed, 1.0e-6f);
}

} // namespace

int main() {
  ErrorStats errors;
  const bool scalar_stages_ok = CheckScalarStageGoldens(&errors);
  const bool scalar_maps_ok = CheckScalarFullMapGoldens(&errors);
  const bool corpus_ok = CheckDifferentialCorpus(&errors);
  const bool perturbation_ok = CheckPerturbationDetection();
  const bool passed =
      scalar_stages_ok && scalar_maps_ok && corpus_ok && perturbation_ok;
  std::cout << std::setprecision(9)
            << "Maximum native-vs-scalar difference-stage error: abs="
            << errors.scalar_stage << '\n'
            << "Maximum native-vs-scalar full-map error: abs="
            << errors.scalar_map << " score=" << errors.scalar_score << '\n'
            << "Maximum native-vs-" << kOracleName
            << " complete-corpus difference-stage error: abs="
            << errors.oracle_stage << '\n'
            << "Maximum native-vs-" << kOracleName
            << " complete-corpus full-map error: abs=" << errors.oracle_map
            << " score=" << errors.oracle_score << '\n'
            << (passed
                    ? "All native Butteraugli distance differentials passed.\n"
                    : "Native Butteraugli distance differentials failed.\n");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
