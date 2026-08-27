// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates always-available native Butteraugli difference and map stages.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "butteraugli_test_tolerances.h"
#include "codec/butteraugli_distance_internal.h"

namespace {

namespace bi = gjxl::butteraugli_internal;
namespace bt = gjxl::butteraugli_test;

constexpr float kPoison = -991.0f;
constexpr double kScorePoison = -313.0;

struct PlaneStorage {
  explicit PlaneStorage(gjxl::Extent2D image_extent, size_t padding = 4)
      : extent(image_extent), stride(extent.width + padding),
        values(stride * extent.height, kPoison) {}

  [[nodiscard]] gjxl::PlaneF32View View() {
    return {values.data(), extent, stride};
  }
  [[nodiscard]] gjxl::ConstPlaneF32View ConstView() const {
    return {values.data(), extent, stride};
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

  gjxl::Extent2D extent;
  size_t stride;
  std::vector<float> values;
};

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D image_extent, size_t padding = 4)
      : extent(image_extent), stride(extent.width + padding) {
    for (std::vector<float> &values : plane) {
      values.assign(stride * extent.height, kPoison);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{{{plane[0].data(), extent, stride},
              {plane[1].data(), extent, stride},
              {plane[2].data(), extent, stride}}}};
  }
  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{{{plane[0].data(), extent, stride},
              {plane[1].data(), extent, stride},
              {plane[2].data(), extent, stride}}}};
  }
  [[nodiscard]] bool PaddingIsUntouched() const {
    for (const std::vector<float> &values : plane) {
      for (size_t y = 0; y < extent.height; ++y) {
        for (size_t x = extent.width; x < stride; ++x) {
          if (values[y * stride + x] != kPoison) {
            return false;
          }
        }
      }
    }
    return true;
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

[[nodiscard]] bool IsInvalid(const gjxl::Status &status) {
  return status.code() == gjxl::StatusCode::kInvalidArgument;
}

void FillImage(ImageStorage *image, float offset = 0.0f) {
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < image->extent.height; ++y) {
      for (size_t x = 0; x < image->extent.width; ++x) {
        image->plane[channel][y * image->stride + x] =
            offset + 0.02f * static_cast<float>(channel + 1) +
            0.003f * static_cast<float>(x + 5 * y) +
            0.04f *
                std::sin(0.31f * static_cast<float>(3 * x + 2 * y + channel));
      }
    }
  }
}

[[nodiscard]] bool CheckDifferenceStorage() {
  bi::OwnedDifferenceStages stages;
  if (stages.StageView(bi::DifferenceStage::kMask).valid() ||
      !stages.extent().empty() || !stages.Resize({4, 3}).ok() ||
      stages.plane_size() != 12) {
    return false;
  }
  float *const first =
      stages.StageView(bi::DifferenceStage::kMaltaMediumFrequencyY).data;
  if (stages.StageView(bi::DifferenceStage::kMaltaMediumFrequencyX).data !=
          first + 12 ||
      stages.StageView(bi::DifferenceStage::kFinalComposition).data !=
          first + 8 * 12 ||
      stages.StageView(bi::DifferenceStage::kCount).valid()) {
    return false;
  }
  stages.StageView(bi::DifferenceStage::kMask).data[7] = 19.0f;
  if (!stages.Resize({4, 3}).ok() ||
      stages.StageView(bi::DifferenceStage::kMaltaMediumFrequencyY).data !=
          first ||
      stages.StageView(bi::DifferenceStage::kMask).data[7] != 19.0f ||
      !stages.Resize({2, 5}).ok()) {
    return false;
  }
  stages.StageView(bi::DifferenceStage::kFinalComposition).data[0] = 23.0f;
  float *const resized =
      stages.StageView(bi::DifferenceStage::kMaltaMediumFrequencyY).data;
  if (!IsInvalid(stages.Resize({0, 5})) ||
      !IsInvalid(stages.Resize(
          {std::numeric_limits<size_t>::max() / bi::kDifferenceStageCount + 1,
           1})) ||
      stages.extent() != gjxl::Extent2D{2, 5} ||
      stages.StageView(bi::DifferenceStage::kMaltaMediumFrequencyY).data !=
          resized ||
      stages.StageView(bi::DifferenceStage::kFinalComposition).data[0] !=
          23.0f) {
    return false;
  }
  bi::OwnedDifferenceStages moved = std::move(stages);
  return moved.StageView(bi::DifferenceStage::kMask).valid() &&
         stages.Resize({2, 5}).ok();
}

[[nodiscard]] bool CheckL2Terms() {
  constexpr gjxl::Extent2D kExtent{6, 1};
  PlaneStorage image0(kExtent);
  PlaneStorage image1(kExtent);
  PlaneStorage output(kExtent);
  const std::array<float, 6> values0 = {-2.0f, -2.0f, -2.0f, 2.0f, 2.0f, 2.0f};
  const std::array<float, 6> values1 = {-0.8f, -2.0f, -2.1f, 0.8f, 2.0f, 2.1f};
  std::copy(values0.begin(), values0.end(), image0.View().Row(0));
  std::copy(values1.begin(), values1.end(), image1.View().Row(0));
  std::fill_n(output.View().Row(0), kExtent.width, 1.0f);
  if (!bi::L2Diff(image0.ConstView(), image1.ConstView(), 2.0f, output.View())
           .ok()) {
    return false;
  }
  for (size_t x = 0; x < kExtent.width; ++x) {
    const float difference = values0[x] - values1[x];
    if (output.View().Row(0)[x] != difference * difference * 2.0f + 1.0f) {
      return false;
    }
  }
  if (!bi::SetL2Diff(image0.ConstView(), image1.ConstView(), 3.0f,
                     output.View())
           .ok()) {
    return false;
  }
  for (size_t x = 0; x < kExtent.width; ++x) {
    const float difference = values0[x] - values1[x];
    if (output.View().Row(0)[x] != difference * difference * 3.0f) {
      return false;
    }
  }
  std::fill_n(output.View().Row(0), kExtent.width, 0.0f);
  if (!bi::L2DiffAsymmetric(image0.ConstView(), image1.ConstView(), 4.0f, 5.0f,
                            output.View())
           .ok() ||
      output.View().Row(0)[0] <= output.View().Row(0)[1] ||
      output.View().Row(0)[3] <= output.View().Row(0)[4] ||
      output.View().Row(0)[2] <= 0.0f || output.View().Row(0)[5] <= 0.0f) {
    return false;
  }

  const std::vector<float> original = output.values;
  gjxl::ConstPlaneF32View invalid = image0.ConstView();
  invalid.data = nullptr;
  return IsInvalid(
             bi::L2Diff(invalid, image1.ConstView(), 1.0f, output.View())) &&
         IsInvalid(bi::L2Diff(image0.ConstView(), image1.ConstView(), -1.0f,
                              output.View())) &&
         IsInvalid(bi::L2DiffAsymmetric(image0.ConstView(), image1.ConstView(),
                                        std::numeric_limits<float>::quiet_NaN(),
                                        1.0f, output.View())) &&
         output.values == original && output.PaddingIsUntouched();
}

[[nodiscard]] bool CheckMaltaAndMaskPrimitives() {
  constexpr gjxl::Extent2D kExtent{9, 9};
  PlaneStorage image0(kExtent);
  PlaneStorage image1(kExtent);
  PlaneStorage low(kExtent);
  PlaneStorage full(kExtent);
  for (size_t y = 0; y < kExtent.height; ++y) {
    std::fill_n(image0.View().Row(y), kExtent.width, 0.0f);
    std::fill_n(image1.View().Row(y), kExtent.width, 0.0f);
    std::fill_n(low.View().Row(y), kExtent.width, 0.0f);
    std::fill_n(full.View().Row(y), kExtent.width, 0.0f);
  }
  image0.View().Row(4)[4] = 1.0f;
  bi::OwnedPlaneF32 diffs;
  if (!bi::MaltaDiffMap(image0.ConstView(), image1.ConstView(), true, 2.0, 3.0,
                        5.0, &diffs, low.View())
           .ok() ||
      !bi::MaltaDiffMap(image0.ConstView(), image1.ConstView(), false, 2.0, 3.0,
                        5.0, &diffs, full.View())
           .ok() ||
      low.View().Row(4)[4] <= 0.0f || full.View().Row(4)[4] <= 0.0f ||
      low.View().Row(4)[4] == full.View().Row(4)[4] ||
      low.View().Row(0)[0] < 0.0f || full.View().Row(0)[0] < 0.0f ||
      !low.PaddingIsUntouched() || !full.PaddingIsUntouched()) {
    return false;
  }
  PlaneStorage too_small({7, 9});
  const std::vector<float> original = full.values;
  if (!IsInvalid(bi::MaltaDiffMap(too_small.ConstView(), too_small.ConstView(),
                                  true, 1.0, 1.0, 1.0, &diffs,
                                  too_small.View())) ||
      !IsInvalid(bi::MaltaDiffMap(image0.ConstView(), image1.ConstView(), true,
                                  -1.0, 1.0, 1.0, &diffs, full.View())) ||
      full.values != original) {
    return false;
  }

  PlaneStorage erosion_input({7, 7});
  PlaneStorage erosion_output({7, 7});
  for (size_t y = 0; y < 7; ++y) {
    std::fill_n(erosion_input.View().Row(y), 7, 4.0f);
  }
  if (!bi::FuzzyErosion(erosion_input.ConstView(), erosion_output.View())
           .ok()) {
    return false;
  }
  for (size_t y = 0; y < 7; ++y) {
    for (size_t x = 0; x < 7; ++x) {
      if (erosion_output.View().Row(y)[x] != 4.0f) {
        return false;
      }
    }
  }
  erosion_input.View().Row(3)[3] = 1.0f;
  if (!bi::FuzzyErosion(erosion_input.ConstView(), erosion_output.View())
           .ok() ||
      std::abs(erosion_output.View().Row(3)[3] - 1.55f) > 1.0e-6f ||
      !erosion_output.PaddingIsUntouched() ||
      !IsInvalid(
          bi::FuzzyErosion(erosion_input.ConstView(), erosion_input.View())) ||
      !(bi::MaskY(0.0f) > bi::MaskY(1.0f)) ||
      !(bi::MaskDcY(0.0f) > bi::MaskDcY(1.0f))) {
    return false;
  }
  return true;
}

void FillPsycho(bi::OwnedPsychoImage *image, float offset) {
  const std::array<gjxl::PlaneF32View, 10> planes = {
      image->LowFrequencyView().plane[0],
      image->LowFrequencyView().plane[1],
      image->LowFrequencyView().plane[2],
      image->MediumFrequencyView().plane[0],
      image->MediumFrequencyView().plane[1],
      image->MediumFrequencyView().plane[2],
      image->HighFrequencyView(0),
      image->HighFrequencyView(1),
      image->UltraHighFrequencyView(0),
      image->UltraHighFrequencyView(1),
  };
  for (size_t plane = 0; plane < planes.size(); ++plane) {
    for (size_t y = 0; y < planes[plane].extent.height; ++y) {
      for (size_t x = 0; x < planes[plane].extent.width; ++x) {
        planes[plane].Row(y)[x] = offset + 0.01f * static_cast<float>(plane) +
                                  0.001f * static_cast<float>(x + 3 * y);
      }
    }
  }
}

[[nodiscard]] std::vector<float>
SnapshotStages(const bi::OwnedDifferenceStages &stages) {
  std::vector<float> result;
  result.reserve(bi::kDifferenceStageCount * stages.plane_size());
  for (size_t stage = 0; stage < bi::kDifferenceStageCount; ++stage) {
    const gjxl::ConstPlaneF32View plane =
        stages.StageView(static_cast<bi::DifferenceStage>(stage));
    for (size_t y = 0; y < plane.extent.height; ++y) {
      result.insert(result.end(), plane.Row(y),
                    plane.Row(y) + plane.extent.width);
    }
  }
  return result;
}

[[nodiscard]] bool CheckDifferenceStageValidation() {
  bi::OwnedPsychoImage reference;
  bi::OwnedPsychoImage distorted;
  bi::OwnedDifferenceStages output;
  bi::DifferenceScratch scratch;
  if (!reference.Resize({8, 8}).ok() || !distorted.Resize({8, 8}).ok() ||
      !output.Resize({3, 2}).ok()) {
    return false;
  }
  FillPsycho(&reference, 0.0f);
  FillPsycho(&distorted, 0.02f);
  for (size_t stage = 0; stage < bi::kDifferenceStageCount; ++stage) {
    gjxl::PlaneF32View plane =
        output.StageView(static_cast<bi::DifferenceStage>(stage));
    std::fill_n(plane.data, output.plane_size(), -17.0f);
  }
  const std::vector<float> original = SnapshotStages(output);
  const auto check = [&](const bi::OwnedPsychoImage &requested_reference,
                         const bi::OwnedPsychoImage &requested_distorted,
                         bi::NativeButteraugliParams params,
                         bi::DifferenceScratch *requested_scratch) {
    return IsInvalid(bi::ComputeDifferenceStages(requested_reference,
                                                 requested_distorted, params,
                                                 requested_scratch, &output)) &&
           SnapshotStages(output) == original;
  };
  if (!check(reference, distorted, {.hf_asymmetry = 0.0f}, &scratch) ||
      !check(reference, distorted,
             {.intensity_target = std::numeric_limits<float>::quiet_NaN()},
             &scratch) ||
      !check(reference, distorted, {}, nullptr)) {
    return false;
  }
  bi::OwnedPsychoImage mismatched;
  bi::OwnedPsychoImage too_small;
  if (!mismatched.Resize({9, 8}).ok() || !too_small.Resize({7, 8}).ok()) {
    return false;
  }
  FillPsycho(&mismatched, 0.0f);
  FillPsycho(&too_small, 0.0f);
  if (!check(reference, mismatched, {}, &scratch) ||
      !check(too_small, too_small, {}, &scratch)) {
    return false;
  }
  reference.HighFrequencyView(0).Row(2)[3] =
      std::numeric_limits<float>::infinity();
  if (!check(reference, distorted, {}, &scratch)) {
    return false;
  }
  FillPsycho(&reference, 0.0f);
  if (!bi::ComputeDifferenceStages(reference, distorted, {}, &scratch, &output)
           .ok() ||
      output.extent() != gjxl::Extent2D{8, 8}) {
    return false;
  }
  for (float value : SnapshotStages(output)) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RunNative(const ImageStorage &reference,
                             const ImageStorage &distorted,
                             bi::NativeButteraugliParams params,
                             bi::NativeButteraugliScratch *scratch,
                             PlaneStorage *map, double *score) {
  return bi::ComputeButteraugliDistanceNative(reference.ConstView(),
                                              distorted.ConstView(), params,
                                              scratch, map->View(), score)
      .ok();
}

[[nodiscard]] bool CheckCompleteMapBehavior() {
  constexpr std::array<gjxl::Extent2D, 8> kExtents = {
      gjxl::Extent2D{1, 1},   gjxl::Extent2D{3, 7},   gjxl::Extent2D{7, 3},
      gjxl::Extent2D{7, 8},   gjxl::Extent2D{8, 8},   gjxl::Extent2D{14, 17},
      gjxl::Extent2D{15, 15}, gjxl::Extent2D{17, 19},
  };
  bi::NativeButteraugliScratch scratch;
  for (gjxl::Extent2D extent : kExtents) {
    ImageStorage reference(extent, 3);
    ImageStorage distorted(extent, 5);
    FillImage(&reference);
    FillImage(&distorted, 0.01f);
    PlaneStorage map(extent, 7);
    double score = kScorePoison;
    if (!RunNative(reference, distorted, {}, &scratch, &map, &score) ||
        !std::isfinite(score) || score < 0.0 || !map.PaddingIsUntouched() ||
        !reference.PaddingIsUntouched() || !distorted.PaddingIsUntouched()) {
      std::cerr << "Native complete-map case failed: " << extent.width << 'x'
                << extent.height << '\n';
      return false;
    }
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        if (!std::isfinite(map.View().Row(y)[x]) ||
            map.View().Row(y)[x] < 0.0f) {
          return false;
        }
      }
    }

    PlaneStorage identity_map(extent, 2);
    double identity_score = kScorePoison;
    if (!RunNative(reference, reference, {}, &scratch, &identity_map,
                   &identity_score) ||
        std::abs(identity_score) > bt::kIdentityTolerance) {
      return false;
    }
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        if (std::abs(identity_map.View().Row(y)[x]) > bt::kIdentityTolerance) {
          return false;
        }
      }
    }
  }

  ImageStorage reference({1, 1});
  ImageStorage distorted({1, 1});
  ImageStorage expanded_reference({8, 8});
  ImageStorage expanded_distorted({8, 8});
  FillImage(&reference);
  FillImage(&distorted, 0.02f);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < 8; ++y) {
      std::fill_n(expanded_reference.View().plane[channel].Row(y), 8,
                  reference.ConstView().plane[channel].Row(0)[0]);
      std::fill_n(expanded_distorted.View().plane[channel].Row(y), 8,
                  distorted.ConstView().plane[channel].Row(0)[0]);
    }
  }
  PlaneStorage small_map({1, 1});
  PlaneStorage expanded_map({8, 8});
  double small_score = 0.0;
  double expanded_score = 0.0;
  if (!RunNative(reference, distorted, {}, &scratch, &small_map,
                 &small_score) ||
      !RunNative(expanded_reference, expanded_distorted, {}, &scratch,
                 &expanded_map, &expanded_score) ||
      std::abs(small_map.View().Row(0)[0] - expanded_map.View().Row(3)[3]) >
          1.0e-6f ||
      std::abs(small_score - small_map.View().Row(0)[0]) > 1.0e-6) {
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckValidationAtomicityAndAliasing() {
  constexpr gjxl::Extent2D kExtent{9, 8};
  ImageStorage reference(kExtent);
  ImageStorage distorted(kExtent);
  FillImage(&reference);
  FillImage(&distorted, 0.01f);
  bi::NativeButteraugliScratch scratch;
  const auto check = [&](gjxl::ConstImage3FView requested_reference,
                         gjxl::ConstImage3FView requested_distorted,
                         bi::NativeButteraugliParams params,
                         bi::NativeButteraugliScratch *requested_scratch,
                         gjxl::PlaneF32View requested_map, double *score) {
    const std::vector<float> original = std::vector<float>(
        requested_map.data,
        requested_map.data +
            requested_map.stride * requested_map.extent.height);
    const double original_score = score == nullptr ? 0.0 : *score;
    const gjxl::Status status = bi::ComputeButteraugliDistanceNative(
        requested_reference, requested_distorted, params, requested_scratch,
        requested_map, score);
    return IsInvalid(status) &&
           std::equal(original.begin(), original.end(), requested_map.data) &&
           (score == nullptr || *score == original_score);
  };

  PlaneStorage output(kExtent);
  double score = kScorePoison;
  gjxl::ConstImage3FView invalid_reference = reference.ConstView();
  invalid_reference.plane[0].data = nullptr;
  if (!check(invalid_reference, distorted.ConstView(), {}, &scratch,
             output.View(), &score) ||
      !check(reference.ConstView(), distorted.ConstView(),
             {.hf_asymmetry = 0.0f}, &scratch, output.View(), &score) ||
      !check(reference.ConstView(), distorted.ConstView(),
             {.x_multiplier = std::numeric_limits<float>::infinity()}, &scratch,
             output.View(), &score) ||
      !check(reference.ConstView(), distorted.ConstView(), {}, nullptr,
             output.View(), &score) ||
      !check(reference.ConstView(), distorted.ConstView(), {}, &scratch,
             output.View(), nullptr)) {
    return false;
  }
  reference.View().plane[1].Row(2)[3] = std::numeric_limits<float>::quiet_NaN();
  if (!check(reference.ConstView(), distorted.ConstView(), {}, &scratch,
             output.View(), &score)) {
    return false;
  }
  FillImage(&reference);

  PlaneStorage expected(kExtent);
  double expected_score = 0.0;
  if (!RunNative(reference, distorted, {}, &scratch, &expected,
                 &expected_score)) {
    return false;
  }
  double alias_score = 0.0;
  gjxl::PlaneF32View alias{reference.plane[0].data(), kExtent,
                           reference.stride};
  if (!bi::ComputeButteraugliDistanceNative(reference.ConstView(),
                                            distorted.ConstView(), {}, &scratch,
                                            alias, &alias_score)
           .ok() ||
      std::abs(alias_score - expected_score) > 1.0e-7) {
    return false;
  }
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      if (reference.plane[0][y * reference.stride + x] !=
          expected.View().Row(y)[x]) {
        return false;
      }
    }
  }
  return reference.PaddingIsUntouched() && distorted.PaddingIsUntouched();
}

} // namespace

int main() {
  if (!CheckDifferenceStorage() || !CheckL2Terms() ||
      !CheckMaltaAndMaskPrimitives() || !CheckDifferenceStageValidation() ||
      !CheckCompleteMapBehavior() || !CheckValidationAtomicityAndAliasing()) {
    return EXIT_FAILURE;
  }
  std::cout << "All native Butteraugli distance tests passed.\n";
  return EXIT_SUCCESS;
}
