// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <vector>

#include "codec/butteraugli_internal.h"

namespace gjxl::butteraugli_internal {

/// Parameters used by the native Butteraugli distance implementation.
struct NativeButteraugliParams {
  float hf_asymmetry = 1.0f;
  float x_multiplier = 1.0f;
  float intensity_target = 80.0f;
};

/// Observable single-scale stages retained for differential validation.
enum class DifferenceStage : size_t {
  kMaltaMediumFrequencyY,
  kMaltaMediumFrequencyX,
  kMaltaHighFrequencyY,
  kMaltaHighFrequencyX,
  kMaltaUltraHighFrequencyY,
  kMaltaUltraHighFrequencyX,
  kMask,
  kMaskedAcY,
  kFinalComposition,
  kCount,
};

inline constexpr size_t kDifferenceStageCount =
    static_cast<size_t>(DifferenceStage::kCount);

class NativePreparedButteraugliReference;

/// Contiguous storage for all observable native difference stages.
class OwnedDifferenceStages {
public:
  OwnedDifferenceStages() = default;

  OwnedDifferenceStages(const OwnedDifferenceStages &) = delete;
  OwnedDifferenceStages &operator=(const OwnedDifferenceStages &) = delete;
  OwnedDifferenceStages(OwnedDifferenceStages &&) noexcept = default;
  OwnedDifferenceStages &operator=(OwnedDifferenceStages &&) noexcept = default;

  /// Reuses storage for the same extent and otherwise commits a new allocation
  /// only after it succeeds.
  [[nodiscard]] Status Resize(Extent2D extent);

  [[nodiscard]] PlaneF32View StageView(DifferenceStage stage) noexcept;
  [[nodiscard]] ConstPlaneF32View
  StageView(DifferenceStage stage) const noexcept;
  [[nodiscard]] Extent2D extent() const noexcept { return extent_; }
  [[nodiscard]] size_t plane_size() const noexcept { return plane_size_; }

private:
  Extent2D extent_;
  size_t plane_size_ = 0;
  std::vector<float> values_;
};

/// Reusable allocation state for one single-scale difference computation.
class DifferenceScratch {
public:
  DifferenceScratch() = default;

  DifferenceScratch(const DifferenceScratch &) = delete;
  DifferenceScratch &operator=(const DifferenceScratch &) = delete;
  DifferenceScratch(DifferenceScratch &&) noexcept = default;
  DifferenceScratch &operator=(DifferenceScratch &&) noexcept = default;

private:
  [[nodiscard]] Status Prepare(Extent2D extent);
  [[nodiscard]] Status PrepareOutputStaging(Extent2D extent) {
    return staged_output_.Resize(extent);
  }

  OwnedPlaneF32 malta_diffs_;
  OwnedPlaneF32 mask_activity0_;
  OwnedPlaneF32 mask_activity1_;
  OwnedPlaneF32 mask_precomputed0_;
  OwnedPlaneF32 mask_precomputed1_;
  OwnedPlaneF32 mask_blurred0_;
  OwnedPlaneF32 mask_blurred1_;
  OwnedImage3F block_diff_ac_;
  OwnedImage3F block_diff_dc_;
  BlurScratch blur_;
  OwnedDifferenceStages staged_output_;

  friend Status ComputeDifferenceStages(const OwnedPsychoImage &,
                                        const OwnedPsychoImage &,
                                        NativeButteraugliParams,
                                        DifferenceScratch *,
                                        OwnedDifferenceStages *);
  friend Status PrepareButteraugliReferenceNative(
    ConstImage3FView,
    NativeButteraugliParams,
    NativePreparedButteraugliReference*);
};

/// Reusable allocation state for complete native map and score computation.
class NativeButteraugliScratch {
public:
  NativeButteraugliScratch() = default;

  NativeButteraugliScratch(const NativeButteraugliScratch &) = delete;
  NativeButteraugliScratch &
  operator=(const NativeButteraugliScratch &) = delete;
  NativeButteraugliScratch(NativeButteraugliScratch &&) noexcept = default;
  NativeButteraugliScratch &
  operator=(NativeButteraugliScratch &&) noexcept = default;

private:
  OwnedImage3F expanded0_;
  OwnedImage3F expanded1_;
  OwnedImage3F subsampled0_;
  OwnedImage3F subsampled1_;
  OwnedImage3F xyb0_;
  OwnedImage3F xyb1_;
  OwnedPsychoImage psycho0_;
  OwnedPsychoImage psycho1_;
  OwnedDifferenceStages main_stages_;
  OwnedDifferenceStages sub_stages_;
  OwnedPlaneF32 final_map_;
  OpsinScratch opsin_;
  FrequencyScratch frequency_;
  DifferenceScratch difference_;

  friend Status ComputeButteraugliDistanceNative(ConstImage3FView,
                                                 ConstImage3FView,
                                                 NativeButteraugliParams,
                                                 NativeButteraugliScratch *,
                                                 PlaneF32View, double *);
};

/// Target-invariant native reference representation plus reusable comparison
/// scratch. This is wrapped by the public PreparedButteraugliReference API.
class NativePreparedButteraugliReference {
public:
  NativePreparedButteraugliReference() = default;

  NativePreparedButteraugliReference(
    const NativePreparedButteraugliReference&) = delete;
  NativePreparedButteraugliReference& operator=(
    const NativePreparedButteraugliReference&) = delete;
  NativePreparedButteraugliReference(
    NativePreparedButteraugliReference&&) noexcept = default;
  NativePreparedButteraugliReference& operator=(
    NativePreparedButteraugliReference&&) noexcept = default;

  [[nodiscard]] Extent2D extent() const noexcept {
    return requested_extent_;
  }
  [[nodiscard]] NativeButteraugliParams params() const noexcept {
    return params_;
  }
  [[nodiscard]] bool ready() const noexcept { return ready_; }

private:
  NativeButteraugliParams params_;
  Extent2D requested_extent_;
  Extent2D working_extent_;
  size_t xborder_ = 0;
  size_t yborder_ = 0;
  bool expanded_ = false;
  bool has_subscale_ = false;
  bool ready_ = false;

  OwnedImage3F main_input_;
  OwnedImage3F sub_input_;
  OwnedImage3F main_xyb_;
  OwnedImage3F sub_xyb_;
  OwnedPsychoImage main_reference_;
  OwnedPsychoImage sub_reference_;
  OwnedPsychoImage main_distorted_;
  OwnedPsychoImage sub_distorted_;
  OwnedDifferenceStages main_stages_;
  OwnedDifferenceStages sub_stages_;
  OwnedPlaneF32 final_map_;
  OpsinScratch main_opsin_;
  OpsinScratch sub_opsin_;
  FrequencyScratch main_frequency_;
  FrequencyScratch sub_frequency_;
  DifferenceScratch main_difference_;
  DifferenceScratch sub_difference_;

  friend Status PrepareButteraugliReferenceNative(
    ConstImage3FView,
    NativeButteraugliParams,
    NativePreparedButteraugliReference*);
  friend Status CompareButteraugliReferenceNative(
    NativePreparedButteraugliReference*,
    ConstImage3FView,
    PlaneF32View,
    double*);
};

/// Adds the symmetric weighted L2 term to `output`.
[[nodiscard]] Status L2Diff(ConstPlaneF32View image0, ConstPlaneF32View image1,
                            float weight, PlaneF32View output);

/// Replaces `output` with the symmetric weighted L2 term.
[[nodiscard]] Status SetL2Diff(ConstPlaneF32View image0,
                               ConstPlaneF32View image1, float weight,
                               PlaneF32View output);

/// Adds Butteraugli's asymmetric high-frequency L2 term to `output`.
[[nodiscard]] Status L2DiffAsymmetric(ConstPlaneF32View image0,
                                      ConstPlaneF32View image1,
                                      float weight_0_gt_1, float weight_0_lt_1,
                                      PlaneF32View output);

/// Adds a full or low-frequency Malta response to `output`. Malta support is
/// zero outside the image rather than mirrored or clamped.
[[nodiscard]] Status MaltaDiffMap(ConstPlaneF32View image0,
                                  ConstPlaneF32View image1, bool low_frequency,
                                  double weight_0_gt_1, double weight_0_lt_1,
                                  double norm, OwnedPlaneF32 *diffs,
                                  PlaneF32View output);

/// Applies Butteraugli's step-three, three-minimum fuzzy erosion.
[[nodiscard]] Status FuzzyErosion(ConstPlaneF32View input, PlaneF32View output);

[[nodiscard]] float MaskY(float delta) noexcept;
[[nodiscard]] float MaskDcY(float delta) noexcept;

/// Computes every observable single-scale difference stage. Both psycho
/// images must have equal dimensions of at least 8x8. The prior output remains
/// unchanged on failure.
[[nodiscard]] Status ComputeDifferenceStages(const OwnedPsychoImage &reference,
                                             const OwnedPsychoImage &distorted,
                                             NativeButteraugliParams params,
                                             DifferenceScratch *scratch,
                                             OwnedDifferenceStages *output);

/// Computes the complete native Butteraugli map and maximum score. Dimensions
/// below eight are edge-expanded and cropped; dimensions at least 15x15 use
/// exactly one additional half-resolution scale. Output writes occur only
/// after every input has been consumed, so input/output overlap is supported.
[[nodiscard]] Status ComputeButteraugliDistanceNative(
    ConstImage3FView reference, ConstImage3FView distorted,
    NativeButteraugliParams params, NativeButteraugliScratch *scratch,
    PlaneF32View distance_map, double *score);

[[nodiscard]] Status PrepareButteraugliReferenceNative(
    ConstImage3FView reference, NativeButteraugliParams params,
    NativePreparedButteraugliReference* prepared);

[[nodiscard]] Status CompareButteraugliReferenceNative(
    NativePreparedButteraugliReference* prepared,
    ConstImage3FView distorted, PlaneF32View distance_map, double* score);

} // namespace gjxl::butteraugli_internal
