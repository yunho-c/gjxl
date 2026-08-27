// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "core/image.h"
#include "core/status.h"

namespace gjxl::butteraugli_internal {

inline constexpr float kOpsinBlurSigma = 1.2f;
inline constexpr float kLowFrequencyBlurSigma = 7.15593339443f;
inline constexpr float kHighFrequencyBlurSigma = 3.22489901262f;
inline constexpr float kUltraHighFrequencyBlurSigma = 1.56416327805f;
inline constexpr float kMaskBlurSigma = 2.7f;

inline constexpr std::array<float, 5> kPinnedBlurSigmas = {
    kOpsinBlurSigma,         kLowFrequencyBlurSigma,
    kHighFrequencyBlurSigma, kUltraHighFrequencyBlurSigma,
    kMaskBlurSigma,
};

/// Contiguous single-plane float storage for internal Butteraugli stages.
class OwnedPlaneF32 {
public:
  /// Reuses storage for the same extent and otherwise commits a new allocation
  /// only after it succeeds.
  [[nodiscard]] Status Resize(Extent2D extent);

  [[nodiscard]] PlaneF32View View() noexcept;
  [[nodiscard]] ConstPlaneF32View ConstView() const noexcept;
  [[nodiscard]] Extent2D extent() const noexcept { return extent_; }
  [[nodiscard]] size_t size() const noexcept { return values_.size(); }

private:
  Extent2D extent_;
  std::vector<float> values_;
};

/// Contiguous planar float storage for three internal Butteraugli channels.
class OwnedImage3F {
public:
  /// Reuses storage for the same extent and otherwise commits a new allocation
  /// only after it succeeds.
  [[nodiscard]] Status Resize(Extent2D extent);

  [[nodiscard]] Image3FView View() noexcept;
  [[nodiscard]] ConstImage3FView ConstView() const noexcept;
  [[nodiscard]] Extent2D extent() const noexcept { return extent_; }
  [[nodiscard]] size_t plane_size() const noexcept { return plane_size_; }

private:
  Extent2D extent_;
  size_t plane_size_ = 0;
  std::vector<float> values_;
};

/// Reusable allocation state for scalar Gaussian blur calls.
class BlurScratch {
public:
  BlurScratch() = default;

  BlurScratch(const BlurScratch &) = delete;
  BlurScratch &operator=(const BlurScratch &) = delete;
  BlurScratch(BlurScratch &&) noexcept = default;
  BlurScratch &operator=(BlurScratch &&) noexcept = default;

private:
  [[nodiscard]] Status Prepare(Extent2D input_extent, size_t kernel_size,
                               bool needs_transposed);

  OwnedPlaneF32 transposed_;
  std::vector<float> kernel_;

  friend Status GaussianBlur(ConstPlaneF32View, float, BlurScratch *,
                             PlaneF32View);
};

/// Applies one of the five Gaussian kernels used by pinned Butteraugli.
/// Input and output must have equal extents and must not start at the same
/// address. Partial overlap is outside this internal contract.
[[nodiscard]] Status GaussianBlur(ConstPlaneF32View input, float sigma,
                                  BlurScratch *scratch, PlaneF32View output);

/// Complete frequency decomposition used by Butteraugli after XYB conversion.
/// LF and MF have X/Y/B planes; HF and UHF only retain X/Y.
class OwnedPsychoImage {
public:
  OwnedPsychoImage() = default;

  OwnedPsychoImage(const OwnedPsychoImage &) = delete;
  OwnedPsychoImage &operator=(const OwnedPsychoImage &) = delete;
  OwnedPsychoImage(OwnedPsychoImage &&) noexcept = default;
  OwnedPsychoImage &operator=(OwnedPsychoImage &&) noexcept = default;

  /// Reuses storage for the same extent and otherwise commits a new allocation
  /// only after it succeeds.
  [[nodiscard]] Status Resize(Extent2D extent);

  [[nodiscard]] Image3FView LowFrequencyView() noexcept;
  [[nodiscard]] ConstImage3FView LowFrequencyView() const noexcept;
  [[nodiscard]] Image3FView MediumFrequencyView() noexcept;
  [[nodiscard]] ConstImage3FView MediumFrequencyView() const noexcept;
  [[nodiscard]] PlaneF32View HighFrequencyView(size_t channel) noexcept;
  [[nodiscard]] ConstPlaneF32View
  HighFrequencyView(size_t channel) const noexcept;
  [[nodiscard]] PlaneF32View UltraHighFrequencyView(size_t channel) noexcept;
  [[nodiscard]] ConstPlaneF32View
  UltraHighFrequencyView(size_t channel) const noexcept;

  [[nodiscard]] Extent2D extent() const noexcept { return extent_; }
  [[nodiscard]] size_t plane_size() const noexcept { return plane_size_; }

private:
  [[nodiscard]] PlaneF32View Plane(size_t index) noexcept;
  [[nodiscard]] ConstPlaneF32View Plane(size_t index) const noexcept;
  [[nodiscard]] Image3FView Image(size_t first_plane) noexcept;
  [[nodiscard]] ConstImage3FView Image(size_t first_plane) const noexcept;

  Extent2D extent_;
  size_t plane_size_ = 0;
  std::vector<float> values_;
};

/// Reusable storage for native opsin conversion.
class OpsinScratch {
public:
  OpsinScratch() = default;

  OpsinScratch(const OpsinScratch &) = delete;
  OpsinScratch &operator=(const OpsinScratch &) = delete;
  OpsinScratch(OpsinScratch &&) noexcept = default;
  OpsinScratch &operator=(OpsinScratch &&) noexcept = default;

private:
  [[nodiscard]] Status Prepare(Extent2D extent);

  OwnedImage3F blurred_;
  OwnedImage3F result_;
  BlurScratch blur_;

  friend Status OpsinDynamicsImage(ConstImage3FView, float, OpsinScratch *,
                                   Image3FView);
};

/// Reusable storage for native frequency decomposition.
class FrequencyScratch {
public:
  FrequencyScratch() = default;

  FrequencyScratch(const FrequencyScratch &) = delete;
  FrequencyScratch &operator=(const FrequencyScratch &) = delete;
  FrequencyScratch(FrequencyScratch &&) noexcept = default;
  FrequencyScratch &operator=(FrequencyScratch &&) noexcept = default;

private:
  [[nodiscard]] Status Prepare(Extent2D extent);

  OwnedPlaneF32 blurred_;
  BlurScratch blur_;

  friend Status SeparateFrequencies(ConstImage3FView, FrequencyScratch *,
                                    OwnedPsychoImage *);
};

/// Converts linear RGB to Butteraugli's opsin-dynamics XYB representation.
/// Exact input/output aliasing is supported; partial overlap is outside this
/// internal contract. Output pixels and padding are unchanged on failure.
[[nodiscard]] Status OpsinDynamicsImage(ConstImage3FView linear_rgb,
                                        float intensity_target,
                                        OpsinScratch *scratch, Image3FView xyb);

/// Splits an XYB image into Butteraugli LF/MF/HF/UHF planes. The prior output
/// remains unchanged if validation, allocation, or computation fails.
[[nodiscard]] Status SeparateFrequencies(ConstImage3FView xyb,
                                         FrequencyScratch *scratch,
                                         OwnedPsychoImage *output);

/// Scalar forms of Butteraugli's point transforms, exposed only internally so
/// threshold and sign behavior can be checked without a public API.
[[nodiscard]] float MaximumClamp(float value, float maximum) noexcept;
[[nodiscard]] float RemoveRangeAroundZero(float value, float width) noexcept;
[[nodiscard]] float AmplifyRangeAroundZero(float value, float width) noexcept;
[[nodiscard]] float SuppressXByY(float y, float x) noexcept;

} // namespace gjxl::butteraugli_internal
