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

} // namespace gjxl::butteraugli_internal
