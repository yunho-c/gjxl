// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// The scalar difference, Malta, masking, and multiscale logic is adapted from
// pinned JPEG XL Butteraugli code distributed under its BSD-style license.
// See third_party/libjxl/LICENSE.

#include "codec/butteraugli_distance_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <ranges>
#include <stdexcept>
#include <utility>
#include "core/managed_allocator.h"

namespace gjxl::butteraugli_internal {
using resource_budget_internal::ManagedVector;

namespace {

constexpr std::array<double, 6> kMaltaWeights = {
    37.0819870399, 8246.75321353, 18.7237414387,
    6923.99476109, 1.10039032555, 173.5,
};
constexpr std::array<double, 6> kMaltaNorms = {
    130262059.556, 1009002.70582, 4498534.45232,
    8051.15833247, 71.7800275169, 5.0,
};
constexpr std::array<float, 9> kL2Weights = {
    400.0f,         1.50815703118f,  0.0f,
    2150.0f,        10.6195433239f,  16.2176043152f,
    29.2353797994f, 0.844626970982f, 0.703646627719f,
};

[[nodiscard]] bool ValidExtent(Extent2D extent, size_t *area = nullptr) {
  size_t computed_area = 0;
  if (extent.empty() || !extent.try_area(&computed_area)) {
    return false;
  }
  if (area != nullptr) {
    *area = computed_area;
  }
  return true;
}

template <typename T> [[nodiscard]] bool ValidPlane(PlaneView<T> plane) {
  if (!plane.valid() ||
      plane.extent.width >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      plane.extent.height >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  return plane.extent.height == 1 ||
         plane.stride <=
             (std::numeric_limits<size_t>::max() - plane.extent.width) /
                 (plane.extent.height - 1);
}

template <typename T> [[nodiscard]] bool ValidImage(Image3View<T> image) {
  if (!ValidPlane(image.plane[0])) {
    return false;
  }
  for (size_t channel = 1; channel < 3; ++channel) {
    if (!ValidPlane(image.plane[channel]) ||
        image.plane[channel].extent != image.plane[0].extent) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool PlaneIsFinite(ConstPlaneF32View plane) {
  for (size_t y = 0; y < plane.extent.height; ++y) {
    for (size_t x = 0; x < plane.extent.width; ++x) {
      if (!std::isfinite(plane.Row(y)[x])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool ImageIsFinite(ConstImage3FView image) {
  return PlaneIsFinite(image.plane[0]) && PlaneIsFinite(image.plane[1]) &&
         PlaneIsFinite(image.plane[2]);
}

[[nodiscard]] bool PsychoImageIsFinite(const OwnedPsychoImage &image) {
  const std::array<ConstPlaneF32View, 10> planes = {
      image.LowFrequencyView().plane[0],
      image.LowFrequencyView().plane[1],
      image.LowFrequencyView().plane[2],
      image.MediumFrequencyView().plane[0],
      image.MediumFrequencyView().plane[1],
      image.MediumFrequencyView().plane[2],
      image.HighFrequencyView(0),
      image.HighFrequencyView(1),
      image.UltraHighFrequencyView(0),
      image.UltraHighFrequencyView(1),
  };
  return std::ranges::all_of(planes, PlaneIsFinite);
}

[[nodiscard]] bool ValidParams(NativeButteraugliParams params) {
  return std::isfinite(params.hf_asymmetry) && params.hf_asymmetry > 0.0f &&
         std::isfinite(params.x_multiplier) && params.x_multiplier > 0.0f &&
         std::isfinite(params.intensity_target) &&
         params.intensity_target > 0.0f;
}

void CopyPlane(ConstPlaneF32View source, PlaneF32View destination) {
  for (size_t y = 0; y < source.extent.height; ++y) {
    std::copy_n(source.Row(y), source.extent.width, destination.Row(y));
  }
}

void FillPlane(PlaneF32View plane, float value) {
  for (size_t y = 0; y < plane.extent.height; ++y) {
    std::fill_n(plane.Row(y), plane.extent.width, value);
  }
}

void FillImage(Image3FView image, float value) {
  for (PlaneF32View plane : image.plane) {
    FillPlane(plane, value);
  }
}

[[nodiscard]] ConstPlaneF32View AsConst(PlaneF32View plane) {
  return {plane.data, plane.extent, plane.stride};
}

// Highway's scalar MulAdd rounds the product before the addition.
[[nodiscard]] float UnfusedMultiplyAdd(float multiplier, float value,
                                       float addend) {
  volatile float product = multiplier * value;
  return product + addend;
}

[[nodiscard]] float Sum4(float a, float b, float c, float d) {
  return (a + b) + (c + d);
}

[[nodiscard]] float Sum5(float a, float b, float c, float d, float e) {
  return Sum4(a, b, c, d + e);
}

[[nodiscard]] float Sum7(float a, float b, float c, float d, float e, float f,
                         float g) {
  return Sum4(a, b, c, Sum4(d, e, f, g));
}

[[nodiscard]] float Sum9(float a, float b, float c, float d, float e, float f,
                         float g, float h, float i) {
  return (Sum4(a, b, c, d) + Sum4(e, f, g, h)) + i;
}

[[nodiscard]] float ReadWithZeroBorder(ConstPlaneF32View plane, size_t x,
                                       size_t y, int dx, int dy) {
  const int64_t source_x = static_cast<int64_t>(x) + dx;
  const int64_t source_y = static_cast<int64_t>(y) + dy;
  if (source_x < 0 || source_y < 0 ||
      source_x >= static_cast<int64_t>(plane.extent.width) ||
      source_y >= static_cast<int64_t>(plane.extent.height)) {
    return 0.0f;
  }
  return plane.Row(
      static_cast<size_t>(source_y))[static_cast<size_t>(source_x)];
}

[[nodiscard]] float MaltaUnitLf(ConstPlaneF32View diffs, size_t x, size_t y) {
  const auto p = [&](int dx, int dy) {
    return ReadWithZeroBorder(diffs, x, y, dx, dy);
  };
  float sum = Sum5(p(-4, 0), p(-2, 0), p(0, 0), p(2, 0), p(4, 0));
  float result = sum * sum;
  const auto add_squared = [&](float value, float *total) {
    *total = UnfusedMultiplyAdd(value, value, *total);
  };
  sum = Sum5(p(0, -4), p(0, -2), p(0, 0), p(0, 2), p(0, 4));
  add_squared(sum, &result);
  sum = Sum5(p(-3, -3), p(-2, -2), p(0, 0), p(2, 2), p(3, 3));
  add_squared(sum, &result);
  sum = Sum5(p(3, -3), p(2, -2), p(0, 0), p(-2, 2), p(-3, 3));
  add_squared(sum, &result);
  sum = Sum5(p(1, -4), p(1, -2), p(0, 0), p(-1, 2), p(-1, 4));
  add_squared(sum, &result);
  sum = Sum5(p(-1, -4), p(-1, -2), p(0, 0), p(1, 2), p(1, 4));
  add_squared(sum, &result);
  sum = Sum5(p(-4, -1), p(-2, -1), p(0, 0), p(2, 1), p(4, 1));
  add_squared(sum, &result);
  sum = Sum5(p(-4, 1), p(-2, 1), p(0, 0), p(2, -1), p(4, -1));
  add_squared(sum, &result);
  sum = Sum5(p(-2, -3), p(-1, -2), p(0, 0), p(1, 2), p(2, 3));
  add_squared(sum, &result);
  sum = Sum5(p(2, -3), p(1, -2), p(0, 0), p(-1, 2), p(-2, 3));
  add_squared(sum, &result);
  sum = Sum5(p(-3, -2), p(-2, -1), p(0, 0), p(2, 1), p(3, 2));
  add_squared(sum, &result);
  sum = Sum5(p(3, -2), p(2, -1), p(0, 0), p(-2, 1), p(-3, 2));
  add_squared(sum, &result);
  sum = Sum5(p(-4, 2), p(-2, 1), p(0, 0), p(2, -1), p(4, -2));
  add_squared(sum, &result);
  sum = Sum5(p(-4, -2), p(-2, -1), p(0, 0), p(2, 1), p(4, 2));
  add_squared(sum, &result);
  sum = Sum5(p(-2, -4), p(-1, -2), p(0, 0), p(1, 2), p(2, 4));
  add_squared(sum, &result);
  sum = Sum5(p(2, -4), p(1, -2), p(0, 0), p(-1, 2), p(-2, 4));
  add_squared(sum, &result);
  return result;
}

[[nodiscard]] float MaltaUnitFull(ConstPlaneF32View diffs, size_t x, size_t y) {
  const auto p = [&](int dx, int dy) {
    return ReadWithZeroBorder(diffs, x, y, dx, dy);
  };
  float sum = Sum9(p(-4, 0), p(-3, 0), p(-2, 0), p(-1, 0), p(0, 0), p(1, 0),
                   p(2, 0), p(3, 0), p(4, 0));
  float result = sum * sum;
  const auto add_squared = [&](float value, float *total) {
    *total = UnfusedMultiplyAdd(value, value, *total);
  };
  sum = Sum9(p(0, -4), p(0, -3), p(0, -2), p(0, -1), p(0, 0), p(0, 1), p(0, 2),
             p(0, 3), p(0, 4));
  add_squared(sum, &result);
  sum =
      Sum7(p(-3, -3), p(-2, -2), p(-1, -1), p(0, 0), p(1, 1), p(2, 2), p(3, 3));
  add_squared(sum, &result);
  sum =
      Sum7(p(3, -3), p(2, -2), p(1, -1), p(0, 0), p(-1, 1), p(-2, 2), p(-3, 3));
  add_squared(sum, &result);
  sum = Sum9(p(1, -4), p(1, -3), p(1, -2), p(0, -1), p(0, 0), p(0, 1), p(-1, 2),
             p(-1, 3), p(-1, 4));
  add_squared(sum, &result);
  sum = Sum9(p(-1, -4), p(-1, -3), p(-1, -2), p(0, -1), p(0, 0), p(0, 1),
             p(1, 2), p(1, 3), p(1, 4));
  add_squared(sum, &result);
  sum = Sum9(p(-4, -1), p(-3, -1), p(-2, -1), p(-1, 0), p(0, 0), p(1, 0),
             p(2, 1), p(3, 1), p(4, 1));
  add_squared(sum, &result);
  sum = Sum9(p(-4, 1), p(-3, 1), p(-2, 1), p(-1, 0), p(0, 0), p(1, 0), p(2, -1),
             p(3, -1), p(4, -1));
  add_squared(sum, &result);
  sum =
      Sum7(p(-2, -3), p(-1, -2), p(-1, -1), p(0, 0), p(1, 1), p(1, 2), p(2, 3));
  add_squared(sum, &result);
  sum =
      Sum7(p(2, -3), p(1, -2), p(1, -1), p(0, 0), p(-1, 1), p(-1, 2), p(-2, 3));
  add_squared(sum, &result);
  sum =
      Sum7(p(-3, -2), p(-2, -1), p(-1, -1), p(0, 0), p(1, 1), p(2, 1), p(3, 2));
  add_squared(sum, &result);
  sum =
      Sum7(p(3, -2), p(2, -1), p(1, -1), p(0, 0), p(-1, 1), p(-2, 1), p(-3, 2));
  add_squared(sum, &result);
  sum = Sum9(p(-4, 1), p(-3, 1), p(-2, 1), p(-1, 0), p(0, 0), p(1, 0), p(2, -1),
             p(3, -1), p(4, -1));
  add_squared(sum, &result);
  sum = Sum9(p(-4, -1), p(-3, -1), p(-2, -1), p(-1, 0), p(0, 0), p(1, 0),
             p(2, 1), p(3, 1), p(4, 1));
  add_squared(sum, &result);
  sum = Sum9(p(-1, -4), p(-1, -3), p(-1, -2), p(0, -1), p(0, 0), p(0, 1),
             p(1, 2), p(1, 3), p(1, 4));
  add_squared(sum, &result);
  sum = Sum9(p(1, -4), p(1, -3), p(1, -2), p(0, -1), p(0, 0), p(0, 1), p(-1, 2),
             p(-1, 3), p(-1, 4));
  add_squared(sum, &result);
  return result;
}

void CombineChannelsForMasking(const OwnedPsychoImage &psycho,
                               PlaneF32View output) {
  const ConstPlaneF32View hf_x = psycho.HighFrequencyView(0);
  const ConstPlaneF32View hf_y = psycho.HighFrequencyView(1);
  const ConstPlaneF32View uhf_x = psycho.UltraHighFrequencyView(0);
  const ConstPlaneF32View uhf_y = psycho.UltraHighFrequencyView(1);
  for (size_t y = 0; y < output.extent.height; ++y) {
    for (size_t x = 0; x < output.extent.width; ++x) {
      const float xdiff = (uhf_x.Row(y)[x] + hf_x.Row(y)[x]) * 2.5f;
      const float ydiff = uhf_y.Row(y)[x] * 0.4f + hf_y.Row(y)[x] * 0.4f;
      output.Row(y)[x] = std::sqrt(xdiff * xdiff + ydiff * ydiff);
    }
  }
}

void DiffPrecompute(ConstPlaneF32View input, PlaneF32View output) {
  constexpr float kMultiplier = 6.19424080439f;
  constexpr float kBiasArgument = 12.61050594197f;
  const float bias = kMultiplier * kBiasArgument;
  const float sqrt_bias = std::sqrt(bias);
  for (size_t y = 0; y < input.extent.height; ++y) {
    for (size_t x = 0; x < input.extent.width; ++x) {
      output.Row(y)[x] =
          std::sqrt(kMultiplier * std::abs(input.Row(y)[x]) + bias) - sqrt_bias;
    }
  }
}

void StoreMin3(float value, float &minimum0, float &minimum1, float &minimum2) {
  if (value < minimum2) {
    if (value < minimum0) {
      minimum2 = minimum1;
      minimum1 = minimum0;
      minimum0 = value;
    } else if (value < minimum1) {
      minimum2 = minimum1;
      minimum1 = value;
    } else {
      minimum2 = value;
    }
  }
}

void ExpandImage(ConstImage3FView input, size_t xborder, size_t yborder,
                 Image3FView output) {
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < output.height(); ++y) {
      const size_t source_y =
          std::min(input.height() - 1, y > yborder ? y - yborder : size_t{0});
      for (size_t x = 0; x < output.width(); ++x) {
        const size_t source_x =
            std::min(input.width() - 1, x > xborder ? x - xborder : size_t{0});
        output.plane[channel].Row(y)[x] =
            input.plane[channel].Row(source_y)[source_x];
      }
    }
  }
}

void Subsample2x(ConstImage3FView input, Image3FView output) {
  FillImage(output, 0.0f);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < input.height(); ++y) {
      for (size_t x = 0; x < input.width(); ++x) {
        output.plane[channel].Row(y / 2)[x / 2] +=
            0.25f * input.plane[channel].Row(y)[x];
      }
    }
    if ((input.width() & 1U) != 0) {
      const size_t last_column = output.width() - 1;
      for (size_t y = 0; y < output.height(); ++y) {
        output.plane[channel].Row(y)[last_column] *= 2.0f;
      }
    }
    if ((input.height() & 1U) != 0) {
      const size_t last_row = output.height() - 1;
      for (size_t x = 0; x < output.width(); ++x) {
        output.plane[channel].Row(last_row)[x] *= 2.0f;
      }
    }
  }
}

} // namespace

Status OwnedDifferenceStages::Resize(Extent2D extent) {
  size_t area = 0;
  if (!ValidExtent(extent, &area) ||
      area > std::numeric_limits<size_t>::max() / kDifferenceStageCount) {
    return Status::InvalidArgument(
        "Butteraugli difference-stage extent is empty or overflows");
  }
  const size_t value_count = area * kDifferenceStageCount;
  if (value_count > values_.max_size()) {
    return Status::InvalidArgument(
        "Butteraugli difference-stage extent exceeds the container limit");
  }
  if (extent_ == extent && values_.size() == value_count) {
    return Status::Ok();
  }
  try {
    ManagedVector<float> replacement(value_count);
    values_.swap(replacement);
    extent_ = extent;
    plane_size_ = area;
    return Status::Ok();
  } catch (const std::length_error &) {
    return Status::InvalidArgument(
        "Butteraugli difference-stage extent exceeds the container limit");
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
        "Unable to allocate Butteraugli difference-stage storage");
  }
}

PlaneF32View OwnedDifferenceStages::StageView(DifferenceStage stage) noexcept {
  const size_t index = static_cast<size_t>(stage);
  if (index >= kDifferenceStageCount || values_.empty()) {
    return {};
  }
  return {values_.data() + index * plane_size_, extent_, extent_.width};
}

ConstPlaneF32View
OwnedDifferenceStages::StageView(DifferenceStage stage) const noexcept {
  const size_t index = static_cast<size_t>(stage);
  if (index >= kDifferenceStageCount || values_.empty()) {
    return {};
  }
  return {values_.data() + index * plane_size_, extent_, extent_.width};
}

Status DifferenceScratch::Prepare(Extent2D extent) {
  std::array<OwnedPlaneF32 *, 7> planes = {
      &malta_diffs_,       &mask_activity0_,    &mask_activity1_,
      &mask_precomputed0_, &mask_precomputed1_, &mask_blurred0_,
      &mask_blurred1_,
  };
  for (OwnedPlaneF32 *plane : planes) {
    Status status = plane->Resize(extent);
    if (!status.ok()) {
      return status;
    }
  }
  Status status = block_diff_ac_.Resize(extent);
  if (!status.ok()) {
    return status;
  }
  return block_diff_dc_.Resize(extent);
}

Status L2Diff(ConstPlaneF32View image0, ConstPlaneF32View image1, float weight,
              PlaneF32View output) {
  if (!ValidPlane(image0) || !ValidPlane(image1) || !ValidPlane(output) ||
      image0.extent != image1.extent || image0.extent != output.extent ||
      !std::isfinite(weight) || weight < 0.0f || !PlaneIsFinite(image0) ||
      !PlaneIsFinite(image1) || !PlaneIsFinite(AsConst(output))) {
    return Status::InvalidArgument("Butteraugli L2 inputs are invalid");
  }
  if (weight == 0.0f) {
    return Status::Ok();
  }
  for (size_t y = 0; y < image0.extent.height; ++y) {
    for (size_t x = 0; x < image0.extent.width; ++x) {
      const float difference = image0.Row(y)[x] - image1.Row(y)[x];
      const float squared = difference * difference;
      output.Row(y)[x] = UnfusedMultiplyAdd(squared, weight, output.Row(y)[x]);
    }
  }
  return Status::Ok();
}

Status SetL2Diff(ConstPlaneF32View image0, ConstPlaneF32View image1,
                 float weight, PlaneF32View output) {
  if (!ValidPlane(image0) || !ValidPlane(image1) || !ValidPlane(output) ||
      image0.extent != image1.extent || image0.extent != output.extent ||
      !std::isfinite(weight) || weight < 0.0f || !PlaneIsFinite(image0) ||
      !PlaneIsFinite(image1)) {
    return Status::InvalidArgument("Butteraugli set-L2 inputs are invalid");
  }
  if (weight == 0.0f) {
    return Status::Ok();
  }
  for (size_t y = 0; y < image0.extent.height; ++y) {
    for (size_t x = 0; x < image0.extent.width; ++x) {
      const float difference = image0.Row(y)[x] - image1.Row(y)[x];
      output.Row(y)[x] = difference * difference * weight;
    }
  }
  return Status::Ok();
}

Status L2DiffAsymmetric(ConstPlaneF32View image0, ConstPlaneF32View image1,
                        float weight_0_gt_1, float weight_0_lt_1,
                        PlaneF32View output) {
  if (!ValidPlane(image0) || !ValidPlane(image1) || !ValidPlane(output) ||
      image0.extent != image1.extent || image0.extent != output.extent ||
      !std::isfinite(weight_0_gt_1) || weight_0_gt_1 < 0.0f ||
      !std::isfinite(weight_0_lt_1) || weight_0_lt_1 < 0.0f ||
      !PlaneIsFinite(image0) || !PlaneIsFinite(image1) ||
      !PlaneIsFinite(AsConst(output))) {
    return Status::InvalidArgument(
        "Butteraugli asymmetric L2 inputs are invalid");
  }
  if (weight_0_gt_1 == 0.0f && weight_0_lt_1 == 0.0f) {
    return Status::Ok();
  }
  const float primary_weight = weight_0_gt_1 * 0.8f;
  const float secondary_weight = weight_0_lt_1 * 0.8f;
  for (size_t y = 0; y < image0.extent.height; ++y) {
    for (size_t x = 0; x < image0.extent.width; ++x) {
      const float value0 = image0.Row(y)[x];
      const float value1 = image1.Row(y)[x];
      const float difference = value0 - value1;
      float total = UnfusedMultiplyAdd(difference * difference, primary_weight,
                                       output.Row(y)[x]);
      const float magnitude = std::abs(value0);
      const float too_small = 0.4f * magnitude;
      float secondary = 0.0f;
      if (value0 < 0.0f) {
        if (value1 > -too_small) {
          secondary = value1 + too_small;
        } else if (value1 < -magnitude) {
          secondary = -value1 - magnitude;
        }
      } else if (value1 < too_small) {
        secondary = too_small - value1;
      } else if (value1 > magnitude) {
        secondary = value1 - magnitude;
      }
      output.Row(y)[x] =
          UnfusedMultiplyAdd(secondary_weight, secondary * secondary, total);
    }
  }
  return Status::Ok();
}

Status MaltaDiffMap(ConstPlaneF32View image0, ConstPlaneF32View image1,
                    bool low_frequency, double weight_0_gt_1,
                    double weight_0_lt_1, double norm, OwnedPlaneF32 *diffs,
                    PlaneF32View output) {
  if (!ValidPlane(image0) || !ValidPlane(image1) || !ValidPlane(output) ||
      image0.extent != image1.extent || image0.extent != output.extent ||
      image0.extent.width < 8 || image0.extent.height < 8 || diffs == nullptr ||
      !std::isfinite(weight_0_gt_1) || weight_0_gt_1 < 0.0 ||
      !std::isfinite(weight_0_lt_1) || weight_0_lt_1 < 0.0 ||
      !std::isfinite(norm) || norm <= 0.0 || !PlaneIsFinite(image0) ||
      !PlaneIsFinite(image1) || !PlaneIsFinite(AsConst(output))) {
    return Status::InvalidArgument("Butteraugli Malta inputs are invalid");
  }
  Status status = diffs->Resize(image0.extent);
  if (!status.ok()) {
    return status;
  }

  constexpr float kWeight0 = 0.5f;
  constexpr float kWeight1 = 0.33f;
  constexpr double kLength = 3.75;
  const double multiplier = low_frequency ? 0.611612573796 : 0.39905817637;
  const double pre_0_gt_1 =
      multiplier * std::sqrt(kWeight0 * weight_0_gt_1) / (kLength * 2.0 + 1.0);
  const double pre_0_lt_1 =
      multiplier * std::sqrt(kWeight1 * weight_0_lt_1) / (kLength * 2.0 + 1.0);
  const float norm2_0_gt_1 = static_cast<float>(pre_0_gt_1 * norm);
  const float norm2_0_lt_1 = static_cast<float>(pre_0_lt_1 * norm);
  PlaneF32View scaled = diffs->View();
  for (size_t y = 0; y < image0.extent.height; ++y) {
    for (size_t x = 0; x < image0.extent.width; ++x) {
      const float value0 = image0.Row(y)[x];
      const float value1 = image1.Row(y)[x];
      const float absolute = 0.5f * (std::abs(value0) + std::abs(value1));
      const float difference = value0 - value1;
      const float scaler = norm2_0_gt_1 / (static_cast<float>(norm) + absolute);
      scaled.Row(y)[x] = scaler * difference;
      const float scaler2 =
          norm2_0_lt_1 / (static_cast<float>(norm) + absolute);
      const double magnitude = std::fabs(value0);
      const double too_small = 0.55 * magnitude;
      const double too_big = 1.05 * magnitude;
      if (value0 < 0.0f) {
        if (value1 > -too_small) {
          const double impact = scaler2 * (value1 + too_small);
          scaled.Row(y)[x] -= impact;
        } else if (value1 < -too_big) {
          const double impact = scaler2 * (-value1 - too_big);
          scaled.Row(y)[x] += impact;
        }
      } else if (value1 < too_small) {
        const double impact = scaler2 * (too_small - value1);
        scaled.Row(y)[x] += impact;
      } else if (value1 > too_big) {
        const double impact = scaler2 * (value1 - too_big);
        scaled.Row(y)[x] -= impact;
      }
    }
  }

  const ConstPlaneF32View scaled_const = diffs->ConstView();
  for (size_t y = 0; y < image0.extent.height; ++y) {
    for (size_t x = 0; x < image0.extent.width; ++x) {
      output.Row(y)[x] += low_frequency ? MaltaUnitLf(scaled_const, x, y)
                                        : MaltaUnitFull(scaled_const, x, y);
    }
  }
  return Status::Ok();
}

Status FuzzyErosion(ConstPlaneF32View input, PlaneF32View output) {
  if (!ValidPlane(input) || !ValidPlane(output) ||
      input.extent != output.extent || !PlaneIsFinite(input) ||
      input.data == output.data) {
    return Status::InvalidArgument(
        "Butteraugli fuzzy-erosion inputs are invalid");
  }
  constexpr size_t kStep = 3;
  for (size_t y = 0; y < input.extent.height; ++y) {
    for (size_t x = 0; x < input.extent.width; ++x) {
      float minimum0 = input.Row(y)[x];
      float minimum1 = 2.0f * minimum0;
      float minimum2 = minimum1;
      if (x >= kStep) {
        StoreMin3(input.Row(y)[x - kStep], minimum0, minimum1, minimum2);
        if (y >= kStep) {
          StoreMin3(input.Row(y - kStep)[x - kStep], minimum0, minimum1,
                    minimum2);
        }
        if (y < input.extent.height - kStep) {
          StoreMin3(input.Row(y + kStep)[x - kStep], minimum0, minimum1,
                    minimum2);
        }
      }
      if (x < input.extent.width - kStep) {
        StoreMin3(input.Row(y)[x + kStep], minimum0, minimum1, minimum2);
        if (y >= kStep) {
          StoreMin3(input.Row(y - kStep)[x + kStep], minimum0, minimum1,
                    minimum2);
        }
        if (y < input.extent.height - kStep) {
          StoreMin3(input.Row(y + kStep)[x + kStep], minimum0, minimum1,
                    minimum2);
        }
      }
      if (y >= kStep) {
        StoreMin3(input.Row(y - kStep)[x], minimum0, minimum1, minimum2);
      }
      if (y < input.extent.height - kStep) {
        StoreMin3(input.Row(y + kStep)[x], minimum0, minimum1, minimum2);
      }
      output.Row(y)[x] = 0.45f * minimum0 + 0.3f * minimum1 + 0.25f * minimum2;
    }
  }
  return Status::Ok();
}

float MaskY(float delta) noexcept {
  constexpr double kIntensityTargetNormalization = 0.79079917404;
  constexpr double kGlobalScale = 1.0 / (17.83 * kIntensityTargetNormalization);
  constexpr double kOffset = 0.829591754942;
  constexpr double kScaler = 0.451936922203;
  constexpr double kMultiplier = 2.5485944793;
  const double value =
      kGlobalScale * (1.0 + kMultiplier / (kScaler * delta + kOffset));
  return static_cast<float>(value * value);
}

float MaskDcY(float delta) noexcept {
  constexpr double kIntensityTargetNormalization = 0.79079917404;
  constexpr double kGlobalScale = 1.0 / (17.83 * kIntensityTargetNormalization);
  constexpr double kOffset = 0.20025578522;
  constexpr double kScaler = 3.87449418804;
  constexpr double kMultiplier = 0.505054525019;
  const double value =
      kGlobalScale * (1.0 + kMultiplier / (kScaler * delta + kOffset));
  return static_cast<float>(value * value);
}

Status ComputeDifferenceStages(const OwnedPsychoImage &reference,
                               const OwnedPsychoImage &distorted,
                               NativeButteraugliParams params,
                               DifferenceScratch *scratch,
                               OwnedDifferenceStages *output) {
  const Extent2D extent = reference.extent();
  if (scratch == nullptr || output == nullptr || extent != distorted.extent() ||
      extent.width < 8 || extent.height < 8 || !ValidParams(params) ||
      !PsychoImageIsFinite(reference) || !PsychoImageIsFinite(distorted)) {
    return Status::InvalidArgument(
        "Butteraugli difference images, parameters, or scratch are invalid");
  }

  OwnedDifferenceStages candidate = std::move(scratch->staged_output_);
  struct CandidateGuard {
    OwnedDifferenceStages* staging = nullptr;
    OwnedDifferenceStages* candidate = nullptr;
    bool committed = false;

    ~CandidateGuard() {
      if (!committed) {
        *staging = std::move(*candidate);
      }
    }
  } guard{&scratch->staged_output_, &candidate};
  Status status = candidate.Resize(extent);
  if (!status.ok()) {
    return status;
  }
  status = scratch->Prepare(extent);
  if (!status.ok()) {
    return status;
  }

  const float asymmetry = params.hf_asymmetry;
  const float sqrt_asymmetry = std::sqrt(asymmetry);
  const std::array<ConstPlaneF32View, 6> planes0 = {
      reference.MediumFrequencyView().plane[1],
      reference.MediumFrequencyView().plane[0],
      reference.HighFrequencyView(1),
      reference.HighFrequencyView(0),
      reference.UltraHighFrequencyView(1),
      reference.UltraHighFrequencyView(0),
  };
  const std::array<ConstPlaneF32View, 6> planes1 = {
      distorted.MediumFrequencyView().plane[1],
      distorted.MediumFrequencyView().plane[0],
      distorted.HighFrequencyView(1),
      distorted.HighFrequencyView(0),
      distorted.UltraHighFrequencyView(1),
      distorted.UltraHighFrequencyView(0),
  };
  const std::array<double, 6> weight_up = {
      kMaltaWeights[0],
      kMaltaWeights[1],
      kMaltaWeights[2] * sqrt_asymmetry,
      kMaltaWeights[3] * sqrt_asymmetry,
      kMaltaWeights[4] * asymmetry,
      kMaltaWeights[5] * asymmetry,
  };
  const std::array<double, 6> weight_down = {
      kMaltaWeights[0],
      kMaltaWeights[1],
      kMaltaWeights[2] / sqrt_asymmetry,
      kMaltaWeights[3] / sqrt_asymmetry,
      kMaltaWeights[4] / asymmetry,
      kMaltaWeights[5] / asymmetry,
  };
  for (size_t index = 0; index < planes0.size(); ++index) {
    PlaneF32View stage =
        candidate.StageView(static_cast<DifferenceStage>(index));
    FillPlane(stage, 0.0f);
    status = MaltaDiffMap(planes0[index], planes1[index], index < 4,
                          weight_up[index], weight_down[index],
                          kMaltaNorms[index], &scratch->malta_diffs_, stage);
    if (!status.ok()) {
      return status;
    }
  }

  FillImage(scratch->block_diff_ac_.View(), 0.0f);
  FillImage(scratch->block_diff_dc_.View(), 0.0f);
  constexpr std::array<size_t, 6> kMaltaAccumulationOrder = {4, 5, 2, 3, 0, 1};
  for (size_t index : kMaltaAccumulationOrder) {
    const size_t channel = index % 2 == 0 ? 1 : 0;
    const ConstPlaneF32View stage =
        std::as_const(candidate).StageView(static_cast<DifferenceStage>(index));
    PlaneF32View accumulated = scratch->block_diff_ac_.View().plane[channel];
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        accumulated.Row(y)[x] += stage.Row(y)[x];
      }
    }
  }

  for (size_t channel = 0; channel < 3; ++channel) {
    if (channel < 2) {
      status = L2DiffAsymmetric(reference.HighFrequencyView(channel),
                                distorted.HighFrequencyView(channel),
                                kL2Weights[channel] * asymmetry,
                                kL2Weights[channel] / asymmetry,
                                scratch->block_diff_ac_.View().plane[channel]);
      if (!status.ok()) {
        return status;
      }
    }
    status = L2Diff(reference.MediumFrequencyView().plane[channel],
                    distorted.MediumFrequencyView().plane[channel],
                    kL2Weights[3 + channel],
                    scratch->block_diff_ac_.View().plane[channel]);
    if (!status.ok()) {
      return status;
    }
    status = SetL2Diff(reference.LowFrequencyView().plane[channel],
                       distorted.LowFrequencyView().plane[channel],
                       kL2Weights[6 + channel],
                       scratch->block_diff_dc_.View().plane[channel]);
    if (!status.ok()) {
      return status;
    }
  }

  CombineChannelsForMasking(reference, scratch->mask_activity0_.View());
  CombineChannelsForMasking(distorted, scratch->mask_activity1_.View());
  DiffPrecompute(scratch->mask_activity0_.ConstView(),
                 scratch->mask_precomputed0_.View());
  DiffPrecompute(scratch->mask_activity1_.ConstView(),
                 scratch->mask_precomputed1_.View());
  status = GaussianBlur(scratch->mask_precomputed0_.ConstView(), kMaskBlurSigma,
                        &scratch->blur_, scratch->mask_blurred0_.View());
  if (!status.ok()) {
    return status;
  }
  status = FuzzyErosion(scratch->mask_blurred0_.ConstView(),
                        candidate.StageView(DifferenceStage::kMask));
  if (!status.ok()) {
    return status;
  }
  status = GaussianBlur(scratch->mask_precomputed1_.ConstView(), kMaskBlurSigma,
                        &scratch->blur_, scratch->mask_blurred1_.View());
  if (!status.ok()) {
    return status;
  }
  PlaneF32View ac_y = scratch->block_diff_ac_.View().plane[1];
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const float difference = scratch->mask_blurred0_.ConstView().Row(y)[x] -
                               scratch->mask_blurred1_.ConstView().Row(y)[x];
      ac_y.Row(y)[x] += 10.0f * difference * difference;
    }
  }
  CopyPlane(AsConst(ac_y), candidate.StageView(DifferenceStage::kMaskedAcY));

  const ConstImage3FView dc = scratch->block_diff_dc_.ConstView();
  const ConstImage3FView ac = scratch->block_diff_ac_.ConstView();
  const ConstPlaneF32View mask =
      std::as_const(candidate).StageView(DifferenceStage::kMask);
  PlaneF32View final = candidate.StageView(DifferenceStage::kFinalComposition);
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const float mask_value = MaskY(mask.Row(y)[x]);
      const float dc_mask_value = MaskDcY(mask.Row(y)[x]);
      const float diff_dc0 = dc.plane[0].Row(y)[x] * params.x_multiplier;
      const float diff_ac0 = ac.plane[0].Row(y)[x] * params.x_multiplier;
      const float masked_dc = diff_dc0 * dc_mask_value +
                              dc.plane[1].Row(y)[x] * dc_mask_value +
                              dc.plane[2].Row(y)[x] * dc_mask_value;
      const float masked_ac = diff_ac0 * mask_value +
                              ac.plane[1].Row(y)[x] * mask_value +
                              ac.plane[2].Row(y)[x] * mask_value;
      final.Row(y)[x] = std::sqrt(masked_dc + masked_ac);
    }
  }

  for (size_t stage = 0; stage < kDifferenceStageCount; ++stage) {
    if (!PlaneIsFinite(std::as_const(candidate).StageView(
            static_cast<DifferenceStage>(stage)))) {
      return Status::Internal(
          "Butteraugli difference computation produced non-finite values");
    }
  }
  scratch->staged_output_ = std::move(*output);
  *output = std::move(candidate);
  guard.committed = true;
  return Status::Ok();
}

Status ComputeButteraugliDistanceNative(ConstImage3FView reference,
                                        ConstImage3FView distorted,
                                        NativeButteraugliParams params,
                                        NativeButteraugliScratch *scratch,
                                        PlaneF32View distance_map,
                                        double *score) {
  if (!ValidImage(reference) || !ValidImage(distorted) ||
      !ValidPlane(distance_map) || reference.extent() != distorted.extent() ||
      reference.extent() != distance_map.extent || !ValidParams(params) ||
      scratch == nullptr || score == nullptr || !ImageIsFinite(reference) ||
      !ImageIsFinite(distorted)) {
    return Status::InvalidArgument("Native Butteraugli inputs, output, "
                                   "parameters, or scratch are invalid");
  }

  const auto compute_single_scale =
      [&](ConstImage3FView scale_reference, ConstImage3FView scale_distorted,
          OwnedDifferenceStages *scale_output) -> Status {
    Status scale_status = scratch->xyb0_.Resize(scale_reference.extent());
    if (!scale_status.ok()) {
      return scale_status;
    }
    scale_status = scratch->xyb1_.Resize(scale_reference.extent());
    if (!scale_status.ok()) {
      return scale_status;
    }
    scale_status = OpsinDynamicsImage(scale_reference, params.intensity_target,
                                      &scratch->opsin_, scratch->xyb0_.View());
    if (!scale_status.ok()) {
      return scale_status;
    }
    scale_status = OpsinDynamicsImage(scale_distorted, params.intensity_target,
                                      &scratch->opsin_, scratch->xyb1_.View());
    if (!scale_status.ok()) {
      return scale_status;
    }
    scale_status = SeparateFrequencies(
        scratch->xyb0_.ConstView(), &scratch->frequency_, &scratch->psycho0_);
    if (!scale_status.ok()) {
      return scale_status;
    }
    scale_status = SeparateFrequencies(
        scratch->xyb1_.ConstView(), &scratch->frequency_, &scratch->psycho1_);
    if (!scale_status.ok()) {
      return scale_status;
    }
    return ComputeDifferenceStages(scratch->psycho0_, scratch->psycho1_, params,
                                   &scratch->difference_, scale_output);
  };

  const Extent2D requested_extent = reference.extent();
  Status status = scratch->final_map_.Resize(requested_extent);
  if (!status.ok()) {
    return status;
  }
  ConstImage3FView working_reference = reference;
  ConstImage3FView working_distorted = distorted;
  size_t xborder = 0;
  size_t yborder = 0;
  bool expanded = false;
  if (requested_extent.width < 8 || requested_extent.height < 8) {
    expanded = true;
    const Extent2D expanded_extent{
        std::max<size_t>(8, requested_extent.width),
        std::max<size_t>(8, requested_extent.height)};
    xborder = requested_extent.width < 8 ? (8 - requested_extent.width) / 2 : 0;
    yborder =
        requested_extent.height < 8 ? (8 - requested_extent.height) / 2 : 0;
    status = scratch->expanded0_.Resize(expanded_extent);
    if (!status.ok()) {
      return status;
    }
    status = scratch->expanded1_.Resize(expanded_extent);
    if (!status.ok()) {
      return status;
    }
    ExpandImage(reference, xborder, yborder, scratch->expanded0_.View());
    ExpandImage(distorted, xborder, yborder, scratch->expanded1_.View());
    working_reference = scratch->expanded0_.ConstView();
    working_distorted = scratch->expanded1_.ConstView();
  }

  status = compute_single_scale(working_reference, working_distorted,
                                &scratch->main_stages_);
  if (!status.ok()) {
    return status;
  }

  PlaneF32View staged_map = scratch->final_map_.View();
  const ConstPlaneF32View main_map =
      std::as_const(scratch->main_stages_)
          .StageView(DifferenceStage::kFinalComposition);
  if (expanded) {
    for (size_t y = 0; y < requested_extent.height; ++y) {
      std::copy_n(main_map.Row(y + yborder) + xborder, requested_extent.width,
                  staged_map.Row(y));
    }
  } else if (requested_extent.width >= 15 && requested_extent.height >= 15) {
    const Extent2D sub_extent{(requested_extent.width + 1) / 2,
                              (requested_extent.height + 1) / 2};
    status = scratch->subsampled0_.Resize(sub_extent);
    if (!status.ok()) {
      return status;
    }
    status = scratch->subsampled1_.Resize(sub_extent);
    if (!status.ok()) {
      return status;
    }
    Subsample2x(reference, scratch->subsampled0_.View());
    Subsample2x(distorted, scratch->subsampled1_.View());
    status = compute_single_scale(scratch->subsampled0_.ConstView(),
                                  scratch->subsampled1_.ConstView(),
                                  &scratch->sub_stages_);
    if (!status.ok()) {
      return status;
    }
    const ConstPlaneF32View sub_map =
        std::as_const(scratch->sub_stages_)
            .StageView(DifferenceStage::kFinalComposition);
    for (size_t y = 0; y < requested_extent.height; ++y) {
      for (size_t x = 0; x < requested_extent.width; ++x) {
        staged_map.Row(y)[x] =
            main_map.Row(y)[x] * 0.85f + 0.5f * sub_map.Row(y / 2)[x / 2];
      }
    }
  } else {
    CopyPlane(main_map, staged_map);
  }

  float maximum = 0.0f;
  for (size_t y = 0; y < requested_extent.height; ++y) {
    for (size_t x = 0; x < requested_extent.width; ++x) {
      const float value = staged_map.Row(y)[x];
      if (!std::isfinite(value) || value < 0.0f) {
        return Status::Internal(
            "Native Butteraugli produced an invalid distance map");
      }
      maximum = std::max(maximum, value);
    }
  }

  CopyPlane(scratch->final_map_.ConstView(), distance_map);
  *score = maximum;
  return Status::Ok();
}

Status PrepareButteraugliReferenceNative(
    ConstImage3FView reference, NativeButteraugliParams params,
    NativePreparedButteraugliReference* prepared) {
  if (!ValidImage(reference) || !ValidParams(params) || prepared == nullptr ||
      !ImageIsFinite(reference)) {
    return Status::InvalidArgument(
        "Native Butteraugli reference preparation is invalid");
  }

  NativePreparedButteraugliReference candidate;
  candidate.params_ = params;
  candidate.requested_extent_ = reference.extent();
  candidate.working_extent_ = reference.extent();
  candidate.expanded_ = reference.width() < 8 || reference.height() < 8;
  if (candidate.expanded_) {
    candidate.working_extent_ = {
        std::max<size_t>(8, reference.width()),
        std::max<size_t>(8, reference.height())};
    candidate.xborder_ = reference.width() < 8
        ? (8 - reference.width()) / 2
        : 0;
    candidate.yborder_ = reference.height() < 8
        ? (8 - reference.height()) / 2
        : 0;
  }
  candidate.has_subscale_ = !candidate.expanded_ &&
      reference.width() >= 15 && reference.height() >= 15;

  const auto prepare_scale = [&params](
      ConstImage3FView scale_reference, OwnedImage3F* xyb,
      OpsinScratch* opsin, FrequencyScratch* frequency,
      OwnedPsychoImage* reference_psycho,
      OwnedPsychoImage* distorted_psycho,
      DifferenceScratch* difference,
      OwnedDifferenceStages* stages) -> Status {
    Status status = xyb->Resize(scale_reference.extent());
    if (!status.ok()) {
      return status;
    }
    status = OpsinDynamicsImage(
        scale_reference, params.intensity_target, opsin, xyb->View());
    if (!status.ok()) {
      return status;
    }
    status = SeparateFrequencies(
        xyb->ConstView(), frequency, reference_psycho);
    if (!status.ok()) {
      return status;
    }
    status = distorted_psycho->Resize(scale_reference.extent());
    if (!status.ok()) {
      return status;
    }
    // Prepares all difference storage during reference preparation. The
    // zero self-comparison is internal and is overwritten by Compare.
    return ComputeDifferenceStages(
        *reference_psycho, *reference_psycho, params, difference, stages);
  };

  ConstImage3FView working_reference = reference;
  if (candidate.expanded_) {
    Status status = candidate.main_input_.Resize(candidate.working_extent_);
    if (!status.ok()) {
      return status;
    }
    ExpandImage(reference, candidate.xborder_, candidate.yborder_,
                candidate.main_input_.View());
    working_reference = candidate.main_input_.ConstView();
  }
  Status status = prepare_scale(
      working_reference, &candidate.main_xyb_, &candidate.main_opsin_,
      &candidate.main_frequency_, &candidate.main_reference_,
      &candidate.main_distorted_, &candidate.main_difference_,
      &candidate.main_stages_);
  if (!status.ok()) {
    return status;
  }
  status = candidate.main_difference_.PrepareOutputStaging(
      candidate.working_extent_);
  if (!status.ok()) {
    return status;
  }

  if (candidate.has_subscale_) {
    const Extent2D sub_extent{
        (reference.width() + 1) / 2, (reference.height() + 1) / 2};
    status = candidate.sub_input_.Resize(sub_extent);
    if (!status.ok()) {
      return status;
    }
    Subsample2x(reference, candidate.sub_input_.View());
    status = prepare_scale(
        candidate.sub_input_.ConstView(), &candidate.sub_xyb_,
        &candidate.sub_opsin_, &candidate.sub_frequency_,
        &candidate.sub_reference_, &candidate.sub_distorted_,
        &candidate.sub_difference_, &candidate.sub_stages_);
    if (!status.ok()) {
      return status;
    }
    status = candidate.sub_difference_.PrepareOutputStaging(sub_extent);
    if (!status.ok()) {
      return status;
    }
  }
  status = candidate.final_map_.Resize(reference.extent());
  if (!status.ok()) {
    return status;
  }
  candidate.ready_ = true;
  *prepared = std::move(candidate);
  return Status::Ok();
}

Status CompareButteraugliReferenceNative(
    NativePreparedButteraugliReference* prepared,
    ConstImage3FView distorted, PlaneF32View distance_map, double* score) {
  if (prepared == nullptr || !prepared->ready_ || !ValidImage(distorted) ||
      distorted.extent() != prepared->requested_extent_ ||
      !ValidPlane(distance_map) ||
      distance_map.extent != prepared->requested_extent_ || score == nullptr ||
      !ImageIsFinite(distorted)) {
    return Status::InvalidArgument(
        "Prepared native Butteraugli comparison is invalid");
  }

  const auto transform_distorted = [prepared](
      ConstImage3FView input, OwnedImage3F* xyb, OpsinScratch* opsin,
      FrequencyScratch* frequency,
      OwnedPsychoImage* psycho) -> Status {
    Status status = OpsinDynamicsImage(
        input, prepared->params_.intensity_target, opsin, xyb->View());
    if (!status.ok()) {
      return status;
    }
    return SeparateFrequencies(xyb->ConstView(), frequency, psycho);
  };

  ConstImage3FView working_distorted = distorted;
  if (prepared->expanded_) {
    ExpandImage(distorted, prepared->xborder_, prepared->yborder_,
                prepared->main_input_.View());
    working_distorted = prepared->main_input_.ConstView();
  }
  Status status = transform_distorted(
      working_distorted, &prepared->main_xyb_, &prepared->main_opsin_,
      &prepared->main_frequency_, &prepared->main_distorted_);
  if (!status.ok()) {
    return status;
  }
  status = ComputeDifferenceStages(
      prepared->main_reference_, prepared->main_distorted_, prepared->params_,
      &prepared->main_difference_, &prepared->main_stages_);
  if (!status.ok()) {
    return status;
  }

  PlaneF32View staged_map = prepared->final_map_.View();
  const ConstPlaneF32View main_map =
      std::as_const(prepared->main_stages_)
          .StageView(DifferenceStage::kFinalComposition);
  if (prepared->expanded_) {
    for (size_t y = 0; y < prepared->requested_extent_.height; ++y) {
      std::copy_n(
          main_map.Row(y + prepared->yborder_) + prepared->xborder_,
          prepared->requested_extent_.width, staged_map.Row(y));
    }
  } else if (prepared->has_subscale_) {
    Subsample2x(distorted, prepared->sub_input_.View());
    status = transform_distorted(
        prepared->sub_input_.ConstView(), &prepared->sub_xyb_,
        &prepared->sub_opsin_, &prepared->sub_frequency_,
        &prepared->sub_distorted_);
    if (!status.ok()) {
      return status;
    }
    status = ComputeDifferenceStages(
        prepared->sub_reference_, prepared->sub_distorted_,
        prepared->params_, &prepared->sub_difference_,
        &prepared->sub_stages_);
    if (!status.ok()) {
      return status;
    }
    const ConstPlaneF32View sub_map =
        std::as_const(prepared->sub_stages_)
            .StageView(DifferenceStage::kFinalComposition);
    for (size_t y = 0; y < prepared->requested_extent_.height; ++y) {
      for (size_t x = 0; x < prepared->requested_extent_.width; ++x) {
        staged_map.Row(y)[x] =
            main_map.Row(y)[x] * 0.85f +
            0.5f * sub_map.Row(y / 2)[x / 2];
      }
    }
  } else {
    CopyPlane(main_map, staged_map);
  }

  float maximum = 0.0f;
  for (size_t y = 0; y < prepared->requested_extent_.height; ++y) {
    for (size_t x = 0; x < prepared->requested_extent_.width; ++x) {
      const float value = staged_map.Row(y)[x];
      if (!std::isfinite(value) || value < 0.0f) {
        return Status::Internal(
            "Prepared native Butteraugli produced an invalid distance map");
      }
      maximum = std::max(maximum, value);
    }
  }
  CopyPlane(prepared->final_map_.ConstView(), distance_map);
  *score = maximum;
  return Status::Ok();
}

} // namespace gjxl::butteraugli_internal
