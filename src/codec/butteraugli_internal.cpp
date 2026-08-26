// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// The scalar Gaussian, opsin-dynamics, and frequency-separation logic is
// adapted from pinned JPEG XL Butteraugli code distributed under its BSD-style
// license. See third_party/libjxl/LICENSE.

#include "codec/butteraugli_internal.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gjxl::butteraugli_internal {
namespace {

[[nodiscard]] bool ValidExtent(Extent2D extent, size_t *area) {
  return !extent.empty() && extent.try_area(area);
}

template <typename T> [[nodiscard]] bool ValidPlane(PlaneView<T> plane) {
  if (!plane.valid() ||
      plane.extent.width >
          static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
      plane.extent.height >
          static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
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

[[nodiscard]] bool PixelsAreFinite(ConstImage3FView image) {
  for (ConstPlaneF32View plane : image.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.Row(y)[x])) {
          return false;
        }
      }
    }
  }
  return true;
}

void CopyPlane(ConstPlaneF32View source, PlaneF32View destination) {
  for (size_t y = 0; y < source.extent.height; ++y) {
    std::copy_n(source.Row(y), source.extent.width, destination.Row(y));
  }
}

[[nodiscard]] ConstPlaneF32View AsConst(PlaneF32View plane) {
  return {plane.data, plane.extent, plane.stride};
}

void CopyImage(ConstImage3FView source, Image3FView destination) {
  for (size_t channel = 0; channel < 3; ++channel) {
    CopyPlane(source.plane[channel], destination.plane[channel]);
  }
}

// Highway's scalar MulAdd deliberately rounds the product before the add.
// Volatile prevents contraction so scalar goldens remain reproducible even
// when the compiler's default contraction mode targets fused instructions.
[[nodiscard]] float UnfusedMultiplyAdd(float multiplier, float value,
                                       float addend) {
  volatile float product = multiplier * value;
  return product + addend;
}

[[nodiscard]] float ButteraugliFastLog2(float value) {
  constexpr float kP0 = -1.8503833400518310e-06f;
  constexpr float kP1 = 1.4287160470083755f;
  constexpr float kP2 = 0.74245873327820566f;
  constexpr float kQ0 = 0.99032814277590719f;
  constexpr float kQ1 = 1.0096718572241148f;
  constexpr float kQ2 = 0.17409343003366853f;

  const uint32_t value_bits = std::bit_cast<uint32_t>(value);
  const int32_t shifted_exponent =
      static_cast<int32_t>(value_bits - 0x3f2aaaabu) >> 23;
  const uint32_t mantissa_bits =
      value_bits - (static_cast<uint32_t>(shifted_exponent) << 23);
  const float x = std::bit_cast<float>(mantissa_bits) - 1.0f;

  float numerator = UnfusedMultiplyAdd(kP2, x, kP1);
  numerator = UnfusedMultiplyAdd(numerator, x, kP0);
  float denominator = UnfusedMultiplyAdd(kQ2, x, kQ1);
  denominator = UnfusedMultiplyAdd(denominator, x, kQ0);
  return numerator / denominator + static_cast<float>(shifted_exponent);
}

[[nodiscard]] float Gamma(float value) {
  constexpr float kRetMul = 19.245013259874995f * 0.6931471805599453f;
  constexpr float kRetAdd = -23.16046239805755f;
  constexpr float kBias = 9.9710635769299145f;
  value = std::max(value, 0.0f);
  return UnfusedMultiplyAdd(kRetMul, ButteraugliFastLog2(value + kBias),
                            kRetAdd);
}

template <bool Clamp>
void OpsinAbsorbance(float red, float green, float blue, float *out0,
                     float *out1, float *out2) {
  constexpr float kMix0 = 0.29956550340058319f;
  constexpr float kMix1 = 0.63373087833825936f;
  constexpr float kMix2 = 0.077705617820981968f;
  constexpr float kMix3 = 1.7557483643287353f;
  constexpr float kMix4 = 0.22158691104574774f;
  constexpr float kMix5 = 0.69391388044116142f;
  constexpr float kMix6 = 0.0987313588422f;
  constexpr float kMix7 = 1.7557483643287353f;
  constexpr float kMix8 = 0.02f;
  constexpr float kMix9 = 0.02f;
  constexpr float kMix10 = 0.20480129041026129f;
  constexpr float kMix11 = 12.226454707163354f;

  *out0 = UnfusedMultiplyAdd(
      kMix0, red,
      UnfusedMultiplyAdd(kMix1, green, UnfusedMultiplyAdd(kMix2, blue, kMix3)));
  *out1 = UnfusedMultiplyAdd(
      kMix4, red,
      UnfusedMultiplyAdd(kMix5, green, UnfusedMultiplyAdd(kMix6, blue, kMix7)));
  *out2 = UnfusedMultiplyAdd(
      kMix8, red,
      UnfusedMultiplyAdd(kMix9, green,
                         UnfusedMultiplyAdd(kMix10, blue, kMix11)));
  if constexpr (Clamp) {
    *out0 = std::max(*out0, kMix3);
    *out1 = std::max(*out1, kMix7);
    *out2 = std::max(*out2, kMix11);
  }
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
  for (ConstPlaneF32View plane : planes) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.Row(y)[x])) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] Status ResizeStorage(Extent2D extent, size_t channel_count,
                                   Extent2D *stored_extent,
                                   size_t *stored_plane_size,
                                   std::vector<float> *values) {

  size_t area = 0;
  if (!ValidExtent(extent, &area) || channel_count == 0 ||
      area > std::numeric_limits<size_t>::max() / channel_count) {
    return Status::InvalidArgument(
        "Butteraugli storage extent is empty or overflows");
  }
  const size_t value_count = area * channel_count;
  if (value_count > values->max_size()) {
    return Status::InvalidArgument(
        "Butteraugli storage extent exceeds the container limit");
  }
  if (*stored_extent == extent && values->size() == value_count) {
    return Status::Ok();
  }

  try {
    std::vector<float> replacement(value_count);
    values->swap(replacement);
    *stored_extent = extent;
    if (stored_plane_size != nullptr) {
      *stored_plane_size = area;
    }
    return Status::Ok();
  } catch (const std::length_error &) {
    return Status::InvalidArgument(
        "Butteraugli storage extent exceeds the container limit");
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Unable to allocate Butteraugli image storage");
  }
}

[[nodiscard]] int64_t Mirror(int64_t coordinate, int64_t size) {
  while (coordinate < 0 || coordinate >= size) {
    if (coordinate < 0) {
      coordinate = -coordinate - 1;
    } else {
      coordinate = 2 * size - 1 - coordinate;
    }
  }
  return coordinate;
}

[[nodiscard]] float HorizontalSeparable5(const float *row, int64_t x,
                                         int64_t width, float weight0,
                                         float weight1, float weight2) {

  float result = row[x] * weight0;
  result = UnfusedMultiplyAdd(
      row[Mirror(x - 1, width)] + row[Mirror(x + 1, width)], weight1, result);
  result = UnfusedMultiplyAdd(
      row[Mirror(x - 2, width)] + row[Mirror(x + 2, width)], weight2, result);
  return result;
}

void Separable5(ConstPlaneF32View input, const std::vector<float> &kernel,
                PlaneF32View output) {

  float sum_weights = 0.0f;
  for (float weight : kernel) {
    sum_weights += weight;
  }
  const float scale = 1.0f / sum_weights;
  const float weight0 = kernel[2] * scale;
  const float weight1 = kernel[1] * scale;
  const float weight2 = kernel[0] * scale;
  const int64_t width = static_cast<int64_t>(input.extent.width);
  const int64_t height = static_cast<int64_t>(input.extent.height);

  for (int64_t y = 0; y < height; ++y) {
    float *destination = output.Row(static_cast<size_t>(y));
    for (int64_t x = 0; x < width; ++x) {
      const float center =
          HorizontalSeparable5(input.Row(static_cast<size_t>(y)), x, width,
                               weight0, weight1, weight2);
      const float top1 = HorizontalSeparable5(
          input.Row(static_cast<size_t>(Mirror(y - 1, height))), x, width,
          weight0, weight1, weight2);
      const float bottom1 = HorizontalSeparable5(
          input.Row(static_cast<size_t>(Mirror(y + 1, height))), x, width,
          weight0, weight1, weight2);
      const float top2 = HorizontalSeparable5(
          input.Row(static_cast<size_t>(Mirror(y - 2, height))), x, width,
          weight0, weight1, weight2);
      const float bottom2 = HorizontalSeparable5(
          input.Row(static_cast<size_t>(Mirror(y + 2, height))), x, width,
          weight0, weight1, weight2);
      float result = center * weight0;
      result = UnfusedMultiplyAdd(top1 + bottom1, weight1, result);
      result = UnfusedMultiplyAdd(top2 + bottom2, weight2, result);
      destination[x] = result;
    }
  }
}

void ConvolveBorderColumn(ConstPlaneF32View input,
                          const std::vector<float> &kernel, size_t x,
                          float *destination) {

  const size_t radius = kernel.size() / 2;
  const size_t first = x < radius ? 0 : x - radius;
  const size_t last = std::min(input.extent.width - 1, x + radius);
  float weight = 0.0f;
  for (size_t source_x = first; source_x <= last; ++source_x) {
    weight += kernel[source_x + radius - x];
  }
  const float scale = 1.0f / weight;
  for (size_t y = 0; y < input.extent.height; ++y) {
    const float *source = input.Row(y);
    float sum = 0.0f;
    for (size_t source_x = first; source_x <= last; ++source_x) {
      sum += source[source_x] * kernel[source_x + radius - x];
    }
    destination[y] = sum * scale;
  }
}

void ConvolveInterior7(ConstPlaneF32View input, size_t first_x, size_t last_x,
                       const std::array<float, 17> &scaled_kernel,
                       PlaneF32View output) {

  const float sk0 = scaled_kernel[0];
  const float sk1 = scaled_kernel[1];
  const float sk2 = scaled_kernel[2];
  const float sk3 = scaled_kernel[3];
  for (size_t y = 0; y < input.extent.height; ++y) {
    const float *source = input.Row(y) + first_x - 3;
    for (size_t x = first_x; x < last_x; ++x, ++source) {
      const float sum0 = (source[0] + source[6]) * sk0;
      const float sum1 = (source[1] + source[5]) * sk1;
      const float sum2 = (source[2] + source[4]) * sk2;
      output.Row(x)[y] = source[3] * sk3 + sum0 + sum1 + sum2;
    }
  }
}

void ConvolveInterior13(ConstPlaneF32View input, size_t first_x, size_t last_x,
                        const std::array<float, 17> &scaled_kernel,
                        PlaneF32View output) {

  for (size_t y = 0; y < input.extent.height; ++y) {
    const float *source = input.Row(y) + first_x - 6;
    for (size_t x = first_x; x < last_x; ++x, ++source) {
      float sum0 = (source[0] + source[12]) * scaled_kernel[0];
      float sum1 = (source[1] + source[11]) * scaled_kernel[1];
      float sum2 = (source[2] + source[10]) * scaled_kernel[2];
      float sum3 = (source[3] + source[9]) * scaled_kernel[3];
      sum0 += (source[4] + source[8]) * scaled_kernel[4];
      sum1 += (source[5] + source[7]) * scaled_kernel[5];
      const float sum = source[6] * scaled_kernel[6];
      output.Row(x)[y] = sum + sum0 + sum1 + sum2 + sum3;
    }
  }
}

void ConvolveInterior15(ConstPlaneF32View input, size_t first_x, size_t last_x,
                        const std::array<float, 17> &scaled_kernel,
                        PlaneF32View output) {

  for (size_t y = 0; y < input.extent.height; ++y) {
    const float *source = input.Row(y) + first_x - 7;
    for (size_t x = first_x; x < last_x; ++x, ++source) {
      float sum0 = (source[0] + source[14]) * scaled_kernel[0];
      float sum1 = (source[1] + source[13]) * scaled_kernel[1];
      float sum2 = (source[2] + source[12]) * scaled_kernel[2];
      float sum3 = (source[3] + source[11]) * scaled_kernel[3];
      sum0 += (source[4] + source[10]) * scaled_kernel[4];
      sum1 += (source[5] + source[9]) * scaled_kernel[5];
      sum2 += (source[6] + source[8]) * scaled_kernel[6];
      const float sum = source[7] * scaled_kernel[7];
      output.Row(x)[y] = sum + sum0 + sum1 + sum2 + sum3;
    }
  }
}

void ConvolveInterior33(ConstPlaneF32View input, size_t first_x, size_t last_x,
                        const std::array<float, 17> &scaled_kernel,
                        PlaneF32View output) {

  for (size_t y = 0; y < input.extent.height; ++y) {
    const float *source = input.Row(y) + first_x - 16;
    for (size_t x = first_x; x < last_x; ++x, ++source) {
      float sum0 = (source[0] + source[32]) * scaled_kernel[0];
      float sum1 = (source[1] + source[31]) * scaled_kernel[1];
      float sum2 = (source[2] + source[30]) * scaled_kernel[2];
      float sum3 = (source[3] + source[29]) * scaled_kernel[3];
      sum0 += (source[4] + source[28]) * scaled_kernel[4];
      sum1 += (source[5] + source[27]) * scaled_kernel[5];
      sum2 += (source[6] + source[26]) * scaled_kernel[6];
      sum3 += (source[7] + source[25]) * scaled_kernel[7];
      sum0 += (source[8] + source[24]) * scaled_kernel[8];
      sum1 += (source[9] + source[23]) * scaled_kernel[9];
      sum2 += (source[10] + source[22]) * scaled_kernel[10];
      sum3 += (source[11] + source[21]) * scaled_kernel[11];
      sum0 += (source[12] + source[20]) * scaled_kernel[12];
      sum1 += (source[13] + source[19]) * scaled_kernel[13];
      sum2 += (source[14] + source[18]) * scaled_kernel[14];
      sum3 += (source[15] + source[17]) * scaled_kernel[15];
      const float sum = source[16] * scaled_kernel[16];
      output.Row(x)[y] = sum + sum0 + sum1 + sum2 + sum3;
    }
  }
}

void ConvolutionWithTranspose(ConstPlaneF32View input,
                              const std::vector<float> &kernel,
                              PlaneF32View output) {

  const size_t radius = kernel.size() / 2;
  float full_weight = 0.0f;
  for (float weight : kernel) {
    full_weight += weight;
  }
  const float scale = 1.0f / full_weight;
  std::array<float, 17> scaled_kernel{};
  for (size_t index = 0; index <= radius; ++index) {
    scaled_kernel[index] = kernel[index] * scale;
  }

  const size_t first_interior = std::min(input.extent.width, radius);
  const size_t last_interior =
      input.extent.width > radius ? input.extent.width - radius : 0;
  if (first_interior < last_interior) {
    switch (kernel.size()) {
    case 7:
      ConvolveInterior7(input, first_interior, last_interior, scaled_kernel,
                        output);
      break;
    case 13:
      ConvolveInterior13(input, first_interior, last_interior, scaled_kernel,
                         output);
      break;
    case 15:
      ConvolveInterior15(input, first_interior, last_interior, scaled_kernel,
                         output);
      break;
    case 33:
      ConvolveInterior33(input, first_interior, last_interior, scaled_kernel,
                         output);
      break;
    }
  }

  for (size_t x = 0; x < first_interior; ++x) {
    ConvolveBorderColumn(input, kernel, x, output.Row(x));
  }
  for (size_t x = last_interior; x < input.extent.width; ++x) {
    ConvolveBorderColumn(input, kernel, x, output.Row(x));
  }
}

} // namespace

Status OwnedPlaneF32::Resize(Extent2D extent) {
  return ResizeStorage(extent, 1, &extent_, nullptr, &values_);
}

PlaneF32View OwnedPlaneF32::View() noexcept {
  return {values_.data(), extent_, extent_.width};
}

ConstPlaneF32View OwnedPlaneF32::ConstView() const noexcept {
  return {values_.data(), extent_, extent_.width};
}

Status OwnedImage3F::Resize(Extent2D extent) {
  return ResizeStorage(extent, 3, &extent_, &plane_size_, &values_);
}

Image3FView OwnedImage3F::View() noexcept {
  float *const base = values_.data();
  return {{{
      {base, extent_, extent_.width},
      {base == nullptr ? nullptr : base + plane_size_, extent_, extent_.width},
      {base == nullptr ? nullptr : base + 2 * plane_size_, extent_,
       extent_.width},
  }}};
}

ConstImage3FView OwnedImage3F::ConstView() const noexcept {
  const float *const base = values_.data();
  return {{{
      {base, extent_, extent_.width},
      {base == nullptr ? nullptr : base + plane_size_, extent_, extent_.width},
      {base == nullptr ? nullptr : base + 2 * plane_size_, extent_,
       extent_.width},
  }}};
}

Status BlurScratch::Prepare(Extent2D input_extent, size_t kernel_size,
                            bool needs_transposed) {

  if (kernel_size > kernel_.max_size()) {
    return Status::InvalidArgument(
        "Butteraugli Gaussian kernel exceeds the container limit");
  }
  try {
    kernel_.resize(kernel_size);
  } catch (const std::length_error &) {
    return Status::InvalidArgument(
        "Butteraugli Gaussian kernel exceeds the container limit");
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory(
        "Unable to allocate the Butteraugli Gaussian kernel");
  }

  if (!needs_transposed) {
    return Status::Ok();
  }
  return transposed_.Resize({input_extent.height, input_extent.width});
}

Status OwnedPsychoImage::Resize(Extent2D extent) {
  return ResizeStorage(extent, 10, &extent_, &plane_size_, &values_);
}

PlaneF32View OwnedPsychoImage::Plane(size_t index) noexcept {
  if (index >= 10 || values_.empty()) {
    return {};
  }
  return {values_.data() + index * plane_size_, extent_, extent_.width};
}

ConstPlaneF32View OwnedPsychoImage::Plane(size_t index) const noexcept {
  if (index >= 10 || values_.empty()) {
    return {};
  }
  return {values_.data() + index * plane_size_, extent_, extent_.width};
}

Image3FView OwnedPsychoImage::Image(size_t first_plane) noexcept {
  return {
      {{Plane(first_plane), Plane(first_plane + 1), Plane(first_plane + 2)}}};
}

ConstImage3FView OwnedPsychoImage::Image(size_t first_plane) const noexcept {
  return {
      {{Plane(first_plane), Plane(first_plane + 1), Plane(first_plane + 2)}}};
}

Image3FView OwnedPsychoImage::LowFrequencyView() noexcept { return Image(0); }

ConstImage3FView OwnedPsychoImage::LowFrequencyView() const noexcept {
  return Image(0);
}

Image3FView OwnedPsychoImage::MediumFrequencyView() noexcept {
  return Image(3);
}

ConstImage3FView OwnedPsychoImage::MediumFrequencyView() const noexcept {
  return Image(3);
}

PlaneF32View OwnedPsychoImage::HighFrequencyView(size_t channel) noexcept {
  return channel < 2 ? Plane(6 + channel) : PlaneF32View{};
}

ConstPlaneF32View
OwnedPsychoImage::HighFrequencyView(size_t channel) const noexcept {
  return channel < 2 ? Plane(6 + channel) : ConstPlaneF32View{};
}

PlaneF32View OwnedPsychoImage::UltraHighFrequencyView(size_t channel) noexcept {
  return channel < 2 ? Plane(8 + channel) : PlaneF32View{};
}

ConstPlaneF32View
OwnedPsychoImage::UltraHighFrequencyView(size_t channel) const noexcept {
  return channel < 2 ? Plane(8 + channel) : ConstPlaneF32View{};
}

Status OpsinScratch::Prepare(Extent2D extent) {
  Status status = blurred_.Resize(extent);
  if (!status.ok()) {
    return status;
  }
  return result_.Resize(extent);
}

Status FrequencyScratch::Prepare(Extent2D extent) {
  return blurred_.Resize(extent);
}

Status GaussianBlur(ConstPlaneF32View input, float sigma, BlurScratch *scratch,
                    PlaneF32View output) {

  if (!ValidPlane(input) || !ValidPlane(output) ||
      input.extent != output.extent || input.data == output.data ||
      scratch == nullptr || !std::isfinite(sigma) || sigma <= 0.0f) {
    return Status::InvalidArgument(
        "Butteraugli blur views, scratch, or sigma are invalid");
  }

  const float scaled_radius = 2.25f * std::abs(sigma);
  if (!std::isfinite(scaled_radius) || scaled_radius >= 17.0f) {
    return Status::InvalidArgument(
        "Butteraugli blur sigma produces an unsupported radius");
  }
  const int radius = std::max(1, static_cast<int>(scaled_radius));
  const size_t kernel_size = 2 * static_cast<size_t>(radius) + 1;
  if (kernel_size != 5 && kernel_size != 7 && kernel_size != 13 &&
      kernel_size != 15 && kernel_size != 33) {
    return Status::InvalidArgument(
        "Butteraugli blur kernel radius is unsupported");
  }

  Status status = scratch->Prepare(input.extent, kernel_size, kernel_size != 5);
  if (!status.ok()) {
    return status;
  }

  const double exponent_scale = -1.0 / (2.0 * sigma * sigma);
  for (int index = -radius; index <= radius; ++index) {
    scratch->kernel_[static_cast<size_t>(index + radius)] =
        static_cast<float>(std::exp(exponent_scale * index * index));
  }

  if (kernel_size == 5) {
    Separable5(input, scratch->kernel_, output);
    return Status::Ok();
  }

  ConvolutionWithTranspose(input, scratch->kernel_,
                           scratch->transposed_.View());
  ConvolutionWithTranspose(scratch->transposed_.ConstView(), scratch->kernel_,
                           output);
  return Status::Ok();
}

Status OpsinDynamicsImage(ConstImage3FView linear_rgb, float intensity_target,
                          OpsinScratch *scratch, Image3FView xyb) {
  if (!ValidImage(linear_rgb) || !ValidImage(xyb) ||
      linear_rgb.extent() != xyb.extent() || scratch == nullptr ||
      !std::isfinite(intensity_target) || intensity_target <= 0.0f ||
      !PixelsAreFinite(linear_rgb)) {
    return Status::InvalidArgument(
        "Butteraugli opsin views, scratch, intensity, or pixels are invalid");
  }

  Status status = scratch->Prepare(linear_rgb.extent());
  if (!status.ok()) {
    return status;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    status =
        GaussianBlur(linear_rgb.plane[channel], kOpsinBlurSigma,
                     &scratch->blur_, scratch->blurred_.View().plane[channel]);
    if (!status.ok()) {
      return status;
    }
  }

  constexpr float kMinimum = 1.0e-4f;
  constexpr float kMinimum01 = 1.7557483643287353f;
  constexpr float kMinimum2 = 12.226454707163354f;
  Image3FView result = scratch->result_.View();
  const ConstImage3FView blurred = scratch->blurred_.ConstView();
  for (size_t y = 0; y < linear_rgb.height(); ++y) {
    for (size_t x = 0; x < linear_rgb.width(); ++x) {
      float pre_mixed0 = 0.0f;
      float pre_mixed1 = 0.0f;
      float pre_mixed2 = 0.0f;
      OpsinAbsorbance<true>(blurred.plane[0].Row(y)[x] * intensity_target,
                            blurred.plane[1].Row(y)[x] * intensity_target,
                            blurred.plane[2].Row(y)[x] * intensity_target,
                            &pre_mixed0, &pre_mixed1, &pre_mixed2);
      pre_mixed0 = std::max(pre_mixed0, kMinimum);
      pre_mixed1 = std::max(pre_mixed1, kMinimum);
      pre_mixed2 = std::max(pre_mixed2, kMinimum);
      const float sensitivity0 =
          std::max(Gamma(pre_mixed0) / pre_mixed0, kMinimum);
      const float sensitivity1 =
          std::max(Gamma(pre_mixed1) / pre_mixed1, kMinimum);
      const float sensitivity2 =
          std::max(Gamma(pre_mixed2) / pre_mixed2, kMinimum);

      float current0 = 0.0f;
      float current1 = 0.0f;
      float current2 = 0.0f;
      OpsinAbsorbance<false>(linear_rgb.plane[0].Row(y)[x] * intensity_target,
                             linear_rgb.plane[1].Row(y)[x] * intensity_target,
                             linear_rgb.plane[2].Row(y)[x] * intensity_target,
                             &current0, &current1, &current2);
      current0 = std::max(current0 * sensitivity0, kMinimum01);
      current1 = std::max(current1 * sensitivity1, kMinimum01);
      current2 = std::max(current2 * sensitivity2, kMinimum2);
      result.plane[0].Row(y)[x] = current0 - current1;
      result.plane[1].Row(y)[x] = current0 + current1;
      result.plane[2].Row(y)[x] = current2;
      if (!std::isfinite(result.plane[0].Row(y)[x]) ||
          !std::isfinite(result.plane[1].Row(y)[x]) ||
          !std::isfinite(result.plane[2].Row(y)[x])) {
        return Status::Internal(
            "Butteraugli opsin computation produced non-finite values");
      }
    }
  }

  CopyImage(scratch->result_.ConstView(), xyb);
  return Status::Ok();
}

float MaximumClamp(float value, float maximum) noexcept {
  constexpr float kMultiplier = 0.724216145665f;
  if (value >= maximum) {
    return UnfusedMultiplyAdd(value - maximum, kMultiplier, maximum);
  }
  if (value < -maximum) {
    return UnfusedMultiplyAdd(value + maximum, kMultiplier, -maximum);
  }
  return value;
}

float RemoveRangeAroundZero(float value, float width) noexcept {
  if (value > width) {
    return value - width;
  }
  if (value < -width) {
    return value + width;
  }
  return 0.0f;
}

float AmplifyRangeAroundZero(float value, float width) noexcept {
  if (value > width) {
    return value + width;
  }
  if (value < -width) {
    return value - width;
  }
  return value + value;
}

float SuppressXByY(float y, float x) noexcept {
  constexpr float kSuppress = 46.0f;
  constexpr float kMinimumScale = 0.653020556257f;
  const float scaler =
      UnfusedMultiplyAdd(kSuppress / UnfusedMultiplyAdd(y, y, kSuppress),
                         1.0f - kMinimumScale, kMinimumScale);
  return scaler * x;
}

Status SeparateFrequencies(ConstImage3FView xyb, FrequencyScratch *scratch,
                           OwnedPsychoImage *output) {
  if (!ValidImage(xyb) || scratch == nullptr || output == nullptr ||
      !PixelsAreFinite(xyb)) {
    return Status::InvalidArgument(
        "Butteraugli frequency input, scratch, output, or pixels are invalid");
  }

  OwnedPsychoImage candidate;
  Status status = candidate.Resize(xyb.extent());
  if (!status.ok()) {
    return status;
  }
  status = scratch->Prepare(xyb.extent());
  if (!status.ok()) {
    return status;
  }

  Image3FView low = candidate.LowFrequencyView();
  Image3FView medium = candidate.MediumFrequencyView();
  for (size_t channel = 0; channel < 3; ++channel) {
    status = GaussianBlur(xyb.plane[channel], kLowFrequencyBlurSigma,
                          &scratch->blur_, low.plane[channel]);
    if (!status.ok()) {
      return status;
    }
    for (size_t y = 0; y < xyb.height(); ++y) {
      for (size_t x = 0; x < xyb.width(); ++x) {
        medium.plane[channel].Row(y)[x] =
            xyb.plane[channel].Row(y)[x] - low.plane[channel].Row(y)[x];
      }
    }
  }

  constexpr float kLowXMultiplier = 33.832837186260f;
  constexpr float kLowYMultiplier = 14.458268100570f;
  constexpr float kLowBMultiplier = 49.87984651440f;
  constexpr float kLowYToBMultiplier = -0.362267051518f;
  for (size_t y = 0; y < xyb.height(); ++y) {
    for (size_t x = 0; x < xyb.width(); ++x) {
      const float low_y = low.plane[1].Row(y)[x];
      low.plane[2].Row(y)[x] = UnfusedMultiplyAdd(kLowYToBMultiplier, low_y,
                                                  low.plane[2].Row(y)[x]) *
                               kLowBMultiplier;
      low.plane[0].Row(y)[x] *= kLowXMultiplier;
      low.plane[1].Row(y)[x] = low_y * kLowYMultiplier;
    }
  }

  for (size_t channel = 0; channel < 3; ++channel) {
    if (channel < 2) {
      CopyPlane(AsConst(medium.plane[channel]),
                candidate.HighFrequencyView(channel));
    }
    status =
        GaussianBlur(AsConst(medium.plane[channel]), kHighFrequencyBlurSigma,
                     &scratch->blur_, scratch->blurred_.View());
    if (!status.ok()) {
      return status;
    }
    if (channel == 2) {
      CopyPlane(scratch->blurred_.ConstView(), medium.plane[channel]);
      continue;
    }
    PlaneF32View high = candidate.HighFrequencyView(channel);
    for (size_t y = 0; y < xyb.height(); ++y) {
      for (size_t x = 0; x < xyb.width(); ++x) {
        const float medium_value = scratch->blurred_.ConstView().Row(y)[x];
        high.Row(y)[x] -= medium_value;
        medium.plane[channel].Row(y)[x] =
            channel == 0 ? RemoveRangeAroundZero(medium_value, 0.29f)
                         : AmplifyRangeAroundZero(medium_value, 0.1f);
      }
    }
  }

  PlaneF32View high_x = candidate.HighFrequencyView(0);
  const ConstPlaneF32View high_y = AsConst(candidate.HighFrequencyView(1));
  for (size_t y = 0; y < xyb.height(); ++y) {
    for (size_t x = 0; x < xyb.width(); ++x) {
      high_x.Row(y)[x] = SuppressXByY(high_y.Row(y)[x], high_x.Row(y)[x]);
    }
  }

  for (size_t channel = 0; channel < 2; ++channel) {
    PlaneF32View high = candidate.HighFrequencyView(channel);
    PlaneF32View ultra = candidate.UltraHighFrequencyView(channel);
    CopyPlane(AsConst(high), ultra);
    status = GaussianBlur(AsConst(high), kUltraHighFrequencyBlurSigma,
                          &scratch->blur_, scratch->blurred_.View());
    if (!status.ok()) {
      return status;
    }
    for (size_t y = 0; y < xyb.height(); ++y) {
      for (size_t x = 0; x < xyb.width(); ++x) {
        float high_value = scratch->blurred_.ConstView().Row(y)[x];
        if (channel == 0) {
          ultra.Row(y)[x] =
              RemoveRangeAroundZero(ultra.Row(y)[x] - high_value, 0.04f);
          high.Row(y)[x] = RemoveRangeAroundZero(high_value, 1.5f);
        } else {
          high_value = MaximumClamp(high_value, 28.4691806922f);
          float ultra_value = ultra.Row(y)[x] - high_value;
          ultra_value = MaximumClamp(ultra_value, 5.19175294647f);
          ultra.Row(y)[x] = ultra_value * 2.69313763794f;
          high.Row(y)[x] = AmplifyRangeAroundZero(high_value * 2.155f, 0.132f);
        }
      }
    }
  }

  if (!PsychoImageIsFinite(candidate)) {
    return Status::Internal(
        "Butteraugli frequency computation produced non-finite values");
  }
  *output = std::move(candidate);
  return Status::Ok();
}

} // namespace gjxl::butteraugli_internal
