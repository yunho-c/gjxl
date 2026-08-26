// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// The scalar Gaussian kernel construction and convolution structure are
// adapted from pinned JPEG XL Butteraugli code distributed under its BSD-style
// license. See third_party/libjxl/LICENSE.

#include "codec/butteraugli_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
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
  result = (row[Mirror(x - 1, width)] + row[Mirror(x + 1, width)]) * weight1 +
           result;
  result = (row[Mirror(x - 2, width)] + row[Mirror(x + 2, width)]) * weight2 +
           result;
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
      result = (top1 + bottom1) * weight1 + result;
      result = (top2 + bottom2) * weight2 + result;
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

} // namespace gjxl::butteraugli_internal
