// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/dc_quantization.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gjxl {
namespace {

constexpr size_t kX = 0;
constexpr size_t kY = 1;
constexpr size_t kB = 2;

struct DcSample {
  std::array<int32_t, 3> quantized;
  std::array<float, 3> reconstructed;
};

Status RoundDc(
  float value,
  int32_t* quantized) {

  if (!std::isfinite(value)) {
    return Status::InvalidArgument(
      "Scaled DC coefficient is not finite");
  }

  // libjxl's default 4:4:4 path uses std::round on the float result.
  const double rounded = static_cast<double>(std::round(value));
  if (rounded < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
      rounded > static_cast<double>(std::numeric_limits<int32_t>::max())) {
    return Status::InvalidArgument(
      "Quantized DC coefficient exceeds int32 range");
  }

  *quantized = static_cast<int32_t>(rounded);
  return Status::Ok();
}

Status QuantizeDcSample(
  float source_x,
  float source_y,
  float source_b,
  const std::array<float, 3>& inverse_steps,
  const std::array<float, 3>& steps,
  DcSample* out) {

  if (!std::isfinite(source_y) ||
      !std::isfinite(source_x) ||
      !std::isfinite(source_b)) {
    return Status::InvalidArgument(
      "DC coefficients must be finite");
  }

  DcSample result;
  Status status = RoundDc(
    source_y * inverse_steps[kY],
    &result.quantized[kY]);
  if (!status.ok()) {
    return status;
  }
  result.reconstructed[kY] =
    static_cast<float>(result.quantized[kY]) * steps[kY];

  // Default JPEG XL DC CfL predicts no X and predicts B from Y with a factor
  // of one. Use quantized/dequantized Y exactly as the decoder.
  status = RoundDc(
    source_x * inverse_steps[kX],
    &result.quantized[kX]);
  if (!status.ok()) {
    return status;
  }
  status = RoundDc(
    (source_b - result.reconstructed[kY]) * inverse_steps[kB],
    &result.quantized[kB]);
  if (!status.ok()) {
    return status;
  }

  result.reconstructed[kX] =
    static_cast<float>(result.quantized[kX]) * steps[kX];
  result.reconstructed[kB] =
    static_cast<float>(result.quantized[kB]) * steps[kB] +
    result.reconstructed[kY];
  if (!std::isfinite(result.reconstructed[kX]) ||
      !std::isfinite(result.reconstructed[kY]) ||
      !std::isfinite(result.reconstructed[kB])) {
    return Status::InvalidArgument(
      "Reconstructed DC coefficient is not finite");
  }

  *out = result;
  return Status::Ok();
}

}  // namespace

Status QuantizeDcCoefficients(
  ConstImage3FView dc,
  const Quantizer& quantizer,
  DcQuantizationOutput output) {

  if (!dc.valid() ||
      !quantizer.valid() ||
      !output.quantized.valid() ||
      !output.reconstructed.valid() ||
      output.quantized.extent() != dc.extent() ||
      output.reconstructed.extent() != dc.extent()) {
    return Status::InvalidArgument(
      "DC quantization inputs are invalid or differently sized");
  }

  size_t sample_count = 0;
  if (!dc.extent().try_area(&sample_count)) {
    return Status::InvalidArgument(
      "DC quantization dimensions are too large");
  }

  const std::array<float, 3>& inverse_steps = quantizer.inverse_dc_steps();
  const std::array<float, 3>& steps = quantizer.dc_steps();
  for (size_t y = 0; y < dc.height(); ++y) {
    for (size_t x = 0; x < dc.width(); ++x) {
      DcSample unused;
      Status status = QuantizeDcSample(
        dc.plane[kX].Row(y)[x],
        dc.plane[kY].Row(y)[x],
        dc.plane[kB].Row(y)[x],
        inverse_steps,
        steps,
        &unused);
      if (!status.ok()) {
        return status;
      }
    }
  }

  // Validation above makes this write pass infallible and preserves atomic
  // failure without allocating a second set of DC planes.
  for (size_t y = 0; y < dc.height(); ++y) {
    for (size_t x = 0; x < dc.width(); ++x) {
      DcSample result;
      const Status status = QuantizeDcSample(
        dc.plane[kX].Row(y)[x],
        dc.plane[kY].Row(y)[x],
        dc.plane[kB].Row(y)[x],
        inverse_steps,
        steps,
        &result);
      if (!status.ok()) {
        return Status::Internal(
          "Validated DC coefficient changed during quantization");
      }
      for (size_t channel = 0; channel < 3; ++channel) {
        output.quantized.plane[channel].Row(y)[x] =
          result.quantized[channel];
        output.reconstructed.plane[channel].Row(y)[x] =
          result.reconstructed[channel];
      }
    }
  }

  return Status::Ok();
}

}  // namespace gjxl
