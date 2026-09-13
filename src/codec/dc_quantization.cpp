// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/dc_quantization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>

#include "codec/weighted_dc_predictor_internal.h"
#include "core/managed_allocator.h"

namespace gjxl {
namespace {

constexpr size_t kX = 0;
constexpr size_t kY = 1;
constexpr size_t kB = 2;

struct DcSample {
  std::array<int32_t, 3> quantized;
  std::array<float, 3> reconstructed;
};

Status RoundDc(float value, int32_t *quantized) {

  if (!std::isfinite(value)) {
    return Status::InvalidArgument("Scaled DC coefficient is not finite");
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

Status QuantizeDcSample(float source_x, float source_y, float source_b,
                        const std::array<float, 3> &inverse_steps,
                        const std::array<float, 3> &steps, DcSample *out,
                        bool precise_chroma) {

  if (!std::isfinite(source_y) || !std::isfinite(source_x) ||
      !std::isfinite(source_b)) {
    return Status::InvalidArgument("DC coefficients must be finite");
  }

  DcSample result;
  Status status = RoundDc(source_y * inverse_steps[kY], &result.quantized[kY]);
  if (!status.ok()) {
    return status;
  }
  result.reconstructed[kY] =
      static_cast<float>(result.quantized[kY]) * steps[kY];

  // Default JPEG XL DC CfL predicts no X and predicts B from Y with a factor
  // of one. Use quantized/dequantized Y exactly as the decoder.
  status = RoundDc(source_x * inverse_steps[kX], &result.quantized[kX]);
  if (!status.ok()) {
    return status;
  }
  status = RoundDc((source_b - result.reconstructed[kY]) * inverse_steps[kB],
                   &result.quantized[kB]);
  if (!status.ok()) {
    return status;
  }

  result.reconstructed[kX] =
      static_cast<float>(result.quantized[kX]) * steps[kX];
  result.reconstructed[kB] =
      precise_chroma
          ? std::fma(result.reconstructed[kY], 1.0f,
                     static_cast<float>(result.quantized[kB]) * steps[kB])
          : static_cast<float>(result.quantized[kB]) * steps[kB] +
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

} // namespace

Status QuantizeDcCoefficients(ConstImage3FView dc, const Quantizer &quantizer,
                              DcQuantizationOutput output) {
  return QuantizeDcCoefficients(dc, quantizer, output, {});
}

Status QuantizeDcCoefficients(ConstImage3FView dc, const Quantizer &quantizer,
                              DcQuantizationOutput output,
                              DcQuantizationOptions options) {

  if (!IsValidDcQuantization(options) || !dc.valid() || !quantizer.valid() ||
      !output.quantized.valid() || !output.reconstructed.valid() ||
      output.quantized.extent() != dc.extent() ||
      output.reconstructed.extent() != dc.extent()) {
    return Status::InvalidArgument(
        "DC quantization inputs are invalid or differently sized");
  }

  size_t sample_count = 0;
  if (!dc.extent().try_area(&sample_count) ||
      sample_count >
          std::numeric_limits<size_t>::max() / (3 * sizeof(int32_t))) {
    return Status::InvalidArgument("DC quantization dimensions are too large");
  }

  std::array<float, 3> inverse_steps = quantizer.inverse_dc_steps();
  std::array<float, 3> steps = quantizer.dc_steps();
  const float precision = static_cast<float>(1u << options.extra_dc_precision);
  for (size_t c = 0; c < 3; ++c) {
    inverse_steps[c] *= precision;
    steps[c] /= precision;
  }
  if (options.mode == DcQuantizationMode::kPredictionAware) {
    try {
      // Retain candidate integers until every channel/group has succeeded.
      // Reconstructing the validated integers during commit needs no allocation
      // and permits the input to alias the reconstructed output.
      resource_budget_internal::ManagedVector<int32_t> candidate(3 *
                                                                 sample_count);
      using WeightedState = dc_prediction_internal::WeightedDcPredictor<
          resource_budget_internal::ResourceClass::kCount>;
      for (size_t group_y = 0; group_y < dc.height();
           group_y += kDcPredictionGroupBlockDimension) {
        const size_t height =
            std::min(kDcPredictionGroupBlockDimension, dc.height() - group_y);
        for (size_t group_x = 0; group_x < dc.width();
             group_x += kDcPredictionGroupBlockDimension) {
          const size_t width =
              std::min(kDcPredictionGroupBlockDimension, dc.width() - group_x);
          // No predictor allocation for the gradient alternative. The optional
          // state owns its five managed arrays, never an untracked heap object.
          std::optional<WeightedState> weighted;
          if (options.prediction == VarDctDcPrediction::kWeighted)
            weighted.emplace(width);
          for (size_t c : {kY, kX, kB}) {
            if (weighted)
              weighted->Reset();
            int32_t *plane = candidate.data() + c * sample_count;
            for (size_t y = 0; y < height; ++y) {
              for (size_t x = 0; x < width; ++x) {
                const size_t index = (group_y + y) * dc.width() + group_x + x;
                const int64_t w = x   ? plane[index - 1]
                                  : y ? plane[index - dc.width()]
                                      : 0;
                const int64_t n = y ? plane[index - dc.width()] : w;
                const int64_t nw = x && y ? plane[index - dc.width() - 1] : w;
                int64_t prediction;
                if (weighted) {
                  const int64_t ne =
                      y && x + 1 < width ? plane[index - dc.width() + 1] : n;
                  const int64_t nn = y > 1 ? plane[index - 2 * dc.width()] : n;
                  prediction = weighted->Predict(x, y, n, w, ne, nw, nn).first;
                } else {
                  prediction =
                      std::clamp(n + w - nw, std::min(n, w), std::max(n, w));
                }
                float value = dc.plane[c].Row(group_y + y)[group_x + x];
                if (c == kB)
                  // Pin the multiply-subtract association used by the native
                  // libjxl quantization prefix on our qualified toolchain.
                  value = std::fma(
                      -static_cast<float>(candidate[kY * sample_count + index]),
                      steps[kY], value);
                float residual = value * inverse_steps[c];
                residual -= static_cast<float>(prediction);
                if (residual > -0.62f && residual < 0.62f)
                  residual = 0;
                float rounded = std::round(residual);
                if (rounded > 2 || rounded < -2)
                  rounded = std::round(residual * 0.5f) * 2;
                int32_t integer_residual = 0;
                if (Status status = RoundDc(rounded, &integer_residual);
                    !status.ok())
                  return status;
                const int64_t quantized = prediction + integer_residual;
                if (quantized < std::numeric_limits<int32_t>::min() ||
                    quantized > std::numeric_limits<int32_t>::max()) {
                  return Status::InvalidArgument(
                      "Prediction-aware DC exceeds int32 range");
                }
                plane[index] = static_cast<int32_t>(quantized);
                if (weighted && !weighted->Update(plane[index], x, y)) {
                  return Status::InvalidArgument(
                      "Prediction-aware DC predictor range exceeded");
                }
              }
            }
          }
        }
      }
      for (size_t y = 0; y < dc.height(); ++y) {
        for (size_t x = 0; x < dc.width(); ++x) {
          const size_t index = y * dc.width() + x;
          const float reconstructed_y =
              static_cast<float>(candidate[kY * sample_count + index]) *
              steps[kY];
          for (size_t c = 0; c < 3; ++c) {
            const int32_t quantized = candidate[c * sample_count + index];
            output.quantized.plane[c].Row(y)[x] = quantized;
            float reconstructed = static_cast<float>(quantized) * steps[c];
            if (c == kB)
              reconstructed = std::fma(reconstructed_y, 1.0f, reconstructed);
            output.reconstructed.plane[c].Row(y)[x] = reconstructed;
          }
        }
      }
      return Status::Ok();
    } catch (const resource_budget_internal::ManagedAllocationFailure &error) {
      return error.status();
    } catch (const std::bad_alloc &) {
      return Status::OutOfMemory("Prediction-aware DC allocation failed");
    } catch (const std::length_error &) {
      return Status::OutOfMemory("Prediction-aware DC allocation is too large");
    }
  }
  for (size_t y = 0; y < dc.height(); ++y) {
    for (size_t x = 0; x < dc.width(); ++x) {
      DcSample unused;
      Status status = QuantizeDcSample(
          dc.plane[kX].Row(y)[x], dc.plane[kY].Row(y)[x],
          dc.plane[kB].Row(y)[x], inverse_steps, steps, &unused,
          options.extra_dc_precision != 0);
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
          dc.plane[kX].Row(y)[x], dc.plane[kY].Row(y)[x],
          dc.plane[kB].Row(y)[x], inverse_steps, steps, &result,
          options.extra_dc_precision != 0);
      if (!status.ok()) {
        return Status::Internal(
            "Validated DC coefficient changed during quantization");
      }
      for (size_t channel = 0; channel < 3; ++channel) {
        output.quantized.plane[channel].Row(y)[x] = result.quantized[channel];
        output.reconstructed.plane[channel].Row(y)[x] =
            result.reconstructed[channel];
      }
    }
  }

  return Status::Ok();
}

} // namespace gjxl
