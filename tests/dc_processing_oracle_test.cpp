// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Calls the pinned library's actual quantization and smoothing functions.
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "codec/dc_quantization.h"
#include "codec/dc_smoothing.h"
#include "core/image_buffer.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {
// Internal functions with external linkage in pinned enc_modular.cc. Keep
// these declarations test-only; the native encoder never links libjxl.
int QuantizeWP(const int32_t *, size_t, size_t, size_t, size_t, size_t,
               weighted::State *, float, float, bool *);
int QuantizeGradient(const int32_t *, size_t, size_t, size_t, size_t, size_t,
                     float, float);
} // namespace jxl

namespace {
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void QuantizationCase(gjxl::Extent2D extent, const gjxl::Quantizer &quantizer,
                      gjxl::VarDctDcPrediction prediction, unsigned pattern, uint8_t extra_precision) {
  const size_t stride = extent.width + 7;
  const size_t area = stride * extent.height;
  std::array<std::vector<float>, 3> source, reconstructed;
  std::array<std::vector<int32_t>, 3> actual, expected;
  gjxl::ConstImage3FView input;
  gjxl::DcQuantizationOutput output;
  uint32_t random = 1729;
  for (size_t c = 0; c < 3; ++c) {
    source[c].assign(area, -900);
    reconstructed[c].assign(area, -901);
    actual[c].assign(area, -777);
    expected[c].assign(area, -777);
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        random = random * 1664525 + 1013904223;
        const int value =
            pattern == 0   ? 0
            : pattern == 1 ? 13
            : (pattern == 2 || pattern == 5)
                ? static_cast<int>((x * 11 + y * 3 + c * 7) % 97) - 48
            : pattern == 3      ? static_cast<int>(random >> 16) - 32768
            : ((x + y + c) & 1) ? 1023
                                : -1024;
        source[c][y * stride + x] = pattern == 5
          ? (static_cast<float>(value) * 0.17f + 2.5f) * quantizer.dc_steps()[c]
          : static_cast<float>(value) * quantizer.dc_steps()[c] * 0.17f;
      }
    }
    input.plane[c] = {source[c].data(), extent, stride};
    output.quantized.plane[c] = {actual[c].data(), extent, stride};
    output.reconstructed.plane[c] = {reconstructed[c].data(), extent, stride};
  }
  const gjxl::DcQuantizationOptions options{
      gjxl::DcQuantizationMode::kPredictionAware, prediction, extra_precision};
  const float precision = float(1u << extra_precision);
  Check(gjxl::QuantizeDcCoefficients(input, quantizer, output, options).ok(),
        "Native DC quantization failed");
  for (size_t gy = 0; gy < extent.height; gy += 256) {
    for (size_t gx = 0; gx < extent.width; gx += 256) {
      const size_t width = std::min<size_t>(256, extent.width - gx);
      const size_t height = std::min<size_t>(256, extent.height - gy);
      for (size_t c : {size_t{1}, size_t{0}, size_t{2}}) {
        jxl::weighted::Header header;
        jxl::weighted::PredictorMode(0, &header);
        jxl::weighted::State state(header, width, height);
        for (size_t y = 0; y < height; ++y) {
          int32_t *row = expected[c].data() + (gy + y) * stride + gx;
          for (size_t x = 0; x < width; ++x) {
            const size_t index = (gy + y) * stride + gx + x;
            float value = source[c][index];
            if (c == 2)
              value -= static_cast<float>(expected[1][index]) *
                       (quantizer.dc_steps()[1] / precision);
            bool outlier = false;
            const float inverse = quantizer.inverse_dc_steps()[c] * precision;
            row[x] = prediction == gjxl::VarDctDcPrediction::kWeighted
                         ? jxl::QuantizeWP(row, stride, c, x, y, width, &state,
                                           value, inverse, &outlier)
                         : jxl::QuantizeGradient(row, stride, c, x, y, width,
                                                 value, inverse);
            Check(!outlier, "Oracle input has an outlier");
            if (prediction == gjxl::VarDctDcPrediction::kWeighted)
              state.UpdateErrors(row[x], x, y, width);
            if (row[x] != actual[c][index]) {
              std::cerr << "Quantization mismatch at " << extent.width << 'x'
                        << extent.height << " c=" << c << " x=" << gx + x
                        << " y=" << gy + y << " pattern=" << pattern
                        << " expected=" << row[x]
                        << " actual=" << actual[c][index] << '\n';
              throw std::runtime_error(
                  "Native quantization differs from pinned libjxl");
            }
          }
        }
      }
    }
  }
  Check(expected == actual, "Quantization touched row padding");
  JxlMemoryManager manager;
  Check(jxl::MemoryManagerInit(&manager, nullptr),
        "Oracle memory manager failed");
  auto integer_or =
      jxl::Image::Create(&manager, extent.width, extent.height, 8, 3);
  auto float_or = jxl::Image3F::Create(&manager, extent.width, extent.height);
  auto context_or = jxl::ImageB::Create(&manager, extent.width, extent.height);
  Check(integer_or.ok() && float_or.ok() && context_or.ok(),
        "Dequantization oracle allocation failed");
  auto integers = std::move(integer_or).value_();
  auto floats = std::move(float_or).value_();
  auto contexts = std::move(context_or).value_();
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      std::copy_n(expected[c].data() + y * stride, extent.width,
                  integers.channel[c < 2 ? c ^ 1 : c].plane.Row(y));
  const float cfl[] = {0, 0, 1};
  jxl::DequantDC(jxl::Rect(0, 0, extent.width, extent.height), &floats,
                 &contexts, integers, quantizer.dc_steps().data(), 1.0f / precision, cfl,
                 jxl::YCbCrChromaSubsampling{}, jxl::BlockCtxMap{});
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        if (floats.ConstPlaneRow(c, y)[x] != reconstructed[c][y * stride + x]) {
          std::cerr << std::hexfloat << "DC reconstruction mismatch at "
                    << extent.width << 'x' << extent.height << " c=" << c
                    << " x=" << x << " y=" << y << " pattern=" << pattern
                    << " expected=" << floats.ConstPlaneRow(c, y)[x]
                    << " actual=" << reconstructed[c][y * stride + x] << '\n';
          throw std::runtime_error(
              "Reconstructed DC differs from pinned decoder dequantization");
        }
}
void SmoothingCase(gjxl::Extent2D extent, const gjxl::Quantizer &quantizer,
                   unsigned pattern) {
  gjxl::Image3FBuffer source(extent), output(extent);
  JxlMemoryManager manager;
  Check(jxl::MemoryManagerInit(&manager, nullptr),
        "Oracle memory manager failed");
  auto oracle_or = jxl::Image3F::Create(&manager, extent.width, extent.height);
  Check(oracle_or.ok(), "Oracle allocation failed");
  auto oracle = std::move(oracle_or).value_();
  uint32_t random = 2718;
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        random = random * 1664525 + 1013904223;
        const int delta = pattern == 0   ? 0
                          : pattern == 1 ? static_cast<int>((x + y) % 3) - 1
                          : (pattern == 2 || pattern == 5)
                              ? static_cast<int>((random >> 16) % 17) - 8
                          : ((x + y) & 1) ? 100
                                          : -100;
        const float value = (static_cast<float>(delta) * 0.17f + 2.5f) *
                            quantizer.dc_steps()[c];
        source.view().plane[c].Row(y)[x] = value;
        oracle.PlaneRow(c, y)[x] = value;
      }
    }
  }
  Check(
      gjxl::SmoothDcCoefficients(source.const_view(), quantizer, output.view())
          .ok(),
      "Native smoothing failed");
  Check(jxl::AdaptiveDCSmoothing(&manager, quantizer.dc_steps().data(), &oracle,
                                 nullptr),
        "Pinned smoothing failed");
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        if (output.const_view().plane[c].Row(y)[x] !=
            oracle.ConstPlaneRow(c, y)[x]) {
          std::cerr << std::hexfloat << "Smoothing mismatch at " << extent.width
                    << 'x' << extent.height << " c=" << c << " x=" << x
                    << " y=" << y << " pattern=" << pattern
                    << " expected=" << oracle.ConstPlaneRow(c, y)[x]
                    << " actual=" << output.const_view().plane[c].Row(y)[x]
                    << '\n';
          throw std::runtime_error(
              "Native smoothing differs from pinned libjxl");
        }
}
} // namespace

int main() {
  try {
    size_t quantization_cases = 0, smoothing_cases = 0;
    for (auto params : std::array<gjxl::QuantizerParams, 3>{
             {{1024, 64}, {799, 37}, {32768, 65536}}}) {
      gjxl::Quantizer quantizer;
      Check(gjxl::Quantizer::Create(params, &quantizer).ok(),
            "Quantizer failed");
      for (auto extent : std::array<gjxl::Extent2D, 8>{{{1, 1},
                                                        {1, 257},
                                                        {257, 1},
                                                        {2, 2},
                                                        {7, 13},
                                                        {255, 256},
                                                        {256, 257},
                                                        {259, 259}}}) {
        for (auto prediction : {gjxl::VarDctDcPrediction::kGradient,
                                gjxl::VarDctDcPrediction::kWeighted})
          for (unsigned pattern = 0; pattern < 6; ++pattern) {
            for (uint8_t precision : {1, 2, 3}) {
              QuantizationCase(extent, quantizer, prediction, pattern, precision);
              ++quantization_cases;
            }
          }
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
          SmoothingCase(extent, quantizer, pattern);
          ++smoothing_cases;
        }
      }
    }
    std::cout << quantization_cases << " quantization and " << smoothing_cases
              << " smoothing cases match pinned libjxl\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
