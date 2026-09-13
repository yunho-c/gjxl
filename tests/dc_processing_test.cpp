// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "codec/dc_quantization.h"
#include "codec/dc_smoothing.h"
#include "core/managed_allocator.h"

namespace {
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Storage {
  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> floats;
  std::array<std::vector<int32_t>, 3> ints;
  explicit Storage(gjxl::Extent2D size) : extent(size), stride(size.width + 5) {
    for (auto &plane : floats)
      plane.assign(stride * extent.height, -901.0f);
    for (auto &plane : ints)
      plane.assign(stride * extent.height, -777);
  }
  gjxl::Image3FView FloatView() {
    gjxl::Image3FView result;
    for (size_t c = 0; c < 3; ++c)
      result.plane[c] = {floats[c].data(), extent, stride};
    return result;
  }
  gjxl::ConstImage3FView Input() const {
    gjxl::ConstImage3FView result;
    for (size_t c = 0; c < 3; ++c)
      result.plane[c] = {floats[c].data(), extent, stride};
    return result;
  }
  gjxl::Image3I32View IntView() {
    gjxl::Image3I32View result;
    for (size_t c = 0; c < 3; ++c)
      result.plane[c] = {ints[c].data(), extent, stride};
    return result;
  }
  void Pattern(const gjxl::Quantizer &quantizer) {
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x)
          floats[c][y * stride + x] =
              static_cast<float>(
                  static_cast<int>((x * 7 + y * 31 + c * 19) % 101) - 50) *
              quantizer.dc_steps()[c] * 0.17f;
  }
  bool PaddingIntact() const {
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = extent.width; x < stride; ++x)
          if (floats[c][y * stride + x] != -901.0f ||
              ints[c][y * stride + x] != -777)
            return false;
    return true;
  }
};
void QuantizationChecks(const gjxl::Quantizer &quantizer) {
  for (auto prediction : {gjxl::VarDctDcPrediction::kGradient,
                          gjxl::VarDctDcPrediction::kWeighted}) {
    const gjxl::DcQuantizationOptions options{
        gjxl::DcQuantizationMode::kPredictionAware, prediction, 1};
    Storage input({259, 259}), output(input.extent);
    input.Pattern(quantizer);
    auto run = [&] {
      Storage result(input.extent);
      Check(gjxl::QuantizeDcCoefficients(input.Input(), quantizer,
                                         {result.IntView(), result.FloatView()},
                                         options)
                .ok(),
            "Prediction-aware quantization failed");
      Check(result.PaddingIntact(), "Quantization changed padding");
      return result;
    };
    output = run();
    auto concurrent = std::async(std::launch::async, run);
    Storage in_place = input;
    Check(gjxl::QuantizeDcCoefficients(
              in_place.Input(), quantizer,
              {in_place.IntView(), in_place.FloatView()}, options)
              .ok(),
          "In-place quantization failed");
    Check(output.ints == in_place.ints && output.floats == in_place.floats &&
              concurrent.get().ints == output.ints,
          "Quantization aliases or shares state");

    // At a one-pixel group, both predictors guess zero. These points pin the
    // deadzone and the transition to even-integer residual quantization.
    for (auto [value, expected] :
         std::array<std::pair<float, int32_t>, 8>{{{-3.0f, -4},
                                                   {-2.5f, -2},
                                                   {-0.62f, -1},
                                                   {-0.6f, 0},
                                                   {0.6f, 0},
                                                   {0.62f, 1},
                                                   {2.5f, 2},
                                                   {3.0f, 4}}}) {
      Storage tiny({1, 1});
      for (size_t c = 0; c < 3; ++c)
        tiny.floats[c][0] = 0;
      tiny.floats[1][0] = value * quantizer.dc_steps()[1] * 0.5f;
      Check(gjxl::QuantizeDcCoefficients(tiny.Input(), quantizer,
                                         {tiny.IntView(), tiny.FloatView()},
                                         options)
                .ok(),
            "Deadzone case failed");
      Check(tiny.ints[1][0] == expected,
            "Wrong prediction-aware deadzone/even rounding");
    }
  }
}
void SmoothingChecks(const gjxl::Quantizer &quantizer) {
  for (auto extent : std::array<gjxl::Extent2D, 5>{
           {{1, 1}, {2, 17}, {17, 2}, {3, 3}, {259, 17}}}) {
    Storage input(extent), output(extent);
    input.Pattern(quantizer);
    Check(
        gjxl::SmoothDcCoefficients(input.Input(), quantizer, output.FloatView())
            .ok(),
        "Smoothing failed");
    Storage in_place = input;
    Check(gjxl::SmoothDcCoefficients(in_place.Input(), quantizer,
                                     in_place.FloatView())
              .ok(),
          "Aliased smoothing failed");
    Check(in_place.floats == output.floats && output.PaddingIntact(),
          "Smoothing alias/padding mismatch");
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x)
          if (x == 0 || y == 0 || x + 1 == extent.width ||
              y + 1 == extent.height)
            Check(output.floats[c][y * output.stride + x] ==
                      input.floats[c][y * input.stride + x],
                  "Smoothing changed border");
  }
  Storage input({3, 3}), output({3, 3});
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < 3; ++y)
      for (size_t x = 0; x < 3; ++x)
        input.floats[c][y * input.stride + x] = 0;
  const auto center = input.stride + 1;
  input.floats[1][center] = quantizer.dc_steps()[1] * 0.25f;
  Check(gjxl::SmoothDcCoefficients(input.Input(), quantizer, output.FloatView())
            .ok(),
        "Smooth impulse failed");
  Check(output.floats[1][center] < input.floats[1][center],
        "Smooth region was not filtered");
  input.floats[0][center] = quantizer.dc_steps()[0];
  Check(gjxl::SmoothDcCoefficients(input.Input(), quantizer, output.FloatView())
            .ok(),
        "Coupled impulse failed");
  Check(output.floats[1][center] == input.floats[1][center],
        "X edge did not protect Y from smoothing");
}
void FailureChecks(const gjxl::Quantizer &quantizer) {
  Storage input({259, 3}), output(input.extent);
  input.Pattern(quantizer);
  const auto original_floats = output.floats;
  const auto original_ints = output.ints;
  gjxl::DcQuantizationOptions options{
      gjxl::DcQuantizationMode::kPredictionAware,
      gjxl::VarDctDcPrediction::kWeighted, 1};
  for (bool smooth : {false, true}) {
    bool success = false;
    for (size_t failure = 0; failure < 64; ++failure) {
      gjxl::resource_budget_internal::
          ArmManagedHostAllocationFailureAfterForTest(failure);
      const auto status =
          smooth ? gjxl::SmoothDcCoefficients(input.Input(), quantizer,
                                              output.FloatView())
                 : gjxl::QuantizeDcCoefficients(
                       input.Input(), quantizer,
                       {output.IntView(), output.FloatView()}, options);
      gjxl::resource_budget_internal::
          DisarmManagedHostAllocationFailureForTest();
      if (status.ok()) {
        success = true;
        break;
      }
      Check(output.floats == original_floats && output.ints == original_ints,
            "Allocation failure changed output");
    }
    Check(success, "Allocation failure sweep never succeeded");
    output = Storage(input.extent);
    input.floats[2].at((input.extent.height - 1) * input.stride +
                       input.extent.width - 1) =
        std::numeric_limits<float>::quiet_NaN();
    const auto invalid =
        smooth ? gjxl::SmoothDcCoefficients(input.Input(), quantizer,
                                            output.FloatView())
               : gjxl::QuantizeDcCoefficients(
                     input.Input(), quantizer,
                     {output.IntView(), output.FloatView()}, options);
    Check(!invalid.ok() && output.floats == original_floats &&
              output.ints == original_ints,
          "Late invalid input changed output");
    input.Pattern(quantizer);
  }
  for (unsigned kind = 0; kind < 3; ++kind) {
    auto invalid = options;
    if (kind == 0)
      invalid.extra_dc_precision = 4;
    if (kind == 1)
      invalid.mode = static_cast<gjxl::DcQuantizationMode>(99);
    if (kind == 2)
      invalid.prediction = static_cast<gjxl::VarDctDcPrediction>(99);
    Check(!gjxl::QuantizeDcCoefficients(input.Input(), quantizer,
                                        {output.IntView(), output.FloatView()},
                                        invalid)
                  .ok() &&
              output.floats == original_floats && output.ints == original_ints,
          "Invalid DC mode changed output");
  }
}
} // namespace

int main() {
  try {
    gjxl::Quantizer quantizer;
    Check(gjxl::Quantizer::Create({1024, 64}, &quantizer).ok(),
          "Quantizer initialization failed");
    QuantizationChecks(quantizer);
    SmoothingChecks(quantizer);
    FailureChecks(quantizer);
    std::cout
        << "DC prediction-aware quantization and smoothing checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
