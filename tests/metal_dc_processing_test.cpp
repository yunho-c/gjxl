// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Raw-DC shader oracle, independent of forward-transform differences.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "codec/dc_quantization.h"
#include "codec/dc_smoothing.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/metal/metal_dc_processing_internal.h"

namespace gjxl::metal_internal {
struct MetalDcProcessingTestAccess {
  struct Invocation {
    AqDcProcessingParams params;
    bool quantize;
    std::unique_ptr<DeviceBuffer> dc, integers, scratch, filtered, error,
        resident;
  };
  static void Encode(MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
                     const void *context) {
    const auto &invocation = *static_cast<const Invocation *>(context);
    const auto bind = [&](const auto &buffer, size_t index) {
      encoder->setBuffer(static_cast<MetalBuffer &>(*buffer).handle(), 0,
                         index);
    };
    if (invocation.quantize) {
      const bool simd_wave = invocation.params.quantization_mode == 1u &&
        invocation.params.predictor == 1u &&
        backend.aq_pipelines_.dc_quantize_simd_wave->threadExecutionWidth() == 32;
      encoder->setComputePipelineState(simd_wave
        ? backend.aq_pipelines_.dc_quantize_simd_wave.get()
        : backend.aq_pipelines_.dc_quantize.get());
      bind(invocation.dc, 0);
      bind(invocation.integers, 1);
      bind(invocation.scratch, 2);
      bind(invocation.error, 3);
      encoder->setBytes(&invocation.params, sizeof(invocation.params), 4);
      bind(invocation.resident, 5);
      const size_t groups = ((invocation.params.width + 255) / 256) *
                            ((invocation.params.height + 255) / 256);
      const MTL::Size threads(simd_wave ? std::min<size_t>(256, (invocation.params.height + 31) / 32 * 32) : std::min<size_t>(256, invocation.params.height), 1, 1);
      uint32_t channel_base = 0;
      encoder->setBytes(&channel_base, sizeof(channel_base), 6);
      DispatchMetalThreadgroups(encoder, MTL::Size(groups, 2, 1), threads);
      channel_base = 2;
      encoder->setBytes(&channel_base, sizeof(channel_base), 6);
      DispatchMetalThreadgroups(encoder, MTL::Size(groups, 1, 1), threads);
    }
    encoder->setComputePipelineState(backend.aq_pipelines_.dc_smooth.get());
    bind(invocation.dc, 0);
    bind(invocation.filtered, 1);
    bind(invocation.error, 2);
    encoder->setBytes(&invocation.params, sizeof(invocation.params), 3);
    bind(invocation.resident, 4);
    DispatchMetalThreads(
        encoder,
        MTL::Size(invocation.params.width * invocation.params.height, 1, 1),
        MTL::Size(64, 1, 1));
  }
  static Status Run(MetalBackend &backend, Extent2D extent,
                    const Quantizer &quantizer, DcQuantizationOptions options,
                    const std::vector<float> &input, bool resident,
                    bool quantize, std::vector<int32_t> *integers,
                    std::vector<float> *dc, std::vector<float> *smoothed) {
    const size_t area = extent.width * extent.height;
    Invocation invocation;
    invocation.quantize = quantize;
    invocation.params = {uint32_t(extent.width),
                         uint32_t(extent.height),
                         resident ? 0u : quantizer.params().global_scale,
                         resident ? 0u : quantizer.params().quant_dc,
                         uint32_t(options.mode),
                         uint32_t(options.prediction),
                         options.extra_dc_precision,
                         resident ? 1u : 0u};
    const size_t groups =
        ((extent.width + 255) / 256) * ((extent.height + 255) / 256);
    const size_t scratch_values =
        groups * 5 * std::min<size_t>(256, extent.width) *
        std::min<size_t>(256, extent.height);
    for (auto [buffer, bytes] :
         std::array<std::pair<std::unique_ptr<DeviceBuffer> *, size_t>, 6>{{
             {&invocation.dc, 3 * area * sizeof(float)},
             {&invocation.integers, 3 * area * sizeof(int32_t)},
             {&invocation.scratch, scratch_values * sizeof(uint32_t)},
             {&invocation.filtered, 3 * area * sizeof(float)},
             {&invocation.error, sizeof(uint32_t)},
             {&invocation.resident, 2 * sizeof(uint32_t)},
         }}) {
      auto status = backend.Allocate(bytes, buffer);
      if (!status.ok())
        return status;
    }
    const std::array<uint32_t, 2> quantizer_words = {
        quantizer.params().global_scale, quantizer.params().quant_dc};
    const uint32_t zero = 0;
    auto status = backend.CopyHostToDevice(*invocation.dc, input.data(),
                                           input.size() * sizeof(float), 0);
    if (status.ok())
      status =
          backend.CopyHostToDevice(*invocation.error, &zero, sizeof(zero), 0);
    if (status.ok())
      status =
          backend.CopyHostToDevice(*invocation.resident, quantizer_words.data(),
                                   sizeof(quantizer_words), 0);
    if (!status.ok())
      return status;
    std::unique_ptr<GpuSubmission> submission;
    status = backend.SubmitCompute("DC primitive oracle", Encode, &invocation,
                                   &submission);
    if (!status.ok())
      return status;
    status = submission->Wait();
    if (!status.ok())
      return status;
    uint32_t error = 0;
    status =
        backend.CopyDeviceToHost(*invocation.error, &error, sizeof(error), 0);
    if (!status.ok())
      return status;
    if (error)
      return Status::DeviceError("DC shader rejected arithmetic");
    std::vector<int32_t> q(3 * area);
    std::vector<float> raw(3 * area), filtered(3 * area);
    if (quantize)
      status = backend.CopyDeviceToHost(*invocation.integers, q.data(),
                                        q.size() * sizeof(int32_t), 0);
    if (status.ok())
      status = backend.CopyDeviceToHost(*invocation.dc, raw.data(),
                                        raw.size() * sizeof(float), 0);
    if (status.ok())
      status = backend.CopyDeviceToHost(*invocation.filtered, filtered.data(),
                                        filtered.size() * sizeof(float), 0);
    if (!status.ok())
      return status;
    *integers = std::move(q);
    *dc = std::move(raw);
    *smoothed = std::move(filtered);
    return Status::Ok();
  }
};
} // namespace gjxl::metal_internal

namespace {
using namespace gjxl;
void Check(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
template <typename T>
Image3View<T> View(std::vector<T> &values, Extent2D extent) {
  const size_t area = extent.width * extent.height;
  Image3View<T> view;
  for (size_t c = 0; c < 3; ++c)
    view.plane[c] = {values.data() + c * area, extent, extent.width};
  return view;
}
ConstImage3FView Input(const std::vector<float> &values, Extent2D extent) {
  const size_t area = extent.width * extent.height;
  ConstImage3FView view;
  for (size_t c = 0; c < 3; ++c)
    view.plane[c] = {values.data() + c * area, extent, extent.width};
  return view;
}
void Equal(const std::vector<float> &expected, const std::vector<float> &actual,
           const char *operation, Extent2D extent) {
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      std::cerr << operation << " mismatch " << extent.width << 'x'
                << extent.height << " index=" << i << std::hexfloat
                << " expected=" << expected[i] << " actual=" << actual[i]
                << '\n';
      throw std::runtime_error("DC shader differs from CPU oracle");
    }
  }
}
void Case(metal_internal::MetalBackend &backend, Extent2D extent,
          const Quantizer &quantizer, DcQuantizationOptions options,
          bool resident, bool quantize, unsigned pattern) {
  const size_t area = extent.width * extent.height;
  std::vector<float> source(3 * area), expected(3 * area), filtered(3 * area);
  std::vector<int32_t> q(3 * area);
  uint32_t random = 1729;
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x) {
        random = random * 1664525 + 1013904223;
        const int value = pattern == 0 ? 0
                          : pattern == 1
                              ? int((x * 11 + y * 3 + c * 7) % 97) - 48
                          : pattern == 2      ? int((random >> 16) % 17) - 8
                          : pattern == 4      ? int(random >> 16) - 32768
                          : ((x + y + c) & 1) ? 1023
                                              : -1024;
        source[c * area + y * extent.width + x] =
            (float(value) * 0.17f + 2.5f) * quantizer.dc_steps()[c];
      }
  if (quantize)
    Check(QuantizeDcCoefficients(Input(source, extent), quantizer,
                                 {View(q, extent), View(expected, extent)},
                                 options));
  else
    expected = source;
  Check(SmoothDcCoefficients(Input(expected, extent), quantizer,
                             View(filtered, extent)));
  std::vector<int32_t> actual_q;
  std::vector<float> actual_dc, actual_filtered;
  Check(metal_internal::MetalDcProcessingTestAccess::Run(
      backend, extent, quantizer, options, source, resident, quantize,
      &actual_q, &actual_dc, &actual_filtered));
  if (quantize && q != actual_q) {
    const auto mismatch = std::mismatch(q.begin(), q.end(), actual_q.begin());
    const size_t i = size_t(mismatch.first - q.begin());
    std::cerr << "DC integer mismatch " << extent.width << 'x' << extent.height
              << " index=" << i << " expected=" << q[i]
              << " actual=" << actual_q[i] << " mode=" << unsigned(options.mode)
              << " predictor=" << unsigned(options.prediction)
              << " precision=" << unsigned(options.extra_dc_precision)
              << " resident=" << resident
              << " global_scale=" << quantizer.params().global_scale
              << " quant_dc=" << quantizer.params().quant_dc << '\n';
    throw std::runtime_error("DC shader integers differ from CPU oracle");
  }
  Equal(expected, actual_dc, "Dequantization", extent);
  Equal(filtered, actual_filtered, "Smoothing", extent);
}
} // namespace

int main() {
  try {
    std::unique_ptr<GpuBackend> gpu;
    Check(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    auto &backend = static_cast<metal_internal::MetalBackend &>(*gpu);
    size_t count = 0;
    for (auto params : std::array<QuantizerParams, 3>{
             {{1024, 64}, {799, 37}, {32768, 65536}}}) {
      Quantizer quantizer;
      Check(Quantizer::Create(params, &quantizer));
      for (auto extent : std::array<Extent2D, 11>{
               {{1, 1}, {7, 13}, {1, 257}, {257, 1}, {259, 259},
                {31, 31}, {32, 32}, {33, 33}, {63, 65}, {255, 256}, {257, 65}}}) {
        for (unsigned mode = 0; mode < 3; ++mode) {
          DcQuantizationOptions options{
              mode == 0 ? DcQuantizationMode::kRound
                        : DcQuantizationMode::kPredictionAware,
              mode == 2 ? VarDctDcPrediction::kWeighted
                        : VarDctDcPrediction::kGradient,
              1};
          for (uint8_t precision : {1, 2, 3}) {
            options.extra_dc_precision = precision;
            for (bool resident : {false, true}) {
              for (unsigned pattern = 0; pattern < 5; ++pattern) {
                Case(backend, extent, quantizer, options, resident, true, pattern);
                ++count;
              }
            }
          }
        }
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
          Case(backend, extent, quantizer, {}, true, false, pattern);
          ++count;
        }
      }
    }
    Quantizer quantizer;
    Check(Quantizer::Create({1024, 64}, &quantizer));
    size_t rejected = 0;
    for (bool quantize : {false, true}) {
      for (size_t location : {size_t{0}, size_t{12}, size_t{74}}) {
        for (float value : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
          std::vector<float> input(75, 0.1f), dc{123}, filtered{456};
          std::vector<int32_t> integers{789};
          input[location] = value;
          const auto status = metal_internal::MetalDcProcessingTestAccess::Run(
              backend, {5, 5}, quantizer,
              {DcQuantizationMode::kPredictionAware,
               VarDctDcPrediction::kWeighted, 1},
              input, true, quantize, &integers, &dc, &filtered);
          if (status.code() != StatusCode::kDeviceError ||
              dc != std::vector<float>{123} ||
              filtered != std::vector<float>{456} ||
              integers != std::vector<int32_t>{789})
            throw std::runtime_error(
                "Invalid DC shader input escaped atomic rejection");
          ++rejected;
        }
      }
    }
    for (auto predictor :
         {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted}) {
      std::vector<float> input(75, 1e30f), dc{123}, filtered{456};
      std::vector<int32_t> integers{789};
      const auto status = metal_internal::MetalDcProcessingTestAccess::Run(
          backend, {5, 5}, quantizer,
          {DcQuantizationMode::kPredictionAware, predictor, 1}, input, false,
          true, &integers, &dc, &filtered);
      if (status.code() != StatusCode::kDeviceError ||
          dc != std::vector<float>{123} ||
          filtered != std::vector<float>{456} ||
          integers != std::vector<int32_t>{789})
        throw std::runtime_error(
            "DC integer overflow escaped atomic rejection");
      ++rejected;
    }
    std::cout << count << " raw-DC Metal cases match CPU exactly; " << rejected
              << " invalid cases reject atomically\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
