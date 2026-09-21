// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/dc_quantization.h"
#include "codec/dc_smoothing.h"
#include "gpu/cuda/cuda_dc_processing_kernels.h"
#include <array>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::cuda_internal;
void Check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
void Check(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
template <class T> struct Device {
  T *data = nullptr;
  explicit Device(size_t count) { Check(cudaMalloc(&data, count * sizeof(T))); }
  ~Device() { cudaFree(data); }
};
template <class T> Image3View<T> View(std::vector<T> &data, Extent2D e) {
  const size_t n = e.width * e.height;
  return {{PlaneView<T>{data.data(), e, e.width},
           PlaneView<T>{data.data() + n, e, e.width},
           PlaneView<T>{data.data() + 2 * n, e, e.width}}};
}
ConstImage3FView ConstView(const std::vector<float> &data, Extent2D e) {
  const size_t n = e.width * e.height;
  return {{ConstPlaneF32View{data.data(), e, e.width},
           ConstPlaneF32View{data.data() + n, e, e.width},
           ConstPlaneF32View{data.data() + 2 * n, e, e.width}}};
}
void Run(Extent2D e, QuantizerParams qp, DcQuantizationOptions options,
         unsigned pattern, bool resident) {
  const size_t count = 3 * e.width * e.height;
  Quantizer quantizer;
  Check(Quantizer::Create(qp, &quantizer));
  std::mt19937 random(391 + unsigned(e.width * 97 + e.height));
  std::vector<float> input(count), reconstructed(count), smoothed(count),
      result(count);
  std::vector<int32_t> integers(count), actual(count);
  const auto steps = quantizer.dc_steps();
  for (size_t i = 0; i < count; ++i) {
    const size_t channel = i / (e.width * e.height);
    const float step = steps[channel] / float(1u << options.extra_dc_precision);
    float value = float(int(random() % 4001) - 2000) * step;
    if (pattern == 0)
      value += step * (float(random() % 1001) / 1000.0f - 0.5f);
    if (pattern == 1)
      value = step * (float(int(i % 53) - 26) + 0.5f);
    if (pattern == 2)
      value = step * (2.0f + 0.01f * float(i % e.width));
    input[i] = value;
  }
  Check(QuantizeDcCoefficients(ConstView(input, e), quantizer,
                               {View(integers, e), View(reconstructed, e)},
                               options));
  Check(SmoothDcCoefficients(ConstView(reconstructed, e), quantizer,
                             View(smoothed, e)));
  Device<float> dc(count), smooth(count);
  Device<int32_t> quantized(count);
  Device<unsigned> error(1), quant(2);
  const unsigned params[2] = {qp.global_scale, qp.quant_dc};
  Check(cudaMemcpy(dc.data, input.data(), count * sizeof(float),
                   cudaMemcpyHostToDevice));
  Check(cudaMemcpy(quant.data, params, sizeof(params), cudaMemcpyHostToDevice));
  Check(cudaMemset(error.data, 0, sizeof(unsigned)));
  CudaDcProcessingParams p{unsigned(e.width),
                           unsigned(e.height),
                           qp.global_scale,
                           qp.quant_dc,
                           unsigned(options.mode),
                           unsigned(options.prediction),
                           options.extra_dc_precision,
                           unsigned(resident)};
  Check(LaunchCudaDcQuantization(dc.data, quantized.data, error.data, p,
                                 resident ? quant.data : nullptr, nullptr));
  Check(cudaMemcpy(actual.data(), quantized.data, count * sizeof(int32_t),
                   cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(result.data(), dc.data, count * sizeof(float),
                   cudaMemcpyDeviceToHost));
  unsigned errors = 0;
  Check(
      cudaMemcpy(&errors, error.data, sizeof(errors), cudaMemcpyDeviceToHost));
  if (errors)
    throw std::runtime_error("device quantization error");
  for (size_t i = 0; i < count; ++i) {
    if (actual[i] != integers[i] || result[i] != reconstructed[i]) {
      std::cerr << "DC mismatch " << e.width << 'x' << e.height
                << " mode=" << unsigned(options.mode)
                << " predictor=" << unsigned(options.prediction)
                << " precision=" << unsigned(options.extra_dc_precision)
                << " pattern=" << pattern << " index=" << i << " integers "
                << actual[i] << '/' << integers[i] << " reconstructed "
                << result[i] << '/' << reconstructed[i] << '\n';
      throw std::runtime_error("CPU/CUDA DC mismatch");
    }
  }
  Check(LaunchCudaDcSmoothing(dc.data, smooth.data, error.data, p,
                              resident ? quant.data : nullptr, nullptr));
  Check(cudaMemcpy(result.data(), smooth.data, count * sizeof(float),
                   cudaMemcpyDeviceToHost));
  Check(
      cudaMemcpy(&errors, error.data, sizeof(errors), cudaMemcpyDeviceToHost));
  if (errors)
    throw std::runtime_error("device smoothing error");
  for (size_t i = 0; i < count; ++i)
    if (result[i] != smoothed[i])
      throw std::runtime_error("CPU/CUDA smoothing mismatch");
  // Non-finite samples must be reported for both prediction variants.
  input[count / 2] = std::numeric_limits<float>::quiet_NaN();
  Check(cudaMemcpy(dc.data, input.data(), count * sizeof(float),
                   cudaMemcpyHostToDevice));
  Check(LaunchCudaDcQuantization(dc.data, quantized.data, error.data, p,
                                 resident ? quant.data : nullptr, nullptr));
  Check(
      cudaMemcpy(&errors, error.data, sizeof(errors), cudaMemcpyDeviceToHost));
  if (!errors)
    throw std::runtime_error("non-finite DC accepted");
}
} // namespace
int main(int argc, char **argv) {
  // A completion file lets sanitizer jobs prove that the instrumented child
  // reached the assertions even when its console handles are not inherited.
  const char *report = nullptr;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--report" && i + 1 < argc)
      report = argv[++i];
    else if (std::string_view(argv[i]) == "--quick")
      quick = true;
    else {
      std::cerr << "Usage: cuda_dc_processing_test [--quick] [--report PATH]\n";
      return 1;
    }
  }
  int devices = 0;
  const auto available = cudaGetDeviceCount(&devices);
  if (available != cudaSuccess || devices == 0) {
    std::cout << "CUDA DC test skipped: " << cudaGetErrorString(available)
              << ", devices=" << devices << '\n';
    return 77;
  }
  try {
    size_t cases = 0;
    const std::vector<gjxl::Extent2D> shapes =
        quick ? std::vector<gjxl::Extent2D>{{19, 33}, {257, 4}, {4, 257}}
              : std::vector<gjxl::Extent2D>{{1, 1},   {1, 257},   {257, 1},
                                            {19, 33}, {256, 256}, {257, 259},
                                            {513, 4}};
    for (auto e : shapes)
      for (auto qp :
           std::array<gjxl::QuantizerParams, 2>{{{1024, 64}, {731, 53}}})
        for (unsigned mode = 0; mode < 3; ++mode)
          for (uint8_t precision : {uint8_t(0), uint8_t(1), uint8_t(3)})
            for (unsigned pattern = 0; pattern < 3; ++pattern) {
              if (quick &&
                  (qp.global_scale != 731 || precision == 0 || pattern != 1))
                continue;
              Run(e, qp,
                  {mode ? gjxl::DcQuantizationMode::kPredictionAware
                        : gjxl::DcQuantizationMode::kRound,
                   mode == 2 ? gjxl::VarDctDcPrediction::kWeighted
                             : gjxl::VarDctDcPrediction::kGradient,
                   precision},
                  pattern, (cases & 1) != 0);
              ++cases;
            }
    std::cout << cases
              << " exact CPU/CUDA DC quantization, reconstruction and "
                 "smoothing cases passed.\n";
    if (report != nullptr) {
      std::ofstream completion(report);
      completion << "PASS cases=" << cases << '\n';
      completion.close();
      if (!completion)
        throw std::runtime_error("cannot write completion report");
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
