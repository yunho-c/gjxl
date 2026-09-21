// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec/color_transform_internal.h"
#include "codec/dct.h"
#include "codec/dct_internal.h"
#include "codec/quantization_pipeline.h"
#include "codestream/encoder.h"
#include "codestream/workflow_internal.h"
#include "core/frame_geometry.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_aq_cpu_order_kernels.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/ops/quantization_pipeline.h"
#include "io/pfm.h"

namespace {
using namespace gjxl;
using namespace gjxl::cuda_internal;
void Check(Status s) {
  if (!s.ok()) throw std::runtime_error(std::string(s.message()));
}
void CheckCuda(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class T>
struct DeviceArray {
  explicit DeviceArray(const std::vector<T>& values) : count(values.size()) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    const auto s = cudaMemcpy(data, values.data(), count * sizeof(T),
                              cudaMemcpyHostToDevice);
    if (s != cudaSuccess) {
      (void)cudaFree(data);
      CheckCuda(s);
    }
  }
  ~DeviceArray() { (void)cudaFree(data); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  std::vector<T> Read() const {
    std::vector<T> values(count);
    CheckCuda(cudaMemcpy(values.data(), data, count * sizeof(T),
                         cudaMemcpyDeviceToHost));
    return values;
  }
  T* data = nullptr;
  size_t count;
};

size_t CheckDct() {
  std::vector<double> basis;
  for (size_t length : {8u, 16u, 32u}) {
    const auto values = dct_internal::InverseBasis(length);
    basis.insert(basis.end(), values.begin(), values.end());
  }
  Require(basis.size() == kCudaCpuOrderBasisCount,
          "Exact DCT basis size changed");
  DeviceArray<double> device_basis(basis);
  size_t cases = 0;
  for (auto strategy : {AcStrategyType::kDct8, AcStrategyType::kDct16x8,
                        AcStrategyType::kDct8x16, AcStrategyType::kDct16x16,
                        AcStrategyType::kDct32x16, AcStrategyType::kDct16x32,
                        AcStrategyType::kDct32x32}) {
    const auto extent = GetAcStrategyInfo(strategy)->pixel_extent();
    const size_t area = extent.width * extent.height;
    for (size_t count : {1u, 3u, 17u}) {
      constexpr size_t guard = 13;
      constexpr float poison = -9876.25f;
      std::vector<float> input(guard + count * area + guard, poison);
      auto expected = input;
      for (size_t i = 0; i < count * area; ++i) {
        const size_t transform = i / area;
        // DC-only, impulse, cancellation, tiny and large signed coefficients.
        const float value =
            float(int((i * 104729 + 13) % 65521) - 32760) / 32760.f;
        input[guard + i] = transform % 5 == 0 ? (i % area == 0 ? .75f : 0.f)
                           : transform % 5 == 1
                               ? (i % area == 19 ? -1.f : 0.f)
                               : value * (transform % 5 == 2   ? 1.f
                                          : transform % 5 == 3 ? 1.e-6f
                                                               : 1.e3f);
      }
      for (size_t i = 0; i < count; ++i)
        Check(InverseDctCpu(
            strategy,
            std::span<const float>(input).subspan(guard + i * area, area),
            std::span<float>(expected).subspan(guard + i * area, area)));
      DeviceArray<float> device_input(input),
          device_output(std::vector<float>(input.size(), poison));
      CheckCuda(LaunchCudaCpuOrderInverseDct(
          device_input.data + guard, device_output.data + guard, count,
          unsigned(extent.width), unsigned(extent.height), device_basis.data,
          nullptr));
      const auto actual = device_output.Read();
      Require(std::memcmp(actual.data(), expected.data(),
                          actual.size() * sizeof(float)) == 0,
              "Exact CUDA inverse DCT differs from CPU or overwrote a guard");
      ++cases;
    }
  }
  return cases;
}

struct Output {
  Extent2D b, p;
  std::vector<float> initial, mask, pixels, quant, block;
  Image3FBuffer rgb;
  VarDctEncoderFrame frame;
  std::vector<double> scores;
  Output(Extent2D source, Extent2D padded)
      : b{padded.width / 8, padded.height / 8},
        p(padded),
        initial(b.width * b.height),
        mask(initial.size()),
        pixels(p.width * p.height),
        quant(initial.size()),
        block(initial.size()),
        rgb(source) {}
  CpuQuantizationPipelineOutput view() {
    return {.initial_quantization = {{initial.data(), b, b.width},
                                     {mask.data(), b, b.width},
                                     {pixels.data(), p, p.width}},
            .adaptive_quantization = {
                .quant_field = {quant.data(), b, b.width},
                .block_distance_map = {block.data(), b, b.width},
                .reconstructed_linear_rgb = rgb.view(),
                .frame = &frame,
                .score_history = &scores}};
  }
};

void CheckPipeline(GpuBackend& gpu, ConstImage3FView source, size_t iterations,
                   bool new_dc, unsigned filter) {
  const Extent2D extent = source.plane[0].extent;
  FrameGeometry geometry;
  Check(FrameGeometry::Create(extent, &geometry));
  Image3FBuffer opsin(geometry.padded_frame());
  Check(color_transform_internal::LinearRgbToPaddedOpsin(source, 255.f,
                                                         opsin.view()));
  CpuQuantizationPipelineOptions options;
  options.butteraugli_target = 1.2f;
  auto& aq = options.adaptive_quantization;
  aq.iterations = iterations;
  if (new_dc) {
    aq.dc_quantization = DcQuantizationMode::kPredictionAware;
    aq.dc_prediction = VarDctDcPrediction::kWeighted;
    aq.profile.extra_dc_precision = 1;
    aq.profile.adaptive_dc_smoothing = true;
  }
  aq.profile.loop_filter.gaborish = filter != 0;
  codestream_internal::QuantizationMatrixScaleStats stats;
  codestream_internal::QuantizationMatrixScales scales;
  Check(codestream_internal::ComputeQuantizationMatrixScaleStats(
      opsin.cropped_view(extent), &stats));
  Check(codestream_internal::SelectQuantizationMatrixScales(
      stats, VarDctRateControlMode::kButteraugliTarget, 1.2f, &scales));
  aq.profile.x_qm_scale = scales.x;
  aq.profile.b_qm_scale = scales.b;
  Output cpu(extent, geometry.padded_frame()),
      cuda(extent, geometry.padded_frame());
  Check(RunCpuQuantizationPipeline(source, opsin.const_view(), options,
                                   cpu.view()));
  Check(RunGpuQuantizationPipeline(
      gpu, source, opsin.const_view(), options,
      GpuAdaptiveQuantizationMode::kExactCoefficients, cuda.view()));
  std::vector<uint8_t> cpu_bytes, cuda_bytes;
  Check(EncodeVarDctCodestream(cpu.frame, &cpu_bytes));
  Check(EncodeVarDctCodestream(cuda.frame, &cuda_bytes));
  if (cpu_bytes != cuda_bytes) {
    std::cerr << "Byte mismatch: " << extent.width << 'x' << extent.height
              << " updates=" << iterations << " dc=" << new_dc
              << " filter=" << filter << '\n';
    throw std::runtime_error("Exact CUDA codestream differs from CPU");
  }
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      Require(std::memcmp(cpu.rgb.const_view().plane[c].Row(y),
                          cuda.rgb.const_view().plane[c].Row(y),
                          extent.width * sizeof(float)) == 0,
              "Exact CUDA reconstructed pixels differ from CPU");
  for (size_t i = 0; i < cpu.block.size(); ++i)
    Require(std::isfinite(cuda.block[i]) &&
                std::abs(cpu.block[i] - cuda.block[i]) <= 2.e-5f,
            "Exact CUDA metric feedback differs from CPU");
}

}  // namespace

int main(int argc, char** argv) try {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  const auto status = gjxl::CreateCudaBackend(&gpu);
  if (status.code() == gjxl::StatusCode::kUnavailable) return 77;
  Check(status);
  const size_t dct_cases = CheckDct();
  size_t pipeline_cases = 0;
  for (auto extent :
       {Extent2D{1, 1}, {7, 33}, {33, 7}, {16, 16}, {35, 65}, {129, 97}}) {
    for (bool new_dc : {false, true}) {
      Image3FBuffer source(extent);
      for (size_t c = 0; c < 3; ++c)
        for (size_t y = 0; y < extent.height; ++y)
          for (size_t x = 0; x < extent.width; ++x) {
            const uint32_t bits = uint32_t((x + 7 * y + 31 * c + 13) * 104729u);
            source.view().plane[c].Row(y)[x] =
                0.1f + 0.5f * float(x) / float(extent.width) +
                float(bits % 257) / 1024.f;
          }
      for (size_t iterations : {0u, 2u, 4u}) {
        CheckPipeline(*gpu, source.const_view(), iterations, new_dc,
                      unsigned((pipeline_cases / 3) % 4));
        ++pipeline_cases;
      }
    }
  }
  const bool quick = argc > 1 && std::string_view(argv[1]) == "--quick";
  if (!quick) {
    Image3FBuffer image;
    const char* input = argc > 2 && std::string_view(argv[1]) == "--photo"
                            ? argv[2]
                            : GJXL_SAMPLE_PFM_PATH;
    Check(io::ReadPfm(input, &image));
    const auto source = image.const_view();
    for (size_t iterations = 0; iterations <= 4; ++iterations) {
      CheckPipeline(*gpu, source, iterations, true, 1);
      ++pipeline_cases;
    }
  }
  const int report_index = quick ? 2 : 3;
  if (argc > report_index) {
    std::ofstream report(argv[report_index]);
    report << "PASS CUDA exact arithmetic\n";
    report.close();
    Require(bool(report),
            "Cannot write CUDA exact arithmetic completion marker");
  }
  std::cout << "CUDA exact arithmetic: " << dct_cases << " DCT cases, "
            << pipeline_cases << " CPU byte/pixel pipeline pairs passed.\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
