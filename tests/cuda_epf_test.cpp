// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "codec/epf.h"
#include "gpu/cuda/cuda_aq_exact_kernels.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu_test_utils.h"

namespace {

using DeviceImage = std::array<gjxl::test::GuardedDevicePlane, 3>;

bool Check(gjxl::Status status, const char* operation) {
  if (status.ok()) return true;
  std::cerr << operation << ": " << status.message() << '\n';
  return false;
}

bool CheckCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

float* Pointer(gjxl::test::GuardedDevicePlane& plane) {
  auto& buffer = static_cast<gjxl::cuda_internal::CudaBuffer&>(plane.buffer());
  return static_cast<float*>(buffer.pointer()) +
    plane.View().offset_bytes / sizeof(float);
}

bool Prepare(gjxl::GpuBackend& gpu, gjxl::Extent2D extent, size_t stride,
             DeviceImage* image) {
  for (size_t channel = 0; channel < image->size(); ++channel) {
    if (!Check((*image)[channel].Prepare(gpu, extent, stride, 7 + channel, 11),
          "Prepare guarded EPF plane")) {
      return false;
    }
    (*image)[channel].PoisonLogical();
    if (!Check((*image)[channel].Upload(), "Poison EPF plane")) return false;
  }
  return true;
}

bool CheckSequence(gjxl::GpuBackend& gpu, gjxl::Extent2D extent,
                   uint32_t iterations, bool custom_options) {
  const size_t count = extent.width * extent.height;
  std::array<std::vector<float>, 3> pixels;
  std::array<std::vector<float>, 3> expected;
  gjxl::ConstImage3FView input_view;
  gjxl::Image3FView expected_view;
  for (size_t channel = 0; channel < pixels.size(); ++channel) {
    pixels[channel].resize(count);
    expected[channel].resize(count);
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        pixels[channel][y * extent.width + x] =
          0.15f * static_cast<float>(channel) +
          0.01f * static_cast<float>(x % 17) +
          0.002f * static_cast<float>(y % 11) +
          0.0001f * static_cast<float>((x * 17 + y * 29 + channel * 13) % 37);
      }
    }
    input_view.plane[channel] = {pixels[channel].data(), extent, extent.width};
    expected_view.plane[channel] = {expected[channel].data(), extent, extent.width};
  }
  const gjxl::Extent2D sigma_extent{(extent.width + 7) / 8, (extent.height + 7) / 8};
  constexpr std::array<float, 5> kSigma = {
    -0.3f, -5.0f, -3.905242919921875f, -3.8f, -0.025f};
  std::vector<float> sigma(sigma_extent.width * sigma_extent.height);
  for (size_t i = 0; i < sigma.size(); ++i) sigma[i] = kSigma[i % kSigma.size()];
  gjxl::EpfFilterOptions options{.iterations = iterations};
  if (custom_options) {
    options.channel_scale = {0.0f, 4.0f, 2.2f};
    options.pass0_sigma_scale = 0.75f;
    options.pass2_sigma_scale = 4.5f;
    options.border_sad_multiplier = 0.625f;
  }
  if (!Check(gjxl::ApplyEpf(input_view,
        {sigma.data(), sigma_extent, sigma_extent.width}, options, expected_view),
        "CPU EPF reference")) {
    return false;
  }

  DeviceImage input;
  DeviceImage scratch_a;
  DeviceImage scratch_b;
  gjxl::test::GuardedDevicePlane device_sigma;
  std::unique_ptr<gjxl::DeviceBuffer> device_error;
  if (!Prepare(gpu, extent, extent.width + 3, &input) ||
      !Prepare(gpu, extent, extent.width + 7, &scratch_a) ||
      !Prepare(gpu, extent, extent.width + 11, &scratch_b) ||
      !Check(device_sigma.Prepare(gpu, sigma_extent, sigma_extent.width + 5),
        "Prepare EPF sigma") ||
      !Check(gpu.Allocate(sizeof(uint32_t), &device_error), "Allocate EPF error")) {
    return false;
  }
  for (size_t channel = 0; channel < input.size(); ++channel) {
    input[channel].SetLogical(pixels[channel]);
    if (!Check(input[channel].Upload(), "Upload EPF input")) return false;
  }
  device_sigma.SetLogical(sigma);
  uint32_t error = 0;
  if (!Check(device_sigma.Upload(), "Upload EPF sigma") ||
      !Check(gpu.CopyHostToDevice(*device_error, &error, sizeof(error)),
        "Initialize EPF error")) {
    return false;
  }
  auto* error_pointer = static_cast<uint32_t*>(
    static_cast<gjxl::cuda_internal::CudaBuffer&>(*device_error).pointer());
  DeviceImage* current = &input;
  const uint32_t first_pass = iterations == 3 ? 0 : 1;
  const uint32_t last_pass = iterations == 1 ? 1 : 2;
  for (uint32_t pass = first_pass; pass <= last_pass; ++pass) {
    DeviceImage* next = current == &scratch_a ? &scratch_b : &scratch_a;
    gjxl::cuda_internal::CudaAqEpfParams params{
      .width = static_cast<uint32_t>(extent.width),
      .height = static_cast<uint32_t>(extent.height),
      .input_stride = static_cast<uint32_t>((*current)[0].View().row_stride),
      .output_stride = static_cast<uint32_t>((*next)[0].View().row_stride),
      .inverse_sigma_stride = static_cast<uint32_t>(device_sigma.View().row_stride),
      .pass = pass,
      .sigma_scale = 1.65f * (pass == 0 ? options.pass0_sigma_scale :
        (pass == 2 ? options.pass2_sigma_scale : 1.0f)),
      .border_sad_multiplier = options.border_sad_multiplier,
      .channel_scale = {options.channel_scale[0], options.channel_scale[1],
        options.channel_scale[2]},
    };
    if (!CheckCuda(gjxl::cuda_internal::LaunchCudaAqEpf(
          {Pointer((*current)[0]), Pointer((*current)[1]), Pointer((*current)[2])},
          Pointer(device_sigma), {Pointer((*next)[0]), Pointer((*next)[1]),
            Pointer((*next)[2])}, error_pointer, params, nullptr), "Launch EPF") ||
        !CheckCuda(cudaDeviceSynchronize(), "Complete EPF")) {
      return false;
    }
    current = next;
  }
  if (!Check(gpu.CopyDeviceToHost(*device_error, &error, sizeof(error)),
        "Read EPF error") || error != 0) {
    std::cerr << "Unexpected EPF error flags: " << error << '\n';
    return false;
  }
  for (DeviceImage* image : {&input, &scratch_a, &scratch_b}) {
    for (auto& plane : *image) {
      if (!Check(plane.Download(), "Download EPF plane") || !plane.GuardsIntact()) {
        std::cerr << "EPF changed a plane guard\n";
        return false;
      }
    }
  }
  if (!Check(device_sigma.Download(), "Download EPF sigma") ||
      !device_sigma.GuardsIntact() || device_sigma.Logical() != sigma) {
    return false;
  }
  float maximum_error = 0.0f;
  for (size_t channel = 0; channel < current->size(); ++channel) {
    if (input[channel].Logical() != pixels[channel]) return false;
    const std::vector<float> actual = (*current)[channel].Logical();
    for (size_t i = 0; i < count; ++i) {
      const float difference = std::abs(actual[i] - expected[channel][i]);
      maximum_error = std::max(maximum_error, difference);
      if (!std::isfinite(actual[i]) ||
          difference > 2e-6f + 2e-6f * std::abs(expected[channel][i])) {
        std::cerr << "EPF reference mismatch at " << extent.width << 'x' << extent.height
                  << " iterations=" << iterations << " channel=" << channel
                  << " index=" << i << " difference=" << difference << '\n';
        return false;
      }
    }
  }
  std::cout << "EPF " << extent.width << 'x' << extent.height
            << " iterations=" << iterations << " custom=" << custom_options
            << " maximum error=" << maximum_error << '\n';
  return true;
}

// Filtering reports and clears non-finite output, but bypass is an exact
// copy (including NaN payloads). Mix bypass and active rows inside one tile
// so bypass must skip one pixel, not terminate the thread's remaining rows.
bool CheckNonFinite(gjxl::GpuBackend& gpu, float non_finite, bool bypass_all) {
  constexpr gjxl::Extent2D kExtent{33, 33};
  constexpr gjxl::Extent2D kSigmaExtent{5, 5};
  DeviceImage input;
  DeviceImage output;
  gjxl::test::GuardedDevicePlane sigma;
  std::unique_ptr<gjxl::DeviceBuffer> device_error;
  if (!Prepare(gpu, kExtent, 36, &input) ||
      !Prepare(gpu, kExtent, 40, &output) ||
      !Check(sigma.Prepare(gpu, kSigmaExtent, 8), "Prepare non-finite sigma") ||
      !Check(gpu.Allocate(sizeof(uint32_t), &device_error), "Allocate EPF error")) {
    return false;
  }
  const std::array<float, 3> values{non_finite, 0.25f, 0.5f};
  for (size_t channel = 0; channel < input.size(); ++channel) {
    input[channel].SetLogical(std::vector<float>(33 * 33, values[channel]));
    if (!Check(input[channel].Upload(), "Upload non-finite input")) return false;
  }
  std::vector<float> sigma_values(25);
  for (size_t y = 0; y < 5; ++y) {
    for (size_t x = 0; x < 5; ++x) {
      sigma_values[y * 5 + x] = !bypass_all && y % 2 ? -0.3f : -5.0f;
    }
  }
  sigma.SetLogical(sigma_values);
  if (!Check(sigma.Upload(), "Upload non-finite sigma")) return false;
  auto* error_pointer = static_cast<uint32_t*>(
    static_cast<gjxl::cuda_internal::CudaBuffer&>(*device_error).pointer());
  for (uint32_t pass = 0; pass <= 3; ++pass) {
    for (auto& plane : output) {
      plane.PoisonLogical();
      if (!Check(plane.Upload(), "Poison non-finite output")) return false;
    }
    uint32_t error = 8;
    if (!Check(gpu.CopyHostToDevice(*device_error, &error, sizeof(error)),
          "Initialize non-finite error")) return false;
    gjxl::cuda_internal::CudaAqEpfParams params{
      .width = 33, .height = 33, .input_stride = 36, .output_stride = 40,
      .inverse_sigma_stride = 8, .pass = pass, .sigma_scale = 1.65f,
      .border_sad_multiplier = 2.0f / 3.0f, .channel_scale = {40.0f, 5.0f, 3.5f},
    };
    const cudaError_t launch = gjxl::cuda_internal::LaunchCudaAqEpf(
      {Pointer(input[0]), Pointer(input[1]), Pointer(input[2])}, Pointer(sigma),
      {Pointer(output[0]), Pointer(output[1]), Pointer(output[2])},
      error_pointer, params, nullptr);
    if (pass == 3 ? launch != cudaErrorInvalidValue : launch != cudaSuccess) {
      std::cerr << "Unexpected EPF launch status for pass " << pass << '\n';
      return false;
    }
    if (!CheckCuda(cudaDeviceSynchronize(), "Complete non-finite EPF") ||
        !Check(gpu.CopyDeviceToHost(*device_error, &error, sizeof(error)),
          "Read non-finite error") || error != (pass == 3 || bypass_all ? 8u : 10u)) {
      return false;
    }
    for (size_t channel = 0; channel < output.size(); ++channel) {
      if (!Check(output[channel].Download(), "Read non-finite output") ||
          !output[channel].GuardsIntact()) return false;
      const auto actual = output[channel].Logical();
      for (size_t i = 0; i < actual.size(); ++i) {
        const bool bypass = bypass_all || (i / 33 / 8) % 2 == 0;
        const uint32_t expected_bits = pass == 3 ? 0x7fc00001u :
          std::bit_cast<uint32_t>(channel == 0 && !bypass ? 0.0f : values[channel]);
        if (std::bit_cast<uint32_t>(actual[i]) != expected_bits) {
          std::cerr << "EPF non-finite/bypass mismatch: pass=" << pass
                    << " channel=" << channel << " index=" << i << '\n';
          return false;
        }
      }
    }
  }
  for (size_t channel = 0; channel < input.size(); ++channel) {
    if (!Check(input[channel].Download(), "Read non-finite input") ||
        !input[channel].GuardsIntact()) return false;
    for (const float value : input[channel].Logical()) {
      if (std::bit_cast<uint32_t>(value) !=
          std::bit_cast<uint32_t>(values[channel])) return false;
    }
  }
  return Check(sigma.Download(), "Read non-finite sigma") &&
    sigma.GuardsIntact() && sigma.Logical() == sigma_values;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  const gjxl::Status status = gjxl::CreateCudaBackend(&gpu);
  if (status.code() == gjxl::StatusCode::kUnavailable) return 77;
  if (!Check(status, "Create CUDA EPF backend")) return EXIT_FAILURE;
  constexpr std::array<gjxl::Extent2D, 19> kExtents = {{{1, 1}, {1, 2}, {2, 1},
    {2, 2}, {3, 5}, {7, 9}, {8, 8}, {9, 7}, {31, 7}, {32, 8}, {33, 9},
    {63, 17}, {65, 33}, {129, 17}, {31, 31}, {32, 32}, {33, 33}, {1, 65}, {65, 1}}};
  for (const auto extent : kExtents) {
    for (uint32_t iterations = 1; iterations <= 3; ++iterations) {
      for (const bool custom : {false, true}) {
        if (!CheckSequence(*gpu, extent, iterations, custom)) return EXIT_FAILURE;
      }
    }
  }
  for (const float non_finite : {std::bit_cast<float>(0x7fc12345u),
         std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()}) {
    for (const bool bypass_all : {false, true}) {
      if (!CheckNonFinite(*gpu, non_finite, bypass_all)) return EXIT_FAILURE;
    }
  }
  std::cout << "All CUDA EPF tests passed." << std::endl;
  return EXIT_SUCCESS;
}
