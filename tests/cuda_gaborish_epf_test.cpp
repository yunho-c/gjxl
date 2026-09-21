// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/cuda/cuda_aq_exact_kernels.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu_test_utils.h"

namespace {

using gjxl::cuda_internal::CudaAqGaborishParams;
using gjxl::cuda_internal::CudaAqEpfParams;
using gjxl::cuda_internal::LaunchCudaAqEpf;
using gjxl::cuda_internal::LaunchCudaAqGaborishEpf;
using gjxl::cuda_internal::LaunchCudaAqGaborish;
using Plane = gjxl::test::GuardedDevicePlane;
using Image = std::array<Plane, 3>;
using Pixels = std::array<std::vector<float>, 3>;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void Check(gjxl::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

bool Equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
    std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

float* Pointer(Plane& plane) {
  auto& buffer = static_cast<gjxl::cuda_internal::CudaBuffer&>(plane.buffer());
  return static_cast<float*>(buffer.pointer()) +
    plane.View().offset_bytes / sizeof(float);
}

std::array<float*, 3> Pointers(Image& image) {
  return {Pointer(image[0]), Pointer(image[1]), Pointer(image[2])};
}

std::array<const float*, 3> ConstPointers(Image& image) {
  return {Pointer(image[0]), Pointer(image[1]), Pointer(image[2])};
}

class Fixture {
public:
  Fixture(gjxl::GpuBackend& gpu, gjxl::Extent2D extent, uint32_t padding)
      : extent_(extent), sigma_extent_{(extent.width + 7) / 8,
                                      (extent.height + 7) / 8} {
    const uint32_t width = static_cast<uint32_t>(extent.width);
    const uint32_t height = static_cast<uint32_t>(extent.height);
    const uint32_t coding_stride = (width + 7) / 8 * 8 + padding;
    gaborish_ = {width, height, coding_stride, coding_stride + 3};
    epf_ = {width, height, gaborish_.output_stride, width + padding,
            static_cast<uint32_t>(sigma_extent_.width) + padding, 1,
            1.65f, 0.81f, {40.0f, 5.0f, 3.5f}};
    for (size_t channel = 0; channel < 3; ++channel) {
      Check(input_[channel].Prepare(gpu, extent, gaborish_.input_stride, 19 + channel, 37));
      Check(scratch_[channel].Prepare(gpu, extent, gaborish_.output_stride, 13 + channel, 29));
      Check(output_[channel].Prepare(gpu, extent, epf_.output_stride, 7 + channel, 11));
    }
    Check(sigma_.Prepare(gpu, sigma_extent_, epf_.inverse_sigma_stride));
    Check(error_.Prepare(gpu, {1, 1}, 1));
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~Fixture() { if (stream_) (void)cudaStreamDestroy(stream_); }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  size_t Run(uint32_t pattern, bool custom) {
    for (uint32_t channel = 0; channel < 3; ++channel) {
      const float axis = custom ? 0.071f + 0.011f * channel : 0.115169525f;
      const float diagonal = custom ? 0.039f + 0.007f * channel : 0.061248592f;
      const float normalization = 1.0f / (1.0f + 4.0f * axis + 4.0f * diagonal);
      gaborish_.center_weight[channel] = normalization;
      gaborish_.axis_weight[channel] = axis * normalization;
      gaborish_.diagonal_weight[channel] = diagonal * normalization;
    }
    // Reuse allocations with changed inputs; a bypass must continue to later
    // rows handled by the same thread, not return from the tile loop.
    for (uint32_t stage = 0; stage < 3; ++stage) {
      Fill(pattern, stage);
      Pixels reference;
      uint32_t reference_flags = 0;
      for (bool fused : {false, true}) {
        Reset();
        auto* error = reinterpret_cast<uint32_t*>(Pointer(error_));
        if (fused) {
          auto gaborish = gaborish_;
          auto epf = epf_;
          // No global intermediate exists: these two strides are unused.
          gaborish.output_stride = epf.input_stride = 0;
          Check(LaunchCudaAqGaborishEpf(ConstPointers(input_), Pointer(sigma_),
              Pointers(output_), error, gaborish, epf, stream_));
        } else {
          Check(LaunchCudaAqGaborish(ConstPointers(input_), Pointers(scratch_),
              error, gaborish_, stream_));
          Check(LaunchCudaAqEpf(ConstPointers(scratch_), Pointer(sigma_),
              Pointers(output_), error, epf_, stream_));
        }
        Check(cudaStreamSynchronize(stream_));
        Check(error_.Download());
        Require(error_.GuardsIntact(), "Error flag guard changed");
        const uint32_t flags = std::bit_cast<uint32_t>(error_.Logical()[0]);
        if (!fused) reference_flags = flags;
        Require(flags == reference_flags && (flags & 256u), "Error flags differ");
        // Gaborish sanitizes the nonfinite channel before EPF, including
        // sigma-bypassed pixels. The flag must not become an Gaborish/EPF flag.
        if ((pattern >= 6 && pattern <= 8) || pattern == 10)
          Require(flags == 257u, "Nonfinite Gaborish input must set bit 1 only");
        for (size_t channel = 0; channel < 3; ++channel) {
          Check(output_[channel].Download());
          Require(output_[channel].GuardsIntact(), "Filtered XYB guard changed");
          const auto actual = output_[channel].Logical();
          if (!fused) reference[channel] = actual;
          Require(Equal(actual, reference[channel]), "Filtered XYB is not bitwise equal");
          Check(scratch_[channel].Download());
          Require(scratch_[channel].GuardsIntact(), "Gaborish guard changed");
          if (fused) {
            for (float value : scratch_[channel].Logical())
              Require(std::bit_cast<uint32_t>(value) == 0x7fc00001u,
                      "Fused path wrote the unused Gaborish intermediate");
          }
          Check(input_[channel].Download());
          Require(input_[channel].GuardsIntact() &&
                  Equal(input_[channel].Logical(), pixels_[channel]), "Input changed");
        }
        Check(sigma_.Download());
        Require(sigma_.GuardsIntact() && Equal(sigma_.Logical(), sigmas_),
                "Inverse sigma changed");
      }
    }
    return 6;
  }

  void CheckArguments() {
    Reset();
    auto inputs = ConstPointers(input_);
    auto outputs = Pointers(output_);
    auto* error = reinterpret_cast<uint32_t*>(Pointer(error_));
    for (unsigned variant = 0; variant < 18; ++variant) {
      auto epf = epf_;
      auto gaborish = gaborish_;
      auto in = inputs;
      auto out = outputs;
      const float* sigma = Pointer(sigma_);
      auto* flags = error;
      switch (variant) {
        case 0: epf.pass = 0; break;
        case 1: epf.pass = 2; break;
        case 2: epf.pass = 3; break;
        case 3: ++gaborish.width; break;
        case 4: ++gaborish.height; break;
        case 5: gaborish.input_stride = epf.width - 1; break;
        case 6: epf.output_stride = epf.width - 1; break;
        case 7: epf.inverse_sigma_stride = 0; break;
        case 8: case 9: case 10: in[variant - 8] = nullptr; break;
        case 11: case 12: case 13: out[variant - 11] = nullptr; break;
        case 14: sigma = nullptr; break;
        case 15: flags = nullptr; break;
        case 16:
          epf.width = gaborish.width = 0xffffffffu;
          epf.height = gaborish.height = 0xffffffffu;
          gaborish.input_stride = epf.output_stride = 0xffffffffu;
          epf.inverse_sigma_stride = 0x20000000u;
          break;
        case 17:
          epf.width = 0;
          break;
      }
      Require(LaunchCudaAqGaborishEpf(in, sigma, out, flags, gaborish, epf,
                                     stream_) == cudaErrorInvalidValue,
              "Invalid fused arguments accepted");
    }
    for (bool empty_width : {false, true}) {
      auto epf = epf_;
      auto gaborish = gaborish_;
      if (empty_width) epf.width = gaborish.width = 0;
      else epf.height = gaborish.height = 0;
      Check(LaunchCudaAqGaborishEpf({}, nullptr, {}, nullptr, gaborish, epf, stream_));
    }
    Check(cudaStreamSynchronize(stream_));
    Check(error_.Download());
    Require(error_.GuardsIntact() &&
            std::bit_cast<uint32_t>(error_.Logical()[0]) == 256u,
            "Invalid or empty launch modified error flags");
    for (auto& plane : output_) {
      Check(plane.Download());
      Require(plane.GuardsIntact(), "Invalid launch changed guards");
      for (float value : plane.Logical())
        Require(std::bit_cast<uint32_t>(value) == 0x7fc00001u,
                "Invalid or empty launch wrote output");
    }
  }

private:
  void Fill(uint32_t pattern, uint32_t stage) {
    uint32_t rng = 125001 + pattern * 1297 + stage * 167 +
      static_cast<uint32_t>(extent_.width * 13 + extent_.height);
    for (size_t c = 0; c < 3; ++c) {
      pixels_[c].resize(extent_.width * extent_.height);
      for (size_t y = 0; y < extent_.height; ++y) {
        for (size_t x = 0; x < extent_.width; ++x) {
          rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
          float value = (static_cast<int>(rng & 65535) - 16384) / 65536.0f;
          switch (pattern) {
            case 0: value = (x + y + c + stage) % 2 ? -0.0f : 0.0f; break;
            case 2: value = static_cast<float>((x * 3 + y * 5 + c + stage) % 1024) / 2048.0f; break;
            case 3: value = (x + y + stage) % 2 ? 0.9f : -0.3f; break;
            case 4: value *= 4096.0f; break;
            case 5: value = std::bit_cast<float>(rng & 0x807fffffu); break;
            case 6: case 10: if (c == 0) value = std::bit_cast<float>(0x7fc12345u); break;
            case 7: if (c == 0) value = std::numeric_limits<float>::infinity(); break;
            case 8: if (c == 0) value = -std::numeric_limits<float>::infinity(); break;
            case 9: value = std::numeric_limits<float>::max() * ((x + y + c) % 2 ? 1.0f : -1.0f); break;
            case 15: value = std::bit_cast<float>((rng & 0x807fffffu) | (((rng >> 24) % 254 + 1) << 23)); break;
          }
          pixels_[c][y * extent_.width + x] = value;
        }
      }
      input_[c].SetLogical(pixels_[c]);
      Check(input_[c].Upload());
    }
    sigmas_.resize(sigma_extent_.width * sigma_extent_.height);
    for (size_t y = 0; y < sigma_extent_.height; ++y) {
      for (size_t x = 0; x < sigma_extent_.width; ++x) {
        float value = -0.02f - static_cast<float>((x + 3 * y + stage) % 11) * 0.025f;
        if (pattern == 10) value = -5.0f;
        if (pattern == 11) {
          constexpr float kThreshold = -3.905242919921875f;
          const float infinity = std::numeric_limits<float>::infinity();
          value = (x + y + stage) % 3 == 0 ? kThreshold :
            std::nextafter(kThreshold, (x + y + stage) % 3 == 1 ? -infinity : infinity);
        }
        if (pattern == 12 && (y + stage) % 2 == 0) value = -5.0f;
        if (pattern == 13 && (x + y) % 2 == 0) value = std::bit_cast<float>(0x7fc12345u);
        if (pattern == 14) value = 0.3f;
        sigmas_[y * sigma_extent_.width + x] = value;
      }
    }
    sigma_.SetLogical(sigmas_);
    Check(sigma_.Upload());
  }

  void Reset() {
    for (Image* image : {&scratch_, &output_}) {
      for (auto& plane : *image) {
        plane.PoisonLogical();
        Check(plane.Upload());
      }
    }
    error_.SetLogical(std::array{std::bit_cast<float>(256u)});
    Check(error_.Upload());
    // Backend uploads complete before returning. Kernels use a separate,
    // nonblocking stream and must complete before backend downloads begin.
  }

  gjxl::Extent2D extent_, sigma_extent_;
  Image input_, scratch_, output_;
  Plane sigma_, error_;
  Pixels pixels_;
  std::vector<float> sigmas_;
  CudaAqEpfParams epf_;
  CudaAqGaborishParams gaborish_;
  cudaStream_t stream_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const bool sanitizer = argc == 2 && std::string_view(argv[1]) == "--sanitizer";
    Require(argc == 1 || sanitizer, "Usage: cuda_gaborish_epf_test [--sanitizer]");
    std::unique_ptr<gjxl::GpuBackend> gpu;
    const auto status = gjxl::CreateCudaBackend(&gpu);
    if (status.code() == gjxl::StatusCode::kUnavailable) return 77;
    Check(status);
    const std::vector<gjxl::Extent2D> extents = sanitizer ?
      std::vector<gjxl::Extent2D>{{1, 1}, {33, 33}, {65, 17}} :
      std::vector<gjxl::Extent2D>{{1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 5},
        {7, 9}, {8, 8}, {9, 7}, {31, 7}, {32, 8}, {33, 9}, {63, 17},
        {65, 33}, {129, 17}, {31, 31}, {32, 32}, {33, 33}, {1, 65}, {65, 1},
        {127, 95}, {128, 96}, {129, 97}, {257, 67}, {4096, 3}};
    size_t checks = 0;
    for (const auto extent : extents) {
      for (uint32_t padding : {0u, 7u}) {
        Fixture fixture(*gpu, extent, padding);
        fixture.CheckArguments();
        for (uint32_t pattern = 0; pattern < 16; ++pattern) {
          for (bool custom : {false, true}) {
            try { checks += fixture.Run(pattern, custom); }
            catch (...) {
              std::cerr << "extent=" << extent.width << 'x' << extent.height
                        << " padding=" << padding << " pattern=" << pattern
                        << " custom=" << custom << '\n';
              throw;
            }
          }
        }
      }
      std::cout << "Gaborish/EPF " << extent.width << 'x' << extent.height
                << " pipeline checks=" << checks << std::endl;
    }
    std::cout << "All CUDA Gaborish/EPF tests passed. Pipeline checks=" << checks << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "CUDA Gaborish/EPF test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
