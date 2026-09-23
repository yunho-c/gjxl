// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/cuda_butteraugli_kernels.h"

namespace {
using namespace gjxl::cuda_internal;
std::string context;
constexpr float kPoison = std::bit_cast<float>(uint32_t{0x7fc12345});

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

void Equal(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size())
    throw std::runtime_error("Diagnostic vector size differs");
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) {
      std::cerr << context << " index=" << i << " legacy=0x" << std::hex
                << std::bit_cast<uint32_t>(a[i]) << " candidate=0x"
                << std::bit_cast<uint32_t>(b[i]) << std::dec << '\n'
                << std::flush;
      throw std::runtime_error("Direct short bitwise mismatch");
    }
  }
}

struct Plane {
  uint32_t width, height, stride;
  size_t offset;
  std::vector<float> initial;
  float* allocation = nullptr;
  Plane(uint32_t w, uint32_t h, uint32_t padding = 0, size_t prefix = 7)
      : width(w),
        height(h),
        stride(w + padding),
        offset(prefix),
        initial(prefix + size_t{stride} * h + 29, kPoison) {
    Check(cudaMalloc(reinterpret_cast<void**>(&allocation),
                     initial.size() * sizeof(float)));
    Reset();
  }
  ~Plane() { (void)cudaFree(allocation); }
  Plane(const Plane&) = delete;
  Plane& operator=(const Plane&) = delete;
  float* Data() const { return allocation + offset; }
  void Write(const std::vector<float>& values) {
    if (values.size() != initial.size())
      throw std::runtime_error("Fixture size differs");
    Check(cudaMemcpy(allocation, values.data(), values.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    // Pageable H2D copies may return after staging. The test launches its
    // kernels on a nonblocking stream, so make fixture initialization complete.
    Check(cudaStreamSynchronize(nullptr));
  }
  void Reset() { Write(initial); }
  std::vector<float> Read() const {
    std::vector<float> values(initial.size());
    Check(cudaMemcpy(values.data(), allocation, values.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    return values;
  }
  void Guards() const {
    const auto values = Read();
    for (size_t i = 0; i < values.size(); ++i) {
      const bool logical = i >= offset &&
                           i < offset + size_t{stride} * height &&
                           (i - offset) % stride < width;
      if (!logical && std::memcmp(&values[i], &initial[i], sizeof(float)))
        throw std::runtime_error("Direct short fixture guard changed");
    }
  }
  void Finite() const {
    const auto values = Read();
    for (uint32_t y = 0; y < height; ++y)
      for (uint32_t x = 0; x < width; ++x)
        if (!std::isfinite(values[offset + size_t{y} * stride + x]))
          throw std::runtime_error(
              "Prepared Direct short comparison produced nonfinite output");
  }
};

void RawCase(uint32_t width, uint32_t height, bool padded, unsigned pattern,
             unsigned channel, unsigned tile, cudaStream_t stream) {
  const unsigned taps = channel < 3 ? 15 : 7;
  Plane input(width, height, padded ? 3 : 0, 3), weights(taps, 1, 0, 5);
  Plane expected_low(width, height, padded ? 7 : 0, 11);
  Plane expected_high(width, height, padded ? 11 : 0, 13);
  Plane actual_low(width, height, padded ? 7 : 0, 11);
  Plane actual_high(width, height, padded ? 11 : 0, 13);
  Plane horizontal(width, height, 0, 17), blurred(width, height, 0, 19);
  std::mt19937 rng(7717u + width + height * 31 + pattern);
  std::uniform_real_distribution<float> random(-3, 3);
  constexpr std::array<uint32_t, 12> special{
      0,          0x80000000, 1,          0x80000001, 0x00800000, 0x80800000,
      0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321};
  constexpr std::array<float, 6> thresholds{
      0.29f, 0.1f, 0.04f, 1.5f, 28.4691806922f, 5.19175294647f};
  for (unsigned tap = 0; tap < taps; ++tap) {
    const float delta = static_cast<float>(tap) - taps / 2;
    float weight = std::exp(-0.13f * delta * delta);
    if (pattern == 2 || pattern == 3) weight = tap == taps / 2 ? 1.0f : 0.0f;
    if (pattern == 5 && (tap == 0 || tap == taps - 1))
      weight = tap == 0 ? kPoison : std::numeric_limits<float>::infinity();
    if (pattern == 6)
      weight = tap == taps / 2 ? 1.25f : tap % 3 ? 0.0625f : -0.03125f;
    weights.initial[weights.offset + tap] = weight;
  }
  for (uint32_t y = 0; y < height; ++y)
    for (uint32_t x = 0; x < width; ++x) {
      float value = random(rng);
      if (pattern == 0) value = (x + y) % 2 ? -0.0f : 0.0f;
      if (pattern == 2 || pattern == 3) {
        value = thresholds[(x + y) % thresholds.size()] *
                ((x + y) % 2 ? -1.0f : 1.0f);
        if (pattern == 3)
          value = std::nextafter(value, x % 2 ? -INFINITY : INFINITY);
      }
      if (pattern == 4)
        value = std::bit_cast<float>(special[(x + y) % special.size()]);
      if (pattern == 7) {
        value = -0.0f;
        if ((x % 32 == 31 || x == width - 1) &&
            (y % 16 == 15 || y == height - 1))
          value = x % 2 ? 1.0f : kPoison;
      }
      input.initial[input.offset + size_t{y} * input.stride + x] = value;
    }
  weights.Reset();
  const CudaButteraugliDirectShortPlan oracle{
      input.Data(),
      weights.Data(),
      expected_low.Data(),
      channel == 2 ? nullptr : expected_high.Data(),
      width,
      height,
      input.stride,
      expected_low.stride,
      channel == 2 ? 0 : expected_high.stride,
      channel};
  auto candidate = oracle;
  candidate.low = actual_low.Data();
  candidate.high = channel == 2 ? nullptr : actual_high.Data();
  std::vector<float> first_low, first_high;
  for (unsigned reuse = 0; reuse < 3; ++reuse) {
    context = "raw " + std::to_string(width) + "x" + std::to_string(height) +
              " padded=" + std::to_string(padded) +
              " pattern=" + std::to_string(pattern) +
              " channel=" + std::to_string(channel) +
              " tile=" + std::to_string(tile) +
              " reuse=" + std::to_string(reuse);
    auto values = input.initial;
    if (reuse == 1)
      for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
          values[input.offset + size_t{y} * input.stride + x] *= -0.75f;
    input.Write(values);
    for (auto* plane : {&expected_low, &expected_high, &actual_low,
                        &actual_high, &horizontal, &blurred})
      plane->Reset();
    Check(LaunchCudaButteraugliDirectShortReferenceForTesting(
        oracle, horizontal.Data(), blurred.Data(), stream));
    Check(tile ? LaunchCudaButteraugliDirectShortForTesting(candidate, tile,
                                                            stream)
               : LaunchCudaButteraugliDirectShort(candidate, stream));
    Check(cudaStreamSynchronize(stream));
    Equal(expected_low.Read(), actual_low.Read());
    Equal(expected_high.Read(), actual_high.Read());
    Equal(values, input.Read());
    Equal(weights.initial, weights.Read());
    for (const auto* plane : {&input, &weights, &expected_low, &expected_high,
                              &actual_low, &actual_high, &horizontal, &blurred})
      plane->Guards();
    if (channel == 2) Equal(actual_high.initial, actual_high.Read());
    if (reuse == 0) {
      first_low = actual_low.Read();
      first_high = actual_high.Read();
    }
    if (reuse == 2) {
      Equal(first_low, actual_low.Read());
      Equal(first_high, actual_high.Read());
    }
  }
}

size_t RejectMalformed(cudaStream_t stream) {
  for (bool zero_width : {false, true}) {
    CudaButteraugliDirectShortPlan empty;
    empty.width = zero_width ? 0 : 17;
    empty.height = zero_width ? 17 : 0;
    Check(LaunchCudaButteraugliDirectShort(empty, stream));
    Check(LaunchCudaButteraugliDirectShortReferenceForTesting(empty, nullptr,
                                                              nullptr, stream));
    for (unsigned tile : {16u, 32u, 64u}) {
      const auto expected =
          tile == 32 || CudaButteraugliDirectShortTestSchedulesAvailable()
              ? cudaSuccess
              : cudaErrorInvalidValue;
      if (LaunchCudaButteraugliDirectShortForTesting(empty, tile, stream) !=
          expected)
        throw std::runtime_error(
            "Empty direct short schedule availability differs");
    }
  }
  Plane input(17, 19), weights(15, 1), low(17, 19), high(17, 19);
  Plane horizontal(17, 19), blurred(17, 19);
  const CudaButteraugliDirectShortPlan valid{input.Data(), weights.Data(),
                                             low.Data(),   high.Data(),
                                             17,           19,
                                             17,           17,
                                             17,           0};
  size_t rejected = 0;
  for (unsigned error = 0; error < 18; ++error) {
    auto bad = valid;
    switch (error) {
      case 0:
        bad.input = nullptr;
        break;
      case 1:
        bad.weights = nullptr;
        break;
      case 2:
        bad.low = nullptr;
        break;
      case 3:
        bad.high = nullptr;
        break;
      case 4:
        bad.channel = 5;
        break;
      case 5:
        bad.input_stride = 16;
        break;
      case 6:
        bad.low_stride = 16;
        break;
      case 7:
        bad.high_stride = 16;
        break;
      case 8:
        bad.low = input.Data();
        break;
      case 9:
        bad.high = low.Data() + 1;
        break;
      case 10:
        bad.weights = low.Data() + 1;
        break;
      case 11:
        bad.input = low.Data() + 1;
        break;
      case 12:
        bad.width = 0x80000000u;
        break;
      case 13:
        bad.height = 0x80000000u;
        break;
      case 14:
        bad.low = reinterpret_cast<float*>(UINTPTR_MAX - 3);
        break;
      case 15:
        bad.high = reinterpret_cast<float*>(UINTPTR_MAX - 3);
        break;
      case 16:
        bad.weights = reinterpret_cast<const float*>(UINTPTR_MAX - 3);
        break;
      case 17:
        bad.width = bad.height = 0x7fffff7fu;
        bad.input_stride = bad.low_stride = bad.high_stride = bad.width;
        break;
    }
    if (LaunchCudaButteraugliDirectShort(bad, stream) !=
            cudaErrorInvalidValue ||
        LaunchCudaButteraugliDirectShortReferenceForTesting(
            bad, horizontal.Data(), blurred.Data(), stream) !=
            cudaErrorInvalidValue)
      throw std::runtime_error("Malformed direct short plan accepted");
    for (unsigned tile : {16u, 32u, 64u})
      if (LaunchCudaButteraugliDirectShortForTesting(bad, tile, stream) !=
          cudaErrorInvalidValue)
        throw std::runtime_error("Malformed forced direct short plan accepted");
    ++rejected;
  }
  for (unsigned tile : {0u, 15u, 48u, 65u}) {
    if (LaunchCudaButteraugliDirectShortForTesting(valid, tile, stream) !=
        cudaErrorInvalidValue)
      throw std::runtime_error("Malformed direct short tile accepted");
    ++rejected;
  }
  if (!CudaButteraugliDirectShortTestSchedulesAvailable()) {
    const auto before_low = low.Read();
    const auto before_high = high.Read();
    for (unsigned tile : {16u, 64u}) {
      if (LaunchCudaButteraugliDirectShortForTesting(valid, tile, stream) !=
          cudaErrorInvalidValue)
        throw std::runtime_error("Unavailable direct short schedule accepted");
      Equal(before_low, low.Read());
      Equal(before_high, high.Read());
      ++rejected;
    }
  }
  for (unsigned error = 0; error < 5; ++error) {
    float* h = horizontal.Data();
    float* b = blurred.Data();
    if (error == 0) h = nullptr;
    if (error == 1) b = nullptr;
    if (error == 2) h = input.Data();
    if (error == 3) b = low.Data() + 1;
    if (error == 4) b = h + 1;
    if (LaunchCudaButteraugliDirectShortReferenceForTesting(
            valid, h, b, stream) != cudaErrorInvalidValue)
      throw std::runtime_error(
          "Malformed direct short oracle scratch accepted");
    ++rejected;
  }
  Check(cudaStreamSynchronize(stream));
  for (const auto* plane :
       {&input, &weights, &low, &high, &horizontal, &blurred})
    Equal(plane->initial, plane->Read());
  return rejected;
}
struct PreparedCase {
  CudaButteraugliPlan plan;
  std::vector<std::unique_ptr<Plane>> storage;
  std::array<Plane*, 3> input, distorted;
  Plane* output;
  Plane* score;
  std::vector<std::pair<Plane*, std::vector<float>>> cached;

  Plane& Add(uint32_t w, uint32_t h, uint32_t padding = 0, size_t prefix = 7) {
    storage.push_back(std::make_unique<Plane>(w, h, padding, prefix));
    return *storage.back();
  }
  PreparedCase(uint32_t width, uint32_t height, bool padded, unsigned profile,
               bool cpu_order, bool fused) {
    plan.width = width;
    plan.height = height;
    plan.working_width = std::max(8u, width);
    plan.working_height = std::max(8u, height);
    plan.expanded = width < 8 || height < 8;
    plan.xborder = width < 8 ? (8 - width) / 2 : 0;
    plan.yborder = height < 8 ? (8 - height) / 2 : 0;
    plan.multiscale = !plan.expanded && width >= 15 && height >= 15;
    plan.sub_width = (width + 1) / 2;
    plan.sub_height = (height + 1) / 2;
    plan.hf_asymmetry = profile == 0 ? 1.0f : (profile == 1 ? 0.6f : 2.5f);
    plan.x_multiplier = profile == 0 ? 1.0f : 0.73f;
    plan.intensity_target = profile == 2 ? 120.0f : 255.0f;
    plan.cpu_order = cpu_order;
    // Exercise the production default unless this is the legacy oracle.
    if (!fused) plan.direct_short = 0;
    for (size_t c = 0; c < 3; ++c) {
      input[c] =
          &Add(width, height, padded ? 3 + static_cast<uint32_t>(c) : 0, 3 + c);
      distorted[c] = &Add(width, height,
                          padded ? 11 + static_cast<uint32_t>(c) : 0, 9 + c);
      for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
          const float value =
              0.05f + 0.1f * static_cast<float>(c) +
              0.003f * static_cast<float>((x * 13 + y * 7) % 53);
          input[c]
              ->initial[input[c]->offset + size_t{y} * input[c]->stride + x] =
              value;
          distorted[c]->initial[distorted[c]->offset +
                                size_t{y} * distorted[c]->stride + x] =
              value + 0.015f * std::sin(static_cast<float>(x * 3 + y * 5 + c));
        }
      input[c]->Reset();
      distorted[c]->Reset();
      plan.reference[c] = input[c]->Data();
      plan.reference_stride[c] = input[c]->stride;
    }
    for (auto& plane : plan.planes)
      plane = Add(plan.working_width, plan.working_height).Data();
    if (plan.multiscale)
      for (auto& plane : plan.reference_sub)
        plane = Add(plan.sub_width, plan.sub_height).Data();
    for (auto& plane : plan.reduction)
      plane = Add((width * height + 255) / 256, 1).Data();
    constexpr std::array<float, 5> sigmas{1.2f, 7.15593339443f, 3.22489901262f,
                                          1.56416327805f, 2.7f};
    constexpr std::array<uint32_t, 5> taps{5, 33, 15, 7, 13};
    for (size_t i = 0; i < taps.size(); ++i) {
      auto& weights = Add(taps[i], 1, 0, 5);
      const double scale = -1.0 / (2.0 * sigmas[i] * sigmas[i]);
      for (uint32_t tap = 0; tap < taps[i]; ++tap) {
        const int delta = static_cast<int>(tap) - static_cast<int>(taps[i] / 2);
        const float value = static_cast<float>(std::exp(scale * delta * delta));
        weights.initial[weights.offset + tap] = value;
        if (i == 1) plan.low_medium_weights.taps[tap] = value;
      }
      weights.Reset();
      plan.kernels[i] = weights.Data();
    }
    output = &Add(width, height, padded ? 19 : 0, 13);
    score = &Add(1, 1, 0, 5);
  }
  void Prepare(cudaStream_t stream) {
    Check(LaunchCudaButteraugliPrepare(plan, stream));
  }
  void SnapshotCaches() {
    for (const auto& plane : storage) {
      bool reference_plane = false;
      for (size_t i = 0; i < 10; ++i)
        reference_plane |= plane->Data() == plan.planes[i] ||
                           plane->Data() == plan.reference_sub[i];
      reference_plane |= plane->Data() == plan.planes[20];
      if (reference_plane) cached.emplace_back(plane.get(), plane->Read());
    }
  }
  void Compare(unsigned reuse, cudaStream_t stream) {
    for (size_t c = 0; c < distorted.size(); ++c) {
      auto values = distorted[c]->initial;
      if (reuse == 1)
        for (uint32_t y = 0; y < plan.height; ++y)
          for (uint32_t x = 0; x < plan.width; ++x)
            values[distorted[c]->offset + size_t{y} * distorted[c]->stride +
                   x] *= 0.93f;
      distorted[c]->Write(values);
    }
    output->Reset();
    score->Reset();
    Check(LaunchCudaButteraugliCompare(
        plan,
        {distorted[0]->Data(), distorted[1]->Data(), distorted[2]->Data()},
        {distorted[0]->stride, distorted[1]->stride, distorted[2]->stride},
        output->Data(), output->stride, score->Data(), stream));
  }
  void Guards() const {
    for (const auto& plane : storage) plane->Guards();
    for (const auto* plane : input) Equal(plane->initial, plane->Read());
    for (const auto& [plane, values] : cached) Equal(values, plane->Read());
    output->Finite();
    score->Finite();
  }
};

void ParentCase(uint32_t width, uint32_t height, bool padded, unsigned profile,
                bool cpu_order, cudaStream_t stream, unsigned tile = 32) {
  PreparedCase legacy(width, height, padded, profile, cpu_order, false);
  PreparedCase fused(width, height, padded, profile, cpu_order, true);
  if (tile != 32) fused.plan.direct_short = tile;
  legacy.Prepare(stream);
  fused.Prepare(stream);
  Check(cudaStreamSynchronize(stream));
  legacy.SnapshotCaches();
  fused.SnapshotCaches();
  std::vector<float> first_map, first_score;
  for (unsigned reuse = 0; reuse < 3; ++reuse) {
    legacy.Compare(reuse, stream);
    fused.Compare(reuse, stream);
    Check(cudaStreamSynchronize(stream));
    context = "prepared " + std::to_string(width) + "x" +
              std::to_string(height) + " padded=" + std::to_string(padded) +
              " profile=" + std::to_string(profile) +
              " tile=" + std::to_string(tile) +
              " cpu_order=" + std::to_string(cpu_order) +
              " reuse=" + std::to_string(reuse);
    Equal(legacy.output->Read(), fused.output->Read());
    Equal(legacy.score->Read(), fused.score->Read());
    legacy.Guards();
    fused.Guards();
    if (reuse == 0) {
      first_map = fused.output->Read();
      first_score = fused.score->Read();
    }
    if (reuse == 2) {
      Equal(first_map, fused.output->Read());
      Equal(first_score, fused.score->Read());
    }
  }
}

void Benchmark(cudaStream_t stream) {
  const std::vector<unsigned> variants =
      CudaButteraugliDirectShortTestSchedulesAvailable()
          ? std::vector<unsigned>{0, 1, 2, 3, 4}
          : std::vector<unsigned>{0, 1, 3};
  const unsigned variant_count = static_cast<unsigned>(variants.size());
  cudaEvent_t start{}, stop{};
  Check(cudaEventCreate(&start));
  Check(cudaEventCreate(&stop));
  std::cout << "width,height,phase,round,variant,iterations,gpu_us,wall_us\n"
            << std::setprecision(10);
  for (const auto shape :
       std::array<std::array<uint32_t, 2>, 6>{{{7, 11},
                                               {33, 65},
                                               {768, 512},
                                               {1920, 1080},
                                               {2048, 1641},
                                               {3840, 2160}}}) {
    PreparedCase legacy(shape[0], shape[1], false, 0, false, false);
    PreparedCase fused(shape[0], shape[1], false, 0, false, true);
    legacy.Prepare(stream);
    fused.Prepare(stream);
    legacy.Compare(0, stream);
    fused.Compare(0, stream);
    Check(cudaStreamSynchronize(stream));
    Equal(legacy.output->Read(), fused.output->Read());
    Equal(legacy.score->Read(), fused.score->Read());
    for (bool prepare : {true, false}) {
      for (unsigned round = 0; round < 2 * variant_count; ++round) {
        for (unsigned position = 0; position < variant_count; ++position) {
          const unsigned rotated =
              (position + round % variant_count) % variant_count;
          const unsigned variant =
              variants[round < variant_count ? rotated
                                             : variant_count - 1 - rotated];
          const bool candidate = variant >= 2;
          constexpr unsigned tiles[] = {0, 0, 16, 32, 64};
          constexpr const char* labels[] = {"legacy", "control", "direct16",
                                            "direct32", "direct64"};
          fused.plan.direct_short = tiles[variant];
          auto& fixture = candidate ? fused : legacy;
          const auto launch = [&] {
            if (prepare) {
              fixture.Prepare(stream);
            } else {
              Check(LaunchCudaButteraugliCompare(
                  fixture.plan,
                  {fixture.distorted[0]->Data(), fixture.distorted[1]->Data(),
                   fixture.distorted[2]->Data()},
                  {fixture.distorted[0]->stride, fixture.distorted[1]->stride,
                   fixture.distorted[2]->stride},
                  fixture.output->Data(), fixture.output->stride,
                  fixture.score->Data(), stream));
            }
          };
          for (unsigned warm = 0; warm < 5; ++warm) launch();
          Check(cudaStreamSynchronize(stream));
          const unsigned iterations =
              uint64_t{shape[0]} * shape[1] < 100000 ? 200 : 20;
          const auto begin = std::chrono::steady_clock::now();
          Check(cudaEventRecord(start, stream));
          for (unsigned i = 0; i < iterations; ++i) launch();
          Check(cudaEventRecord(stop, stream));
          Check(cudaEventSynchronize(stop));
          const auto end = std::chrono::steady_clock::now();
          float milliseconds = 0;
          Check(cudaEventElapsedTime(&milliseconds, start, stop));
          const double wall =
              std::chrono::duration<double, std::micro>(end - begin).count();
          std::cout << shape[0] << ',' << shape[1] << ','
                    << (prepare ? "prepare" : "compare") << ',' << round << ','
                    << labels[variant] << ',' << iterations << ','
                    << milliseconds * 1000.0 / iterations << ','
                    << wall / iterations << '\n'
                    << std::flush;
        }
      }
    }
    // Timed reference preparation must still leave a usable cached reference.
    legacy.Compare(0, stream);
    fused.Compare(0, stream);
    Check(cudaStreamSynchronize(stream));
    Equal(legacy.output->Read(), fused.output->Read());
    Equal(legacy.score->Read(), fused.score->Read());
    legacy.Guards();
    fused.Guards();
  }
  Check(cudaEventDestroy(start));
  Check(cudaEventDestroy(stop));
}
}  // namespace

int main(int argc, char** argv) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  try {
    Check(cudaSetDevice(0));
    cudaStream_t stream{};
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const std::string_view mode = argc > 1 ? argv[1] : "";
    if (mode == "--benchmark") {
      Benchmark(stream);
      Check(cudaStreamDestroy(stream));
      return 0;
    }
    if (mode == "--tall-only") {
      for (unsigned channel = 0; channel < 5; ++channel) {
        RawCase(1, 4194305, false, 1, channel,
                CudaButteraugliDirectShortTestSchedulesAvailable() ? 64 : 32,
                stream);
        RawCase(33, 262145, false, 1, channel, 32, stream);
      }
      Check(cudaStreamDestroy(stream));
      std::cout << "Verified 10 tall direct short fixtures\n" << std::flush;
      return 0;
    }
    const bool small = mode == "--sanitizer";
    if (!mode.empty() && !small) throw std::runtime_error("Unknown argument");
    const size_t rejects = RejectMalformed(stream);
    constexpr std::array<std::array<uint32_t, 2>, 12> shapes{{{1, 1},
                                                              {1, 19},
                                                              {19, 1},
                                                              {7, 11},
                                                              {15, 15},
                                                              {31, 15},
                                                              {32, 16},
                                                              {33, 17},
                                                              {31, 31},
                                                              {33, 33},
                                                              {65, 65},
                                                              {127, 129}}};
    size_t raw_cases = 0, parents = 0;
    for (const auto shape : shapes) {
      if (small && shape != shapes[0] && shape != shapes[7] &&
          shape != shapes[10])
        continue;
      for (bool padded : {false, true}) {
        if (small && !padded) continue;
        for (unsigned pattern = 0; pattern < 8; ++pattern) {
          if (small && pattern != 1 && pattern != 4 && pattern != 5) continue;
          for (unsigned channel = 0; channel < 5; ++channel)
            for (unsigned tile : {0u, 16u, 32u, 64u}) {
              if ((tile == 16 || tile == 64) &&
                  !CudaButteraugliDirectShortTestSchedulesAvailable())
                continue;
              if (small && tile == 0) continue;
              RawCase(shape[0], shape[1], padded, pattern, channel, tile,
                      stream);
              ++raw_cases;
            }
        }
      }
    }
    constexpr std::array<std::array<uint32_t, 2>, 10> parent_shapes{
        {{1, 1},
         {1, 19},
         {7, 11},
         {8, 8},
         {14, 15},
         {15, 15},
         {31, 63},
         {33, 65},
         {65, 129},
         {127, 131}}};
    for (const auto shape : parent_shapes) {
      if (small && shape != parent_shapes[0] && shape != parent_shapes[5] &&
          shape != parent_shapes[7])
        continue;
      for (bool padded : {false, true})
        for (unsigned profile = 0; profile < (small ? 2u : 3u); ++profile)
          for (bool cpu_order : {false, true})
            for (unsigned tile : {16u, 32u, 64u}) {
              if (tile != 32 &&
                  !CudaButteraugliDirectShortTestSchedulesAvailable())
                continue;
              if (cpu_order && tile != 32) continue;
              ParentCase(shape[0], shape[1], padded, profile, cpu_order, stream,
                         tile);
              ++parents;
            }
    }
    const bool alternatives =
        CudaButteraugliDirectShortTestSchedulesAvailable();
    const size_t expected_raw =
        alternatives ? (small ? 135 : 3840) : (small ? 45 : 1920);
    const size_t expected_parents =
        alternatives ? (small ? 48 : 240) : (small ? 24 : 120);
    if (raw_cases != expected_raw || parents != expected_parents ||
        rejects != (alternatives ? 27u : 29u))
      throw std::runtime_error("Direct short fixture coverage differs");
    Check(cudaStreamDestroy(stream));
    std::cout
        << "Verified " << raw_cases << " direct short fixtures and " << parents
        << " prepared legacy/fused pairs with guarded A/B/A reuse; rejected="
        << rejects << '\n'
        << std::flush;
  } catch (const std::exception& error) {
    std::cerr << context << ": " << error.what() << '\n';
    return 1;
  }
}
