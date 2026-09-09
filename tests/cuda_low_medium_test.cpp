// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gpu/cuda/cuda_butteraugli_kernels.h"

namespace {
void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct DeviceArray {
  explicit DeviceArray(const std::vector<float>& values) : count(values.size()) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    Write(values);
  }
  ~DeviceArray() { (void)cudaFree(data); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  void Write(const std::vector<float>& values) {
    CheckCuda(cudaMemcpy(data, values.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  }
  std::vector<float> Read() const {
    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
  }
  float* data = nullptr;
  size_t count;
};
void Equal(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::runtime_error("Array size mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      std::cerr << "Mismatch at " << i << ": " << a[i] << " / " << b[i] << '\n';
      throw std::runtime_error("Bitwise low/medium mismatch");
    }
  }
}
struct Case {
  // Three input, horizontal, blurred, low, and medium planes, then weights.
  std::array<std::vector<float>, 16> planes;
  std::array<uint32_t, 15> strides{};
  gjxl::cuda_internal::CudaButteraugliLowMediumPlan plan;
  static size_t Offset(size_t plane) { return 3 + plane % 7; }
  Case(uint32_t width, uint32_t height, bool padded, unsigned pattern) {
    plan.width = width;
    plan.height = height;
    plan.input_stride = width + (padded ? 11 : 0);
    plan.blurred_stride = width + (padded ? 17 : 0);
    plan.output_stride = width + (padded ? 23 : 0);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (size_t p = 0; p < 15; ++p) {
      strides[p] = p < 3 ? plan.input_stride : p < 6 ? width
                    : p < 9 ? plan.blurred_stride : plan.output_stride;
      planes[p].assign(Offset(p) + static_cast<size_t>(strides[p]) * height + 31, poison);
    }
    planes[15].assign(Offset(15) + 33 + 31, poison);
    for (unsigned tap = 0; tap < 33; ++tap) {
      const float delta = static_cast<float>(tap) - 16.0f;
      planes[15][Offset(15) + tap] = pattern == 3 ? (tap == 16 ? 1.0f : 0.0f)
          : pattern == 4 ? 0.03f + (tap % 3) * 0.11f : std::exp(-0.015f * delta * delta);
    }
    std::mt19937 rng(3731u + width + height * 31 + pattern);
    std::uniform_real_distribution<float> random(-3.0f, 3.0f);
    for (size_t p = 0; p < 3; ++p)
      for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
          float value = random(rng);
          if (pattern == 0) value = (x + y) % 2 ? -0.0f : 0.0f;
          if (pattern == 2) value *= (x + y) % 3 ? 1.0e-37f : 1.0e37f;
          if (pattern == 3 && p == 1) value = 1.0f;
          if (pattern == 3 && p == 2) value = std::nextafter(
              0.362267051518f, (x + y) % 2 ? 0.0f : 1.0f);
          if (pattern == 4 && (x + y + p) % 7 == 0)
            value = (x + p) % 2 ? poison : std::numeric_limits<float>::infinity();
          planes[p][Offset(p) + static_cast<size_t>(y) * strides[p] + x] = value;
        }
  }
};
struct DeviceCase {
  std::array<std::unique_ptr<DeviceArray>, 16> planes;
  gjxl::cuda_internal::CudaButteraugliLowMediumPlan plan;
  explicit DeviceCase(const Case& c) : plan(c.plan) {
    for (unsigned tap = 0; tap < 33; ++tap)
      plan.weights.taps[tap] = c.planes[15][Case::Offset(15) + tap];
    for (size_t p = 0; p < planes.size(); ++p)
      planes[p] = std::make_unique<DeviceArray>(c.planes[p]);
    const auto pointer = [&](size_t p) { return planes[p]->data + Case::Offset(p); };
    for (size_t p = 0; p < 3; ++p) {
      plan.input[p] = pointer(p);
      plan.intermediate[p] = pointer(3 + p);
      plan.blurred[p] = pointer(6 + p);
      plan.low[p] = pointer(9 + p);
      plan.medium[p] = pointer(12 + p);
    }
    plan.device_weights = pointer(15);
  }
  void Launch(bool reference, cudaStream_t stream = nullptr, int schedule = -1) {
    using namespace gjxl::cuda_internal;
    CheckCuda(reference ? LaunchCudaButteraugliLowMediumReference(plan, stream)
                        : schedule < 0 ? LaunchCudaButteraugliLowMedium(plan, stream)
                        : LaunchCudaButteraugliLowMediumForTesting(
                              plan, static_cast<unsigned>(schedule), stream));
  }
};
void Verify(uint32_t width, uint32_t height, bool padded, unsigned pattern,
            int schedule = -1) {
  Case c(width, height, padded, pattern);
  DeviceCase reference(c), sequential(c), candidate(c);
  for (unsigned reuse = 0; reuse < 3; ++reuse) {
    if (reuse == 2) {
      for (size_t p = 0; p < 3; ++p)
        for (uint32_t y = 0; y < height; ++y)
          for (uint32_t x = 0; x < width; ++x) {
            auto& value = c.planes[p][Case::Offset(p) + static_cast<size_t>(y) * c.strides[p] + x];
            value = -0.75f * value + 0.013f;
          }
      candidate.plan.blurred.fill(nullptr);
      candidate.plan.blurred_stride = 0;
      sequential.plan.blurred.fill(nullptr);
      sequential.plan.blurred_stride = 0;
    }
    for (size_t p = 0; p < 16; ++p) {
      reference.planes[p]->Write(c.planes[p]);
      sequential.planes[p]->Write(c.planes[p]);
      candidate.planes[p]->Write(c.planes[p]);
    }
    reference.Launch(true);
    CheckCuda(gjxl::cuda_internal::LaunchCudaButteraugliLowMediumSequentialReference(
        sequential.plan, nullptr));
    candidate.Launch(false, nullptr, schedule);
    CheckCuda(cudaDeviceSynchronize());
    for (size_t p = 0; p < 16; ++p) {
      const auto a = reference.planes[p]->Read();
      const auto b = candidate.planes[p]->Read();
      const auto s = sequential.planes[p]->Read();
      if (p >= 6 && p < 9) {
        Equal(c.planes[p], b);
        Equal(c.planes[p], s);
      } else {
        Equal(a, b);
        Equal(a, s);
      }
      if (p < 3 || p == 15) Equal(c.planes[p], a);
      if (p == 15) continue;
      for (size_t i = 0; i < a.size(); ++i) {
        const bool active = i >= Case::Offset(p) &&
            i < Case::Offset(p) + static_cast<size_t>(c.strides[p]) * height &&
            (i - Case::Offset(p)) % c.strides[p] < width;
        if (!active && (std::memcmp(&a[i], &c.planes[p][i], sizeof(float)) != 0 ||
                        std::memcmp(&b[i], &c.planes[p][i], sizeof(float)) != 0))
          throw std::runtime_error("Low/medium padding guard overwritten");
      }
    }
  }
}
struct OwnedCase {
  Case host;
  DeviceCase device, oracle;
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  std::array<std::vector<float>,16> expected;
  OwnedCase(unsigned width,unsigned height,unsigned pattern)
      :host(width,height,true,pattern),device(host),oracle(host) {
    CheckCuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  }
  void Prepare() {
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(gjxl::cuda_internal::LaunchCudaButteraugliLowMediumForTesting(device.plan,0,stream));
    CheckCuda(gjxl::cuda_internal::LaunchCudaButteraugliLowMediumReference(oracle.plan,stream));
    CheckCuda(cudaStreamSynchronize(stream));
    for(unsigned p=0;p<16;++p) {
      expected[p]=device.planes[p]->Read();
      if(p>=6&&p<9)Equal(expected[p],host.planes[p]);
      else Equal(expected[p],oracle.planes[p]->Read());
    }
  }
  void Capture(int variant) {
    // The payload dies on return; both it and the caller-owned copy are
    // poisoned before either independent graph is launched.
    auto temporary=device.plan;
    CheckCuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(unsigned i=0;i<4;++i)CheckCuda(gjxl::cuda_internal::LaunchCudaButteraugliLowMediumForTesting(temporary,variant,stream));
    CheckCuda(cudaStreamEndCapture(stream,&graph));size_t nodes=0;
    CheckCuda(cudaGraphGetNodes(graph,nullptr,&nodes));
    if(nodes!=8)throw std::runtime_error("Ownership graph node count mismatch");
    CheckCuda(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    std::fill(std::begin(temporary.weights.taps),std::end(temporary.weights.taps),-9876.0f);
    std::fill(std::begin(device.plan.weights.taps),std::end(device.plan.weights.taps),-5432.0f);
  }
  void Verify() {for(unsigned p=0;p<16;++p)Equal(expected[p],device.planes[p]->Read());}
  void ReplaceWeights(unsigned salt) {
    for(unsigned tap=0;tap<33;++tap) {
      float value=tap==16?1.25f:((tap+salt)%3==0?-0.03125f:0.0625f);
      host.planes[15][Case::Offset(15)+tap]=value;
      device.plan.weights.taps[tap]=value;oracle.plan.weights.taps[tap]=value;
    }
    device.planes[15]->Write(host.planes[15]);oracle.planes[15]->Write(host.planes[15]);
  }
  void CloseGraph() {CheckCuda(cudaGraphExecDestroy(executable));CheckCuda(cudaGraphDestroy(graph));}
  void CloseStream() {CheckCuda(cudaStreamDestroy(stream));}
};
void VerifyWeightOwnership() {
    unsigned checked=0;
    for(int variant:{64,96})for(unsigned pattern:{1u,3u,4u}) {
      OwnedCase a(33,97,pattern),b(65,193,(pattern+1)%5);
      for(unsigned generation=0;generation<2;++generation) {
        if(generation){a.ReplaceWeights(0);b.ReplaceWeights(1);}
        a.Prepare();b.Prepare();a.Capture(variant);b.Capture(variant);
        for(unsigned repeat=0;repeat<3;++repeat) {
          CheckCuda(cudaGraphLaunch(b.executable,b.stream));
          CheckCuda(cudaGraphLaunch(a.executable,a.stream));
          CheckCuda(cudaStreamSynchronize(a.stream));CheckCuda(cudaStreamSynchronize(b.stream));
          a.Verify();b.Verify();checked+=2;
        }
        a.CloseGraph();b.CloseGraph();
      }
      a.CloseStream();b.CloseStream();
      std::cout<<"Low/medium ownership CASE PASS variant="<<variant<<" pattern="<<pattern<<'\n'<<std::flush;
    }
    std::cout<<"Low/medium ownership PASS checked="<<checked<<" arrays=16 exact=1\n"<<std::flush;
}
void VerifyPolicy() {
  using gjxl::cuda_internal::CudaButteraugliLowMediumRollingTileHeight;
  constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
  constexpr std::array<std::array<uint32_t, 3>, 18> choices{{
      {0,maximum,0}, {maximum,0,0}, {1,maximum,0}, {31,maximum,0},
      {maximum,95,0}, {32,62499,0}, {32,62500,64}, {32,124999,64},
      {32,125000,96}, {20833,96,0}, {20834,96,64}, {41667,96,96},
      {1919,1079,64}, {1920,1080,64}, {1999,2000,64}, {2000,2000,96},
      {500,500,0}, {maximum,maximum,96}}};
  for (const auto& choice : choices)
    if (CudaButteraugliLowMediumRollingTileHeight(choice[0],choice[1]) != choice[2])
      throw std::runtime_error("Low/medium geometry policy mismatch");
}

}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  const std::string_view mode = argc > 1 ? argv[1] : "";
  try {
    CheckCuda(cudaSetDevice(0));
    using namespace gjxl::cuda_internal;
    VerifyPolicy();
    for (bool zero_width : {false,true}) {
      CudaButteraugliLowMediumPlan empty;
      empty.width = zero_width ? 0 : 17;
      empty.height = zero_width ? 17 : 0;
      CheckCuda(LaunchCudaButteraugliLowMedium(empty, nullptr));
      CheckCuda(LaunchCudaButteraugliLowMediumReference(empty, nullptr));
      CheckCuda(LaunchCudaButteraugliLowMediumSequentialReference(empty, nullptr));
      for (unsigned tile : {0u,64u,96u})
        CheckCuda(LaunchCudaButteraugliLowMediumForTesting(empty, tile, nullptr));
    }
    for (unsigned invalid = 0; invalid < 3; ++invalid) {
      CudaButteraugliLowMediumPlan bad;
      bad.width = bad.height = 1;
      bad.input_stride = bad.blurred_stride = bad.output_stride = 1;
      uint32_t* const strides[] = {&bad.input_stride, &bad.blurred_stride, &bad.output_stride};
      *strides[invalid] = 0;
      if (LaunchCudaButteraugliLowMediumReference(bad, nullptr) != cudaErrorInvalidValue ||
          (invalid != 1 && LaunchCudaButteraugliLowMedium(bad, nullptr) != cudaErrorInvalidValue) ||
          (invalid != 1 && LaunchCudaButteraugliLowMediumSequentialReference(bad, nullptr) != cudaErrorInvalidValue))
        throw std::runtime_error("Invalid low/medium stride not rejected");
      if (invalid != 1)
        for (unsigned tile : {0u,64u,96u})
          if (LaunchCudaButteraugliLowMediumForTesting(bad, tile, nullptr) != cudaErrorInvalidValue)
            throw std::runtime_error("Invalid forced low/medium stride not rejected");
    }
    CudaButteraugliLowMediumPlan empty;
    for (unsigned tile : {0u,64u,96u})
      CheckCuda(LaunchCudaButteraugliLowMediumForTesting(empty,tile,nullptr));
    for (unsigned tile : {1u,24u,48u,128u})
      if (LaunchCudaButteraugliLowMediumForTesting(empty,tile,nullptr) != cudaErrorInvalidValue)
        throw std::runtime_error("Invalid low/medium tile not rejected");
    if (mode == "--ownership-only") {
      VerifyWeightOwnership();
      return 0;
    }
    if (mode == "--policy-only") {
      constexpr std::array<std::array<uint32_t,2>,8> boundaries{{
          {1562,1280},{1563,1280},{1999,2000},{2000,2000},
          {31,65536},{32,62500},{42106,95},{41667,96}}};
      unsigned cases = 0;
      for (const auto& shape : boundaries) {
        Verify(shape[0],shape[1],cases % 2 != 0,1);++cases;
        std::cout << "Verified low/medium policy geometry " << shape[0] << 'x' << shape[1]
                  << " tile=" << CudaButteraugliLowMediumRollingTileHeight(shape[0],shape[1])
                  << '\n' << std::flush;
      }
      std::cout << "Verified " << cases << " low/medium production-policy cases\n" << std::flush;
      return 0;
    }
    if (mode == "--tall-only") {
      Verify(1, 4194305, false, 1);
      std::cout << "Verified tall low/medium case above 65535 tile rows\n" << std::flush;
      Verify(33, 262145, true, 1);
      std::cout << "Verified paired horizontal case above 65535 tile rows\n" << std::flush;
      return 0;
    }
    constexpr std::array<std::array<uint32_t, 2>, 64> shapes{{
        {1,1}, {1,19}, {19,1}, {7,11}, {15,15}, {31,31}, {32,32}, {33,33},
        {47,47}, {48,48}, {49,49}, {63,63}, {64,64}, {65,65}, {95,95},
        {96,96}, {97,97}, {127,65}, {255,3}, {256,4}, {257,67}, {511,129},
        {1,3}, {1,4}, {1,5}, {255,5}, {256,3}, {256,5}, {257,3},
        {257,4}, {257,5}, {513,9}, {1,2}, {1,6}, {1,7},
        {33,16}, {33,17}, {33,18}, {2,3}, {3,4}, {4,5},
        {16,3}, {17,4}, {18,5}, {34,3}, {35,5},
        {33,127}, {33,128}, {33,129}, {65,127}, {65,128}, {65,129},
        {33,191}, {33,192}, {33,193}, {65,191}, {65,192}, {65,193},
        {33,255}, {33,256}, {33,257}, {65,255}, {65,256}, {65,257}}};
    size_t cases = 0;
    for (const auto& shape : shapes) {
      // Include tiny, all-border, and horizontal-tile-boundary cases without
      // paying for the tall fixture under race instrumentation.
      if (mode == "--sanitizer" && shape[0] != 33 &&
          shape[0] != 34 && shape[0] != 35 &&
          !(shape[0] == 1 && shape[1] == 5) &&
          !(shape[0] == 257 && shape[1] == 5)) continue;
      for (bool padded : {false,true})
        for (unsigned pattern = 0; pattern < 5; ++pattern)
          for (int schedule : {-1, 0, 64, 96}) {
            Verify(shape[0], shape[1], padded, pattern, schedule);
            ++cases;
          }
      std::cout << "Verified low/medium geometry " << shape[0] << 'x' << shape[1]
                << " (" << cases << " cases)\n" << std::flush;
    }
    std::cout << "Verified " << cases << " guarded low/medium cases with three-stage reuse\n" << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
