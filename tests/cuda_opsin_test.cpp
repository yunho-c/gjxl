// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

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
    std::vector<float> values(count);
    CheckCuda(cudaMemcpy(values.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
    return values;
  }
  float* data = nullptr;
  size_t count;
};
void Equal(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::runtime_error("Opsin array size mismatch");
  for (size_t i=0; i<a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      std::cerr << "Mismatch at " << i << ": " << a[i] << " / " << b[i] << '\n';
      throw std::runtime_error("Bitwise Opsin mismatch");
    }
  }
}
struct Case {
  // RGB, horizontal RGB, blurred RGB, XYB, then weights.
  std::array<std::vector<float>, 13> planes;
  std::array<uint32_t, 12> strides{};
  gjxl::cuda_internal::CudaButteraugliOpsinPlan plan;
  static size_t Offset(size_t plane) { return 3 + plane % 7; }
  Case(uint32_t width, uint32_t height, bool padded, unsigned pattern) {
    plan.width=width; plan.height=height;
    plan.output_stride=width+(padded ? 23 : 0);
    const float poison=std::numeric_limits<float>::quiet_NaN();
    for (size_t p=0; p<12; ++p) {
      strides[p]=p<3 ? width+(padded ? 11+3*static_cast<uint32_t>(p) : 0)
                     : p<6 ? width : plan.output_stride;
      planes[p].assign(Offset(p)+static_cast<size_t>(strides[p])*height+31,poison);
      if (p<3) plan.input_stride[p]=strides[p];
    }
    planes[12].assign(Offset(12)+5+31,poison);
    for (unsigned tap=0; tap<5; ++tap) {
      const float delta=static_cast<float>(tap)-2.0f;
      planes[12][Offset(12)+tap]=pattern==3 ? (tap==2 ? 1.0f : 0.0f)
          : pattern==5 ? 0.03f+(tap%3)*0.11f : std::exp(-0.5f*delta*delta);
    }
    std::mt19937 rng(4973u+width+height*31+pattern);
    std::uniform_real_distribution<float> random(-1.0f,3.0f);
    for (size_t p=0; p<3; ++p)
      for (uint32_t y=0; y<height; ++y)
        for (uint32_t x=0; x<width; ++x) {
          float value=random(rng);
          if (pattern==0) value=(x+y+p)%2 ? -0.0f : 0.0f;
          if (pattern==2) value *= (x+y)%3 ? 1.0e-37f : 1.0e37f;
          if (pattern==3) value=std::nextafter(0.0f,(x+y+p)%2 ? -1.0f : 1.0f);
          if (pattern==4 && (x+y+p)%7==0)
            value=(x+p)%2 ? poison : (y%2 ? -1.0f : 1.0f)*std::numeric_limits<float>::infinity();
          planes[p][Offset(p)+static_cast<size_t>(y)*strides[p]+x]=value;
        }
  }
};
struct DeviceCase {
  std::array<std::unique_ptr<DeviceArray>,13> planes;
  gjxl::cuda_internal::CudaButteraugliOpsinPlan plan;
  explicit DeviceCase(const Case& c) : plan(c.plan) {
    for (size_t p=0; p<13; ++p) planes[p]=std::make_unique<DeviceArray>(c.planes[p]);
    const auto pointer=[&](size_t p) { return planes[p]->data+Case::Offset(p); };
    for (size_t p=0; p<3; ++p) {
      plan.input[p]=pointer(p); plan.intermediate[p]=pointer(3+p);
      plan.blurred[p]=pointer(6+p); plan.output[p]=pointer(9+p);
    }
    plan.weights=pointer(12);
  }
  void Launch(bool reference, cudaStream_t stream=nullptr) {
    using namespace gjxl::cuda_internal;
    CheckCuda(reference ? LaunchCudaButteraugliOpsinReference(plan,stream)
                        : LaunchCudaButteraugliOpsin(plan,stream));
  }
};
void Verify(uint32_t width, uint32_t height, bool padded, unsigned pattern) {
  Case c(width,height,padded,pattern);
  DeviceCase reference(c),candidate(c);
  const auto original_reference=reference.plan;
  for (unsigned reuse=0; reuse<3; ++reuse) {
    reference.plan=original_reference;
    reference.plan.intensity_target=candidate.plan.intensity_target=
        std::array<float,3>{80.0f,255.0f,1000.0f}[reuse];
    if (reuse==1) {
      // Exact S47 storage policy: one reused horizontal plane and blurred RGB
      // overwritten by pointwise XYB. The fused path needs three horizontals.
      reference.plan.intermediate.fill(reference.plan.intermediate[0]);
      reference.plan.blurred=reference.plan.output;
    }
    if (reuse==2) {
      for (size_t p=0; p<3; ++p)
        for (uint32_t y=0; y<height; ++y)
          for (uint32_t x=0; x<width; ++x) {
            auto& value=c.planes[p][Case::Offset(p)+static_cast<size_t>(y)*c.strides[p]+x];
            value=-0.75f*value+0.013f;
          }
      candidate.plan.blurred.fill(nullptr);
    }
    for (size_t p=0; p<13; ++p) {
      reference.planes[p]->Write(c.planes[p]);
      candidate.planes[p]->Write(c.planes[p]);
    }
    reference.Launch(true); candidate.Launch(false);
    CheckCuda(cudaDeviceSynchronize());
    for (size_t p=0; p<13; ++p) {
      const auto a=reference.planes[p]->Read(), b=candidate.planes[p]->Read();
      if (p<3 || p==12) { Equal(c.planes[p],a); Equal(c.planes[p],b); }
      if (p>=9 && p<12) Equal(a,b);
      if (p>=3 && p<6 && reuse!=1) Equal(a,b);
      if (p>=6 && p<9) { Equal(c.planes[p],b); if (reuse==1) Equal(c.planes[p],a); }
      if (reuse==1 && (p==4 || p==5)) Equal(c.planes[p],a);
      if (p==12) continue;
      for (size_t i=0; i<a.size(); ++i) {
        const bool active=i>=Case::Offset(p) &&
            i<Case::Offset(p)+static_cast<size_t>(c.strides[p])*height &&
            (i-Case::Offset(p))%c.strides[p]<width;
        if (!active && (std::memcmp(&a[i],&c.planes[p][i],sizeof(float))!=0 ||
                        std::memcmp(&b[i],&c.planes[p][i],sizeof(float))!=0))
          throw std::runtime_error("Opsin padding guard overwritten");
      }
    }
  }
}
}  // namespace

int main(int argc,char** argv) {
  int devices=0;
  if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
  const std::string_view mode=argc>1 ? argv[1] : "";
  try {
    CheckCuda(cudaSetDevice(0));
    using namespace gjxl::cuda_internal;
    for (bool zero_width : {false,true}) {
      CudaButteraugliOpsinPlan empty;
      empty.width=zero_width ? 0 : 17; empty.height=zero_width ? 17 : 0;
      CheckCuda(LaunchCudaButteraugliOpsin(empty,nullptr));
      CheckCuda(LaunchCudaButteraugliOpsinReference(empty,nullptr));
    }
    for (unsigned invalid=0; invalid<4; ++invalid) {
      CudaButteraugliOpsinPlan bad;
      bad.width=bad.height=bad.output_stride=1; bad.input_stride.fill(1);
      (invalid<3 ? bad.input_stride[invalid] : bad.output_stride)=0;
      if (LaunchCudaButteraugliOpsin(bad,nullptr)!=cudaErrorInvalidValue ||
          LaunchCudaButteraugliOpsinReference(bad,nullptr)!=cudaErrorInvalidValue)
        throw std::runtime_error("Invalid Opsin stride not rejected");
    }
    if (mode=="--tall-only") {
      Verify(1,4194305,false,1);
      std::cout << "Verified tall Opsin case above 65535 tile rows\n" << std::flush;
      return 0;
    }
    constexpr std::array<std::array<uint32_t,2>,20> shapes{{
        {1,1},{1,3},{3,1},{1,19},{19,1},{2,2},{3,7},{7,3},{8,8},{15,15},
        {16,16},{17,17},{31,31},{32,32},{33,33},{63,17},{64,16},{65,33},{257,67},{511,129}}};
    size_t cases=0;
    for (const auto& shape:shapes) {
      if (mode=="--sanitizer" && shape[0]!=1 && shape[0]!=33 && shape[0]!=65) continue;
      for (bool padded:{false,true}) for (unsigned pattern=0;pattern<6;++pattern) {
        Verify(shape[0],shape[1],padded,pattern); ++cases;
      }
      std::cout << "Verified Opsin geometry " << shape[0] << 'x' << shape[1]
                << " (" << cases << " cases)\n" << std::flush;
    }
    std::cout << "Verified " << cases << " guarded Opsin cases with three-stage reuse\n" << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
  return 0;
}
