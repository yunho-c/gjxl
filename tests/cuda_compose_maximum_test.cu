// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/cuda_butteraugli_kernels.h"

namespace {
constexpr uint32_t kWidth = 256;
constexpr size_t kPrefix = 13;
constexpr size_t kSuffix = 19;
constexpr float kGuard = -12345.0f;

void Check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct Buffer {
  explicit Buffer(size_t size) : host(size + kPrefix + kSuffix, kGuard) {
    Check(cudaMalloc(&allocation, host.size() * sizeof(float)));
    data = allocation + kPrefix;
  }
  ~Buffer() { if (allocation) cudaFree(allocation); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void Upload() {
    Check(cudaMemcpy(allocation, host.data(), host.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  }
  std::vector<float> Download() const {
    std::vector<float> result(host.size());
    Check(cudaMemcpy(result.data(), allocation, result.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < kPrefix; ++i)
      if (result[i] != kGuard) throw std::runtime_error("prefix guard");
    for (size_t i = result.size() - kSuffix; i < result.size(); ++i)
      if (result[i] != kGuard) throw std::runtime_error("suffix guard");
    return result;
  }
  float* allocation = nullptr;
  float* data = nullptr;
  std::vector<float> host;
};

// Independent copies of the separate passes, deliberately not sharing the
// production reduction helper or composition function.
__global__ void ReferenceCompose(const float* main_map, const float* sub_map,
                                 float* output, uint32_t width, uint32_t height,
                                 uint32_t main_stride, uint32_t sub_stride,
                                 uint32_t output_stride) {
  const size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (flat >= size_t(width) * height) return;
  const uint32_t y = uint32_t(flat / width);
  const uint32_t x = uint32_t(flat - size_t(y) * width);
  output[size_t(y) * output_stride + x] =
      main_map[size_t(y) * main_stride + x] * 0.85f +
      0.5f * sub_map[size_t(y / 2) * sub_stride + x / 2];
}

__global__ void ReferenceMaximum(const float* input, float* output,
                                 uint32_t width, uint32_t stride, uint32_t count) {
  __shared__ float values[kWidth];
  const uint32_t i = blockIdx.x * kWidth + threadIdx.x;
  float value = -INFINITY;
  if (i < count) {
    const uint32_t y = i / width;
    value = input[size_t(y) * stride + i - y * width];
    if (!isfinite(value) || value < 0.0f) value = NAN;
  }
  values[threadIdx.x] = value;
  __syncthreads();
  for (uint32_t step = kWidth / 2; step; step /= 2) {
    if (threadIdx.x < step) {
      const float other = values[threadIdx.x + step];
      values[threadIdx.x] = isnan(values[threadIdx.x]) || isnan(other)
          ? NAN : fmaxf(values[threadIdx.x], other);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = values[0];
}

void Reference(const gjxl::cuda_internal::CudaButteraugliComposePlan& p,
               cudaStream_t stream) {
  uint32_t count = p.width * p.height;
  uint32_t width = p.width, stride = p.output_stride;
  ReferenceCompose<<<(count + 255) / 256, 256, 0, stream>>>(
      p.main_map, p.sub_map, p.output, p.width, p.height, p.main_stride,
      p.sub_stride, p.output_stride);
  Check(cudaGetLastError());
  const float* input = p.output;
  bool use_a = true;
  for (;;) {
    const uint32_t blocks = (count + 255) / 256;
    float* output = blocks == 1 ? p.score : p.reduction[use_a ? 0 : 1];
    ReferenceMaximum<<<blocks, 256, 0, stream>>>(input, output, width, stride, count);
    Check(cudaGetLastError());
    if (blocks == 1) return;
    input = output;
    count = width = stride = blocks;
    use_a = !use_a;
  }
}

void Same(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
    throw std::runtime_error("bitwise map/input mismatch");
}

size_t Run(uint32_t width, uint32_t height, uint32_t pad, bool inplace,
           cudaStream_t stream) {
  const uint32_t main_stride = width + pad;
  const uint32_t sub_stride = (width + 1) / 2 + 2 * pad;
  const uint32_t output_stride = inplace ? main_stride : width + 3 * pad;
  const uint32_t partial_count = (width * height + 255) / 256;
  Buffer main(size_t(main_stride) * height);
  Buffer sub(size_t(sub_stride) * ((height + 1) / 2));
  Buffer ref(size_t(output_stride) * height), actual(size_t(output_stride) * height);
  Buffer a(partial_count), b(partial_count), ref_score(1), score(1);
  for (int pattern = 0; pattern < 8; ++pattern) {
    for (uint32_t y = 0; y < height; ++y)
      for (uint32_t x = 0; x < width; ++x)
        main.host[kPrefix + size_t(y) * main_stride + x] =
            pattern == 1 ? 0.0f : pattern == 2 ? -0.0f :
            float((uint64_t(x) * 137 + uint64_t(y) * 79) % 10001) / 2000.0f;
    for (uint32_t y = 0; y < (height + 1) / 2; ++y)
      for (uint32_t x = 0; x < (width + 1) / 2; ++x)
        sub.host[kPrefix + size_t(y) * sub_stride + x] =
            pattern == 1 ? 0.0f : pattern == 2 ? -0.0f :
            float((uint64_t(x) * 89 + uint64_t(y) * 53) % 10001) / 3000.0f;
    if (pattern == 3) main.host[kPrefix] = NAN;
    if (pattern == 4)
      main.host[kPrefix + size_t(height / 2) * main_stride + width / 2] = INFINITY;
    if (pattern == 5)
      main.host[kPrefix + size_t(height - 1) * main_stride + width - 1] = -100.0f;
    if (pattern == 6) sub.host[kPrefix + size_t((height - 1) / 2) * sub_stride + (width - 1) / 2] = NAN;
    if (pattern == 7) {
      main.host[kPrefix] = std::numeric_limits<float>::max();
      sub.host[kPrefix] = std::numeric_limits<float>::max();
    }
    if (inplace) ref.host = actual.host = main.host;
    main.Upload(); sub.Upload(); ref.Upload(); actual.Upload();
    a.Upload(); b.Upload(); ref_score.Upload(); score.Upload();
    gjxl::cuda_internal::CudaButteraugliComposePlan p{
        inplace ? ref.data : main.data, sub.data, ref.data,
        partial_count == 1 ? std::array<float*, 2>{} : std::array<float*, 2>{a.data, b.data},
        ref_score.data, width, height, main_stride, sub_stride, output_stride};
    Reference(p, stream);
    Check(cudaStreamSynchronize(stream));
    // Re-poison reusable scratch so correctness never depends on the oracle.
    a.Upload(); b.Upload();
    p.main_map = inplace ? actual.data : main.data;
    p.output = actual.data;
    p.score = score.data;
    Check(gjxl::cuda_internal::LaunchCudaButteraugliCompose(p, stream));
    Check(cudaStreamSynchronize(stream));
    const auto expected = ref.Download(), result = actual.Download();
    Same(expected, result);
    for (uint32_t y = 0; y < height; ++y)
      for (uint32_t x = width; x < output_stride; ++x)
        if (result[kPrefix + size_t(y) * output_stride + x] != kGuard)
          throw std::runtime_error("row guard");
    Same(main.host, main.Download()); Same(sub.host, sub.Download());
    const float f0 = ref_score.Download()[kPrefix], f1 = score.Download()[kPrefix];
    if (pattern >= 3) {
      if (!std::isnan(f0) || !std::isnan(f1)) throw std::runtime_error("invalid score");
    } else if (std::memcmp(&f0, &f1, sizeof(float))) {
      throw std::runtime_error("bitwise score mismatch");
    }
    (void)a.Download(); (void)b.Download();
  }
  return 8;
}
}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    const bool sanitizer = argc == 2 && std::string(argv[1]) == "--sanitizer";
    std::vector<std::pair<uint32_t, uint32_t>> shapes{
        {1, 1}, {7, 3}, {15, 15}, {16, 16}, {17, 17}, {31, 33},
        {255, 1}, {256, 1}, {257, 1}, {65535, 1}, {65536, 1}, {65537, 1}};
    if (!sanitizer) shapes.insert(shapes.end(), {{500, 500}, {1919, 1079}, {3839, 2159}});
    cudaStream_t stream;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    size_t cases = 0;
    for (auto shape : shapes) for (uint32_t pad : {0u, 7u}) for (bool inplace : {false, true})
      cases += Run(shape.first, shape.second, pad, inplace, stream);
    Check(cudaStreamDestroy(stream));
    std::cout << "CUDA compose maximum PASS cases=" << cases << '\n' << std::flush;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "CUDA compose maximum ERROR: " << e.what() << '\n';
    return 1;
  }
}
