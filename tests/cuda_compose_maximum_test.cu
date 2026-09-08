// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "cuda_compose_test_utils.cuh"

namespace {
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
