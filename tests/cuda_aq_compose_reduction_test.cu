// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "cuda_compose_test_utils.cuh"
#include "gpu/cuda/cuda_aq_butteraugli_kernels.h"
#include <random>
using gjxl::cuda_internal::CudaAqAnchor;
using gjxl::cuda_internal::CudaAqExactBatch;

namespace {
struct AqComposeFixture {
  uint32_t width, height, bw, bh, main_stride, sub_stride, block_stride;
  std::vector<CudaAqAnchor> anchors;
  std::vector<CudaAqExactBatch> batches;
  AqComposeFixture(uint32_t w, uint32_t h, unsigned preference, unsigned pad)
      : width(w), height(h), bw((w + 7) / 8), bh((h + 7) / 8),
        main_stride(w + pad), sub_stride((w + 1) / 2 + 2 * pad),
        block_stride(bw + 3 * pad) {
    const std::array<std::pair<unsigned, unsigned>, 7> sizes{
        {{1, 1}, {2, 2}, {4, 4}, {2, 1}, {1, 2}, {4, 2}, {2, 4}}};
    std::array<std::vector<CudaAqAnchor>, 7> groups;
    std::vector<bool> used(size_t(bw) * bh);
    for (uint32_t y = 0; y < bh; ++y)
      for (uint32_t x = 0; x < bw; ++x) {
        if (used[size_t(y) * bw + x])
          continue;
        unsigned choice = preference == 7 ? (x * 13 + y * 7) % 7 : preference;
        auto fits = [&](unsigned s) {
          auto [sx, sy] = sizes[s];
          if (x + sx > bw || y + sy > bh)
            return false;
          for (unsigned j = 0; j < sy; ++j)
            for (unsigned i = 0; i < sx; ++i)
              if (used[size_t(y + j) * bw + x + i])
                return false;
          return true;
        };
        if (!fits(choice))
          choice = 0;
        auto [sx, sy] = sizes[choice];
        for (unsigned j = 0; j < sy; ++j)
          for (unsigned i = 0; i < sx; ++i)
            used[size_t(y + j) * bw + x + i] = true;
        groups[choice].push_back({x, y});
      }
    for (unsigned i = 0; i < 7; ++i) {
      const auto [sx, sy] = sizes[i];
      batches.push_back({uint32_t(anchors.size()), uint32_t(groups[i].size()),
                         0, 0, sx * 8, sy * 8, sx, sy});
      anchors.insert(anchors.end(), groups[i].begin(), groups[i].end());
    }
  }
};
struct AqComposeProbe {
  AqComposeFixture f;
  Buffer main, sub, map, blocks, score, partial_a, partial_b, maxima, anchors,
      error;
  cudaStream_t stream = nullptr;
  AqComposeProbe(uint32_t w, uint32_t h, unsigned strategy, unsigned pad)
      : f(w, h, strategy, pad), main(size_t(f.main_stride) * h),
        sub(size_t(f.sub_stride) * ((h + 1) / 2)),
        map(size_t(f.main_stride) * h), blocks(size_t(f.block_stride) * f.bh),
        score(1), partial_a((w * h + 255) / 256),
        partial_b((w * h + 255) / 256), maxima(f.anchors.size()),
        anchors(2 * f.anchors.size()), error(1) {
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::memcpy(anchors.host.data() + kPrefix, f.anchors.data(),
                f.anchors.size() * sizeof(CudaAqAnchor));
    anchors.Upload();
  }
  ~AqComposeProbe() {
    if (stream)
      cudaStreamDestroy(stream);
  }
  void Fill(unsigned pattern) {
    std::mt19937 random(1110000 + f.width * 17 + f.height);
    for (uint32_t y = 0; y < f.height; ++y)
      for (uint32_t x = 0; x < f.width; ++x) {
        float value = float(random() % 100001) / 40000.0f;
        if (pattern == 1)
          value = std::ldexp(float(1 + random() % 10001) / 10001.0f,
                             int(random() % 32) - 20);
        if (pattern == 2)
          value = (random() % 2) ? 0.0f : -0.0f;
        if (pattern == 3)
          value = -0.0f;
        if (pattern == 9)
          value = std::numeric_limits<float>::denorm_min() *
                  float(1 + random() % 16);
        main.host[kPrefix + size_t(y) * f.main_stride + x] = value;
      }
    for (uint32_t y = 0; y < (f.height + 1) / 2; ++y)
      for (uint32_t x = 0; x < (f.width + 1) / 2; ++x) {
        float value = float(random() % 100001) / 60000.0f;
        if (pattern == 2)
          value = (random() % 2) ? 0.0f : -0.0f;
        if (pattern == 3)
          value = -0.0f;
        if (pattern == 9)
          value = std::numeric_limits<float>::denorm_min() *
                  float(1 + random() % 16);
        sub.host[kPrefix + size_t(y) * f.sub_stride + x] = value;
      }
    if (pattern == 4)
      main.host[kPrefix] = NAN;
    if (pattern == 5)
      main.host[kPrefix + size_t(f.height / 2) * f.main_stride + f.width / 2] =
          INFINITY;
    if (pattern == 6)
      main.host[kPrefix + size_t(f.height - 1) * f.main_stride + f.width - 1] =
          -100.0f;
    if (pattern == 7)
      sub.host[kPrefix + size_t((f.height - 1) / 2) * f.sub_stride +
               (f.width - 1) / 2] = NAN;
    if (pattern == 8) {
      main.host[kPrefix] = std::numeric_limits<float>::max();
      sub.host[kPrefix] = std::numeric_limits<float>::max();
    }
    main.Upload();
    sub.Upload();
    map.Upload();
    partial_a.Upload();
    partial_b.Upload();
    maxima.Upload();
  }
  void Run(bool fused, bool seeded = false) {
    Check(cudaMemsetAsync(error.data, seeded ? 0x40 : 0, sizeof(unsigned),
                          stream));
    const auto *anchor = reinterpret_cast<const CudaAqAnchor *>(anchors.data);
    if (!fused) {
      Reference({main.data,
                 sub.data,
                 map.data,
                 {partial_a.data, partial_b.data},
                 score.data,
                 f.width,
                 f.height,
                 f.main_stride,
                 f.sub_stride,
                 f.main_stride},
                stream);
      for (auto batch : f.batches)
        Check(gjxl::cuda_internal::LaunchCudaAqReduceButteraugli(
            map.data, f.main_stride, anchor, blocks.data, f.block_stride,
            reinterpret_cast<unsigned *>(error.data), f.width, f.height, batch,
            stream));
    } else {
      for (auto batch : f.batches)
        Check(gjxl::cuda_internal::LaunchCudaAqComposeReduction(
            {main.data, sub.data, anchor, blocks.data, maxima.data,
             reinterpret_cast<unsigned *>(error.data), f.width, f.height,
             f.main_stride, f.sub_stride, f.block_stride, batch},
            stream));
      uint32_t count = uint32_t(f.anchors.size());
      const float *input = maxima.data;
      bool use_a = true;
      for (;;) {
        const uint32_t n = (count + 255) / 256;
        float *output = n == 1  ? score.data
                        : use_a ? partial_a.data
                                : partial_b.data;
        ReferenceMaximum<<<n, 256, 0, stream>>>(input, output, count, count,
                                                count);
        Check(cudaGetLastError());
        if (n == 1)
          break;
        input = output;
        count = n;
        use_a = !use_a;
      }
    }
  }
  unsigned Flags() {
    const auto values = error.Download();
    unsigned result;
    std::memcpy(&result, values.data() + kPrefix, 4);
    return result;
  }
  size_t Verify(unsigned pattern) {
    Fill(pattern);
    blocks.Upload();
    score.Upload();
    error.Upload();
    Run(false);
    Check(cudaStreamSynchronize(stream));
    const auto expected = blocks.Download();
    const float expected_score = score.Download()[kPrefix];
    const unsigned expected_flags = Flags();
    for (unsigned reuse = 0; reuse < 2; ++reuse) {
      blocks.Upload();
      score.Upload();
      error.Upload();
      maxima.Upload();
      Run(true, reuse != 0);
      Check(cudaStreamSynchronize(stream));
      const auto actual = blocks.Download();
      const float actual_score = score.Download()[kPrefix];
      if ((expected_flags | (reuse ? 0x40404040u : 0u)) != Flags())
        throw std::runtime_error("flags differ reuse=" + std::to_string(reuse));
      if (std::memcmp(actual.data(), expected.data(),
                      actual.size() * sizeof(float))) {
        for (size_t i = 0; i < actual.size(); ++i)
          if (std::memcmp(&actual[i], &expected[i], 4))
            std::cerr << "BLOCK mismatch index=" << i
                      << " expected=" << std::hexfloat << expected[i]
                      << " actual=" << actual[i] << std::defaultfloat << '\n';
        throw std::runtime_error("block map differs reuse=" +
                                 std::to_string(reuse));
      }
      if (!(std::isnan(expected_score) && std::isnan(actual_score)) &&
          std::memcmp(&expected_score, &actual_score, 4)) {
        std::cerr << "SCORE expected=" << std::hexfloat << expected_score
                  << " actual=" << actual_score << std::defaultfloat << '\n';
        throw std::runtime_error("score differs reuse=" +
                                 std::to_string(reuse));
      }
      Same(main.host, main.Download());
      Same(sub.host, sub.Download());
      Same(anchors.host, anchors.Download());
      (void)maxima.Download();
      (void)partial_a.Download();
      (void)partial_b.Download();
    }
    return 2;
  }
};
} // namespace

int main(int argc, char **argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    return 77;
  try {
    const bool sanitizer = argc == 2 && std::string(argv[1]) == "--sanitizer";
    size_t cases = 0;
    const std::vector<std::pair<unsigned, unsigned>> sizes{
        {1, 1}, {7, 3}, {15, 17}, {33, 35}, {65, 67}, {127, 129}};
    for (auto [w, h] : sizes)
      for (unsigned strategy = 0; strategy < 8; ++strategy)
        for (unsigned pad : {0u, 7u}) {
          AqComposeProbe probe(w, h, strategy, pad);
          // Restore finite input after exceptional values on the same
          // allocation.
          for (unsigned pattern = 0; pattern < 11; ++pattern)
            cases += probe.Verify(pattern % 10);
        }
    if (!sanitizer)
      for (auto [w, h] : std::vector<std::pair<unsigned, unsigned>>{
               {500, 500}, {1919, 1079}, {3839, 2159}})
        for (unsigned strategy : {0u, 2u, 7u})
          for (unsigned pad : {0u, 7u}) {
            AqComposeProbe probe(w, h, strategy, pad);
            cases += probe.Verify(0);
          }
    Check(gjxl::cuda_internal::LaunchCudaAqComposeReduction({}, nullptr));
    std::cout << "CUDA AQ compose PASS cases=" << cases << '\n' << std::flush;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CUDA AQ compose ERROR " << error.what() << '\n';
    return 1;
  }
}
