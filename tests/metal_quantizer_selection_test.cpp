// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Exact median/MAD, quantizer, raw-quant and scratch comparison with guarded
// buffers.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
struct Params {
  uint32_t value_count, padded_count, median_index, width, height, stride,
      raw_stride, scaled_dc;
  float quant_dc;
};
struct Pass {
  uint32_t shift, deviation;
};
static_assert(sizeof(Params) == 36);
constexpr size_t kGuard = 16;
constexpr uint32_t kPoison = 0x7fc12345u;
int main() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "No Metal device available.\n";
    return 77;
  }
  NS::Error *error = nullptr;
  auto library = NS::TransferPtr(device->newLibrary(
      NS::String::string(GJXL_METALLIB_PATH, NS::UTF8StringEncoding), &error));
  if (!library) {
    std::cerr << error->localizedDescription()->utf8String() << '\n';
    return 3;
  }
  auto queue = NS::TransferPtr(device->newCommandQueue());
  const char *names[] = {"gjxl_aq_resident_quant_select_initialize",
                         "gjxl_aq_resident_quant_histogram",
                         "gjxl_aq_resident_quant_select_bucket",
                         "gjxl_aq_resident_quant_finalize_quantizer",
                         "gjxl_aq_resident_quant_small",
                         "gjxl_aq_initial_quant_raw_quant"};
  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 6> pipelines;
  for (size_t i = 0; i < 6; ++i) {
    auto fn = NS::TransferPtr(library->newFunction(
        NS::String::string(names[i], NS::UTF8StringEncoding)));
    if (!fn)
      return 4;
    pipelines[i] =
        NS::TransferPtr(device->newComputePipelineState(fn.get(), &error));
    if (!pipelines[i])
      return 5;
    if (pipelines[i]->maxTotalThreadsPerThreadgroup() < 256 ||
        (i == 4 && pipelines[i]->threadExecutionWidth() != 32)) {
      std::cerr << "Single-group quantizer is unavailable on this device.\n";
      return 77;
    }
  }
  const std::array<std::array<uint32_t, 2>, 24> extents{
      {{1, 1},     {2, 1},     {3, 1},     {7, 3},     {15, 17},   {16, 16},
       {257, 1},   {31, 33},   {63, 65},   {64, 96},   {8191, 1},  {128, 64},
       {8193, 1},  {64, 127},  {64, 129},  {127, 129}, {128, 128}, {129, 127},
       {129, 129}, {256, 128}, {257, 256}, {512, 256}, {513, 255}, {513, 256}}};
  size_t cases = 0;
  for (auto extent : extents)
    for (unsigned pattern = 0; pattern < 7; ++pattern)
      for (float dc : std::array<float, 4>{0.0f, 0.001f, 1.0f, 16.0f}) {
        Params p{extent[0] * extent[1],
                 extent[0] * extent[1],
                 extent[0] * extent[1] / 2,
                 extent[0],
                 extent[1],
                 extent[0] + 5,
                 extent[0] + 7,
                 uint32_t(double(dc * 4096.0f) * 1.6),
                 dc};
        const std::array<size_t, 7> sizes{
            p.stride * p.height, 2, 2, 1, 3, 256, p.raw_stride * p.height};
        std::array<std::vector<uint32_t>, 7> initial;
        for (size_t b = 0; b < 7; ++b)
          initial[b].assign(sizes[b] + 2 * kGuard, kPoison);
        initial[3][kGuard] = 8u;
        std::vector<float> logical;
        uint32_t rng = 0x73845162u;
        for (uint32_t y = 0; y < p.height; ++y)
          for (uint32_t x = 0; x < p.width; ++x) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            float value = 0.125f;
            if (pattern == 1)
              value = 0.0001f + float(rng % 100001) * 0.00013f;
            if (pattern == 2)
              value = std::bit_cast<float>(((100u + rng % 55u) << 23u) |
                                           (rng & 0x007fffffu));
            if (pattern == 3)
              value = float((x + 3 * y) % 7 + 1) * 0.25f;
            if (pattern == 4)
              value = std::bit_cast<float>(std::bit_cast<uint32_t>(0.3f) +
                                           (rng % 3u));
            if (pattern == 5)
              value = 0.0f;
            if (pattern == 6)
              value = -1.0f;
            initial[0][kGuard + y * p.stride + x] =
                std::bit_cast<uint32_t>(value);
            logical.push_back(value);
          }
        // Independent CPU order-statistic oracle; upper median for even
        // lengths.
        std::vector<float> sorted = logical;
        std::nth_element(sorted.begin(), sorted.begin() + p.median_index,
                         sorted.end());
        const float median = sorted[p.median_index];
        for (auto &v : sorted)
          v = std::abs(v - median);
        std::nth_element(sorted.begin(), sorted.begin() + p.median_index,
                         sorted.end());
        const float mad = sorted[p.median_index];
        std::array<std::array<std::vector<uint32_t>, 7>, 2> outputs;
        for (size_t side = 0; side < 2; ++side) {
          std::array<NS::SharedPtr<MTL::Buffer>, 7> b;
          for (size_t i = 0; i < 7; ++i)
            b[i] = NS::TransferPtr(
                device->newBuffer(initial[i].data(), initial[i].size() * 4,
                                  MTL::ResourceStorageModeShared));
          auto command = queue->commandBuffer();
          auto encoder = command->computeCommandEncoder();
          auto bind = [&](size_t buffer, size_t index) {
            encoder->setBuffer(b[buffer].get(), kGuard * 4, index);
          };
          auto launch = [&](uint32_t count) {
            encoder->dispatchThreadgroups(MTL::Size((count + 255) / 256, 1, 1),
                                          MTL::Size(256, 1, 1));
          };
          // Compare the established 19-dispatch path against the fused
          // selection.
          if (side == 0) {
            for (uint32_t deviation = 0; deviation < 2; ++deviation) {
              encoder->setComputePipelineState(pipelines[0].get());
              bind(4, 0);
              bind(5, 1);
              encoder->setBytes(&p, sizeof(p), 2);
              launch(256);
              for (uint32_t shift : std::array<uint32_t, 4>{24, 16, 8, 0}) {
                Pass pass{shift, deviation};
                encoder->setComputePipelineState(pipelines[1].get());
                bind(0, 0);
                bind(1, 1);
                bind(5, 2);
                bind(4, 3);
                encoder->setBytes(&p, sizeof(p), 4);
                encoder->setBytes(&pass, sizeof(pass), 5);
                launch(p.value_count);
                encoder->setComputePipelineState(pipelines[2].get());
                bind(5, 0);
                bind(4, 1);
                bind(1, 2);
                encoder->setBytes(&pass, sizeof(pass), 3);
                launch(256);
              }
            }
            encoder->setComputePipelineState(pipelines[3].get());
            bind(1, 0);
            bind(2, 1);
            bind(3, 2);
            encoder->setBytes(&p, sizeof(p), 3);
            launch(1);
          } else {
            encoder->setComputePipelineState(pipelines[4].get());
            bind(0, 0);
            bind(1, 1);
            bind(2, 2);
            bind(3, 3);
            encoder->setBytes(&p, sizeof(p), 4);
            bind(4, 5);
            bind(5, 6);
            launch(256);
          }
          encoder->setComputePipelineState(pipelines[5].get());
          bind(0, 0);
          bind(2, 1);
          bind(6, 2);
          bind(3, 3);
          encoder->setBytes(&p, sizeof(p), 4);
          encoder->dispatchThreadgroups(
              MTL::Size((p.width + 7) / 8, (p.height + 7) / 8, 1),
              MTL::Size(8, 8, 1));
          encoder->endEncoding();
          command->commit();
          command->waitUntilCompleted();
          if (command->status() != MTL::CommandBufferStatusCompleted) {
            std::cerr << command->error()->localizedDescription()->utf8String()
                      << '\n';
            return 6;
          }
          for (size_t i = 0; i < 7; ++i) {
            const auto *words = static_cast<const uint32_t *>(b[i]->contents());
            outputs[side][i].assign(words, words + initial[i].size());
            for (size_t j = 0; j < initial[i].size(); ++j) {
              // Inputs, guard regions, and row padding must remain untouched.
              const bool outside = j < kGuard || j >= kGuard + sizes[i];
              const bool padding =
                  i == 6 && !outside && (j - kGuard) % p.raw_stride >= p.width;
              if ((i == 0 || outside || padding) && words[j] != initial[i][j])
                return 7;
            }
          }
          if (outputs[side][1][kGuard] != std::bit_cast<uint32_t>(median) ||
              outputs[side][1][kGuard + 1] != std::bit_cast<uint32_t>(mad))
            return 8;
          if ((outputs[side][3][kGuard] & 8u) == 0u)
            return 9;
          if ((pattern >= 5 || dc == 0.0f) &&
              (outputs[side][3][kGuard] & 524288u) == 0u)
            return 10;
        }
        if (outputs[0] != outputs[1]) {
          std::cerr << "Mismatch " << p.width << 'x' << p.height << " pattern "
                    << pattern << " dc " << dc << '\n';
          return 11;
        }
        ++cases;
      }
  std::cout << "Passed " << cases
            << " exact guarded quantizer cases, including CPU median/MAD and "
               "invalid selected-value/DC checks.\n";
}
