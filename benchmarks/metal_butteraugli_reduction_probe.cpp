// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "metal_probe.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using namespace gjxl::benchmark;
// Shader ABI, deliberately independent of the production dispatch selection.
struct Params {
  uint32_t source_width, source_height, work_stride, sub_stride, block_stride;
  uint32_t anchor_offset, anchor_count, pixel_width, pixel_height;
  uint32_t covered_width, covered_height;
  float x_multiplier, asymmetry;
};
static_assert(sizeof(Params) == 52);
struct Anchor {
  uint32_t x, y;
};
struct Fixture {
  Params params;
  std::vector<Buffer> buffers;
  Fixture(MTL::Device *device, unsigned width, unsigned height, unsigned pw,
          unsigned ph, unsigned pattern) {
    const unsigned columns = (width + pw - 1) / pw;
    const unsigned rows = (height + ph - 1) / ph;
    params = {width,
              height,
              width + 19,
              (width + 1) / 2 + 7,
              columns * (pw / 8) + 5,
              3,
              columns * rows,
              pw,
              ph,
              pw / 8,
              ph / 8,
              0.87f,
              0.93f};
    for (unsigned i = 0; i < 21; ++i)
      buffers.emplace_back(device, size_t(params.work_stride) * height * 4);
    buffers.emplace_back(device,
                         size_t(params.sub_stride) * ((height + 1) / 2) * 4);
    buffers.emplace_back(device, (params.anchor_count + 8) * sizeof(Anchor));
    buffers.emplace_back(device,
                         size_t(params.block_stride) * rows * (ph / 8) * 4);
    buffers.emplace_back(device, (params.anchor_count + 8) * 4);
    buffers.emplace_back(device, 4);
    uint32_t rng = 0x71834213;
    for (unsigned c = 0; c < 21; ++c)
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          float value = float(rng % 2001) * 0.000001f;
          if (c >= 18)
            value *= 50.0f;
          if (pattern == 1)
            value = 0.0f;
          if (pattern == 2)
            value = x == width / 2 && y == height / 2 ? 0.1f : 0.0f;
          if (pattern == 5)
            value *= 1.0e10f;
          buffers[c].as<float>()[y * params.work_stride + x] = value;
        }
    for (unsigned y = 0; y < (height + 1) / 2; ++y)
      for (unsigned x = 0; x < (width + 1) / 2; ++x)
        buffers[21].as<float>()[y * params.sub_stride + x] =
            pattern == 1 ? 0.0f : float((x + 13 * y) % 111) * 0.00001f;
    for (unsigned i = 0; i < params.anchor_count; ++i) {
      const unsigned spatial = params.anchor_count - 1 - i;
      buffers[22].as<Anchor>()[params.anchor_offset + i] = {
          spatial % columns * (pw / 8), spatial / columns * (ph / 8)};
    }
    if (pattern == 3)
      buffers[21].as<float>()[0] = -1000.0f;
    if (pattern == 4)
      buffers[0].as<float>()[0] = std::numeric_limits<float>::infinity();
    if (pattern == 6)
      buffers[22].as<Anchor>()[params.anchor_offset] = {width, height};
    if (pattern == 8)
      params.anchor_count = 0;
    buffers[25].as<uint32_t>()[0] = pattern == 7 ? 16u : 0u;
  }
  double run(MTL::CommandQueue *queue, const Kernel &kernel, unsigned threads,
             unsigned repeats = 1) {
    auto command = NS::RetainPtr(queue->commandBuffer());
    auto encoder = NS::RetainPtr(command->computeCommandEncoder());
    encoder->setComputePipelineState(kernel.pipeline.get());
    for (unsigned i = 0; i < buffers.size(); ++i)
      encoder->setBuffer(buffers[i].object.get(), kGuard, i);
    encoder->setBytes(&params, sizeof(params), 26);
    for (unsigned i = 0; i < repeats; ++i)
      encoder->dispatchThreadgroups(MTL::Size(params.anchor_count + 2, 1, 1),
                                    MTL::Size(threads, 1, 1));
    encoder->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    Check(command->status() == MTL::CommandBufferStatusCompleted,
          "metric dispatch failed");
    return (command->GPUEndTime() - command->GPUStartTime()) * 1e9 / repeats;
  }
  void compare(const Fixture &other) const {
    for (unsigned i = 0; i < buffers.size(); ++i) {
      buffers[i].guards();
      other.buffers[i].guards();
      if (std::memcmp(buffers[i].data(), other.buffers[i].data(),
                      buffers[i].size)) {
        const auto *a = buffers[i].as<unsigned char>();
        const auto *b = other.buffers[i].as<unsigned char>();
        size_t byte = 0;
        while (byte < buffers[i].size && a[byte] == b[byte])
          ++byte;
        throw std::runtime_error("metric mismatch buffer=" + std::to_string(i) +
                                 " byte=" + std::to_string(byte));
      }
    }
  }
};
} // namespace
int main(int argc, char **argv) try {
  Check(argc >= 3, "usage: gjxl_metal_butteraugli_reduction_probe "
                   "BASE.metallib CAND.metallib [--timing] [--extent WxH]");
  bool timing = false;
  std::vector<std::pair<unsigned, unsigned>> sizes;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--timing")
      timing = true;
    else if (arg == "--extent" && i + 1 < argc) {
      const std::string value = argv[++i];
      const auto pos = value.find('x');
      Check(pos != std::string::npos, "expected WxH");
      const auto w = std::stoul(value.substr(0, pos));
      const auto h = std::stoul(value.substr(pos + 1));
      Check(w >= 1 && h >= 1 && w <= 4096 && h <= 4096,
            "extent outside probe limits");
      sizes.emplace_back(w, h);
    } else
      throw std::runtime_error("unknown option: " + arg);
  }
  if (sizes.empty())
    sizes = timing ? std::vector<std::pair<unsigned, unsigned>>{{512, 512},
                                                                {3839, 2159}}
                   : std::vector<std::pair<unsigned, unsigned>>{
                         {1, 1}, {7, 9}, {17, 31}, {79, 67}, {128, 96}};
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  Check(bool(device), "Metal device");
  auto queue = NS::TransferPtr(device->newCommandQueue());
  Kernel baseline(device.get(), argv[1],
                  "gjxl_butteraugli_resident_l2_reduce_f32");
  unsigned cases = 0;
  for (const unsigned threads : {64u, 128u, 256u})
    for (const std::string mode : {"tree", "simd"}) {
      const std::string name = "gjxl_butteraugli_resident_l2_reduce_w" +
                               std::to_string(threads) + "_" + mode;
      Kernel candidate(device.get(), argv[2], name.c_str());
      if (mode == "simd")
        Check(candidate.pipeline->threadExecutionWidth() == 32,
              "shuffle requires 32 lanes");
      for (auto [pw, ph] :
           std::array<std::pair<unsigned, unsigned>, 7>{{{8, 8},
                                                         {16, 8},
                                                         {8, 16},
                                                         {16, 16},
                                                         {32, 16},
                                                         {16, 32},
                                                         {32, 32}}}) {
        if (threads < 256 && pw * ph > threads)
          continue;
        for (auto [width, height] : sizes)
          for (unsigned pattern = 0; pattern < (timing ? 1u : 9u); ++pattern) {
            Fixture a(device.get(), width, height, pw, ph, pattern),
                b(device.get(), width, height, pw, ph, pattern);
            a.run(queue.get(), baseline, 256);
            b.run(queue.get(), candidate, threads);
            try {
              a.compare(b);
            } catch (const std::exception &e) {
              throw std::runtime_error(
                  name + " block=" + std::to_string(pw) + "x" +
                  std::to_string(ph) + " extent=" + std::to_string(width) +
                  "x" + std::to_string(height) +
                  " pattern=" + std::to_string(pattern) + ": " + e.what());
            }
            const auto error = a.buffers[25].as<uint32_t>()[0];
            if (pattern <= 2 || pattern == 8)
              Check(error == 0, "unexpected metric error");
            if (pattern == 3 || pattern == 4)
              Check(error & 128u, "missing invalid metric error");
            if (pattern == 5)
              Check(error & 256u, "missing overflow error");
            if (pattern == 6)
              Check(error & 64u, "missing invalid anchor error");
            if (pattern == 7)
              Check(error == 16u, "existing error flag lost");
            ++cases;
            if (timing)
              for (unsigned pair = 0; pair < 5; ++pair)
                for (unsigned side = 0; side < 2; ++side) {
                  const bool cand = (pair + side) % 2;
                  for (unsigned sample = 0; sample < 11; ++sample) {
                    const double ns =
                        cand ? b.run(queue.get(), candidate, threads, 3)
                             : a.run(queue.get(), baseline, 256, 3);
                    if (sample >= 4)
                      std::cout
                          << "{\"variant\":\"" << name
                          << "\",\"block_width\":" << pw
                          << ",\"block_height\":" << ph
                          << ",\"width\":" << width << ",\"height\":" << height
                          << ",\"pair\":" << pair
                          << ",\"candidate\":" << (cand ? "true" : "false")
                          << ",\"sample\":" << sample - 4
                          << ",\"gpu_ns\":" << ns << "}\n";
                  }
                }
            a.compare(b);
          }
      }
    }
  if (!timing)
    std::cout << "{\"bitwise_cases\":" << cases << ",\"guards\":true}\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
