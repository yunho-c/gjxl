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
struct Params {
  uint32_t width, height, input_stride, output_stride;
};
struct Fixture {
  Params params;
  std::vector<Buffer> buffers;
  Fixture(MTL::Device *device, unsigned w, unsigned h, unsigned pattern)
      : params{w, h, w + 19, w + 23} {
    for (unsigned i = 0; i < 3; ++i)
      buffers.emplace_back(device, size_t(params.input_stride) * h * 4);
    buffers.emplace_back(device, 33 * 4);
    for (unsigned i = 0; i < 6; ++i)
      buffers.emplace_back(device, size_t(params.output_stride) * h * 4);
    for (int i = 0; i < 33; ++i)
      buffers[3].as<float>()[i] =
          float(std::exp(-double((i - 16) * (i - 16)) /
                         (2.0 * 7.15593339443 * 7.15593339443)));
    uint32_t rng = 0x17937391;
    for (unsigned c = 0; c < 3; ++c)
      for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          float v = float(int(rng % 2001) - 1000) * 0.00003125f;
          if (pattern == 1)
            v = 0.25f;
          if (pattern == 2)
            v = (x == w / 2 && y == h / 2) ? 4.0f : 0.0f;
          if (pattern == 3)
            v *= 1.0e38f;
          buffers[c].as<float>()[y * params.input_stride + x] = v;
        }
    if (pattern == 4)
      buffers[0].as<float>()[0] = std::numeric_limits<float>::infinity();
    if (pattern == 5)
      buffers[1].as<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  }
  double run(MTL::CommandQueue *queue, const Kernel &kernel, unsigned pixels,
             unsigned repeats = 1) {
    auto cb = NS::RetainPtr(queue->commandBuffer());
    auto e = NS::RetainPtr(cb->computeCommandEncoder());
    e->setComputePipelineState(kernel.pipeline.get());
    for (unsigned i = 0; i < buffers.size(); ++i)
      e->setBuffer(buffers[i].object.get(), kGuard, i);
    e->setBytes(&params, sizeof(params), 10);
    e->setThreadgroupMemoryLength(18432, 0);
    for (unsigned i = 0; i < repeats; ++i)
      e->dispatchThreadgroups(
          MTL::Size((params.width + 15) / 16, (params.height + 63) / 64, 1),
          MTL::Size(16, 64 / pixels, 1));
    e->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    Check(cb->status() == MTL::CommandBufferStatusCompleted,
          "filter dispatch failed");
    return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e9 / repeats;
  }
  void compare(const Fixture &other) const {
    for (unsigned i = 0; i < buffers.size(); ++i) {
      buffers[i].guards();
      other.buffers[i].guards();
      if (std::memcmp(buffers[i].data(), other.buffers[i].data(),
                      buffers[i].size)) {
        size_t offset = 0;
        const auto *a = buffers[i].as<unsigned char>();
        const auto *b = other.buffers[i].as<unsigned char>();
        while (offset < buffers[i].size && a[offset] == b[offset])
          ++offset;
        throw std::runtime_error("filter mismatch buffer=" + std::to_string(i) +
                                 " byte=" + std::to_string(offset));
      }
    }
  }
};
struct MaltaParams {
  uint32_t width, height, reference_stride, distorted_stride, response_stride,
      accumulation_stride, low_frequency, initialize_accumulation,
      write_response;
  float norm2_0_gt_1, norm2_0_lt_1, norm;
};
struct MaltaFixture {
  MaltaParams params;
  std::vector<Buffer> buffers;
  MaltaFixture(MTL::Device *device, unsigned w, unsigned h, unsigned flags,
               unsigned pattern)
      : params{w,
               h,
               w + 7,
               w + 11,
               w + 17,
               w + 23,
               flags & 1u,
               (flags >> 1) & 1u,
               (flags >> 2) & 1u,
               0.75f,
               0.91f,
               0.3f} {
    for (unsigned stride : {params.reference_stride, params.distorted_stride,
                            params.response_stride, params.accumulation_stride})
      buffers.emplace_back(device, size_t(stride) * h * 4);
    uint32_t rng = 0x92421853;
    for (unsigned c : {0u, 1u, 3u}) {
      const unsigned stride = c == 0   ? params.reference_stride
                              : c == 1 ? params.distorted_stride
                                       : params.accumulation_stride;
      for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          float v = float(int(rng % 2001) - 1000) * 0.00003125f;
          if (pattern == 1)
            v = 0.25f;
          if (pattern == 2)
            v = (x == w / 2 && y == h / 2) ? 4.0f : 0.0f;
          if (pattern == 3)
            v *= 1.0e38f;
          buffers[c].as<float>()[y * stride + x] = v;
        }
    }
    if (pattern == 4)
      buffers[0].as<float>()[0] = std::numeric_limits<float>::infinity();
    if (pattern == 5)
      buffers[1].as<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  }
  void run(MTL::CommandQueue *queue, const Kernel &kernel) {
    auto cb = NS::RetainPtr(queue->commandBuffer());
    auto e = NS::RetainPtr(cb->computeCommandEncoder());
    e->setComputePipelineState(kernel.pipeline.get());
    for (unsigned i = 0; i < buffers.size(); ++i)
      e->setBuffer(buffers[i].object.get(), kGuard, i);
    e->setBytes(&params, sizeof(params), 4);
    e->setThreadgroupMemoryLength(40 * 16 * sizeof(float), 0);
    e->dispatchThreadgroups(
        MTL::Size((params.width + 31) / 32, (params.height + 7) / 8, 1),
        MTL::Size(32, 8, 1));
    e->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    Check(cb->status() == MTL::CommandBufferStatusCompleted,
          "Malta dispatch failed");
  }
  void compare(const MaltaFixture &other) const {
    for (unsigned i = 0; i < buffers.size(); ++i) {
      buffers[i].guards();
      other.buffers[i].guards();
      Check(std::memcmp(buffers[i].data(), other.buffers[i].data(),
                        buffers[i].size) == 0,
            "Malta mismatch buffer=" + std::to_string(i));
    }
    if (!params.write_response) {
      const auto *data = buffers[2].as<unsigned char>();
      Check(std::all_of(data, data + buffers[2].size,
                        [](unsigned char x) { return x == 0xa5; }),
            "Malta response suppression failed");
    }
  }
};
} // namespace
int main(int argc, char **argv) try {
  Check(argc >= 3, "usage: gjxl_metal_butteraugli_filter_probe BASE.metallib "
                   "CAND.metallib [--timing] [--malta] [--extent WxH]");
  bool timing = false;
  bool malta = false;
  std::vector<std::pair<unsigned, unsigned>> sizes;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--timing")
      timing = true;
    else if (arg == "--malta")
      malta = true;
    else if (arg == "--extent" && i + 1 < argc) {
      const std::string v = argv[++i];
      const auto pos = v.find('x');
      Check(pos != std::string::npos, "expected WxH");
      const auto w = std::stoul(v.substr(0, pos)),
                 h = std::stoul(v.substr(pos + 1));
      Check(w >= 1 && h >= 1 && w <= 4096 && h <= 4096,
            "extent outside probe limits");
      sizes.emplace_back(w, h);
    } else
      throw std::runtime_error("invalid option " + arg);
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
  if (malta) {
    Check(!timing,
          "Malta probe supports parity only; use integrated stage timing");
    Kernel base(device.get(), argv[1], "gjxl_butteraugli_malta_fused_f32");
    Kernel cand(device.get(), argv[2], "gjxl_butteraugli_malta_fixed_f32");
    unsigned cases = 0;
    for (auto [w, h] : sizes)
      for (unsigned flags = 0; flags < 8; ++flags)
        for (unsigned pattern = 0; pattern < 6; ++pattern) {
          MaltaFixture a(device.get(), w, h, flags, pattern),
              b(device.get(), w, h, flags, pattern);
          a.run(queue.get(), base);
          b.run(queue.get(), cand);
          try {
            a.compare(b);
          } catch (const std::exception &e) {
            throw std::runtime_error(
                "Malta " + std::to_string(w) + "x" + std::to_string(h) +
                " flags=" + std::to_string(flags) +
                " pattern=" + std::to_string(pattern) + ": " + e.what());
          }
          ++cases;
        }
    std::cout << "{\"bitwise_cases\":" << cases << ",\"guards\":true}\n";
    return 0;
  }
  Kernel base(device.get(), argv[1],
              "gjxl_butteraugli_frequency_low_medium_tiled_f32");
  unsigned cases = 0;
  for (unsigned pixels : {1u, 2u, 4u})
    for (const std::string space : {"device", "constant"}) {
      const std::string name = "gjxl_butteraugli_low_medium_p" +
                               std::to_string(pixels) + "_" + space;
      Kernel cand(device.get(), argv[2], name.c_str());
      for (auto [w, h] : sizes)
        for (unsigned pattern = 0; pattern < (timing ? 1u : 6u); ++pattern) {
          Fixture a(device.get(), w, h, pattern),
              b(device.get(), w, h, pattern);
          a.run(queue.get(), base, 1);
          b.run(queue.get(), cand, pixels);
          try {
            a.compare(b);
          } catch (const std::exception &e) {
            throw std::runtime_error(name + " extent=" + std::to_string(w) +
                                     "x" + std::to_string(h) + " pattern=" +
                                     std::to_string(pattern) + ": " + e.what());
          }
          ++cases;
          if (timing)
            for (unsigned pair = 0; pair < 5; ++pair)
              for (unsigned side = 0; side < 2; ++side) {
                const bool candidate = (pair + side) % 2;
                for (unsigned sample = 0; sample < 11; ++sample) {
                  const double ns = candidate
                                        ? b.run(queue.get(), cand, pixels, 3)
                                        : a.run(queue.get(), base, 1, 3);
                  if (sample >= 4)
                    std::cout
                        << "{\"variant\":\"" << name << "\",\"width\":" << w
                        << ",\"height\":" << h << ",\"pair\":" << pair
                        << ",\"candidate\":" << (candidate ? "true" : "false")
                        << ",\"sample\":" << sample - 4 << ",\"gpu_ns\":" << ns
                        << "}\n";
                }
              }
          a.compare(b);
        }
    }
  if (!timing)
    std::cout << "{\"bitwise_cases\":" << cases << ",\"guards\":true}\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
