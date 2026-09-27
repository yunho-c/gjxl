// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "metal_probe.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
using namespace gjxl::benchmark;
struct Params {
  uint32_t w, h, ix, iy, mx, my, hx, hy;
};
struct High {
  uint32_t w, h, in, medium, high, channel;
};
struct Plane {
  uint32_t w, h, in, out;
};
struct Fixture {
  Params p;
  std::vector<Buffer> b;
  std::array<std::vector<unsigned char>, 3> immutable;
  Fixture(MTL::Device *d, unsigned w, unsigned h, unsigned pattern)
      : p{w, h, w + 19, w + 23, w + 29, w + 31, w + 37, w + 41} {
    for (auto stride : {p.ix, p.iy})
      b.emplace_back(d, size_t(stride) * h * 4);
    b.emplace_back(d, 15 * 4);
    for (auto stride : {p.mx, p.my, p.hx, p.hy})
      b.emplace_back(d, size_t(stride) * h * 4);
    const float sigma = 3.22489901262f;
    const double exponent_scale = -1.0 / (2.0 * sigma * sigma);
    for (int i = 0; i < 15; ++i)
      b[2].as<float>()[i] = float(std::exp(exponent_scale * (i - 7) * (i - 7)));
    uint32_t rng = 0x38514f39;
    for (unsigned c = 0; c < 2; ++c)
      for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          float v = float(int(rng % 20001) - 10000) * 0.003125f;
          if (pattern == 1)
            v = c ? 0.5f : -0.25f;
          if (pattern == 2)
            v = (x == w / 2 && y == h / 2) ? 4.0f : 0;
          if (pattern == 3)
            v = ((x + y + c) & 1) ? 7.0f : -11.0f;
          if (pattern == 4)
            v = float(x) * 0.001f - float(y) * 0.003f + c;
          if (pattern == 5)
            v *= 1.0e20f;
          b[c].as<float>()[size_t(y) * (c ? p.iy : p.ix) + x] = v;
        }
    if (pattern == 6)
      b[0].as<float>()[0] = std::numeric_limits<float>::infinity();
    if (pattern == 7)
      b[1].as<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    for (unsigned i = 0; i < 3; ++i)
      immutable[i].assign(b[i].as<unsigned char>(),
                          b[i].as<unsigned char>() + b[i].size);
  }
  void encode(MTL::ComputeCommandEncoder *e, const std::array<Kernel *, 4> &k,
              unsigned mode, bool suppress) {
    if (mode == 0) {
      e->setComputePipelineState(k[0]->pipeline.get());
      for (unsigned c = 0; c < 2; ++c) {
        for (unsigned i = 0; i < 4; ++i) {
          unsigned bi = i == 0 ? c : i == 1 ? 2 : i == 2 ? 3 + c : 5 + c;
          e->setBuffer(b[bi].object.get(), kGuard, i);
        }
        High h{p.w, p.h, c ? p.iy : p.ix, c ? p.my : p.mx, c ? p.hy : p.hx, c};
        e->setBytes(&h, sizeof(h), 4);
        e->setThreadgroupMemoryLength(4992, 0);
        e->dispatchThreadgroups(MTL::Size((p.w + 15) / 16, (p.h + 63) / 64, 1),
                                MTL::Size(16, 16, 1));
      }
    } else {
      e->setComputePipelineState(k[mode == 1 ? 1 : 2]->pipeline.get());
      for (unsigned i = 0; i < 7; ++i)
        e->setBuffer(b[i].object.get(), kGuard, i);
      e->setBytes(&p, sizeof(p), 7);
      e->setThreadgroupMemoryLength(mode == 1 ? 4992 : 9984, 0);
      e->dispatchThreadgroups(
          MTL::Size((p.w + 15) / 16, (p.h + 63) / 64, mode == 1 ? 2 : 1),
          MTL::Size(16, 16, 1));
    }
    if (suppress && mode != 3) {
      e->setComputePipelineState(k[3]->pipeline.get());
      e->setBuffer(b[5].object.get(), kGuard, 0);
      e->setBuffer(b[6].object.get(), kGuard, 1);
      Plane v{p.w, p.h, p.hy, p.hx};
      e->setBytes(&v, sizeof(v), 2);
      e->dispatchThreads(MTL::Size(p.w, p.h, 1), MTL::Size(16, 16, 1));
    }
  }
  double run(MTL::CommandQueue *q, const std::array<Kernel *, 4> &k,
             unsigned mode, bool suppress, unsigned repeats = 1) {
    auto cb = NS::RetainPtr(q->commandBuffer());
    auto e = NS::RetainPtr(cb->computeCommandEncoder());
    for (unsigned i = 0; i < repeats; ++i)
      encode(e.get(), k, mode, suppress);
    e->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    Check(cb->status() == MTL::CommandBufferStatusCompleted,
          "HighXY dispatch failed");
    return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e9 / repeats;
  }
  void compare(const Fixture &o) {
    for (unsigned i = 0; i < 7; ++i) {
      b[i].guards();
      o.b[i].guards();
      if (i < 3) {
        Check(std::memcmp(b[i].data(), immutable[i].data(), b[i].size) == 0,
              "Input/weight modified");
        Check(std::memcmp(o.b[i].data(), o.immutable[i].data(), o.b[i].size) ==
                  0,
              "Candidate input/weight modified");
      }
      if (std::memcmp(b[i].data(), o.b[i].data(), b[i].size)) {
        size_t at = 0;
        auto *a = b[i].as<unsigned char>();
        auto *z = o.b[i].as<unsigned char>();
        while (at < b[i].size && a[at] == z[at])
          ++at;
        throw std::runtime_error("HighXY mismatch buffer=" + std::to_string(i) +
                                 " byte=" + std::to_string(at));
      }
      if (i >= 3) {
        unsigned stride = i == 3 ? p.mx : i == 4 ? p.my : i == 5 ? p.hx : p.hy;
        for (unsigned y = 0; y < p.h; ++y)
          for (unsigned x = p.w * 4; x < stride * 4; ++x)
            Check(b[i].as<unsigned char>()[size_t(y) * stride * 4 + x] == 0xa5,
                  "Row padding modified");
      }
    }
  }
};
int main(int argc, char **argv) try {
  Check(argc == 2, "usage metal_high_xy_test METALLIB");
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto d = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!d || !d->name() ||
      std::string(d->name()->utf8String()) != "Apple M4 Pro") {
    std::cout << "SKIP: fusion qualification is restricted to M4 Pro\n";
    return 0;
  }
  auto q = NS::TransferPtr(d->newCommandQueue());
  Kernel a(d.get(), argv[1], "gjxl_butteraugli_high_shared_f32"),
      c(d.get(), argv[1], "gjxl_butteraugli_high_xy_suppress_f32"),
      s(d.get(), argv[1], "gjxl_butteraugli_frequency_suppress_x_f32");
  std::array<Kernel *, 4> k{&a, &c, &c, &s};
  unsigned cases = 0;
  for (auto [w, h] : std::vector<std::pair<unsigned, unsigned>>{{1, 1},
                                                                {2, 3},
                                                                {7, 9},
                                                                {15, 15},
                                                                {16, 64},
                                                                {17, 65},
                                                                {31, 63},
                                                                {79, 67},
                                                                {128, 96},
                                                                {193, 131}})
    for (unsigned pattern = 0; pattern < 8; ++pattern)
      for (unsigned mode = 3; mode < 4; ++mode)
        for (bool suppress : {false, true}) {
          if (mode == 3 && !suppress)
            continue;
          Fixture base(d.get(), w, h, pattern), cand(d.get(), w, h, pattern);
          base.run(q.get(), k, 0, suppress);
          cand.run(q.get(), k, mode, suppress);
          try {
            base.compare(cand);
          } catch (const std::exception &e) {
            throw std::runtime_error(
                std::to_string(w) + "x" + std::to_string(h) +
                " pattern=" + std::to_string(pattern) +
                " mode=" + std::to_string(mode) + ": " + e.what());
          }
          ++cases;
        }
  std::cout << "{\"bitwise_cases\":" << cases
            << ",\"guards\":true,\"padding\":true}\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
