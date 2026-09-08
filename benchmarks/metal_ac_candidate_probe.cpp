// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "metal_probe.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace {
using namespace gjxl::benchmark;
struct Candidate {
  uint32_t x, y;
  float quant, entropy, cfl_x, cfl_b;
};
struct Params {
  uint32_t width, height, opsin_stride, mask_stride, quant_stride, candidates,
      count, transform_width, transform_height, blocks_x, blocks_y, blocks,
      quant_source;
  float loss, zeros, delta;
};
static_assert(sizeof(Params) == 64 && sizeof(Candidate) == 24);
struct Fixture {
  Params p;
  // X, Y, B, candidates, quant, matrices, mask, norm/cost, loss, rate,
  // coefficients.
  std::vector<Buffer> b;
  Fixture(MTL::Device *device, unsigned rows, unsigned cols, unsigned count,
          unsigned pattern, unsigned source, unsigned padding)
      : p{72,          56,   72 + padding, 75 + padding, 12,       count,
          rows * cols, cols, rows,         cols / 8,     rows / 8,
          rows * cols / 64,
          source,      0.9f, 1.2f,         0.7f} {
    for (size_t size : std::array<size_t, 11>{
             p.opsin_stride * p.height * 4, p.opsin_stride * p.height * 4,
             p.opsin_stride * p.height * 4, count * sizeof(Candidate),
             p.quant_stride * 7 * 4, 6 * p.count * 4,
             p.mask_stride * p.height * 4, count * 4, count * 3 * p.count * 4,
             count * 3 * 8, count * 3 * p.count * 4})
      b.emplace_back(device, size);
    uint32_t rng = 0x34f296d1;
    for (unsigned c = 0; c < 3; ++c)
      for (unsigned y = 0; y < p.height; ++y)
        for (unsigned x = 0; x < p.width; ++x) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          float value = float(int(rng % 2001) - 1000) / 4096.f;
          if (pattern == 1)
            value = 0.125f;
          if (pattern == 2)
            value = (x % 8 == 0 && y % 8 == 0) ? 4.f : 0.f;
          if (pattern == 3)
            value = (x + y) % 2 ? -0.5f : 0.5f;
          if (pattern == 4)
            value *= 1.e10f;
          if (pattern == 5 && x % 8 == 0 && y % 8 == 0)
            value = std::numeric_limits<float>::infinity();
          b[c].as<float>()[y * p.opsin_stride + x] = value + c * 0.03125f;
        }
    for (unsigned y = 0; y < p.height; ++y)
      for (unsigned x = 0; x < p.width; ++x)
        b[6].as<float>()[y * p.mask_stride + x] =
            pattern == 6 ? 0.f : 0.25f + float((x + y * 3) % 37) / 32.f;
    for (unsigned i = 0; i < p.quant_stride * 7; ++i)
      b[4].as<float>()[i] = 0.5f + float(i % 23) / 16.f;
    for (unsigned i = 0; i < p.count * 3; ++i) {
      const float v = 0.2f + float(i % 41) / 16.f;
      b[5].as<float>()[i] = v;
      b[5].as<float>()[p.count * 3 + i] = 1.f / v;
    }
    for (unsigned i = 0; i < count; ++i) {
      auto &c = b[3].as<Candidate>()[i];
      c = {
          i % ((p.width - cols) / 8 + 1), (i * 3) % ((p.height - rows) / 8 + 1),
          0.5f + (i % 13) / 8.f,          0.75f + (i % 7) / 16.f,
          (int(i % 9) - 4) / 8.f,         (int(i % 11) - 5) / 8.f};
      if (source == 1)
        b[7].as<float>()[i] = 0.75f + (i % 17) / 16.f;
      if (pattern == 8)
        c.x = p.width / 8;
      if (pattern == 9)
        c.quant = -0.5f;
      if (pattern == 7)
        c.cfl_x = std::numeric_limits<float>::quiet_NaN();
    }
  }
  void bind(MTL::ComputeCommandEncoder *e, unsigned buffer, unsigned index) {
    e->setBuffer(b[buffer].object.get(), kGuard, index);
  }
  void run(MTL::CommandQueue *queue, const Kernel &first, const Kernel *inverse,
           const Kernel *finalizer, unsigned inverse_workers = 0,
           bool grouped_forward = false) {
    auto command = NS::RetainPtr(queue->commandBuffer());
    auto e = NS::RetainPtr(command->computeCommandEncoder());
    const unsigned workers = p.transform_height / 8 * 32;
    if (finalizer) {
      e->setComputePipelineState(finalizer->pipeline.get());
      for (unsigned i = 0; i < 6; ++i)
        bind(e.get(), std::array<unsigned, 6>{8, 6, 3, 9, 7, 4}[i], i);
      e->setBytes(&p, sizeof(p), 6);
      e->dispatchThreads(MTL::Size(p.candidates, 1, 1), MTL::Size(64, 1, 1));
    } else if (inverse) {
      e->setComputePipelineState(first.pipeline.get());
      for (unsigned i = 0; i < 7; ++i)
        bind(e.get(), std::array<unsigned, 7>{0, 1, 2, 3, 10, 4, 7}[i], i);
      e->setBytes(&p, sizeof(p), 7);
      e->dispatchThreadgroups(
          MTL::Size(p.candidates * (grouped_forward ? 1 : 3), 1, 1),
          MTL::Size(workers * (grouped_forward ? 3 : 1), 1, 1));
      e->setComputePipelineState(inverse->pipeline.get());
      for (unsigned i = 0; i < 6; ++i)
        bind(e.get(), std::array<unsigned, 6>{10, 5, 3, 4, 8, 9}[i], i);
      e->setBytes(&p, sizeof(p), 6);
      bind(e.get(), 7, 7);
      bind(e.get(), 6, 8);
      for (unsigned i = 0; i < 3; ++i)
        e->setThreadgroupMemoryLength(p.count * 4, i);
      e->dispatchThreadgroups(
          MTL::Size(p.candidates * 3, 1, 1),
          MTL::Size(inverse_workers ? inverse_workers : workers, 1, 1));
    } else {
      e->setComputePipelineState(first.pipeline.get());
      for (unsigned i = 0; i < 10; ++i)
        bind(e.get(), i, i);
      e->setBytes(&p, sizeof(p), 10);
      e->dispatchThreadgroups(MTL::Size(p.candidates, 1, 1),
                              MTL::Size(workers * 3, 1, 1));
    }
    e->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    Check(command->status() == MTL::CommandBufferStatusCompleted,
          "AC candidate dispatch failed");
  }
  void run_forward(MTL::CommandQueue *queue, const Kernel &kernel,
                   bool grouped) {
    auto command = NS::RetainPtr(queue->commandBuffer());
    auto e = NS::RetainPtr(command->computeCommandEncoder());
    e->setComputePipelineState(kernel.pipeline.get());
    for (unsigned i = 0; i < 7; ++i)
      bind(e.get(), std::array<unsigned, 7>{0, 1, 2, 3, 10, 4, 7}[i], i);
    e->setBytes(&p, sizeof(p), 7);
    e->dispatchThreadgroups(
        MTL::Size(p.candidates * (grouped ? 1 : 3), 1, 1),
        MTL::Size(p.transform_height / 8 * 32 * (grouped ? 3 : 1), 1, 1));
    e->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    Check(command->status() == MTL::CommandBufferStatusCompleted,
          "AC forward dispatch failed");
  }
  void compare(const Fixture &other, bool baseline_is_fused,
               bool candidate_is_fused = true) const {
    for (size_t i = 0; i < b.size(); ++i) {
      b[i].guards();
      other.b[i].guards();
      if (i == 10 && candidate_is_fused) {
        for (size_t j = 0; j < other.b[i].size; ++j)
          Check(other.b[i].as<unsigned char>()[j] == 0xa5 &&
                    (!baseline_is_fused || b[i].as<unsigned char>()[j] == 0xa5),
                "Fused candidate wrote coefficient scratch");
        continue;
      }
      Check(std::memcmp(b[i].data(), other.b[i].data(), b[i].size) == 0,
            "AC candidate bitwise mismatch in buffer " + std::to_string(i));
    }
  }
};

// These paths retain separate forward and residual/inverse/loss dispatches.
// Tuned DCT32 and 16x32 use wider inverse launches than their forward DCTs.
int RunStagedReductions(MTL::Device *device, MTL::CommandQueue *queue,
                        const char *baseline, const char *candidate,
                        bool grouped_forward = false) {
  unsigned cases = 0;
  unsigned grouped_shapes = 0;
  for (const auto shape : std::array<std::array<unsigned, 3>, 4>{
           {{8, 8, 32}, {32, 32, 512}, {32, 16, 128}, {16, 32, 256}}}) {
    const unsigned rows = shape[0], cols = shape[1], workers = shape[2];
    const std::string prefix = "gjxl_ac_strategy_dct" + std::to_string(rows) +
                              (rows == cols ? "" : "x" + std::to_string(cols));
    const std::string inverse_name = prefix +
        (workers > rows / 8 * 32 ? "_residual_inverse_tuned_loss"
                                : "_residual_inverse_compact_loss");
    Kernel forward_base(device, baseline, (prefix + "_forward_fused").c_str());
    Kernel forward_new(device, candidate, (prefix + "_forward_fused").c_str());
    bool candidate_grouped = false;
    if (grouped_forward) {
      const std::string name = prefix + "_forward_grouped";
      auto function = NS::TransferPtr(forward_new.library->newFunction(
          NS::String::string(name.c_str(), NS::UTF8StringEncoding)));
      if (function) {
        forward_new = Kernel(device, candidate, name.c_str());
        candidate_grouped = true;
        ++grouped_shapes;
      }
    }
    const unsigned forward_workers = rows / 8 * 32;
    for (const Kernel *kernel : {&forward_base, &forward_new}) {
      const unsigned required_threads = forward_workers *
          (kernel == &forward_new && candidate_grouped ? 3 : 1);
      Check(kernel->pipeline->threadExecutionWidth() == 32 &&
                kernel->pipeline->maxTotalThreadsPerThreadgroup() >=
                    required_threads,
            "Forward launch exceeds pipeline SIMD/thread limits");
      Check(kernel->pipeline->staticThreadgroupMemoryLength() <=
                device->maxThreadgroupMemoryLength(),
            "Forward launch exceeds threadgroup storage limit");
    }
    Kernel inverse_base(device, baseline, inverse_name.c_str());
    Kernel inverse_new(device, candidate, inverse_name.c_str());
    Kernel finish_base(device, baseline, "gjxl_ac_strategy_cost_from_loss");
    Kernel finish_new(device, candidate, "gjxl_ac_strategy_cost_from_loss");
    const unsigned dynamic_bytes = 3 * rows * cols * sizeof(float);
    for (const Kernel *kernel : {&inverse_base, &inverse_new}) {
      Check(kernel->pipeline->threadExecutionWidth() == 32,
            "Staged reductions require 32-lane SIMD groups");
      Check(kernel->pipeline->maxTotalThreadsPerThreadgroup() >= workers,
            "Staged reduction launch exceeds pipeline limit");
      Check(kernel->pipeline->staticThreadgroupMemoryLength() + dynamic_bytes <=
                device->maxThreadgroupMemoryLength(),
            "Staged reduction exceeds threadgroup storage limit");
    }
    std::cout << "{\"rows\":" << rows << ",\"columns\":" << cols
              << ",\"forward_threads\":" << rows / 8 * 32
              << ",\"candidate_forward_threads\":"
              << forward_workers * (candidate_grouped ? 3 : 1)
              << ",\"forward_grouped\":" << candidate_grouped
              << ",\"baseline_forward_bytes\":"
              << forward_base.pipeline->staticThreadgroupMemoryLength()
              << ",\"candidate_forward_bytes\":"
              << forward_new.pipeline->staticThreadgroupMemoryLength()
              << ",\"inverse_threads\":" << workers
              << ",\"dynamic_threadgroup_bytes\":" << dynamic_bytes
              << ",\"baseline_threadgroup_bytes\":"
              << inverse_base.pipeline->staticThreadgroupMemoryLength()
              << ",\"static_threadgroup_bytes\":"
              << inverse_new.pipeline->staticThreadgroupMemoryLength() << "}\n";
    for (unsigned count : {1, 2, 7, 33, 257})
      for (unsigned pattern = 0; pattern < 10; ++pattern)
        for (unsigned source : {0, 1, 2})
          for (unsigned padding : {0, 19}) {
            Fixture a(device, rows, cols, count, pattern, source, padding);
            Fixture b(device, rows, cols, count, pattern, source, padding);
            if (grouped_forward) {
              a.run_forward(queue, forward_base, false);
              b.run_forward(queue, forward_new, candidate_grouped);
              a.compare(b, false, false);
            }
            a.run(queue, forward_base, &inverse_base, nullptr, workers);
            b.run(queue, forward_new, &inverse_new, nullptr, workers,
                  candidate_grouped);
            a.compare(b, false, false);
            a.run(queue, forward_base, nullptr, &finish_base);
            b.run(queue, forward_new, nullptr, &finish_new);
            a.compare(b, false, false);
            ++cases;
          }
  }
  Check(!grouped_forward || grouped_shapes != 0,
        "Grouped-forward probe found no grouped kernels");
  std::cout << "{\"cases\":" << cases << ",\"bitwise\":true,\"guards\":true}\n";
  return 0;
}
} // namespace

int main(int argc, char **argv) try {
  Check(argc == 3 ||
            (argc == 4 && (std::string_view(argv[3]) == "--fused-baseline" ||
                           std::string_view(argv[3]) == "--staged-reductions" ||
                           std::string_view(argv[3]) == "--grouped-forward")),
        "usage: gjxl_metal_ac_candidate_probe BASELINE.metallib "
        "CANDIDATE.metallib [--fused-baseline|--staged-reductions|--grouped-forward]");
  const bool fused_baseline = argc == 4 &&
                              std::string_view(argv[3]) == "--fused-baseline";
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  Check(bool(device), "Metal device unavailable");
  auto queue = NS::TransferPtr(device->newCommandQueue());
  if (argc == 4 && std::string_view(argv[3]) == "--grouped-forward")
    return RunStagedReductions(device.get(), queue.get(), argv[1], argv[2], true);
  if (argc == 4 && std::string_view(argv[3]) == "--staged-reductions")
    return RunStagedReductions(device.get(), queue.get(), argv[1], argv[2]);
  unsigned cases = 0;
  for (const auto shape :
       std::array<std::array<unsigned, 2>, 3>{{{16, 16}, {16, 8}, {8, 16}}}) {
    const unsigned rows = shape[0], cols = shape[1];
    const std::string prefix = "gjxl_ac_strategy_dct" + std::to_string(rows) +
                              (rows == cols ? "" : "x" + std::to_string(cols));
    Kernel forward(device.get(), argv[1], (prefix + "_forward_fused").c_str());
    Kernel inverse(device.get(), argv[1],
                   (prefix + "_residual_inverse_compact_loss").c_str());
    // Keep the split oracle as the default. The optional mode compares two
    // fused implementations directly, including their untouched scratch.
    std::optional<Kernel> previous;
    if (fused_baseline)
      previous.emplace(device.get(), argv[1],
                       (prefix + "_candidate_loss_parallel").c_str());
    Kernel fused(device.get(), argv[2],
                 (prefix + "_candidate_loss_parallel").c_str());
    Kernel finish_base(device.get(), argv[1],
                       "gjxl_ac_strategy_cost_from_loss");
    Kernel finish_new(device.get(), argv[2], "gjxl_ac_strategy_cost_from_loss");
    std::cout << "{\"rows\":" << rows << ",\"columns\":" << cols
              << ",\"threads\":" << rows / 8 * 96;
    if (previous)
      std::cout << ",\"baseline_threadgroup_bytes\":"
                << previous->pipeline->staticThreadgroupMemoryLength();
    std::cout << ",\"static_threadgroup_bytes\":"
              << fused.pipeline->staticThreadgroupMemoryLength() << "}\n";
    for (unsigned count : {1, 2, 7, 33, 257})
      for (unsigned pattern = 0; pattern < 10; ++pattern)
        for (unsigned source : {0, 1, 2})
          for (unsigned padding : {0, 19}) {
            Fixture a(device.get(), rows, cols, count, pattern, source,
                      padding);
            Fixture b(device.get(), rows, cols, count, pattern, source,
                      padding);
            a.run(queue.get(), previous ? *previous : forward,
                  previous ? nullptr : &inverse, nullptr);
            b.run(queue.get(), fused, nullptr, nullptr);
            // Also compare quant norm, per-channel rate and loss.
            a.compare(b, fused_baseline);
            a.run(queue.get(), forward, nullptr, &finish_base);
            b.run(queue.get(), fused, nullptr, &finish_new);
            a.compare(b, fused_baseline);
            ++cases;
          }
  }
  std::cout << "{\"cases\":" << cases << ",\"bitwise\":true,\"guards\":true}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
