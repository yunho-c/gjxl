// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

// Independent metallib comparison: only the coefficient dispatch is timed.
// Inputs, guarded output initialization, readback comparisons and library setup
// are outside GPU timestamps. Both libraries receive identical input bytes.
#include "codec/quantization.h"
#include "gpu/metal/metal_aq_evaluation_internal.h"
#include "metal_probe.h"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Params = gjxl::metal_internal::AqReconstructionParams;
using gjxl::benchmark::Buffer;
using gjxl::benchmark::Check;
using gjxl::benchmark::Kernel;
using gjxl::benchmark::kGuard;
struct Fixture {
  Params params{};
  std::vector<Buffer> buffers;
  Fixture(MTL::Device *device, gjxl::AcStrategyType strategy, size_t width,
          size_t height, bool adjusted, bool group_major, unsigned pattern) {
    const auto &info = *gjxl::GetAcStrategyInfo(strategy);
    const auto shape = info.pixel_extent();
    width = (width + shape.width - 1) / shape.width * shape.width;
    height = (height + shape.height - 1) / shape.height * shape.height;
    const size_t nx = width / shape.width, ny = height / shape.height;
    const size_t anchors = nx * ny, count = info.coefficient_count();
    const size_t blocks = width * height / 64;
    const size_t tiles = ((width + 63) / 64) * ((height + 63) / 64);
    // Sparse group-major destinations exercise channel stride and unused slots.
    const size_t quantized_count =
        group_major ? anchors * 3 * 65536 : anchors * count * 3;
    const std::array<size_t, 17> sizes = {anchors * 8,
                                          11904 * 4,
                                          blocks * 4,
                                          tiles,
                                          tiles,
                                          anchors * count * 3 * 4,
                                          quantized_count * 4,
                                          anchors * count * 3 * 4,
                                          blocks * 3 * 4,
                                          blocks * 3 * 4,
                                          4,
                                          sizeof(Params),
                                          blocks * 4,
                                          blocks,
                                          8,
                                          anchors * 4 * 4,
                                          anchors * 4};
    for (auto size : sizes)
      buffers.emplace_back(device, size);
    params = {.coding_width = uint32_t(width),
              .coding_height = uint32_t(height),
              .coding_stride = uint32_t(width),
              .block_width = uint32_t(width / 8),
              .block_height = uint32_t(height / 8),
              .raw_quant_stride = uint32_t(width / 8),
              .color_width = uint32_t((width + 63) / 64),
              .color_stride = uint32_t((width + 63) / 64),
              .anchor_offset = 0,
              .anchor_count = uint32_t(anchors),
              .coefficient_offset = 0,
              .coefficient_count = uint32_t(count),
              .pixel_width = uint32_t(shape.width),
              .pixel_height = uint32_t(shape.height),
              .covered_width = uint32_t(shape.width / 8),
              .covered_height = uint32_t(shape.height / 8),
              .strategy = uint32_t(strategy),
              .global_scale = 8192,
              .quant_dc = 48,
              .x_matrix_multiplier = 0.8125f,
              .b_matrix_multiplier = 1.125f,
              .adjust_ac_quant = uint32_t(adjusted),
              .inverse_sigma_stride = uint32_t(width / 8),
              .epf_sharpness_stride = uint32_t(width / 8),
              .epf_quant_multiplier = 1.5f,
              .epf_sharpness_lut = {0.0f, 0.2f, 0.4f, 0.5f, 0.6f, 0.7f, 0.9f, 1.0f},
              .use_resident_quantizer = uint32_t(pattern % 2),
              .group_major_output = uint32_t(group_major)};
    auto *points = buffers[0].as<uint32_t>();
    for (size_t a = 0; a < anchors; ++a) {
      points[2 * a] = uint32_t(a % nx * shape.width / 8);
      points[2 * a + 1] = uint32_t(a / nx * shape.height / 8);
      buffers[16].as<uint32_t>()[a] = uint32_t(a * 3 * 65536);
      for (size_t q = 0; q < 4; ++q)
        buffers[15].as<float>()[4 * a + q] = 0.5f + float(q) * 0.041f;
    }
    for (size_t i = 0; i < blocks; ++i) {
      buffers[2].as<int32_t>()[i] = 1 + int32_t((i * 37 + pattern * 19) % 255);
      buffers[13].as<uint8_t>()[i] = uint8_t(i % 8);
    }
    for (size_t i = 0; i < tiles; ++i) {
      buffers[3].as<int8_t>()[i] = int8_t(int(i % 61) - 30);
      buffers[4].as<int8_t>()[i] = int8_t(int(i % 43) - 21);
    }
    const size_t table_base = strategy == gjxl::AcStrategyType::kDct8       ? 0
                              : strategy == gjxl::AcStrategyType::kDct16x16 ? 384
                              : strategy == gjxl::AcStrategyType::kDct32x32 ? 1920
                              : count == 128                                ? 8064
                                                                            : 8832;
    for (size_t channel = 0; channel < 3; ++channel) {
      gjxl::QuantizationMatrixView matrix;
      Check(gjxl::GetDefaultQuantizationMatrix(strategy, gjxl::XybChannel(channel),
                                               &matrix)
                .ok(),
            "quant matrix");
      std::copy(matrix.dequant.begin(), matrix.dequant.end(),
                buffers[1].as<float>() + table_base + channel * count);
      std::copy(matrix.inverse_dequant.begin(), matrix.inverse_dequant.end(),
                buffers[1].as<float>() + table_base + 3 * count + channel * count);
    }
    uint32_t rng = 0x913ab45u + pattern;
    for (size_t i = 0; i < anchors * count * 3; ++i) {
      rng ^= rng << 13;
      rng ^= rng >> 17;
      rng ^= rng << 5;
      float value = float(int32_t(rng % 20001) - 10000) * 0.00003125f;
      if (pattern == 1)
        value = 0;
      if (pattern == 2)
        value *= 100.0f;
      buffers[5].as<float>()[i] = value;
    }
    if (pattern == 3)
      buffers[5].as<float>()[17] = std::numeric_limits<float>::infinity();
    if (pattern == 4)
      buffers[13].as<uint8_t>()[0] = 9;
    if (pattern == 5)
      buffers[5].as<float>()[count + 21] = std::numeric_limits<float>::quiet_NaN();
    if (pattern == 6)
      buffers[1].as<float>()[table_base + 19] = std::numeric_limits<float>::max();
    if (pattern == 7)
      buffers[5].as<float>()[18] = std::numeric_limits<float>::max();
    buffers[10].as<uint32_t>()[0] = 0;
    buffers[14].as<uint32_t>()[0] = params.global_scale;
    buffers[14].as<uint32_t>()[1] = params.quant_dc;
  }
  double run(MTL::CommandQueue *queue, const Kernel &kernel, unsigned repeats = 1,
             unsigned workers = 0) {
    auto command = NS::RetainPtr(queue->commandBuffer());
    auto encoder = NS::RetainPtr(command->computeCommandEncoder());
    encoder->setComputePipelineState(kernel.pipeline.get());
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (i == 11)
        encoder->setBytes(&params, sizeof(params), i);
      else
        encoder->setBuffer(buffers[i].object.get(), kGuard, i);
    }
    for (unsigned i = 0; i < repeats; ++i)
      encoder->dispatchThreadgroups(
          MTL::Size(params.anchor_count, 1, 1),
          MTL::Size(workers ? workers : std::min(256u, params.coefficient_count), 1,
                    1));
    encoder->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    Check(command->status() == MTL::CommandBufferStatusCompleted,
          "GPU submission failed");
    return (command->GPUEndTime() - command->GPUStartTime()) * 1e9 / repeats;
  }
  void compare(const Fixture &other, const std::string &mode = "full") const {
    for (size_t b = 0; b < buffers.size(); ++b) {
      buffers[b].guards();
      other.buffers[b].guards();
      if (b == 11)
        continue;
      if ((mode == "scored" && b == 6) || (mode == "final" && b == 7)) {
        const auto *bytes = other.buffers[b].as<unsigned char>();
        for (size_t i = 0; i < other.buffers[b].size; ++i) {
          if (bytes[i] != 0xa5)
            throw std::runtime_error("unrequested output was written");
        }
        continue;
      }
      if (std::memcmp(buffers[b].data(), other.buffers[b].data(), buffers[b].size)) {
        const auto *a = buffers[b].as<unsigned char>();
        const auto *c = other.buffers[b].as<unsigned char>();
        size_t i = 0;
        while (i < buffers[b].size && a[i] == c[i])
          ++i;
        throw std::runtime_error("bitwise mismatch buffer=" + std::to_string(b) +
                                 " byte=" + std::to_string(i));
      }
    }
  }
};
} // namespace
int main(int argc, char **argv) try {
  Check(argc >= 3,
        "usage: gjxl_metal_coefficient_probe BASELINE.metallib CANDIDATE.metallib "
        "[--timing] [--mode full|scored|final] [--threads N]");
  bool timing = false;
  std::string mode = "full";
  unsigned workers = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--timing")
      timing = true;
    else if (arg == "--mode" && i + 1 < argc)
      mode = argv[++i];
    else if (arg == "--threads" && i + 1 < argc)
      workers = unsigned(std::stoul(argv[++i]));
    else
      throw std::runtime_error("invalid option: " + arg);
  }
  Check(mode == "full" || mode == "scored" || mode == "final", "invalid mode");
  Check(workers <= 1024 && workers % 32 == 0, "workers must be a SIMD multiple <=1024");
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  Check(bool(device), "Metal device unavailable");
  auto queue = NS::TransferPtr(device->newCommandQueue());
  Kernel baseline(device.get(), argv[1]),
      candidate(device.get(), argv[2],
                mode == "scored"  ? "gjxl_aq_encode_scored_coefficients"
                : mode == "final" ? "gjxl_aq_encode_final_coefficients"
                                  : "gjxl_aq_encode_reconstruction_coefficients");
  const std::array strategies = {
      gjxl::AcStrategyType::kDct8,     gjxl::AcStrategyType::kDct16x16,
      gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
      gjxl::AcStrategyType::kDct8x16,  gjxl::AcStrategyType::kDct32x16,
      gjxl::AcStrategyType::kDct16x32};
  size_t cases = 0;
  for (auto strategy : strategies) {
    if (!timing) {
      for (size_t size : {size_t(32), size_t(96)})
        for (bool adjusted : {false, true})
          for (bool group : {false, true})
            for (unsigned pattern = 0; pattern < 8; ++pattern) {
              Fixture a(device.get(), strategy, size, size, adjusted, group, pattern);
              Fixture b(device.get(), strategy, size, size, adjusted, group, pattern);
              a.run(queue.get(), baseline);
              b.run(queue.get(), candidate, 1, workers);
              a.compare(b, mode);
              const auto error = a.buffers[10].as<uint32_t>()[0];
              const bool invalid = pattern == 3 || pattern == 5 || pattern == 7 ||
                                   (pattern == 4 && adjusted);
              if (pattern != 6)
                Check((error != 0) == invalid, "unexpected numeric validation result");
              ++cases;
            }
    } else {
      for (size_t size : {size_t(512), size_t(3840)}) {
        Fixture a(device.get(), strategy, size, size == 512 ? 512 : 2160, true, false,
                  0);
        Fixture b(device.get(), strategy, size, size == 512 ? 512 : 2160, true, false,
                  0);
        a.run(queue.get(), baseline);
        b.run(queue.get(), candidate, 1, workers);
        a.compare(b, mode);
        for (unsigned pair = 0; pair < 5; ++pair)
          for (unsigned side = 0; side < 2; ++side) {
            const bool use_candidate = (side + pair) % 2;
            auto &fixture = use_candidate ? b : a;
            const auto &kernel = use_candidate ? candidate : baseline;
            for (unsigned sample = 0; sample < 11; ++sample) {
              const double ns =
                  fixture.run(queue.get(), kernel, 3, use_candidate ? workers : 0);
              if (sample >= 4)
                std::cout << "{\"strategy\":" << unsigned(strategy)
                          << ",\"width\":" << size << ",\"pair\":" << pair
                          << ",\"candidate\":" << (use_candidate ? "true" : "false")
                          << ",\"sample\":" << sample - 4 << ",\"gpu_ns\":" << ns
                          << "}\n";
            }
          }
        a.compare(b, mode);
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
