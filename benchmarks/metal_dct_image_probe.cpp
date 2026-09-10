// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Exact image-I/O comparison against packed kernels in a frozen metallib.
#include "core/ac_strategy.h"
#include "gpu/metal/metal_aq_evaluation_internal.h"
#include "metal_probe.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

namespace {
using namespace gjxl::benchmark;
using Params = gjxl::metal_internal::AqDctImageParams;
void Submit(MTL::CommandQueue *queue, const Kernel &kernel,
            std::vector<Buffer> &buffers, size_t count, size_t threads,
            const Params *params = nullptr) {
  auto command = NS::RetainPtr(queue->commandBuffer());
  auto encoder = NS::RetainPtr(command->computeCommandEncoder());
  encoder->setComputePipelineState(kernel.pipeline.get());
  for (size_t i = 0; i < buffers.size(); ++i)
    encoder->setBuffer(buffers[i].object.get(), kGuard, i);
  if (params)
    encoder->setBytes(params, sizeof(*params), 5);
  encoder->dispatchThreadgroups(MTL::Size(count, 1, 1), MTL::Size(threads, 1, 1));
  encoder->endEncoding();
  command->commit();
  command->waitUntilCompleted();
  Check(command->status() == MTL::CommandBufferStatusCompleted,
        "DCT submission failed");
  for (const auto &b : buffers)
    b.guards();
}
void Test(MTL::Device *device, MTL::CommandQueue *queue, const char *reference,
          const char *candidate, gjxl::AcStrategyType strategy, unsigned pattern) {
  const auto &info = *gjxl::GetAcStrategyInfo(strategy);
  const auto shape = info.pixel_extent();
  const size_t n = info.coefficient_count();
  const size_t width = pattern % 2 ? 128 : 64, height = pattern % 2 ? 96 : 64;
  const size_t stride = width + 16, planes = stride * (height + 8),
               anchors = width * height / n;
  const Params params{3, uint32_t(anchors), 32, uint32_t(stride)};
  std::vector<Buffer> direct;
  for (size_t i = 0; i < 3; ++i)
    direct.emplace_back(device, planes * 4);
  direct.emplace_back(device, (anchors + 6) * 8);
  direct.emplace_back(device, (3 * anchors * n + 64) * 4);
  std::vector<Buffer> packed;
  packed.emplace_back(device, 3 * anchors * n * 4);
  packed.emplace_back(device, 3 * anchors * n * 4);
  // The existing wide inverse kernels bind their bases as device buffers.
  // Match production's host basis generation exactly.
  for (size_t length : {shape.height, shape.width}) {
    packed.emplace_back(device, length * length * 4);
    auto *basis = packed.back().as<float>();
    const double scale = std::sqrt(2.0 / double(length));
    for (size_t frequency = 0; frequency < length; ++frequency) {
      const double alpha = frequency == 0 ? 1.0 / std::sqrt(2.0) : 1.0;
      for (size_t sample = 0; sample < length; ++sample) {
        const double angle = (double(sample) + 0.5) * double(frequency) *
                             std::numbers::pi_v<double> / double(length);
        basis[frequency * length + sample] = float(alpha * scale * std::cos(angle));
      }
    }
  }
  std::vector<size_t> order(anchors);
  for (size_t i = 0; i < anchors; ++i)
    order[i] = i;
  std::mt19937 random(193 + pattern);
  std::shuffle(order.begin(), order.end(), random);
  auto *points = direct[3].as<uint32_t>();
  for (size_t i = 0; i < anchors; ++i) {
    const size_t x = order[i] % (width / shape.width) * shape.width;
    const size_t y = order[i] / (width / shape.width) * shape.height;
    points[2 * (params.anchor_offset + i)] = uint32_t(x / 8);
    points[2 * (params.anchor_offset + i) + 1] = uint32_t(y / 8);
    for (size_t c = 0; c < 3; ++c)
      for (size_t py = 0; py < shape.height; ++py)
        for (size_t px = 0; px < shape.width; ++px) {
          float value = float(int(random() % 20001) - 10000) * 0.0000625f;
          if (pattern == 2)
            value = 0;
          if (pattern == 3)
            value *= 1024;
          direct[c].as<float>()[(y + py) * stride + x + px] = value;
          packed[0].as<float>()[(c * anchors + i) * n + py * shape.width + px] = value;
        }
  }
  const std::string name =
      "gjxl_dct" + (shape.width == shape.height ? std::to_string(shape.width)
                                                : std::to_string(shape.height) + "x" +
                                                      std::to_string(shape.width));
  // Names use rows x columns, while pixel extents use width x height.
  const std::string forward = name + "_forward_simdgroup_2d_matmul";
  const std::string inverse = name + "_inverse_simdgroup_2d_matmul";
  Kernel old_forward(device, reference, forward.c_str()),
      new_forward(device, candidate, (forward + "_image").c_str());
  Kernel old_inverse(device, reference, inverse.c_str()),
      new_inverse(device, candidate, (inverse + "_image").c_str());
  const size_t threads = shape.height / 8 * 32;
  Submit(queue, old_forward, packed, 3 * anchors, threads);
  Submit(queue, new_forward, direct, 3 * anchors, threads, &params);
  Check(!std::memcmp(packed[1].data(),
                     direct[4].as<float>() + params.coefficient_offset, packed[1].size),
        "forward image DCT differs: " + name);
  for (size_t i = 0; i < params.coefficient_offset; ++i)
    Check(direct[4].as<uint32_t>()[i] == 0xa5a5a5a5u, "coefficient prefix overwritten");
  for (size_t i = params.coefficient_offset + 3 * anchors * n; i < direct[4].size / 4;
       ++i)
    Check(direct[4].as<uint32_t>()[i] == 0xa5a5a5a5u, "coefficient suffix overwritten");
  std::memcpy(packed[0].data(), packed[1].data(), packed[0].size);
  for (size_t c = 0; c < 3; ++c)
    std::memset(direct[c].data(), 0xa5, direct[c].size);
  Submit(queue, old_inverse, packed, 3 * anchors, threads);
  Submit(queue, new_inverse, direct, 3 * anchors, threads, &params);
  for (size_t c = 0; c < 3; ++c) {
    std::vector<float> expected(planes);
    std::memset(expected.data(), 0xa5, planes * 4);
    for (size_t i = 0; i < anchors; ++i) {
      const size_t x = points[2 * (params.anchor_offset + i)] * 8,
                   y = points[2 * (params.anchor_offset + i) + 1] * 8;
      for (size_t py = 0; py < shape.height; ++py)
        std::copy_n(packed[1].as<float>() + (c * anchors + i) * n + py * shape.width,
                    shape.width, expected.data() + (y + py) * stride + x);
    }
    Check(!std::memcmp(expected.data(), direct[c].data(), planes * 4),
          "inverse image DCT or image padding differs: " + name);
  }
}
} // namespace
int main(int argc, char **argv) try {
  Check(argc == 3,
        "usage: gjxl_metal_dct_image_probe BASELINE.metallib CANDIDATE.metallib");
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  Check(bool(device), "Metal device");
  auto queue = NS::TransferPtr(device->newCommandQueue());
  const std::array strategies = {
      gjxl::AcStrategyType::kDct8,     gjxl::AcStrategyType::kDct16x16,
      gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
      gjxl::AcStrategyType::kDct8x16,  gjxl::AcStrategyType::kDct32x16,
      gjxl::AcStrategyType::kDct16x32};
  for (auto strategy : strategies)
    for (unsigned pattern = 0; pattern < 8; ++pattern)
      Test(device.get(), queue.get(), argv[1], argv[2], strategy, pattern);
  std::cout << "56 forward/inverse image pairs agree bitwise, including guards, "
               "padding and shuffled anchors\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
