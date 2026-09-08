// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/metal_aq_evaluation_internal.h"
#include "metal_probe.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
namespace {
using namespace gjxl::benchmark;
using Params = gjxl::metal_internal::AqEpfParams;
struct Variant {
  const char *suffix;
  size_t width, height, pixels;
};
constexpr std::array variants = {
    Variant{"direct", 8, 8, 1},       Variant{"tile16x8_p1", 16, 8, 1},
    Variant{"tile16x8_p2", 16, 8, 2}, Variant{"tile16x8_p4", 16, 8, 4},
    Variant{"tile32x4_p2", 32, 4, 2}, Variant{"tile32x4_p4", 32, 4, 4}};
struct Fixture {
  Params params;
  gjxl::metal_internal::AqOpsinToLinearParams linear_params;
  bool linear;
  std::vector<Buffer> buffers;
  Fixture(MTL::Device *device, size_t width, size_t height, unsigned pass,
          unsigned pattern, bool linear_output = false)
      : params{uint32_t(width),
               uint32_t(height),
               uint32_t(width + 19),
               uint32_t(width + 23),
               uint32_t((width + 7) / 8 + 3),
               pass,
               pass == 0   ? 1.2f
               : pass == 1 ? 1.0f
                           : 0.85f,
               0.91f,
               {1.0f, 0.72f, 0.63f}},
        linear_params{uint32_t(width), uint32_t(height), uint32_t(width + 23),
                      uint32_t(width + 29), 0.91f},
        linear(linear_output) {
    const std::array<size_t, 8> sizes = {params.input_stride * height * 4,
                                         params.input_stride * height * 4,
                                         params.input_stride * height * 4,
                                         params.inverse_sigma_stride *
                                             ((height + 7) / 8) * 4,
                                         params.output_stride * height * 4,
                                         params.output_stride * height * 4,
                                         params.output_stride * height * 4,
                                         4};
    for (auto size : sizes)
      buffers.emplace_back(device, size);
    if (linear)
      for (unsigned i = 0; i < 3; ++i)
        buffers.emplace_back(device, linear_params.output_stride * height * 4);
    uint32_t rng = 0x7193123u + pattern;
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < width; ++x) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          float value =
              float(int32_t(rng % 2001) - 1000) * 0.00003125f + float(c) * 0.1f;
          if (pattern == 1)
            value = 0.3125f;
          if (pattern == 2)
            value = (x == width / 2 && y == height / 2) ? 32.0f : 0.0f;
          if (pattern == 3)
            value *= 1.0e20f;
          buffers[c].as<float>()[y * params.input_stride + x] = value;
        }
    for (size_t y = 0; y < (height + 7) / 8; ++y)
      for (size_t x = 0; x < (width + 7) / 8; ++x)
        buffers[3].as<float>()[y * params.inverse_sigma_stride + x] =
            pattern == 4 || ((x + 3 * y) % 13 == 0 && pattern != 1)
                ? -4.0f
                : -0.035f - float((x + 7 * y) % 31) * 0.003f;
    if (pattern == 5)
      buffers[0]
          .as<float>()[width > 1 && height > 1 ? params.input_stride + 1 : 0] =
          std::numeric_limits<float>::infinity();
    buffers[7].as<uint32_t>()[0] = 0;
  }
  double run(MTL::CommandQueue *queue, const Kernel &kernel,
             const Variant &variant, unsigned repeats = 1,
             const Kernel *conversion = nullptr, bool fused_linear = false) {
    auto command = NS::RetainPtr(queue->commandBuffer());
    auto encoder = NS::RetainPtr(command->computeCommandEncoder());
    const MTL::Size threads(variant.width, variant.height, 1);
    for (unsigned i = 0; i < repeats; ++i) {
      encoder->setComputePipelineState(kernel.pipeline.get());
      for (size_t j = 0; j < 8; ++j) {
        const size_t index = fused_linear && j >= 4 && j <= 6 ? j + 4 : j;
        encoder->setBuffer(buffers[index].object.get(), kGuard, j);
      }
      Params output_params = params;
      if (fused_linear)
        output_params.output_stride = linear_params.output_stride;
      encoder->setBytes(&output_params, sizeof(output_params), 8);
      if (fused_linear)
        encoder->setBytes(&linear_params.scale, sizeof(float), 9);
      if (std::string_view(variant.suffix) == "direct")
        encoder->dispatchThreads(MTL::Size(params.width, params.height, 1),
                                 threads);
      else
        encoder->dispatchThreadgroups(
            MTL::Size((params.width + variant.width - 1) / variant.width,
                      (params.height + variant.height * variant.pixels - 1) /
                          (variant.height * variant.pixels),
                      1),
            threads);
      if (conversion) {
        encoder->setComputePipelineState(conversion->pipeline.get());
        for (size_t c = 0; c < 3; ++c) {
          encoder->setBuffer(buffers[4 + c].object.get(), kGuard, c);
          encoder->setBuffer(buffers[8 + c].object.get(), kGuard, 3 + c);
        }
        encoder->setBuffer(buffers[7].object.get(), kGuard, 6);
        encoder->setBytes(&linear_params, sizeof(linear_params), 7);
        encoder->dispatchThreads(MTL::Size(params.width, params.height, 1),
                                 MTL::Size(8, 8, 1));
      }
    }
    encoder->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    Check(command->status() == MTL::CommandBufferStatusCompleted,
          "EPF dispatch failed");
    return (command->GPUEndTime() - command->GPUStartTime()) * 1e9 / repeats;
  }
  void compare(const Fixture &other) const {
    for (size_t i = 0; i < buffers.size(); ++i) {
      buffers[i].guards();
      other.buffers[i].guards();
      if (linear && i >= 4 && i <= 6) {
        const auto *bytes = other.buffers[i].as<unsigned char>();
        Check(std::all_of(bytes, bytes + other.buffers[i].size,
                          [](unsigned char v) { return v == 0xa5; }),
              "Fused EPF wrote the suppressed intermediate");
        continue;
      }
      if (std::memcmp(buffers[i].data(), other.buffers[i].data(),
                      buffers[i].size)) {
        const auto *a = buffers[i].as<unsigned char>();
        const auto *b = other.buffers[i].as<unsigned char>();
        size_t byte = 0;
        while (byte < buffers[i].size && a[byte] == b[byte])
          ++byte;
        throw std::runtime_error("EPF mismatch buffer=" + std::to_string(i) +
                                 " byte=" + std::to_string(byte));
      }
    }
  }
};
} // namespace
int main(int argc, char **argv) try {
  Check(argc >= 3,
        "usage: gjxl_metal_epf_probe BASELINE.metallib CANDIDATE.metallib "
        "[--timing] [--linear] [--extent WxH] [--pattern N]");
  bool timing = false;
  bool linear = false;
  int selected_pattern = -1;
  std::vector<std::pair<size_t, size_t>> selected_extents;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--timing")
      timing = true;
    else if (arg == "--linear")
      linear = true;
    else if (arg == "--pattern" && i + 1 < argc)
      selected_pattern = std::stoi(argv[++i]);
    else if (arg == "--extent" && i + 1 < argc) {
      const std::string value = argv[++i];
      const auto separator = value.find('x');
      Check(separator != std::string::npos, "extent requires WxH");
      const size_t w = std::stoul(value.substr(0, separator)),
                   h = std::stoul(value.substr(separator + 1));
      Check(w >= 8 && h >= 8 && w <= 8192 && h <= 8192,
            "extent outside probe limits");
      selected_extents.emplace_back(w, h);
    } else
      throw std::runtime_error("invalid option: " + arg);
  }
  Check(selected_pattern >= -1 && selected_pattern < 6,
        "pattern outside probe limits");
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  Check(bool(device), "Metal device");
  auto queue = NS::TransferPtr(device->newCommandQueue());
  Kernel conversion(device.get(), argv[1], "gjxl_aq_opsin_to_linear_rgb_f32");
  size_t cases = 0;
  for (unsigned pass = 0; pass < 3; ++pass)
    for (const auto &variant : variants) {
      if (linear &&
          (pass == 0 || (std::string_view(variant.suffix) != "direct" &&
                         std::string_view(variant.suffix) != "tile32x4_p2")))
        continue;
      const std::string baseline_name =
          linear
              ? "gjxl_aq_epf_pass" + std::to_string(pass) + "_" + variant.suffix
              : "gjxl_aq_epf_f32";
      Kernel baseline(device.get(), argv[1], baseline_name.c_str());
      const Variant &baseline_variant = linear ? variant : variants[0];
      const std::string name = "gjxl_aq_epf_pass" + std::to_string(pass) +
                               (linear ? "_linear_" : "_") + variant.suffix;
      Kernel candidate(device.get(), argv[2], name.c_str());
      const std::string tiny_control_name =
          "gjxl_aq_epf_pass" + std::to_string(pass) + "_tile32x4_p2";
      Kernel tiny_control(device.get(), argv[1], tiny_control_name.c_str());
      const std::vector<std::pair<size_t, size_t>> sizes =
          !selected_extents.empty() ? selected_extents
          : linear && !timing
              ? std::vector<std::pair<size_t, size_t>>{{1, 1},   {1, 9},
                                                       {2, 2},   {8, 8},
                                                       {17, 9},  {79, 67},
                                                       {128, 96}}
          : timing
              ? std::vector<std::pair<size_t, size_t>>{{512, 512}, {3840, 2160}}
              : std::vector<std::pair<size_t, size_t>>{
                    {8, 8}, {17, 9}, {79, 67}, {128, 96}};
      for (auto [width, height] : sizes)
        for (unsigned pattern = 0; pattern < 6; ++pattern) {
          if (selected_pattern >= 0 ? pattern != unsigned(selected_pattern)
                                    : (timing && pattern != 0))
            continue;
          Fixture a(device.get(), width, height, pass, pattern, linear),
              b(device.get(), width, height, pass, pattern, linear);
          // The parent's direct sampler only reflects once. Its tiled kernel
          // is the valid control for dimensions smaller than the stencil.
          const bool tiny = linear && (width < 3 || height < 3);
          const Kernel &control = tiny ? tiny_control : baseline;
          const Variant &control_variant =
              tiny ? variants[4] : baseline_variant;
          a.run(queue.get(), control, control_variant, 1,
                linear ? &conversion : nullptr);
          b.run(queue.get(), candidate, variant, 1, nullptr, linear);
          try {
            a.compare(b);
          } catch (const std::exception &e) {
            throw std::runtime_error(name + " " + std::to_string(width) + "x" +
                                     std::to_string(height) + " pattern=" +
                                     std::to_string(pattern) + ": " + e.what());
          }
          if (pattern < 5 && !(linear && pattern == 3))
            Check(a.buffers[7].as<uint32_t>()[0] == 0,
                  "unexpected EPF error flag");
          ++cases;
          if (timing)
            for (unsigned pair = 0; pair < 5; ++pair)
              for (unsigned side = 0; side < 2; ++side) {
                const bool use_candidate = (pair + side) % 2;
                for (unsigned sample = 0; sample < 11; ++sample) {
                  const double ns =
                      use_candidate
                          ? b.run(queue.get(), candidate, variant, 3, nullptr,
                                  linear)
                          : a.run(queue.get(), control, control_variant, 3,
                                  linear ? &conversion : nullptr);
                  if (sample >= 4)
                    std::cout << "{\"pass\":" << pass << ",\"variant\":\""
                              << variant.suffix << "\",\"width\":" << width
                              << ",\"height\":" << height
                              << ",\"pattern\":" << pattern
                              << ",\"pair\":" << pair << ",\"candidate\":"
                              << (use_candidate ? "true" : "false")
                              << ",\"sample\":" << sample - 4
                              << ",\"gpu_ns\":" << ns << "}\n";
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
