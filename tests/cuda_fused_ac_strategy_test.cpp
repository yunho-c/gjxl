// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "codec/ac_strategy_search_policy.h"
#include "codec/quantization.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_kernels.h"
#include "gpu/ops/ac_strategy.h"

namespace {
using namespace gjxl;
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Check(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
struct Range { size_t offset, bytes; };
struct ArenaPlan {
  size_t bytes = 64;
  Range Add(size_t count) {
    const size_t offset = (bytes + 255) & ~size_t{255};
    bytes = offset + count + 64;
    return {offset, count};
  }
};
size_t cases = 0, descriptors = 0;
void Case(GpuBackend& gpu, Extent2D blocks,
          ac_strategy_internal::CandidateStage stage, bool device_cfl) {
  const auto covered = GetAcStrategyInfo(stage.strategy)->covered_blocks;
  const Extent2D pixels{blocks.width * 8, blocks.height * 8};
  const Extent2D tiles{(blocks.width + 7) / 8, (blocks.height + 7) / 8};
  std::vector<AcStrategyCandidate> expected;
  // Enumerate complete candidates independently of the device tile mapping.
  for (size_t by = 0; by < blocks.height; by += 8) {
    for (size_t bx = 0; bx < blocks.width; bx += 8) {
      const size_t height = std::min<size_t>(8, blocks.height - by);
      const size_t width = std::min<size_t>(8, blocks.width - bx);
      for (size_t y = 0; y + covered.height <= height; y += stage.anchor_step) {
        for (size_t x = 0; x + covered.width <= width; x += stage.anchor_step) {
          expected.push_back({static_cast<uint32_t>(bx + x),
            static_cast<uint32_t>(by + y), 1.0f, stage.entropy_multiplier, 0.0f, 0.0f});
        }
      }
    }
  }
  const size_t count = expected.size();
  if (count == 0) return;
  for (size_t i = 0; i < count; ++i) {
    expected[i].quant_norm = .31f + .017f * static_cast<float>(i % 29);
    expected[i].cfl_x = .03125f * static_cast<float>(int(i % 13) - 6);
    expected[i].cfl_b = .5f + .0625f * static_cast<float>(int(i % 11) - 5);
  }
  const size_t pixel_stride = pixels.width + 5;
  const size_t tile_stride = tiles.width + 2;
  ArenaPlan plan;
  std::array<Range, 3> image;
  for (auto& r : image) r = plan.Add(pixel_stride * pixels.height * sizeof(float));
  const auto mask = plan.Add(pixel_stride * pixels.height * sizeof(float));
  const auto norms = plan.Add(count * sizeof(float));
  const auto cfl_x = plan.Add(tile_stride * tiles.height);
  const auto cfl_b = plan.Add(tile_stride * tiles.height);
  const size_t coefficient_count = GetAcStrategyInfo(stage.strategy)->coefficient_count();
  const auto matrices = plan.Add(6 * coefficient_count * sizeof(float));
  const auto candidates = plan.Add(count * sizeof(AcStrategyCandidate));
  AcStrategyScratchRequirements sizes;
  Check(GetAcStrategyScratchRequirements(gpu, stage.strategy, count, &sizes));
  const auto scratch_a = plan.Add(sizes.scratch_a_bytes);
  Require(sizes.scratch_b_bytes == 0, "Fused batch still requires forward scratch");
  // Only the independent two-kernel reference uses this test-owned range.
  const auto coefficients = plan.Add(count * 3 * coefficient_count * sizeof(float));
  const auto rates = plan.Add(sizes.rate_scratch_bytes);
  const auto costs = plan.Add(count * sizeof(float));
  std::vector<uint8_t> initial(plan.bytes, 0xa5);
  const auto WriteFloat = [&](size_t byte_offset, float value) {
    std::memcpy(initial.data() + byte_offset, &value, sizeof(value));
  };
  for (size_t y = 0; y < pixels.height; ++y) for (size_t x = 0; x < pixels.width; ++x) {
    const float base = .18f + .013f * std::sin(.13f * static_cast<float>(x)) +
      .007f * std::cos(.19f * static_cast<float>(y));
    for (size_t channel = 0; channel < 3; ++channel)
      WriteFloat(image[channel].offset + (y * pixel_stride + x) * sizeof(float),
        base * (channel == 0 ? .31f : channel == 1 ? 1.0f : 1.12f));
    WriteFloat(mask.offset + (y * pixel_stride + x) * sizeof(float),
      42.0f + .2f * static_cast<float>((x + 3 * y) % 19));
  }
  for (size_t i = 0; i < count; ++i)
    WriteFloat(norms.offset + i * sizeof(float), expected[i].quant_norm);
  for (size_t y = 0; y < tiles.height; ++y) for (size_t x = 0; x < tiles.width; ++x) {
    initial[cfl_x.offset + y * tile_stride + x] = static_cast<uint8_t>(static_cast<int8_t>(int((x + y * 3) % 39) - 19));
    initial[cfl_b.offset + y * tile_stride + x] = static_cast<uint8_t>(static_cast<int8_t>(int((x * 3 + y) % 51) - 25));
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    QuantizationMatrixView matrix;
    Check(GetDefaultQuantizationMatrix(stage.strategy, static_cast<XybChannel>(channel), &matrix));
    std::memcpy(initial.data() + matrices.offset + channel * coefficient_count * sizeof(float),
      matrix.dequant.data(), coefficient_count * sizeof(float));
    std::memcpy(initial.data() + matrices.offset + (3 + channel) * coefficient_count * sizeof(float),
      matrix.inverse_dequant.data(), coefficient_count * sizeof(float));
  }
  std::memcpy(initial.data() + candidates.offset, expected.data(), candidates.bytes);
  std::unique_ptr<DeviceBuffer> arena;
  Check(gpu.Allocate(plan.bytes, &arena));
  const auto Plane = [&](Range r, DeviceElementType type, Extent2D extent, size_t stride) {
    return ConstDevicePlaneView{arena.get(), r.offset, type, extent, stride};
  };
  AcStrategyCandidateBatch batch{
    .strategy = stage.strategy, .matrices = arena.get(), .candidates = arena.get(),
    .resident_opsin = {{Plane(image[0], DeviceElementType::kF32, pixels, pixel_stride),
      Plane(image[1], DeviceElementType::kF32, pixels, pixel_stride),
      Plane(image[2], DeviceElementType::kF32, pixels, pixel_stride)}},
    .resident_pixel_mask = Plane(mask, DeviceElementType::kF32, pixels, pixel_stride),
    .resident_y_to_x = device_cfl ? Plane(cfl_x, DeviceElementType::kI8, tiles, tile_stride) : ConstDevicePlaneView{},
    .resident_y_to_b = device_cfl ? Plane(cfl_b, DeviceElementType::kI8, tiles, tile_stride) : ConstDevicePlaneView{},
    .scratch_a = arena.get(), .rate_scratch = arena.get(), .costs = arena.get(),
    .matrices_offset_bytes = matrices.offset, .candidates_offset_bytes = candidates.offset,
    .scratch_a_offset_bytes = scratch_a.offset,
    .rate_scratch_offset_bytes = rates.offset, .costs_offset_bytes = costs.offset,
    .pixel_extent = pixels, .candidate_count = count, .butteraugli_target = 1.2f};
  std::unique_ptr<GpuSubmission> submission;
  const auto Run = [&](const auto& request) {
    Check(EvaluateAcStrategyCandidates(gpu, request, &submission));
    Require(submission != nullptr, "Candidate grid returned no submission");
    Check(submission->Wait());
  };
  Check(gpu.CopyHostToDevice(*arena, initial.data(), initial.size()));
  using namespace cuda_internal;
  auto* cuda_arena = dynamic_cast<CudaBuffer*>(arena.get());
  Require(cuda_arena != nullptr, "Expected a CUDA arena");
  const auto* state = cuda_arena->state();
  ScopedCudaDevice device(state->ordinal);
  Check(CudaRuntimeStatus(device.status(), "Select fused-test device"));
  auto* bytes = static_cast<uint8_t*>(cuda_arena->pointer());
  const auto Float = [&](Range r) {return reinterpret_cast<float*>(bytes + r.offset);};
  const auto* x = device_cfl ? reinterpret_cast<const signed char*>(bytes + cfl_x.offset) : nullptr;
  const auto* b = device_cfl ? reinterpret_cast<const signed char*>(bytes + cfl_b.offset) : nullptr;
  const CudaAcStrategyBatchParams params{
    .pixel_width = static_cast<uint32_t>(pixels.width),
    .pixel_height = static_cast<uint32_t>(pixels.height),
    .opsin_row_stride = static_cast<uint32_t>(pixel_stride),
    .pixel_mask_row_stride = static_cast<uint32_t>(pixel_stride),
    .candidate_count = static_cast<uint32_t>(count),
    .coefficient_count = static_cast<uint32_t>(coefficient_count),
    .transform_width = static_cast<uint32_t>(covered.width * 8),
    .transform_height = static_cast<uint32_t>(covered.height * 8),
    .color_tile_row_stride = static_cast<uint32_t>(tile_stride),
    .use_device_cfl = device_cfl ? 1u : 0u};
  Check(CudaRuntimeStatus(LaunchCudaAcStrategyForward(
    Float(image[0]), Float(image[1]), Float(image[2]), bytes + candidates.offset,
    Float(coefficients), params, state->stream), "Launch reference forward"));
  Check(CudaRuntimeStatus(LaunchCudaAcStrategyResidualInverseLoss(
    Float(coefficients), Float(matrices), Float(norms), x, b, Float(mask),
    bytes + candidates.offset, bytes + rates.offset, Float(scratch_a),
    params, state->stream), "Launch reference residual/inverse/loss"));
  Check(CudaRuntimeStatus(cudaStreamSynchronize(state->stream), "Wait reference"));
  std::vector<uint8_t> reference_losses(scratch_a.bytes), reference_rates(rates.bytes);
  Check(gpu.CopyDeviceToHost(*arena, reference_losses.data(), scratch_a.bytes, scratch_a.offset));
  Check(gpu.CopyDeviceToHost(*arena, reference_rates.data(), rates.bytes, rates.offset));
  std::vector<float> reference;
  // No batch points at the reference coefficients or norms. Reset the whole
  // arena before each fused evaluation and preserve those unused ranges.
  const std::array outputs{scratch_a, rates, costs};
  const auto Verify = [&] {
    std::vector<uint8_t> result(initial.size());
    Check(gpu.CopyDeviceToHost(*arena, result.data(), result.size()));
    Require(std::memcmp(result.data() + candidates.offset, expected.data(), candidates.bytes) == 0,
      "Fused batch changed candidate descriptors");
    if (reference.empty()) {
      reference.resize(count);
      std::memcpy(reference.data(), result.data() + costs.offset, costs.bytes);
      Require(std::ranges::all_of(reference, [](float v) {return std::isfinite(v) && v >= 0;}),
        "Fused batch produced an invalid cost");
    } else {
      Require(std::memcmp(result.data() + costs.offset, reference.data(), costs.bytes) == 0,
        "Fused batch cost bits changed on repeat");
    }
    Require(std::memcmp(result.data()+scratch_a.offset, reference_losses.data(), scratch_a.bytes)==0,
      "Fused channel loss bits differ");
    Require(std::memcmp(result.data()+rates.offset, reference_rates.data(), rates.bytes)==0,
      "Fused channel rate bits differ");
    for (size_t i = 0; i < result.size(); ++i) {
      const bool output = std::ranges::any_of(outputs, [&](Range r) {return i >= r.offset && i - r.offset < r.bytes;});
      Require(output || result[i] == initial[i], "Fused batch changed input or guard bytes");
    }
    return result;
  };
  for (int repeat = 0; repeat < 3; ++repeat) {
    Check(gpu.CopyHostToDevice(*arena, initial.data(), initial.size()));
    Run(batch);(void)Verify();
  }
  ++cases; descriptors += count;
}
}

int main() {
  try {
    std::unique_ptr<gjxl::GpuBackend> gpu;
    const auto status = gjxl::CreateCudaBackend(&gpu);
    if (!status.ok()) { std::cerr << status.message() << '\n'; return 77; }
    // Include one-candidate and odd-tail square batches: all inactive channel
    // groups must reach the deferred Y-read barrier before any tile is reused.
    for (gjxl::Extent2D blocks : std::array<gjxl::Extent2D, 23>{{
      {1,1},{1,9},{9,1},{3,3},{7,7},{8,8},{9,9},{15,17},{17,15},
      {33,3},{3,33},{33,17},{65,33},{257,1},{1,257},
      {2,2},{4,4},{4,5},{5,4},{4,6},{6,4},{4,9},{9,4}}}) {
      for (const auto& stage : gjxl::ac_strategy_internal::kCandidateStages)
        for (const bool device_cfl : {false, true}) Case(*gpu, blocks, stage, device_cfl);
    }
    std::cout << "CUDA fused AC grids passed: cases=" << cases << " descriptors=" << descriptors
      << " reference_batches=" << cases << " fused_batches=" << 3 * cases << "\n" << std::flush;
  } catch (const std::exception& error) {
    std::cerr << "Candidate grid failure: " << error.what() << '\n'; return 1;
  }
  return 0;
}
