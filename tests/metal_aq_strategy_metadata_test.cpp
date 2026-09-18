// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

#include "core/ac_strategy.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_aq_strategy_metadata.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_backend_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::metal_internal;
constexpr uint32_t kPoison = 0xa5a5a5a5u;
bool Ok(Status s) {
  if (s.ok())
    return true;
  std::cerr << s.message() << '\n';
  return false;
}

struct Arena {
  DeviceScratchArena storage;
  AqStrategyMetadataStoragePlan plan;
  AqStrategyMetadataDescriptor descriptor;
  std::vector<uint8_t> before, after;
  std::array<std::vector<uint32_t>, kMetadataPlaneCount> result;

  bool Prepare(GpuBackend &gpu, Extent2D blocks) {
    if (!Ok(ComputeAqStrategyMetadataStoragePlan(blocks, &plan)))
      return false;
    descriptor.blocks = blocks;
    DeviceScratchLayoutPlan layout(256);
    std::array<DevicePlaneLayout, kMetadataPlaneCount + 1> planes;
    for (size_t i = 0; i < planes.size(); ++i) {
      const size_t n = i == 0 ? plan.selection_bytes : plan.elements[i - 1];
      if (!Ok(layout.AddPlane(i == 0 ? DeviceElementType::kU8
                                     : DeviceElementType::kI32,
                              {n + 64, 1}, n + 64, 256, &planes[i])))
        return false;
    }
    if (!Ok(storage.Prepare(gpu, layout.capacity_bytes())))
      return false;
    for (size_t i = 0; i < planes.size(); ++i) {
      DevicePlaneView view;
      if (!Ok(storage.BindPlane(planes[i], &view)))
        return false;
      const size_t n = i == 0 ? plan.selection_bytes : plan.elements[i - 1];
      view.extent = {n, 1};
      view.row_stride = n;
      if (i == 0)
        descriptor.selection = view;
      else
        descriptor.planes[i - 1] = view;
    }
    before.assign(storage.capacity_bytes(), 0xa5);
    after.resize(before.size());
    return true;
  }

  bool Run(GpuBackend &gpu, const std::vector<uint8_t> &cells) {
    if (cells.size() != plan.selection_bytes)
      return false;
    std::fill(before.begin(), before.end(), 0xa5);
    std::copy(cells.begin(), cells.end(),
              before.begin() + descriptor.selection.offset_bytes);
    if (!Ok(gpu.CopyHostToDevice(*storage.backing_buffer(), before.data(),
                                 before.size())))
      return false;
    std::unique_ptr<GpuSubmission> submission;
    if (!Ok(MetalAqStrategyMetadata::Submit(gpu, descriptor, &submission)) ||
        !submission || !Ok(submission->Wait()) ||
        !Ok(gpu.CopyDeviceToHost(*storage.backing_buffer(), after.data(),
                                 after.size())))
      return false;
    std::vector<bool> writable(after.size(), false);
    for (size_t i = 0; i < descriptor.planes.size(); ++i) {
      const auto view = descriptor.planes[i];
      result[i].resize(plan.elements[i]);
      std::memcpy(result[i].data(), after.data() + view.offset_bytes,
                  4 * plan.elements[i]);
      std::fill(writable.begin() + view.offset_bytes,
                writable.begin() + view.offset_bytes + 4 * plan.elements[i],
                true);
    }
    for (size_t i = 0; i < after.size(); ++i) {
      if (!writable[i] && after[i] != before[i]) {
        std::cerr << "Metadata overwrote a guard/input byte at " << i << '\n';
        return false;
      }
    }
    return true;
  }

  template <class Container>
  bool Match(size_t plane, const Container &expected) const {
    const auto &actual = result[plane];
    if (expected.size() > actual.size())
      return false;
    for (size_t i = 0; i < actual.size(); ++i) {
      const uint32_t value = i < expected.size() ? expected[i] : kPoison;
      if (actual[i] != value) {
        std::cerr << "Metadata plane " << plane << " differs at " << i << ": "
                  << actual[i] << " expected " << value << '\n';
        return false;
      }
    }
    return true;
  }

  bool Match(const MetalAqStrategyMetadataSnapshot &expected) const {
    return Match(kMetadataFamilies, expected.families) &&
           Match(kMetadataControl, expected.control) &&
           Match(kMetadataStrategies, expected.strategies) &&
           Match(kMetadataAnchors, expected.anchors) &&
           Match(kMetadataColorRecords, expected.color_records) &&
           Match(kMetadataColorOffsets, expected.color_offsets) &&
           Match(kMetadataDestinations, expected.destinations);
  }
};

bool Grid(Extent2D blocks, uint32_t seed, AcStrategyGrid *grid) {
  if (!Ok(AcStrategyGrid::Create(blocks, grid)))
    return false;
  if (seed == 0) {
    grid->fill_dct8();
    return true;
  }
  for (size_t y = 0; y < blocks.height; ++y)
    for (size_t x = 0; x < blocks.width; ++x) {
      if (grid->occupied(x, y))
        continue;
      seed = seed * 1664525u + 1013904223u;
      for (size_t attempt = 0; attempt < 7; ++attempt) {
        const auto type = kSupportedAqStrategies[(seed % 7 + attempt) % 7];
        const auto size = GetAcStrategyInfo(type)->covered_blocks;
        if (x + size.width > std::min((x / 8 + 1) * 8, blocks.width) ||
            y + size.height > std::min((y / 8 + 1) * 8, blocks.height))
          continue;
        if (grid->Set(x, y, type).ok())
          break;
      }
    }
  return grid->complete();
}

std::vector<uint8_t> Cells(const AcStrategyGrid &grid) {
  const auto blocks = grid.extent(), tiles = blocks.ceil_div(8);
  std::vector<uint8_t> cells(
      blocks.width * blocks.height + tiles.width * tiles.height, 0);
  for (size_t y = 0; y < blocks.height; ++y)
    for (size_t x = 0; x < blocks.width; ++x) {
      AcStrategyCell cell;
      if (!grid.Get(x, y, &cell).ok())
        std::abort();
      cells[y * blocks.width + x] =
          (uint8_t(cell.strategy) << 1) | cell.is_anchor;
    }
  return cells;
}

struct Oracle {
  std::array<std::vector<float>, 3> image;
  std::vector<uint8_t> sharpness;
  std::unique_ptr<PreparedAqEvaluation> prepared;
  bool Prepare(GpuBackend &gpu, const AcStrategyGrid &grid) {
    const auto blocks = grid.extent();
    const Extent2D pixels{blocks.width * 8, blocks.height * 8};
    for (auto &p : image)
      p.assign(pixels.width * pixels.height, 0.25f);
    sharpness.assign(blocks.width * blocks.height, 4);
    ConstImage3FView view{{{{image[0].data(), pixels, pixels.width},
                            {image[1].data(), pixels, pixels.width},
                            {image[2].data(), pixels, pixels.width}}}};
    AqEvaluationOptions options;
    options.evaluation_free = true;
    return Ok(PrepareAqEvaluation(
        gpu,
        {.original_linear_rgb = view,
         .coding_opsin = view,
         .strategies = &grid,
         .epf_sharpness = {sharpness.data(), blocks, blocks.width},
         .options = options,
         .resident_quantization = true,
         .coefficient_decision_mode =
             AcCoefficientDecisionMode::kAdjustedSharedQuant},
        &prepared));
  }
  bool Get(const AcStrategyGrid &grid,
           MetalAqStrategyMetadataSnapshot *result) {
    return Ok(prepared->Reconfigure(
               grid, {sharpness.data(), grid.extent(), grid.extent().width})) &&
           Ok(GetMetalAqStrategyMetadataForTesting(*prepared, result));
  }
};

bool CheckShape(GpuBackend &gpu, Extent2D blocks, size_t *cases) {
  Arena arena;
  Oracle oracle;
  AcStrategyGrid grid;
  if (!arena.Prepare(gpu, blocks) || !Grid(blocks, 0, &grid) ||
      !oracle.Prepare(gpu, grid))
    return false;
  for (uint32_t seed : {0u, 1u, 19u, 77287u}) {
    MetalAqStrategyMetadataSnapshot expected;
    if (!Grid(blocks, seed, &grid) || !oracle.Get(grid, &expected) ||
        !arena.Run(gpu, Cells(grid)) || !arena.Match(expected)) {
      std::cerr << "Metadata shape " << blocks.width << 'x' << blocks.height
                << " seed " << seed << '\n';
      std::cerr << "control:";
      for (auto x : arena.result[kMetadataControl])
        std::cerr << ' ' << x;
      std::cerr << " families:";
      for (auto x : arena.result[kMetadataFamilies])
        std::cerr << ' ' << x;
      std::cerr << '\n';
      return false;
    }
    ++*cases;
  }
  return true;
}

bool CheckFailure(GpuBackend &gpu) {
  const Extent2D blocks{17, 19};
  Arena arena;
  AcStrategyGrid grid;
  if (!arena.Prepare(gpu, blocks) || !Grid(blocks, 0, &grid))
    return false;
  const auto valid = Cells(grid);
  for (size_t failure = 0; failure < 6; ++failure) {
    auto cells = valid;
    if (failure == 0)
      cells[0] = 255; // Unsupported strategy.
    if (failure == 1)
      cells[13] = 0; // Orphan non-anchor.
    if (failure == 2)
      cells[blocks.width - 1] = 9; // DCT16 crossing edge.
    if (failure == 3)
      cells[7] = 9; // DCT16 crossing tile.
    if (failure == 4)
      cells[blocks.width * blocks.height + 1] = 1; // Selector error.
    if (failure == 5) { // Equal-area overlap plus hole; area alone cannot
                        // validate a cover.
      for (size_t dy = 0; dy < 2; ++dy)
        for (size_t dx = 0; dx < 2; ++dx) {
          cells[dy * blocks.width + 1 + dx] = 8;
          cells[(1 + dy) * blocks.width + dx] = 8;
        }
      cells[1] = cells[blocks.width] = 9;
      cells[0] = 0;
    }
    if (!arena.Run(gpu, cells) || arena.result[kMetadataControl][0] == 0)
      return false;
    for (size_t i = 1; i < 4; ++i)
      if (arena.result[kMetadataControl][i] != 0)
        return false;
    for (size_t i = 0; i < blocks.width * blocks.height; ++i)
      if (arena.result[kMetadataStrategies][2 * i] != 0 ||
          arena.result[kMetadataStrategies][2 * i + 1] != 1)
        return false;
    for (size_t f = 0; f < 7; ++f)
      for (size_t i : {1u, 2u, 3u})
        if (arena.result[kMetadataFamilies][5 * f + i] != 0)
          return false;
    for (size_t plane :
         {size_t(kMetadataAnchors), size_t(kMetadataColorRecords),
          size_t(kMetadataDestinations)})
      if (!std::ranges::all_of(arena.result[plane],
                               [](uint32_t x) { return x == kPoison; }))
        return false;
    if (!arena.Run(gpu, valid) || arena.result[kMetadataControl][0] != 0)
      return false;
  }
  std::unique_ptr<GpuSubmission> submission;
  auto bad = arena.descriptor;
  --bad.selection.extent.width;
  if (MetalAqStrategyMetadata::Submit(gpu, bad, &submission).ok() || submission)
    return false;
  bad = arena.descriptor;
  bad.planes[kMetadataControl] = bad.planes[kMetadataFamilies];
  if (MetalAqStrategyMetadata::Submit(gpu, bad, &submission).ok() || submission)
    return false;
  if (!Ok(ArmNextMetalSubmissionFailureForTest(gpu, true, false)) ||
      MetalAqStrategyMetadata::Submit(gpu, arena.descriptor, &submission)
              .code() != StatusCode::kSubmissionFailed ||
      submission)
    return false;
  if (!Ok(ArmNextMetalSubmissionFailureForTest(gpu, false, true)) ||
      !Ok(MetalAqStrategyMetadata::Submit(gpu, arena.descriptor,
                                          &submission)) ||
      !submission || submission->Wait().code() != StatusCode::kDeviceError)
    return false;
  submission.reset();
  if (!arena.Run(gpu, valid) || arena.result[kMetadataControl][0] != 0)
    return false;
  AqStrategyMetadataStoragePlan plan;
  plan.capacity_bytes = 123;
  for (auto shape :
       {Extent2D{0, 8}, Extent2D{1, std::numeric_limits<size_t>::max()},
        Extent2D{1, std::numeric_limits<int32_t>::max()}}) {
    if (ComputeAqStrategyMetadataStoragePlan(shape, &plan).ok() ||
        plan.capacity_bytes != 123)
      return false;
  }
  return true;
}

bool Benchmark(GpuBackend &gpu, unsigned rotation) {
  const std::array shapes{Extent2D{250, 188}, Extent2D{480, 270},
                          Extent2D{750, 500}};
  for (size_t shape = 0; shape < shapes.size(); ++shape) {
    const Extent2D blocks =
        shapes[(shape + rotation % shapes.size()) % shapes.size()];
    Arena arena;
    AcStrategyGrid grid;
    if (!arena.Prepare(gpu, blocks) || !Grid(blocks, 779731, &grid) ||
        !arena.Run(gpu, Cells(grid)) || arena.result[kMetadataControl][0] != 0)
      return false;
    std::vector<double> times;
    for (size_t i = 0; i < 24; ++i) {
      std::unique_ptr<GpuSubmission> submission;
      uint64_t duration = 0;
      if (!Ok(MetalAqStrategyMetadata::Submit(gpu, arena.descriptor,
                                              &submission)) ||
          !Ok(submission->Wait()) ||
          !Ok(GetMetalSubmissionGpuDuration(*submission, &duration)))
        return false;
      if (i >= 3)
        times.push_back(double(duration) / 1e6);
    }
    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    std::cout << "{\"pixels\":" << blocks.width * blocks.height * 64
              << ",\"gpu_median_ms\":" << sorted[sorted.size() / 2]
              << ",\"scratch_bytes\":" << arena.plan.capacity_bytes
              << ",\"samples_ms\":[";
    for (size_t i = 0; i < times.size(); ++i)
      std::cout << (i ? "," : "") << times[i];
    std::cout << "]}\n";
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  bool benchmark = false;
  unsigned rotation = 0;
  const char *library = GJXL_METALLIB_PATH;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--benchmark")
      benchmark = true;
    else if (arg == "--metallib" && i + 1 < argc)
      library = argv[++i];
    else if (arg == "--rotate" && i + 1 < argc)
      rotation = unsigned(std::strtoul(argv[++i], nullptr, 10));
    else {
      std::cerr << "Unknown metadata test argument\n";
      return EXIT_FAILURE;
    }
  }
  std::unique_ptr<GpuBackend> gpu;
  if (!Ok(CreateMetalBackend(library, &gpu)))
    return EXIT_FAILURE;
  if (benchmark)
    return Benchmark(*gpu, rotation) ? EXIT_SUCCESS : EXIT_FAILURE;
  size_t cases = 0;
  for (size_t w = 1; w <= 8; ++w)
    for (size_t h = 1; h <= 8; ++h)
      if (!CheckShape(*gpu, {8 + w, 8 + h}, &cases))
        return EXIT_FAILURE;
  for (const auto shape : {Extent2D{1, 1},
                           {1, 17},
                           {17, 1},
                           {8, 8},
                           {31, 33},
                           {33, 35},
                           {63, 65},
                           {64, 64},
                           {129, 131}})
    if (!CheckShape(*gpu, shape, &cases))
      return EXIT_FAILURE;
  if (!CheckFailure(*gpu))
    return EXIT_FAILURE;
  std::cout << cases
            << " exact CPU-builder/GPU-metadata cases passed; guards, failure "
               "and reuse passed\n";
  return EXIT_SUCCESS;
}
