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

#include "codec/adaptive_quantization.h"
#include "codec/vardct_frame.h"
#include "codec/vardct_frame_view_internal.h"
#include "core/ac_strategy.h"
#include "gpu/metal/kernels/aq_strategy_dispatch.h"
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
  bool Prepare(GpuBackend &gpu, const AcStrategyGrid &grid,
               bool evaluate = false) {
    const auto blocks = grid.extent();
    const Extent2D pixels{blocks.width * 8, blocks.height * 8};
    for (auto &p : image)
      p.assign(pixels.width * pixels.height, 0.25f);
    if (evaluate) {
      for (size_t c = 0; c < 3; ++c)
        for (size_t y = 0; y < pixels.height; ++y)
          for (size_t x = 0; x < pixels.width; ++x)
            image[c][y * pixels.width + x] =
                c == 0   ? .002f * float((x + 7 * y) % 9)
                : c == 1 ? .19f + .00012f * float(x) - .00007f * float(y)
                         : .15f + .00004f * float(x + 2 * y);
    }
    sharpness.assign(blocks.width * blocks.height, 4);
    ConstImage3FView view{{{{image[0].data(), pixels, pixels.width},
                            {image[1].data(), pixels, pixels.width},
                            {image[2].data(), pixels, pixels.width}}}};
    AqEvaluationOptions options;
    options.evaluation_free = !evaluate;
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

bool QuantizedCoefficientsEqual(const auto &expected, const auto &actual) {
  if (!expected.valid() || !actual.valid() ||
      expected.ac_group_count() != actual.ac_group_count()) {
    return false;
  }
  const gjxl::ConstImage3I32View expected_dc = expected.quantized_dc();
  const gjxl::ConstImage3I32View actual_dc = actual.quantized_dc();
  if (expected_dc.extent() != actual_dc.extent())
    return false;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < expected_dc.extent().height; ++y) {
      if (!std::equal(expected_dc.plane[channel].Row(y),
                      expected_dc.plane[channel].Row(y) +
                          expected_dc.extent().width,
                      actual_dc.plane[channel].Row(y))) {
        return false;
      }
    }
  }
  for (size_t group_index = 0; group_index < expected.ac_group_count();
       ++group_index) {
    gjxl::VarDctAcGroupView expected_group;
    gjxl::VarDctAcGroupView actual_group;
    if (!expected.GetAcGroup(group_index, &expected_group).ok() ||
        !actual.GetAcGroup(group_index, &actual_group).ok() ||
        expected_group.used_coefficient_count !=
            actual_group.used_coefficient_count) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      if (!std::equal(expected_group.coefficients[channel].begin(),
                      expected_group.coefficients[channel].end(),
                      actual_group.coefficients[channel].begin())) {
        return false;
      }
    }
  }

  return true;
}

bool CheckDispatchConsumers(GpuBackend &gpu) {
  using gjxl_aq_dispatch::Record;
  for (size_t trial = 0; trial < 27; ++trial) {
    const Extent2D blocks = trial >= 24 ? Extent2D{1, trial - 23}
                            : trial < 3 ? Extent2D{13, 11}
                                        : Extent2D{69, 69};
    AcStrategyGrid grid;
    if (trial < 3 || trial >= 24) {
      if (!Grid(blocks, uint32_t(trial), &grid))
        return false;
    } else {
      if (!Ok(AcStrategyGrid::Create(blocks, &grid)))
        return false;
      // Empty families, pure DCT8, and either side of the 256-anchor switch.
      const size_t family = (trial - 3) / 3;
      const auto type = kSupportedAqStrategies[family];
      size_t left = 255 + (trial - 3) % 3;
      const auto covered = GetAcStrategyInfo(type)->covered_blocks;
      for (size_t y = 0; y < blocks.height; ++y)
        for (size_t x = 0; x < blocks.width; ++x) {
          if (grid.occupied(x, y))
            continue;
          if (left &&
              x + covered.width <= std::min((x / 8 + 1) * 8, blocks.width) &&
              y + covered.height <= std::min((y / 8 + 1) * 8, blocks.height) &&
              grid.Set(x, y, type).ok())
            --left;
          else if (!Ok(grid.Set(x, y, AcStrategyType::kDct8)))
            return false;
        }
    }
    Arena metadata;
    Oracle oracle;
    if (!metadata.Prepare(gpu, blocks) || !metadata.Run(gpu, Cells(grid)) ||
        !oracle.Prepare(gpu, grid, true))
      return false;
    if (trial >= 6 && trial < 24 &&
        metadata.result[kMetadataFamilies][5 * ((trial - 3) / 3) + 2] !=
            255 + (trial - 3) % 3) {
      std::cerr << "Fixture missed the requested family-count boundary\n";
      return false;
    }
    DeviceScratchLayoutPlan layout;
    DevicePlaneLayout parameter_layout;
    constexpr size_t words = 7 * sizeof(Record) / 4;
    if (!Ok(layout.AddPlane(DeviceElementType::kI32, {words + 64, 1},
                            words + 64, 256, &parameter_layout)))
      return false;
    DeviceScratchArena parameters;
    DevicePlaneView view;
    if (!Ok(parameters.Prepare(gpu, layout.capacity_bytes())) ||
        !Ok(parameters.BindPlane(parameter_layout, &view)))
      return false;
    std::vector<uint32_t> poison(words + 64, kPoison);
    if (!Ok(gpu.CopyHostToDevice(*view.buffer, poison.data(), poison.size() * 4,
                                 view.offset_bytes)))
      return false;
    view.extent = {words, 1};
    view.row_stride = words;
    if (trial == 0) {
      const auto families = metadata.descriptor.planes[kMetadataFamilies];
      auto short_view = view;
      --short_view.extent.width;
      auto alias = view;
      alias.extent = {35, 1};
      if (BindMetalAqStrategyDispatchForTesting(*oracle.prepared, families,
                                                short_view)
                  .code() != StatusCode::kInvalidArgument ||
          BindMetalAqStrategyDispatchForTesting(*oracle.prepared, alias, view)
                  .code() != StatusCode::kInvalidArgument)
        return false;
    }
    const size_t count = blocks.width * blocks.height;
    std::vector<float> initial(count), expected(count), actual(count);
    for (size_t i = 0; i < count; ++i)
      initial[i] = .25f + float(i % 37) * .09f;
    float quant_dc = 0;
    if (!Ok(ComputeInitialQuantDc(2.4f, &quant_dc)))
      return false;
    const ConstPlaneF32View quant{initial.data(), blocks, blocks.width};
    const AqResidentButteraugliPolicyInput input{
        .adjusted_initial_quant_field = quant,
        .quant_dc = quant_dc,
        .butteraugli_target = 2.4f,
        .iterations = trial % 5,
        .evaluate_final_field = trial % 2 == 0,
        .adjust_initial_field = true};
    VarDctEncoderFrame expected_frame, actual_frame;
    const bool completed = trial % 3 == 0;
    std::unique_ptr<vardct_frame_internal::CompletedVarDctFrame>
        expected_completed, actual_completed;
    std::vector<double> expected_scores, actual_scores;
    if (!Ok(oracle.prepared->PrepareInvariantColorCorrelationResident(
            quant, quant_dc)) ||
        !Ok(oracle.prepared->EvaluateResidentButteraugliPolicy(
            input,
            {.quant_field = {expected.data(), blocks, blocks.width},
             .score_history = &expected_scores,
             .frame = completed ? nullptr : &expected_frame,
             .completed_frame = completed ? &expected_completed : nullptr})) ||
        !Ok(BindMetalAqStrategyDispatchForTesting(
            *oracle.prepared, metadata.descriptor.planes[kMetadataFamilies],
            view)) ||
        !Ok(oracle.prepared->PrepareInvariantColorCorrelationResident(
            quant, quant_dc)))
      return false;
    const auto before = gpu.stats();
    if (!Ok(oracle.prepared->EvaluateResidentButteraugliPolicy(
            input,
            {.quant_field = {actual.data(), blocks, blocks.width},
             .score_history = &actual_scores,
             .frame = completed ? nullptr : &actual_frame,
             .completed_frame = completed ? &actual_completed : nullptr})) ||
        expected != actual || expected_scores != actual_scores ||
        !(completed
              ? expected_completed && actual_completed &&
                    QuantizedCoefficientsEqual(expected_completed->view(),
                                               actual_completed->view())
              : QuantizedCoefficientsEqual(expected_frame, actual_frame)) ||
        gpu.stats().committed_submissions != before.committed_submissions + 1 ||
        (!completed &&
         gpu.stats().successful_allocations != before.successful_allocations)) {
      std::cerr << "Indirect AQ dispatch changed output on trial " << trial
                << '\n';
      return false;
    }
    if (trial == 0) {
      std::vector<double> rejected_scores{-99.0};
      gpu_profile_internal::GpuExecutionProfile profile;
      const auto submissions = gpu.stats().committed_submissions;
      auto *profiler =
          dynamic_cast<gpu_profile_internal::PreparedAqEvaluationProfiler *>(
              oracle.prepared.get());
      if (!profiler ||
          profiler->EvaluateResidentButteraugliPolicyProfiled(
                      input, {.score_history = &rejected_scores},
                      gpu_profile_internal::GpuProfilingMode::kStage, &profile)
                  .code() != StatusCode::kUnavailable ||
          rejected_scores != std::vector<double>{-99.0} ||
          gpu.stats().committed_submissions != submissions)
        return false;
    }
    if (completed) {
      const auto a = expected_completed->view().coefficient_order_population();
      const auto b = actual_completed->view().coefficient_order_population();
      if (a.present_mask != b.present_mask ||
          !std::ranges::equal(a.counts, b.counts)) {
        std::cerr << "Indirect coefficient populations differ on trial "
                  << trial << '\n';
        return false;
      }
    }
    if (!Ok(gpu.CopyDeviceToHost(*view.buffer, poison.data(), poison.size() * 4,
                                 view.offset_bytes)))
      return false;
    if (!std::all_of(poison.begin() + words, poison.end(),
                     [](auto v) { return v == kPoison; })) {
      std::cerr << "Indirect parameters overwrote guards\n";
      return false;
    }
    if (trial < 5) {
      std::fill(actual.begin(), actual.end(), -123.0f);
      std::vector<double> failed_scores{-91.0};
      VarDctEncoderFrame failed_frame;
      if (trial == 0 && !Ok(FailNextMetalAqUploadForTesting(*oracle.prepared)))
        return false;
      if (trial == 1 &&
          !Ok(ArmNextMetalSubmissionFailureForTest(gpu, true, false)))
        return false;
      if (trial == 2 &&
          !Ok(ArmNextMetalSubmissionFailureForTest(gpu, false, true)))
        return false;
      if (trial == 3 &&
          !Ok(FailNextMetalAqReadbackForTesting(*oracle.prepared)))
        return false;
      if (trial == 4 && !Ok(FailNextMetalAqNumericForTesting(*oracle.prepared)))
        return false;
      const AqResidentButteraugliPolicyOutput out{
          .quant_field = {actual.data(), blocks, blocks.width},
          .score_history = &failed_scores,
          .frame = &failed_frame};
      const auto status =
          oracle.prepared->EvaluateResidentButteraugliPolicy(input, out);
      if (status.code() != (trial == 1 ? StatusCode::kSubmissionFailed
                                       : StatusCode::kDeviceError) ||
          failed_frame.valid() || failed_scores != std::vector<double>{-91.0} ||
          !std::ranges::all_of(actual, [](float v) { return v == -123.0f; }) ||
          oracle.prepared->EvaluateResidentButteraugliPolicy(input, out)
                  .code() != StatusCode::kFailedPrecondition) {
        std::cerr << "Indirect dispatch failure was not atomic: " << trial
                  << '\n';
        return false;
      }
      oracle.prepared.reset(); // Release borrower before external buffers.
    } else {
      if (!Ok(BindMetalAqStrategyDispatchForTesting(*oracle.prepared, {}, {})))
        return false;
    }
  }
  std::cout << "27 GPU-family/indirect AQ consumer cases matched CPU dispatch "
               "exactly\n";
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
  MetalBackendOptions options;
  options.forward_dct8 = options.inverse_dct8 = options.forward_dct16x16 =
      options.inverse_dct16x16 = options.forward_dct32x32 =
          options.inverse_dct32x32 = options.forward_dct16x8 =
              options.inverse_dct16x8 = options.forward_dct8x16 =
                  options.inverse_dct8x16 = options.forward_dct32x16 =
                      options.inverse_dct32x16 = options.forward_dct16x32 =
                          options.inverse_dct16x32 =
                              MetalDctImplementation::kSimdgroupMatmul;
  if (!Ok(CreateMetalBackend(library, options, &gpu)))
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
  if (!CheckFailure(*gpu) || !CheckDispatchConsumers(*gpu))
    return EXIT_FAILURE;
  std::cout << cases
            << " exact CPU-builder/GPU-metadata cases passed; guards, failure "
               "and reuse passed\n";
  return EXIT_SUCCESS;
}
