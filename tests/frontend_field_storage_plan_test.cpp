// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#ifndef GJXL_FIELD_PLAN_ORACLE_ONLY
#include "codec/frontend_storage_plan.h"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

#include "codec/adaptive_quantization.h"
#include "codec/chroma_from_luma_internal.h"
#include "codec/prepared_coefficients_internal.h"
#include "codec/quantization.h"
#include "core/image_buffer.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::prepared_coefficients_internal;
namespace cfl = gjxl::chroma_from_luma_internal;

bool Check(bool value, const char *message) {
  if (!value)
    std::cerr << message << '\n';
  return value;
}
bool Ok(const Status &status) {
  if (!status.ok())
    std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                   s.total.pending_count == 0 && s.open_reservations == 0 &&
                   s.waiting_requests == 0,
               "Field operation leaked a reservation or backing");
}

struct Case {
  Extent2D pixels;
  size_t family;
};
constexpr std::array<Case, 13> cases{{{{8, 8}, 0},
                                      {{24, 16}, 0},
                                      {{64, 64}, 1},
                                      {{64, 64}, 2},
                                      {{64, 16}, 3},
                                      {{16, 64}, 4},
                                      {{64, 40}, 5},
                                      {{40, 64}, 6},
                                      {{248, 264}, 7},
                                      {{256, 256}, 7},
                                      {{264, 264}, 7},
                                      {{8200, 8}, 7},
                                      {{8, 8200}, 7}}};
constexpr std::array<AcStrategyType, 7> families{
    AcStrategyType::kDct8,     AcStrategyType::kDct16x16,
    AcStrategyType::kDct32x32, AcStrategyType::kDct16x8,
    AcStrategyType::kDct8x16,  AcStrategyType::kDct32x16,
    AcStrategyType::kDct16x32};
constexpr size_t kOperations = 10;

struct Output {
  Extent2D pixels, blocks;
  size_t block_stride, pixel_stride;
  std::vector<float> quant, strategy, mask;
  std::vector<int32_t> raw;
  Quantizer quantizer;
  ColorCorrelationMap color;
  explicit Output(Extent2D extent)
      : pixels(extent), blocks(BlockGrid::FromPaddedPixelExtent(extent).blocks),
        block_stride(blocks.width + 2), pixel_stride(pixels.width + 3),
        quant(block_stride * blocks.height, -17), strategy(quant),
        mask(pixel_stride * pixels.height, -19), raw(quant.size(), -23) {
    for (size_t y = 0; y < blocks.height; ++y)
      for (size_t x = 0; x < blocks.width; ++x)
        quant[y * block_stride + x] =
            0.1f + 0.01f * ((y * blocks.width + x) % 31);
    if (!Ok(Quantizer::Create({3541, 10}, &quantizer)))
      std::abort();
    const auto tiles = ColorTileExtent(pixels);
    std::vector<int8_t> x(tiles.width * tiles.height, -7), b(x.size(), 11);
    if (!Ok(cfl::CreateColorCorrelationMap({x.data(), tiles, tiles.width},
                                           {b.data(), tiles, tiles.width},
                                           &color)))
      std::abort();
  }
  InitialQuantFieldOutput view() {
    return {{quant.data(), blocks, block_stride},
            {strategy.data(), blocks, block_stride},
            {mask.data(), pixels, pixel_stride}};
  }
};

struct Fixture {
  Image3FBuffer opsin;
  AcStrategyGrid grid;
  PreparedForwardDctCoefficients prepared;
  std::vector<float> field;
  std::vector<int32_t> raw;
  Quantizer quantizer;
  bool Create(Case c) {
    opsin.resize(c.pixels);
    for (size_t channel = 0; channel < 3; ++channel)
      for (size_t i = 0; i < opsin.plane(channel).size(); ++i)
        opsin.plane(channel)[i] =
            0.05f * channel +
            0.002f * static_cast<float>(
                         static_cast<int>((i * 17 + channel * 13) % 113) - 56);
    const auto blocks = BlockGrid::FromPaddedPixelExtent(c.pixels).blocks;
    if (!Ok(AcStrategyGrid::Create(blocks, &grid)) ||
        !Ok(Quantizer::Create({3541, 10}, &quantizer)))
      return false;
    size_t sequence = 0;
    for (size_t y = 0; y < blocks.height; ++y)
      for (size_t x = 0; x < blocks.width; ++x) {
        if (grid.occupied(x, y))
          continue;
        auto strategy = families[c.family < 7 ? c.family : sequence++ % 7];
        const auto covered = GetAcStrategyInfo(strategy)->covered_blocks;
        bool fits = covered.width <= std::min(8 - x % 8, blocks.width - x) &&
                    covered.height <= std::min(8 - y % 8, blocks.height - y);
        for (size_t dy = 0; fits && dy < covered.height; ++dy)
          for (size_t dx = 0; fits && dx < covered.width; ++dx)
            fits = !grid.occupied(x + dx, y + dy);
        if (!fits)
          strategy = AcStrategyType::kDct8;
        if (!Ok(grid.Set(x, y, strategy)))
          return false;
      }
    field.resize(blocks.width * blocks.height);
    raw.resize(field.size());
    for (size_t i = 0; i < field.size(); ++i) {
      field[i] = 0.1f + 0.01f * (i % 31);
      raw[i] = 1 + i % 256;
    }
    thread_budget_internal::EncodeScope serial(1);
    return Ok(
        PrepareForwardDctCoefficients(opsin.const_view(), grid, &prepared));
  }
  Status Run(size_t operation, float target, bool alias, Output *out) const {
    const auto blocks = grid.extent();
    const ConstPlaneI32View raw_view{raw.data(), blocks, blocks.width};
    switch (operation) {
    case 0:
      return ComputeInitialQuantField(opsin.const_view(), {target, 0.9f},
                                      out->view());
    case 1:
      return AdjustQuantField(
          grid, target,
          alias
              ? ConstPlaneF32View{out->quant.data(), blocks, out->block_stride}
              : ConstPlaneF32View{field.data(), blocks, blocks.width},
          out->view().quant_field);
    case 2:
      return CreateQuantizerFromField(
          1.2f, {field.data(), blocks, blocks.width},
          {out->raw.data(), blocks, out->block_stride}, &out->quantizer);
    case 3: // Self-copy also exercises old-output/input aliasing.
      return cfl::CreateColorCorrelationMap(
          out->color.y_to_x_map(), out->color.y_to_b_map(), &out->color);
    case 4:
      return cfl::ComputeInitialColorCorrelationMapFast(opsin.const_view(),
                                                        &out->color);
    case 5:
      return ComputeInitialColorCorrelationMap(opsin.const_view(), &out->color);
    case 6:
    case 7:
      return ComputeFinalColorCorrelationMap(opsin.const_view(), grid, raw_view,
                                             quantizer, operation == 7,
                                             &out->color);
    case 8:
    case 9:
      return cfl::ComputeFinalColorCorrelationMapPrepared(
          prepared, raw_view, quantizer, operation == 9, &out->color);
    default:
      return Status::Internal("Unknown field test operation");
    }
  }
};

// Canonical little-endian exact scalar bits, never struct padding or tolerance.
struct Bytes {
  std::vector<uint8_t> data;
  void Integer(uint64_t value, size_t bytes = 8) {
    for (size_t i = 0; i < bytes; ++i)
      data.push_back((value >> (i * 8)) & 255);
  }
  template <class T> void Range(const T &values) {
    Integer(values.size());
    for (auto value : values) {
      if constexpr (std::is_same_v<decltype(value), float>)
        Integer(std::bit_cast<uint32_t>(value), 4);
      else
        Integer(static_cast<std::make_unsigned_t<decltype(value)>>(value),
                sizeof(value));
    }
  }
};
Bytes Snapshot(const Output &out) {
  Bytes bytes;
  bytes.Range(out.quant);
  bytes.Range(out.strategy);
  bytes.Range(out.mask);
  bytes.Range(out.raw);
  bytes.Integer(out.quantizer.params().global_scale);
  bytes.Integer(out.quantizer.params().quant_dc);
  bytes.Range(out.quantizer.dc_steps());
  bytes.Range(out.quantizer.inverse_dc_steps());
  bytes.Integer(out.color.tile_extent().width);
  bytes.Integer(out.color.tile_extent().height);
  for (auto plane : {out.color.y_to_x_map(), out.color.y_to_b_map()})
    for (size_t y = 0; y < plane.extent.height; ++y)
      bytes.Range(std::span(plane.Row(y), plane.extent.width));
  return bytes;
}
void Emit(const Bytes &bytes) {
  Bytes size;
  size.Integer(bytes.data.size());
  std::cout.write(reinterpret_cast<const char *>(size.data.data()),
                  size.data.size());
  std::cout.write(reinterpret_cast<const char *>(bytes.data.data()),
                  bytes.data.size());
}
bool Oracle() {
  thread_budget_internal::EncodeScope serial(1);
  for (auto c : cases) {
    Fixture f;
    if (!f.Create(c))
      return false;
    for (float target : {1.2f, 4.0f})
      for (size_t operation = 0; operation < kOperations; ++operation) {
        Output out(c.pixels);
        if (!Ok(f.Run(operation, target, false, &out)))
          return false;
        Emit(Snapshot(out));
      }
  }
  return std::cout.good() && Empty(DefaultResourceBudget());
}

#ifndef GJXL_FIELD_PLAN_ORACLE_ONLY
using namespace gjxl::frontend_storage_internal;

bool Plan(Extent2D pixels, size_t threads, size_t operation,
          HostStorageBound *working, HostStorageBound *output) {
  *output = {};
  if (operation == 0) {
    InitialQuantStoragePlan plan;
    if (!Ok(ComputeInitialQuantStoragePlan(pixels, threads, &plan)))
      return false;
    *working = plan.working;
    return true;
  }
  const auto blocks = BlockGrid::FromPaddedPixelExtent(pixels).blocks;
  if (operation == 1)
    return Ok(ComputeQuantFieldAdjustmentStorageBound(blocks, working));
  if (operation == 2)
    return Ok(ComputeQuantizerSelectionStorageBound(blocks, working));
  ColorCorrelationStoragePlan plan;
  const auto mode = operation == 3 ? ColorCorrelationStorageMode::kCopy
                    : operation == 4
                        ? ColorCorrelationStorageMode::kInitialPixel
                        : ColorCorrelationStorageMode::kTransform;
  if (!Ok(ComputeColorCorrelationStoragePlan(pixels, mode, &plan)))
    return false;
  *working = plan.working;
  *output = plan.output;
  return true;
}
bool PurePlans() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.Reserve(1, &job)))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    ArmNextManagedHostAllocationFailureForTest();
    for (auto pixels : {Extent2D{8, 8},
                        {248, 264},
                        {256, 256},
                        {8200, 8},
                        {8, 8200},
                        {3840, 2160},
                        {1ul << 24, 8}}) {
      size_t last = 0;
      for (size_t threads : {1ul, 2ul, 12ul, SIZE_MAX, 0ul}) {
        InitialQuantStoragePlan plan;
        if (!Ok(ComputeInitialQuantStoragePlan(pixels, threads, &plan)))
          return false;
        const size_t n = pixels.width * pixels.height, rows = pixels.height / 4;
        const size_t p =
            n < 65536
                ? 1
                : std::min(rows, threads == 0 ? 12 : std::min(threads, 12ul));
        const size_t row_bytes =
            4 * (n + n / 16 + p * pixels.width) +
            (p > 1 ? rows * sizeof(Status) + p * sizeof(std::thread) : 0);
        const size_t finish_bytes = 4 * (2 * n + n / 16 + 2 * (n / 64));
        if (!Check(plan.maximum_participants == p &&
                       plan.working.peak_bytes ==
                           std::max(row_bytes, finish_bytes) &&
                       plan.working.peak_bytes >= last,
                   "Initial quant phase/worker bound differs"))
          return false;
        last = plan.working.peak_bytes;
      }
      for (size_t op = 0; op < kOperations; ++op) {
        HostStorageBound work, output;
        if (!Plan(pixels, 0, op, &work, &output))
          return false;
        const auto tiles = ColorTileExtent(pixels);
        const size_t t = tiles.width * tiles.height;
        if (!Check(
                work.peak_bytes >= output.peak_bytes &&
                    (op < 3 || output.retained_bytes == 2 * t) &&
                    (op != 1 ||
                     work.peak_bytes == pixels.width * pixels.height / 16) &&
                    (op != 2 ||
                     work.peak_bytes == pixels.width * pixels.height / 8) &&
                    (op != 3 || work.peak_bytes == 2 * t) &&
                    (op != 4 || work.peak_bytes == 4 * t) &&
                    (op < 5 || work.peak_bytes ==
                                   2 * t + 16 * std::min(pixels.width, 64ul) *
                                               std::min(pixels.height, 64ul)),
                "Field or CfL allocation recipe differs"))
          return false;
      }
    }
    if (!Check(ManagedHostAllocationFailurePendingForTest() &&
                   budget.snapshot().peak_backing_bytes == 0,
               "Planning allocated backing"))
      return false;
    DisarmManagedHostAllocationFailureForTest();
  }
  job.Reset();
  InitialQuantStoragePlan initial{17, {23, 29}};
  ColorCorrelationStoragePlan color{19, {31, 37}, {41, 43}};
  HostStorageBound bound{47, 53};
  const auto old_initial = initial;
  const auto old_color = color;
  const auto old_bound = bound;
  for (auto extent :
       {Extent2D{}, {8, 0}, {SIZE_MAX - 7, 8}, {SIZE_MAX - 7, SIZE_MAX - 7}})
    if (!Check(
            !ComputeInitialQuantStoragePlan(extent, 0, &initial).ok() &&
                initial == old_initial &&
                !ComputeColorCorrelationStoragePlan(
                     extent, ColorCorrelationStorageMode::kTransform, &color)
                     .ok() &&
                color == old_color &&
                !ComputeQuantFieldAdjustmentStorageBound(extent, &bound).ok() &&
                bound == old_bound &&
                !ComputeQuantizerSelectionStorageBound(extent, &bound).ok() &&
                bound == old_bound,
            "Invalid field plan changed output"))
      return false;
  return Check(!ComputeInitialQuantStoragePlan({9, 8}, 0, &initial).ok() &&
                   initial == old_initial &&
                   !ComputeInitialQuantStoragePlan({8, 8}, 0, nullptr).ok() &&
                   !ComputeColorCorrelationStoragePlan(
                        {9, 8}, ColorCorrelationStorageMode::kCopy, &color)
                        .ok() &&
                   color == old_color &&
                   !ComputeColorCorrelationStoragePlan(
                        {8, 8}, static_cast<ColorCorrelationStorageMode>(-1),
                        &color)
                        .ok() &&
                   color == old_color &&
                   !ComputeColorCorrelationStoragePlan(
                        {8, 8}, ColorCorrelationStorageMode::kCopy, nullptr)
                        .ok() &&
                   !ComputeQuantFieldAdjustmentStorageBound({1, 1}, nullptr)
                        .ok() &&
                   !ComputeQuantizerSelectionStorageBound({1, 1}, nullptr).ok(),
               "Field plan accepted null, unpadded or invalid mode") &&
         Empty(budget);
}

bool RuntimeCases() {
  size_t checks = 0;
  for (auto c : cases) {
    Fixture f;
    if (!f.Create(c))
      return false;
    for (float target : {1.2f, 4.0f}) {
      std::array<Bytes, kOperations> expected;
      for (size_t op = 0; op < kOperations; ++op) {
        thread_budget_internal::EncodeScope serial(1);
        Output reference(c.pixels);
        if (!Ok(f.Run(op, target, false, &reference)))
          return false;
        expected[op] = Snapshot(reference);
      }
      if (!Check(expected[6].data == expected[8].data &&
                     expected[7].data == expected[9].data,
                 "Prepared/direct final CfL differs"))
        return false;
      for (size_t op = 0; op < kOperations; ++op)
        for (size_t threads : {1ul, 2ul, 12ul, 0ul}) {
          if (op != 0 && threads != 1)
            continue;
          HostStorageBound work, retained;
          if (!Plan(c.pixels, threads, op, &work, &retained))
            return false;
          ResourceBudget budget(work.peak_bytes);
          ResourceReservation job;
          if (!Ok(budget.Reserve(work.peak_bytes, &job)))
            return false;
          {
            Output actual(c.pixels);
            {
              ResourceContextScope resources(
                  {&job, ResourceClass::kPreparation});
              thread_budget_internal::EncodeScope scope(threads);
              if (!Ok(f.Run(op, target, true, &actual)))
                return false;
            }
            const auto s = budget.snapshot();
            if (!Check(Snapshot(actual).data == expected[op].data &&
                           s.peak_backing_bytes <= work.peak_bytes &&
                           s.total.live_capacity_bytes ==
                               retained.retained_bytes &&
                           s.total.backing_count == (op >= 3 ? 2 : 0) &&
                           s.total.pending_count == 0 &&
                           s.classes[static_cast<size_t>(
                                         ResourceClass::kPreparation)]
                                   .live_capacity_bytes ==
                               s.total.live_capacity_bytes,
                       "Field output or live allocation bound differs"))
              return false;
            job.Reset();
            if (!Check(budget.snapshot().total.live_capacity_bytes ==
                           retained.retained_bytes,
                       "Producer closure uncharged retained CfL"))
              return false;
          }
          if (!Empty(budget))
            return false;
          ++checks;
        }
    }
  }
  std::cerr << "Field reservation cases: " << checks << '\n';
  return true;
}

bool LifetimeAndWorkerBoundary() {
  Fixture f;
  if (!f.Create({{264, 264}, 7}))
    return false;
  // The output map from the first call remains charged while its replacement
  // is built. Composition must add the old owner to the operation's working
  // bound, including when the old output is also the copy's borrowed source.
  ColorCorrelationStoragePlan plan;
  if (!Ok(ComputeColorCorrelationStoragePlan(
          f.opsin.extent(), ColorCorrelationStorageMode::kCopy, &plan)))
    return false;
  HostStorageBound overlap = plan.working;
  if (!overlap.Add(plan.output))
    return false;
  ResourceBudget budget(overlap.peak_bytes);
  ResourceReservation job;
  if (!Ok(budget.Reserve(overlap.peak_bytes, &job)))
    return false;
  {
    Output actual(f.opsin.extent());
    const auto before = Snapshot(actual);
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    if (!Ok(f.Run(3, 1.2f, true, &actual)) ||
        !Ok(f.Run(3, 1.2f, true, &actual)) ||
        !Check(Snapshot(actual).data == before.data &&
                   budget.snapshot().peak_backing_bytes == overlap.peak_bytes &&
                   budget.snapshot().total.live_capacity_bytes ==
                       plan.output.retained_bytes,
               "CfL replacement did not account old/new overlap"))
      return false;
    job.Reset();
  }
  if (!Empty(budget))
    return false;

  if (std::thread::hardware_concurrency() < 2)
    return true;
  // Admit the two common arrays and dispatcher, but no worker's row scratch.
  // Failure must propagate from the worker's inherited reservation unchanged.
  const auto pixels = f.opsin.extent();
  const size_t n = pixels.width * pixels.height;
  const size_t credit = 4 * (n + n / 16) +
                        (pixels.height / 4) * sizeof(Status) +
                        sizeof(std::thread);
  ResourceBudget worker_budget(credit);
  ResourceReservation worker_job;
  if (!Ok(worker_budget.Reserve(credit, &worker_job)))
    return false;
  {
    Output actual(pixels);
    const auto before = Snapshot(actual);
    ResourceContextScope resources({&worker_job, ResourceClass::kPreparation});
    thread_budget_internal::EncodeScope threads(2);
    const auto status = f.Run(0, 1.2f, false, &actual);
    if (!Check(
            status.resource_plan_exceeded() &&
                Snapshot(actual).data == before.data &&
                worker_budget.snapshot().total.backing_count == 0 &&
                worker_budget.snapshot().total.pending_count == 0,
            "Initial quant worker escaped its reservation or changed output"))
      return false;
  }
  worker_job.Reset();
  return Empty(worker_budget);
}

bool Faults() {
  Fixture f;
  if (!f.Create({{72, 16}, 7}))
    return false;
  thread_budget_internal::EncodeScope serial(1);
  for (size_t op = 0; op < kOperations; ++op) {
    HostStorageBound work, retained;
    if (!Plan(f.opsin.extent(), 1, op, &work, &retained))
      return false;
    bool complete = false, underplan_verified = false;
    for (size_t failure = 0; failure < 256; ++failure) {
      const bool underplan = complete;
      ResourceBudget budget(work.peak_bytes);
      ResourceReservation job;
      if (!Ok(budget.Reserve(work.peak_bytes, &job)) ||
          (underplan && !Ok(job.ReduceCapacity(0))))
        return false;
      bool injected = false;
      {
        Output actual(f.opsin.extent());
        const auto before = Snapshot(actual);
        ResourceContextScope resources({&job, ResourceClass::kPreparation});
        ArmManagedHostAllocationFailureAfterForTest(underplan ? 0 : failure);
        const auto status = f.Run(op, 4.0f, true, &actual);
        injected = !ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (underplan || injected) {
          if (!Check(
                  status.code() == StatusCode::kOutOfMemory &&
                      status.resource_plan_exceeded() == underplan &&
                      injected != underplan &&
                      Snapshot(actual).data == before.data &&
                      budget.snapshot().total.backing_count == 0 &&
                      budget.snapshot().total.pending_count == 0,
                  "Field failure changed output, leaked or escaped its plan"))
            return false;
        } else if (!Ok(status))
          return false;
      }
      job.Reset();
      if (!Empty(budget))
        return false;
      if (underplan) {
        underplan_verified = true;
        break;
      }
      if (!injected) {
        complete = true;
        std::cerr << "Field allocation failure positions " << op << ": "
                  << failure << '\n';
      }
    }
    if (!Check(complete && underplan_verified,
               "Field fault sweep did not finish"))
      return false;
  }
  return true;
}
#endif
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--oracle")
    return Oracle() ? EXIT_SUCCESS : EXIT_FAILURE;
#ifndef GJXL_FIELD_PLAN_ORACLE_ONLY
  return PurePlans() && RuntimeCases() && LifetimeAndWorkerBoundary() &&
                 Faults() && Empty(DefaultResourceBudget()) &&
                 Check(DefaultResourceBudget().snapshot().peak_backing_bytes ==
                           0,
                       "Field test escaped to the default domain")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
#else
  return EXIT_FAILURE;
#endif
}
