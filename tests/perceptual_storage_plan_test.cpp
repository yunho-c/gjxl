// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#ifndef GJXL_PERCEPTUAL_PLAN_ORACLE_ONLY
#include "codec/frontend_storage_plan.h"
#include "codec/perceptual_storage_plan.h"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

#include "codec/butteraugli.h"
#include "codec/butteraugli_distance_internal.h"
#include "codec/color_transform.h"
#include "codec/color_transform_internal.h"
#include "codec/loop_filter.h"
#include "codec/maximum_error.h"
#include "core/image_buffer.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
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
               "Perceptual operation leaked backing or admission");
}
constexpr std::array<Extent2D, 13> cases{{{1, 1},
                                          {1, 19},
                                          {19, 1},
                                          {7, 9},
                                          {8, 8},
                                          {14, 16},
                                          {15, 15},
                                          {16, 17},
                                          {33, 35},
                                          {64, 40},
                                          {247, 263},
                                          {256, 256},
                                          {263, 263}}};
constexpr size_t kOperations = 18;

struct Image {
  Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> planes;
  explicit Image(Extent2D e) : extent(e), stride(e.width + 3) {
    for (auto &p : planes)
      p.resize(stride * e.height, -17);
    Fill(0);
  }
  void Fill(size_t variant) {
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x)
          planes[c][y * stride + x] =
              0.05f + 0.1f * c + 0.003f * ((x * 17 + y * 23 + c * 13) % 173) +
              0.0003f * variant * (1 + (x + y + c) % 7);
  }
  Image3FView View() {
    return {{{{planes[0].data(), extent, stride},
              {planes[1].data(), extent, stride},
              {planes[2].data(), extent, stride}}}};
  }
  ConstImage3FView ConstView() const {
    return {{{{planes[0].data(), extent, stride},
              {planes[1].data(), extent, stride},
              {planes[2].data(), extent, stride}}}};
  }
};
struct Output {
  Image image;
  Extent2D blocks, pixels;
  size_t block_stride, map_stride;
  std::vector<float> block, map;
  double score = -29;
  MaximumErrorReduction maximum{{-1, -2, -3}, -4};
  Output(Extent2D source, Extent2D destination)
      : image(destination), blocks(source.ceil_div(8)), pixels(source),
        block_stride(blocks.width + 2), map_stride(source.width + 3),
        block(block_stride * blocks.height, -19),
        map(map_stride * pixels.height, -23) {}
  PlaneF32View BlockView() { return {block.data(), blocks, block_stride}; }
  PlaneF32View MapView() { return {map.data(), pixels, map_stride}; }
};
struct Fixture {
  Extent2D pixels, padded, blocks;
  Image reference, distorted, padded_reference;
  AcStrategyGrid grid;
  Quantizer quantizer;
  std::vector<float> sigma, distance;
  std::vector<int32_t> raw;
  std::vector<uint8_t> sharp;
  explicit Fixture(Extent2D e)
      : pixels(e), padded{e.ceil_div(8).width * 8, e.ceil_div(8).height * 8},
        blocks(e.ceil_div(8)), reference(e), distorted(e),
        padded_reference(padded), sigma(blocks.width * blocks.height, -1.0f),
        distance(e.width * e.height), raw(sigma.size(), 17),
        sharp(sigma.size(), 4) {
    distorted.Fill(1);
    for (size_t i = 0; i < distance.size(); ++i)
      distance[i] = 0.01f * (i % 29);
    if (!Ok(Quantizer::Create({3541, 10}, &quantizer)) ||
        !Ok(AcStrategyGrid::Create(blocks, &grid)))
      std::abort();
    for (size_t y = 0; y < blocks.height; ++y)
      for (size_t x = 0; x < blocks.width; ++x)
        if (!grid.occupied(x, y)) {
          const auto strategy = x + 1 < blocks.width && y + 1 < blocks.height &&
                                        x % 2 == 0 && y % 2 == 0
                                    ? AcStrategyType::kDct16x16
                                    : AcStrategyType::kDct8;
          if (!Ok(grid.Set(x, y, strategy)))
            std::abort();
        }
  }
  Extent2D Destination(size_t op) const { return op == 2 ? padded : pixels; }
  Status Run(size_t op, bool alias, Output *out) const {
    const auto input =
        alias && op != 2 ? out->image.ConstView() : reference.ConstView();
    if (op == 0)
      return LinearRgbToOpsin(input, 80, out->image.View());
    if (op == 1)
      return OpsinToLinearRgb(input, 80, out->image.View());
    if (op == 2)
      return color_transform_internal::LinearRgbToPaddedOpsin(
          reference.ConstView(), 80, out->image.View());
    if (op == 3)
      return ApplyGaborishInverse(input, {0.97f, 1.0f, 1.03f},
                                  out->image.View());
    if (op == 4)
      return ApplyGaborish(input, {}, out->image.View());
    if (op >= 5 && op <= 12) {
      LoopFilterOptions options;
      options.gaborish = op >= 9;
      options.epf_options.iterations = (op - 5) % 4;
      return ApplyLoopFilters(input, {sigma.data(), blocks, blocks.width},
                              options, out->image.View());
    }
    if (op == 13)
      return ComputeEpfInverseSigma(
          grid, {raw.data(), blocks, blocks.width}, quantizer,
          {sharp.data(), blocks, blocks.width}, {}, out->BlockView());
    if (op == 14)
      return ReduceButteraugliDistanceMap(
          {distance.data(), pixels, pixels.width}, grid, out->BlockView());
    if (op == 15)
      return ReduceMaximumError(
          padded_reference.ConstView(), distorted.ConstView(), pixels, grid,
          {0.01f, 0.02f, 0.03f}, out->BlockView(), &out->maximum);
    if (op == 16)
      return ComputeButteraugliDistance(reference.ConstView(),
                                        distorted.ConstView(), {},
                                        out->MapView(), &out->score);
    PreparedButteraugliReference prepared;
    auto status = prepared.Prepare(reference.ConstView(), {});
    if (!status.ok())
      return status;
    return prepared.Compare(distorted.ConstView(), out->MapView(), &out->score);
  }
};

struct Bytes {
  std::vector<uint8_t> data;
  void Integer(uint64_t v, size_t n = 8) {
    for (size_t i = 0; i < n; ++i)
      data.push_back((v >> (i * 8)) & 255);
  }
  void Float(float v) { Integer(std::bit_cast<uint32_t>(v), 4); }
  template <class T> void Range(const T &a) {
    Integer(a.size());
    for (float v : a)
      Float(v);
  }
};
Bytes Snapshot(const Output &out) {
  Bytes b;
  for (const auto &p : out.image.planes)
    b.Range(p);
  b.Range(out.block);
  b.Range(out.map);
  b.Integer(std::bit_cast<uint64_t>(out.score));
  b.Range(out.maximum.channel_maximum);
  b.Float(out.maximum.normalized_maximum);
  return b;
}
bool Oracle() {
  thread_budget_internal::EncodeScope serial(1);
  for (auto e : cases) {
    Fixture f(e);
    for (size_t op = 0; op < kOperations; ++op) {
      Output out(e, f.Destination(op));
      if (!Ok(f.Run(op, false, &out)))
        return false;
      const auto b = Snapshot(out);
      Bytes prefix;
      prefix.Integer(b.data.size());
      std::cout.write(reinterpret_cast<const char *>(prefix.data.data()),
                      prefix.data.size());
      std::cout.write(reinterpret_cast<const char *>(b.data.data()),
                      b.data.size());
    }
  }
  return std::cout.good() && Empty(DefaultResourceBudget());
}
// Runnable against both revisions to demonstrate the six wrapper fixes.
bool UnderplanOracle(bool require_typed = false) {
  Fixture f({263, 263});
  thread_budget_internal::EncodeScope threads(2);
  for (size_t op : {0ul, 1ul, 2ul, 3ul, 4ul, 10ul}) {
    // The direct padded path has no managed allocations when run serially.
    if (op == 2 && std::thread::hardware_concurrency() < 2)
      continue;
    ResourceBudget budget(1);
    ResourceReservation job;
    if (!Ok(budget.Reserve(1, &job)) || !Ok(job.ReduceCapacity(0)))
      return false;
    Output out(f.pixels, f.Destination(op));
    const auto before = Snapshot(out);
    Status status;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      ArmNextManagedHostAllocationFailureForTest();
      status = f.Run(op, false, &out);
      if (!Check(ManagedHostAllocationFailurePendingForTest(),
                 "Underplan reached physical allocation"))
        return false;
      DisarmManagedHostAllocationFailureForTest();
    }
    if (!Check(status.code() == StatusCode::kOutOfMemory &&
                   Snapshot(out).data == before.data,
               "Underplan changed output or status code"))
      return false;
    if (!Check(!require_typed || status.resource_plan_exceeded(),
               "Wrapper erased the typed underplan failure"))
      return false;
    std::cout << op << ' ' << status.resource_plan_exceeded() << '\n';
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  return true;
}

#ifndef GJXL_PERCEPTUAL_PLAN_ORACLE_ONLY
using namespace gjxl::frontend_storage_internal;
bool Plan(const Fixture &f, size_t threads, size_t op, HostStorageBound *out) {
  if (op < 3) {
    ColorTransformStoragePlan plan;
    if (!Ok(ComputeColorTransformStoragePlan(f.pixels, f.Destination(op),
                                             op == 2, threads, &plan)))
      return false;
    *out = plan.working;
    return true;
  }
  if (op <= 4)
    return Ok(ComputeLoopFilterStorageBound(f.pixels, true, 0, out));
  if (op <= 12)
    return Ok(
        ComputeLoopFilterStorageBound(f.pixels, op >= 9, (op - 5) % 4, out));
  if (op <= 15)
    return Ok(ComputeBlockReductionStorageBound(f.blocks, out));
  NativeButteraugliStoragePlan plan;
  if (!Ok(ComputeNativeButteraugliStoragePlan(f.pixels, &plan)))
    return false;
  *out = op == 16 ? plan.one_shot : plan.comparison;
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
    for (auto e : {Extent2D{1, 1},
                   {7, 19},
                   {14, 16},
                   {15, 15},
                   {263, 263},
                   {3840, 2160},
                   {1ul << 24, 1}}) {
      NativeButteraugliStoragePlan native;
      if (!Ok(ComputeNativeButteraugliStoragePlan(e, &native)))
        return false;
      const size_t n = e.width * e.height;
      const size_t m =
          native.working_extent.width * native.working_extent.height;
      const size_t s = native.sub_extent.width * native.sub_extent.height;
      const size_t expected =
          4 * (63 * m + 51 + n + (native.working_extent != e ? 3 * m : 0) +
               (s ? 66 * s + 51 : 0));
      const size_t one_retained =
          4 *
          (66 * m + 15 * s + n + 51 + (native.working_extent != e ? 6 * m : 0));
      if (!Check(native.prepared.retained_bytes == expected &&
                     native.preparation.peak_bytes == expected &&
                     native.comparison.peak_bytes == expected + 40 * m &&
                     native.one_shot.retained_bytes == one_retained &&
                     native.one_shot.peak_bytes == one_retained + 44 * m,
                 "Native Butteraugli owner formula differs"))
        return false;
      for (size_t threads : {0ul, 1ul, 2ul, 12ul, SIZE_MAX}) {
        ColorTransformStoragePlan color;
        if (!Ok(ComputeColorTransformStoragePlan(e, e, false, threads, &color)))
          return false;
        const size_t p =
            n < 65536 ? 1
                      : std::min(e.height,
                                 threads == 0 ? 12 : std::min(threads, 12ul));
        if (!Check(color.maximum_participants == p &&
                       color.working.peak_bytes ==
                           12 * n + (p > 1 ? e.height * sizeof(Status) +
                                                 p * sizeof(std::thread)
                                           : 0),
                   "Color-transform dispatch bound differs"))
          return false;
        if (!Ok(ComputeColorTransformStoragePlan(e, e, true, threads,
                                                 &color)) ||
            !Check(
                color.working.peak_bytes ==
                    (p > 1 ? e.height * sizeof(Status) + p * sizeof(std::thread)
                           : 0),
                "Direct color transform added a temporary image"))
          return false;
      }
      for (bool g : {false, true})
        for (size_t iterations = 0; iterations <= 3; ++iterations) {
          HostStorageBound filter;
          const size_t images = g ? (iterations == 0   ? 1
                                     : iterations == 3 ? 3
                                                       : 2)
                                  : (iterations == 0   ? 0
                                     : iterations == 3 ? 2
                                                       : 1);
          if (!Ok(ComputeLoopFilterStorageBound(e, g, iterations, &filter)) ||
              !Check(filter.peak_bytes == 12 * n * images,
                     "Nested filter scratch differs"))
            return false;
        }
    }
    if (!Check(ManagedHostAllocationFailurePendingForTest() &&
                   budget.snapshot().peak_backing_bytes == 0,
               "Preprocessing/perceptual planning allocated backing"))
      return false;
    DisarmManagedHostAllocationFailureForTest();
  }
  job.Reset();
  NativeButteraugliStoragePlan native;
  native.working_extent = {23, 29};
  ColorTransformStoragePlan color{17, {31, 37}};
  HostStorageBound bound{41, 43};
  const auto old_native = native;
  const auto old_color = color;
  const auto old_bound = bound;
  for (auto e :
       {Extent2D{}, {1, 0}, {SIZE_MAX, 8}, {SIZE_MAX - 7, SIZE_MAX - 7}})
    if (!Check(!ComputeNativeButteraugliStoragePlan(e, &native).ok() &&
                   native == old_native &&
                   !ComputeColorTransformStoragePlan(e, e, false, 0, &color)
                        .ok() &&
                   color == old_color &&
                   !ComputeLoopFilterStorageBound(e, true, 3, &bound).ok() &&
                   bound == old_bound &&
                   !ComputeBlockReductionStorageBound(e, &bound).ok() &&
                   bound == old_bound,
               "Invalid perceptual plan changed output"))
      return false;
  return Check(
             !ComputeNativeButteraugliStoragePlan({1, 1}, nullptr).ok() &&
                 !ComputeColorTransformStoragePlan({2, 1}, {1, 1}, true, 0,
                                                   &color)
                      .ok() &&
                 color == old_color &&
                 !ComputeColorTransformStoragePlan({1, 1}, {2, 1}, false, 0,
                                                   &color)
                      .ok() &&
                 color == old_color &&
                 !ComputeColorTransformStoragePlan({1, 1}, {1, 1}, false, 0,
                                                   nullptr)
                      .ok() &&
                 !ComputeLoopFilterStorageBound({1, 1}, true, 4, &bound).ok() &&
                 bound == old_bound &&
                 !ComputeLoopFilterStorageBound({1, 1}, true, 0, nullptr)
                      .ok() &&
                 !ComputeBlockReductionStorageBound({1, 1}, nullptr).ok(),
             "Invalid mode or null planner accepted") &&
         Empty(budget);
}
bool RuntimeCases() {
  size_t checks = 0;
  for (auto e : cases) {
    Fixture f(e);
    std::array<Bytes, kOperations> expected;
    for (size_t op = 0; op < kOperations; ++op) {
      thread_budget_internal::EncodeScope serial(1);
      Output reference(e, f.Destination(op));
      if (!Ok(f.Run(op, false, &reference)))
        return false;
      expected[op] = Snapshot(reference);
    }
    if (!Check(expected[16].data == expected[17].data,
               "Prepared and one-shot native outputs differ"))
      return false;
    for (size_t op = 0; op < kOperations; ++op)
      for (size_t threads : {1ul, 2ul, 12ul, 0ul}) {
        if (op >= 3 && threads != 1)
          continue;
        HostStorageBound work;
        if (!Plan(f, threads, op, &work))
          return false;
        ResourceBudget budget(std::max(size_t{1}, work.peak_bytes));
        ResourceReservation job;
        if (!Ok(budget.Reserve(std::max(size_t{1}, work.peak_bytes), &job)) ||
            (work.peak_bytes == 0 && !Ok(job.ReduceCapacity(0))))
          return false;
        Output actual(e, f.Destination(op));
        {
          ResourceContextScope resources({&job, ResourceClass::kPreparation});
          thread_budget_internal::EncodeScope scope(threads);
          if (!Ok(f.Run(op, true, &actual)))
            return false;
        }
        if (!Check(Snapshot(actual).data == expected[op].data &&
                       budget.snapshot().peak_backing_bytes <=
                           work.peak_bytes &&
                       budget.snapshot().total.backing_count == 0 &&
                       budget.snapshot().total.pending_count == 0,
                   "Preprocessing/perceptual output or working bound differs"))
          return false;
        job.Reset();
        if (!Empty(budget))
          return false;
        ++checks;
      }
  }
  std::cerr << "Perceptual reservation cases: " << checks << '\n';
  return true;
}
bool PreparedLifetimes() {
  for (auto e : {Extent2D{1, 1}, {7, 19}, {14, 16}, {15, 15}, {33, 35}}) {
    Fixture f(e);
    NativeButteraugliStoragePlan plan;
    if (!Ok(ComputeNativeButteraugliStoragePlan(e, &plan)))
      return false;
    // A second producer below reserves a COMPLETE envelope while the first
    // producer's reference remains charged. Do not infer cross-reservation
    // incremental credit by subtracting retained bytes from a peak bound:
    // newly replaced psychoimages remain owned by the second reservation.
    ResourceBudget budget(plan.prepared.retained_bytes +
                          plan.comparison.peak_bytes);
    ResourceReservation job;
    if (!Ok(budget.Reserve(plan.comparison.peak_bytes, &job)))
      return false;
    {
      PreparedButteraugliReference prepared;
      {
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        if (!Ok(prepared.Prepare(f.reference.ConstView(), {})))
          return false;
      }
      if (!Check(budget.snapshot().total.live_capacity_bytes ==
                         plan.prepared.retained_bytes &&
                     budget.snapshot().peak_backing_bytes ==
                         plan.preparation.peak_bytes,
                 "Eager native reference backing differs from its exact plan"))
        return false;
      for (size_t variant = 1; variant <= 3; ++variant) {
        f.distorted.Fill(variant);
        Output expected(e, e), actual(e, e);
        if (!Ok(f.Run(16, false, &expected)))
          return false;
        {
          ResourceContextScope scope({&job, ResourceClass::kPreparation});
          if (!Ok(prepared.Compare(f.distorted.ConstView(), actual.MapView(),
                                   &actual.score)))
            return false;
        }
        if (!Check(Snapshot(actual).data == Snapshot(expected).data &&
                       budget.snapshot().total.live_capacity_bytes ==
                           plan.prepared.retained_bytes &&
                       budget.snapshot().peak_backing_bytes ==
                           plan.comparison.peak_bytes,
                   "Native comparison lost backing or omitted replacement "
                   "overlap"))
          return false;
      }
      job.Reset();
      if (!Check(budget.snapshot().total.live_capacity_bytes ==
                     plan.prepared.retained_bytes,
                 "Closing producer uncharged prepared reference"))
        return false;
      ResourceReservation next;
      if (!Ok(budget.Reserve(plan.comparison.peak_bytes, &next)))
        return false;
      {
        Output actual(e, e);
        ResourceContextScope scope({&next, ResourceClass::kPreparation});
        if (!Ok(prepared.Compare(f.distorted.ConstView(), actual.MapView(),
                                 &actual.score)))
          return false;
      }
      next.Reset();
    }
    if (!Empty(budget))
      return false;
    // Internal one-shot scratch keeps differently sized main/subscale arrays
    // between calls. Cover successful reuse, not only public fresh scratch.
    ResourceBudget reuse(plan.one_shot.peak_bytes);
    ResourceReservation work;
    if (!Ok(reuse.Reserve(plan.one_shot.peak_bytes, &work)))
      return false;
    {
      butteraugli_internal::NativeButteraugliScratch scratch;
      for (size_t i = 0; i < 3; ++i) {
        Output actual(e, e), expected(e, e);
        f.distorted.Fill(i + 1);
        if (!Ok(f.Run(16, false, &expected)))
          return false;
        {
          ResourceContextScope scope({&work, ResourceClass::kPreparation});
          if (!Ok(butteraugli_internal::ComputeButteraugliDistanceNative(
                  f.reference.ConstView(), f.distorted.ConstView(), {},
                  &scratch, actual.MapView(), &actual.score)))
            return false;
        }
        if (!Check(Snapshot(actual).data == Snapshot(expected).data &&
                       reuse.snapshot().peak_backing_bytes <=
                           plan.one_shot.peak_bytes,
                   "Reused multiscale scratch exceeded plan or changed output"))
          return false;
      }
    }
    work.Reset();
    if (!Empty(reuse))
      return false;
  }
  return true;
}
bool ReferenceFailures() {
  Fixture f({17, 19});
  NativeButteraugliStoragePlan plan;
  if (!Ok(ComputeNativeButteraugliStoragePlan(f.pixels, &plan)))
    return false;
  Output expected(f.pixels, f.pixels);
  if (!Ok(f.Run(16, false, &expected)))
    return false;
  for (bool reprepare : {true, false}) {
    bool complete = false;
    for (size_t failure = 0; failure < 128; ++failure) {
      ResourceBudget budget(plan.comparison.peak_bytes);
      ResourceReservation job;
      if (!Ok(budget.Reserve(plan.comparison.peak_bytes, &job)))
        return false;
      bool injected = false;
      {
        PreparedButteraugliReference prepared;
        // A previous caller-owned reference is separate from re-preparation's
        // new owner. Comparison instead retains the reference on this job.
        if (reprepare) {
          if (!Ok(prepared.Prepare(f.reference.ConstView(), {})))
            return false;
        } else {
          ResourceContextScope scope({&job, ResourceClass::kPreparation});
          if (!Ok(prepared.Prepare(f.reference.ConstView(), {})))
            return false;
        }
        Output actual(f.pixels, f.pixels);
        const auto before = Snapshot(actual);
        Status status;
        {
          ResourceContextScope scope({&job, ResourceClass::kPreparation});
          ArmManagedHostAllocationFailureAfterForTest(failure);
          status =
              reprepare
                  ? prepared.Prepare(f.distorted.ConstView(), {1.2f, 0.7f, 160})
                  : prepared.Compare(f.distorted.ConstView(), actual.MapView(),
                                     &actual.score);
          injected = !ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
        }
        if (injected) {
          if (!Check(status.code() == StatusCode::kOutOfMemory &&
                         !status.resource_plan_exceeded() && prepared.ready() &&
                         prepared.extent() == f.pixels &&
                         prepared.options() == ButteraugliOptions{} &&
                         Snapshot(actual).data == before.data &&
                         budget.snapshot().total.pending_count == 0 &&
                         budget.snapshot().total.live_capacity_bytes ==
                             (reprepare ? 0 : plan.prepared.retained_bytes),
                     "Native reference failure changed reference/output or "
                     "leaked staging"))
            return false;
          {
            ResourceContextScope scope({&job, ResourceClass::kPreparation});
            if (!Ok(prepared.Compare(f.distorted.ConstView(), actual.MapView(),
                                     &actual.score)))
              return false;
          }
          if (!Check(
                  Snapshot(actual).data == Snapshot(expected).data,
                  "Native reference was not reusable after allocation failure"))
            return false;
        } else if (!Ok(status))
          return false;
      }
      job.Reset();
      if (!Empty(budget))
        return false;
      if (!injected) {
        complete = true;
        std::cerr << "Native reference failure positions "
                  << (reprepare ? "prepare" : "compare") << ": " << failure
                  << '\n';
        break;
      }
    }
    if (!Check(complete, "Native reference failure sweep did not finish"))
      return false;
  }
  return true;
}

bool Faults() {
  Fixture f({17, 19});
  thread_budget_internal::EncodeScope serial(1);
  for (size_t op = 0; op < kOperations; ++op) {
    HostStorageBound work;
    if (!Plan(f, 1, op, &work))
      return false;
    if (work.peak_bytes == 0)
      continue;
    bool complete = false, underplan_verified = false;
    for (size_t failure = 0; failure < 512; ++failure) {
      const bool underplan = complete;
      ResourceBudget budget(work.peak_bytes);
      ResourceReservation job;
      if (!Ok(budget.Reserve(work.peak_bytes, &job)) ||
          (underplan && !Ok(job.ReduceCapacity(0))))
        return false;
      Output actual(f.pixels, f.Destination(op));
      const auto before = Snapshot(actual);
      bool injected;
      {
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        ArmManagedHostAllocationFailureAfterForTest(underplan ? 0 : failure);
        const auto status = f.Run(op, true, &actual);
        injected = !ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (underplan || injected) {
          if (!Check(status.code() == StatusCode::kOutOfMemory &&
                         status.resource_plan_exceeded() == underplan &&
                         injected != underplan &&
                         Snapshot(actual).data == before.data,
                     "Perceptual failure changed output or erased the "
                     "underplan reason"))
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
        std::cerr << "Perceptual allocation failure positions " << op << ": "
                  << failure << '\n';
      }
    }
    if (!Check(complete && underplan_verified,
               "Incomplete perceptual failure sweep"))
      return false;
  }
  return true;
}
#endif
} // namespace
int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--oracle")
    return Oracle() ? EXIT_SUCCESS : EXIT_FAILURE;
  if (argc == 2 && std::string_view(argv[1]) == "--underplan")
    return UnderplanOracle() ? EXIT_SUCCESS : EXIT_FAILURE;
#ifndef GJXL_PERCEPTUAL_PLAN_ORACLE_ONLY
  return PurePlans() && UnderplanOracle(true) && RuntimeCases() &&
                 PreparedLifetimes() && ReferenceFailures() && Faults() &&
                 Empty(DefaultResourceBudget()) &&
                 Check(DefaultResourceBudget().snapshot().peak_backing_bytes ==
                           0,
                       "Perceptual test escaped to default domain")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
#else
  return EXIT_FAILURE;
#endif
}
