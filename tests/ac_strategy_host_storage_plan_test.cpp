// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/ac_strategy_search_internal.h"
#include "gpu/ops/ac_strategy.h"
#include "gpu/ops/ac_strategy_search.h"
#include "gpu/ops/ac_strategy_storage_plan.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using ac_strategy_internal::ComputeSearchStoragePlan;
using ac_strategy_internal::SearchStoragePlan;
using ac_strategy_search_internal::ComputeHostStoragePlan;
using ac_strategy_search_internal::HostStoragePlan;

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
               "AC search leaked backing or admission");
}

// Isolate the backend-independent host contribution. These test-only device
// bytes and synchronous submission are deliberately NOT managed allocations.
// This exercises the production candidate packing, retained vectors, readback
// and CPU placement, not Metal kernel arithmetic or its submission metadata.
class Buffer final : public DeviceBuffer {
public:
  Buffer(BackendId id, size_t bytes)
      : DeviceBuffer(BackendKind::kMetal, id, bytes), data(bytes) {}
  std::vector<std::byte> data;
};
class Submission final : public GpuSubmission {
public:
  Status Wait() override { return Status::Ok(); }
};
float Cost(AcStrategyType strategy, size_t x, size_t y, size_t variant) {
  // Include exact ties as well as changed decisions without floating error.
  return variant == 0 ? 64.0f
                      : float(1 + (size_t(strategy) * 17 + x * 13 + y * 7 +
                                   variant * 19) %
                                      257);
}
class Backend final : public GpuBackend, public GpuAcStrategyEvaluation {
public:
  size_t variant = 0;
  bool fail_submission = false;
  BackendKind kind() const noexcept override { return BackendKind::kMetal; }
  std::string_view name() const noexcept override {
    return "host-plan fixture";
  }
  Status Allocate(size_t bytes, std::unique_ptr<DeviceBuffer> *out) override {
    *out = std::make_unique<Buffer>(id(), bytes);
    RecordSuccessfulAllocation();
    return Status::Ok();
  }
  Status CopyHostToDevice(DeviceBuffer &dst, const void *src, size_t bytes,
                          size_t offset = 0) override {
    if (!owns(dst) || offset > dst.size_bytes() ||
        bytes > dst.size_bytes() - offset)
      return Status::InvalidArgument("Test upload range");
    std::memcpy(static_cast<Buffer &>(dst).data.data() + offset, src, bytes);
    return Status::Ok();
  }
  Status CopyDeviceToHost(const DeviceBuffer &src, void *dst, size_t bytes,
                          size_t offset = 0) override {
    if (!owns(src) || offset > src.size_bytes() ||
        bytes > src.size_bytes() - offset)
      return Status::InvalidArgument("Test readback range");
    std::memcpy(dst, static_cast<const Buffer &>(src).data.data() + offset,
                bytes);
    return Status::Ok();
  }
  Status ForwardTransform(const TransformBatch &,
                          std::unique_ptr<GpuSubmission> *) override {
    return Status::Unavailable("Test backend has no transforms");
  }
  Status InverseTransform(const TransformBatch &,
                          std::unique_ptr<GpuSubmission> *) override {
    return Status::Unavailable("Test backend has no transforms");
  }
  Status EvaluateAcStrategyCandidateBatches(
      std::span<const AcStrategyCandidateBatch> batches,
      std::unique_ptr<GpuSubmission> *out) override {
    if (fail_submission)
      return Status::Internal("Injected test submission failure");
    for (const auto &batch : batches) {
      for (size_t i = 0; i < batch.candidate_count; ++i) {
        AcStrategyCandidate candidate;
        auto status =
            CopyDeviceToHost(*batch.candidates, &candidate, sizeof(candidate),
                             i * sizeof(candidate));
        if (!status.ok())
          return status;
        const float cost =
            Cost(batch.strategy, candidate.block_x, candidate.block_y, variant);
        status = CopyHostToDevice(*batch.costs, &cost, sizeof(cost),
                                  i * sizeof(cost));
        if (!status.ok())
          return status;
      }
    }
    *out = std::make_unique<Submission>();
    RecordCommittedSubmission();
    return Status::Ok();
  }
};

struct Fixture {
  Extent2D pixels, blocks;
  size_t stride;
  std::array<std::vector<float>, 3> image;
  std::vector<float> quant, mask;
  std::array<std::vector<float>, kAcStrategyCount> costs;
  ColorCorrelationMap cfl;
  Buffer device;
  Fixture(Backend &backend, Extent2D e, size_t variant)
      : pixels(e), blocks(e.width / 8, e.height / 8), stride(e.width + 3),
        quant((blocks.width + 2) * blocks.height, 0.4f),
        mask(stride * e.height, 46.0f),
        device(backend.id(), stride * e.height * sizeof(float)) {
    for (size_t c = 0; c < 3; ++c) {
      image[c].resize(stride * e.height, -777.0f);
      for (size_t y = 0; y < e.height; ++y)
        for (size_t x = 0; x < e.width; ++x)
          image[c][y * stride + x] =
              0.1f + 0.05f * c + 0.001f * ((x * 7 + y * 11 + variant) % 37);
    }
    for (size_t s = 0; s < costs.size(); ++s) {
      costs[s].resize(blocks.width * blocks.height);
      for (size_t y = 0; y < blocks.height; ++y)
        for (size_t x = 0; x < blocks.width; ++x)
          costs[s][y * blocks.width + x] =
              Cost(AcStrategyType(s), x, y, variant);
    }
    if (!Ok(ComputeInitialColorCorrelationMap(Image(), &cfl)))
      std::abort();
  }
  ConstImage3FView Image() const {
    return {{{{image[0].data(), pixels, stride},
              {image[1].data(), pixels, stride},
              {image[2].data(), pixels, stride}}}};
  }
  ConstPlaneF32View Quant() const {
    return {quant.data(), blocks, blocks.width + 2};
  }
  ConstPlaneF32View Mask() const { return {mask.data(), pixels, stride}; }
  ResidentAcStrategySearchInputs Resident() const {
    const ConstDevicePlaneView plane{&device, 0, DeviceElementType::kF32,
                                     pixels, stride};
    return {.opsin = {{plane, plane, plane}},
            .quant_field = {&device, 0, DeviceElementType::kF32, blocks,
                            blocks.width + 2},
            .pixel_mask = plane};
  }
  Status Run(size_t mode, Backend &backend, AcStrategyGrid *out,
             PreparedAcStrategySearch *prepared = nullptr,
             AcStrategyGpuSearchStats *stats = nullptr) const {
    constexpr AcStrategySearchOptions options{.butteraugli_target = 1.2f};
    if (mode == 0)
      return FindAcStrategyGrid(Image(), Quant(), Mask(), cfl, options, out);
    if (mode == 3)
      return FindAcStrategyGridGpu(backend, Image(), Quant(), Mask(), cfl,
                                   options, out, stats);
    if (mode >= 4)
      return FindAcStrategyGridGpuResident(backend, {}, Quant(), {}, cfl,
                                           Resident(), options, out, stats,
                                           prepared);
    ac_strategy_internal::CandidateCostTableView table{.block_extent = blocks};
    for (size_t i = 0; i < costs.size(); ++i)
      table.strategy_costs[i] = costs[i];
    if (mode == 1)
      return ac_strategy_internal::FindAcStrategyGridFromCandidateCosts(
          Image(), Quant(), Mask(), cfl, options, table, out);
    return ac_strategy_internal::FindAcStrategyGridFromResidentCandidateCosts(
        pixels, Quant(), {}, cfl, options, table, out);
  }
};
std::vector<uint8_t> Cells(const AcStrategyGrid &grid) {
  std::vector<uint8_t> cells;
  for (size_t y = 0; y < grid.extent().height; ++y)
    for (size_t x = 0; x < grid.extent().width; ++x) {
      AcStrategyCell cell;
      if (!Ok(grid.Get(x, y, &cell)))
        std::abort();
      cells.push_back((uint8_t(cell.strategy) << 1) | cell.is_anchor);
    }
  return cells;
}
HostStorageBound Bound(Extent2D pixels, size_t mode, bool reuse = false) {
  SearchStoragePlan cpu;
  HostStoragePlan gpu;
  if (mode < 3) {
    if (!Ok(ComputeSearchStoragePlan(pixels, &cpu)))
      std::abort();
    return cpu.working;
  }
  if (!Ok(ComputeHostStoragePlan(pixels, mode >= 4, reuse, &gpu)))
    std::abort();
  return gpu.working;
}

bool PlanCases() {
  static_assert(sizeof(float) == 4 && sizeof(AcStrategyCandidate) == 24);
  for (size_t by = 1; by <= 64; ++by)
    for (size_t bx = 1; bx <= 64; ++bx) {
      const Extent2D e{bx * 8, by * 8};
      ac_strategy_search_internal::StoragePlan device;
      if (!Ok(ac_strategy_search_internal::ComputeStoragePlan(e, true,
                                                              &device)))
        return false;
      size_t candidates = 0;
      for (const auto &stage : device.stages)
        candidates += stage.candidate_count;
      for (bool resident : {false, true})
        for (bool reuse : {false, true}) {
          HostStoragePlan plan;
          if (!Ok(ComputeHostStoragePlan(e, resident, reuse, &plan)))
            return false;
          const size_t b = bx * by;
          const size_t retained =
              (reuse ? 32 : 28) * candidates + 62976 + 28 * b;
          const size_t peak =
              (reuse ? 60 : 28) * candidates + 62976 + (reuse ? 56 : 28) * b;
          const size_t staging = resident ? 0 : 1024 * b;
          if (!Check(plan.prepared == HostStorageBound{retained, peak} &&
                         plan.staging == HostStorageBound{staging, staging} &&
                         plan.merge.output == HostStorageBound{b, b} &&
                         plan.merge.working == HostStorageBound{2 * b, 2 * b} &&
                         plan.working ==
                             HostStorageBound{retained + staging + 2 * b,
                                              peak + staging + 2 * b},
                     "AC host plan disagrees with independent byte formula"))
            return false;
        }
    }
  HostStoragePlan plan;
  plan.prepared = {17, 23};
  const auto old = plan;
  for (Extent2D e :
       {Extent2D{}, {7, 8}, {8, 9}, {65536, 65536}, {size_t{1} << 32, 8}})
    if (!Check(!ComputeHostStoragePlan(e, true, true, &plan).ok() &&
                   plan == old,
               "Invalid host plan changed output"))
      return false;
  SearchStoragePlan cpu{{11, 11}, {22, 22}};
  const auto old_cpu = cpu;
  for (Extent2D e : {Extent2D{},
                     {7, 8},
                     {8, 9},
                     {std::numeric_limits<size_t>::max() - 7,
                      std::numeric_limits<size_t>::max() - 7}})
    if (!Check(!ComputeSearchStoragePlan(e, &cpu).ok() && cpu == old_cpu,
               "Invalid CPU plan changed output"))
      return false;
  if (!Check(!ComputeSearchStoragePlan({8, 8}, nullptr).ok() &&
                 !ComputeHostStoragePlan({8, 8}, true, false, nullptr).ok(),
             "Null AC host plan accepted"))
    return false;
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(1, &job)))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kAcSearch});
    ArmNextManagedHostAllocationFailureForTest();
    const bool ok =
        ComputeHostStoragePlan({536870904, 8}, true, true, &plan).ok();
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(ok && pending && budget.snapshot().peak_backing_bytes == 0,
               "Large AC host plan allocated backing"))
      return false;
  }
  job.Reset();
  return Empty(budget);
}

bool FreshCases() {
  Backend backend;
  size_t runs = 0;
  for (Extent2D e : {Extent2D{8, 8},
                     {8, 72},
                     {72, 8},
                     {16, 16},
                     {24, 24},
                     {32, 32},
                     {56, 64},
                     {64, 64},
                     {72, 80},
                     {128, 96},
                     {136, 144},
                     {256, 264},
                     {512, 256}})
    for (size_t variant : {size_t{0}, size_t{1}, size_t{5}}) {
      backend.variant = variant;
      Fixture fixture(backend, e, variant);
      AcStrategyGrid reference, direct;
      if (!Ok(fixture.Run(1, backend, &reference)) ||
          !Ok(fixture.Run(0, backend, &direct)))
        return false;
      for (size_t mode = 0; mode < 6; ++mode) {
        const auto bound = Bound(e, mode);
        ResourceBudget budget(bound.peak_bytes);
        ResourceReservation job;
        AcStrategyGrid result;
        // Caller-owned old output is not part of this job's accounting.
        if (!Ok(AcStrategyGrid::Create({1, 1}, &result)) ||
            !Ok(result.Set(0, 0, AcStrategyType::kDct8)) ||
            !Ok(budget.TryReserve(bound.peak_bytes, &job)))
          return false;
        {
          ResourceContextScope scope({&job, ResourceClass::kAcSearch});
          PreparedAcStrategySearch prepared;
          if (!Ok(fixture.Run(mode, backend, &result,
                              mode == 5 ? &prepared : nullptr)) ||
              !Check(Cells(result) == Cells(mode == 0 ? direct : reference),
                     "Reserved AC search changed placement"))
            return false;
        }
        const size_t b = e.width / 8 * (e.height / 8);
        if (!Check(budget.snapshot().peak_backing_bytes == bound.peak_bytes &&
                       budget.snapshot().total.live_capacity_bytes == b,
                   "Fresh AC bound differs from actual backing peak/output"))
          return false;
        job.Reset();
        if (!Check(budget.snapshot().total.live_capacity_bytes == b,
                   "Closing AC producer uncharged retained output"))
          return false;
        result = {};
        if (!Empty(budget))
          return false;
        ++runs;
      }
    }
  std::cout << "Fresh AC runtime/reservation cases: " << runs << '\n';
  return true;
}

bool ReuseCases() {
  Backend first, second;
  const Extent2D maximum{136, 144};
  const auto bound = Bound(maximum, 5, true);
  const size_t max_blocks = maximum.width / 8 * (maximum.height / 8);
  // Include the old, tracked output until the next grid publishes atomically.
  ResourceBudget budget(bound.peak_bytes + max_blocks);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(bound.peak_bytes + max_blocks, &job)))
    return false;
  AcStrategyGrid result;
  auto prepared = std::make_unique<PreparedAcStrategySearch>();
  size_t iteration = 0;
  for (Extent2D e : {Extent2D{64, 64},
                     {72, 72},
                     {8, 8},
                     {128, 136},
                     {136, 144},
                     {136, 144},
                     {8, 144},
                     {136, 8},
                     {8, 8},
                     {136, 144},
                     {72, 72},
                     {136, 144}}) {
    Backend &backend = iteration < 9 ? first : second;
    backend.variant = ++iteration;
    Fixture fixture(backend, e, backend.variant);
    AcStrategyGrid reference;
    if (!Ok(fixture.Run(1, backend, &reference)))
      return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kAcSearch});
      if (!Ok(fixture.Run(5, backend, &result, prepared.get())) ||
          !Check(Cells(result) == Cells(reference), "AC reuse changed output"))
        return false;
    }
    if (!Check(budget.snapshot().peak_backing_bytes <=
                   bound.peak_bytes + max_blocks,
               "Prepared AC growth/replacement exceeded its bound"))
      return false;
  }
  const size_t retained = budget.snapshot().total.live_capacity_bytes;
  job.Reset();
  if (!Check(budget.snapshot().total.live_capacity_bytes == retained &&
                 retained > max_blocks,
             "Closed AC producer lost prepared/output charges"))
    return false;
  prepared.reset();
  if (!Check(budget.snapshot().total.live_capacity_bytes == max_blocks,
             "Destroying prepared search affected output ownership"))
    return false;
  result = {};
  return Empty(budget);
}

bool FailureCases() {
  Backend backend;
  Fixture fixture(backend, {64, 64}, 0);
  size_t positions = 0;
  for (size_t mode = 0; mode < 6; ++mode) {
    const auto bound = Bound(fixture.pixels, mode, true);
    bool completed = false;
    for (size_t fail_after = 0; fail_after < 80; ++fail_after) {
      ResourceBudget budget(bound.peak_bytes);
      ResourceReservation job;
      AcStrategyGrid output;
      AcStrategyGpuSearchStats stats;
      stats.total_candidate_count = 777;
      if (!Ok(AcStrategyGrid::Create({1, 1}, &output)) ||
          !Ok(output.Set(0, 0, AcStrategyType::kDct8)) ||
          !Ok(budget.TryReserve(bound.peak_bytes, &job)))
        return false;
      const auto before = Cells(output);
      {
        ResourceContextScope scope({&job, ResourceClass::kAcSearch});
        PreparedAcStrategySearch prepared;
        ArmManagedHostAllocationFailureAfterForTest(fail_after);
        const auto status = fixture.Run(
            mode, backend, &output, mode == 5 ? &prepared : nullptr, &stats);
        const bool pending = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (status.ok()) {
          if (!Check(pending, "Successful AC call consumed failure hook"))
            return false;
          completed = true;
        } else {
          if (!Check(status.code() == StatusCode::kOutOfMemory &&
                         !status.resource_plan_exceeded() && !pending &&
                         Cells(output) == before &&
                         stats.total_candidate_count == 777,
                     "Physical AC failure changed outputs or error type"))
            return false;
          ++positions;
          // Prepared state may retain partial backing on failure; recovery on
          // that SAME state/reservation must work and drain at destruction.
          if (!Ok(fixture.Run(mode, backend, &output,
                              mode == 5 ? &prepared : nullptr)))
            return false;
        }
      }
      output = {};
      job.Reset();
      if (!Empty(budget))
        return false;
      if (completed)
        break;
    }
    if (!Check(completed, "AC physical failure sweep did not reach success"))
      return false;
    ResourceBudget budget(1);
    ResourceReservation job;
    AcStrategyGrid output;
    if (!Ok(budget.TryReserve(1, &job)) || !Ok(job.ReduceCapacity(0)))
      return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kAcSearch});
      ArmNextManagedHostAllocationFailureForTest();
      const auto status = fixture.Run(mode, backend, &output);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (!Check(status.resource_plan_exceeded() && pending &&
                     !output.valid() &&
                     budget.snapshot().peak_backing_bytes == 0,
                 "AC underplan escaped its domain or consumed physical hook"))
        return false;
    }
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  // Failure after all preparation, retaining reusable state but no new output.
  const auto bound = Bound(fixture.pixels, 5, true);
  ResourceBudget budget(bound.peak_bytes);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(bound.peak_bytes, &job)))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kAcSearch});
    PreparedAcStrategySearch prepared;
    AcStrategyGrid output;
    backend.fail_submission = true;
    const auto status = fixture.Run(5, backend, &output, &prepared);
    backend.fail_submission = false;
    if (!Check(!status.ok() && !output.valid(),
               "Failed AC submission published output") ||
        !Ok(fixture.Run(5, backend, &output, &prepared)))
      return false;
  }
  job.Reset();
  if (!Empty(budget))
    return false;
  std::cout << "Physical AC allocation failures/recovery: " << positions
            << '\n';
  return true;
}

bool LateFailures() {
  Backend backend;
  Fixture small(backend, {64, 64}, 0), large(backend, {72, 72}, 0);
  AcStrategyGrid reference;
  if (!Ok(large.Run(1, backend, &reference)))
    return false;
  for (size_t mode = 0; mode < 6; ++mode) {
    // The fresh peak is reached at export, after every candidate/merge has
    // finished. One missing byte must fail atomically there, not fall back.
    const auto bound = Bound(large.pixels, mode);
    ResourceBudget budget(bound.peak_bytes - 1);
    ResourceReservation job;
    AcStrategyGrid output;
    AcStrategyGpuSearchStats stats;
    stats.total_candidate_count = 777;
    if (!Ok(small.Run(1, backend, &output)) ||
        !Ok(budget.TryReserve(bound.peak_bytes - 1, &job)))
      return false;
    const auto before = Cells(output);
    {
      ResourceContextScope scope({&job, ResourceClass::kAcSearch});
      PreparedAcStrategySearch prepared;
      const auto status = large.Run(mode, backend, &output,
                                    mode == 5 ? &prepared : nullptr, &stats);
      if (!Check(status.resource_plan_exceeded() && Cells(output) == before &&
                     stats.total_candidate_count == 777 &&
                     budget.snapshot().peak_backing_bytes > 0,
                 "Late AC underplan changed outputs or lost typed status"))
        return false;
    }
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  // Growth on an already successful prepared owner has different allocation
  // positions than fresh preparation. Retain the old output and recover from
  // each physical failure on the same partially updated prepared state.
  const auto bound = Bound(large.pixels, 5, true);
  const size_t envelope = bound.peak_bytes + 81; // Old output, at most 9x9.
  size_t failures = 0;
  bool completed = false;
  for (size_t fail_after = 0; fail_after < 80; ++fail_after) {
    ResourceBudget budget(envelope);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(envelope, &job)))
      return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kAcSearch});
      PreparedAcStrategySearch prepared;
      AcStrategyGrid output;
      if (!Ok(small.Run(5, backend, &output, &prepared)))
        return false;
      const auto before = Cells(output);
      ArmManagedHostAllocationFailureAfterForTest(fail_after);
      const auto status = large.Run(5, backend, &output, &prepared);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (status.ok()) {
        if (!Check(pending && Cells(output) == Cells(reference),
                   "Successful AC growth changed placement or consumed hook"))
          return false;
        completed = true;
      } else {
        if (!Check(status.code() == StatusCode::kOutOfMemory &&
                       !status.resource_plan_exceeded() && !pending &&
                       Cells(output) == before,
                   "Failed AC growth changed old output or error type") ||
            !Ok(large.Run(5, backend, &output, &prepared)) ||
            !Check(Cells(output) == Cells(reference),
                   "AC growth failure corrupted recovery placement"))
          return false;
        ++failures;
      }
    }
    job.Reset();
    if (!Empty(budget))
      return false;
    if (completed)
      break;
  }
  std::cout << "Prepared AC growth failures/recovery: " << failures
            << "; late underplans: 6\n";
  return Check(completed, "AC growth failure sweep did not reach success");
}
} // namespace

int main() {
  if (!PlanCases() || !FreshCases() || !ReuseCases() || !FailureCases() ||
      !LateFailures() ||
      !Check(DefaultResourceBudget().snapshot().peak_backing_bytes == 0,
             "Host AC test escaped to default domain"))
    return EXIT_FAILURE;
  std::cout << "AC host formula cases: 16384; reuse transitions: 12\n";
  return EXIT_SUCCESS;
}
