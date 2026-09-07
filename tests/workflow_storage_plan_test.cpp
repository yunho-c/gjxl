// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codestream/batch_workflow.h"
#include "codestream/compatibility_workflow_storage_plan.h"
#include "codestream/encoding_result_internal.h"
#include "codestream/resident_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "codestream/workflow_storage_plan.h"
#include "core/image_buffer.h"
#include "gjxl/gjxl.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
using enum VectorCapacityPolicy;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Ok(const Status &s) {
  if (!s.ok())
    std::cerr << s.message() << '\n';
  return s.ok();
}
bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                   s.total.pending_count == 0 && s.open_reservations == 0 &&
                   s.waiting_requests == 0,
               "Workflow plan leaked managed storage");
}
struct CBuffer {
  GJXLBuffer value{};
  CBuffer() = default;
  CBuffer(const CBuffer &) = delete;
  CBuffer &operator=(const CBuffer &) = delete;
  ~CBuffer() { gjxl_buffer_free(&value); }
};
WorkflowStorageOptions Options(size_t mode) {
  WorkflowStorageOptions o;
  o.encoding.effort = 1;
  o.encoding.cpu_thread_count = 1;
  o.encoding.backend = mode == 0 ? VarDctBackendPreference::kCpu
                                 : VarDctBackendPreference::kMetal;
  o.route =
      mode == 0 ? WorkflowStorageRoute::kCpu : WorkflowStorageRoute::kMetal;
  if (mode == 2)
    o.encoding.metal_aq_mode = GpuAdaptiveQuantizationMode::kExactCoefficients;
  if (mode == 3)
    o.encoding.metal_aq_mode = GpuAdaptiveQuantizationMode::kMaximumThroughput;
  if (mode == 4) {
    o.encoding.rate_control_mode = VarDctRateControlMode::kMaximumError;
    o.encoding.maximum_error = {0.05f, 0.05f, 0.05f};
  }
  if (mode == 5) {
    o.encoding.backend = VarDctBackendPreference::kAutomatic;
    o.encoding.metal_aq_mode = GpuAdaptiveQuantizationMode::kExactCoefficients;
    o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
    o.encoding.target_bytes = 1;
    o.encoding.target_size_maximum_attempts = 64;
    o.route = WorkflowStorageRoute::kAutomaticExactSearch;
  }
  return o;
}

bool CheckComposition() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(1, &job)) || !Ok(job.ReduceCapacity(0)))
    return false;
  ResourceContextScope scope({&job, ResourceClass::kPreparation});
  size_t count = 0;
  for (Extent2D source :
       {Extent2D{1, 1}, {17, 9}, {128, 96}, {257, 257}, {3839, 2159}})
    for (size_t mode = 0; mode < 6; ++mode)
      for (int effort : {1, 7, 10})
        for (size_t diagnostic = 0; diagnostic < 3; ++diagnostic) {
          auto o = Options(mode);
          o.encoding.effort = effort;
          o.collect_timing = diagnostic != 0;
          o.collect_profile = diagnostic == 2;
          WorkflowStoragePlan p;
          ArmNextManagedHostAllocationFailureForTest();
          const auto status = ComputeWorkflowStoragePlan(source, o, &p);
          const bool pending = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          if (!Ok(status) ||
              !Check(pending, "Unified planning allocated backing"))
            return false;
          HostStorageBound working, output;
          std::array<size_t, 4> pools{};
          const CpuWorkflowStorageOptions plain{o.encoding, o.collect_timing,
                                                o.collect_profile};
          if (mode == 0) {
            CpuWorkflowStoragePlan base;
            if (!Ok(ComputeCpuWorkflowStoragePlan(source, plain, &base)))
              return false;
            working = base.working;
            output = base.output;
          } else if (mode == 5) {
            AutomaticExactSearchStoragePlan base;
            if (!Ok(ComputeAutomaticExactSearchStoragePlan(source, plain,
                                                           &base)))
              return false;
            working = base.working;
            output = base.output;
            pools = base.metal.idle_pool_capacity;
          } else if (mode == 1) {
            ResidentWorkflowStoragePlan base;
            if (!Ok(ComputeResidentWorkflowStoragePlan(
                    source,
                    {o.encoding, o.collect_timing, o.collect_profile, false},
                    &base)))
              return false;
            working = base.working;
            output = base.output;
            pools = base.idle_pool_capacity;
          } else {
            MetalCompatibilityWorkflowStoragePlan base;
            if (!Ok(ComputeMetalCompatibilityWorkflowStoragePlan(source, plain,
                                                                 &base)))
              return false;
            working = base.working;
            output = base.output;
            pools = base.idle_pool_capacity;
          }
          if (!Check(p.backend_working == working && p.working == working &&
                         p.output == output && p.idle_pool_capacity == pools &&
                         p.input.peak_bytes == 0 &&
                         p.publication.peak_bytes == 0,
                     "Unified selector changed its underlying policy recipe"))
            return false;
          if (diagnostic == 0) {
            o.adapter = WorkflowStorageAdapter::kPackedSrgbC;
            WorkflowStoragePlan c;
            if (!Ok(ComputeWorkflowStoragePlan(source, o, &c)) ||
                !Check(c.backend_working == p.backend_working &&
                           c.input.peak_bytes == source.width * source.height *
                                                     3 * sizeof(float) &&
                           c.publication.peak_bytes ==
                               p.maximum_codestream_bytes &&
                           c.output == c.publication &&
                           c.working.peak_bytes == p.working.peak_bytes +
                                                       c.input.peak_bytes +
                                                       c.publication.peak_bytes,
                       "C conversion or output-copy overlap missing from plan"))
              return false;
          }
          ++count;
        }
  WorkflowStoragePlan sentinel;
  sentinel.working = {19, 23};
  for (size_t invalid = 0; invalid < 8; ++invalid) {
    auto o = Options(0);
    Extent2D source{17, 9};
    switch (invalid) {
    case 0:
      o.route = static_cast<WorkflowStorageRoute>(99);
      break;
    case 1:
      o.adapter = static_cast<WorkflowStorageAdapter>(99);
      break;
    case 2:
      o.adapter = WorkflowStorageAdapter::kPackedSrgbC;
      o.collect_timing = true;
      break;
    case 3:
      o.collect_gpu_profile = true;
      break;
    case 4:
      o = Options(1);
      o.encoding.backend = VarDctBackendPreference::kCpu;
      break;
    case 5:
      o = Options(5);
      o.route = WorkflowStorageRoute::kMetal;
      break;
    case 6:
      source = {std::numeric_limits<size_t>::max(), 9};
      break;
    case 7:
      source = {};
      break;
    }
    auto p = sentinel;
    if (!Check(!ComputeWorkflowStoragePlan(source, o, &p).ok() && p == sentinel,
               "Invalid unified plan changed output"))
      return false;
  }
  if (!Check(!ComputeWorkflowStoragePlan({17, 9}, Options(0), nullptr).ok(),
             "Null plan output was accepted"))
    return false;
  std::cout << "Unified recipe cases: " << count << '\n';
  return true;
}

bool CheckBatchBounds() {
  std::array<WorkflowStoragePlan, 6> plans;
  BatchWorkflowStorageAccumulator batch;
  HostStorageBound retained, metadata;
  std::array<size_t, 4> pools{};
  size_t largest = 0;
  for (size_t i = 0; i < plans.size(); ++i) {
    auto o = Options(i);
    o.collect_timing = true;
    if (!Ok(ComputeWorkflowStoragePlan({128 + i * 8, 96}, o, &plans[i])) ||
        !Ok(batch.AddRequest(&plans[i])))
      return false;
    const auto &p = plans[i];
    if (!retained.Add({p.output.retained_bytes, p.output.retained_bytes}))
      return false;
    largest = std::max(largest, p.working.peak_bytes);
    for (size_t j = 0; j < 4; ++j)
      pools[j] = std::max(pools[j], p.idle_pool_capacity[j]);
  }
  if (!Ok(batch.AddRequest(nullptr)))
    return false;
  constexpr size_t requests = 7;
  if (!metadata.AddVector<VarDctBatchEncodingResult>(requests, kFreshExact) ||
      !metadata.AddVector<OwnedEncodingResult>(requests, kFreshExact) ||
      !metadata.AddVector<std::array<ResourceAllocation, 3>>(requests,
                                                             kFreshExact))
    return false;
  const size_t base = retained.peak_bytes + metadata.peak_bytes;
  size_t idle = 0;
  for (auto n : pools)
    idle += n;
  for (size_t workers = 1; workers <= 8; ++workers) {
    BatchWorkflowStoragePlan unlimited;
    if (!Ok(batch.Finish(workers, 0, &unlimited)) ||
        !Check(unlimited.request_count == requests &&
                   unlimited.encodable_count == 6 &&
                   unlimited.in_flight == std::min(workers, size_t{6}) &&
                   unlimited.result_metadata == metadata &&
                   unlimited.retained_results == retained &&
                   unlimited.idle_pools.peak_bytes == idle &&
                   !unlimited.trim_after_each_image &&
                   unlimited.working.peak_bytes ==
                       base + idle + unlimited.in_flight * largest,
               "Unlimited batch lost work, per-pool idle maxima or retained "
               "results"))
      return false;
    for (size_t slots = 1; slots <= 6; ++slots) {
      BatchWorkflowStoragePlan p;
      const size_t limit = base + idle + slots * largest;
      if (!Ok(batch.Finish(workers, limit, &p)) ||
          !Check(p.in_flight == std::min(slots, workers) &&
                     !p.trim_after_each_image && p.working.peak_bytes <= limit,
                 "Limited batch chose an invalid concurrency"))
        return false;
    }
    auto p = unlimited;
    if (!Check(!batch.Finish(workers, base + largest - 1, &p).ok() &&
                   p == unlimited,
               "Impossible batch was admitted or changed output") ||
        !Ok(batch.Finish(workers, base + largest, &p)) ||
        !Check(p.in_flight == 1 && p.trim_after_each_image &&
                   p.working.peak_bytes == base + largest &&
                   p.idle_pools.peak_bytes == 0,
               "Tight batch did not require trim before reusing its sole work "
               "slot"))
      return false;
  }
  BatchWorkflowStoragePlan before, after;
  if (!Ok(batch.Finish(4, 0, &before)))
    return false;
  auto invalid = plans[0];
  invalid.collect_timing = false;
  if (!Check(!batch.AddRequest(&invalid).ok(), "Untimed batch plan accepted") ||
      !Ok(batch.Finish(4, 0, &after)) ||
      !Check(after == before, "Rejected request mutated preflight"))
    return false;
  BatchWorkflowStorageAccumulator empty, rejected;
  if (!Ok(empty.Finish(4, 0, &after)) ||
      !Check(after.working.peak_bytes == 0 && after.in_flight == 0,
             "Empty batch reserved work") ||
      !Ok(rejected.AddRequest(nullptr)) || !Ok(rejected.Finish(4, 0, &after)) ||
      !Check(after.in_flight == 0 && after.retained_results.peak_bytes == 0 &&
                 after.working == after.result_metadata,
             "All-invalid batch reserved encoding work"))
    return false;
  return Check(!batch.Finish(0, 0, &after).ok() &&
                   !batch.Finish(1, 0, nullptr).ok(),
               "Invalid batch output or concurrency was accepted");
}

bool CheckCAdapter() {
  size_t count = 0;
  for (auto backend : {GJXL_BACKEND_CPU, GJXL_BACKEND_METAL})
    for (Extent2D source : {Extent2D{17, 9}, {128, 96}})
      for (size_t channels : {size_t{3}, size_t{4}}) {
        const size_t stride = source.width * channels + 7;
        std::vector<uint8_t> packed(stride * source.height, 255);
        for (size_t y = 0; y < source.height; ++y)
          for (size_t x = 0; x < source.width; ++x)
            for (size_t c = 0; c < 3; ++c)
              packed[y * stride + x * channels + c] =
                  (x * 3 + y * 5 + c * 19) % 256;
        GJXLContextOptions co{};
        GJXLEncoderOptions eo{};
        if (!Check(gjxl_context_options_init(&co, sizeof(co)) == GJXL_OK &&
                       gjxl_encoder_options_init(&eo, sizeof(eo)) == GJXL_OK,
                   "C options failed"))
          return false;
        co.backend = backend;
        co.num_cpu_threads = 1;
        eo.effort = 1;
        eo.distance = 1.0f;
        GJXLContext *raw = nullptr;
        if (!Check(gjxl_context_create(&co, &raw) == GJXL_OK,
                   "C context failed"))
          return false;
        std::unique_ptr<GJXLContext, decltype(&gjxl_context_destroy)> context(
            raw, gjxl_context_destroy);
        const GJXLImageView image{sizeof(GJXLImageView),
                                  static_cast<uint32_t>(source.width),
                                  static_cast<uint32_t>(source.height),
                                  channels == 3 ? GJXL_PIXEL_FORMAT_RGB8_SRGB
                                                : GJXL_PIXEL_FORMAT_RGBA8_SRGB,
                                  packed.data(),
                                  packed.size(),
                                  stride};
        auto o = Options(backend == GJXL_BACKEND_CPU ? 0 : 1);
        o.adapter = WorkflowStorageAdapter::kPackedSrgbC;
        WorkflowStoragePlan p;
        if (!Ok(ComputeWorkflowStoragePlan(source, o, &p)) ||
            !Ok(TrimVarDctPreparationCache()))
          return false;
        std::vector<uint8_t> oracle;
        // First run uses an ample independent envelope; next two use only the
        // precomputed bound, cold then warm, while previous C output stays
        // alive.
        CBuffer previous;
        ResourceBudget budget(p.working.peak_bytes * 3);
        for (size_t repeat = 0; repeat < 3; ++repeat) {
          ResourceReservation job;
          if (!Ok(budget.TryReserve(
                  p.working.peak_bytes * (repeat == 0 ? 2 : 1), &job)))
            return false;
          CBuffer owned;
          auto &bytes = owned.value;
          {
            ResourceContextScope scope({&job, ResourceClass::kInput});
            if (!Check(gjxl_encode(context.get(), &image, &eo, &bytes) ==
                           GJXL_OK,
                       gjxl_get_last_error()))
              return false;
          }
          if (repeat == 0)
            oracle.assign(bytes.data, bytes.data + bytes.size);
          else if (!Check(std::equal(bytes.data, bytes.data + bytes.size,
                                     oracle.begin(), oracle.end()),
                          "Bounded C adapter bytes changed"))
            return false;
          if (repeat == 0 && !Ok(TrimVarDctPreparationCache()))
            return false;
          job.Reset();
          if (!Check(budget.snapshot().total.live_capacity_bytes == 0,
                     "C output remained charged after publication") ||
              !Empty(DefaultResourceBudget()))
            return false;
          std::swap(previous.value, bytes);
        }
        if (!Ok(TrimVarDctPreparationCache()) || !Empty(budget))
          return false;
        for (size_t failure = 0; failure < 3; ++failure) {
          ResourceReservation job;
          if (!Ok(budget.TryReserve(failure == 0 ? 1 : p.working.peak_bytes,
                                    &job)))
            return false;
          CBuffer bytes;
          {
            ResourceContextScope scope({&job, ResourceClass::kInput});
            if (failure != 0)
              ArmManagedHostClassAllocationFailureAfterForTest(
                  failure == 1 ? ResourceClass::kInput
                               : ResourceClass::kRetainedResult,
                  0);
            const auto status =
                gjxl_encode(context.get(), &image, &eo, &bytes.value);
            const bool injected =
                failure == 0 || !ManagedHostAllocationFailurePendingForTest();
            DisarmManagedHostAllocationFailureForTest();
            if (!Check(status == (failure == 0 ? GJXL_ERROR_RESOURCE_PLAN_EXCEEDED
                                               : GJXL_ERROR_OUT_OF_MEMORY) &&
                           injected && bytes.value.data == nullptr && bytes.value.size == 0,
                       "C input/publication failure was not atomic"))
              return false;
            if (failure != 0 &&
                !Check(gjxl_encode(context.get(), &image, &eo, &bytes.value) ==
                           GJXL_OK,
                       "C adapter failed to recover in the same reservation"))
              return false;
          }
          if (!Ok(TrimVarDctPreparationCache()))
            return false;
          job.Reset();
          if (!Empty(budget))
            return false;
        }
        ++count;
      }
  std::cout << "Complete C adapter cases (three calls each): " << count << '\n';
  return true;
}

bool CheckRealBatch() {
  std::array<Image3FBuffer, 4> images;
  std::array<VarDctBatchEncodingRequest, 5> requests;
  BatchWorkflowStorageAccumulator accumulator;
  for (size_t i = 0; i < images.size(); ++i) {
    images[i] = Image3FBuffer({17 + i * 16, 9 + i * 8});
    for (size_t c = 0; c < 3; ++c)
      for (size_t j = 0; j < images[i].plane(c).size(); ++j)
        images[i].plane(c)[j] =
            0.05f + 0.8f * ((j * (c + 3) + i) % 127) / 127.0f;
    auto o = Options(i);
    o.collect_timing = true;
    requests[i] = {images[i].const_view(), o.encoding};
    WorkflowStoragePlan p;
    if (!Ok(ComputeWorkflowStoragePlan(images[i].extent(), o, &p)) ||
        !Ok(accumulator.AddRequest(&p)))
      return false;
  }
  requests.back().options = Options(0).encoding;
  if (!Ok(accumulator.AddRequest(nullptr)))
    return false;
  std::vector<VarDctBatchEncodingResult> oracle, previous;
  for (size_t workers : {size_t{1}, size_t{2}, size_t{4}}) {
    BatchWorkflowStoragePlan p;
    if (!Ok(accumulator.Finish(workers, 0, &p)) ||
        !Ok(TrimVarDctPreparationCache()))
      return false;
    ResourceBudget budget(p.working.peak_bytes * 2);
    std::unique_ptr<VarDctBatchEncoder> driver;
    if (!Ok(VarDctBatchEncoder::Create(p.in_flight, &driver)))
      return false;
    for (size_t repeat = 0; repeat < 3; ++repeat) {
      ResourceReservation job;
      if (!Ok(budget.TryReserve(p.working.peak_bytes, &job)))
        return false;
      std::vector<VarDctBatchEncodingResult> results;
      {
        ResourceContextScope scope({&job, ResourceClass::kRetainedResult});
        if (!Ok(driver->Encode(requests, &results)))
          return false;
      }
      if (!Check(results.size() == requests.size() &&
                     results.back().status.code() ==
                         StatusCode::kInvalidArgument,
                 "Batch validation/result order changed"))
        return false;
      for (size_t i = 0; i < images.size(); ++i) {
        if (!Ok(results[i].status) ||
            !Check(!results[i].codestream.empty(), "Batch output missing"))
          return false;
        if (!oracle.empty() &&
            !Check(results[i].codestream == oracle[i].codestream &&
                       results[i].summary == oracle[i].summary,
                   "Batch plan or concurrency changed decisions"))
          return false;
      }
      if (oracle.empty())
        oracle = results;
      previous = std::move(results);
      job.Reset();
      if (!Check(budget.snapshot().total.live_capacity_bytes == 0 &&
                     budget.snapshot().peak_backing_bytes <=
                         p.working.peak_bytes,
                 "Batch publication retained a live charge or exceeded planned "
                 "backing"))
        return false;
    }
    if (!Ok(TrimVarDctPreparationCache()) || !Empty(budget))
      return false;
    ResourceReservation tiny;
    if (!Ok(budget.TryReserve(1, &tiny)))
      return false;
    std::vector<VarDctBatchEncodingResult> sentinel(1);
    sentinel[0].codestream = {19, 23};
    {
      ResourceContextScope scope({&tiny, ResourceClass::kRetainedResult});
      const auto status = driver->Encode(requests, &sentinel);
      if (!Check(status.resource_plan_exceeded() && sentinel.size() == 1 &&
                     sentinel[0].codestream == std::vector<uint8_t>({19, 23}),
                 "Batch metadata underplan did not preserve caller output"))
        return false;
    }
    tiny.Reset();
    if (!Empty(budget))
      return false;
  }
  std::cout
      << "Complete heterogeneous batch cases: 3 worker limits, 3 calls each\n";
  return Empty(DefaultResourceBudget());
}
} // namespace

int main() {
  return CheckComposition() && CheckBatchBounds() && CheckCAdapter() &&
                 CheckRealBatch()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
