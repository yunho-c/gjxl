// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include "codestream/compatibility_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "codestream/workflow_lifetime_test.h"
#include "core/image_buffer.h"
#include "gpu/metal/metal_backend.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;

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
               "Compatibility plan leaked backing");
}
CpuWorkflowStorageOptions Options(size_t mode) {
  CpuWorkflowStorageOptions o;
  o.encoding.backend = VarDctBackendPreference::kMetal;
  o.encoding.metal_aq_mode =
      mode < 2    ? GpuAdaptiveQuantizationMode::kExactCoefficients
      : mode == 2 ? GpuAdaptiveQuantizationMode::kFullyResident
      : mode == 3 ? GpuAdaptiveQuantizationMode::kThroughput
                  : GpuAdaptiveQuantizationMode::kMaximumThroughput;
  if (mode >= 1 && mode <= 3) {
    o.encoding.rate_control_mode = VarDctRateControlMode::kMaximumError;
    o.encoding.maximum_error = {0.05f, 0.05f, 0.05f};
  }
  o.encoding.cpu_thread_count = 1;
  return o;
}
bool CheckPlans() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(1, &job)) || !Ok(job.ReduceCapacity(0)))
    return false;
  ResourceContextScope scope({&job, ResourceClass::kPreparation});
  size_t cases = 0;
  for (Extent2D source : {Extent2D{1, 1},
                          {14, 15},
                          {15, 15},
                          {65, 63},
                          {257, 9},
                          {257, 257},
                          {3839, 2159}})
    for (int effort = 1; effort <= 10; ++effort)
      for (size_t mode = 0; mode < 5; ++mode)
        for (size_t flags = 0; flags < 8; ++flags) {
          auto o = Options(mode);
          o.encoding.effort = effort;
          o.encoding.cpu_thread_count = flags & 1 ? 0 : 4;
          if (flags & 2)
            o.encoding.compression_mode =
                VarDctCompressionMode::kMaximumCompression;
          o.collect_timing = true;
          o.collect_profile = bool(flags & 4);
          MetalCompatibilityWorkflowStoragePlan p;
          ArmNextManagedHostAllocationFailureForTest();
          const auto s =
              ComputeMetalCompatibilityWorkflowStoragePlan(source, o, &p);
          const bool pending = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          if (!Ok(s) ||
              !Check(pending && p.working.peak_bytes >= p.output.peak_bytes &&
                         (p.score_count == 0) == (mode == 4),
                     "Compatibility plan was not pure or consistent"))
            return false;
          ++cases;
        }
  for (size_t attempts = 1; attempts <= 64; ++attempts) {
    auto o = Options(0);
    o.encoding.backend = VarDctBackendPreference::kAutomatic;
    o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
    o.encoding.target_size_maximum_attempts = attempts;
    AutomaticExactSearchStoragePlan p;
    if (!Ok(ComputeAutomaticExactSearchStoragePlan({257, 257}, o, &p)) ||
        !Check(p.working.peak_bytes ==
                       p.cpu.working.peak_bytes + p.metal.working.peak_bytes &&
                   p.cpu.retained_best == p.metal.retained_best &&
                   (p.cpu.retained_best.peak_bytes == 0) == (attempts == 1),
               "Mixed plan lost retained state or multiplied best results"))
      return false;
  }
  MetalCompatibilityWorkflowStoragePlan sentinel;
  sentinel.working = {41, 42};
  for (size_t bad = 0; bad < 9; ++bad) {
    auto o = Options(0);
    Extent2D source{65, 63};
    switch (bad) {
    case 0:
      o.encoding.backend = VarDctBackendPreference::kAutomatic;
      break;
    case 1:
      o.encoding.effort = 0;
      break;
    case 2:
      o.encoding.cpu_thread_count = 257;
      break;
    case 3:
      o.encoding.metal_aq_mode = GpuAdaptiveQuantizationMode::kFullyResident;
      break;
    case 4:
      o = Options(4);
      o.encoding.rate_control_mode = VarDctRateControlMode::kMaximumError;
      break;
    case 5:
      o = Options(2);
      o.encoding.density_mode = VarDctDensityMode::kHighDensity;
      break;
    case 6:
      source = {};
      break;
    case 7:
      source = {std::numeric_limits<size_t>::max(), 8};
      break;
    case 8:
      o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
      o.encoding.target_size_maximum_attempts = 65;
      break;
    }
    auto p = sentinel;
    if (!Check(
            !ComputeMetalCompatibilityWorkflowStoragePlan(source, o, &p).ok() &&
                p == sentinel,
            "Invalid compatibility plan changed output"))
      return false;
  }
  AutomaticExactSearchStoragePlan mixed_sentinel;
  mixed_sentinel.working = {43, 44};
  auto p = mixed_sentinel;
  if (!Check(
          !ComputeAutomaticExactSearchStoragePlan({65, 63}, Options(0), &p)
                  .ok() &&
              p == mixed_sentinel &&
              !ComputeMetalCompatibilityWorkflowStoragePlan({8, 8}, Options(0),
                                                            nullptr)
                   .ok() &&
              !ComputeAutomaticExactSearchStoragePlan({8, 8}, {}, nullptr).ok(),
          "Mixed/null plan accepted invalid shape"))
    return false;
  job.Reset();
  std::cout << "Compatibility planning checks: " << cases
            << "; mixed limits: 64\n";
  return Empty(budget);
}
Image3FBuffer MakeImage(Extent2D extent) {
  Image3FBuffer image(extent);
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        image.plane(c)[y * extent.width + x] =
            0.03f + 0.8f * ((x * 11 + y * 7 + c * 23 + x * y) % 251) / 251.0f;
  return image;
}
struct Result {
  std::vector<uint8_t> bytes;
  VarDctEncodingSummary summary;
  VarDctEncodingTiming timing;
  VarDctEncodingProfile profile;
};
Status Encode(GpuBackend &gpu, ConstImage3FView image,
              const CpuWorkflowStorageOptions &o, Result *out) {
  if (o.collect_timing)
    return EncodeLinearRgbVarDctCodestreamProfiled(
        image, o.encoding, &out->bytes, &out->summary, &out->timing);
  if (o.collect_profile)
    return EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
        image, o.encoding, &gpu, IsAutomaticMetalBackendQualified(gpu),
        &out->bytes, &out->summary, &out->profile);
  return EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image, o.encoding, &gpu, IsAutomaticMetalBackendQualified(gpu),
      &out->bytes, &out->summary);
}
bool Trim(GpuBackend &gpu) {
  return Ok(gpu.TrimPreparationCache()) && Ok(TrimVarDctPreparationCache());
}
struct MixedTrace {
  ResourceBudget &budget;
  std::array<size_t, 64> aq_live{};
  size_t count = 0;
  static void Observe(bool, bool, void *context) noexcept {
    auto &trace = *static_cast<MixedTrace *>(context);
    if (trace.count < trace.aq_live.size())
      trace.aq_live[trace.count] =
          trace.budget.snapshot()
              .classes[static_cast<size_t>(ResourceClass::kAqScratch)]
              .live_capacity_bytes;
    ++trace.count;
  }
};
struct ScopedTrace {
  WorkflowLifetimeObserverForTest previous;
  explicit ScopedTrace(MixedTrace *trace)
      : previous(SetWorkflowLifetimeObserverForTest(
            trace ? WorkflowLifetimeObserverForTest{&MixedTrace::Observe, trace}
                  : WorkflowLifetimeObserverForTest{})) {}
  ~ScopedTrace() { (void)SetWorkflowLifetimeObserverForTest(previous); }
};
bool Run(GpuBackend &gpu, ConstImage3FView image,
         const CpuWorkflowStorageOptions &o, bool mixed = false) {
  HostStorageBound working;
  size_t scores = 0;
  if (mixed) {
    AutomaticExactSearchStoragePlan p;
    if (!Ok(ComputeAutomaticExactSearchStoragePlan(image.extent(), o, &p)))
      return false;
    working = p.working;
    scores = p.cpu.score_count;
  } else {
    MetalCompatibilityWorkflowStoragePlan p;
    if (!Ok(ComputeMetalCompatibilityWorkflowStoragePlan(image.extent(), o,
                                                         &p)))
      return false;
    working = p.working;
    scores = p.score_count;
  }
  Result oracle, measured;
  ResourceBudget budget(working.peak_bytes);
  ResourceReservation job;
  for (size_t pass = 0; pass < 3; ++pass) {
    ResourceBudget ample(0);
    ResourceReservation reference;
    if (pass == 0) {
      if (!Ok(ample.TryReserve(std::max(working.peak_bytes, size_t{64} << 20),
                               &reference)))
        return false;
    } else if (pass == 1 && !Ok(budget.TryReserve(working.peak_bytes, &job)))
      return false;
    {
      ResourceContextScope scope(
          {pass == 0 ? &reference : &job, ResourceClass::kPreparation});
      auto options = o;
      if (pass == 0)
        options.collect_timing = options.collect_profile = false;
      MixedTrace trace{pass == 0 ? ample : budget};
      ScopedTrace observe(mixed ? &trace : nullptr);
      const auto s =
          Encode(gpu, image, options, pass == 0 ? &oracle : &measured);
      if (!Ok(s)) {
        std::cerr << image.width() << 'x' << image.height()
                  << " mode=" << static_cast<int>(o.encoding.metal_aq_mode)
                  << " plan=" << working.peak_bytes << '\n';
        return false;
      }
      if (pass != 0 &&
          !Check(measured.bytes == oracle.bytes &&
                     measured.summary == oracle.summary &&
                     measured.summary.score_history.size() == scores,
                 "Bounded compatibility workflow changed output"))
        return false;
      if (pass != 0 && o.collect_timing) {
        if (!Check(measured.timing.attempts.size() ==
                       measured.summary.encode_attempt_count,
                   "Compatibility attempt timings lost backing"))
          return false;
        if (mixed && o.encoding.target_size_maximum_attempts == 64) {
          const bool qualified = IsAutomaticMetalBackendQualified(gpu);
          size_t metal = 0, cpu = 0;
          size_t first_metal = 64;
          for (size_t i = 0; i < measured.timing.attempts.size(); ++i) {
            const auto &attempt = measured.timing.attempts[i];
            if (!Check(attempt.succeeded,
                       "Mixed search lost a planned attempt"))
              return false;
            const bool selects_metal =
                qualified &&
                IsAutomaticMetalTargetEligible(attempt.butteraugli_target);
            if (selects_metal)
              first_metal = std::min(first_metal, i);
            (selects_metal ? metal : cpu)++;
          }
          if (!Check(
                  (metal > 0) == qualified && cpu > 0 &&
                      measured.timing.attempts.size() == 64,
                  "Mixed search did not cross the automatic target interval"))
            return false;
          if (qualified &&
              !Check(
                  trace.count == 64 && first_metal > 0 &&
                      first_metal + 1 < trace.count &&
                      trace.aq_live[first_metal] >
                          trace.aq_live[first_metal - 1] &&
                      trace.aq_live[first_metal + 1] >=
                          trace.aq_live[first_metal],
                  "Mixed search did not retain Metal AQ across a CPU attempt"))
            return false;
          std::cout << "Mixed target routing: Metal=" << metal << " CPU=" << cpu
                    << '\n';
        }
      }
      if ((pass == 0 || pass == 2) && !Trim(gpu))
        return false;
    }
    if (pass == 0) {
      reference.Reset();
      if (!Empty(ample))
        return false;
    }
  }
  if (image.width() >= 3839)
    std::cout << "Large compatibility mode="
              << static_cast<int>(o.encoding.metal_aq_mode)
              << " plan=" << working.peak_bytes
              << " peak=" << budget.snapshot().peak_backing_bytes
              << " bytes=" << measured.bytes.size() << '\n';
  job.Reset();
  return Empty(budget);
}
bool CheckRuntime(GpuBackend &gpu, bool large) {
  size_t cases = 0;
  if (large) {
    auto image = MakeImage({3839, 2159});
    for (size_t mode : {size_t{0}, size_t{2}, size_t{4}}) {
      auto o = Options(mode);
      o.encoding.effort = 1;
      o.encoding.cpu_thread_count = 4;
      if (!Run(gpu, image.const_view(), o))
        return false;
    }
    return true;
  }
  for (Extent2D extent :
       {Extent2D{1, 1}, {14, 15}, {15, 15}, {65, 63}, {257, 257}}) {
    auto image = MakeImage(extent);
    for (size_t mode = 0; mode < 5; ++mode)
      for (int effort : {1, 7}) {
        auto o = Options(mode);
        o.encoding.effort = effort;
        o.encoding.cpu_thread_count = effort == 1 ? 1 : 4;
        o.collect_profile = true;
        if (!Run(gpu, image.const_view(), o))
          return false;
        ++cases;
      }
  }
  auto image = MakeImage({128, 96});
  for (bool high_density : {false, true}) {
    auto o = Options(0);
    o.encoding.effort = 1;
    o.encoding.cpu_thread_count = 0;
    o.encoding.compression_mode = VarDctCompressionMode::kMaximumCompression;
    if (high_density)
      o.encoding.density_mode = VarDctDensityMode::kHighDensity;
    o.collect_profile = true;
    if (!Run(gpu, image.const_view(), o))
      return false;
    ++cases;
  }
  for (size_t mode : {size_t{0}, size_t{4}})
    for (size_t attempts : {size_t{1}, size_t{4}}) {
      auto o = Options(mode);
      o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
      o.encoding.target_bytes = 1;
      o.encoding.target_size_tolerance = 0;
      o.encoding.target_size_maximum_attempts = attempts;
      o.collect_timing = true;
      if (!Run(gpu, image.const_view(), o))
        return false;
      ++cases;
    }
  auto o = Options(0);
  o.encoding.backend = VarDctBackendPreference::kAutomatic;
  o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
  o.encoding.target_bytes = 1;
  o.encoding.target_size_tolerance = 0;
  o.encoding.target_size_maximum_attempts = 64;
  o.encoding.effort = 1;
  o.collect_timing = true;
  if (!Run(gpu, image.const_view(), o, true))
    return false;
  std::cout << "Bounded compatibility workflows: " << cases + 1 << '\n';
  return true;
}
bool CheckFailures(GpuBackend &gpu) {
  auto image = MakeImage({17, 9});
  for (size_t mode = 0; mode < 5; ++mode) {
    auto o = Options(mode);
    o.encoding.effort = 1;
    MetalCompatibilityWorkflowStoragePlan plan;
    if (!Ok(ComputeMetalCompatibilityWorkflowStoragePlan(image.extent(), o,
                                                         &plan)))
      return false;
    ResourceBudget rejected(plan.working.peak_bytes - 1);
    ResourceReservation none;
    if (!Check(!rejected.TryReserve(plan.working.peak_bytes, &none).ok(),
               "Oversized compatibility reservation admitted") ||
        !Empty(rejected))
      return false;
    for (bool underplan : {false, true}) {
      ResourceBudget budget(underplan ? 1 : plan.working.peak_bytes);
      ResourceReservation job;
      if (!Ok(budget.TryReserve(underplan ? 1 : plan.working.peak_bytes, &job)))
        return false;
      Result result;
      result.bytes = {1, 2, 3};
      result.summary.score_history = {42};
      const auto summary = result.summary;
      {
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        if (!underplan)
          ArmNextManagedHostAllocationFailureForTest();
        const auto status = Encode(gpu, image.const_view(), o, &result);
        DisarmManagedHostAllocationFailureForTest();
        if (!Check(
                status.code() == StatusCode::kOutOfMemory &&
                    status.resource_plan_exceeded() == underplan &&
                    result.bytes == std::vector<uint8_t>({1, 2, 3}) &&
                    result.summary == summary,
                "Compatibility allocation failure lost type or atomic output"))
          return false;
        if (!underplan && !Ok(Encode(gpu, image.const_view(), o, &result)))
          return false;
        if (!Trim(gpu))
          return false;
      }
      job.Reset();
      if (!Empty(budget))
        return false;
    }
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  const bool large = argc == 2 && std::string_view(argv[1]) == "--large";
  if (!CheckPlans())
    return EXIT_FAILURE;
  std::unique_ptr<GpuBackend> gpu;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)) ||
      !Ok(EnsureProductionMetalBackendAvailable()) ||
      !CheckRuntime(*gpu, large) || (!large && !CheckFailures(*gpu)))
    return EXIT_FAILURE;
  gpu.reset();
  return Empty(DefaultResourceBudget()) &&
                 Check(DefaultResourceBudget().snapshot().peak_backing_bytes ==
                           0,
                       "Compatibility backing escaped the explicit domain")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
