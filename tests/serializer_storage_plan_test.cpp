// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#ifndef GJXL_SERIALIZER_ORACLE_ONLY
#include "codestream/serializer_storage_plan.h"
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

#include "codestream/encoder_internal.h"
#include "codestream/headers.h"
#include "codestream_frame_fixture.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
using codestream_test_internal::FixtureCheck;
using codestream_test_internal::FixtureOk;
using codestream_test_internal::FrameFixture;
using enum VarDctEntropyBehavior;
using enum VarDctCoefficientOrderBehavior;

struct FixtureCase {
  Extent2D blocks;
  size_t family;
  int pattern;
  bool padded;
};
constexpr std::array<FixtureCase, 12> kCases{
    {{{1, 1}, 0, 0, false},
     {{1, 1}, 0, 1, true},
     {{5, 1}, 0, 2, false},
     {{1, 5}, 4, 2, true},
     {{8, 8}, 2, 2, false},
     {{9, 7}, 7, 1, true},
     {{33, 1}, 0, 2, true},
     {{33, 33}, 7, 2, false},
     {{257, 1},
      0,
      0,
      true}, // Two DC groups, including nested auto measurement.
     {{128, 64},
      7,
      0,
      false}, // Six map candidates possible at the split boundary.
     {{129, 65}, 0, 2, true},
     {{257, 33}, 7, 0, true}}};

bool Create(const FixtureCase &c, FrameFixture *f) {
  if (!f->Create(c.blocks, c.family, c.pattern))
    return false;
  if (c.padded &&
      !FixtureOk(FrameGeometry::Create(
          {c.blocks.width * 8 - 1, c.blocks.height * 8 - 3}, &f->geometry)))
    return false;
  return FixtureCheck(f->view().valid(),
                      "Padded serializer fixture is invalid");
}

bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return FixtureCheck(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                          s.total.pending_count == 0 &&
                          s.open_reservations == 0 && s.waiting_requests == 0,
                      "Serializer test leaked a resource charge");
}

bool Oracle(const FrameFixture &f, VarDctCodestreamOptions coding,
            std::vector<uint8_t> *bytes) {
  thread_budget_internal::EncodeScope serial(1);
  return FixtureOk(EncodeVarDctCodestreamFromView(f.view(), coding, bytes));
}

void Emit(size_t value) {
  for (size_t i = 0; i < 8; ++i)
    std::cout.put(
        static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 255));
}

bool EmitOracle() {
  for (const auto &c : kCases) {
    FrameFixture f;
    if (!Create(c, &f))
      return false;
    for (auto entropy : {kBalanced, kHighDensity, kMaximumCompression}) {
      for (auto order : {kFull, kEffort7Dct8Sampled}) {
        std::vector<uint8_t> bytes;
        if (!Oracle(f, {entropy, order}, &bytes))
          return false;
        Emit(bytes.size());
        std::cout.write(reinterpret_cast<const char *>(bytes.data()),
                        bytes.size());
      }
    }
  }
  return std::cout.good() && Empty(DefaultResourceBudget());
}

#ifndef GJXL_SERIALIZER_ORACLE_ONLY
bool HeaderBounds() {
  BlockContextMapStoragePlan maps;
  SerializerHeaderStoragePlan headers;
  if (!FixtureOk(ComputeBlockContextMapStoragePlan({1, 1}, false, &maps)) ||
      !FixtureOk(ComputeSerializerHeaderStoragePlan(1, 1, maps, 0, &headers)))
    return false;
  HostStorageBound backing = headers.frame_scratch;
  HostStorageBound destination;
  if (!FixtureOk(ComputeEntropyWriterStorageBound(headers.frame_prefix_bits,
                                                  &destination)) ||
      !backing.Add(destination))
    return false;
  // Exercise every size selector transition without allocating giant images.
  constexpr std::array<size_t, 8> dimensions{1,    512,    513,    8192,
                                             8193, 262144, 262145, 0x3fffffff};
  for (size_t width : dimensions) {
    for (size_t height : dimensions) {
      for (uint8_t scale : {0, 7}) {
        ResourceBudget budget(backing.peak_bytes);
        ResourceReservation job;
        if (!FixtureOk(budget.Reserve(backing.peak_bytes, &job)))
          return false;
        {
          ResourceContextScope scope({&job, ResourceClass::kSerializer});
          BitWriter writer;
          SimpleVarDctCodestreamProfile profile;
          profile.x_qm_scale = scale;
          profile.b_qm_scale = 7 - scale;
          if (!FixtureOk(
                  WriteSimpleCodestreamHeader({width, height}, &writer)) ||
              !FixtureOk(WriteSimpleFrameHeader(profile, &writer)) ||
              !FixtureCheck(writer.bits_written() <=
                                    headers.frame_prefix_bits &&
                                budget.snapshot().peak_backing_bytes <=
                                    backing.peak_bytes,
                            "Header selectors exceeded the prefix envelope"))
            return false;
        }
        job.Reset();
        if (!Empty(budget))
          return false;
      }
    }
  }
  // The DC-global bound uses 36 bits for the two quantizer selectors.
  HostStorageBound quantizer;
  if (!FixtureOk(ComputeEntropyWriterStorageBound(36, &quantizer)) ||
      !quantizer.Add(quantizer))
    return false;
  for (uint32_t global :
       {1u, 2048u, 2049u, 4096u, 4097u, 8192u, 8193u, kMaxEncoderGlobalScale}) {
    for (uint32_t dc : {1u, 16u, 32u, 33u, 256u, 257u, kMaxQuantDc}) {
      ResourceBudget budget(quantizer.peak_bytes);
      ResourceReservation job;
      if (!FixtureOk(budget.Reserve(quantizer.peak_bytes, &job)))
        return false;
      {
        ResourceContextScope scope({&job, ResourceClass::kSerializer});
        BitWriter writer;
        if (!FixtureOk(WriteSimpleQuantizer({global, dc}, &writer)) ||
            !FixtureCheck(writer.bits_written() <= 36 &&
                              budget.snapshot().peak_backing_bytes <=
                                  quantizer.peak_bytes,
                          "Quantizer exceeded its header envelope"))
          return false;
      }
      job.Reset();
      if (!Empty(budget))
        return false;
    }
  }
  return true;
}

bool PurePlans() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!FixtureOk(budget.Reserve(1, &job)))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kSerializer});
    ArmNextManagedHostAllocationFailureForTest();
    for (Extent2D extent : {Extent2D{1, 1},
                            {31, 31},
                            {256, 256},
                            {257, 259},
                            {2049, 257},
                            {3839, 2159},
                            {1ul << 24, 1}}) {
      for (auto entropy : {kBalanced, kHighDensity, kMaximumCompression}) {
        for (auto order : {kFull, kEffort7Dct8Sampled}) {
          size_t previous = 0;
          for (size_t threads : {1ul, 2ul, 8ul, SIZE_MAX}) {
            SerializerStoragePlan plan;
            const SerializerStorageOptions options{
                {entropy, order}, threads, true};
            if (!FixtureOk(
                    ComputeSerializerStoragePlan(extent, options, &plan)))
              return false;
            const auto blocks = extent.ceil_div(8);
            size_t b = 0, g = 0, d = 0, tiles = 0;
            if (!blocks.try_area(&b) || !blocks.ceil_div(32).try_area(&g) ||
                !blocks.ceil_div(256).try_area(&d) ||
                !blocks.ceil_div(8).try_area(&tiles))
              return false;
            if (!FixtureCheck(
                    plan.ac_group_count == g && plan.dc_group_count == d &&
                        plan.maximum_ac_tokens == 195 * b &&
                        plan.maximum_dc_tokens == 6 * b + 2 * tiles &&
                        plan.output.retained_bytes ==
                            plan.maximum_output_bytes &&
                        plan.output.peak_bytes == plan.maximum_output_bytes &&
                        plan.working.peak_bytes >= plan.output.peak_bytes &&
                        plan.working.peak_bytes >= previous,
                    "Serializer count or concurrency bound is inconsistent"))
              return false;
            previous = plan.working.peak_bytes;
            SerializerStoragePlan no_profile;
            auto bare = options;
            bare.collect_profile = false;
            if (!FixtureOk(
                    ComputeSerializerStoragePlan(extent, bare, &no_profile)) ||
                !FixtureCheck(
                    no_profile.working.peak_bytes <= plan.working.peak_bytes &&
                        no_profile.maximum_output_bytes ==
                            plan.maximum_output_bytes,
                    "Profiling changed output bound or reduced storage"))
              return false;
          }
          SerializerStoragePlan automatic;
          if (!FixtureOk(ComputeSerializerStoragePlan(
                  extent, {{entropy, order}, 0, true}, &automatic)) ||
              !FixtureCheck(automatic.working.peak_bytes >= previous,
                            "Auto plan omitted nested dispatch storage"))
            return false;
        }
      }
      SerializerStoragePlan full, sampled;
      if (!FixtureOk(ComputeSerializerStoragePlan(
              extent, {{kMaximumCompression, kFull}}, &full)) ||
          !FixtureOk(ComputeSerializerStoragePlan(
              extent, {{kMaximumCompression, kEffort7Dct8Sampled}},
              &sampled)) ||
          !FixtureCheck(full == sampled,
                        "Maximum compression did not normalize order policy"))
        return false;
    }
    if (!FixtureCheck(ManagedHostAllocationFailurePendingForTest() &&
                          budget.snapshot().peak_backing_bytes == 0,
                      "Whole-serializer planning allocated managed backing"))
      return false;
    DisarmManagedHostAllocationFailureForTest();
  }
  job.Reset();
  SerializerStoragePlan plan;
  plan.maximum_output_bytes = 17;
  const auto sentinel = plan;
  for (Extent2D extent : {Extent2D{}, {0, 1}, {SIZE_MAX, 1}, {1ul << 30, 1}}) {
    if (!FixtureCheck(!ComputeSerializerStoragePlan(extent, {}, &plan).ok() &&
                          plan == sentinel,
                      "Invalid serializer geometry changed output"))
      return false;
  }
  for (const auto options :
       {SerializerStorageOptions{
            {static_cast<VarDctEntropyBehavior>(99), kFull}},
        SerializerStorageOptions{
            {kBalanced, static_cast<VarDctCoefficientOrderBehavior>(99)}}}) {
    if (!FixtureCheck(
            !ComputeSerializerStoragePlan({32, 32}, options, &plan).ok() &&
                plan == sentinel,
            "Invalid serializer options changed output"))
      return false;
  }
  return FixtureCheck(
             !ComputeSerializerStoragePlan({0x3fffffff, 0x3fffffff}, {}, &plan)
                     .ok() &&
                 plan == sentinel &&
                 !ComputeSerializerStoragePlan({1, 1}, {}, nullptr).ok(),
             "Overflowing serializer plan modified output") &&
         Empty(budget);
}

bool EncodeWithinPlan(const FrameFixture &f, SerializerStorageOptions options,
                      const std::vector<uint8_t> &oracle) {
  SerializerStoragePlan plan;
  if (!FixtureOk(
          ComputeSerializerStoragePlan(f.geometry.frame(), options, &plan)))
    return false;
  ResourceBudget budget(plan.working.peak_bytes);
  ResourceReservation job;
  if (!FixtureOk(budget.Reserve(plan.working.peak_bytes, &job)))
    return false;
  {
    CodestreamBuffer output;
    VarDctCodestreamProfile profile;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      thread_budget_internal::EncodeScope threads(options.cpu_thread_count);
      if (!FixtureOk(EncodeVarDctCodestreamToBuffer(
              f.view(), options.coding, &output,
              options.collect_profile ? &profile : nullptr)))
        return false;
    }
    const auto s = budget.snapshot();
    const auto serial =
        s.classes[static_cast<size_t>(ResourceClass::kSerializer)];
    if (!FixtureCheck(std::ranges::equal(output.view(), oracle) &&
                          output.capacity() <= plan.maximum_output_bytes &&
                          s.peak_backing_bytes <= plan.working.peak_bytes &&
                          s.total.live_capacity_bytes == output.capacity() &&
                          serial.live_capacity_bytes == output.capacity() &&
                          s.total.backing_count == 1 &&
                          serial.backing_count == 1 &&
                          s.total.pending_count == 0,
                      "Whole serializer exceeded plan, changed bytes, or "
                      "retained hidden owners"))
      return false;
    job.Reset();
    if (!FixtureCheck(budget.snapshot().total.live_capacity_bytes ==
                          output.capacity(),
                      "Serializer result was uncharged before publication"))
      return false;
    std::vector<uint8_t> published{17, 19};
    output.PublishTo(&published);
    if (!FixtureCheck(published == oracle,
                      "Serializer publication changed output") ||
        !Empty(budget))
      return false;
  }
  return Empty(budget);
}

bool RealEncodes() {
  size_t count = 0;
  for (const auto &c : kCases) {
    FrameFixture f;
    if (!Create(c, &f))
      return false;
    for (auto entropy : {kBalanced, kHighDensity, kMaximumCompression}) {
      for (auto order : {kFull, kEffort7Dct8Sampled}) {
        std::vector<uint8_t> oracle;
        if (!Oracle(f, {entropy, order}, &oracle))
          return false;
        for (size_t threads : {0ul, 1ul, 2ul, 8ul}) {
          for (bool profile : {false, true}) {
            if (!EncodeWithinPlan(f, {{entropy, order}, threads, profile},
                                  oracle))
              return false;
            ++count;
          }
        }
      }
    }
  }
  std::cerr << "Whole-serializer reservation cases: " << count << '\n';
  return Empty(DefaultResourceBudget());
}

bool FailureSweep() {
  FrameFixture f;
  if (!f.Create({1, 1}, 0, 1))
    return false;
  const std::array<uint8_t, 3> sentinel{17, 19, 23};
  thread_budget_internal::EncodeScope serial(
      1); // Physical hooks are thread-local.
  for (auto entropy : {kBalanced, kHighDensity, kMaximumCompression}) {
    SerializerStorageOptions options{{entropy, kFull}, 1, true};
    SerializerStoragePlan plan;
    std::vector<uint8_t> oracle;
    if (!FixtureOk(
            ComputeSerializerStoragePlan(f.geometry.frame(), options, &plan)) ||
        !Oracle(f, options.coding, &oracle))
      return false;
    bool completed = false;
    for (size_t fail = 0; fail < 32768; ++fail) {
      ResourceBudget budget(plan.working.peak_bytes);
      ResourceReservation job;
      if (!FixtureOk(budget.Reserve(plan.working.peak_bytes, &job)))
        return false;
      bool injected = false;
      {
        CodestreamBuffer output;
        if (!FixtureOk(CodestreamBuffer::CopyFrom(sentinel, &output)))
          return false;
        VarDctCodestreamProfile profile;
        profile.total_nanoseconds = 17;
        profile.assembly.output_copy_nanoseconds = 19;
        profile.selected_block_context_candidate_index = 23;
        const auto before = profile;
        {
          ResourceContextScope scope({&job, ResourceClass::kPreparation});
          ArmManagedHostAllocationFailureAfterForTest(fail);
          const Status status = EncodeVarDctCodestreamToBuffer(
              f.view(), options.coding, &output, &profile);
          injected = !ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          if (injected) {
            if (!FixtureCheck(status.code() == StatusCode::kOutOfMemory &&
                                  !status.resource_plan_exceeded() &&
                                  profile == before &&
                                  std::ranges::equal(output.view(), sentinel) &&
                                  budget.snapshot().total.backing_count == 0,
                              "Serializer allocation failure changed "
                              "output/profile or leaked"))
              return false;
          } else if (!FixtureOk(status) ||
                     !FixtureCheck(std::ranges::equal(output.view(), oracle),
                                   "Fault-free final encode changed bytes"))
            return false;
        }
      }
      job.Reset();
      if (!Empty(budget))
        return false;
      if (!injected) {
        std::cerr << "Whole-serializer allocation failure positions "
                  << static_cast<int>(entropy) << ": " << fail << '\n';
        completed = true;
        break;
      }
    }
    if (!FixtureCheck(completed, "Whole-serializer fault sweep did not finish"))
      return false;
  }
  // An underestimated admitted envelope cannot grow, fallback or publish.
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!FixtureOk(budget.Reserve(1, &job)) || !FixtureOk(job.ReduceCapacity(0)))
    return false;
  {
    CodestreamBuffer output;
    if (!FixtureOk(CodestreamBuffer::CopyFrom(sentinel, &output)))
      return false;
    VarDctCodestreamProfile profile;
    profile.total_nanoseconds = 19;
    const auto before = profile;
    const size_t default_peak =
        DefaultResourceBudget().snapshot().peak_backing_bytes;
    ResourceContextScope scope({&job, ResourceClass::kSerializer});
    ArmNextManagedHostAllocationFailureForTest();
    const Status status =
        EncodeVarDctCodestreamToBuffer(f.view(), {}, &output, &profile);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!FixtureCheck(
            status.resource_plan_exceeded() && pending && profile == before &&
                std::ranges::equal(output.view(), sentinel) &&
                budget.snapshot().peak_backing_bytes == 0 &&
                DefaultResourceBudget().snapshot().peak_backing_bytes ==
                    default_peak,
            "Whole serializer escaped an underestimated reservation"))
      return false;
  }
  job.Reset();
  return Empty(budget);
}
#endif
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--oracle")
    return EmitOracle() ? EXIT_SUCCESS : EXIT_FAILURE;
#ifndef GJXL_SERIALIZER_ORACLE_ONLY
  return HeaderBounds() && PurePlans() && RealEncodes() && FailureSweep() &&
                 Empty(DefaultResourceBudget())
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
#else
  return EXIT_FAILURE;
#endif
}
