// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/storage_plan.h"
#include "codestream/modular/workflow.h"
#include "codestream/modular/search.h"
#include "codestream/workflow_admission_scope.h"
#include "codestream/workflow_admission_test.h"
#include "core/cpu_execution.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void Empty(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(s.live_capacity_bytes == 0 && s.reserved_unbacked_bytes == 0 &&
            s.active_reservations == 0 && s.active_cpu_participants == 0,
        "Workflow leaked backing, admission or CPU participation");
}
void Validation() {
  std::array<uint8_t, 64> bytes{};
  const Rgb8View valid{std::span(bytes).subspan(3), {3, 4}, 13};
  Ok(valid.Validate());
  std::vector<uint8_t> output{7, 8, 9};
  for (const Rgb8View bad :
       {Rgb8View{{}, {3, 4}, 13}, Rgb8View{bytes, {0, 1}, 3}, Rgb8View{bytes, {3, 4}, 8},
        Rgb8View{std::span(bytes).first(47), {3, 4}, 13}, Rgb8View{bytes, {3, 4}, SIZE_MAX},
        Rgb8View{bytes, {SIZE_MAX, 1}, SIZE_MAX}}) {
    ArmNextManagedHostAllocationFailureForTest();
    Check(!EncodeRgb8Modular(bad, {}, &output).ok(), "Invalid view accepted");
    Check(ManagedHostAllocationFailurePendingForTest(), "Invalid input allocated backing");
    DisarmManagedHostAllocationFailureForTest();
    Check(output == std::vector<uint8_t>({7, 8, 9}), "Invalid input changed output");
  }
  Check(!EncodeRgb8Modular(valid, {}, nullptr).ok(), "Null output accepted");
  Check(!EncodeRgb8Modular(valid, {.entropy = static_cast<EntropyCodingMode>(255)}, &output).ok(),
        "Invalid entropy mode accepted");
  for (const auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularWorkflowStoragePlan plan;
    ArmNextManagedHostAllocationFailureForTest();
    Ok(ComputeModularWorkflowStoragePlan({257, 259}, mode, &plan));
    Check(ManagedHostAllocationFailurePendingForTest(), "Workflow planner allocated backing");
    DisarmManagedHostAllocationFailureForTest();
    const auto prior = plan;
    Check(!ComputeModularWorkflowStoragePlan({SIZE_MAX, 2}, mode, &plan).ok() && plan == prior,
          "Overflow mutated plan");
  }
}
void FormatValidation() {
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    ModularInputProfile profile;
    Ok(ResolveModularInput({3, 4}, format, &profile));
    const size_t row = 3 * profile.channel_count * profile.bytes_per_sample;
    const size_t stride = row + 1;
    std::vector<uint8_t> bytes(1 + 3 * stride + row, 0xfe);
    PackedModularImageView valid{std::span(bytes).subspan(1), {3, 4}, stride, format};
    Ok(valid.Validate());
    for (unsigned kind = 0; kind < 8; ++kind) {
      auto bad = valid;
      switch (kind) {
      case 0:
        bad.bytes = bad.bytes.first(bad.bytes.size() - 1);
        break;
      case 1:
        bad.row_stride = row - 1;
        break;
      case 2:
        bad.row_stride = SIZE_MAX;
        break;
      case 3:
        bad.extent = {SIZE_MAX, 4};
        break;
      case 4:
        bad.format = static_cast<PackedModularFormat>(255);
        break;
      case 5:
        bad.byte_order = static_cast<SampleByteOrder>(255);
        break;
      case 6:
        bad.extent.height = 0;
        break;
      case 7:
        bad.bytes = {};
        break;
      }
      std::vector<uint8_t> output{7, 8, 9};
      ArmNextManagedHostAllocationFailureForTest();
      Check(!EncodeModularImage(bad, {}, &output).ok() &&
                ManagedHostAllocationFailurePendingForTest() &&
                output == std::vector<uint8_t>({7, 8, 9}),
            "Invalid packed input allocated or changed output");
      DisarmManagedHostAllocationFailureForTest();
    }
    for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
      ModularWorkflowStoragePlan plan;
      ArmNextManagedHostAllocationFailureForTest();
      Ok(ComputeModularWorkflowStoragePlan({257, 259}, format, mode, &plan));
      Check(ManagedHostAllocationFailurePendingForTest() &&
                plan.tokens == 257 * 259 * profile.channel_count,
            "Format planner allocated or counted channels incorrectly");
      DisarmManagedHostAllocationFailureForTest();
      const auto previous = plan;
      Check(!ComputeModularWorkflowStoragePlan({SIZE_MAX, 2}, format, mode, &plan).ok() &&
                plan == previous,
            "Format overflow mutated plan");
    }
    // Byte order has no effect on 8-bit input, but both enum values are accepted.
    if (profile.bytes_per_sample == 1) {
      std::vector<uint8_t> little, big;
      Ok(EncodeModularImage(valid, {}, &little));
      valid.byte_order = SampleByteOrder::kBigEndian;
      Ok(EncodeModularImage(valid, {}, &big));
      Check(little == big, "8-bit input depends on byte order");
      if (format == PackedModularFormat::kRgb8) {
        Ok(EncodeRgb8Modular({valid.bytes, valid.extent, valid.row_stride}, {}, &big));
        Check(little == big, "RGB8 adapter changed output");
      }
    }
  }
}
void PolicyValidation() {
  std::array<uint8_t, 12> bytes{};
  PackedModularImageView view{bytes, {2, 2}, 6};
  for (unsigned i = 0; i < 12; ++i) {
    ModularCodingPolicy policy;
    switch (i) {
    case 0:
      policy.tree.size = 0;
      break;
    case 1:
      policy.tree.size = 32;
      break;
    case 2:
      policy.tree.nodes[0].predictor = static_cast<Predictor>(14);
      break;
    case 3:
      policy.tree.nodes[0].multiplier = 0;
      break;
    case 4:
      policy.tree.nodes[0].multiplier = UINT32_MAX;
      break;
    case 5:
      policy.weighted.coefficients[0] = 32;
      break;
    case 6:
      policy.weighted.weights[0] = 16;
      break;
    case 7:
      policy.rct = 42;
      break;
    case 8:
      policy.tree.size = 2;
      break; // Unreachable node.
    case 9:
      policy.tree.size = 3;
      policy.tree.nodes[0] = {.property = 0, .left = 0, .right = 2};
      break;
    case 10:
      policy.tree.size = 3;
      policy.tree.nodes[0] = {.property = 16, .left = 1, .right = 2};
      break;
    case 11:
      policy.tree.size = 5;
      policy.tree.nodes[0] = {.property = 0, .split = 10, .left = 1, .right = 2};
      policy.tree.nodes[1] = {.property = 0, .split = 9, .left = 3, .right = 4};
      break;
    }
    std::vector<uint8_t> output{7, 8, 9};
    ArmNextManagedHostAllocationFailureForTest();
    Check(!EncodeModularImage(view, {.coding = policy}, &output).ok() &&
              ManagedHostAllocationFailurePendingForTest() &&
              output == std::vector<uint8_t>({7, 8, 9}),
          "Invalid policy allocated or changed output");
    DisarmManagedHostAllocationFailureForTest();
  }
  ModularCodingPolicy policy;
  policy.rct = 6;
  auto gray = view;
  gray.format = PackedModularFormat::kGray8;
  std::vector<uint8_t> output{7, 8, 9};
  ArmNextManagedHostAllocationFailureForTest();
  Check(!EncodeModularImage(gray, {.coding = policy}, &output).ok() &&
            ManagedHostAllocationFailurePendingForTest(),
        "Gray RCT accepted or allocated");
  Check(!EncodeModularImage(view, {.coding = policy, .search = true}, &output).ok() &&
            ManagedHostAllocationFailurePendingForTest(),
        "Search overrode prescribed policy");
  DisarmManagedHostAllocationFailureForTest();
  policy = {};
  policy.tree.nodes[0].offset = 1;
  policy.tree.nodes[0].multiplier = 2;
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({}, &domain));
  Check(!EncodeModularImage(view, {domain, EntropyCodingMode::kPrefix, policy}, &output).ok() &&
            output == std::vector<uint8_t>({7, 8, 9}),
        "Nonintegral residual changed output");
  Empty(*domain);
}
void SharedDomain() {
  const Extent2D extent{17, 19};
  std::vector<uint8_t> pixels(17 * 19 * 3, 117);
  Rgb8View input{pixels, extent, 51};
  ModularWorkflowStoragePlan plan;
  Ok(ComputeModularWorkflowStoragePlan(extent, EntropyCodingMode::kPrefix, &plan));
  std::shared_ptr<const ExecutionDomain> domain, other;
  Ok(ExecutionDomain::Create(
      {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 1}, &domain));
  Ok(ExecutionDomain::Create({}, &other));
  std::vector<uint8_t> first, second;
  Status a, b;
  std::thread worker([&] { a = EncodeRgb8Modular(input, {domain}, &first); });
  b = EncodeRgb8Modular(input, {domain}, &second);
  worker.join();
  Ok(a);
  Ok(b);
  Check(first == second && domain->snapshot().peak_cpu_participants == 1 &&
            domain->snapshot().peak_committed_bytes <= plan.working.peak_bytes,
        "Simultaneous calls bypassed their shared domain");
  Empty(*domain);
  {
    codestream_internal::WorkflowAdmission outer;
    Ok(outer.Start(plan.working.peak_bytes, domain));
    thread_budget_internal::CpuExecutionScope participant;
    Ok(participant.Start(domain, 1));
    Ok(EncodeRgb8Modular(input, {domain}, &second));
    const auto saved = second;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = EncodeRgb8Modular(input, {other}, &second);
    Check(status.code() == StatusCode::kInvalidArgument && second == saved &&
              ManagedHostAllocationFailurePendingForTest(),
          "Foreign nested domain bypassed admission");
    DisarmManagedHostAllocationFailureForTest();
    Check(domain->snapshot().active_cpu_participants == 1,
          "Nested workflow released its outer CPU participant");
  }
  Empty(*domain);
  Empty(*other);
}
void Workflow(Extent2D extent, EntropyCodingMode mode, bool faults,
              PackedModularFormat format = PackedModularFormat::kRgb8,
              ModularCodingPolicy coding = {}, bool search = false) {
  ModularInputProfile profile;
  Ok(ResolveModularInput(extent, format, &profile));
  const size_t row_bytes = extent.width * profile.channel_count * profile.bytes_per_sample;
  const size_t stride = row_bytes + 7;
  std::vector<uint8_t> bytes(5 + stride * extent.height, 0xcd);
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < row_bytes; ++x)
      bytes[5 + y * stride + x] = static_cast<uint8_t>(x * 37 + y * 71);
  const auto unchanged = bytes;
  PackedModularImageView view{std::span(bytes).subspan(5), extent, stride, format,
                              SampleByteOrder::kBigEndian};
  ModularWorkflowStoragePlan plan;
  Ok(search ? ComputeModularSearchStoragePlan(extent, format, mode, &plan)
            : ComputeModularWorkflowStoragePlan(extent, format, mode, coding, &plan));
  std::shared_ptr<const ExecutionDomain> domain;
  // Allow a retained result to coexist with a fresh complete reservation.
  Ok(ExecutionDomain::Create(
      {.managed_memory_bytes = plan.working.peak_bytes + plan.output.peak_bytes,
       .cpu_participant_limit = 1},
      &domain));
  ModularEncodingOptions options{domain, mode, coding, search};
  std::vector<uint8_t> expected;
  Ok(EncodeModularImage(view, options, &expected));
  Empty(*domain);
  Check(domain->snapshot().peak_backing_bytes <= plan.working.peak_bytes &&
            domain->snapshot().peak_cpu_participants == 1,
        "Workflow exceeded planned envelope");
  Check(expected.size() <= plan.maximum_codestream_bytes, "Output exceeded planned size");
  codestream_internal::CodestreamBuffer retained;
  Ok(EncodeModularImageOwned(view, options, &retained));
  Check(std::ranges::equal(retained.view(), expected) &&
            domain->snapshot().live_capacity_bytes == retained.capacity(),
        "Owned output lost its charge");
  Ok(EncodeModularImageOwned(view, options, &retained));
  Check(std::ranges::equal(retained.view(), expected), "Owned replacement changed bytes");
  ArmNextManagedHostAllocationFailureForTest();
  Check(!EncodeModularImageOwned(view, options, &retained).ok() &&
            std::ranges::equal(retained.view(), expected),
        "Failed owned replacement changed output");
  DisarmManagedHostAllocationFailureForTest();
  std::thread destroy([value = std::move(retained)]() mutable { value.Reset(); });
  destroy.join();
  Empty(*domain);
  std::vector<uint8_t> output{7, 8, 9};
  codestream_internal::ArmNextWorkflowAdmissionCapacityForTest(1);
  auto status = EncodeModularImage(view, options, &output);
  codestream_internal::DisarmWorkflowAdmissionCapacityForTest();
  Check(status.resource_plan_exceeded() && output == std::vector<uint8_t>({7, 8, 9}),
        "Underplanning was not terminal and atomic");
  Empty(*domain);
  std::shared_ptr<const ExecutionDomain> small;
  Ok(ExecutionDomain::Create({.managed_memory_bytes = plan.working.peak_bytes - 1}, &small));
  ArmNextManagedHostAllocationFailureForTest();
  Check(!EncodeModularImage(view, {small, mode, coding, search}, &output).ok() &&
            ManagedHostAllocationFailurePendingForTest(),
        "Insufficient admission allocated backing");
  DisarmManagedHostAllocationFailureForTest();
  Empty(*small);
  if (faults) {
    size_t failure_count = 0;
    for (; failure_count < 10000; ++failure_count) {
      ArmManagedHostAllocationFailureAfterForTest(failure_count);
      status = EncodeModularImage(view, options, &output);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      Empty(*domain);
      if (status.ok()) {
        Check(pending && output == expected, "Fault injection did not cover every allocation");
        break;
      }
      Check(!pending && status.code() == StatusCode::kOutOfMemory &&
                output == std::vector<uint8_t>({7, 8, 9}),
            "Allocation failure changed output or status");
    }
    Check(failure_count > 10 && failure_count < 10000, "Allocation sweep did not finish");
    std::cout << "allocation failures " << (mode == EntropyCodingMode::kPrefix ? "prefix " : "ANS ")
              << "format " << static_cast<int>(format) << ' ' << extent.width << 'x'
              << extent.height << " policy "
              << (search                            ? "search"
                  : coding == ModularCodingPolicy{} ? "baseline"
                                                    : "prescribed")
              << ": " << failure_count << '\n';
  }
  Ok(EncodeModularImage(view, options, &output));
  Check(output == expected && bytes == unchanged,
        "Recovery, determinism or input immutability failed");
  Empty(*domain);
}
} // namespace
int main() try {
  Validation();
  FormatValidation();
  PolicyValidation();
  SharedDomain();
  for (const auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    Workflow({1, 1}, mode, true);
    Workflow({17, 19}, mode, true);
    Workflow({257, 3}, mode, true);
    Workflow({2049, 1}, mode, false);
    for (auto format :
         {PackedModularFormat::kGray8, PackedModularFormat::kGray16, PackedModularFormat::kRgb16,
          PackedModularFormat::kRgba8, PackedModularFormat::kRgba16}) {
      Workflow({17, 19}, mode, true, format);
      Workflow({257, 3}, mode, true, format);
      Workflow({2049, 1}, mode, false, format);
    }
  }
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularCodingPolicy policy;
    policy.rct = 6;
    policy.tree.size = 3;
    policy.tree.nodes[0] = {.property = 15, .left = 1, .right = 2};
    policy.tree.nodes[1].predictor = Predictor::kWeighted;
    policy.tree.nodes[2].predictor = Predictor::kAverage4;
    policy.weighted.coefficients = {17, 9, 6, 8, 7, 1, 2};
    Workflow({17, 19}, mode, true, PackedModularFormat::kRgba16, policy);
    Workflow({257, 3}, mode, true, PackedModularFormat::kRgba16, policy);
    Workflow({17, 19}, mode, true, PackedModularFormat::kRgba16, {}, true);
    Workflow({257, 3}, mode, true, PackedModularFormat::kRgba16, {}, true);
  }
  std::cout << "Modular workflow contracts passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
