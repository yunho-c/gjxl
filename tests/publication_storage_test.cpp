// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>

#include "codestream/rate_control_internal.h"
#include "core/publication_vector.h"
#include "core/publication_output.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::codestream_internal;

bool Check(bool good, const char* message) {
  if (!good) std::cerr << message << '\n';
  return good;
}
bool Ok(const Status& status) {
  if (!status.ok()) std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget& budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
    s.total.pending_count == 0 && s.open_reservations == 0,
    "Publication backing or reservation leaked");
}

bool CheckOwnership() {
  ResourceBudget budget(1024), other(1024);
  ResourceReservation producer, consumer, foreign;
  CodestreamBuffer bytes;
  const std::array<uint8_t, 17> source{3, 5, 8};
  if (!Ok(budget.Reserve(128, &producer))) return false;
  {
    ResourceContextScope scope({&producer, ResourceClass::kInput});
    if (!Ok(CodestreamBuffer::CopyFrom(source, &bytes))) return false;
  }
  const auto* pointer = bytes.data();
  const auto s = budget.snapshot();
  if (!Check(bytes.capacity() == source.size() &&
      std::ranges::equal(bytes.view(), source) && s.total.live_capacity_bytes == 17 &&
      s.total.backing_count == 1 && s.total.pending_count == 0 &&
      s.classes[static_cast<size_t>(ResourceClass::kSerializer)].live_capacity_bytes == 17,
      "Publication backing capacity or category differs")) return false;
  producer.Reset();
  if (!Ok(budget.TryReserve(17, &consumer)) || !Ok(bytes.TransferTo(consumer)) ||
      !Ok(other.TryReserve(17, &foreign)) ||
      !Check(!bytes.TransferTo(foreign).ok(), "Publication crossed resource domains") ||
      !Ok(bytes.Reclassify(ResourceClass::kRetainedResult))) return false;
  foreign.Reset();
  consumer.Reset();
  if (!Check(budget.snapshot().committed_bytes() == 17 &&
      budget.snapshot().classes[static_cast<size_t>(ResourceClass::kRetainedResult)]
        .live_capacity_bytes == 17, "Closed producer lost retained result")) return false;
  std::vector<uint8_t> published{99, 100};
  std::thread thread([&published, retained = std::move(bytes)]() mutable {
    retained.PublishTo(&published);
  });
  thread.join();
  return Check(published.data() == pointer && std::ranges::equal(published, source),
    "Publication copied or changed its backing") && Empty(budget) && Empty(other);
}

bool CheckFailures() {
  ResourceBudget budget(64);
  ResourceReservation job;
  CodestreamBuffer bytes;
  if (!Ok(budget.TryReserve(64, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kSerializer});
    if (!Ok(CodestreamBuffer::Create(40, &bytes))) return false;
    const auto* pointer = bytes.data();
    // A replacement must authorize its entire new backing while the old one
    // still exists. The physical-allocation hook must not be consumed.
    ArmNextManagedHostAllocationFailureForTest();
    const Status underplan = CodestreamBuffer::Create(25, &bytes);
    const bool preauthorized = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(underplan.resource_plan_exceeded() && preauthorized &&
        bytes.data() == pointer && bytes.size() == 40 &&
        budget.snapshot().total.pending_count == 0,
        "Publication replacement allocated before authorization or lost output")) return false;
    ArmNextManagedHostAllocationFailureForTest();
    const Status physical = CodestreamBuffer::Create(24, &bytes);
    const bool injected = !ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(physical.code() == StatusCode::kOutOfMemory &&
        !physical.resource_plan_exceeded() && injected && bytes.data() == pointer &&
        budget.snapshot().total.live_capacity_bytes == 40 &&
        budget.snapshot().total.pending_count == 0,
        "Physical publication failure leaked or changed its reason")) return false;
    if (!Check(CodestreamBuffer::Create(1, nullptr).code() == StatusCode::kInvalidArgument &&
        PublicationVector<double>::Create(std::numeric_limits<size_t>::max(), nullptr)
          .code() == StatusCode::kInvalidArgument, "Null publication output accepted")) return false;
    PublicationVector<double> large;
    if (!Check(PublicationVector<double>::Create(std::numeric_limits<size_t>::max(), &large)
        .code() == StatusCode::kInvalidArgument, "Publication extent overflow accepted")) return false;
    if (!Ok(CodestreamBuffer::Create(24, &bytes)) ||
        !Check(budget.snapshot().peak_backing_bytes == 64 &&
          budget.snapshot().total.live_capacity_bytes == 24,
          "Replacement failed to count overlapping old and new backing")) return false;
    if (!Ok(CodestreamBuffer::CopyFrom({}, &bytes)) ||
        !Check(bytes.empty() && bytes.capacity() == 0 &&
          budget.snapshot().total.backing_count == 0, "Empty publication retained backing")) return false;
  }
  job.Reset();
  return Empty(budget);
}

struct DestructionProbe {
  static inline const ResourceBudget* budget = nullptr;
  static inline size_t expected_bytes = 0;
  static inline bool ordered = true;
  uint64_t value = 0;
  ~DestructionProbe() {
    if (budget != nullptr)
      ordered &= budget->snapshot().total.live_capacity_bytes == expected_bytes;
  }
};

struct ProbeRecord {
  std::vector<DestructionProbe> values;
};

bool CheckRecordDestruction() {
  using Owner = PublicationRecord<ProbeRecord, DestructionProbe, &ProbeRecord::values>;
  ResourceBudget budget(4096);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(4096, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kRetainedResult});
    PublicationVector<DestructionProbe> a, b, c;
    if (!Ok(PublicationVector<DestructionProbe>::Create(3, &a)) ||
        !Ok(PublicationVector<DestructionProbe>::Create(5, &b)) ||
        !Ok(PublicationVector<DestructionProbe>::Create(2, &c))) return false;
    Owner first, second;
    first.SetField(std::move(a));
    second.SetField(std::move(b));
    DestructionProbe::budget = &budget;
    DestructionProbe::ordered = true;
    DestructionProbe::expected_bytes = 10 * sizeof(DestructionProbe);
    first.SetField(std::move(c));
    DestructionProbe::expected_bytes = 7 * sizeof(DestructionProbe);
    first = std::move(second);
    DestructionProbe::expected_bytes = 5 * sizeof(DestructionProbe);
    // Closing the producer must not affect backing/charge destruction order.
    job.Reset();
    first.Reset();
    DestructionProbe::budget = nullptr;
    if (!Check(DestructionProbe::ordered, "Record replacement freed its ticket before backing")) return false;
  }
  return Empty(budget);
}

bool CheckDestructionAndEscrow() {
  ResourceBudget budget(4096);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(4096, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kRetainedResult});
    PublicationVector<DestructionProbe> first, second;
    if (!Ok(PublicationVector<DestructionProbe>::Create(3, &first)) ||
        !Ok(PublicationVector<DestructionProbe>::Create(5, &second))) return false;
    DestructionProbe::budget = &budget;
    DestructionProbe::expected_bytes = 8 * sizeof(DestructionProbe);
    first = std::move(second);
    DestructionProbe::expected_bytes = 5 * sizeof(DestructionProbe);
    first.Reset();
    DestructionProbe::budget = nullptr;
    if (!Check(DestructionProbe::ordered, "Move assignment freed ticket before backing")) return false;

    // The real batch publishes nested vectors before the outer result array.
    // Escrow must outlive unpublished nested backing on every rollback path.
    std::array<ResourceAllocation, 2> charges;
    PublicationVector<std::vector<uint8_t>> outer;
    std::array<CodestreamBuffer, 2> nested;
    if (!Ok(PublicationVector<std::vector<uint8_t>>::Create(2, &outer)) ||
        !Ok(CodestreamBuffer::Create(31, &nested[0])) ||
        !Ok(CodestreamBuffer::Create(47, &nested[1]))) return false;
    const size_t expected = 2 * sizeof(std::vector<uint8_t>) + 78;
    const auto* pointer = nested[0].data();
    for (size_t i = 0; i < 2; ++i) charges[i] = nested[i].MoveToPublication(&outer[i]);
    job.Reset();
    if (!Check(budget.snapshot().committed_bytes() == expected,
        "Nested result was uncharged before outer publication")) return false;
    std::vector<std::vector<uint8_t>> published;
    outer.PublishTo(&published);
    if (!Check(budget.snapshot().committed_bytes() == 78 && published[0].data() == pointer,
        "Outer publication lost nested escrow or copied backing")) return false;
    for (auto& charge : charges) charge.Reset();
    if (!Empty(budget)) return false;
  }
  // Destruction without publication on another thread must also release once.
  if (!Ok(budget.TryReserve(64, &job))) return false;
  CodestreamBuffer discarded;
  {
    ResourceContextScope scope({&job, ResourceClass::kSerializer});
    if (!Ok(CodestreamBuffer::Create(64, &discarded))) return false;
  }
  job.Reset();
  std::thread thread([retained = std::move(discarded)] {});
  thread.join();
  return Empty(budget);
}

bool CheckBoundedHistoryAndRecord() {
  using History = PublicationVector<double>;
  ResourceBudget budget(256);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(256, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kAqScratch});
    History history, internal;
    if (!Ok(History::CreateForAppend(3, &history, ResourceClass::kAqScratch)) ||
        !Check(history.empty() && history.capacity() == 3 &&
          budget.snapshot().total.live_capacity_bytes == 24,
          "Empty bounded history lost its backing charge")) return false;
    const auto* pointer = history.data();
    ArmNextManagedHostAllocationFailureForTest();
    const bool appended = history.Append(1.0).ok() && history.Append(2.0).ok() &&
      history.Append(3.0).ok();
    const Status full = history.Append(4.0);
    const bool allocation_free = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(appended && allocation_free && full.resource_plan_exceeded() &&
        history.size() == 3 && history.data() == pointer && history.back() == 3.0,
        "Bounded append grew, changed its previous values, or lost the plan error")) return false;
    PublicationOutput<double> owned_output(&internal), absent;
    if (!Check(absent == nullptr && owned_output != nullptr, "History output validity differs")) return false;
    owned_output.Publish(std::move(history));
    if (!Check(internal.data() == pointer && budget.snapshot().total.live_capacity_bytes == 24,
        "Internal history publication discarded its charge")) return false;

    OwnedEncodingSummary summary, replacement;
    summary.value().encoded_bytes = 123;
    summary.SetField(std::move(internal));
    if (!Check(summary.value().score_history.data() == pointer,
        "Summary adoption copied history")) return false;
    const std::array<double, 2> next{5.0, 6.0};
    if (!Ok(History::CopyFrom(next, &history, ResourceClass::kAqScratch))) return false;
    replacement.SetField(std::move(history));
    replacement.value().encoded_bytes = 456;
    if (!Check(budget.snapshot().total.live_capacity_bytes == 40, "Record overlap was not charged")) return false;
    summary = std::move(replacement);
    if (!Check(summary.value().encoded_bytes == 456 &&
        budget.snapshot().total.live_capacity_bytes == 16,
        "Record move assignment lost scalar metadata or leaked old history")) return false;
    if (!Ok(summary.Reclassify(ResourceClass::kRetainedResult))) return false;
    job.Reset();
    VarDctEncodingSummary published;
    auto charge = summary.MoveToPublication(&published);
    if (!Check(budget.snapshot().committed_bytes() == 16 &&
        std::ranges::equal(published.score_history, next), "Record publication lost escrow")) return false;
    charge.Reset();
    if (!Empty(budget)) return false;
  }
  // Empty logical size must not make an untracked capacity appear managed.
  History untracked;
  if (!Ok(History::CreateForAppend(1, &untracked)) ||
      !Check(!untracked.Reclassify(ResourceClass::kRetainedResult).ok(),
        "Empty untracked history bypassed ownership validation")) return false;
  return true;
}

bool CheckManagedScoreTies() {
  const auto score = [](float target) {
    return target == kMinimumTargetSizeButteraugliTarget ? 5.0 :
      target == kMaximumTargetSizeButteraugliTarget ? 2.0 : 0.0;
  };
  const TargetSizeSearchOptions options{.target_bytes = 90, .maximum_attempts = 3};
  TargetSizeSearchResult expected;
  const TargetSizeEvaluator oracle = [&](float target, std::vector<uint8_t>* bytes,
                                        VarDctEncodingSummary* summary) {
    bytes->assign(80, 0);
    summary->encoded_bytes = 80;
    summary->selected_butteraugli_target = target;
    summary->score_history = {score(target)};
    summary->final_butteraugli_score_evaluated = true;
    return Status::Ok();
  };
  if (!Ok(SearchTargetSize(options, oracle, &expected))) return false;
  ResourceBudget budget(1024);
  ResourceReservation job;
  ManagedTargetSizeSearchResult actual;
  if (!Ok(budget.TryReserve(1024, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kSerializer});
    const ManagedTargetSizeEvaluator evaluator = [&](float target, CodestreamBuffer* bytes,
                                                   OwnedEncodingSummary* owner) {
      Status status = CodestreamBuffer::Create(80, bytes, ResourceClass::kSerializer);
      if (!status.ok()) return status;
      PublicationVector<double> history;
      const std::array<double, 1> scores{score(target)};
      status = PublicationVector<double>::CopyFrom(scores, &history, ResourceClass::kAqScratch);
      if (!status.ok()) return status;
      owner->SetField(std::move(history));
      auto& summary = owner->value();
      summary.encoded_bytes = 80;
      summary.selected_butteraugli_target = target;
      summary.final_butteraugli_score_evaluated = true;
      return Status::Ok();
    };
    if (!Ok(SearchTargetSize(options, evaluator, &actual)) ||
        !Check(actual.summary.value() == expected.summary &&
          std::ranges::equal(actual.codestream.view(), expected.codestream) &&
          actual.attempt_count == expected.attempt_count &&
          budget.snapshot().total.live_capacity_bytes == 88 &&
          budget.snapshot().total.backing_count == 2 && budget.snapshot().peak_backing_bytes >= 176,
          "Managed score tie-break or retained candidate histories differ")) return false;
  }
  job.Reset();
  if (!Check(budget.snapshot().committed_bytes() == 88, "Closed search lost score ownership")) return false;
  std::thread consumer([retained = std::move(actual)] {});
  consumer.join();
  return Empty(budget);
}

bool CheckManagedSearch() {
  for (bool underplan : {false, true}) {
    ResourceBudget budget(1024);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(1024, &job))) return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kSerializer});
      ManagedTargetSizeSearchResult result;
      if (!Ok(CodestreamBuffer::Create(7, &result.codestream))) return false;
      const auto* previous = result.codestream.data();
      size_t attempts = 0;
      bool overlap = false;
      const ManagedTargetSizeEvaluator evaluator = [&](float target, CodestreamBuffer* bytes,
                                                       OwnedEncodingSummary* summary) {
        ++attempts;
        if (attempts == 2) {
          if (underplan) return CodestreamBuffer::Create(1024, bytes);
          return Status::OutOfMemory("Candidate-local physical allocation failure");
        }
        const size_t count = attempts == 1 ? 100 : 80;
        Status status = CodestreamBuffer::Create(count, bytes, ResourceClass::kSerializer);
        if (!status.ok()) return status;
        if (attempts == 3) overlap = budget.snapshot().total.live_capacity_bytes >= 7 + 100 + 80;
        summary->value().encoded_bytes = count;
        summary->value().selected_butteraugli_target = target;
        return Status::Ok();
      };
      const Status status = SearchTargetSize({.target_bytes = 90, .maximum_attempts = 3},
                                            evaluator, &result);
      if (underplan) {
        if (!Check(status.resource_plan_exceeded() && attempts == 2 &&
            result.codestream.data() == previous && result.codestream.size() == 7 &&
            result.attempt_count == 0 && budget.snapshot().total.live_capacity_bytes == 7,
            "Underplanned search continued, published, or retained candidates")) return false;
      } else if (!Ok(status) || !Check(attempts == 3 && overlap &&
          result.failed_attempt_count == 1 && result.codestream.size() == 80 &&
          budget.snapshot().total.live_capacity_bytes == 80,
          "Managed search changed candidate failure policy or lost retained charge")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;
  }
  return true;
}
}  // namespace

int main() {
  return CheckOwnership() && CheckFailures() && CheckDestructionAndEscrow() && CheckManagedSearch() &&
    CheckRecordDestruction() && CheckBoundedHistoryAndRecord() && CheckManagedScoreTies()
    ? EXIT_SUCCESS : EXIT_FAILURE;
}
