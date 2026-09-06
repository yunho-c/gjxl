// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cstdlib>
#include <iostream>
#include <thread>

#include "gpu/ops/gpu_execution_profile_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::gpu_profile_internal;

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
    s.total.pending_count == 0 && s.open_reservations == 0, "Diagnostic resource leak");
}

struct Backings {
  size_t bytes = 0;
  size_t count = 0;
  template <typename T> void Add(const ProfileStorage<T>& values) {
    bytes += values.capacity() * sizeof(T);
    count += values.capacity() != 0;
  }
  void Add(const ProfileString& value) {
    // Independent accounting oracle: SSO capacity needs no character backing;
    // long capacity includes one extra allocated character for the terminator.
    if (value.capacity() > ProfileString{}.capacity()) {
      bytes += value.capacity() + 1;
      ++count;
    }
  }
  void Add(const GpuExecutionProfile& profile) {
    Add(profile.wall_stages);
    Add(profile.submissions);
    for (const auto& wall : profile.wall_stages) Add(wall.stage_id);
    for (const auto& submission : profile.submissions) {
      Add(submission.submission_id);
      Add(submission.stages);
      for (const auto& stage : submission.stages) {
        Add(stage.stage_id); Add(stage.group_id); Add(stage.dispatches);
        for (const auto& dispatch : stage.dispatches) Add(dispatch.kernel_id);
      }
    }
  }
  bool Matches(const ResourceBudget& budget) const {
    const auto s = budget.snapshot();
    const auto& diagnostic = s.classes[static_cast<size_t>(ResourceClass::kDiagnostics)];
    return Check(s.total.live_capacity_bytes == bytes && s.total.backing_count == count &&
      s.total.pending_count == 0 && diagnostic.live_capacity_bytes == bytes &&
      diagnostic.backing_count == count, "Nested diagnostic backing/capacity differs");
  }
};

GpuExecutionProfile MakeProfile() {
  GpuExecutionProfile profile;
  profile.mode = GpuProfilingMode::kStage;
  profile.capabilities = {true, true, true};
  profile.wall_stages.reserve(4);
  profile.wall_stages.push_back({.stage_id = ProfileString(96, 'w')});
  profile.submissions.resize(2);
  for (auto& submission : profile.submissions) {
    submission.submission_id = ProfileString(80, 's');
    submission.stages.resize(3);
    for (auto& stage : submission.stages) {
      stage.stage_id = "SSO";
      stage.group_id = ProfileString(64, 'g');
      stage.dispatches.resize(2);
      stage.dispatches[0].kernel_id = "short";
      stage.dispatches[1].kernel_id = ProfileString(128, 'k');
    }
  }
  // Empty logical containers/labels can still own capacity.
  profile.submissions[0].stages[0].dispatches.clear();
  profile.submissions[0].submission_id.clear();
  return profile;
}

bool CheckNestedOwnershipAndPublication() {
  ResourceBudget budget(65536);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(65536, &job))) return false;
  GpuExecutionProfile first, copy;
  Backings expected, copied;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    first = MakeProfile();
    expected.Add(first);
    if (!expected.Matches(budget)) return false;
    copy = first;
    copied.Add(copy);
    expected.Add(copy);
    if (!expected.Matches(budget) || !Check(first == copy, "Diagnostic copy changed data")) return false;
  }
  job.Reset();
  const auto* wall_pointer = first.wall_stages.data();
  const auto* long_pointer = first.wall_stages[0].stage_id.data();
  GpuExecutionProfile published;
  std::thread publisher([&published, retained = std::move(first)]() mutable {
    published = std::move(retained);
    published.ReleaseResourceChargesAfterPublication();
    published.ReleaseResourceChargesAfterPublication();  // Idempotent.
  });
  publisher.join();
  if (!copied.Matches(budget) || !Check(published == copy &&
      published.wall_stages.data() == wall_pointer &&
      published.wall_stages[0].stage_id.data() == long_pointer,
      "Diagnostic publication copied or damaged nested backing")) return false;
  std::thread consumer([retained = std::move(copy)] {});
  consumer.join();
  if (!Empty(budget)) return false;
  // Published backing still has a valid allocator header for later frees and
  // growth, but the external owner is no longer charged to the closed job.
  published.wall_stages[0].stage_id.append(1000, 'x');
  published.wall_stages.reserve(32);
  return Empty(budget);
}

bool CheckPublicationStringBoundaries() {
  for (size_t length : {size_t{0}, size_t{1}, ProfileString{}.capacity(),
                       ProfileString{}.capacity() + 1, size_t{4096}}) {
    ResourceBudget budget(16384);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(16384, &job))) return false;
    ProfileString value;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      value.assign(length, 'a');
      Backings expected;
      expected.Add(value);
      if (!expected.Matches(budget)) return false;
    }
    job.Reset();
    const auto* pointer = value.data();
    ReleaseManagedBackingAfterPublication(value);
    if (!Check(value.data() == pointer && value.size() == length, "String publication changed data") ||
        !Empty(budget)) return false;
  }
  return true;
}

Status MakeSessionProfile(GpuExecutionProfile* output) {
  try {
    GpuProfilingSession session(GpuProfilingMode::kStage, {true, true, true});
    Status status = session.EndWallStage("a deliberately non-SSO diagnostic wall stage",
      GpuWallStageKind::kPreparation, GpuProfilingSession::BeginWallStage());
    if (!status.ok()) return status;
    auto child = MakeProfile();
    child.submissions[0].submission_id = "child";
    status = session.Append(std::move(child));
    if (!status.ok()) return status;
    *output = std::move(session).Finish();
    return Status::Ok();
  } catch (const ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Injected diagnostic allocation failure");
  }
}

bool CheckFailures() {
  size_t fail_at = 0;
  for (; fail_at < 128; ++fail_at) {
    ResourceBudget budget(65536);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(65536, &job))) return false;
    bool injected;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      ArmManagedHostClassAllocationFailureAfterForTest(ResourceClass::kDiagnostics, fail_at);
      ManagedVector<uint8_t> unrelated(16);  // Must not consume a diagnostic-only hook.
      GpuExecutionProfile output;
      output.mode = GpuProfilingMode::kDispatch;
      const Status status = MakeSessionProfile(&output);
      injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (injected) {
        if (!Check(status.code() == StatusCode::kOutOfMemory &&
            output.mode == GpuProfilingMode::kDispatch && output.submissions.empty() &&
            budget.snapshot().total.live_capacity_bytes == 16,
            "Diagnostic failure was not atomic or ignored its class filter")) return false;
      } else if (!Ok(status) || !Check(!output.submissions.empty(), "Diagnostic fixture is empty")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;
    if (!injected) break;
  }
  if (!Check(fail_at > 20 && fail_at < 128, "Diagnostic failure enumeration did not finish")) return false;
  std::cout << "Diagnostic graph failure positions: " << fail_at << '\n';
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(1, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    GpuExecutionProfile output;
    output.mode = GpuProfilingMode::kDispatch;
    const Status status = MakeSessionProfile(&output);
    if (!Check(status.resource_plan_exceeded() && output.mode == GpuProfilingMode::kDispatch &&
        output.wall_stages.empty(), "Diagnostic session swallowed a resource-plan overrun")) return false;
  }
  job.Reset();
  return Empty(budget);
}
}  // namespace

int main() {
  return CheckNestedOwnershipAndPublication() && CheckPublicationStringBoundaries() && CheckFailures() &&
    Check(DefaultResourceBudget().snapshot().peak_backing_bytes == 0, "Diagnostic domain escape")
    ? EXIT_SUCCESS : EXIT_FAILURE;
}
