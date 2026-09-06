// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>

#include "c_api/image_conversion.h"
#include "codestream/batch_workflow_test.h"
#include "codestream/workflow_internal.h"
#include "gjxl/gjxl.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
constexpr size_t kTestEnvelope = 64 * 1024 * 1024;

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
    "Workflow publication leaked managed backing");
}

struct Fixture {
  std::array<uint8_t, 17 * 9 * 3> packed;
  Image3FBuffer linear;
  VarDctEncodingOptions options{
    .butteraugli_target = 1.0f, .effort = 1,
    .backend = VarDctBackendPreference::kCpu, .cpu_thread_count = 1};
  bool Init() {
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = 37 + i % 183;
    return Ok(c_api_internal::ConvertPackedSrgbToLinearRgb(
      {packed.data(), packed.size(), 17, 9, 51, c_api_internal::PackedPixelFormat::kRgb8Srgb},
      &linear));
  }
  GJXLImageView View() const {
    return {sizeof(GJXLImageView), 17, 9, GJXL_PIXEL_FORMAT_RGB8_SRGB,
            packed.data(), packed.size(), 51};
  }
};

bool CheckWorkflow(const Fixture& fixture) {
  ResourceBudget budget(kTestEnvelope);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(kTestEnvelope, &job))) return false;
  CodestreamBuffer retained;
  OwnedEncodingSummary summary;
  std::vector<uint8_t> published;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    if (!Ok(EncodeLinearRgbVarDctCodestreamOwned(fixture.linear.const_view(),
        fixture.options, &retained, &summary))) return false;
    const auto s = budget.snapshot();
    if (!Check(s.total.live_capacity_bytes == retained.capacity() +
        summary.value().score_history.capacity() * sizeof(double) && s.total.backing_count == 2 &&
        s.classes[static_cast<size_t>(ResourceClass::kSerializer)].live_capacity_bytes == retained.capacity(),
        "Owned workflow released result bytes or retained temporary backing")) return false;
    VarDctEncodingSummary reference;
    if (!Ok(EncodeLinearRgbVarDctCodestream(fixture.linear.const_view(), fixture.options,
        &published, &reference)) || !Check(std::ranges::equal(retained.view(), published) &&
        summary.value() == reference, "Owned/public workflow outputs differ")) return false;
    if (!Check(budget.snapshot().total.live_capacity_bytes == retained.capacity() +
        summary.value().score_history.capacity() * sizeof(double),
        "Public output remained in managed accounting")) return false;
  }
  job.Reset();
  const auto* pointer = retained.data();
  retained.PublishTo(&published);
  VarDctEncodingSummary published_summary;
  summary.PublishTo(&published_summary);
  return Check(published.data() == pointer, "Workflow publication copied output") && Empty(budget);
}

bool CheckPublicDiagnosticFailures(const Fixture& fixture) {
  for (auto owner : {ResourceClass::kRetainedResult, ResourceClass::kAqScratch}) {
    ResourceBudget budget(kTestEnvelope);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(kTestEnvelope, &job))) return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      std::vector<uint8_t> bytes{23};
      VarDctEncodingSummary summary;
      summary.score_history = {42};
      const auto old_summary = summary;
      VarDctEncodingTiming timing;
      timing.total_nanoseconds = 123;
      timing.attempts.resize(2);
      const auto* old_attempts = timing.attempts.data();
      ArmManagedHostClassAllocationFailureAfterForTest(owner, 0);
      const Status status = EncodeLinearRgbVarDctCodestreamProfiled(fixture.linear.const_view(),
        fixture.options, &bytes, &summary, &timing);
      const bool injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (!Check(injected && status.code() == StatusCode::kOutOfMemory &&
          bytes == std::vector<uint8_t>{23} && summary == old_summary &&
          timing.total_nanoseconds == 123 && timing.attempts.size() == 2 &&
          timing.attempts.data() == old_attempts && budget.snapshot().total.backing_count == 0,
          "Timing/score allocation failure was not atomic")) return false;
      if (!Ok(EncodeLinearRgbVarDctCodestreamProfiled(fixture.linear.const_view(), fixture.options,
          &bytes, &summary, &timing)) || !Check(!bytes.empty() && !summary.score_history.empty() &&
          timing.attempts.size() == 1 && budget.snapshot().total.backing_count == 0,
          "Public diagnostic output remained charged or failed recovery")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;
  }
  return true;
}

struct BatchObservation {
  const ResourceBudget* budget;
  const std::vector<VarDctBatchEncodingResult>* previous;
  size_t calls = 0;
  bool good = true;
  static void Observe(void* opaque, std::span<const VarDctBatchEncodingResult> results,
                      std::span<const OwnedEncodingResult> owned) noexcept {
    auto& self = *static_cast<BatchObservation*>(opaque);
    ++self.calls;
    size_t expected = results.size() * (sizeof(VarDctBatchEncodingResult) +
      sizeof(OwnedEncodingResult) + 3 * sizeof(ResourceAllocation));
    size_t backings = results.empty() ? 0 : 3;
    self.good &= results.size() == owned.size() && self.previous->size() == 1 &&
      (*self.previous)[0].codestream.size() == 1 && (*self.previous)[0].codestream[0] == 0x17;
    for (size_t i = 0; i < results.size(); ++i) {
      self.good &= results[i].codestream.empty();
      const auto& bytes = owned[i].codestream;
      const auto& summary = owned[i].summary.value();
      const auto& timing = owned[i].timing.value();
      if (results[i].status.ok()) {
        self.good &= !bytes.empty() && bytes.size() == summary.encoded_bytes &&
          results[i].summary.score_history.empty() && results[i].timing.attempts.empty();
        expected += bytes.capacity() + summary.score_history.capacity() * sizeof(double) +
          timing.attempts.capacity() * sizeof(VarDctEncodingAttemptTiming);
        backings += 1 + (summary.score_history.capacity() != 0) + (timing.attempts.capacity() != 0);
      } else self.good &= bytes.empty() && summary.score_history.empty() && timing.attempts.empty();
    }
    const auto s = self.budget->snapshot();
    const auto& result = s.classes[static_cast<size_t>(ResourceClass::kRetainedResult)];
    self.good &= s.total.pending_count == 0 && s.total.live_capacity_bytes == expected &&
      s.total.backing_count == backings && result.live_capacity_bytes == expected;
  }
};

bool CheckBatch(const Fixture& fixture) {
  std::array<VarDctBatchEncodingRequest, 4> requests;
  for (auto& request : requests) request = {fixture.linear.const_view(), fixture.options};
  requests[1].linear_rgb = {};  // Failed results must not retain candidate backing.
  for (size_t workers : {1, 3}) {
    ResourceBudget budget(kTestEnvelope);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(kTestEnvelope, &job))) return false;
    std::unique_ptr<VarDctBatchEncoder> encoder;
    if (!Ok(VarDctBatchEncoder::Create(workers, &encoder))) return false;
    std::vector<VarDctBatchEncodingResult> results(1);
    results[0].codestream = {0x17};
    BatchObservation observer{&budget, &results};
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      batch_publication_observer_for_testing = {&observer, BatchObservation::Observe};
      const Status status = encoder->Encode(requests, &results);
      batch_publication_observer_for_testing = {};
      if (!Ok(status) || !Check(observer.calls == 1 && observer.good && results.size() == 4 &&
          results[0].status.ok() && !results[1].status.ok() && results[2].status.ok() &&
          results[3].status.ok() && results[0].codestream == results[2].codestream &&
          results[2].codestream == results[3].codestream &&
          budget.snapshot().total.backing_count == 0,
          "Batch did not retain/account/publish all ordered results")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;

    // The structural arrays fit, but the image workflow does not. A worker
    // must inherit this reservation instead of allocating in the default one.
    if (!Ok(budget.TryReserve(8192, &job))) return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      if (!Ok(encoder->Encode(requests, &results))) return false;
      for (size_t i : {0, 2, 3})
        if (!Check(results[i].status.resource_plan_exceeded() && results[i].codestream.empty(),
            "Batch worker escaped an insufficient resource plan")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;

    // An array-allocation failure remains batch-atomic, including prior results.
    if (!Ok(budget.TryReserve(1, &job))) return false;
    results.resize(1);
    results[0].codestream = {0x17};
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      if (!Check(encoder->Encode(requests, &results).resource_plan_exceeded() &&
          results.size() == 1 && results[0].codestream == std::vector<uint8_t>({0x17}),
          "Batch setup failure published partial results")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;

    if (!Ok(budget.TryReserve(kTestEnvelope, &job))) return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      if (!Ok(encoder->Encode(requests, &results))) return false;
      for (size_t i : {0, 2, 3})
        if (!Check(results[i].status.ok() && !results[i].codestream.empty(),
            "Batch did not recover after resource-plan rejection")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;
  }
  return true;
}

bool CheckC(const Fixture& fixture) {
  GJXLContextOptions options;
  if (gjxl_context_options_init(&options, sizeof(options)) != GJXL_OK) return false;
  options.backend = GJXL_BACKEND_CPU;
  options.num_cpu_threads = 1;
  GJXLContext* raw_context = nullptr;
  if (gjxl_context_create(&options, &raw_context) != GJXL_OK) return false;
  std::unique_ptr<GJXLContext, decltype(&gjxl_context_destroy)> context(raw_context, gjxl_context_destroy);
  GJXLEncoderOptions encoding;
  if (gjxl_encoder_options_init(&encoding, sizeof(encoding)) != GJXL_OK) return false;
  encoding.effort = 1;
  const auto image = fixture.View();
  ResourceBudget budget(kTestEnvelope);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(6, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kInput});
    GJXLBuffer output{};
    ArmNextManagedHostAllocationFailureForTest();
    const GJXLResult result = gjxl_encode(context.get(), &image, &encoding, &output);
    const bool rejected_before_allocating = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(result == GJXL_ERROR_OUT_OF_MEMORY && rejected_before_allocating &&
        output.data == nullptr && output.size == 0 &&
        std::string_view(gjxl_get_last_error()) == "Allocation exceeds admitted resource plan",
        "C conversion bypassed its plan or lost the precise error")) return false;
  }
  job.Reset();
  if (!Empty(budget)) return false;
  std::vector<uint8_t> expected;
  if (!Ok(budget.TryReserve(kTestEnvelope, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    if (!Ok(EncodeLinearRgbVarDctCodestream(fixture.linear.const_view(), fixture.options,
        &expected))) return false;
  }
  job.Reset();
  if (!Empty(budget)) return false;
  // Enumerate all actual host-backing sites on this single-threaded CPU path.
  // The final injected site is the C array copy, after the owned codestream.
  size_t fail_at = 0;
  bool final_publication_failure = false;
  for (; fail_at < 8192; ++fail_at) {
    if (!Ok(budget.TryReserve(kTestEnvelope, &job))) return false;
    bool injected;
    {
      ResourceContextScope scope({&job, ResourceClass::kInput});
      GJXLBuffer output{};
      ArmManagedHostAllocationFailureAfterForTest(fail_at);
      const GJXLResult result = gjxl_encode(context.get(), &image, &encoding, &output);
      injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (injected) {
        final_publication_failure = std::string_view(gjxl_get_last_error()) == "C API allocation failed";
        if (!Check(result == GJXL_ERROR_OUT_OF_MEMORY && output.data == nullptr && output.size == 0,
            "C injected failure was not atomic")) return false;
      } else {
        if (!Check(result == GJXL_OK && output.data != nullptr && output.size != 0 &&
            final_publication_failure, "C final publication site was not exercised")) return false;
        const bool matches = std::ranges::equal(std::span(output.data, output.size), expected);
        gjxl_buffer_free(&output);
        if (!Check(matches, "C publication changed the equivalent C++ codestream")) return false;
      }
      if (!Check(budget.snapshot().total.backing_count == 0 &&
          budget.snapshot().total.pending_count == 0, "C failure/publication retained charges")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;
    if (!injected) break;
  }
  std::cout << "C publication failure positions: " << fail_at << '\n';
  return Check(fail_at > 3 && fail_at < 8192, "C failure enumeration did not finish");
}
}  // namespace

int main() {
  Fixture fixture;
  return fixture.Init() && CheckWorkflow(fixture) && CheckPublicDiagnosticFailures(fixture) &&
    CheckBatch(fixture) && CheckC(fixture) &&
    Check(DefaultResourceBudget().snapshot().peak_committed_bytes == 0,
          "Explicit publication reservation escaped to the default domain")
    ? EXIT_SUCCESS : EXIT_FAILURE;
}
