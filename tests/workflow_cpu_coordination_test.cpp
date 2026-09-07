// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <barrier>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

#include "codestream/batch_workflow_test.h"
#include "codestream/workflow_admission.h"
#include "codestream/workflow_internal.h"
#include "codestream/workflow_lifetime_test.h"
#include "core/cpu_execution.h"
#include "core/image_buffer.h"
#include "gjxl/execution_domain.hpp"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::thread_budget_internal;
void Require(bool good, const char* text) {
  if (!good) { std::cerr << text << '\n'; std::exit(EXIT_FAILURE); }
}
void Ok(const Status& status) {
  if (!status.ok()) { std::cerr << status.message() << '\n'; std::exit(EXIT_FAILURE); }
}
std::shared_ptr<const ExecutionDomain> Domain(size_t limit) {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({0, limit}, &domain));
  return domain;
}
void Empty(const ExecutionDomain& domain) {
  Ok(WorkflowAdmission::TrimIdle(domain));
  const auto s = domain.snapshot();
  Require(s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 && s.suspended_cpu_workers == 0 &&
          s.waiting_cpu_callers == 0 &&
          s.peak_cpu_participants > 0 && s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit &&
          s.active_reservations == 0 && s.waiting_requests == 0 &&
          s.live_capacity_bytes == 0 && s.idle_capacity_bytes == 0,
          "Public workflow leaked/exceeded CPU or memory capacity");
}
Image3FBuffer Image(Extent2D extent, size_t phase) {
  Image3FBuffer image(extent);
  for (size_t c = 0; c < 3; ++c)
    for (size_t i = 0; i < image.plane(c).size(); ++i)
      image.plane(c)[i] = 0.05f + 0.8f * ((i * (c + 3) + phase) % 127) / 127.0f;
  return image;
}
VarDctEncodingOptions Options(bool metal, size_t threads,
                              std::shared_ptr<const ExecutionDomain> domain) {
  VarDctEncodingOptions options;
  options.backend = metal ? VarDctBackendPreference::kMetal : VarDctBackendPreference::kCpu;
  options.effort = 1;
  options.cpu_thread_count = threads;
  options.execution_domain = std::move(domain);
  return options;
}

void SingleLimits(bool metal) {
  const auto image = Image({320, 272}, 0); // Cross row and codestream-group parallel thresholds.
  std::vector<uint8_t> expected;
  VarDctEncodingSummary expected_summary;
  Ok(EncodeLinearRgbVarDctCodestream(image.const_view(), Options(metal, 1, {}), &expected, &expected_summary));
  for (size_t limit : {1, 2, 4}) {
    for (size_t threads : {0, 1, 2, 8}) {
      auto domain = Domain(limit);
      auto options = Options(metal, threads, domain);
      std::vector<uint8_t> actual;
      VarDctEncodingSummary summary;
      bool reached_tail = false;
      const auto previous = SetWorkflowLifetimeObserverForTest({
        +[](bool, bool, void* p) noexcept {
          Require(HasCpuParticipation(), "CPU serialization began without a ticket");
          *static_cast<bool*>(p) = true;
        }, &reached_tail});
      if (metal) {
        Ok(EncodeLinearRgbVarDctCodestream(image.const_view(), options, &actual, &summary));
      } else {
        VarDctEncodingProfile profile;
        Ok(EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
          image.const_view(), options, nullptr, false, &actual, &summary, &profile));
        Require(profile.peak_cpu_participants > 0 && profile.peak_cpu_participants <= limit &&
                (threads == 0 || profile.peak_cpu_participants <= threads),
                "Observed per-image CPU participants exceeded a limit");
      }
      (void)SetWorkflowLifetimeObserverForTest(previous);
      Require(reached_tail && actual == expected && summary == expected_summary,
              "Domain/per-image CPU limits changed bytes or decisions");
      VarDctEncodingTiming timing;
      Ok(EncodeLinearRgbVarDctCodestreamProfiled(image.const_view(), options, &actual, &summary, &timing));
      Require(actual == expected && summary == expected_summary && timing.total_nanoseconds > 0 &&
              timing.cpu_admission_wait_nanoseconds + timing.cpu_resume_wait_nanoseconds +
                timing.cpu_blocked_nanoseconds <= timing.total_nanoseconds &&
              (!metal || timing.cpu_blocked_nanoseconds > 0), "CPU timing attribution is invalid");
      Empty(*domain);
    }
  }
}

struct CEncoder {
  std::unique_ptr<GJXLExecutionDomain, decltype(&gjxl_execution_domain_destroy)> domain{nullptr, gjxl_execution_domain_destroy};
  std::unique_ptr<GJXLContext, decltype(&gjxl_context_destroy)> context{nullptr, gjxl_context_destroy};
  GJXLEncoderOptions options{};
  std::array<uint8_t, 17 * 9 * 4> pixels{};
  GJXLImageView view{};
  explicit CEncoder(std::shared_ptr<const ExecutionDomain> shared) {
    GJXLExecutionDomain* raw_domain = nullptr;
    Ok(CreateCExecutionDomain(shared, &raw_domain)); domain.reset(raw_domain);
    GJXLContextOptions co{};
    Require(gjxl_context_options_init(&co, sizeof(co)) == GJXL_OK &&
            gjxl_encoder_options_init(&options, sizeof(options)) == GJXL_OK, "C initialization failed");
    co.backend = GJXL_BACKEND_CPU; co.num_cpu_threads = 8; co.execution_domain = domain.get();
    options.effort = 1;
    GJXLContext* raw_context = nullptr;
    Require(gjxl_context_create(&co, &raw_context) == GJXL_OK, "C context failed"); context.reset(raw_context);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = i % 4 == 3 ? 255 : (i * 7) % 251;
    view = {sizeof(view), 17, 9, GJXL_PIXEL_FORMAT_RGBA8_SRGB, pixels.data(), pixels.size(), 17 * 4};
  }
  std::vector<uint8_t> Encode() {
    GJXLBuffer output{};
    Require(gjxl_encode(context.get(), &view, &options, &output) == GJXL_OK, gjxl_get_last_error());
    std::vector<uint8_t> bytes(output.data, output.data + output.size);
    gjxl_buffer_free(&output);
    return bytes;
  }
};

void ConcurrentPublicAndBatch(bool metal, size_t limit) {
  auto domain = Domain(limit);
  const auto small = Image({33, 25}, 1), large = Image({320, 272}, 4);
  const std::array<VarDctBatchEncodingRequest, 4> requests{{
    {small.const_view(), Options(false, 8, domain)},
    {large.const_view(), Options(metal, 0, domain)},
    {large.const_view(), Options(false, 2, domain)},
    {small.const_view(), Options(metal, 1, domain)},
  }};
  std::array<std::vector<uint8_t>, 4> expected;
  for (size_t i = 0; i < requests.size(); ++i)
    Ok(EncodeLinearRgbVarDctCodestream(requests[i].linear_rgb, requests[i].options, &expected[i]));
  CEncoder c(domain);
  const auto expected_c = c.Encode();
  // The context independently retains the exact C++ domain.
  c.domain.reset();
  std::array<std::unique_ptr<VarDctBatchEncoder>, 2> drivers;
  for (auto& driver : drivers) Ok(VarDctBatchEncoder::Create(4, &driver));
  std::barrier start(6);
  std::array<std::thread, 6> callers;
  for (size_t i = 0; i < callers.size(); ++i) callers[i] = std::thread([&, i] {
    start.arrive_and_wait();
    if (i < 2) {
      Require(c.Encode() == expected_c, "Concurrent C output changed");
    } else if (i < 4) {
      const size_t index = i - 2;
      std::vector<uint8_t> bytes;
      Ok(EncodeLinearRgbVarDctCodestream(requests[index].linear_rgb, requests[index].options, &bytes));
      Require(bytes == expected[index], "Concurrent C++ output changed");
    } else {
      bool published = false;
      const auto previous = batch_publication_observer_for_testing;
      const auto previous_execution = batch_execution_observer_for_testing;
      batch_execution_observer_for_testing = {nullptr, +[](void*, size_t, bool entering) noexcept {
        if (!entering) Require(HasCpuParticipation(), "Batch image epilogue has no CPU ticket");
      }};
      batch_publication_observer_for_testing = {&published,
        +[](void* p, std::span<const VarDctBatchEncodingResult>, std::span<const OwnedEncodingResult>) noexcept {
          Require(HasCpuParticipation(), "Batch publication has no CPU ticket");
          *static_cast<bool*>(p) = true;
        }};
      std::vector<VarDctBatchEncodingResult> results;
      Ok(drivers[i - 4]->Encode(requests, &results));
      batch_publication_observer_for_testing = previous;
      batch_execution_observer_for_testing = previous_execution;
      Require(published && results.size() == requests.size(), "Batch did not publish all results");
      for (size_t j = 0; j < results.size(); ++j)
        Require(results[j].status.ok() && results[j].codestream == expected[j], "Concurrent batch output changed order/bytes");
    }
  });
  for (auto& caller : callers) caller.join();
  Empty(*domain);
  Require(domain->snapshot().peak_cpu_participants == limit, "Concurrent work did not exercise the domain cap");
}

void FailureRecovery() {
  auto domain = Domain(1);
  const auto image = Image({33, 25}, 7);
  auto options = Options(false, 0, domain);
  std::vector<uint8_t> expected;
  Ok(EncodeLinearRgbVarDctCodestream(image.const_view(), options, &expected));
  std::vector<uint8_t> bytes{3, 1, 4};
  resource_budget_internal::ArmNextManagedHostAllocationFailureForTest();
  auto status = EncodeLinearRgbVarDctCodestream(image.const_view(), options, &bytes);
  resource_budget_internal::DisarmManagedHostAllocationFailureForTest();
  Require(!status.ok() && bytes == std::vector<uint8_t>({3, 1, 4}), "Allocation failure published output");
  Empty(*domain);
  ArmNextWorkflowAdmissionCapacityForTest(1);
  status = EncodeLinearRgbVarDctCodestream(image.const_view(), options, &bytes);
  Require(status.resource_plan_exceeded() && bytes == std::vector<uint8_t>({3, 1, 4}),
          "Underplan failure lost atomic output");
  Empty(*domain);
  Ok(EncodeLinearRgbVarDctCodestream(image.const_view(), options, &bytes));
  Require(bytes == expected, "Domain did not recover after failure");
  Empty(*domain);
}

void PublicQueueTiming() {
  auto domain = Domain(1);
  const auto image = Image({33, 25}, 9);
  VarDctEncodingTiming timing;
  {
    CpuExecutionScope blocker;
    Ok(blocker.Start(domain));
    std::thread caller([&] {
      std::vector<uint8_t> bytes;
      Ok(EncodeLinearRgbVarDctCodestreamProfiled(image.const_view(), Options(false, 1, domain),
                                               &bytes, nullptr, &timing));
      Require(!bytes.empty(), "Queued public call failed to publish");
    });
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (domain->snapshot().waiting_cpu_callers != 1) {
      Require(std::chrono::steady_clock::now() < end, "Public caller never queued for CPU");
      std::this_thread::yield();
    }
    { CpuSuspension suspension; caller.join(); }
  }
  Require(timing.cpu_admission_wait_nanoseconds > 0 &&
          timing.total_nanoseconds >= timing.cpu_admission_wait_nanoseconds,
          "Profiled public call did not report its initial CPU queue");
  Empty(*domain);
}
} // namespace

int main(int argc, char** argv) {
  const bool cpu_only = argc > 1 && std::string_view(argv[1]) == "--cpu-only";
  SingleLimits(false);
  FailureRecovery();
  PublicQueueTiming();
  for (size_t limit : {1, 3}) ConcurrentPublicAndBatch(false, limit);
  if (!cpu_only) {
    Ok(EnsureProductionMetalBackendAvailable());
    SingleLimits(true);
    for (size_t limit : {1, 3}) ConcurrentPublicAndBatch(true, limit);
  }
  return EXIT_SUCCESS;
}
