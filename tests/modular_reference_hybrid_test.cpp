// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/workflow_admission_scope.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "gpu/metal/metal_backend.h"
#include "modular_reference_workflow_fixture.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using codestream_internal::CodestreamBuffer;
using test::modular_workflow_fixture::Fixture;
using test::modular_workflow_fixture::Make;
namespace reference = gjxl::test::modular_reference;
size_t files = 0, samples = 0, gpu_batches = 0, faults = 0;
bool pipeline = false;
HistogramRowBackend combined_rows = HistogramRowBackend::kRasterCpu;
ReferenceFittingBackend combined_fitter = ReferenceFittingBackend::kCpu;
bool combined_unique = false;
size_t pipeline_window = 16, participants = 4;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
void Released(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  if (s.committed_bytes() || s.total.backing_count || s.total.pending_count || s.open_reservations)
    std::cerr << "Retained bytes " << s.committed_bytes() << " idle " << s.total.idle_capacity_bytes
              << " backings " << s.total.backing_count << " pending " << s.total.pending_count
              << " reservations " << s.open_reservations << '\n';
  Check(!s.committed_bytes() && !s.total.backing_count && !s.total.pending_count &&
            !s.open_reservations,
        "Hybrid workflow leaked resources");
}
void SameDecisions(const ReferenceModularFrame &cpu, const ReferenceModularFrame &hybrid) {
  const auto &a = cpu.stats(), &b = hybrid.stats();
  Check(a.local_trees == b.local_trees && a.selection_calls == b.selection_calls &&
            a.learning_calls == b.learning_calls && a.fits == b.fits &&
            a.empty_learning_calls == b.empty_learning_calls &&
            a.scanned_pixels == b.scanned_pixels && a.selected_samples == b.selected_samples &&
            a.distinct_samples == b.distinct_samples && a.tree_nodes == b.tree_nodes,
        "Hybrid workflow changed learning work or scope");
  Check(b.feature_batches == b.token_batches && b.feature_samples == b.token_samples &&
            b.feature_samples == b.scanned_pixels &&
            b.feature_batches == b.gpu_feature_batches + b.cpu_feature_batches,
        "Hybrid workflow repeated or omitted features/tokens");
  if (b.local_trees)
    Check(b.peak_feature_samples <= (pipeline ? pipeline_window : 8) * 1024 * 1024,
          "Local workflow exceeded the aggregate feature-cohort limit");
  Check(cpu.streams().size() == hybrid.streams().size(), "Stream count changed");
  for (size_t i = 0; i < cpu.streams().size(); ++i) {
    const auto &x = cpu.streams()[i], &y = hybrid.streams()[i];
    Check(x.id == y.id && x.channel_count == y.channel_count &&
              x.metadata_channels == y.metadata_channels &&
              x.learning_extent == y.learning_extent && x.tree == y.tree &&
              x.selection.wp_mode == y.selection.wp_mode &&
              x.selection.transforms == y.selection.transforms &&
              x.learning.first_group == y.learning.first_group &&
              x.learning.stop_group == y.learning.stop_group &&
              x.learning.preparation.population == y.learning.preparation.population &&
              x.learning.preparation.quantization_samples ==
                  y.learning.preparation.quantization_samples,
          "Hybrid workflow changed stream selection, tree or sampling");
    Check(hybrid.view().streams[i].weighted_features.empty(),
          "Published frame retained feature borrows");
  }
}
void Encode(const ReferenceModularFrame &frame, const PreparedModularEntropyOptions &options,
            const ResourceBudget &budget, CodestreamBuffer *out) {
  ModularWorkflowStoragePlan plan;
  Ok(ComputePreparedModularStreamsStoragePlan(frame.view(), options, &plan));
  ResourceReservation reservation;
  Ok(budget.TryReserve(plan.working.peak_bytes, &reservation));
  ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
  Ok(EncodePreparedModularStreams(frame.view(), options, out));
}
void Case(GpuBackend &gpu, Fixture f, ReferenceWorkflowOptions options,
          WeightedFeatureBackend producer, bool multi_batch = false, bool wide_palette = false,
          bool mixed_cohort = false) {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{2} << 30, participants}, &domain));
  const auto &budget = codestream_internal::WorkflowAdmission::Budget(*domain);
  auto resources = CurrentResourceContext();
  resources.domain = domain.get();
  ResourceContextScope context(resources);
  thread_budget_internal::CpuExecutionScope execution;
  Ok(execution.Start(domain, participants));
  thread_budget_internal::EncodeScope participation(participants);
  ReferenceModularFrame cpu, hybrid;
  ModularProfile cpu_profile, gpu_profile;
  options.learning_cpu_participants = 1;
  {
    ProfileSession session(&cpu_profile);
    Ok(PrepareReferenceModularFrame(f.view(), options, budget, &cpu));
  }
  options.gpu = &gpu;
  options.sample_rows = combined_rows;
  options.use_unique_row_proof = combined_unique;
  options.defer_gpu_row_materialization = combined_rows == HistogramRowBackend::kGpu;
  options.gpu_row_diagnostics = combined_rows == HistogramRowBackend::kGpu;
  options.fitting_backend = combined_fitter;
  if (combined_fitter != ReferenceFittingBackend::kCpu) options.fitting_minimum_rows = 64;
  if (pipeline) {
    options.learning_schedule = ReferenceLearningSchedule::kPipeline;
    options.feature_window_samples = pipeline_window * 1024 * 1024;
  }
  options.learning_cpu_participants = participants;
  options.feature_producer = producer;
  {
    ProfileSession session(&gpu_profile);
    Ok(PrepareReferenceModularFrame(f.view(), options, budget, &hybrid));
  }
  SameDecisions(cpu, hybrid);
  if (combined_rows == HistogramRowBackend::kGpu && hybrid.stats().local_trees) {
    const auto &rows = hybrid.stats().rows;
    Check(rows.gpu_calls && rows.gpu_rows == hybrid.stats().selected_samples &&
              rows.retained_rows == hybrid.stats().distinct_samples &&
              rows.deferred_calls == rows.gpu_calls,
          "Combined workflow lost row ownership or preparation diagnostics");
  }
  if (combined_unique)
    Check(hybrid.stats().rows.uniqueness_checks != 0, "Combined workflow lost uniqueness control");
  if (combined_fitter == ReferenceFittingBackend::kGpu ||
      combined_fitter == ReferenceFittingBackend::kGpuResident ||
      combined_fitter == ReferenceFittingBackend::kGpuFrontier ||
      combined_fitter == ReferenceFittingBackend::kGpuTiledFrontier)
    Check(hybrid.stats().histogram_gpu_batches != 0, "Combined workflow did not exercise GPU fitting");
  Check(hybrid.stats().learning_peak_participants >= 1 &&
            hybrid.stats().learning_peak_participants <= participants &&
            (!hybrid.stats().local_trees || hybrid.stats().learning_peak_participants == participants),
        "Hybrid learning exceeded its allowance or lost local concurrency");
  Check(cpu_profile.training_work.weighted_prediction_samples == cpu.stats().scanned_pixels &&
            cpu_profile.training_work.reused_prediction_samples == 0 &&
            gpu_profile.training_work.reused_prediction_samples == hybrid.stats().scanned_pixels &&
            gpu_profile.training_work.weighted_prediction_samples == 0,
        "Native learner did not consume the prepared maps exactly once");
  if (multi_batch)
    Check(hybrid.stats().token_batches > 1 && hybrid.stats().gpu_feature_batches > 0,
          "Large automatic workflow did not exercise bounded GPU batches");
  if (pipeline && multi_batch && pipeline_window == 16)
    Check(hybrid.stats().pipeline_peak_batches > 1 && hybrid.stats().learning_waves == 1,
          "Pipeline did not retain independent owners across cohorts");
  if (wide_palette) {
    bool found = false;
    for (const auto &record : hybrid.streams())
      for (size_t c = 0; c < record.metadata_channels; ++c)
        found |= record.channels[c].descriptor.extent.width > 256;
    Check(found && hybrid.stats().cpu_feature_batches && hybrid.stats().gpu_feature_batches,
          "Wide palette did not exercise separated CPU metadata and GPU pixel batches");
  }
  if (mixed_cohort) {
    Check(hybrid.stats().local_trees && hybrid.stats().cpu_feature_batches > 1 &&
              hybrid.stats().gpu_feature_batches > 1 && hybrid.stats().learning_waves == 1 &&
              hybrid.stats().feature_cohorts == 1 &&
              hybrid.stats().peak_feature_samples == hybrid.stats().feature_samples,
          "Mixed feature producers still split the ready fitting cohort");
  }
  gpu_batches += hybrid.stats().gpu_feature_batches;
  // Moves preserve source/tree identities and the independent completed token owners.
  ReferenceModularFrame moved(std::move(hybrid));
  hybrid = std::move(moved);
  std::fill(f.bytes.begin(), f.bytes.end(), 0);
  for (auto entropy :
       {options.entropy, PreparedModularEntropyOptions{PreparedModularEntropyPolicy::kPrefix}}) {
    CodestreamBuffer a, b;
    Encode(cpu, entropy, budget, &a);
    Encode(hybrid, entropy, budget, &b);
    Check(std::ranges::equal(a.view(), b.view()),
          "Hybrid complete bytes differ from improved CPU policy");
#if GJXL_REFERENCE_WORKFLOW_ORACLE
    Check(reference::DecodeLossless(b.view()) == f.original, "Hybrid exact decode differs");
#endif
    ++files;
    samples += f.original.samples.size();
  }
  cpu = {};
  hybrid = {};
  // The capacity-reuse fitter deliberately retains idle backing in its
  // backend. Exercise explicit cache release before asserting no ownership.
  Ok(gpu.TrimPreparationCache());
  Released(budget);
}
void WidePalette(GpuBackend &gpu) {
  auto f = Make({1025, 257}, 3, 16, 1);
  for (size_t y = 0; y < f.original.height; ++y)
    for (size_t x = 0; x < f.original.width; ++x)
      for (size_t c = 0; c < 3; ++c) {
        const auto value = static_cast<uint16_t>(((x + y * 31) % 600) * (109 + c * 191));
        f.original.samples[(y * f.original.width + x) * 3 + c] = value;
        const size_t pos = y * f.stride + (x * 3 + c) * 2;
        f.bytes[pos] = value >> 8;
        f.bytes[pos + 1] = value;
      }
  ReferenceWorkflowOptions options;
  options.buffering = ReferenceBuffering::kBuffered;
  Case(gpu, std::move(f), options, WeightedFeatureBackend::kGpu, false, true);
}
Fixture MixedCohortFixture(bool alternating = true, uint32_t height = 257) {
  auto f = Make({alternating ? 2049u : 5121u, height}, 3, 16, 3);
  // Mix wide local palette metadata (CPU features) and unpaletted groups
  // (GPU features), alternating or in two long runs. All maps fit within
  // the local cohort's sample ceiling.
  for (size_t y = 0; y < f.original.height; ++y)
    for (size_t x = 0; x < f.original.width; ++x) {
      if (alternating ? (x / 256) % 2 != 0 : x < 2048)
        continue;
      for (size_t c = 0; c < 3; ++c) {
        const auto value = static_cast<uint16_t>(((x % 256 + y * 31) % 600) * (109 + c * 191));
        f.original.samples[(y * f.original.width + x) * 3 + c] = value;
        const size_t pos = y * f.stride + (x * 3 + c) * 2;
        f.bytes[pos] = value >> 8;
        f.bytes[pos + 1] = value;
      }
    }
  return f;
}
void MixedCohort(GpuBackend &gpu) {
  Case(gpu, MixedCohortFixture(), {}, WeightedFeatureBackend::kGpu, false, false, true);
}
void CohortMemoryAdmission(GpuBackend &gpu) {
  // A large ordinary run followed by a wide-palette run makes the second
  // owner's admission cost larger than one fit's scratch or the first run's
  // token output. Completing that run leaves room for the remaining owner.
  const auto f = MixedCohortFixture(false);
  ResourceBudget control(size_t{1} << 30);
  ReferenceWorkflowOptions options;
  options.learning_cpu_participants = 1;
  options.feature_cpu_participants = 1;
  ReferenceModularFrame cpu;
  Ok(PrepareReferenceModularFrame(f.view(), options, control, &cpu));
  options.gpu = &gpu;
  options.feature_producer = WeightedFeatureBackend::kGpu;
  if (pipeline) {
    options.learning_schedule = ReferenceLearningSchedule::kPipeline;
    options.feature_window_samples = 16 * 1024 * 1024;
  }
  bool flushed = false;
  // Find the narrow range in which another producer owner cannot be admitted
  // but the already prepared cohort can still fit/tokenize. The exact cutoff
  // depends on the platform's managed storage plans, not physical free RAM.
  for (size_t mib : {56, 60, 64, 68, 72, 80}) {
    ResourceBudget budget(mib << 20);
    ReferenceModularFrame hybrid;
    const auto status = PrepareReferenceModularFrame(f.view(), options, budget, &hybrid);
    if (status.ok()) {
      SameDecisions(cpu, hybrid);
      Check(budget.snapshot().peak_committed_bytes <= (mib << 20),
            "Feature cohort exceeded its managed memory limit");
      if (hybrid.stats().feature_memory_limited_cohorts || hybrid.stats().pipeline_admission_stalls) {
        if (!pipeline)
          Check(hybrid.stats().feature_cohorts > 1, "Memory pressure did not split the cohort");
        // The small budget tests preparation admission. Use the separate
        // control budget for entropy serialization of both completed frames.
        CodestreamBuffer a, b;
        Encode(cpu, options.entropy, control, &a);
        Encode(hybrid, options.entropy, control, &b);
        Check(std::ranges::equal(a.view(), b.view()), "Flushed cohort changed complete bytes");
#if GJXL_REFERENCE_WORKFLOW_ORACLE
        Check(reference::DecodeLossless(b.view()) == f.original, "Flushed cohort decode differs");
#endif
        ++files;
        samples += f.original.samples.size();
        gpu_batches += hybrid.stats().gpu_feature_batches;
        flushed = true;
      }
    } else {
      Check((status.code() == StatusCode::kOutOfMemory || status.code() == StatusCode::kUnavailable) &&
                !status.resource_plan_exceeded() && hybrid.streams().empty(),
            "Constrained feature preparation leaked a partial frame or exceeded its plan");
    }
    hybrid = {};
    Released(budget);
    if (flushed)
      break;
  }
  Check(flushed, "Feature cohort memory-pressure fallback was not exercised");
  cpu = {};
  Released(control);
}
void Failures(GpuBackend &gpu) {
  const auto f = Make({7, 3}, 3, 8, 3);
  ReferenceWorkflowOptions options;
  options.gpu = &gpu;
  if (pipeline) options.learning_schedule = ReferenceLearningSchedule::kPipeline;
  options.feature_producer = WeightedFeatureBackend::kGpu;
  ResourceBudget budget(size_t{1} << 30);
  ReferenceModularFrame frame;
  Ok(PrepareReferenceModularFrame(f.view(), options, budget, &frame));
  const auto *old_frame = frame.streams().data();
  CodestreamBuffer output;
  Ok(CodestreamBuffer::CopyFrom(std::array<uint8_t, 3>{7, 8, 9}, &output));
  const auto old_output = output.view();
  const auto *old_bytes = old_output.data();
  const size_t retained = budget.snapshot().committed_bytes();
  // Small-fixture allocation sweep covers the new owner/descriptor/handoff
  // boundaries without exercising the legacy candidate-search fault matrix.
  bool recovered = false;
  for (size_t failure = 0; failure < 512; ++failure) {
    ArmManagedHostAllocationFailureAfterForTest(failure);
    const auto s = PrepareReferenceModularFrame(f.view(), options, budget, &frame);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (s.ok()) {
      Check(pending, "Ignored an injected allocation failure");
      recovered = true;
      break;
    }
    ++faults;
    Check(s.code() == StatusCode::kOutOfMemory && !pending && frame.streams().data() == old_frame &&
              budget.snapshot().committed_bytes() == retained &&
              !budget.snapshot().total.pending_count,
          "Failed hybrid preparation changed output or leaked storage");
  }
  Check(recovered, "Bounded preparation allocation sweep did not finish");
  for (size_t fault = 0; fault < 3; ++fault) {
    if (fault == 0)
      Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, fault == 1, fault == 2));
    ReferenceWorkflowStats stats;
    stats.fits = 777;
    const auto before = budget.snapshot().committed_bytes();
    const auto s = EncodeReferenceModularImage(f.view(), options, budget, &output, &stats);
    Check(!s.ok() && output.view().data() == old_bytes &&
              std::ranges::equal(output.view(), old_output) && stats.fits == 777 &&
              budget.snapshot().committed_bytes() == before &&
              !budget.snapshot().total.pending_count,
          "Failed hybrid encode changed output/stats or leaked storage");
    ++faults;
  }
  // Capacity already used by a different stage/call must return without waiting.
  ResourceReservation busy;
  const auto remaining = (size_t{1} << 30) - budget.snapshot().committed_bytes();
  Ok(budget.TryReserve(remaining, &busy));
  Check(EncodeReferenceModularImage(f.view(), options, budget, &output).code() ==
                StatusCode::kUnavailable &&
            output.view().data() == old_bytes,
        "Busy hybrid workflow waited or published");
  busy = {};
  for (bool shared : {false, true}) {
    using namespace thread_budget_internal;
    CpuExecutionScope execution;
    Ok(execution.Start({}, 4));
    EncodeScope participation(4);
    const auto local = shared ? Make({257, 129}, 3, 8, 3) : MixedCohortFixture();
    WorkerLaunchFaultForTesting fault{shared ? WorkerLaunchSite::kModularHistogramProperties
                                            : WorkerLaunchSite::kModularTreeLearning, 1};
    WorkerLaunchFaultScopeForTesting inject(&fault);
    ReferenceWorkflowStats stats;
    stats.fits = 777;
    const auto before = budget.snapshot().committed_bytes();
    const auto status = EncodeReferenceModularImage(local.view(), options, budget, &output, &stats);
    Check(!status.ok() && fault.triggered && fault.launched_in_group == 1 &&
              output.view().data() == old_bytes && stats.fits == 777 &&
              budget.snapshot().committed_bytes() == before,
          "Parallel hybrid failure did not join before releasing feature ownership");
    ++faults;
  }
  frame = {};
  output = {};
  Released(budget);
}
void ResidentPipelineFailures(GpuBackend &gpu) {
  using namespace thread_budget_internal;
  CpuExecutionScope execution;
  Ok(execution.Start({}, 4));
  EncodeScope participation(4);
  // Nine groups select local trees; a smaller group count would exercise a
  // shared fit and could exceed resident per-fit admission before the test.
  const auto f = Make({2049, 17}, 3, 8, 3);
  ReferenceWorkflowOptions options;
  options.gpu = &gpu;
  options.fitting_backend = ReferenceFittingBackend::kGpuTiledFrontier;
  options.learning_schedule = ReferenceLearningSchedule::kPipeline;
  options.learning_cpu_participants = 4;
  options.feature_window_samples = 16 * 1024 * 1024;
  // No preceding GPU submission: CPU features/rows make the injected device
  // fault hit resident fitting after the CPU preparation workers have run.
  options.feature_producer = WeightedFeatureBackend::kCpu;
  options.sample_rows = HistogramRowBackend::kSelectedCpu;
  ResourceBudget budget(size_t{1} << 30);
  ReferenceModularFrame frame;
  Ok(PrepareReferenceModularFrame(f.view(), options, budget, &frame));
  Check(frame.stats().local_trees && frame.stats().histogram_resident_fits == frame.stats().fits &&
            frame.stats().fits > 1 && !frame.stats().histogram_cpu_nodes,
        "Resident pipeline failure fixture did not use local GPU fits");
  const auto *old = frame.streams().data();
  const size_t retained = budget.snapshot().committed_bytes();
  for (bool submission : {false, true}) {
    const auto before = gpu.stats().committed_submissions;
    Ok(ArmNextMetalSubmissionFailureForTest(gpu, submission, !submission));
    const auto s = PrepareReferenceModularFrame(f.view(), options, budget, &frame);
    Check(!s.ok() && frame.streams().data() == old &&
              budget.snapshot().committed_bytes() == retained &&
              !budget.snapshot().total.pending_count &&
              gpu.stats().committed_submissions == before + size_t(!submission),
          "Resident pipeline GPU failure published or retained partial work");
    ++faults;
  }
  {
    WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularTreeLearning, 1};
    WorkerLaunchFaultScopeForTesting inject(&fault);
    const auto s = PrepareReferenceModularFrame(f.view(), options, budget, &frame);
    Check(!s.ok() && fault.triggered && fault.launched_in_group == 1 &&
              frame.streams().data() == old && budget.snapshot().committed_bytes() == retained,
          "Resident pipeline launch failure did not join and preserve output");
    ++faults;
  }
  {
    ResourceBudget small(size_t{32} << 20);
    const auto s = PrepareReferenceModularFrame(f.view(), options, small, &frame);
    Check((s.code() == StatusCode::kUnavailable || s.code() == StatusCode::kOutOfMemory) &&
              !s.resource_plan_exceeded() && frame.streams().data() == old &&
              small.snapshot().peak_committed_bytes <= (size_t{32} << 20),
          "Resident pipeline exceeded admission or published a partial frame");
    Released(small);
    ++faults;
  }
  {
    // Six groups select one shared tree with more than 131072 distinct rows.
    // Reject the resident contract violation atomically, without truncation.
    const auto unsupported = Make({513, 257}, 3, 8, 3);
    const auto s = PrepareReferenceModularFrame(unsupported.view(), options, budget, &frame);
    if (s.code() != StatusCode::kUnsupported || frame.streams().data() != old ||
        budget.snapshot().committed_bytes() != retained)
      std::cerr << "Eligibility status: " << s.message() << "; retained "
                << budget.snapshot().committed_bytes() << " versus " << retained
                << "; idle " << budget.snapshot().total.idle_capacity_bytes << '\n';
    Check(s.code() == StatusCode::kUnsupported && frame.streams().data() == old &&
              budget.snapshot().committed_bytes() == retained,
          "Resident pipeline changed eligibility or retained rejected samples");
    ++faults;
  }
  Ok(PrepareReferenceModularFrame(f.view(), options, budget, &frame));
  frame = {};
  Released(budget);
}
} // namespace
int main(int argc, char **argv) try {
  const bool focused = argc == 2 && std::string_view(argv[1]) == "--pipeline-focused";
  pipeline = focused || (argc == 2 && std::string_view(argv[1]) == "--pipeline");
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
  if (argc == 2 && std::string_view(argv[1]) == "--tiled-pipeline-failures") {
    ResidentPipelineFailures(*gpu);
    std::cout << "Resident pipeline failure checks passed: " << faults << '\n';
    return 0;
  } else if (argc == 2 && std::string_view(argv[1]) == "--tiled-pipeline") {
    pipeline = true;
    combined_fitter = ReferenceFittingBackend::kGpuTiledFrontier;
    combined_rows = HistogramRowBackend::kGpu;
    combined_unique = true;
    ReferenceWorkflowOptions all;
    all.predictors = ReferencePredictors::kAll;
    for (size_t p : {1, 2, 4}) {
      participants = p;
      for (size_t window : {8, 16}) {
        pipeline_window = window;
        Case(*gpu, Make({2049, 17}, 4, 16, 1), all, WeightedFeatureBackend::kGpu);
      }
    }
    pipeline_window = 16;
    Case(*gpu, Make({2049, 1537}, 3, 8, 3), {}, WeightedFeatureBackend::kAuto, true);
    Case(*gpu, MixedCohortFixture(true, 129), {}, WeightedFeatureBackend::kGpu, false, false, true);
    ResidentPipelineFailures(*gpu);
  } else if (argc == 2 && (std::string_view(argv[1]) == "--resident" || std::string_view(argv[1]) == "--frontier" || std::string_view(argv[1]) == "--tiled")) {
    combined_fitter = std::string_view(argv[1]) == "--tiled" ? ReferenceFittingBackend::kGpuTiledFrontier : std::string_view(argv[1]) == "--frontier" ? ReferenceFittingBackend::kGpuFrontier : ReferenceFittingBackend::kGpuResident;
    Case(*gpu, Make({17, 9}, 1, 8, 1), {}, WeightedFeatureBackend::kGpu);
    Case(*gpu, Make({257, 257}, 3, 8, 1), {}, WeightedFeatureBackend::kGpu);
    ReferenceWorkflowOptions all;
    all.predictors = ReferencePredictors::kAll;
    combined_unique = true;
    combined_rows = HistogramRowBackend::kGpu;
    Case(*gpu, Make({2049, 17}, 4, 16, 1), all, WeightedFeatureBackend::kGpu);
  } else if (argc == 2 && std::string_view(argv[1]) == "--combined") {
    combined_unique = true;
    ReferenceWorkflowOptions all;
    all.predictors = ReferencePredictors::kAll;
    for (auto rows : {HistogramRowBackend::kSelectedCpu, HistogramRowBackend::kGpu}) {
      combined_rows = rows;
      for (auto fitter : {ReferenceFittingBackend::kCpu,
                          ReferenceFittingBackend::kResumableCpu, ReferenceFittingBackend::kGpu}) {
        combined_fitter = fitter;
        Case(*gpu, Make({2049, 17}, 4, 16, 1), all, WeightedFeatureBackend::kGpu);
      }
      combined_fitter = ReferenceFittingBackend::kCpu;
      pipeline = true;
      Case(*gpu, Make({2049, 17}, 4, 16, 1), all, WeightedFeatureBackend::kGpu);
      MixedCohort(*gpu);
      pipeline = false;
      MixedCohort(*gpu);
    }
    combined_fitter = ReferenceFittingBackend::kGpu;
    MixedCohort(*gpu);
    combined_fitter = ReferenceFittingBackend::kCpu;
    pipeline = true;
    Case(*gpu, Make({2049, 1537}, 3, 8, 3), {}, WeightedFeatureBackend::kAuto, true);
    auto unsupported = all;
    unsupported.learning_schedule = ReferenceLearningSchedule::kPipeline;
    unsupported.fitting_backend = ReferenceFittingBackend::kGpu;
    ReferenceModularFrame frame;
    ResourceBudget budget;
    Check(PrepareReferenceModularFrame({}, unsupported, budget, &frame).code() ==
              StatusCode::kUnsupported, "Pipeline silently accepted GPU fitting");
  } else if (focused) {
    ReferenceWorkflowOptions all;
    all.predictors = ReferencePredictors::kAll;
    for (size_t p : {1, 2, 4}) {
      participants = p;
      for (size_t window : {8, 16}) {
        pipeline_window = window;
        Case(*gpu, Make({2049, 17}, 4, 16, 1), all, WeightedFeatureBackend::kGpu);
      }
    }
    pipeline_window = 16;
    Case(*gpu, Make({2049, 1537}, 3, 8, 3), {}, WeightedFeatureBackend::kAuto, true);
    MixedCohort(*gpu);
    CohortMemoryAdmission(*gpu);
    Failures(*gpu);
  } else if (argc == 2 && std::string_view(argv[1]) == "--palette") {
    WidePalette(*gpu);
    MixedCohort(*gpu);
    CohortMemoryAdmission(*gpu);
  } else {
    for (size_t channels : {1, 3, 4})
      for (unsigned bits : {8, 16})
        for (auto extent : {Extent2D{17, 9}, {2048, 1}, {2049, 1}, {257, 257}}) {
          const bool wide = extent.width > 256;
          const auto order =
              bits == 16 && !wide ? SampleByteOrder::kLittleEndian : SampleByteOrder::kBigEndian;
          Case(*gpu, Make(extent, channels, bits, wide ? 3 : 1, order), {},
               WeightedFeatureBackend::kGpu);
        }
    ReferenceWorkflowOptions buffered;
    buffered.buffering = ReferenceBuffering::kBuffered;
    buffered.predictors = ReferencePredictors::kAll;
    Case(*gpu, Make({2049, 1}, 3, 16, 1), buffered, WeightedFeatureBackend::kCpu);
    auto local_all = buffered;
    local_all.buffering = ReferenceBuffering::kAutomatic;
    Case(*gpu, Make({2049, 17}, 4, 16, 1), local_all, WeightedFeatureBackend::kGpu);
    Case(*gpu, Make({257, 257}, 4, 8, 1), {}, WeightedFeatureBackend::kAuto);
    Case(*gpu, Make({2049, 1537}, 3, 8, 3), {}, WeightedFeatureBackend::kAuto, true);
    Failures(*gpu);
    if (pipeline) {
      MixedCohort(*gpu);
      CohortMemoryAdmission(*gpu);
    }
  }
  Check(gpu_batches != 0, "No GPU features qualified");
  std::cout << "{\"files\":" << files << ",\"decoded_samples\":" << samples
            << ",\"gpu_feature_batches\":" << gpu_batches << ",\"faults\":" << faults
#if GJXL_REFERENCE_WORKFLOW_ORACLE
            << ",\"exact_decode\":true"
#else
            << ",\"exact_decode\":false"
#endif
            << "}\n";
  return 0;
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n';
  return 1;
}
