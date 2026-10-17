// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_tree_parallel.h"
#include "codec/modular/histogram_tree_profile.h"
#include "codec/modular/profile.h"
#include "codestream/modular/histogram_fit_queue.h"
#include "codestream/modular/reference_workflow_internal.h"
#include "gpu/ops/modular_resident_histogram_fitting.h"
#include <algorithm>
#include <optional>

namespace gjxl::modular_internal::reference_workflow_internal {
namespace {
using namespace resource_budget_internal;
using namespace histogram_gpu_internal;
struct Job {
  ResourceReservation reservation;
  HistogramFitRecord fit_record;
  PreparedHistogramSamples samples;
  HistogramTreeFitter fitter;
  ReferenceWorkflowStats stats;
  Status status;
  size_t gpu_fit = SIZE_MAX, result_first = 0, computed = 0, reused = 0;
  std::array<HistogramPropertyResult, kPropertyCount> gpu_results;
  bool done = false, has_gpu_result = false, force_cpu = false;
};
Status Invalid() {
  return Status::InvalidArgument("Invalid histogram fitting batch");
}
bool AdmissionFailure(const Status &s) {
  return s.code() == StatusCode::kUnavailable ||
         s.code() == StatusCode::kOutOfMemory;
}
HistogramSamplingOptions RequestSampling(const HistogramFitRequest &request,
                                         HistogramSamplingOptions sampling) {
  if (request.row_source) sampling.row_source = request.row_source;
  if (sampling.row_backend == HistogramRowBackend::kGpu && !sampling.row_source)
    sampling.row_backend = HistogramRowBackend::kSelectedCpu;
  return sampling;
}
Status JobBound(const HistogramFitRequest &request,
                HistogramSamplingOptions sampling, size_t *bytes) {
  HostStorageBound preparation, fitting, extra;
  if (auto s = ComputeHistogramSamplePreparationStorageBound(
          request.images, RequestSampling(request, sampling), &preparation);
      !s.ok())
    return s;
  size_t rows = 0;
  for (const auto &image : request.images)
    for (const auto &channel : image.channels) {
      size_t area;
      if (!channel.descriptor.extent.try_area(&area) || area > SIZE_MAX - rows)
        return Invalid();
      rows += area;
    }
  rows = std::min(rows, sampling.maximum_samples);
  if (auto s = ComputeHistogramTreeWorkspaceStorageBound(
          sampling.predictor_count, sampling.maximum_property_values + 1,
          request.maximum_leaves, kMaximumTreeDepth, &fitting);
      !s.ok())
    return s;
  if (auto s = ComputeHistogramTreeFitterExtraStorageBound(rows, &extra);
      !s.ok())
    return s;
  // The whole-job envelope protects preparation, paused CPU state, row order
  // and tree compaction. GPU columns/evaluation have separate reservations.
  if (!preparation.Add(fitting) || !preparation.Add(extra) ||
      !preparation.AddVector<DecisionNode>(2 * request.maximum_leaves - 1,
                                           VectorCapacityPolicy::kFreshExact))
    return Invalid();
  *bytes = preparation.peak_bytes;
  return Status::Ok();
}
Status Finish(Job &job, ReferenceModularStream &record) {
  if (auto s = job.fitter.Publish(&record.tree); !s.ok())
    return s;
  job.samples = {};
  if (record.tree.nodes.size() > kInlineTreeNodes &&
      record.tree.nodes.size() != record.tree.size)
    record.tree.nodes.resize(record.tree.size);
  record.learning.tree_nodes = record.tree.size;
  ++job.stats.fits;
  job.stats.tree_nodes += record.tree.size;
  job.done = true;
  return Status::Ok();
}
} // namespace

Status FitResidentHistogramBatch(std::span<const ResidentHistogramRequest> requests,
                                 const ReferenceWorkflowOptions &options,
                                 const ResourceBudget &budget,
                                 ReferenceWorkflowStats *stats,
                                 ResidentHistogramResult *out, bool *admission_limited) {
  if (admission_limited) *admission_limited = false;
  auto *op = options.gpu ? QueryGpuModularHistogramFitting(*options.gpu) : nullptr;
  if (!op || !out || !stats) return Invalid();
  ResidentHistogramInput input{requests};
  input.maximum_active_nodes = options.resident_active_nodes;
  input.maximum_scratch_bytes = options.resident_scratch_bytes;
  if (options.fitting_backend == ReferenceFittingBackend::kGpuFrontier)
    input.executor = ResidentHistogramExecutor::kFrontier;
  else if (options.fitting_backend == ReferenceFittingBackend::kGpuTiledFrontier)
    input.executor = ResidentHistogramExecutor::kTiledFrontier;
  else if (options.fitting_backend != ReferenceFittingBackend::kGpuResident)
    return Invalid();
  ResidentHistogramPlan plan;
  const auto limits = op->histogram_fitting_limits();
  if (auto s = ComputeResidentHistogramPlan(input, &plan, &limits); !s.ok()) return s;
  ResourceReservation reservation;
  if (auto s = budget.TryReserve(plan.working.peak_bytes, &reservation); !s.ok()) {
    if (admission_limited) *admission_limited = AdmissionFailure(s);
    return s;
  }
  ResidentHistogramResult result;
  {
    auto resources = CurrentResourceContext(); resources.reservation = &reservation;
    ResourceContextScope scope(resources);
    Seconds timer(stats->histogram_resident_seconds);
    if (auto s = op->FitHistogramTrees(input, &result); !s.ok()) return s;
  }
  stats->histogram_resident_fits += requests.size();
  stats->histogram_resident_gpu_nanoseconds += result.gpu_nanoseconds;
  stats->histogram_resident_submissions += result.submissions;
  stats->histogram_frontier_rounds += result.frontier_rounds;
  stats->histogram_frontier_maximum_active = std::max(stats->histogram_frontier_maximum_active, result.maximum_active_nodes);
  stats->histogram_frontier_maximum_same_fit = std::max(stats->histogram_frontier_maximum_same_fit, result.maximum_same_fit_nodes);
  ++stats->histogram_gpu_preparations;
  ++stats->histogram_gpu_batches;
  stats->histogram_gpu_nodes += result.visited_nodes;
  stats->histogram_prepared_bytes += plan.prepared.storage_bytes;
  stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, plan.working.peak_bytes);
  *out = std::move(result);
  return Status::Ok();
}

Status TryLearnHistogramBatch(std::span<const HistogramFitRequest> requests,
                              HistogramSamplingOptions sampling,
                              size_t participants,
                              const ReferenceWorkflowOptions &options,
                              const ResourceBudget &budget,
                              ReferenceWorkflowStats *stats, bool *handled,
                              bool measure_learning) {
  using namespace thread_budget_internal;
  *handled = false;
  if (options.fitting_backend == ReferenceFittingBackend::kCpu ||
      requests.empty())
    return Status::Ok();
  if (requests.size() > kMaximumHistogramFittingFits || !participants)
    return Invalid();
  const bool tiled = options.fitting_backend == ReferenceFittingBackend::kGpuTiledFrontier;
  const bool frontier = tiled || options.fitting_backend == ReferenceFittingBackend::kGpuFrontier;
  const bool resident = options.fitting_backend == ReferenceFittingBackend::kGpuResident || frontier;
  if (resident && (active_histogram_fitting_profile || options.fitting_gpu_profile))
    return Status::Unsupported("Resident fitting has complete-call diagnostics only");
  const bool gpu_requested =
      options.fitting_backend != ReferenceFittingBackend::kResumableCpu;
  if (options.fitting_backend == ReferenceFittingBackend::kAuto &&
      (!stats->local_trees || requests.size() < 4 || !options.gpu))
    return Status::Ok();
  auto *op =
      options.gpu ? QueryGpuModularHistogramFitting(*options.gpu) : nullptr;
  if (options.fitting_backend == ReferenceFittingBackend::kAuto && !op)
    return Status::Ok();
  if (gpu_requested && !op)
    return Status::Unsupported(
        "Native histogram fitting capability is unavailable");
  auto *const parent_profile = active_modular_profile;
  auto *const parent_fit_profile = active_histogram_fitting_profile;
  HostStorageBound metadata;
  if (!metadata.AddVector<Job>(requests.size(),
                               VectorCapacityPolicy::kFreshExact) ||
      !metadata.AddVector<std::thread>(participants - 1,
                                       VectorCapacityPolicy::kFreshExact) ||
      (parent_fit_profile &&
       !metadata.AddVector<HistogramFittingProfile>(
           requests.size(), VectorCapacityPolicy::kFreshExact)))
    return Invalid();
  ResourceReservation metadata_reservation;
  if (auto s = budget.TryReserve(metadata.peak_bytes, &metadata_reservation);
      !s.ok()) {
    if (!AdmissionFailure(s))
      return s;
    ++stats->histogram_admission_fallbacks;
    return Status::Ok();
  }
  auto context = CurrentResourceContext();
  context.reservation = &metadata_reservation;
  context.resource_class = ResourceClass::kPreparation;
  ResourceContextScope metadata_scope(context);
  // Profiles outlive jobs, including error-path fitter destruction.
  ManagedVector<HistogramFittingProfile> profiles(
      parent_fit_profile ? requests.size() : 0);
  ManagedVector<Job> jobs(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    size_t bytes;
    if (!requests[i].record || !requests[i].maximum_leaves)
      return Invalid();
    if (auto s = JobBound(requests[i], sampling, &bytes); !s.ok())
      return s;
    if (auto s = budget.TryReserve(bytes, &jobs[i].reservation); !s.ok()) {
      if (!AdmissionFailure(s))
        return s;
      ++stats->histogram_admission_fallbacks;
      return Status::Ok();
    }
    jobs[i].stats.maximum_stage_bytes = bytes;
  }
  *handled = true;
  ProfileScope training(ProfileStage::kTraining);
  double unused_elapsed = 0;
  Seconds learning(measure_learning ? stats->learning_seconds : unused_elapsed);
  stats->maximum_stage_bytes =
      std::max(stats->maximum_stage_bytes, metadata.peak_bytes);
  std::optional<histogram_parallel_internal::PropertyTeam> team;
  team.emplace(participants, WorkerLaunchSite::kModularTreeLearning);
  if (auto s = team->Start(); !s.ok())
    return s;
  ++stats->learning_waves;
  stats->learning_peak_participants =
      std::max(stats->learning_peak_participants, team->participants());
  const auto run = [&](auto &&function) -> Status {
    team->Run(jobs.size(), [&](size_t i, size_t worker) noexcept {
      auto &job = jobs[i];
      if (!job.status.ok() || job.done)
        return;
      try {
        auto resources = CurrentResourceContext();
        resources.reservation = &job.reservation;
        ResourceContextScope scope(resources);
        job.status = function(i, worker);
      } catch (const ManagedAllocationFailure &e) {
        job.status = e.status();
      } catch (const std::bad_alloc &) {
        job.status =
            Status::OutOfMemory("Batched histogram worker allocation failed");
      } catch (const std::length_error &) {
        job.status = Invalid();
      } catch (...) {
        job.status = Status::Internal("Batched histogram worker failed");
      }
      if (job.done)
        job.reservation.Reset();
    });
    for (const auto &job : jobs)
      if (!job.status.ok())
        return job.status;
    return Status::Ok();
  };
  if (auto s = run([&](size_t i, size_t) -> Status {
        auto &job = jobs[i];
        auto &record = *requests[i].record;
        ModularProfile profile;
        ProfileSession session(parent_profile ? &profile : nullptr);
        {
          Seconds preparation(job.stats.sample_preparation_seconds);
          if (auto s = PrepareHistogramSamples(requests[i].images, &job.samples,
                                               RequestSampling(requests[i], sampling));
              !s.ok())
            return s;
        }
        job.computed = profile.training_work.weighted_prediction_samples;
        job.reused = profile.training_work.reused_prediction_samples;
        const auto &prepared = job.samples.stats();
        record.learning = {0, requests[i].images.size(), prepared,
                           job.samples.samples().weights.size(), 1};
        job.stats.rows = prepared.rows;
        job.stats.learning_calls = 1;
        job.stats.scanned_pixels = prepared.scanned_pixels;
        job.stats.selected_samples = prepared.selected_samples;
        job.stats.distinct_samples = job.samples.samples().weights.size();
        if (!job.samples.samples().total_samples) {
          record.tree.nodes[0].predictor = sampling.predictors[0];
          job.stats.empty_learning_calls = 1;
          job.stats.tree_nodes = 1;
          job.samples = {};
          job.done = true;
          return Status::Ok();
        }
        if (resident) return Status::Ok();
        Seconds fitting(job.stats.fitting_seconds);
        auto *profile_sink = parent_fit_profile ? &profiles[i] : nullptr;
        if (profile_sink)
          profile_sink->records = {&job.fit_record, 1};
        return InitializePreparedHistogramTreeFitter(
            job.samples, &job.fitter, requests[i].maximum_leaves,
            kMaximumTreeDepth, profile_sink);
      });
      !s.ok())
    return s;

  const auto merge_stats = [&] {
    for (size_t i = 0; i < jobs.size(); ++i) {
      const auto &job = jobs[i];
      for (auto member : {&ReferenceWorkflowStats::learning_calls,
                          &ReferenceWorkflowStats::fits,
                          &ReferenceWorkflowStats::empty_learning_calls,
                          &ReferenceWorkflowStats::scanned_pixels,
                          &ReferenceWorkflowStats::selected_samples,
                          &ReferenceWorkflowStats::distinct_samples,
                          &ReferenceWorkflowStats::tree_nodes,
                          &ReferenceWorkflowStats::histogram_cpu_nodes})
        stats->*member += job.stats.*member;
      stats->maximum_stage_bytes =
          std::max(stats->maximum_stage_bytes, job.stats.maximum_stage_bytes);
      MergeHistogramRowProfile(stats->rows, job.stats.rows);
      stats->sample_preparation_seconds += job.stats.sample_preparation_seconds;
      stats->fitting_seconds += job.stats.fitting_seconds;
      if (parent_profile) {
        parent_profile->training_work.weighted_prediction_samples += job.computed;
        parent_profile->training_work.reused_prediction_samples += job.reused;
      }
      if (parent_fit_profile) {
        profiles[i].records = {};
        profiles[i].records_written = 0;
        MergeHistogramFittingProfile(*parent_fit_profile, profiles[i]);
        if (job.stats.fits) {
          if (parent_fit_profile->records_written <
              parent_fit_profile->records.size())
            parent_fit_profile->records[parent_fit_profile->records_written++] =
                job.fit_record;
          else
            ++parent_fit_profile->records_dropped;
        }
      }
    }
  };
  if (resident) {
    // Initial integration keeps coarse cohorts. Every nonempty tree, including
    // its small tails, is completed by one resident GPU operation.
    for (size_t first = 0; first < jobs.size(); first += kMaximumResidentHistogramFits) {
      std::array<ResidentHistogramRequest, kMaximumResidentHistogramFits> batch;
      std::array<size_t, kMaximumResidentHistogramFits> indices;
      size_t count = 0;
      for (size_t i = first; i < std::min(jobs.size(), first + kMaximumResidentHistogramFits); ++i) {
        if (jobs[i].done) continue;
        auto learning = jobs[i].samples.tree_options();
        learning.maximum_leaves = requests[i].maximum_leaves;
        batch[count] = {{static_cast<uint32_t>(i), &jobs[i].samples.samples()}, learning};
        indices[count++] = i;
      }
      if (!count) continue;
      ResidentHistogramResult result;
      if (auto s = FitResidentHistogramBatch(std::span(batch).first(count), options,
              budget, stats, &result); !s.ok()) return s;
      for (size_t j = 0; j < count; ++j) {
        const size_t i = indices[j];
        auto &job = jobs[i];
        auto &record = *requests[i].record;
        record.tree = std::move(result.trees[j]);
        record.learning.tree_nodes = record.tree.size;
        ++job.stats.fits;
        job.stats.tree_nodes += record.tree.size;
        job.samples = {};
        job.done = true;
        job.reservation.Reset();
      }
    }
    merge_stats();
    return Status::Ok();
  }

  std::unique_ptr<PreparedGpuHistogramRows> prepared;
  HistogramFittingDeviceLimits limits{};
  if (gpu_requested) {
    limits = op->histogram_fitting_limits();
    std::array<HistogramFittingSource, kMaximumHistogramFittingFits> sources;
    std::array<HistogramFittingShape, kMaximumHistogramFittingFits> shapes;
    size_t count = 0, rows = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
      const auto &s = jobs[i].samples.samples();
      if (jobs[i].done || s.weights.size() < options.fitting_minimum_rows ||
          count >= limits.maximum_fits ||
          s.weights.size() > limits.maximum_rows - rows)
        continue;
      sources[count] = {static_cast<uint32_t>(requests[i].record->id), &s};
      if (auto status =
              ResolveHistogramFittingSource(sources[count], &shapes[count]);
          !status.ok()) {
        if (status.code() == StatusCode::kUnsupported)
          continue;
        return status;
      }
      jobs[i].gpu_fit = count++;
      rows += s.weights.size();
    }
    if (count) {
      HistogramFittingPreparationPlan plan;
      if (auto s = ComputeHistogramFittingPreparationPlan(
              std::span(shapes).first(count), &plan, &limits);
          !s.ok())
        return s;
      bool started = false;
      Seconds timer(stats->histogram_preparation_seconds);
      const auto status =
          Stage(budget, plan.preparation.peak_bytes, stats, [&] {
            started = true;
            return op->PrepareHistogramRows(std::span(sources).first(count),
                                            &prepared);
          });
      if (!status.ok()) {
        if (started || !AdmissionFailure(status))
          return status;
        ++stats->histogram_admission_fallbacks;
      } else {
        if (!prepared || prepared->backend_id() != options.gpu->id() ||
            !std::ranges::equal(prepared->shapes(),
                                std::span(shapes).first(count)))
          return Status::Internal(
              "Histogram preparation returned an invalid owner");
        ++stats->histogram_gpu_preparations;
        stats->histogram_prepared_bytes += plan.storage_bytes;
      }
    }
  }
  const auto process = [&](size_t i) noexcept -> HistogramFitQueue::Result {
    auto &job = jobs[i];
    if (job.done)
      return HistogramFitQueue::Result::kDone;
    try {
      auto resources = CurrentResourceContext();
      resources.reservation = &job.reservation;
      ResourceContextScope scope(resources);
      Seconds fitting(job.stats.fitting_seconds);
      job.status = [&]() -> Status {
        if (job.has_gpu_result) {
          if (auto status = ApplyHistogramNodeEvaluation(
                  job.fitter, static_cast<uint32_t>(requests[i].record->id),
                  std::span(job.gpu_results)
                      .first(job.samples.samples().property_count));
              !status.ok())
            return status;
          job.has_gpu_result = false;
        }
        std::array<HistogramPropertyEvaluation, kPropertyCount> results;
        for (;;) {
          if (!job.fitter.ready())
            if (auto status = job.fitter.Advance(); !status.ok())
              return status;
          if (job.fitter.complete())
            return Finish(job, *requests[i].record);
          if (!job.force_cpu && prepared && job.gpu_fit != SIZE_MAX &&
              job.fitter.node().original_rows.size() >=
                  options.fitting_minimum_rows) {
            auto node = HistogramFittingNodeView(0, job.fitter.node());
            const auto shape = prepared->shapes()[job.gpu_fit];
            HistogramFittingEvaluationPlan plan;
            const auto status = ComputeHistogramFittingEvaluationPlan(
                {&shape, 1}, {&node.shape, 1}, options.fitting_execution, &plan,
                &limits);
            if (status.ok())
              return Status::Ok();
            if (status.code() != StatusCode::kUnsupported)
              return status;
          }
          job.force_cpu = false;
          auto output =
              std::span(results).first(job.samples.samples().property_count);
          if (auto status = job.fitter.EvaluateCpu(output); !status.ok())
            return status;
          if (auto status = job.fitter.Apply(output); !status.ok())
            return status;
          ++job.stats.histogram_cpu_nodes;
        }
      }();
    } catch (const ManagedAllocationFailure &e) {
      job.status = e.status();
    } catch (const std::bad_alloc &) {
      job.status =
          Status::OutOfMemory("Histogram fit worker allocation failed");
    } catch (const std::length_error &) {
      job.status = Invalid();
    } catch (...) {
      job.status = Status::Internal("Histogram fit worker failed");
    }
    if (!job.status.ok())
      return HistogramFitQueue::Result::kFailed;
    if (job.done) {
      job.reservation.Reset();
      return HistogramFitQueue::Result::kDone;
    }
    return HistogramFitQueue::Result::kGpu;
  };
  if (!prepared) {
    if (auto s = run([&](size_t i, size_t) {
          (void)process(i);
          return jobs[i].status;
        });
        !s.ok())
      return s;
  } else {
    // Release the preparation team's grants before acquiring the fitting team.
    // The coordinator submits ready nodes while other workers drain CPU tails.
    team.reset();
    auto *const diagnostic = options.fitting_gpu_profile;
    const auto profile_mode = diagnostic ? diagnostic->mode
                                        : HistogramFittingProfileMode::kDisabled;
    const bool trace_wait = diagnostic && diagnostic->trace_wait &&
                           profile_mode != HistogramFittingProfileMode::kDisabled;
    HistogramFitQueue queue(participants, options.fitting_ready_target,
                            options.fitting_flush_microseconds);
    if (auto s = queue.Start(jobs.size(), process); !s.ok())
      return s;
    stats->learning_peak_participants =
        std::max(stats->learning_peak_participants, queue.participants());
    std::array<size_t, kMaximumHistogramFittingFits> indices;
    std::array<HistogramFittingNodeInput, kMaximumHistogramFittingFits> nodes;
    std::array<HistogramFittingNodeShape, kMaximumHistogramFittingFits> shapes;
    size_t count = 0;
    HistogramFitQueue::Trace queue_trace;
    while (queue.Next(indices, &count, trace_wait ? &queue_trace : nullptr) && count) {
      for (size_t j = 0; j < count; ++j) {
        const auto &job = jobs[indices[j]];
        nodes[j] = HistogramFittingNodeView(job.gpu_fit, job.fitter.node());
        shapes[j] = nodes[j].shape;
      }
      // Heterogeneous maxima can exceed a scratch slot even if each node fits.
      for (size_t first = 0; first < count;) {
        size_t length = count - first;
        HistogramFittingEvaluationPlan plan;
        for (;;) {
          const auto status = ComputeHistogramFittingEvaluationPlan(
              prepared->shapes(), std::span(shapes).subspan(first, length),
              options.fitting_execution, &plan, &limits);
          if (status.ok())
            break;
          if (status.code() != StatusCode::kUnsupported || length == 1)
            return status;
          length = (length + 1) / 2;
        }
        HostStorageBound profile_bound;
        if (auto s = op->HistogramFittingProfileStorageBound(plan, profile_mode,
                                                             &profile_bound);
            !s.ok())
          return s;
        auto capacities = op->HistogramFittingBufferCapacities(
            plan, options.fitting_execution);
        HostStorageBound evaluation_bound;
        if (auto s = ComputeHistogramFittingCapacityBound(plan, capacities,
                                                          &evaluation_bound); !s.ok())
          return s;
        if (!evaluation_bound.Add(profile_bound))
          return Status::OutOfMemory("Histogram diagnostic admission overflow");
        ResourceReservation evaluation_reservation;
        auto admitted = budget.TryReserve(evaluation_bound.peak_bytes,
                                          &evaluation_reservation);
        // A larger cache entry is optional. If its full capacity cannot be
        // admitted, retry the original logical envelope before CPU fallback.
        if (!admitted.ok() && AdmissionFailure(admitted) &&
            options.fitting_execution.reuse_larger_buffers) {
          ++stats->histogram_capacity_retries;
          capacities = {};
          evaluation_bound = plan.evaluation;
          if (!evaluation_bound.Add(profile_bound))
            return Status::OutOfMemory("Histogram diagnostic admission overflow");
          admitted = budget.TryReserve(evaluation_bound.peak_bytes,
                                        &evaluation_reservation);
        }
        if (!admitted.ok() && !AdmissionFailure(admitted))
          return admitted;
        std::unique_ptr<PreparedGpuHistogramEvaluation> output;
        HistogramFittingBatchProfile batch_profile;
        std::chrono::steady_clock::time_point evaluated;
        size_t result_first = 0;
        for (size_t j = first; j < first + length; ++j) {
          auto &job = jobs[indices[j]];
          job.result_first = result_first;
          result_first += job.samples.samples().property_count;
        }
        if (admitted.ok()) {
          stats->maximum_stage_bytes =
              std::max(stats->maximum_stage_bytes, evaluation_bound.peak_bytes);
          auto resources = CurrentResourceContext();
          resources.reservation = &evaluation_reservation;
          ResourceContextScope scope(resources);
          Seconds timer(stats->histogram_evaluation_seconds);
          if (auto status = op->EvaluateHistogramNodes(
                  {prepared.get(), std::span(nodes).subspan(first, length),
                   options.fitting_execution, profile_mode, &batch_profile, trace_wait,
                   capacities},
                  &output);
              !status.ok())
            return status;
          if (trace_wait) evaluated = std::chrono::steady_clock::now();
          if (!output || output->backend_id() != options.gpu->id() ||
              output->results().size() != result_first)
            return Status::Internal(
                "Histogram evaluation returned an invalid owner");
          ++stats->histogram_gpu_batches;
          stats->histogram_gpu_nodes += length;
          stats->histogram_gpu_rows += plan.rows;
          stats->histogram_gpu_properties += plan.properties;
          stats->histogram_dispatch_bytes += plan.dispatch_bytes;
          stats->histogram_peak_scratch_bytes =
              std::max(stats->histogram_peak_scratch_bytes, plan.scratch_bytes);
        } else {
          ++stats->histogram_admission_fallbacks;
        }
        for (size_t j = first; j < first + length; ++j) {
          auto &job = jobs[indices[j]];
          job.has_gpu_result = bool(output);
          job.force_cpu = !output;
          if (output) {
            const auto results = output->results().subspan(
                job.result_first, job.samples.samples().property_count);
            std::copy(results.begin(), results.end(), job.gpu_results.begin());
          }
        }
        // Accounted per-job result storage lets the output/dispatch reservation
        // end before another GPU batch while CPU workers apply these results.
        output.reset();
        evaluation_reservation.Reset();
        // A ready set can be split by scratch/admission limits. Attribute the
        // queue wait once; each sub-batch still gets its own resume snapshot.
        if (trace_wait && first == 0) batch_profile.queue_trace = queue_trace;
        queue.Resume(std::span(indices).subspan(first, length),
                     trace_wait ? &batch_profile.queue_trace : nullptr);
        if (admitted.ok() && diagnostic &&
            profile_mode != HistogramFittingProfileMode::kDisabled) {
          if (trace_wait)
            batch_profile.queue_trace.publish_nanoseconds =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - evaluated).count();
          if (diagnostic->written < diagnostic->batches.size())
            diagnostic->batches[diagnostic->written++] = batch_profile;
          else
            ++diagnostic->dropped;
        }
        first += length;
      }
    }
    queue.Stop();
    for (const auto &job : jobs)
      if (!job.status.ok())
        return job.status;
  }
  merge_stats();
  return Status::Ok();
}
} // namespace gjxl::modular_internal::reference_workflow_internal
