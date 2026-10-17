// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
#include "codec/modular/profile.h"
#include "codec/modular/histogram_tree_profile.h"
#include "codestream/modular/reference_learning_profile.h"
#include "codestream/modular/local_learning_queue.h"
#include "codestream/modular/reference_workflow_internal.h"
#include "core/parallel_work_internal.h"
#include <mutex>

namespace gjxl::modular_internal::reference_workflow_internal {
namespace {
using namespace resource_budget_internal;
using namespace modular_gpu_internal;
Status Invalid() { return Status::InvalidArgument("Invalid reference learning workflow"); }
HistogramSamplingOptions SamplingOptions(const ReferenceWorkflowOptions &options) {
  HistogramSamplingOptions sampling;
  sampling.row_backend = options.sample_rows;
  sampling.use_unique_row_proof = options.use_unique_row_proof;
  sampling.defer_gpu_row_materialization = options.defer_gpu_row_materialization;
  sampling.gpu_row_diagnostics = options.gpu_row_diagnostics;
  sampling.row_generator = options.row_generator;
  sampling.row_source = options.row_source;
  if (sampling.row_backend == HistogramRowBackend::kGpu && !sampling.row_source)
    sampling.row_backend = HistogramRowBackend::kSelectedCpu;
  sampling.maximum_samples = options.maximum_samples;
  sampling.maximum_channel_size = kGroupDimension;
  if (options.predictors == ReferencePredictors::kAll) {
    sampling.predictor_count = kHistogramPredictors;
    for (size_t i = 0; i < kHistogramPredictors; ++i)
      sampling.predictors[i] = static_cast<Predictor>(i);
    std::swap(sampling.predictors[0], sampling.predictors[size_t(Predictor::kWeighted)]);
    std::swap(sampling.predictors[1], sampling.predictors[size_t(Predictor::kGradient)]);
  }
  return sampling;
}
Status Learn(std::span<const HistogramTrainingImage> images, size_t maximum_leaves,
             const ReferenceWorkflowOptions &options, const ResourceBudget &budget,
             ReferenceWorkflowStats *stats, ReferenceModularStream *record,
             bool admitted = false, bool parallel_fit = false) {
  ProfileScope profile(ProfileStage::kTraining);
  // A parallel job has already reserved its entire preparation/fit/compaction
  // lifetime. Serial/global learning retains the tighter per-stage admission.
  const auto stage = [&](size_t bytes, auto &&function) {
    if (!admitted)
      return Stage(budget, bytes, stats, function);
    stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, bytes);
    return function();
  };
  const auto sampling = SamplingOptions(options);
  HostStorageBound bound;
  PreparedHistogramSamples prepared;
  {
    Seconds timer(stats->sample_preparation_seconds);
    if (auto s = ComputeHistogramSamplePreparationStorageBound(images, sampling, &bound); !s.ok())
      return s;
    if (auto s = stage(bound.peak_bytes,
                       [&] { return PrepareHistogramSamples(images, &prepared, sampling); });
        !s.ok())
      return s;
  }
  Seconds fitting(stats->fitting_seconds);
  record->learning = {0, images.size(), prepared.stats(), prepared.samples().weights.size(), 1};
  ++stats->learning_calls;
  MergeHistogramRowProfile(stats->rows, prepared.stats().rows);
  stats->scanned_pixels += prepared.stats().scanned_pixels;
  stats->selected_samples += prepared.stats().selected_samples;
  stats->distinct_samples += prepared.samples().weights.size();
  if (prepared.samples().total_samples == 0) {
    record->tree.nodes[0].predictor = sampling.predictors[0];
    ++stats->empty_learning_calls;
  } else {
    auto fitting = prepared.tree_options();
    fitting.maximum_leaves = maximum_leaves;
    fitting.cpu_participants = parallel_fit ? options.learning_cpu_participants : 1;
    fitting.cpu_participants = HistogramTreeLearningParticipants(prepared.samples(), fitting);
    HistogramTreeLearningStats execution;
    for (;;) {
      if (auto s = ComputeHistogramTreeLearningStorageBound(prepared.samples(), fitting, &bound);
          !s.ok())
        return s;
      bool started = false;
      auto status = stage(bound.peak_bytes, [&] {
        started = true;
        return LearnPreparedHistogramTree(prepared, &record->tree, maximum_leaves,
                                          fitting.maximum_depth, fitting.cpu_participants, &execution);
      });
      if (status.ok())
        break;
      // Retry only failed admission, before any sample mutation or worker
      // launch. Allocation/launch/fit failures keep their original status.
      if (!started && fitting.cpu_participants > 1 &&
          (status.code() == StatusCode::kOutOfMemory || status.code() == StatusCode::kUnavailable)) {
        fitting.cpu_participants = std::max(size_t{1}, fitting.cpu_participants / 2);
        ++stats->learning_memory_limited_waves;
        continue;
      }
      return status;
    }
    stats->learning_peak_participants = std::max(stats->learning_peak_participants,
                                                execution.participants);
    ++stats->fits;
    record->learning.tree_nodes = record->tree.size;
  }
  // The fit reserves its maximum tree size. Keep only live nodes between
  // scopes, after releasing the preparation columns. No refit or pruning.
  prepared = {};
  if (record->tree.nodes.size() > kInlineTreeNodes &&
      record->tree.nodes.size() != record->tree.size) {
    HostStorageBound compact;
    if (!compact.AddVector<DecisionNode>(record->tree.size, VectorCapacityPolicy::kFreshExact))
      return Invalid();
    if (auto s = stage(compact.peak_bytes,
                       [&] {
                         record->tree.nodes.resize(record->tree.size);
                         return Status::Ok();
                       });
        !s.ok())
      return s;
  }
  stats->tree_nodes += record->tree.size;
  return Status::Ok();
}

struct ChannelFeatures {
  std::array<WeightedFeatureView, kReferenceStreamChannels> views;
  // Borrowed from the owning feature batch through joined fitting/tokenization.
  const HistogramRowSource *row_source = nullptr;
};
ReferenceWorkflowOptions LocalOptions(const ReferenceWorkflowOptions &options,
                                      const ChannelFeatures *features) {
  auto local = options;
  if (features && options.sample_rows == HistogramRowBackend::kGpu) {
    local.row_source = features->row_source;
    local.row_generator = dynamic_cast<HistogramRowGenerator *>(options.gpu);
  }
  return local;
}
size_t LocalTreeLeaves(const PreparedModularStreamView &stream) {
  size_t coded_samples = 0;
  for (auto v : stream.channels)
    coded_samples += v.descriptor.extent.width * v.descriptor.extent.height;
  return std::min(kMaximumTreeLeaves, (1024 + coded_samples + 1) / 2);
}
Status LearnLocal(ReferenceModularStream &record, const PreparedModularStreamView &stream,
                  const ReferenceWorkflowOptions &options, const ResourceBudget &budget,
                  ReferenceWorkflowStats *stats,
                  std::span<const WeightedFeatureView> features = {}, bool admitted = false) {
  if (!record.channel_count)
    return Status::Ok();
  const HistogramTrainingImage image{record.learning_extent, record.training_channels(),
                                     record.metadata_channels, stream.weighted, features};
  return Learn({&image, 1}, LocalTreeLeaves(stream), options, budget, stats, &record, admitted);
}

// One reservation per whole job: selected rows are bounded without reading
// pixels; no other worker can consume this job's admitted fit workspace.
Status LocalLearningBound(const ReferenceModularStream &record,
                          const PreparedModularStreamView &stream,
                          const ReferenceWorkflowOptions &options,
                          std::span<const WeightedFeatureView> features, size_t *bytes) {
  *bytes = 1;
  if (!record.channel_count)
    return Status::Ok();
  const HistogramTrainingImage image{record.learning_extent, record.training_channels(),
                                     record.metadata_channels, stream.weighted, features};
  const auto sampling = SamplingOptions(options);
  HostStorageBound bound, fit;
  if (auto s = ComputeHistogramSamplePreparationStorageBound({&image, 1}, sampling, &bound);
      !s.ok())
    return s;
  const size_t leaves = LocalTreeLeaves(stream);
  // This workflow has no multiplier boxes. Both static quantization and the
  // dynamic properties have at most maximum_property_values cuts (WP has 15).
  if (auto s = ComputeHistogramTreeWorkspaceStorageBound(
          sampling.predictor_count, sampling.maximum_property_values + 1, leaves,
          kMaximumTreeDepth, &fit); !s.ok())
    return s;
  if (!bound.Add(fit) ||
      !bound.AddVector<DecisionNode>(2 * leaves - 1, VectorCapacityPolicy::kFreshExact))
    return Invalid();
  // Includes overlap with the maximum-size tree during compaction. Released
  // preparation scratch makes this conservative, including on tiny images.
  *bytes = bound.peak_bytes;
  return Status::Ok();
}
struct LocalLearningJob {
  ResourceReservation reservation;
  ReferenceWorkflowStats stats;
  HistogramFitRecord fit;
  size_t bytes = 0;
  size_t computed_features = 0, reused_features = 0;
};
template <typename T> using LearningStorage = ManagedVector<T, ResourceClass::kPreparation>;
size_t LearningParticipants(const ReferenceWorkflowOptions &options, size_t count) {
  using namespace thread_budget_internal;
  if (!HasCpuParticipation() || InExplicitParallelScope())
    return 1;
  size_t desired = options.learning_cpu_participants;
  if (!desired)
    desired = std::max(size_t{1}, size_t{std::thread::hardware_concurrency()});
  if (CpuThreadCount())
    desired = std::min(desired, CpuThreadCount());
  return std::min({desired, count, size_t{256}, current_cpu_execution.job.limit(),
                   current_cpu_execution.domain.limit()});
}
void MergeLearningStats(ReferenceWorkflowStats &to, const ReferenceWorkflowStats &from) {
  for (auto member : {&ReferenceWorkflowStats::learning_calls, &ReferenceWorkflowStats::fits,
                     &ReferenceWorkflowStats::empty_learning_calls,
                     &ReferenceWorkflowStats::scanned_pixels,
                     &ReferenceWorkflowStats::selected_samples,
                     &ReferenceWorkflowStats::distinct_samples, &ReferenceWorkflowStats::tree_nodes})
    to.*member += from.*member;
  to.maximum_stage_bytes = std::max(to.maximum_stage_bytes, from.maximum_stage_bytes);
  to.sample_preparation_seconds += from.sample_preparation_seconds;
  to.fitting_seconds += from.fitting_seconds;
  MergeHistogramRowProfile(to.rows, from.rows);
}
Status LearnLocalScopes(std::span<ReferenceModularStream> records,
                        std::span<const PreparedModularStreamView> streams,
                        const ReferenceWorkflowOptions &options, const ResourceBudget &budget,
                        ReferenceWorkflowStats *stats,
                        std::span<const ChannelFeatures> features = {}) {
  using namespace thread_budget_internal;
  if (options.fitting_backend != ReferenceFittingBackend::kCpu) {
    size_t first = 0;
    while (first < records.size()) {
      std::array<HistogramTrainingImage, histogram_gpu_internal::kMaximumHistogramFittingFits> images;
      std::array<HistogramFitRequest, histogram_gpu_internal::kMaximumHistogramFittingFits> requests;
      size_t stop = first, count = 0;
      for (; stop < records.size() && count < options.fitting_batch_fits; ++stop) {
        auto &record = records[stop];
        if (!record.channel_count) continue;
        images[count] = {record.learning_extent, record.training_channels(),
            record.metadata_channels, streams[stop].weighted,
            features.empty() ? std::span<const WeightedFeatureView>{}
                             : std::span(features[stop].views).first(record.channel_count)};
        requests[count] = {{&images[count], 1}, &record, LocalTreeLeaves(streams[stop]),
                           features.empty() ? nullptr : features[stop].row_source};
        ++count;
      }
      if (count) {
        bool handled = false;
        auto sampling = SamplingOptions(options);
        if (options.sample_rows == HistogramRowBackend::kGpu) {
          sampling.row_backend = HistogramRowBackend::kGpu;
          sampling.row_generator = dynamic_cast<HistogramRowGenerator *>(options.gpu);
        }
        if (auto s = TryLearnHistogramBatch(std::span(requests).first(count), sampling,
                LearningParticipants(options, count), options, budget, stats, &handled); !s.ok()) return s;
        if (!handled) break;
      }
      first = stop;
    }
    if (first == records.size()) return Status::Ok();
    records = records.subspan(first);
    streams = streams.subspan(first);
    if (!features.empty()) features = features.subspan(first);
  }
  ProfileScope profile(ProfileStage::kTraining);
  Seconds wall(stats->learning_seconds);
  const auto feature = [&](size_t i) -> std::span<const WeightedFeatureView> {
    return features.empty() ? std::span<const WeightedFeatureView>{}
                            : std::span(features[i].views).first(records[i].channel_count);
  };
  const auto local_options = [&](size_t i) {
    return LocalOptions(options, features.empty() ? nullptr : &features[i]);
  };
  const auto serial = [&](size_t first, size_t stop) {
    LearningWaveProbe trace;
    trace.Dispatch(stop - first, 1);
    ++stats->learning_waves;
    stats->learning_peak_participants = std::max(size_t{1}, stats->learning_peak_participants);
    for (size_t i = first; i < stop; ++i) {
      LearningWaveProbe::Job job(trace, i - first, records[i].id, 0, *stats);
      if (auto s = LearnLocal(records[i], streams[i], local_options(i), budget, stats, feature(i)); !s.ok())
        return s;
    }
    trace.Joined();
    return Status::Ok();
  };
  const size_t desired = LearningParticipants(options, records.size());
  if (desired < 2)
    return serial(0, records.size());

  auto *const parent_profile = active_modular_profile;
  auto *const parent_fit_profile = active_histogram_fitting_profile;
  HostStorageBound metadata;
  if (!metadata.AddVector<LocalLearningJob>(records.size(), VectorCapacityPolicy::kFreshExact) ||
      !metadata.AddVector<Status>(desired, VectorCapacityPolicy::kFreshExact) ||
      !metadata.AddVector<std::thread>(desired - 1, VectorCapacityPolicy::kFreshExact) ||
      (parent_fit_profile && !metadata.AddVector<HistogramFittingProfile>(
                                 desired, VectorCapacityPolicy::kFreshExact)))
    return Invalid();
  ResourceReservation metadata_reservation;
  if (auto s = budget.TryReserve(metadata.peak_bytes, &metadata_reservation); !s.ok()) {
    if (s.code() != StatusCode::kUnavailable && s.code() != StatusCode::kOutOfMemory)
      return s;
    ++stats->learning_memory_limited_waves;
    return serial(0, records.size());
  }
  stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, metadata.peak_bytes);
  auto context = CurrentResourceContext();
  context.reservation = &metadata_reservation;
  context.resource_class = ResourceClass::kPreparation;
  ResourceContextScope metadata_scope(context);
  LearningStorage<LocalLearningJob> jobs(records.size());
  LearningStorage<HistogramFittingProfile> profiles(parent_fit_profile ? desired : 0);
  for (size_t i = 0; i < records.size(); ++i)
    if (auto s = LocalLearningBound(records[i], streams[i], local_options(i), feature(i), &jobs[i].bytes);
        !s.ok())
      return s;
  constexpr ParallelWorkErrors errors{
      "Local tree worker allocation failed", "Local tree worker failed",
      StatusCode::kInvalidArgument, "Local tree worker storage overflow",
      "Local tree worker launch allocation failed", LaunchFailureAction::kReturnError,
      "Unable to start local tree workers"};
  size_t first = 0;
  while (first < records.size()) {
    LearningWaveProbe trace;
    size_t count = 0;
    for (; count < desired && first + count < records.size(); ++count) {
      auto &job = jobs[first + count];
      if (auto s = budget.TryReserve(job.bytes, &job.reservation); !s.ok()) {
        if (s.code() != StatusCode::kUnavailable && s.code() != StatusCode::kOutOfMemory)
          return s;
        ++stats->learning_memory_limited_waves;
        break;
      }
      job.stats.maximum_stage_bytes = job.bytes;
    }
    if (!count) {
      // The conservative whole-job envelope may not fit while the original
      // prepare/fit stage envelopes do. Drop even the scheduler metadata so
      // the tighter serial route has its original capacity available.
      jobs = LearningStorage<LocalLearningJob>{};
      profiles = LearningStorage<HistogramFittingProfile>{};
      metadata_reservation.Reset();
      return serial(first, records.size());
    }
    CpuWorkerGroup group(count);
    trace.Dispatch(records.size() - first, group.participants());
    ++stats->learning_waves;
    stats->learning_peak_participants = std::max(stats->learning_peak_participants,
                                                group.participants());
    // Only admission is serialized. A worker releases its completed whole-job
    // reservation before taking the next canonical job, without a wave barrier.
    // Retained tree tickets remain charged after their reservation closes.
    size_t next = first + count, memory_limited = 0;
    std::mutex admission;
    std::atomic<bool> launch_cancelled{false};
    const auto run = [&](size_t index, size_t worker) -> Status {
      size_t i = first + index;
      while (true) {
        if (launch_cancelled.load(std::memory_order_relaxed))
          return Status::Ok();
        auto &job = jobs[i];
        Status status;
        {
          LearningWaveProbe::Job timing(trace, i - first, records[i].id, worker, job.stats);
          auto resources = CurrentResourceContext();
          resources.reservation = &job.reservation;
          ResourceContextScope scope(resources);
          ModularProfile worker_profile;
          ProfileSession session(parent_profile ? &worker_profile : nullptr);
          auto *fit_profile = parent_fit_profile ? &profiles[worker] : nullptr;
          if (fit_profile) {
            // Counters accumulate per worker; each fit has its own canonical
            // record, published by the caller after the join.
            fit_profile->records = {&job.fit, 1};
            fit_profile->records_written = 0;
          }
          HistogramFittingProfileSession fitting_session(fit_profile);
          status = LearnLocal(records[i], streams[i], local_options(i), budget, &job.stats,
                              feature(i), true);
          job.computed_features = worker_profile.training_work.weighted_prediction_samples;
          job.reused_features = worker_profile.training_work.reused_prediction_samples;
        }
        job.reservation.Reset();
        if (!status.ok())
          return status;
        std::lock_guard lock(admission);
        if (next == records.size() || launch_cancelled.load(std::memory_order_relaxed))
          return Status::Ok();
        auto &candidate = jobs[next];
        if (auto s = budget.TryReserve(candidate.bytes, &candidate.reservation); !s.ok()) {
          if (s.code() != StatusCode::kUnavailable && s.code() != StatusCode::kOutOfMemory)
            return s;
          // Never wait while holding image/features/worker grants. Other
          // finishing workers may admit this job; otherwise the caller retries
          // the untouched suffix after every active job has joined.
          ++memory_limited;
          return Status::Ok();
        }
        candidate.stats.maximum_stage_bytes = candidate.bytes;
        i = next++;
      }
    };
    Status status;
    if (group.participants() > 1) {
      status = RunParallelWork<LearningStorage>(count, group, group.participants() - 1,
                                                WorkerLaunchSite::kModularTreeLearning,
                                                errors, run, &launch_cancelled);
    } else {
      for (size_t i = 0; i < count && status.ok(); ++i)
        status = run(i, 0);
    }
    // All workers are joined before reading counters, releasing reservations,
    // publishing trees to tokenization, or returning a failure.
    trace.Joined(next - first);
    stats->learning_memory_limited_waves += memory_limited != 0;
    if (parent_fit_profile)
      for (auto &fit_profile : profiles) {
        fit_profile.records = {};
        fit_profile.records_written = 0;
        MergeHistogramFittingProfile(*parent_fit_profile, fit_profile);
        fit_profile = {};
      }
    for (size_t i = first; i < next; ++i) {
      MergeLearningStats(*stats, jobs[i].stats);
      if (parent_profile) {
        parent_profile->training_work.weighted_prediction_samples += jobs[i].computed_features;
        parent_profile->training_work.reused_prediction_samples += jobs[i].reused_features;
      }
      if (parent_fit_profile && jobs[i].stats.fits) {
        if (parent_fit_profile->records_written < parent_fit_profile->records.size())
          parent_fit_profile->records[parent_fit_profile->records_written++] = jobs[i].fit;
        else
          ++parent_fit_profile->records_dropped;
      }
      jobs[i] = {};
    }
    if (!status.ok())
      return status;
    first = next;
  }
  return Status::Ok();
}
Status LearnGlobal(std::span<ReferenceModularStream> records,
                   std::span<const PreparedModularStreamView> streams, uint64_t population,
                   const ReferenceWorkflowOptions &options, const ResourceBudget &budget,
                   ReferenceWorkflowStats *stats, std::span<const ChannelFeatures> features = {}) {
  Seconds wall(stats->learning_seconds);
  // Preserve excluded global descriptors, canonical ID holes, and the first
  // image's population contribution, exactly as in the scalar workflow.
  size_t stop = 1;
  for (const auto &record : records)
    if (record.channel_count)
      stop = std::max(stop, record.id + 1);
  HostStorageBound bound;
  if (!bound.AddVector<HistogramTrainingImage>(stop, VectorCapacityPolicy::kFreshExact))
    return Invalid();
  ManagedVector<HistogramTrainingImage> images;
  if (auto s = Stage(budget, bound.peak_bytes, stats,
                     [&] {
                       images.resize(stop);
                       return Status::Ok();
                     });
      !s.ok())
    return s;
  for (size_t i = 0; i < records.size(); ++i) {
    const auto &record = records[i];
    if (record.id < stop)
      images[record.id] = {record.learning_extent, record.training_channels(),
                           record.metadata_channels, streams[i].weighted,
                           features.empty() ? std::span<const WeightedFeatureView>{}
                                            : std::span(features[i].views).first(record.channel_count)};
  }
  if (options.fitting_backend != ReferenceFittingBackend::kCpu) {
    const HistogramFitRequest request{images, &records[0], MaximumGlobalTreeLeaves(population)};
    bool handled = false;
    if (auto s = TryLearnHistogramBatch({&request, 1}, SamplingOptions(options), 1,
            options, budget, stats, &handled, false); !s.ok()) return s;
    if (handled) return Status::Ok();
  }
  stats->learning_peak_participants = 1;
  ++stats->learning_waves;
  return Learn(images, MaximumGlobalTreeLeaves(population), options, budget, stats, &records[0], false, true);
}

// Batching is independent of compression decisions. Local fitting can share a
// bounded cohort of feature owners, including both CPU and GPU producers.
// Cohort mode retains the original aggregate sample ceiling. Pipeline mode
// preserves batch shapes and admits owners under its separate live window.
// Explicit buffered/global learning retains its entire scope until the one fit.
constexpr size_t kFeatureBatchSamples = 8 * 1024 * 1024;
struct FeatureBatch {
  size_t first = 0, stop = 0, slices = 0, samples = 0, retained_bytes = 0;
  size_t remaining = 0;
  bool released = false;
  LearningBatchTiming *timing = nullptr;
  ResourceReservation completion;
  std::unique_ptr<PreparedGpuModularStreamFeatures> owner;
};
struct HybridLearning {
  std::span<ReferenceModularStream> records;
  std::span<PreparedModularStreamView> streams;
  const ReferenceWorkflowOptions &options;
  const ResourceBudget &budget;
  ReferenceWorkflowStats *stats;
  GpuModularStreamProcessing &op;
  GpuTokenOwners &tokens;
  ManagedVector<ChannelFeatures> features;
  ManagedVector<ModularStreamFeatureSlice> jobs;
  ManagedVector<ModularStreamTokenTree> trees;
  ManagedVector<FeatureBatch> batches;
  size_t live_samples = 0, live_bytes = 0, token_owners = 0;

  Status Initialize() {
    if (options.sample_rows == HistogramRowBackend::kGpu &&
        !dynamic_cast<HistogramRowGenerator *>(options.gpu))
      return Status::Unsupported("Backend has no histogram row generator");
    if (records.size() > SIZE_MAX / kReferenceStreamChannels)
      return Invalid();
    const size_t job_capacity =
        std::min(kMaximumModularFeatureBatchSlices, records.size() * kReferenceStreamChannels);
    // A cohort may alternate producer eligibility at every stream. Owners are
    // charged individually and released after their tokens are completed.
    const size_t batch_capacity = records.size();
    HostStorageBound bound;
    if (!bound.AddVector<ChannelFeatures>(records.size(), VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<ModularStreamFeatureSlice>(job_capacity,
                                                    VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<ModularStreamTokenTree>(job_capacity, VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<FeatureBatch>(batch_capacity, VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<std::unique_ptr<PreparedGpuModularTokens>>(
            records.size(), VectorCapacityPolicy::kFreshExact))
      return Invalid();
    return Stage(budget, bound.peak_bytes, stats, [&] {
      features.resize(records.size());
      jobs.reserve(job_capacity);
      trees.reserve(job_capacity);
      batches.resize(batch_capacity);
      tokens.resize(records.size());
      return Status::Ok();
    });
  }
  Status CohortStop(size_t first, size_t *out) const {
    size_t stop = first, population = 0, fits = 0, aligned = first;
    const size_t participants = LearningParticipants(options, records.size() - first);
    for (; stop < streams.size(); ++stop) {
      size_t samples = 0;
      for (const auto &channel : streams[stop].channels) {
        size_t area;
        if (!channel.descriptor.extent.try_area(&area) || area > SIZE_MAX - samples)
          return Invalid();
        samples += area;
      }
      if (samples > kFeatureBatchSamples)
        return Status::Unsupported("Selected stream exceeds histogram feature batch admission");
      if (samples > kFeatureBatchSamples - population)
        break;
      population += samples;
      // The excluded global learning record can have channels but no coded
      // samples. It must not shift a boundary intended for actual local fits.
      fits += samples != 0;
      if (participants && fits && fits % participants == 0)
        aligned = stop + 1;
    }
    // Avoid a one-fit spill at a capacity boundary, without shrinking a small
    // cohort below two participant groups or changing the final tail.
    if (stop < streams.size() && participants > 1 && fits >= 2 * participants && aligned > first)
      stop = aligned;
    *out = stop;
    return Status::Ok();
  }
  Status Describe(size_t first, size_t limit, FeatureBatch *batch) const {
    FeatureBatch result;
    result.first = first;
    bool wavefront = true;
    size_t stop = first;
    for (; stop < limit; ++stop) {
      const auto &stream = streams[stop];
      size_t samples = 0, slices = 0;
      bool supported = true;
      for (const auto &channel : stream.channels) {
        size_t area;
        if (!channel.descriptor.extent.try_area(&area) || area > SIZE_MAX - samples)
          return Invalid();
        samples += area;
        slices += area != 0;
        supported &=
            channel.descriptor.extent.width <= 256 && channel.descriptor.extent.height <= 256;
      }
      if (!samples)
        continue;
      if (samples > kFeatureBatchSamples || slices > kMaximumModularFeatureBatchSlices)
        return Status::Unsupported("Selected stream exceeds histogram feature batch admission");
      if (result.slices &&
          (samples > kFeatureBatchSamples - result.samples ||
           slices > kMaximumModularFeatureBatchSlices - result.slices || supported != wavefront))
        break;
      wavefront = supported;
      result.slices += slices;
      result.samples += samples;
    }
    result.stop = stop;
    result.remaining = stop - first;
    *batch = std::move(result);
    return Status::Ok();
  }
  Status Prepare(size_t first, size_t limit, FeatureBatch *batch, bool *admission_limited,
                 bool protect_completion = false) {
    ProfileScope profile(ProfileStage::kTraining);
    Seconds timer(stats->feature_seconds);
    jobs.clear();
    FeatureBatch result;
    if (auto s = Describe(first, limit, &result); !s.ok()) return s;
    const size_t stop = result.stop;
    for (size_t i = first; i < stop; ++i)
      for (size_t c = 0; c < streams[i].channels.size(); ++c) {
        const auto &channel = streams[i].channels[c];
        if (!channel.descriptor.extent.empty())
          jobs.push_back({channel, streams[i].weighted, static_cast<uint32_t>(records[i].id),
                          static_cast<uint32_t>(c)});
      }
    auto *trace = active_learning_schedule_profile;
    const double prepare_start = trace ? trace->Now() : 0;
    if (jobs.empty()) {
      *batch = std::move(result);
      return Status::Ok();
    }
    if (protect_completion) {
      size_t maximum_nodes = 0;
      for (size_t i = first; i < stop; ++i)
        if (!streams[i].channels.empty()) maximum_nodes += 2 * LocalTreeLeaves(streams[i]) - 1;
      HostStorageBound completion;
      if (auto s = ComputeModularStreamTokenStorageBound(result.samples, result.slices,
              maximum_nodes, &completion); !s.ok()) return s;
      if (auto s = budget.TryReserve(completion.peak_bytes, &result.completion); !s.ok()) {
        *admission_limited = s.code() == StatusCode::kUnavailable || s.code() == StatusCode::kOutOfMemory;
        return s;
      }
    }
    ModularStreamFeatureStoragePlan plan;
    const auto limits = op.modular_stream_limits();
    if (auto s = ComputeModularStreamFeatureStoragePlan(jobs, options.feature_cpu_participants,
                                                        &plan, &limits);
        !s.ok())
      return s;
    const size_t bytes = std::max(plan.cpu_preparation.peak_bytes, plan.gpu_preparation.peak_bytes);
    {
      ResourceReservation reservation;
      if (auto s = budget.TryReserve(std::max(size_t{1}, bytes), &reservation); !s.ok()) {
        *admission_limited = s.code() == StatusCode::kUnavailable || s.code() == StatusCode::kOutOfMemory;
        return s;
      }
      stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, bytes);
      auto context = CurrentResourceContext();
      context.reservation = &reservation;
      context.resource_class = ResourceClass::kPreparation;
      ResourceContextScope resources(context);
      if (auto s = op.PrepareModularStreamFeatures(jobs, &result.owner,
                                                   options.feature_cpu_participants,
                                                   options.feature_producer); !s.ok())
        return s;
    }
    if (!result.owner || result.owner->backend_id() != options.gpu->id() ||
        result.owner->sample_count() != result.samples ||
        result.owner->weighted_views().size() != jobs.size())
      return Status::Internal("Histogram feature producer returned an invalid owner");
    for (size_t i = first; i < stop; ++i)
      features[i].row_source = result.owner.get();
    const auto views = result.owner->weighted_views();
    size_t next = 0;
    for (size_t i = first; i < stop; ++i)
      for (size_t c = 0; c < streams[i].channels.size(); ++c) {
        const auto &channel = streams[i].channels[c];
        if (channel.descriptor.extent.empty())
          continue;
        if (views[next].values.empty())
          return Status::Internal("Histogram feature map is empty");
        if (auto s = ValidateWeightedFeatureView(views[next], channel, streams[i].weighted);
            !s.ok())
          return s;
        features[i].views[c] = views[next++];
      }
    result.retained_bytes = plan.retained.retained_bytes;
    if (result.samples > SIZE_MAX - live_samples || result.retained_bytes > SIZE_MAX - live_bytes)
      return Invalid();
    live_samples += result.samples;
    live_bytes += result.retained_bytes;
    stats->peak_feature_samples = std::max(stats->peak_feature_samples, live_samples);
    stats->peak_feature_bytes = std::max(stats->peak_feature_bytes, live_bytes);
    ++stats->feature_batches;
    stats->feature_samples += result.samples;
    if (result.owner->producer() == WeightedFeatureBackend::kGpu) {
      ++stats->gpu_feature_batches;
      stats->gpu_feature_samples += result.samples;
    } else if (result.owner->producer() == WeightedFeatureBackend::kCpu) {
      ++stats->cpu_feature_batches;
    } else {
      return Status::Internal("Histogram feature producer returned an invalid route");
    }
    *batch = std::move(result);
    if (trace && !trace->batches.empty()) {
      if (trace->batches_written == trace->batches.size()) ++trace->dropped;
      else {
        batch->timing = &trace->batches[trace->batches_written++];
        *batch->timing = {records[first].id, records[stop - 1].id + 1, batch->samples,
                          prepare_start, trace->Now(), 0, 0, 0};
      }
    }
    return Status::Ok();
  }
  Status Tokenize(FeatureBatch &batch, bool *admission_limited = nullptr) {
    if (!batch.owner)
      return Status::Ok();
    ProfileScope profile(ProfileStage::kTokens);
    Seconds timer(stats->token_seconds);
    if (batch.timing) {
      const double now = active_learning_schedule_profile->Now();
      if (!batch.timing->fits_observed_done) batch.timing->fits_observed_done = now;
      batch.timing->tokenize_start = now;
    }
    trees.clear();
    if (stats->local_trees) {
      size_t next = 0;
      for (size_t i = batch.first; i < batch.stop; ++i) {
        size_t count = 0;
        for (const auto &channel : streams[i].channels)
          count += !channel.descriptor.extent.empty();
        if (count) {
          trees.push_back({next, count, &records[i].tree});
          next += count;
        }
      }
    } else {
      trees.push_back({0, batch.slices, &records[0].tree});
    }
    ModularStreamTokenStoragePlan plan;
    const auto limits = op.modular_stream_limits();
    if (auto s = ComputeModularStreamTokenStoragePlan(batch.samples, batch.slices, trees, &plan,
                                                      &limits);
        !s.ok())
      return s;
    std::unique_ptr<PreparedGpuModularTokens> owner;
    bool started = false;
    const auto generate = [&] {
      if (batch.completion.valid()) {
        // Protected before this batch's feature allocation: lookahead and fit
        // reservations cannot consume the capacity needed to retire its owner.
        if (auto s = batch.completion.ReduceCapacity(plan.generation.peak_bytes); !s.ok()) return s;
        auto context = CurrentResourceContext();
        context.reservation = &batch.completion;
        ResourceContextScope scope(context);
        stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, plan.generation.peak_bytes);
        started = true;
        return op.GenerateModularStreamTokens(*batch.owner, trees, &owner);
      }
      return Stage(budget, plan.generation.peak_bytes, stats, [&] {
          started = true;
          return op.GenerateModularStreamTokens(*batch.owner, trees, &owner);
        });
    };
    if (auto s = generate(); !s.ok()) {
      if (admission_limited) *admission_limited = !started &&
          (s.code() == StatusCode::kUnavailable || s.code() == StatusCode::kOutOfMemory);
      return s;
    }
    if (!owner || owner->backend_id() != options.gpu->id() ||
        owner->token_count() != batch.samples || owner->context_count() != plan.maximum_contexts)
      return Status::Internal("Histogram token producer returned an invalid owner");
    ModularTokenHostView host;
    if (auto s = owner->BorrowHostView(&host); !s.ok())
      return s;
    if (host.values.size() != batch.samples || host.contexts.size() != batch.samples ||
        !host.values.data() || !host.contexts.data())
      return Status::Internal("Histogram token producer returned an invalid host view");
    size_t next = 0;
    for (size_t i = batch.first; i < batch.stop; ++i) {
      auto &stream = streams[i];
      size_t count = 0;
      for (const auto &channel : stream.channels)
        count += channel.descriptor.extent.width * channel.descriptor.extent.height;
      if (count) {
        stream.completed_tokens = PreparedModularStreamTokenView{
            EntropyTokenStreamView::Split(host.values.subspan(next, count),
                                          host.contexts.subspan(next, count)),
            stream.channels, stats->local_trees ? &records[i].tree : &records[0].tree,
            stream.weighted, records[i].id};
        next += count;
      }
      features[i] = {};
    }
    if (next != batch.samples || token_owners >= tokens.size())
      return Status::Internal("Histogram batch token coverage mismatch");
    tokens[token_owners++] = std::move(owner);
    ++stats->token_batches;
    stats->token_samples += batch.samples;
    live_samples -= batch.samples;
    live_bytes -= batch.retained_bytes;
    batch.owner.reset();
    batch.completion.Reset();
    if (batch.timing) batch.timing->released = active_learning_schedule_profile->Now();
    return Status::Ok();
  }
  Status RunPipeline(bool *handled) {
    *handled = false;
    const bool resident = options.fitting_backend == ReferenceFittingBackend::kGpuResident ||
        options.fitting_backend == ReferenceFittingBackend::kGpuFrontier ||
        options.fitting_backend == ReferenceFittingBackend::kGpuTiledFrontier;
    if (resident && (active_histogram_fitting_profile || options.fitting_gpu_profile))
      return Status::Unsupported("Resident fitting has complete-call diagnostics only");
    struct PipelineJob : LocalLearningJob {
      Status status;
      size_t batch = 0, group = 0;
      bool admitted = true, prepared = false;
      double start = 0, finish = 0;
    };
    struct ResidentGroup { size_t first = 0, stop = 0, unprepared = 0; };
    const size_t desired = LearningParticipants(options, records.size());
    auto *const parent_profile = active_modular_profile;
    auto *const parent_fit_profile = active_histogram_fitting_profile;
    HostStorageBound metadata;
    if (resident && (!metadata.AddVector<PreparedHistogramSamples>(records.size(), VectorCapacityPolicy::kFreshExact) ||
        !metadata.AddVector<ResidentGroup>(records.size(), VectorCapacityPolicy::kFreshExact))) return Invalid();
    if (!metadata.AddVector<PipelineJob>(records.size(), VectorCapacityPolicy::kFreshExact) ||
        !metadata.AddVector<std::thread>(desired - 1, VectorCapacityPolicy::kFreshExact) ||
        (parent_fit_profile && !metadata.AddVector<HistogramFittingProfile>(
            desired, VectorCapacityPolicy::kFreshExact))) return Invalid();
    ResourceReservation reservation;
    if (auto s = budget.TryReserve(metadata.peak_bytes, &reservation); !s.ok()) {
      if (s.code() != StatusCode::kUnavailable && s.code() != StatusCode::kOutOfMemory) return s;
      ++stats->pipeline_admission_stalls;
      return Status::Ok();  // No feature/fit has started: retain the cohort route.
    }
    *handled = true;
    ProfileScope training(ProfileStage::kTraining);
    Seconds elapsed(stats->pipeline_seconds);
    auto context = CurrentResourceContext();
    context.reservation = &reservation;
    context.resource_class = ResourceClass::kPreparation;
    ResourceContextScope metadata_scope(context);
    LearningStorage<PipelineJob> fit_jobs(records.size());
    LearningStorage<PreparedHistogramSamples> prepared(resident ? records.size() : 0);
    LearningStorage<ResidentGroup> groups(resident ? records.size() : 0);
    size_t group_count = 0;
    if (resident) for (size_t first = 0; first < records.size();) {
      size_t stop;
      if (auto s = CohortStop(first, &stop); !s.ok()) return s;
      while (first < stop) {
        auto &group = groups[group_count];
        group.first = first;
        group.stop = std::min(stop, first + std::min(options.fitting_batch_fits,
            histogram_gpu_internal::kMaximumResidentHistogramFits));
        group.unprepared = group.stop - first;
        for (size_t i = first; i < group.stop; ++i) fit_jobs[i].group = group_count;
        first = group.stop;
        ++group_count;
      }
    }
    LearningStorage<HistogramFittingProfile> profiles(parent_fit_profile ? desired : 0);
    LearningWaveProbe trace;
    using Clock = std::chrono::steady_clock;
    const auto epoch = Clock::now();
    const auto now = [&] { return std::chrono::duration<double>(Clock::now() - epoch).count(); };
    const auto work = [&](size_t i, size_t worker) noexcept {
      auto &job = fit_jobs[i];
      job.start = now();
      try {
        LearningWaveProbe::Job timing(trace, i, records[i].id, worker, job.stats);
        auto resources = CurrentResourceContext();
        resources.reservation = job.admitted ? &job.reservation : nullptr;
        ResourceContextScope scope(resources);
        ModularProfile worker_profile;
        ProfileSession session(parent_profile ? &worker_profile : nullptr);
        auto *fit_profile = parent_fit_profile ? &profiles[worker] : nullptr;
        if (fit_profile) {
          fit_profile->records = {&job.fit, 1};
          fit_profile->records_written = 0;
        }
        HistogramFittingProfileSession fitting_session(fit_profile);
        if (resident) {
          auto &record = records[i];
          if (record.channel_count) {
            const auto sampling = SamplingOptions(LocalOptions(options, &features[i]));
            const HistogramTrainingImage image{record.learning_extent, record.training_channels(),
                record.metadata_channels, streams[i].weighted,
                std::span(features[i].views).first(record.channel_count)};
            {
              Seconds preparation(job.stats.sample_preparation_seconds);
              job.status = PrepareHistogramSamples({&image, 1}, &prepared[i], sampling);
            }
            if (job.status.ok()) {
              const auto &p = prepared[i].stats();
              record.learning = {0, 1, p, prepared[i].samples().weights.size(), 1};
              job.stats.rows = p.rows;
              job.stats.learning_calls = 1;
              job.stats.scanned_pixels = p.scanned_pixels;
              job.stats.selected_samples = p.selected_samples;
              job.stats.distinct_samples = prepared[i].samples().weights.size();
              if (!prepared[i].samples().total_samples) {
                record.tree.nodes[0].predictor = sampling.predictors[0];
                job.stats.empty_learning_calls = 1;
                job.stats.tree_nodes = 1;
              }
            }
          }
        } else {
          job.status = LearnLocal(records[i], streams[i], LocalOptions(options, &features[i]), budget, &job.stats,
              std::span(features[i].views).first(records[i].channel_count), job.admitted);
        }
        job.computed_features = worker_profile.training_work.weighted_prediction_samples;
        job.reused_features = worker_profile.training_work.reused_prediction_samples;
      } catch (const ManagedAllocationFailure &e) {
        job.status = e.status();
      } catch (const std::bad_alloc &) {
        job.status = Status::OutOfMemory("Pipelined local fit allocation failed");
      } catch (const std::length_error &) {
        job.status = Invalid();
      } catch (...) {
        job.status = Status::Internal("Pipelined local fit failed");
      }
      job.reservation.Reset();
      job.finish = now();
    };
    // Declared last: error paths join before destroying callback/jobs/profiles
    // or returning to HybridLearning's feature-owner destructor.
    LocalLearningQueue queue(desired);
    trace.Dispatch(records.size(), queue.participants());
    if (auto s = queue.Start(work); !s.ok()) return s;
    ++stats->learning_waves;
    stats->learning_peak_participants = queue.participants();
    stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, metadata.peak_bytes);
    const size_t slots = std::min(LocalLearningQueue::kCapacity,
        resident ? 2 * std::min(options.fitting_batch_fits, histogram_gpu_internal::kMaximumResidentHistogramFits)
                 : 2 * queue.participants());
    size_t next = 0, next_fit = 0, complete = 0, active = 0, count = 0, live_batches = 0;
    size_t cohort_stop = 0;
    bool new_cohort = false;
    const auto finished = [&](size_t i) -> Status {
      auto &job = fit_jobs[i];
      if (!job.status.ok()) return job.status;
      auto &batch = batches[job.batch];
      if (!--batch.remaining && batch.timing)
        batch.timing->fits_observed_done = active_learning_schedule_profile->Now();
      ++complete;
      return Status::Ok();
    };
    while (complete != records.size() || live_batches) {
      Status deferred_admission;
      size_t i;
      while (queue.Take(&i)) {
        --active;
        if (!fit_jobs[i].status.ok()) return fit_jobs[i].status;
        if (resident) {
          fit_jobs[i].prepared = true;
          --groups[fit_jobs[i].group].unprepared;
        } else if (auto s = finished(i); !s.ok()) return s;
      }
      // Only a batch's own completed fits gate tokenization. Other batches may
      // still be fitting; their disjoint records and feature borrows stay live.
      for (size_t b = 0; b < count; ++b) {
        auto &batch = batches[b];
        if (batch.remaining || batch.released) continue;
        bool limited = false;
        if (auto s = Tokenize(batch, &limited); !s.ok()) {
          if (!limited || !active) return s;
          ++stats->pipeline_admission_stalls;
          continue;
        }
        batch.released = true;
        --live_batches;
      }
      while (next_fit < next && active < slots) {
        auto &job = fit_jobs[next_fit];
        if (!job.bytes) {
          if (resident && records[next_fit].channel_count) {
            const auto &record = records[next_fit];
            const HistogramTrainingImage image{record.learning_extent, record.training_channels(),
                record.metadata_channels, streams[next_fit].weighted,
                std::span(features[next_fit].views).first(record.channel_count)};
            HostStorageBound bound;
            if (auto s = ComputeHistogramSamplePreparationStorageBound({&image, 1},
                    SamplingOptions(LocalOptions(options, &features[next_fit])), &bound); !s.ok()) return s;
            job.bytes = std::max(size_t{1}, bound.peak_bytes);
          } else if (auto s = LocalLearningBound(records[next_fit], streams[next_fit], LocalOptions(options, &features[next_fit]),
              std::span(features[next_fit].views).first(records[next_fit].channel_count), &job.bytes);
              !s.ok()) return s;
        }
        if (auto s = budget.TryReserve(job.bytes, &job.reservation); !s.ok()) {
          if (s.code() != StatusCode::kUnavailable && s.code() != StatusCode::kOutOfMemory) return s;
          ++stats->pipeline_admission_stalls;
          if (resident) { deferred_admission = s; break; }
          if (active) break;
          // With all admitted jobs drained, the original tighter stage bounds
          // may still succeed. Never retry a job that actually started.
          ++stats->pipeline_serial_fallbacks;
          job.admitted = false;
          work(next_fit, 0);
          if (auto s = finished(next_fit++); !s.ok()) return s;
          break;  // Release a completed owner before admitting more work.
        }
        job.stats.maximum_stage_bytes = job.bytes;
        if (auto s = queue.Submit(next_fit++); !s.ok()) return s;
        ++active;
        stats->pipeline_peak_jobs = std::max(stats->pipeline_peak_jobs, active);
      }
      if (next < records.size()) {
        if (next == cohort_stop) {
          if (auto s = CohortStop(next, &cohort_stop); !s.ok()) return s;
          new_cohort = true;
        }
        FeatureBatch plan;
        if (auto s = Describe(next, cohort_stop, &plan); !s.ok()) return s;
        if (plan.samples <= options.feature_window_samples - live_samples) {
          bool limited = false;
          if (auto s = Prepare(next, cohort_stop, &batches[count], &limited, true); !s.ok()) {
            if (!limited) return s;
            ++stats->pipeline_admission_stalls;
            if (resident) deferred_admission = s;
            if (!active && !live_batches) {
              if (next) return s;
              // Protected completion may be too conservative for a tight
              // domain. Nothing has started; discard the queue and use the
              // original per-stage cohort admission without recomputing work.
              queue.Stop();
              trace.Joined(0);
              --stats->learning_waves;
              stats->learning_peak_participants = 0;
              *handled = false;
              return Status::Ok();
            }
          } else {
            auto &batch = batches[count];
            if (batch.stop <= next) return Status::Internal("Pipeline batch did not advance");
            for (size_t j = next; j < batch.stop; ++j) fit_jobs[j].batch = count;
            next = batch.stop;
            ++count;
            ++live_batches;
            stats->pipeline_peak_batches = std::max(stats->pipeline_peak_batches, live_batches);
            if (new_cohort) { ++stats->feature_cohorts; new_cohort = false; }
            continue;  // Publish this batch's jobs before doing more device work.
          }
        }
      }
      if (resident) {
        bool dispatched = false;
        for (size_t g = 0; g < group_count; ++g) {
          auto &group = groups[g];
          if (group.first == group.stop || (group.unprepared && active)) continue;
          size_t stop = group.first;
          while (stop < group.stop && fit_jobs[stop].prepared) ++stop;
          if (stop == group.first) continue;
          // Prefer complete cohorts. Under admission pressure a prepared
          // prefix can retire owners without recomputing samples or CPU fits.
          std::array<histogram_gpu_internal::ResidentHistogramRequest,
                     histogram_gpu_internal::kMaximumResidentHistogramFits> requests;
          std::array<size_t, histogram_gpu_internal::kMaximumResidentHistogramFits> indices;
          size_t count = 0;
          for (size_t j = group.first; j < stop; ++j) {
            if (!records[j].channel_count || !prepared[j].samples().total_samples) continue;
            auto learning = prepared[j].tree_options();
            learning.maximum_leaves = LocalTreeLeaves(streams[j]);
            requests[count] = {{static_cast<uint32_t>(records[j].id), &prepared[j].samples()}, learning};
            indices[count++] = j;
          }
          histogram_gpu_internal::ResidentHistogramResult result;
          bool limited = false;
          if (count) {
            auto s = FitResidentHistogramBatch(std::span(requests).first(count), options, budget, stats, &result, &limited);
            if (!s.ok()) {
              if (!limited || !active) return s;
              ++stats->pipeline_admission_stalls;
              deferred_admission = s;
              break;
            }
          }
          for (size_t j = 0; j < count; ++j) {
            const size_t index = indices[j];
            auto &record = records[index];
            record.tree = std::move(result.trees[j]);
            record.learning.tree_nodes = record.tree.size;
            ++fit_jobs[index].stats.fits;
            fit_jobs[index].stats.tree_nodes += record.tree.size;
          }
          for (size_t j = group.first; j < stop; ++j) {
            prepared[j] = {};
            fit_jobs[j].finish = now();
            trace.CompleteDeferred(j, fit_jobs[j].stats);
            if (auto s = finished(j); !s.ok()) return s;
          }
          group.first = stop;
          dispatched = true;
          break;  // Retire completed owners before fitting another cohort.
        }
        if (dispatched) continue;
        if (!active && !deferred_admission.ok()) return deferred_admission;
      }
      if (active) queue.Progress();
    }
    queue.Stop();
    trace.Joined();
    double first_start = now(), last_finish = 0;
    for (auto &job : fit_jobs) {
      first_start = std::min(first_start, job.start);
      last_finish = std::max(last_finish, job.finish);
      MergeLearningStats(*stats, job.stats);
      if (parent_profile) {
        parent_profile->training_work.weighted_prediction_samples += job.computed_features;
        parent_profile->training_work.reused_prediction_samples += job.reused_features;
      }
      if (parent_fit_profile && job.stats.fits) {
        if (parent_fit_profile->records_written < parent_fit_profile->records.size())
          parent_fit_profile->records[parent_fit_profile->records_written++] = job.fit;
        else ++parent_fit_profile->records_dropped;
      }
    }
    stats->learning_seconds += std::max(0.0, last_finish - first_start);
    if (parent_fit_profile)
      for (auto &profile : profiles) {
        profile.records = {};
        profile.records_written = 0;
        MergeHistogramFittingProfile(*parent_fit_profile, profile);
      }
    return Status::Ok();
  }
  Status Run(uint64_t population) {
    if (auto s = Initialize(); !s.ok())
      return s;
    bool pipelined = false;
    if (stats->local_trees && options.learning_schedule == ReferenceLearningSchedule::kPipeline)
      if (auto s = RunPipeline(&pipelined); !s.ok()) return s;
    size_t next = 0, count = 0;
    while (!pipelined && next < records.size()) {
      const size_t first = next;
      size_t stop = records.size();
      if (stats->local_trees) {
        count = 0;
        if (auto s = CohortStop(first, &stop); !s.ok())
          return s;
      }
      // Eligibility remains a feature-producer boundary, not a fitting
      // barrier. Every owner outlives the joined fits and its token generation.
      while (next < stop) {
        auto &batch = batches[count];
        bool admission_limited = false;
        if (auto s = Prepare(next, stop, &batch, &admission_limited); !s.ok()) {
          if (stats->local_trees && count && admission_limited) {
            ++stats->feature_memory_limited_cohorts;
            stop = next;
            break;
          }
          return s;
        }
        if (batch.stop <= next)
          return Status::Internal("Histogram batch did not advance");
        next = batch.stop;
        ++count;
      }
      ++stats->feature_cohorts;
      if (stats->local_trees) {
        const size_t length = stop - first;
        if (auto s = LearnLocalScopes(records.subspan(first, length), streams.subspan(first, length),
                                      options, budget, stats,
                                      std::span(features).subspan(first, length)); !s.ok())
          return s;
        for (size_t i = 0; i < count; ++i)
          if (auto s = Tokenize(batches[i]); !s.ok())
            return s;
      }
    }
    if (!stats->local_trees) {
      if (auto s = LearnGlobal(records, streams, population, options, budget, stats, features);
          !s.ok())
        return s;
      for (size_t i = 0; i < count; ++i)
        if (auto s = Tokenize(batches[i]); !s.ok())
          return s;
    }
    if (live_samples || live_bytes)
      return Status::Internal("Histogram features survived tokenization");
    if (active_modular_profile) {
      auto &profile = active_modular_profile->adaptive_gpu;
      profile.feature_preparations += stats->feature_batches;
      profile.feature_reuses += stats->token_batches;
      profile.feature_samples += stats->feature_samples;
      profile.learning_feature_uses += stats->feature_batches;
      profile.encoded_candidates += stats->token_batches;
      profile.feature_seconds += stats->feature_seconds;
      profile.token_seconds += stats->token_seconds;
    }
    return Status::Ok();
  }
};
} // namespace

Status LearnReferenceModularScopes(std::span<ReferenceModularStream> records,
                                   std::span<PreparedModularStreamView> streams,
                                   uint64_t population, const ReferenceWorkflowOptions &options,
                                   const ResourceBudget &budget, ReferenceWorkflowStats *stats,
                                   GpuTokenOwners *tokens) {
  if (options.gpu) {
    auto *op = QueryGpuModularStreamProcessing(*options.gpu);
    if (!op)
      return Status::Unsupported("Backend does not support histogram stream processing");
    return HybridLearning{records, streams, options, budget, stats, *op, *tokens}.Run(population);
  }
  if (stats->local_trees) {
    return LearnLocalScopes(records, streams, options, budget, stats);
  }
  return LearnGlobal(records, streams, population, options, budget, stats);
}
} // namespace gjxl::modular_internal::reference_workflow_internal
