// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/kernels/modular_histogram_fitting_params.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/ops/modular_histogram_fitting.h"
#include "gpu/ops/modular_resident_histogram_fitting.h"
#include <chrono>
#include <cstring>

namespace gjxl::metal_internal {
using namespace histogram_gpu_internal;
using namespace resource_budget_internal;

// Completed operations obey the backend lifetime contract. Cache only after
// the caller releases the result view; no later batch can overwrite it early.
class MetalHistogramEvaluation final : public PreparedGpuHistogramEvaluation {
public:
  explicit MetalHistogramEvaluation(MetalBackend &b) : owner(b) {}
  ~MetalHistogramEvaluation() override {
    owner.ReleaseAqScratchArena(MetalAqScratchArena::kHistogramOutput,
                                std::move(buffer), reusable, prefer_larger);
  }
  BackendId backend_id() const noexcept override { return backend; }
  std::span<const HistogramPropertyResult> results() const noexcept override {
    return host;
  }
  MetalBackend &owner;
  BackendId backend = 0;
  DeviceScratchArena buffer;
  std::span<const HistogramPropertyResult> host;
  bool reusable = false;
  bool prefer_larger = false;
};
struct MetalHistogramScratchLease {
  MetalBackend &owner;
  DeviceScratchArena dispatch, scratch;
  bool reusable = false;
  bool prefer_larger = false;
  ~MetalHistogramScratchLease() { Release(); }
  void Release() noexcept {
    owner.ReleaseAqScratchArena(MetalAqScratchArena::kHistogramDispatch,
                                std::move(dispatch), reusable, prefer_larger);
    owner.ReleaseAqScratchArena(MetalAqScratchArena::kHistogramScratch,
                                std::move(scratch), reusable, prefer_larger);
    dispatch = DeviceScratchArena{};
    scratch = DeviceScratchArena{};
  }
};
namespace {
namespace wire = gjxl_histogram_fitting;
inline constexpr char kSubmissionId[] = "modular.histogram_fitting";
static_assert(sizeof(wire::Params) == 72);
static_assert(sizeof(wire::TaskDescriptor) == 8);
static_assert(static_cast<uint32_t>(modular_internal::Predictor::kWeighted) ==
              6);
class Prepared final : public PreparedGpuHistogramRows {
public:
  BackendId backend_id() const noexcept override { return backend; }
  std::span<const HistogramFittingShape> shapes() const noexcept override {
    return metadata;
  }
  BackendId backend = 0;
  HistogramFittingPreparationPlan plan;
  ManagedVector<HistogramFittingShape> metadata;
  std::unique_ptr<DeviceBuffer> buffer;
};
struct Pending {
  std::unique_ptr<GpuSubmission> submission;
  bool completed = false;
  ~Pending() {
    if (submission && !completed)
      DrainMetalSubmission(*submission);
  }
  Status Wait(HistogramFittingWaitTrace *trace = nullptr) {
    auto s = trace ? WaitMetalSubmissionTraced(*submission, trace)
                   : submission->Wait();
    completed = true;
    return s;
  }
};
struct Context {
  MTL::Buffer *prepared, *dispatch, *output, *scratch;
  std::array<MTL::ComputePipelineState *, 5> pipelines;
  const HistogramFittingEvaluationPlan *plan;
  wire::Params params;
};
struct ProfileContext {
  const Context *context;
  size_t phase, first, count;
};
size_t Waves(const HistogramFittingEvaluationPlan &plan) {
  return plan.tasks ? (plan.tasks + plan.active_properties - 1) /
                          plan.active_properties
                    : 0;
}
size_t PrefixGroups(const Context &c) {
  const auto &plan = *c.plan;
  const size_t columns =
      plan.maximum_predictors * (plan.maximum_symbols + 1) + 1;
  const size_t simd_groups = 128 / c.pipelines[2]->threadExecutionWidth();
  return (columns + simd_groups - 1) / simd_groups;
}
void EncodeProfilePhase(MetalBackend &, MTL::ComputeCommandEncoder *encoder,
                        const void *opaque) {
  const auto &stage = *static_cast<const ProfileContext *>(opaque);
  const auto &c = *stage.context;
  const auto &plan = *c.plan;
  auto params = c.params;
  params.first_task = static_cast<uint32_t>(stage.first);
  params.tasks = static_cast<uint32_t>(stage.count);
  encoder->setBuffer(c.prepared, 0, 0);
  encoder->setBuffer(c.dispatch, 0, 1);
  encoder->setBuffer(c.output, 0, 2);
  if (c.scratch)
    encoder->setBuffer(c.scratch, 0, 3);
  const size_t tile_bins =
      plan.tasks ? std::min(size_t{256},
                            wire::kHistogramTileCells / plan.maximum_symbols)
                 : 1;
  const size_t tiles = (plan.maximum_bins + tile_bins - 1) / tile_bins;
  const std::array<MTL::Size, 5> grids = {
      MTL::Size(plan.nodes, 1, 1),
      MTL::Size(plan.histogram_shards * tiles, plan.maximum_predictors,
                stage.count),
      MTL::Size(PrefixGroups(c), stage.count, 1),
      MTL::Size((plan.maximum_bins * plan.maximum_predictors + 127) / 128,
                stage.count, 1),
      MTL::Size(stage.count, 1, 1)};
  encoder->setBytes(&params, sizeof(params), 4);
  encoder->setComputePipelineState(c.pipelines[stage.phase]);
  RecordMetalComputePipelineState(c.pipelines[stage.phase]);
  DispatchMetalThreadgroups(encoder, grids[stage.phase], MTL::Size(128, 1, 1));
  const MTL::Resource *dependencies[] = {c.dispatch, c.output, c.scratch};
  encoder->memoryBarrier(dependencies, stage.phase == 0 ? 2 : 3);
}
struct EncoderSequence {
  const Context *context;
  mutable size_t next = 0;
};
void EncodeSeparate(MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
                    const void *opaque) {
  const auto &sequence = *static_cast<const EncoderSequence *>(opaque);
  const size_t index = sequence.next++;
  const auto &plan = *sequence.context->plan;
  const size_t first = index ? ((index - 1) / 4) * plan.active_properties : 0;
  const ProfileContext stage{
      sequence.context, index ? 1 + (index - 1) % 4 : 0, first,
      index ? std::min(plan.active_properties, plan.tasks - first) : 0};
  EncodeProfilePhase(backend, encoder, &stage);
}
struct WallProfile {
  using Clock = std::chrono::steady_clock;
  bool enabled;
  HistogramFittingBatchProfile value;
  Clock::time_point last;
  explicit WallProfile(bool active) : enabled(active) {
    if (enabled)
      last = Clock::now();
  }
  void Lap(size_t phase) {
    if (!enabled)
      return;
    const auto now = Clock::now();
    value.wall_nanoseconds[phase] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - last)
            .count();
    last = now;
  }
};
void Encode(MetalBackend &, MTL::ComputeCommandEncoder *encoder,
            const void *opaque) {
  const auto &c = *static_cast<const Context *>(opaque);
  const auto &plan = *c.plan;
  auto params = c.params;
  encoder->setBuffer(c.prepared, 0, 0);
  encoder->setBuffer(c.dispatch, 0, 1);
  encoder->setBuffer(c.output, 0, 2);
  if (c.scratch)
    encoder->setBuffer(c.scratch, 0, 3);
  const auto launch = [&](size_t pipeline, MTL::Size groups) {
    encoder->setBytes(&params, sizeof(params), 4);
    encoder->setComputePipelineState(c.pipelines[pipeline]);
    RecordMetalComputePipelineState(c.pipelines[pipeline]);
    DispatchMetalThreadgroups(encoder, groups, MTL::Size(128, 1, 1));
  };
  launch(0, MTL::Size(plan.nodes, 1, 1));
  const MTL::Resource *initial[] = {c.dispatch, c.output};
  encoder->memoryBarrier(initial, 2);
  if (!plan.tasks)
    return;
  const MTL::Resource *dependencies[] = {c.dispatch, c.output, c.scratch};
  for (size_t first = 0; first < plan.tasks; first += plan.active_properties) {
    const size_t count = std::min(plan.active_properties, plan.tasks - first);
    params.first_task = static_cast<uint32_t>(first);
    params.tasks = static_cast<uint32_t>(count);
    const size_t tile_bins =
        std::min(size_t{256}, wire::kHistogramTileCells / plan.maximum_symbols);
    const size_t tiles = (plan.maximum_bins + tile_bins - 1) / tile_bins;
    launch(1, MTL::Size(plan.histogram_shards * tiles, plan.maximum_predictors,
                        count));
    encoder->memoryBarrier(dependencies, 3);
    launch(2, MTL::Size(PrefixGroups(c), count, 1));
    encoder->memoryBarrier(dependencies, 3);
    launch(3,
           MTL::Size((plan.maximum_bins * plan.maximum_predictors + 127) / 128,
                     count, 1));
    encoder->memoryBarrier(dependencies, 3);
    launch(4, MTL::Size(count, 1, 1));
    encoder->memoryBarrier(dependencies, 3);
  }
}
} // namespace

HistogramFittingDeviceLimits
MetalBackend::histogram_fitting_limits() const noexcept {
  const size_t buffer = std::min(
      size_t{UINT32_MAX}, static_cast<size_t>(device_->maxBufferLength()));
  return {kMaximumHistogramFittingFits, kMaximumHistogramFittingRows,
          kMaximumHistogramFittingFits, buffer,
          std::min(buffer, size_t{256 * 1024 * 1024})};
}

std::array<size_t, 3> MetalBackend::HistogramFittingBufferCapacities(
    const HistogramFittingEvaluationPlan &plan,
    HistogramFittingExecutionOptions options) {
  std::array capacities{plan.output_bytes, plan.dispatch_bytes, plan.scratch_bytes};
  if (!options.reuse_buffers || !options.reuse_larger_buffers) return capacities;
  const std::array kinds{MetalAqScratchArena::kHistogramOutput,
                         MetalAqScratchArena::kHistogramDispatch,
                         MetalAqScratchArena::kHistogramScratch};
  std::lock_guard lock(preparation_cache_mutex_);
  for (size_t i = 0; i < kinds.size(); ++i) {
    const auto &idle = idle_aq_scratch_[static_cast<size_t>(kinds[i])];
    if (capacities[i] && idle &&
        (i != 2 || idle->capacity_bytes() <= options.maximum_scratch_bytes))
      capacities[i] = std::max(capacities[i], idle->capacity_bytes());
  }
  return capacities;
}

Status MetalBackend::HistogramFittingProfileStorageBound(
    const HistogramFittingEvaluationPlan &plan,
    HistogramFittingProfileMode mode, HostStorageBound *out) const {
  if (!out || !plan.nodes || plan.nodes > kMaximumHistogramFittingFits ||
      plan.tasks > kMaximumHistogramFittingFits * 16 ||
      (plan.tasks && !plan.active_properties) ||
      (mode != HistogramFittingProfileMode::kDisabled &&
       mode != HistogramFittingProfileMode::kWall &&
       mode != HistogramFittingProfileMode::kStage))
    return Status::InvalidArgument("Invalid histogram profile plan");
  HostStorageBound bound;
  if (mode == HistogramFittingProfileMode::kStage) {
    const size_t stages = 1 + 4 * Waves(plan);
    gpu_profile_internal::SubmissionProfileStoragePlan profile;
    // Registered kernel labels are bounded by the common Metal registry.
    if (auto s = gpu_profile_internal::ComputeSubmissionProfileStoragePlan(
            {stages, stages, 8, sizeof(kSubmissionId) - 1,
             kMaximumKernelIdBytes, sizeof(kSubmissionId) - 1},
            &profile);
        !s.ok())
      return s;
    profile.resolution.peak_bytes =
        std::max(profile.resolution.peak_bytes, profile.recorded.peak_bytes);
    if (!bound.AddVector<ProfileContext>(stages,
                                         VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<MetalProfiledComputeStage>(
            stages, VectorCapacityPolicy::kFreshExact) ||
        !bound.Add(profile.resolution))
      return Status::OutOfMemory("Histogram profiling storage overflow");
  }
  *out = bound;
  return Status::Ok();
}

Status MetalBackend::PrepareHistogramRows(
    std::span<const HistogramFittingSource> sources,
    std::unique_ptr<PreparedGpuHistogramRows> *out) try {
  if (!out || sources.empty())
    return Status::InvalidArgument(
        "Invalid histogram preparation output/sources");
  if (sources.size() > kMaximumHistogramFittingFits)
    return Status::Unsupported("Histogram preparation cohort is too large");
  std::array<HistogramFittingShape, kMaximumHistogramFittingFits> shapes;
  for (size_t i = 0; i < sources.size(); ++i)
    if (auto s = ResolveHistogramFittingSource(sources[i], &shapes[i]); !s.ok())
      return s;
  const auto view = std::span(shapes).first(sources.size());
  HistogramFittingPreparationPlan plan;
  const auto limits = histogram_fitting_limits();
  if (auto s = ComputeHistogramFittingPreparationPlan(view, &plan, &limits);
      !s.ok())
    return s;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<Prepared>();
  candidate->backend = id();
  candidate->plan = plan;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    candidate->metadata.assign(view.begin(), view.end());
    if (auto s = Allocate(plan.storage_bytes, &candidate->buffer); !s.ok())
      return s;
  }
  auto *bytes =
      static_cast<std::byte *>(AsMetalBuffer(*candidate->buffer)->contents());
  auto *descriptors = reinterpret_cast<wire::FitDescriptor *>(bytes);
  size_t offset = plan.columns_offset;
  for (size_t i = 0; i < sources.size(); ++i) {
    const auto &shape = shapes[i];
    const auto &samples = *sources[i].samples;
    HistogramFittingColumnLayout layout;
    if (auto s = ComputeHistogramFittingColumnLayout(shape, &layout); !s.ok())
      return s;
    wire::FitDescriptor descriptor{};
    descriptor.rows = static_cast<uint32_t>(shape.rows);
    descriptor.population = static_cast<uint32_t>(shape.population);
    descriptor.predictors = static_cast<uint32_t>(shape.predictors);
    descriptor.properties = static_cast<uint32_t>(shape.properties);
    descriptor.static_properties =
        static_cast<uint32_t>(shape.static_properties);
    descriptor.source_id = shape.source_id;
    descriptor.weights_offset =
        static_cast<uint32_t>(offset + layout.weights_offset);
    std::memcpy(bytes + descriptor.weights_offset, samples.weights.data(),
                samples.weights.size() * sizeof(uint16_t));
    for (size_t pred = 0; pred < shape.predictors; ++pred) {
      descriptor.predictor_ids[pred] =
          static_cast<uint32_t>(shape.predictor_ids[pred]);
      descriptor.residual_offsets[pred] =
          static_cast<uint32_t>(offset + layout.residual_offsets[pred]);
      std::memcpy(bytes + descriptor.residual_offsets[pred],
                  samples.residuals[pred].data(),
                  shape.rows *
                      sizeof(modular_internal::HistogramResidualToken));
    }
    for (size_t prop = 0; prop < shape.properties; ++prop) {
      descriptor.property_ids[prop] = shape.property_ids[prop];
      descriptor.bins[prop] = static_cast<uint32_t>(shape.bins[prop]);
      descriptor.property_offsets[prop] =
          static_cast<uint32_t>(offset + layout.property_offsets[prop]);
      const bool stat = prop < shape.static_properties;
      const void *column =
          stat ? static_cast<const void *>(samples.static_values[prop].data())
               : static_cast<const void *>(
                     samples.values[prop - shape.static_properties].data());
      std::memcpy(bytes + descriptor.property_offsets[prop], column,
                  shape.rows * (stat ? sizeof(uint32_t) : sizeof(uint8_t)));
    }
    descriptors[i] = descriptor;
    offset += layout.bytes;
  }
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Histogram preparation allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Histogram preparation storage overflow");
}

Status MetalBackend::EvaluateHistogramNodes(
    const HistogramFittingInput &input,
    std::unique_ptr<PreparedGpuHistogramEvaluation> *out) try {
  const auto mode = input.profile_mode;
  WallProfile profile(mode != HistogramFittingProfileMode::kDisabled);
  if (input.trace_wait && !profile.enabled)
    return Status::InvalidArgument("Wait tracing requires histogram profiling");
  if ((profile.enabled && !input.profile) ||
      (mode != HistogramFittingProfileMode::kDisabled &&
       mode != HistogramFittingProfileMode::kWall &&
       mode != HistogramFittingProfileMode::kStage))
    return Status::InvalidArgument("Invalid histogram profile request");
  if (!out)
    return Status::InvalidArgument("Null histogram evaluation output");
  HistogramFittingEvaluationPlan plan;
  const auto limits = histogram_fitting_limits();
  if (auto s = ValidateHistogramFittingInput(input, id(), &plan, &limits);
      !s.ok())
    return s;
  HostStorageBound capacity_bound;
  if (auto s = ComputeHistogramFittingCapacityBound(
          plan, input.buffer_capacity_limits, &capacity_bound); !s.ok())
    return s;
  if (input.options.reuse_larger_buffers && !input.options.reuse_buffers)
    return Status::InvalidArgument("Larger histogram buffers require reuse");
  if (input.buffer_capacity_limits[2] > input.options.maximum_scratch_bytes)
    return Status::InvalidArgument("Histogram scratch capacity exceeds limit");
  const auto *prepared = dynamic_cast<const Prepared *>(input.prepared);
  if (!prepared ||
      plan.prepared_bytes != prepared->plan.retained.retained_bytes)
    return Status::InvalidArgument(
        "Histogram evaluation requires its native prepared owner");
  const MetalBuffer *source = nullptr;
  if (auto s = RequireMetalBuffer(prepared->buffer.get(),
                                  prepared->plan.storage_bytes, 0,
                                  "Histogram prepared rows", &source);
      !s.ok())
    return s;
  profile.Lap(0);
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<MetalHistogramEvaluation>(*this);
  candidate->backend = id();
  candidate->prefer_larger = input.options.reuse_larger_buffers;
  MetalHistogramScratchLease backing{*this};
  backing.prefer_larger = input.options.reuse_larger_buffers;
  Pending pending;
  const auto allocate = [&](MetalAqScratchArena kind, size_t bytes,
                            DeviceScratchArena *arena, size_t index) {
    bool reused = false;
    const size_t maximum = input.options.reuse_larger_buffers
        ? input.buffer_capacity_limits[index] : bytes;
    const auto status = input.options.reuse_buffers
        ? AcquireAqScratchArena(kind, bytes, arena, &reused, maximum)
        : arena->Prepare(*this, bytes);
    if (!status.ok()) return status;
    profile.value.buffer_requested_bytes[index] = bytes;
    profile.value.buffer_capacity_bytes[index] = arena->capacity_bytes();
    if (reused) {
      profile.value.reused_buffers |= uint32_t{1} << index;
      profile.value.reused_buffer_bytes += bytes;
    }
    return status;
  };
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = allocate(MetalAqScratchArena::kHistogramOutput,
                          plan.output_bytes, &candidate->buffer, 0); !s.ok())
      return s;
  }
  if (auto s = allocate(MetalAqScratchArena::kHistogramDispatch,
                        plan.dispatch_bytes, &backing.dispatch, 1); !s.ok())
    return s;
  if (plan.scratch_bytes)
    if (auto s = allocate(MetalAqScratchArena::kHistogramScratch,
                          plan.scratch_bytes, &backing.scratch, 2); !s.ok())
      return s;
  auto *dispatch = backing.dispatch.backing_buffer();
  auto *scratch = backing.scratch.backing_buffer();
  auto *result_buffer = candidate->buffer.backing_buffer();
  profile.Lap(1);
  auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
  std::memset(bytes, 0, plan.dispatch_bytes);
  auto *descriptors = reinterpret_cast<wire::NodeDescriptor *>(bytes);
  auto *tasks =
      reinterpret_cast<wire::TaskDescriptor *>(bytes + plan.tasks_offset);
  size_t indices = plan.indices_offset, counts = plan.parent_counts_offset,
         extra = plan.parent_extra_offset,
         penalties = plan.predictor_penalties_offset, result_first = 0,
         task_count = 0;
  for (size_t i = 0; i < input.nodes.size(); ++i) {
    const auto &node = input.nodes[i];
    const auto &n = node.shape;
    const auto &f = prepared->metadata[n.fit];
    descriptors[i] = {static_cast<uint32_t>(n.fit),
                      n.node_id,
                      static_cast<uint32_t>(n.rows),
                      static_cast<uint32_t>(n.symbols),
                      static_cast<uint32_t>(indices),
                      static_cast<uint32_t>(counts),
                      static_cast<uint32_t>(extra),
                      static_cast<uint32_t>(result_first),
                      n.inherited_constants,
                      static_cast<uint32_t>(n.parent_predictor),
                      static_cast<uint32_t>(penalties),
                      0};
    std::memcpy(bytes + indices, node.rows.data(), node.rows.size_bytes());
    std::memcpy(bytes + counts, node.parent_counts.data(),
                node.parent_counts.size_bytes());
    std::memcpy(bytes + extra, node.parent_extra_bits.data(),
                node.parent_extra_bits.size_bytes());
    if (auto s = ComputeHistogramFittingPredictorPenalties(
            std::span(f.predictor_ids).first(f.predictors), n.parent_predictor,
            n.change_predictor_penalty,
            {reinterpret_cast<float *>(bytes + penalties), f.predictors});
        !s.ok())
      return s;
    for (size_t prop = 0; prop < f.properties; ++prop)
      if (!(n.inherited_constants & (uint32_t{1} << prop)) && f.bins[prop] > 1)
        tasks[task_count++] = {static_cast<uint32_t>(i),
                               static_cast<uint32_t>(prop)};
    indices += node.rows.size_bytes();
    counts += node.parent_counts.size_bytes();
    extra += node.parent_extra_bits.size_bytes();
    penalties += f.predictors * sizeof(float);
    result_first += f.properties;
  }
  const auto &s = plan.scratch;
  Context context{source->handle(),
                  AsMetalBuffer(*dispatch)->handle(),
                  AsMetalBuffer(*result_buffer)->handle(),
                  scratch ? AsMetalBuffer(*scratch)->handle() : nullptr,
                  {},
                  &plan,
                  {static_cast<uint32_t>(plan.nodes),
                   static_cast<uint32_t>(plan.tasks_offset),
                   static_cast<uint32_t>(plan.status_offset),
                   static_cast<uint32_t>(plan.histogram_shards), 0, 0,
                   static_cast<uint32_t>(s.bytes),
                   static_cast<uint32_t>(plan.maximum_bins),
                   static_cast<uint32_t>(plan.maximum_predictors),
                   static_cast<uint32_t>(plan.maximum_symbols),
                   static_cast<uint32_t>(s.shard_histograms_offset),
                   static_cast<uint32_t>(s.shard_extra_offset),
                   static_cast<uint32_t>(s.shard_occupancy_offset),
                   static_cast<uint32_t>(s.histograms_offset),
                   static_cast<uint32_t>(s.extra_offset),
                   static_cast<uint32_t>(s.occupancy_offset),
                   static_cast<uint32_t>(s.candidate_costs_offset),
                   static_cast<uint32_t>(s.side_winners_offset)}};
  for (size_t i = 0; i < context.pipelines.size(); ++i)
    context.pipelines[i] = histogram_fitting_pipelines_[i].get();
  profile.Lap(2);
  if (mode == HistogramFittingProfileMode::kStage) {
    gpu_profile_internal::ProfileStorage<ProfileContext> contexts;
    gpu_profile_internal::ProfileStorage<MetalProfiledComputeStage> stages;
    const size_t count = 1 + 4 * Waves(plan);
    contexts.reserve(count);
    stages.reserve(count);
    const auto append = [&](size_t phase, size_t first, size_t tasks) {
      contexts.push_back({&context, phase, first, tasks});
      stages.push_back({kHistogramFittingGpuStages[phase],
                        "modular.histogram_fitting",
                        static_cast<uint32_t>(first), 0, EncodeProfilePhase,
                        &contexts.back()});
    };
    append(0, 0, 0);
    for (size_t first = 0; first < plan.tasks; first += plan.active_properties)
      for (size_t phase = 1; phase < 5; ++phase)
        append(phase, first,
               std::min(plan.active_properties, plan.tasks - first));
    if (auto s = SubmitComputeProfiled(
            "modular.histogram_fitting", stages,
            gpu_profile_internal::GpuProfilingMode::kStage,
            &pending.submission, input.trace_wait);
        !s.ok())
      return s;
  } else if (input.options.multiple_encoders) {
    const EncoderSequence sequence{&context};
    if (auto s = SubmitCompute("modular.histogram_fitting", EncodeSeparate,
                               &sequence, &pending.submission, input.trace_wait,
                               1 + 4 * Waves(plan)); !s.ok())
      return s;
  } else {
    if (auto s = SubmitCompute("modular.histogram_fitting", Encode, &context,
                               &pending.submission, input.trace_wait);
        !s.ok())
      return s;
  }
  profile.Lap(3);
  if (!pending.submission)
    return Status::Internal("Histogram fitting returned no submission");
  if (auto status = pending.Wait(input.trace_wait ? &profile.value.wait_trace : nullptr);
      !status.ok())
    return status;
  profile.Lap(4);
  uint32_t status_word;
  std::memcpy(&status_word, bytes + plan.status_offset, sizeof(status_word));
  if (auto status = CheckHistogramFittingDeviceStatus(status_word);
      !status.ok())
    return status;
  std::span<const std::byte> output;
  if (auto status = BorrowCompletedReadOnly(*result_buffer,
                                            plan.output_bytes, 0, &output);
      !status.ok())
    return status;
  candidate->host = {
      reinterpret_cast<const HistogramPropertyResult *>(output.data()),
      plan.properties};
  profile.Lap(5);
  if (mode == HistogramFittingProfileMode::kStage) {
    gpu_profile_internal::GpuSubmissionProfile resolved;
    if (auto s = GetMetalSubmissionGpuProfile(*pending.submission, &resolved);
        !s.ok())
      return s;
    if (resolved.stages.size() != 1 + 4 * Waves(plan))
      return Status::Internal("Histogram profile stage coverage differs");
    for (size_t i = 0; i < resolved.stages.size(); ++i) {
      const size_t phase = i ? 1 + (i - 1) % 4 : 0;
      const auto &stage = resolved.stages[i];
      if (!stage.timestamp_valid ||
          stage.stage_id != kHistogramFittingGpuStages[phase] ||
          stage.dispatches.size() != 1)
        return Status::DeviceError("Histogram profile stage is incomplete");
      profile.value.gpu_nanoseconds[phase] += stage.gpu_nanoseconds;
      ++profile.value.dispatches[phase];
    }
    profile.value.command_buffer_gpu_nanoseconds =
        resolved.command_buffer_gpu_nanoseconds;
  } else if (profile.enabled) {
    // Completed command-buffer timing needs no counter samples or additional
    // encoder boundaries, so wall mode also measures the ordinary device span.
    if (auto s = GetMetalSubmissionGpuDuration(
            *pending.submission, &profile.value.command_buffer_gpu_nanoseconds);
        !s.ok())
      return s;
  }
  profile.Lap(6);
  pending.submission.reset();
  backing.reusable = input.options.reuse_buffers;
  backing.Release();
  candidate->reusable = input.options.reuse_buffers;
  profile.Lap(7);
  if (profile.enabled) {
    profile.value.nodes = plan.nodes;
    profile.value.rows = plan.rows;
    profile.value.tasks = plan.tasks;
    profile.value.active_properties = plan.active_properties;
    profile.value.waves = Waves(plan);
    profile.value.maximum_bins = plan.maximum_bins;
    profile.value.maximum_symbols = plan.maximum_symbols;
    profile.value.scratch_bytes = plan.scratch_bytes;
    profile.value.compute_encoders =
        mode == HistogramFittingProfileMode::kStage || input.options.multiple_encoders
            ? 1 + 4 * Waves(plan) : 1;
    *input.profile = profile.value;
  }
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Histogram evaluation allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Histogram evaluation storage overflow");
}
Status MetalBackend::FitHistogramTrees(const ResidentHistogramInput &input,
                                      ResidentHistogramResult *out) try {
  if (!out) return Status::InvalidArgument("Null resident histogram output");
  ResidentHistogramPlan plan;
  const auto limits = histogram_fitting_limits();
  if (auto s = ComputeResidentHistogramPlan(input, &plan, &limits); !s.ok()) return s;
  auto *pipeline = histogram_fitting_pipelines_[5].get();
  if (pipeline->threadExecutionWidth() != 32 ||
      pipeline->maxTotalThreadsPerThreadgroup() < 128)
    return Status::Unsupported("Resident histogram prototype requires four 32-lane SIMD groups");
  const ManagedHostScope resources(ResourceClass::kPreparation);
  ResidentHistogramResult candidate;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    candidate.trees.resize(plan.fits);
    for (size_t i = 0; i < plan.fits; ++i)
      candidate.trees[i].nodes.resize(plan.layout[i].maximum_nodes);
    if (input.capture_row_order) {
      candidate.row_order.resize(plan.rows);
      candidate.row_offsets.resize(plan.fits + 1);
    }
  }
  std::array<HistogramFittingSource, kMaximumResidentHistogramFits> sources;
  for (size_t i = 0; i < plan.fits; ++i) sources[i] = input.requests[i].source;
  std::unique_ptr<PreparedGpuHistogramRows> prepared;
  if (auto s = PrepareHistogramRows(std::span(sources).first(plan.fits), &prepared); !s.ok()) return s;
  auto *rows = dynamic_cast<Prepared *>(prepared.get());
  if (!rows) return Status::Internal("Missing resident histogram prepared owner");
  std::unique_ptr<DeviceBuffer> work;
  if (auto s = Allocate(plan.work_bytes, &work); !s.ok()) return s;
  auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*work)->contents());
  std::memcpy(bytes, plan.layout.data(), plan.fits * sizeof(plan.layout[0]));
  for (size_t i = 0; i < plan.fits; ++i) {
    const auto &r = input.requests[i];
    const auto &samples = *r.source.samples;
    const auto &d = plan.layout[i];
    for (size_t p = 0; p < samples.property_count; ++p)
      if (!samples.thresholds[p].empty())
        std::memcpy(bytes + d.thresholds[p], samples.thresholds[p].data(),
                    samples.thresholds[p].size() * sizeof(int32_t));
    // Native predictor penalties contain deliberate double-literal rounding.
    // Prepare this small policy table once; GPU node state chooses its row.
    const float change = 800.0f / (100.0f + r.learning.split_threshold);
    for (size_t p = 0; p < samples.predictor_count; ++p)
      if (auto s = ComputeHistogramFittingPredictorPenalties(
              std::span(samples.predictors).first(samples.predictor_count), samples.predictors[p],
              change, {reinterpret_cast<float *>(bytes + d.penalties) + p * samples.predictor_count,
                       samples.predictor_count}); !s.ok()) return s;
  }
  struct ResidentContext {
    MTL::Buffer *prepared, *work;
    MTL::ComputePipelineState *pipeline;
    size_t fits;
  } context{AsMetalBuffer(*rows->buffer)->handle(), AsMetalBuffer(*work)->handle(), pipeline, plan.fits};
  const auto encode = [](MetalBackend &, MTL::ComputeCommandEncoder *encoder, const void *opaque) {
    const auto &c = *static_cast<const ResidentContext *>(opaque);
    encoder->setBuffer(c.prepared, 0, 0);
    encoder->setBuffer(c.work, 0, 1);
    encoder->setComputePipelineState(c.pipeline);
    RecordMetalComputePipelineState(c.pipeline);
    DispatchMetalThreadgroups(encoder, MTL::Size(c.fits, 1, 1), MTL::Size(128, 1, 1));
  };
  Pending pending;
  if (auto s = SubmitCompute("modular.histogram_resident", encode, &context, &pending.submission); !s.ok()) return s;
  if (!pending.submission) return Status::Internal("Missing resident histogram submission");
  if (auto s = pending.Wait(); !s.ok()) return s;
  if (auto s = GetMetalSubmissionGpuDuration(*pending.submission, &candidate.gpu_nanoseconds); !s.ok()) return s;
  std::span<const std::byte> completed;
  if (auto s = BorrowCompletedReadOnly(*work, plan.work_bytes, 0, &completed); !s.ok()) return s;
  size_t first_row = 0;
  for (size_t i = 0; i < plan.fits; ++i) {
    const auto &d = plan.layout[i];
    const auto &state = *reinterpret_cast<const gjxl_histogram_resident::State *>(completed.data() + d.state);
    if (state.status == gjxl_histogram_resident::kCapacityError)
      return Status::Unsupported("Resident histogram fit exceeds tree capacity or depth");
    if (state.status || state.pending || !state.size || state.size > d.maximum_nodes || state.visited > state.size)
      return Status::DeviceError("Resident histogram state or partition failed");
    candidate.visited_nodes += state.visited;
    const auto *nodes = reinterpret_cast<const gjxl_histogram_resident::Node *>(completed.data() + d.nodes);
    auto &tree = candidate.trees[i];
    tree.size = state.size;
    for (size_t n = 0; n < tree.size; ++n) {
      if (nodes[n].left >= tree.size || nodes[n].right >= tree.size || nodes[n].predictor >= 14)
        return Status::DeviceError("Resident histogram produced invalid tree links");
      tree.nodes[n] = {nodes[n].property, nodes[n].split,
                      static_cast<modular_internal::TreeIndex>(nodes[n].left),
                      static_cast<modular_internal::TreeIndex>(nodes[n].right),
                      static_cast<modular_internal::Predictor>(nodes[n].predictor), 0, 1};
    }
    modular_internal::TreeLayout layout;
    if (auto s = modular_internal::ValidateTree(tree, &layout); !s.ok()) return s;
    if (tree.size <= modular_internal::kInlineTreeNodes) tree.nodes.resize(tree.size);
    if (input.capture_row_order) {
      const size_t count = input.requests[i].source.samples->weights.size();
      candidate.row_offsets[i] = first_row;
      if (count) std::memcpy(candidate.row_order.data() + first_row, completed.data() + d.order, count * sizeof(uint32_t));
      first_row += count;
      candidate.row_offsets[i + 1] = first_row;
    }
  }
  candidate.device_bytes = plan.prepared.storage_bytes + plan.work_bytes;
  candidate.planned_peak_bytes = plan.working.peak_bytes;
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Resident histogram fitting allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Resident histogram fitting storage overflow");
}
} // namespace gjxl::metal_internal
