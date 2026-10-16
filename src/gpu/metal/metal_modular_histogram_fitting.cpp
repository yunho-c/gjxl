// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/kernels/modular_histogram_fitting_params.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/ops/modular_histogram_fitting.h"
#include <cstring>

namespace gjxl::metal_internal {
using namespace histogram_gpu_internal;
using namespace resource_budget_internal;
namespace {
namespace wire = gjxl_histogram_fitting;
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
class Completed final : public PreparedGpuHistogramEvaluation {
public:
  BackendId backend_id() const noexcept override { return backend; }
  std::span<const HistogramPropertyResult> results() const noexcept override {
    return host;
  }
  BackendId backend = 0;
  std::unique_ptr<DeviceBuffer> buffer;
  std::span<const HistogramPropertyResult> host;
};
struct Pending {
  std::unique_ptr<GpuSubmission> submission;
  bool completed = false;
  ~Pending() {
    if (submission && !completed)
      DrainMetalSubmission(*submission);
  }
  Status Wait() {
    auto s = submission->Wait();
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
    launch(2, MTL::Size((plan.maximum_predictors * plan.maximum_symbols + 127) /
                            128,
                        count, 1));
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
  if (!out)
    return Status::InvalidArgument("Null histogram evaluation output");
  HistogramFittingEvaluationPlan plan;
  const auto limits = histogram_fitting_limits();
  if (auto s = ValidateHistogramFittingInput(input, id(), &plan, &limits);
      !s.ok())
    return s;
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
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<Completed>();
  candidate->backend = id();
  std::unique_ptr<DeviceBuffer> dispatch, scratch;
  Pending pending;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = Allocate(plan.output_bytes, &candidate->buffer); !s.ok())
      return s;
  }
  if (auto s = Allocate(plan.dispatch_bytes, &dispatch); !s.ok())
    return s;
  if (plan.scratch_bytes)
    if (auto s = Allocate(plan.scratch_bytes, &scratch); !s.ok())
      return s;
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
                  AsMetalBuffer(*candidate->buffer)->handle(),
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
  if (auto status = SubmitCompute("modular.histogram_fitting", Encode, &context,
                                  &pending.submission);
      !status.ok())
    return status;
  if (!pending.submission)
    return Status::Internal("Histogram fitting returned no submission");
  if (auto status = pending.Wait(); !status.ok())
    return status;
  uint32_t status_word;
  std::memcpy(&status_word, bytes + plan.status_offset, sizeof(status_word));
  if (auto status = CheckHistogramFittingDeviceStatus(status_word);
      !status.ok())
    return status;
  std::span<const std::byte> output;
  if (auto status = BorrowCompletedReadOnly(*candidate->buffer,
                                            plan.output_bytes, 0, &output);
      !status.ok())
    return status;
  candidate->host = {
      reinterpret_cast<const HistogramPropertyResult *>(output.data()),
      plan.properties};
  pending.submission.reset();
  scratch.reset();
  dispatch.reset();
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Histogram evaluation allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Histogram evaluation storage overflow");
}
} // namespace gjxl::metal_internal
