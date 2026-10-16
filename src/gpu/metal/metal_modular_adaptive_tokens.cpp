// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codec/modular/weighted_feature_math.h"
#include "gpu/metal/kernels/histogram_sample_rows_params.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/ops/modular_adaptive_layout.h"
#include <cstring>

namespace gjxl::metal_internal {
using namespace modular_gpu_internal;
using namespace modular_internal;
using namespace resource_budget_internal;
namespace {
using Params = gjxl_modular_adaptive::Params;
using Slice = gjxl_modular_adaptive::Slice;
using Feature = gjxl_modular_adaptive::WeightedFeature;
using Node = gjxl_modular_adaptive::Node;
constexpr size_t kGuard = 256;
static_assert(sizeof(Params) == 40 && alignof(Params) == 8);
static_assert(sizeof(Slice) == sizeof(AdaptiveModularTokenSlice) && alignof(Slice) == 8);
static_assert(sizeof(Feature) == sizeof(AdaptiveModularWeightedFeature) && alignof(Feature) == 8);
static_assert(sizeof(Node) == sizeof(ModularTokenNode));
static_assert(sizeof(gjxl_weighted_features::Parameters) == 44 &&
              alignof(gjxl_weighted_features::Parameters) == 4);
static_assert(sizeof(gjxl_weighted_features::State) == 20);
static_assert(sizeof(AdaptiveModularDiagnostic) == 272 &&
              offsetof(AdaptiveModularDiagnostic, leaf) == 248);
#define OFFSET(A, B, f) static_assert(offsetof(A, f) == offsetof(B, f))
OFFSET(Slice, AdaptiveModularTokenSlice, token_offset);
OFFSET(Slice, AdaptiveModularTokenSlice, plane_offset_elements);
OFFSET(Slice, AdaptiveModularTokenSlice, plane_stride_elements);
OFFSET(Slice, AdaptiveModularTokenSlice, stream_id);
OFFSET(Slice, AdaptiveModularTokenSlice, property_channel);
OFFSET(Slice, AdaptiveModularTokenSlice, width);
OFFSET(Slice, AdaptiveModularTokenSlice, height);
OFFSET(Slice, AdaptiveModularTokenSlice, reserved);
OFFSET(Feature, AdaptiveModularWeightedFeature, prediction);
OFFSET(Feature, AdaptiveModularWeightedFeature, property15);
#undef OFFSET
bool GuardIntact(const std::byte *p, size_t count) {
  return std::all_of(p, p + count, [](std::byte b) { return b == std::byte{0xa5}; });
}
class CompletedFeatures final : public PreparedGpuAdaptiveModularFeatures {
public:
  BackendId backend_id() const noexcept override { return backend; }
  const codec_internal::ImageMetadata &metadata() const noexcept override { return source; }
  size_t token_count() const noexcept override { return layout.tokens; }
  std::span<const AdaptiveModularTokenSlice> slices() const noexcept override {
    return descriptors;
  }
  bool has_weighted_features() const noexcept override { return weighted != nullptr; }
  Status
  BorrowWeightedHostView(std::span<const AdaptiveModularWeightedFeature> *out) const override {
    if (!out)
      return Status::InvalidArgument("Null adaptive Modular weighted view");
    *out = host_weighted;
    return Status::Ok();
  }
  BackendId backend;
  codec_internal::ImageMetadata source;
  AdaptiveModularLayout layout;
  ManagedVector<AdaptiveModularTokenSlice> descriptors;
  std::unique_ptr<DeviceBuffer> planes, weighted;
  std::span<const AdaptiveModularWeightedFeature> host_weighted;
};
class CompletedTokens final : public PreparedGpuModularTokens {
public:
  CompletedTokens(BackendId backend, AdaptiveModularTokenStoragePlan plan, size_t contexts)
      : backend_(backend), plan_(plan), contexts_(contexts) {}
  BackendId backend_id() const noexcept override { return backend_; }
  size_t token_count() const noexcept override { return plan_.token_count; }
  size_t context_count() const noexcept override { return contexts_; }
  DeviceMemoryRange values() const noexcept override {
    return {buffer.get(), 0, plan_.values_bytes};
  }
  DeviceMemoryRange contexts() const noexcept override {
    return {buffer.get(), plan_.contexts_offset, plan_.token_count * sizeof(uint16_t)};
  }
  Status BorrowHostView(ModularTokenHostView *out) const override {
    if (!out)
      return Status::InvalidArgument("Null adaptive Modular token view");
    *out = host;
    return Status::Ok();
  }
  std::unique_ptr<DeviceBuffer> buffer;
  ModularTokenHostView host;

private:
  BackendId backend_;
  AdaptiveModularTokenStoragePlan plan_;
  size_t contexts_;
};
class CompletedStreamFeatures final : public PreparedGpuModularStreamFeatures {
public:
  BackendId backend_id() const noexcept override { return backend; }
  size_t sample_count() const noexcept override { return plan.samples; }
  WeightedFeatureBackend producer() const noexcept override { return selected_producer; }
  std::span<const WeightedFeatureView> weighted_views() const noexcept override { return views; }
  BackendId backend;
  const MetalBackend *owner = nullptr;
  WeightedFeatureBackend selected_producer = WeightedFeatureBackend::kCpu;
  ModularStreamFeatureStoragePlan plan;
  ManagedVector<AdaptiveModularTokenSlice> slices;
  ManagedVector<WeightedFeatureView> views;
  std::unique_ptr<DeviceBuffer> planes, weighted;
};
struct Pending {
  std::unique_ptr<GpuSubmission> submission;
  bool completed = false;
  ~Pending() {
    if (submission && !completed)
      DrainMetalSubmission(*submission);
  }
  Status Wait() {
    const auto s = submission->Wait();
    completed = true;
    return s;
  }
};
struct RowTimer {
  double *seconds;
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  ~RowTimer() {
    if (seconds)
      *seconds += std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  }
};
struct RowPlan {
  size_t output_bytes, slices_offset, positions_offset, status_offset,
      dispatch_bytes;
};
Status PlanRows(size_t rows, size_t slices, size_t statics, size_t properties,
                size_t predictors, size_t maximum_buffer, RowPlan *out) {
  if (rows > kHistogramMaximumSamples ||
      slices > kMaximumModularFeatureBatchSlices || statics > 2 ||
      statics > properties || !properties || properties > 16 || !predictors ||
      predictors > 14 || (rows && !slices))
    return Status::InvalidArgument("Invalid histogram row storage request");
  const auto align = [](size_t bytes) { return (bytes + 255) & ~size_t{255}; };
  RowPlan p{};
  p.output_bytes = rows * (4 * statics + properties - statics + 2 * predictors);
  p.slices_offset = align(sizeof(gjxl_histogram_rows::Mapping));
  p.positions_offset =
      align(p.slices_offset + slices * sizeof(gjxl_histogram_rows::Slice));
  p.status_offset = align(p.positions_offset + rows * sizeof(uint32_t));
  p.dispatch_bytes = p.status_offset + 256;
  if (p.output_bytes > maximum_buffer || p.dispatch_bytes > maximum_buffer)
    return Status::Unsupported("Histogram row buffers exceed device admission");
  *out = p;
  return Status::Ok();
}
class CompletedRows final : public PreparedHistogramRows {
public:
  std::unique_ptr<DeviceBuffer> buffer;
  std::span<const std::byte> host;
  std::span<const std::byte> bytes() const override { return host; }
};
struct RowContext {
  MTL::Buffer *planes, *weighted, *output, *dispatch;
  MTL::ComputePipelineState *pipeline;
  gjxl_histogram_rows::Params params;
  size_t maximum_rows;
};
void EncodeRows(MetalBackend &, MTL::ComputeCommandEncoder *encoder,
                const void *opaque) {
  const auto &c = *static_cast<const RowContext *>(opaque);
  encoder->setComputePipelineState(c.pipeline);
  RecordMetalComputePipelineState(c.pipeline);
  encoder->setBuffer(c.planes, 0, 0);
  encoder->setBuffer(c.weighted, 0, 1);
  encoder->setBuffer(c.output, 0, 2);
  encoder->setBuffer(c.dispatch, 0, 3);
  encoder->setBytes(&c.params, sizeof(c.params), 4);
  DispatchMetalThreads(encoder, MTL::Size(c.maximum_rows, c.params.slices, 1),
                       MTL::Size(64, 1, 1));
}
static_assert(sizeof(gjxl_histogram_rows::Slice) == 64);
static_assert(sizeof(gjxl_histogram_rows::Params) == 48);
static_assert(sizeof(gjxl_histogram_rows::Mapping) == 20580);
struct EncodeContext {
  MTL::Buffer *planes, *weighted, *output, *dispatch, *diagnostics = nullptr;
  MTL::ComputePipelineState *pipeline;
  Params params;
  size_t maximum_slice_tokens;
};
void Encode(MetalBackend &, MTL::ComputeCommandEncoder *encoder, const void *opaque) {
  const auto &c = *static_cast<const EncodeContext *>(opaque);
  encoder->setComputePipelineState(c.pipeline);
  RecordMetalComputePipelineState(c.pipeline);
  encoder->setBuffer(c.planes, 0, 0);
  encoder->setBuffer(c.weighted, 0, 1);
  encoder->setBuffer(c.output, 0, 2);
  encoder->setBuffer(c.dispatch, 0, 3);
  encoder->setBytes(&c.params, sizeof(Params), 4);
  if (c.diagnostics)
    encoder->setBuffer(c.diagnostics, kGuard, 5);
  DispatchMetalThreads(encoder, MTL::Size(c.maximum_slice_tokens, c.params.slice_count, 1),
                       MTL::Size(64, 1, 1));
}
struct WeightedContext {
  MTL::Buffer *planes, *features, *dispatch;
  MTL::ComputePipelineState *pipeline;
  gjxl_weighted_features::Parameters parameters;
  size_t slices, status_offset;
  MTL::Buffer *parameter_buffer = nullptr;
  size_t parameters_offset = 0;
};
void EncodeWeighted(MetalBackend &, MTL::ComputeCommandEncoder *encoder, const void *opaque) {
  const auto &c = *static_cast<const WeightedContext *>(opaque);
  encoder->setComputePipelineState(c.pipeline);
  RecordMetalComputePipelineState(c.pipeline);
  encoder->setBuffer(c.planes, 0, 0);
  encoder->setBuffer(c.features, 0, 1);
  encoder->setBuffer(c.dispatch, 0, 2);
  encoder->setBuffer(c.dispatch, c.status_offset, 3);
  if (c.parameter_buffer)
    encoder->setBuffer(c.parameter_buffer, c.parameters_offset, 4);
  else
    encoder->setBytes(&c.parameters, sizeof(c.parameters), 4);
  DispatchMetalThreadgroups(encoder, MTL::Size(c.slices, 1, 1), MTL::Size(128, 1, 1));
}
} // namespace
ModularTokenDeviceLimits MetalBackend::adaptive_modular_limits() const noexcept {
  return {size_t(PTRDIFF_MAX) / sizeof(uint32_t), UINT32_MAX,
          static_cast<size_t>(device_->maxBufferLength())};
}
ModularTokenDeviceLimits MetalBackend::modular_stream_limits() const noexcept {
  auto limits = adaptive_modular_limits();
  limits.maximum_slices = kMaximumModularFeatureBatchSlices;
  return limits;
}
Status MetalBackend::HistogramRowStorageBound(size_t rows, size_t slices,
                                              size_t statics, size_t properties,
                                              size_t predictors,
                                              HostStorageBound *out) const {
  if (!out)
    return Status::InvalidArgument("Null histogram row storage bound");
  RowPlan plan;
  if (auto s = PlanRows(rows, slices, statics, properties, predictors,
                        modular_stream_limits().maximum_buffer_bytes, &plan);
      !s.ok())
    return s;
  HostStorageBound bound;
  if (rows && !bound.Add({plan.output_bytes + plan.dispatch_bytes,
                          plan.output_bytes + plan.dispatch_bytes}))
    return Status::InvalidArgument("Histogram row storage overflow");
  *out = bound;
  return Status::Ok();
}
Status
MetalBackend::GenerateHistogramRows(const HistogramRowRequest &request,
                                    std::unique_ptr<PreparedHistogramRows> *out,
                                    HistogramRowProfile *profile) try {
  const auto *source =
      dynamic_cast<const CompletedStreamFeatures *>(request.source);
  if (!out || !source || source->owner != this || request.positions.empty() ||
      request.mapping.size() + request.static_mapping.size() !=
          request.properties.size())
    return Status::InvalidArgument("Invalid completed histogram row source");
  RowPlan plan;
  if (auto s = PlanRows(request.positions.size(), request.slices.size(),
                        request.static_mapping.size(),
                        request.properties.size(), request.predictors.size(),
                        modular_stream_limits().maximum_buffer_bytes, &plan);
      !s.ok())
    return s;
  for (size_t p = 0; p < request.properties.size(); ++p)
    if (request.properties[p] >= 16 ||
        (p < request.static_mapping.size() && request.properties[p] != p))
      return Status::InvalidArgument("Invalid histogram row properties");
  for (const auto predictor : request.predictors)
    if (static_cast<size_t>(predictor) >= 14)
      return Status::InvalidArgument("Invalid histogram row predictor");
  ManagedHostScope allocation_scope(ResourceClass::kPreparation);
  auto candidate = std::make_unique<CompletedRows>();
  std::unique_ptr<DeviceBuffer> dispatch;
  size_t maximum_rows = 0;
  {
    RowTimer timer{profile ? &profile->allocation_copy_seconds : nullptr};
    if (auto s = Allocate(plan.output_bytes, &candidate->buffer); !s.ok())
      return s;
    if (auto s = Allocate(plan.dispatch_bytes, &dispatch); !s.ok())
      return s;
    auto *bytes =
        static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
    auto *mapping = reinterpret_cast<gjxl_histogram_rows::Mapping *>(bytes);
    for (size_t p = 0; p < request.properties.size(); ++p)
      mapping->properties[p] = request.properties[p];
    for (size_t p = 0; p < request.predictors.size(); ++p)
      mapping->predictors[p] = static_cast<uint32_t>(request.predictors[p]);
    for (size_t p = 0; p < request.static_mapping.size(); ++p)
      std::memcpy(mapping->statics[p], request.static_mapping[p].data(),
                  sizeof(mapping->statics[p]));
    for (size_t p = 0; p < request.mapping.size(); ++p)
      std::memcpy(mapping->dynamics[p], request.mapping[p].data(),
                  sizeof(mapping->dynamics[p]));
    auto *slices = reinterpret_cast<gjxl_histogram_rows::Slice *>(
        bytes + plan.slices_offset);
    size_t next = 0;
    for (size_t i = 0; i < request.slices.size(); ++i) {
      const auto &slice = request.slices[i];
      if (slice.first_row != next ||
          slice.row_count > request.positions.size() - next)
        return Status::InvalidArgument(
            "Histogram row slices do not partition selected positions");
      size_t found = source->views.size();
      for (size_t j = 0; j < source->views.size(); ++j)
        if (source->views[j].values.data() == slice.features.values.data()) {
          found = j;
          break;
        }
      if (found == source->views.size())
        return Status::InvalidArgument(
            "Histogram row feature map is not owned by this source");
      if (auto s = ValidateWeightedFeatureView(slice.features,
                                               source->views[found].source,
                                               source->views[found].parameters);
          !s.ok())
        return s;
      std::memcpy(&slices[i].source, &source->slices[found], sizeof(Slice));
      slices[i].first_row = static_cast<uint32_t>(next);
      slices[i].row_count = static_cast<uint32_t>(slice.row_count);
      slices[i].channel = slice.channel;
      slices[i].group = slice.group;
      next += slice.row_count;
      maximum_rows = std::max(maximum_rows, slice.row_count);
    }
    if (next != request.positions.size())
      return Status::InvalidArgument(
          "Histogram row slices do not cover selected positions");
    std::memcpy(bytes + plan.positions_offset, request.positions.data(),
                request.positions.size_bytes());
    std::memset(bytes + plan.status_offset, 0, sizeof(uint32_t));
  }
  RowContext context{AsMetalBuffer(*source->planes)->handle(),
                     AsMetalBuffer(*source->weighted)->handle(),
                     AsMetalBuffer(*candidate->buffer)->handle(),
                     AsMetalBuffer(*dispatch)->handle(),
                     modular_token_pipelines_[6].get(),
                     {plan.slices_offset, plan.positions_offset,
                      plan.status_offset,
                      static_cast<uint32_t>(request.positions.size()),
                      static_cast<uint32_t>(request.slices.size()),
                      static_cast<uint32_t>(request.static_mapping.size()),
                      static_cast<uint32_t>(request.mapping.size()),
                      static_cast<uint32_t>(request.predictors.size()), 0},
                     maximum_rows};
  {
    RowTimer timer{profile ? &profile->submit_wait_seconds : nullptr};
    Pending pending;
    if (auto s = SubmitCompute("modular.histogram_rows", EncodeRows, &context,
                               &pending.submission);
        !s.ok())
      return s;
    if (!pending.submission)
      return Status::Internal(
          "Histogram row generation returned no submission");
    if (auto s = pending.Wait(); !s.ok())
      return s;
  }
  uint32_t status;
  std::memcpy(&status,
              static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents()) +
                  plan.status_offset,
              sizeof(status));
  if (status & 2)
    return Status::InvalidArgument(
        "Histogram row position outside its channel");
  if (status)
    return Status::Unsupported(
        "Histogram training residual exceeds supported range");
  if (auto s = BorrowCompletedReadOnly(*candidate->buffer, plan.output_bytes, 0,
                                       &candidate->host);
      !s.ok())
    return s;
  if (profile) {
    ++profile->gpu_calls;
    profile->gpu_rows += request.positions.size();
    profile->output_bytes += plan.output_bytes;
    profile->dispatch_bytes += plan.dispatch_bytes;
  }
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Histogram row allocation failed");
}

Status
MetalBackend::PrepareModularStreamFeatures(std::span<const ModularStreamFeatureSlice> slices,
                                           std::unique_ptr<PreparedGpuModularStreamFeatures> *out,
                                           size_t cpu_participants,
                                           WeightedFeatureBackend producer) try {
  if (!out ||
      (producer != WeightedFeatureBackend::kCpu && producer != WeightedFeatureBackend::kGpu &&
       producer != WeightedFeatureBackend::kAuto))
    return Status::InvalidArgument("Invalid Modular stream feature request");
  const auto limits = modular_stream_limits();
  ModularStreamFeatureStoragePlan plan;
  if (auto s = ComputeModularStreamFeatureStoragePlan(slices, cpu_participants, &plan, &limits);
      !s.ok())
    return s;
  const bool use_gpu = modular_token_pipelines_[5] && plan.wavefront_shapes &&
                       (producer == WeightedFeatureBackend::kGpu ||
                        (producer == WeightedFeatureBackend::kAuto &&
                         PreferGpuWeightedFeatures(plan.samples, slices.size())));
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto *profile =
      active_modular_profile ? &active_modular_profile->adaptive_gpu.feature_preparation : nullptr;
  auto candidate = std::make_unique<CompletedStreamFeatures>();
  candidate->backend = id();
  candidate->owner = this;
  candidate->selected_producer = use_gpu ? WeightedFeatureBackend::kGpu : WeightedFeatureBackend::kCpu;
  candidate->plan = plan;
  {
    const FeaturePreparationScope allocation(profile,
                                             &FeaturePreparationProfile::allocation_seconds);
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    candidate->slices.resize(slices.size());
    candidate->views.resize(slices.size());
    if (auto s = Allocate(plan.plane_bytes, &candidate->planes); !s.ok())
      return s;
    if (auto s = Allocate(plan.feature_bytes, &candidate->weighted); !s.ok())
      return s;
  }
  {
    const FeaturePreparationScope copy(profile, &FeaturePreparationProfile::copy_seconds);
    auto *pixels = static_cast<int32_t *>(AsMetalBuffer(*candidate->planes)->contents());
    size_t next = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto &slice = slices[i];
      const auto [w, h] = slice.input.descriptor.extent;
      candidate->slices[i] = {next,
                              next,
                              w,
                              slice.stream_id,
                              slice.channel,
                              static_cast<uint32_t>(w),
                              static_cast<uint32_t>(h),
                              {static_cast<uint32_t>(i), 0}};
      for (size_t y = 0; y < h; ++y)
        std::memcpy(pixels + next + y * w, slice.input.Row(y), w * sizeof(int32_t));
      next += w * h;
    }
  }
  if (use_gpu) {
    std::unique_ptr<DeviceBuffer> dispatch;
    {
      const FeaturePreparationScope allocation(profile,
                                               &FeaturePreparationProfile::allocation_seconds);
      if (auto s = Allocate(plan.dispatch_bytes, &dispatch); !s.ok())
        return s;
    }
    auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
    {
      const FeaturePreparationScope copy(profile, &FeaturePreparationProfile::copy_seconds);
      std::memcpy(bytes, candidate->slices.data(), plan.slices_bytes);
      std::memset(bytes + plan.status_offset, 0, sizeof(uint32_t));
      auto *parameters =
          reinterpret_cast<gjxl_weighted_features::Parameters *>(bytes + plan.parameters_offset);
      for (size_t i = 0; i < slices.size(); ++i) {
        for (size_t p = 0; p < 7; ++p)
          parameters[i].coefficients[p] = slices[i].parameters.coefficients[p];
        for (size_t p = 0; p < 4; ++p)
          parameters[i].weights[p] = slices[i].parameters.weights[p];
      }
    }
    WeightedContext context{AsMetalBuffer(*candidate->planes)->handle(),
                            AsMetalBuffer(*candidate->weighted)->handle(),
                            AsMetalBuffer(*dispatch)->handle(),
                            modular_token_pipelines_[5].get(),
                            {},
                            slices.size(),
                            plan.status_offset,
                            AsMetalBuffer(*dispatch)->handle(),
                            plan.parameters_offset};
    if (profile) {
      ++profile->gpu_calls;
      profile->gpu_samples += plan.samples;
      profile->slice_jobs += slices.size();
    }
    const FeaturePreparationScope compute(profile, &FeaturePreparationProfile::prediction_seconds);
    Pending pending;
    if (auto s = SubmitCompute("modular.stream_weighted_features", EncodeWeighted, &context,
                               &pending.submission);
        !s.ok())
      return s;
    if (!pending.submission)
      return Status::Internal("Modular stream features returned no submission");
    if (auto s = pending.Wait(); !s.ok())
      return s;
    uint32_t status;
    std::memcpy(&status, bytes + plan.status_offset, sizeof(status));
    if (status == 1)
      return Status::Unsupported("Modular weighted state exceeds supported range");
    if (status)
      return Status::Internal("Unknown Modular weighted feature status");
  } else {
    if (profile && producer != WeightedFeatureBackend::kCpu)
      ++profile->cpu_fallback_calls;
    ManagedVector<WeightedFeatureSlice> jobs;
    ManagedVector<WeightedPredictorParameters> parameters;
    {
      const FeaturePreparationScope allocation(profile,
                                               &FeaturePreparationProfile::allocation_seconds);
      jobs.resize(slices.size());
      parameters.resize(slices.size());
    }
    for (size_t i = 0; i < slices.size(); ++i) {
      jobs[i] = {slices[i].input, static_cast<size_t>(candidate->slices[i].token_offset)};
      parameters[i] = slices[i].parameters;
    }
    auto *features =
        static_cast<WeightedPredictorFeature *>(AsMetalBuffer(*candidate->weighted)->contents());
    if (auto s = PrepareWeightedPredictorFeatureBatchWithParameters(
            jobs, parameters, {features, plan.samples}, cpu_participants, profile);
        !s.ok())
      return s;
  }
  {
    const FeaturePreparationScope publication(profile,
                                              &FeaturePreparationProfile::publication_seconds);
    std::span<const std::byte> bytes;
    if (auto s = BorrowCompletedReadOnly(*candidate->weighted, plan.feature_bytes, 0, &bytes);
        !s.ok())
      return s;
    const auto features =
        std::span(reinterpret_cast<const WeightedPredictorFeature *>(bytes.data()), plan.samples);
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto &s = candidate->slices[i];
      candidate->views[i] = {slices[i].input, slices[i].parameters,
                             features.subspan(s.token_offset, size_t{s.width} * s.height)};
    }
    *out = std::move(candidate);
  }
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular stream feature allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular stream feature storage overflow");
}

Status
MetalBackend::GenerateModularStreamTokens(const PreparedGpuModularStreamFeatures &features,
                                          std::span<const ModularStreamTokenTree> trees,
                                          std::unique_ptr<PreparedGpuModularTokens> *out) try {
  const auto *input = dynamic_cast<const CompletedStreamFeatures *>(&features);
  if (!out || !input || input->owner != this)
    return Status::InvalidArgument("Invalid Modular stream feature owner/output");
  ModularStreamTokenStoragePlan plan;
  const auto limits = modular_stream_limits();
  if (auto s = ComputeModularStreamTokenStoragePlan(input->plan.samples, input->slices.size(),
                                                    trees, &plan, &limits);
      !s.ok())
    return s;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  AdaptiveModularTokenStoragePlan token_plan;
  token_plan.token_count = input->plan.samples;
  token_plan.values_bytes = plan.values_bytes;
  token_plan.contexts_offset = plan.contexts_offset;
  token_plan.output_bytes = plan.output_bytes;
  auto candidate = std::make_unique<CompletedTokens>(id(), token_plan, plan.maximum_contexts);
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = Allocate(plan.output_bytes, &candidate->buffer); !s.ok())
      return s;
  }
  std::unique_ptr<DeviceBuffer> dispatch;
  if (auto s = Allocate(plan.dispatch_bytes, &dispatch); !s.ok())
    return s;
  auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
  std::memcpy(bytes, input->slices.data(), input->plan.slices_bytes);
  auto *slices = reinterpret_cast<AdaptiveModularTokenSlice *>(bytes);
  auto *nodes = reinterpret_cast<ModularTokenNode *>(bytes + plan.nodes_offset);
  size_t next = 0;
  for (const auto &binding : trees) {
    TreeLayout layout;
    if (auto s = ValidateTree(*binding.tree, &layout); !s.ok())
      return s;
    for (size_t i = binding.first_slice; i < binding.first_slice + binding.slice_count; ++i)
      slices[i].reserved[0] = static_cast<uint32_t>(next);
    for (size_t i = 0; i < binding.tree->size; ++i) {
      const auto &n = binding.tree->nodes[i];
      nodes[next + i] = {
          n.property, n.split,      n.left,           n.right, static_cast<uint32_t>(n.predictor),
          n.offset,   n.multiplier, layout.context[i]};
    }
    next += binding.tree->size;
  }
  std::memset(bytes + plan.status_offset, 0, sizeof(uint32_t));
  EncodeContext context{};
  context.planes = AsMetalBuffer(*input->planes)->handle();
  context.weighted = AsMetalBuffer(*input->weighted)->handle();
  context.output = AsMetalBuffer(*candidate->buffer)->handle();
  context.dispatch = AsMetalBuffer(*dispatch)->handle();
  context.pipeline = modular_token_pipelines_[3].get();
  context.params = {plan.contexts_offset,
                    plan.nodes_offset,
                    plan.status_offset,
                    static_cast<uint32_t>(input->slices.size()),
                    static_cast<uint32_t>(plan.nodes),
                    1,
                    1};
  context.maximum_slice_tokens = input->plan.maximum_slice_samples;
  Pending pending;
  if (auto s = SubmitCompute("modular.stream_tokens", Encode, &context, &pending.submission);
      !s.ok())
    return s;
  if (!pending.submission)
    return Status::Internal("Modular stream tokens returned no submission");
  if (auto s = pending.Wait(); !s.ok())
    return s;
  uint32_t status;
  std::memcpy(&status, bytes + plan.status_offset, sizeof(status));
  if (auto s = CheckModularTokenDeviceStatus(status); !s.ok())
    return s;
  std::span<const std::byte> output;
  if (auto s = BorrowCompletedReadOnly(*candidate->buffer, plan.output_bytes, 0, &output); !s.ok())
    return s;
  candidate->host = {{reinterpret_cast<const uint32_t *>(output.data()), input->plan.samples},
                     {reinterpret_cast<const uint16_t *>(output.data() + plan.contexts_offset),
                      input->plan.samples}};
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular stream token allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular stream token storage overflow");
}

Status MetalBackend::PrepareAdaptiveModularFeatures(
    const ModularEncoderFrame &frame, std::span<const AdaptiveModularTokenSlice> slices,
    const WeightedPredictorParameters &parameters, bool weighted,
    std::unique_ptr<PreparedGpuAdaptiveModularFeatures> *out, size_t maximum_cpu_participants,
    WeightedFeatureBackend feature_backend) try {
  if (!out || !parameters.valid() ||
      (feature_backend != WeightedFeatureBackend::kCpu &&
       feature_backend != WeightedFeatureBackend::kGpu &&
       feature_backend != WeightedFeatureBackend::kAuto))
    return Status::InvalidArgument("Invalid adaptive Modular feature output/parameters");
  AdaptiveModularLayout layout;
  if (auto s = ComputeAdaptiveModularLayout(frame, &layout); !s.ok())
    return s;
  if (auto s = ValidateAdaptiveModularSlices(layout, slices); !s.ok())
    return s;
  const bool gpu_eligible =
      modular_token_pipelines_[5] && GpuWeightedFeatureSlicesSupported(slices);
  const bool use_gpu = weighted && gpu_eligible &&
                       (feature_backend == WeightedFeatureBackend::kGpu ||
                        (feature_backend == WeightedFeatureBackend::kAuto &&
                         PreferGpuWeightedFeatures(layout.tokens, slices.size())));
  AdaptiveModularTokenStoragePlan storage;
  const auto limits = adaptive_modular_limits();
  if (auto s = ComputeAdaptiveModularTokenStoragePlan(
          layout.shape.channels(), layout.shape.metadata, slices.size(), 1, weighted,
          maximum_cpu_participants, &storage, &limits);
      !s.ok())
    return s;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto *profile =
      active_modular_profile ? &active_modular_profile->adaptive_gpu.feature_preparation : nullptr;
  std::unique_ptr<CompletedFeatures> candidate;
  {
    const FeaturePreparationScope allocation(profile,
                                             &FeaturePreparationProfile::allocation_seconds);
    candidate = std::make_unique<CompletedFeatures>();
    candidate->backend = id();
    candidate->source = frame.metadata();
    candidate->layout = layout;
    candidate->descriptors.assign(slices.begin(), slices.end());
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = Allocate(storage.plane_bytes, &candidate->planes); !s.ok())
      return s;
    if (weighted)
      if (auto s = Allocate(storage.weighted_bytes, &candidate->weighted); !s.ok())
        return s;
  }
  {
    const FeaturePreparationScope copy(profile, &FeaturePreparationProfile::copy_seconds);
    auto *planes = static_cast<std::byte *>(AsMetalBuffer(*candidate->planes)->contents());
    std::memset(planes, 0, storage.plane_bytes);
    for (size_t c = 0; c < layout.shape.count; ++c) {
      const auto source = frame.image().view(c);
      const size_t bytes =
          source.descriptor.extent.width * source.descriptor.extent.height * sizeof(int32_t);
      std::memcpy(planes + layout.plane_offsets[c], source.backing.data(), bytes);
    }
  }
  if (use_gpu) {
    WeightedContext context{AsMetalBuffer(*candidate->planes)->handle(),
                            AsMetalBuffer(*candidate->weighted)->handle(),
                            nullptr,
                            modular_token_pipelines_[5].get(),
                            {},
                            slices.size(),
                            storage.nodes_offset};
    for (size_t i = 0; i < 7; ++i)
      context.parameters.coefficients[i] = parameters.coefficients[i];
    for (size_t i = 0; i < 4; ++i)
      context.parameters.weights[i] = parameters.weights[i];
    std::unique_ptr<DeviceBuffer> dispatch;
    Pending pending;
    {
      const FeaturePreparationScope allocation(profile,
                                               &FeaturePreparationProfile::allocation_seconds);
      if (auto s = Allocate(storage.feature_dispatch_bytes, &dispatch); !s.ok())
        return s;
    }
    auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
    {
      const FeaturePreparationScope copy(profile, &FeaturePreparationProfile::copy_seconds);
      std::memcpy(bytes, slices.data(), storage.slices_bytes);
      std::memset(bytes + storage.nodes_offset, 0, sizeof(uint32_t));
    }
    context.dispatch = AsMetalBuffer(*dispatch)->handle();
    if (profile) {
      ++profile->gpu_calls;
      profile->gpu_samples += storage.token_count;
      profile->slice_jobs += slices.size();
    }
    {
      const FeaturePreparationScope computation(profile,
                                                &FeaturePreparationProfile::prediction_seconds);
      if (auto s = SubmitCompute("modular.weighted_features", EncodeWeighted, &context,
                                 &pending.submission);
          !s.ok())
        return s;
      if (!pending.submission)
        return Status::Internal("Weighted features returned no submission");
      if (auto s = pending.Wait(); !s.ok())
        return s;
      uint32_t status;
      std::memcpy(&status, bytes + storage.nodes_offset, sizeof(status));
      if (status & ~uint32_t{1})
        return Status::DeviceError("Invalid weighted feature status");
      if (status)
        return Status::Unsupported("Adaptive Modular weighted state exceeds supported range");
    }
  } else if (weighted) {
    if (profile && feature_backend != WeightedFeatureBackend::kCpu)
      ++profile->cpu_fallback_calls;
    // Shared storage is initialized directly: no full-image feature staging or
    // hidden tokenization. Each scalar weighted recurrence resets at its slice.
    auto *features = static_cast<AdaptiveModularWeightedFeature *>(
        AsMetalBuffer(*candidate->weighted)->contents());
    ManagedVector<WeightedFeatureSlice> jobs;
    {
      const FeaturePreparationScope allocation(profile,
                                               &FeaturePreparationProfile::allocation_seconds);
      jobs.resize(slices.size());
      size_t index = 0;
      if (auto s = VisitAdaptiveModularSlices(
              layout,
              [&](const auto &slice, size_t c, ModularRect rect) {
                auto view = frame.image().view(c);
                view.backing = view.backing.subspan(rect.y * view.stride + rect.x);
                view.descriptor.extent = rect.extent;
                jobs[index++] = {view, static_cast<size_t>(slice.token_offset)};
                return Status::Ok();
              });
          !s.ok())
        return s;
    }
    if (auto s = PrepareWeightedPredictorFeatureBatch(jobs, parameters,
                                                      std::span(features, storage.token_count),
                                                      maximum_cpu_participants, profile);
        !s.ok())
      return s;
  }
  {
    const FeaturePreparationScope publication(profile,
                                              &FeaturePreparationProfile::publication_seconds);
    if (weighted) {
      std::span<const std::byte> bytes;
      if (auto s = BorrowCompletedReadOnly(*candidate->weighted, storage.weighted_bytes, 0, &bytes);
          !s.ok())
        return s;
      candidate->host_weighted = {
          reinterpret_cast<const AdaptiveModularWeightedFeature *>(bytes.data()),
          storage.token_count};
    }
    *out = std::move(candidate);
  }
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Adaptive Modular feature allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Adaptive Modular feature storage overflow");
}

Status
MetalBackend::GenerateAdaptiveModularTokens(const PreparedGpuAdaptiveModularFeatures &features,
                                            const DecisionTree &tree,
                                            std::unique_ptr<PreparedGpuModularTokens> *out) {
  return GenerateAdaptiveModularTokensImpl(features, tree, out);
}
Status MetalBackend::GenerateAdaptiveModularTokensImpl(
    const PreparedGpuAdaptiveModularFeatures &features, const DecisionTree &tree,
    std::unique_ptr<PreparedGpuModularTokens> *out, DeviceBuffer *diagnostics) try {
  const auto *input = dynamic_cast<const CompletedFeatures *>(&features);
  if (!out || !input || input->backend_id() != id())
    return Status::InvalidArgument("Invalid adaptive Modular feature owner/output");
  TreeLayout tree_layout;
  if (auto s = ValidateTree(tree, &tree_layout); !s.ok())
    return s;
  const auto &metadata = input->metadata();
  const size_t channels = (metadata.color == codec_internal::SourceColor::kGraySrgb ? 1 : 3) +
                          static_cast<size_t>(metadata.alpha.has_value());
  if (tree_layout.leaves >
      MaximumGlobalTreeLeaves(uint64_t{metadata.extent.width} * metadata.extent.height * channels))
    return Status::Unsupported("Adaptive Modular tree exceeds the image-dependent decoder limit");
  if (tree_layout.weighted && !input->has_weighted_features())
    return Status::Unsupported("Adaptive Modular tree needs weighted features");
  AdaptiveModularTokenStoragePlan plan;
  const auto limits = adaptive_modular_limits();
  if (auto s = ComputeAdaptiveModularTokenStoragePlan(
          input->layout.shape.channels(), input->layout.shape.metadata, input->slices().size(),
          tree.size, input->has_weighted_features(), 1, &plan, &limits);
      !s.ok())
    return s;
  EncodeContext context{};
  context.planes = AsMetalBuffer(*input->planes)->handle();
  context.weighted = input->weighted ? AsMetalBuffer(*input->weighted)->handle() : nullptr;
  context.params = {
      plan.contexts_offset,
      plan.nodes_offset,
      plan.status_offset,
      static_cast<uint32_t>(plan.slice_count),
      static_cast<uint32_t>(tree.size),
      static_cast<uint32_t>(diagnostics ? input->has_weighted_features() : tree_layout.weighted),
      0};
  context.maximum_slice_tokens = input->layout.maximum_slice_tokens;
  context.pipeline = modular_token_pipelines_[diagnostics ? 4 : 3].get();
  if (diagnostics) {
    if (plan.token_count > (size_t(PTRDIFF_MAX) - 2 * kGuard) / sizeof(AdaptiveModularDiagnostic) ||
        plan.output_bytes >
            limits.maximum_buffer_bytes - std::min(kGuard, limits.maximum_buffer_bytes))
      return Status::Unsupported("Adaptive Modular diagnostic storage exceeds limits");
    MetalBuffer *native = nullptr;
    if (auto s = RequireMetalBuffer(
            diagnostics, plan.token_count * sizeof(AdaptiveModularDiagnostic) + 2 * kGuard, 0,
            "adaptive Modular diagnostics", &native);
        !s.ok())
      return s;
    context.diagnostics = native->handle();
  }
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<CompletedTokens>(id(), plan, tree_layout.leaves);
  std::unique_ptr<DeviceBuffer> dispatch;
  Pending pending;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = Allocate(plan.output_bytes + (diagnostics ? kGuard : 0), &candidate->buffer);
        !s.ok())
      return s;
  }
  if (diagnostics)
    std::memset(AsMetalBuffer(*candidate->buffer)->contents(), 0xa5,
                candidate->buffer->size_bytes());
  if (auto s = Allocate(plan.dispatch_bytes, &dispatch); !s.ok())
    return s;
  auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
  std::memset(bytes, 0, plan.dispatch_bytes);
  std::memcpy(bytes, input->slices().data(), plan.slices_bytes);
  auto *nodes = reinterpret_cast<ModularTokenNode *>(bytes + plan.nodes_offset);
  for (size_t i = 0; i < tree.size; ++i) {
    const auto &n = tree.nodes[i];
    nodes[i] = {n.property,
                n.split,
                n.left,
                n.right,
                static_cast<uint32_t>(n.predictor),
                n.offset,
                n.multiplier,
                tree_layout.context[i]};
  }
  context.output = AsMetalBuffer(*candidate->buffer)->handle();
  context.dispatch = AsMetalBuffer(*dispatch)->handle();
  if (!context.weighted)
    context.weighted = context.dispatch; // Unused, safely sized binding.
  if (auto s = SubmitCompute("modular.adaptive_tokens", Encode, &context, &pending.submission);
      !s.ok())
    return s;
  if (!pending.submission)
    return Status::Internal("Adaptive Modular operation returned no submission");
  if (auto s = pending.Wait(); !s.ok())
    return s;
  uint32_t status;
  std::memcpy(&status, bytes + plan.status_offset, sizeof(status));
  if (auto s = CheckModularTokenDeviceStatus(status); !s.ok())
    return s;
  if (diagnostics) {
    const auto *p = static_cast<const std::byte *>(AsMetalBuffer(*candidate->buffer)->contents());
    if (!GuardIntact(p + plan.values_bytes, plan.contexts_offset - plan.values_bytes) ||
        !GuardIntact(p + plan.output_bytes, kGuard))
      return Status::DeviceError("Adaptive Modular token guard overwritten");
  }
  std::span<const std::byte> output;
  if (auto s = BorrowCompletedReadOnly(*candidate->buffer, plan.output_bytes, 0, &output); !s.ok())
    return s;
  candidate->host = {
      {reinterpret_cast<const uint32_t *>(output.data()), plan.token_count},
      {reinterpret_cast<const uint16_t *>(output.data() + plan.contexts_offset), plan.token_count}};
  pending.submission.reset();
  dispatch.reset();
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Adaptive Modular token allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Adaptive Modular token storage overflow");
}

Status
MetalBackend::DiagnoseAdaptiveModularTokens(const PreparedGpuAdaptiveModularFeatures &features,
                                            const DecisionTree &tree,
                                            ManagedVector<AdaptiveModularDiagnostic> *out) try {
  if (!out || !dynamic_cast<const CompletedFeatures *>(&features) ||
      features.backend_id() != id() ||
      features.token_count() >
          (size_t(PTRDIFF_MAX) - 2 * kGuard) / sizeof(AdaptiveModularDiagnostic))
    return Status::InvalidArgument("Invalid adaptive Modular diagnostics");
  const ManagedHostScope resources(ResourceClass::kDiagnostics);
  const size_t bytes = features.token_count() * sizeof(AdaptiveModularDiagnostic);
  std::unique_ptr<DeviceBuffer> diagnostic;
  if (auto s = Allocate(bytes + 2 * kGuard, &diagnostic); !s.ok())
    return s;
  auto *data = static_cast<std::byte *>(AsMetalBuffer(*diagnostic)->contents());
  std::memset(data, 0xa5, bytes + 2 * kGuard);
  std::unique_ptr<PreparedGpuModularTokens> tokens;
  if (auto s = GenerateAdaptiveModularTokensImpl(features, tree, &tokens, diagnostic.get());
      !s.ok())
    return s;
  if (!GuardIntact(data, kGuard) || !GuardIntact(data + kGuard + bytes, kGuard))
    return Status::DeviceError("Adaptive Modular diagnostic guard overwritten");
  ManagedVector<AdaptiveModularDiagnostic> candidate(features.token_count());
  std::memcpy(candidate.data(), data + kGuard, bytes);
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Adaptive Modular diagnostic allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Adaptive Modular diagnostic storage overflow");
}
Status DiagnoseMetalAdaptiveModularTokens(GpuBackend &backend,
                                          const PreparedGpuAdaptiveModularFeatures &features,
                                          const DecisionTree &tree,
                                          ManagedVector<AdaptiveModularDiagnostic> *out) {
  auto *metal = dynamic_cast<MetalBackend *>(&backend);
  if (!metal)
    return Status::Unsupported("Adaptive Modular diagnostics require Metal");
  return metal->DiagnoseAdaptiveModularTokens(features, tree, out);
}
} // namespace gjxl::metal_internal
