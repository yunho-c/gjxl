// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Lossless scope policy follows pinned enc_frame.cc and enc_modular.cc.
#include "codestream/modular/reference_workflow_internal.h"
#include "codestream/modular/reference_learning_profile.h"
#include "codec/modular/profile.h"
#include <algorithm>
#include <stdexcept>

namespace gjxl::modular_internal {
namespace {
using namespace resource_budget_internal;
Status Invalid() {
  return Status::InvalidArgument("Invalid reference Modular workflow");
}
using reference_workflow_internal::Stage;
Status Fill(const ModularImage &image, ReferenceModularStream *record) {
  if (image.channel_count() > record->channels.size())
    return Invalid();
  record->channel_count = image.channel_count();
  record->metadata_channels = image.metadata_channels();
  for (size_t c = 0; c < image.channel_count(); ++c)
    record->channels[c] = image.view(c);
  return Status::Ok();
}
} // namespace

PreparedModularStreamsView ReferenceModularFrame::view() const {
  return {extent_, format_,
          records_.empty() || stats_.local_trees ? nullptr : &records_[0].tree,
          streams_};
}
Status PrepareReferenceModularFrame(PackedModularImageView input,
                                    const ReferenceWorkflowOptions &options,
                                    const ResourceBudget &budget,
                                    ReferenceModularFrame *out) try {
  if (!out ||
      options.learning_cpu_participants > 256 ||
      options.selection_cpu_participants > 256 ||
      options.tokenization.cpu_participants > 256 ||
      (options.learning_schedule != ReferenceLearningSchedule::kCohorts &&
       options.learning_schedule != ReferenceLearningSchedule::kPipeline) ||
      (options.feature_window_samples != 8 * 1024 * 1024 &&
       options.feature_window_samples != 16 * 1024 * 1024) ||
      !options.fitting_batch_fits || options.fitting_batch_fits > 256 ||
      options.fitting_minimum_rows > kHistogramMaximumSamples ||
      (options.fitting_backend != ReferenceFittingBackend::kCpu &&
       options.fitting_backend != ReferenceFittingBackend::kResumableCpu &&
       options.fitting_backend != ReferenceFittingBackend::kGpu &&
       options.fitting_backend != ReferenceFittingBackend::kAuto &&
       options.fitting_backend != ReferenceFittingBackend::kGpuResident &&
       options.fitting_backend != ReferenceFittingBackend::kGpuFrontier) ||
      (options.buffering != ReferenceBuffering::kAutomatic &&
       options.buffering != ReferenceBuffering::kBuffered) ||
      (options.predictors != ReferencePredictors::kBest &&
       options.predictors != ReferencePredictors::kAll) ||
      !options.maximum_samples ||
      options.maximum_samples > kHistogramMaximumSamples ||
      !options.feature_cpu_participants || options.feature_cpu_participants > 256 ||
      (options.feature_producer != WeightedFeatureBackend::kCpu &&
       options.feature_producer != WeightedFeatureBackend::kGpu &&
       options.feature_producer != WeightedFeatureBackend::kAuto))
    return Invalid();
  if (options.learning_schedule == ReferenceLearningSchedule::kPipeline &&
      options.fitting_backend != ReferenceFittingBackend::kCpu)
    return Status::Unsupported("Local learning pipeline currently requires the ordinary CPU fitter");
  if (options.gpu && !modular_gpu_internal::QueryGpuModularStreamProcessing(*options.gpu))
    return Status::Unsupported("Backend does not support histogram workflow stream processing");
  if ((options.fitting_backend == ReferenceFittingBackend::kGpu ||
       options.fitting_backend == ReferenceFittingBackend::kGpuResident ||
       options.fitting_backend == ReferenceFittingBackend::kGpuFrontier) &&
      (!options.gpu || !histogram_gpu_internal::QueryGpuModularHistogramFitting(*options.gpu)))
    return Status::Unsupported("Backend does not support native histogram fitting");
  if ((options.fitting_backend == ReferenceFittingBackend::kGpu ||
       options.fitting_backend == ReferenceFittingBackend::kAuto) &&
      (!options.fitting_execution.histogram_shards ||
       options.fitting_execution.histogram_shards >
           histogram_gpu_internal::kMaximumHistogramFittingShards ||
       !options.fitting_execution.maximum_active_properties ||
       !options.fitting_execution.maximum_scratch_bytes))
    return Invalid();
  if (auto s = ValidatePreparedModularEntropyOptions(options.entropy); !s.ok())
    return s;
  if (auto s = input.Validate(); !s.ok())
    return s;
  ModularInputProfile source;
  if (auto s = ResolveModularInput(input.extent, input.format, &source);
      !s.ok())
    return s;
  const uint64_t population =
      uint64_t{input.extent.width} * input.extent.height * source.channel_count;
  if (population > kHistogramMaximumPopulation)
    return Status::Unsupported(
        "Reference workflow source population exceeds private admission");
  ModularFrameGeometry geometry;
  if (auto s = ModularFrameGeometry::Create(input.extent, &geometry); !s.ok())
    return s;
  size_t final_id;
  if (auto s = ModularStreamId(geometry, StreamRole::kGroup,
                               geometry.group_count() - 1, &final_id);
      !s.ok())
    return s;
  if (final_id >= 65535)
    return Status::Unsupported(
        "Reference workflow group IDs exceed learning admission");
  ReferenceModularFrame result;
  result.extent_ = input.extent;
  result.format_ = input.format;
  auto &stats = result.stats_;
  stats.local_trees = options.buffering == ReferenceBuffering::kAutomatic &&
                      geometry.group_count() > 8;
  HostStorageBound bound;
  const size_t count = 1 + geometry.dc_group_count() + geometry.group_count();
  if (!bound.AddVector<ReferenceModularStream>(
          count, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<PreparedModularStreamView>(
          count, VectorCapacityPolicy::kFreshExact))
    return Invalid();
  if (auto s = Stage(budget, bound.peak_bytes, &stats,
                     [&] {
                       result.records_.resize(count);
                       result.streams_.resize(count);
                       return Status::Ok();
                     });
      !s.ok())
    return s;
  if (auto s = ComputeModularImageStorageBound(source.channels(), 0, &bound);
      !s.ok())
    return s;
  if (auto s = Stage(
          budget, bound.peak_bytes, &stats,
          [&] { return ModularEncoderFrame::Prepare(input, &result.source_); });
      !s.ok())
    return s;
  auto &global = result.records_[0];
  if (auto s = Fill(result.source_.image(), &global); !s.ok())
    return s;
  global.learning_extent = input.extent;
  if (!stats.local_trees) {
    const bool single = geometry.group_count() == 1;
    if (auto s = reference_workflow_internal::SelectReferenceModularScopes(
            {&global, 1}, {ReferencePaletteStage::kGlobal, source.metadata.bits,
                           1024, single, single},
            options.selection_cpu_participants, budget, &stats);
        !s.ok())
      return s;
  }
  const ModularImage *base = &result.source_.image();
  if (global.selection.image.channel_count()) {
    base = &global.selection.image;
    result
        .source_ = {}; // All future group views now borrow the selected image.
  }
  ChannelShape shape;
  if (auto s = DescribeImage(*base, &shape); !s.ok())
    return s;
  ModularStreamStoragePlan placement;
  if (auto s = ComputeModularStreamStoragePlan(geometry, shape.channels(),
                                               shape.metadata, &placement);
      !s.ok())
    return s;
  ModularStreamPlan plan;
  if (auto s = Stage(budget, placement.owned.peak_bytes, &stats,
                     [&] {
                       return BuildModularStreamPlan(geometry, shape.channels(),
                                                     shape.metadata, &plan);
                     });
      !s.ok())
    return s;
  if (stats.local_trees) {
    // The global header is produced from the first streaming DC chunk. Its
    // normal planes remain present as excluded learning descriptors.
    global.learning_extent = {std::min(input.extent.width, size_t{2048}),
                              std::min(input.extent.height, size_t{2048})};
    for (size_t c = 0; c < global.channel_count; ++c) {
      ModularChannelView view;
      if (auto s = BorrowChannelSlice(base->view(c),
                                      {0, 0, global.learning_extent}, &view);
          !s.ok())
        return s;
      global.channels[c] = view;
    }
  }
  for (size_t i = 0; i < plan.streams.size(); ++i) {
    const auto &entry = plan.streams[i];
    auto &record = result.records_[i];
    record.id = entry.id;
    if (i != 0) {
      record.learning_extent = entry.role == StreamRole::kDcGroup
                                   ? Extent2D{2048, 2048}
                                   : Extent2D{256, 256};
      for (auto slice : std::span(plan.slices)
                            .subspan(entry.slice_begin, entry.slice_count)) {
        if (record.channel_count == record.channels.size())
          return Invalid();
        if (auto s =
                BorrowChannelSlice(base->view(slice.channel), slice.rect,
                                   &record.channels[record.channel_count++]);
            !s.ok())
          return s;
      }
    }
  }
  if (auto s = reference_workflow_internal::SelectReferenceModularScopes(
          std::span(result.records_).subspan(1),
          {ReferencePaletteStage::kLocal, source.metadata.bits},
          options.selection_cpu_participants, budget, &stats); !s.ok())
    return s;
  // Selection may replace channel backing and add metadata. Publish views only
  // after every selection job has joined and its owning record is final.
  for (size_t i = 0; i < plan.streams.size(); ++i) {
    const auto &entry = plan.streams[i];
    auto &record = result.records_[i];
    auto &stream = result.streams_[i];
    stream.channels = record.training_channels().first(
        i == 0 ? entry.slice_count : record.channel_count);
    stream.transforms = record.selection.transforms;
    stream.weighted = kSelectionWeightedPresets[record.selection.wp_mode];
    if (stats.local_trees && (i == 0 || record.channel_count))
      stream.local_tree = &record.tree;
  }
  if (auto s = reference_workflow_internal::LearnReferenceModularScopes(
          result.records_, result.streams_, population, options, budget, &stats,
          &result.gpu_tokens_); !s.ok())
    return s;
  // Reuse the coding planner's full shape/tree/header validation before
  // publishing this self-contained result, without allocating coding backing.
  ModularWorkflowStoragePlan coding;
  if (auto s = ComputePreparedModularStreamsStoragePlan(
          result.view(), options.entropy, &coding, options.tokenization);
      !s.ok())
    return s;
  *out = std::move(result);
  return Status::Ok();
} catch (const ManagedAllocationFailure &error) {
  return error.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Reference Modular workflow allocation failed");
} catch (const std::length_error &) {
  return Invalid();
}

Status EncodeReferenceModularImage(PackedModularImageView input,
                                   const ReferenceWorkflowOptions &options,
                                   const ResourceBudget &budget,
                                   codestream_internal::CodestreamBuffer *out,
                                   ReferenceWorkflowStats *stats) {
  if (!out)
    return Invalid();
  ReferenceModularFrame prepared;
  if (auto s = PrepareReferenceModularFrame(input, options, budget, &prepared);
      !s.ok())
    return s;
  ModularWorkflowStoragePlan plan;
  auto result_stats = prepared.stats();
  auto entropy = options.entropy;
  auto tokenization = options.tokenization;
  ResourceReservation reservation;
  for (;;) {
    if (auto s = ComputePreparedModularStreamsStoragePlan(prepared.view(), entropy, &plan, tokenization);
        !s.ok())
      return s;
    const auto status = budget.TryReserve(std::max(size_t{1}, plan.working.peak_bytes), &reservation);
    if (status.ok())
      break;
    if ((plan.participants == 1 && plan.tokenization_participants == 1) ||
        (status.code() != StatusCode::kUnavailable && status.code() != StatusCode::kOutOfMemory))
      return status;
    // Retry admission only, before tokenization or any model/output is built.
    // Reduce the larger parallel stage first. Tight budgets
    // retain the original serial envelope and identical compression decisions.
    if (plan.tokenization_participants > 1 &&
        (plan.participants == 1 ||
         plan.preparation.peak_bytes >= std::max(plan.modeling.peak_bytes, plan.emission.peak_bytes))) {
      tokenization.cpu_participants = std::max(size_t{1}, plan.tokenization_participants / 2);
      ++result_stats.tokenization_memory_reductions;
    } else {
      entropy.cpu_participants = std::max(size_t{1}, plan.participants / 2);
      ++result_stats.entropy_memory_reductions;
    }
  }
  result_stats.entropy_cpu_participants = plan.participants;
  result_stats.tokenization_cpu_participants = plan.tokenization_participants;
  result_stats.maximum_stage_bytes = std::max(result_stats.maximum_stage_bytes,
                                              plan.working.peak_bytes);
  auto context = CurrentResourceContext();
  context.reservation = &reservation;
  context.resource_class = ResourceClass::kPreparation;
  ResourceContextScope resources(context);
  codestream_internal::CodestreamBuffer result;
  if (auto s = EncodePreparedModularStreams(prepared.view(), entropy, &result, tokenization);
      !s.ok())
    return s;
  *out = std::move(result);
  if (stats)
    *stats = result_stats;
  if (active_reference_workflow_stats)
    *active_reference_workflow_stats = result_stats;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
