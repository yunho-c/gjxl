// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/metal_aq_strategy_metadata.h"

#include "gpu/metal/metal_backend_internal.h"
#include <limits>

#define setComputePipelineState(state)                                         \
  setComputePipelineState(state);                                              \
  ::gjxl::metal_internal::RecordMetalComputePipelineState(state)

namespace gjxl::metal_internal {
namespace {
struct Params {
  uint32_t width, height, chunks, tiles_x, tiles_y, groups_x, groups_y;
};
static_assert(sizeof(Params) == 28);
Params Parameters(Extent2D blocks) {
  const auto tiles = blocks.ceil_div(8), groups = blocks.ceil_div(32);
  return {uint32_t(blocks.width),
          uint32_t(blocks.height),
          uint32_t((blocks.width * blocks.height + 255) / 256),
          uint32_t(tiles.width),
          uint32_t(tiles.height),
          uint32_t(groups.width),
          uint32_t(groups.height)};
}
} // namespace

Status
ComputeAqStrategyMetadataStoragePlan(Extent2D blocks,
                                     AqStrategyMetadataStoragePlan *out) {
  size_t count = 0, groups = 0;
  if (out == nullptr || blocks.empty() || !blocks.try_area(&count) ||
      count > size_t(std::numeric_limits<int32_t>::max()) / 192u ||
      !blocks.ceil_div(32).try_area(&groups) ||
      groups > std::numeric_limits<uint32_t>::max() / (3u * 65536u))
    return Status::InvalidArgument(
        "AQ strategy metadata geometry exceeds shader limits");
  const Params p = Parameters(blocks);
  const size_t tiles = size_t(p.tiles_x) * p.tiles_y;
  AqStrategyMetadataStoragePlan candidate;
  candidate.selection_bytes = count + tiles;
  candidate.elements = {count,     size_t(p.chunks) * 7,
                        tiles,     35,
                        4,         2 * count,
                        2 * count, 6 * count,
                        tiles + 1, count};
  DeviceScratchLayoutPlan arena;
  for (size_t i = 0; i < candidate.scratch.size(); ++i) {
    const size_t n = candidate.elements[i];
    const auto status = arena.AddPlane(DeviceElementType::kI32, {n, 1}, n, 256,
                                       &candidate.scratch[i]);
    if (!status.ok())
      return status;
  }
  candidate.capacity_bytes = arena.capacity_bytes();
  *out = candidate;
  return Status::Ok();
}

Status MetalAqStrategyMetadata::Validate(
    GpuBackend &backend, const AqStrategyMetadataDescriptor &descriptor) {
  auto *metal = dynamic_cast<MetalBackend *>(&backend);
  if (!metal)
    return Status::Unavailable("AQ metadata requires Metal");
  AqStrategyMetadataStoragePlan plan;
  Status status =
      ComputeAqStrategyMetadataStoragePlan(descriptor.blocks, &plan);
  if (!status.ok())
    return status;
  std::array<DeviceMemoryRange, kMetadataPlaneCount + 1> ranges;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const ConstDevicePlaneView view =
        i == 0 ? descriptor.selection : descriptor.planes[i - 1];
    const size_t n = i == 0 ? plan.selection_bytes : plan.elements[i - 1];
    if (view.element_type !=
            (i == 0 ? DeviceElementType::kU8 : DeviceElementType::kI32) ||
        view.extent.height != 1 || view.extent.width < n ||
        view.offset_bytes % DeviceElementSize(view.element_type) != 0)
      return Status::InvalidArgument(
          "AQ metadata plane geometry or type is invalid");
    status = ComputeDevicePlaneRange(view, backend.id(), &ranges[i]);
    if (!status.ok())
      return status;
    if (!MetalBackend::AsMetalBuffer(*view.buffer))
      return Status::InvalidArgument("AQ metadata requires Metal buffers");
    for (size_t j = 0; j < i; ++j)
      if (DeviceRangesOverlap(ranges[i], ranges[j]))
        return Status::InvalidArgument("AQ metadata buffers overlap");
  }
  for (const auto &pipeline : metal->aq_pipelines_.strategy_metadata) {
    if (!pipeline || pipeline->maxTotalThreadsPerThreadgroup() < 256 ||
        pipeline->threadExecutionWidth() != 32)
      return Status::Unavailable(
          "AQ metadata needs 256 threads and 32-lane SIMD groups");
  }
  return Status::Ok();
}

void MetalAqStrategyMetadata::Encode(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const AqStrategyMetadataDescriptor &descriptor) {
  const Params p = Parameters(descriptor.blocks);
  const size_t count = descriptor.blocks.width * descriptor.blocks.height;
  const size_t tiles = size_t(p.tiles_x) * p.tiles_y;
  // Reset, stable family ranks, tile counts, global prefixes, grouped anchors,
  // tile-local CfL records, group-local coefficient destinations.
  const std::array<size_t, 7> threads{
      4,     size_t(p.chunks) * 256,         tiles, 32, count,
      tiles, size_t(p.groups_x) * p.groups_y};
  for (size_t stage = 0; stage < threads.size(); ++stage) {
    encoder->setComputePipelineState(
        backend.aq_pipelines_.strategy_metadata[stage].get());
    encoder->setBuffer(
        MetalBackend::AsMetalBuffer(*descriptor.selection.buffer)->handle(),
        descriptor.selection.offset_bytes, 0);
    for (size_t i = 0; i < descriptor.planes.size(); ++i) {
      const auto &view = descriptor.planes[i];
      encoder->setBuffer(MetalBackend::AsMetalBuffer(*view.buffer)->handle(),
                         view.offset_bytes, i + 1);
    }
    encoder->setBytes(&p, sizeof(p), 11);
    if (stage == 1 || stage == 3)
      DispatchMetalThreadgroups(encoder,
                                MTL::Size(stage == 1 ? p.chunks : 1, 1, 1),
                                MTL::Size(stage == 1 ? 256 : 32, 1, 1));
    else
      DispatchMetalThreads(encoder, MTL::Size(threads[stage], 1, 1),
                           MTL::Size(256, 1, 1));
  }
}

void MetalAqStrategyMetadata::EncodeSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {
  Encode(backend, encoder,
         *static_cast<const AqStrategyMetadataDescriptor *>(context));
}

Status
MetalAqStrategyMetadata::Submit(GpuBackend &backend,
                                const AqStrategyMetadataDescriptor &descriptor,
                                std::unique_ptr<GpuSubmission> *submission) {
  if (!submission)
    return Status::InvalidArgument("AQ metadata submission is null");
  submission->reset();
  Status status = Validate(backend, descriptor);
  if (!status.ok())
    return status;
  return static_cast<MetalBackend &>(backend).SubmitCompute(
      "gjxl AQ strategy metadata", &EncodeSubmission, &descriptor, submission);
}
} // namespace gjxl::metal_internal
