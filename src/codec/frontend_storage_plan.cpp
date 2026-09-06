// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/frontend_storage_plan.h"

#include <algorithm>
#include <limits>
#include <thread>

#include "codec/chroma_from_luma.h"
#include "codec/dct.h"
#include "codec/prepared_coefficients_internal.h"
#include "codec/vardct_frame.h"
#include "core/frame_geometry.h"

namespace gjxl::frontend_storage_internal {
namespace {
using enum resource_budget_internal::VectorCapacityPolicy;
using resource_budget_internal::ManagedVector;

Status Overflow() {
  return Status::OutOfMemory("Frontend representation storage bound overflows");
}

Status Geometry(Extent2D pixels, FrameGeometry *geometry, size_t *blocks,
                size_t *tiles, size_t *groups) {
  Status status = FrameGeometry::Create(pixels, geometry);
  if (!status.ok())
    return status;
  if (!geometry->block_grid().blocks.try_area(blocks) ||
      !ColorTileExtent(geometry->padded_frame()).try_area(tiles) ||
      !geometry->block_grid()
           .blocks.ceil_div(kVarDctAcGroupBlockDimension)
           .try_area(groups))
    return Overflow();
  return Status::Ok();
}

Status FieldScratch(Extent2D blocks, size_t planes, HostStorageBound *out) {
  size_t count = 0;
  if (out == nullptr || blocks.empty() || !blocks.try_area(&count))
    return Status::InvalidArgument("Quantization field extent is invalid");
  HostStorageBound bound;
  if (!bound.AddVector<float>(count, kFreshExact, planes))
    return Overflow();
  *out = bound;
  return Status::Ok();
}
} // namespace

Status ComputeColorTransformStoragePlan(Extent2D source, Extent2D destination,
                                        bool direct_padded,
                                        size_t cpu_thread_count,
                                        ColorTransformStoragePlan *out) {
  size_t dispatch_pixels = 0, destination_pixels = 0;
  if (out == nullptr || source.empty() || destination.empty() ||
      source.width > destination.width || source.height > destination.height ||
      (!direct_padded && source != destination))
    return Status::InvalidArgument("Color-transform plan extents are invalid");
  if (!Extent2D{destination.width, source.height}.try_area(&dispatch_pixels) ||
      !destination.try_area(&destination_pixels))
    return Overflow();
  ColorTransformStoragePlan plan;
  const size_t workers = cpu_thread_count == 0
                             ? kMaximumColorWorkers
                             : std::min(cpu_thread_count, kMaximumColorWorkers);
  plan.maximum_participants = dispatch_pixels < kMinimumParallelColorPixels
                                  ? 1
                                  : std::min(workers, source.height);
  if ((!direct_padded &&
       !plan.working.AddVector<float>(destination_pixels, kFreshExact, 3)) ||
      (plan.maximum_participants > 1 &&
       (!plan.working.AddVector<Status>(source.height, kFreshExact) ||
        !plan.working.AddVector<std::thread>(plan.maximum_participants,
                                             kFreshExact))))
    return Overflow();
  *out = plan;
  return Status::Ok();
}

Status ComputeLoopFilterStorageBound(Extent2D pixels, bool gaborish,
                                     size_t epf_iterations,
                                     HostStorageBound *out) {
  size_t count = 0;
  if (out == nullptr || pixels.empty() || epf_iterations > 3)
    return Status::InvalidArgument("Loop-filter storage options are invalid");
  if (!pixels.try_area(&count))
    return Overflow();
  HostStorageBound bound;
  // EPF uses one image at iterations 1/2, two at 3, and none at 0. When both
  // filters run, the outer image coexists first with Gaborish's one temporary,
  // then with EPF's temporaries. All destination/input backing is separate.
  const size_t epf_images = epf_iterations == 0   ? 0
                            : epf_iterations == 3 ? 2
                                                  : 1;
  const size_t images = !gaborish ? epf_images
                        : epf_iterations == 0
                            ? 1
                            : 1 + std::max(size_t{1}, epf_images);
  if (!bound.AddVector<float>(count, kFreshExact, 3 * images))
    return Overflow();
  *out = bound;
  return Status::Ok();
}

Status ComputeBlockReductionStorageBound(Extent2D blocks,
                                         HostStorageBound *out) {
  return FieldScratch(blocks, 1, out);
}

Status ComputeInitialQuantStoragePlan(Extent2D padded_extent,
                                      size_t cpu_thread_count,
                                      InitialQuantStoragePlan *out) {
  if (out == nullptr || !BlockGrid::IsPaddedPixelExtent(padded_extent))
    return Status::InvalidArgument("Initial quantization extent is invalid");
  size_t pixels = 0;
  if (!padded_extent.try_area(&pixels))
    return Overflow();
  const size_t row_groups = padded_extent.height / 4;
  const size_t workers =
      cpu_thread_count == 0
          ? kMaximumInitialQuantWorkers
          : std::min(cpu_thread_count, kMaximumInitialQuantWorkers);
  InitialQuantStoragePlan plan;
  plan.maximum_participants = pixels < kMinimumParallelInitialQuantValues
                                  ? 1
                                  : std::min(workers, row_groups);
  // The pixel mask and quarter-resolution erosion input survive both phases.
  HostStorageBound common;
  if (!common.AddVector<float>(pixels, kFreshExact) ||
      !common.AddVector<float>(pixels / 16, kFreshExact))
    return Overflow();
  auto rows = common;
  if (!rows.AddVector<float>(padded_extent.width, kFreshExact,
                             plan.maximum_participants) ||
      (plan.maximum_participants > 1 &&
       (!rows.AddVector<Status>(row_groups, kFreshExact) ||
        !rows.AddVector<std::thread>(plan.maximum_participants, kFreshExact))))
    return Overflow();
  // Workers (including their row arrays, statuses and thread vector) have
  // joined and returned before the quant/strategy fields and blurred mask.
  // Erosion, modulations and convolution have no other heap scratch.
  auto finish = common;
  if (!finish.AddVector<float>(pixels / 64, kFreshExact, 2) ||
      !finish.AddVector<float>(pixels, kFreshExact))
    return Overflow();
  plan.working = {std::max(rows.retained_bytes, finish.retained_bytes),
                  std::max(rows.peak_bytes, finish.peak_bytes)};
  *out = plan;
  return Status::Ok();
}

Status ComputeQuantFieldAdjustmentStorageBound(Extent2D block_extent,
                                               HostStorageBound *out) {
  return FieldScratch(block_extent, 1, out);
}

Status ComputeQuantizerSelectionStorageBound(Extent2D block_extent,
                                             HostStorageBound *out) {
  return FieldScratch(block_extent, 2, out);
}

Status ComputeColorCorrelationStoragePlan(Extent2D padded_extent,
                                          ColorCorrelationStorageMode mode,
                                          ColorCorrelationStoragePlan *out) {
  if (out == nullptr || !BlockGrid::IsPaddedPixelExtent(padded_extent))
    return Status::InvalidArgument("Color-correlation extent is invalid");
  ColorCorrelationStoragePlan plan;
  size_t pixels = 0;
  if (!padded_extent.try_area(&pixels) ||
      !ColorTileExtent(padded_extent).try_area(&plan.color_tiles))
    return Overflow();
  if (!plan.output.AddVector<int8_t>(plan.color_tiles, kFreshExact, 2))
    return Overflow();
  plan.working = plan.output;
  switch (mode) {
  case ColorCorrelationStorageMode::kCopy:
    break;
  case ColorCorrelationStorageMode::kInitialPixel:
    if (!plan.working.Add(plan.output))
      return Overflow();
    break;
  case ColorCorrelationStorageMode::kTransform: {
    // All four vectors reserve exactly the tile's coefficient count. Anchors
    // partition the tile, so pushes never exceed that reserve. Tile processing
    // is serial; DCT/multiplier-search scratch is entirely on stacks.
    const size_t coefficients =
        std::min(padded_extent.width, kColorTileDimension) *
        std::min(padded_extent.height, kColorTileDimension);
    if (!plan.working.AddVector<float>(coefficients, kFreshExact, 4))
      return Overflow();
    break;
  }
  default:
    return Status::InvalidArgument("Color-correlation storage mode is invalid");
  }
  *out = plan;
  return Status::Ok();
}

Status ComputeImage3FStorageBound(Extent2D extent, HostStorageBound *out) {
  size_t pixels = 0;
  if (out == nullptr || extent.empty() || !extent.try_area(&pixels))
    return Status::InvalidArgument("Image backing extent is invalid");
  HostStorageBound bound;
  if (!bound.AddVector<float>(pixels, kFreshExact, 3))
    return Overflow();
  *out = bound;
  return Status::Ok();
}

Status ComputeOwnedFrameStoragePlan(Extent2D frame_extent,
                                    OwnedFrameStoragePlan *out) {
  if (out == nullptr)
    return Status::InvalidArgument("Owned frame storage output is null");
  OwnedFrameStoragePlan plan;
  FrameGeometry geometry;
  Status status = Geometry(frame_extent, &geometry, &plan.blocks,
                           &plan.color_tiles, &plan.ac_groups);
  if (!status.ok())
    return status;
  constexpr size_t group_coefficients = 3 * kVarDctAcGroupCoefficientCapacity;
  if (plan.ac_groups > std::numeric_limits<size_t>::max() / group_coefficients)
    return Overflow();
  plan.ac_coefficients = plan.ac_groups * group_coefficients;
  auto &output = plan.output;
  // Strategy cells and sharpness; raw quantization plus three quantized DC
  // planes; three reconstructed DC planes; two CfL planes; group counts; AC.
  if (!output.AddVector<uint8_t>(plan.blocks, kFreshExact, 2) ||
      !output.AddVector<int32_t>(plan.blocks, kFreshExact, 4) ||
      !output.AddVector<float>(plan.blocks, kFreshExact, 3) ||
      !output.AddVector<int8_t>(plan.color_tiles, kFreshExact, 2) ||
      !output.AddVector<size_t>(plan.ac_groups, kFreshExact) ||
      !output.AddVector<int32_t>(plan.ac_coefficients, kFreshExact))
    return Overflow();
  *out = plan;
  return Status::Ok();
}

Status ComputePreparedForwardStoragePlan(Extent2D padded_extent,
                                         size_t cpu_thread_count,
                                         PreparedForwardStoragePlan *out) {
  if (out == nullptr || !BlockGrid::IsPaddedPixelExtent(padded_extent))
    return Status::InvalidArgument("Prepared coefficient extent is invalid");
  size_t pixels = 0, tiles = 0;
  PreparedForwardStoragePlan plan;
  if (!padded_extent.try_area(&pixels) ||
      !BlockGrid::FromPaddedPixelExtent(padded_extent)
           .blocks.try_area(&plan.maximum_transforms) ||
      !ColorTileExtent(padded_extent).try_area(&tiles) ||
      tiles == std::numeric_limits<size_t>::max())
    return Overflow();
  const size_t workers =
      cpu_thread_count == 0
          ? kMaximumForwardWorkers
          : std::min(cpu_thread_count, kMaximumForwardWorkers);
  plan.maximum_participants = pixels < kMinimumParallelForwardCoefficients
                                  ? 1
                                  : std::min(workers, plan.maximum_transforms);
  using prepared_coefficients_internal::PreparedTransform;
  auto &output = plan.output;
  if (!output.AddVector<float>(pixels, kFreshExact, 3) ||
      !output.AddVector<PreparedTransform>(plan.maximum_transforms, kGrowing) ||
      !output.AddVector<size_t>(tiles + 1, kFreshExact) ||
      !output.AddVector<size_t>(plan.maximum_transforms, kGrowing))
    return Overflow();
  plan.working = output;
  // Every anchor occurs in exactly one per-tile list; sum of their growth
  // bounds <= the bound on the summed anchor counts, including replacement.
  // Lists remain alive during transform dispatch and flattened-index creation.
  if (!plan.working.AddVector<ManagedVector<size_t>>(tiles, kFreshExact) ||
      !plan.working.AddVector<size_t>(plan.maximum_transforms, kGrowing) ||
      (plan.maximum_participants > 1 &&
       (!plan.working.AddVector<Status>(plan.maximum_transforms, kFreshExact) ||
        !plan.working.AddVector<std::thread>(plan.maximum_participants,
                                             kFreshExact))))
    return Overflow();
  *out = plan;
  return Status::Ok();
}

Status ComputeCoefficientReconstructionStorageBound(Extent2D frame_extent,
                                                    HostStorageBound *out) {
  if (out == nullptr)
    return Status::InvalidArgument("Reconstruction storage output is null");
  FrameGeometry geometry;
  size_t blocks = 0, tiles = 0, groups = 0;
  Status status = Geometry(frame_extent, &geometry, &blocks, &tiles, &groups);
  if (!status.ok())
    return status;
  HostStorageBound bound;
  status = ComputeImage3FStorageBound(geometry.padded_frame(), &bound);
  if (!status.ok())
    return status;
  const Extent2D block_extent = geometry.block_grid().blocks;
  size_t coefficients = 0, dc = 0;
  // Fixed format table: geometry and the current CPU support predicate bound
  // every possible strategy, including both rectangular orientations.
  for (const auto &info : kAcStrategyInfos) {
    if (SupportsCpuDct(info.type) &&
        info.covered_blocks.width <= block_extent.width &&
        info.covered_blocks.height <= block_extent.height) {
      coefficients = std::max(coefficients, info.coefficient_count());
      dc = std::max(dc, info.covered_blocks.width * info.covered_blocks.height);
    }
  }
  if (!bound.AddVector<size_t>(groups, kFreshExact) ||
      !bound.AddVector<float>(coefficients, kFreshExact, 4) ||
      !bound.AddVector<float>(dc, kFreshExact))
    return Overflow();
  *out = bound;
  return Status::Ok();
}
} // namespace gjxl::frontend_storage_internal
