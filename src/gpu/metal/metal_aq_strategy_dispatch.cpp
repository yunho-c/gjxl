// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/coefficient_order_population_internal.h"
#include "gpu/metal/metal_aq_evaluation_internal.h"
#include <cstring>

namespace gjxl::metal_internal {
using gjxl_aq_dispatch::Record;
static_assert(sizeof(Record) == 544);
static_assert(7 * sizeof(Record) <= 4096);
static_assert(sizeof(AqReconstructionParams) == sizeof(Record::reconstruction));
static_assert(offsetof(AqReconstructionParams, anchor_offset) == 8 * 4);
static_assert(offsetof(AqReconstructionParams, anchor_count) == 9 * 4);
static_assert(offsetof(AqReconstructionParams, coefficient_offset) == 10 * 4);
static_assert(sizeof(AqBlockReductionParams) ==
              sizeof(Record::block_reduction));
static_assert(offsetof(AqBlockReductionParams, anchor_offset) == 4 * 4);
static_assert(offsetof(AqBlockReductionParams, anchor_count) == 5 * 4);
static_assert(sizeof(AqQuantFieldAdjustmentParams) ==
              sizeof(Record::adjustment));
static_assert(offsetof(AqQuantFieldAdjustmentParams, anchor_offset) == 4);
static_assert(offsetof(AqQuantFieldAdjustmentParams, anchor_count) == 8);

Status MetalPreparedAqEvaluation::BindStrategyDispatchForTesting(
    ConstDevicePlaneView families, DevicePlaneView parameters) {
  if (families.buffer || parameters.buffer) {
    if (!resident_quantization_ || frame_only_ ||
        final_transform_metadata_pending_)
      return Status::Unavailable(
          "Device strategy dispatch requires ordinary resident AQ");
    if (families.element_type != DeviceElementType::kI32 ||
        families.extent != Extent2D{35, 1} || families.offset_bytes % 4 ||
        parameters.element_type != DeviceElementType::kI32 ||
        parameters.extent != Extent2D{7 * sizeof(Record) / 4, 1} ||
        parameters.offset_bytes % 4)
      return Status::InvalidArgument(
          "Device strategy dispatch geometry is invalid");
    DeviceMemoryRange a, b;
    Status status = ComputeDevicePlaneRange(families, backend_->id(), &a);
    if (!status.ok())
      return status;
    status = ComputeDevicePlaneRange(parameters, backend_->id(), &b);
    if (!status.ok())
      return status;
    if (DeviceRangesOverlap(a, b) ||
        !MetalBackend::AsMetalBuffer(*families.buffer) ||
        !MetalBackend::AsMetalBuffer(*parameters.buffer))
      return Status::InvalidArgument(
          "Device strategy dispatch buffers are invalid");
    for (auto type : kSupportedAqStrategies) {
      const auto &pipeline = backend_->transform_pipelines_[size_t(type)];
      if (!pipeline.forward_image.state || !pipeline.inverse_image.state)
        return Status::Unavailable(
            "Device strategy dispatch needs direct image transforms");
    }
  }
  Status status = BeginOperation();
  if (!status.ok())
    return status;
  strategy_dispatch_families_ = families;
  strategy_dispatch_ = parameters;
  CompleteOperation();
  return Status::Ok();
}

void MetalPreparedAqEvaluation::EncodeStrategyDispatch(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder) const {
  std::array<Record, 7> templates{};
  for (size_t f = 0; f < templates.size(); ++f) {
    auto &r = templates[f];
    auto reconstruction = reconstruction_params_[f];
    reconstruction.group_major_output = 0;
    std::memcpy(r.reconstruction, &reconstruction, sizeof(reconstruction));
    reconstruction.group_major_output = 1;
    std::memcpy(r.completed_reconstruction, &reconstruction,
                sizeof(reconstruction));
    const auto &source =
        resident_ac_strategy_inputs_ && options_.profile.loop_filter.gaborish
            ? reconstructed_
            : coding_;
    r.forward[3] = uint32_t(source[0].row_stride);
    r.inverse[3] = uint32_t(reconstructed_[0].row_stride);
    std::memcpy(r.adjustment, &quant_field_adjustment_params_[f],
                sizeof(r.adjustment));
    const auto family = vardct_frame_internal::OrderPopulationFamily(
        batches_[f].coefficient_count);
    r.population[2] =
        uint32_t(vardct_frame_internal::kOrderPopulationOffsets[family]);
    r.population[3] = completed_sample_dct8_ ? 1u : 0u;
    std::memcpy(r.block_reduction, &block_reduction_params_[f],
                sizeof(r.block_reduction));
  }
  encoder->setComputePipelineState(
      backend.aq_pipelines_.strategy_metadata[7].get());
  RecordMetalComputePipelineState(
      backend.aq_pipelines_.strategy_metadata[7].get());
  encoder->setBuffer(
      MetalBackend::AsMetalBuffer(*strategy_dispatch_families_.buffer)
          ->handle(),
      strategy_dispatch_families_.offset_bytes, 0);
  encoder->setBytes(templates.data(), sizeof(templates), 1);
  encoder->setBuffer(
      MetalBackend::AsMetalBuffer(*strategy_dispatch_.buffer)->handle(),
      strategy_dispatch_.offset_bytes, 2);
  DispatchMetalThreads(encoder, MTL::Size(7, 1, 1), MTL::Size(32, 1, 1));
}

void MetalPreparedAqEvaluation::BindStrategyParameters(
    MTL::ComputeCommandEncoder *encoder, size_t batch, size_t member_offset,
    size_t binding) const {
  encoder->setBuffer(
      MetalBackend::AsMetalBuffer(*strategy_dispatch_.buffer)->handle(),
      strategy_dispatch_.offset_bytes + batch * sizeof(Record) + member_offset,
      binding);
}

void MetalPreparedAqEvaluation::DispatchStrategy(
    MTL::ComputeCommandEncoder *encoder, size_t batch,
    gjxl_aq_dispatch::Dispatch dispatch, MTL::Size threads) const {
  // Enabled only for unprofiled resident policy: current profile records
  // require host-known dimensions and cannot truthfully represent indirect
  // grids.
  encoder->dispatchThreadgroups(
      MetalBackend::AsMetalBuffer(*strategy_dispatch_.buffer)->handle(),
      strategy_dispatch_.offset_bytes + batch * sizeof(Record) +
          offsetof(Record, groups) + size_t(dispatch) * 3 * sizeof(uint32_t),
      threads);
}

Status BindMetalAqStrategyDispatchForTesting(PreparedAqEvaluation &prepared,
                                             ConstDevicePlaneView families,
                                             DevicePlaneView parameters) {
  auto *metal = dynamic_cast<MetalPreparedAqEvaluation *>(&prepared);
  if (!metal)
    return Status::InvalidArgument(
        "Device strategy dispatch requires Metal AQ");
  return metal->BindStrategyDispatchForTesting(families, parameters);
}
} // namespace gjxl::metal_internal

namespace gjxl::metal_internal {
bool MetalPreparedAqEvaluation::SupportsResidentStrategies() const noexcept {
  if (!resident_strategy_metadata_enabled_ || !resident_quantization_ ||
      frame_only_ || options_.metric != AqEvaluationMetric::kButteraugli)
    return false;
  for (auto type : kSupportedAqStrategies) {
    const auto &p = backend_->transform_pipelines_[size_t(type)];
    if (!p.forward_image.state || !p.inverse_image.state)
      return false;
  }
  return true;
}

Status MetalPreparedAqEvaluation::ReconfigureResidentStrategies(
    ConstDevicePlaneView selection, ConstPlaneU8View sharpness) {
  if (!SupportsResidentStrategies())
    return Status::Unavailable("Resident strategy metadata was not prepared");
  if (!sharpness.valid() || sharpness.extent != block_extent_ ||
      sharpness.stride < block_extent_.width ||
      selection.buffer == persistent_.backing_buffer() ||
      selection.buffer == staging_.backing_buffer())
    return Status::InvalidArgument(
        "Resident strategy input geometry or ownership is invalid");
  auto descriptor = resident_strategy_metadata_;
  descriptor.selection = selection;
  auto flat = [](DevicePlaneView plane, size_t count) {
    plane.extent = {count, 1};
    plane.row_stride = count;
    return plane;
  };
  descriptor.planes[kMetadataStrategies] = flat(strategies_, 2 * block_count_);
  descriptor.planes[kMetadataAnchors] = flat(anchors_, 2 * block_count_);
  descriptor.planes[kMetadataColorRecords] =
      flat(color_transform_records_, 6 * block_count_);
  descriptor.planes[kMetadataColorOffsets] = color_tile_offsets_;
  // This destination is unused for owned-frame output; coefficient coding may
  // overwrite it after metadata construction. Completed output replaces it.
  descriptor.planes[kMetadataDestinations] =
      flat(quantized_coefficients_, block_count_);
  Status status = MetalAqStrategyMetadata::Validate(*backend_, descriptor);
  if (!status.ok())
    return status;
  for (size_t y = 0; y < block_extent_.height; ++y)
    for (size_t x = 0; x < block_extent_.width; ++x)
      if (sharpness.Row(y)[x] >= 8)
        return Status::InvalidArgument(
            "Resident strategy sharpness is invalid");
  status = BeginOperation();
  if (!status.ok())
    return status;
  try {
    resource_budget_internal::ManagedVector<uint8_t> packed(block_count_);
    for (size_t y = 0; y < block_extent_.height; ++y)
      std::copy_n(sharpness.Row(y), block_extent_.width,
                  packed.data() + y * block_extent_.width);
    status =
        backend_->CopyHostToDevice(*epf_sharpness_.buffer, packed.data(),
                                   block_count_, epf_sharpness_.offset_bytes);
    if (!status.ok()) {
      Invalidate();
      return status;
    }
    epf_sharpness_host_ = std::move(packed);
    resident_strategy_metadata_ = descriptor;
    strategy_dispatch_families_ = descriptor.planes[kMetadataFamilies];
    strategy_dispatch_ = resident_strategy_parameters_;
    resident_strategy_pending_ = true;
    resident_search_batch_count_ = 0;
    final_transform_metadata_pending_ = true;
    anchor_count_ = block_count_;
    final_cfl_params_.transform_count = uint32_t(block_count_);
    invariant_color_correlation_ready_ = false;
    resident_forward_coefficients_ready_ = false;
    resident_color_correlation_pending_ = false;
    resident_color_correlation_readback_needed_ = false;
    CompleteOperation();
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure &e) {
    Invalidate();
    return e.status();
  } catch (const std::bad_alloc &) {
    Invalidate();
    return Status::OutOfMemory("Resident strategy sharpness allocation failed");
  }
}

Status MetalPreparedAqEvaluation::ReconfigureResidentStrategySearch(
    std::span<const AcStrategyCandidateBatch> batches,
    AcStrategyDeviceSelection selection, ConstPlaneU8View sharpness) {
  if (!SupportsResidentStrategies())
    return Status::Unavailable("Resident strategy search was not prepared");
  if (batches.size() != resident_search_batches_.size() ||
      selection.block_extent != block_extent_)
    return Status::InvalidArgument(
        "Resident strategy search geometry is invalid");
  const auto aliases_aq = [&](const DeviceBuffer *buffer) {
    if (!buffer)
      return false;
    if (buffer == persistent_.backing_buffer() ||
        buffer == staging_.backing_buffer())
      return true;
    for (const auto &image : {original_, coding_})
      for (const auto &plane : image)
        if (buffer == plane.buffer)
          return true;
    return false;
  };
  if (aliases_aq(selection.output))
    return Status::InvalidArgument(
        "Resident selection output aliases AQ storage");
  std::array<MetalBackend::ValidatedAcStrategyBatch, 7> validated;
  size_t count = 0;
  for (const auto &batch : batches) {
    for (auto *buffer :
         {batch.scratch_a, batch.scratch_b, batch.rate_scratch, batch.costs})
      if (aliases_aq(buffer))
        return Status::InvalidArgument(
            "Resident search scratch aliases AQ storage");
    MetalBackend::ValidatedAcStrategyBatch candidate;
    Status status =
        backend_->ValidateAcStrategyCandidateBatch(batch, &candidate);
    if (!status.ok())
      return status;
    if (batch.candidate_count)
      validated[count++] = candidate;
  }
  MetalBackend::AcStrategyEncodeContext::Selection selected;
  Status status =
      backend_->ValidateAcStrategySelection(batches, selection, &selected);
  if (!status.ok())
    return status;
  const size_t bytes = block_count_ + tile_extent_.width * tile_extent_.height;
  status = ReconfigureResidentStrategies({selection.output,
                                          selection.offset_bytes,
                                          DeviceElementType::kU8,
                                          {bytes, 1},
                                          bytes},
                                         sharpness);
  if (!status.ok())
    return status;
  resident_search_batches_ = validated;
  resident_search_batch_count_ = count;
  resident_search_selection_ = selected;
  return Status::Ok();
}

void MetalPreparedAqEvaluation::EncodeResidentStrategyMetadata(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder) {
  if (resident_search_batch_count_) {
    const MetalBackend::AcStrategyEncodeContext context{
        {resident_search_batches_.data(), resident_search_batch_count_},
        &resident_search_selection_};
    MetalBackend::EncodeAcStrategySubmission(backend, encoder, &context);
  }
  // Reset before importing the metadata error; all subsequent policy resets
  // preserve it while resident_strategy_pending_ is true.
  EncodeReconstructionReset(backend, encoder);
  MetalAqStrategyMetadata::Encode(backend, encoder,
                                  resident_strategy_metadata_);
  encoder->setComputePipelineState(
      backend.aq_pipelines_.strategy_metadata[8].get());
  RecordMetalComputePipelineState(
      backend.aq_pipelines_.strategy_metadata[8].get());
  const auto control = resident_strategy_metadata_.planes[kMetadataControl];
  encoder->setBuffer(MetalBackend::AsMetalBuffer(*control.buffer)->handle(),
                     control.offset_bytes, 0);
  encoder->setBuffer(
      MetalBackend::AsMetalBuffer(*reconstruction_error_.buffer)->handle(),
      reconstruction_error_.offset_bytes, 1);
  DispatchMetalThreads(encoder, MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

Status MetalPreparedAqEvaluation::FinishResidentStrategyMetadata(
    AcStrategyGrid *selected) {
  const auto plane = resident_strategy_metadata_.selection;
  const size_t tiles = tile_extent_.width * tile_extent_.height;
  std::span<const std::byte> bytes;
  Status status = backend_->BorrowCompletedReadOnly(
      *plane.buffer, block_count_ + tiles, plane.offset_bytes, &bytes);
  if (!status.ok())
    return status;
  const auto *cells = reinterpret_cast<const uint8_t *>(bytes.data());
  for (size_t t = 0; t < tiles; ++t)
    if (cells[block_count_ + t])
      return Status::DeviceError(
          "Resident strategy selector reported an invalid tile");
  AcStrategyGrid grid;
  status = AcStrategyGrid::Create(block_extent_, &grid);
  if (!status.ok())
    return status;
  for (size_t y = 0; y < block_extent_.height; ++y)
    for (size_t x = 0; x < block_extent_.width; ++x) {
      const uint8_t cell = cells[y * block_extent_.width + x];
      if (cell & 1u) {
        status = grid.Set(x, y, AcStrategyType(cell >> 1));
        if (!status.ok())
          return Status::DeviceError("Resident strategy cover is invalid");
      }
    }
  if (!grid.complete())
    return Status::DeviceError("Resident strategy cover is incomplete");
  for (size_t y = 0; y < block_extent_.height; ++y)
    for (size_t x = 0; x < block_extent_.width; ++x) {
      AcStrategyCell cell;
      status = grid.Get(x, y, &cell);
      if (!status.ok() || cells[y * block_extent_.width + x] !=
                              ((uint8_t(cell.strategy) << 1) | cell.is_anchor))
        return Status::DeviceError(
            "Resident strategy ownership is inconsistent");
    }
  status = ReconfigureImpl(
      grid, {epf_sharpness_host_.data(), block_extent_, block_extent_.width},
      true);
  if (!status.ok())
    return status;
  resident_strategy_pending_ = false;
  resident_search_batch_count_ = 0;
  resident_strategy_metadata_.selection = {};
  if (selected)
    *selected = std::move(grid);
  return Status::Ok();
}
} // namespace gjxl::metal_internal
