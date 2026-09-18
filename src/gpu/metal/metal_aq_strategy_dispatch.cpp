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
