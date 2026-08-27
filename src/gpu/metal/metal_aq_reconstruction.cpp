// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_aq_evaluation_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/quantizer.h"

namespace gjxl::metal_internal {
namespace {

inline constexpr NS::UInteger kAqThreadCount = 256;

template <typename T>
Status UploadContiguous(MetalBackend &backend, std::span<const T> source,
                        DevicePlaneView destination) {

  return backend.CopyHostToDevice(*destination.buffer, source.data(),
                                  source.size_bytes(),
                                  destination.offset_bytes);
}

void BindPlane(MTL::ComputeCommandEncoder *encoder, DevicePlaneView plane,
               NS::UInteger index) {

  MetalBuffer *buffer = dynamic_cast<MetalBuffer *>(plane.buffer);
  encoder->setBuffer(buffer->handle(), plane.offset_bytes, index);
}

void DispatchThreads1d(MTL::ComputeCommandEncoder *encoder,
                       size_t thread_count) {

  encoder->dispatchThreads(
      MTL::Size(static_cast<NS::UInteger>(thread_count), 1, 1),
      MTL::Size(kAqThreadCount, 1, 1));
}

Status CopyReadback(MetalBackend &backend, DevicePlaneView source,
                    void *destination, size_t size_bytes) {

  return backend.CopyDeviceToHost(*source.buffer, destination, size_bytes,
                                  source.offset_bytes);
}

MetalPreparedAqEvaluation *
AsMetalPrepared(PreparedAqEvaluation &prepared) noexcept {

  return dynamic_cast<MetalPreparedAqEvaluation *>(&prepared);
}

} // namespace

void MetalPreparedAqEvaluation::EncodeReconstructionSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {

  const auto &self = *static_cast<const MetalPreparedAqEvaluation *>(context);
  encoder->setComputePipelineState(
      backend.aq_pipelines_.reset_reconstruction.get());
  BindPlane(encoder, self.gathered_pixels_, 0);
  BindPlane(encoder, self.forward_coefficients_, 1);
  BindPlane(encoder, self.quantized_coefficients_, 2);
  BindPlane(encoder, self.reconstruction_coefficients_, 3);
  BindPlane(encoder, self.dc_, 4);
  for (size_t channel = 0; channel < 3; ++channel) {
    BindPlane(encoder, self.reconstructed_[channel], channel + 5);
  }
  BindPlane(encoder, self.reconstruction_error_, 8);
  encoder->setBytes(&self.reset_params_, sizeof(self.reset_params_), 9);
  DispatchThreads1d(encoder,
                    std::max({self.coefficient_value_count_,
                              3 * self.block_count_, self.pixel_count_}));

  const MetalBuffer *anchors =
      MetalBackend::AsMetalBuffer(*self.anchors_.buffer);
  const MetalBuffer *gathered =
      MetalBackend::AsMetalBuffer(*self.gathered_pixels_.buffer);
  MetalBuffer *forward =
      MetalBackend::AsMetalBuffer(*self.forward_coefficients_.buffer);
  const MetalBuffer *reconstruction =
      MetalBackend::AsMetalBuffer(*self.reconstruction_coefficients_.buffer);
  MetalBuffer *inverse_output =
      MetalBackend::AsMetalBuffer(*self.gathered_pixels_.buffer);

  for (size_t batch_index = 0; batch_index < self.batches_.size();
       ++batch_index) {
    const AqStrategyBatch &batch = self.batches_[batch_index];
    if (batch.anchor_count == 0)
      continue;
    const AqReconstructionParams &params =
        self.reconstruction_params_[batch_index];
    const size_t batch_value_count =
        3 * batch.anchor_count * batch.coefficient_count;
    const size_t coefficient_offset_bytes =
        batch.coefficient_offset * sizeof(float);

    encoder->setComputePipelineState(
        backend.aq_pipelines_.gather_transform_pixels.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      BindPlane(encoder, self.coding_[channel], channel);
    }
    encoder->setBuffer(anchors->handle(), self.anchors_.offset_bytes, 3);
    BindPlane(encoder, self.gathered_pixels_, 4);
    encoder->setBytes(&params, sizeof(params), 5);
    DispatchThreads1d(encoder, batch_value_count);

    backend.EncodeTransformBatch(
        encoder, TransformDirection::kForward, batch.strategy, *gathered,
        self.gathered_pixels_.offset_bytes + coefficient_offset_bytes, *forward,
        self.forward_coefficients_.offset_bytes + coefficient_offset_bytes,
        3 * batch.anchor_count);

    encoder->setComputePipelineState(
        backend.aq_pipelines_.encode_reconstruction_coefficients.get());
    BindPlane(encoder, self.anchors_, 0);
    BindPlane(encoder, self.quant_tables_, 1);
    BindPlane(encoder, self.raw_quant_, 2);
    BindPlane(encoder, self.y_to_x_, 3);
    BindPlane(encoder, self.y_to_b_, 4);
    BindPlane(encoder, self.forward_coefficients_, 5);
    BindPlane(encoder, self.quantized_coefficients_, 6);
    BindPlane(encoder, self.reconstruction_coefficients_, 7);
    BindPlane(encoder, self.dc_, 8);
    BindPlane(encoder, self.reconstruction_error_, 9);
    encoder->setBytes(&params, sizeof(params), 10);
    encoder->dispatchThreadgroups(
        MTL::Size(static_cast<NS::UInteger>(batch.anchor_count), 1, 1),
        MTL::Size(kAqThreadCount, 1, 1));

    backend.EncodeTransformBatch(
        encoder, TransformDirection::kInverse, batch.strategy, *reconstruction,
        self.reconstruction_coefficients_.offset_bytes +
            coefficient_offset_bytes,
        *inverse_output,
        self.gathered_pixels_.offset_bytes + coefficient_offset_bytes,
        3 * batch.anchor_count);

    encoder->setComputePipelineState(
        backend.aq_pipelines_.scatter_reconstructed_pixels.get());
    BindPlane(encoder, self.anchors_, 0);
    BindPlane(encoder, self.gathered_pixels_, 1);
    for (size_t channel = 0; channel < 3; ++channel) {
      BindPlane(encoder, self.reconstructed_[channel], channel + 2);
    }
    encoder->setBytes(&params, sizeof(params), 5);
    DispatchThreads1d(encoder, batch_value_count);
  }
}

Status MetalPreparedAqEvaluation::RunReconstruction(
    AqEvaluationInput input,
    MetalAqReconstructionSnapshotForTesting *snapshot) {

  if (snapshot == nullptr) {
    return Status::InvalidArgument("AQ reconstruction snapshot output is null");
  }
  Status status = ValidateInput(input);
  if (!status.ok())
    return status;
  status = BeginOperation();
  if (!status.ok())
    return status;
  status = UploadInput(input);
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  for (AqReconstructionParams &params : reconstruction_params_) {
    params.global_scale = input.quantizer.global_scale;
  }

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
      "gjxl prepared AQ coefficient reconstruction",
      &MetalPreparedAqEvaluation::EncodeReconstructionSubmission, this,
      &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
               ? Status::Internal("AQ reconstruction returned no submission")
               : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  status = WaitForOperation();
  if (!status.ok())
    return status;

  uint32_t device_error = 0;
  status = CopyReadback(*backend_, reconstruction_error_, &device_error,
                        sizeof(device_error));
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  if (device_error != 0) {
    Invalidate();
    return Status::DeviceError(
        "Metal AQ coefficient reconstruction detected invalid numeric input");
  }
  status =
      CopyReadback(*backend_, forward_coefficients_, forward_readback_.data(),
                   forward_readback_.size() * sizeof(float));
  if (status.ok()) {
    status = CopyReadback(*backend_, quantized_coefficients_,
                          quantized_readback_.data(),
                          quantized_readback_.size() * sizeof(int32_t));
  }
  if (status.ok()) {
    status = CopyReadback(*backend_, dc_, dc_readback_.data(),
                          dc_readback_.size() * sizeof(float));
  }
  for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
    status =
        CopyReadback(*backend_, reconstructed_[channel],
                     reconstructed_readback_[channel].data(),
                     reconstructed_readback_[channel].size() * sizeof(float));
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  try {
    MetalAqReconstructionSnapshotForTesting result;
    result.block_extent = block_extent_;
    result.pixel_extent = coding_extent_;
    result.transforms.reserve(row_major_anchors_.size());
    for (const AqAnchor &anchor : row_major_anchors_) {
      const AqStrategyBatch &batch = batches_[anchor.batch_index];
      MetalAqTransformSnapshotForTesting transform{
          .block_x = anchor.block_x,
          .block_y = anchor.block_y,
          .strategy = anchor.strategy,
      };
      const size_t channel_stride =
          batch.anchor_count * batch.coefficient_count;
      for (size_t channel = 0; channel < 3; ++channel) {
        const size_t offset = batch.coefficient_offset +
                              channel * channel_stride +
                              anchor.index_in_batch * batch.coefficient_count;
        transform.forward_coefficients[channel].assign(
            forward_readback_.begin() + static_cast<std::ptrdiff_t>(offset),
            forward_readback_.begin() +
                static_cast<std::ptrdiff_t>(offset + batch.coefficient_count));
        transform.quantized_coefficients[channel].assign(
            quantized_readback_.begin() + static_cast<std::ptrdiff_t>(offset),
            quantized_readback_.begin() +
                static_cast<std::ptrdiff_t>(offset + batch.coefficient_count));
      }
      result.transforms.push_back(std::move(transform));
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      result.dc[channel].assign(
          dc_readback_.begin() +
              static_cast<std::ptrdiff_t>(channel * block_count_),
          dc_readback_.begin() +
              static_cast<std::ptrdiff_t>((channel + 1) * block_count_));
      result.reconstructed_opsin[channel] = reconstructed_readback_[channel];
    }
    *snapshot = std::move(result);
  } catch (const std::bad_alloc &) {
    CompleteOperation();
    return Status::OutOfMemory(
        "Unable to allocate AQ reconstruction diagnostic snapshot");
  } catch (const std::length_error &) {
    CompleteOperation();
    return Status::InvalidArgument(
        "AQ reconstruction diagnostic snapshot is too large");
  }

  CompleteOperation();
  return Status::Ok();
}

void MetalPreparedAqEvaluation::EncodeQuantizationProbeSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {

  const auto &self = *static_cast<const MetalPreparedAqEvaluation *>(context);
  encoder->setComputePipelineState(
      backend.aq_pipelines_.quantization_probe.get());
  BindPlane(encoder, self.quant_probe_input_, 0);
  BindPlane(encoder, self.quant_tables_, 1);
  BindPlane(encoder, self.quant_probe_quantized_, 2);
  BindPlane(encoder, self.quant_probe_dequantized_, 3);
  BindPlane(encoder, self.reconstruction_error_, 4);
  encoder->setBytes(&self.quant_probe_params_, sizeof(self.quant_probe_params_),
                    5);
  DispatchThreads1d(encoder, self.quant_probe_params_.coefficient_count);
}

Status MetalPreparedAqEvaluation::RunQuantizationProbe(
    const MetalAqQuantizationProbeForTesting &probe,
    std::vector<int32_t> *quantized, std::vector<float> *dequantized) {

  if (quantized == nullptr || dequantized == nullptr) {
    return Status::InvalidArgument("AQ quantization probe output is null");
  }
  const AcStrategyInfo *info = GetAcStrategyInfo(probe.strategy);
  if (info == nullptr || info->coefficient_count() == 0 ||
      info->coefficient_count() > maximum_coefficient_count_ ||
      probe.coefficients.size() != info->coefficient_count()) {
    return Status::InvalidArgument(
        "AQ quantization probe strategy or coefficient count is invalid");
  }
  if (static_cast<uint8_t>(probe.channel) >
          static_cast<uint8_t>(XybChannel::kB) ||
      probe.raw_quant < 1 || probe.raw_quant > kMaxRawQuant ||
      !std::isfinite(probe.matrix_multiplier) ||
      probe.matrix_multiplier <= 0.0f) {
    return Status::InvalidArgument(
        "AQ quantization probe parameters are invalid");
  }
  Quantizer quantizer;
  Status status = Quantizer::Create(probe.quantizer, &quantizer);
  if (!status.ok())
    return status;
  status = BeginOperation();
  if (!status.ok())
    return status;
  status = UploadContiguous(*backend_, probe.coefficients, quant_probe_input_);
  const uint32_t zero = 0;
  if (status.ok()) {
    status = backend_->CopyHostToDevice(*reconstruction_error_.buffer, &zero,
                                        sizeof(zero),
                                        reconstruction_error_.offset_bytes);
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  quant_probe_params_ = {
      static_cast<uint32_t>(info->coefficient_count()),
      static_cast<uint32_t>(probe.strategy),
      static_cast<uint32_t>(probe.channel),
      probe.raw_quant,
      probe.quantizer.global_scale,
      probe.matrix_multiplier,
  };

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
      "gjxl AQ quantization probe",
      &MetalPreparedAqEvaluation::EncodeQuantizationProbeSubmission, this,
      &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok() ? Status::Internal(
                             "AQ quantization probe returned no submission")
                       : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  status = WaitForOperation();
  if (!status.ok())
    return status;

  uint32_t device_error = 0;
  status = CopyReadback(*backend_, reconstruction_error_, &device_error,
                        sizeof(device_error));
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  if (device_error != 0) {
    Invalidate();
    return Status::DeviceError(
        "Metal AQ quantization probe detected non-finite or overflowing input");
  }
  const size_t count = info->coefficient_count();
  status = CopyReadback(*backend_, quant_probe_quantized_,
                        quant_probe_quantized_readback_.data(),
                        count * sizeof(int32_t));
  if (status.ok()) {
    status = CopyReadback(*backend_, quant_probe_dequantized_,
                          quant_probe_dequantized_readback_.data(),
                          count * sizeof(float));
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  try {
    std::vector<int32_t> quantized_result(
        quant_probe_quantized_readback_.begin(),
        quant_probe_quantized_readback_.begin() +
            static_cast<std::ptrdiff_t>(count));
    std::vector<float> dequantized_result(
        quant_probe_dequantized_readback_.begin(),
        quant_probe_dequantized_readback_.begin() +
            static_cast<std::ptrdiff_t>(count));
    *quantized = std::move(quantized_result);
    *dequantized = std::move(dequantized_result);
  } catch (const std::bad_alloc &) {
    CompleteOperation();
    return Status::OutOfMemory(
        "Unable to allocate AQ quantization probe output");
  } catch (const std::length_error &) {
    CompleteOperation();
    return Status::InvalidArgument("AQ quantization probe output is too large");
  }
  CompleteOperation();
  return Status::Ok();
}

Status RunMetalAqReconstructionForTesting(
    PreparedAqEvaluation &prepared, AqEvaluationInput input,
    MetalAqReconstructionSnapshotForTesting *snapshot) {

  MetalPreparedAqEvaluation *metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ reconstruction requires a Metal prepared evaluation");
  }
  return metal->RunReconstruction(input, snapshot);
}

Status RunMetalAqQuantizationProbeForTesting(
    PreparedAqEvaluation &prepared,
    const MetalAqQuantizationProbeForTesting &probe,
    std::vector<int32_t> *quantized, std::vector<float> *dequantized) {

  MetalPreparedAqEvaluation *metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ quantization probe requires a Metal prepared evaluation");
  }
  return metal->RunQuantizationProbe(probe, quantized, dequantized);
}

} // namespace gjxl::metal_internal
