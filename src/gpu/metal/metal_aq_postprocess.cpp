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
#include <type_traits>
#include <utility>

namespace gjxl::metal_internal {
namespace {

template <typename T> bool ValidHostPlaneLayout(PlaneView<T> plane) noexcept {
  if (!plane.valid())
    return false;
  if (plane.extent.height - 1 >
      (std::numeric_limits<size_t>::max() - plane.extent.width) /
          plane.stride) {
    return false;
  }
  const size_t elements =
      (plane.extent.height - 1) * plane.stride + plane.extent.width;
  using Value = std::remove_const_t<T>;
  return elements <= std::numeric_limits<size_t>::max() / sizeof(Value);
}

Status ValidateDirectInput(ConstImage3FView image,
                           ConstPlaneF32View inverse_sigma,
                           Extent2D coding_extent, Extent2D block_extent) {
  if (!image.valid() || image.extent() != coding_extent ||
      !std::ranges::all_of(image.plane,
                           [](ConstPlaneF32View plane) {
                             return ValidHostPlaneLayout(plane);
                           }) ||
      !ValidHostPlaneLayout(inverse_sigma) ||
      inverse_sigma.extent != block_extent) {
    return Status::InvalidArgument(
        "Metal AQ postprocess input geometry is invalid");
  }
  for (ConstPlaneF32View plane : image.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.Row(y)[x])) {
          return Status::InvalidArgument(
              "Metal AQ postprocess input contains a non-finite sample");
        }
      }
    }
  }
  for (size_t y = 0; y < inverse_sigma.extent.height; ++y) {
    for (size_t x = 0; x < inverse_sigma.extent.width; ++x) {
      const float value = inverse_sigma.Row(y)[x];
      if (!std::isfinite(value) || value >= 0.0f) {
        return Status::InvalidArgument(
            "Metal AQ postprocess inverse sigma is invalid");
      }
    }
  }
  return Status::Ok();
}

template <typename T>
Status UploadPlane(MetalBackend &backend, PlaneView<const T> source,
                   DevicePlaneView destination) {
  const size_t row_bytes = source.extent.width * sizeof(T);
  for (size_t y = 0; y < source.extent.height; ++y) {
    Status status = backend.CopyHostToDevice(
        *destination.buffer, source.Row(y), row_bytes,
        destination.offset_bytes + y * destination.row_stride * sizeof(T));
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

void BindImage(MTL::ComputeCommandEncoder *encoder,
               const std::array<DevicePlaneView, 3> &image,
               NS::UInteger first_index) {
  for (size_t channel = 0; channel < image.size(); ++channel) {
    MetalBuffer *buffer = dynamic_cast<MetalBuffer *>(image[channel].buffer);
    encoder->setBuffer(buffer->handle(), image[channel].offset_bytes,
                       first_index + channel);
  }
}

void BindPlane(MTL::ComputeCommandEncoder *encoder, DevicePlaneView plane,
               NS::UInteger index) {
  MetalBuffer *buffer = dynamic_cast<MetalBuffer *>(plane.buffer);
  encoder->setBuffer(buffer->handle(), plane.offset_bytes, index);
}

Status CopyPlane(MetalBackend &backend, DevicePlaneView source,
                 std::vector<float> *destination) {
  return backend.CopyDeviceToHost(*source.buffer, destination->data(),
                                  destination->size() * sizeof(float),
                                  source.offset_bytes);
}

MetalPreparedAqEvaluation *
AsMetalPrepared(PreparedAqEvaluation &prepared) noexcept {
  return dynamic_cast<MetalPreparedAqEvaluation *>(&prepared);
}

} // namespace

std::array<DevicePlaneView, 3>
MetalPreparedAqEvaluation::FinalFilteredImage() const noexcept {
  return final_filter_scratch_index_ < 0 ? reconstructed_
                                         : filter_scratch_[static_cast<size_t>(
                                               final_filter_scratch_index_)];
}

void MetalPreparedAqEvaluation::EncodePostprocess(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder) const {
  std::array<DevicePlaneView, 3> current = reconstructed_;
  size_t filter_stage = 0;

  if (options_.profile.loop_filter.gaborish) {
    const std::array<DevicePlaneView, 3> output = filter_scratch_[0];
    encoder->setComputePipelineState(backend.aq_pipelines_.gaborish.get());
    BindImage(encoder, current, 0);
    BindImage(encoder, output, 3);
    BindPlane(encoder, reconstruction_error_, 6);
    encoder->setBytes(&gaborish_params_, sizeof(gaborish_params_), 7);
    MetalBackend::DispatchPlane(encoder, coding_extent_);
    current = output;
    ++filter_stage;
  }

  const uint32_t iterations =
    options_.profile.loop_filter.epf_options.iterations;
  const uint32_t first_pass = iterations == 3 ? 0 : 1;
  for (uint32_t pass = first_pass; pass < first_pass + iterations; ++pass) {
    const size_t scratch_index = filter_stage % 2;
    const std::array<DevicePlaneView, 3> output =
        filter_scratch_[scratch_index];
    encoder->setComputePipelineState(backend.aq_pipelines_.epf.get());
    BindImage(encoder, current, 0);
    BindPlane(encoder, inverse_sigma_, 3);
    BindImage(encoder, output, 4);
    BindPlane(encoder, reconstruction_error_, 7);
    encoder->setBytes(&epf_params_[pass], sizeof(epf_params_[pass]), 8);
    MetalBackend::DispatchPlane(encoder, coding_extent_);
    current = output;
    ++filter_stage;
  }

  encoder->setComputePipelineState(backend.aq_pipelines_.opsin_to_linear.get());
  BindImage(encoder, current, 0);
  BindImage(encoder, reconstructed_linear_, 3);
  BindPlane(encoder, reconstruction_error_, 6);
  encoder->setBytes(&opsin_to_linear_params_, sizeof(opsin_to_linear_params_),
                    7);
  MetalBackend::DispatchPlane(encoder, source_extent_);
}

void MetalPreparedAqEvaluation::EncodePostprocessSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {
  const auto &self = *static_cast<const MetalPreparedAqEvaluation *>(context);
  self.EncodePostprocess(backend, encoder);
}

void MetalPreparedAqEvaluation::EncodeReconstructionAndPostprocessSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {
  EncodeReconstructionSubmission(backend, encoder, context);
  const auto &self = *static_cast<const MetalPreparedAqEvaluation *>(context);
  self.EncodePostprocess(backend, encoder);
}

Status MetalPreparedAqEvaluation::FinishPostprocess(
    MetalAqPostprocessSnapshotForTesting *snapshot) {
  Status status = WaitForOperation();
  if (!status.ok())
    return status;

  uint32_t device_error = 0;
  status = backend_->CopyDeviceToHost(*reconstruction_error_.buffer,
                                      &device_error, sizeof(device_error),
                                      reconstruction_error_.offset_bytes);
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  if (device_error != 0) {
    Invalidate();
    return Status::DeviceError(
        "Metal AQ postprocessing detected an invalid numeric result");
  }

  const std::array<DevicePlaneView, 3> filtered = FinalFilteredImage();
  for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
    status = CopyPlane(*backend_, reconstructed_[channel],
                       &reconstructed_readback_[channel]);
  }
  for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
    status =
        CopyPlane(*backend_, filtered[channel], &filtered_readback_[channel]);
  }
  for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
    status = CopyPlane(*backend_, reconstructed_linear_[channel],
                       &linear_readback_[channel]);
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  try {
    MetalAqPostprocessSnapshotForTesting result;
    result.coding_extent = coding_extent_;
    result.source_extent = source_extent_;
    for (size_t channel = 0; channel < 3; ++channel) {
      result.reconstructed_opsin[channel] = reconstructed_readback_[channel];
      result.filtered_opsin[channel] = filtered_readback_[channel];
      result.reconstructed_linear[channel] = linear_readback_[channel];
    }
    *snapshot = std::move(result);
  } catch (const std::bad_alloc &) {
    CompleteOperation();
    return Status::OutOfMemory(
        "Unable to allocate AQ postprocess diagnostic snapshot");
  } catch (const std::length_error &) {
    CompleteOperation();
    return Status::InvalidArgument(
        "AQ postprocess diagnostic snapshot is too large");
  }

  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::RunPostprocess(
    ConstImage3FView reconstructed_opsin, ConstPlaneF32View epf_inverse_sigma,
    MetalAqPostprocessSnapshotForTesting *snapshot) {
  if (snapshot == nullptr) {
    return Status::InvalidArgument(
        "AQ postprocess diagnostic snapshot output is null");
  }
  Status status = ValidateDirectInput(reconstructed_opsin, epf_inverse_sigma,
                                      coding_extent_, block_extent_);
  if (!status.ok())
    return status;
  status = BeginOperation();
  if (!status.ok())
    return status;

  for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
    status = UploadPlane(*backend_, reconstructed_opsin.plane[channel],
                         reconstructed_[channel]);
  }
  if (status.ok()) {
    status = UploadPlane(*backend_, epf_inverse_sigma, inverse_sigma_);
  }
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

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
      "gjxl prepared AQ postprocessing",
      &MetalPreparedAqEvaluation::EncodePostprocessSubmission, this,
      &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
               ? Status::Internal("AQ postprocess returned no submission")
               : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  return FinishPostprocess(snapshot);
}

Status MetalPreparedAqEvaluation::RunReconstructionAndPostprocess(
    AqEvaluationInput input, MetalAqPostprocessSnapshotForTesting *snapshot) {
  if (snapshot == nullptr) {
    return Status::InvalidArgument(
        "AQ chained postprocess snapshot output is null");
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
    params.quant_dc = input.quantizer.quant_dc;
  }

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
      "gjxl prepared AQ reconstruction and postprocessing",
      &MetalPreparedAqEvaluation::EncodeReconstructionAndPostprocessSubmission,
      this, &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok() ? Status::Internal(
                             "AQ chained postprocess returned no submission")
                       : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  return FinishPostprocess(snapshot);
}

Status MetalPreparedAqEvaluation::GetPostprocessPlan(
    MetalAqPostprocessPlanForTesting *plan) const {
  if (plan == nullptr) {
    return Status::InvalidArgument("AQ postprocess plan output is null");
  }
  *plan = {
      .filter_scratch_images = filter_scratch_image_count_,
      .gaborish_dispatches = options_.profile.loop_filter.gaborish ? 1u : 0u,
      .epf_dispatches = options_.profile.loop_filter.epf_options.iterations,
      .color_dispatches = 1,
      .copy_dispatches = 0,
  };
  return Status::Ok();
}

Status RunMetalAqPostprocessForTesting(
    PreparedAqEvaluation &prepared, ConstImage3FView reconstructed_opsin,
    ConstPlaneF32View epf_inverse_sigma,
    MetalAqPostprocessSnapshotForTesting *snapshot) {
  MetalPreparedAqEvaluation *metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ postprocessing requires a Metal prepared evaluation");
  }
  return metal->RunPostprocess(reconstructed_opsin, epf_inverse_sigma,
                               snapshot);
}

Status RunMetalAqReconstructionAndPostprocessForTesting(
    PreparedAqEvaluation &prepared, AqEvaluationInput input,
    MetalAqPostprocessSnapshotForTesting *snapshot) {
  MetalPreparedAqEvaluation *metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ chained postprocessing requires a Metal prepared evaluation");
  }
  return metal->RunReconstructionAndPostprocess(input, snapshot);
}

Status
GetMetalAqPostprocessPlanForTesting(PreparedAqEvaluation &prepared,
                                    MetalAqPostprocessPlanForTesting *plan) {
  MetalPreparedAqEvaluation *metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ postprocess plan requires a Metal prepared evaluation");
  }
  return metal->GetPostprocessPlan(plan);
}

} // namespace gjxl::metal_internal
