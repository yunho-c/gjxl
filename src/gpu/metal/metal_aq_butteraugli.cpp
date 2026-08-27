// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_aq_evaluation_internal.h"

#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gjxl::metal_internal {

Status MetalPreparedAqEvaluation::RunButteraugli(
    AqEvaluationInput input, MetalAqButteraugliSnapshotForTesting *snapshot) {
  if (snapshot == nullptr) {
    return Status::InvalidArgument(
        "AQ Butteraugli diagnostic snapshot output is null");
  }
  Status status = ValidateInput(input);
  if (!status.ok())
    return status;

  MetalAqButteraugliSnapshotForTesting result;
  result.source_extent = source_extent_;
  size_t pixel_count = 0;
  if (!source_extent_.try_area(&pixel_count)) {
    return Status::InvalidArgument(
        "AQ Butteraugli diagnostic extent overflows");
  }
  try {
    result.distance_map.resize(pixel_count);
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory(
        "Unable to allocate AQ Butteraugli diagnostic snapshot");
  } catch (const std::length_error &) {
    return Status::InvalidArgument(
        "AQ Butteraugli diagnostic snapshot is too large");
  }

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
      "gjxl prepared AQ reconstruction and postprocessing",
      &MetalPreparedAqEvaluation::EncodeReconstructionAndPostprocessSubmission,
      this, &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
               ? Status::Internal("AQ reconstruction and postprocessing "
                                  "returned no submission")
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
        "Metal AQ reconstruction or postprocessing produced invalid numerics "
        "(flag " +
        std::to_string(device_error) + ")");
  }

  bool fail_submission = false;
  bool fail_completion = false;
  bool fail_readback = false;
  {
    std::lock_guard lock(mutex_);
    fail_submission = fail_next_butteraugli_submission_;
    fail_completion = fail_next_butteraugli_completion_;
    fail_readback = fail_next_butteraugli_readback_;
    fail_next_butteraugli_submission_ = false;
    fail_next_butteraugli_completion_ = false;
    fail_next_butteraugli_readback_ = false;
  }
  if (fail_submission || fail_completion) {
    backend_->ArmNextSubmissionFailureForTest(fail_submission, fail_completion);
  }

  if (butteraugli_ == nullptr) {
    Invalidate();
    return Status::Internal("Prepared AQ Butteraugli state is missing");
  }
  status = butteraugli_->Compare(
      {.distorted_linear_rgb = {{{reconstructed_linear_[0],
                                  reconstructed_linear_[1],
                                  reconstructed_linear_[2]}}},
       .distance_map = distance_map_,
       .score = score_});
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  if (fail_readback) {
    Invalidate();
    return Status::DeviceError("Injected AQ Butteraugli readback failure");
  }

  status = butteraugli_->ReadDistanceMap(
      {result.distance_map.data(), source_extent_, source_extent_.width});
  if (status.ok()) {
    status = butteraugli_->ReadScore(&result.score);
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  *snapshot = std::move(result);
  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::FailNextButteraugli(bool fail_submission,
                                                      bool fail_completion,
                                                      bool fail_readback) {
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ != State::kReady) {
    return Status::FailedPrecondition(
        "AQ Butteraugli failure injection requires a ready object");
  }
  fail_next_butteraugli_submission_ = fail_submission;
  fail_next_butteraugli_completion_ = fail_completion;
  fail_next_butteraugli_readback_ = fail_readback;
  return Status::Ok();
}

Status RunMetalAqButteraugliForTesting(
    PreparedAqEvaluation &prepared, AqEvaluationInput input,
    MetalAqButteraugliSnapshotForTesting *snapshot) {
  auto *metal = dynamic_cast<MetalPreparedAqEvaluation *>(&prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ Butteraugli diagnostic requires a Metal prepared evaluation");
  }
  return metal->RunButteraugli(input, snapshot);
}

Status FailNextMetalAqButteraugliForTesting(PreparedAqEvaluation &prepared,
                                            bool fail_submission,
                                            bool fail_completion,
                                            bool fail_readback) {
  auto *metal = dynamic_cast<MetalPreparedAqEvaluation *>(&prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument("AQ Butteraugli failure injection requires "
                                   "a Metal prepared evaluation");
  }
  return metal->FailNextButteraugli(fail_submission, fail_completion,
                                    fail_readback);
}

} // namespace gjxl::metal_internal
