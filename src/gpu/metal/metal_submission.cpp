// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_backend_internal.h"

#include <mutex>
#include <utility>

#include "gpu/metal/metal_status.h"

namespace gjxl::metal_internal {
namespace {

class MetalSubmission final : public GpuSubmission {
public:
  MetalSubmission(
    NS::SharedPtr<MTL::CommandBuffer> command_buffer,
    NS::SharedPtr<MTL::CommandQueue> command_queue,
    NS::SharedPtr<MTL::Device> device,
    bool test_fail_completion)
    : command_buffer_(std::move(command_buffer)),
      command_queue_(std::move(command_queue)),
      device_(std::move(device)),
      test_fail_completion_(test_fail_completion) {}

  ~MetalSubmission() override = default;

  Status Wait() override {
    std::call_once(wait_once_, [this] {
      auto pool = NS::TransferPtr(
        NS::AutoreleasePool::alloc()->init());
      command_buffer_->waitUntilCompleted();
      if (test_fail_completion_) {
        completion_status_ = Status::DeviceError(
          "Injected Metal command-buffer completion failure");
        return;
      }
      if (command_buffer_->status() == MTL::CommandBufferStatusError) {
        completion_status_ = metal::ErrorToDeviceStatus(
          command_buffer_->error(), "Metal command buffer");
      }
    });
    return completion_status_;
  }

private:
  NS::SharedPtr<MTL::CommandBuffer> command_buffer_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
  NS::SharedPtr<MTL::Device> device_;
  bool test_fail_completion_ = false;
  std::once_flag wait_once_;
  Status completion_status_;
};

}  // namespace

Status MetalBackend::SubmitCompute(
  const char* label,
  ComputeEncodeCallback encode,
  const void* context,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr) {
    return Status::InvalidArgument(
      "GPU submission output pointer is null");
  }
  submission->reset();
  if (label == nullptr || encode == nullptr) {
    return Status::Internal(
      "Metal compute submission callback is invalid");
  }
  const bool fail_submission =
    test_fail_submission_ || test_fail_next_submission_.exchange(false);
  const bool fail_completion =
    test_fail_completion_ || test_fail_next_completion_.exchange(false);
  if (fail_submission) {
    return Status::SubmissionFailed(
      "Injected Metal submission failure");
  }

  auto pool = NS::TransferPtr(
    NS::AutoreleasePool::alloc()->init());
  MTL::CommandBuffer* raw_command_buffer = command_queue_->commandBuffer();
  if (raw_command_buffer == nullptr) {
    return Status::SubmissionFailed(
      "Failed to create Metal command buffer");
  }
  auto command_buffer = NS::RetainPtr(raw_command_buffer);
  raw_command_buffer->setLabel(NS::String::string(
    label, NS::UTF8StringEncoding));
  MTL::ComputeCommandEncoder* encoder =
    raw_command_buffer->computeCommandEncoder();
  if (encoder == nullptr) {
    return Status::SubmissionFailed(
      "Failed to create Metal compute encoder");
  }

  encode(*this, encoder, context);
  encoder->endEncoding();
  std::unique_ptr<GpuSubmission> pending(
    new MetalSubmission(
      command_buffer,
      command_queue_,
      device_,
      fail_completion));
  raw_command_buffer->commit();
  RecordCommittedSubmission();
  *submission = std::move(pending);
  return Status::Ok();
}

void MetalBackend::ArmNextSubmissionFailureForTest(
  bool fail_submission,
  bool fail_completion) noexcept {

  test_fail_next_submission_.store(fail_submission);
  test_fail_next_completion_.store(fail_completion);
}

}  // namespace gjxl::metal_internal

namespace gjxl {

Status ArmNextMetalSubmissionFailureForTest(
  GpuBackend& backend,
  bool fail_submission,
  bool fail_completion) {

  auto* metal = dynamic_cast<metal_internal::MetalBackend*>(&backend);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "Submission failure injection requires a Metal backend");
  }
  metal->ArmNextSubmissionFailureForTest(
    fail_submission, fail_completion);
  return Status::Ok();
}

}  // namespace gjxl
