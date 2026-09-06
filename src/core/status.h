// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace gjxl {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kUnsupported,
  kUnavailable,
  kOutOfMemory,
  kFailedPrecondition,
  kSubmissionFailed,
  kDeviceError,
  kInternal,
};

class Status {
public:
  Status() = default;

  Status(StatusCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

  [[nodiscard]] static Status Ok() {
    return {};
  }

  [[nodiscard]] static Status InvalidArgument(std::string message) {
    return {StatusCode::kInvalidArgument, std::move(message)};
  }

  [[nodiscard]] static Status Unsupported(std::string message) {
    return {StatusCode::kUnsupported, std::move(message)};
  }

  [[nodiscard]] static Status Unavailable(std::string message) {
    return {StatusCode::kUnavailable, std::move(message)};
  }

  [[nodiscard]] static Status OutOfMemory(std::string message) {
    return {StatusCode::kOutOfMemory, std::move(message)};
  }

  // A planner/coverage contract failure, not a candidate-local allocation
  // failure. Keep the existing public error code while preserving the reason
  // through internal retries and exception adapters.
  [[nodiscard]] static Status ResourcePlanExceeded(std::string message) {
    Status status = OutOfMemory(std::move(message));
    status.resource_plan_exceeded_ = true;
    return status;
  }

  [[nodiscard]] static Status FailedPrecondition(std::string message) {
    return {StatusCode::kFailedPrecondition, std::move(message)};
  }

  [[nodiscard]] static Status SubmissionFailed(std::string message) {
    return {StatusCode::kSubmissionFailed, std::move(message)};
  }

  [[nodiscard]] static Status DeviceError(std::string message) {
    return {StatusCode::kDeviceError, std::move(message)};
  }

  [[nodiscard]] static Status Internal(std::string message) {
    return {StatusCode::kInternal, std::move(message)};
  }

  [[nodiscard]] bool ok() const noexcept {
    return code_ == StatusCode::kOk;
  }

  [[nodiscard]] StatusCode code() const noexcept {
    return code_;
  }

  [[nodiscard]] bool resource_plan_exceeded() const noexcept {
    return resource_plan_exceeded_;
  }

  [[nodiscard]] std::string_view message() const noexcept {
    return message_;
  }

private:
  StatusCode code_ = StatusCode::kOk;
  bool resource_plan_exceeded_ = false;
  std::string message_;
};

} // gjxl
