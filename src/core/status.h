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
  kUnavailable,
  kOutOfMemory,
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

  [[nodiscard]] static Status Unavailable(std::string message) {
    return {StatusCode::kUnavailable, std::move(message)};
  }

  [[nodiscard]] static Status OutOfMemory(std::string message) {
    return {StatusCode::kOutOfMemory, std::move(message)};
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

  [[nodiscard]] std::string_view message() const noexcept {
    return message_;
  }

private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

}
