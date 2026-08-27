// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <Foundation/Foundation.hpp>

#include <string>
#include <string_view>

#include "core/status.h"

namespace gjxl::metal {

inline Status ErrorToStatus(
  const NS::Error* error,
  std::string_view operation) {

    std::string message(operation);
    message += ": ";

  if (error != nullptr) {
    const NS::String* description =
      error->localizedDescription();

    if (description != nullptr) {
      const char* utf8 =
        description->utf8String();

      if (utf8 != nullptr) {
        message += utf8;
        return Status::Internal(
          std::move(message));
      }
    }
  }

  message += "unknown Metal error";

  return Status::Internal(
    std::move(message));
}

inline Status ErrorToDeviceStatus(
  const NS::Error* error,
  std::string_view operation) {

  const Status status = ErrorToStatus(error, operation);
  return Status::DeviceError(std::string(status.message()));
}

}  // namespace gjxl::metal
