// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#import <Foundation/Foundation.h>

#include <string>
#include <string_view>

#include "core/status.h"

namespace gjxl::metal {

inline Status ErrorToStatus(
  NSError* error,
  std::string_view operation) {

    std::string message(operation);

    message += ": ";

    if (error != nil) {
      const char* description =
        [[error localizedDescription] UTF8String];

      if (description != nullptr) {
        message += description;
      } else {
        message += "unknown Metal error";
      }
    } else {
      message += "unknown Metal error";
    }

    return Status::Internal(std::move(message));
}

}
