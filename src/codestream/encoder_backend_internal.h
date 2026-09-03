// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "codestream/encoder_internal.h"
#include "codestream/libjxl_tail_internal.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

namespace codestream_internal {

enum class VarDctCodestreamBackend {
  kGjxl,
  kLibjxl,
};

struct VarDctCodestreamBackendOptions {
  VarDctCodestreamBackend backend = VarDctCodestreamBackend::kGjxl;
  VarDctEntropyBehavior entropy_behavior =
    VarDctEntropyBehavior::kBalanced;
  VarDctCoefficientOrderBehavior coefficient_order_behavior =
    VarDctCoefficientOrderBehavior::kFull;
  int libjxl_effort = 7;
  float butteraugli_distance = 1.0f;
  size_t libjxl_thread_count = 1;
  LibjxlTailContext* libjxl_context = nullptr;
};

struct VarDctCodestreamBackendProfile {
  VarDctCodestreamBackend backend = VarDctCodestreamBackend::kGjxl;
  uint64_t total_nanoseconds = 0;
  VarDctCodestreamProfile gjxl;
  LibjxlTailProfile libjxl;

  friend bool operator==(const VarDctCodestreamBackendProfile&,
                         const VarDctCodestreamBackendProfile&) = default;
};

/// Serializes a completed frame through the explicitly requested backend.
/// There is no fallback. Failure leaves every caller-visible output unchanged.
[[nodiscard]] Status EncodeVarDctCodestreamWithBackend(
  const VarDctEncoderFrame& frame,
  VarDctCodestreamBackendOptions options,
  std::vector<uint8_t>* output,
  VarDctCodestreamBackendProfile* profile = nullptr);

}  // namespace codestream_internal
}  // namespace gjxl
