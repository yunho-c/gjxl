// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>

#include "codec/profile_timing_internal.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl::gpu_profile_internal {

struct CommandBufferProfile {
  uint64_t start_host_nanoseconds = 0;
  uint64_t end_host_nanoseconds = 0;
  bool host_envelope_aligned = false;

  [[nodiscard]] bool available() const noexcept {
    return start_host_nanoseconds != 0 &&
      end_host_nanoseconds >= start_host_nanoseconds;
  }

  [[nodiscard]] uint64_t duration_nanoseconds() const noexcept {
    return available() ? end_host_nanoseconds - start_host_nanoseconds : 0;
  }
};

struct FrameEncodingProfile {
  profile_internal::HostInterval input_upload;
  profile_internal::HostInterval submission;
  profile_internal::HostInterval completion_wait;
  CommandBufferProfile command_buffer;
  profile_internal::HostInterval readback;
  profile_internal::HostInterval frame_assembly;
};

/// Optional internal diagnostic implemented by prepared backends that can
/// expose host/device timing without changing PreparedAqEvaluation's API.
class ProfiledFrameEncoder {
public:
  virtual ~ProfiledFrameEncoder() = default;

  [[nodiscard]] virtual Status EncodeFrameProfiled(
    AqEvaluationInput input,
    VarDctEncoderFrame* frame,
    FrameEncodingProfile* profile) = 0;

  [[nodiscard]] virtual Status ComputeInitialQuantizationProfiled(
    InitialQuantizationOptions options,
    InitialQuantFieldOutput output,
    QuantizerParams* quantizer,
    float quant_dc,
    FrameEncodingProfile* profile) = 0;

protected:
  ProfiledFrameEncoder() = default;
};

[[nodiscard]] inline ProfiledFrameEncoder* QueryProfiledFrameEncoder(
  PreparedAqEvaluation& prepared) noexcept {

  return dynamic_cast<ProfiledFrameEncoder*>(&prepared);
}

}  // namespace gjxl::gpu_profile_internal
