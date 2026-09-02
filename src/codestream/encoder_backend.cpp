// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/encoder_backend_internal.h"

#include <utility>
#include <vector>

#include "codec/vardct_frame.h"
#include "codestream/encoder.h"
#include "codestream/workflow_internal.h"

namespace gjxl::codestream_internal {

Status EncodeVarDctCodestreamWithBackend(
  const VarDctEncoderFrame& frame,
  VarDctCodestreamBackendOptions options,
  std::vector<uint8_t>* output,
  VarDctCodestreamBackendProfile* profile) {

  if (output == nullptr) {
    return Status::InvalidArgument("Codestream backend output is null");
  }
  VarDctCodestreamBackendProfile candidate_profile;
  candidate_profile.backend = options.backend;
  Status status;
  switch (options.backend) {
    case VarDctCodestreamBackend::kGjxl:
      status = profile == nullptr
        ? EncodeVarDctCodestream(frame, output)
        : EncodeVarDctCodestreamProfiled(
            frame, output, &candidate_profile.gjxl);
      if (status.ok()) {
        candidate_profile.total_nanoseconds =
          candidate_profile.gjxl.total_nanoseconds;
      }
      break;

    case VarDctCodestreamBackend::kLibjxl: {
      const LibjxlTailOptions libjxl_options{
        .effort = options.libjxl_effort,
        .butteraugli_distance = options.butteraugli_distance,
        .thread_count = options.libjxl_thread_count,
      };
      status = profile == nullptr
        ? EncodeVarDctCodestreamWithLibjxl(frame, libjxl_options, output)
        : EncodeVarDctCodestreamWithLibjxlProfiled(
            frame, libjxl_options, output, &candidate_profile.libjxl);
      if (status.ok()) {
        candidate_profile.total_nanoseconds =
          candidate_profile.libjxl.total_nanoseconds;
      }
      break;
    }

    default:
      return Status::InvalidArgument("Codestream backend is invalid");
  }
  if (!status.ok()) {
    return status;
  }
  if (profile != nullptr) {
    *profile = std::move(candidate_profile);
  }
  return Status::Ok();
}

Status EncodeLinearRgbVarDctCodestreamWithCodestreamBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  VarDctCodestreamBackendOptions codestream_options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingProfile* profile) {

  if (codestream_options.backend == VarDctCodestreamBackend::kLibjxl) {
    if (codestream_options.libjxl_effort < 1 ||
        codestream_options.libjxl_effort > 10 ||
        codestream_options.libjxl_thread_count == 0 ||
        codestream_options.libjxl_thread_count > kMaximumCpuThreadCount) {
      return Status::InvalidArgument(
        "Libjxl tail effort or thread count is invalid");
    }
    if (options.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
        options.rate_control_mode ==
          VarDctRateControlMode::kTargetBitsPerPixel) {
      return Status::InvalidArgument(
        "The libjxl tail experiment does not support target-size control");
    }
    if (!LibjxlTailExperimentAvailable()) {
      return Status::Unavailable(
        "GJXL was built without the libjxl tail experiment");
    }
  }
  return EncodeLinearRgbVarDctCodestreamWithSerializerForTesting(
    linear_rgb, options, codestream_options,
    EncodeVarDctCodestreamWithBackend, backend,
    backend_is_qualified_for_automatic, codestream, summary, profile);
}

}  // namespace gjxl::codestream_internal
