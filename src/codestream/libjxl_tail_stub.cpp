// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/libjxl_tail_internal.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "codec/codestream.h"

namespace gjxl::codestream_internal {

LibjxlTailContext::~LibjxlTailContext() = default;

bool LibjxlTailExperimentAvailable() noexcept { return false; }

Status CreateLibjxlTailContext(
    size_t thread_count, std::unique_ptr<LibjxlTailContext>* context) {
  if (context == nullptr || thread_count == 0) {
    return Status::InvalidArgument("Libjxl tail context request is invalid");
  }
  return Status::Unavailable(
      "GJXL was built without the libjxl tail experiment");
}

Status AuditVarDctStateWithLibjxl(const VarDctEncoderFrame &frame,
                                  LibjxlTailOptions options,
                                  LibjxlTailStateAudit *audit) {
  if (audit == nullptr) {
    return Status::InvalidArgument("Libjxl state-audit output is null");
  }
  if (options.effort < 1 || options.effort > 10 ||
      !std::isfinite(options.butteraugli_distance) ||
      options.butteraugli_distance <= 0.0f || options.thread_count == 0) {
    return Status::InvalidArgument("Libjxl tail options are invalid");
  }
  const Status validation = ValidateSimpleCodestreamFrame(frame);
  if (!validation.ok()) {
    return validation;
  }
  return Status::Unavailable(
      "GJXL was built without the libjxl tail experiment");
}

Status EncodeVarDctCodestreamWithLibjxl(const VarDctEncoderFrame &frame,
                                        LibjxlTailOptions options,
                                        std::vector<uint8_t> *output) {

  if (output == nullptr) {
    return Status::InvalidArgument("Libjxl tail output is null");
  }
  if (options.effort < 1 || options.effort > 10 ||
      !std::isfinite(options.butteraugli_distance) ||
      options.butteraugli_distance <= 0.0f || options.thread_count == 0) {
    return Status::InvalidArgument("Libjxl tail options are invalid");
  }
  const Status validation = ValidateSimpleCodestreamFrame(frame);
  if (!validation.ok()) {
    return validation;
  }
  return Status::Unavailable(
      "GJXL was built without the libjxl tail experiment");
}

Status EncodeVarDctCodestreamWithLibjxlProfiled(
    const VarDctEncoderFrame &frame, LibjxlTailOptions options,
    std::vector<uint8_t> *output, LibjxlTailProfile *profile) {
  if (profile == nullptr) {
    return Status::InvalidArgument("Libjxl tail profile output is null");
  }
  return EncodeVarDctCodestreamWithLibjxl(frame, options, output);
}

Status EncodeVarDctCodestreamWithLibjxlContextProfiled(
    const VarDctEncoderFrame& frame, LibjxlTailOptions options,
    LibjxlTailContext&, std::vector<uint8_t>* output,
    LibjxlTailProfile* profile) {
  if (profile == nullptr) {
    return Status::InvalidArgument("Libjxl tail profile output is null");
  }
  return EncodeVarDctCodestreamWithLibjxl(frame, options, output);
}

Status DecodeCodestreamPixelsWithLibjxl(
    const std::vector<uint8_t>& codestream, Extent2D expected_extent,
    std::vector<float>* pixels) {
  if (pixels == nullptr || codestream.empty() || expected_extent.empty()) {
    return Status::InvalidArgument("Libjxl decode request is invalid");
  }
  return Status::Unavailable(
      "GJXL was built without the libjxl tail experiment");
}

} // namespace gjxl::codestream_internal
