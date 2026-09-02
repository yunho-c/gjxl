// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/libjxl_tail_internal.h"

#include <cstdint>
#include <vector>

#include "codec/codestream.h"

namespace gjxl::codestream_internal {

bool LibjxlTailExperimentAvailable() noexcept { return false; }

Status EncodeVarDctCodestreamWithLibjxl(const VarDctEncoderFrame &frame,
                                        LibjxlTailOptions options,
                                        std::vector<uint8_t> *output) {

  if (output == nullptr) {
    return Status::InvalidArgument("Libjxl tail output is null");
  }
  if (options.effort < 1 || options.effort > 10 || options.thread_count == 0) {
    return Status::InvalidArgument("Libjxl tail options are invalid");
  }
  const Status validation = ValidateSimpleCodestreamFrame(frame);
  if (!validation.ok()) {
    return validation;
  }
  return Status::Unavailable(
      "GJXL was built without the libjxl tail experiment");
}

} // namespace gjxl::codestream_internal
