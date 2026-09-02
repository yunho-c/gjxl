// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/libjxl_tail_internal.h"

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/codestream.h"
#include "codec/vardct_frame.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_precomputed_vardct.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/padded_bytes.h"

namespace gjxl::codestream_internal {

bool LibjxlTailExperimentAvailable() noexcept { return true; }

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

  const Extent2D pixels = frame.geometry().frame();
  const Extent2D blocks = frame.geometry().block_grid().blocks;
  const SimpleVarDctCodestreamProfile &profile = frame.profile();
  const jxl::PrecomputedVarDctFrame bridge_frame{
      .xsize = pixels.width,
      .ysize = pixels.height,
      .xsize_blocks = blocks.width,
      .ysize_blocks = blocks.height,
      .color_channel_count = profile.color_channel_count,
      .extra_channel_count = profile.extra_channel_count,
      .pass_count = profile.pass_count,
      .xyb_encoded = true,
  };
  const jxl::PrecomputedVarDctEncodeOptions bridge_options{
      .effort = options.effort,
      .num_threads = options.thread_count,
  };

  JxlMemoryManager memory_manager;
  if (!jxl::MemoryManagerInit(&memory_manager, nullptr)) {
    return Status::OutOfMemory(
        "Could not initialize the libjxl tail memory manager");
  }
  jxl::PaddedBytes bridge_output(&memory_manager);
  const jxl::Status bridge_status = jxl::EncodePrecomputedVarDctFrame(
      &memory_manager, bridge_frame, bridge_options, &bridge_output);
  if (!bridge_status) {
    if (bridge_status.code() == jxl::StatusCode::kUnsupported) {
      return Status::Unsupported(
          "Pinned libjxl precomputed VarDCT tail is not implemented");
    }
    return Status::Internal("Pinned libjxl VarDCT tail bridge failed");
  }
  if (bridge_output.empty()) {
    return Status::Internal("Pinned libjxl tail produced an empty codestream");
  }

  try {
    std::vector<uint8_t> candidate(bridge_output.begin(), bridge_output.end());
    *output = std::move(candidate);
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Could not copy the libjxl tail output");
  } catch (const std::length_error &) {
    return Status::OutOfMemory("Libjxl tail output is too large");
  }
  return Status::Ok();
}

} // namespace gjxl::codestream_internal
