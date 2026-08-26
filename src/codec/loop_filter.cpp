// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/loop_filter.h"

#include <new>
#include <stdexcept>

#include "core/image_buffer.h"
#include "core/image_ops.h"

namespace gjxl {

Status ApplyLoopFilters(
  ConstImage3FView input,
  ConstPlaneF32View inverse_sigma,
  LoopFilterOptions options,
  Image3FView output) {

  if (!input.valid() ||
      !output.valid() ||
      input.extent() != output.extent()) {
    return Status::InvalidArgument(
      "Loop-filter images are invalid or differently sized");
  }

  const bool apply_epf = options.epf_options.iterations != 0;
  if (!options.gaborish) {
    if (apply_epf) {
      return ApplyEpf(input, inverse_sigma, options.epf_options, output);
    }
    CopyImage(input, output);
    return Status::Ok();
  }

  if (!apply_epf) {
    return ApplyGaborish(input, options.gaborish_options, output);
  }

  try {
    Image3FBuffer scratch(input.extent());
    Status status = ApplyGaborish(
      input,
      options.gaborish_options,
      scratch.view());
    if (!status.ok()) {
      return status;
    }
    return ApplyEpf(
      scratch.const_view(),
      inverse_sigma,
      options.epf_options,
      output);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate loop-filter scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Loop-filter image dimensions are too large");
  }
}

}  // namespace gjxl
