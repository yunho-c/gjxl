// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/loop_filter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <vector>

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

  try {
    size_t pixel_count = 0;
    if (!input.extent().try_area(&pixel_count)) {
      return Status::InvalidArgument(
        "Loop-filter image dimensions are too large");
    }
    std::array<std::vector<float>, 3> scratch;
    for (std::vector<float>& plane : scratch) {
      plane.resize(pixel_count);
    }
    const Image3FView scratch_view{{
      PlaneF32View{scratch[0].data(), input.extent(), input.width()},
      PlaneF32View{scratch[1].data(), input.extent(), input.width()},
      PlaneF32View{scratch[2].data(), input.extent(), input.width()},
    }};
    const ConstImage3FView scratch_const_view{{
      ConstPlaneF32View{scratch[0].data(), input.extent(), input.width()},
      ConstPlaneF32View{scratch[1].data(), input.extent(), input.width()},
      ConstPlaneF32View{scratch[2].data(), input.extent(), input.width()},
    }};

    if (options.gaborish) {
      Status status = ApplyGaborish(
        input,
        options.gaborish_options,
        scratch_view);
      if (!status.ok()) {
        return status;
      }
    } else {
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t y = 0; y < input.height(); ++y) {
          std::copy_n(
            input.plane[channel].Row(y),
            input.width(),
            scratch_view.plane[channel].Row(y));
        }
      }
    }

    if (options.epf_options.iterations != 0) {
      return ApplyEpf(
        scratch_const_view,
        inverse_sigma,
        options.epf_options,
        output);
    }

    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < input.height(); ++y) {
        std::copy_n(
          scratch_view.plane[channel].Row(y),
          input.width(),
          output.plane[channel].Row(y));
      }
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate loop-filter scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Loop-filter image dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace gjxl
