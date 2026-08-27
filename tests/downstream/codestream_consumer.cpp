// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "codestream/workflow.h"

int main() {
  constexpr gjxl::Extent2D kExtent{1, 1};
  const std::array<float, 3> pixel = {0.2f, 0.3f, 0.4f};
  const gjxl::ConstImage3FView image{{
    gjxl::ConstPlaneF32View{&pixel[0], kExtent, 1},
    gjxl::ConstPlaneF32View{&pixel[1], kExtent, 1},
    gjxl::ConstPlaneF32View{&pixel[2], kExtent, 1},
  }};
  std::vector<uint8_t> codestream;
  gjxl::VarDctEncodingSummary summary;
  const gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image, {}, &codestream, &summary);
  return status.ok() && codestream.size() >= 2 &&
      codestream[0] == 0xff && codestream[1] == 0x0a &&
      summary.extent == kExtent
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
}
