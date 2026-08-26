// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstdlib>
#include <iostream>

#include "codec/butteraugli.h"

int main() {
  constexpr gjxl::Extent2D kExtent{1, 1};
  std::array<float, 3> reference = {0.1f, 0.2f, 0.3f};
  std::array<float, 3> distorted = {0.1f, 0.2f, 0.3f};
  const gjxl::ConstImage3FView reference_view{{
    gjxl::ConstPlaneF32View{&reference[0], kExtent, 1},
    gjxl::ConstPlaneF32View{&reference[1], kExtent, 1},
    gjxl::ConstPlaneF32View{&reference[2], kExtent, 1},
  }};
  const gjxl::ConstImage3FView distorted_view{{
    gjxl::ConstPlaneF32View{&distorted[0], kExtent, 1},
    gjxl::ConstPlaneF32View{&distorted[1], kExtent, 1},
    gjxl::ConstPlaneF32View{&distorted[2], kExtent, 1},
  }};
  float distance = 17.0f;
  double score = 19.0;

  const gjxl::Status status = gjxl::ComputeButteraugliDistance(
    reference_view,
    distorted_view,
    {},
    {&distance, kExtent, 1},
    &score);
  if (status.code() != gjxl::StatusCode::kUnavailable ||
      distance != 17.0f ||
      score != 19.0) {
    std::cerr << "Disabled Butteraugli backend did not fail atomically\n";
    return EXIT_FAILURE;
  }

  std::cout << "Disabled Butteraugli backend reports unavailable.\n";
  return EXIT_SUCCESS;
}
