// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "gpu/metal/metal_embedded_library_internal.h"

int main() {
  std::ifstream file(GJXL_METALLIB_PATH, std::ios::binary);
  if (!file) { std::cerr << "Unable to read complete metallib\n"; return EXIT_FAILURE; }
  const std::vector<uint8_t> expected((std::istreambuf_iterator<char>(file)), {});
  const auto embedded = gjxl::metal_internal::EmbeddedMetalLibrary();
  if (expected.empty() || !std::equal(embedded.begin(), embedded.end(), expected.begin(), expected.end())) {
    std::cerr << "Embedded Metal payload differs: linked=" << embedded.size()
              << " file=" << expected.size() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
