#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include "gpu/metal/metal_embedded_library_internal.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1], std::ios::binary);
  if (!file) return 2;
  const std::vector<uint8_t> expected((std::istreambuf_iterator<char>(file)), {});
  const auto actual = gjxl::metal_internal::EmbeddedMetalLibrary();
  std::cout << "embedded=" << actual.size() << " metallib=" << expected.size() << '\n';
  if (actual.size() != expected.size()) return 1;
  for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) return 1;
  return 0;
}
