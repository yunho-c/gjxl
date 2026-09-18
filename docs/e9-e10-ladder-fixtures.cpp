// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Bounded synthetic fixtures for comparing revisions and checking decoding.
// See e9-e10-ladder-validation.json for build/run commands. No timing loop.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "codestream/workflow.h"
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  std::filesystem::create_directories(argv[1]);
  constexpr gjxl::Extent2D extent{80, 72};
  std::array<std::vector<float>, 3> planes;
  for (size_t c = 0; c < 3; ++c) {
    planes[c].resize(extent.width * extent.height);
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        planes[c][y * extent.width + x] =
          0.25f + 0.15f * std::sin(float(x) * 0.13f + float(y) * 0.07f + float(c)) +
          0.05f * std::cos(float(x * y % 97) * 0.2f);
  }
  gjxl::ConstImage3FView image{{
    gjxl::ConstPlaneF32View{planes[0].data(), extent, extent.width},
    gjxl::ConstPlaneF32View{planes[1].data(), extent, extent.width},
    gjxl::ConstPlaneF32View{planes[2].data(), extent, extent.width}}};
  for (auto backend : {gjxl::VarDctBackendPreference::kCpu, gjxl::VarDctBackendPreference::kMetal}) {
    const std::string name = backend == gjxl::VarDctBackendPreference::kCpu ? "cpu" : "metal";
    for (int effort = 1; effort <= 10; ++effort) {
      std::vector<uint8_t> bytes;
      gjxl::VarDctEncodingSummary summary;
      const auto status = gjxl::EncodeLinearRgbVarDctCodestream(image,
        {.butteraugli_target = 1.2f, .effort = effort, .backend = backend}, &bytes, &summary);
      if (!status.ok()) { std::cerr << status.message() << '\n'; return 1; }
      const std::string path = std::string(argv[1]) + "/" + name + "-e" + std::to_string(effort) + ".jxl";
      std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      if (!out) return 1;
      std::cout << name << " e" << effort << " bytes=" << bytes.size() << " scores=" << summary.score_history.size() << '\n';
    }
  }
}
