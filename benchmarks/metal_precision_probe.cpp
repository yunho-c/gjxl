// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Precision-study encoder and independent decoded-image evaluator.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "codec/butteraugli.h"
#include "codestream/workflow_internal.h"
#include "gpu/metal/metal_backend.h"
#include "io/pfm.h"

namespace {
void Check(gjxl::Status s) {
  if (!s.ok()) throw std::runtime_error(std::string(s.message()));
}
gjxl::MetalBackendOptions Options() {
  const auto m = gjxl::MetalDctImplementation::kSimdgroupMatmul;
  return {.forward_dct8=m, .inverse_dct8=m,
    .forward_dct16x16=m, .inverse_dct16x16=m,
    .forward_dct32x32=m, .inverse_dct32x32=m,
    .forward_dct16x8=m, .inverse_dct16x8=m,
    .forward_dct8x16=m, .inverse_dct8x16=m,
    .forward_dct32x16=m, .inverse_dct32x16=m,
    .forward_dct16x32=m, .inverse_dct16x32=m,
    .ac_residual_inverse=gjxl::MetalAcResidualInverseMode::kFusedTuned};
}
}
int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(17);
    if (argc == 7 && std::string(argv[1]) == "encode") {
      gjxl::Image3FBuffer input;
      Check(gjxl::io::ReadPfm(argv[3], &input));
      std::unique_ptr<gjxl::GpuBackend> backend;
      Check(gjxl::CreateMetalBackend(argv[2], Options(), &backend));
      std::vector<uint8_t> bytes;
      gjxl::VarDctEncodingSummary summary;
      Check(gjxl::codestream_internal::EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
        input.const_view(), {.butteraugli_target=std::stof(argv[5]),
          .effort=std::stoi(argv[6]), .backend=gjxl::VarDctBackendPreference::kMetal,
          .metal_aq_mode=gjxl::GpuAdaptiveQuantizationMode::kFullyResident},
        backend.get(), true, &bytes, &summary));
      std::ofstream file(argv[4], std::ios::binary);
      file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      if (!file) throw std::runtime_error("Unable to write codestream");
      std::cout << "{\"bytes\":" << bytes.size() << ",\"strategies\":[";
      for (size_t i=0; i<summary.strategy_counts.size(); ++i)
        std::cout << (i ? "," : "") << summary.strategy_counts[i];
      std::cout << "],\"scores\":[";
      for (size_t i=0; i<summary.score_history.size(); ++i)
        std::cout << (i ? "," : "") << summary.score_history[i];
      std::cout << "]}\n";
    } else if (argc == 4 && std::string(argv[1]) == "metric") {
      gjxl::Image3FBuffer reference, decoded;
      Check(gjxl::io::ReadPfm(argv[2], &reference));
      Check(gjxl::io::ReadPfm(argv[3], &decoded));
      if (reference.extent() != decoded.extent())
        throw std::runtime_error("Image dimensions differ");
      const auto e = reference.extent();
      std::vector<float> map(e.width * e.height);
      double score = 0, squared = 0, maximum = 0;
      Check(gjxl::ComputeButteraugliDistance(reference.const_view(), decoded.const_view(),
        {}, {map.data(), e, e.width}, &score));
      for (size_t c=0; c<3; ++c) for (size_t i=0; i<map.size(); ++i) {
        const double delta = double(reference.plane(c)[i]) - decoded.plane(c)[i];
        squared += delta * delta;
        maximum = std::max(maximum, std::abs(delta));
      }
      const double mse = squared / (3 * map.size());
      if (!std::isfinite(score) || !std::isfinite(mse))
        throw std::runtime_error("Nonfinite metric");
      std::cout << "{\"butteraugli\":" << score << ",\"mse\":" << mse
                << ",\"max_abs_rgb\":" << maximum << "}\n";
    } else {
      throw std::runtime_error("usage: probe encode METALLIB INPUT.pfm OUTPUT.jxl DISTANCE EFFORT | metric REFERENCE.pfm DECODED.pfm");
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
