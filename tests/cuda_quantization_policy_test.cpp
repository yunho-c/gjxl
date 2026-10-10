// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Fixed-raw decisions must preserve reconstruction and scores across reuse.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include "codec/color_transform.h"
#include "codec/quantization_pipeline_internal.h"
#include "codestream/encoder.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/quantization_pipeline.h"
namespace {
constexpr gjxl::Extent2D kOriginalExtent{257, 17};
constexpr gjxl::Extent2D kPaddedExtent{264, 24};

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D extent, float fill = -777.0f)
      : extent(extent), stride(extent.width + 3) {
    for (std::vector<float> &values : plane) {
      values.assign(stride * extent.height, fill);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
        gjxl::PlaneF32View{plane[0].data(), extent, stride},
        gjxl::PlaneF32View{plane[1].data(), extent, stride},
        gjxl::PlaneF32View{plane[2].data(), extent, stride},
    }};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
        gjxl::ConstPlaneF32View{plane[0].data(), extent, stride},
        gjxl::ConstPlaneF32View{plane[1].data(), extent, stride},
        gjxl::ConstPlaneF32View{plane[2].data(), extent, stride},
    }};
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

void FillImages(ImageStorage *original, ImageStorage *padded) {
  for (size_t y = 0; y < kPaddedExtent.height; ++y) {
    const size_t source_y = std::min(y, kOriginalExtent.height - 1);
    for (size_t x = 0; x < kPaddedExtent.width; ++x) {
      const size_t source_x = std::min(x, kOriginalExtent.width - 1);
      const float fx = static_cast<float>(source_x);
      const float fy = static_cast<float>(source_y);
      const std::array<float, 3> rgb = {
          std::clamp(0.08f + 0.0028f * fx + 0.06f * std::sin(0.31f * fy), 0.0f,
                     1.0f),
          std::clamp(0.13f + 0.021f * fy + 0.04f * std::cos(0.071f * fx), 0.0f,
                     1.0f),
          ((source_x / 7 + source_y / 3) & 1u) == 0 ? 0.11f : 0.83f,
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        padded->plane[channel][y * padded->stride + x] = rgb[channel];
        if (x < kOriginalExtent.width && y < kOriginalExtent.height) {
          original->plane[channel][y * original->stride + x] = rgb[channel];
        }
      }
    }
  }
}

struct PipelineStorage {
  PipelineStorage(gjxl::Extent2D original_extent,
                  gjxl::Extent2D padded_extent)
      : block_extent{padded_extent.width / 8, padded_extent.height / 8},
        padded_extent(padded_extent),
        initial_quant(block_extent.width * block_extent.height),
        strategy_mask(block_extent.width * block_extent.height),
        pixel_mask(padded_extent.width * padded_extent.height),
        final_quant(block_extent.width * block_extent.height),
        block_distance(block_extent.width * block_extent.height),
        reconstructed(original_extent) {}

  gjxl::Extent2D block_extent;
  gjxl::Extent2D padded_extent;
  std::vector<float> initial_quant;
  std::vector<float> strategy_mask;
  std::vector<float> pixel_mask;
  std::vector<float> final_quant;
  std::vector<float> block_distance;
  ImageStorage reconstructed;
  gjxl::VarDctEncoderFrame frame;
  std::vector<double> scores;
  gjxl::MaximumErrorResult maximum_error_result;

  [[nodiscard]] gjxl::CpuQuantizationPipelineOutput Output() {
    return {
        .initial_quantization =
            {
                .quant_field = {initial_quant.data(), block_extent,
                                block_extent.width},
                .strategy_mask = {strategy_mask.data(), block_extent,
                                  block_extent.width},
                .pixel_mask = {pixel_mask.data(), padded_extent,
                               padded_extent.width},
            },
        .adaptive_quantization =
            {
                .quant_field = {final_quant.data(), block_extent,
                                block_extent.width},
                .block_distance_map = {block_distance.data(), block_extent,
                                       block_extent.width},
                .reconstructed_linear_rgb = reconstructed.View(),
                .frame = &frame,
                .score_history = &scores,
                .maximum_error_result = &maximum_error_result,
            },
    };
  }
};

double MaximumImageError(const ImageStorage& left,
                         const ImageStorage& right) {
  double maximum = 0.0;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < left.extent.height; ++y) {
      for (size_t x = 0; x < left.extent.width; ++x) {
        maximum = std::max(
          maximum,
          std::abs(static_cast<double>(
            left.plane[channel][y * left.stride + x]) -
            right.plane[channel][y * right.stride + x]));
      }
    }
  }
  return maximum;
}

bool CheckFixedRawQuantizationReuse(gjxl::GpuBackend& gpu) {
  using namespace gjxl;
  using namespace gjxl::quantization_pipeline_internal;
  ImageStorage original(kOriginalExtent), padded(kPaddedExtent), opsin(kPaddedExtent);
  FillImages(&original, &padded);
  if (!LinearRgbToOpsin(padded.ConstView(), 255.0f, opsin.View()).ok()) return false;
  CpuQuantizationPipelineOptions options;
  options.butteraugli_target = 1.9f;
  options.fixed_dct8 = true;
  options.uniform_initial_quantization = true;
  options.adaptive_quantization.iterations = 0;
  options.adaptive_quantization.profile.loop_filter.gaborish = false;
  options.adaptive_quantization.profile.adaptive_dc_smoothing = true;
  for (auto mode : {GpuAdaptiveQuantizationMode::kExactCoefficients,
                    GpuAdaptiveQuantizationMode::kFullyResident,
                    GpuAdaptiveQuantizationMode::kThroughput}) {
    PreparedQuantizationPipeline host;
    adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization cached;
    auto status = PrepareQuantizationPipeline(
      original.ConstView(), opsin.ConstView(), options, &host, false);
    if (!status.ok()) return false;
    std::vector<uint8_t> adjusted_bytes;
    for (auto decision : {AcCoefficientDecisionMode::kAdjustedSharedQuant,
                          AcCoefficientDecisionMode::kFixedRawQuant,
                          AcCoefficientDecisionMode::kFixedRawQuant,
                          AcCoefficientDecisionMode::kAdjustedSharedQuant}) {
      options.adaptive_quantization.coefficient_decision_mode = decision;
      PipelineStorage fresh(kOriginalExtent, kPaddedExtent);
      PipelineStorage reused(kOriginalExtent, kPaddedExtent);
      status = RunGpuQuantizationPipeline(gpu, original.ConstView(),
        opsin.ConstView(), options, mode, fresh.Output());
      if (status.ok()) status = RunPreparedGpuQuantizationPipeline(
        gpu, original.ConstView(), host, options, mode,
        reused.Output(), nullptr, &cached);
      std::vector<uint8_t> bytes;
      if (status.ok()) status = EncodeVarDctCodestream(reused.frame, &bytes);
      std::vector<uint8_t> fresh_bytes;
      if (status.ok()) status = EncodeVarDctCodestream(fresh.frame, &fresh_bytes);
      if (!status.ok() || fresh_bytes != bytes ||
          fresh.scores != reused.scores ||
          MaximumImageError(fresh.reconstructed, reused.reconstructed) != 0.0) {
        std::cerr << "AC decision mode reused stale GPU state: "
                  << status.message() << " mode=" << int(mode)
                  << " decision=" << int(decision) << '\n';
        return false;
      }
      if (decision == AcCoefficientDecisionMode::kFixedRawQuant) {
        const auto raw = reused.frame.raw_quant_field();
        for (size_t y = 0; y < raw.extent.height; ++y)
          for (size_t x = 0; x < raw.extent.width; ++x)
            // The low-level throughput API requests one AQ update; its input
            // field is spatial even when initialization was uniform.
            if (mode != GpuAdaptiveQuantizationMode::kThroughput &&
                raw.Row(y)[x] != raw.Row(0)[0]) {
              std::cerr << "Fixed raw coding changed the uniform quantizer\n";
              return false;
            }
        if (bytes == adjusted_bytes) {
          std::cerr << "AC bypass did not change the discriminating fixture\n";
          return false;
        }
      } else if (adjusted_bytes.empty()) {
        adjusted_bytes = bytes;
      } else if (bytes != adjusted_bytes) {
        std::cerr << "Restoring AC adjustment retained fixed-raw output\n";
        return false;
      }
      if (mode == GpuAdaptiveQuantizationMode::kExactCoefficients) {
        PipelineStorage cpu(kOriginalExtent, kPaddedExtent);
        status = RunCpuQuantizationPipeline(original.ConstView(),
          opsin.ConstView(), options, cpu.Output());
        std::vector<uint8_t> cpu_bytes;
        if (status.ok()) status = EncodeVarDctCodestream(cpu.frame, &cpu_bytes);
        if (!status.ok() || cpu_bytes != bytes) {
          std::cerr << "CPU/exact GPU AC policy mismatch\n";
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace
int main(int argc, char** argv) {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  const auto status = gjxl::CreateCudaBackend(&gpu);
  if (status.code() == gjxl::StatusCode::kUnavailable) return 77;
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return EXIT_FAILURE;
  }
  if (!CheckFixedRawQuantizationReuse(*gpu)) return EXIT_FAILURE;
  std::cout << "CUDA fixed-raw reuse, reconstruction and CPU/exact parity passed.\n";
  if (argc == 2) {
    std::ofstream report(argv[1]);
    report << "PASS CUDA fixed-raw quantization policy\n";
    if (!report) return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
