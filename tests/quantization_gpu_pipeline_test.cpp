// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates GPU AC search feeding the owned VarDCT encoder frame.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "codec/color_transform.h"
#include "codec/quantization_pipeline.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/quantization_pipeline.h"

#ifndef GJXL_METALLIB_PATH
#error "GJXL_METALLIB_PATH must point to the test metallib"
#endif

namespace {

constexpr gjxl::Extent2D kOriginalExtent{257, 17};
constexpr gjxl::Extent2D kPaddedExtent{264, 24};
constexpr gjxl::Extent2D kBlockExtent{33, 3};
constexpr size_t kPaddedPixelCount = kPaddedExtent.width * kPaddedExtent.height;
constexpr size_t kBlockCount = kBlockExtent.width * kBlockExtent.height;

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
  std::vector<float> initial_quant = std::vector<float>(kBlockCount);
  std::vector<float> strategy_mask = std::vector<float>(kBlockCount);
  std::vector<float> pixel_mask = std::vector<float>(kPaddedPixelCount);
  std::vector<float> final_quant = std::vector<float>(kBlockCount);
  std::vector<float> block_distance = std::vector<float>(kBlockCount);
  ImageStorage reconstructed{kOriginalExtent};
  gjxl::VarDctEncoderFrame frame;
  std::vector<double> scores;

  [[nodiscard]] gjxl::CpuQuantizationPipelineOutput Output() {
    return {
        .initial_quantization =
            {
                .quant_field = {initial_quant.data(), kBlockExtent,
                                kBlockExtent.width},
                .strategy_mask = {strategy_mask.data(), kBlockExtent,
                                  kBlockExtent.width},
                .pixel_mask = {pixel_mask.data(), kPaddedExtent,
                               kPaddedExtent.width},
            },
        .adaptive_quantization =
            {
                .quant_field = {final_quant.data(), kBlockExtent,
                                kBlockExtent.width},
                .block_distance_map = {block_distance.data(), kBlockExtent,
                                       kBlockExtent.width},
                .reconstructed_linear_rgb = reconstructed.View(),
                .frame = &frame,
                .score_history = &scores,
            },
    };
  }
};

bool GridsEqual(const gjxl::AcStrategyGrid &left,
                const gjxl::AcStrategyGrid &right) {

  if (left.extent() != right.extent()) {
    return false;
  }
  for (size_t y = 0; y < left.extent().height; ++y) {
    for (size_t x = 0; x < left.extent().width; ++x) {
      gjxl::AcStrategyCell left_cell;
      gjxl::AcStrategyCell right_cell;
      if (!left.Get(x, y, &left_cell).ok() ||
          !right.Get(x, y, &right_cell).ok() ||
          left_cell.strategy != right_cell.strategy ||
          left_cell.is_anchor != right_cell.is_anchor) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
bool PlanesEqual(gjxl::PlaneView<const T> left,
                 gjxl::PlaneView<const T> right) {
  if (left.extent != right.extent) {
    return false;
  }
  for (size_t y = 0; y < left.extent.height; ++y) {
    if (!std::equal(left.Row(y), left.Row(y) + left.extent.width,
                    right.Row(y))) {
      return false;
    }
  }
  return true;
}

bool FramesEqual(const gjxl::VarDctEncoderFrame &left,
                 const gjxl::VarDctEncoderFrame &right) {

  if (!left.valid() || !right.valid() ||
      left.geometry().frame() != right.geometry().frame() ||
      left.geometry().padded_frame() != right.geometry().padded_frame() ||
      !GridsEqual(left.strategies(), right.strategies()) ||
      !PlanesEqual(left.raw_quant_field(), right.raw_quant_field()) ||
      !PlanesEqual(left.epf_sharpness(), right.epf_sharpness()) ||
      left.quantizer().params().global_scale !=
          right.quantizer().params().global_scale ||
      left.quantizer().params().quant_dc !=
          right.quantizer().params().quant_dc ||
      left.coding_options().x_matrix_multiplier !=
          right.coding_options().x_matrix_multiplier ||
      left.coding_options().b_matrix_multiplier !=
          right.coding_options().b_matrix_multiplier ||
      left.ac_group_extent() != right.ac_group_extent() ||
      left.ac_group_count() != right.ac_group_count()) {
    return false;
  }

  const gjxl::ColorCorrelationMap &left_cfl = left.color_correlation();
  const gjxl::ColorCorrelationMap &right_cfl = right.color_correlation();
  if (left_cfl.tile_extent() != right_cfl.tile_extent() ||
      !PlanesEqual(left_cfl.y_to_x_map(), right_cfl.y_to_x_map()) ||
      !PlanesEqual(left_cfl.y_to_b_map(), right_cfl.y_to_b_map())) {
    return false;
  }
  const gjxl::ConstImage3FView left_dc = left.dc();
  const gjxl::ConstImage3FView right_dc = right.dc();
  const gjxl::ConstImage3I32View left_quantized_dc = left.quantized_dc();
  const gjxl::ConstImage3I32View right_quantized_dc = right.quantized_dc();
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!PlanesEqual(
          left_quantized_dc.plane[channel],
          right_quantized_dc.plane[channel]) ||
        !PlanesEqual(left_dc.plane[channel], right_dc.plane[channel])) {
      return false;
    }
  }
  for (size_t group_index = 0; group_index < left.ac_group_count();
       ++group_index) {
    gjxl::VarDctAcGroupView left_group;
    gjxl::VarDctAcGroupView right_group;
    if (!left.GetAcGroup(group_index, &left_group).ok() ||
        !right.GetAcGroup(group_index, &right_group).ok() ||
        left_group.block_x != right_group.block_x ||
        left_group.block_y != right_group.block_y ||
        left_group.block_extent != right_group.block_extent ||
        left_group.used_coefficient_count !=
            right_group.used_coefficient_count) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      if (!std::equal(left_group.coefficients[channel].begin(),
                      left_group.coefficients[channel].end(),
                      right_group.coefficients[channel].begin())) {
        return false;
      }
    }
  }
  return true;
}

gjxl::MetalBackendOptions FactoredOptions() {
  constexpr auto implementation = gjxl::MetalDctImplementation::kFactoredRadix2;
  return {
      .forward_dct8 = implementation,
      .inverse_dct8 = implementation,
      .forward_dct16x16 = implementation,
      .inverse_dct16x16 = implementation,
      .forward_dct32x32 = implementation,
      .inverse_dct32x32 = implementation,
      .forward_dct16x8 = implementation,
      .inverse_dct16x8 = implementation,
      .forward_dct8x16 = implementation,
      .inverse_dct8x16 = implementation,
      .forward_dct32x16 = implementation,
      .inverse_dct32x16 = implementation,
      .forward_dct16x32 = implementation,
      .inverse_dct16x32 = implementation,
  };
}

bool CheckGpuPipelineParity() {
  ImageStorage original(kOriginalExtent);
  ImageStorage padded_linear(kPaddedExtent);
  ImageStorage opsin(kPaddedExtent);
  FillImages(&original, &padded_linear);
  if (!gjxl::LinearRgbToOpsin(padded_linear.ConstView(), 255.0f, opsin.View())
           .ok()) {
    return false;
  }

  gjxl::CpuQuantizationPipelineOptions options;
  options.butteraugli_target = 1.2f;
  options.adaptive_quantization.iterations = 0;
  PipelineStorage cpu;
  const gjxl::Status cpu_status = gjxl::RunCpuQuantizationPipeline(
      original.ConstView(), opsin.ConstView(), options, cpu.Output());

  std::unique_ptr<gjxl::GpuBackend> gpu;
  const gjxl::Status create_status =
      gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, FactoredOptions(), &gpu);
  if (!cpu_status.ok() || !create_status.ok()) {
    std::cerr << "Unable to initialize pipeline parity test: CPU="
              << cpu_status.message() << ", GPU=" << create_status.message()
              << '\n';
    return false;
  }

  PipelineStorage accelerated;
  gjxl::AcStrategyGpuSearchStats stats;
  const gjxl::Status gpu_status = gjxl::RunGpuQuantizationPipeline(
      *gpu, original.ConstView(), opsin.ConstView(), options,
      accelerated.Output(), &stats);
  if (!gpu_status.ok() || stats.total_candidate_count == 0 ||
      cpu.initial_quant != accelerated.initial_quant ||
      cpu.strategy_mask != accelerated.strategy_mask ||
      cpu.pixel_mask != accelerated.pixel_mask ||
      cpu.final_quant != accelerated.final_quant ||
      cpu.block_distance != accelerated.block_distance ||
      cpu.reconstructed.plane != accelerated.reconstructed.plane ||
      cpu.scores != accelerated.scores ||
      !FramesEqual(cpu.frame, accelerated.frame)) {
    std::cerr << "GPU-search pipeline differs from CPU: "
              << gpu_status.message() << '\n';
    return false;
  }

  gjxl::VarDctAcGroupView edge_group;
  if (accelerated.frame.ac_group_extent() != gjxl::Extent2D{2, 1} ||
      !accelerated.frame.GetAcGroup(1, &edge_group).ok() ||
      edge_group.block_x != 32 || edge_group.block_y != 0 ||
      edge_group.block_extent != gjxl::Extent2D{1, 3} ||
      edge_group.used_coefficient_count != 192) {
    std::cerr << "GPU-search frame has an invalid edge AC group\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  if (!CheckGpuPipelineParity()) {
    return EXIT_FAILURE;
  }
  std::cout << "GPU-search quantization pipeline matches CPU.\n";
  return EXIT_SUCCESS;
}
