// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates GPU AC search feeding the owned VarDCT encoder frame.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "codec/color_transform.h"
#include "codec/chroma_from_luma_internal.h"
#include "codec/epf.h"
#include "codec/gaborish.h"
#include "codec/quantization_pipeline.h"
#include "codec/quantization_pipeline_internal.h"
#include "codec/vardct_frame_view_internal.h"
#include "codestream/encoder.h"
#include "codestream/encoder_internal.h"
#include "codestream/workflow.h"
#include "codestream/workflow_internal.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/gaborish.h"
#include "gpu/ops/quantization_pipeline.h"

#ifndef GJXL_METALLIB_PATH
#error "GJXL_METALLIB_PATH must point to the test metallib"
#endif

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

bool ImagesShareStorage(gjxl::ConstImage3FView left,
                        gjxl::ConstImage3FView right) {
  for (size_t channel = 0; channel < 3; ++channel) {
    if (left.plane[channel].data != right.plane[channel].data ||
        left.plane[channel].extent != right.plane[channel].extent ||
        left.plane[channel].stride != right.plane[channel].stride) {
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
      left.profile() != right.profile() ||
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

double MaximumError(const std::vector<float>& left,
                    const std::vector<float>& right) {
  double maximum = 0.0;
  for (size_t index = 0; index < left.size(); ++index) {
    maximum = std::max(
      maximum,
      std::abs(static_cast<double>(left[index]) - right[index]));
  }
  return maximum;
}

double MaximumScoreError(const std::vector<double>& left,
                         const std::vector<double>& right) {
  if (left.size() != right.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double maximum = 0.0;
  for (size_t index = 0; index < left.size(); ++index) {
    maximum = std::max(maximum, std::abs(left[index] - right[index]));
  }
  return maximum;
}

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

bool CheckGpuGaborish() {
  constexpr gjxl::Extent2D kExtent{17, 11};
  constexpr std::array<float, 3> kMultipliers{1.0f, 0.8f, 1.1f};
  ImageStorage input(kExtent);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        input.plane[channel][y * input.stride + x] =
          0.01f * static_cast<float>(3 * x + 5 * y + 7 * channel) +
          0.2f * std::sin(static_cast<float>(x + channel * y));
      }
    }
  }
  ImageStorage expected(kExtent);
  if (!gjxl::ApplyGaborishInverse(
        input.ConstView(), kMultipliers, expected.View()).ok()) {
    return false;
  }

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu).ok()) {
    return false;
  }
  ImageStorage output(kExtent);
  const gjxl::Status status = gjxl::ApplyGaborishInverseGpu(
    *gpu, input.ConstView(), kMultipliers, output.View());
  ImageStorage aliased = input;
  const gjxl::Status alias_status = gjxl::ApplyGaborishInverseGpu(
    *gpu, aliased.ConstView(), kMultipliers, aliased.View());
  const double output_error = MaximumImageError(expected, output);
  const double alias_error = MaximumImageError(expected, aliased);
  if (!status.ok() || !alias_status.ok() || output_error > 2.0e-6 ||
      alias_error > 2.0e-6) {
    std::cerr << "GPU Gaborish mismatch: " << status.message()
              << " output=" << output_error << " alias=" << alias_error
              << '\n';
    return false;
  }

  ImageStorage nonfinite = input;
  nonfinite.plane[1][2 * nonfinite.stride + 3] =
    std::numeric_limits<float>::quiet_NaN();
  ImageStorage atomic_output(kExtent, 91.0f);
  const auto original_output = atomic_output.plane;
  const gjxl::GpuBackendStats before = gpu->stats();
  const gjxl::Status invalid = gjxl::ApplyGaborishInverseGpu(
    *gpu, nonfinite.ConstView(), kMultipliers, atomic_output.View());
  const gjxl::GpuBackendStats after = gpu->stats();
  if (invalid.code() != gjxl::StatusCode::kInvalidArgument ||
      atomic_output.plane != original_output ||
      before.successful_allocations != after.successful_allocations ||
      before.committed_submissions != after.committed_submissions) {
    std::cerr << "GPU Gaborish invalid input was not atomic\n";
    return false;
  }
  std::cout << "GPU Gaborish max error=" << output_error << '\n';
  return true;
}

class BackendWithoutAq final : public gjxl::GpuBackend {
public:
  gjxl::BackendKind kind() const noexcept override {
    return gjxl::BackendKind::kMetal;
  }
  std::string_view name() const noexcept override { return "no AQ"; }
  gjxl::Status Allocate(
      size_t, std::unique_ptr<gjxl::DeviceBuffer>* out) override {
    if (out != nullptr) out->reset();
    return gjxl::Status::Unavailable("not implemented");
  }
  gjxl::Status CopyHostToDevice(
      gjxl::DeviceBuffer&, const void*, size_t, size_t) override {
    return gjxl::Status::Unavailable("not implemented");
  }
  gjxl::Status CopyDeviceToHost(
      const gjxl::DeviceBuffer&, void*, size_t, size_t) override {
    return gjxl::Status::Unavailable("not implemented");
  }
  gjxl::Status ForwardTransform(
      const gjxl::TransformBatch&,
      std::unique_ptr<gjxl::GpuSubmission>* submission) override {
    if (submission != nullptr) submission->reset();
    return gjxl::Status::Unavailable("not implemented");
  }
  gjxl::Status InverseTransform(
      const gjxl::TransformBatch&,
      std::unique_ptr<gjxl::GpuSubmission>* submission) override {
    if (submission != nullptr) submission->reset();
    return gjxl::Status::Unavailable("not implemented");
  }
};

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
  PipelineStorage cpu(kOriginalExtent, kPaddedExtent);
  const gjxl::Status cpu_status = gjxl::RunCpuQuantizationPipeline(
      original.ConstView(), opsin.ConstView(), options, cpu.Output());

  std::unique_ptr<gjxl::GpuBackend> gpu;
  const gjxl::Status create_status =
      gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu);
  if (!cpu_status.ok() || !create_status.ok()) {
    std::cerr << "Unable to initialize pipeline parity test: CPU="
              << cpu_status.message() << ", GPU=" << create_status.message()
              << '\n';
    return false;
  }

  PipelineStorage accelerated(kOriginalExtent, kPaddedExtent);
  gjxl::AcStrategyGpuSearchStats stats;
  const gjxl::Status gpu_status = gjxl::RunGpuQuantizationPipeline(
      *gpu, original.ConstView(), opsin.ConstView(), options,
      accelerated.Output(), &stats);
  constexpr double kTolerance = 2.0e-3;
  constexpr double kNarrowAccumulatedTolerance = 2.5e-2;
  const double quant_error = MaximumError(
    cpu.final_quant, accelerated.final_quant);
  const double block_error = MaximumError(
    cpu.block_distance, accelerated.block_distance);
  const double score_error = MaximumScoreError(cpu.scores, accelerated.scores);
  const double image_error = MaximumImageError(
    cpu.reconstructed, accelerated.reconstructed);
  const bool frames_equal = FramesEqual(cpu.frame, accelerated.frame);
  if (!gpu_status.ok() || stats.total_candidate_count == 0 ||
      cpu.initial_quant != accelerated.initial_quant ||
      cpu.strategy_mask != accelerated.strategy_mask ||
      cpu.pixel_mask != accelerated.pixel_mask ||
      quant_error > kTolerance ||
      block_error > kNarrowAccumulatedTolerance ||
      score_error > kTolerance ||
      image_error > kNarrowAccumulatedTolerance ||
      !frames_equal) {
    std::cerr << "GPU-search pipeline differs from CPU: "
              << gpu_status.message() << ", quant=" << quant_error
              << ", block=" << block_error << ", score=" << score_error
              << ", image=" << image_error
              << ", frame=" << (frames_equal ? "exact" : "different")
              << '\n';
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

  std::vector<uint8_t> cpu_codestream;
  std::vector<uint8_t> first_gpu_codestream;
  std::vector<uint8_t> second_gpu_codestream;
  if (!gjxl::EncodeVarDctCodestream(cpu.frame, &cpu_codestream).ok() ||
      !gjxl::EncodeVarDctCodestream(
        accelerated.frame, &first_gpu_codestream).ok() ||
      !gjxl::EncodeVarDctCodestream(
        accelerated.frame, &second_gpu_codestream).ok() ||
      cpu_codestream.empty() || cpu_codestream != first_gpu_codestream ||
      first_gpu_codestream != second_gpu_codestream) {
    std::cerr << "CPU/GPU-search codestream bytes are not deterministic\n";
    return false;
  }
  std::cout << "Complete GPU pipeline errors: quant=" << quant_error
            << " block=" << block_error << " score=" << score_error
            << " image=" << image_error
            << "; frame and codestream exact\n";
  return true;
}

bool CheckMaximumThroughputFrontendParity() {
  ImageStorage original(kOriginalExtent);
  ImageStorage padded_linear(kPaddedExtent);
  ImageStorage opsin(kPaddedExtent);
  FillImages(&original, &padded_linear);
  if (!gjxl::LinearRgbToOpsin(
        padded_linear.ConstView(), 255.0f, opsin.View()).ok()) {
    return false;
  }
  const gjxl::Extent2D blocks{
      kPaddedExtent.width / gjxl::kJxlBlockDimension,
      kPaddedExtent.height / gjxl::kJxlBlockDimension};
  const size_t block_count = blocks.width * blocks.height;
  const size_t pixel_count = kPaddedExtent.width * kPaddedExtent.height;
  gjxl::CpuQuantizationPipelineOptions pipeline_options;
  pipeline_options.butteraugli_target = 1.2f;
  const float initial_target =
      pipeline_options.adaptive_quantization.profile.loop_filter.gaborish
          ? pipeline_options.butteraugli_target
          : 0.62f * pipeline_options.butteraugli_target;
  const gjxl::InitialQuantizationOptions initial_options{
      .butteraugli_target = initial_target,
      .rescale = pipeline_options.initial_quant_rescale,
  };
  std::vector<float> expected_quant(block_count);
  std::vector<float> expected_strategy(block_count);
  std::vector<float> expected_pixel(pixel_count);
  if (!gjxl::ComputeInitialQuantField(
           opsin.ConstView(), initial_options,
           {
             .quant_field = {
               expected_quant.data(), blocks, blocks.width},
             .strategy_mask = {
               expected_strategy.data(), blocks, blocks.width},
             .pixel_mask = {
               expected_pixel.data(), kPaddedExtent, kPaddedExtent.width},
           }).ok()) {
    return false;
  }
  gjxl::ColorCorrelationMap color;
  gjxl::AcStrategyGrid strategies;
  std::vector<uint8_t> sharpness(block_count);
  if (!gjxl::chroma_from_luma_internal::ComputeInitialColorCorrelationMapFast(
        opsin.ConstView(), &color).ok() ||
      !gjxl::AcStrategyGrid::Create(blocks, &strategies).ok() ||
      !gjxl::FillDefaultEpfSharpness(
        {sharpness.data(), blocks, blocks.width}).ok()) {
    return false;
  }
  strategies.fill_dct8();
  gjxl::AdaptiveQuantizationOptions adaptive_options =
      pipeline_options.adaptive_quantization;
  adaptive_options.butteraugli_target =
      pipeline_options.butteraugli_target;

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu).ok()) {
    return false;
  }
  std::vector<float> expected_final(block_count);
  gjxl::VarDctEncoderFrame expected_frame;
  if (!gjxl::RunGpuFrameOnlyQuantization(
        *gpu, original.ConstView(), opsin.ConstView(), strategies,
        {expected_quant.data(), blocks, blocks.width},
        {sharpness.data(), blocks, blocks.width}, color, adaptive_options,
        {
          .quant_field = {expected_final.data(), blocks, blocks.width},
          .frame = &expected_frame,
        }).ok()) {
    return false;
  }

  std::vector<float> actual_quant(block_count);
  std::vector<float> actual_strategy(block_count);
  std::vector<float> actual_pixel(pixel_count);
  std::vector<float> actual_final(block_count);
  gjxl::VarDctEncoderFrame actual_frame;
  if (!gjxl::RunGpuFrameOnlyQuantizationResidentFrontend(
        *gpu, original.ConstView(), opsin.ConstView(), strategies,
        {sharpness.data(), blocks, blocks.width}, initial_options,
        adaptive_options,
        {
          .quant_field = {actual_quant.data(), blocks, blocks.width},
          .strategy_mask = {actual_strategy.data(), blocks, blocks.width},
          .pixel_mask = {
            actual_pixel.data(), kPaddedExtent, kPaddedExtent.width},
        },
        {
          .quant_field = {actual_final.data(), blocks, blocks.width},
          .frame = &actual_frame,
        }).ok()) {
    return false;
  }
  std::vector<uint8_t> expected_bytes;
  std::vector<uint8_t> actual_bytes;
  const double quant_error = MaximumError(expected_quant, actual_quant);
  const double strategy_error =
      MaximumError(expected_strategy, actual_strategy);
  const double pixel_error = MaximumError(expected_pixel, actual_pixel);
  const double final_error = MaximumError(expected_final, actual_final);
  if (quant_error > 2.0e-6 || strategy_error > 2.0e-6 ||
      pixel_error > 2.0e-5 || final_error > 2.0e-6 ||
      !FramesEqual(expected_frame, actual_frame) ||
      !gjxl::EncodeVarDctCodestream(expected_frame, &expected_bytes).ok() ||
      !gjxl::EncodeVarDctCodestream(actual_frame, &actual_bytes).ok() ||
      expected_bytes != actual_bytes) {
    std::cerr << "Resident maximum-throughput frontend differs: quant="
              << quant_error << " strategy=" << strategy_error
              << " pixel=" << pixel_error << " final=" << final_error
              << '\n';
    return false;
  }
  return true;
}

bool CheckDefaultUpdatePipelineParity() {
  constexpr gjxl::Extent2D kExtent{96, 64};
  ImageStorage original(kExtent);
  ImageStorage padded_linear(kExtent);
  ImageStorage opsin(kExtent);
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      const std::array<float, 3> rgb = {
        0.12f + 0.006f * fx + 0.004f * fy,
        0.18f + 0.003f * fx + 0.006f * fy,
        0.08f + 0.005f * fx + 0.004f * fy,
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        original.plane[channel][y * original.stride + x] = rgb[channel];
        padded_linear.plane[channel][y * padded_linear.stride + x] =
          rgb[channel];
      }
    }
  }
  if (!gjxl::LinearRgbToOpsin(
        padded_linear.ConstView(), 255.0f, opsin.View()).ok()) {
    return false;
  }

  gjxl::CpuQuantizationPipelineOptions options;
  options.butteraugli_target = 1.0f;
  PipelineStorage cpu(kExtent, kExtent);
  if (!gjxl::RunCpuQuantizationPipeline(
        original.ConstView(), opsin.ConstView(), options, cpu.Output()).ok()) {
    return false;
  }

  BackendWithoutAq unavailable;
  PipelineStorage unavailable_output(kExtent, kExtent);
  const std::vector<float> unavailable_initial =
    unavailable_output.initial_quant;
  gjxl::AcStrategyGpuSearchStats unavailable_stats;
  unavailable_stats.total_candidate_count = 987654;
  const gjxl::Status unavailable_status = gjxl::RunGpuQuantizationPipeline(
    unavailable, original.ConstView(), opsin.ConstView(), options,
    unavailable_output.Output(), &unavailable_stats);
  if (unavailable_status.code() != gjxl::StatusCode::kUnavailable ||
      unavailable_output.initial_quant != unavailable_initial ||
      unavailable_output.frame.valid() ||
      !unavailable_output.scores.empty() ||
      unavailable_stats.total_candidate_count != 987654) {
    std::cerr << "Missing prepared AQ capability changed pipeline output\n";
    return false;
  }

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu).ok()) {
    return false;
  }
  PipelineStorage accelerated(kExtent, kExtent);
  gjxl::AcStrategyGpuSearchStats stats;
  const gjxl::GpuBackendStats before = gpu->stats();
  const gjxl::Status status = gjxl::RunGpuQuantizationPipeline(
    *gpu, original.ConstView(), opsin.ConstView(), options,
    accelerated.Output(), &stats);
  const gjxl::GpuBackendStats after = gpu->stats();
  constexpr double kTolerance = 2.0e-3;
  const double quant_error = MaximumError(
    cpu.final_quant, accelerated.final_quant);
  const double block_error = MaximumError(
    cpu.block_distance, accelerated.block_distance);
  const double score_error = MaximumScoreError(cpu.scores, accelerated.scores);
  const double image_error = MaximumImageError(
    cpu.reconstructed, accelerated.reconstructed);
  if (!status.ok() || stats.total_candidate_count == 0 ||
      after.committed_submissions !=
        before.committed_submissions + options.adaptive_quantization.iterations +
          3 ||
      cpu.initial_quant != accelerated.initial_quant ||
      cpu.strategy_mask != accelerated.strategy_mask ||
      cpu.pixel_mask != accelerated.pixel_mask ||
      quant_error > kTolerance || block_error > kTolerance ||
      score_error > kTolerance || image_error > kTolerance ||
      !FramesEqual(cpu.frame, accelerated.frame)) {
    std::cerr << "Default-update GPU pipeline differs: " << status.message()
              << ", quant=" << quant_error << ", block=" << block_error
              << ", score=" << score_error << ", image=" << image_error
              << '\n';
    return false;
  }

  std::vector<uint8_t> cpu_codestream;
  std::vector<uint8_t> gpu_codestream;
  if (!gjxl::EncodeVarDctCodestream(cpu.frame, &cpu_codestream).ok() ||
      !gjxl::EncodeVarDctCodestream(
        accelerated.frame, &gpu_codestream).ok() ||
      cpu_codestream != gpu_codestream) {
    std::cerr << "Default-update GPU pipeline codestream differs\n";
    return false;
  }

  PipelineStorage resident(kExtent, kExtent);
  gjxl::AcStrategyGpuSearchStats resident_stats;
  const gjxl::GpuBackendStats before_resident = gpu->stats();
  const gjxl::Status resident_status = gjxl::RunGpuQuantizationPipeline(
      *gpu, original.ConstView(), opsin.ConstView(), options,
      gjxl::GpuAdaptiveQuantizationMode::kFullyResident, resident.Output(),
      &resident_stats);
  const gjxl::GpuBackendStats after_resident = gpu->stats();
  std::vector<uint8_t> resident_codestream;
  const double resident_initial_error = MaximumError(
      cpu.initial_quant, resident.initial_quant);
  const double resident_strategy_mask_error = MaximumError(
      cpu.strategy_mask, resident.strategy_mask);
  const double resident_pixel_mask_error = MaximumError(
      cpu.pixel_mask, resident.pixel_mask);
  if (!resident_status.ok() || resident_stats.total_candidate_count == 0 ||
      after_resident.committed_submissions !=
          before_resident.committed_submissions + 5 ||
      resident_initial_error > 2.0e-6 ||
      resident_strategy_mask_error > 2.0e-6 ||
      resident_pixel_mask_error > 3.0e-5 || !resident.frame.valid() ||
      resident.scores.size() != options.adaptive_quantization.iterations + 1 ||
      !gjxl::EncodeVarDctCodestream(
           resident.frame, &resident_codestream).ok() ||
      resident_codestream.empty()) {
    std::cerr << "Fully resident complete GPU pipeline failed: "
              << resident_status.message()
              << " initial=" << resident_initial_error
              << " strategy_mask=" << resident_strategy_mask_error
              << " pixel_mask=" << resident_pixel_mask_error << '\n';
    return false;
  }

  gjxl::quantization_pipeline_internal::PreparedQuantizationPipeline
    encoding_prepared;
  gjxl::adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    encoding_aq;
  gjxl::Status encoding_status =
    gjxl::quantization_pipeline_internal::PrepareQuantizationPipeline(
      original.ConstView(), opsin.ConstView(), options, &encoding_prepared,
      false, false,
      gjxl::quantization_pipeline_internal::
        QuantizationPipelineInputProvenance::kFiniteLinearRgbAndOpsin);
  const bool resident_preprocessing_was_deferred =
    !encoding_prepared.preprocessed_opsin.const_view().valid();
  gjxl::VarDctEncoderFrame encoding_frame;
  std::vector<double> encoding_scores;
  if (encoding_status.ok()) {
    encoding_status = gjxl::quantization_pipeline_internal::
      RunPreparedGpuQuantizationPipelineForEncoding(
        *gpu, original.ConstView(), encoding_prepared, options,
        gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
        {
          .frame = &encoding_frame,
          .score_history = &encoding_scores,
        },
        nullptr, &encoding_aq);
  }
  std::vector<uint8_t> encoding_codestream;
  if (encoding_status.ok()) {
    encoding_status = gjxl::EncodeVarDctCodestream(
      encoding_frame, &encoding_codestream);
  }
  if (!encoding_status.ok() || !resident_preprocessing_was_deferred ||
      encoding_prepared.preprocessed_opsin.const_view().valid() ||
      !ImagesShareStorage(
        encoding_prepared.validated_original_linear_rgb,
        original.ConstView()) ||
      !ImagesShareStorage(
        encoding_prepared.validated_coding_opsin,
        opsin.ConstView()) ||
      encoding_scores != resident.scores ||
      !FramesEqual(encoding_frame, resident.frame) ||
      encoding_codestream != resident_codestream) {
    std::cerr << "Encoding-only resident pipeline materialized host "
                 "preprocessing or changed frame output: "
              << encoding_status.message() << '\n';
    return false;
  }

  std::unique_ptr<gjxl::vardct_frame_internal::CompletedVarDctFrame>
    encoding_lease;
  gjxl::VarDctEncoderFrame leased_fallback;
  std::vector<double> leased_scores;
  encoding_status = gjxl::quantization_pipeline_internal::
    RunPreparedGpuQuantizationPipelineForEncoding(
      *gpu, original.ConstView(), encoding_prepared, options,
      gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
      {
        .frame = &leased_fallback,
        .score_history = &leased_scores,
        .completed_frame = &encoding_lease,
      }, nullptr, &encoding_aq);
  std::vector<uint8_t> leased_bytes;
  if (encoding_status.ok() && encoding_lease != nullptr) {
    // The public-workflow handoff must also survive releasing its reusable
    // preparation, not only direct prepared-evaluator unit tests.
    encoding_aq.evaluation.reset();
    encoding_status = gjxl::codestream_internal::EncodeVarDctCodestreamFromView(
      encoding_lease->view(), {}, &leased_bytes);
  }
  if (!encoding_status.ok() || encoding_lease == nullptr ||
      leased_fallback.valid() || leased_scores != resident.scores ||
      leased_bytes != resident_codestream) {
    std::cerr << "Encoding-only completed frame handoff failed: "
              << encoding_status.message() << '\n';
    return false;
  }

  ImageStorage mismatched_original = original;
  mismatched_original.plane[0][7] =
    std::numeric_limits<float>::quiet_NaN();
  gjxl::adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    mismatched_aq;
  gjxl::VarDctEncoderFrame mismatched_frame;
  std::vector<double> mismatched_scores;
  const gjxl::Status mismatched_status =
    gjxl::quantization_pipeline_internal::
      RunPreparedGpuQuantizationPipelineForEncoding(
        *gpu, mismatched_original.ConstView(), encoding_prepared, options,
        gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
        {
          .frame = &mismatched_frame,
          .score_history = &mismatched_scores,
        },
        nullptr, &mismatched_aq);
  if (mismatched_status.code() != gjxl::StatusCode::kInvalidArgument ||
      mismatched_aq.evaluation != nullptr || mismatched_frame.valid() ||
      !mismatched_scores.empty()) {
    std::cerr << "Validated preparation provenance escaped its bound source "
                 "identity: "
              << mismatched_status.message() << '\n';
    return false;
  }

  gjxl::quantization_pipeline_internal::PreparedQuantizationPipeline
    exact_encoding_prepared;
  gjxl::adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    exact_encoding_aq;
  gjxl::Status exact_encoding_status =
    gjxl::quantization_pipeline_internal::PrepareQuantizationPipeline(
      original.ConstView(), opsin.ConstView(), options,
      &exact_encoding_prepared, false, false);
  const bool exact_preprocessing_was_deferred =
    !exact_encoding_prepared.preprocessed_opsin.const_view().valid();
  gjxl::VarDctEncoderFrame exact_encoding_frame;
  std::vector<double> exact_encoding_scores;
  if (exact_encoding_status.ok()) {
    exact_encoding_status = gjxl::quantization_pipeline_internal::
      RunPreparedGpuQuantizationPipelineForEncoding(
        *gpu, original.ConstView(), exact_encoding_prepared, options,
        gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
        {
          .frame = &exact_encoding_frame,
          .score_history = &exact_encoding_scores,
        },
        nullptr, &exact_encoding_aq);
  }
  std::vector<uint8_t> exact_encoding_codestream;
  if (exact_encoding_status.ok()) {
    exact_encoding_status = gjxl::EncodeVarDctCodestream(
      exact_encoding_frame, &exact_encoding_codestream);
  }
  gjxl::metal_internal::MetalAqReadbackStatsForTesting
    exact_encoding_readback;
  if (exact_encoding_status.ok() && exact_encoding_aq.evaluation != nullptr) {
    exact_encoding_status =
      gjxl::metal_internal::GetMetalAqReadbackStatsForTesting(
        *exact_encoding_aq.evaluation, &exact_encoding_readback);
  }
  if (!exact_encoding_status.ok() ||
      !exact_preprocessing_was_deferred ||
      !exact_encoding_prepared.preprocessed_opsin.const_view().valid() ||
      exact_encoding_scores != accelerated.scores ||
      !FramesEqual(exact_encoding_frame, accelerated.frame) ||
      exact_encoding_codestream != gpu_codestream ||
      exact_encoding_readback.score_history_bytes != sizeof(float) ||
      exact_encoding_readback.block_distance_map_bytes == 0 ||
      exact_encoding_readback.frame_bytes != 0 ||
      exact_encoding_readback.mapped_frame_bytes != 0 ||
      exact_encoding_readback.reconstructed_rgb_bytes != 0) {
    std::cerr << "Encoding-only exact pipeline changed output or read "
                 "diagnostics: " << exact_encoding_status.message() << '\n';
    return false;
  }

  gjxl::CpuQuantizationPipelineOptions maximum_error_options = options;
  maximum_error_options.adaptive_quantization.control_mode =
    gjxl::AdaptiveQuantizationControlMode::kMaximumError;
  maximum_error_options.adaptive_quantization.maximum_error =
    {0.05f, 0.05f, 0.05f};
  PipelineStorage maximum_error_full(kExtent, kExtent);
  gjxl::Status maximum_error_status = gjxl::RunGpuQuantizationPipeline(
    *gpu, original.ConstView(), opsin.ConstView(), maximum_error_options,
    gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
    maximum_error_full.Output());
  std::vector<uint8_t> maximum_error_codestream;
  if (maximum_error_status.ok()) {
    maximum_error_status = gjxl::EncodeVarDctCodestream(
      maximum_error_full.frame, &maximum_error_codestream);
  }

  gjxl::quantization_pipeline_internal::PreparedQuantizationPipeline
    maximum_encoding_prepared;
  gjxl::adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    maximum_encoding_aq;
  if (maximum_error_status.ok()) {
    maximum_error_status =
      gjxl::quantization_pipeline_internal::PrepareQuantizationPipeline(
        original.ConstView(), opsin.ConstView(), maximum_error_options,
        &maximum_encoding_prepared, false, false);
  }
  gjxl::VarDctEncoderFrame maximum_encoding_frame;
  std::vector<double> maximum_encoding_scores;
  gjxl::MaximumErrorResult maximum_encoding_result;
  if (maximum_error_status.ok()) {
    maximum_error_status = gjxl::quantization_pipeline_internal::
      RunPreparedGpuQuantizationPipelineForEncoding(
        *gpu, original.ConstView(), maximum_encoding_prepared,
        maximum_error_options,
        gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
        {
          .frame = &maximum_encoding_frame,
          .score_history = &maximum_encoding_scores,
          .maximum_error_result = &maximum_encoding_result,
        },
        nullptr, &maximum_encoding_aq);
  }
  std::vector<uint8_t> maximum_encoding_codestream;
  if (maximum_error_status.ok()) {
    maximum_error_status = gjxl::EncodeVarDctCodestream(
      maximum_encoding_frame, &maximum_encoding_codestream);
  }
  gjxl::metal_internal::MetalAqReadbackStatsForTesting
    maximum_encoding_readback;
  if (maximum_error_status.ok() &&
      maximum_encoding_aq.evaluation != nullptr) {
    maximum_error_status =
      gjxl::metal_internal::GetMetalAqReadbackStatsForTesting(
        *maximum_encoding_aq.evaluation, &maximum_encoding_readback);
  }
  if (!maximum_error_status.ok() ||
      maximum_encoding_scores != maximum_error_full.scores ||
      maximum_encoding_result != maximum_error_full.maximum_error_result ||
      !FramesEqual(maximum_encoding_frame, maximum_error_full.frame) ||
      maximum_encoding_codestream != maximum_error_codestream ||
      maximum_encoding_readback.score_history_bytes != 0 ||
      maximum_encoding_readback.maximum_error_bytes == 0 ||
      maximum_encoding_readback.block_distance_map_bytes == 0 ||
      maximum_encoding_readback.frame_bytes != 0 ||
      maximum_encoding_readback.mapped_frame_bytes != 0 ||
      maximum_encoding_readback.reconstructed_rgb_bytes != 0) {
    std::cerr << "Encoding-only maximum-error pipeline changed output or "
                 "read diagnostics: " << maximum_error_status.message()
              << '\n';
    return false;
  }

  PipelineStorage throughput(kExtent, kExtent);
  gjxl::AcStrategyGpuSearchStats throughput_stats;
  const gjxl::GpuBackendStats before_throughput = gpu->stats();
  const gjxl::Status throughput_status = gjxl::RunGpuQuantizationPipeline(
      *gpu, original.ConstView(), opsin.ConstView(), options,
      gjxl::GpuAdaptiveQuantizationMode::kThroughput, throughput.Output(),
      &throughput_stats);
  const gjxl::GpuBackendStats after_throughput = gpu->stats();
  std::vector<uint8_t> throughput_codestream;
  const double throughput_initial_error = MaximumError(
      cpu.initial_quant, throughput.initial_quant);
  const double throughput_strategy_mask_error = MaximumError(
      cpu.strategy_mask, throughput.strategy_mask);
  const double throughput_pixel_mask_error = MaximumError(
      cpu.pixel_mask, throughput.pixel_mask);
  if (!throughput_status.ok() ||
      throughput_stats.total_candidate_count == 0 ||
      after_throughput.committed_submissions !=
          before_throughput.committed_submissions + 5 ||
      throughput_initial_error > 2.0e-6 ||
      throughput_strategy_mask_error > 2.0e-6 ||
      throughput_pixel_mask_error > 3.0e-5 || !throughput.frame.valid() ||
      throughput.scores.size() != 2 ||
      !gjxl::EncodeVarDctCodestream(
           throughput.frame, &throughput_codestream).ok() ||
      throughput_codestream.empty()) {
    std::cerr << "Throughput GPU pipeline failed: "
              << throughput_status.message()
              << " initial=" << throughput_initial_error
              << " strategy_mask=" << throughput_strategy_mask_error
              << " pixel_mask=" << throughput_pixel_mask_error << '\n';
    return false;
  }

  PipelineStorage maximum_output(kExtent, kExtent);
  gjxl::AcStrategyGpuSearchStats maximum_stats;
  maximum_stats.total_candidate_count = 31337;
  const gjxl::Status maximum_pipeline_status =
      gjxl::RunGpuQuantizationPipeline(
          *gpu, original.ConstView(), opsin.ConstView(), options,
          gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput,
          maximum_output.Output(), &maximum_stats);
  if (maximum_pipeline_status.code() !=
          gjxl::StatusCode::kInvalidArgument ||
      maximum_output.frame.valid() || !maximum_output.scores.empty() ||
      maximum_stats.total_candidate_count != 31337) {
    std::cerr << "Maximum-throughput mode reached the iterative pipeline\n";
    return false;
  }

  PipelineStorage invalid_mode_output(kExtent, kExtent);
  gjxl::AcStrategyGpuSearchStats invalid_mode_stats;
  invalid_mode_stats.total_candidate_count = 424242;
  const std::vector<float> invalid_initial = invalid_mode_output.initial_quant;
  const gjxl::GpuBackendStats before_invalid = gpu->stats();
  const gjxl::Status invalid_mode_status = gjxl::RunGpuQuantizationPipeline(
      *gpu, original.ConstView(), opsin.ConstView(), options,
      static_cast<gjxl::GpuAdaptiveQuantizationMode>(99),
      invalid_mode_output.Output(), &invalid_mode_stats);
  if (invalid_mode_status.code() != gjxl::StatusCode::kInvalidArgument ||
      invalid_mode_output.initial_quant != invalid_initial ||
      invalid_mode_output.frame.valid() ||
      !invalid_mode_output.scores.empty() ||
      invalid_mode_stats.total_candidate_count != 424242 ||
      gpu->stats().successful_allocations !=
          before_invalid.successful_allocations ||
      gpu->stats().committed_submissions !=
          before_invalid.committed_submissions) {
    std::cerr << "Invalid GPU pipeline AQ mode was not rejected atomically\n";
    return false;
  }
  std::cout << "Default-update GPU pipeline errors: quant=" << quant_error
            << " block=" << block_error << " score=" << score_error
            << " image=" << image_error
            << "; frame and codestream exact\n"
            << "Fully resident diagnostic errors: quant="
            << MaximumError(cpu.final_quant, resident.final_quant)
            << " block="
            << MaximumError(cpu.block_distance, resident.block_distance)
            << " score=" << MaximumScoreError(cpu.scores, resident.scores)
            << " image="
            << MaximumImageError(cpu.reconstructed, resident.reconstructed)
            << " frame="
            << (FramesEqual(cpu.frame, resident.frame) ? "exact" : "different")
            << " codestream="
            << (cpu_codestream == resident_codestream ? "exact" : "different")
            << '\n';
  return true;
}

bool CheckPreparedGpuAttemptReuse() {
  constexpr gjxl::Extent2D kExtent{96, 64};
  ImageStorage original(kExtent);
  ImageStorage padded_linear(kExtent);
  ImageStorage opsin(kExtent);
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      const std::array<float, 3> rgb = {
        0.08f + 0.005f * fx + 0.003f * fy,
        0.12f + 0.002f * fx + 0.007f * fy,
        ((x / 9 + y / 7) & 1u) == 0 ? 0.09f : 0.78f,
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        original.plane[channel][y * original.stride + x] = rgb[channel];
        padded_linear.plane[channel][y * padded_linear.stride + x] =
          rgb[channel];
      }
    }
  }
  if (!gjxl::LinearRgbToOpsin(
        padded_linear.ConstView(), 255.0f, opsin.View()).ok()) {
    return false;
  }

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu).ok()) {
    return false;
  }
  gjxl::CpuQuantizationPipelineOptions preparation_options;
  gjxl::quantization_pipeline_internal::PreparedQuantizationPipeline
    host_prepared;
  gjxl::Status status =
    gjxl::quantization_pipeline_internal::PrepareQuantizationPipeline(
      original.ConstView(), opsin.ConstView(), preparation_options,
      &host_prepared, false);
  if (!status.ok()) {
    std::cerr << "Prepared GPU host setup failed: "
              << status.message() << '\n';
    return false;
  }
  if (host_prepared.generation == 0 ||
      !ImagesShareStorage(host_prepared.coding_opsin, opsin.ConstView())) {
    std::cerr << "Prepared GPU pipeline did not borrow its coding image\n";
    return false;
  }
  gjxl::adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization
    gpu_prepared;
  constexpr std::array<float, 2> kTargets = {0.8f, 2.0f};
  std::array<std::vector<uint8_t>, 2> reused_codestreams;
  gjxl::PreparedAqEvaluation* first_evaluation = nullptr;
  for (size_t index = 0; index < kTargets.size(); ++index) {
    gjxl::CpuQuantizationPipelineOptions options = preparation_options;
    options.butteraugli_target = kTargets[index];
    PipelineStorage one_shot(kExtent, kExtent);
    PipelineStorage reused(kExtent, kExtent);
    status = gjxl::RunGpuQuantizationPipeline(
      *gpu, original.ConstView(), opsin.ConstView(), options,
      one_shot.Output());
    if (status.ok()) {
      status = gjxl::quantization_pipeline_internal::
        RunPreparedGpuQuantizationPipeline(
          *gpu, original.ConstView(), host_prepared, options,
          gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
          reused.Output(), nullptr, &gpu_prepared);
    }
    std::vector<uint8_t> one_shot_codestream;
    if (status.ok()) {
      status = gjxl::EncodeVarDctCodestream(
        one_shot.frame, &one_shot_codestream);
    }
    if (status.ok()) {
      status = gjxl::EncodeVarDctCodestream(
        reused.frame, &reused_codestreams[index]);
    }
    if (!status.ok() || gpu_prepared.evaluation == nullptr ||
        one_shot_codestream != reused_codestreams[index] ||
        one_shot.initial_quant != reused.initial_quant ||
        one_shot.strategy_mask != reused.strategy_mask ||
        one_shot.pixel_mask != reused.pixel_mask ||
        one_shot.final_quant != reused.final_quant ||
        one_shot.block_distance != reused.block_distance ||
        one_shot.scores != reused.scores ||
        MaximumImageError(one_shot.reconstructed, reused.reconstructed) !=
          0.0 ||
        !FramesEqual(one_shot.frame, reused.frame)) {
      std::cerr << "Prepared GPU attempt differs at target "
                << kTargets[index] << ": " << status.message() << '\n';
      return false;
    }
    if (index == 0) {
      first_evaluation = gpu_prepared.evaluation.get();
    } else if (gpu_prepared.evaluation.get() != first_evaluation) {
      std::cerr << "Prepared GPU AQ allocation was replaced between targets\n";
      return false;
    }
  }
  if (reused_codestreams[0] == reused_codestreams[1]) {
    std::cerr << "Prepared GPU pipeline cached target-dependent output\n";
    return false;
  }
  gjxl::PreparedAqEvaluation* first_resident_evaluation = nullptr;
  for (size_t index = 0; index < kTargets.size(); ++index) {
    gjxl::CpuQuantizationPipelineOptions options = preparation_options;
    options.butteraugli_target = kTargets[index];
    PipelineStorage resident_reused(kExtent, kExtent);
    status = gjxl::quantization_pipeline_internal::
      RunPreparedGpuQuantizationPipeline(
        *gpu, original.ConstView(), host_prepared, options,
        gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
        resident_reused.Output(), nullptr, &gpu_prepared);
    if (!status.ok() || gpu_prepared.evaluation == nullptr ||
        !resident_reused.frame.valid()) {
      std::cerr << "Prepared resident GPU attempt failed at target "
                << kTargets[index] << ": " << status.message() << '\n';
      return false;
    }
    if (index == 0) {
      first_resident_evaluation = gpu_prepared.evaluation.get();
    } else if (gpu_prepared.evaluation.get() !=
               first_resident_evaluation) {
      std::cerr << "Prepared resident evaluator was replaced between targets\n";
      return false;
    }
  }
  if (gpu_prepared.quantization_pipeline_generation !=
      host_prepared.generation) {
    std::cerr << "Prepared resident evaluator lost its source generation\n";
    return false;
  }

  const uint64_t first_generation = host_prepared.generation;
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      const float fx = static_cast<float>(x) /
        static_cast<float>(kExtent.width - 1);
      const float fy = static_cast<float>(y) /
        static_cast<float>(kExtent.height - 1);
      const std::array<float, 3> rgb = {
        0.67f - 0.42f * fx + 0.08f * fy,
        0.09f + 0.18f * fx + 0.53f * fy,
        ((x / 5 + y / 3) & 1u) == 0 ? 0.74f : 0.06f,
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        original.plane[channel][y * original.stride + x] = rgb[channel];
        padded_linear.plane[channel][y * padded_linear.stride + x] =
          rgb[channel];
      }
    }
  }
  if (!gjxl::LinearRgbToOpsin(
        padded_linear.ConstView(), 255.0f, opsin.View()).ok()) {
    return false;
  }
  status =
    gjxl::quantization_pipeline_internal::PrepareQuantizationPipeline(
      original.ConstView(), opsin.ConstView(), preparation_options,
      &host_prepared, false);
  if (!status.ok() || host_prepared.generation == first_generation ||
      !ImagesShareStorage(host_prepared.coding_opsin, opsin.ConstView()) ||
      gpu_prepared.quantization_pipeline_generation != first_generation) {
    std::cerr << "Re-prepared GPU pipeline did not advance its borrowed "
                 "source generation: "
              << status.message() << '\n';
    return false;
  }

  gjxl::CpuQuantizationPipelineOptions changed_options = preparation_options;
  changed_options.butteraugli_target = 1.1f;
  PipelineStorage changed_one_shot(kExtent, kExtent);
  PipelineStorage changed_reused(kExtent, kExtent);
  status = gjxl::RunGpuQuantizationPipeline(
    *gpu, original.ConstView(), opsin.ConstView(), changed_options,
    gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
    changed_one_shot.Output());
  if (status.ok()) {
    status = gjxl::quantization_pipeline_internal::
      RunPreparedGpuQuantizationPipeline(
        *gpu, original.ConstView(), host_prepared, changed_options,
        gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
        changed_reused.Output(), nullptr, &gpu_prepared);
  }
  std::vector<uint8_t> changed_one_shot_codestream;
  std::vector<uint8_t> changed_reused_codestream;
  if (status.ok()) {
    status = gjxl::EncodeVarDctCodestream(
      changed_one_shot.frame, &changed_one_shot_codestream);
  }
  if (status.ok()) {
    status = gjxl::EncodeVarDctCodestream(
      changed_reused.frame, &changed_reused_codestream);
  }
  if (!status.ok() ||
      gpu_prepared.quantization_pipeline_generation !=
        host_prepared.generation ||
      changed_one_shot_codestream != changed_reused_codestream ||
      changed_one_shot.initial_quant != changed_reused.initial_quant ||
      changed_one_shot.strategy_mask != changed_reused.strategy_mask ||
      changed_one_shot.pixel_mask != changed_reused.pixel_mask ||
      changed_one_shot.final_quant != changed_reused.final_quant ||
      changed_one_shot.block_distance != changed_reused.block_distance ||
      changed_one_shot.scores != changed_reused.scores ||
      MaximumImageError(
        changed_one_shot.reconstructed, changed_reused.reconstructed) !=
        0.0 ||
      !FramesEqual(changed_one_shot.frame, changed_reused.frame)) {
    std::cerr << "Prepared GPU pipeline reused stale borrowed pixels: "
              << status.message() << '\n';
    return false;
  }
  std::cout << "Prepared GPU attempts reuse exact and resident evaluator "
               "allocations and invalidate changed source generations\n";
  return true;
}

bool CheckWorkflowBackendSelection() {
  constexpr gjxl::Extent2D kExtent{128, 96};
  ImageStorage original(kExtent);
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      const float fx = static_cast<float>(x) /
          static_cast<float>(kExtent.width - 1);
      const float fy = static_cast<float>(y) /
          static_cast<float>(kExtent.height - 1);
      original.plane[0][y * original.stride + x] =
          0.06f + 0.78f * fx;
      original.plane[1][y * original.stride + x] =
          0.08f + 0.72f * fy + 0.03f * std::sin(19.0f * fx);
      original.plane[2][y * original.stride + x] =
          0.04f + 0.31f * fx + 0.46f * fy;
    }
  }

  if (gjxl::codestream_internal::IsAutomaticMetalGeometryEligible({64, 48}) ||
      gjxl::codestream_internal::IsAutomaticMetalGeometryEligible({96, 64}) ||
      !gjxl::codestream_internal::IsAutomaticMetalGeometryEligible(kExtent) ||
      !gjxl::codestream_internal::IsAutomaticMetalGeometryEligible({128, 96}) ||
      gjxl::codestream_internal::IsAutomaticMetalTargetEligible(0.5f) ||
      gjxl::codestream_internal::IsAutomaticMetalTargetEligible(0.999f) ||
      !gjxl::codestream_internal::IsAutomaticMetalTargetEligible(1.0f) ||
      !gjxl::codestream_internal::IsAutomaticMetalTargetEligible(1.2f) ||
      gjxl::codestream_internal::IsAutomaticMetalTargetEligible(1.201f) ||
      gjxl::codestream_internal::IsAutomaticMetalTargetEligible(2.0f) ||
      gjxl::codestream_internal::IsAutomaticMetalTargetEligible(
          std::numeric_limits<float>::infinity()) ||
      gjxl::codestream_internal::IsAutomaticMetalTargetEligible(
          std::numeric_limits<float>::quiet_NaN())) {
    std::cerr << "Automatic Metal geometry or quality policy changed\n";
    return false;
  }

  std::vector<uint8_t> cpu_bytes;
  std::vector<uint8_t> forced_bytes;
  std::vector<uint8_t> automatic_bytes;
  std::vector<uint8_t> unqualified_bytes;
  std::vector<uint8_t> exact_bytes;
  gjxl::VarDctEncodingSummary cpu_summary;
  gjxl::VarDctEncodingSummary forced_summary;
  gjxl::VarDctEncodingSummary automatic_summary;
  gjxl::VarDctEncodingSummary unqualified_summary;
  gjxl::VarDctEncodingSummary exact_summary;
  const auto encode = [&](gjxl::VarDctBackendPreference preference,
                          gjxl::GpuBackend* backend, bool qualified,
                          std::vector<uint8_t>* bytes,
                          gjxl::VarDctEncodingSummary* summary) {
    return gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
            original.ConstView(),
            {.butteraugli_target = 1.0f, .backend = preference}, backend,
            qualified, bytes, summary);
  };
  if (!encode(gjxl::VarDctBackendPreference::kCpu, nullptr, false,
              &cpu_bytes, &cpu_summary).ok()) {
    return false;
  }
  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu).ok() ||
      !encode(gjxl::VarDctBackendPreference::kMetal, gpu.get(), false,
              &forced_bytes, &forced_summary).ok() ||
      !encode(gjxl::VarDctBackendPreference::kAutomatic, gpu.get(), true,
              &automatic_bytes, &automatic_summary).ok() ||
      !encode(gjxl::VarDctBackendPreference::kAutomatic, gpu.get(), false,
              &unqualified_bytes, &unqualified_summary).ok() ||
      !gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          original.ConstView(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kMetal,
           .metal_aq_mode =
             gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients},
          gpu.get(), false, &exact_bytes, &exact_summary).ok()) {
    std::cerr << "Public workflow backend selection failed\n";
    return false;
  }
  if (forced_bytes != automatic_bytes || cpu_bytes != unqualified_bytes ||
      cpu_bytes != exact_bytes ||
      MaximumScoreError(cpu_summary.score_history,
                        exact_summary.score_history) > 2.0e-3 ||
      forced_summary.score_history != automatic_summary.score_history ||
      cpu_summary.score_history != unqualified_summary.score_history ||
      forced_summary.strategy_counts != automatic_summary.strategy_counts ||
      cpu_summary.strategy_counts != exact_summary.strategy_counts ||
      cpu_summary.strategy_counts != unqualified_summary.strategy_counts ||
      cpu_summary.execution_backend != gjxl::VarDctExecutionBackend::kCpu ||
      !cpu_summary.final_butteraugli_score_evaluated ||
      forced_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      forced_summary.final_butteraugli_score_evaluated ||
      forced_summary.score_history.size() != 2 ||
      automatic_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      automatic_summary.final_butteraugli_score_evaluated ||
      unqualified_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kCpu ||
      !unqualified_summary.final_butteraugli_score_evaluated ||
      forced_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kFullyResident ||
      automatic_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kFullyResident ||
      exact_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      !exact_summary.final_butteraugli_score_evaluated ||
      exact_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients) {
    std::cerr << "Public workflow default or fallback policy changed\n";
    return false;
  }

  const gjxl::VarDctEncodingOptions maximum_error_options{
    .rate_control_mode = gjxl::VarDctRateControlMode::kMaximumError,
    .maximum_error = {0.05f, 0.05f, 0.05f},
  };
  std::vector<uint8_t> maximum_cpu_bytes;
  std::vector<uint8_t> maximum_metal_bytes;
  std::vector<uint8_t> maximum_automatic_bytes;
  gjxl::VarDctEncodingSummary maximum_cpu_summary;
  gjxl::VarDctEncodingSummary maximum_metal_summary;
  gjxl::VarDctEncodingSummary maximum_automatic_summary;
  auto maximum_cpu_options = maximum_error_options;
  maximum_cpu_options.backend = gjxl::VarDctBackendPreference::kCpu;
  auto maximum_metal_options = maximum_error_options;
  maximum_metal_options.backend = gjxl::VarDctBackendPreference::kMetal;
  maximum_metal_options.metal_aq_mode =
    gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients;
  auto maximum_automatic_options = maximum_error_options;
  maximum_automatic_options.backend =
    gjxl::VarDctBackendPreference::kAutomatic;
  if (!gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          original.ConstView(), maximum_cpu_options, nullptr, false,
          &maximum_cpu_bytes, &maximum_cpu_summary).ok() ||
      !gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          original.ConstView(), maximum_metal_options, gpu.get(), false,
          &maximum_metal_bytes, &maximum_metal_summary).ok() ||
      !gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          original.ConstView(), maximum_automatic_options, gpu.get(), true,
          &maximum_automatic_bytes, &maximum_automatic_summary).ok() ||
      maximum_cpu_bytes != maximum_metal_bytes ||
      maximum_cpu_bytes != maximum_automatic_bytes ||
      maximum_cpu_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kCpu ||
      maximum_metal_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kMetal ||
      maximum_automatic_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kCpu ||
      maximum_metal_summary.maximum_error_evaluation_count != 6 ||
      maximum_metal_summary.maximum_error_outcome !=
        maximum_cpu_summary.maximum_error_outcome ||
      MaximumScoreError(maximum_cpu_summary.score_history,
                        maximum_metal_summary.score_history) > 2.0e-3) {
    std::cerr << "Maximum-error workflow backend parity failed\n";
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (std::abs(
          maximum_cpu_summary.achieved_maximum_error[channel] -
          maximum_metal_summary.achieved_maximum_error[channel]) > 2.0e-3f) {
      std::cerr << "Maximum-error workflow diagnostics differ\n";
      return false;
    }
  }

  std::vector<uint8_t> maximum_resident_bytes;
  gjxl::VarDctEncodingSummary maximum_resident_summary;
  auto maximum_resident_options = maximum_metal_options;
  maximum_resident_options.metal_aq_mode =
    gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  if (!gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          original.ConstView(), maximum_resident_options, gpu.get(), false,
          &maximum_resident_bytes, &maximum_resident_summary).ok() ||
      maximum_resident_bytes.empty() ||
      maximum_resident_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kMetal ||
      maximum_resident_summary.metal_aq_mode !=
        gjxl::GpuAdaptiveQuantizationMode::kFullyResident ||
      maximum_resident_summary.maximum_error_evaluation_count != 6 ||
      !std::isfinite(
        maximum_resident_summary.achieved_maximum_error_ratio)) {
    std::cerr << "Resident maximum-error workflow failed\n";
    return false;
  }

  std::vector<uint8_t> resident_bytes;
  gjxl::VarDctEncodingSummary resident_summary;
  if (!gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = 1.0f,
               .backend = gjxl::VarDctBackendPreference::kMetal,
               .metal_aq_mode =
                   gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
               .collect_final_butteraugli_score = true},
              gpu.get(), false, &resident_bytes, &resident_summary)
          .ok() ||
      resident_bytes.empty() || resident_summary.score_history.size() != 3 ||
      !resident_summary.final_butteraugli_score_evaluated ||
      resident_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      resident_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kFullyResident) {
    std::cerr << "Forced fully resident public workflow failed\n";
    return false;
  }

  std::vector<uint8_t> resident_default_bytes;
  gjxl::VarDctEncodingSummary resident_default_summary;
  if (!gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = 1.0f,
               .backend = gjxl::VarDctBackendPreference::kMetal},
              gpu.get(), false, &resident_default_bytes,
              &resident_default_summary)
          .ok() ||
      resident_default_bytes != resident_bytes ||
      resident_default_summary.score_history.size() != 2 ||
      resident_default_summary.final_butteraugli_score_evaluated ||
      !std::equal(
        resident_default_summary.score_history.begin(),
        resident_default_summary.score_history.end(),
        resident_summary.score_history.begin()) ||
      resident_default_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      resident_default_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kFullyResident) {
    std::cerr << "Default fully resident workflow changed encoded output\n";
    return false;
  }

  const gjxl::VarDctEncodingOptions automatic_target_options{
    .rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
    // An impossible one-byte target forces the deterministic subdivision to
    // attempt target 1.102... at attempt 37, inside the automatic Metal gate.
    .target_bytes = 1,
    .target_size_tolerance = 0.0,
    .target_size_maximum_attempts = 37,
    .backend = gjxl::VarDctBackendPreference::kAutomatic,
  };
  auto cpu_target_options = automatic_target_options;
  cpu_target_options.backend = gjxl::VarDctBackendPreference::kCpu;
  std::vector<uint8_t> automatic_target_bytes;
  std::vector<uint8_t> cpu_target_bytes;
  gjxl::VarDctEncodingSummary automatic_target_summary;
  gjxl::VarDctEncodingSummary cpu_target_summary;
  const gjxl::GpuBackendStats before_automatic_target = gpu->stats();
  if (!gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          original.ConstView(), automatic_target_options, gpu.get(), true,
          &automatic_target_bytes, &automatic_target_summary).ok() ||
      gpu->stats().committed_submissions !=
        before_automatic_target.committed_submissions ||
      !gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          original.ConstView(), cpu_target_options, nullptr, false,
          &cpu_target_bytes, &cpu_target_summary).ok() ||
      automatic_target_bytes != cpu_target_bytes ||
      automatic_target_summary != cpu_target_summary ||
      automatic_target_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kCpu) {
    std::cerr << "Automatic resident target-size workflow did not stay on CPU\n";
    return false;
  }

  std::vector<uint8_t> maximum_bytes;
  gjxl::VarDctEncodingSummary maximum_summary;
  if (!gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = 1.0f,
               .backend = gjxl::VarDctBackendPreference::kMetal,
               .metal_aq_mode =
                   gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput},
              gpu.get(), false, &maximum_bytes, &maximum_summary)
          .ok() ||
      maximum_bytes.empty() || !maximum_summary.score_history.empty() ||
      maximum_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      maximum_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput ||
      maximum_summary.strategy_counts[
          static_cast<size_t>(gjxl::AcStrategyType::kDct8)] !=
          (kExtent.width / gjxl::kJxlBlockDimension) *
            (kExtent.height / gjxl::kJxlBlockDimension)) {
    std::cerr << "Forced maximum-throughput public workflow failed\n";
    return false;
  }

  std::vector<uint8_t> maximum_target_bytes;
  gjxl::VarDctEncodingSummary maximum_target_summary;
  gjxl::VarDctEncodingOptions maximum_target_options{
      .rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
      .target_bytes = 280,
      .target_size_tolerance = 0.1,
      .target_size_selection =
          gjxl::TargetSizeSelectionPolicy::kClosestAbsolute,
      .backend = gjxl::VarDctBackendPreference::kMetal,
      .metal_aq_mode =
          gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput,
  };
  if (!gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(), maximum_target_options, gpu.get(), false,
              &maximum_target_bytes, &maximum_target_summary)
          .ok() ||
      maximum_target_bytes.empty() ||
      maximum_target_summary.rate_control_mode !=
          gjxl::VarDctRateControlMode::kTargetBytes ||
      maximum_target_summary.encode_attempt_count == 0 ||
      !maximum_target_summary.score_history.empty() ||
      maximum_target_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      maximum_target_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput) {
    std::cerr << "Maximum-throughput target-size workflow failed\n";
    return false;
  }

  std::vector<uint8_t> maximum_failed_bytes{4, 2, 1};
  const std::vector<uint8_t> maximum_failed_bytes_original =
      maximum_failed_bytes;
  gjxl::VarDctEncodingSummary maximum_failed_summary{
      .extent = {9, 7}, .encoded_bytes = 23, .score_history = {5.0}};
  const gjxl::VarDctEncodingSummary maximum_failed_summary_original =
      maximum_failed_summary;
  if (!gjxl::ArmNextMetalSubmissionFailureForTest(*gpu, true, false).ok()) {
    std::cerr << "Could not arm maximum-throughput submission failure\n";
    return false;
  }
  const gjxl::Status maximum_failure = gjxl::codestream_internal::
      EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
          original.ConstView(),
          {.butteraugli_target = 1.0f,
           .backend = gjxl::VarDctBackendPreference::kMetal,
           .metal_aq_mode =
               gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput},
          gpu.get(), false, &maximum_failed_bytes, &maximum_failed_summary);
  if (maximum_failure.ok() ||
      maximum_failed_bytes != maximum_failed_bytes_original ||
      maximum_failed_summary != maximum_failed_summary_original) {
    std::cerr << "Maximum-throughput failure was not atomic\n";
    return false;
  }

  std::vector<uint8_t> throughput_bytes;
  gjxl::VarDctEncodingSummary throughput_summary;
  if (!gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = 1.0f,
               .backend = gjxl::VarDctBackendPreference::kMetal,
               .metal_aq_mode =
                   gjxl::GpuAdaptiveQuantizationMode::kThroughput},
              gpu.get(), false, &throughput_bytes, &throughput_summary)
          .ok() ||
      throughput_bytes.empty() ||
      throughput_bytes != resident_bytes ||
      throughput_summary.score_history.size() != 2 ||
      throughput_summary.final_butteraugli_score_evaluated ||
      !std::equal(
        throughput_summary.score_history.begin(),
        throughput_summary.score_history.end(),
        resident_summary.score_history.begin()) ||
      throughput_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      throughput_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kThroughput) {
    std::cerr << "Forced throughput public workflow failed\n";
    return false;
  }

  std::vector<uint8_t> throughput_scored_bytes;
  gjxl::VarDctEncodingSummary throughput_scored_summary;
  if (!gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = 1.0f,
               .backend = gjxl::VarDctBackendPreference::kMetal,
               .metal_aq_mode =
                   gjxl::GpuAdaptiveQuantizationMode::kThroughput,
               .collect_final_butteraugli_score = true},
              gpu.get(), false, &throughput_scored_bytes,
              &throughput_scored_summary)
          .ok() ||
      throughput_scored_bytes != throughput_bytes ||
      throughput_scored_summary.score_history !=
        resident_summary.score_history ||
      !throughput_scored_summary.final_butteraugli_score_evaluated ||
      throughput_scored_summary.metal_aq_mode !=
        gjxl::GpuAdaptiveQuantizationMode::kThroughput) {
    std::cerr << "Throughput final-score collection changed encoded output\n";
    return false;
  }

  {
    std::vector<uint8_t> upper_forced_bytes;
    std::vector<uint8_t> upper_automatic_bytes;
    gjxl::VarDctEncodingSummary upper_forced_summary;
    gjxl::VarDctEncodingSummary upper_automatic_summary;
    if (!gjxl::codestream_internal::
            EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                original.ConstView(),
                {.butteraugli_target = 1.2f,
                 .backend = gjxl::VarDctBackendPreference::kMetal},
                gpu.get(), false, &upper_forced_bytes, &upper_forced_summary)
            .ok() ||
        !gjxl::codestream_internal::
            EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                original.ConstView(),
                {.butteraugli_target = 1.2f,
                 .backend = gjxl::VarDctBackendPreference::kAutomatic},
                gpu.get(), true, &upper_automatic_bytes,
                &upper_automatic_summary)
            .ok() ||
        upper_forced_bytes != upper_automatic_bytes ||
        upper_forced_summary.score_history !=
          upper_automatic_summary.score_history ||
        upper_automatic_summary.execution_backend !=
            gjxl::VarDctExecutionBackend::kMetal ||
        upper_automatic_summary.metal_aq_mode !=
            gjxl::GpuAdaptiveQuantizationMode::kFullyResident) {
      std::cerr << "Automatic Metal did not accept the upper quality bound\n";
      return false;
    }
  }

  for (float target : {0.5f, 0.999f, 1.201f, 2.0f}) {
    std::vector<uint8_t> target_cpu_bytes;
    std::vector<uint8_t> target_automatic_bytes;
    gjxl::VarDctEncodingSummary target_cpu_summary;
    gjxl::VarDctEncodingSummary target_automatic_summary;
    if (!gjxl::codestream_internal::
            EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                original.ConstView(),
                {.butteraugli_target = target,
                 .backend = gjxl::VarDctBackendPreference::kCpu},
                nullptr, false, &target_cpu_bytes, &target_cpu_summary)
            .ok() ||
        !gjxl::codestream_internal::
            EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                original.ConstView(),
                {.butteraugli_target = target,
                 .backend = gjxl::VarDctBackendPreference::kAutomatic},
                gpu.get(), true, &target_automatic_bytes,
                &target_automatic_summary)
            .ok() ||
        target_cpu_bytes != target_automatic_bytes ||
        target_cpu_summary.score_history !=
            target_automatic_summary.score_history ||
        target_automatic_summary.execution_backend !=
            gjxl::VarDctExecutionBackend::kCpu) {
      std::cerr << "Automatic Metal did not preserve CPU outside its quality "
                   "window at distance "
                << target << '\n';
      return false;
    }
  }

  std::vector<uint8_t> failed_bytes{9, 2, 6};
  const std::vector<uint8_t> failed_bytes_original = failed_bytes;
  gjxl::VarDctEncodingSummary failed_summary{
      .extent = {5, 4}, .encoded_bytes = 71, .score_history = {3.0}};
  const gjxl::VarDctEncodingSummary failed_summary_original = failed_summary;
  if (!gjxl::ArmNextMetalSubmissionFailureForTest(
           *gpu, true, false).ok()) {
    return false;
  }
  const gjxl::Status operational_failure = encode(
      gjxl::VarDctBackendPreference::kAutomatic, gpu.get(), true,
      &failed_bytes, &failed_summary);
  if (operational_failure.code() != gjxl::StatusCode::kSubmissionFailed ||
      failed_bytes != failed_bytes_original ||
      failed_summary != failed_summary_original) {
    std::cerr << "Automatic Metal operational failure fell back or committed\n";
    return false;
  }
  std::vector<uint8_t> retry_bytes;
  gjxl::VarDctEncodingSummary retry_summary;
  if (!encode(gjxl::VarDctBackendPreference::kMetal, gpu.get(), false,
              &retry_bytes, &retry_summary).ok() ||
      retry_bytes != forced_bytes) {
    std::cerr << "Metal backend was not reusable after AC submission failure\n";
    return false;
  }

  std::unique_ptr<gjxl::GpuBackend> embedded;
  if (!gjxl::CreateEmbeddedMetalBackend({}, &embedded).ok() ||
      embedded == nullptr) {
    std::cerr << "Embedded production Metal backend is unavailable\n";
    return false;
  }
  const gjxl::VarDctExecutionBackend expected_production_backend =
      gjxl::codestream_internal::IsAutomaticMetalBackendQualified(*embedded)
          ? gjxl::VarDctExecutionBackend::kMetal
          : gjxl::VarDctExecutionBackend::kCpu;
  std::vector<uint8_t> production_auto_bytes;
  gjxl::VarDctEncodingSummary production_auto_summary;
  if (!gjxl::EncodeLinearRgbVarDctCodestream(
           original.ConstView(),
           {.butteraugli_target = 1.0f,
            .backend = gjxl::VarDctBackendPreference::kAutomatic},
           &production_auto_bytes, &production_auto_summary).ok() ||
      production_auto_bytes !=
        (expected_production_backend == gjxl::VarDctExecutionBackend::kMetal
           ? forced_bytes
           : cpu_bytes) ||
      production_auto_summary.execution_backend !=
          expected_production_backend ||
      (expected_production_backend == gjxl::VarDctExecutionBackend::kMetal &&
       production_auto_summary.metal_aq_mode !=
         gjxl::GpuAdaptiveQuantizationMode::kFullyResident)) {
    std::cerr << "Production automatic Metal selection failed\n";
    return false;
  }
  const gjxl::Status empty_library = gjxl::CreateMetalBackend(
      std::span<const uint8_t>{}, &embedded);
  if (empty_library.code() != gjxl::StatusCode::kInvalidArgument ||
      embedded != nullptr) {
    std::cerr << "Empty in-memory Metal library was accepted\n";
    return false;
  }

  BackendWithoutAq missing;
  std::vector<uint8_t> missing_auto_bytes;
  gjxl::VarDctEncodingSummary missing_auto_summary;
  if (!encode(gjxl::VarDctBackendPreference::kAutomatic, &missing, true,
              &missing_auto_bytes, &missing_auto_summary).ok() ||
      missing_auto_bytes != cpu_bytes ||
      missing_auto_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kCpu) {
    std::cerr << "Automatic workflow did not fall back before GPU execution\n";
    return false;
  }
  std::vector<uint8_t> sentinel_bytes{3, 1, 4};
  const std::vector<uint8_t> original_sentinel = sentinel_bytes;
  gjxl::VarDctEncodingSummary sentinel_summary{
      .extent = {7, 5}, .encoded_bytes = 19, .score_history = {2.0}};
  const gjxl::VarDctEncodingSummary original_summary = sentinel_summary;
  const gjxl::Status missing_status =
      gjxl::codestream_internal::
          EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
              original.ConstView(),
              {.butteraugli_target = 1.0f,
               .backend = gjxl::VarDctBackendPreference::kMetal},
              &missing, true, &sentinel_bytes, &sentinel_summary);
  if (missing_status.code() != gjxl::StatusCode::kUnavailable ||
      sentinel_bytes != original_sentinel ||
      sentinel_summary != original_summary) {
    std::cerr << "Forced Metal capability failure was not atomic\n";
    return false;
  }

  std::cout << "Automatic and forced Metal preserve public workflow bytes\n";
  return true;
}

} // namespace

int main() {
  if (!CheckGpuGaborish() || !CheckGpuPipelineParity() ||
      !CheckMaximumThroughputFrontendParity() ||
      !CheckDefaultUpdatePipelineParity() ||
      !CheckPreparedGpuAttemptReuse() ||
      !CheckWorkflowBackendSelection()) {
    return EXIT_FAILURE;
  }
  std::cout << "Complete GPU quantization pipeline matches CPU.\n";
  return EXIT_SUCCESS;
}
