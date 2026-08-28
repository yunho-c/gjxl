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
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "codec/color_transform.h"
#include "codec/quantization_pipeline.h"
#include "codec/quantization_pipeline_internal.h"
#include "codestream/encoder.h"
#include "codestream/workflow.h"
#include "codestream/workflow_internal.h"
#include "gpu/metal/metal_backend.h"
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
  if (!resident_status.ok() || resident_stats.total_candidate_count == 0 ||
      after_resident.committed_submissions !=
          before_resident.committed_submissions +
              options.adaptive_quantization.iterations + 3 ||
      cpu.initial_quant != resident.initial_quant ||
      cpu.strategy_mask != resident.strategy_mask ||
      cpu.pixel_mask != resident.pixel_mask || !resident.frame.valid() ||
      resident.scores.size() != options.adaptive_quantization.iterations + 1 ||
      !gjxl::EncodeVarDctCodestream(
           resident.frame, &resident_codestream).ok() ||
      resident_codestream.empty()) {
    std::cerr << "Fully resident complete GPU pipeline failed: "
              << resident_status.message() << '\n';
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
  std::cout << "Prepared GPU attempts reuse one AQ allocation exactly\n";
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
  gjxl::VarDctEncodingSummary cpu_summary;
  gjxl::VarDctEncodingSummary forced_summary;
  gjxl::VarDctEncodingSummary automatic_summary;
  gjxl::VarDctEncodingSummary unqualified_summary;
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
              &unqualified_bytes, &unqualified_summary).ok()) {
    std::cerr << "Public workflow backend selection failed\n";
    return false;
  }
  if (cpu_bytes != forced_bytes || cpu_bytes != automatic_bytes ||
      cpu_bytes != unqualified_bytes ||
      MaximumScoreError(cpu_summary.score_history,
                        forced_summary.score_history) > 2.0e-3 ||
      MaximumScoreError(cpu_summary.score_history,
                        automatic_summary.score_history) > 2.0e-3 ||
      cpu_summary.score_history != unqualified_summary.score_history ||
      cpu_summary.strategy_counts != forced_summary.strategy_counts ||
      cpu_summary.strategy_counts != automatic_summary.strategy_counts ||
      cpu_summary.strategy_counts != unqualified_summary.strategy_counts ||
      cpu_summary.execution_backend != gjxl::VarDctExecutionBackend::kCpu ||
      forced_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      automatic_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      unqualified_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kCpu ||
      forced_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients ||
      automatic_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients) {
    std::cerr << "Public workflow backend changed accepted decisions: "
              << "forced_bytes=" << (cpu_bytes == forced_bytes)
              << " automatic_bytes=" << (cpu_bytes == automatic_bytes)
              << " unqualified_bytes=" << (cpu_bytes == unqualified_bytes)
              << " forced_score_error="
              << MaximumScoreError(cpu_summary.score_history,
                                   forced_summary.score_history)
              << " automatic_score_error="
              << MaximumScoreError(cpu_summary.score_history,
                                   automatic_summary.score_history)
              << " unqualified_scores="
              << (cpu_summary.score_history ==
                  unqualified_summary.score_history)
              << " forced_strategies="
              << (cpu_summary.strategy_counts ==
                  forced_summary.strategy_counts)
              << " automatic_strategies="
              << (cpu_summary.strategy_counts ==
                  automatic_summary.strategy_counts)
              << " unqualified_strategies="
              << (cpu_summary.strategy_counts ==
                  unqualified_summary.strategy_counts)
              << " backends="
              << static_cast<int>(cpu_summary.execution_backend) << ','
              << static_cast<int>(forced_summary.execution_backend) << ','
              << static_cast<int>(automatic_summary.execution_backend) << ','
              << static_cast<int>(unqualified_summary.execution_backend)
              << '\n';
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
                   gjxl::GpuAdaptiveQuantizationMode::kFullyResident},
              gpu.get(), false, &resident_bytes, &resident_summary)
          .ok() ||
      resident_bytes.empty() || resident_summary.score_history.size() != 3 ||
      resident_summary.execution_backend !=
          gjxl::VarDctExecutionBackend::kMetal ||
      resident_summary.metal_aq_mode !=
          gjxl::GpuAdaptiveQuantizationMode::kFullyResident) {
    std::cerr << "Forced fully resident public workflow failed\n";
    return false;
  }

  for (gjxl::VarDctBackendPreference backend : {
           gjxl::VarDctBackendPreference::kAutomatic,
           gjxl::VarDctBackendPreference::kCpu}) {
    std::vector<uint8_t> rejected_bytes{8, 6, 7};
    const std::vector<uint8_t> original_rejected_bytes = rejected_bytes;
    gjxl::VarDctEncodingSummary rejected_summary{
        .extent = {3, 2}, .encoded_bytes = 99, .score_history = {4.0}};
    const gjxl::VarDctEncodingSummary original_rejected_summary =
        rejected_summary;
    const gjxl::Status rejected = gjxl::codestream_internal::
        EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
            original.ConstView(),
            {.butteraugli_target = 1.0f,
             .backend = backend,
             .metal_aq_mode =
                 gjxl::GpuAdaptiveQuantizationMode::kFullyResident},
            gpu.get(), true, &rejected_bytes, &rejected_summary);
    if (rejected.code() != gjxl::StatusCode::kInvalidArgument ||
        rejected_bytes != original_rejected_bytes ||
        rejected_summary != original_rejected_summary) {
      std::cerr << "Fully resident workflow was not explicit or atomic\n";
      return false;
    }
  }

  {
    std::vector<uint8_t> upper_cpu_bytes;
    std::vector<uint8_t> upper_automatic_bytes;
    gjxl::VarDctEncodingSummary upper_cpu_summary;
    gjxl::VarDctEncodingSummary upper_automatic_summary;
    if (!gjxl::codestream_internal::
            EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                original.ConstView(),
                {.butteraugli_target = 1.2f,
                 .backend = gjxl::VarDctBackendPreference::kCpu},
                nullptr, false, &upper_cpu_bytes, &upper_cpu_summary)
            .ok() ||
        !gjxl::codestream_internal::
            EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                original.ConstView(),
                {.butteraugli_target = 1.2f,
                 .backend = gjxl::VarDctBackendPreference::kAutomatic},
                gpu.get(), true, &upper_automatic_bytes,
                &upper_automatic_summary)
            .ok() ||
        upper_cpu_bytes != upper_automatic_bytes ||
        MaximumScoreError(upper_cpu_summary.score_history,
                          upper_automatic_summary.score_history) > 2.0e-3 ||
        upper_automatic_summary.execution_backend !=
            gjxl::VarDctExecutionBackend::kMetal) {
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
      retry_bytes != cpu_bytes) {
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
      production_auto_bytes != cpu_bytes ||
      production_auto_summary.execution_backend !=
          expected_production_backend) {
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
  if (!CheckGpuPipelineParity() || !CheckDefaultUpdatePipelineParity() ||
      !CheckPreparedGpuAttemptReuse() ||
      !CheckWorkflowBackendSelection()) {
    return EXIT_FAILURE;
  }
  std::cout << "Complete GPU quantization pipeline matches CPU.\n";
  return EXIT_SUCCESS;
}
