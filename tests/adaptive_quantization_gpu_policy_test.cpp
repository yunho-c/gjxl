// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

#include "codec/adaptive_quantization.h"
#include "codec/color_transform.h"
#include "codec/quantization.h"
#include "codestream/encoder.h"
#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/adaptive_quantization.h"

namespace {

constexpr gjxl::Extent2D kOriginalExtent{96, 64};
constexpr gjxl::Extent2D kPaddedExtent{96, 64};
constexpr gjxl::Extent2D kBlockExtent{12, 8};
constexpr uint32_t kPoisonBits = 0x7fc12345u;
constexpr float kPoison = std::bit_cast<float>(kPoisonBits);
constexpr double kTolerance = 2.0e-3;
double g_max_score_error = 0.0;
double g_max_quant_error = 0.0;
double g_max_block_error = 0.0;
double g_max_image_error = 0.0;

bool CheckStatus(gjxl::Status status, std::string_view operation) {
  if (status.ok()) return true;
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

bool ExpectCode(gjxl::Status status, gjxl::StatusCode expected,
                std::string_view operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " returned " << static_cast<int>(status.code())
            << ", expected " << static_cast<int>(expected) << ": "
            << status.message() << '\n';
  return false;
}

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D image_extent, float fill = -777.0f)
      : extent(image_extent), stride(image_extent.width + 3) {
    for (std::vector<float>& values : plane) {
      values.assign(stride * extent.height, fill);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

void FillPaddedLinear(ImageStorage* padded, ImageStorage* original) {
  for (size_t y = 0; y < kPaddedExtent.height; ++y) {
    const size_t source_y = std::min(y, kOriginalExtent.height - 1);
    for (size_t x = 0; x < kPaddedExtent.width; ++x) {
      const size_t source_x = std::min(x, kOriginalExtent.width - 1);
      const float fx = static_cast<float>(source_x);
      const float fy = static_cast<float>(source_y);
      const std::array<float, 3> rgb = {
        0.12f + 0.006f * fx + 0.004f * fy,
        0.18f + 0.003f * fx + 0.006f * fy,
        0.08f + 0.005f * fx + 0.004f * fy,
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

struct CpuOutputStorage {
  static constexpr size_t kBlockStride = kBlockExtent.width + 2;

  CpuOutputStorage()
      : reconstructed(kOriginalExtent),
        quant_field(kBlockStride * kBlockExtent.height, kPoison),
        block_distance(kBlockStride * kBlockExtent.height, kPoison) {}

  [[nodiscard]] gjxl::AdaptiveQuantizationOutput Output() {
    return {
      .quant_field = {quant_field.data(), kBlockExtent, kBlockStride},
      .block_distance_map = {
        block_distance.data(), kBlockExtent, kBlockStride},
      .reconstructed_linear_rgb = reconstructed.View(),
      .frame = &frame,
      .score_history = &score_history,
      .maximum_error_result = &maximum_error,
    };
  }

  [[nodiscard]] bool PaddingPoisoned() const {
    for (size_t y = 0; y < kBlockExtent.height; ++y) {
      for (size_t x = kBlockExtent.width; x < kBlockStride; ++x) {
        if (std::bit_cast<uint32_t>(
              quant_field[y * kBlockStride + x]) != kPoisonBits ||
            std::bit_cast<uint32_t>(
              block_distance[y * kBlockStride + x]) != kPoisonBits) {
          return false;
        }
      }
    }
    for (const std::vector<float>& plane : reconstructed.plane) {
      for (size_t y = 0; y < reconstructed.extent.height; ++y) {
        for (size_t x = reconstructed.extent.width;
             x < reconstructed.stride; ++x) {
          if (plane[y * reconstructed.stride + x] != -777.0f) {
            return false;
          }
        }
      }
    }
    return true;
  }

  [[nodiscard]] bool Poisoned() const {
    return std::ranges::all_of(quant_field, [](float value) {
             return std::bit_cast<uint32_t>(value) == kPoisonBits;
           }) &&
           std::ranges::all_of(block_distance, [](float value) {
             return std::bit_cast<uint32_t>(value) == kPoisonBits;
           }) &&
           std::ranges::all_of(
             reconstructed.plane,
             [](const std::vector<float>& plane) {
               return std::ranges::all_of(
                 plane, [](float value) { return value == -777.0f; });
             }) &&
           !frame.valid() && score_history.empty();
  }

  ImageStorage reconstructed;
  std::vector<float> quant_field;
  std::vector<float> block_distance;
  gjxl::VarDctEncoderFrame frame;
  std::vector<double> score_history;
  gjxl::MaximumErrorResult maximum_error;
};

struct GpuOutputStorage {
  static constexpr size_t kBlockStride = kBlockExtent.width + 4;
  std::vector<float> quant_field =
    std::vector<float>(kBlockStride * kBlockExtent.height, kPoison);
  std::vector<float> block_distance =
    std::vector<float>(kBlockStride * kBlockExtent.height, kPoison);
  std::vector<double> score_history;

  [[nodiscard]] gjxl::GpuAdaptiveQuantizationPolicyOutput Output() {
    return {
      .quant_field = {quant_field.data(), kBlockExtent, kBlockStride},
      .block_distance_map = {
        block_distance.data(), kBlockExtent, kBlockStride},
      .score_history = &score_history,
    };
  }

  [[nodiscard]] bool PaddingPoisoned() const {
    for (size_t y = 0; y < kBlockExtent.height; ++y) {
      for (size_t x = kBlockExtent.width; x < kBlockStride; ++x) {
        if (std::bit_cast<uint32_t>(
              quant_field[y * kBlockStride + x]) != kPoisonBits ||
            std::bit_cast<uint32_t>(
              block_distance[y * kBlockStride + x]) != kPoisonBits) {
          return false;
        }
      }
    }
    return true;
  }

  [[nodiscard]] bool Poisoned() const {
    return std::ranges::all_of(quant_field, [](float value) {
             return std::bit_cast<uint32_t>(value) == kPoisonBits;
           }) &&
           std::ranges::all_of(block_distance, [](float value) {
             return std::bit_cast<uint32_t>(value) == kPoisonBits;
           }) &&
           score_history.empty();
  }
};

gjxl::AdaptiveQuantizationOptions MakeOptions(bool non_default,
                                               size_t iterations) {
  gjxl::AdaptiveQuantizationOptions options;
  options.butteraugli_target = 1.1f;
  options.iterations = iterations;
  if (non_default) {
    options.fast_color_correlation = false;
    options.profile.x_qm_scale = 3;
    options.profile.b_qm_scale = 1;
    options.profile.loop_filter.gaborish = true;
    options.profile.loop_filter.gaborish_options.weight1 =
      {0.071f, 0.093f, 0.057f};
    options.profile.loop_filter.gaborish_options.weight2 =
      {0.039f, 0.027f, 0.045f};
    options.profile.loop_filter.epf_options.iterations = 3;
    options.profile.loop_filter.epf_options.channel_scale =
      {31.0f, 7.0f, 4.25f};
    options.profile.loop_filter.epf_options.pass0_sigma_scale = 1.17f;
    options.profile.loop_filter.epf_options.pass2_sigma_scale = 4.75f;
    options.profile.loop_filter.epf_options.border_sad_multiplier = 0.81f;
    options.profile.intensity_target = 183.0f;
    options.butteraugli = {0.91f, 1.07f, 80.0f};
  }
  return options;
}

bool MakeMixedStrategies(gjxl::AcStrategyGrid* strategies) {
  if (!CheckStatus(gjxl::AcStrategyGrid::Create(kBlockExtent, strategies),
                   "mixed policy strategy creation") ||
      !CheckStatus(strategies->Set(
        0, 0, gjxl::AcStrategyType::kDct32x32), "policy DCT32x32") ||
      !CheckStatus(strategies->Set(
        4, 0, gjxl::AcStrategyType::kDct16x32), "policy DCT16x32") ||
      !CheckStatus(strategies->Set(
        8, 0, gjxl::AcStrategyType::kDct32x16), "policy DCT32x16") ||
      !CheckStatus(strategies->Set(
        10, 0, gjxl::AcStrategyType::kDct16x16), "policy DCT16x16") ||
      !CheckStatus(strategies->Set(
        10, 2, gjxl::AcStrategyType::kDct16x8), "policy DCT16x8") ||
      !CheckStatus(strategies->Set(
        10, 4, gjxl::AcStrategyType::kDct8x16), "policy DCT8x16")) {
    return false;
  }
  strategies->fill_empty_dct8();
  return strategies->complete();
}

template <typename T>
bool PlanesEqual(gjxl::PlaneView<const T> left,
                 gjxl::PlaneView<const T> right) {
  if (left.extent != right.extent) return false;
  for (size_t y = 0; y < left.extent.height; ++y) {
    if (!std::equal(
          left.Row(y), left.Row(y) + left.extent.width, right.Row(y))) {
      return false;
    }
  }
  return true;
}

bool GridsEqual(const gjxl::AcStrategyGrid& left,
                const gjxl::AcStrategyGrid& right) {
  if (left.extent() != right.extent()) return false;
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

bool FramesEqual(const gjxl::VarDctEncoderFrame& left,
                 const gjxl::VarDctEncoderFrame& right) {
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
  const auto& left_cfl = left.color_correlation();
  const auto& right_cfl = right.color_correlation();
  if (!PlanesEqual(left_cfl.y_to_x_map(), right_cfl.y_to_x_map()) ||
      !PlanesEqual(left_cfl.y_to_b_map(), right_cfl.y_to_b_map())) {
    return false;
  }
  const gjxl::ConstImage3I32View left_dc = left.quantized_dc();
  const gjxl::ConstImage3I32View right_dc = right.quantized_dc();
  const gjxl::ConstImage3FView left_reconstructed_dc = left.dc();
  const gjxl::ConstImage3FView right_reconstructed_dc = right.dc();
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!PlanesEqual(left_dc.plane[channel], right_dc.plane[channel]) ||
        !PlanesEqual(
          left_reconstructed_dc.plane[channel],
          right_reconstructed_dc.plane[channel])) {
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
      if (!std::equal(
            left_group.coefficients[channel].begin(),
            left_group.coefficients[channel].end(),
            right_group.coefficients[channel].begin())) {
        return false;
      }
    }
  }
  return true;
}

bool ComparePolicyResult(const CpuOutputStorage& cpu,
                         const GpuOutputStorage& gpu,
                         gjxl::AdaptiveQuantizationOptions options) {
  if (cpu.score_history.size() != gpu.score_history.size() ||
      cpu.score_history.size() != options.iterations + 1 ||
      !cpu.frame.valid() || !gpu.PaddingPoisoned()) {
    std::cerr << "CPU/GPU AQ policy result shape mismatch\n";
    return false;
  }
  bool within_tolerance = true;
  for (size_t index = 0; index < cpu.score_history.size(); ++index) {
    const double error =
      std::abs(cpu.score_history[index] - gpu.score_history[index]);
    g_max_score_error = std::max(g_max_score_error, error);
    if (error > kTolerance) {
      std::cerr << "CPU/GPU score mismatch at " << index
                << ": CPU " << cpu.score_history[index] << ", GPU "
                << gpu.score_history[index] << ", error " << error << '\n';
      within_tolerance = false;
    }
  }
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      const float cpu_quant =
        cpu.quant_field[y * CpuOutputStorage::kBlockStride + x];
      const float gpu_quant =
        gpu.quant_field[y * GpuOutputStorage::kBlockStride + x];
      const float cpu_block =
        cpu.block_distance[y * CpuOutputStorage::kBlockStride + x];
      const float gpu_block =
        gpu.block_distance[y * GpuOutputStorage::kBlockStride + x];
      const double quant_error = std::abs(cpu_quant - gpu_quant);
      const double block_error = std::abs(cpu_block - gpu_block);
      g_max_quant_error = std::max(g_max_quant_error, quant_error);
      g_max_block_error = std::max(g_max_block_error, block_error);
      if (quant_error > kTolerance || block_error > kTolerance) {
        std::cerr << "CPU/GPU bounded policy mismatch at " << x << ',' << y
                  << ": quant error " << quant_error << ", block error "
                  << block_error << '\n';
        within_tolerance = false;
      }
    }
  }

  if (!within_tolerance) return false;

  float quant_dc = 0.0f;
  if (!CheckStatus(gjxl::ComputeInitialQuantDc(
        options.butteraugli_target, &quant_dc), "policy quant DC")) {
    return false;
  }
  std::array<int32_t, kBlockExtent.width * kBlockExtent.height> raw{};
  gjxl::Quantizer quantizer;
  if (!CheckStatus(gjxl::CreateQuantizerFromField(
        quant_dc,
        {gpu.quant_field.data(), kBlockExtent,
         GpuOutputStorage::kBlockStride},
        {raw.data(), kBlockExtent, kBlockExtent.width},
        &quantizer), "GPU final raw-quant reconstruction")) {
    return false;
  }
  if (quantizer.params().global_scale !=
        cpu.frame.quantizer().params().global_scale ||
      quantizer.params().quant_dc !=
        cpu.frame.quantizer().params().quant_dc) {
    std::cerr << "CPU/GPU final quantizer differs\n";
    return false;
  }
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      gjxl::AcStrategyCell cell;
      if (!cpu.frame.strategies().Get(x, y, &cell).ok()) return false;
      const int32_t frame_raw = cpu.frame.raw_quant_field().Row(y)[x];
      if (frame_raw < 1 || frame_raw > gjxl::kMaxRawQuant ||
          (!cell.is_anchor &&
           raw[y * kBlockExtent.width + x] != frame_raw)) {
        std::cerr << "CPU final adjusted raw quant is invalid at "
                  << x << ',' << y << '\n';
        return false;
      }
    }
  }
  return true;
}

bool CompareFullResult(const CpuOutputStorage& cpu,
                       const GpuOutputStorage& bounded,
                       const CpuOutputStorage& full) {
  if (!full.frame.valid() || !full.PaddingPoisoned() ||
      bounded.score_history != full.score_history ||
      !FramesEqual(cpu.frame, full.frame)) {
    std::cerr << "GPU full AQ output shape or bounded result differs\n";
    return false;
  }
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      if (bounded.quant_field[y * GpuOutputStorage::kBlockStride + x] !=
            full.quant_field[y * CpuOutputStorage::kBlockStride + x] ||
          bounded.block_distance[y * GpuOutputStorage::kBlockStride + x] !=
            full.block_distance[y * CpuOutputStorage::kBlockStride + x]) {
        std::cerr << "GPU bounded and full AQ fields differ\n";
        return false;
      }
    }
  }

  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kOriginalExtent.height; ++y) {
      for (size_t x = 0; x < kOriginalExtent.width; ++x) {
        const double error = std::abs(
          static_cast<double>(full.reconstructed.plane[channel][
            y * full.reconstructed.stride + x]) -
          cpu.reconstructed.plane[channel][
            y * cpu.reconstructed.stride + x]);
        g_max_image_error = std::max(g_max_image_error, error);
        if (error > kTolerance) {
          std::cerr << "CPU/GPU reconstructed RGB mismatch at " << x << ','
                    << y << " channel " << channel << ": " << error << '\n';
          return false;
        }
      }
    }
  }

  if (cpu.frame.profile().intensity_target == 255.0f &&
      cpu.frame.profile().x_qm_scale == 2 &&
      cpu.frame.profile().b_qm_scale == 2) {
    std::vector<uint8_t> cpu_codestream;
    std::vector<uint8_t> gpu_codestream;
    if (!CheckStatus(
          gjxl::EncodeVarDctCodestream(cpu.frame, &cpu_codestream),
          "CPU full AQ codestream") ||
        !CheckStatus(
          gjxl::EncodeVarDctCodestream(full.frame, &gpu_codestream),
          "GPU full AQ codestream") ||
        cpu_codestream != gpu_codestream) {
      std::cerr << "CPU/GPU full AQ frame codestream differs\n";
      return false;
    }
  }
  return true;
}

bool CheckCase(gjxl::GpuBackend& gpu, bool non_default, size_t iterations,
               gjxl::ConstImage3FView original,
               gjxl::ConstImage3FView opsin,
               const gjxl::AcStrategyGrid& strategies,
               gjxl::ConstPlaneF32View initial,
               gjxl::ConstPlaneU8View sharpness) {
  const gjxl::AdaptiveQuantizationOptions options =
    MakeOptions(non_default, iterations);
  CpuOutputStorage cpu;
  GpuOutputStorage bounded;
  if (!CheckStatus(gjxl::FindBestQuantization(
        original, opsin, strategies, initial, sharpness, options,
        cpu.Output()), "CPU AQ policy")) {
    return false;
  }
  const gjxl::GpuBackendStats before = gpu.stats();
  if (!CheckStatus(gjxl::RunGpuAdaptiveQuantizationPolicy(
        gpu, original, opsin, strategies, initial, sharpness, options,
        bounded.Output()), "GPU AQ policy")) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu.stats();
  // All three arenas may now be leased; cold preparation allocates at most three.
  const uint64_t policy_allocations =
      after.successful_allocations - before.successful_allocations;
  if (policy_allocations > 3 ||
      after.committed_submissions !=
        before.committed_submissions + iterations + 2) {
    std::cerr << "GPU AQ policy preparation/evaluation resource count differs\n";
    return false;
  }
  if (!ComparePolicyResult(cpu, bounded, options)) {
    return false;
  }

  CpuOutputStorage full;
  CpuOutputStorage rejected_full;
  gjxl::AdaptiveQuantizationOutput invalid_full = rejected_full.Output();
  for (gjxl::PlaneF32View& plane :
       invalid_full.reconstructed_linear_rgb.plane) {
    --plane.extent.width;
  }
  const gjxl::GpuBackendStats before_rejected = gpu.stats();
  if (!ExpectCode(gjxl::RunGpuAdaptiveQuantization(
        gpu, original, opsin, strategies, initial, sharpness, options,
        invalid_full), gjxl::StatusCode::kInvalidArgument,
        "invalid GPU full AQ output") ||
      !rejected_full.Poisoned() || gpu.stats().successful_allocations !=
        before_rejected.successful_allocations ||
      gpu.stats().committed_submissions !=
        before_rejected.committed_submissions) {
    return false;
  }
  const gjxl::GpuBackendStats before_full = gpu.stats();
  if (!CheckStatus(gjxl::RunGpuAdaptiveQuantization(
        gpu, original, opsin, strategies, initial, sharpness, options,
        full.Output()), "GPU full AQ")) {
    return false;
  }
  const gjxl::GpuBackendStats after_full = gpu.stats();
  const uint64_t full_allocations =
      after_full.successful_allocations -
      before_full.successful_allocations;
  if (full_allocations > 3 ||
      after_full.committed_submissions !=
        before_full.committed_submissions + iterations + 2) {
    std::cerr << "GPU final materialization added a submission or allocation\n";
    return false;
  }
  return CompareFullResult(cpu, bounded, full);
}

bool CheckFullyResidentCase(
    gjxl::GpuBackend& gpu, bool non_default, size_t iterations,
    gjxl::ConstImage3FView original, gjxl::ConstImage3FView opsin,
    const gjxl::AcStrategyGrid& strategies,
    gjxl::ConstPlaneF32View initial,
    gjxl::ConstPlaneU8View sharpness) {
  const gjxl::AdaptiveQuantizationOptions options =
      MakeOptions(non_default, iterations);
  constexpr auto mode =
      gjxl::GpuAdaptiveQuantizationMode::kFullyResident;
  GpuOutputStorage bounded;
  CpuOutputStorage full;
  const gjxl::GpuBackendStats before_bounded = gpu.stats();
  if (!CheckStatus(gjxl::RunGpuAdaptiveQuantizationPolicy(
          gpu, original, opsin, strategies, initial, sharpness, options, mode,
          bounded.Output()), "fully resident GPU AQ policy")) {
    return false;
  }
  const gjxl::GpuBackendStats after_bounded = gpu.stats();
  const uint64_t bounded_allocations =
      after_bounded.successful_allocations -
      before_bounded.successful_allocations;
  if (bounded_allocations > 3 ||
      after_bounded.committed_submissions !=
          before_bounded.committed_submissions + 3) {
    std::cerr << "Fully resident bounded resource count differs\n";
    return false;
  }
  const gjxl::GpuBackendStats before_full = gpu.stats();
  if (!CheckStatus(gjxl::RunGpuAdaptiveQuantization(
          gpu, original, opsin, strategies, initial, sharpness, options, mode,
          full.Output()), "fully resident GPU full AQ")) {
    return false;
  }
  const gjxl::GpuBackendStats after_full = gpu.stats();
  const uint64_t full_allocations =
      after_full.successful_allocations -
      before_full.successful_allocations;
  if (full_allocations > 3 ||
      after_full.committed_submissions !=
          before_full.committed_submissions + 3 ||
      !bounded.PaddingPoisoned() || !full.PaddingPoisoned() ||
      !full.frame.valid() ||
      bounded.score_history.size() != iterations + 1 ||
      bounded.score_history != full.score_history) {
    std::cerr << "Fully resident output or resource contract differs\n";
    return false;
  }
  for (double score : full.score_history) {
    if (!std::isfinite(score) || score < 0.0) {
      std::cerr << "Fully resident score is invalid\n";
      return false;
    }
  }
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      const float bounded_quant =
          bounded.quant_field[y * GpuOutputStorage::kBlockStride + x];
      const float bounded_block =
          bounded.block_distance[y * GpuOutputStorage::kBlockStride + x];
      if (bounded_quant !=
              full.quant_field[y * CpuOutputStorage::kBlockStride + x] ||
          bounded_block !=
              full.block_distance[y * CpuOutputStorage::kBlockStride + x] ||
          !std::isfinite(bounded_quant) || bounded_quant <= 0.0f ||
          !std::isfinite(bounded_block) || bounded_block < 0.0f) {
        std::cerr << "Fully resident bounded and full fields differ\n";
        return false;
      }
    }
  }
  for (const std::vector<float>& plane : full.reconstructed.plane) {
    for (size_t y = 0; y < full.reconstructed.extent.height; ++y) {
      for (size_t x = 0; x < full.reconstructed.extent.width; ++x) {
        if (!std::isfinite(
                plane[y * full.reconstructed.stride + x])) {
          std::cerr << "Fully resident reconstruction is invalid\n";
          return false;
        }
      }
    }
  }
  if (non_default) {
    return true;
  }
  std::vector<uint8_t> codestream;
  return CheckStatus(
      gjxl::EncodeVarDctCodestream(full.frame, &codestream),
      "fully resident frame codestream") && !codestream.empty();
}

bool CheckMaximumErrorCase(
    gjxl::GpuBackend& gpu,
    gjxl::ConstImage3FView original,
    gjxl::ConstImage3FView opsin,
    const gjxl::AcStrategyGrid& strategies,
    gjxl::ConstPlaneF32View initial,
    gjxl::ConstPlaneU8View sharpness,
    gjxl::GpuAdaptiveQuantizationMode mode) {
  gjxl::AdaptiveQuantizationOptions options = MakeOptions(false, 0);
  options.control_mode =
    gjxl::AdaptiveQuantizationControlMode::kMaximumError;
  options.maximum_error = {0.05f, 0.05f, 0.05f};

  CpuOutputStorage cpu;
  CpuOutputStorage gpu_output;
  if (!CheckStatus(gjxl::FindBestQuantization(
        original, opsin, strategies, initial, sharpness, options,
        cpu.Output()), "CPU maximum-error AQ") ||
      !CheckStatus(gjxl::RunGpuAdaptiveQuantization(
        gpu, original, opsin, strategies, initial, sharpness, options, mode,
        gpu_output.Output()), "Metal maximum-error AQ")) {
    return false;
  }
  if (!cpu.PaddingPoisoned() || !gpu_output.PaddingPoisoned() ||
      cpu.score_history.size() != 6 ||
      gpu_output.score_history.size() != 6 ||
      gpu_output.maximum_error.evaluation_count != 6 ||
      gpu_output.maximum_error.outcome ==
        gjxl::MaximumErrorOutcome::kNotApplicable) {
    std::cerr << "Maximum-error AQ result shape is invalid\n";
    return false;
  }

  if (mode == gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients) {
    const bool frame_equal = FramesEqual(cpu.frame, gpu_output.frame);
    if (!frame_equal ||
        cpu.maximum_error.evaluation_count !=
          gpu_output.maximum_error.evaluation_count ||
        cpu.maximum_error.outcome != gpu_output.maximum_error.outcome) {
      std::cerr << "Exact Metal maximum-error frame or outcome differs "
                   "from CPU\n";
      return false;
    }
    double maximum_difference = std::abs(
      static_cast<double>(cpu.maximum_error.normalized_maximum) -
      gpu_output.maximum_error.normalized_maximum);
    for (size_t channel = 0; channel < 3; ++channel) {
      maximum_difference = std::max(
        maximum_difference,
        std::abs(
          static_cast<double>(cpu.maximum_error.achieved[channel]) -
          gpu_output.maximum_error.achieved[channel]));
      for (size_t y = 0; y < kOriginalExtent.height; ++y) {
        for (size_t x = 0; x < kOriginalExtent.width; ++x) {
          maximum_difference = std::max(
            maximum_difference,
            std::abs(static_cast<double>(
              cpu.reconstructed.plane[channel][
                y * cpu.reconstructed.stride + x]) -
              gpu_output.reconstructed.plane[channel][
                y * gpu_output.reconstructed.stride + x]));
        }
      }
    }
    for (size_t index = 0; index < cpu.score_history.size(); ++index) {
      maximum_difference = std::max(
        maximum_difference,
        std::abs(cpu.score_history[index] -
                 gpu_output.score_history[index]));
    }
    for (size_t y = 0; y < kBlockExtent.height; ++y) {
      for (size_t x = 0; x < kBlockExtent.width; ++x) {
        maximum_difference = std::max(
          maximum_difference,
          std::abs(static_cast<double>(
            cpu.quant_field[y * CpuOutputStorage::kBlockStride + x]) -
            gpu_output.quant_field[
              y * CpuOutputStorage::kBlockStride + x]));
        maximum_difference = std::max(
          maximum_difference,
          std::abs(static_cast<double>(
            cpu.block_distance[y * CpuOutputStorage::kBlockStride + x]) -
            gpu_output.block_distance[
              y * CpuOutputStorage::kBlockStride + x]));
      }
    }
    if (maximum_difference > kTolerance) {
      std::cerr << "Exact Metal maximum-error diagnostics differ from CPU "
                << "by " << maximum_difference << '\n';
      return false;
    }
    std::vector<uint8_t> cpu_codestream;
    std::vector<uint8_t> gpu_codestream;
    if (!CheckStatus(gjxl::EncodeVarDctCodestream(
          cpu.frame, &cpu_codestream),
          "CPU maximum-error codestream") ||
        !CheckStatus(gjxl::EncodeVarDctCodestream(
          gpu_output.frame, &gpu_codestream),
          "Metal maximum-error codestream") ||
        cpu_codestream != gpu_codestream) {
      std::cerr << "Exact Metal maximum-error codestream differs from CPU\n";
      return false;
    }
  } else {
    for (double score : gpu_output.score_history) {
      if (!std::isfinite(score) || score < 0.0) {
        std::cerr << "Resident maximum-error score is invalid\n";
        return false;
      }
    }
  }
  return true;
}

bool CheckInvalidModeIsAtomic(
    gjxl::GpuBackend& gpu,
    gjxl::ConstImage3FView original,
    gjxl::ConstImage3FView opsin,
    const gjxl::AcStrategyGrid& strategies,
    gjxl::ConstPlaneF32View initial,
    gjxl::ConstPlaneU8View sharpness) {
  const auto invalid =
      static_cast<gjxl::GpuAdaptiveQuantizationMode>(99);
  const gjxl::AdaptiveQuantizationOptions options = MakeOptions(false, 0);
  GpuOutputStorage bounded;
  CpuOutputStorage full;
  const gjxl::GpuBackendStats before = gpu.stats();
  return ExpectCode(gjxl::RunGpuAdaptiveQuantizationPolicy(
          gpu, original, opsin, strategies, initial, sharpness, options,
          invalid, bounded.Output()), gjxl::StatusCode::kInvalidArgument,
          "invalid bounded GPU AQ mode") &&
      ExpectCode(gjxl::RunGpuAdaptiveQuantization(
          gpu, original, opsin, strategies, initial, sharpness, options,
          invalid, full.Output()), gjxl::StatusCode::kInvalidArgument,
          "invalid full GPU AQ mode") &&
      bounded.Poisoned() && full.Poisoned() &&
      gpu.stats().successful_allocations == before.successful_allocations &&
      gpu.stats().committed_submissions == before.committed_submissions;
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

bool CheckAtomicCapabilityFailure(
    gjxl::ConstImage3FView original,
    gjxl::ConstImage3FView opsin,
    const gjxl::AcStrategyGrid& strategies,
    gjxl::ConstPlaneF32View initial,
    gjxl::ConstPlaneU8View sharpness) {
  BackendWithoutAq backend;
  GpuOutputStorage output;
  CpuOutputStorage full_output;
  return ExpectCode(gjxl::RunGpuAdaptiveQuantizationPolicy(
      backend, original, opsin, strategies, initial, sharpness,
      MakeOptions(false, 0), output.Output()),
    gjxl::StatusCode::kUnavailable, "missing GPU AQ policy capability") &&
    output.Poisoned() &&
    ExpectCode(gjxl::RunGpuAdaptiveQuantization(
      backend, original, opsin, strategies, initial, sharpness,
      MakeOptions(false, 0), full_output.Output()),
      gjxl::StatusCode::kUnavailable, "missing GPU full AQ capability") &&
    full_output.Poisoned();
}

}  // namespace

int main() {
  ImageStorage original(kOriginalExtent);
  ImageStorage padded_linear(kPaddedExtent);
  ImageStorage opsin(kPaddedExtent);
  FillPaddedLinear(&padded_linear, &original);
  gjxl::AcStrategyGrid strategies;
  std::vector<float> initial_values(
    kBlockExtent.width * kBlockExtent.height);
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      initial_values[y * kBlockExtent.width + x] =
        0.41f + 0.01f * static_cast<float>((7 * x + 3 * y) % 9);
    }
  }
  std::vector<uint8_t> sharpness(
    kBlockExtent.width * kBlockExtent.height, 4);
  const gjxl::ConstPlaneF32View initial{
    initial_values.data(), kBlockExtent, kBlockExtent.width};
  const gjxl::ConstPlaneU8View sharpness_view{
    sharpness.data(), kBlockExtent, kBlockExtent.width};

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(gjxl::LinearRgbToOpsin(
        padded_linear.ConstView(), 255.0f, opsin.View()),
        "policy opsin conversion") ||
      !MakeMixedStrategies(&strategies) ||
      !CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "GPU AQ policy backend") ||
      !CheckAtomicCapabilityFailure(
        original.ConstView(), opsin.ConstView(), strategies, initial,
        sharpness_view) ||
      !CheckInvalidModeIsAtomic(
        *gpu, original.ConstView(), opsin.ConstView(), strategies, initial,
        sharpness_view)) {
    return EXIT_FAILURE;
  }

  for (bool non_default : {false, true}) {
    for (size_t iterations = 0; iterations <= 2; ++iterations) {
      if (!CheckCase(*gpu, non_default, iterations,
                     original.ConstView(), opsin.ConstView(), strategies,
                     initial, sharpness_view)) {
        return EXIT_FAILURE;
      }
    }
    for (size_t iterations = 0; iterations <= 4; ++iterations) {
      if (!CheckFullyResidentCase(
              *gpu, non_default, iterations, original.ConstView(),
              opsin.ConstView(), strategies, initial, sharpness_view)) {
        return EXIT_FAILURE;
      }
    }
  }
  if (!CheckMaximumErrorCase(
        *gpu, original.ConstView(), opsin.ConstView(), strategies, initial,
        sharpness_view,
        gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients) ||
      !CheckMaximumErrorCase(
        *gpu, original.ConstView(), opsin.ConstView(), strategies, initial,
        sharpness_view,
        gjxl::GpuAdaptiveQuantizationMode::kFullyResident)) {
    return EXIT_FAILURE;
  }

  std::cout << "GPU AQ policy parity passed; max score error "
            << g_max_score_error << ", quant-field error "
            << g_max_quant_error << ", block-map error "
            << g_max_block_error << ", reconstructed RGB error "
            << g_max_image_error
            << "; raw quant, frame, and supported codestream exact\n";
  return EXIT_SUCCESS;
}
