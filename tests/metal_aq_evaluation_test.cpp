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
#include <limits>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

#include "codec/butteraugli.h"
#include "codec/color_transform.h"
#include "codec/loop_filter.h"
#include "codec/maximum_error.h"
#include "codec/reconstruction.h"
#include "codec/vardct_frame.h"
#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_aq_evaluation_profile.h"
#include "gpu/metal/metal_aq_butteraugli_test.h"
#include "gpu/metal/metal_aq_postprocess_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/aq_evaluation.h"

namespace {

constexpr uint32_t kPoisonBits = 0x7fc12345u;
constexpr float kPoison = std::bit_cast<float>(kPoisonBits);
constexpr float kReductionTolerance = 2.0e-6f;
float g_max_reduction_error = 0.0f;
gjxl::AqEvaluationMemoryStats g_memory_stats;

struct MemoryObservation {
  std::string_view label;
  gjxl::Extent2D source;
  gjxl::Extent2D coding;
  gjxl::AqEvaluationMemoryStats stats;
};

std::vector<MemoryObservation> g_memory_observations;

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

struct HostImage {
  gjxl::Extent2D extent;
  size_t stride = 0;
  std::array<std::vector<float>, 3> plane;

  HostImage(gjxl::Extent2D image_extent, size_t row_stride,
            float fill = -777.0f)
      : extent(image_extent), stride(row_stride) {
    for (std::vector<float>& values : plane) {
      values.assign(stride * extent.height, fill);
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView View() const {
    return {{{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }

  [[nodiscard]] gjxl::Image3FView MutableView() {
    return {{{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }
};

void FillOriginal(HostImage* image) {
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      image->plane[0][y * image->stride + x] =
        static_cast<float>((11 * x + 3 * y) % 101) / 100.0f;
      image->plane[1][y * image->stride + x] =
        static_cast<float>((5 * x + 13 * y + 17) % 103) / 102.0f;
      image->plane[2][y * image->stride + x] =
        static_cast<float>((19 * x + 7 * y + 29) % 107) / 106.0f;
    }
  }
}

void FillOpsin(HostImage* image) {
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      const float wave =
        0.025f * std::sin(0.071f * static_cast<float>(3 * x + 5 * y));
      image->plane[0][y * image->stride + x] =
        wave + 0.002f * static_cast<float>((x + 7 * y) % 9);
      image->plane[1][y * image->stride + x] =
        0.19f + 0.0012f * static_cast<float>(x) -
        0.0007f * static_cast<float>(y);
      image->plane[2][y * image->stride + x] =
        0.15f + 0.0004f * static_cast<float>(x + 2 * y);
    }
  }
}

gjxl::AqEvaluationOptions MakeOptions() {
  gjxl::AqEvaluationOptions options;
  options.profile.x_qm_scale = 3;
  options.profile.b_qm_scale = 1;
  options.profile.loop_filter.gaborish = true;
  options.profile.loop_filter.epf_options.iterations = 3;
  options.profile.loop_filter.epf_options.channel_scale =
    {31.0f, 7.0f, 4.25f};
  options.profile.intensity_target = 183.0f;
  options.butteraugli = {0.91f, 1.07f, 80.0f};
  return options;
}

bool MakeMixedStrategies(gjxl::AcStrategyGrid* strategies) {
  constexpr gjxl::Extent2D blocks{12, 8};
  if (!CheckStatus(gjxl::AcStrategyGrid::Create(blocks, strategies),
                   "mixed strategy creation") ||
      !CheckStatus(strategies->Set(
        0, 0, gjxl::AcStrategyType::kDct32x32), "DCT32x32 placement") ||
      !CheckStatus(strategies->Set(
        4, 0, gjxl::AcStrategyType::kDct32x16), "DCT32x16 placement") ||
      !CheckStatus(strategies->Set(
        6, 0, gjxl::AcStrategyType::kDct16x32), "DCT16x32 placement") ||
      !CheckStatus(strategies->Set(
        10, 0, gjxl::AcStrategyType::kDct16x16), "DCT16x16 placement") ||
      !CheckStatus(strategies->Set(
        6, 2, gjxl::AcStrategyType::kDct16x8), "DCT16x8 placement") ||
      !CheckStatus(strategies->Set(
        7, 2, gjxl::AcStrategyType::kDct8x16), "DCT8x16 placement")) {
    return false;
  }
  strategies->fill_empty_dct8();
  return strategies->complete();
}

struct EvaluationInputStorage {
  gjxl::Extent2D blocks;
  gjxl::Extent2D tiles;
  size_t raw_stride = 0;
  size_t sigma_stride = 0;
  size_t color_stride = 0;
  std::vector<int32_t> raw_quant;
  std::vector<float> inverse_sigma;
  std::vector<int8_t> y_to_x;
  std::vector<int8_t> y_to_b;
  gjxl::QuantizerParams quantizer{1173, 43};

  explicit EvaluationInputStorage(gjxl::Extent2D coding_extent)
      : blocks{coding_extent.width / 8, coding_extent.height / 8},
        tiles{(coding_extent.width + 63) / 64,
              (coding_extent.height + 63) / 64},
        raw_stride(blocks.width + 3),
        sigma_stride(blocks.width + 5),
        color_stride(tiles.width + 2),
        raw_quant(raw_stride * blocks.height, -12345),
        inverse_sigma(sigma_stride * blocks.height, 12345.0f),
        y_to_x(color_stride * tiles.height, 99),
        y_to_b(color_stride * tiles.height, 99) {
    for (size_t y = 0; y < blocks.height; ++y) {
      for (size_t x = 0; x < blocks.width; ++x) {
        raw_quant[y * raw_stride + x] =
          1 + static_cast<int32_t>((17 * x + 11 * y) % 96);
        inverse_sigma[y * sigma_stride + x] =
          -0.045f - 0.002f * static_cast<float>((x + 3 * y) % 11);
      }
    }
    for (size_t y = 0; y < tiles.height; ++y) {
      for (size_t x = 0; x < tiles.width; ++x) {
        y_to_x[y * color_stride + x] =
          static_cast<int8_t>(-9 + 5 * x + 3 * y);
        y_to_b[y * color_stride + x] =
          static_cast<int8_t>(7 - 4 * x - 2 * y);
      }
    }
  }

  [[nodiscard]] gjxl::AqEvaluationInput View() const {
    return {
      .raw_quant_field = {raw_quant.data(), blocks, raw_stride},
      .quantizer = quantizer,
      .y_to_x = {y_to_x.data(), tiles, color_stride},
      .y_to_b = {y_to_b.data(), tiles, color_stride},
      .epf_inverse_sigma = {
        inverse_sigma.data(), blocks, sigma_stride},
    };
  }
};

struct EvaluationOutputStorage {
  gjxl::Extent2D blocks;
  size_t stride = 0;
  std::vector<float> map;
  double score = -987654.25;

  explicit EvaluationOutputStorage(gjxl::Extent2D block_extent)
      : blocks(block_extent),
        stride(block_extent.width + 7),
        map(stride * block_extent.height, kPoison) {}

  [[nodiscard]] gjxl::AqEvaluationOutput View() {
    return {{map.data(), blocks, stride}, &score};
  }

  [[nodiscard]] bool Poisoned() const {
    return std::ranges::all_of(map, [](float value) {
      return std::bit_cast<uint32_t>(value) == kPoisonBits;
    }) && score == -987654.25;
  }

  [[nodiscard]] bool ValidAndPadded() const {
    if (!std::isfinite(score) || score < 0.0) return false;
    for (size_t y = 0; y < blocks.height; ++y) {
      for (size_t x = 0; x < blocks.width; ++x) {
        const float value = map[y * stride + x];
        if (!std::isfinite(value) || value < 0.0f) return false;
      }
      for (size_t x = blocks.width; x < stride; ++x) {
        if (std::bit_cast<uint32_t>(map[y * stride + x]) != kPoisonBits) {
          return false;
        }
      }
    }
    return true;
  }
};

bool Prepare(gjxl::GpuBackend& gpu, const HostImage& original,
             const HostImage& coding, const gjxl::AcStrategyGrid& strategies,
             std::unique_ptr<gjxl::PreparedAqEvaluation>* prepared,
             gjxl::AqEvaluationOptions options = MakeOptions()) {
  const gjxl::Extent2D blocks = strategies.extent();
  const std::vector<uint8_t> sharpness(blocks.width * blocks.height, 4);
  return CheckStatus(gjxl::PrepareAqEvaluation(
    gpu,
    {
      .original_linear_rgb = original.View(),
      .coding_opsin = coding.View(),
      .strategies = &strategies,
      .epf_sharpness = {sharpness.data(), blocks, blocks.width},
      .options = options,
    },
    prepared), "Metal AQ preparation");
}

struct Fixture {
  HostImage original{{89, 57}, 96};
  HostImage coding{{96, 64}, 103};
  gjxl::AcStrategyGrid strategies;
  EvaluationInputStorage input{{96, 64}};

  bool Initialize() {
    FillOriginal(&original);
    FillOpsin(&coding);
    return MakeMixedStrategies(&strategies);
  }
};

bool CompareOutputs(const EvaluationOutputStorage& left,
                    const EvaluationOutputStorage& right) {
  if (!left.ValidAndPadded() || !right.ValidAndPadded() ||
      left.score != right.score) {
    return false;
  }
  for (size_t y = 0; y < left.blocks.height; ++y) {
    if (!std::equal(left.map.data() + y * left.stride,
                    left.map.data() + y * left.stride + left.blocks.width,
                    right.map.data() + y * right.stride)) {
      return false;
    }
  }
  return true;
}

bool QuantizedCoefficientsEqual(const gjxl::VarDctEncoderFrame& expected,
                                const gjxl::VarDctEncoderFrame& actual) {
  if (!expected.valid() || !actual.valid() ||
      expected.ac_group_count() != actual.ac_group_count()) {
    return false;
  }
  const gjxl::ConstImage3I32View expected_dc = expected.quantized_dc();
  const gjxl::ConstImage3I32View actual_dc = actual.quantized_dc();
  if (expected_dc.extent() != actual_dc.extent()) return false;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < expected_dc.extent().height; ++y) {
      if (!std::equal(expected_dc.plane[channel].Row(y),
                      expected_dc.plane[channel].Row(y) +
                          expected_dc.extent().width,
                      actual_dc.plane[channel].Row(y))) {
        return false;
      }
    }
  }
  for (size_t group_index = 0; group_index < expected.ac_group_count();
       ++group_index) {
    gjxl::VarDctAcGroupView expected_group;
    gjxl::VarDctAcGroupView actual_group;
    if (!expected.GetAcGroup(group_index, &expected_group).ok() ||
        !actual.GetAcGroup(group_index, &actual_group).ok() ||
        expected_group.used_coefficient_count !=
            actual_group.used_coefficient_count) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      if (!std::equal(expected_group.coefficients[channel].begin(),
                      expected_group.coefficients[channel].end(),
                      actual_group.coefficients[channel].begin())) {
        return false;
      }
    }
  }
  return true;
}

bool CheckReductionCase(gjxl::GpuBackend& gpu, gjxl::Extent2D source_extent,
                        gjxl::Extent2D coding_extent,
                        const gjxl::AcStrategyGrid& strategies,
                        std::string_view label) {
  const gjxl::Extent2D blocks = strategies.extent();
  HostImage original(source_extent, source_extent.width + 3);
  HostImage coding(coding_extent, coding_extent.width + 5);
  FillOriginal(&original);
  FillOpsin(&coding);
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Prepare(gpu, original, coding, strategies, &prepared)) return false;

  const size_t distance_stride = source_extent.width + 7;
  std::vector<float> distance(
    distance_stride * source_extent.height, kPoison);
  for (size_t y = 0; y < source_extent.height; ++y) {
    for (size_t x = 0; x < source_extent.width; ++x) {
      distance[y * distance_stride + x] =
        std::abs(0.01f + 1.7f * std::sin(
          0.037f * static_cast<float>(13 * x + 7 * y)));
    }
  }
  const gjxl::ConstPlaneF32View distance_view{
    distance.data(), source_extent, distance_stride};
  std::vector<float> expected(blocks.width * blocks.height);
  if (!CheckStatus(gjxl::ReduceButteraugliDistanceMap(
        distance_view, strategies,
        {expected.data(), blocks, blocks.width}), "CPU reduction oracle")) {
    return false;
  }

  const size_t output_stride = blocks.width + 4;
  std::vector<float> actual(output_stride * blocks.height, kPoison);
  const gjxl::GpuBackendStats before = gpu.stats();
  if (!CheckStatus(gjxl::metal_internal::RunMetalAqBlockReductionForTesting(
        *prepared, distance_view,
        {actual.data(), blocks, output_stride}), label)) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu.stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 1) {
    std::cerr << label << " allocated or submitted more than once\n";
    return false;
  }

  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      const float error = std::abs(
        actual[y * output_stride + x] - expected[y * blocks.width + x]);
      g_max_reduction_error = std::max(g_max_reduction_error, error);
      if (error > kReductionTolerance) {
        std::cerr << label << " mismatch at " << x << ',' << y
                  << ": error " << error << '\n';
        return false;
      }
    }
    for (size_t x = blocks.width; x < output_stride; ++x) {
      if (std::bit_cast<uint32_t>(actual[y * output_stride + x]) !=
          kPoisonBits) {
        std::cerr << label << " changed host padding\n";
        return false;
      }
    }
  }
  return true;
}

bool CheckReductionCorpus(gjxl::GpuBackend& gpu) {
  constexpr std::array<gjxl::AcStrategyType, 7> strategies = {
    gjxl::AcStrategyType::kDct8,
    gjxl::AcStrategyType::kDct16x16,
    gjxl::AcStrategyType::kDct32x32,
    gjxl::AcStrategyType::kDct16x8,
    gjxl::AcStrategyType::kDct8x16,
    gjxl::AcStrategyType::kDct32x16,
    gjxl::AcStrategyType::kDct16x32,
  };
  for (gjxl::AcStrategyType strategy : strategies) {
    const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
    gjxl::AcStrategyGrid grid;
    const gjxl::Extent2D blocks = info->covered_blocks;
    const gjxl::Extent2D coding = info->pixel_extent();
    if (!CheckStatus(gjxl::AcStrategyGrid::Create(blocks, &grid),
                     "isolated reduction grid") ||
        !CheckStatus(grid.Set(0, 0, strategy),
                     "isolated reduction strategy") ||
        !CheckReductionCase(gpu, coding, coding, grid, info->name)) {
      return false;
    }
  }

  gjxl::AcStrategyGrid mixed;
  return MakeMixedStrategies(&mixed) &&
    CheckReductionCase(
      gpu, {91, 57}, {96, 64}, mixed, "mixed partial-edge reduction");
}

bool CheckMaximumErrorReduction(gjxl::GpuBackend& gpu) {
  Fixture fixture;
  if (!fixture.Initialize()) return false;

  gjxl::AqEvaluationOptions options = MakeOptions();
  options.metric = gjxl::AqEvaluationMetric::kMaximumError;
  options.maximum_error = {0.004f, 0.007f, 0.01f};
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               &prepared, options)) {
    return false;
  }

  gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting snapshot;
  if (!CheckStatus(
        gjxl::metal_internal::
          RunMetalAqReconstructionAndPostprocessForTesting(
            *prepared, fixture.input.View(), &snapshot),
        "maximum-error oracle reconstruction")) {
    return false;
  }
  const gjxl::ConstImage3FView filtered{{{
    {snapshot.filtered_opsin[0].data(), snapshot.coding_extent,
     snapshot.coding_extent.width},
    {snapshot.filtered_opsin[1].data(), snapshot.coding_extent,
     snapshot.coding_extent.width},
    {snapshot.filtered_opsin[2].data(), snapshot.coding_extent,
     snapshot.coding_extent.width},
  }}};
  const gjxl::Extent2D blocks = fixture.strategies.extent();
  std::vector<float> expected_map(blocks.width * blocks.height);
  gjxl::MaximumErrorReduction expected;
  if (!CheckStatus(gjxl::ReduceMaximumError(
        fixture.coding.View(), filtered, fixture.original.extent,
        fixture.strategies, options.maximum_error,
        {expected_map.data(), blocks, blocks.width}, &expected),
        "CPU maximum-error reduction oracle")) {
    return false;
  }

  EvaluationOutputStorage actual(blocks);
  gjxl::MaximumErrorReduction actual_reduction{{-1.0f, -1.0f, -1.0f}, -1.0f};
  gjxl::AqEvaluationOutput output = actual.View();
  output.maximum_error = &actual_reduction;
  const gjxl::GpuBackendStats before = gpu.stats();
  if (!CheckStatus(prepared->Evaluate(fixture.input.View(), output),
                   "Metal maximum-error reduction") ||
      !actual.ValidAndPadded()) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu.stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 1) {
    std::cerr << "Maximum-error evaluation violated residency\n";
    return false;
  }

  float maximum_difference = 0.0f;
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      maximum_difference = std::max(
        maximum_difference,
        std::abs(actual.map[y * actual.stride + x] -
                 expected_map[y * blocks.width + x]));
    }
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    maximum_difference = std::max(
      maximum_difference,
      std::abs(actual_reduction.channel_maximum[channel] -
               expected.channel_maximum[channel]));
  }
  maximum_difference = std::max(
    maximum_difference,
    std::abs(actual_reduction.normalized_maximum -
             expected.normalized_maximum));
  if (maximum_difference > 1.0e-6f ||
      actual.score != actual_reduction.normalized_maximum) {
    std::cerr << "Metal maximum-error reduction differs from CPU oracle by "
              << maximum_difference << '\n';
    return false;
  }

  EvaluationOutputStorage rejected(blocks);
  const uint64_t submissions = gpu.stats().committed_submissions;
  if (!ExpectCode(
        prepared->Evaluate(fixture.input.View(), rejected.View()),
        gjxl::StatusCode::kInvalidArgument,
        "missing maximum-error result output") ||
      !rejected.Poisoned() ||
      gpu.stats().committed_submissions != submissions) {
    return false;
  }
  return true;
}

bool CheckProductionEvaluation(gjxl::GpuBackend& gpu) {
  Fixture fixture;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!fixture.Initialize() ||
      !Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               &prepared)) {
    return false;
  }
  g_memory_stats = prepared->memory_stats();
  const gjxl::GpuBackendStats before = gpu.stats();
  EvaluationOutputStorage output(fixture.strategies.extent());
  if (!CheckStatus(prepared->Evaluate(fixture.input.View(), output.View()),
                   "production AQ evaluation") ||
      !output.ValidAndPadded()) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu.stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 1) {
    std::cerr << "Production AQ evaluation violated residency\n";
    return false;
  }

  EvaluationOutputStorage profiled_output(fixture.strategies.extent());
  gjxl::metal_internal::MetalAqEvaluationProfile profile;
  const gjxl::GpuBackendStats before_profile = gpu.stats();
  if (!ExpectCode(
          gjxl::metal_internal::EvaluateMetalAqProfiled(
              *prepared, fixture.input.View(), profiled_output.View(),
              nullptr),
          gjxl::StatusCode::kInvalidArgument, "null AQ profile") ||
      !profiled_output.Poisoned() ||
      gpu.stats().committed_submissions !=
          before_profile.committed_submissions ||
      !CheckStatus(
          gjxl::metal_internal::EvaluateMetalAqProfiled(
              *prepared, fixture.input.View(), profiled_output.View(),
              &profile),
          "profiled AQ evaluation") ||
      !profiled_output.ValidAndPadded() ||
      profile.input_upload_bytes !=
          2 * fixture.input.blocks.width * fixture.input.blocks.height *
              sizeof(float) +
          2 * fixture.input.tiles.width * fixture.input.tiles.height *
              sizeof(int8_t) ||
      profile.input_upload_nanoseconds == 0 ||
      profile.submission_nanoseconds == 0 ||
      profile.completion_wait_nanoseconds == 0 ||
      profile.command_buffer_gpu_nanoseconds == 0 ||
      profile.bounded_readback_nanoseconds == 0 ||
      profile.output_commit_nanoseconds == 0 ||
      profile.final_readback_nanoseconds != 0 ||
      gpu.stats().successful_allocations !=
          before_profile.successful_allocations ||
      gpu.stats().committed_submissions !=
          before_profile.committed_submissions + 1) {
    std::cerr << "Profiled AQ evaluation did not preserve its contract\n";
    return false;
  }

  EvaluationOutputStorage rejected_final(fixture.strategies.extent());
  gjxl::VarDctEncoderFrame rejected_frame;
  gjxl::AqEvaluationOutput::Final invalid_final{
    .reconstructed_linear_rgb = {},
    .frame = &rejected_frame,
  };
  gjxl::AqEvaluationOutput invalid_final_output = rejected_final.View();
  invalid_final_output.final = &invalid_final;
  const uint64_t before_rejected_final =
    gpu.stats().committed_submissions;
  if (!ExpectCode(
        prepared->Evaluate(fixture.input.View(), invalid_final_output),
        gjxl::StatusCode::kInvalidArgument,
        "invalid final AQ output") ||
      !rejected_final.Poisoned() || rejected_frame.valid() ||
      gpu.stats().committed_submissions != before_rejected_final) {
    return false;
  }

  EvaluationOutputStorage final_bounded(fixture.strategies.extent());
  HostImage final_rgb(fixture.original.extent, fixture.original.extent.width + 5);
  gjxl::VarDctEncoderFrame final_frame;
  gjxl::AqEvaluationOutput::Final final_output{
    .reconstructed_linear_rgb = final_rgb.MutableView(),
    .frame = &final_frame,
  };
  gjxl::AqEvaluationOutput complete_output = final_bounded.View();
  complete_output.final = &final_output;
  const gjxl::GpuBackendStats before_final = gpu.stats();
  if (!CheckStatus(
        prepared->Evaluate(fixture.input.View(), complete_output),
        "production final AQ evaluation") ||
      !final_bounded.ValidAndPadded() || !final_frame.valid()) {
    return false;
  }
  const gjxl::GpuBackendStats after_final = gpu.stats();
  if (after_final.successful_allocations !=
        before_final.successful_allocations ||
      after_final.committed_submissions !=
        before_final.committed_submissions + 1) {
    std::cerr << "Final AQ materialization added an allocation or submission\n";
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < final_rgb.extent.height; ++y) {
      for (size_t x = 0; x < final_rgb.extent.width; ++x) {
        if (!std::isfinite(final_rgb.plane[channel][y * final_rgb.stride + x])) {
          std::cerr << "Final AQ RGB contains a non-finite pixel\n";
          return false;
        }
      }
      for (size_t x = final_rgb.extent.width; x < final_rgb.stride; ++x) {
        if (final_rgb.plane[channel][y * final_rgb.stride + x] != -777.0f) {
          std::cerr << "Final AQ RGB changed host padding\n";
          return false;
        }
      }
    }
  }

  EvaluationOutputStorage exact_linear_bounded(fixture.strategies.extent());
  HostImage exact_linear_rgb(
      fixture.original.extent, fixture.original.extent.width + 3);
  gjxl::VarDctEncoderFrame exact_linear_frame;
  gjxl::AqEvaluationOutput::Final exact_linear_final{
    .reconstructed_linear_rgb = exact_linear_rgb.MutableView(),
    .frame = &exact_linear_frame,
  };
  gjxl::AqEvaluationOutput exact_linear_output = exact_linear_bounded.View();
  exact_linear_output.final = &exact_linear_final;
  gjxl::AqEvaluationInput exact_linear_input = fixture.input.View();
  exact_linear_input.exact_coefficients = &final_frame;
  HostImage exact_cpu_reconstruction(
      fixture.coding.extent, fixture.coding.extent.width + 4);
  HostImage exact_cpu_cropped(
      fixture.original.extent, fixture.original.extent.width + 2);
  HostImage exact_cpu_filtered(
      fixture.original.extent, fixture.original.extent.width + 1);
  HostImage exact_cpu_linear(
      fixture.original.extent, fixture.original.extent.width + 4);
  if (!CheckStatus(gjxl::ReconstructQuantizedCoefficients(
                       final_frame, exact_cpu_reconstruction.MutableView()),
                   "exact-linear CPU reconstruction")) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < fixture.original.extent.height; ++y) {
      std::copy_n(
          exact_cpu_reconstruction.plane[channel].data() +
              y * exact_cpu_reconstruction.stride,
          fixture.original.extent.width,
          exact_cpu_cropped.plane[channel].data() +
              y * exact_cpu_cropped.stride);
    }
  }
  const gjxl::AqEvaluationOptions exact_options = MakeOptions();
  if (!CheckStatus(gjxl::ApplyLoopFilters(
                       exact_cpu_cropped.View(),
                       fixture.input.View().epf_inverse_sigma,
                       exact_options.profile.loop_filter,
                       exact_cpu_filtered.MutableView()),
                   "exact-linear CPU loop filters") ||
      !CheckStatus(gjxl::OpsinToLinearRgb(
                       exact_cpu_filtered.View(),
                       exact_options.profile.intensity_target,
                       exact_cpu_linear.MutableView()),
                   "exact-linear CPU color conversion")) {
    return false;
  }
  exact_linear_input.exact_reconstructed_linear_rgb = exact_cpu_linear.View();
  const gjxl::GpuBackendStats before_exact_linear = gpu.stats();
  if (!CheckStatus(prepared->Evaluate(exact_linear_input, exact_linear_output),
                   "exact-linear AQ evaluation") ||
      !exact_linear_bounded.ValidAndPadded() || !exact_linear_frame.valid()) {
    return false;
  }
  const gjxl::GpuBackendStats after_exact_linear = gpu.stats();
  if (after_exact_linear.successful_allocations !=
          before_exact_linear.successful_allocations ||
      after_exact_linear.committed_submissions !=
          before_exact_linear.committed_submissions + 1) {
    std::cerr << "Exact-linear AQ evaluation violated residency\n";
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < exact_linear_rgb.extent.height; ++y) {
      for (size_t x = 0; x < exact_linear_rgb.extent.width; ++x) {
        if (!std::isfinite(
                exact_linear_rgb.plane[channel][y * exact_linear_rgb.stride +
                                                x])) {
          std::cerr << "Exact-linear AQ RGB contains a non-finite pixel\n";
          return false;
        }
      }
      for (size_t x = exact_linear_rgb.extent.width;
           x < exact_linear_rgb.stride; ++x) {
        if (exact_linear_rgb.plane[channel][y * exact_linear_rgb.stride + x] !=
            -777.0f) {
          std::cerr << "Exact-linear AQ RGB changed host padding\n";
          return false;
        }
      }
    }
  }

  EvaluationOutputStorage exact_coeff_bounded(fixture.strategies.extent());
  HostImage exact_coeff_rgb(
      fixture.original.extent, fixture.original.extent.width + 4);
  gjxl::VarDctEncoderFrame exact_coeff_frame;
  gjxl::AqEvaluationOutput::Final exact_coeff_final{
    .reconstructed_linear_rgb = exact_coeff_rgb.MutableView(),
    .frame = &exact_coeff_frame,
  };
  gjxl::AqEvaluationOutput exact_coeff_output = exact_coeff_bounded.View();
  exact_coeff_output.final = &exact_coeff_final;
  gjxl::AqEvaluationInput exact_coeff_input = fixture.input.View();
  exact_coeff_input.exact_coefficients = &final_frame;
  const gjxl::GpuBackendStats before_exact_coeff = gpu.stats();
  if (!CheckStatus(prepared->Evaluate(exact_coeff_input, exact_coeff_output),
                   "exact-coefficient AQ evaluation") ||
      !exact_coeff_bounded.ValidAndPadded() ||
      !QuantizedCoefficientsEqual(final_frame, exact_coeff_frame)) {
    return false;
  }
  const gjxl::GpuBackendStats after_exact_coeff = gpu.stats();
  if (after_exact_coeff.successful_allocations !=
          before_exact_coeff.successful_allocations ||
      after_exact_coeff.committed_submissions !=
          before_exact_coeff.committed_submissions + 1) {
    std::cerr << "Exact-coefficient AQ evaluation violated residency\n";
    return false;
  }
  double exact_coeff_block_error = 0.0;
  double exact_coeff_rgb_error = 0.0;
  for (size_t y = 0; y < exact_coeff_bounded.blocks.height; ++y) {
    for (size_t x = 0; x < exact_coeff_bounded.blocks.width; ++x) {
      exact_coeff_block_error = std::max(
          exact_coeff_block_error,
          std::abs(static_cast<double>(
                   exact_coeff_bounded.map[
                           y * exact_coeff_bounded.stride + x]) -
                   exact_linear_bounded.map[
                       y * exact_linear_bounded.stride + x]));
    }
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < exact_coeff_rgb.extent.height; ++y) {
      for (size_t x = 0; x < exact_coeff_rgb.extent.width; ++x) {
        exact_coeff_rgb_error = std::max(
            exact_coeff_rgb_error,
            std::abs(static_cast<double>(
                         exact_coeff_rgb.plane[channel][
                             y * exact_coeff_rgb.stride + x]) -
                     exact_linear_rgb.plane[channel][
                         y * exact_linear_rgb.stride + x]));
      }
    }
  }
  if (exact_coeff_block_error > 2.0e-3 ||
      std::abs(exact_coeff_bounded.score - exact_linear_bounded.score) >
          2.0e-3 ||
      exact_coeff_rgb_error > 2.0e-3) {
    std::cerr << "Exact-coefficient AQ reconstruction exceeded tolerance: "
              << "block=" << exact_coeff_block_error
              << " score="
              << std::abs(exact_coeff_bounded.score -
                          exact_linear_bounded.score)
              << " RGB=" << exact_coeff_rgb_error << '\n';
    return false;
  }

  EvaluationOutputStorage rejected_exact(fixture.strategies.extent());
  gjxl::AqEvaluationInput orphan_exact = fixture.input.View();
  orphan_exact.exact_reconstructed_linear_rgb = exact_cpu_linear.View();
  const uint64_t before_rejected_exact = gpu.stats().committed_submissions;
  if (!ExpectCode(prepared->Evaluate(orphan_exact, rejected_exact.View()),
                  gjxl::StatusCode::kInvalidArgument,
                  "exact AQ image without coefficients") ||
      !rejected_exact.Poisoned() ||
      gpu.stats().committed_submissions != before_rejected_exact) {
    return false;
  }
  gjxl::AqEvaluationInput malformed_exact = exact_linear_input;
  malformed_exact.exact_reconstructed_linear_rgb.plane[1].stride =
      malformed_exact.exact_reconstructed_linear_rgb.width() - 1;
  if (!ExpectCode(prepared->Evaluate(malformed_exact, rejected_exact.View()),
                  gjxl::StatusCode::kInvalidArgument,
                  "malformed exact AQ image") ||
      !rejected_exact.Poisoned() ||
      gpu.stats().committed_submissions != before_rejected_exact) {
    return false;
  }
  HostImage wrong_exact_linear(
      {fixture.original.extent.width + 1, fixture.original.extent.height},
      fixture.original.extent.width + 3);
  FillOriginal(&wrong_exact_linear);
  gjxl::AqEvaluationInput mismatched_exact = exact_linear_input;
  mismatched_exact.exact_reconstructed_linear_rgb = wrong_exact_linear.View();
  if (!ExpectCode(prepared->Evaluate(mismatched_exact, rejected_exact.View()),
                  gjxl::StatusCode::kInvalidArgument,
                  "mismatched exact-linear geometry") ||
      !rejected_exact.Poisoned() ||
      gpu.stats().committed_submissions != before_rejected_exact) {
    return false;
  }

  gjxl::metal_internal::MetalAqButteraugliSnapshotForTesting staged;
  if (!CheckStatus(gjxl::metal_internal::RunMetalAqButteraugliForTesting(
        *prepared, fixture.input.View(), &staged),
        "staged AQ comparison after production") ||
      std::abs(staged.score - output.score) > 1.0e-6) {
    std::cerr << "Fused/staged AQ score mismatch: fused " << output.score
              << ", staged " << staged.score << '\n';
    return false;
  }

  EvaluationInputStorage invalid = fixture.input;
  invalid.raw_quant[0] = 0;
  EvaluationOutputStorage rejected(fixture.strategies.extent());
  const uint64_t submissions = gpu.stats().committed_submissions;
  if (!ExpectCode(prepared->Evaluate(invalid.View(), rejected.View()),
                  gjxl::StatusCode::kInvalidArgument,
                  "invalid production input") ||
      !rejected.Poisoned() ||
      gpu.stats().committed_submissions != submissions) {
    return false;
  }
  gjxl::AqEvaluationInput overflowing = fixture.input.View();
  overflowing.raw_quant_field.stride =
    std::numeric_limits<size_t>::max();
  if (!ExpectCode(prepared->Evaluate(overflowing, rejected.View()),
                  gjxl::StatusCode::kInvalidArgument,
                  "overflowing production input stride") ||
      !rejected.Poisoned() ||
      gpu.stats().committed_submissions != submissions) {
    return false;
  }
  gjxl::AqEvaluationOutput wrong_output = rejected.View();
  wrong_output.block_distance_map.extent.width -= 1;
  if (!ExpectCode(prepared->Evaluate(fixture.input.View(), wrong_output),
                  gjxl::StatusCode::kInvalidArgument,
                  "invalid production output") ||
      !rejected.Poisoned() ||
      gpu.stats().committed_submissions != submissions) {
    return false;
  }
  EvaluationOutputStorage reused(fixture.strategies.extent());
  return CheckStatus(prepared->Evaluate(fixture.input.View(), reused.View()),
                     "reuse after rejected production input") &&
         reused.ValidAndPadded();
}

bool CheckMemoryScaling(gjxl::GpuBackend& gpu) {
  constexpr std::array<MemoryObservation, 5> cases = {{
    {"128x96", {128, 96}, {128, 96}, {}},
    {"Flower", {510, 532}, {512, 536}, {}},
    {"480p", {854, 480}, {856, 480}, {}},
    {"720p", {1280, 720}, {1280, 720}, {}},
    {"1080p", {1920, 1080}, {1920, 1080}, {}},
  }};
  size_t previous_persistent = 0;
  size_t previous_staging = 0;
  size_t previous_peak = 0;
  for (const MemoryObservation& test_case : cases) {
    HostImage original(
      test_case.source, test_case.source.width, 0.0f);
    HostImage coding(
      test_case.coding, test_case.coding.width, 0.0f);
    const gjxl::Extent2D blocks{
      test_case.coding.width / 8, test_case.coding.height / 8};
    gjxl::AcStrategyGrid strategies;
    if (!CheckStatus(
          gjxl::AcStrategyGrid::Create(blocks, &strategies),
          "memory strategy grid")) {
      return false;
    }
    strategies.fill_dct8();
    std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
    const gjxl::GpuBackendStats before = gpu.stats();
    if (!Prepare(gpu, original, coding, strategies, &prepared)) {
      return false;
    }
    const gjxl::GpuBackendStats after = gpu.stats();
    const gjxl::AqEvaluationMemoryStats stats = prepared->memory_stats();
    if (after.successful_allocations != before.successful_allocations + 3 ||
        after.committed_submissions != before.committed_submissions + 1 ||
        stats.persistent_bytes <= previous_persistent ||
        stats.staging_bytes <= previous_staging ||
        stats.peak_scratch_bytes <= previous_peak) {
      std::cerr << "Prepared AQ memory accounting did not scale monotonically\n";
      return false;
    }
    previous_persistent = stats.persistent_bytes;
    previous_staging = stats.staging_bytes;
    previous_peak = stats.peak_scratch_bytes;
    g_memory_observations.push_back(
      {test_case.label, test_case.source, test_case.coding, stats});
  }
  return true;
}

bool CheckInvariantColorCorrelation(gjxl::GpuBackend& gpu) {
  Fixture fixture;
  if (!fixture.Initialize()) return false;
  const gjxl::Extent2D blocks = fixture.strategies.extent();
  const size_t block_count = blocks.width * blocks.height;
  const std::vector<uint8_t> sharpness(block_count, 4);
  std::vector<float> quant_field(block_count);
  for (size_t index = 0; index < block_count; ++index) {
    quant_field[index] = 0.7f +
        0.015f * static_cast<float>((13 * index) % 29);
  }
  const auto prepare_resident = [&](std::unique_ptr<
                                    gjxl::PreparedAqEvaluation>* output) {
    return CheckStatus(gjxl::PrepareAqEvaluation(
        gpu,
        {
          .original_linear_rgb = fixture.original.View(),
          .coding_opsin = fixture.coding.View(),
          .strategies = &fixture.strategies,
          .epf_sharpness = {sharpness.data(), blocks, blocks.width},
          .options = MakeOptions(),
          .resident_quantization = true,
          .coefficient_decision_mode =
              gjxl::AcCoefficientDecisionMode::kAdjustedSharedQuant,
        },
        output), "resident invariant-CfL preparation");
  };
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  std::unique_ptr<gjxl::PreparedAqEvaluation> baseline;
  if (!prepare_resident(&prepared) || !prepare_resident(&baseline)) {
    return false;
  }

  gjxl::AqEvaluationInput uploaded_input{
    .y_to_x = fixture.input.View().y_to_x,
    .y_to_b = fixture.input.View().y_to_b,
    .quant_field = {quant_field.data(), blocks, blocks.width},
    .quant_dc = 1.0f,
  };
  EvaluationOutputStorage expected(fixture.strategies.extent());
  if (!CheckStatus(baseline->Evaluate(uploaded_input, expected.View()),
                   "invariant-CfL baseline evaluation")) {
    return false;
  }

  const gjxl::GpuBackendStats before_invalid = gpu.stats();
  if (!ExpectCode(prepared->SetInvariantColorCorrelation(
                      fixture.input.View().y_to_x, {}),
                  gjxl::StatusCode::kInvalidArgument,
                  "partial invariant-CfL binding") ||
      gpu.stats().successful_allocations !=
          before_invalid.successful_allocations ||
      gpu.stats().committed_submissions !=
          before_invalid.committed_submissions) {
    return false;
  }

  const std::vector<int8_t> retained_x = fixture.input.y_to_x;
  const std::vector<int8_t> retained_b = fixture.input.y_to_b;
  const gjxl::AqEvaluationMemoryStats memory = prepared->memory_stats();
  const gjxl::GpuBackendStats before_binding = gpu.stats();
  if (!CheckStatus(prepared->SetInvariantColorCorrelation(
                       fixture.input.View().y_to_x,
                       fixture.input.View().y_to_b),
                   "invariant-CfL binding") ||
      prepared->memory_stats().persistent_bytes != memory.persistent_bytes ||
      prepared->memory_stats().staging_bytes != memory.staging_bytes ||
      prepared->memory_stats().peak_scratch_bytes !=
          memory.peak_scratch_bytes ||
      gpu.stats().successful_allocations !=
          before_binding.successful_allocations ||
      gpu.stats().committed_submissions !=
          before_binding.committed_submissions) {
    std::cerr << "Invariant CfL binding changed prepared residency\n";
    return false;
  }

  gjxl::AqEvaluationInput resident_input = uploaded_input;
  resident_input.y_to_x = {};
  resident_input.y_to_b = {};
  EvaluationOutputStorage rejected(fixture.strategies.extent());
  const uint64_t submissions = gpu.stats().committed_submissions;
  if (!ExpectCode(prepared->Evaluate(uploaded_input, rejected.View()),
                  gjxl::StatusCode::kInvalidArgument,
                  "host CfL after invariant binding") ||
      !rejected.Poisoned() ||
      gpu.stats().committed_submissions != submissions) {
    return false;
  }

  std::fill(fixture.input.y_to_x.begin(), fixture.input.y_to_x.end(), 97);
  std::fill(fixture.input.y_to_b.begin(), fixture.input.y_to_b.end(), -97);
  EvaluationOutputStorage first(fixture.strategies.extent());
  gjxl::metal_internal::MetalAqEvaluationProfile profile;
  if (!CheckStatus(gjxl::metal_internal::EvaluateMetalAqProfiled(
                       *prepared, resident_input, first.View(), &profile),
                   "profiled invariant-CfL evaluation") ||
      profile.input_upload_bytes !=
          fixture.input.blocks.width * fixture.input.blocks.height *
              sizeof(float) ||
      !CompareOutputs(first, expected)) {
    std::cerr << "Invariant CfL was reuploaded or changed the evaluation\n";
    return false;
  }

  if (!CheckStatus(prepared->Reconfigure(
                       fixture.strategies,
                       {sharpness.data(), blocks, blocks.width}),
                   "invariant-CfL strategy reconfiguration") ||
      !ExpectCode(prepared->Evaluate(resident_input, rejected.View()),
                  gjxl::StatusCode::kInvalidArgument,
                  "stale invariant CfL after reconfiguration")) {
    return false;
  }

  const gjxl::ConstPlaneI8View retained_x_view{
      retained_x.data(), fixture.input.tiles, fixture.input.color_stride};
  const gjxl::ConstPlaneI8View retained_b_view{
      retained_b.data(), fixture.input.tiles, fixture.input.color_stride};
  EvaluationOutputStorage rebound(fixture.strategies.extent());
  return CheckStatus(prepared->SetInvariantColorCorrelation(
                         retained_x_view, retained_b_view),
                     "rebound invariant CfL") &&
         CheckStatus(prepared->Evaluate(resident_input, rebound.View()),
                     "evaluation after invariant-CfL rebound") &&
         CompareOutputs(rebound, expected);
}

bool CheckReconfiguration(gjxl::GpuBackend& gpu) {
  Fixture fixture;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!fixture.Initialize() ||
      !Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               &prepared)) {
    return false;
  }

  gjxl::AcStrategyGrid dct8;
  const gjxl::Extent2D blocks = fixture.strategies.extent();
  if (!CheckStatus(gjxl::AcStrategyGrid::Create(blocks, &dct8),
                   "DCT8 reconfiguration grid")) {
    return false;
  }
  dct8.fill_empty_dct8();
  std::vector<uint8_t> sharpness(blocks.width * blocks.height);
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      sharpness[y * blocks.width + x] =
        static_cast<uint8_t>(2 + (x + 3 * y) % 4);
    }
  }

  const gjxl::AqEvaluationMemoryStats memory = prepared->memory_stats();
  const gjxl::GpuBackendStats before = gpu.stats();
  if (!CheckStatus(prepared->Reconfigure(
        dct8, {sharpness.data(), blocks, blocks.width}),
        "Metal AQ reconfiguration")) {
    return false;
  }
  const gjxl::GpuBackendStats after_reconfigure = gpu.stats();
  if (after_reconfigure.successful_allocations !=
        before.successful_allocations ||
      after_reconfigure.committed_submissions !=
        before.committed_submissions ||
      prepared->memory_stats().persistent_bytes != memory.persistent_bytes ||
      prepared->memory_stats().staging_bytes != memory.staging_bytes ||
      prepared->memory_stats().peak_scratch_bytes !=
        memory.peak_scratch_bytes) {
    std::cerr << "Metal AQ reconfiguration changed allocations or memory\n";
    return false;
  }

  EvaluationOutputStorage reconfigured(blocks);
  if (!CheckStatus(prepared->Evaluate(
        fixture.input.View(), reconfigured.View()),
        "reconfigured Metal AQ evaluation") ||
      !reconfigured.ValidAndPadded() ||
      gpu.stats().successful_allocations != before.successful_allocations ||
      gpu.stats().committed_submissions != before.committed_submissions + 1) {
    std::cerr << "Reconfigured Metal AQ evaluation violated residency\n";
    return false;
  }

  std::unique_ptr<gjxl::PreparedAqEvaluation> fresh;
  if (!CheckStatus(gjxl::PrepareAqEvaluation(
        gpu,
        {
          .original_linear_rgb = fixture.original.View(),
          .coding_opsin = fixture.coding.View(),
          .strategies = &dct8,
          .epf_sharpness = {sharpness.data(), blocks, blocks.width},
          .options = MakeOptions(),
        },
        &fresh), "fresh DCT8 AQ preparation")) {
    return false;
  }
  EvaluationOutputStorage expected(blocks);
  if (!CheckStatus(fresh->Evaluate(fixture.input.View(), expected.View()),
                   "fresh DCT8 AQ evaluation") ||
      !CompareOutputs(reconfigured, expected)) {
    std::cerr << "Reconfigured AQ output differs from fresh preparation\n";
    return false;
  }

  sharpness[0] = 8;
  const gjxl::GpuBackendStats before_invalid = gpu.stats();
  if (!ExpectCode(prepared->Reconfigure(
        dct8, {sharpness.data(), blocks, blocks.width}),
        gjxl::StatusCode::kInvalidArgument,
        "invalid AQ reconfiguration") ||
      gpu.stats().successful_allocations !=
        before_invalid.successful_allocations ||
      gpu.stats().committed_submissions !=
        before_invalid.committed_submissions) {
    return false;
  }
  EvaluationOutputStorage retained(blocks);
  if (!CheckStatus(prepared->Evaluate(fixture.input.View(), retained.View()),
                   "AQ evaluation after invalid reconfiguration") ||
      !CompareOutputs(reconfigured, retained)) {
    std::cerr << "Invalid AQ reconfiguration changed the prepared state\n";
    return false;
  }
  return true;
}

bool CheckSplitSeamAndDestruction(gjxl::GpuBackend& gpu) {
  Fixture fixture;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!fixture.Initialize() ||
      !Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               &prepared) ||
      !CheckStatus(gjxl::metal_internal::SubmitMetalAqEvaluationForTesting(
        *prepared, fixture.input.View()), "split production submit")) {
    return false;
  }
  EvaluationOutputStorage reentrant(fixture.strategies.extent());
  if (!ExpectCode(prepared->Evaluate(fixture.input.View(), reentrant.View()),
                  gjxl::StatusCode::kFailedPrecondition,
                  "same-object AQ reentry") ||
      !reentrant.Poisoned()) {
    return false;
  }
  gjxl::AqEvaluationOutput invalid_output{
    .block_distance_map = {}, .score = &reentrant.score};
  if (!ExpectCode(gjxl::metal_internal::FinishMetalAqEvaluationForTesting(
        *prepared, invalid_output), gjxl::StatusCode::kInvalidArgument,
        "invalid split finish") ||
      !CheckStatus(gjxl::metal_internal::FinishMetalAqEvaluationForTesting(
        *prepared, reentrant.View()), "valid split finish") ||
      !reentrant.ValidAndPadded()) {
    return false;
  }

  if (!Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               &prepared)) {
    return false;
  }
  bool waited = false;
  if (!CheckStatus(gjxl::metal_internal::SetMetalAqWaitObserverForTesting(
        *prepared, &waited), "AQ destruction wait observer") ||
      !CheckStatus(gjxl::metal_internal::SubmitMetalAqEvaluationForTesting(
        *prepared, fixture.input.View()), "destructor production submit")) {
    return false;
  }
  prepared.reset();
  if (!waited) {
    std::cerr << "Prepared AQ destructor did not wait for outstanding work\n";
    return false;
  }
  return true;
}

bool CheckFailure(gjxl::StatusCode expected, bool submission,
                  bool completion, bool readback) {
  Fixture fixture;
  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!fixture.Initialize() ||
      !CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "failure backend") ||
      !Prepare(*gpu, fixture.original, fixture.coding, fixture.strategies,
               &prepared) ||
      !CheckStatus(gjxl::ArmNextMetalSubmissionFailureForTest(
        *gpu, submission, completion), "AQ failure injection") ||
      (readback &&
       !CheckStatus(gjxl::metal_internal::FailNextMetalAqReadbackForTesting(
         *prepared), "AQ readback injection"))) {
    return false;
  }
  EvaluationOutputStorage output(fixture.strategies.extent());
  if (!ExpectCode(prepared->Evaluate(fixture.input.View(), output.View()),
                  expected, "injected production failure") ||
      !output.Poisoned() ||
      !ExpectCode(prepared->Evaluate(fixture.input.View(), output.View()),
                  gjxl::StatusCode::kFailedPrecondition,
                  "reuse after production failure") ||
      !output.Poisoned()) {
    return false;
  }
  return true;
}

bool CheckFinalReadbackFailure() {
  Fixture fixture;
  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!fixture.Initialize() ||
      !CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "final failure backend") ||
      !Prepare(*gpu, fixture.original, fixture.coding, fixture.strategies,
               &prepared) ||
      !CheckStatus(gjxl::metal_internal::FailNextMetalAqReadbackForTesting(
        *prepared), "final readback injection")) {
    return false;
  }
  EvaluationOutputStorage bounded(fixture.strategies.extent());
  HostImage reconstructed(
    fixture.original.extent, fixture.original.extent.width + 3);
  gjxl::VarDctEncoderFrame frame;
  gjxl::AqEvaluationOutput::Final final{
    .reconstructed_linear_rgb = reconstructed.MutableView(),
    .frame = &frame,
  };
  gjxl::AqEvaluationOutput output = bounded.View();
  output.final = &final;
  if (!ExpectCode(prepared->Evaluate(fixture.input.View(), output),
                  gjxl::StatusCode::kDeviceError,
                  "final readback failure") ||
      !bounded.Poisoned() || frame.valid() ||
      !std::ranges::all_of(
        reconstructed.plane,
        [](const std::vector<float>& plane) {
          return std::ranges::all_of(
            plane, [](float value) { return value == -777.0f; });
        }) ||
      !ExpectCode(prepared->Evaluate(fixture.input.View(), output),
                  gjxl::StatusCode::kFailedPrecondition,
                  "reuse after final readback failure")) {
    return false;
  }
  return true;
}

bool CheckUploadOrNumericFailure(bool upload) {
  Fixture fixture;
  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!fixture.Initialize() ||
      !CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "operational-boundary backend") ||
      !Prepare(*gpu, fixture.original, fixture.coding, fixture.strategies,
               &prepared)) {
    return false;
  }
  const gjxl::Status injection = upload
    ? gjxl::metal_internal::FailNextMetalAqUploadForTesting(*prepared)
    : gjxl::metal_internal::FailNextMetalAqNumericForTesting(*prepared);
  if (!CheckStatus(injection, upload ? "AQ upload injection"
                                    : "AQ numeric injection")) {
    return false;
  }
  const gjxl::GpuBackendStats before = gpu->stats();
  EvaluationOutputStorage output(fixture.strategies.extent());
  if (!ExpectCode(prepared->Evaluate(fixture.input.View(), output.View()),
                  gjxl::StatusCode::kDeviceError,
                  upload ? "injected production upload failure"
                         : "injected production numeric failure") ||
      !output.Poisoned()) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu->stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions !=
        before.committed_submissions + (upload ? 0 : 1) ||
      !ExpectCode(prepared->Evaluate(fixture.input.View(), output.View()),
                  gjxl::StatusCode::kFailedPrecondition,
                  "reuse after operational-boundary failure") ||
      !output.Poisoned()) {
    return false;
  }
  return true;
}

bool CheckIndependentConcurrency(gjxl::GpuBackend& gpu) {
  Fixture fixture;
  std::unique_ptr<gjxl::PreparedAqEvaluation> first;
  std::unique_ptr<gjxl::PreparedAqEvaluation> second;
  if (!fixture.Initialize() ||
      !Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               &first) ||
      !Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               &second)) {
    return false;
  }
  EvaluationOutputStorage first_output(fixture.strategies.extent());
  EvaluationOutputStorage second_output(fixture.strategies.extent());
  gjxl::Status first_status;
  gjxl::Status second_status;
  const gjxl::GpuBackendStats before = gpu.stats();
  std::thread first_thread([&] {
    first_status = first->Evaluate(fixture.input.View(), first_output.View());
  });
  std::thread second_thread([&] {
    second_status = second->Evaluate(fixture.input.View(), second_output.View());
  });
  first_thread.join();
  second_thread.join();
  const gjxl::GpuBackendStats after = gpu.stats();
  return CheckStatus(first_status, "first concurrent AQ evaluation") &&
         CheckStatus(second_status, "second concurrent AQ evaluation") &&
         after.successful_allocations == before.successful_allocations &&
         after.committed_submissions == before.committed_submissions + 2 &&
         CompareOutputs(first_output, second_output);
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

bool CheckCapabilityBoundary() {
  Fixture fixture;
  if (!fixture.Initialize()) return false;
  BackendWithoutAq backend;
  const gjxl::Extent2D blocks = fixture.strategies.extent();
  const std::vector<uint8_t> sharpness(blocks.width * blocks.height, 4);
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  return gjxl::QueryGpuAqEvaluation(backend) == nullptr &&
    ExpectCode(gjxl::PrepareAqEvaluation(
      backend,
      {
        .original_linear_rgb = fixture.original.View(),
        .coding_opsin = fixture.coding.View(),
        .strategies = &fixture.strategies,
        .epf_sharpness = {
          sharpness.data(), blocks, blocks.width},
        .options = MakeOptions(),
      },
      &prepared), gjxl::StatusCode::kUnavailable, "missing AQ capability") &&
    prepared == nullptr;
}

bool CheckInvalidCoefficientDecisionMode(gjxl::GpuBackend& gpu) {
  Fixture fixture;
  if (!fixture.Initialize()) return false;
  const gjxl::Extent2D blocks = fixture.strategies.extent();
  const std::vector<uint8_t> sharpness(blocks.width * blocks.height, 4);
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  return ExpectCode(gjxl::PrepareAqEvaluation(
      gpu,
      {
        .original_linear_rgb = fixture.original.View(),
        .coding_opsin = fixture.coding.View(),
        .strategies = &fixture.strategies,
        .epf_sharpness = {sharpness.data(), blocks, blocks.width},
        .options = MakeOptions(),
        .coefficient_decision_mode =
            static_cast<gjxl::AcCoefficientDecisionMode>(99),
      },
      &prepared), gjxl::StatusCode::kInvalidArgument,
      "invalid coefficient decision mode") &&
    prepared == nullptr;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "Metal AQ backend") ||
      !CheckCapabilityBoundary() ||
      !CheckInvalidCoefficientDecisionMode(*gpu) ||
      !CheckReductionCorpus(*gpu) ||
      !CheckMaximumErrorReduction(*gpu) ||
      !CheckProductionEvaluation(*gpu) ||
      !CheckInvariantColorCorrelation(*gpu) ||
      !CheckReconfiguration(*gpu) ||
      !CheckMemoryScaling(*gpu) ||
      !CheckSplitSeamAndDestruction(*gpu) ||
      !CheckUploadOrNumericFailure(true) ||
      !CheckFailure(gjxl::StatusCode::kSubmissionFailed, true, false, false) ||
      !CheckFailure(gjxl::StatusCode::kDeviceError, false, true, false) ||
      !CheckUploadOrNumericFailure(false) ||
      !CheckFailure(gjxl::StatusCode::kDeviceError, false, false, true) ||
      !CheckFinalReadbackFailure() ||
      !CheckIndependentConcurrency(*gpu)) {
    return EXIT_FAILURE;
  }
  std::cout << "Metal AQ Milestone 7 evaluation tests passed; max block "
            << "reduction error " << g_max_reduction_error
            << "; memory persistent " << g_memory_stats.persistent_bytes
            << ", staging " << g_memory_stats.staging_bytes
            << ", peak scratch " << g_memory_stats.peak_scratch_bytes
            << " bytes\n";
  for (const MemoryObservation& observation : g_memory_observations) {
    std::cout << "AQ memory " << observation.label << " ("
              << observation.source.width << 'x' << observation.source.height
              << " -> " << observation.coding.width << 'x'
              << observation.coding.height << "): persistent="
              << observation.stats.persistent_bytes << " staging="
              << observation.stats.staging_bytes << " peak="
              << observation.stats.peak_scratch_bytes << " bytes\n";
  }
  return EXIT_SUCCESS;
}
