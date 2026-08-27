// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/color_transform.h"
#include "codec/loop_filter.h"
#include "codec/reconstruction.h"
#include "core/ac_strategy.h"
#include "core/quantizer.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_aq_postprocess_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/aq_evaluation.h"

namespace {

constexpr double kFilterAbsolute = 2.0e-5;
constexpr double kFilterRelative = 2.0e-5;
constexpr double kColorAbsolute = 1.0e-4;
constexpr double kColorRelative = 5.0e-5;
constexpr double kCombinedAbsolute = 2.0e-4;
constexpr double kCombinedRelative = 1.0e-4;

double g_max_filter_error = 0.0;
double g_max_color_error = 0.0;
double g_max_combined_error = 0.0;

bool CheckStatus(gjxl::Status status, std::string_view operation) {
  if (status.ok())
    return true;
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

bool ExpectCode(gjxl::Status status, gjxl::StatusCode expected,
                std::string_view operation) {
  if (status.code() == expected)
    return true;
  std::cerr << operation << " returned " << static_cast<int>(status.code())
            << ", expected " << static_cast<int>(expected) << ": "
            << status.message() << '\n';
  return false;
}

bool Near(float actual, float expected, double absolute, double relative,
          double *maximum_error) {
  const double error =
      std::abs(static_cast<double>(actual) - static_cast<double>(expected));
  *maximum_error = std::max(*maximum_error, error);
  return std::isfinite(actual) &&
         error <= absolute + relative * std::abs(expected);
}

struct HostImage {
  gjxl::Extent2D extent;
  size_t stride = 0;
  std::array<std::vector<float>, 3> plane;

  HostImage(gjxl::Extent2D image_extent, size_t row_stride, float fill = 0.0f)
      : extent(image_extent), stride(row_stride) {
    for (std::vector<float> &values : plane) {
      values.assign(stride * extent.height, fill);
    }
  }

  gjxl::ConstImage3FView ConstView() const {
    return {{{
        {plane[0].data(), extent, stride},
        {plane[1].data(), extent, stride},
        {plane[2].data(), extent, stride},
    }}};
  }

  gjxl::Image3FView View() {
    return {{{
        {plane[0].data(), extent, stride},
        {plane[1].data(), extent, stride},
        {plane[2].data(), extent, stride},
    }}};
  }

  gjxl::ConstImage3FView Cropped(gjxl::Extent2D cropped) const {
    return {{{
        {plane[0].data(), cropped, stride},
        {plane[1].data(), cropped, stride},
        {plane[2].data(), cropped, stride},
    }}};
  }
};

void FillLinear(HostImage *image) {
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

void FillOpsin(HostImage *image, uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> noise(-0.012f, 0.012f);
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      const float wave =
          0.025f * std::sin(0.071f * static_cast<float>(3 * x + 5 * y));
      image->plane[0][y * image->stride + x] = wave + noise(generator);
      image->plane[1][y * image->stride + x] =
          0.19f + 0.0012f * static_cast<float>(x) -
          0.0007f * static_cast<float>(y) + noise(generator);
      image->plane[2][y * image->stride + x] =
          0.15f + 0.0004f * static_cast<float>(x + 2 * y) + noise(generator);
    }
  }
  const size_t impulse_x = image->extent.width / 3;
  const size_t impulse_y = image->extent.height / 2;
  image->plane[0][impulse_y * image->stride + impulse_x] += 0.31f;
  image->plane[1][impulse_y * image->stride + impulse_x] -= 0.17f;
  image->plane[2][impulse_y * image->stride + impulse_x] += 0.23f;
}

bool MakeDct8Strategies(gjxl::Extent2D coding_extent,
                        gjxl::AcStrategyGrid *strategies) {
  const gjxl::Extent2D blocks{coding_extent.width / 8,
                              coding_extent.height / 8};
  if (!CheckStatus(gjxl::AcStrategyGrid::Create(blocks, strategies),
                   "DCT8 strategy creation")) {
    return false;
  }
  strategies->fill_empty_dct8();
  return strategies->complete();
}

gjxl::AqEvaluationOptions MakeOptions(bool gaborish, uint32_t epf_iterations,
                                      bool non_default) {
  gjxl::AqEvaluationOptions options;
  options.coefficient_coding = {1.31f, 0.77f};
  options.loop_filter.gaborish = gaborish;
  options.loop_filter.epf_options.iterations = epf_iterations;
  if (non_default) {
    options.loop_filter.gaborish_options.weight1 = {0.071f, 0.093f, 0.057f};
    options.loop_filter.gaborish_options.weight2 = {0.039f, 0.027f, 0.045f};
    options.loop_filter.epf_options.channel_scale = {31.0f, 7.0f, 4.25f};
    options.loop_filter.epf_options.pass0_sigma_scale = 1.17f;
    options.loop_filter.epf_options.pass2_sigma_scale = 4.75f;
    options.loop_filter.epf_options.border_sad_multiplier = 0.81f;
    options.opsin_intensity_target = 183.0f;
  }
  return options;
}

bool Prepare(gjxl::GpuBackend &gpu, const HostImage &original,
             const HostImage &coding, const gjxl::AcStrategyGrid &strategies,
             gjxl::AqEvaluationOptions options,
             std::unique_ptr<gjxl::PreparedAqEvaluation> *prepared) {
  return CheckStatus(
      gjxl::PrepareAqEvaluation(gpu,
                                {.original_linear_rgb = original.ConstView(),
                                 .coding_opsin = coding.ConstView(),
                                 .strategies = &strategies,
                                 .options = options},
                                prepared),
      "prepared AQ postprocess creation");
}

std::vector<float> MakeSigma(gjxl::Extent2D coding_extent, uint32_t variant) {
  const gjxl::Extent2D blocks{coding_extent.width / 8,
                              coding_extent.height / 8};
  std::vector<float> sigma(blocks.width * blocks.height);
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      sigma[y * blocks.width + x] =
          -0.035f - 0.011f * static_cast<float>((x + 3 * y + variant) % 9);
    }
  }
  if (!sigma.empty())
    sigma.back() = -4.25f;
  return sigma;
}

gjxl::ConstPlaneF32View SigmaView(const std::vector<float> &sigma,
                                  gjxl::Extent2D coding_extent) {
  const gjxl::Extent2D blocks{coding_extent.width / 8,
                              coding_extent.height / 8};
  return {sigma.data(), blocks, blocks.width};
}

bool CompareSnapshotPlane(const std::vector<float> &actual,
                          const HostImage &expected, size_t channel,
                          double absolute, double relative,
                          double *maximum_error, std::string_view stage) {
  const size_t count = expected.extent.width * expected.extent.height;
  if (actual.size() != count) {
    std::cerr << stage << " output size mismatch\n";
    return false;
  }
  for (size_t y = 0; y < expected.extent.height; ++y) {
    for (size_t x = 0; x < expected.extent.width; ++x) {
      const size_t index = y * expected.extent.width + x;
      const float oracle = expected.plane[channel][y * expected.stride + x];
      if (!Near(actual[index], oracle, absolute, relative, maximum_error)) {
        std::cerr << stage << " mismatch at channel " << channel << ", (" << x
                  << ", " << y << "): actual " << actual[index] << ", expected "
                  << oracle << '\n';
        return false;
      }
    }
  }
  return true;
}

bool CompareExactInput(
    const gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting &snapshot,
    const HostImage &input) {
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!CompareSnapshotPlane(snapshot.reconstructed_opsin[channel], input,
                              channel, 0.0, 0.0, &g_max_filter_error,
                              "uploaded reconstruction")) {
      return false;
    }
  }
  return true;
}

bool ComparePostprocessOracle(
    const gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting &snapshot,
    const HostImage &reconstructed, gjxl::ConstPlaneF32View sigma,
    gjxl::AqEvaluationOptions options) {
  HostImage filtered(reconstructed.extent, reconstructed.extent.width + 5,
                     -777.0f);
  if (!CheckStatus(gjxl::ApplyLoopFilters(reconstructed.ConstView(), sigma,
                                          options.loop_filter, filtered.View()),
                   "CPU loop-filter oracle")) {
    return false;
  }
  HostImage linear(snapshot.source_extent, snapshot.source_extent.width + 3,
                   -888.0f);
  if (!CheckStatus(
          gjxl::OpsinToLinearRgb(filtered.Cropped(snapshot.source_extent),
                                 options.opsin_intensity_target, linear.View()),
          "CPU color-conversion oracle")) {
    return false;
  }
  HostImage actual_filtered(reconstructed.extent, reconstructed.extent.width);
  for (size_t channel = 0; channel < 3; ++channel) {
    actual_filtered.plane[channel] = snapshot.filtered_opsin[channel];
  }
  HostImage isolated_linear(snapshot.source_extent,
                            snapshot.source_extent.width + 1, -999.0f);
  if (!CheckStatus(gjxl::OpsinToLinearRgb(
                       actual_filtered.Cropped(snapshot.source_extent),
                       options.opsin_intensity_target, isolated_linear.View()),
                   "CPU isolated color-conversion oracle")) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!CompareSnapshotPlane(snapshot.filtered_opsin[channel], filtered,
                              channel, kFilterAbsolute, kFilterRelative,
                              &g_max_filter_error, "filtered opsin") ||
        !CompareSnapshotPlane(snapshot.reconstructed_linear[channel],
                              isolated_linear, channel, kColorAbsolute,
                              kColorRelative, &g_max_color_error,
                              "isolated reconstructed linear") ||
        !CompareSnapshotPlane(snapshot.reconstructed_linear[channel], linear,
                              channel, kCombinedAbsolute, kCombinedRelative,
                              &g_max_combined_error,
                              "combined reconstructed linear")) {
      return false;
    }
  }
  return true;
}

bool CheckDirectCase(gjxl::GpuBackend &gpu, gjxl::Extent2D source_extent,
                     gjxl::Extent2D coding_extent, bool gaborish,
                     uint32_t epf_iterations, bool non_default, uint32_t seed) {
  HostImage original(source_extent, source_extent.width + 4, -91.0f);
  HostImage coding(coding_extent, coding_extent.width + 7, -92.0f);
  FillLinear(&original);
  FillOpsin(&coding, seed);
  gjxl::AcStrategyGrid strategies;
  if (!MakeDct8Strategies(coding_extent, &strategies))
    return false;
  const gjxl::AqEvaluationOptions options =
      MakeOptions(gaborish, epf_iterations, non_default);
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Prepare(gpu, original, coding, strategies, options, &prepared))
    return false;

  gjxl::metal_internal::MetalAqPostprocessPlanForTesting plan;
  const size_t filter_stages =
      (gaborish ? size_t{1} : size_t{0}) + epf_iterations;
  if (!CheckStatus(gjxl::metal_internal::GetMetalAqPostprocessPlanForTesting(
                       *prepared, &plan),
                   "postprocess routing plan") ||
      plan.filter_scratch_images != std::min<size_t>(2, filter_stages) ||
      plan.gaborish_dispatches != (gaborish ? 1u : 0u) ||
      plan.epf_dispatches != epf_iterations || plan.color_dispatches != 1u ||
      plan.copy_dispatches != 0u) {
    std::cerr << "Prepared postprocess routing plan mismatch\n";
    return false;
  }

  const std::vector<float> sigma = MakeSigma(coding_extent, seed);
  const gjxl::GpuBackendStats before = gpu.stats();
  gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting snapshot;
  if (!CheckStatus(gjxl::metal_internal::RunMetalAqPostprocessForTesting(
                       *prepared, coding.ConstView(),
                       SigmaView(sigma, coding_extent), &snapshot),
                   "direct Metal AQ postprocess")) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu.stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 1 ||
      snapshot.source_extent != source_extent ||
      snapshot.coding_extent != coding_extent) {
    std::cerr << "Direct AQ postprocess residency contract failed\n";
    return false;
  }
  return CompareExactInput(snapshot, coding) &&
         ComparePostprocessOracle(snapshot, coding,
                                  SigmaView(sigma, coding_extent), options);
}

bool CheckDirectCorpus(gjxl::GpuBackend &gpu) {
  constexpr std::array<std::pair<gjxl::Extent2D, gjxl::Extent2D>, 5> extents = {
      std::pair{gjxl::Extent2D{5, 3}, gjxl::Extent2D{8, 8}},
      std::pair{gjxl::Extent2D{16, 8}, gjxl::Extent2D{16, 8}},
      std::pair{gjxl::Extent2D{3, 17}, gjxl::Extent2D{8, 24}},
      std::pair{gjxl::Extent2D{17, 9}, gjxl::Extent2D{24, 16}},
      std::pair{gjxl::Extent2D{121, 89}, gjxl::Extent2D{128, 96}},
  };
  uint32_t seed = 0x51a20u;
  for (const auto &[source, coding] : extents) {
    for (bool gaborish : {false, true}) {
      for (uint32_t iterations = 0; iterations <= 3; ++iterations) {
        if (!CheckDirectCase(gpu, source, coding, gaborish, iterations, false,
                             seed++)) {
          return false;
        }
      }
    }
  }
  for (bool gaborish : {false, true}) {
    for (uint32_t iterations = 0; iterations <= 3; ++iterations) {
      if (!CheckDirectCase(gpu, {121, 89}, {128, 96}, gaborish, iterations,
                           true, seed++)) {
        return false;
      }
    }
  }
  return true;
}

bool MakeMixedStrategies(gjxl::AcStrategyGrid *strategies) {
  constexpr gjxl::Extent2D blocks{12, 8};
  if (!CheckStatus(gjxl::AcStrategyGrid::Create(blocks, strategies),
                   "mixed strategy creation") ||
      !CheckStatus(strategies->Set(0, 0, gjxl::AcStrategyType::kDct32x32),
                   "DCT32x32 placement") ||
      !CheckStatus(strategies->Set(4, 0, gjxl::AcStrategyType::kDct32x16),
                   "DCT32x16 placement") ||
      !CheckStatus(strategies->Set(6, 0, gjxl::AcStrategyType::kDct16x32),
                   "DCT16x32 placement") ||
      !CheckStatus(strategies->Set(10, 0, gjxl::AcStrategyType::kDct16x16),
                   "DCT16x16 placement") ||
      !CheckStatus(strategies->Set(6, 2, gjxl::AcStrategyType::kDct16x8),
                   "DCT16x8 placement") ||
      !CheckStatus(strategies->Set(7, 2, gjxl::AcStrategyType::kDct8x16),
                   "DCT8x16 placement")) {
    return false;
  }
  strategies->fill_empty_dct8();
  return strategies->complete();
}

struct EvaluationInputStorage {
  std::vector<int32_t> raw_quant;
  std::vector<float> inverse_sigma;
  gjxl::ColorCorrelationMap color;
  gjxl::QuantizerParams quantizer{8192, 48};

  bool Initialize(const HostImage &coding) {
    constexpr gjxl::Extent2D blocks{12, 8};
    raw_quant.resize(blocks.width * blocks.height);
    inverse_sigma.resize(blocks.width * blocks.height);
    return CheckStatus(
        gjxl::ComputeInitialColorCorrelationMap(coding.ConstView(), &color),
        "chained color correlation");
  }

  void SetVariant(uint32_t variant) {
    constexpr gjxl::Extent2D blocks{12, 8};
    for (size_t y = 0; y < blocks.height; ++y) {
      for (size_t x = 0; x < blocks.width; ++x) {
        raw_quant[y * blocks.width + x] =
            1 + static_cast<int32_t>((37 * x + 19 * y + 71 * variant) % 256);
        inverse_sigma[y * blocks.width + x] =
            -0.041f - 0.003f * static_cast<float>((x + y + variant) % 7);
      }
    }
    quantizer.global_scale = 7168 + 1024 * variant;
  }

  gjxl::AqEvaluationInput View() const {
    constexpr gjxl::Extent2D blocks{12, 8};
    return {
        .raw_quant_field = {raw_quant.data(), blocks, blocks.width},
        .quantizer = quantizer,
        .y_to_x = color.y_to_x_map(),
        .y_to_b = color.y_to_b_map(),
        .epf_inverse_sigma = {inverse_sigma.data(), blocks, blocks.width},
    };
  }
};

bool CompareChainedReconstruction(
    const gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting &snapshot,
    const HostImage &coding, const gjxl::AcStrategyGrid &strategies,
    const EvaluationInputStorage &input, gjxl::AqEvaluationOptions options) {
  gjxl::Quantizer quantizer;
  gjxl::QuantizedCoefficientFrame frame;
  HostImage expected(coding.extent, coding.extent.width);
  if (!CheckStatus(gjxl::Quantizer::Create(input.quantizer, &quantizer),
                   "chained CPU quantizer") ||
      !CheckStatus(gjxl::ComputeQuantizedCoefficients(
                       coding.ConstView(), strategies,
                       input.View().raw_quant_field, quantizer, input.color,
                       options.coefficient_coding, &frame),
                   "chained CPU coefficient oracle") ||
      !CheckStatus(gjxl::ReconstructQuantizedCoefficients(
                       frame, quantizer, input.color,
                       options.coefficient_coding, expected.View()),
                   "chained CPU reconstruction oracle")) {
    return false;
  }
  double maximum_reconstruction_error = 0.0;
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!CompareSnapshotPlane(snapshot.reconstructed_opsin[channel], expected,
                              channel, 7.5e-4, 1.0e-4,
                              &maximum_reconstruction_error,
                              "chained reconstructed opsin")) {
      return false;
    }
  }

  HostImage actual_reconstruction(coding.extent, coding.extent.width);
  for (size_t channel = 0; channel < 3; ++channel) {
    actual_reconstruction.plane[channel] =
        snapshot.reconstructed_opsin[channel];
  }
  return ComparePostprocessOracle(snapshot, actual_reconstruction,
                                  input.View().epf_inverse_sigma, options);
}

bool CheckChainedPath(gjxl::GpuBackend &gpu) {
  constexpr gjxl::Extent2D source_extent{91, 57};
  constexpr gjxl::Extent2D coding_extent{96, 64};
  HostImage original(source_extent, source_extent.width + 3);
  HostImage coding(coding_extent, coding_extent.width + 5);
  FillLinear(&original);
  FillOpsin(&coding, 0x8192u);
  gjxl::AcStrategyGrid strategies;
  EvaluationInputStorage input;
  const gjxl::AqEvaluationOptions options = MakeOptions(true, 3, true);
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!MakeMixedStrategies(&strategies) || !input.Initialize(coding) ||
      !Prepare(gpu, original, coding, strategies, options, &prepared)) {
    return false;
  }
  const gjxl::GpuBackendStats before = gpu.stats();
  for (uint32_t variant = 0; variant < 3; ++variant) {
    input.SetVariant(variant);
    gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting snapshot;
    if (!CheckStatus(gjxl::metal_internal::
                         RunMetalAqReconstructionAndPostprocessForTesting(
                             *prepared, input.View(), &snapshot),
                     "chained Metal AQ postprocess") ||
        !CompareChainedReconstruction(snapshot, coding, strategies, input,
                                      options)) {
      return false;
    }
  }
  const gjxl::GpuBackendStats after = gpu.stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 3) {
    std::cerr << "Chained AQ postprocess did not preserve residency\n";
    return false;
  }
  return true;
}

gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting PoisonedSnapshot() {
  gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting snapshot;
  snapshot.coding_extent = {77, 88};
  snapshot.source_extent = {99, 111};
  snapshot.reconstructed_opsin[0] = {123.0f};
  snapshot.filtered_opsin[1] = {456.0f};
  snapshot.reconstructed_linear[2] = {789.0f};
  return snapshot;
}

bool IsPoisoned(const gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting
                    &snapshot) {
  return snapshot.coding_extent == gjxl::Extent2D{77, 88} &&
         snapshot.source_extent == gjxl::Extent2D{99, 111} &&
         snapshot.reconstructed_opsin[0] == std::vector<float>{123.0f} &&
         snapshot.filtered_opsin[1] == std::vector<float>{456.0f} &&
         snapshot.reconstructed_linear[2] == std::vector<float>{789.0f};
}

struct FailureFixture {
  HostImage original{{17, 9}, 20};
  HostImage coding{{24, 16}, 28};
  gjxl::AcStrategyGrid strategies;
  std::vector<float> sigma;
  gjxl::AqEvaluationOptions options = MakeOptions(false, 0, false);

  bool Initialize() {
    FillLinear(&original);
    FillOpsin(&coding, 0xdeadbu);
    sigma = MakeSigma(coding.extent, 0);
    return MakeDct8Strategies(coding.extent, &strategies);
  }
};

bool CheckValidationAndNumericFailure(gjxl::GpuBackend &gpu) {
  FailureFixture fixture;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!fixture.Initialize() ||
      !Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               fixture.options, &prepared)) {
    return false;
  }
  auto snapshot = PoisonedSnapshot();
  HostImage invalid = fixture.coding;
  invalid.plane[1][5] = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> invalid_sigma = fixture.sigma;
  invalid_sigma[0] = 0.0f;
  const uint64_t submissions = gpu.stats().committed_submissions;
  if (!ExpectCode(gjxl::metal_internal::RunMetalAqPostprocessForTesting(
                      *prepared, invalid.ConstView(),
                      SigmaView(fixture.sigma, fixture.coding.extent),
                      &snapshot),
                  gjxl::StatusCode::kInvalidArgument,
                  "non-finite direct postprocess input") ||
      !IsPoisoned(snapshot) ||
      gpu.stats().committed_submissions != submissions ||
      !ExpectCode(gjxl::metal_internal::RunMetalAqPostprocessForTesting(
                      *prepared, fixture.coding.ConstView(),
                      SigmaView(invalid_sigma, fixture.coding.extent),
                      &snapshot),
                  gjxl::StatusCode::kInvalidArgument,
                  "invalid direct postprocess sigma") ||
      !IsPoisoned(snapshot) ||
      gpu.stats().committed_submissions != submissions ||
      !CheckStatus(gjxl::metal_internal::RunMetalAqPostprocessForTesting(
                       *prepared, fixture.coding.ConstView(),
                       SigmaView(fixture.sigma, fixture.coding.extent),
                       &snapshot),
                   "reuse after rejected postprocess input")) {
    return false;
  }

  if (!Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               fixture.options, &prepared)) {
    return false;
  }
  snapshot = PoisonedSnapshot();
  HostImage overflow = fixture.coding;
  for (std::vector<float> &plane : overflow.plane) {
    std::fill(plane.begin(), plane.end(), std::numeric_limits<float>::max());
  }
  return ExpectCode(gjxl::metal_internal::RunMetalAqPostprocessForTesting(
                        *prepared, overflow.ConstView(),
                        SigmaView(fixture.sigma, fixture.coding.extent),
                        &snapshot),
                    gjxl::StatusCode::kDeviceError,
                    "finite-overflow postprocess input") &&
         IsPoisoned(snapshot) &&
         ExpectCode(gjxl::metal_internal::RunMetalAqPostprocessForTesting(
                        *prepared, fixture.coding.ConstView(),
                        SigmaView(fixture.sigma, fixture.coding.extent),
                        &snapshot),
                    gjxl::StatusCode::kFailedPrecondition,
                    "postprocess reuse after numeric failure") &&
         IsPoisoned(snapshot);
}

bool CheckInjectedFailure(gjxl::MetalBackendOptions backend_options,
                          gjxl::StatusCode expected, bool inject_readback) {
  FailureFixture fixture;
  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!fixture.Initialize() ||
      !CheckStatus(
          gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, backend_options, &gpu),
          "postprocess failure backend") ||
      !Prepare(*gpu, fixture.original, fixture.coding, fixture.strategies,
               fixture.options, &prepared)) {
    return false;
  }
  if (inject_readback &&
      !CheckStatus(
          gjxl::metal_internal::FailNextMetalAqReadbackForTesting(*prepared),
          "postprocess readback injection")) {
    return false;
  }
  auto snapshot = PoisonedSnapshot();
  return ExpectCode(gjxl::metal_internal::RunMetalAqPostprocessForTesting(
                        *prepared, fixture.coding.ConstView(),
                        SigmaView(fixture.sigma, fixture.coding.extent),
                        &snapshot),
                    expected, "injected postprocess failure") &&
         IsPoisoned(snapshot) &&
         ExpectCode(gjxl::metal_internal::RunMetalAqPostprocessForTesting(
                        *prepared, fixture.coding.ConstView(),
                        SigmaView(fixture.sigma, fixture.coding.extent),
                        &snapshot),
                    gjxl::StatusCode::kFailedPrecondition,
                    "postprocess reuse after injected failure") &&
         IsPoisoned(snapshot);
}

bool CheckIndependentConcurrency(gjxl::GpuBackend &gpu) {
  FailureFixture fixture;
  std::unique_ptr<gjxl::PreparedAqEvaluation> first;
  std::unique_ptr<gjxl::PreparedAqEvaluation> second;
  if (!fixture.Initialize() ||
      !Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               fixture.options, &first) ||
      !Prepare(gpu, fixture.original, fixture.coding, fixture.strategies,
               fixture.options, &second)) {
    return false;
  }
  gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting first_snapshot;
  gjxl::metal_internal::MetalAqPostprocessSnapshotForTesting second_snapshot;
  bool first_ok = false;
  bool second_ok = false;
  const gjxl::GpuBackendStats before = gpu.stats();
  std::thread first_thread([&] {
    first_ok =
        gjxl::metal_internal::RunMetalAqPostprocessForTesting(
            *first, fixture.coding.ConstView(),
            SigmaView(fixture.sigma, fixture.coding.extent), &first_snapshot)
            .ok();
  });
  std::thread second_thread([&] {
    second_ok =
        gjxl::metal_internal::RunMetalAqPostprocessForTesting(
            *second, fixture.coding.ConstView(),
            SigmaView(fixture.sigma, fixture.coding.extent), &second_snapshot)
            .ok();
  });
  first_thread.join();
  second_thread.join();
  const gjxl::GpuBackendStats after = gpu.stats();
  return first_ok && second_ok &&
         after.successful_allocations == before.successful_allocations &&
         after.committed_submissions == before.committed_submissions + 2 &&
         CompareExactInput(first_snapshot, fixture.coding) &&
         CompareExactInput(second_snapshot, fixture.coding);
}

} // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "Metal AQ postprocess backend") ||
      !CheckDirectCorpus(*gpu) || !CheckChainedPath(*gpu) ||
      !CheckValidationAndNumericFailure(*gpu) ||
      !CheckInjectedFailure({.test_fail_submission = true},
                            gjxl::StatusCode::kSubmissionFailed, false) ||
      !CheckInjectedFailure({.test_fail_completion = true},
                            gjxl::StatusCode::kDeviceError, false) ||
      !CheckInjectedFailure({}, gjxl::StatusCode::kDeviceError, true) ||
      !CheckIndependentConcurrency(*gpu)) {
    return EXIT_FAILURE;
  }
  std::cout
      << "Metal AQ Milestone 4 postprocess tests passed; max filter error "
      << g_max_filter_error << ", max color error " << g_max_color_error
      << ", max combined error " << g_max_combined_error << '\n';
  return EXIT_SUCCESS;
}
