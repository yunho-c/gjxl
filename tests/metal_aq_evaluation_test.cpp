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
#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_aq_butteraugli_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/aq_evaluation.h"

namespace {

constexpr uint32_t kPoisonBits = 0x7fc12345u;
constexpr float kPoison = std::bit_cast<float>(kPoisonBits);
constexpr float kReductionTolerance = 2.0e-6f;
float g_max_reduction_error = 0.0f;
gjxl::AqEvaluationMemoryStats g_memory_stats;

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
             std::unique_ptr<gjxl::PreparedAqEvaluation>* prepared) {
  return CheckStatus(gjxl::PrepareAqEvaluation(
    gpu,
    {
      .original_linear_rgb = original.View(),
      .coding_opsin = coding.View(),
      .strategies = &strategies,
      .options = MakeOptions(),
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
  BackendWithoutAq backend;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  return fixture.Initialize() &&
    gjxl::QueryGpuAqEvaluation(backend) == nullptr &&
    ExpectCode(gjxl::PrepareAqEvaluation(
      backend,
      {
        .original_linear_rgb = fixture.original.View(),
        .coding_opsin = fixture.coding.View(),
        .strategies = &fixture.strategies,
        .options = MakeOptions(),
      },
      &prepared), gjxl::StatusCode::kUnavailable, "missing AQ capability") &&
    prepared == nullptr;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "Metal AQ backend") ||
      !CheckCapabilityBoundary() ||
      !CheckReductionCorpus(*gpu) ||
      !CheckProductionEvaluation(*gpu) ||
      !CheckSplitSeamAndDestruction(*gpu) ||
      !CheckUploadOrNumericFailure(true) ||
      !CheckFailure(gjxl::StatusCode::kSubmissionFailed, true, false, false) ||
      !CheckFailure(gjxl::StatusCode::kDeviceError, false, true, false) ||
      !CheckUploadOrNumericFailure(false) ||
      !CheckFailure(gjxl::StatusCode::kDeviceError, false, false, true) ||
      !CheckIndependentConcurrency(*gpu)) {
    return EXIT_FAILURE;
  }
  std::cout << "Metal AQ Milestone 6 evaluation tests passed; max block "
            << "reduction error " << g_max_reduction_error
            << "; memory persistent " << g_memory_stats.persistent_bytes
            << ", staging " << g_memory_stats.staging_bytes
            << ", peak scratch " << g_memory_stats.peak_scratch_bytes
            << " bytes\n";
  return EXIT_SUCCESS;
}
