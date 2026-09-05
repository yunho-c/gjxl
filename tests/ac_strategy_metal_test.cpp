// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/quantization.h"
#include "gpu/backend.h"
#ifdef GJXL_TEST_CUDA
#include "gpu/cuda/cuda_backend.h"
#else
#include "gpu/metal/metal_backend.h"
#endif
#include "gpu/ops/ac_strategy.h"

namespace {

constexpr gjxl::Extent2D kPixelExtent{64, 64};
constexpr gjxl::Extent2D kBlockExtent{8, 8};
#ifdef GJXL_TEST_CUDA
constexpr size_t kCandidateCount = 33;
#else
constexpr size_t kCandidateCount = 7;
#endif

constexpr std::array kStrategies = {
  gjxl::AcStrategyType::kDct8,
  gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16,
  gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
  gjxl::AcStrategyType::kDct32x32,
};

struct Fixture {
  std::array<std::vector<float>, 3> plane;
  std::vector<float> packed_opsin;
  std::vector<float> quant_field;
  std::vector<float> pixel_mask;

  Fixture()
    : packed_opsin(3 * kPixelExtent.width * kPixelExtent.height),
      quant_field(kBlockExtent.width * kBlockExtent.height),
      pixel_mask(kPixelExtent.width * kPixelExtent.height) {

    const size_t plane_size = kPixelExtent.width * kPixelExtent.height;
    for (size_t channel = 0; channel < plane.size(); ++channel) {
      plane[channel].resize(plane_size);
      for (size_t y = 0; y < kPixelExtent.height; ++y) {
        for (size_t x = 0; x < kPixelExtent.width; ++x) {
          const float value =
            0.08f +
            0.019f * std::sin(
              0.13f * static_cast<float>(x + 3 * channel + 1)) +
            0.016f * std::cos(
              0.17f * static_cast<float>(2 * y + channel + 2)) +
            0.004f * static_cast<float>((x * 7 + y * 5 + channel) % 13);
          plane[channel][y * kPixelExtent.width + x] = value;
        }
      }
      std::copy(
        plane[channel].begin(),
        plane[channel].end(),
        packed_opsin.begin() + channel * plane_size);
    }

    for (size_t i = 0; i < quant_field.size(); ++i) {
      quant_field[i] = 0.31f + 0.009f * static_cast<float>((i * 11) % 37);
    }
    for (size_t y = 0; y < kPixelExtent.height; ++y) {
      for (size_t x = 0; x < kPixelExtent.width; ++x) {
        pixel_mask[y * kPixelExtent.width + x] =
          39.0f + 0.23f * static_cast<float>((x * 3 + y * 17) % 29);
      }
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView Image() const {
    return {{
      gjxl::ConstPlaneF32View{
        plane[0].data(), kPixelExtent, kPixelExtent.width},
      gjxl::ConstPlaneF32View{
        plane[1].data(), kPixelExtent, kPixelExtent.width},
      gjxl::ConstPlaneF32View{
        plane[2].data(), kPixelExtent, kPixelExtent.width},
    }};
  }

  [[nodiscard]] gjxl::ConstPlaneF32View QuantField() const {
    return {quant_field.data(), kBlockExtent, kBlockExtent.width};
  }

  [[nodiscard]] gjxl::ConstPlaneF32View PixelMask() const {
    return {pixel_mask.data(), kPixelExtent, kPixelExtent.width};
  }
};

bool CheckStatus(
  const gjxl::Status& status,
  std::string_view operation) {

  if (status.ok()) {
    return true;
  }
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

class BackendWithoutAc final : public gjxl::GpuBackend {
public:
  [[nodiscard]] gjxl::BackendKind kind() const noexcept override {
#ifdef GJXL_TEST_CUDA
    return gjxl::BackendKind::kCuda;
#else
    return gjxl::BackendKind::kMetal;
#endif
  }

  [[nodiscard]] std::string_view name() const noexcept override {
    return "backend without AC evaluation";
  }

  gjxl::Status Allocate(
    size_t,
    std::unique_ptr<gjxl::DeviceBuffer>*) override {

    return gjxl::Status::Unavailable("Allocation is not implemented");
  }

  gjxl::Status CopyHostToDevice(
    gjxl::DeviceBuffer&,
    const void*,
    size_t,
    size_t) override {

    return gjxl::Status::Unavailable("Upload is not implemented");
  }

  gjxl::Status CopyDeviceToHost(
    const gjxl::DeviceBuffer&,
    void*,
    size_t,
    size_t) override {

    return gjxl::Status::Unavailable("Readback is not implemented");
  }

  gjxl::Status ForwardTransform(
    const gjxl::TransformBatch&,
    std::unique_ptr<gjxl::GpuSubmission>* submission) override {

    if (submission != nullptr)
      submission->reset();
    return gjxl::Status::Unavailable("Transform is not implemented");
  }

  gjxl::Status InverseTransform(
    const gjxl::TransformBatch&,
    std::unique_ptr<gjxl::GpuSubmission>* submission) override {

    if (submission != nullptr)
      submission->reset();
    return gjxl::Status::Unavailable("Transform is not implemented");
  }
};

std::vector<gjxl::AcStrategyCandidate> MakeCandidates(
  gjxl::AcStrategyType strategy,
  const Fixture& fixture) {

  const gjxl::Extent2D covered =
    gjxl::GetAcStrategyInfo(strategy)->covered_blocks;
  const size_t position_count_x = kBlockExtent.width - covered.width + 1;
  const size_t position_count_y = kBlockExtent.height - covered.height + 1;
  std::vector<gjxl::AcStrategyCandidate> candidates(kCandidateCount);
  for (size_t i = 0; i < candidates.size(); ++i) {
    const size_t block_x = i == 0 ? 0 : i == 1 ? position_count_x - 1 :
      (3 * i + 1) % position_count_x;
    const size_t block_y = i == 0 ? 0 : i == 1 ? position_count_y - 1 :
      (5 * i + 2) % position_count_y;
    float quant_norm = 0.0f;
    if (!CheckStatus(
          gjxl::ComputeAcStrategyQuantNorm(
            strategy,
            block_x,
            block_y,
            fixture.QuantField(),
            &quant_norm),
          "ComputeAcStrategyQuantNorm")) {
      return {};
    }
    candidates[i] = {
      .block_x = static_cast<uint32_t>(block_x),
      .block_y = static_cast<uint32_t>(block_y),
      .quant_norm = quant_norm,
      .entropy_multiplier = 0.92f + 0.035f * static_cast<float>(i),
      .cfl_x = -0.16f + 0.04f * static_cast<float>(i),
      .cfl_b = 0.13f - 0.027f * static_cast<float>(i),
    };
  }
  return candidates;
}

bool PackMatrices(
  gjxl::AcStrategyType strategy,
  std::vector<float>* matrices) {

  const size_t coefficient_count =
    gjxl::GetAcStrategyInfo(strategy)->coefficient_count();
  matrices->resize(
    gjxl::kAcStrategyCostMatrixCount * coefficient_count);
  for (size_t channel = 0; channel < 3; ++channel) {
    gjxl::QuantizationMatrixView matrix;
    if (!CheckStatus(
          gjxl::GetDefaultQuantizationMatrix(
            strategy,
            static_cast<gjxl::XybChannel>(channel),
            &matrix),
          "GetDefaultQuantizationMatrix")) {
      return false;
    }
    std::copy(
      matrix.dequant.begin(),
      matrix.dequant.end(),
      matrices->begin() + channel * coefficient_count);
    std::copy(
      matrix.inverse_dequant.begin(),
      matrix.inverse_dequant.end(),
      matrices->begin() + (3 + channel) * coefficient_count);
  }
  return true;
}

bool Allocate(
  gjxl::GpuBackend& gpu,
  size_t bytes,
  std::string_view role,
  std::unique_ptr<gjxl::DeviceBuffer>* buffer) {

  return CheckStatus(gpu.Allocate(bytes, buffer), role);
}

#ifdef GJXL_TEST_CUDA
bool CheckCompactScratchRanges(gjxl::GpuBackend& gpu,
  const gjxl::AcStrategyCandidateBatch& batch,
  const std::vector<float>& expected) {
  constexpr float kGuard = -123456.0f;
  std::unique_ptr<gjxl::GpuSubmission> submission;
  for (const size_t count : {1, 8, 11, 16, 32, 33}) {
    gjxl::AcStrategyScratchRequirements scratch;
    if (!CheckStatus(gjxl::GetAcStrategyScratchRequirements(
          gpu, batch.strategy, count, &scratch), "Query compact scratch")) {
      return false;
    }
    // Pack all four live output ranges back-to-back, with guards only outside
    // the complete allocation. No obsolete coefficient-sized A gap is left.
    const std::array<size_t, 4> sizes = {scratch.scratch_a_bytes,
      scratch.scratch_b_bytes, scratch.rate_scratch_bytes, count * sizeof(float)};
    std::array<size_t, 4> offsets{7 * sizeof(float)};
    for (size_t i = 1; i < offsets.size(); ++i) {
      offsets[i] = offsets[i - 1] + sizes[i - 1];
    }
    const size_t live_end = offsets.back() + sizes.back();
    const size_t arena_bytes = live_end + 11 * sizeof(float);
    std::vector<float> words(arena_bytes / sizeof(float), kGuard);
    std::unique_ptr<gjxl::DeviceBuffer> arena;
    if (!Allocate(gpu, arena_bytes, "Allocate compact arena", &arena) ||
        !CheckStatus(gpu.CopyHostToDevice(*arena, words.data(), arena_bytes),
          "Initialize compact guards")) return false;
    auto compact = batch;
    compact.candidate_count = count;
    compact.scratch_a = compact.scratch_b = compact.rate_scratch =
      compact.costs = arena.get();
    compact.scratch_a_offset_bytes = offsets[0];
    compact.scratch_b_offset_bytes = offsets[1];
    compact.rate_scratch_offset_bytes = offsets[2];
    compact.costs_offset_bytes = offsets[3];
    // Ordered batches are allowed to share exactly the same scratch ranges.
    const std::array batches{compact, compact};
    if (!CheckStatus(gjxl::EvaluateAcStrategyCandidateBatches(
          gpu, batches, &submission), "Submit compact arena") ||
        submission == nullptr || !CheckStatus(submission->Wait(), "Wait compact") ||
        !CheckStatus(gpu.CopyDeviceToHost(*arena, words.data(), arena_bytes),
          "Read compact arena")) return false;
    for (size_t i = 0; i < words.size(); ++i) {
      if (((i < 7 || i >= live_end / sizeof(float)) && words[i] != kGuard) ||
          (i >= offsets[3] / sizeof(float) && i < live_end / sizeof(float) &&
           words[i] != expected[i - offsets[3] / sizeof(float)])) {
        std::cerr << "Compact scratch changed a cost or guard\n";
        return false;
      }
    }
    const std::array offset_members = {
      &gjxl::AcStrategyCandidateBatch::scratch_a_offset_bytes,
      &gjxl::AcStrategyCandidateBatch::scratch_b_offset_bytes,
      &gjxl::AcStrategyCandidateBatch::rate_scratch_offset_bytes,
      &gjxl::AcStrategyCandidateBatch::costs_offset_bytes,
    };
    const auto committed = gpu.stats().committed_submissions;
    const auto Rejected = [&](const auto& invalid) {
      return gjxl::EvaluateAcStrategyCandidates(gpu, invalid, &submission).code() ==
          gjxl::StatusCode::kInvalidArgument && submission == nullptr &&
          gpu.stats().committed_submissions == committed;
    };
    for (size_t i = 0; i < sizes.size(); ++i) {
      auto invalid = compact;
      invalid.*offset_members[i] = arena_bytes - sizes[i] + sizeof(float);
      if (!Rejected(invalid)) {
        std::cerr << "One-float-short scratch range was accepted\n";
        return false;
      }
      invalid = compact;
      invalid.*offset_members[i] += 1;
      if (!Rejected(invalid)) return false;
      for (size_t j = i + 1; j < sizes.size(); ++j) {
        invalid = compact;
        invalid.*offset_members[j] = offsets[i] + sizes[i] - sizeof(float);
        if (!Rejected(invalid)) {
          std::cerr << "Partially overlapping scratch ranges were accepted\n";
          return false;
        }
      }
    }
    auto input_alias = compact;
    input_alias.scratch_a = const_cast<gjxl::DeviceBuffer*>(batch.opsin);
    input_alias.scratch_a_offset_bytes = batch.opsin_offset_bytes;
    if (!Rejected(input_alias)) return false;
  }
  std::unique_ptr<gjxl::GpuBackend> foreign_gpu;
  std::unique_ptr<gjxl::DeviceBuffer> foreign_a;
  if (!CheckStatus(gjxl::CreateCudaBackend(&foreign_gpu), "Create foreign backend") ||
      !Allocate(*foreign_gpu, batch.candidate_count * 3 * sizeof(float),
        "Allocate foreign compact scratch", &foreign_a)) return false;
  auto foreign = batch;
  foreign.scratch_a = foreign_a.get();
  if (gjxl::EvaluateAcStrategyCandidates(gpu, foreign, &submission).code() !=
        gjxl::StatusCode::kInvalidArgument || submission != nullptr) {
    std::cerr << "Foreign compact scratch was accepted\n";
    return false;
  }
  // Old callers may continue allocating full coefficient storage for A.
  std::unique_ptr<gjxl::DeviceBuffer> full_a;
  const size_t full_bytes = batch.candidate_count * 3 * sizeof(float) *
    gjxl::GetAcStrategyInfo(batch.strategy)->coefficient_count();
  if (!Allocate(gpu, full_bytes, "Allocate conservative scratch", &full_a)) {
    return false;
  }
  auto conservative = batch;
  conservative.scratch_a = full_a.get();
  if (!CheckStatus(gjxl::EvaluateAcStrategyCandidates(gpu, conservative,
        &submission), "Submit conservative scratch") || submission == nullptr ||
      !CheckStatus(submission->Wait(), "Wait conservative scratch")) return false;
  std::vector<float> costs(expected.size());
  return CheckStatus(gpu.CopyDeviceToHost(*batch.costs, costs.data(),
           costs.size() * sizeof(float)), "Read conservative costs") && costs == expected;
}
#endif

bool RunStrategyCase(
  gjxl::GpuBackend& gpu,
  std::string_view implementation,
  gjxl::AcStrategyType strategy,
  const Fixture& fixture) {

  constexpr float kButteraugliTarget = 1.3f;
  const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
  const size_t coefficient_count = info->coefficient_count();
  std::vector<gjxl::AcStrategyCandidate> candidates =
    MakeCandidates(strategy, fixture);
  std::vector<float> matrices;
  if (!PackMatrices(strategy, &matrices)) {
    return false;
  }

  std::vector<float> expected(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    const gjxl::AcStrategyCandidate& candidate = candidates[i];
    if (!CheckStatus(
          gjxl::EstimateAcStrategyCost(
            strategy,
            candidate.block_x,
            candidate.block_y,
            fixture.Image(),
            fixture.QuantField(),
            fixture.PixelMask(),
            {
              .butteraugli_target = kButteraugliTarget,
              .entropy_multiplier = candidate.entropy_multiplier,
              .cfl_factors = {candidate.cfl_x, 0.0f, candidate.cfl_b},
            },
            &expected[i]),
          "CPU candidate evaluation")) {
      return false;
    }
  }

  const size_t image_bytes = fixture.packed_opsin.size() * sizeof(float);
  const size_t quant_bytes = fixture.quant_field.size() * sizeof(float);
  const size_t mask_bytes = fixture.pixel_mask.size() * sizeof(float);
  const size_t matrix_bytes = matrices.size() * sizeof(float);
  const size_t candidate_bytes =
    candidates.size() * sizeof(gjxl::AcStrategyCandidate);
  const size_t packed_bytes =
    candidates.size() * 3 * coefficient_count * sizeof(float);
  const size_t rate_bytes = candidates.size() * 3 *
    gjxl::kAcStrategyRateScratchBytesPerChannel;
  const size_t cost_bytes = candidates.size() * sizeof(float);
  gjxl::AcStrategyScratchRequirements scratch;
  if (!CheckStatus(gjxl::GetAcStrategyScratchRequirements(
        gpu, strategy, candidates.size(), &scratch), "Query scratch sizes") ||
      scratch.scratch_b_bytes != packed_bytes ||
      scratch.rate_scratch_bytes != rate_bytes ||
#ifdef GJXL_TEST_CUDA
      scratch.scratch_a_bytes != candidates.size() * 3 * sizeof(float)) {
#else
      scratch.scratch_a_bytes != packed_bytes) {
#endif
    std::cerr << "Unexpected AC scratch requirements\n";
    return false;
  }

  std::unique_ptr<gjxl::DeviceBuffer> device_opsin;
  std::unique_ptr<gjxl::DeviceBuffer> device_mask;
  std::unique_ptr<gjxl::DeviceBuffer> device_quant;
  std::unique_ptr<gjxl::DeviceBuffer> device_matrices;
  std::unique_ptr<gjxl::DeviceBuffer> device_candidates;
  std::unique_ptr<gjxl::DeviceBuffer> scratch_a;
  std::unique_ptr<gjxl::DeviceBuffer> scratch_b;
  std::unique_ptr<gjxl::DeviceBuffer> rate_scratch;
  std::unique_ptr<gjxl::DeviceBuffer> device_costs;
  if (!Allocate(gpu, image_bytes, "Allocate opsin", &device_opsin) ||
      !Allocate(gpu, mask_bytes, "Allocate mask", &device_mask) ||
      !Allocate(gpu, quant_bytes, "Allocate quant field", &device_quant) ||
      !Allocate(gpu, matrix_bytes, "Allocate matrices", &device_matrices) ||
      !Allocate(
        gpu, candidate_bytes, "Allocate candidates", &device_candidates) ||
      !Allocate(gpu, scratch.scratch_a_bytes, "Allocate scratch A", &scratch_a) ||
      !Allocate(gpu, packed_bytes, "Allocate scratch B", &scratch_b) ||
      !Allocate(gpu, rate_bytes, "Allocate rate scratch", &rate_scratch) ||
      !Allocate(gpu, cost_bytes, "Allocate costs", &device_costs) ||
      !CheckStatus(
        gpu.CopyHostToDevice(
          *device_opsin, fixture.packed_opsin.data(), image_bytes),
        "Upload opsin") ||
      !CheckStatus(
        gpu.CopyHostToDevice(
          *device_mask, fixture.pixel_mask.data(), mask_bytes),
        "Upload mask") ||
      !CheckStatus(
        gpu.CopyHostToDevice(
          *device_quant, fixture.quant_field.data(), quant_bytes),
        "Upload quant field") ||
      !CheckStatus(
        gpu.CopyHostToDevice(
          *device_matrices, matrices.data(), matrix_bytes),
        "Upload matrices") ||
      !CheckStatus(
        gpu.CopyHostToDevice(
          *device_candidates, candidates.data(), candidate_bytes),
        "Upload candidates")) {
    return false;
  }

  std::vector<float> poisoned_costs(
    candidates.size(),
    std::numeric_limits<float>::quiet_NaN());
  if (!CheckStatus(
        gpu.CopyHostToDevice(
          *device_costs, poisoned_costs.data(), cost_bytes),
        "Poison cost output")) {
    return false;
  }

  const gjxl::AcStrategyCandidateBatch batch{
    .strategy = strategy,
    .opsin = device_opsin.get(),
    .pixel_mask = device_mask.get(),
    .matrices = device_matrices.get(),
    .candidates = device_candidates.get(),
    .scratch_a = scratch_a.get(),
    .scratch_b = scratch_b.get(),
    .rate_scratch = rate_scratch.get(),
    .costs = device_costs.get(),
    .pixel_extent = kPixelExtent,
    .opsin_row_stride = kPixelExtent.width,
    .opsin_plane_stride = kPixelExtent.width * kPixelExtent.height,
    .pixel_mask_row_stride = kPixelExtent.width,
    .candidate_count = candidates.size(),
    .butteraugli_target = kButteraugliTarget,
  };
  std::unique_ptr<gjxl::GpuSubmission> submission;
  if (!CheckStatus(
        gjxl::EvaluateAcStrategyCandidates(gpu, batch, &submission),
        "Submit candidate batch") ||
      submission == nullptr ||
      !CheckStatus(submission->Wait(), "Wait for candidate batch") ||
      !CheckStatus(
        gpu.CopyDeviceToHost(
          *device_costs, poisoned_costs.data(), cost_bytes),
        "Download candidate costs")) {
    return false;
  }

  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const double actual = poisoned_costs[i];
    const double reference = expected[i];
    const double absolute_error = std::abs(actual - reference);
    const double relative_error = absolute_error / std::max(1.0, reference);
    max_absolute_error = std::max(max_absolute_error, absolute_error);
    max_relative_error = std::max(max_relative_error, relative_error);
    if (!std::isfinite(actual) ||
        absolute_error > 0.005 + 1.0e-6 * std::abs(reference)) {
      std::cerr << implementation << ' ' << info->name
                << " cost mismatch at candidate " << i
                << ": expected " << reference
                << ", got " << actual << '\n';
      return false;
    }
  }

  std::cout << implementation << ' ' << info->name
            << " max absolute cost error " << max_absolute_error
            << ", max relative error " << max_relative_error << '\n';

#ifdef GJXL_TEST_CUDA
  // Changing batch length must not change an individual candidate's result.
  // Cross the 8/16/32-transform packed DCT boundaries (three transforms per
  // candidate), as well as partially occupied packed residual blocks.
  const std::vector<float> complete_costs = poisoned_costs;
  if (!CheckCompactScratchRanges(gpu, batch, complete_costs)) return false;
  for (const size_t count : {1, 2, 3, 5, 6, 10, 11, 16, 17, 21, 22, 32}) {
    gjxl::AcStrategyCandidateBatch prefix = batch;
    prefix.candidate_count = count;
    std::ranges::fill(poisoned_costs, -1.0f);
    if (!CheckStatus(gpu.CopyHostToDevice(
          *device_costs, poisoned_costs.data(), cost_bytes), "Poison prefix costs") ||
        !CheckStatus(gjxl::EvaluateAcStrategyCandidates(gpu, prefix, &submission),
          "Submit candidate prefix") ||
        submission == nullptr ||
        !CheckStatus(submission->Wait(), "Wait for candidate prefix") ||
        !CheckStatus(gpu.CopyDeviceToHost(
          *device_costs, poisoned_costs.data(), cost_bytes), "Download prefix costs")) {
      return false;
    }
    for (size_t i = 0; i < poisoned_costs.size(); ++i) {
      if (poisoned_costs[i] != (i < count ? complete_costs[i] : -1.0f)) {
        std::cerr << implementation << ' ' << info->name
                  << " candidate prefix changed a cost or its output guard\n";
        return false;
      }
    }
  }
#endif

#ifdef GJXL_TEST_CUDA
  // Fused gathering and cost reduction must honor independent padded rows,
  // plane gaps, nonzero base offsets, and resident plane offsets. NaN padding
  // catches misplaced reads; exact costs isolate addressing from rounding.
  constexpr size_t kOpsinOffset = 13;
  constexpr size_t kOpsinStride = kPixelExtent.width + 11;
  constexpr size_t kPlaneStride = kOpsinStride * kPixelExtent.height + 19;
  std::vector<float> strided_opsin(kOpsinOffset + 3 * kPlaneStride + 7,
    std::numeric_limits<float>::quiet_NaN());
  std::unique_ptr<gjxl::DeviceBuffer> device_strided_opsin;
  const size_t strided_bytes = strided_opsin.size() * sizeof(float);
  constexpr size_t kMaskOffset = 7;
  constexpr size_t kMaskStride = kPixelExtent.width + 5;
  std::vector<float> strided_mask(kMaskOffset + kMaskStride * kPixelExtent.height + 9,
    std::numeric_limits<float>::quiet_NaN());
  for (size_t y = 0; y < kPixelExtent.height; ++y) {
    std::copy_n(fixture.pixel_mask.data() + y * kPixelExtent.width,
      kPixelExtent.width, strided_mask.data() + kMaskOffset + y * kMaskStride);
  }
  const size_t strided_mask_bytes = strided_mask.size() * sizeof(float);
  std::unique_ptr<gjxl::DeviceBuffer> device_strided_mask;
  if (!Allocate(gpu, strided_bytes, "Allocate strided opsin",
        &device_strided_opsin) ||
      !Allocate(gpu, strided_mask_bytes, "Allocate strided mask",
        &device_strided_mask) ||
      !CheckStatus(gpu.CopyHostToDevice(*device_strided_mask,
        strided_mask.data(), strided_mask_bytes), "Upload strided mask")) {
    return false;
  }
  for (const bool resident : {false, true}) {
    gjxl::AcStrategyCandidateBatch strided_batch = batch;
    strided_batch.opsin = device_strided_opsin.get();
    strided_batch.opsin_offset_bytes = kOpsinOffset * sizeof(float);
    strided_batch.opsin_row_stride = kOpsinStride;
    strided_batch.opsin_plane_stride = kPlaneStride;
    strided_batch.pixel_mask = device_strided_mask.get();
    strided_batch.pixel_mask_offset_bytes = kMaskOffset * sizeof(float);
    strided_batch.pixel_mask_row_stride = kMaskStride;
    for (size_t channel = 0; channel < 3; ++channel) {
      const size_t physical_channel = resident ? (channel + 2) % 3 : channel;
      const size_t offset = kOpsinOffset + physical_channel * kPlaneStride;
      for (size_t y = 0; y < kPixelExtent.height; ++y) {
        std::copy_n(fixture.plane[channel].data() + y * kPixelExtent.width,
          kPixelExtent.width, strided_opsin.data() + offset + y * kOpsinStride);
      }
      if (resident) {
        strided_batch.resident_opsin.plane[channel] = {
          device_strided_opsin.get(), offset * sizeof(float),
          gjxl::DeviceElementType::kF32, kPixelExtent, kOpsinStride};
      }
    }
    if (resident) {
      strided_batch.resident_pixel_mask = {
        device_strided_mask.get(), kMaskOffset * sizeof(float),
        gjxl::DeviceElementType::kF32, kPixelExtent, kMaskStride};
    }
    if (!CheckStatus(gpu.CopyHostToDevice(*device_strided_opsin,
          strided_opsin.data(), strided_bytes), "Upload strided opsin") ||
        !CheckStatus(gjxl::EvaluateAcStrategyCandidates(
          gpu, strided_batch, &submission), "Submit strided candidate batch") ||
        submission == nullptr ||
        !CheckStatus(submission->Wait(), "Wait for strided candidate batch") ||
        !CheckStatus(gpu.CopyDeviceToHost(*device_costs,
          poisoned_costs.data(), cost_bytes), "Download strided costs")) {
      return false;
    }
    if (poisoned_costs != complete_costs) {
      std::cerr << implementation << ' ' << info->name
                << " strided candidate cost changed (resident=" << resident
                << ")\n";
      return false;
    }
    std::vector<float> readback(strided_opsin.size());
    if (!CheckStatus(gpu.CopyDeviceToHost(*device_strided_opsin,
          readback.data(), strided_bytes), "Check strided input guards")) {
      return false;
    }
    for (size_t i = 0; i < readback.size(); ++i) {
      if (!(readback[i] == strided_opsin[i] ||
            (std::isnan(readback[i]) && std::isnan(strided_opsin[i])))) {
        std::cerr << "Candidate evaluation modified strided input\n";
        return false;
      }
    }
    std::vector<float> mask_readback(strided_mask.size());
    if (!CheckStatus(gpu.CopyDeviceToHost(*device_strided_mask,
          mask_readback.data(), strided_mask_bytes), "Check strided mask guards")) {
      return false;
    }
    for (size_t i = 0; i < mask_readback.size(); ++i) {
      if (!(mask_readback[i] == strided_mask[i] ||
            (std::isnan(mask_readback[i]) && std::isnan(strided_mask[i])))) {
        std::cerr << "Candidate evaluation modified strided mask\n";
        return false;
      }
    }
  }

  // A bad mask pixel must poison every candidate covering it, but no others.
  // Exercise the first/last pixels and an interior pixel across every shape.
  constexpr std::array<std::array<size_t, 2>, 3> bad_positions{{
    {0, 0}, {kPixelExtent.width - 1, kPixelExtent.height - 1}, {31, 27}}};
  for (const auto position : bad_positions) {
    for (const float bad : {0.0f, -1.0f,
         std::numeric_limits<float>::quiet_NaN(),
         std::numeric_limits<float>::infinity()}) {
      std::vector<float> invalid_mask = fixture.pixel_mask;
      invalid_mask[position[1] * kPixelExtent.width + position[0]] = bad;
      if (!CheckStatus(gpu.CopyHostToDevice(*device_mask,
            invalid_mask.data(), mask_bytes), "Upload invalid mask") ||
          !CheckStatus(gjxl::EvaluateAcStrategyCandidates(gpu, batch, &submission),
            "Submit invalid mask batch") ||
          submission == nullptr ||
          !CheckStatus(submission->Wait(), "Wait for invalid mask batch") ||
          !CheckStatus(gpu.CopyDeviceToHost(*device_costs,
            poisoned_costs.data(), cost_bytes), "Download invalid mask costs")) {
        return false;
      }
      for (size_t i = 0; i < candidates.size(); ++i) {
        const size_t x = candidates[i].block_x * 8;
        const size_t y = candidates[i].block_y * 8;
        const bool covers = position[0] >= x && position[1] >= y &&
          position[0] < x + info->pixel_extent().width &&
          position[1] < y + info->pixel_extent().height;
        if (covers ? !std::isnan(poisoned_costs[i])
                   : poisoned_costs[i] != complete_costs[i]) {
          std::cerr << implementation << ' ' << info->name
                    << " invalid mask affected the wrong candidate " << i << '\n';
          return false;
        }
      }
    }
  }
  if (!CheckStatus(gpu.CopyHostToDevice(*device_mask,
        fixture.pixel_mask.data(), mask_bytes), "Restore valid mask")) {
    return false;
  }
#endif

  std::vector<gjxl::AcStrategyCandidate> resident_candidates = candidates;
  for (gjxl::AcStrategyCandidate& candidate : resident_candidates) {
    candidate.quant_norm = 1.0f;
  }
  const size_t plane_elements = kPixelExtent.width * kPixelExtent.height;
  gjxl::AcStrategyCandidateBatch resident_batch = batch;
  resident_batch.resident_opsin = gjxl::ConstDeviceImage3View{{{
    {device_opsin.get(), 0, gjxl::DeviceElementType::kF32,
     kPixelExtent, kPixelExtent.width},
    {device_opsin.get(), plane_elements * sizeof(float),
     gjxl::DeviceElementType::kF32, kPixelExtent, kPixelExtent.width},
    {device_opsin.get(), 2 * plane_elements * sizeof(float),
     gjxl::DeviceElementType::kF32, kPixelExtent, kPixelExtent.width},
  }}};
  resident_batch.resident_pixel_mask = {
    device_mask.get(), 0, gjxl::DeviceElementType::kF32,
    kPixelExtent, kPixelExtent.width};
  resident_batch.resident_quant_field = {
    device_quant.get(), 0, gjxl::DeviceElementType::kF32,
    kBlockExtent, kBlockExtent.width};
  std::ranges::fill(poisoned_costs,
                    std::numeric_limits<float>::quiet_NaN());
  if (!CheckStatus(
        gpu.CopyHostToDevice(
          *device_candidates, resident_candidates.data(), candidate_bytes),
        "Upload resident candidates") ||
      !CheckStatus(
        gpu.CopyHostToDevice(
          *device_costs, poisoned_costs.data(), cost_bytes),
        "Poison resident costs") ||
      !CheckStatus(
        gjxl::EvaluateAcStrategyCandidates(
          gpu, resident_batch, &submission),
        "Submit resident candidate batch") ||
      submission == nullptr ||
      !CheckStatus(submission->Wait(), "Wait for resident candidate batch") ||
      !CheckStatus(
        gpu.CopyDeviceToHost(
          *device_costs, poisoned_costs.data(), cost_bytes),
        "Download resident candidate costs")) {
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    const double error = std::abs(
        static_cast<double>(poisoned_costs[i]) - expected[i]);
    if (!std::isfinite(poisoned_costs[i]) ||
        error > 0.005 + 1.0e-6 * std::abs(expected[i])) {
      std::cerr << implementation << ' ' << info->name
                << " resident quant-norm mismatch at candidate " << i
                << ": expected " << expected[i]
                << ", got " << poisoned_costs[i] << '\n';
      return false;
    }
  }

#ifdef GJXL_TEST_CUDA
  // A device CfL map supersedes descriptor factors, even non-finite ones.
  // All candidates in this fixture share one 64x64 color tile.
  const std::array<signed char, 2> cfl_map{-7, 0};
  std::unique_ptr<gjxl::DeviceBuffer> device_cfl;
  if (!Allocate(gpu, cfl_map.size(), "Allocate CfL map", &device_cfl) ||
      !CheckStatus(gpu.CopyHostToDevice(*device_cfl, cfl_map.data(),
        cfl_map.size()), "Upload CfL map")) {
    return false;
  }
  std::vector<float> host_cfl_costs;
  for (const bool device_factors : {false, true}) {
    auto cfl_candidates = resident_candidates;
    for (auto& candidate : cfl_candidates) {
      candidate.cfl_x = device_factors
        ? std::numeric_limits<float>::quiet_NaN()
        : static_cast<float>(cfl_map[0]) * (1.0f / 84.0f);
      candidate.cfl_b = device_factors
        ? std::numeric_limits<float>::infinity()
        : 1.0f + static_cast<float>(cfl_map[1]) * (1.0f / 84.0f);
    }
    auto cfl_batch = resident_batch;
    if (device_factors) {
      cfl_batch.resident_y_to_x = {
        device_cfl.get(), 0, gjxl::DeviceElementType::kI8, {1, 1}, 1};
      cfl_batch.resident_y_to_b = {
        device_cfl.get(), 1, gjxl::DeviceElementType::kI8, {1, 1}, 1};
    }
    if (!CheckStatus(gpu.CopyHostToDevice(*device_candidates,
          cfl_candidates.data(), candidate_bytes), "Upload CfL candidates") ||
        !CheckStatus(gjxl::EvaluateAcStrategyCandidates(
          gpu, cfl_batch, &submission), "Submit CfL candidate batch") ||
        submission == nullptr ||
        !CheckStatus(submission->Wait(), "Wait for CfL candidate batch") ||
        !CheckStatus(gpu.CopyDeviceToHost(*device_costs,
          poisoned_costs.data(), cost_bytes), "Download CfL costs")) {
      return false;
    }
    if (!device_factors) {
      host_cfl_costs = poisoned_costs;
    } else if (poisoned_costs != host_cfl_costs) {
      std::cerr << implementation << ' ' << info->name
                << " device CfL map did not supersede descriptor factors\n";
      return false;
    }
  }
  if (!CheckStatus(gpu.CopyHostToDevice(*device_candidates, candidates.data(),
        candidate_bytes), "Restore candidates")) {
    return false;
  }
#endif

  const gjxl::GpuBackendStats before_submissions = gpu.stats();
  std::unique_ptr<gjxl::GpuSubmission> first_submission;
  std::unique_ptr<gjxl::GpuSubmission> second_submission;
  if (!CheckStatus(
        gjxl::EvaluateAcStrategyCandidates(
          gpu, batch, &first_submission),
        "Submit first independent candidate batch") ||
      !CheckStatus(
        gjxl::EvaluateAcStrategyCandidates(
          gpu, batch, &second_submission),
        "Submit second independent candidate batch") ||
      first_submission == nullptr || second_submission == nullptr ||
      !CheckStatus(
        second_submission->Wait(), "Wait for second candidate batch") ||
      !CheckStatus(
        first_submission->Wait(), "Wait for first candidate batch") ||
      !CheckStatus(
        first_submission->Wait(), "Repeat first candidate wait") ||
      gpu.stats().committed_submissions !=
        before_submissions.committed_submissions + 2) {
    std::cerr << implementation << ' ' << info->name
              << " did not return independent submission handles\n";
    return false;
  }

  candidates[0].block_x = std::numeric_limits<uint32_t>::max();
  candidates[1].quant_norm = std::numeric_limits<float>::quiet_NaN();
  size_t invalid_count = 2;
#ifdef GJXL_TEST_CUDA
  candidates[2].block_y = std::numeric_limits<uint32_t>::max();
  candidates[3].quant_norm = 0.0f;
  candidates[4].entropy_multiplier = std::numeric_limits<float>::infinity();
  candidates[5].entropy_multiplier = -1.0f;
  candidates[6].cfl_x = std::numeric_limits<float>::quiet_NaN();
  candidates[7].cfl_b = std::numeric_limits<float>::infinity();
  candidates[8].block_x = static_cast<uint32_t>(
    kBlockExtent.width - info->covered_blocks.width + 1);
  candidates[9].block_y = static_cast<uint32_t>(
    kBlockExtent.height - info->covered_blocks.height + 1);
  invalid_count = 10;
#endif
  if (!CheckStatus(
        gpu.CopyHostToDevice(
          *device_candidates, candidates.data(), candidate_bytes),
        "Upload invalid candidates") ||
      !CheckStatus(
        gjxl::EvaluateAcStrategyCandidates(gpu, batch, &submission),
        "Submit invalid candidate batch") ||
      submission == nullptr ||
      !CheckStatus(submission->Wait(), "Wait for invalid candidate batch") ||
      !CheckStatus(
        gpu.CopyDeviceToHost(
          *device_costs, poisoned_costs.data(), cost_bytes),
        "Download invalid candidate costs")) {
    std::cerr << implementation << ' ' << info->name
              << " invalid device descriptor did not produce NaN\n";
    return false;
  }

  for (size_t i = 0; i < invalid_count; ++i) {
    if (!std::isnan(poisoned_costs[i])) {
      std::cerr << implementation << ' ' << info->name
                << " invalid device descriptor " << i << " did not produce NaN\n";
      return false;
    }
  }
  for (size_t i = invalid_count; i < candidates.size(); ++i) {
    const double error =
      std::abs(static_cast<double>(poisoned_costs[i]) - expected[i]);
    if (!std::isfinite(poisoned_costs[i]) ||
        error > 0.005 + 1.0e-6 * std::abs(expected[i])) {
      std::cerr << implementation << ' ' << info->name
                << " invalid descriptor contaminated another candidate\n";
      return false;
    }
  }

  gjxl::AcStrategyCandidateBatch aliased_batch = batch;
  aliased_batch.scratch_b = aliased_batch.scratch_a;
  if (gjxl::EvaluateAcStrategyCandidates(
        gpu, aliased_batch, &submission).ok() || submission != nullptr) {
    std::cerr << implementation << ' ' << info->name
              << " aliased scratch buffers were accepted\n";
    return false;
  }
  return true;
}

#ifndef GJXL_TEST_CUDA
gjxl::MetalBackendOptions OptionsFor(
  gjxl::MetalDctImplementation implementation) {

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

bool CheckImplementation(
  gjxl::MetalDctImplementation implementation,
  std::string_view name,
  const Fixture& fixture) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(
        gjxl::CreateMetalBackend(
          GJXL_METALLIB_PATH,
          OptionsFor(implementation),
          &gpu),
        std::string("Create ") + std::string(name) + " backend")) {
    return false;
  }
  for (const gjxl::AcStrategyType strategy : kStrategies) {
    if (!RunStrategyCase(*gpu, name, strategy, fixture)) {
      return false;
    }
  }
  return true;
}
#else
bool CheckImplementation(
  std::string_view name,
  const Fixture& fixture) {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(
        gjxl::CreateCudaBackend(&gpu),
        std::string("Create ") + std::string(name) + " backend")) {
    return false;
  }
  for (const gjxl::AcStrategyType strategy : kStrategies) {
    if (!RunStrategyCase(*gpu, name, strategy, fixture)) return false;
  }
  return true;
}
#endif

bool CheckValidation() {
  BackendWithoutAc without_ac;
  std::unique_ptr<gjxl::GpuSubmission> submission;
  gjxl::AcStrategyCandidateBatch nonempty;
  nonempty.candidate_count = 1;
  const gjxl::Status unavailable = gjxl::EvaluateAcStrategyCandidates(
    without_ac, nonempty, &submission);
  if (gjxl::QueryGpuAcStrategyEvaluation(without_ac) != nullptr ||
      unavailable.code() != gjxl::StatusCode::kUnavailable ||
      submission != nullptr ||
      gjxl::EvaluateAcStrategyCandidates(
        without_ac, nonempty, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Optional AC-strategy capability contract failed\n";
    return false;
  }

  std::unique_ptr<gjxl::GpuBackend> gpu;
#ifdef GJXL_TEST_CUDA
  if (!CheckStatus(
        gjxl::CreateCudaBackend(&gpu),
        "Create validation backend")) {
#else
  if (!CheckStatus(
        gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
        "Create validation backend")) {
#endif
    return false;
  }

  if (gjxl::QueryGpuAcStrategyEvaluation(*gpu) == nullptr ||
      !gjxl::EvaluateAcStrategyCandidates(*gpu, {}, &submission).ok() ||
      submission != nullptr) {
    std::cerr << "Zero-sized candidate batch was rejected\n";
    return false;
  }
#ifdef GJXL_TEST_CUDA
  const auto before_query = gpu->stats();
  for (const auto strategy : kStrategies) {
    gjxl::AcStrategyScratchRequirements scratch{1, 2, 3};
    if (!CheckStatus(gjxl::GetAcStrategyScratchRequirements(
          *gpu, strategy, 0, &scratch), "Query empty scratch") ||
        scratch.scratch_a_bytes != 0 || scratch.scratch_b_bytes != 0 ||
        scratch.rate_scratch_bytes != 0) return false;
    scratch = {1, 2, 3};
    if (gjxl::GetAcStrategyScratchRequirements(*gpu, strategy,
          std::numeric_limits<size_t>::max(), &scratch).code() !=
          gjxl::StatusCode::kInvalidArgument ||
        scratch.scratch_a_bytes != 1 || scratch.scratch_b_bytes != 2 ||
        scratch.rate_scratch_bytes != 3) return false;
  }
  gjxl::AcStrategyScratchRequirements untouched{1, 2, 3};
  if (gjxl::GetAcStrategyScratchRequirements(*gpu,
        gjxl::AcStrategyType::kDct64x64, 0, &untouched).code() !=
        gjxl::StatusCode::kUnavailable ||
      gjxl::GetAcStrategyScratchRequirements(*gpu,
        gjxl::AcStrategyType::kCount, 0, &untouched).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      gjxl::GetAcStrategyScratchRequirements(*gpu,
        gjxl::AcStrategyType::kDct8, 1, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      untouched.scratch_a_bytes != 1 || untouched.scratch_b_bytes != 2 ||
      untouched.rate_scratch_bytes != 3 ||
      gpu->stats().successful_allocations != before_query.successful_allocations ||
      gpu->stats().committed_submissions != before_query.committed_submissions) {
    std::cerr << "CUDA scratch query contract failed\n";
    return false;
  }
#endif
  gjxl::AcStrategyCandidateBatch invalid;
  invalid.candidate_count = 1;
  if (gjxl::EvaluateAcStrategyCandidates(
        *gpu, invalid, &submission).ok() || submission != nullptr) {
    std::cerr << "Candidate batch with missing buffers was accepted\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
#ifdef GJXL_TEST_CUDA
  std::unique_ptr<gjxl::GpuBackend> probe;
  const gjxl::Status create = gjxl::CreateCudaBackend(&probe);
  if (!create.ok()) {
    std::cerr << "CUDA backend unavailable: " << create.message() << '\n';
    return 77;
  }
#endif
  const Fixture fixture;
#ifdef GJXL_TEST_CUDA
  if (!CheckValidation() ||
      !CheckImplementation("CUDA", fixture)) {
    return EXIT_FAILURE;
  }
  std::cout << "All CUDA AC-strategy candidate tests passed.\n";
#else
  if (!CheckValidation() ||
      !CheckImplementation(
        gjxl::MetalDctImplementation::kScalarMatmul,
        "scalar matmul",
        fixture) ||
      !CheckImplementation(
        gjxl::MetalDctImplementation::kSimdgroupMatmul,
        "simdgroup matmul",
        fixture) ||
      !CheckImplementation(
        gjxl::MetalDctImplementation::kFactoredRadix2,
        "factored radix-2",
        fixture)) {
    return EXIT_FAILURE;
  }

  std::cout << "All Metal AC-strategy candidate tests passed.\n";
#endif
  return EXIT_SUCCESS;
}
