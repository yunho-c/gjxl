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
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/ac_strategy.h"

namespace {

constexpr gjxl::Extent2D kPixelExtent{64, 64};
constexpr gjxl::Extent2D kBlockExtent{8, 8};
constexpr size_t kCandidateCount = 7;

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
    return gjxl::BackendKind::kMetal;
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
    const size_t block_x = (3 * i + 1) % position_count_x;
    const size_t block_y = (5 * i + 2) % position_count_y;
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
      !Allocate(gpu, packed_bytes, "Allocate scratch A", &scratch_a) ||
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
        "Download invalid candidate costs") ||
      !std::isnan(poisoned_costs[0]) ||
      !std::isnan(poisoned_costs[1])) {
    std::cerr << implementation << ' ' << info->name
              << " invalid device descriptor did not produce NaN\n";
    return false;
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

static_assert(
  gjxl::MetalBackendOptions{}.ac_residual_inverse ==
  gjxl::MetalAcResidualInverseMode::kFusedTuned);

gjxl::MetalBackendOptions OptionsFor(
  gjxl::MetalDctImplementation implementation,
  gjxl::MetalAcResidualInverseMode ac_residual_inverse =
    gjxl::MetalAcResidualInverseMode::kFusedTuned) {

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
    .ac_residual_inverse = ac_residual_inverse,
  };
}

bool CheckImplementation(
  gjxl::MetalDctImplementation implementation,
  std::string_view name,
  const Fixture& fixture,
  gjxl::MetalAcResidualInverseMode ac_residual_inverse =
    gjxl::MetalAcResidualInverseMode::kFusedTuned) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(
        gjxl::CreateMetalBackend(
          GJXL_METALLIB_PATH,
          OptionsFor(implementation, ac_residual_inverse),
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
  if (!CheckStatus(
        gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
        "Create validation backend")) {
    return false;
  }

  if (gjxl::QueryGpuAcStrategyEvaluation(*gpu) == nullptr ||
      !gjxl::EvaluateAcStrategyCandidates(*gpu, {}, &submission).ok() ||
      submission != nullptr) {
    std::cerr << "Zero-sized candidate batch was rejected\n";
    return false;
  }
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
  const Fixture fixture;
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
        gjxl::MetalDctImplementation::kSimdgroupMatmul,
        "simdgroup matmul compact AC residual/inverse",
        fixture,
        gjxl::MetalAcResidualInverseMode::kFusedCompact) ||
      !CheckImplementation(
        gjxl::MetalDctImplementation::kSimdgroupMatmul,
        "simdgroup matmul wide AC residual/inverse",
        fixture,
        gjxl::MetalAcResidualInverseMode::kFusedWide) ||
      !CheckImplementation(
        gjxl::MetalDctImplementation::kSimdgroupMatmul,
        "simdgroup matmul split AC residual/inverse",
        fixture,
        gjxl::MetalAcResidualInverseMode::kSplit) ||
      !CheckImplementation(
        gjxl::MetalDctImplementation::kFactoredRadix2,
        "factored radix-2",
        fixture)) {
    return EXIT_FAILURE;
  }

  std::cout << "All Metal AC-strategy candidate tests passed.\n";
  return EXIT_SUCCESS;
}
