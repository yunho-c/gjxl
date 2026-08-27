// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

#include "codec/quantization.h"
#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/aq_evaluation.h"

namespace {

constexpr gjxl::Extent2D kSourceExtent{89, 57};
constexpr gjxl::Extent2D kCodingExtent{96, 64};
constexpr gjxl::Extent2D kBlockExtent{12, 8};
constexpr gjxl::Extent2D kTileExtent{2, 1};
constexpr uint32_t kPoisonBits = 0x7fc12345u;
constexpr float kPoison = std::bit_cast<float>(kPoisonBits);

bool CheckStatus(gjxl::Status status, std::string_view operation) {
  if (status.ok()) return true;
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

bool ExpectCode(
  gjxl::Status status,
  gjxl::StatusCode expected,
  std::string_view operation) {

  if (status.code() == expected) return true;
  std::cerr << operation << " returned code "
            << static_cast<int>(status.code()) << ", expected "
            << static_cast<int>(expected) << ": " << status.message() << '\n';
  return false;
}

struct HostImage3F {
  gjxl::Extent2D extent;
  size_t stride = 0;
  std::array<std::vector<float>, 3> plane;

  static HostImage3F Make(
    gjxl::Extent2D extent,
    size_t stride,
    float seed) {

    HostImage3F image{.extent = extent, .stride = stride};
    for (size_t channel = 0; channel < 3; ++channel) {
      image.plane[channel].assign(stride * extent.height, -9999.0f);
      for (size_t y = 0; y < extent.height; ++y) {
        for (size_t x = 0; x < extent.width; ++x) {
          image.plane[channel][y * stride + x] =
            seed + static_cast<float>(channel) * 0.17f +
            static_cast<float>(x) * 0.0025f +
            static_cast<float>(y) * 0.004f;
        }
      }
    }
    return image;
  }

  gjxl::ConstImage3FView View() const {
    return {{{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }

  void OverwriteLogical(float value) {
    for (std::vector<float>& values : plane) {
      for (size_t y = 0; y < extent.height; ++y) {
        std::fill_n(values.data() + y * stride, extent.width, value);
      }
    }
  }
};

struct EvaluationInputStorage {
  size_t raw_stride = kBlockExtent.width + 3;
  size_t sigma_stride = kBlockExtent.width + 5;
  size_t color_stride = kTileExtent.width + 4;
  std::vector<int32_t> raw_quant;
  std::vector<float> inverse_sigma;
  std::vector<int8_t> y_to_x;
  std::vector<int8_t> y_to_b;
  gjxl::QuantizerParams quantizer;

  static EvaluationInputStorage Make(size_t variant) {
    EvaluationInputStorage storage;
    storage.raw_quant.assign(
      storage.raw_stride * kBlockExtent.height, -123456);
    storage.inverse_sigma.assign(
      storage.sigma_stride * kBlockExtent.height, 12345.0f);
    storage.y_to_x.assign(storage.color_stride * kTileExtent.height, 99);
    storage.y_to_b.assign(storage.color_stride * kTileExtent.height, 99);
    for (size_t y = 0; y < kBlockExtent.height; ++y) {
      for (size_t x = 0; x < kBlockExtent.width; ++x) {
        storage.raw_quant[y * storage.raw_stride + x] =
          1 + static_cast<int32_t>((x * 17 + y * 11 + variant * 23) % 256);
        storage.inverse_sigma[y * storage.sigma_stride + x] =
          -0.05f - static_cast<float>(x + 3 * y + variant) * 0.001f;
      }
    }
    for (size_t x = 0; x < kTileExtent.width; ++x) {
      storage.y_to_x[x] = static_cast<int8_t>(-11 + x * 15 + variant);
      storage.y_to_b[x] = static_cast<int8_t>(7 - x * 9 - variant);
    }
    storage.quantizer = {
      1000u + static_cast<uint32_t>(variant * 137),
      31u + static_cast<uint32_t>(variant * 7),
    };
    return storage;
  }

  gjxl::AqEvaluationInput View() const {
    return {
      .raw_quant_field = {
        raw_quant.data(), kBlockExtent, raw_stride},
      .quantizer = quantizer,
      .y_to_x = {y_to_x.data(), kTileExtent, color_stride},
      .y_to_b = {y_to_b.data(), kTileExtent, color_stride},
      .epf_inverse_sigma = {
        inverse_sigma.data(), kBlockExtent, sigma_stride},
    };
  }
};

struct EvaluationOutputStorage {
  size_t stride = kBlockExtent.width + 7;
  std::vector<float> map =
    std::vector<float>(stride * kBlockExtent.height, kPoison);
  double score = -987654.25;

  gjxl::AqEvaluationOutput View() {
    return {{map.data(), kBlockExtent, stride}, &score};
  }

  bool Poisoned() const {
    return std::ranges::all_of(map, [](float value) {
      return std::bit_cast<uint32_t>(value) == kPoisonBits;
    }) && score == -987654.25;
  }

  bool PaddingPoisoned() const {
    for (size_t y = 0; y < kBlockExtent.height; ++y) {
      for (size_t x = kBlockExtent.width; x < stride; ++x) {
        if (std::bit_cast<uint32_t>(map[y * stride + x]) != kPoisonBits) {
          return false;
        }
      }
    }
    return true;
  }
};

gjxl::AqEvaluationOptions MakeOptions() {
  gjxl::AqEvaluationOptions options;
  options.coefficient_coding = {1.25f, 0.75f};
  options.opsin_intensity_target = 255.0f;
  options.butteraugli = {0.91f, 1.07f, 80.0f};
  options.loop_filter.gaborish = true;
  options.loop_filter.epf_options.iterations = 3;
  options.loop_filter.epf_options.channel_scale = {39.0f, 5.5f, 3.25f};
  return options;
}

bool MakeMixedStrategies(gjxl::AcStrategyGrid* strategies) {
  if (!CheckStatus(
        gjxl::AcStrategyGrid::Create(kBlockExtent, strategies),
        "strategy-grid creation") ||
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

float OptionProbeValue(const gjxl::AqEvaluationOptions& options) {
  float value = options.coefficient_coding.x_matrix_multiplier * 0.0625f;
  value += options.coefficient_coding.b_matrix_multiplier * 0.03125f;
  value += options.opsin_intensity_target * (1.0f / 1024.0f);
  value += options.butteraugli.hf_asymmetry * (1.0f / 64.0f);
  value += options.butteraugli.x_multiplier * (1.0f / 128.0f);
  value += options.butteraugli.intensity_target * (1.0f / 2048.0f);
  value += options.loop_filter.gaborish ? 0.25f : 0.0f;
  value += static_cast<float>(options.loop_filter.epf_options.iterations) *
    (1.0f / 64.0f);
  return value;
}

bool ComputeOracle(
  const HostImage3F& original,
  const HostImage3F& coding,
  const gjxl::AcStrategyGrid& strategies,
  const gjxl::AqEvaluationOptions& options,
  const EvaluationInputStorage& input,
  std::vector<float>* expected,
  float* expected_score) {

  expected->resize(kBlockExtent.width * kBlockExtent.height);
  *expected_score = -std::numeric_limits<float>::infinity();
  for (size_t block_y = 0; block_y < kBlockExtent.height; ++block_y) {
    for (size_t block_x = 0; block_x < kBlockExtent.width; ++block_x) {
      gjxl::AcStrategyCell cell;
      if (!CheckStatus(
            strategies.Get(block_x, block_y, &cell),
            "oracle strategy lookup")) {
        return false;
      }
      gjxl::QuantizationMatrixView matrix;
      if (!CheckStatus(gjxl::GetDefaultQuantizationMatrix(
            cell.strategy, gjxl::XybChannel::kX, &matrix),
            "oracle quantization matrix")) {
        return false;
      }
      const size_t source_x = std::min(block_x * 8, kSourceExtent.width - 1);
      const size_t source_y = std::min(block_y * 8, kSourceExtent.height - 1);
      const size_t coding_x = block_x * 8;
      const size_t coding_y = block_y * 8;
      const size_t color_x = std::min(block_x / 8, kTileExtent.width - 1);
      const size_t color_y = std::min(block_y / 8, kTileExtent.height - 1);

      float value = original.plane[0][source_y * original.stride + source_x];
      value += original.plane[1][source_y * original.stride + source_x] * 0.5f;
      value += original.plane[2][source_y * original.stride + source_x] * 0.25f;
      value += coding.plane[0][coding_y * coding.stride + coding_x] * 0.125f;
      value += coding.plane[1][coding_y * coding.stride + coding_x] * 0.0625f;
      value += coding.plane[2][coding_y * coding.stride + coding_x] * 0.03125f;
      value += static_cast<float>(
        input.raw_quant[block_y * input.raw_stride + block_x]) *
        (1.0f / 256.0f);
      value += input.inverse_sigma[
        block_y * input.sigma_stride + block_x];
      value += static_cast<float>(cell.strategy) * (1.0f / 32.0f);
      value += static_cast<float>(cell.is_anchor) * 0.5f;
      value += static_cast<float>(
        input.y_to_x[color_y * input.color_stride + color_x]) *
        (1.0f / 256.0f);
      value += static_cast<float>(
        input.y_to_b[color_y * input.color_stride + color_x]) *
        (1.0f / 256.0f);
      value += matrix.dequant[0];
      value += matrix.inverse_dequant[1] * (1.0f / 256.0f);
      value += static_cast<float>(input.quantizer.global_scale) *
        (1.0f / 65536.0f);
      value += static_cast<float>(input.quantizer.quant_dc) *
        (1.0f / 65536.0f);
      value += OptionProbeValue(options);
      (*expected)[block_y * kBlockExtent.width + block_x] = value;
      *expected_score = std::max(*expected_score, value);
    }
  }
  return true;
}

bool CompareOutput(
  const EvaluationOutputStorage& output,
  const std::vector<float>& expected,
  float expected_score,
  std::string_view label) {

  float max_error = 0.0f;
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      const float actual = output.map[y * output.stride + x];
      const float oracle = expected[y * kBlockExtent.width + x];
      const float error = std::abs(actual - oracle);
      max_error = std::max(max_error, error);
      if (!std::isfinite(actual) || error > 2.0e-5f) {
        std::cerr << label << " mismatch at " << x << ',' << y
                  << ": expected " << oracle << ", got " << actual << '\n';
        return false;
      }
    }
  }
  if (std::abs(output.score - static_cast<double>(expected_score)) > 2.0e-5 ||
      !output.PaddingPoisoned()) {
    std::cerr << label << " score or output padding mismatch\n";
    return false;
  }
  std::cout << label << " max error: " << max_error << '\n';
  return true;
}

gjxl::AqEvaluationPreparation MakePreparation(
  const HostImage3F& original,
  const HostImage3F& coding,
  const gjxl::AcStrategyGrid& strategies,
  const gjxl::AqEvaluationOptions& options) {

  return {
    original.View(), coding.View(), &strategies, options,
  };
}

bool Prepare(
  gjxl::GpuBackend& gpu,
  const HostImage3F& original,
  const HostImage3F& coding,
  const gjxl::AcStrategyGrid& strategies,
  const gjxl::AqEvaluationOptions& options,
  std::unique_ptr<gjxl::PreparedAqEvaluation>* prepared) {

  return CheckStatus(gjxl::PrepareAqEvaluation(
    gpu, MakePreparation(original, coding, strategies, options), prepared),
    "Metal AQ preparation");
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

bool CheckPublicAndPreparationValidation(
  gjxl::GpuBackend& gpu,
  const HostImage3F& original,
  const HostImage3F& coding,
  const gjxl::AcStrategyGrid& strategies,
  const gjxl::AqEvaluationOptions& options) {

  BackendWithoutAq without_aq;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (gjxl::QueryGpuAqEvaluation(without_aq) != nullptr ||
      !ExpectCode(gjxl::PrepareAqEvaluation(
        without_aq, MakePreparation(original, coding, strategies, options),
        &prepared), gjxl::StatusCode::kUnavailable,
        "missing AQ capability") ||
      prepared != nullptr ||
      !ExpectCode(gjxl::PrepareAqEvaluation(
        gpu, MakePreparation(original, coding, strategies, options), nullptr),
        gjxl::StatusCode::kInvalidArgument,
        "null prepared output")) {
    return false;
  }

  const uint64_t allocations = gpu.stats().successful_allocations;
  gjxl::AcStrategyGrid incomplete;
  if (!CheckStatus(
        gjxl::AcStrategyGrid::Create(kBlockExtent, &incomplete),
        "incomplete strategy creation")) {
    return false;
  }
  if (!ExpectCode(gjxl::PrepareAqEvaluation(
        gpu, MakePreparation(original, coding, incomplete, options), &prepared),
        gjxl::StatusCode::kInvalidArgument, "incomplete strategies") ||
      prepared != nullptr) {
    return false;
  }

  gjxl::AcStrategyGrid unsupported;
  if (!CheckStatus(
        gjxl::AcStrategyGrid::Create(kBlockExtent, &unsupported),
        "unsupported strategy creation") ||
      !CheckStatus(unsupported.Set(
        0, 0, gjxl::AcStrategyType::kIdentity),
        "unsupported strategy placement")) {
    return false;
  }
  unsupported.fill_empty_dct8();
  if (!ExpectCode(gjxl::PrepareAqEvaluation(
        gpu, MakePreparation(original, coding, unsupported, options), &prepared),
        gjxl::StatusCode::kInvalidArgument, "unsupported strategies")) {
    return false;
  }

  gjxl::AqEvaluationOptions bad_options = options;
  bad_options.loop_filter.epf_options.pass0_sigma_scale =
    std::numeric_limits<float>::quiet_NaN();
  if (!ExpectCode(gjxl::PrepareAqEvaluation(
        gpu, MakePreparation(original, coding, strategies, bad_options),
        &prepared), gjxl::StatusCode::kInvalidArgument, "invalid AQ options")) {
    return false;
  }

  HostImage3F non_finite = original;
  non_finite.plane[2][3 * non_finite.stride + 5] =
    std::numeric_limits<float>::infinity();
  if (!ExpectCode(gjxl::PrepareAqEvaluation(
        gpu, MakePreparation(non_finite, coding, strategies, options),
        &prepared), gjxl::StatusCode::kInvalidArgument,
        "non-finite prepared image") ||
      gpu.stats().successful_allocations != allocations) {
    std::cerr << "Rejected preparation allocated device storage\n";
    return false;
  }

  using gjxl::metal_internal::ValidateMetalAqGeometryForTesting;
  const size_t u32_max = std::numeric_limits<uint32_t>::max();
  return CheckStatus(
           ValidateMetalAqGeometryForTesting(
             {u32_max - 14, 1}, {u32_max - 7, 8}),
           "maximum representable AQ geometry") &&
         ExpectCode(
           ValidateMetalAqGeometryForTesting(
             {u32_max - 6, 1}, {u32_max + size_t{1}, 8}),
           gjxl::StatusCode::kInvalidArgument,
           "overflowing AQ geometry") &&
         ExpectCode(
           ValidateMetalAqGeometryForTesting({89, 57}, {104, 64}),
           gjxl::StatusCode::kInvalidArgument,
           "excessive AQ padding");
}

bool CheckRepeatedProbe(
  gjxl::GpuBackend& gpu,
  HostImage3F* original,
  HostImage3F* coding,
  const gjxl::AcStrategyGrid& strategies,
  const gjxl::AqEvaluationOptions& options) {

  const HostImage3F original_snapshot = *original;
  const HostImage3F coding_snapshot = *coding;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Prepare(
        gpu, *original, *coding, strategies, options, &prepared)) {
    return false;
  }
  const gjxl::AqEvaluationMemoryStats memory = prepared->memory_stats();
  if (memory.persistent_bytes == 0 || memory.staging_bytes == 0 ||
      memory.peak_scratch_bytes == 0) {
    std::cerr << "Prepared AQ memory accounting is empty\n";
    return false;
  }
  std::cout << "prepared AQ persistent=" << memory.persistent_bytes
            << " staging=" << memory.staging_bytes
            << " peak scratch=" << memory.peak_scratch_bytes << '\n';

  original->OverwriteLogical(777.0f);
  coding->OverwriteLogical(-333.0f);

  EvaluationInputStorage warmup_input = EvaluationInputStorage::Make(0);
  EvaluationOutputStorage production_output;
  const uint64_t before_unavailable = gpu.stats().committed_submissions;
  if (!ExpectCode(prepared->Evaluate(
        warmup_input.View(), production_output.View()),
        gjxl::StatusCode::kUnavailable, "Milestone 2 production Evaluate") ||
      !production_output.Poisoned() ||
      gpu.stats().committed_submissions != before_unavailable) {
    std::cerr << "Unavailable production Evaluate changed state\n";
    return false;
  }

  EvaluationOutputStorage warmup_output;
  if (!CheckStatus(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, warmup_input.View(), warmup_output.View()),
        "AQ contract-probe warmup")) {
    return false;
  }
  std::vector<float> expected;
  float expected_score = 0.0f;
  if (!ComputeOracle(
        original_snapshot, coding_snapshot, strategies, options, warmup_input,
        &expected, &expected_score) ||
      !CompareOutput(warmup_output, expected, expected_score, "AQ warmup")) {
    return false;
  }

  const gjxl::GpuBackendStats before = gpu.stats();
  for (size_t variant = 1; variant <= 3; ++variant) {
    EvaluationInputStorage input = EvaluationInputStorage::Make(variant);
    EvaluationOutputStorage output;
    if (!CheckStatus(
          gjxl::metal_internal::RunMetalAqContractProbeForTesting(
            *prepared, input.View(), output.View()),
          "measured AQ contract probe") ||
        !ComputeOracle(
          original_snapshot, coding_snapshot, strategies, options, input,
          &expected, &expected_score) ||
        !CompareOutput(output, expected, expected_score, "measured AQ probe")) {
      return false;
    }
    const gjxl::GpuBackendStats after = gpu.stats();
    if (after.successful_allocations != before.successful_allocations ||
        after.committed_submissions !=
          before.committed_submissions + variant) {
      std::cerr << "Repeated AQ probe allocated or used multiple submissions\n";
      return false;
    }
  }

  EvaluationInputStorage invalid = EvaluationInputStorage::Make(4);
  invalid.raw_quant[0] = 0;
  EvaluationOutputStorage rejected_output;
  const uint64_t before_rejection = gpu.stats().committed_submissions;
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, invalid.View(), rejected_output.View()),
        gjxl::StatusCode::kInvalidArgument, "invalid raw quant") ||
      !rejected_output.Poisoned() ||
      gpu.stats().committed_submissions != before_rejection) {
    return false;
  }
  invalid = EvaluationInputStorage::Make(4);
  invalid.inverse_sigma[0] = 0.0f;
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, invalid.View(), rejected_output.View()),
        gjxl::StatusCode::kInvalidArgument, "invalid inverse sigma")) {
    return false;
  }
  invalid = EvaluationInputStorage::Make(4);
  invalid.quantizer.global_scale = 0;
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, invalid.View(), rejected_output.View()),
        gjxl::StatusCode::kInvalidArgument, "invalid quantizer")) {
    return false;
  }
  gjxl::AqEvaluationOutput null_score = rejected_output.View();
  null_score.score = nullptr;
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, warmup_input.View(), null_score),
        gjxl::StatusCode::kInvalidArgument, "null AQ score") ||
      gpu.stats().committed_submissions != before_rejection) {
    return false;
  }
  gjxl::AqEvaluationInput mismatched = warmup_input.View();
  mismatched.y_to_x.extent = {1, 1};
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, mismatched, rejected_output.View()),
        gjxl::StatusCode::kInvalidArgument, "mismatched AQ input") ||
      gpu.stats().committed_submissions != before_rejection) {
    return false;
  }
  gjxl::AqEvaluationInput overflowing = warmup_input.View();
  overflowing.raw_quant_field.stride =
    std::numeric_limits<size_t>::max();
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, overflowing, rejected_output.View()),
        gjxl::StatusCode::kInvalidArgument, "overflowing AQ host stride") ||
      gpu.stats().committed_submissions != before_rejection) {
    return false;
  }
  gjxl::AqEvaluationOutput mismatched_output = rejected_output.View();
  mismatched_output.block_distance_map.extent = {11, 8};
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, warmup_input.View(), mismatched_output),
        gjxl::StatusCode::kInvalidArgument, "mismatched AQ output") ||
      gpu.stats().committed_submissions != before_rejection) {
    return false;
  }

  EvaluationInputStorage outstanding_input = EvaluationInputStorage::Make(5);
  if (!CheckStatus(
        gjxl::metal_internal::SubmitMetalAqContractProbeForTesting(
          *prepared, outstanding_input.View()),
        "outstanding AQ probe submission")) {
    return false;
  }
  EvaluationOutputStorage reentrant_output;
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, warmup_input.View(), reentrant_output.View()),
        gjxl::StatusCode::kFailedPrecondition, "reentrant AQ probe") ||
      !CheckStatus(
        gjxl::metal_internal::FinishMetalAqContractProbeForTesting(
          *prepared, reentrant_output.View()),
        "outstanding AQ probe finish") ||
      !ExpectCode(
        gjxl::metal_internal::FinishMetalAqContractProbeForTesting(
          *prepared, reentrant_output.View()),
        gjxl::StatusCode::kFailedPrecondition, "duplicate AQ probe finish")) {
    return false;
  }
  return true;
}

bool CheckConcurrentPreparedObjects(
  gjxl::GpuBackend& gpu,
  const HostImage3F& original,
  const HostImage3F& coding,
  const gjxl::AcStrategyGrid& strategies,
  const gjxl::AqEvaluationOptions& options) {

  std::unique_ptr<gjxl::PreparedAqEvaluation> first;
  std::unique_ptr<gjxl::PreparedAqEvaluation> second;
  if (!Prepare(gpu, original, coding, strategies, options, &first) ||
      !Prepare(gpu, original, coding, strategies, options, &second)) {
    return false;
  }
  EvaluationInputStorage first_input = EvaluationInputStorage::Make(6);
  EvaluationInputStorage second_input = EvaluationInputStorage::Make(7);
  EvaluationOutputStorage first_output;
  EvaluationOutputStorage second_output;
  gjxl::Status first_status;
  gjxl::Status second_status;
  const gjxl::GpuBackendStats before = gpu.stats();
  std::thread first_thread([&] {
    first_status = gjxl::metal_internal::RunMetalAqContractProbeForTesting(
      *first, first_input.View(), first_output.View());
  });
  std::thread second_thread([&] {
    second_status = gjxl::metal_internal::RunMetalAqContractProbeForTesting(
      *second, second_input.View(), second_output.View());
  });
  first_thread.join();
  second_thread.join();
  if (!CheckStatus(first_status, "first concurrent AQ probe") ||
      !CheckStatus(second_status, "second concurrent AQ probe") ||
      gpu.stats().successful_allocations != before.successful_allocations ||
      gpu.stats().committed_submissions != before.committed_submissions + 2) {
    return false;
  }
  std::vector<float> expected;
  float expected_score = 0.0f;
  if (!ComputeOracle(
        original, coding, strategies, options, first_input,
        &expected, &expected_score) ||
      !CompareOutput(first_output, expected, expected_score,
                     "first concurrent AQ probe") ||
      !ComputeOracle(
        original, coding, strategies, options, second_input,
        &expected, &expected_score) ||
      !CompareOutput(second_output, expected, expected_score,
                     "second concurrent AQ probe")) {
    return false;
  }
  return true;
}

bool CheckOperationalFailure(
  gjxl::MetalBackendOptions backend_options,
  gjxl::StatusCode expected,
  const HostImage3F& original,
  const HostImage3F& coding,
  const gjxl::AcStrategyGrid& strategies,
  const gjxl::AqEvaluationOptions& options) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(gjxl::CreateMetalBackend(
        GJXL_METALLIB_PATH, backend_options, &gpu),
        "failure-injection backend creation")) {
    return false;
  }
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Prepare(
        *gpu, original, coding, strategies, options, &prepared)) {
    return false;
  }
  EvaluationInputStorage input = EvaluationInputStorage::Make(8);
  EvaluationOutputStorage output;
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, input.View(), output.View()),
        expected, "injected AQ operational failure") ||
      !output.Poisoned() ||
      !ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, input.View(), output.View()),
        gjxl::StatusCode::kFailedPrecondition,
        "reuse after AQ operational failure")) {
    return false;
  }
  return true;
}

bool CheckReadbackFailureAndDestruction(
  const HostImage3F& original,
  const HostImage3F& coding,
  const gjxl::AcStrategyGrid& strategies,
  const gjxl::AqEvaluationOptions& options) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "readback backend creation")) {
    return false;
  }
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Prepare(
        *gpu, original, coding, strategies, options, &prepared) ||
      !CheckStatus(
        gjxl::metal_internal::FailNextMetalAqReadbackForTesting(*prepared),
        "readback failure injection")) {
    return false;
  }
  EvaluationInputStorage input = EvaluationInputStorage::Make(9);
  EvaluationOutputStorage output;
  if (!ExpectCode(
        gjxl::metal_internal::RunMetalAqContractProbeForTesting(
          *prepared, input.View(), output.View()),
        gjxl::StatusCode::kDeviceError, "injected AQ readback failure") ||
      !output.Poisoned() ||
      !ExpectCode(
        prepared->Evaluate(input.View(), output.View()),
        gjxl::StatusCode::kFailedPrecondition,
        "production reuse after readback failure")) {
    return false;
  }

  if (!Prepare(
        *gpu, original, coding, strategies, options, &prepared)) {
    return false;
  }
  bool waited = false;
  if (!CheckStatus(
        gjxl::metal_internal::SetMetalAqWaitObserverForTesting(
          *prepared, &waited),
        "AQ wait observation") ||
      !CheckStatus(
        gjxl::metal_internal::SubmitMetalAqContractProbeForTesting(
          *prepared, input.View()),
        "destructor AQ probe submission")) {
    return false;
  }
  prepared.reset();
  if (!waited) {
    std::cerr << "Prepared AQ destructor did not wait for outstanding work\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  HostImage3F original = HostImage3F::Make(kSourceExtent, 97, 0.15f);
  HostImage3F coding = HostImage3F::Make(kCodingExtent, 103, -0.08f);
  gjxl::AcStrategyGrid strategies;
  const gjxl::AqEvaluationOptions options = MakeOptions();
  if (!MakeMixedStrategies(&strategies)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "Metal backend creation") ||
      gjxl::QueryGpuAqEvaluation(*gpu) == nullptr ||
      !CheckPublicAndPreparationValidation(
        *gpu, original, coding, strategies, options)) {
    return EXIT_FAILURE;
  }

  HostImage3F mutable_original = original;
  HostImage3F mutable_coding = coding;
  if (!CheckRepeatedProbe(
        *gpu, &mutable_original, &mutable_coding, strategies, options) ||
      !CheckConcurrentPreparedObjects(
        *gpu, original, coding, strategies, options) ||
      !CheckOperationalFailure(
        {.test_fail_submission = true},
        gjxl::StatusCode::kSubmissionFailed,
        original, coding, strategies, options) ||
      !CheckOperationalFailure(
        {.test_fail_completion = true},
        gjxl::StatusCode::kDeviceError,
        original, coding, strategies, options) ||
      !CheckReadbackFailureAndDestruction(
        original, coding, strategies, options)) {
    return EXIT_FAILURE;
  }

  std::cout << "Metal prepared AQ Milestone 2 tests passed\n";
  return EXIT_SUCCESS;
}
