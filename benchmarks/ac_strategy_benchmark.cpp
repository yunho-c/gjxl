// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Compares equal batched AC candidate workloads on the CPU and Metal GPU.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/quantization.h"
#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/ac_strategy.h"

namespace {

constexpr gjxl::Extent2D kPixelExtent{512, 512};
constexpr gjxl::Extent2D kBlockExtent{64, 64};
constexpr float kButteraugliTarget = 1.3f;
constexpr size_t kMinimumCandidatesPerSample = 256;
constexpr size_t kMinimumGpuBatchesPerSample = 16;
constexpr double kMaximumRelativeCostError = 0.01;

constexpr std::array kStrategies = {
  gjxl::AcStrategyType::kDct8,
  gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16,
  gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
  gjxl::AcStrategyType::kDct32x32,
};

volatile double g_checksum = 0.0;

struct ParitySummary {
  size_t evaluated_count = 0;
  size_t tight_mismatch_count = 0;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  gjxl::AcStrategyType worst_strategy = gjxl::AcStrategyType::kDct8;
  size_t worst_candidate = 0;
};

ParitySummary g_parity;

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
            0.09f +
            0.021f * std::sin(
              0.031f * static_cast<float>(x + 5 * channel + 1)) +
            0.017f * std::cos(
              0.027f * static_cast<float>(2 * y + channel + 3)) +
            0.005f * static_cast<float>((x * 11 + y * 7 + channel) % 17);
          plane[channel][y * kPixelExtent.width + x] = value;
        }
      }
      std::copy(
        plane[channel].begin(),
        plane[channel].end(),
        packed_opsin.begin() + channel * plane_size);
    }

    for (size_t i = 0; i < quant_field.size(); ++i) {
      quant_field[i] = 0.3f + 0.006f * static_cast<float>((i * 29) % 53);
    }
    for (size_t y = 0; y < kPixelExtent.height; ++y) {
      for (size_t x = 0; x < kPixelExtent.width; ++x) {
        pixel_mask[y * kPixelExtent.width + x] =
          38.0f + 0.19f * static_cast<float>((x * 5 + y * 23) % 31);
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

struct TimingStats {
  double median_ms = 0.0;
  double minimum_ms = 0.0;
  double maximum_ms = 0.0;
};

void Require(
  const gjxl::Status& status,
  std::string_view operation) {

  if (status.ok()) {
    return;
  }
  std::cerr << operation << " failed: " << status.message() << '\n';
  std::exit(EXIT_FAILURE);
}

size_t ParsePositive(
  const char* text,
  std::string_view name) {

  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (text[0] == '-' || errno == ERANGE ||
      end == text || *end != '\0' || value == 0 ||
      value > std::numeric_limits<size_t>::max()) {
    std::cerr << "Invalid " << name << ": " << text << '\n';
    std::exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(value);
}

std::vector<gjxl::AcStrategyCandidate> MakeCandidates(
  gjxl::AcStrategyType strategy,
  size_t candidate_count,
  const Fixture& fixture) {

  const gjxl::Extent2D covered =
    gjxl::GetAcStrategyInfo(strategy)->covered_blocks;
  const size_t position_count_x = kBlockExtent.width - covered.width + 1;
  const size_t position_count_y = kBlockExtent.height - covered.height + 1;
  std::vector<gjxl::AcStrategyCandidate> candidates(candidate_count);
  for (size_t i = 0; i < candidate_count; ++i) {
    const size_t block_x = (37 * i + 11) % position_count_x;
    const size_t block_y = (53 * i + 7 + i / position_count_x) %
      position_count_y;
    float quant_norm = 0.0f;
    Require(
      gjxl::ComputeAcStrategyQuantNorm(
        strategy,
        block_x,
        block_y,
        fixture.QuantField(),
        &quant_norm),
      "ComputeAcStrategyQuantNorm");
    candidates[i] = {
      .block_x = static_cast<uint32_t>(block_x),
      .block_y = static_cast<uint32_t>(block_y),
      .quant_norm = quant_norm,
      .entropy_multiplier = 0.96f + 0.01f * static_cast<float>(i % 9),
      .cfl_x = -0.14f + 0.017f * static_cast<float>(i % 13),
      .cfl_b = 0.11f - 0.013f * static_cast<float>(i % 11),
    };
  }
  return candidates;
}

std::vector<float> PackMatrices(gjxl::AcStrategyType strategy) {
  const size_t coefficient_count =
    gjxl::GetAcStrategyInfo(strategy)->coefficient_count();
  std::vector<float> matrices(
    gjxl::kAcStrategyCostMatrixCount * coefficient_count);
  for (size_t channel = 0; channel < 3; ++channel) {
    gjxl::QuantizationMatrixView matrix;
    Require(
      gjxl::GetDefaultQuantizationMatrix(
        strategy,
        static_cast<gjxl::XybChannel>(channel),
        &matrix),
      "GetDefaultQuantizationMatrix");
    std::copy(
      matrix.dequant.begin(),
      matrix.dequant.end(),
      matrices.begin() + channel * coefficient_count);
    std::copy(
      matrix.inverse_dequant.begin(),
      matrix.inverse_dequant.end(),
      matrices.begin() + (3 + channel) * coefficient_count);
  }
  return matrices;
}

gjxl::MetalBackendOptions SimdgroupOptions() {
  constexpr auto implementation =
    gjxl::MetalDctImplementation::kSimdgroupMatmul;
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

class CandidateBenchmarkCase {
public:
  CandidateBenchmarkCase(
    gjxl::GpuBackend& gpu,
    const Fixture& fixture,
    gjxl::AcStrategyType strategy,
    size_t candidate_count)
    : gpu_(gpu),
      fixture_(fixture),
      strategy_(strategy),
      candidates_(MakeCandidates(strategy, candidate_count, fixture)),
      matrices_(PackMatrices(strategy)),
      cpu_costs_(candidate_count),
      metal_costs_(candidate_count) {

    const size_t coefficient_count =
      gjxl::GetAcStrategyInfo(strategy)->coefficient_count();
    AllocateAndUpload(
      fixture.packed_opsin,
      "opsin",
      &device_opsin_);
    AllocateAndUpload(
      fixture.pixel_mask,
      "pixel mask",
      &device_mask_);
    AllocateAndUpload(matrices_, "matrices", &device_matrices_);
    AllocateAndUpload(candidates_, "candidates", &device_candidates_);

    const size_t packed_bytes =
      candidate_count * 3 * coefficient_count * sizeof(float);
    const size_t rate_bytes = candidate_count * 3 *
      gjxl::kAcStrategyRateScratchBytesPerChannel;
    const size_t cost_bytes = candidate_count * sizeof(float);
    Require(gpu_.Allocate(packed_bytes, &scratch_a_), "Allocate scratch A");
    Require(gpu_.Allocate(packed_bytes, &scratch_b_), "Allocate scratch B");
    Require(gpu_.Allocate(rate_bytes, &rate_scratch_), "Allocate rate scratch");
    Require(gpu_.Allocate(cost_bytes, &device_costs_), "Allocate costs");

    batch_ = {
      .strategy = strategy,
      .opsin = device_opsin_.get(),
      .pixel_mask = device_mask_.get(),
      .matrices = device_matrices_.get(),
      .candidates = device_candidates_.get(),
      .scratch_a = scratch_a_.get(),
      .scratch_b = scratch_b_.get(),
      .rate_scratch = rate_scratch_.get(),
      .costs = device_costs_.get(),
      .pixel_extent = kPixelExtent,
      .opsin_row_stride = kPixelExtent.width,
      .opsin_plane_stride = kPixelExtent.width * kPixelExtent.height,
      .pixel_mask_row_stride = kPixelExtent.width,
      .candidate_count = candidate_count,
      .butteraugli_target = kButteraugliTarget,
    };
  }

  void EvaluateCpu() {
    for (size_t i = 0; i < candidates_.size(); ++i) {
      const gjxl::AcStrategyCandidate& candidate = candidates_[i];
      Require(
        gjxl::EstimateAcStrategyCost(
          strategy_,
          candidate.block_x,
          candidate.block_y,
          fixture_.Image(),
          fixture_.QuantField(),
          fixture_.PixelMask(),
          {
            .butteraugli_target = kButteraugliTarget,
            .entropy_multiplier = candidate.entropy_multiplier,
            .cfl_factors = {candidate.cfl_x, 0.0f, candidate.cfl_b},
          },
          &cpu_costs_[i]),
        "CPU candidate evaluation");
    }
    g_checksum = g_checksum + cpu_costs_.front() + cpu_costs_.back();
  }

  void EvaluateMetalResident() {
    std::unique_ptr<gjxl::GpuSubmission> submission;
    Require(
      gjxl::EvaluateAcStrategyCandidates(gpu_, batch_, &submission),
      "Metal candidate submission");
    if (submission == nullptr) {
      std::cerr << "Metal candidate submission is null\n";
      std::exit(EXIT_FAILURE);
    }
    Require(submission->Wait(), "Metal candidate synchronization");
  }

  void EvaluateMetalRoundTrip() {
    const size_t candidate_bytes =
      candidates_.size() * sizeof(gjxl::AcStrategyCandidate);
    const size_t cost_bytes = metal_costs_.size() * sizeof(float);
    Require(
      gpu_.CopyHostToDevice(
        *device_candidates_, candidates_.data(), candidate_bytes),
      "Upload candidate descriptors");
    EvaluateMetalResident();
    Require(
      gpu_.CopyDeviceToHost(
        *device_costs_, metal_costs_.data(), cost_bytes),
      "Download candidate costs");
    g_checksum = g_checksum + metal_costs_.front() + metal_costs_.back();
  }

  void Validate() {
    EvaluateCpu();
    EvaluateMetalRoundTrip();
    g_parity.evaluated_count += cpu_costs_.size();
    for (size_t i = 0; i < cpu_costs_.size(); ++i) {
      const double reference = cpu_costs_[i];
      const double actual = metal_costs_[i];
      const double error = std::abs(actual - reference);
      const double relative_error = error / std::max(1.0, reference);
      if (relative_error > g_parity.max_relative_error) {
        g_parity.max_relative_error = relative_error;
        g_parity.max_absolute_error = error;
        g_parity.worst_strategy = strategy_;
        g_parity.worst_candidate = i;
      }
      if (error > 0.005 + 1.0e-6 * std::abs(reference)) {
        ++g_parity.tight_mismatch_count;
      }
      if (!std::isfinite(actual) ||
          relative_error > kMaximumRelativeCostError) {
        std::cerr << "Metal candidate result failed validation for "
                  << gjxl::GetAcStrategyInfo(strategy_)->name
                  << " at candidate " << i
                  << ": expected " << reference
                  << ", got " << actual << '\n';
        std::exit(EXIT_FAILURE);
      }
    }
  }

private:
  template <typename T>
  void AllocateAndUpload(
    const std::vector<T>& values,
    std::string_view role,
    std::unique_ptr<gjxl::DeviceBuffer>* buffer) {

    const size_t bytes = values.size() * sizeof(T);
    Require(gpu_.Allocate(bytes, buffer), std::string("Allocate ") +
      std::string(role));
    Require(
      gpu_.CopyHostToDevice(**buffer, values.data(), bytes),
      std::string("Upload ") + std::string(role));
  }

  gjxl::GpuBackend& gpu_;
  const Fixture& fixture_;
  gjxl::AcStrategyType strategy_;
  std::vector<gjxl::AcStrategyCandidate> candidates_;
  std::vector<float> matrices_;
  std::vector<float> cpu_costs_;
  std::vector<float> metal_costs_;
  std::unique_ptr<gjxl::DeviceBuffer> device_opsin_;
  std::unique_ptr<gjxl::DeviceBuffer> device_mask_;
  std::unique_ptr<gjxl::DeviceBuffer> device_matrices_;
  std::unique_ptr<gjxl::DeviceBuffer> device_candidates_;
  std::unique_ptr<gjxl::DeviceBuffer> scratch_a_;
  std::unique_ptr<gjxl::DeviceBuffer> scratch_b_;
  std::unique_ptr<gjxl::DeviceBuffer> rate_scratch_;
  std::unique_ptr<gjxl::DeviceBuffer> device_costs_;
  gjxl::AcStrategyCandidateBatch batch_;
};

template <typename Function>
double TimeRepeated(size_t repetitions, Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < repetitions; ++i) {
    function();
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() /
    static_cast<double>(repetitions);
}

TimingStats Summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t midpoint = samples.size() / 2;
  const double median = samples.size() % 2 == 0
    ? 0.5 * (samples[midpoint - 1] + samples[midpoint])
    : samples[midpoint];
  return {
    .median_ms = median,
    .minimum_ms = samples.front(),
    .maximum_ms = samples.back(),
  };
}

void PrintRow(
  std::string_view strategy,
  size_t candidate_count,
  const TimingStats& cpu,
  const TimingStats& metal_resident,
  const TimingStats& metal_roundtrip) {

  std::cout << std::left << std::setw(11) << strategy
            << std::right << std::setw(7) << candidate_count
            << std::fixed << std::setprecision(3)
            << std::setw(11) << cpu.median_ms
            << std::setw(11) << metal_resident.median_ms
            << std::setw(10) << cpu.median_ms / metal_resident.median_ms
            << std::setw(11) << metal_roundtrip.median_ms
            << std::setw(10) << cpu.median_ms / metal_roundtrip.median_ms
            << "   CPU[" << cpu.minimum_ms << ',' << cpu.maximum_ms << ']'
            << " GPU[" << metal_resident.minimum_ms << ','
            << metal_resident.maximum_ms << ']'
            << " E2E[" << metal_roundtrip.minimum_ms << ','
            << metal_roundtrip.maximum_ms << "]\n";
}

void BenchmarkCase(
  gjxl::GpuBackend& gpu,
  const Fixture& fixture,
  gjxl::AcStrategyType strategy,
  size_t candidate_count,
  size_t sample_count) {

  CandidateBenchmarkCase benchmark_case(
    gpu, fixture, strategy, candidate_count);
  benchmark_case.Validate();
  const size_t cpu_repetitions = std::max(
    size_t{1},
    kMinimumCandidatesPerSample / candidate_count);
  const size_t gpu_repetitions = std::max(
    kMinimumGpuBatchesPerSample,
    kMinimumCandidatesPerSample / candidate_count);

  // Warm both execution paths after correctness is established.
  benchmark_case.EvaluateCpu();
  benchmark_case.EvaluateMetalResident();

  std::vector<double> cpu_samples;
  std::vector<double> resident_samples;
  std::vector<double> roundtrip_samples;
  cpu_samples.reserve(sample_count);
  resident_samples.reserve(sample_count);
  roundtrip_samples.reserve(sample_count);

  enum class TimedPath {
    kCpu,
    kMetalResident,
    kMetalRoundTrip,
  };
  constexpr std::array kOrders = {
    std::array{
      TimedPath::kCpu,
      TimedPath::kMetalResident,
      TimedPath::kMetalRoundTrip},
    std::array{
      TimedPath::kCpu,
      TimedPath::kMetalRoundTrip,
      TimedPath::kMetalResident},
    std::array{
      TimedPath::kMetalResident,
      TimedPath::kCpu,
      TimedPath::kMetalRoundTrip},
    std::array{
      TimedPath::kMetalResident,
      TimedPath::kMetalRoundTrip,
      TimedPath::kCpu},
    std::array{
      TimedPath::kMetalRoundTrip,
      TimedPath::kCpu,
      TimedPath::kMetalResident},
    std::array{
      TimedPath::kMetalRoundTrip,
      TimedPath::kMetalResident,
      TimedPath::kCpu},
  };
  for (size_t sample = 0; sample < sample_count; ++sample) {
    for (const TimedPath path : kOrders[sample % kOrders.size()]) {
      switch (path) {
        case TimedPath::kCpu:
          cpu_samples.push_back(TimeRepeated(
            cpu_repetitions, [&] { benchmark_case.EvaluateCpu(); }));
          break;
        case TimedPath::kMetalResident:
          benchmark_case.EvaluateMetalResident();
          resident_samples.push_back(TimeRepeated(
            gpu_repetitions,
            [&] { benchmark_case.EvaluateMetalResident(); }));
          break;
        case TimedPath::kMetalRoundTrip:
          benchmark_case.EvaluateMetalRoundTrip();
          roundtrip_samples.push_back(TimeRepeated(
            gpu_repetitions,
            [&] { benchmark_case.EvaluateMetalRoundTrip(); }));
          break;
      }
    }
  }

  PrintRow(
    gjxl::GetAcStrategyInfo(strategy)->name,
    candidate_count,
    Summarize(std::move(cpu_samples)),
    Summarize(std::move(resident_samples)),
    Summarize(std::move(roundtrip_samples)));
}

}  // namespace

int main(int argc, char** argv) {
  const size_t dct8_equivalent_candidates =
    argc > 1 ? ParsePositive(argv[1], "DCT8-equivalent count") : 4096;
  const size_t sample_count =
    argc > 2 ? ParsePositive(argv[2], "sample count") : 12;
  if (argc > 3) {
    std::cerr << "Usage: " << argv[0]
              << " [DCT8-equivalent candidates] [samples]\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<gjxl::GpuBackend> gpu;
  Require(
    gjxl::CreateMetalBackend(
      GJXL_METALLIB_PATH,
      SimdgroupOptions(),
      &gpu),
    "Create Metal backend");
  const Fixture fixture;

  std::cout << "Backend: " << gpu->name() << " [simdgroup matmul]\n"
            << "Image residency and quantization matrices are outside timing; "
               "Metal E2E includes candidate upload and cost download.\n"
            << "Each row reports median milliseconds per batch and CPU/GPU "
               "speedup; ranges show all samples balanced across the six "
               "measurement orders. Each GPU sample has an untimed warm-up "
               "submission.\n\n"
            << std::left << std::setw(11) << "strategy"
            << std::right << std::setw(7) << "count"
            << std::setw(11) << "CPU ms"
            << std::setw(11) << "GPU ms"
            << std::setw(10) << "speedup"
            << std::setw(11) << "E2E ms"
            << std::setw(10) << "speedup" << "   ranges\n";

  for (const gjxl::AcStrategyType strategy : kStrategies) {
    const size_t coefficient_count =
      gjxl::GetAcStrategyInfo(strategy)->coefficient_count();
    const size_t maximum_count = std::max(
      size_t{1},
      dct8_equivalent_candidates / (coefficient_count / 64));
    std::set<size_t> batch_sizes = {1, 8, 32, maximum_count};
    for (const size_t candidate_count : batch_sizes) {
      if (candidate_count <= maximum_count) {
        BenchmarkCase(
          *gpu,
          fixture,
          strategy,
          candidate_count,
          sample_count);
      }
    }
  }

  std::cout << std::defaultfloat
            << "Validation: " << g_parity.evaluated_count
            << " CPU/GPU costs, " << g_parity.tight_mismatch_count
            << " quantization-boundary differences above the tight unit-test "
               "tolerance; max relative error "
            << g_parity.max_relative_error << " ("
            << gjxl::GetAcStrategyInfo(g_parity.worst_strategy)->name
            << " candidate " << g_parity.worst_candidate
            << ", absolute " << g_parity.max_absolute_error
            << "; hard gate " << kMaximumRelativeCostError << ").\n";
  std::cout << "checksum " << g_checksum << '\n';
  return EXIT_SUCCESS;
}
