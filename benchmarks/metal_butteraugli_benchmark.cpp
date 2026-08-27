// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Rotated native CPU and prepared Metal Butteraugli path timings.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "butteraugli_fixtures.h"
#include "codec/butteraugli.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_butteraugli_test.h"
#include "gpu/ops/butteraugli.h"

#ifndef GJXL_FLOWER_PPM_PATH
#error "GJXL_FLOWER_PPM_PATH must identify the pinned Flower PPM"
#endif

namespace {

namespace bt = gjxl::butteraugli_test;
using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWarmups = 3;
constexpr size_t kDefaultSamples = 15;

struct CommandLineOptions {
  std::string workload = "all";
  size_t warmups = kDefaultWarmups;
  size_t samples = kDefaultSamples;
};

struct WorkloadSpec {
  std::string_view name;
  gjxl::Extent2D extent;
  bool flower = false;
};

constexpr std::array<WorkloadSpec, 6> kWorkloads{{
    {"synthetic_128x96", {128, 96}, false},
    {"odd_121x89", {121, 89}, false},
    {"flower_510x532", {510, 532}, true},
    {"padded_480p", {854, 479}, false},
    {"padded_720p", {1279, 719}, false},
    {"padded_1080p", {1919, 1079}, false},
}};

constexpr std::array<gjxl::Extent2D, 11> kCrossoverExtents{{
    {8, 8},
    {16, 12},
    {32, 24},
    {64, 48},
    {96, 72},
    {128, 96},
    {192, 144},
    {256, 192},
    {384, 288},
    {512, 384},
    {854, 479},
}};

enum class Phase : size_t {
  kNativeOneShot,
  kReferenceUpload,
  kMetalPreparation,
  kDistortedUpload,
  kResidentComparison,
  kScoreReadback,
  kResidentConsumerE2E,
  kMapReadback,
  kFirstComparisonE2E,
  kCount,
};

constexpr size_t kPhaseCount = static_cast<size_t>(Phase::kCount);
constexpr std::array<std::string_view, kPhaseCount> kPhaseNames{{
    "native_cpu_one_shot",
    "reference_upload",
    "metal_preparation",
    "distorted_upload",
    "resident_comparison",
    "score_readback",
    "resident_consumer_e2e",
    "map_readback",
    "first_comparison_e2e",
}};

struct TimingStats {
  double minimum_ms = 0.0;
  double median_ms = 0.0;
  double maximum_ms = 0.0;
};

[[nodiscard]] size_t ParseSize(std::string_view text, bool allow_zero) {
  if (text.empty() || text.front() == '-') {
    throw std::runtime_error("Benchmark count argument is invalid");
  }
  size_t parsed = 0;
  const unsigned long long value = std::stoull(std::string(text), &parsed);
  if (parsed != text.size() || value > std::numeric_limits<size_t>::max() ||
      (!allow_zero && value == 0)) {
    throw std::runtime_error("Benchmark count argument is invalid");
  }
  return static_cast<size_t>(value);
}

[[nodiscard]] CommandLineOptions ParseCommandLine(int argc, char** argv) {
  CommandLineOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << "usage: gjxl_metal_butteraugli_benchmark "
                   "[--workload NAME|all|crossover] [--warmups N] "
                   "[--samples N]\n";
      std::exit(EXIT_SUCCESS);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("Benchmark option is missing its value");
    }
    const std::string_view value = argv[++index];
    if (argument == "--workload") {
      options.workload = value;
    } else if (argument == "--warmups") {
      options.warmups = ParseSize(value, true);
    } else if (argument == "--samples") {
      options.samples = ParseSize(value, false);
    } else {
      throw std::runtime_error("Unknown Metal Butteraugli benchmark option: " +
                               std::string(argument));
    }
  }
  return options;
}

[[nodiscard]] gjxl::Extent2D ToExtent(bt::OracleExtent extent) {
  return {extent.width, extent.height};
}

[[nodiscard]] gjxl::ConstImage3FView ToHostImage(bt::ConstOracleImage3 image) {

  return {{{
      {image.plane[0].data, ToExtent(image.plane[0].extent),
       image.plane[0].stride},
      {image.plane[1].data, ToExtent(image.plane[1].extent),
       image.plane[1].stride},
      {image.plane[2].data, ToExtent(image.plane[2].extent),
       image.plane[2].stride},
  }}};
}

[[nodiscard]] gjxl::ButteraugliOptions ToOptions(bt::OracleOptions options) {
  return {
      .hf_asymmetry = options.hf_asymmetry,
      .x_multiplier = options.x_multiplier,
      .intensity_target = options.intensity_target,
  };
}

void Require(const gjxl::Status& status, std::string_view operation) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(status.message()));
  }
}

class DeviceImage {
public:
  DeviceImage(gjxl::GpuBackend& backend, const bt::ImageStorage& source)
      : backend_(backend), source_(source), extent_(ToExtent(source.extent())) {
    const size_t plane_bytes =
        source_.stride() * extent_.height * sizeof(float);
    for (auto& buffer : buffer_) {
      Require(backend_.Allocate(plane_bytes, &buffer), "image allocation");
    }
  }

  void Upload() {
    const size_t plane_bytes =
        source_.stride() * extent_.height * sizeof(float);
    for (size_t channel = 0; channel < buffer_.size(); ++channel) {
      Require(backend_.CopyHostToDevice(*buffer_[channel],
                                        source_.planes()[channel].data(),
                                        plane_bytes),
              "image upload");
    }
  }

  [[nodiscard]] gjxl::ConstDeviceImage3View View() const noexcept {
    return {{{
        Plane(0),
        Plane(1),
        Plane(2),
    }}};
  }

  [[nodiscard]] size_t bytes() const noexcept {
    return 3 * source_.stride() * extent_.height * sizeof(float);
  }

private:
  [[nodiscard]] gjxl::ConstDevicePlaneView
  Plane(size_t channel) const noexcept {
    return {
        buffer_[channel].get(), 0, gjxl::DeviceElementType::kF32, extent_,
        source_.stride(),
    };
  }

  gjxl::GpuBackend& backend_;
  const bt::ImageStorage& source_;
  gjxl::Extent2D extent_;
  std::array<std::unique_ptr<gjxl::DeviceBuffer>, 3> buffer_;
};

class DeviceOutput {
public:
  DeviceOutput(gjxl::GpuBackend& backend, gjxl::Extent2D extent)
      : extent_(extent) {
    size_t area = 0;
    if (!extent.try_area(&area)) {
      throw std::runtime_error("Output extent overflows");
    }
    Require(backend.Allocate(area * sizeof(float), &map_), "map allocation");
    Require(backend.Allocate(sizeof(float), &score_), "score allocation");
  }

  [[nodiscard]] gjxl::DevicePlaneView Map() noexcept {
    return {
        map_.get(), 0, gjxl::DeviceElementType::kF32, extent_, extent_.width,
    };
  }

  [[nodiscard]] gjxl::DevicePlaneView Score() noexcept {
    return {
        score_.get(), 0, gjxl::DeviceElementType::kF32, {1, 1}, 1,
    };
  }

  [[nodiscard]] size_t bytes() const noexcept {
    return extent_.width * extent_.height * sizeof(float) + sizeof(float);
  }

private:
  gjxl::Extent2D extent_;
  std::unique_ptr<gjxl::DeviceBuffer> map_;
  std::unique_ptr<gjxl::DeviceBuffer> score_;
};

[[nodiscard]] TimingStats Summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return {samples.front(), samples[samples.size() / 2], samples.back()};
}

class WorkloadRunner {
public:
  WorkloadRunner(gjxl::GpuBackend& backend, const bt::FixturePair& fixture)
      : backend_(backend), fixture_(fixture),
        extent_(ToExtent(fixture.reference.extent())),
        options_(ToOptions(fixture.options)),
        reference_(backend, fixture.reference),
        distorted_(backend, fixture.distorted), output_(backend, extent_),
        host_map_(extent_.width * extent_.height),
        readback_map_(extent_.width * extent_.height) {

    reference_.Upload();
    distorted_.Upload();
    Prepare(reference_, &prepared_);
    Compare(*prepared_, distorted_, output_);
    double score = 0.0;
    Require(prepared_->ReadScore(&score), "initial score readback");
    sink_ += score;
    Require(gjxl::QueryMetalButteraugliResourceUsageForTest(*prepared_,
                                                            &resource_usage_),
            "prepared resource query");
  }

  void Run(Phase phase) {
    switch (phase) {
    case Phase::kNativeOneShot:
      RunNative();
      return;
    case Phase::kReferenceUpload:
      reference_.Upload();
      return;
    case Phase::kMetalPreparation: {
      std::unique_ptr<gjxl::PreparedDeviceButteraugli> temporary;
      Prepare(reference_, &temporary);
      sink_ += temporary->valid() ? 1.0 : 0.0;
      return;
    }
    case Phase::kDistortedUpload:
      distorted_.Upload();
      return;
    case Phase::kResidentComparison:
      Compare(*prepared_, distorted_, output_);
      return;
    case Phase::kScoreReadback: {
      double score = 0.0;
      Require(prepared_->ReadScore(&score), "score readback");
      sink_ += score;
      return;
    }
    case Phase::kResidentConsumerE2E: {
      distorted_.Upload();
      Compare(*prepared_, distorted_, output_);
      double score = 0.0;
      Require(prepared_->ReadScore(&score), "resident score readback");
      sink_ += score;
      return;
    }
    case Phase::kMapReadback:
      Require(prepared_->ReadDistanceMap(
                  {readback_map_.data(), extent_, extent_.width}),
              "map readback");
      sink_ += readback_map_[readback_map_.size() / 2];
      return;
    case Phase::kFirstComparisonE2E:
      RunFirstComparison();
      return;
    case Phase::kCount:
      break;
    }
    throw std::runtime_error("Invalid benchmark phase");
  }

  [[nodiscard]] const gjxl::MetalButteraugliResourceUsage&
  resources() const noexcept {
    return resource_usage_;
  }

  [[nodiscard]] size_t caller_input_bytes() const noexcept {
    return reference_.bytes() + distorted_.bytes();
  }

  [[nodiscard]] size_t caller_output_bytes() const noexcept {
    return output_.bytes();
  }

  [[nodiscard]] double sink() const noexcept { return sink_; }

private:
  void Prepare(const DeviceImage& reference,
               std::unique_ptr<gjxl::PreparedDeviceButteraugli>* prepared) {
    Require(gjxl::PrepareDeviceButteraugli(
                backend_, {reference.View(), options_}, prepared),
            "Metal reference preparation");
  }

  static void Compare(gjxl::PreparedDeviceButteraugli& prepared,
                      const DeviceImage& distorted, DeviceOutput& output) {
    Require(prepared.Compare({distorted.View(), output.Map(), output.Score()}),
            "Metal comparison");
  }

  void RunNative() {
    double score = 0.0;
    Require(gjxl::ComputeButteraugliDistance(
                ToHostImage(fixture_.reference.ConstView()),
                ToHostImage(fixture_.distorted.ConstView()), options_,
                {host_map_.data(), extent_, extent_.width}, &score),
            "native CPU comparison");
    sink_ += score + host_map_[host_map_.size() / 2];
  }

  void RunFirstComparison() {
    DeviceImage reference(backend_, fixture_.reference);
    DeviceImage distorted(backend_, fixture_.distorted);
    DeviceOutput output(backend_, extent_);
    reference.Upload();
    distorted.Upload();
    std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
    Prepare(reference, &prepared);
    Compare(*prepared, distorted, output);
    double score = 0.0;
    Require(prepared->ReadScore(&score), "first-comparison score readback");
    sink_ += score;
  }

  gjxl::GpuBackend& backend_;
  const bt::FixturePair& fixture_;
  gjxl::Extent2D extent_;
  gjxl::ButteraugliOptions options_;
  DeviceImage reference_;
  DeviceImage distorted_;
  DeviceOutput output_;
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared_;
  std::vector<float> host_map_;
  std::vector<float> readback_map_;
  gjxl::MetalButteraugliResourceUsage resource_usage_;
  double sink_ = 0.0;
};

[[nodiscard]] std::array<TimingStats, kPhaseCount>
Benchmark(WorkloadRunner* runner, size_t warmups, size_t sample_count) {

  for (size_t round = 0; round < warmups; ++round) {
    for (size_t offset = 0; offset < kPhaseCount; ++offset) {
      runner->Run(static_cast<Phase>((round + offset) % kPhaseCount));
    }
  }
  std::array<std::vector<double>, kPhaseCount> samples;
  for (auto& phase_samples : samples)
    phase_samples.reserve(sample_count);
  for (size_t sample = 0; sample < sample_count; ++sample) {
    for (size_t offset = 0; offset < kPhaseCount; ++offset) {
      const Phase phase = static_cast<Phase>((sample + offset) % kPhaseCount);
      const auto begin = Clock::now();
      runner->Run(phase);
      const auto end = Clock::now();
      samples[static_cast<size_t>(phase)].push_back(
          std::chrono::duration<double, std::milli>(end - begin).count());
    }
  }
  std::array<TimingStats, kPhaseCount> result;
  for (size_t phase = 0; phase < kPhaseCount; ++phase) {
    result[phase] = Summarize(std::move(samples[phase]));
  }
  return result;
}

[[nodiscard]] bt::FixturePair MakeWorkload(const WorkloadSpec& spec) {
  if (spec.flower)
    return bt::LoadFlowerFixture(GJXL_FLOWER_PPM_PATH);
  return bt::MakeFixture({
      std::string(spec.name),
      {spec.extent.width, spec.extent.height},
      bt::FixtureKind::kTexture,
  });
}

void PrintWorkload(gjxl::GpuBackend& backend, const WorkloadSpec& spec,
                   size_t warmups, size_t samples, double* sink) {

  const bt::FixturePair fixture = MakeWorkload(spec);
  WorkloadRunner runner(backend, fixture);
  const auto timings = Benchmark(&runner, warmups, samples);
  const auto& resources = runner.resources();
  const double native =
      timings[static_cast<size_t>(Phase::kNativeOneShot)].median_ms;
  const double resident =
      timings[static_cast<size_t>(Phase::kResidentConsumerE2E)].median_ms;
  std::cout << "workload " << spec.name << ' ' << spec.extent.width << 'x'
            << spec.extent.height << '\n';
  for (size_t phase = 0; phase < kPhaseCount; ++phase) {
    std::cout << "  timing_ms " << kPhaseNames[phase]
              << " median=" << timings[phase].median_ms << " range=["
              << timings[phase].minimum_ms << ',' << timings[phase].maximum_ms
              << "]\n";
  }
  std::cout << "  speedup native_over_resident_consumer=" << native / resident
            << '\n'
            << "  metal_prepared_bytes allocation="
            << resources.prepared_allocation_bytes
            << " cached_reference=" << resources.cached_reference_bytes
            << " gaussian_kernels=" << resources.gaussian_kernel_bytes
            << " peak_logical_comparison_scratch="
            << resources.peak_comparison_scratch_bytes << '\n'
            << "  caller_device_bytes inputs=" << runner.caller_input_bytes()
            << " outputs=" << runner.caller_output_bytes() << '\n';
  *sink += runner.sink();
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CommandLineOptions options = ParseCommandLine(argc, argv);
    std::unique_ptr<gjxl::GpuBackend> backend;
    Require(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &backend),
            "Metal backend creation");
    std::cout << std::fixed << std::setprecision(6)
              << "Metal Butteraugli benchmark: " << options.warmups
              << " warmups, " << options.samples << " rotated samples across "
              << kPhaseCount << " phases\n"
              << "resident_consumer_e2e includes distorted upload, prepared "
                 "comparison/synchronization, and score readback; the map "
                 "remains device-resident. Backend creation is excluded.\n";
    double sink = 0.0;
    bool matched = false;
    if (options.workload == "all") {
      for (const WorkloadSpec& spec : kWorkloads) {
        PrintWorkload(*backend, spec, options.warmups, options.samples, &sink);
      }
      matched = true;
    } else if (options.workload == "crossover") {
      for (gjxl::Extent2D extent : kCrossoverExtents) {
        const std::string name = "crossover_" + std::to_string(extent.width) +
                                 "x" + std::to_string(extent.height);
        PrintWorkload(*backend, {name, extent, false}, options.warmups,
                      options.samples, &sink);
      }
      matched = true;
    } else {
      for (const WorkloadSpec& spec : kWorkloads) {
        if (options.workload == spec.name) {
          PrintWorkload(*backend, spec, options.warmups, options.samples,
                        &sink);
          matched = true;
          break;
        }
      }
    }
    if (!matched) {
      throw std::runtime_error("Unknown Metal Butteraugli workload: " +
                               options.workload);
    }
    std::cout << "sink=" << sink << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
