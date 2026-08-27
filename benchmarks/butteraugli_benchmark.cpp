// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Interleaved native/libjxl timings and libjxl-managed-memory baselines.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <jxl/memory_manager.h>

#include "butteraugli_fixtures.h"
#include "butteraugli_oracle.h"
#include "codec/butteraugli.h"

namespace {

namespace bt = gjxl::butteraugli_test;
using Clock = std::chrono::steady_clock;

constexpr size_t kWarmupRounds = 3;
constexpr size_t kSamples = 15;

[[nodiscard]] gjxl::Extent2D ToGjxlExtent(bt::OracleExtent extent) {
  return {extent.width, extent.height};
}

[[nodiscard]] gjxl::ConstImage3FView ToGjxlImage(bt::ConstOracleImage3 image) {
  return gjxl::ConstImage3FView{{{
      {image.plane[0].data, ToGjxlExtent(image.plane[0].extent),
       image.plane[0].stride},
      {image.plane[1].data, ToGjxlExtent(image.plane[1].extent),
       image.plane[1].stride},
      {image.plane[2].data, ToGjxlExtent(image.plane[2].extent),
       image.plane[2].stride},
  }}};
}

[[nodiscard]] gjxl::ButteraugliOptions
ToGjxlOptions(bt::OracleOptions options) {
  return {
      .hf_asymmetry = options.hf_asymmetry,
      .x_multiplier = options.x_multiplier,
      .intensity_target = options.intensity_target,
  };
}

struct TrackingAllocator {
  struct Header {
    size_t size;
  };

  static void *Allocate(void *opaque, size_t size) {
    auto *self = static_cast<TrackingAllocator *>(opaque);
    if (size > std::numeric_limits<size_t>::max() - sizeof(Header)) {
      return nullptr;
    }
    auto *header = static_cast<Header *>(std::malloc(sizeof(Header) + size));
    if (header == nullptr)
      return nullptr;
    header->size = size;
    self->live_bytes += size;
    self->peak_bytes = std::max(self->peak_bytes, self->live_bytes);
    return header + 1;
  }
  static void Free(void *opaque, void *address) {
    if (address == nullptr)
      return;
    auto *self = static_cast<TrackingAllocator *>(opaque);
    auto *header = static_cast<Header *>(address) - 1;
    self->live_bytes -= header->size;
    std::free(header);
  }
  [[nodiscard]] JxlMemoryManager Manager() { return {this, Allocate, Free}; }
  void ResetPeak() { peak_bytes = live_bytes; }

  size_t live_bytes = 0;
  size_t peak_bytes = 0;
};

enum class Phase : size_t {
  kNativeOneShot,
  kLibjxlOneShot,
  kReferencePreparation,
  kPreparedComparison,
  kCount,
};

constexpr size_t kPhaseCount = static_cast<size_t>(Phase::kCount);
constexpr std::array<std::string_view, kPhaseCount> kPhaseNames = {
    "native_one_shot",
    "libjxl_one_shot",
    "libjxl_reference_preparation",
    "libjxl_prepared_comparison",
};

struct TimingStats {
  double minimum_ms;
  double median_ms;
  double maximum_ms;
};

[[nodiscard]] TimingStats Summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return {
      samples.front(),
      samples[samples.size() / 2],
      samples.back(),
  };
}

struct MemoryStats {
  size_t libjxl_one_shot_peak = 0;
  size_t preparation_peak = 0;
  size_t prepared_retained = 0;
  size_t comparison_peak_total = 0;
  size_t comparison_incremental_peak = 0;
};

class WorkloadRunner {
public:
  explicit WorkloadRunner(const bt::FixturePair &fixture)
      : fixture_(fixture), map_(fixture.reference.extent().width *
                                fixture.reference.extent().height) {
    if (!bt::PrepareLiveButteraugliReference(fixture_.reference.ConstView(),
                                             fixture_.options, &prepared_)) {
      throw std::runtime_error("Unable to prepare benchmark reference");
    }
  }

  void Run(Phase phase) {
    double score = 0.0;
    const bt::OraclePlane output{map_.data(), fixture_.reference.extent(),
                                 fixture_.reference.extent().width};
    bool ok = false;
    if (phase == Phase::kNativeOneShot) {
      ok = gjxl::ComputeButteraugliDistance(
               ToGjxlImage(fixture_.reference.ConstView()),
               ToGjxlImage(fixture_.distorted.ConstView()),
               ToGjxlOptions(fixture_.options),
               {map_.data(), ToGjxlExtent(fixture_.reference.extent()),
                fixture_.reference.extent().width},
               &score)
               .ok();
    } else if (phase == Phase::kLibjxlOneShot) {
      ok = bt::ComputeLiveButteraugli(
          fixture_.reference.ConstView(), fixture_.distorted.ConstView(),
          fixture_.options, output, &score);
    } else if (phase == Phase::kReferencePreparation) {
      bt::PreparedReference temporary;
      ok = bt::PrepareLiveButteraugliReference(fixture_.reference.ConstView(),
                                               fixture_.options, &temporary);
      score = temporary.valid() ? 1.0 : 0.0;
    } else {
      ok = bt::CompareLiveButteraugliPrepared(
          prepared_, fixture_.distorted.ConstView(), output, &score);
    }
    if (!ok || !std::isfinite(score)) {
      throw std::runtime_error("Butteraugli benchmark phase failed");
    }
    sink_ += score + map_[map_.size() / 2];
  }

  [[nodiscard]] double sink() const { return sink_; }

private:
  const bt::FixturePair &fixture_;
  std::vector<float> map_;
  bt::PreparedReference prepared_;
  double sink_ = 0.0;
};

[[nodiscard]] std::array<TimingStats, kPhaseCount>
BenchmarkTimings(const bt::FixturePair &fixture, double *sink) {
  WorkloadRunner runner(fixture);
  for (size_t round = 0; round < kWarmupRounds; ++round) {
    for (size_t offset = 0; offset < kPhaseCount; ++offset) {
      runner.Run(static_cast<Phase>((round + offset) % kPhaseCount));
    }
  }

  std::array<std::vector<double>, kPhaseCount> samples;
  for (std::vector<double> &phase_samples : samples) {
    phase_samples.reserve(kSamples);
  }
  for (size_t sample = 0; sample < kSamples; ++sample) {
    for (size_t offset = 0; offset < kPhaseCount; ++offset) {
      const Phase phase =
          static_cast<Phase>((sample + offset) % kPhaseCount);
      const auto begin = Clock::now();
      runner.Run(phase);
      const auto end = Clock::now();
      samples[static_cast<size_t>(phase)].push_back(
          std::chrono::duration<double, std::milli>(end - begin).count());
    }
  }
  *sink += runner.sink();
  return {
      Summarize(std::move(samples[0])),
      Summarize(std::move(samples[1])),
      Summarize(std::move(samples[2])),
      Summarize(std::move(samples[3])),
  };
}

[[nodiscard]] MemoryStats MeasureMemory(const bt::FixturePair &fixture) {
  MemoryStats result;
  std::vector<float> map(fixture.reference.extent().width *
                         fixture.reference.extent().height);
  const bt::OraclePlane output{map.data(), fixture.reference.extent(),
                               fixture.reference.extent().width};
  double score = 0.0;

  {
    TrackingAllocator tracker;
    JxlMemoryManager manager = tracker.Manager();
    if (!bt::ComputeLiveButteraugli(
            fixture.reference.ConstView(), fixture.distorted.ConstView(),
            fixture.options, output, &score, &manager) ||
        tracker.live_bytes != 0) {
      throw std::runtime_error("One-shot memory measurement failed");
    }
    result.libjxl_one_shot_peak = tracker.peak_bytes;
  }
  {
    TrackingAllocator tracker;
    JxlMemoryManager manager = tracker.Manager();
    {
      bt::PreparedReference prepared;
      if (!bt::PrepareLiveButteraugliReference(fixture.reference.ConstView(),
                                               fixture.options, &prepared,
                                               &manager)) {
        throw std::runtime_error("Preparation memory measurement failed");
      }
      result.preparation_peak = tracker.peak_bytes;
      result.prepared_retained = tracker.live_bytes;
      tracker.ResetPeak();
      if (!bt::CompareLiveButteraugliPrepared(
              prepared, fixture.distorted.ConstView(), output, &score)) {
        throw std::runtime_error("Comparison memory measurement failed");
      }
      result.comparison_peak_total = tracker.peak_bytes;
      result.comparison_incremental_peak =
          tracker.peak_bytes - result.prepared_retained;
      if (tracker.live_bytes != result.prepared_retained) {
        throw std::runtime_error("Prepared comparison retained scratch memory");
      }
    }
    if (tracker.live_bytes != 0) {
      throw std::runtime_error("Prepared reference leaked managed memory");
    }
  }
  return result;
}

void PrintWorkload(const bt::FixturePair &fixture, double *sink) {
  const auto timings = BenchmarkTimings(fixture, sink);
  const MemoryStats memory = MeasureMemory(fixture);
  std::cout << "workload " << fixture.name << ' '
            << fixture.reference.extent().width << 'x'
            << fixture.reference.extent().height << '\n';
  for (size_t phase = 0; phase < timings.size(); ++phase) {
    std::cout << "  timing_ms " << kPhaseNames[phase]
              << " median=" << timings[phase].median_ms << " range=["
              << timings[phase].minimum_ms << ',' << timings[phase].maximum_ms
              << "]\n";
  }
  std::cout << "  libjxl_managed_bytes libjxl_one_shot_peak="
            << memory.libjxl_one_shot_peak
            << " preparation_peak=" << memory.preparation_peak
            << " prepared_retained=" << memory.prepared_retained
            << " comparison_peak_total=" << memory.comparison_peak_total
            << " comparison_incremental_peak="
            << memory.comparison_incremental_peak << '\n';
}

} // namespace

int main() {
  try {
    const bt::FixturePair synthetic = bt::MakeFixture({
        "synthetic",
        {128, 96},
        bt::FixtureKind::kTexture,
    });
    const bt::FixturePair flower = bt::LoadFlowerFixture(GJXL_FLOWER_PPM_PATH);
    double sink = 0.0;
    std::cout << std::fixed << std::setprecision(3)
              << "Butteraugli benchmark: 3 warmup rounds, 15 rotated samples "
                 "across 4 equal-fixture phases\n"
              << "libjxl-managed-byte peaks apply only to libjxl phases and "
                 "exclude standard-library and process allocations.\n";
    PrintWorkload(synthetic, &sink);
    PrintWorkload(flower, &sink);
    std::cout << "sink=" << sink << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
