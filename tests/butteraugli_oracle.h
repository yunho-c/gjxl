// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <jxl/memory_manager.h>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace gjxl::butteraugli_test {

struct OracleExtent {
  size_t width = 0;
  size_t height = 0;

  friend constexpr bool operator==(OracleExtent, OracleExtent) = default;
};

struct ConstOraclePlane {
  const float *data = nullptr;
  OracleExtent extent;
  size_t stride = 0;
};

struct OraclePlane {
  float *data = nullptr;
  OracleExtent extent;
  size_t stride = 0;
};

struct ConstOracleImage3 {
  std::array<ConstOraclePlane, 3> plane;
};

struct OracleOptions {
  float hf_asymmetry = 1.0f;
  float x_multiplier = 1.0f;
  float intensity_target = 80.0f;
};

/// Calls pinned libjxl's dynamically dispatched internal blur directly.
/// The output plane is committed only after all validation and allocation
/// succeeds.
[[nodiscard]] bool
ComputeLiveGaussianBlur(ConstOraclePlane input, float sigma, OraclePlane output,
                        const JxlMemoryManager *memory_manager = nullptr);

/// Calls pinned libjxl's public, dynamically dispatched full-map path.
/// The output map and score are committed only after the entire call succeeds.
[[nodiscard]] bool
ComputeLiveButteraugli(ConstOracleImage3 reference, ConstOracleImage3 distorted,
                       OracleOptions options, OraclePlane distance_map,
                       double *score,
                       const JxlMemoryManager *memory_manager = nullptr);

class PreparedReference {
public:
  PreparedReference();
  ~PreparedReference();
  PreparedReference(PreparedReference &&) noexcept;
  PreparedReference &operator=(PreparedReference &&) noexcept;

  PreparedReference(const PreparedReference &) = delete;
  PreparedReference &operator=(const PreparedReference &) = delete;

  [[nodiscard]] bool valid() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend bool PrepareLiveButteraugliReference(ConstOracleImage3, OracleOptions,
                                              PreparedReference *,
                                              const JxlMemoryManager *);
  friend bool CompareLiveButteraugliPrepared(const PreparedReference &,
                                             ConstOracleImage3, OraclePlane,
                                             double *);
};

/// Builds pinned libjxl's reusable comparator for a reference image.
[[nodiscard]] bool PrepareLiveButteraugliReference(
    ConstOracleImage3 reference, OracleOptions options,
    PreparedReference *prepared,
    const JxlMemoryManager *memory_manager = nullptr);

/// Compares against a prepared reference and atomically commits map and score.
[[nodiscard]] bool
CompareLiveButteraugliPrepared(const PreparedReference &prepared,
                               ConstOracleImage3 distorted,
                               OraclePlane distance_map, double *score);

enum class IntermediateStage : size_t {
  kBlurSigma1p2,
  kBlurSigma7p15593339443,
  kBlurSigma3p22489901262,
  kBlurSigma1p56416327805,
  kBlurSigma2p7,
  kOpsinX,
  kOpsinY,
  kOpsinB,
  kLowFrequencyX,
  kLowFrequencyY,
  kLowFrequencyB,
  kMediumFrequencyX,
  kMediumFrequencyY,
  kMediumFrequencyB,
  kHighFrequencyX,
  kHighFrequencyY,
  kUltraHighFrequencyX,
  kUltraHighFrequencyY,
  kMaltaMediumFrequencyY,
  kMaltaMediumFrequencyX,
  kMaltaHighFrequencyY,
  kMaltaHighFrequencyX,
  kMaltaUltraHighFrequencyY,
  kMaltaUltraHighFrequencyX,
  kMask,
  kMaskedAcY,
  kFinalComposition,
  kCount,
};

inline constexpr size_t kIntermediateStageCount =
    static_cast<size_t>(IntermediateStage::kCount);

struct IntermediateStageOutput {
  OracleExtent extent;
  std::array<std::vector<float>, kIntermediateStageCount> plane;
};

[[nodiscard]] const char *IntermediateStageName(IntermediateStage stage);

/// Calls the pinned static Highway target for intermediate-stage fixtures.
/// Defining HWY_COMPILE_ONLY_SCALAR on this adapter selects scalar symbols.
[[nodiscard]] bool ComputePinnedIntermediateStages(
    ConstOracleImage3 reference, ConstOracleImage3 distorted,
    OracleOptions options, IntermediateStageOutput *output,
    const JxlMemoryManager *memory_manager = nullptr);

} // namespace gjxl::butteraugli_test
