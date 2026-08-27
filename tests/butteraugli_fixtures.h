// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "butteraugli_oracle_types.h"

namespace gjxl::butteraugli_test {

inline constexpr float kFixturePoison = -777.0f;

class ImageStorage {
public:
  explicit ImageStorage(OracleExtent extent, size_t extra_stride = 3,
                        float fill = kFixturePoison);

  [[nodiscard]] ConstOracleImage3 ConstView() const;
  [[nodiscard]] OracleExtent extent() const { return extent_; }
  [[nodiscard]] size_t stride() const { return stride_; }

  [[nodiscard]] const std::array<std::vector<float>, 3> &planes() const {
    return plane_;
  }
  [[nodiscard]] std::array<std::vector<float>, 3> &planes() { return plane_; }

private:
  OracleExtent extent_;
  size_t stride_;
  std::array<std::vector<float>, 3> plane_;
};

enum class FixtureKind {
  kFlat,
  kTexture,
  kContrast,
  kChromatic,
  kGradient,
  kIntermediate,
  kIdentityTexture,
  kImpulse,
  kRandomLow,
  kRandomNormal,
  kRandomWide,
};

struct FixtureSpec {
  std::string name;
  OracleExtent extent;
  FixtureKind kind;
  OracleOptions options;
  size_t impulse_channel = 0;
  uint32_t random_seed = 0x12345678u;
};

struct FixturePair {
  std::string name;
  ImageStorage reference;
  ImageStorage distorted;
  OracleOptions options;
};

[[nodiscard]] FixturePair MakeFixture(const FixtureSpec &spec);

/// Loads the libjxl flower PPM, decodes standard sRGB to linear samples, and
/// compares it with a clamped-edge 3x3 box blur. A zero crop extent returns the
/// complete image.
[[nodiscard]] FixturePair LoadFlowerFixture(const std::string &ppm_path,
                                            size_t crop_x = 0,
                                            size_t crop_y = 0,
                                            OracleExtent crop_extent = {});

[[nodiscard]] std::vector<FixturePair>
BuildDifferentialCorpus(const std::string &flower_ppm_path);

/// Builds the deterministic corpus that has no external test-data dependency.
[[nodiscard]] std::vector<FixturePair> BuildSyntheticDifferentialCorpus();

[[nodiscard]] bool PaddingIsPoisoned(const ImageStorage &image);

} // namespace gjxl::butteraugli_test
