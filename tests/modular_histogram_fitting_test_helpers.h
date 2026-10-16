// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gpu/ops/modular_histogram_fitting.h"
#include <bit>
#include <numeric>
#include <stdexcept>

namespace histogram_fitting_test {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::histogram_gpu_internal;
using namespace gjxl::resource_budget_internal;
inline void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
inline void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
inline HistogramTrainingSamples Fixture(size_t seed, size_t rows,
                                        size_t predictors) {
  HistogramTrainingSamples s;
  s.predictor_count = predictors;
  s.property_count = 16;
  s.static_property_count = 2;
  std::iota(s.properties.begin(), s.properties.end(), uint8_t{0});
  // Weighted first, then Gradient, then remaining predictors in wire order.
  s.predictors[0] = Predictor::kWeighted;
  size_t next = 1;
  if (predictors > 1)
    s.predictors[next++] = Predictor::kGradient;
  for (size_t pred = 0; next < predictors; ++pred)
    if (Predictor(pred) != Predictor::kWeighted &&
        Predictor(pred) != Predictor::kGradient)
      s.predictors[next++] = Predictor(pred);
  for (size_t prop = 0; prop < s.property_count; ++prop) {
    const size_t bins =
        prop < 2 ? (prop == 0 ? 4 : 8) : (prop == 15 ? 16 : 256);
    s.thresholds[prop].resize(bins - 1);
    std::iota(s.thresholds[prop].begin(), s.thresholds[prop].end(),
              prop < 2 ? 0 : -127);
  }
  uint32_t rng = static_cast<uint32_t>(seed + 1);
  for (size_t row = 0; row < rows; ++row) {
    rng = rng * 1664525u + 1013904223u;
    const size_t pattern = (row * 11 + (seed % 3 ? (rng >> 24) : 0)) % 32;
    const uint16_t weight = seed % 2 ? 1 + row % 7 : 1;
    s.weights.push_back(weight);
    s.total_samples += weight;
    s.static_values[0].push_back(seed % 4 ? 0 : pattern % 4);
    s.static_values[1].push_back(seed % 5 ? 7 : pattern % 8);
    for (size_t prop = 2; prop < s.property_count; ++prop)
      s.values[prop - 2].push_back(prop == 15   ? pattern % 16
                                   : prop == 14 ? 255
                                   : prop < 6   ? (pattern % 4) * 85
                                                : (pattern * (prop + 1)) % 256);
    for (size_t pred = 0; pred < predictors; ++pred) {
      // Some exact predictor/property ties, and different symbol spans as
      // trees partition. Other cases cover all 256 residual symbols.
      const size_t token = seed % 3 == 0
                               ? pattern * (pred % 2 + 1)
                               : (pattern * (pred + 1) + (rng >> 24)) % 256;
      s.residuals[pred].push_back(
          {static_cast<uint8_t>(token),
           static_cast<uint8_t>(seed % 7 == 0 ? 0 : (row + pred) % 6)});
    }
  }
  return s;
}
inline bool SameRows(const HistogramTrainingSamples &a,
                     const HistogramTrainingSamples &b) {
  return a.weights == b.weights && a.residuals == b.residuals &&
         a.static_values == b.static_values && a.values == b.values &&
         a.thresholds == b.thresholds;
}
inline void CheckOriginalRows(const HistogramTrainingSamples &original,
                              const HistogramTrainingSamples &current,
                              std::span<const uint32_t> order) {
  Check(order.size() == original.weights.size(),
        "Row permutation length differs");
  for (size_t i = 0; i < order.size(); ++i) {
    const size_t r = order[i];
    Check(r < order.size() && current.weights[i] == original.weights[r],
          "Weight permutation differs");
    for (size_t p = 0; p < original.predictor_count; ++p)
      Check(current.residuals[p][i] == original.residuals[p][r],
            "Residual permutation differs");
    for (size_t p = 0; p < original.static_property_count; ++p)
      Check(current.static_values[p][i] == original.static_values[p][r],
            "Static permutation differs");
    for (size_t p = 0;
         p < original.property_count - original.static_property_count; ++p)
      Check(current.values[p][i] == original.values[p][r],
            "Property permutation differs");
  }
}
inline bool SameResults(std::span<const HistogramPropertyResult> a,
                        std::span<const HistogramPropertyResult> b) {
  if (a.size() != b.size())
    return false;
  for (size_t p = 0; p < a.size(); ++p) {
    if (a[p].source_id != b[p].source_id || a[p].node_id != b[p].node_id ||
        a[p].property_index != b[p].property_index ||
        a[p].is_constant != b[p].is_constant)
      return false;
    for (size_t c = 0; c < 4; ++c) {
      const auto &x = a[p].winners[c], &y = b[p].winners[c];
      if (x.bin != y.bin || x.low_rows != y.low_rows ||
          x.left_predictor != y.left_predictor ||
          x.right_predictor != y.right_predictor || x.valid != y.valid ||
          x.reserved != y.reserved ||
          std::bit_cast<uint32_t>(x.left_cost) !=
              std::bit_cast<uint32_t>(y.left_cost) ||
          std::bit_cast<uint32_t>(x.right_cost) !=
              std::bit_cast<uint32_t>(y.right_cost))
        return false;
    }
  }
  return true;
}
} // namespace histogram_fitting_test
