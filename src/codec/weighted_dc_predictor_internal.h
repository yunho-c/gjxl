// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Default weighted predictor adapted from pinned libjxl context_predict.h.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

#include "core/managed_allocator.h"

namespace gjxl::dc_prediction_internal {
constexpr std::array<uint32_t, 64> Reciprocals() {
  std::array<uint32_t, 64> values{};
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (uint32_t{1} << 24) / (i + 1);
  return values;
}
constexpr auto kReciprocal = Reciprocals();

template <resource_budget_internal::ResourceClass Owner>
class WeightedDcPredictor {
public:
  explicit WeightedDcPredictor(size_t width)
      : width_(width), error_((width + 2) * 2) {
    for (auto &errors : prediction_errors_)
      errors.resize((width + 2) * 2);
  }
  void Reset() {
    std::fill(error_.begin(), error_.end(), 0);
    for (auto &errors : prediction_errors_)
      std::fill(errors.begin(), errors.end(), 0);
    predictions_ = {};
    prediction_ = 0;
  }
  std::pair<int64_t, int64_t> Predict(size_t x, size_t y, int64_t n, int64_t w,
                                      int64_t ne, int64_t nw, int64_t nn) {
    const size_t current = (y & 1) ? 0 : width_ + 2;
    const size_t previous = (y & 1) ? width_ + 2 : 0;
    const size_t north = previous + x;
    const size_t northeast = x + 1 < width_ ? north + 1 : north;
    const size_t northwest = x ? north - 1 : north;
    std::array<uint32_t, 4> weights{};
    constexpr std::array<uint32_t, 4> max_weights = {13, 12, 12, 12};
    uint32_t weight_sum = 0;
    for (size_t i = 0; i < 4; ++i) {
      // These additions have the decoder's unsigned 32-bit semantics.
      const uint32_t error = prediction_errors_[i][north] +
                             prediction_errors_[i][northeast] +
                             prediction_errors_[i][northwest];
      const int shift = std::max(
          0, static_cast<int>(std::bit_width(uint64_t{error} + 1)) - 6);
      weights[i] =
          4 + ((max_weights[i] * kReciprocal[error >> shift]) >> shift);
      weight_sum += weights[i];
    }
    const unsigned log_weight = std::bit_width(weight_sum) - 1;
    weight_sum = 0;
    for (auto &weight : weights) {
      weight >>= log_weight - 4;
      weight_sum += weight;
    }
    const int64_t ew = x ? error_[current + x - 1] : 0;
    const int64_t en = error_[north], enw = error_[northwest],
                  ene = error_[northeast];
    int64_t property = ew;
    for (const auto error : {en, enw, ene})
      if (std::abs(error) > std::abs(property))
        property = error;
    n *= 8;
    w *= 8;
    ne *= 8;
    nw *= 8;
    nn *= 8;
    predictions_[0] = w + ne - n;
    predictions_[1] = n - (((en + ew + ene) * 16) >> 5);
    predictions_[2] = w - (((en + ew + enw) * 10) >> 5);
    predictions_[3] = n - (((enw + en + ene) * 7) >> 5);
    int64_t sum = (weight_sum >> 1) - 1;
    for (size_t i = 0; i < 4; ++i)
      sum += predictions_[i] * weights[i];
    prediction_ = (sum * kReciprocal[weight_sum - 1]) >> 24;
    if (((en ^ ew) | (en ^ enw)) <= 0)
      prediction_ =
          std::clamp(prediction_, std::min({w, ne, n}), std::max({w, ne, n}));
    return {(prediction_ + 3) >> 3, property};
  }
  bool Update(int32_t value, size_t x, size_t y) {
    const size_t current = (y & 1) ? 0 : width_ + 2;
    const size_t previous = (y & 1) ? width_ + 2 : 0;
    const int64_t scaled = int64_t{value} * 8;
    const int64_t error = prediction_ - scaled;
    if (error < std::numeric_limits<int32_t>::min() ||
        error > std::numeric_limits<int32_t>::max())
      return false;
    std::array<uint32_t, 4> errors{};
    for (size_t i = 0; i < 4; ++i) {
      const uint64_t magnitude = (std::abs(predictions_[i] - scaled) + 3) >> 3;
      if (magnitude > std::numeric_limits<uint32_t>::max())
        return false;
      errors[i] = static_cast<uint32_t>(magnitude);
    }
    error_[current + x] = static_cast<int32_t>(error);
    for (size_t i = 0; i < 4; ++i) {
      prediction_errors_[i][current + x] = errors[i];
      prediction_errors_[i][previous + x + 1] += errors[i];
    }
    return true;
  }

private:
  size_t width_;
  std::array<int64_t, 4> predictions_{};
  int64_t prediction_ = 0;
  std::array<resource_budget_internal::ManagedVector<uint32_t, Owner>, 4>
      prediction_errors_;
  resource_budget_internal::ManagedVector<int32_t, Owner> error_;
};

} // namespace gjxl::dc_prediction_internal
