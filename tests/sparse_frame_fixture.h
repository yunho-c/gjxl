// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "coefficient_order_population_fixture.h"
#include "quantized_frame_fixture.h"

namespace gjxl_test {

template <typename T>
gjxl::VarDctEncoderFrame SparseOracle(size_t strategy, size_t pattern, size_t side) {
  auto seed = MakeFrame(strategy, pattern, side);
  if (pattern != 5 || sizeof(T) == 4) return seed;
  PopulationAssembly assembly(seed);
  for (auto& value : assembly.coefficients) {
    if (value != 0) value = value < 0 ? std::numeric_limits<T>::min()
                                     : std::numeric_limits<T>::max();
  }
  gjxl::VarDctEncoderFrame result;
  Check(gjxl::vardct_frame_internal::AssembleVarDctEncoderFrame(assembly.Input(nullptr), &result));
  return result;
}

template <typename T>
gjxl::vardct_frame_internal::SparseAcStorage<T> SparseStorage(const gjxl::VarDctEncoderFrame& frame) {
  using namespace gjxl;
  std::vector<T> logical;
  for (size_t g = 0; g < frame.ac_group_count(); ++g) {
    VarDctNativeAcGroupView native;
    Check(frame.GetNativeAcGroup(g, &native));
    std::visit([&](const auto& group) {
      for (const auto channel : group.coefficients) {
        for (size_t i = 0; i < group.used_coefficient_count; ++i) {
          if (channel[i] < std::numeric_limits<T>::min() || channel[i] > std::numeric_limits<T>::max())
            throw std::runtime_error("Sparse fixture value does not fit");
          logical.push_back(static_cast<T>(channel[i]));
        }
      }
    }, native);
  }
  vardct_frame_internal::SparseAcStorage<T> owner;
  owner.coefficient_count = logical.size();
  const size_t words = (logical.size() + 63) / 64;
  owner.masks.assign(words, 0);
  owner.offsets.ResetForOverwrite(words);
  std::vector<T> values;
  // Deliberately reverse word payload intervals. No consumer may assume
  // monotonic offsets or rely on one GPU tile completion order.
  for (size_t w = words; w != 0;) {
    --w;
    owner.offsets.data()[w] = static_cast<uint32_t>(values.size());
    for (size_t bit = 0; bit < 64 && w * 64 + bit < logical.size(); ++bit) {
      const T value = logical[w * 64 + bit];
      if (value != 0) {
        owner.masks.data()[w] |= uint64_t{1} << bit;
        values.push_back(value);
      }
    }
  }
  owner.values.ResetForOverwrite(values.size());
  std::copy(values.begin(), values.end(), owner.values.begin());
  Check(owner.Validate(false, 0));
  return owner;
}

template <typename T>
gjxl::vardct_frame_internal::QuantizedFrameAssemblyInputT<T> SparseInput(
    const gjxl::VarDctEncoderFrame& frame,
    std::span<const gjxl::vardct_frame_internal::QuantizedAcTransformLayout> layouts,
    gjxl::vardct_frame_internal::SparseAcStorage<T>* owner,
    const gjxl::vardct_frame_internal::CoefficientOrderPopulation* population = nullptr) {
  return {.geometry = frame.geometry(), .strategies = &frame.strategies(),
    .raw_quant_field = frame.raw_quant_field(), .quantizer = &frame.quantizer(),
    .y_to_x = frame.color_correlation().y_to_x_map(), .y_to_b = frame.color_correlation().y_to_b_map(),
    .epf_sharpness = frame.epf_sharpness(), .profile = frame.profile(),
    .quantized_dc = frame.quantized_dc(), .transforms = layouts,
    .coefficient_order_population = population, .sparse_ac_storage = owner};
}

inline void EqualSparseCoefficients(const gjxl::VarDctEncoderFrame& expected,
                                     const gjxl::VarDctEncoderFrame& actual) {
  Check(actual.valid() ? gjxl::Status::Ok() : gjxl::Status::Internal("Sparse fixture frame invalid"));
  if (actual.ac_group_count() != expected.ac_group_count()) throw std::runtime_error("Sparse group count differs");
  for (size_t g = 0; g < actual.ac_group_count(); ++g) {
    gjxl::VarDctNativeAcGroupView a, b;
    Check(expected.GetNativeAcGroup(g, &a)); Check(actual.GetNativeAcGroup(g, &b));
    std::visit([&](const auto& left, const auto& right) {
      if (left.block_x != right.block_x || left.block_y != right.block_y ||
          left.block_extent != right.block_extent || left.used_coefficient_count != right.used_coefficient_count)
        throw std::runtime_error("Sparse group metadata differs");
      for (size_t c = 0; c < 3; ++c) for (size_t i = 0; i < left.used_coefficient_count; ++i)
        if (left.coefficients[c][i] != right.coefficients[c][i]) throw std::runtime_error("Sparse coefficient differs");
    }, a, b);
  }
}

}  // namespace gjxl_test
