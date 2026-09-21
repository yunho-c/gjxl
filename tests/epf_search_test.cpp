// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/epf_search.h"
#include "codec/adaptive_quantization.h"
#include "codec/adaptive_quantization_internal.h"
#include "codec/color_transform.h"
#include "codec/loop_filter.h"
#include "codec/reconstruction.h"
#include "core/image_buffer.h"

namespace {
using namespace gjxl;

// Literal two-raster-pass selector from libjxl 9e7fba5d's
// ComputeARHeuristics, deliberately retaining its size_t histogram quotient.
// This independently checks the production selector's elimination of both
// context-dependent passes rather than duplicating its per-block argmin.
std::vector<uint8_t> LibjxlSelector(
    const std::array<ConstPlaneF32View, 3>& errors, float distance) {
  const std::vector<uint8_t> steps = distance > 4.5f
      ? std::vector<uint8_t>{0, 4} : std::vector<uint8_t>{0, 2, 7};
  const auto extent = errors[0].extent;
  std::vector<uint8_t> result(extent.width * extent.height);
  std::array<size_t, 8> lut{};
  for (size_t i = 0; i < steps.size(); ++i) lut[steps[i]] = i;
  std::array<std::array<size_t, 8>, 9> histogram{};
  std::array<size_t, 9> totals;
  totals.fill(1);
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const uint8_t top = y ? result[(y - 1) * extent.width + x] : 0;
      const uint8_t left = x ? result[y * extent.width + x - 1] : 0;
      const float top_error = errors[lut[top]].Row(y)[x];
      const float left_error = errors[lut[left]].Row(y)[x];
      uint8_t best = 0;
      float best_error = std::numeric_limits<float>::max();
      for (uint8_t value : steps) {
        float error = errors[lut[value]].Row(y)[x];
        if (value == 0) error *= 0.99f;
        if (error < best_error) {
          best = value;
          best_error = error;
        }
      }
      const uint8_t value = best_error < std::min(top_error, left_error)
          ? best : top_error < left_error ? top : left;
      result[y * extent.width + x] = value;
      const size_t context = 3 * lut[top] + lut[left];
      ++histogram[context][value];
      ++totals[context];
    }
  }
  const float clamped = std::min(5.0f, distance);
  const float c3 = std::max(0.85970338919928291f,
                            std::pow(0.98017198824148288f, clamped));
  const float c5 = 0.1087690359555803f;
  std::array<float, 27> multiplier{};
  for (uint8_t top : steps) {
    for (uint8_t left : steps) {
      const size_t context = 3 * lut[top] + lut[left];
      for (uint8_t value : steps) {
        float& mul = multiplier[3 * context + lut[value]];
        mul = 1.0 / (1.0 + c5 * std::log1p(
            histogram[context][value] / totals[context]) / clamped);
        if (value == 0) mul *= c3;
      }
    }
  }
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const uint8_t top = y ? result[(y - 1) * extent.width + x] : 0;
      const uint8_t left = x ? result[y * extent.width + x - 1] : 0;
      const size_t context = 3 * lut[top] + lut[left];
      float best_error = std::numeric_limits<float>::max();
      uint8_t best = 0;
      for (uint8_t value : steps) {
        const float error = errors[lut[value]].Row(y)[x] *
                            multiplier[3 * context + lut[value]];
        if (error < best_error) {
          best_error = error;
          best = value;
        }
      }
      result[y * extent.width + x] = best;
    }
  }
  return result;
}

bool CheckSelector() {
  constexpr Extent2D extent{127, 39};
  constexpr size_t stride = 131;
  std::array<std::vector<float>, 3> data;
  std::array<ConstPlaneF32View, 3> errors;
  uint32_t random = 0x13a93f21u;
  for (size_t i = 0; i < 3; ++i) {
    data[i].resize(stride * extent.height);
    for (size_t j = 0; j < data[i].size(); ++j) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      data[i][j] = j % 11 == 0 ? 0.0f
          : std::ldexp(float(random & 1023u), int(j % 33) - 16);
    }
    errors[i] = {data[i].data(), extent, stride};
  }
  std::vector<uint8_t> selected(stride * extent.height, 255);
  for (float distance : {0.5f, 1.0f, 4.5f, 4.5001f, 5.0f, 15.0f}) {
    const auto expected = LibjxlSelector(errors, distance);
    if (!SelectEpfSharpnessFromErrors(
            errors, distance, {selected.data(), extent, stride}).ok()) return false;
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < stride; ++x) {
        if (selected[y * stride + x] != (x < extent.width
            ? expected[y * extent.width + x] : uint8_t{255})) return false;
      }
    }
  }
  const auto saved = selected;
  data[2][(extent.height - 1) * stride + extent.width - 1] =
      std::numeric_limits<float>::quiet_NaN();
  if (SelectEpfSharpnessFromErrors(
          errors, 1.0f, {selected.data(), extent, stride}).ok() ||
      selected != saved) return false;
  // The unused third plane must not be read for the two-candidate policy.
  if (!SelectEpfSharpnessFromErrors(
          errors, 8.0f, {selected.data(), extent, stride}).ok()) return false;
  EpfSharpnessSearchConfig config;
  return !MakeEpfSharpnessSearchConfig(0.0f, &config).ok() &&
         !MakeEpfSharpnessSearchConfig(
             std::numeric_limits<float>::infinity(), &config).ok();
}

bool CheckBlockError() {
  constexpr Extent2D extent{9, 10};
  Image3FBuffer original(extent), reconstruction(extent);
  for (size_t c = 0; c < 3; ++c) {
    std::fill(original.plane(c).begin(), original.plane(c).end(), float(c + 1));
  }
  std::vector<float> mask(12 * extent.height, 2.0f);
  std::array<float, 6> errors;
  errors.fill(-777.0f);
  const PlaneF32View out{errors.data(), {2, 2}, 3};
  if (!ComputeEpfBlockErrors(original.const_view(), reconstruction.const_view(),
                             {mask.data(), extent, 12}, out).ok()) return false;
  const double per_pixel = 4 * (12.339445295782363 + 4 + 0.2 * 9);
  const std::array<size_t, 4> pixels{64, 8, 16, 2};
  for (size_t y = 0; y < 2; ++y) {
    for (size_t x = 0; x < 2; ++x) {
      const float expected = float(per_pixel * pixels[2 * y + x]);
      if (std::abs(out.Row(y)[x] - expected) > 0.001f) return false;
    }
    if (out.Row(y)[2] != -777.0f) return false;
  }
  const auto saved = errors;
  mask.back() = -1.0f;  // Row padding is not a sample.
  if (!ComputeEpfBlockErrors(original.const_view(), reconstruction.const_view(),
                             {mask.data(), extent, 12}, out).ok()) return false;
  mask[9 * 12 + 8] = -1.0f;
  return !ComputeEpfBlockErrors(original.const_view(), reconstruction.const_view(),
                                {mask.data(), extent, 12}, out).ok() &&
         errors == saved;
}

bool CheckGlobalCandidateSearch() {
  constexpr Extent2D source{23, 19}, padded{24, 24}, blocks{3, 3};
  Image3FBuffer original(padded);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < padded.height; ++y) {
      for (size_t x = 0; x < padded.width; ++x) {
        original.view().plane[c].Row(y)[x] = (c == 0 ? 0.01f : 0.3f) +
            0.017f * std::sin(float(x * (c + 1) + 3 * y) * 0.53f);
      }
    }
  }
  std::vector<float> mask(padded.width * padded.height, 1.0f);
  for (size_t i = 0; i < mask.size(); ++i) mask[i] += float(i % 7) * 0.2f;
  AcStrategyGrid grid;
  Quantizer quantizer;
  ColorCorrelationMap cfl;
  FrameGeometry geometry;
  if (!AcStrategyGrid::Create(blocks, &grid).ok() ||
      !Quantizer::Create({3541, 10}, &quantizer).ok() ||
      !FrameGeometry::Create(source, &geometry).ok() ||
      !ComputeInitialColorCorrelationMap(original.const_view(), &cfl).ok()) return false;
  if (!grid.Set(0, 0, AcStrategyType::kDct16x16).ok()) return false;
  grid.fill_empty_dct8();
  std::array<int32_t, 9> raw{10, 19, 7, 28, 16, 9, 3, 17, 6};
  std::array<uint8_t, 9> neutral;
  neutral.fill(4);
  for (bool gaborish : {false, true}) {
    for (uint32_t iterations : {0u, 1u, 2u, 3u}) {
      SimpleVarDctCodestreamProfile profile;
      profile.loop_filter.gaborish = gaborish;
      profile.loop_filter.epf_options.iterations = iterations;
      VarDctEncoderFrame frame;
      if (!ComputeQuantizedCoefficients(
              original.const_view(),
              {.geometry = geometry, .strategies = &grid,
               .raw_quant_field = {raw.data(), blocks, 3}, .quantizer = &quantizer,
               .color_correlation = &cfl,
               .epf_sharpness = {neutral.data(), blocks, 3}}, profile, &frame).ok()) return false;
      for (float distance : {0.49f, 0.5f, 4.5f, 8.0f}) {
        std::array<uint8_t, 9> selected;
        selected.fill(255);
        if (!SearchEpfSharpness(original.const_view(), {mask.data(), padded, 24},
                                frame, distance, {selected.data(), blocks, 3}).ok()) return false;
        if (iterations == 0 || distance < 0.5f) {
          if (selected != neutral) return false;
          continue;
        }
        EpfSharpnessSearchConfig config;
        if (!MakeEpfSharpnessSearchConfig(distance, &config).ok()) return false;
        std::array<std::vector<float>, 3> error_storage;
        std::array<ConstPlaneF32View, 3> errors;
        for (size_t i = 0; i < config.count; ++i) {
          // Independent unoptimized path: reconstruct and run the complete
          // Gaborish+EPF chain afresh for each globally uniform candidate.
          Image3FBuffer reconstruction(padded), filtered(source);
          std::array<uint8_t, 9> candidate;
          candidate.fill(config.candidates[i]);
          std::array<float, 9> sigma;
          if (!ReconstructQuantizedCoefficients(frame, reconstruction.view()).ok() ||
              !ComputeEpfInverseSigma(grid, frame.raw_quant_field(), frame.quantizer(),
                                     {candidate.data(), blocks, 3}, profile.epf_sigma,
                                     {sigma.data(), blocks, 3}).ok() ||
              !ApplyLoopFilters(reconstruction.cropped_view(source),
                                {sigma.data(), blocks, 3}, profile.loop_filter,
                                filtered.view()).ok()) return false;
          error_storage[i].resize(9);
          if (!ComputeEpfBlockErrors(original.cropped_view(source), filtered.const_view(),
                                    {mask.data(), source, 24},
                                    {error_storage[i].data(), blocks, 3}).ok()) return false;
          errors[i] = {error_storage[i].data(), blocks, 3};
        }
        const auto expected = LibjxlSelector(errors, distance);
        if (!std::equal(selected.begin(), selected.end(), expected.begin())) return false;
        if (!std::equal(neutral.begin(), neutral.end(), frame.epf_sharpness().data)) return false;
      }
    }
  }
  return true;
}

bool CheckCpuPolicy() {
  constexpr Extent2D source{24, 16}, blocks{3, 2};
  Image3FBuffer linear(source), original(source);
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < source.height; ++y)
      for (size_t x = 0; x < source.width; ++x)
        linear.view().plane[c].Row(y)[x] =
            0.05f + float((x * 7 + y * 3 + c * 13) % 81) * 0.01f;
  if (!LinearRgbToOpsin(linear.const_view(), 255.0f, original.view()).ok()) return false;
  std::array<float, 6> initial, strategy_mask;
  std::vector<float> mask(24 * 16);
  std::array<uint8_t, 6> neutral;
  neutral.fill(4);
  if (!ComputeInitialQuantField(original.const_view(), {},
          {{initial.data(), blocks, 3}, {strategy_mask.data(), blocks, 3},
           {mask.data(), source, 24}}).ok()) return false;
  AcStrategyGrid grid;
  if (!AcStrategyGrid::Create(blocks, &grid).ok() ||
      !grid.Set(0, 0, AcStrategyType::kDct16x16).ok()) return false;
  grid.fill_empty_dct8();
  VarDctEncoderFrame baseline;
  std::array<float, 6> baseline_quant;
  std::vector<double> baseline_scores;
  for (bool search : {false, true}) {
    AdaptiveQuantizationOptions options;
    options.profile.loop_filter.gaborish = false;
    options.iterations = 2;
    options.search_epf_sharpness = search;
    options.epf_search_reference = {original.const_view(), {mask.data(), source, 24}};
    VarDctEncoderFrame frame;
    Image3FBuffer rgb(source);
    std::array<float, 6> quant, distance;
    std::vector<double> scores;
    adaptive_quantization_internal::AdaptiveQuantizationProfile profile;
    if (!adaptive_quantization_internal::FindBestQuantizationProfiled(
            linear.const_view(), original.const_view(), grid,
            {initial.data(), blocks, 3}, {neutral.data(), blocks, 3}, options,
            {{quant.data(), blocks, 3}, {distance.data(), blocks, 3},
             rgb.view(), &frame, &scores}, &profile).ok() ||
        scores.size() != 3 || profile.evaluations.size() != 3) return false;
    for (size_t i = 0; i < profile.evaluations.size(); ++i) {
      const auto elapsed = profile.evaluations[i].stage_nanoseconds[static_cast<size_t>(
          adaptive_quantization_internal::EvaluationStage::kEpfSharpnessSearch)];
      if ((elapsed != 0) != (search && i == 2)) return false;
    }
    if (!search) {
      baseline = std::move(frame);
      baseline_quant = quant;
      baseline_scores = std::move(scores);
      continue;
    }
    if (quant != baseline_quant ||
        !std::equal(scores.begin(), scores.begin() + 2, baseline_scores.begin())) return false;
    for (size_t y = 0; y < blocks.height; ++y) {
      if (!std::equal(frame.raw_quant_field().Row(y),
                      frame.raw_quant_field().Row(y) + blocks.width,
                      baseline.raw_quant_field().Row(y))) return false;
      for (size_t c = 0; c < 3; ++c)
        if (!std::equal(frame.quantized_dc().plane[c].Row(y),
                        frame.quantized_dc().plane[c].Row(y) + blocks.width,
                        baseline.quantized_dc().plane[c].Row(y))) return false;
    }
    for (size_t group = 0; group < baseline.ac_group_count(); ++group) {
      VarDctAcGroupView a, b;
      if (!frame.GetAcGroup(group, &a).ok() || !baseline.GetAcGroup(group, &b).ok()) return false;
      for (size_t c = 0; c < 3; ++c)
        if (!std::equal(a.coefficients[c].begin(), a.coefficients[c].end(),
                        b.coefficients[c].begin(), b.coefficients[c].end())) return false;
    }
    std::array<uint8_t, 6> expected_map;
    if (!SearchEpfSharpness(original.const_view(), {mask.data(), source, 24},
                            baseline, 1.0f, {expected_map.data(), blocks, 3}).ok()) return false;
    for (size_t y = 0; y < blocks.height; ++y)
      if (!std::equal(expected_map.data() + 3 * y, expected_map.data() + 3 * (y + 1),
                      frame.epf_sharpness().Row(y))) return false;
    Image3FBuffer reconstructed(source), filtered(source), expected(source);
    std::array<float, 6> sigma;
    if (!ReconstructQuantizedCoefficients(frame, reconstructed.view()).ok() ||
        !ComputeEpfInverseSigma(grid, frame.raw_quant_field(), frame.quantizer(),
              frame.epf_sharpness(), frame.profile().epf_sigma,
              {sigma.data(), blocks, 3}).ok() ||
        !ApplyLoopFilters(reconstructed.const_view(), {sigma.data(), blocks, 3},
                           frame.profile().loop_filter, filtered.view()).ok() ||
        !OpsinToLinearRgb(filtered.const_view(), frame.profile().intensity_target,
                          expected.view()).ok()) return false;
    for (size_t c = 0; c < 3; ++c)
      for (size_t i = 0; i < source.width * source.height; ++i)
        if (rgb.plane(c)[i] != expected.plane(c)[i]) return false;
  }
  return true;
}
}  // namespace

int main() {
  if (!CheckSelector()) { std::cerr << "EPF selector oracle failed\n"; return EXIT_FAILURE; }
  if (!CheckBlockError()) { std::cerr << "EPF block-error check failed\n"; return EXIT_FAILURE; }
  if (!CheckGlobalCandidateSearch()) { std::cerr << "EPF candidate search failed\n"; return EXIT_FAILURE; }
  if (!CheckCpuPolicy()) { std::cerr << "EPF CPU policy integration failed\n"; return EXIT_FAILURE; }
  std::cout << "EPF sharpness search checks passed\n";
  return EXIT_SUCCESS;
}
