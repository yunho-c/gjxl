// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "native_fit.h"
#include "codec/modular/histogram_tree_fitter.h"
#include <optional>
#include <stdexcept>
#include <string>

namespace reference_learning {
namespace native = gjxl::modular_internal;
native::HistogramTrainingSamples
NativeSamples(const jxl::TreeSamples &reference) {
  if (reference.NumProperties() > native::kPropertyCount ||
      reference.NumPredictors() > 14)
    throw std::runtime_error("Captured features exceed native profile");
  native::HistogramTrainingSamples samples;
  samples.total_samples = reference.NumSamples();
  samples.predictor_count = reference.NumPredictors();
  samples.property_count = reference.NumProperties();
  samples.static_property_count = reference.NumStaticProps();
  samples.weights.resize(reference.NumDistinctSamples());
  for (size_t i = 0; i < samples.weights.size(); ++i)
    samples.weights[i] = reference.Count(i);
  for (size_t p = 0; p < samples.predictor_count; ++p) {
    samples.predictors[p] =
        static_cast<native::Predictor>(reference.PredictorFromIndex(p));
    samples.residuals[p].resize(samples.weights.size());
    for (size_t i = 0; i < samples.weights.size(); ++i) {
      const auto token = reference.RToken(p, i);
      samples.residuals[p][i] = {token.tok, token.nbits};
    }
  }
  for (size_t p = 0; p < samples.property_count; ++p) {
    if (reference.PropertyFromIndex(p) >= native::kPropertyCount)
      throw std::runtime_error("Captured property exceeds native profile");
    samples.properties[p] = reference.PropertyFromIndex(p);
    samples.thresholds[p].resize(reference.NumPropertyValues(p) - 1);
    for (size_t i = 0; i < samples.thresholds[p].size(); ++i)
      samples.thresholds[p][i] = reference.UnquantizeProperty(p, i);
    if (p < samples.static_property_count) {
      auto &column = samples.static_values[p];
      column.resize(samples.weights.size());
      for (size_t i = 0; i < column.size(); ++i)
        column[i] = reference.Property<true>(p, i);
    } else {
      auto &column = samples.values[p - samples.static_property_count];
      column.resize(samples.weights.size());
      for (size_t i = 0; i < column.size(); ++i)
        column[i] =
            reference.Property<false>(p - samples.static_property_count, i);
    }
  }
  return samples;
}

NativeFitResult FitNative(const Fixture &fixture, bool track_storage, bool resumable) {
  auto samples = NativeSamples(fixture.samples);
  std::vector<native::HistogramMultiplierRange> multipliers;
  for (const auto &m : fixture.multipliers)
    multipliers.push_back({m.range, m.multiplier});
  native::HistogramTreeLearningOptions options;
  options.split_threshold = fixture.threshold;
  options.fast_decode_multiplier = fixture.fast_decode_multiplier;
  options.static_range = fixture.range;
  options.multipliers = multipliers;
  // Explicit research admission, without changing the existing workflow limit.
  options.maximum_samples = native::kHistogramMaximumSamples;
  gjxl::resource_budget_internal::HostStorageBound bound;
  auto status = resumable
      ? native::ComputeHistogramTreeFitterStorageBound(samples, options, &bound)
      : native::ComputeHistogramTreeLearningStorageBound(samples, options, &bound);
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
  NativeFitResult result;
  auto column_bytes = [](const auto &column) {
    return column.capacity() *
           sizeof(typename std::decay_t<decltype(column)>::value_type);
  };
  result.sample_backing_bytes = column_bytes(samples.weights);
  for (const auto &column : samples.residuals)
    result.sample_backing_bytes += column_bytes(column);
  for (const auto &column : samples.static_values)
    result.sample_backing_bytes += column_bytes(column);
  for (const auto &column : samples.values)
    result.sample_backing_bytes += column_bytes(column);
  for (const auto &column : samples.thresholds)
    result.sample_backing_bytes += column_bytes(column);
  using namespace gjxl::resource_budget_internal;
  std::optional<ResourceBudget> budget;
  ResourceReservation reservation;
  auto context = CurrentResourceContext();
  if (track_storage) {
    budget.emplace(bound.peak_bytes);
    const auto admitted = budget->Reserve(bound.peak_bytes, &reservation);
    if (!admitted.ok())
      throw std::runtime_error(std::string(admitted.message()));
    context = {&reservation, ResourceClass::kPreparation};
  }
  const ResourceContextScope scope(context);
  const Timer timer;
  if (resumable) {
    native::HistogramTreeFitter fit;
    std::array<native::HistogramPropertyEvaluation, native::kPropertyCount> output;
    const auto view = std::span(output).first(samples.property_count);
    status = fit.Initialize(samples, options);
    while (status.ok() && !fit.complete()) {
      status = fit.Advance();
      if (status.ok() && fit.ready()) {
        status = fit.EvaluateCpu(view);
        if (status.ok()) status = fit.Apply(view);
      }
    }
    if (status.ok()) status = fit.Publish(&result.tree);
  } else {
    status = native::LearnHistogramTree(samples, &result.tree, options);
  }
  result.time = timer.Elapsed();
  result.scratch_bound_bytes = bound.peak_bytes;
  if (budget)
    result.scratch_peak_bytes = budget->snapshot().peak_backing_bytes;
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
  if (result.tree.size != fixture.expected.size())
    throw std::runtime_error("Native fitting node count mismatch: native " +
                             std::to_string(result.tree.size) + ", reference " +
                             std::to_string(fixture.expected.size()));
  for (size_t i = 0; i < result.tree.size; ++i) {
    const auto &a = result.tree.nodes[i];
    const auto &b = fixture.expected[i];
    if (a.property != b.property || a.split != b.splitval ||
        a.left != b.lchild || a.right != b.rchild ||
        static_cast<unsigned>(a.predictor) !=
            static_cast<unsigned>(b.predictor) ||
        a.offset != b.predictor_offset || a.multiplier != b.multiplier)
      throw std::runtime_error(
          "Native fitting node mismatch at " + std::to_string(i) +
          ": property " + std::to_string(a.property) + "/" +
          std::to_string(b.property) + ", split " + std::to_string(a.split) +
          "/" + std::to_string(b.splitval));
  }
  return result;
}
} // namespace reference_learning
