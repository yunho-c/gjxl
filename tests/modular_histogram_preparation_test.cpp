// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_sample_preparation.h"
#include "codec/modular/profile.h"
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
bool Same(const HistogramTrainingSamples &a,
          const HistogramTrainingSamples &b) {
  return a.total_samples == b.total_samples &&
         a.predictor_count == b.predictor_count &&
         a.property_count == b.property_count &&
         a.static_property_count == b.static_property_count &&
         a.predictors == b.predictors && a.properties == b.properties &&
         a.weights == b.weights && a.residuals == b.residuals &&
         a.values == b.values && a.static_values == b.static_values &&
         a.thresholds == b.thresholds;
}
struct Image {
  std::vector<int32_t> pixels;
  std::array<ModularChannelView, 2> channels;
  std::array<HistogramTrainingImage, 1> images;
  Image() : pixels(33 * 41 * 2) {
    for (size_t i = 0; i < pixels.size(); ++i)
      pixels[i] = static_cast<int32_t>((i * 197 + i / 29) % 65536) - 32768;
    for (size_t c = 0; c < channels.size(); ++c)
      channels[c] = {
          std::span(pixels).subspan(c * 33 * 41, 33 * 41), {{33, 41}}, 33};
    images[0] = {{33, 41}, channels, 0};
  }
};
void ReusedFeatures() {
  Image image;
  image.images[0].weighted = {{8, 8, 4, 0, 3, 23, 2}, {13, 12, 12, 12}};
  std::vector<WeightedPredictorFeature> features(image.pixels.size());
  for (size_t c = 0; c < image.channels.size(); ++c)
    Ok(PrepareWeightedPredictorFeatures(
        image.channels[c], image.images[0].weighted,
        std::span(features).subspan(c * 33 * 41, 33 * 41)));
  for (bool excluded : {false, true}) {
    HistogramSamplingOptions options;
    options.maximum_channel_size = 256;
    if (excluded)
      image.channels[1].descriptor.extent.width = 257;
    PreparedHistogramSamples scalar, reused;
    image.images[0].weighted_features = {};
    Ok(PrepareHistogramSamples(image.images, &scalar, options));
    std::array<WeightedFeatureView, 2> views{};
    for (size_t c = 0; c < (excluded ? 1 : views.size()); ++c)
      views[c] = {image.channels[c], image.images[0].weighted,
                  std::span(features).subspan(c * 33 * 41, 33 * 41)};
    image.images[0].weighted_features = views;
    ModularProfile profile;
    {
      ProfileSession session(&profile);
      Ok(PrepareHistogramSamples(image.images, &reused, options));
    }
    Check(
        Same(scalar.samples(), reused.samples()) &&
            scalar.tree_options().split_threshold ==
                reused.tree_options().split_threshold &&
            profile.training_work.weighted_prediction_samples == 0 &&
            profile.training_work.reused_prediction_samples ==
                scalar.stats().scanned_pixels,
        "Borrowed features changed samples or recomputed weighted predictions");
    DecisionTree a, b;
    Ok(LearnPreparedHistogramTree(scalar, &a));
    Ok(LearnPreparedHistogramTree(reused, &b));
    Check(a == b, "Borrowed features changed the learned tree");
    const auto *old = reused.samples().weights.data();
    image.images[0].weighted_features =
        image.images[0].weighted_features.first(1);
    ArmNextManagedHostAllocationFailureForTest();
    const auto s = PrepareHistogramSamples(image.images, &reused, options);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(s.code() == StatusCode::kInvalidArgument && pending &&
              reused.samples().weights.data() == old,
          "Incomplete weighted features allocated or changed output");
  }
}
void AdmissionAndPublication(bool unique) {
  Image image;
  HistogramSamplingOptions options;
  options.maximum_property_values = unique ? 256 : 8; // Small WP needs 15 cuts.
  options.use_unique_row_proof = unique;
  PreparedHistogramSamples output;
  Ok(PrepareHistogramSamples(image.images, &output, options));
  const auto saved = output.samples();
  const auto *saved_pointer = output.samples().weights.data();
  const auto threshold = output.tree_options().split_threshold;
  auto unchanged = [&] {
    return saved_pointer == output.samples().weights.data() &&
           Same(saved, output.samples()) &&
           threshold == output.tree_options().split_threshold;
  };
  auto rejects = [&](auto modify) {
    auto bad = options;
    modify(bad);
    ArmNextManagedHostAllocationFailureForTest();
    HostStorageBound bound;
    const auto planned = ComputeHistogramSamplePreparationStorageBound(
        image.images, bad, &bound);
    const auto status = PrepareHistogramSamples(image.images, &output, bad);
    Check(!planned.ok() && status.code() == planned.code() && unchanged() &&
              ManagedHostAllocationFailurePendingForTest(),
          "Invalid options allocated or published");
    DisarmManagedHostAllocationFailureForTest();
  };
  rejects([](auto &o) { o.sampling_fraction = 0; });
  rejects([](auto &o) {
    o.sampling_fraction = std::numeric_limits<float>::quiet_NaN();
  });
  rejects([](auto &o) { o.sampling_fraction = 1.1; });
  rejects([](auto &o) {
    o.sampling_fraction = std::numeric_limits<float>::denorm_min();
  });
  rejects([](auto &o) { o.maximum_samples = 1; });
  rejects([](auto &o) { o.maximum_samples = kHistogramMaximumSamples + 1; });
  rejects([](auto &o) { o.maximum_property_values = 257; });
  rejects([](auto &o) { o.maximum_property_values = 1; });
  rejects([](auto &o) { o.predictor_count = 15; });
  rejects([](auto &o) { o.predictors[1] = o.predictors[0]; });
  rejects([](auto &o) { o.predictors[0] = Predictor::kInvalid; });
  rejects([](auto &o) { o.properties[0] = 1; });
  rejects([](auto &o) { o.properties[2] = 16; });
  rejects([](auto &o) { o.first_group = UINT32_MAX; });
  rejects([](auto &o) { o.split_threshold = -1; });
  rejects([](auto &o) { o.fast_decode_multiplier = 0.99; });
  image.channels[0].stride = 1;
  Check(!PrepareHistogramSamples(image.images, &output, options).ok() &&
            unchanged(),
        "Invalid view changed output");
  image.channels[0].stride = 33;
  image.images[0].metadata_channels = 1;
  Check(!PrepareHistogramSamples(image.images, &output, options).ok() &&
            unchanged(),
        "Invalid metadata changed output");
  image.images[0].metadata_channels = 0;
  Check(!PrepareHistogramSamples({}, &output, options).ok() && unchanged(),
        "Empty image list accepted");
  Check(!PrepareHistogramSamples(image.images, nullptr, options).ok(),
        "Null preparation output accepted");
  PreparedHistogramSamples unprepared;
  DecisionTree tree, old_tree = tree;
  Check(!LearnPreparedHistogramTree(unprepared, &tree).ok() && tree == old_tree,
        "Default unprepared samples fitted");

  HostStorageBound bound;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeHistogramSamplePreparationStorageBound(image.images, options,
                                                   &bound));
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Storage planner allocated");
  DisarmManagedHostAllocationFailureForTest();
  const auto old_bound = bound;
  auto bad_options = options;
  bad_options.property_count = 17;
  Check(!ComputeHistogramSamplePreparationStorageBound(image.images,
                                                       bad_options, &bound)
                .ok() &&
            bound == old_bound,
        "Invalid preparation storage planning changed output");
  for (bool coefficient : {false, true}) {
    auto &wp = image.images[0].weighted;
    if (coefficient)
      wp.coefficients[3] = 32;
    else
      wp.weights[2] = 16;
    ArmNextManagedHostAllocationFailureForTest();
    const auto planned = ComputeHistogramSamplePreparationStorageBound(
        image.images, options, &bound);
    const auto prepared =
        PrepareHistogramSamples(image.images, &output, options);
    Check(planned.code() == StatusCode::kInvalidArgument &&
              prepared.code() == planned.code() && bound == old_bound &&
              unchanged() && ManagedHostAllocationFailurePendingForTest(),
          "Invalid stream WP allocated or changed published output");
    DisarmManagedHostAllocationFailureForTest();
    wp = {};
  }
  // Exercise failure publication with a valid nondefault stream header too.
  image.images[0].weighted = {{8, 8, 4, 0, 3, 23, 2}, {13, 12, 12, 12}};
  size_t failures = 0;
  for (; failures < 100; ++failures) {
    ArmManagedHostAllocationFailureAfterForTest(failures);
    const auto status = PrepareHistogramSamples(image.images, &output, options);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (status.ok()) {
      Check(pending, "Failure injection unexpectedly consumed on success");
      break;
    }
    Check(status.code() == StatusCode::kOutOfMemory && !pending && unchanged(),
          "Allocation failure changed prepared output or escaped");
  }
  Check(failures >= 40 && failures < 100,
        "Allocation failure sweep incomplete");
}
void StorageAndPopulation() {
  Image image;
  for (bool unique : {false, true}) {
    for (bool custom_wp : {false, true}) {
      image.images[0].weighted =
          custom_wp ? WeightedPredictorParameters{{8, 8, 4, 0, 3, 23, 2},
                                                  {13, 12, 12, 12}}
                    : WeightedPredictorParameters{};
      for (size_t bins : {2, 8, 32, 256}) {
        HistogramSamplingOptions options;
        options.use_unique_row_proof = unique;
        options.maximum_property_values = bins;
        HostStorageBound bound;
        Ok(ComputeHistogramSamplePreparationStorageBound(image.images, options,
                                                         &bound));
        ResourceBudget budget(bound.peak_bytes);
        ResourceReservation reservation;
        Ok(budget.Reserve(bound.peak_bytes, &reservation));
        size_t peak;
        {
          ResourceContextScope scope(
              {&reservation, ResourceClass::kPreparation});
          PreparedHistogramSamples prepared;
          Ok(PrepareHistogramSamples(image.images, &prepared, options));
          peak = budget.snapshot().peak_backing_bytes;
          Check(peak <= bound.peak_bytes,
                "Preparation exceeded its storage bound");
          Check(prepared.stats().scanned_pixels == image.pixels.size() &&
                    prepared.stats().population == 2 * image.pixels.size() &&
                    prepared.stats().selected_samples ==
                        prepared.samples().total_samples &&
                    std::accumulate(prepared.samples().weights.begin(),
                                    prepared.samples().weights.end(),
                                    size_t{0}) ==
                        prepared.samples().total_samples,
                "Preparation lost selected population");
        }
        Check(budget.snapshot().total.backing_count == 0,
              "Preparation leaked managed storage");
        ResourceBudget short_budget(peak - 1);
        ResourceReservation short_reservation;
        Ok(short_budget.Reserve(peak - 1, &short_reservation));
        {
          ResourceContextScope scope(
              {&short_reservation, ResourceClass::kPreparation});
          PreparedHistogramSamples prepared;
          Check(PrepareHistogramSamples(image.images, &prepared, options)
                        .resource_plan_exceeded() &&
                    prepared.samples().weights.empty(),
                "Preparation escaped its admitted storage");
        }
        Check(short_budget.snapshot().total.backing_count == 0,
              "Failed preparation leaked managed storage");
      }
    }
  }
  // Tiny planes select every pixel, regardless of the requested fraction.
  std::array<int32_t, 3> pixels{0, -1, 65535};
  std::array<ModularChannelView, 1> channels{{{pixels, {{1, 3}}, 1}}};
  std::array<HistogramTrainingImage, 3> images{{{}, {{1, 3}, channels}, {}}};
  PreparedHistogramSamples prepared;
  HistogramSamplingOptions options;
  options.sampling_fraction = .01f;
  options.first_group = 7;
  Ok(PrepareHistogramSamples(images, &prepared, options));
  Check(prepared.stats().selected_samples == 3 &&
            prepared.stats().population == 4 &&
            prepared.stats().quantization_samples == 0 &&
            prepared.tree_options().static_range[1] ==
                std::array<uint32_t, 2>{7, 10},
        "Width-one quantization, group holes or minimum sampling changed");
}
void UniqueProof() {
  std::vector<int32_t> pixels(257 * 257, 0);
  auto channel = [&](uint32_t w, uint32_t h) {
    return ModularChannelView{
        std::span(pixels).first(size_t{w} * h), {{w, h}}, w};
  };
  auto check = [&](std::span<const HistogramTrainingImage> images,
                   HistogramSamplingOptions options, bool expected) {
    for (auto backend :
         {HistogramRowBackend::kRasterCpu, HistogramRowBackend::kSelectedCpu}) {
      options.row_backend = backend;
      options.use_unique_row_proof = false;
      PreparedHistogramSamples baseline, candidate;
      Ok(PrepareHistogramSamples(images, &baseline, options));
      options.use_unique_row_proof = true;
      Ok(PrepareHistogramSamples(images, &candidate, options));
      const auto &rows = candidate.stats().rows;
      Check(Same(baseline.samples(), candidate.samples()) &&
                baseline.tree_options().split_threshold ==
                    candidate.tree_options().split_threshold,
            "Unique-row proof changed native samples or thresholds");
      Check(rows.unique_calls == size_t{expected} &&
                rows.uniqueness_checks ==
                    size_t{candidate.samples().total_samples != 0} &&
                rows.unique_rows ==
                    (expected ? candidate.samples().total_samples : 0),
            "Unique-row proof accepted or rejected the wrong scope");
      if (expected)
        Check(candidate.samples().weights.size() ==
                      candidate.samples().total_samples &&
                  std::ranges::all_of(candidate.samples().weights,
                                      [](auto w) { return w == 1; }),
              "Proven rows did not retain unit weights");
      DecisionTree a, b;
      Ok(LearnPreparedHistogramTree(baseline, &a));
      Ok(LearnPreparedHistogramTree(candidate, &b));
      Check(a == b, "Unique-row proof changed the learned tree");
    }
  };
  std::array channels{channel(17, 19), channel(31, 23)};
  std::array<HistogramTrainingImage, 4> images{
      {{}, {{256, 256}, channels}, {}, {{256, 256}, channels}}};
  HistogramSamplingOptions options;
  options.first_group = 7;
  check(images, options, true); // Empty group holes and two independent keys.
  options.sampling_fraction = 1;
  options.predictor_count = 14;
  for (size_t p = 0; p < 14; ++p)
    options.predictors[p] = static_cast<Predictor>(13 - p);
  check(images, options, true);
  for (size_t bins : {2, 8}) {
    options.maximum_property_values = bins;
    check(images, options, false);
  }
  options.maximum_property_values = 255;
  check(images, options, true); // Actual maps, not a hard-coded 256-bin flag.
  options = {};
  options.first_group = 511;
  check(images, options, false); // Group IDs collapse through the native clamp.
  options = {};
  options.property_count = 4;
  options.properties = {3, 9, 2, 15};
  HistogramTrainingImage one{{256, 256}, std::span(channels).first(1)};
  check({&one, 1}, options, true); // A single slice needs no static columns.
  one.channels = channels;
  check({&one, 1}, options, false); // Missing channel key, identical pixels.
  options.property_count = 5;
  options.properties = {0, 3, 9, 2, 15};
  check({&one, 1}, options, true); // One group, partial static prefix.
  check(images, options, false);   // Missing group key.
  options.properties = {0, 3, 9, 8, 15};
  check({&one, 1}, options, false); // Missing y.
  options = {};
  std::array edge{channel(256, 256)};
  one.channels = edge;
  check({&one, 1}, options, true);
  options.maximum_property_values = 255;
  check({&one, 1}, options, false);
  options.maximum_property_values = 256;
  edge[0] = channel(257, 256);
  check({&one, 1}, options, false);
  edge[0] = channel(256, 257);
  check({&one, 1}, options, false);
  edge[0] = channel(1, 17);
  check({&one, 1}, options, true);
  std::array thin{channel(1, 17), channel(1, 17)};
  one.channels = thin;
  check({&one, 1}, options, false); // Width-one channels do not populate cuts.
  one.channels = channels;
  HistogramMultiplierRange multiplier;
  multiplier.range = {{{0, 2}, {0, 1}}};
  multiplier.multiplier = 2;
  std::array multipliers{multiplier};
  // A boundary after channel 1 groups channels 0/1 into the same static bin.
  std::array three{channel(17, 19), channel(17, 19), channel(17, 19)};
  one.channels = three;
  options.multipliers = multipliers;
  check({&one, 1}, options, false);
  options = {};
  std::array<ModularChannelView, 514> clamped{};
  clamped[511] = clamped[512] = channel(17, 19);
  one.channels = clamped;
  check({&one, 1}, options, false); // Channel IDs collapse at the same clamp.
  options.maximum_channel_size = 32;
  std::array excluded{channel(17, 19), channel(257, 1), channel(1, 17)};
  one.channels = excluded;
  check({&one, 1}, options, true); // Inclusion ends at first oversized channel.
  excluded[0] = channel(257, 1);
  excluded[0].descriptor.role = ChannelRole::kMetadata;
  one.metadata_channels = 1;
  check({&one, 1}, options, false); // Metadata is included despite size limit.
}
} // namespace
int main() try {
  ReusedFeatures();
  AdmissionAndPublication(false);
  AdmissionAndPublication(true);
  StorageAndPopulation();
  UniqueProof();
  std::cout << "Histogram preparation admission, population, storage and "
               "failure checks passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
