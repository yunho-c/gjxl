// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "native_preparation.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/transform/transform.h"
#include "native_fit.h"
#include <numeric>
#include <stdexcept>
#include <string>

namespace reference_learning {
namespace native = gjxl::modular_internal;
namespace {
void Require(bool good, const std::string &message) {
  if (!good)
    throw std::runtime_error(message);
}
void Ok(const gjxl::Status &status) {
  Require(status.ok(), std::string(status.message()));
}
template <typename A, typename B>
void Equal(const A &a, const B &b, const std::string &name) {
  Require(a.size() == b.size(),
          name + " size mismatch: " + std::to_string(a.size()) + "/" +
              std::to_string(b.size()));
  for (size_t i = 0; i < a.size(); ++i)
    Require(a[i] == b[i], name + " mismatch at " + std::to_string(i));
}
} // namespace

BorrowedTrainingImages
BorrowReferenceImages(std::span<const jxl::Image> images,
                      std::span<const jxl::ModularOptions> options) {
  Require(images.size() == options.size(),
          "Mismatched reference image/options");
  BorrowedTrainingImages result;
  result.channels.resize(images.size());
  result.images.resize(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    const auto &image = images[i];
    bool included = image.w && image.h;
    for (size_t c = 0; c < image.channel.size(); ++c) {
      const auto &channel = image.channel[c];
      if (c >= image.nb_meta_channels &&
          (channel.w > options[i].max_chan_size ||
           channel.h > options[i].max_chan_size))
        included = false;
      const auto stride =
          included ? static_cast<size_t>(channel.plane.PixelsPerRow()) : 0;
      const size_t size = included && channel.w && channel.h
                              ? (channel.h - 1) * stride + channel.w
                              : 0;
      native::ChannelDescriptor descriptor;
      descriptor.extent = {static_cast<uint32_t>(channel.w),
                           static_cast<uint32_t>(channel.h)};
      descriptor.hshift = std::max(0, channel.hshift);
      descriptor.vshift = std::max(0, channel.vshift);
      descriptor.role = c < image.nb_meta_channels
                            ? native::ChannelRole::kMetadata
                            : native::ChannelRole::kColor;
      result.channels[i].push_back(
          {{size ? channel.Row(0) : nullptr, size}, descriptor, stride});
    }
    result.images[i] = {
        {static_cast<uint32_t>(image.w), static_cast<uint32_t>(image.h)},
        result.channels[i],
        static_cast<size_t>(image.nb_meta_channels)};
    jxl::weighted::Header wp;
    if (jxl::PredictorHasWeighted(options[i].predictor)) {
      Require(options[i].wp_mode >= 0 && options[i].wp_mode <= 4,
              "Reference WP preset is outside the e9 profile");
      jxl::weighted::PredictorMode(options[i].wp_mode, &wp);
    }
    const std::array coefficients{wp.p1C,  wp.p2C,  wp.p3Ca, wp.p3Cb,
                                  wp.p3Cc, wp.p3Cd, wp.p3Ce};
    for (size_t j = 0; j < coefficients.size(); ++j) {
      Require(coefficients[j] >= 0 && coefficients[j] <= 31,
              "Invalid reference WP coefficient");
      result.images[i].weighted.coefficients[j] =
          static_cast<uint8_t>(coefficients[j]);
    }
    for (size_t j = 0; j < 4; ++j) {
      Require(wp.w[j] <= 15, "Invalid reference WP weight");
      result.images[i].weighted.weights[j] = static_cast<uint8_t>(wp.w[j]);
    }
  }
  return result;
}
native::HistogramSamplingOptions
NativeSamplingOptions(const jxl::ModularOptions &options) {
  jxl::TreeSamples shape;
  Require(bool(shape.SetPredictor(options.predictor, options.wp_tree_mode)) &&
              bool(shape.SetProperties(options.splitting_heuristics_properties,
                                       options.wp_tree_mode)),
          "Reference preparation options rejected");
  native::HistogramSamplingOptions result;
  result.maximum_samples = native::kHistogramMaximumSamples;
  result.sampling_fraction = options.nb_repeats;
  result.maximum_property_values = options.max_property_values;
  result.maximum_channel_size = options.max_chan_size;
  result.split_threshold = options.splitting_heuristics_node_threshold;
  result.fast_decode_multiplier = options.fast_decode_multiplier;
  result.predictor_count = shape.NumPredictors();
  result.property_count = shape.NumProperties();
  Require(result.predictor_count <= 14 && result.property_count <= 16,
          "Reference preparation exceeds native features");
  for (size_t p = 0; p < result.predictor_count; ++p)
    result.predictors[p] =
        static_cast<native::Predictor>(shape.PredictorFromIndex(p));
  for (size_t p = 0; p < result.property_count; ++p) {
    Require(shape.PropertyFromIndex(p) < 16,
            "Previous-channel properties unsupported");
    result.properties[p] = shape.PropertyFromIndex(p);
  }
  return result;
}
void CheckPreparedSamples(const native::PreparedHistogramSamples &prepared,
                          const Fixture &fixture) {
  const auto &a = prepared.samples();
  const auto b = NativeSamples(fixture.samples);
  Require(a.total_samples == b.total_samples &&
              a.predictor_count == b.predictor_count &&
              a.property_count == b.property_count &&
              a.static_property_count == b.static_property_count,
          "Prepared sample dimensions mismatch");
  Equal(a.weights, b.weights, "weights");
  for (size_t p = 0; p < a.predictor_count; ++p) {
    Require(a.predictors[p] == b.predictors[p], "Predictor order mismatch");
    Equal(a.residuals[p], b.residuals[p], "residual " + std::to_string(p));
  }
  for (size_t p = 0; p < a.property_count; ++p) {
    Require(a.properties[p] == b.properties[p], "Property order mismatch");
    Equal(a.thresholds[p], b.thresholds[p], "cuts " + std::to_string(p));
    if (p < a.static_property_count)
      Equal(a.static_values[p], b.static_values[p],
            "static " + std::to_string(p));
    else
      Equal(a.values[p - a.static_property_count],
            b.values[p - b.static_property_count],
            "property " + std::to_string(p));
  }
  const auto o = prepared.tree_options();
  Require(std::bit_cast<uint32_t>(o.split_threshold) ==
              std::bit_cast<uint32_t>(fixture.threshold),
          "Prepared threshold bits mismatch");
  Require(std::bit_cast<uint32_t>(o.fast_decode_multiplier) ==
              std::bit_cast<uint32_t>(fixture.fast_decode_multiplier),
          "Prepared predictor penalty bits mismatch");
  Require(o.static_range == fixture.range, "Prepared static ranges mismatch");
  Require(o.multipliers.size() == fixture.multipliers.size(),
          "Prepared multiplier count mismatch");
  for (size_t i = 0; i < o.multipliers.size(); ++i)
    Require(o.multipliers[i].range == fixture.multipliers[i].range &&
                o.multipliers[i].multiplier ==
                    fixture.multipliers[i].multiplier,
            "Prepared multiplier mismatch");
}
void CheckNativeTree(const native::DecisionTree &tree,
                     const jxl::Tree &expected) {
  Require(tree.size == expected.size(), "Prepared tree node count mismatch");
  for (size_t i = 0; i < tree.size; ++i) {
    const auto &a = tree.nodes[i];
    const auto &b = expected[i];
    Require(a.property == b.property && a.split == b.splitval &&
                a.left == b.lchild && a.right == b.rchild &&
                static_cast<unsigned>(a.predictor) ==
                    static_cast<unsigned>(b.predictor) &&
                a.offset == b.predictor_offset && a.multiplier == b.multiplier,
            "Prepared tree mismatch at node " + std::to_string(i));
  }
}
native::DecisionTree
CheckPreparedTree(native::PreparedHistogramSamples &prepared,
                  const jxl::Tree &expected) {
  native::DecisionTree tree;
  Ok(native::LearnPreparedHistogramTree(prepared, &tree));
  CheckNativeTree(tree, expected);
  return tree;
}

namespace {
bool CheckPreparationCase(
    const std::vector<jxl::Image> &images,
    std::span<const jxl::ModularOptions> options, size_t start,
    const std::vector<jxl::ModularMultiplierInfo> &multipliers = {}) {
  auto native_options = NativeSamplingOptions(options[start]);
  native_options.first_group = start;
  std::vector<native::HistogramMultiplierRange> native_multipliers;
  for (const auto &m : multipliers)
    native_multipliers.push_back({m.range, m.multiplier});
  native_options.multipliers = native_multipliers;
  const auto borrowed = BorrowReferenceImages(std::span(images).subspan(start),
                                              options.subspan(start));
  std::vector<Fixture> fixtures;
  Session session;
  session.captured = &fixtures;
  jxl::Tree expected;
  {
    const ScopedSession scope(&session);
    auto tree = jxl::LearnTree(images.data(), options.data(), start,
                               images.size(), multipliers);
    Require(tree.ok(), "Synthetic reference preparation failed");
    expected = std::move(tree).value_();
  }
  bool proven = false;
  for (bool proof : {false, true}) {
    native_options.use_unique_row_proof = proof;
    using namespace gjxl::resource_budget_internal;
    HostStorageBound bound;
    Ok(native::ComputeHistogramSamplePreparationStorageBound(
        borrowed.images, native_options, &bound));
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    native::PreparedHistogramSamples prepared;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      Ok(native::PrepareHistogramSamples(borrowed.images, &prepared,
                                         native_options));
    }
    Require(budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
            "Preparation exceeded storage bound");
    if (prepared.samples().total_samples) {
      Require(fixtures.size() == 1,
              "Synthetic reference capture count mismatch");
      CheckPreparedSamples(prepared, fixtures.front());
    } else {
      Require(fixtures.empty(),
              "Empty reference preparation unexpectedly fitted");
    }
    if (proof)
      proven = prepared.stats().rows.unique_calls != 0;
    CheckPreparedTree(prepared, expected);
  }
  return proven;
}
bool CheckPreparationCase(
    const std::vector<jxl::Image> &images, const jxl::ModularOptions &option,
    size_t start,
    const std::vector<jxl::ModularMultiplierInfo> &multipliers = {}) {
  const std::vector<jxl::ModularOptions> options(images.size(), option);
  return CheckPreparationCase(images, options, start, multipliers);
}
} // namespace

size_t RunPreparationDifferentialTests() {
  JxlMemoryManager manager;
  Require(bool(jxl::MemoryManagerInit(&manager, nullptr)),
          "Memory manager initialization failed");
  constexpr size_t cases = 96;
  for (size_t test = 0; test < cases; ++test) {
    try {
      const size_t start = test % 3, count = 1 + test % 4;
      std::vector<jxl::Image> images;
      for (size_t i = 0; i < start + count; ++i)
        images.emplace_back(&manager);
      uint32_t state = test + 1;
      auto random = [&] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
      };
      for (size_t i = start; i < images.size(); ++i) {
        auto &image = images[i];
        if (test % 7 == 0 && i == start)
          continue; // Preserve an empty group before nonempty ones.
        image.w = 129;
        image.h = 33;
        image.bitdepth = 16;
        image.error = false;
        for (size_t c = 0; c < 1 + test % 4; ++c) {
          const size_t w = test % 11 == 0 ? 1 : 2 + (random() % 128);
          const size_t h = 1 + random() % 33;
          auto created = jxl::Channel::Create(&manager, w, h, 0, 0);
          Require(created.ok(), "Synthetic plane allocation failed");
          auto channel = std::move(created).value_();
          for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x) {
              int32_t value = int32_t(random() % 131071) - 65535;
              if (test % 6 == 0)
                value = -1 - (random() % 997 == 0);
              if (test % 6 == 1)
                value = (x + y + c) % 3 - 1;
              if (test % 6 == 2)
                value = 0;
              channel.Row(y)[x] = value;
            }
          image.channel.push_back(std::move(channel));
        }
        if (test % 5 == 0)
          image.nb_meta_channels = 1;
      }
      jxl::ModularOptions option;
      constexpr std::array predictors{
          jxl::Predictor::Best, jxl::Predictor::Variable,
          jxl::Predictor::Gradient, jxl::Predictor::Zero};
      constexpr std::array<float, 4> fractions{0.65f, 1.f, 0.05f, 0.5f};
      constexpr std::array<size_t, 6> bins{2, 8, 16, 32, 96, 256};
      option.predictor = predictors[test % predictors.size()];
      option.nb_repeats = fractions[(test / 4) % fractions.size()];
      option.max_property_values = bins[(test / 3) % bins.size()];
      option.fast_decode_multiplier = 1.001f;
      option.splitting_heuristics_node_threshold = 89;
      option.splitting_heuristics_properties =
          test % 2 ? std::vector<uint32_t>{0,  1, 15, 9, 10, 11, 12, 13,
                                           14, 2, 3,  4, 5,  6,  7,  8}
                   : std::vector<uint32_t>{0, 1,  4,  5,  6,  7,  8, 15,
                                           9, 10, 11, 12, 13, 14, 2, 3};
      CheckPreparationCase(images, option, start);
    } catch (const std::exception &e) {
      throw std::runtime_error("Preparation case " + std::to_string(test) +
                               ": " + e.what());
    }
  }
  auto add_channel = [&](jxl::Image &image, size_t width, size_t height,
                         int32_t value) {
    image.w = std::max(image.w, width);
    image.h = std::max(image.h, height);
    image.bitdepth = 16;
    image.error = false;
    auto created = jxl::Channel::Create(&manager, width, height, 0, 0);
    Require(created.ok(), "Special-case allocation failed");
    auto channel = std::move(created).value_();
    for (size_t y = 0; y < height; ++y)
      std::fill_n(channel.Row(y), width, value);
    image.channel.push_back(std::move(channel));
  };
  jxl::ModularOptions option;
  option.nb_repeats = 1.f;
  option.predictor = jxl::Predictor::Variable;
  option.max_property_values = 8;
  option.fast_decode_multiplier = 1.001f;
  option.splitting_heuristics_node_threshold = 89;
  option.splitting_heuristics_properties = {9};
  std::vector<jxl::Image> special;
  special.emplace_back(&manager);
  add_channel(special[0], 64, 2048, 0);
  CheckPreparationCase(special, option, 0); // Row weights saturate twice.
  // Pinned InitTable retains old slots on growth, then reinserts the same row
  // into its second slot. Saturating one alias leaves the other alias live;
  // the next increment wraps the reference's uint16_t weight. Native must
  // reject this unsupported case rather than publish a corrupt population.
  special.clear();
  special.emplace_back(&manager);
  add_channel(special[0], 2, 15000, 1);
  add_channel(special[0], 2, 25000, 1);
  {
    auto overflow_option = option;
    overflow_option.predictor = jxl::Predictor::Zero;
    const std::vector<jxl::ModularOptions> overflow_options(special.size(),
                                                            overflow_option);
    const auto borrowed = BorrowReferenceImages(special, overflow_options);
    for (bool proof : {false, true}) {
      auto native_options = NativeSamplingOptions(overflow_option);
      native_options.use_unique_row_proof = proof;
      native::PreparedHistogramSamples prepared;
      const auto status = native::PrepareHistogramSamples(
          borrowed.images, &prepared, native_options);
      Require(
          status.code() == gjxl::StatusCode::kUnsupported &&
              prepared.samples().weights.empty(),
          "Saturated duplicate alias was accepted or published: " +
              std::string(status.message()) +
              ", rows=" + std::to_string(prepared.samples().weights.size()) +
              ", selected=" + std::to_string(prepared.samples().total_samples));
    }
    std::vector<Fixture> fixtures;
    Session session;
    session.captured = &fixtures;
    const ScopedSession scope(&session);
    const auto tree = jxl::LearnTree(special.data(), &overflow_option, 0, 1);
    Require(tree.ok() && fixtures.size() == 1,
            "Reference overflow fixture failed");
    const auto &samples = fixtures[0].samples;
    size_t weighted = 0;
    for (size_t i = 0; i < samples.NumDistinctSamples(); ++i)
      weighted += samples.Count(i);
    Require(samples.NumSamples() == 80000 && weighted == 80000 - 65536,
            "Pinned reference weight-overflow reproduction changed");
  }
  special.clear();
  special.emplace_back(&manager);
  add_channel(special[0], 16, 16, -1);
  add_channel(special[0], 17, 32, 2);
  add_channel(special[0], 33, 128, -7);
  CheckPreparationCase(special, option, 0); // Table grows with duplicate rows.
  special[0].nb_meta_channels = 1;
  option.max_chan_size = 16;
  CheckPreparationCase(special, option,
                       0); // Metadata followed by first oversized channel.
  option.max_chan_size = 0xFFFFFF;
  option.splitting_heuristics_properties = {0, 1,  4,  5,  6,  7,  8, 15,
                                            9, 10, 11, 12, 13, 14, 2, 3};
  for (size_t i = 1; i < 3; ++i) {
    special.emplace_back(&manager);
    for (size_t c = 0; c < 3; ++c)
      add_channel(special[i], 17, 19, i + c);
  }
  const std::vector<jxl::ModularMultiplierInfo> multipliers{
      {{{{0, 1}, {0, 3}}}, 2},
      {{{{1, 3}, {0, 1}}}, 3},
      {{{{1, 3}, {1, 3}}}, 4}};
  CheckPreparationCase(special, option, 0, multipliers);
  // Static group values beyond the quantizer's clamp, wide coordinates, and
  // property-cache ordering on a nearly constant negative distribution.
  special.clear();
  for (size_t i = 0; i < 523; ++i)
    special.emplace_back(&manager);
  add_channel(special[520], 1025, 3, -1);
  special[520].channel[0].Row(0)[100] = -2;
  add_channel(special[522], 513, 5, -1);
  CheckPreparationCase(special, option, 520);

  // Five presets plus a mixed-preset batch, with Best, all predictors and a
  // nonweighted predictor. In the last case the reference ignores wp_mode,
  // including for property 15, so the effective header must remain default.
  special.clear();
  for (size_t i = 0; i < 8; ++i) {
    special.emplace_back(&manager);
    if (i < 2 || i == 5)
      continue;
    add_channel(special[i], 862, 3, 0);
    special[i].nb_meta_channels = 1;
    for (size_t c = 0; c < 3; ++c)
      add_channel(special[i], 43 + i, 35 + c, 0);
    for (size_t c = 0; c < special[i].channel.size(); ++c) {
      auto &ch = special[i].channel[c];
      for (size_t y = 0; y < ch.h; ++y)
        for (size_t x = 0; x < ch.w; ++x)
          ch.Row(y)[x] = static_cast<int32_t>((x * 1973 + y * 503 + x * y * 31 +
                                               c * 997 + i * 7) %
                                              131071) -
                         65535;
    }
  }
  option.max_chan_size = 128;
  option.nb_repeats = 0.65f;
  option.max_property_values = 256;
  for (auto predictor : {jxl::Predictor::Best, jxl::Predictor::Variable,
                         jxl::Predictor::Gradient}) {
    option.predictor = predictor;
    for (int mode = 0; mode < 6; ++mode) {
      std::vector<jxl::ModularOptions> options(special.size(), option);
      for (size_t i = 0; i < options.size(); ++i)
        options[i].wp_mode = mode == 5 ? i % 5 : mode;
      CheckPreparationCase(special, options, 2);
    }
  }
  special.clear();
  for (size_t i = 0; i < 3; ++i) {
    special.emplace_back(&manager);
    if (i == 1)
      continue;
    add_channel(special.back(), 31, 37, -1);
    add_channel(special.back(), 32, 37, 2);
  }
  option.predictor = jxl::Predictor::Variable;
  Require(CheckPreparationCase(special, option, 0),
          "Known unique native fixture did not exercise the proof");
  option.max_property_values = 8;
  Require(!CheckPreparationCase(special, option, 0),
          "Coarse coordinate bins incorrectly proved unique");
  return cases + 5 + 18 + 2;
}
} // namespace reference_learning
