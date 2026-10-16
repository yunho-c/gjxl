// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Sampling, quantization and deduplication adapted from pinned libjxl
// e8ff0976 enc_ma.cc, enc_encoding.cc and base/random.h.
#include "codec/modular/histogram_sample_preparation.h"
#include "codec/modular/profile.h"

#include <cmath>
#include <numeric>
#include <optional>

namespace gjxl::modular_internal {
namespace {
using resource_budget_internal::ManagedVector;
constexpr uint32_t kUnused = UINT32_MAX;
constexpr size_t kMapSize = 1023;

struct Random {
  uint64_t s[2];
  explicit Random(uint64_t seed = 0)
      : s{0x94D049BB133111EBull, 0xBF58476D1CE4E5B9ull + seed} {}
  uint64_t Next() {
    auto a = s[0];
    const auto b = s[1], result = a + b;
    s[0] = b;
    a ^= a << 23;
    a ^= b ^ (a >> 18) ^ (b >> 5);
    s[1] = a;
    return result;
  }
  uint32_t Geometric(float inverse_log) {
    const uint32_t bits = static_cast<uint32_t>(Next() >> 41) | 0x3F800000;
    const float f = std::bit_cast<float>(bits) - 1.0f;
    const float value = std::log(1 - f) * inverse_log;
    return static_cast<uint32_t>(value);
  }
};
double PixelFraction(size_t area, float requested) {
  return std::max<double>(std::min(1.0f, requested),
                          std::min(1.0, 1024.0 / area));
}
uint64_t SamplingThreshold(double fraction) {
  return uint64_t{UINT32_MAX} * fraction;
}
size_t CountSelected(size_t area, float requested) {
  const double fraction = PixelFraction(area, requested);
  if (fraction == 1.0)
    return area;
  const auto threshold = SamplingThreshold(fraction);
  Random rng;
  size_t selected = 0;
  for (size_t i = 0; i < area; ++i)
    selected += (rng.Next() >> 32) <= threshold;
  return selected;
}
bool Included(const HistogramTrainingImage &image, size_t c,
              HistogramSamplingOptions options) {
  const auto [w, h] = image.channels[c].descriptor.extent;
  return c < image.metadata_channels || (w <= options.maximum_channel_size &&
                                         h <= options.maximum_channel_size);
}
size_t StaticCount(HistogramSamplingOptions options) {
  size_t count = 0;
  while (count < options.property_count && options.properties[count] == count &&
         count < 2)
    ++count;
  return count;
}
size_t CutCapacity(HistogramSamplingOptions options, size_t property_index) {
  if (property_index < StaticCount(options))
    return std::max(options.maximum_property_values,
                    2 * options.multipliers.size());
  // The fixed WP cuts have 15 entries even with fewer requested quantiles.
  return options.properties[property_index] == 15
             ? std::max<size_t>(options.maximum_property_values, 15)
             : options.maximum_property_values;
}
struct Plan {
  HistogramPreparationStats stats;
  size_t maximum_width = 0, maximum_channels = 0, groups = 0, hash_capacity = 0;
};
Status Preflight(std::span<const HistogramTrainingImage> images,
                 HistogramSamplingOptions options, Plan *out) {
  if (images.empty() || images.size() > 65535 ||
      options.first_group > 65535 - images.size() || !options.maximum_samples ||
      options.maximum_samples > kHistogramMaximumSamples ||
      !std::isfinite(options.sampling_fraction) ||
      options.sampling_fraction <= 0 || options.sampling_fraction > 1 ||
      !std::isfinite(options.split_threshold) || options.split_threshold < 0 ||
      !std::isfinite(options.fast_decode_multiplier) ||
      options.fast_decode_multiplier < 1 || !options.maximum_channel_size ||
      options.maximum_property_values < 2 ||
      options.maximum_property_values > 256 || !options.predictor_count ||
      options.predictor_count > 14 || !options.property_count ||
      options.property_count > 16 || options.multipliers.size() > 65536)
    return Status::InvalidArgument("Invalid histogram sampling options");
  const float quantization_fraction =
      std::min(options.sampling_fraction * 0.1, 0.99);
  const float inverse_log = 1.0 / std::log(1 - quantization_fraction);
  if (!std::isfinite(inverse_log))
    return Status::Unsupported("Histogram sampling fraction is too small for "
                               "reference geometric sampling");
  std::array<bool, 16> seen{};
  for (size_t i = 0; i < options.predictor_count; ++i) {
    const size_t p = static_cast<size_t>(options.predictors[i]);
    if (p >= 14 || seen[p])
      return Status::InvalidArgument("Invalid histogram sampling predictors");
    seen[p] = true;
  }
  seen = {};
  const size_t statics = StaticCount(options);
  for (size_t i = 0; i < options.property_count; ++i) {
    const auto p = options.properties[i];
    if (p >= 16 || seen[p] || (i >= statics && p < 2))
      return Status::InvalidArgument("Invalid histogram sampling properties");
    seen[p] = true;
  }
  Plan plan;
  plan.groups = options.first_group + images.size();
  size_t initial_population = 0, previous_area = 0, previous_selected = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    const auto &image = images[i];
    if (image.channels.size() > 65535 ||
        image.metadata_channels > image.channels.size() ||
        !image.weighted.valid() ||
        (!image.weighted_features.empty() &&
         image.weighted_features.size() != image.channels.size()) ||
        (image.extent.empty() && !image.channels.empty()))
      return Status::InvalidArgument("Invalid histogram training image");
    for (size_t c = 0; c < image.weighted_features.size(); ++c)
      if (auto s = ValidateWeightedFeatureView(image.weighted_features[c],
                                               image.channels[c], image.weighted);
          !s.ok())
        return s;
    plan.maximum_channels =
        std::max(plan.maximum_channels, image.channels.size());
    for (size_t c = 0; c < image.channels.size() && Included(image, c, options);
         ++c) {
      const auto &view = image.channels[c];
      const auto [w, h] = view.descriptor.extent;
      if ((view.descriptor.role == ChannelRole::kMetadata) !=
          (c < image.metadata_channels))
        return Status::InvalidArgument("Invalid histogram metadata prefix");
      if (!w || !h)
        continue;
      if (auto status = view.Validate(); !status.ok())
        return status;
      const uint64_t area64 = uint64_t{w} * h;
      if (area64 > kHistogramMaximumPopulation - plan.stats.scanned_pixels)
        return Status::Unsupported(
            "Histogram training population exceeds private admission");
      const size_t area = static_cast<size_t>(area64);
      plan.stats.scanned_pixels += area;
      if (i == 0)
        initial_population += area;
      ++plan.stats.channels;
      if (image.weighted_features.empty() || image.weighted_features[c].values.empty())
        plan.maximum_width = std::max<size_t>(plan.maximum_width, w);
      if (area != previous_area) {
        previous_area = area;
        previous_selected = CountSelected(area, options.sampling_fraction);
      }
      if (previous_selected >
          options.maximum_samples - plan.stats.selected_samples)
        return Status::Unsupported(
            "Histogram training samples exceed admission; no truncation");
      // Preserve the reference expression's evaluation order and +64 reserve.
      const size_t guess =
          PixelFraction(area, options.sampling_fraction) * h * w + 64;
      const size_t hash_request = (plan.stats.selected_samples + guess) * 3 / 2;
      plan.hash_capacity =
          std::max(plan.hash_capacity, std::bit_ceil(hash_request));
      plan.stats.selected_samples += previous_selected;
    }
  }
  // LearnTree initializes this denominator with the first image's population,
  // then GatherTreeData adds every visited pixel, including that image again.
  plan.stats.population =
      std::max<size_t>(initial_population, 1) + plan.stats.scanned_pixels;
  const HistogramStaticRange range{
      {{0, static_cast<uint32_t>(plan.maximum_channels)},
       {options.first_group, static_cast<uint32_t>(plan.groups)}}};
  for (const auto &m : options.multipliers) {
    if (!m.multiplier || m.multiplier > INT32_MAX)
      return Status::InvalidArgument(
          "Invalid histogram preparation multiplier");
    for (size_t p = 0; p < 2; ++p)
      if (m.range[p][0] < range[p][0] || m.range[p][1] > range[p][1] ||
          m.range[p][0] >= m.range[p][1])
        return Status::InvalidArgument(
            "Multiplier box is outside the training images");
  }
  *out = plan;
  return Status::Ok();
}
template <typename T>
void QuantizeHistogram(std::span<const uint32_t> histogram, int32_t origin,
                       size_t chunks, T &cuts) {
  cuts.clear();
  const uint64_t total =
      std::accumulate(histogram.begin(), histogram.end(), uint64_t{0});
  if (!total)
    return;
  uint64_t cumulative = 0, threshold = 1;
  for (size_t i = 0; i < histogram.size(); ++i) {
    cumulative += histogram[i];
    if (cumulative * chunks >= threshold * total) {
      cuts.push_back(static_cast<int32_t>(i) + origin);
      while (cumulative * chunks >= threshold * total)
        ++threshold;
    }
  }
  cuts.pop_back();
}
HistogramResidualToken ResidualToken(int32_t residual) {
  const uint32_t value = (static_cast<uint32_t>(residual) << 1) ^
                         static_cast<uint32_t>(residual >> 31);
  if (value < 16)
    return {static_cast<uint8_t>(value), 0};
  const uint32_t n = std::bit_width(value) - 1;
  const uint32_t m = value - (uint32_t{1} << n);
  const uint32_t token = 16 + ((n - 4) << 3) + ((m >> (n - 1)) << 2) + (m & 3);
  return {static_cast<uint8_t>(token), static_cast<uint8_t>(n - 3)};
}
} // namespace

struct HistogramSampleBuilder {
  PreparedHistogramSamples *output = nullptr;
  HistogramSamplingOptions options;
  Plan plan;
  ManagedVector<uint32_t> group_counts, channel_counts, dedup;
  ManagedVector<int32_t> pixel_cuts, absolute_cuts, diff_cuts;
  std::array<ManagedVector<int32_t>, 2> multiplier_cuts;
  std::array<uint32_t, 1025> pixels{}, differences{};
  std::array<uint32_t, 513> absolute_pixels{};
  std::array<std::array<uint16_t, kMapSize>, 2> static_mapping{};
  std::array<std::array<uint8_t, kMapSize>, 16> mapping{};

  void Allocate() {
    auto &s = output->samples_;
    s.predictor_count = options.predictor_count;
    s.property_count = options.property_count;
    s.static_property_count = StaticCount(options);
    s.predictors = options.predictors;
    s.properties = options.properties;
    s.weights.reserve(plan.stats.selected_samples);
    for (size_t p = 0; p < s.predictor_count; ++p)
      s.residuals[p].reserve(plan.stats.selected_samples);
    for (size_t p = 0; p < s.static_property_count; ++p)
      s.static_values[p].reserve(plan.stats.selected_samples);
    for (size_t p = 0; p < s.property_count - s.static_property_count; ++p)
      s.values[p].reserve(plan.stats.selected_samples);
    for (size_t p = 0; p < s.property_count; ++p)
      s.thresholds[p].reserve(CutCapacity(options, p));
    group_counts.resize(plan.groups);
    channel_counts.resize(plan.maximum_channels);
    dedup.reserve(plan.hash_capacity);
    pixel_cuts.reserve(options.maximum_property_values);
    absolute_cuts.reserve(options.maximum_property_values);
    diff_cuts.reserve(options.maximum_property_values);
    for (auto &cuts : multiplier_cuts)
      cuts.reserve(2 * options.multipliers.size());
    output->multipliers_.assign(options.multipliers.begin(),
                                options.multipliers.end());
  }
  Status CollectQuantization(std::span<const HistogramTrainingImage> images) {
    const float fraction = std::min(options.sampling_fraction * 0.1, 0.99);
    const float inverse_log = 1.0 / std::log(1 - fraction);
    for (size_t i = 0; i < images.size(); ++i) {
      const auto &image = images[i];
      const size_t group = options.first_group + i;
      Random rng(group);
      size_t skip = rng.Geometric(inverse_log);
      for (size_t c = 0;
           c < image.channels.size() && Included(image, c, options); ++c) {
        const auto &view = image.channels[c];
        const auto [w, h] = view.descriptor.extent;
        if (w <= 1 || !h)
          continue;
        const size_t area = size_t{w} * h;
        group_counts[group] += area;
        channel_counts[c] += area;
        while (skip < area) {
          const size_t x = skip % w, y = skip / w;
          const auto *row = view.Row(y);
          const int32_t value = row[x];
          const int64_t difference = int64_t{value} - row[x == 0 ? 1 : x - 1];
          if (value == INT32_MIN || difference < INT32_MIN ||
              difference > INT32_MAX)
            return Status::Unsupported(
                "Histogram quantization sample exceeds reference arithmetic");
          ++pixels[std::clamp(value, -512, 512) + 512];
          ++absolute_pixels[std::min<int64_t>(std::abs(int64_t{value}), 512)];
          ++differences[std::clamp<int64_t>(difference, -512, 512) + 512];
          ++plan.stats.quantization_samples;
          skip += rng.Geometric(inverse_log) + size_t{1};
        }
        skip -= area;
      }
    }
    return Status::Ok();
  }
  void Quantize() {
    auto &s = output->samples_;
    for (const auto &m : options.multipliers) {
      for (size_t p = 0; p < 2; ++p) {
        if (m.range[p][0] != output->range_[p][0])
          multiplier_cuts[p].push_back(m.range[p][0] - 1);
        if (m.range[p][1] != output->range_[p][1])
          multiplier_cuts[p].push_back(m.range[p][1] - 1);
      }
    }
    for (auto &cuts : multiplier_cuts) {
      std::sort(cuts.begin(), cuts.end());
      cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    }
    bool pixels_are_absolute = false;
    auto quantize_pixel = [&] {
      if (pixel_cuts.empty()) {
        if (pixels_are_absolute)
          QuantizeHistogram(std::span<const uint32_t>(absolute_pixels), 0,
                            options.maximum_property_values, pixel_cuts);
        else
          QuantizeHistogram(std::span<const uint32_t>(pixels), -512,
                            options.maximum_property_values, pixel_cuts);
      }
    };
    for (size_t p = 0; p < s.property_count; ++p) {
      const auto property = s.properties[p];
      auto &cuts = s.thresholds[p];
      if (property < 2) {
        if (!multiplier_cuts[property].empty())
          cuts.assign(multiplier_cuts[property].begin(),
                      multiplier_cuts[property].end());
        else
          QuantizeHistogram(property == 0
                                ? std::span<const uint32_t>(channel_counts)
                                : std::span<const uint32_t>(group_counts),
                            0, options.maximum_property_values, cuts);
      } else if (property == 2 || property == 3) {
        for (size_t i = 0; i + 1 < options.maximum_property_values; ++i)
          cuts.push_back((i + 1) * 256 / options.maximum_property_values - 1);
      } else if (property == 6 || property == 7 || property == 8) {
        quantize_pixel();
        cuts.assign(pixel_cuts.begin(), pixel_cuts.end());
      } else if (property == 4 || property == 5) {
        // Empty is the reference's cache sentinel, including when quantization
        // legitimately produces no cuts. Preserve its order-dependent reuse.
        if (absolute_cuts.empty()) {
          quantize_pixel();
          pixels_are_absolute = true;
          QuantizeHistogram(std::span<const uint32_t>(absolute_pixels), 0,
                            options.maximum_property_values, absolute_cuts);
        }
        cuts.assign(absolute_cuts.begin(), absolute_cuts.end());
      } else if (property == 15) {
        if (options.maximum_property_values < 32)
          cuts = {-127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31, 63, 127};
        else if (options.maximum_property_values < 64)
          cuts = {-255, -191, -127, -95, -63, -47, -31, -23, -15, -11,
                  -7,   -5,   -3,   -1,  0,   1,   3,   5,   7,   11,
                  15,   23,   31,   47,  63,  95,  127, 191, 255};
        else
          cuts = {-255, -223, -191, -159, -127, -111, -95, -79, -63, -55, -47,
                  -39,  -31,  -27,  -23,  -19,  -15,  -13, -11, -9,  -7,  -6,
                  -5,   -4,   -3,   -2,   -1,   0,    1,   2,   3,   4,   5,
                  6,    7,    9,    11,   13,   15,   19,  23,  27,  31,  39,
                  47,   55,   63,   79,   95,   111,  127, 159, 191, 223, 255};
      } else {
        if (diff_cuts.empty())
          QuantizeHistogram(std::span<const uint32_t>(differences), -512,
                            options.maximum_property_values, diff_cuts);
        cuts.assign(diff_cuts.begin(), diff_cuts.end());
      }
      size_t mapped = 0;
      for (size_t i = 0; i < kMapSize; ++i) {
        while (mapped < cuts.size() && static_cast<int>(i) - 511 > cuts[mapped])
          ++mapped;
        if (p < s.static_property_count)
          static_mapping[p][i] = static_cast<uint16_t>(mapped);
        else
          mapping[p - s.static_property_count][i] =
              static_cast<uint8_t>(mapped);
      }
    }
  }
  size_t Hash(size_t row, bool second) const {
    const auto &s = output->samples_;
    uint64_t h = second ? 0x1e35a7bd1e35a7bdull : 0x1e35a7bd;
    const uint64_t factor = h;
    auto residuals = [&] {
      for (size_t p = 0; p < s.predictor_count; ++p) {
        const auto token = s.residuals[p][row];
        h = second ? (h * factor) ^ token.symbol : h * factor + token.symbol;
        h = second ? (h * factor) ^ token.extra_bits
                   : h * factor + token.extra_bits;
      }
    };
    if (!second)
      residuals();
    for (size_t p = 0; p < s.static_property_count; ++p)
      h = second ? (h * factor) ^ s.static_values[p][row]
                 : h * factor + s.static_values[p][row];
    for (size_t p = 0; p < s.property_count - s.static_property_count; ++p)
      h = second ? (h * factor) ^ s.values[p][row]
                 : h * factor + s.values[p][row];
    if (second)
      residuals();
    return (h >> 16) & (dedup.size() - 1);
  }
  bool SameRow(size_t a, size_t b) const {
    const auto &s = output->samples_;
    bool same = true;
    for (size_t p = 0; p < s.predictor_count; ++p)
      same &= s.residuals[p][a] == s.residuals[p][b];
    for (size_t p = 0; p < s.static_property_count; ++p)
      same &= s.static_values[p][a] == s.static_values[p][b];
    for (size_t p = 0; p < s.property_count - s.static_property_count; ++p)
      same &= s.values[p][a] == s.values[p][b];
    return same;
  }
  void Insert(size_t row) {
    const size_t first = Hash(row, false), second = Hash(row, true);
    if (dedup[first] == kUnused)
      dedup[first] = static_cast<uint32_t>(row);
    else if (dedup[second] == kUnused)
      dedup[second] = static_cast<uint32_t>(row);
  }
  void BeginChannel(size_t width, size_t height) {
    const size_t guess =
        PixelFraction(width * height, options.sampling_fraction) * height *
            width +
        64;
    const size_t size =
        std::bit_ceil((output->samples_.weights.size() + guess) * 3 / 2);
    if (dedup.size() == size)
      return;
    // The reference retains the existing prefix when resizing its table.
    dedup.resize(size, kUnused);
    for (size_t row = 0; row < output->samples_.weights.size(); ++row)
      if (output->samples_.weights[row] != UINT16_MAX)
        Insert(row);
  }
  Status Add(const PredictionProperties &properties,
             const std::array<HistogramResidualToken, 14> &tokens) {
    auto &s = output->samples_;
    if (s.total_samples >= plan.stats.selected_samples)
      return Status::Internal("Histogram sample preflight count mismatch");
    for (size_t p = 0; p < s.predictor_count; ++p)
      s.residuals[p].push_back(tokens[p]);
    for (size_t p = 0; p < s.static_property_count; ++p)
      s.static_values[p].push_back(
          static_mapping[p]
                        [std::clamp<int64_t>(properties[p], -511, 511) + 511]);
    for (size_t p = s.static_property_count; p < s.property_count; ++p)
      s.values[p - s.static_property_count].push_back(
          mapping[p - s.static_property_count]
                 [std::clamp<int64_t>(properties[s.properties[p]], -511, 511) +
                  511]);
    const size_t row = s.weights.size();
    s.weights.push_back(1);
    ++s.total_samples;
    for (const size_t position : {Hash(row, false), Hash(row, true)}) {
      const auto old = dedup[position];
      if (old == kUnused || !SameRow(row, old))
        continue;
      if (s.weights[old] == UINT16_MAX)
        return Status::Unsupported(
            "Reference deduplication would overflow a row weight");
      if (++s.weights[old] == UINT16_MAX)
        dedup[position] = kUnused;
      for (size_t p = 0; p < s.predictor_count; ++p)
        s.residuals[p].pop_back();
      for (size_t p = 0; p < s.static_property_count; ++p)
        s.static_values[p].pop_back();
      for (size_t p = 0; p < s.property_count - s.static_property_count; ++p)
        s.values[p].pop_back();
      s.weights.pop_back();
      return Status::Ok();
    }
    Insert(row);
    return Status::Ok();
  }
  Status Gather(std::span<const HistogramTrainingImage> images) {
    for (size_t i = 0; i < images.size(); ++i) {
      const auto &image = images[i];
      for (size_t c = 0;
           c < image.channels.size() && Included(image, c, options); ++c) {
        const auto &view = image.channels[c];
        const auto [w, h] = view.descriptor.extent;
        if (!w || !h)
          continue;
        BeginChannel(w, h);
        Random rng;
        const auto threshold = SamplingThreshold(
            PixelFraction(size_t{w} * h, options.sampling_fraction));
        const auto features = image.weighted_features.empty()
                                  ? std::span<const WeightedPredictorFeature>{}
                                  : image.weighted_features[c].values;
        std::optional<WeightedPredictor<resource_budget_internal::ResourceClass::kPreparation>> wp;
        if (features.empty())
          wp.emplace(w, image.weighted);
        if (active_modular_profile) {
          auto &work = active_modular_profile->training_work;
          (features.empty() ? work.weighted_prediction_samples : work.reused_prediction_samples) +=
              size_t{w} * h;
        }
        for (size_t y = 0; y < h; ++y) {
          int64_t previous_gradient = 0;
          for (size_t x = 0; x < w; ++x) {
            const auto n = Neighbors(view, x, y);
            const auto weighted = wp ? wp->Predict(x, y, n.top, n.left, n.top_right,
                                                   n.top_left, n.top_top)
                                     : std::pair{features[y * w + x].prediction,
                                                 features[y * w + x].property15};
            const auto properties =
                Properties(n, c, options.first_group + i, x, y,
                           previous_gradient, weighted.second);
            previous_gradient = properties[9];
            if ((rng.Next() >> 32) <= threshold) {
              std::array<HistogramResidualToken, 14> tokens{};
              for (size_t p = 0; p < options.predictor_count; ++p) {
                const int64_t residual =
                    int64_t{view.Row(y)[x]} -
                    Predict(options.predictors[p], n, weighted.first);
                if (residual < INT32_MIN || residual > INT32_MAX)
                  return Status::Unsupported(
                      "Histogram training residual exceeds supported range");
                tokens[p] = ResidualToken(static_cast<int32_t>(residual));
              }
              if (auto status = Add(properties, tokens); !status.ok())
                return status;
            }
            if (wp && !wp->Update(view.Row(y)[x], x, y))
              return Status::Unsupported(
                  "Histogram training weighted state exceeds supported range");
          }
        }
      }
    }
    return Status::Ok();
  }
};

Status ComputeHistogramSamplePreparationStorageBound(
    std::span<const HistogramTrainingImage> images,
    HistogramSamplingOptions options,
    resource_budget_internal::HostStorageBound *out) {
  if (!out)
    return Status::InvalidArgument("Null histogram preparation storage output");
  Plan plan;
  if (auto status = Preflight(images, options, &plan); !status.ok())
    return status;
  using namespace resource_budget_internal;
  HostStorageBound bound;
  const size_t statics = StaticCount(options),
               rows = plan.stats.selected_samples;
  if (!bound.AddVector<HistogramSampleBuilder>(
          1, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<uint16_t>(rows, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<HistogramResidualToken>(
          rows, VectorCapacityPolicy::kFreshExact, options.predictor_count) ||
      !bound.AddVector<uint32_t>(rows, VectorCapacityPolicy::kFreshExact,
                                 statics) ||
      !bound.AddVector<uint8_t>(rows, VectorCapacityPolicy::kFreshExact,
                                options.property_count - statics) ||
      !bound.AddVector<uint32_t>(plan.groups,
                                 VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<uint32_t>(plan.maximum_channels,
                                 VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<uint32_t>(plan.hash_capacity,
                                 VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int32_t>(options.maximum_property_values,
                                VectorCapacityPolicy::kFreshExact, 3) ||
      !bound.AddVector<int32_t>(2 * options.multipliers.size(),
                                VectorCapacityPolicy::kFreshExact, 2) ||
      !bound.AddVector<HistogramMultiplierRange>(
          options.multipliers.size(), VectorCapacityPolicy::kFreshExact) ||
      (plan.maximum_width &&
       !bound.AddVector<uint32_t>((plan.maximum_width + 2) * 2,
                                  VectorCapacityPolicy::kFreshExact, 5)))
    return Status::InvalidArgument("Histogram preparation storage overflow");
  for (size_t p = 0; p < options.property_count; ++p)
    if (!bound.AddVector<int32_t>(CutCapacity(options, p),
                                  VectorCapacityPolicy::kFreshExact))
      return Status::InvalidArgument("Histogram property storage overflow");
  *out = bound;
  return Status::Ok();
}
Status PrepareHistogramSamples(std::span<const HistogramTrainingImage> images,
                               PreparedHistogramSamples *out,
                               HistogramSamplingOptions options) try {
  if (!out)
    return Status::InvalidArgument("Null prepared histogram output");
  Plan plan;
  if (auto status = Preflight(images, options, &plan); !status.ok())
    return status;
  PreparedHistogramSamples result;
  result.range_ = {{{0, static_cast<uint32_t>(plan.maximum_channels)},
                    {options.first_group, static_cast<uint32_t>(plan.groups)}}};
  ManagedVector<HistogramSampleBuilder> scratch(1);
  auto &builder = scratch[0];
  builder.output = &result;
  builder.options = options;
  builder.plan = plan;
  builder.Allocate();
  if (auto status = builder.CollectQuantization(images); !status.ok())
    return status;
  builder.Quantize();
  if (auto status = builder.Gather(images); !status.ok())
    return status;
  if (result.samples_.total_samples != plan.stats.selected_samples)
    return Status::Internal("Histogram sampling disagrees with preflight");
  result.stats_ = builder.plan.stats;
  const float fraction =
      result.samples_.total_samples * 1.0f / result.stats_.population;
  const float required_cost = fraction * 0.9 + 0.1;
  result.threshold_ = options.split_threshold * required_cost;
  result.fast_decode_ = options.fast_decode_multiplier;
  result.maximum_samples_ = options.maximum_samples;
  for (auto &range : result.range_)
    if (range[1] == 0)
      range[1] = UINT32_MAX;
  *out = std::move(result);
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Histogram sample preparation allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument(
      "Histogram sample preparation storage overflow");
}
Status LearnPreparedHistogramTree(PreparedHistogramSamples &samples,
                                  DecisionTree *out, size_t maximum_leaves,
                                  size_t maximum_depth, size_t cpu_participants,
                                  HistogramTreeLearningStats *execution_stats) {
  auto options = samples.tree_options();
  options.maximum_leaves = maximum_leaves;
  options.maximum_depth = maximum_depth;
  options.cpu_participants = cpu_participants;
  options.execution_stats = execution_stats;
  return LearnHistogramTree(samples.samples_, out, options);
}
} // namespace gjxl::modular_internal
