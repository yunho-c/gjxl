// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Adapted from pinned libjxl e8ff0976 enc_ma.cc and fast_math-inl.h.
// Native storage, validation and SIMD operations; no libjxl/Highway dependency.
#include "codec/modular/histogram_tree_learning.h"
#include "codec/modular/histogram_tree_parallel.h"
#include "codec/modular/histogram_tree_profile.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace gjxl::modular_internal {
namespace {
using histogram_profile_internal::Phase;
using resource_budget_internal::ManagedVector;
constexpr size_t kSymbols = 256;
// The last (unevaluated) property bin is cleared after all predictors, so its
// histogram may temporarily hold every predictor's weighted population.
static_assert(kHistogramMaximumSamples <= INT32_MAX / kHistogramPredictors);
static_assert(kHistogramMaximumSamples <= UINT32_MAX / 63);
size_t Padded(size_t value) { return (value + 3) & ~size_t{3}; }

// Four-lane order matches the pinned AArch64 reference. Other architectures
// use the same explicit FMA polynomial and reduction order, without promising
// identical decisions to a reference compiled for a different SIMD width.
#if (defined(__aarch64__) || defined(_M_ARM64)) &&                             \
    !defined(GJXL_HISTOGRAM_SCALAR)
float32x4_t FastLog2(float32x4_t x) {
  const auto bits = vreinterpretq_s32_f32(x);
  const auto exponent =
      vshrq_n_s32(vsubq_s32(bits, vdupq_n_s32(0x3f2aaaab)), 23);
  const auto mantissa =
      vreinterpretq_f32_s32(vsubq_s32(bits, vshlq_n_s32(exponent, 23)));
  const auto z = vsubq_f32(mantissa, vdupq_n_f32(1.0f));
  auto p = vfmaq_f32(vdupq_n_f32(1.4287160470083755E+00f),
                     vdupq_n_f32(7.4245873327820566E-01f), z);
  auto q = vfmaq_f32(vdupq_n_f32(1.0096718572241148E+00f),
                     vdupq_n_f32(1.7409343003366853E-01f), z);
  p = vfmaq_f32(vdupq_n_f32(-1.8503833400518310E-06f), p, z);
  q = vfmaq_f32(vdupq_n_f32(9.9032814277590719E-01f), q, z);
  return vaddq_f32(vdivq_f32(p, q), vcvtq_f32_s32(exponent));
}
float EstimateBits(const int32_t *counts, size_t symbols) {
  auto total = vdupq_n_s32(0);
  for (size_t i = 0; i < symbols; i += 4)
    total = vaddq_s32(total, vld1q_s32(counts + i));
  total = vdupq_n_s32(vaddvq_s32(total));
  const auto inv_total = vdupq_n_f32(1.0f / vgetq_lane_s32(total, 0));
  const auto minimum = vdupq_n_f32(1.0f / 4096);
  auto bits = vdupq_n_f32(0);
  for (size_t i = 0; i < symbols; i += 4) {
    const auto c = vld1q_s32(counts + i);
    const auto f = vcvtq_f32_s32(c);
    const auto probability = vmaxq_f32(vmulq_f32(f, inv_total), minimum);
    const auto log =
        vbslq_f32(vceqq_s32(c, total), vdupq_n_f32(0), FastLog2(probability));
    bits = vsubq_f32(bits, vmulq_f32(f, log));
  }
  return vaddvq_f32(bits);
}
#else
float FastLog2(float x) {
  const uint32_t bits = std::bit_cast<uint32_t>(x);
  const int32_t exponent = std::bit_cast<int32_t>(bits - 0x3f2aaaabu) >> 23;
  const float mantissa =
      std::bit_cast<float>(bits - (static_cast<uint32_t>(exponent) << 23));
  const float z = mantissa - 1.0f;
  float p = std::fma(7.4245873327820566E-01f, z, 1.4287160470083755E+00f);
  float q = std::fma(1.7409343003366853E-01f, z, 1.0096718572241148E+00f);
  p = std::fma(p, z, -1.8503833400518310E-06f);
  q = std::fma(q, z, 9.9032814277590719E-01f);
  return p / q + static_cast<float>(exponent);
}
float EstimateBits(const int32_t *counts, size_t symbols) {
  // Match the reference's separate multiply and subtract. Implicit contraction
  // changes rounding in close split/predictor decisions; explicit std::fma in
  // FastLog2 remains intentional. Scope this to the scalar fallback.
#pragma STDC FP_CONTRACT OFF
  int32_t total = 0;
  for (size_t i = 0; i < symbols; ++i)
    total += counts[i];
  const float inv_total = 1.0f / total;
  std::array<float, 4> bits{};
  for (size_t i = 0; i < symbols; i += 4) {
    for (size_t lane = 0; lane < 4; ++lane) {
      const int32_t c = counts[i + lane];
      const float log =
          c == total ? 0.0f : FastLog2(std::max(c * inv_total, 1.0f / 4096));
      bits[lane] -= static_cast<float>(c) * log;
    }
  }
  return (bits[0] + bits[1]) + (bits[2] + bits[3]);
}
#endif

struct SamplesView {
  HistogramTrainingSamples &data;
  size_t NumDistinctSamples() const { return data.weights.size(); }
  size_t NumPredictors() const { return data.predictor_count; }
  size_t NumProperties() const { return data.property_count; }
  size_t NumStaticProps() const { return data.static_property_count; }
  size_t Count(size_t row) const { return data.weights[row]; }
  const auto &RTokens(size_t pred) const { return data.residuals[pred]; }
  size_t Token(size_t pred, size_t row) const {
    return data.residuals[pred][row].symbol;
  }
  size_t NumPropertyValues(size_t prop) const {
    return data.thresholds[prop].size() + 1;
  }
  Predictor PredictorFromIndex(size_t pred) const {
    return data.predictors[pred];
  }
  size_t PropertyFromIndex(size_t prop) const { return data.properties[prop]; }
  size_t PredictorIndex(Predictor pred) const {
    return std::find(data.predictors.begin(),
                     data.predictors.begin() + data.predictor_count, pred) -
           data.predictors.begin();
  }
  template <bool S> size_t Property(size_t prop, size_t row) const {
    if constexpr (S)
      return data.static_values[prop][row];
    else
      return data.values[prop][row];
  }
  int32_t UnquantizeProperty(size_t prop, size_t bin) const {
    return data.thresholds[prop][bin];
  }
  uint32_t QuantizeStaticProperty(size_t prop, uint32_t value) const {
    // This is the reference's clipped lookup mapping, derived from its cuts.
    const int64_t clipped = std::clamp<int64_t>(value, -511, 511);
    const auto &cuts = data.thresholds[prop];
    return std::lower_bound(cuts.begin(), cuts.end(), clipped) - cuts.begin();
  }
  void Swap(size_t a, size_t b) { data.Swap(a, b); }
};

void MakeSplitNode(size_t pos, int property, int splitval, Predictor low,
                   Predictor high, DecisionTree *tree) {
  auto &parent = tree->nodes[pos];
  parent.left = static_cast<TreeIndex>(tree->size);
  parent.right = static_cast<TreeIndex>(tree->size + 1);
  parent.split = splitval;
  parent.property = property;
  // Wire-left is strictly greater. Predictor names refer to property values.
  tree->nodes[tree->size] = DecisionNode{};
  tree->nodes[tree->size++].predictor = high;
  tree->nodes[tree->size] = DecisionNode{};
  tree->nodes[tree->size++].predictor = low;
}

enum class IntersectionType { kNone, kPartial, kInside };
IntersectionType BoxIntersects(HistogramStaticRange needle,
                               HistogramStaticRange haystack, uint32_t &axis,
                               uint32_t &value) {
  bool partial = false;
  for (size_t i = 0; i < kHistogramStaticProperties; ++i) {
    if (haystack[i][0] >= needle[i][1] || haystack[i][1] <= needle[i][0])
      return IntersectionType::kNone;
    if (haystack[i][0] <= needle[i][0] && haystack[i][1] >= needle[i][1])
      continue;
    partial = true;
    axis = static_cast<uint32_t>(i);
    value = haystack[i][0] > needle[i][0] && haystack[i][0] < needle[i][1]
                ? haystack[i][0] - 1
                : haystack[i][1] - 1;
  }
  return partial ? IntersectionType::kPartial : IntersectionType::kInside;
}
template <bool S, bool Profiled>
void SplitTreeSamples(SamplesView &samples, size_t begin, size_t pos,
                      size_t end, size_t prop, uint32_t value,
                      HistogramNodeBucket *profile) {
  size_t low = begin, high = pos;
  do {
    while (low < pos && samples.Property<S>(prop, low) <= value)
      ++low;
    while (high < end && samples.Property<S>(prop, high) > value)
      ++high;
    if (low < pos && high < end) {
      samples.Swap(low, high);
      if constexpr (Profiled)
        ++profile->work.row_swaps;
    }
    ++low;
    ++high;
  } while (low < pos && high < end);
}
template <bool S, bool CollectPopulation = false>
void CollectExtraBitsIncrease(
    const SamplesView &samples,
    const ManagedVector<HistogramResidualToken> &tokens,
    ManagedVector<int32_t> &increase, ManagedVector<size_t> &extra_increase,
    size_t begin, size_t end, size_t prop, size_t symbols,
    int32_t *used = nullptr) {
  for (size_t i = begin; i < end; ++i) {
    const auto token = tokens[i];
    const size_t count = samples.Count(i);
    const size_t bin = samples.Property<S>(prop, i);
    if constexpr (CollectPopulation)
      ++used[bin]; // Partition positions count distinct rows, not their
                   // weights.
    increase[bin * symbols + token.symbol] += count;
    extra_increase[bin] += token.extra_bits * count;
  }
}

struct NodeInfo {
  size_t pos, begin, end;
  HistogramStaticRange static_prop_range;
  size_t depth;
  // A constant property stays constant in every subset of this node's rows.
  uint32_t constant_properties = 0;
};
static_assert(kPropertyCount <= 32);
struct CostInfo {
  float cost = std::numeric_limits<float>::max();
  float extra_cost = 0;
  Predictor pred = Predictor::kZero;
  float Cost() const { return cost + extra_cost; }
};
struct SplitInfo {
  size_t prop = 0;
  uint32_t val = 0;
  size_t pos = 0;
  float lcost = std::numeric_limits<float>::max();
  float rcost = std::numeric_limits<float>::max();
  Predictor lpred = Predictor::kZero;
  Predictor rpred = Predictor::kZero;
  float Cost() const { return lcost + rcost; }
};
struct PropertyProfile {
  HistogramFitSeconds seconds{};
  HistogramNodeBucket bucket;
  std::array<uint64_t, kHistogramBinBuckets> properties_by_active_bins{},
      properties_by_bin_span{};
};
struct PropertyResult {
  SplitInfo best_static_constant, best_static, best_nonstatic, best_nowp;
  uint32_t constant_properties = 0;
  PropertyProfile profile;
};
struct PropertyWorkspace {
  ManagedVector<int32_t> above, below, used, increase;
  ManagedVector<size_t> extra_increase;
  ManagedVector<CostInfo> costs_l, costs_r;
  void Initialize(size_t bins) {
    above.resize(kSymbols);
    below.resize(kSymbols);
    used.resize(bins);
    increase.resize(bins * kSymbols);
    extra_increase.resize(bins);
    costs_l.resize(bins);
    costs_r.resize(bins);
  }
};
struct Workspace {
  ManagedVector<NodeInfo> nodes;
  ManagedVector<int32_t> counts;
  ManagedVector<uint32_t> total_extra;
  PropertyWorkspace property;
  ManagedVector<PropertyWorkspace> workers;
  std::array<PropertyResult, kPropertyCount> results;
};
constexpr size_t kMinimumParallelHistogramRows = 8192; // rows times predictors

Status ValidateShape(const HistogramTrainingSamples &s,
                     HistogramTreeLearningOptions o) {
  if (!std::isfinite(o.split_threshold) || o.split_threshold < 0 ||
      !std::isfinite(o.fast_decode_multiplier) ||
      o.fast_decode_multiplier < 1 || !o.maximum_samples ||
      o.maximum_samples > kHistogramMaximumSamples || !o.maximum_leaves ||
      o.maximum_leaves > kMaximumTreeLeaves || !o.maximum_depth ||
      o.maximum_depth > kMaximumTreeDepth || o.cpu_participants > 256 ||
      o.multipliers.size() > kHistogramMaximumStaticBins ||
      !s.predictor_count || s.predictor_count > kHistogramPredictors ||
      !s.property_count || s.property_count > kPropertyCount ||
      s.static_property_count > kHistogramStaticProperties ||
      s.static_property_count > s.property_count ||
      s.total_samples > o.maximum_samples || s.weights.size() > s.total_samples)
    return Status::InvalidArgument(
        "Invalid histogram Modular learning shape or options");
  std::array<bool, kHistogramPredictors> seen_predictors{};
  for (size_t p = 0; p < s.predictor_count; ++p) {
    const size_t pred = static_cast<size_t>(s.predictors[p]);
    if (pred >= kHistogramPredictors || seen_predictors[pred] ||
        s.residuals[p].size() != s.weights.size())
      return Status::InvalidArgument(
          "Invalid histogram Modular predictor column");
    seen_predictors[pred] = true;
  }
  std::array<bool, kPropertyCount> seen_properties{};
  for (size_t p = 0; p < s.property_count; ++p) {
    const size_t prop = s.properties[p];
    const bool stat = p < s.static_property_count;
    if (prop >= kPropertyCount || seen_properties[prop] ||
        (stat ? prop != p : prop < kHistogramStaticProperties) ||
        s.thresholds[p].size() >= (stat ? kHistogramMaximumStaticBins : 256) ||
        (stat ? s.static_values[p].size()
              : s.values[p - s.static_property_count].size()) !=
            s.weights.size())
      return Status::InvalidArgument(
          "Invalid histogram Modular property column");
    seen_properties[prop] = true;
  }
  for (auto range : o.static_range)
    if (range[0] >= range[1])
      return Status::InvalidArgument("Invalid histogram static range");
  for (const auto &m : o.multipliers) {
    if (!m.multiplier || m.multiplier > INT32_MAX)
      return Status::InvalidArgument("Invalid histogram multiplier");
    for (auto range : m.range)
      if (range[0] >= range[1])
        return Status::InvalidArgument("Invalid histogram multiplier range");
  }
  return Status::Ok();
}
size_t MaximumBins(const HistogramTrainingSamples &s) {
  size_t bins = 1;
  for (size_t p = 0; p < s.property_count; ++p)
    bins = std::max(bins, s.thresholds[p].size() + 1);
  return bins;
}

template <bool Profiled>
void EvaluateProperty(const SamplesView &tree_samples, PropertyWorkspace &w,
                      const ManagedVector<int32_t> &counts,
                      const ManagedVector<uint32_t> &tot_extra_bits,
                      size_t begin, size_t end, size_t prop, size_t max_symbols,
                      Predictor parent_predictor, float change_pred_penalty,
                      uint32_t inherited_constants,
                      PropertyResult &result) noexcept {
  auto &best_split_static_constant = result.best_static_constant;
  auto &best_split_static = result.best_static;
  auto &best_split_nonstatic = result.best_nonstatic;
  auto &best_split_nowp = result.best_nowp;
  best_split_static_constant = best_split_static = best_split_nonstatic =
      best_split_nowp = {};
  auto &constant_properties = result.constant_properties;
  constant_properties = inherited_constants;
  PropertyProfile *profile = nullptr;
  HistogramNodeBucket *bucket = nullptr;
  if constexpr (Profiled) {
    result.profile = {};
    profile = &result.profile;
    bucket = &profile->bucket;
  }
  const size_t num_predictors = tree_samples.NumPredictors();
  auto &prop_value_used_count = w.used;
  auto &count_increase = w.increase;
  auto &extra_bits_increase = w.extra_increase;
  auto &costs_l = w.costs_l;
  auto &costs_r = w.costs_r;
  auto &counts_above = w.above;
  auto &counts_below = w.below;
  const uint32_t property_bit = uint32_t{1} << prop;
  if constexpr (Profiled)
    ++bucket->work.properties;
  if ((constant_properties & property_bit) ||
      tree_samples.NumPropertyValues(prop) == 1) {
    constant_properties |= property_bit;
    if constexpr (Profiled) {
      auto &work = bucket->work;
      ++work.constant_properties;
      ++work.skipped_constant_properties;
      ++work.active_bins;
      ++work.bin_span;
      ++profile->properties_by_active_bins[0];
      ++profile->properties_by_bin_span[0];
    }
    return;
  }
  Phase<Profiled> property_timer(
      profile, HistogramFitStage::kPropertyPopulation, bucket);
  const size_t prop_size = tree_samples.NumPropertyValues(prop);
  std::fill_n(prop_value_used_count.begin(), prop_size, int32_t{0});
  property_timer.Stop();
  // Collect occupancy in the first predictor's histogram pass. The
  // occupied endpoints are found on the bin axis, not on every row.
  Phase<Profiled> first_joint_timer(
      profile, HistogramFitStage::kJointHistograms, bucket);
  const auto &first_tokens = tree_samples.RTokens(0);
  if (prop < tree_samples.NumStaticProps()) {
    CollectExtraBitsIncrease<true, true>(
        tree_samples, first_tokens, count_increase, extra_bits_increase, begin,
        end, prop, max_symbols, prop_value_used_count.data());
  } else {
    CollectExtraBitsIncrease<false, true>(
        tree_samples, first_tokens, count_increase, extra_bits_increase, begin,
        end, prop - tree_samples.NumStaticProps(), max_symbols,
        prop_value_used_count.data());
  }
  first_joint_timer.Stop();
  Phase<Profiled> cost_setup_timer(
      profile, HistogramFitStage::kPropertyPopulation, bucket);
  size_t first_used = 0, last_used = prop_size - 1;
  while (!prop_value_used_count[first_used])
    ++first_used;
  while (!prop_value_used_count[last_used])
    --last_used;
  std::fill_n(costs_l.begin(), last_used - first_used, CostInfo{});
  std::fill_n(costs_r.begin(), last_used - first_used, CostInfo{});
  cost_setup_timer.Stop();
  if constexpr (Profiled) {
    bucket->work.constant_properties += first_used == last_used;
    bucket->work.property_row_visits += end - begin;
    bucket->work.fused_property_row_visits += end - begin;
    bucket->work.joint_histogram_row_visits += end - begin;
    bucket->work.bin_span += last_used - first_used + 1;
    ++profile->properties_by_bin_span[HistogramProfileBucket(
        last_used - first_used + 1, kHistogramBinBuckets)];
  }
  if (first_used == last_used) {
    constant_properties |= property_bit;
    // This property has no candidate threshold. Only the first
    // predictor was collected; discard its last-bin increments.
    extra_bits_increase[last_used] = 0;
    std::fill_n(count_increase.begin() + last_used * max_symbols, max_symbols,
                int32_t{0});
    if constexpr (Profiled) {
      ++bucket->work.active_bins;
      ++profile->properties_by_active_bins[0];
    }
    return;
  }
  // For all predictors, compute the right and left costs of each split.
  for (size_t pred = 0; pred < num_predictors; pred++) {
    if (pred != 0) {
      Phase<Profiled> joint_timer(profile, HistogramFitStage::kJointHistograms,
                                  bucket);
      // Compute cost and histogram increments for each property value.
      const auto &rtokens = tree_samples.RTokens(pred);
      if (prop < tree_samples.NumStaticProps()) {
        CollectExtraBitsIncrease<true>(tree_samples, rtokens, count_increase,
                                       extra_bits_increase, begin, end, prop,
                                       max_symbols);
      } else {
        CollectExtraBitsIncrease<false>(
            tree_samples, rtokens, count_increase, extra_bits_increase, begin,
            end, prop - tree_samples.NumStaticProps(), max_symbols);
      }
      joint_timer.Stop();
      if constexpr (Profiled)
        bucket->work.joint_histogram_row_visits += end - begin;
    }
    Phase<Profiled> candidate_timer(
        profile, HistogramFitStage::kCandidateEvaluation, bucket);
    memcpy(counts_above.data(), counts.data() + pred * max_symbols,
           max_symbols * sizeof counts_above[0]);
    memset(counts_below.data(), 0, max_symbols * sizeof counts_below[0]);
    size_t extra_bits_below = 0;
    // Exclude last used: this ensures neither counts_above nor
    // counts_below is empty.
    for (size_t i = first_used; i < last_used; i++) {
      if (!prop_value_used_count[i])
        continue;
      extra_bits_below += extra_bits_increase[i];
      // The increase for this property value has been used, and will not
      // be used again: clear it. Also below.
      extra_bits_increase[i] = 0;
      for (size_t sym = 0; sym < max_symbols; sym++) {
        counts_above[sym] -= count_increase[i * max_symbols + sym];
        counts_below[sym] += count_increase[i * max_symbols + sym];
        count_increase[i * max_symbols + sym] = 0;
      }
      float rcost = EstimateBits(counts_above.data(), max_symbols) +
                    tot_extra_bits[pred] - extra_bits_below;
      float lcost =
          EstimateBits(counts_below.data(), max_symbols) + extra_bits_below;

      float penalty = 0;
      // Never discourage moving away from the Weighted predictor.
      if (tree_samples.PredictorFromIndex(pred) != parent_predictor &&
          parent_predictor != Predictor::kWeighted) {
        penalty = change_pred_penalty;
      }
      // If everything else is equal, disfavour Weighted (slower) and
      // favour Zero (faster if it's the only predictor used in a
      // group+channel combination)
      if (tree_samples.PredictorFromIndex(pred) == Predictor::kWeighted) {
        penalty += 1e-8;
      }
      if (tree_samples.PredictorFromIndex(pred) == Predictor::kZero) {
        penalty -= 1e-8;
      }
      if (rcost + penalty < costs_r[i - first_used].Cost()) {
        costs_r[i - first_used].cost = rcost;
        costs_r[i - first_used].extra_cost = penalty;
        costs_r[i - first_used].pred = tree_samples.PredictorFromIndex(pred);
      }
      if (lcost + penalty < costs_l[i - first_used].Cost()) {
        costs_l[i - first_used].cost = lcost;
        costs_l[i - first_used].extra_cost = penalty;
        costs_l[i - first_used].pred = tree_samples.PredictorFromIndex(pred);
      }
    }
  }
  // Iterate through the possible splits and find the one with minimum sum
  // of costs of the two sides.
  Phase<Profiled> selection_timer(profile, HistogramFitStage::kSplitSelection,
                                  bucket);
  size_t active_thresholds = 0;
  size_t split = begin;
  for (size_t i = first_used; i < last_used; i++) {
    if (!prop_value_used_count[i])
      continue;
    if constexpr (Profiled)
      ++active_thresholds;
    split += prop_value_used_count[i];
    float rcost = costs_r[i - first_used].cost;
    float lcost = costs_l[i - first_used].cost;

    bool uses_wp = tree_samples.PropertyFromIndex(prop) == 15 ||
                   costs_l[i - first_used].pred == Predictor::kWeighted ||
                   costs_r[i - first_used].pred == Predictor::kWeighted;
    bool zero_entropy_side = rcost == 0 || lcost == 0;

    SplitInfo &best_ref =
        tree_samples.PropertyFromIndex(prop) < kHistogramStaticProperties
            ? (zero_entropy_side ? best_split_static_constant
                                 : best_split_static)
            : (uses_wp ? best_split_nonstatic : best_split_nowp);
    if (lcost + rcost < best_ref.Cost()) {
      best_ref.prop = prop;
      best_ref.val = i;
      best_ref.pos = split;
      best_ref.lcost = lcost;
      best_ref.lpred = costs_l[i - first_used].pred;
      best_ref.rcost = rcost;
      best_ref.rpred = costs_r[i - first_used].pred;
    }
  }
  selection_timer.Stop();
  if constexpr (Profiled) {
    auto &work = bucket->work;
    work.active_bins += active_thresholds + 1;
    work.candidate_thresholds += active_thresholds;
    work.entropy_evaluations += 2 * active_thresholds * num_predictors;
    work.symbol_terms += 2 * active_thresholds * num_predictors * max_symbols;
    ++profile->properties_by_active_bins[HistogramProfileBucket(
        active_thresholds + 1, kHistogramBinBuckets)];
  }
  // Clear extra_bits_increase and cost_increase for last_used.
  extra_bits_increase[last_used] = 0;
  for (size_t sym = 0; sym < max_symbols; sym++) {
    count_increase[last_used * max_symbols + sym] = 0;
  }
}

void MergePropertyProfile(HistogramFittingProfile &to,
                          HistogramNodeBucket &bucket,
                          const PropertyProfile &from) {
  MergeHistogramNodeWork(bucket.work, from.bucket.work);
  for (size_t i = 0; i < to.seconds.size(); ++i) {
    to.seconds[i] += from.seconds[i];
    bucket.seconds[i] += from.seconds[i];
  }
  for (size_t i = 0; i < kHistogramBinBuckets; ++i) {
    to.properties_by_active_bins[i] += from.properties_by_active_bins[i];
    to.properties_by_bin_span[i] += from.properties_by_bin_span[i];
  }
}

template <bool Profiled>
Status FindBestSplit(SamplesView &tree_samples, Workspace &w,
                     histogram_parallel_internal::PropertyTeam &team,
                     HistogramTreeLearningStats &execution,
                     HistogramTreeLearningOptions options, float threshold,
                     std::span<const HistogramMultiplierRange> mul_info,
                     HistogramStaticRange initial_static_prop_range,
                     float fast_decode_multiplier, DecisionTree *tree,
                     HistogramFittingProfile *profile,
                     HistogramFitRecord *record) {
  auto &nodes = w.nodes;
  nodes.push_back(NodeInfo{0, 0, tree_samples.NumDistinctSamples(),
                           initial_static_prop_range, 1});

  size_t num_predictors = tree_samples.NumPredictors();
  size_t num_properties = tree_samples.NumProperties();

  while (!nodes.empty()) {
    size_t pos = nodes.back().pos;
    size_t begin = nodes.back().begin;
    size_t end = nodes.back().end;

    HistogramStaticRange static_prop_range = nodes.back().static_prop_range;
    const size_t depth = nodes.back().depth;
    uint32_t constant_properties = nodes.back().constant_properties;
    nodes.pop_back();
    if (begin == end)
      continue;
    histogram_profile_internal::Node<Profiled> node_profile(
        profile, end - begin, depth, record);

    SplitInfo best_split_static_constant;
    SplitInfo best_split_static;
    SplitInfo best_split_nonstatic;
    SplitInfo best_split_nowp;

    // Compute the maximum token in the range.
    Phase<Profiled> symbol_timer(profile, HistogramFitStage::kSymbolRange,
                                 node_profile.bucket);
    size_t max_symbols = 0;
    for (size_t pred = 0; pred < num_predictors; pred++) {
      for (size_t i = begin; i < end; i++) {
        uint32_t tok = tree_samples.Token(pred, i);
        max_symbols = max_symbols > tok + 1 ? max_symbols : tok + 1;
      }
    }
    max_symbols = Padded(max_symbols);
    symbol_timer.Stop();
    if constexpr (Profiled) {
      ++profile->nodes_by_symbol_span[max_symbols];
      node_profile.bucket->work.symbol_row_visits +=
          (end - begin) * num_predictors;
      node_profile.bucket->work.parent_histogram_row_visits +=
          (end - begin) * num_predictors;
    }
    Phase<Profiled> parent_timer(profile, HistogramFitStage::kParentHistograms,
                                 node_profile.bucket);
    auto &counts = w.counts;
    auto &tot_extra_bits = w.total_extra;
    std::fill_n(counts.begin(), max_symbols * num_predictors, int32_t{0});
    for (size_t pred = 0; pred < num_predictors; pred++) {
      size_t extra_bits = 0;
      const auto &rtokens = tree_samples.RTokens(pred);
      for (size_t i = begin; i < end; i++) {
        const HistogramResidualToken &rt = rtokens[i];
        size_t count = tree_samples.Count(i);
        size_t eb = rt.extra_bits * count;
        counts[pred * max_symbols + rt.symbol] += count;
        extra_bits += eb;
      }
      tot_extra_bits[pred] = extra_bits;
    }

    float base_bits;
    {
      size_t pred = tree_samples.PredictorIndex(tree->nodes[pos].predictor);
      base_bits =
          EstimateBits(counts.data() + pred * max_symbols, max_symbols) +
          tot_extra_bits[pred];
    }
    parent_timer.Stop();

    SplitInfo *best = &best_split_nonstatic;

    SplitInfo forced_split;
    // The multiplier ranges cut halfway through the current ranges of static
    // properties. We do this even if the current node is not a leaf, to
    // minimize the number of nodes in the resulting tree.
    for (const auto &mmi : mul_info) {
      uint32_t axis;
      uint32_t val;
      IntersectionType t =
          BoxIntersects(static_prop_range, mmi.range, axis, val);
      if (t == IntersectionType::kNone)
        continue;
      if (t == IntersectionType::kInside) {
        tree->nodes[pos].multiplier = mmi.multiplier;
        break;
      }
      if (t == IntersectionType::kPartial) {
        if (axis >= tree_samples.NumStaticProps())
          return Status::InvalidArgument(
              "Multiplier split requires a static property");
        forced_split.val = tree_samples.QuantizeStaticProperty(axis, val);
        if (forced_split.val + 1 >= tree_samples.NumPropertyValues(axis) ||
            tree_samples.UnquantizeProperty(axis, forced_split.val) !=
                static_cast<int64_t>(val))
          return Status::InvalidArgument(
              "Multiplier boundary is absent from quantization cuts");
        forced_split.prop = axis;
        forced_split.lcost = forced_split.rcost = base_bits / 2 - threshold;
        forced_split.lpred = forced_split.rpred = tree->nodes[pos].predictor;
        best = &forced_split;
        if constexpr (Profiled)
          ++node_profile.bucket->work.forced_splits;
        best->pos = begin;

        if (best->prop < tree_samples.NumStaticProps()) {
          for (size_t x = begin; x < end; x++) {
            if (tree_samples.Property<true>(best->prop, x) <= best->val) {
              best->pos++;
            }
          }
        } else {
          size_t prop = best->prop - tree_samples.NumStaticProps();
          for (size_t x = begin; x < end; x++) {
            if (tree_samples.Property<false>(prop, x) <= best->val) {
              best->pos++;
            }
          }
        }
        break;
      }
    }

    if (best != &forced_split) {
      // Each property preserves predictor/bin arithmetic and tie order. Only
      // independent properties run concurrently; reduce their category winners
      // in the original property order before applying predictor preferences.
      const float change_pred_penalty = 800.0f / (100.0f + threshold);
      if (base_bits > threshold) {
        size_t varying = 0;
        for (size_t prop = 0; prop < num_properties; ++prop)
          varying += !(constant_properties & (uint32_t{1} << prop)) &&
                     tree_samples.NumPropertyValues(prop) > 1;
        const bool parallel =
            team.participants() > 1 && varying > 1 &&
            (end - begin) * num_predictors >= kMinimumParallelHistogramRows;
        const auto evaluate = [&](size_t prop, size_t worker) noexcept {
          auto &scratch = worker == 0 ? w.property : w.workers[worker - 1];
          EvaluateProperty<Profiled>(
              tree_samples, scratch, counts, tot_extra_bits, begin, end, prop,
              max_symbols, tree->nodes[pos].predictor, change_pred_penalty,
              constant_properties, w.results[prop]);
        };
        if (parallel) {
          team.Run(num_properties, evaluate);
          ++execution.parallel_nodes;
          if constexpr (Profiled)
            ++profile->parallel_property_nodes;
        } else {
          for (size_t prop = 0; prop < num_properties; ++prop)
            evaluate(prop, 0);
        }
        for (size_t prop = 0; prop < num_properties; ++prop) {
          const auto &r = w.results[prop];
          constant_properties |= r.constant_properties;
          for (auto pair :
               {std::pair{&best_split_static_constant, &r.best_static_constant},
                std::pair{&best_split_static, &r.best_static},
                std::pair{&best_split_nonstatic, &r.best_nonstatic},
                std::pair{&best_split_nowp, &r.best_nowp}})
            if (pair.second->Cost() < pair.first->Cost())
              *pair.first = *pair.second;
          if constexpr (Profiled)
            MergePropertyProfile(*profile, *node_profile.bucket, r.profile);
        }
      }

      // Try to avoid introducing WP.
      if (best_split_nowp.Cost() + threshold < base_bits &&
          best_split_nowp.Cost() <= fast_decode_multiplier * best->Cost()) {
        best = &best_split_nowp;
      }
      // Split along static props if possible and not significantly more
      // expensive.
      if (best_split_static.Cost() + threshold < base_bits &&
          best_split_static.Cost() <= fast_decode_multiplier * best->Cost()) {
        best = &best_split_static;
      }
      // Split along static props to create constant nodes if possible.
      if (best_split_static_constant.Cost() + threshold < base_bits) {
        best = &best_split_static_constant;
      }
    }

    if (best->Cost() + threshold < base_bits) {
      uint32_t p = tree_samples.PropertyFromIndex(best->prop);
      int32_t dequant = tree_samples.UnquantizeProperty(best->prop, best->val);
      // Split node and try to split children.
      if (tree->size + 2 > 2 * options.maximum_leaves - 1 ||
          depth >= options.maximum_depth)
        return Status::Unsupported(
            "Histogram Modular fit exceeds tree capacity or depth");
      MakeSplitNode(pos, p, dequant, best->lpred, best->rpred, tree);
      if constexpr (Profiled) {
        ++node_profile.bucket->work.splits;
        node_profile.bucket->work.partition_rows += end - begin;
      }
      // "Sort" according to winning property
      Phase<Profiled> partition_timer(profile, HistogramFitStage::kPartition,
                                      node_profile.bucket);
      if (best->prop < tree_samples.NumStaticProps()) {
        SplitTreeSamples<true, Profiled>(tree_samples, begin, best->pos, end,
                                         best->prop, best->val,
                                         node_profile.bucket);
      } else {
        SplitTreeSamples<false, Profiled>(tree_samples, begin, best->pos, end,
                                          best->prop -
                                              tree_samples.NumStaticProps(),
                                          best->val, node_profile.bucket);
      }
      partition_timer.Stop();
      auto new_sp_range = static_prop_range;
      if (p < kHistogramStaticProperties) {
        new_sp_range[p][1] = dequant + 1;
      }
      nodes.push_back(NodeInfo{tree->nodes[pos].right, begin, best->pos,
                               new_sp_range, depth + 1, constant_properties});
      new_sp_range = static_prop_range;
      if (p < kHistogramStaticProperties) {
        new_sp_range[p][0] = dequant + 1;
      }
      nodes.push_back(NodeInfo{tree->nodes[pos].left, best->pos, end,
                               new_sp_range, depth + 1, constant_properties});
    }
  }
  return Status::Ok();
}

} // namespace

void HistogramTrainingSamples::Swap(size_t a, size_t b) {
  if (a == b)
    return;
  for (size_t p = 0; p < predictor_count; ++p)
    std::swap(residuals[p][a], residuals[p][b]);
  for (size_t p = 0; p < static_property_count; ++p)
    std::swap(static_values[p][a], static_values[p][b]);
  for (size_t p = 0; p < property_count - static_property_count; ++p)
    std::swap(values[p][a], values[p][b]);
  std::swap(weights[a], weights[b]);
}

Status ValidateHistogramTrainingSamples(const HistogramTrainingSamples &s,
                                        HistogramTreeLearningOptions o) {
  if (auto status = ValidateShape(s, o); !status.ok())
    return status;
  // Reductions keep complete admission checks while allowing vectorization.
  // Shape validation bounds row count by 2^24; uint16 weights therefore fit
  // in this uint64 sum even for malformed populations.
  uint64_t total = 0;
  uint16_t minimum_weight = UINT16_MAX;
  for (auto count : s.weights) {
    total += count;
    minimum_weight = std::min(minimum_weight, count);
  }
  if (!minimum_weight || total > o.maximum_samples)
    return Status::InvalidArgument("Invalid histogram sample weights");
  if (total != s.total_samples)
    return Status::InvalidArgument("Histogram sample count mismatch");
  for (size_t pred = 0; pred < s.predictor_count; ++pred) {
    uint8_t extra_bits = 0;
    for (auto token : s.residuals[pred])
      extra_bits |= token.extra_bits;
    if (extra_bits > 63)
      return Status::InvalidArgument("Invalid histogram residual token");
  }
  for (size_t p = 0; p < s.property_count; ++p) {
    const auto &cuts = s.thresholds[p];
    if (!std::is_sorted(cuts.begin(), cuts.end()) ||
        std::adjacent_find(cuts.begin(), cuts.end()) != cuts.end() ||
        (!cuts.empty() && (cuts.back() == INT32_MAX ||
                           (p < s.static_property_count && cuts.front() < 0))))
      return Status::InvalidArgument("Invalid histogram property cuts");
    if (p < s.static_property_count) {
      uint32_t maximum_bin = 0;
      for (auto bin : s.static_values[p])
        maximum_bin = std::max(maximum_bin, bin);
      if (maximum_bin > cuts.size())
        return Status::InvalidArgument("Invalid histogram static bin");
    } else {
      uint8_t maximum_bin = 0;
      for (auto bin : s.values[p - s.static_property_count])
        maximum_bin = std::max(maximum_bin, bin);
      if (maximum_bin > cuts.size())
        return Status::InvalidArgument("Invalid histogram property bin");
    }
  }
  return Status::Ok();
}

size_t HistogramTreeLearningParticipants(const HistogramTrainingSamples &s,
                                         HistogramTreeLearningOptions options) {
  using namespace thread_budget_internal;
  if (options.cpu_participants == 1 || !HasCpuParticipation() ||
      InExplicitParallelScope() || !s.predictor_count ||
      s.predictor_count > kHistogramPredictors ||
      s.property_count > kPropertyCount ||
      s.weights.size() <
          (kMinimumParallelHistogramRows + s.predictor_count - 1) /
              s.predictor_count)
    return 1;
  size_t varying = 0;
  for (size_t prop = 0; prop < s.property_count; ++prop)
    varying += !s.thresholds[prop].empty();
  size_t desired =
      options.cpu_participants
          ? options.cpu_participants
          : std::max(size_t{1}, size_t{std::thread::hardware_concurrency()});
  if (CpuThreadCount())
    desired = std::min(desired, CpuThreadCount());
  return std::max(size_t{1},
                  std::min({desired, varying, current_cpu_execution.job.limit(),
                            current_cpu_execution.domain.limit()}));
}

Status ComputeHistogramTreeLearningStorageBound(
    const HistogramTrainingSamples &s, HistogramTreeLearningOptions options,
    resource_budget_internal::HostStorageBound *out) {
  if (!out)
    return Status::InvalidArgument("Null histogram learning storage output");
  if (auto status = ValidateShape(s, options); !status.ok())
    return status;
  using namespace resource_budget_internal;
  const size_t bins = MaximumBins(s);
  HostStorageBound bound;
  if (auto status = ComputeHistogramTreeWorkspaceStorageBound(
          s.predictor_count, bins, options.maximum_leaves,
          options.maximum_depth, &bound);
      !status.ok())
    return status;
  const size_t workers = HistogramTreeLearningParticipants(s, options) - 1;
  HostStorageBound extra;
  if (!extra.AddVector<PropertyWorkspace>(1,
                                          VectorCapacityPolicy::kFreshExact) ||
      !extra.AddVector<std::thread>(1, VectorCapacityPolicy::kFreshExact) ||
      !extra.AddVector<int32_t>(kSymbols, VectorCapacityPolicy::kFreshExact,
                                2) ||
      !extra.AddVector<int32_t>(bins, VectorCapacityPolicy::kFreshExact) ||
      !extra.AddVector<int32_t>(bins * kSymbols,
                                VectorCapacityPolicy::kFreshExact) ||
      !extra.AddVector<size_t>(bins, VectorCapacityPolicy::kFreshExact) ||
      !extra.AddVector<CostInfo>(bins, VectorCapacityPolicy::kFreshExact, 2) ||
      !bound.Add(extra, workers))
    return Status::InvalidArgument("Histogram parallel storage overflow");
  *out = bound;
  return Status::Ok();
}

Status ComputeHistogramTreeWorkspaceStorageBound(
    size_t predictor_count, size_t bins, size_t maximum_leaves,
    size_t maximum_depth, resource_budget_internal::HostStorageBound *out) {
  if (!out || !predictor_count || predictor_count > kHistogramPredictors ||
      !bins || bins > kHistogramMaximumStaticBins || !maximum_leaves ||
      maximum_leaves > kMaximumTreeLeaves || !maximum_depth ||
      maximum_depth > kMaximumTreeDepth)
    return Status::InvalidArgument("Invalid histogram workspace capacity");
  using namespace resource_budget_internal;
  HostStorageBound bound;
  const size_t nodes = 2 * maximum_leaves - 1;
  if (!bound.AddVector<Workspace>(1, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<NodeInfo>(maximum_depth,
                                 VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int32_t>(kSymbols * predictor_count,
                                VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int32_t>(kSymbols, VectorCapacityPolicy::kFreshExact,
                                2) ||
      !bound.AddVector<int32_t>(bins, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int32_t>(bins * kSymbols,
                                VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<uint32_t>(predictor_count,
                                 VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<size_t>(bins, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<CostInfo>(bins, VectorCapacityPolicy::kFreshExact, 2) ||
      (nodes > kInlineTreeNodes &&
       !bound.AddVector<DecisionNode>(nodes,
                                      VectorCapacityPolicy::kFreshExact)))
    return Status::InvalidArgument(
        "Histogram Modular learning storage overflow");
  *out = bound;
  return Status::Ok();
}

Status LearnHistogramTree(HistogramTrainingSamples &s, DecisionTree *out,
                          HistogramTreeLearningOptions options) try {
  auto *profile = active_histogram_fitting_profile;
  histogram_profile_internal::Fit fit_profile(profile);
  Phase<true> validation_timer(profile, HistogramFitStage::kValidation);
  if (!out)
    return Status::InvalidArgument("Null histogram tree output");
  if (auto status = ValidateHistogramTrainingSamples(s, options); !status.ok())
    return status;
  validation_timer.Stop();
  if (fit_profile.record) {
    auto &record = *fit_profile.record;
    record.rows = s.weights.size();
    record.population = s.total_samples;
    record.predictors = s.predictor_count;
    record.properties = s.property_count;
  }
  Phase<true> setup_timer(profile, HistogramFitStage::kSetup);
  ManagedVector<Workspace> storage(1);
  auto &w = storage[0];
  const size_t bins = MaximumBins(s);
  w.nodes.reserve(options.maximum_depth);
  w.counts.resize(kSymbols * s.predictor_count);
  w.total_extra.resize(s.predictor_count);
  w.property.Initialize(bins);
  histogram_parallel_internal::PropertyTeam team(
      HistogramTreeLearningParticipants(s, options));
  HistogramTreeLearningStats execution{team.participants(), 0};
  if (profile)
    profile->peak_property_participants =
        std::max(profile->peak_property_participants, team.participants());
  w.workers.resize(team.participants() - 1);
  for (auto &worker : w.workers)
    worker.Initialize(bins);
  DecisionTree result;
  result.nodes.resize(2 * options.maximum_leaves - 1);
  result.nodes[0].predictor = s.predictors[0];
  SamplesView view{s};
  if (auto status = team.Start(); !status.ok())
    return status;
  setup_timer.Stop();
  const auto status =
      profile ? FindBestSplit<true>(view, w, team, execution, options,
                                    options.split_threshold,
                                    options.multipliers, options.static_range,
                                    options.fast_decode_multiplier, &result,
                                    profile, fit_profile.record)
              : FindBestSplit<false>(
                    view, w, team, execution, options, options.split_threshold,
                    options.multipliers, options.static_range,
                    options.fast_decode_multiplier, &result, nullptr, nullptr);
  if (!status.ok())
    return status;
  Phase<true> output_timer(profile, HistogramFitStage::kOutput);
  TreeLayout layout;
  if (auto status = ValidateTree(result, &layout); !status.ok())
    return status;
  if (result.size <= kInlineTreeNodes)
    result.nodes.resize(result.size); // Releases only; cannot allocate.
  *out = std::move(result);
  if (options.execution_stats)
    *options.execution_stats = execution;
  fit_profile.Success(out->size);
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Histogram Modular learning allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Histogram Modular learning storage overflow");
}
} // namespace gjxl::modular_internal
