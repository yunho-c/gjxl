// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

#include "codec/adaptive_quantization_internal.h"
#include "codec/color_transform_internal.h"
#include "codec/frontend_storage_plan.h"
#include "codestream/cpu_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::frontend_storage_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::adaptive_quantization_internal;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Ok(const Status &s) {
  if (!s.ok())
    std::cerr << s.message() << '\n';
  return s.ok();
}
bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                   s.total.pending_count == 0 && s.open_reservations == 0 &&
                   s.waiting_requests == 0,
               "CPU plan test leaked backing or admission");
}
struct Image {
  Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> planes;
  explicit Image(Extent2D e) : extent(e), stride(e.width + 3) {
    for (size_t c = 0; c < 3; ++c) {
      planes[c].resize(stride * e.height, -17);
      for (size_t y = 0; y < e.height; ++y)
        for (size_t x = 0; x < e.width; ++x)
          planes[c][y * stride + x] =
              0.03f + 0.8f * ((x * 11 + y * 7 + c * 23 + x * y) % 251) / 251.0f;
    }
  }
  Image3FView view() {
    return {{{{planes[0].data(), extent, stride},
              {planes[1].data(), extent, stride},
              {planes[2].data(), extent, stride}}}};
  }
  ConstImage3FView const_view() const {
    return {{{{planes[0].data(), extent, stride},
              {planes[1].data(), extent, stride},
              {planes[2].data(), extent, stride}}}};
  }
};

bool CheckPlans() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(1, &job)) || !Ok(job.ReduceCapacity(0)))
    return false;
  ResourceContextScope scope({&job, ResourceClass::kPreparation});
  size_t cases = 0;
  for (Extent2D extent : {Extent2D{1, 1},
                          {14, 15},
                          {15, 15},
                          {65, 63},
                          {257, 9},
                          {257, 257},
                          {3839, 2159}}) {
    for (int effort = 1; effort <= 10; ++effort) {
      for (size_t flags = 0; flags < 64; ++flags) {
        CpuWorkflowStorageOptions o;
        o.encoding.backend = VarDctBackendPreference::kCpu;
        o.encoding.effort = effort;
        o.encoding.cpu_thread_count = flags & 1 ? 1 : 0;
        o.encoding.density_mode = flags & 2 ? VarDctDensityMode::kHighDensity
                                            : VarDctDensityMode::kDefault;
        o.encoding.compression_mode =
            flags & 4 ? VarDctCompressionMode::kMaximumCompression
                      : VarDctCompressionMode::kAutomatic;
        o.encoding.rate_control_mode =
            flags & 8    ? VarDctRateControlMode::kMaximumError
            : flags & 16 ? VarDctRateControlMode::kTargetBytes
                         : VarDctRateControlMode::kButteraugliTarget;
        o.collect_profile = bool(flags & 32);
        o.collect_timing = true;
        CpuWorkflowStoragePlan p;
        ArmNextManagedHostAllocationFailureForTest();
        const auto s = ComputeCpuWorkflowStoragePlan(extent, o, &p);
        const bool pending = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (!Ok(s) ||
            !Check(pending &&
                       p.score_count ==
                           (flags & 8
                                ? 6
                                : AdaptiveQuantizationIterations(o.encoding) +
                                      1) &&
                       p.working.peak_bytes >= p.output.peak_bytes &&
                       p.aq.policy.profile.peak_bytes == 0 &&
                       (p.aq.reference.peak_bytes == 0) == bool(flags & 8),
                   "CPU workflow formula/allocation-free contract failed"))
          return false;
        ++cases;
      }
    }
  }
  CpuWorkflowStorageOptions o;
  o.encoding.backend = VarDctBackendPreference::kCpu;
  o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
  for (size_t attempts = 1; attempts <= 64; ++attempts) {
    o.encoding.target_size_maximum_attempts = attempts;
    CpuWorkflowStoragePlan p;
    if (!Ok(ComputeCpuWorkflowStoragePlan({89, 57}, o, &p)) ||
        !Check(p.maximum_attempts == attempts &&
                   p.retained_best.retained_bytes ==
                       (attempts > 1 ? p.serializer.output.retained_bytes +
                                           sizeof(double) * p.score_count
                                     : 0),
               "CPU search multiplied retained output by attempt count"))
      return false;
  }
  CpuWorkflowStoragePlan sentinel;
  sentinel.working = {41, 42};
  for (size_t bad = 0; bad < 8; ++bad) {
    auto invalid = o;
    Extent2D extent{89, 57};
    switch (bad) {
    case 0:
      invalid.encoding.backend = VarDctBackendPreference::kMetal;
      break;
    case 1:
      invalid.encoding.effort = 0;
      break;
    case 2:
      invalid.encoding.cpu_thread_count = 257;
      break;
    case 3:
      invalid.encoding.target_size_maximum_attempts = 0;
      break;
    case 4:
      invalid.encoding.target_size_maximum_attempts = 65;
      break;
    case 5:
      invalid.encoding.rate_control_mode =
          static_cast<VarDctRateControlMode>(99);
      break;
    case 6:
      extent = {};
      break;
    case 7:
      extent = {std::numeric_limits<size_t>::max(), 8};
      break;
    }
    auto p = sentinel;
    if (!Check(!ComputeCpuWorkflowStoragePlan(extent, invalid, &p).ok() &&
                   p == sentinel,
               "Invalid CPU workflow changed output"))
      return false;
  }
  if (!Check(!ComputeCpuWorkflowStoragePlan({8, 8}, o, nullptr).ok(),
             "Null CPU plan accepted"))
    return false;
  CpuAqStoragePlan aq_sentinel;
  aq_sentinel.working = {43, 44};
  for (size_t bad = 0; bad < 5; ++bad) {
    CpuAqStorageOptions options;
    Extent2D extent{65, 63};
    switch (bad) {
    case 0:
      options.control = static_cast<AdaptiveQuantizationControlMode>(99);
      break;
    case 1:
      options.iterations = 5;
      break;
    case 2:
      options.epf_iterations = 4;
      break;
    case 3:
      extent = {};
      break;
    case 4:
      extent = {std::numeric_limits<size_t>::max(), 8};
      break;
    }
    auto p = aq_sentinel;
    if (!Check(!ComputeCpuAqStoragePlan(extent, options, &p).ok() &&
                   p == aq_sentinel,
               "Invalid CPU AQ shape changed its plan"))
      return false;
  }
  CpuAqStorageOptions maximum_options;
  maximum_options.control = AdaptiveQuantizationControlMode::kMaximumError;
  maximum_options.iterations = std::numeric_limits<size_t>::max();
  CpuAqStoragePlan maximum_plan;
  if (!Ok(ComputeCpuAqStoragePlan({17, 9}, maximum_options, &maximum_plan)) ||
      !Check(maximum_plan.policy.evaluations == 6 &&
                 !ComputeCpuAqStoragePlan({8, 8}, {}, nullptr).ok() &&
                 !ComputeAqPolicyStoragePlan(
                      {1, 1}, AdaptiveQuantizationControlMode::kButteraugli, 0,
                      false, nullptr)
                      .ok(),
             "Pinned maximum-error count or null component output failed"))
    return false;
  job.Reset();
  std::cout << "CPU workflow plan shapes: " << cases << "; search bounds: 64\n";
  return Empty(budget);
}

struct MockEvaluator final : AdaptiveQuantizationEvaluator {
  size_t calls = 0;
  Status Evaluate(ConstPlaneF32View field, float, bool,
                  AdaptiveQuantizationEvaluation *out,
                  EvaluationProfile *) override {
    AdaptiveQuantizationEvaluation next;
    const float score = 0.55f + 0.04f * static_cast<float>(calls);
    next.block_distance.assign(field.extent.width * field.extent.height, score);
    auto s = Quantizer::Create({}, &next.quantizer);
    if (!s.ok())
      return s;
    next.score = score;
    next.maximum_error = {{score, score, score}, score};
    *out = std::move(next);
    ++calls;
    return Status::Ok();
  }
};

bool CheckPolicy() {
  ResourceBudget inputs(0);
  ResourceReservation input_job;
  if (!Ok(inputs.TryReserve(64, &input_job)))
    return false;
  AcStrategyGrid grid;
  {
    ResourceContextScope input_scope({&input_job, ResourceClass::kInput});
    if (!Ok(AcStrategyGrid::Create({8, 8}, &grid)) ||
        !Ok(grid.Set(0, 0, AcStrategyType::kDct32x32)))
      return false;
    grid.fill_empty_dct8();
  }
  std::array<float, 64> initial;
  initial.fill(0.5f);
  size_t cases = 0;
  for (bool maximum : {false, true})
    for (size_t iterations = 0; iterations <= 4; ++iterations)
      for (bool adjusted : {false, true})
        for (bool profile : {false, true}) {
          AdaptiveQuantizationOptions options;
          options.control_mode =
              maximum ? AdaptiveQuantizationControlMode::kMaximumError
                      : AdaptiveQuantizationControlMode::kButteraugli;
          options.iterations = iterations;
          AqPolicyStoragePlan p;
          if (!Ok(ComputeAqPolicyStoragePlan(grid.extent(),
                                             options.control_mode, iterations,
                                             profile, &p)))
            return false;
          // The mock's current returned map is outside the policy bound. Its
          // previous map is already included; reserve one extra incoming owner.
          ResourceBudget budget(p.working.peak_bytes + sizeof(initial));
          ResourceReservation job;
          if (!Ok(budget.TryReserve(p.working.peak_bytes + sizeof(initial),
                                    &job)))
            return false;
          AdaptiveQuantizationPolicyResult result;
          AdaptiveQuantizationProfile measured;
          {
            ResourceContextScope scope({&job, ResourceClass::kAqScratch});
            MockEvaluator evaluator;
            const ConstPlaneF32View field{initial.data(), grid.extent(), 8};
            auto s = adjusted ? RunAdaptiveQuantizationPolicyAdjusted(
                                    grid, field, options, evaluator, &result,
                                    profile ? &measured : nullptr)
                              : RunAdaptiveQuantizationPolicy(
                                    grid, field, options, evaluator, &result,
                                    profile ? &measured : nullptr);
            if (!Ok(s) ||
                !Check(evaluator.calls == p.evaluations &&
                           result.score_history.size() == p.evaluations &&
                           measured.evaluations.size() ==
                               (profile ? p.evaluations : 0),
                       "Policy evaluation count exceeded its plan"))
              return false;
          }
          job.Reset();
          if (!Check(budget.snapshot().total.live_capacity_bytes ==
                         p.output.retained_bytes + p.profile.retained_bytes,
                     "Retained AQ result/profile did not retain exact charges"))
            return false;
          result = {};
          measured = {};
          if (!Empty(budget))
            return false;
          ++cases;
        }
  grid = {};
  input_job.Reset();
  std::cout << "Bounded shared-policy cases: " << cases << '\n';
  return Empty(inputs);
}

struct Result {
  std::vector<uint8_t> bytes;
  VarDctEncodingSummary summary;
  VarDctEncodingTiming timing;
  VarDctEncodingProfile profile;
};

struct AqResult {
  Image reconstructed;
  std::vector<float> field, distance;
  std::vector<double> scores;
  std::vector<uint64_t> frame_snapshot;
  MaximumErrorResult maximum;
  AqResult(Extent2D source, size_t blocks)
      : reconstructed(source), field(blocks, -17), distance(blocks, -17) {}
};

bool SnapshotFrame(const VarDctEncoderFrame &frame,
                   std::vector<uint64_t> *out) {
  // Test-owned exact snapshot, not encoder backing. This also supports AQ's
  // diagnostic filter profiles, which the simple serializer rightly rejects.
  const auto append = [&](const auto &plane) {
    out->push_back(plane.extent.width);
    out->push_back(plane.extent.height);
    for (size_t y = 0; y < plane.extent.height; ++y)
      for (size_t x = 0; x < plane.extent.width; ++x) {
        using Value = std::remove_cvref_t<decltype(plane.Row(y)[x])>;
        if constexpr (std::is_same_v<Value, float>)
          out->push_back(std::bit_cast<uint32_t>(plane.Row(y)[x]));
        else
          out->push_back(static_cast<uint64_t>(plane.Row(y)[x]));
      }
  };
  out->push_back(frame.quantizer().params().global_scale);
  out->push_back(frame.quantizer().params().quant_dc);
  append(frame.raw_quant_field());
  append(frame.epf_sharpness());
  append(frame.color_correlation().y_to_x_map());
  append(frame.color_correlation().y_to_b_map());
  for (size_t c = 0; c < 3; ++c) {
    append(frame.quantized_dc().plane[c]);
    append(frame.dc().plane[c]);
  }
  for (size_t y = 0; y < frame.strategies().extent().height; ++y)
    for (size_t x = 0; x < frame.strategies().extent().width; ++x) {
      AcStrategyCell cell;
      if (!Ok(frame.strategies().Get(x, y, &cell)))
        return false;
      out->push_back(static_cast<uint64_t>(cell.strategy));
      out->push_back(cell.is_anchor);
    }
  for (size_t i = 0; i < frame.ac_group_count(); ++i) {
    VarDctAcGroupView group;
    if (!Ok(frame.GetAcGroup(i, &group)))
      return false;
    out->push_back(group.block_x);
    out->push_back(group.block_y);
    out->push_back(group.block_extent.width);
    out->push_back(group.block_extent.height);
    out->push_back(group.used_coefficient_count);
    for (const auto &channel : group.coefficients)
      for (int32_t coefficient : channel)
        out->push_back(static_cast<uint64_t>(coefficient));
  }
  return true;
}

bool CheckEvaluators() {
  size_t cases = 0;
  for (Extent2D source :
       {Extent2D{1, 1}, {14, 15}, {15, 15}, {65, 63}, {257, 257}}) {
    Image original(source);
    FrameGeometry geometry;
    if (!Ok(FrameGeometry::Create(source, &geometry)))
      return false;
    Image opsin(geometry.padded_frame());
    const auto blocks = geometry.block_grid().blocks;
    const size_t count = blocks.width * blocks.height;
    ColorTransformStoragePlan input_color;
    if (!Ok(ComputeColorTransformStoragePlan(source, opsin.extent, true, 1,
                                             &input_color)))
      return false;
    ResourceBudget inputs(input_color.working.peak_bytes + count);
    ResourceReservation input_job;
    if (!Ok(inputs.TryReserve(input_color.working.peak_bytes + count,
                              &input_job)))
      return false;
    AcStrategyGrid grid;
    {
      ResourceContextScope scope({&input_job, ResourceClass::kInput});
      thread_budget_internal::EncodeScope cpu(1);
      if (!Ok(color_transform_internal::LinearRgbToPaddedOpsin(
              original.const_view(), 255.0f, opsin.view())) ||
          !Ok(AcStrategyGrid::Create(blocks, &grid)))
        return false;
      if (blocks.width >= 4 && blocks.height >= 4 &&
          !Ok(grid.Set(0, 0, AcStrategyType::kDct32x32)))
        return false;
      grid.fill_empty_dct8();
    }
    std::vector<float> initial(count, 0.5f);
    std::vector<uint8_t> sharpness(count, 4);
    for (size_t flags = 0; flags < 16; ++flags) {
      const bool maximum = bool(flags & 8);
      CpuAqStorageOptions o;
      o.control = maximum ? AdaptiveQuantizationControlMode::kMaximumError
                          : AdaptiveQuantizationControlMode::kButteraugli;
      o.iterations = flags % 5;
      o.cpu_thread_count = flags & 1 ? 0 : 4;
      o.gaborish = bool(flags & 4);
      o.epf_iterations = flags % 4;
      o.collect_profile = bool(flags & 2);
      // There is no combined prepared-reference/profile public AQ overload.
      o.prepared_reference = !o.collect_profile && bool(flags & 1);
      CpuAqStoragePlan p;
      if (!Ok(ComputeCpuAqStoragePlan(source, o, &p)))
        return false;
      AqResult oracle(source, count), candidate(source, count);
      for (size_t pass = 0; pass < 2; ++pass) {
        const size_t capacity =
            pass == 0 ? std::max(p.working.peak_bytes, size_t{64} << 20)
                      : p.working.peak_bytes;
        ResourceBudget budget(capacity);
        ResourceReservation job;
        if (!Ok(budget.TryReserve(capacity, &job)))
          return false;
        auto &result = pass == 0 ? oracle : candidate;
        {
          ResourceContextScope scope({&job, ResourceClass::kAqScratch});
          thread_budget_internal::EncodeScope cpu(o.cpu_thread_count);
          AdaptiveQuantizationOptions options;
          options.control_mode = o.control;
          options.iterations = o.iterations;
          options.maximum_error = {0.05f, 0.05f, 0.05f};
          options.profile.loop_filter.gaborish = o.gaborish;
          options.profile.loop_filter.epf_options.iterations = o.epf_iterations;
          PreparedButteraugliReference reference;
          if (!maximum && o.prepared_reference &&
              !Ok(reference.Prepare(original.const_view(),
                                    options.butteraugli)))
            return false;
          VarDctEncoderFrame frame;
          AdaptiveQuantizationProfile profile;
          const AdaptiveQuantizationOutput output{
              .quant_field = {result.field.data(), blocks, blocks.width},
              .block_distance_map = {result.distance.data(), blocks,
                                     blocks.width},
              .reconstructed_linear_rgb = result.reconstructed.view(),
              .frame = &frame,
              .score_history = &result.scores,
              .maximum_error_result = &result.maximum};
          const ConstPlaneF32View field{initial.data(), blocks, blocks.width};
          const ConstPlaneU8View sharp{sharpness.data(), blocks, blocks.width};
          const auto status =
              o.collect_profile
                  ? FindBestQuantizationProfiled(
                        original.const_view(), opsin.const_view(), grid, field,
                        sharp, options, output, &profile)
                  : FindBestQuantizationPrepared(
                        original.const_view(), opsin.const_view(), grid, field,
                        sharp, options,
                        !maximum && o.prepared_reference ? &reference : nullptr,
                        output);
          if (!Ok(status) ||
              !Check(frame.valid() &&
                         result.scores.size() == p.policy.evaluations &&
                         profile.evaluations.size() ==
                             (o.collect_profile ? p.policy.evaluations : 0),
                     "CPU evaluator result exceeded its isolated plan")) {
            std::cerr << source.width << 'x' << source.height
                      << " flags=" << flags << " plan=" << capacity << '\n';
            return false;
          }
          if (!Check(frame.profile() == options.profile,
                     "CPU AQ changed its profile metadata") ||
              !SnapshotFrame(frame, &result.frame_snapshot))
            return false;
        }
        job.Reset();
        if (!Empty(budget))
          return false;
      }
      if (!Check(candidate.frame_snapshot == oracle.frame_snapshot &&
                     candidate.scores == oracle.scores &&
                     candidate.field == oracle.field &&
                     candidate.distance == oracle.distance &&
                     candidate.reconstructed.planes ==
                         oracle.reconstructed.planes &&
                     candidate.maximum == oracle.maximum,
                 "Isolated bounded CPU AQ changed output"))
        return false;
      ++cases;
    }
    grid = {};
    input_job.Reset();
    if (!Empty(inputs))
      return false;
  }
  std::cout << "Isolated bounded CPU evaluator cases: " << cases << '\n';
  return true;
}
Status Encode(ConstImage3FView source, const CpuWorkflowStorageOptions &o,
              Result *result) {
  if (o.collect_timing)
    return EncodeLinearRgbVarDctCodestreamProfiled(
        source, o.encoding, &result->bytes, &result->summary, &result->timing);
  if (o.collect_profile)
    return EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
        source, o.encoding, nullptr, false, &result->bytes, &result->summary,
        &result->profile);
  return EncodeLinearRgbVarDctCodestream(source, o.encoding, &result->bytes,
                                         &result->summary);
}
bool RunWorkflow(ConstImage3FView source, const CpuWorkflowStorageOptions &o) {
  CpuWorkflowStoragePlan p;
  if (!Ok(ComputeCpuWorkflowStoragePlan(source.extent(), o, &p)))
    return false;
  Result oracle, measured;
  for (size_t pass = 0; pass < 3; ++pass) {
    const size_t capacity =
        pass == 0 ? std::max(p.working.peak_bytes, size_t{64} << 20)
                  : p.working.peak_bytes;
    ResourceBudget budget(capacity);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(capacity, &job)))
      return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      auto options = o;
      if (pass == 0)
        options.collect_profile = options.collect_timing = false;
      const auto status =
          Encode(source, options, pass == 0 ? &oracle : &measured);
      if (!Ok(status)) {
        std::cerr << source.width() << 'x' << source.height()
                  << " effort=" << o.encoding.effort
                  << " mode=" << static_cast<int>(o.encoding.rate_control_mode)
                  << " plan=" << capacity << '\n';
        return false;
      }
      if (pass != 0 &&
          !Check(
              measured.bytes == oracle.bytes &&
                  measured.summary == oracle.summary &&
                  measured.summary.score_history.size() == p.score_count &&
                  (!o.collect_timing ||
                   measured.timing.attempts.size() ==
                       measured.summary.encode_attempt_count),
              "Bounded CPU workflow changed bytes, decisions or timing count"))
        return false;
    }
    if (source.width() >= 3839) {
      const auto &result = pass == 0 ? oracle : measured;
      std::cout << "Large CPU pass " << pass
                << ": plan=" << p.working.peak_bytes
                << " managed_peak=" << budget.snapshot().peak_backing_bytes
                << " bytes=" << result.bytes.size() << '\n';
    }
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  return true;
}

bool CheckRuntime() {
  size_t cases = 0;
  for (Extent2D extent :
       {Extent2D{1, 1}, {14, 15}, {15, 15}, {65, 63}, {257, 9}}) {
    Image image(extent);
    for (int effort : {1, 4, 7, 9, 10})
      for (bool maximum : {false, true}) {
        CpuWorkflowStorageOptions o;
        o.encoding.backend = VarDctBackendPreference::kCpu;
        o.encoding.effort = effort;
        o.encoding.cpu_thread_count = 1;
        o.encoding.rate_control_mode =
            maximum ? VarDctRateControlMode::kMaximumError
                    : VarDctRateControlMode::kButteraugliTarget;
        o.encoding.maximum_error = {0.05f, 0.05f, 0.05f};
        o.collect_profile = true;
        if (!RunWorkflow(image.const_view(), o))
          return false;
        ++cases;
      }
  }
  Image image({89, 57});
  for (size_t flags = 0; flags < 16; ++flags) {
    CpuWorkflowStorageOptions o;
    o.encoding.backend = flags & 1 ? VarDctBackendPreference::kAutomatic
                                   : VarDctBackendPreference::kCpu;
    o.encoding.rate_control_mode =
        flags & 2 ? VarDctRateControlMode::kTargetBytes
                  : VarDctRateControlMode::kTargetBitsPerPixel;
    o.encoding.target_bytes = 650;
    o.encoding.target_bits_per_pixel = 1.2;
    o.encoding.target_size_tolerance = 0;
    o.encoding.target_size_maximum_attempts = flags & 4 ? 4 : 1;
    o.encoding.target_size_selection =
        flags & 8 ? TargetSizeSelectionPolicy::kClosestAbsolute
                  : TargetSizeSelectionPolicy::kLargestAtOrBelow;
    o.encoding.cpu_thread_count = 1;
    o.collect_timing = true;
    if (!RunWorkflow(image.const_view(), o))
      return false;
    ++cases;
  }
  Image large({257, 257});
  for (size_t flags = 0; flags < 4; ++flags) {
    CpuWorkflowStorageOptions o;
    o.encoding.backend = VarDctBackendPreference::kCpu;
    o.encoding.cpu_thread_count = flags & 1 ? 4 : 0;
    o.encoding.density_mode = flags & 2 ? VarDctDensityMode::kHighDensity
                                        : VarDctDensityMode::kDefault;
    o.encoding.effort = 1;
    o.encoding.compression_mode = VarDctCompressionMode::kMaximumCompression;
    o.collect_profile = true;
    if (!RunWorkflow(large.const_view(), o))
      return false;
    ++cases;
  }
  std::cout << "Bounded CPU whole-workflow cases: " << cases << '\n';
  return true;
}

bool CheckFailure() {
  Image image({17, 9});
  CpuWorkflowStorageOptions o;
  o.encoding.backend = VarDctBackendPreference::kCpu;
  o.encoding.effort = 1;
  o.encoding.cpu_thread_count = 1;
  o.collect_profile = true;
  CpuWorkflowStoragePlan p;
  if (!Ok(ComputeCpuWorkflowStoragePlan(image.extent, o, &p)))
    return false;
  ResourceBudget rejected(p.working.peak_bytes - 1);
  ResourceReservation none;
  if (!Check(!rejected.TryReserve(p.working.peak_bytes, &none).ok(),
             "Oversized CPU job admitted") ||
      !Empty(rejected))
    return false;
  for (bool underplan : {false, true}) {
    ResourceBudget budget(underplan ? 1 : p.working.peak_bytes);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(underplan ? 1 : p.working.peak_bytes, &job)))
      return false;
    Result result;
    result.bytes = {1, 2, 3};
    result.summary.score_history = {42};
    const auto summary = result.summary;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      if (!underplan)
        ArmNextManagedHostAllocationFailureForTest();
      auto s = Encode(image.const_view(), o, &result);
      DisarmManagedHostAllocationFailureForTest();
      if (!Check(s.code() == StatusCode::kOutOfMemory &&
                     s.resource_plan_exceeded() == underplan &&
                     result.bytes == std::vector<uint8_t>({1, 2, 3}) &&
                     result.summary == summary,
                 "CPU allocation failure lost type or atomic output"))
        return false;
      if (!underplan && !Ok(Encode(image.const_view(), o, &result)))
        return false;
    }
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--plans-only")
    return CheckPlans() && CheckPolicy() ? EXIT_SUCCESS : EXIT_FAILURE;
  if (argc == 2 && std::string_view(argv[1]) == "--large") {
    Image image({3839, 2159});
    CpuWorkflowStorageOptions o;
    o.encoding.backend = VarDctBackendPreference::kCpu;
    o.encoding.effort = 1;
    o.encoding.cpu_thread_count = 4;
    return RunWorkflow(image.const_view(), o) ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  return CheckPlans() && CheckPolicy() && CheckEvaluators() && CheckRuntime() &&
                 CheckFailure() && Empty(DefaultResourceBudget()) &&
                 Check(DefaultResourceBudget().snapshot().peak_backing_bytes ==
                           0,
                       "CPU backing escaped its explicit domain")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
