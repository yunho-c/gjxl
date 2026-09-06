// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include "codec/adaptive_quantization_internal.h"
#include "codec/color_transform.h"
#include "codec/frontend_storage_plan.h"
#include "codec/vardct_frame_view_internal.h"
#include "core/image_buffer.h"
#include "gpu/metal/metal_aq_host_storage_plan.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_storage_plan.h"

namespace {
using namespace gjxl;
using namespace gjxl::metal_internal;
using namespace gjxl::resource_budget_internal;
using vardct_frame_internal::BorrowFrame;
using vardct_frame_internal::CompletedVarDctFrame;
using vardct_frame_internal::VarDctFrameView;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Ok(const Status &status) {
  if (!status.ok())
    std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                   s.total.pending_count == 0 && s.open_reservations == 0 &&
                   s.waiting_requests == 0,
               "AQ host plan leaked backing or admission");
}
size_t HostBytes(const ResourceBudget &budget) {
  return budget.snapshot()
      .classes[static_cast<size_t>(ResourceClass::kPreparation)]
      .live_capacity_bytes;
}

bool Equal(const VarDctFrameView &a, const VarDctFrameView &b) {
  if (!a.valid() || !b.valid() ||
      a.strategies().extent() != b.strategies().extent() ||
      a.ac_group_count() != b.ac_group_count() ||
      a.geometry().frame() != b.geometry().frame() ||
      a.profile() != b.profile() ||
      a.quantizer().params().global_scale !=
          b.quantizer().params().global_scale ||
      a.quantizer().params().quant_dc != b.quantizer().params().quant_dc ||
      a.color_correlation().tile_extent() !=
          b.color_correlation().tile_extent())
    return false;
  const auto blocks = a.strategies().extent();
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      AcStrategyCell ac, bc;
      if (!Ok(a.strategies().Get(x, y, &ac)) ||
          !Ok(b.strategies().Get(x, y, &bc)) || ac.strategy != bc.strategy ||
          ac.is_anchor != bc.is_anchor)
        return false;
      if (a.raw_quant_field().Row(y)[x] != b.raw_quant_field().Row(y)[x] ||
          a.epf_sharpness().Row(y)[x] != b.epf_sharpness().Row(y)[x])
        return false;
      for (size_t c = 0; c < 3; ++c) {
        if (a.quantized_dc().plane[c].Row(y)[x] !=
                b.quantized_dc().plane[c].Row(y)[x] ||
            a.dc().plane[c].Row(y)[x] != b.dc().plane[c].Row(y)[x])
          return false;
      }
    }
  }
  const auto tiles = a.color_correlation().tile_extent();
  for (size_t y = 0; y < tiles.height; ++y)
    for (size_t x = 0; x < tiles.width; ++x)
      if (a.color_correlation().y_to_x_map().Row(y)[x] !=
              b.color_correlation().y_to_x_map().Row(y)[x] ||
          a.color_correlation().y_to_b_map().Row(y)[x] !=
              b.color_correlation().y_to_b_map().Row(y)[x])
        return false;
  for (size_t g = 0; g < a.ac_group_count(); ++g) {
    VarDctAcGroupView av, bv;
    if (!Ok(a.GetAcGroup(g, &av)) || !Ok(b.GetAcGroup(g, &bv)) ||
        av.used_coefficient_count != bv.used_coefficient_count)
      return false;
    for (size_t c = 0; c < 3; ++c)
      if (!std::ranges::equal(av.coefficients[c], bv.coefficients[c]))
        return false;
  }
  return true;
}

bool CheckPlans() {
  size_t cases = 0;
  for (const Extent2D source : {Extent2D{1, 1},
                                {8, 8},
                                {17, 9},
                                {65, 63},
                                {257, 9},
                                {89, 57},
                                {1919, 1079},
                                {3839, 2159}}) {
    const Extent2D coding{(source.width + 7) / 8 * 8,
                          (source.height + 7) / 8 * 8};
    const size_t pixels = coding.width * coding.height, blocks = pixels / 64;
    const size_t tiles =
        ((coding.width + 63) / 64) * ((coding.height + 63) / 64);
    const size_t groups =
        ((coding.width + 255) / 256) * ((coding.height + 255) / 256);
    for (size_t flags = 0; flags < 2048; ++flags) {
      AqHostStorageOptions o{
          .source_extent = source,
          .coding_extent = coding,
          .frame_only = bool(flags & 1),
          .resident_initial_quant = bool(flags & 2),
          .resident_ac_strategy_inputs = bool(flags & 4),
          .resident_quantization = bool(flags & 8),
          .defer_final_transform_metadata = bool(flags & 16),
          .metric = flags & 32 ? AqEvaluationMetric::kMaximumError
                               : AqEvaluationMetric::kButteraugli,
          .reconfigure = bool(flags & 64),
          .exact_coefficients = bool(flags & 128),
          .reconstruct_exact_coefficients = bool(flags & 256),
          .reconstructed_rgb_readback = bool(flags & 512),
          .resident_quant_field_readback = bool(flags & 1024)};
      const bool invalid =
          (o.resident_ac_strategy_inputs && !o.resident_initial_quant) ||
          (o.defer_final_transform_metadata &&
           (!o.resident_ac_strategy_inputs || !o.resident_quantization ||
            o.frame_only)) ||
          (o.frame_only &&
           (o.reconfigure || o.exact_coefficients ||
            o.reconstructed_rgb_readback || o.resident_quant_field_readback)) ||
          (o.reconstruct_exact_coefficients && !o.exact_coefficients) ||
          (o.resident_quant_field_readback && !o.resident_quantization);
      AqHostStoragePlan plan;
      plan.working = {42, 43};
      const auto sentinel = plan;
      ArmNextManagedHostAllocationFailureForTest();
      const Status status = ComputeAqHostStoragePlan(o, &plan);
      const bool no_allocation = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (!Check(no_allocation &&
                     (invalid ? !status.ok() && plan == sentinel : status.ok()),
                 "AQ plan validation, atomicity or allocation-free contract "
                 "failed"))
        return false;
      if (invalid)
        continue;
      ++cases;
      if (!Check(
              plan.prepared.retained_bytes <= plan.prepared.peak_bytes &&
                  plan.preparation.peak_bytes >=
                      plan.prepared.peak_bytes + 4 * kAqQuantTableValueCount &&
                  plan.retained.retained_bytes >=
                      plan.prepared.retained_bytes + 4 * blocks &&
                  plan.working.peak_bytes ==
                      std::max(plan.preparation.peak_bytes,
                               plan.operation.peak_bytes) &&
                  plan.working.retained_bytes == plan.retained.retained_bytes,
              "AQ plan phase composition failed"))
        return false;
      if (o.resident_initial_quant) {
        auto masked = o;
        masked.initial_pixel_mask_readback = true;
        AqHostStoragePlan diagnostic;
        if (!Ok(ComputeAqHostStoragePlan(masked, &diagnostic)) ||
            !Check(diagnostic.prepared == plan.prepared &&
                       diagnostic.retained.retained_bytes -
                               plan.retained.retained_bytes ==
                           (o.resident_ac_strategy_inputs ? 4 * pixels : 0),
                   "Lazy initial mask was omitted or counted twice"))
          return false;
      }
    }
    for (const size_t anchors : {size_t{1}, blocks}) {
      CompletedFrameHostStoragePlan completed;
      if (!Ok(ComputeCompletedFrameHostStoragePlan(source, coding, anchors,
                                                   &completed)) ||
          !Check(completed.output.retained_bytes ==
                         30 * blocks + 2 * tiles + sizeof(size_t) * groups &&
                     completed.output.peak_bytes ==
                         completed.output.retained_bytes &&
                     completed.working.peak_bytes ==
                         completed.output.peak_bytes + 4 * anchors,
                 "Completed host snapshot counts changed"))
        return false;
    }
  }
  AqHostStorageOptions invalid{.source_extent = {8, 8},
                               .coding_extent = {8, 8}};
  AqHostStoragePlan plan;
  if (!Ok(ComputeAqHostStoragePlan(invalid, &plan)))
    return false;
  const auto sentinel = plan;
  for (const Extent2D bad : {Extent2D{},
                             {9, 8},
                             {65536, 65536},
                             {std::numeric_limits<size_t>::max(), 8}}) {
    invalid.source_extent = invalid.coding_extent = bad;
    if (!Check(!ComputeAqHostStoragePlan(invalid, &plan).ok() &&
                   plan == sentinel,
               "AQ bad geometry changed output"))
      return false;
  }
  if (!Check(
          !ComputeAqHostStoragePlan({}, nullptr).ok() &&
              !ComputeCompletedFrameHostStoragePlan({8, 8}, {8, 8}, 1, nullptr)
                   .ok(),
          "Null plan output accepted"))
    return false;
  std::cout << "AQ host plan formula cases: " << cases << '\n';
  return true;
}

struct Fixture {
  Extent2D source_extent, coding, blocks, tiles;
  size_t block_count;
  Image3FBuffer source, rgb, opsin, diagnostic;
  AcStrategyGrid grid, mixed;
  std::vector<uint8_t> sharpness;
  std::vector<int8_t> cfl;
  std::vector<int32_t> raw;
  std::vector<float> sigma, field, mask, pixel_mask, map, adjusted;
  explicit Fixture(Extent2D extent)
      : source_extent(extent),
        coding{(extent.width + 7) / 8 * 8, (extent.height + 7) / 8 * 8},
        blocks{coding.width / 8, coding.height / 8},
        tiles{(coding.width + 63) / 64, (coding.height + 63) / 64},
        block_count(blocks.width * blocks.height), source(extent), rgb(coding),
        opsin(coding), diagnostic(extent), sharpness(block_count, 4),
        cfl(tiles.width * tiles.height), raw(block_count, 16),
        sigma(block_count, -0.05f), field(block_count, 0.8f), mask(block_count),
        pixel_mask(coding.width * coding.height), map(block_count),
        adjusted(block_count) {}
  bool Init() {
    if (!Ok(AcStrategyGrid::Create(blocks, &grid)) ||
        !Ok(AcStrategyGrid::Create(blocks, &mixed)))
      return false;
    grid.fill_dct8();
    if (blocks.width >= 12 && blocks.height >= 8) {
      for (const auto [x, y, type] :
           std::array<std::tuple<size_t, size_t, AcStrategyType>, 6>{
               {{0, 0, AcStrategyType::kDct32x32},
                {4, 0, AcStrategyType::kDct32x16},
                {6, 0, AcStrategyType::kDct16x32},
                {10, 0, AcStrategyType::kDct16x16},
                {6, 2, AcStrategyType::kDct16x8},
                {7, 2, AcStrategyType::kDct8x16}}})
        if (!Ok(mixed.Set(x, y, type)))
          return false;
    } else if (blocks.width >= 2 && blocks.height >= 2) {
      if (!Ok(mixed.Set(0, 0, AcStrategyType::kDct16x16)))
        return false;
    }
    mixed.fill_empty_dct8();
    for (size_t y = 0; y < coding.height; ++y) {
      for (size_t x = 0; x < coding.width; ++x) {
        for (size_t c = 0; c < 3; ++c) {
          const float value =
              0.03f +
              0.8f *
                  float((std::min(x, source_extent.width - 1) * (c + 3) +
                         std::min(y, source_extent.height - 1) * 7) %
                        127) /
                  127.0f;
          rgb.view().plane[c].Row(y)[x] = value;
          if (x < source_extent.width && y < source_extent.height)
            source.view().plane[c].Row(y)[x] = value;
        }
      }
    }
    return Ok(LinearRgbToOpsin(rgb.const_view(), 255.0f, opsin.view()));
  }
  AqEvaluationInput Input() const {
    return {.raw_quant_field = {raw.data(), blocks, blocks.width},
            .quantizer = {1173, 43},
            .epf_inverse_sigma = {sigma.data(), blocks, blocks.width}};
  }
  AqEvaluationPreparation Preparation(size_t mode) const {
    AqEvaluationOptions options;
    options.profile.loop_filter.gaborish = false;
    options.profile.loop_filter.epf_options.iterations = 0;
    options.metric = mode == 3 ? AqEvaluationMetric::kMaximumError
                               : AqEvaluationMetric::kButteraugli;
    options.maximum_error = {0.1f, 0.1f, 0.1f};
    return {.original_linear_rgb = source.const_view(),
            .coding_opsin = opsin.const_view(),
            .strategies = &grid,
            .epf_sharpness = {sharpness.data(), blocks, blocks.width},
            .options = options,
            .frame_only = mode == 4 || mode == 5,
            .frame_only_resident_initial_quant = mode == 2 || mode == 5,
            .resident_ac_strategy_inputs = mode == 2,
            .resident_quantization = mode == 1 || mode == 2,
            .coefficient_decision_mode =
                AcCoefficientDecisionMode::kAdjustedSharedQuant,
            .defer_final_transform_metadata = mode == 2};
  }
  AqHostStorageOptions Options(size_t mode) const {
    const auto p = Preparation(mode);
    return {.source_extent = source_extent,
            .coding_extent = coding,
            .frame_only = p.frame_only,
            .resident_initial_quant = p.frame_only_resident_initial_quant,
            .resident_ac_strategy_inputs = p.resident_ac_strategy_inputs,
            .resident_quantization = p.resident_quantization,
            .defer_final_transform_metadata = p.defer_final_transform_metadata,
            .metric = p.options.metric,
            .reconfigure = !p.frame_only,
            .exact_coefficients = mode == 0 || mode == 3 || mode == 6,
            .reconstruct_exact_coefficients = mode == 0 || mode == 3,
            .reconstructed_rgb_readback = !p.frame_only,
            .resident_quant_field_readback = p.resident_quantization,
            .initial_pixel_mask_readback = p.frame_only_resident_initial_quant};
  }
};

struct Envelope {
  AqHostStoragePlan host;
  CompletedFrameHostStoragePlan completed_host;
  CompletedFrameStoragePlan completed_device;
  frontend_storage_internal::OwnedFrameStoragePlan owned;
  size_t device = 0, capacity = 0;
  bool Init(const Fixture &f, size_t mode) {
    const auto options = f.Options(mode);
    AqStoragePlan aq;
    if (!Ok(ComputeAqHostStoragePlan(options, &host)) ||
        !Ok(ComputeCompletedFrameHostStoragePlan(
            f.source_extent, f.coding, f.block_count, &completed_host)) ||
        !Ok(ComputeCompletedFrameStoragePlan(
            f.source_extent, f.coding, f.block_count, &completed_device)) ||
        !Ok(frontend_storage_internal::ComputeOwnedFrameStoragePlan(
            f.source_extent, &owned)) ||
        !Ok(ComputeAqStoragePlan(
            {.source_extent = f.source_extent,
             .coding_extent = f.coding,
             .anchor_capacity_count = f.block_count,
             .maximum_coefficient_count = 1024,
             .frame_only = options.frame_only,
             .needs_reconstructed = !options.frame_only,
             .frame_only_resident_initial_quant =
                 options.resident_initial_quant,
             .resident_quantization = options.resident_quantization,
             .uses_butteraugli_sinks =
                 !options.frame_only &&
                 options.metric == AqEvaluationMetric::kButteraugli &&
                 f.source_extent.width >= 15 && f.source_extent.height >= 15,
             .metric = options.metric},
            &aq)))
      return false;
    device = aq.persistent_bytes + aq.staging_bytes;
    if (!options.frame_only &&
        options.metric == AqEvaluationMetric::kButteraugli) {
      ButteraugliStoragePlan ba;
      if (!Ok(ComputeButteraugliStoragePlan(f.source_extent, false, &ba)))
        return false;
      device += ba.capacity_bytes;
    }
    // Two old owned frames can coexist with a fresh atomic replacement. The
    // independent lease and at most five score doubles are separate owners.
    capacity = device + host.working.peak_bytes + 3 * owned.output.peak_bytes +
               completed_host.working.peak_bytes +
               completed_device.capacity_bytes + 5 * sizeof(double);
    return true;
  }
};

bool RunOperations(GpuBackend &gpu, Fixture &f, size_t mode,
                   ResourceBudget &budget, const Envelope &envelope,
                   std::unique_ptr<PreparedAqEvaluation> &prepared) {
  VarDctEncoderFrame owned, repeat;
  std::unique_ptr<CompletedVarDctFrame> completed;
  const auto options = f.Options(mode);
  if (options.resident_initial_quant) {
    if (!Ok(prepared->ComputeInitialQuantization(
            {}, {{f.field.data(), f.blocks, f.blocks.width},
                 {f.mask.data(), f.blocks, f.blocks.width},
                 options.resident_ac_strategy_inputs
                     ? PlaneF32View{}
                     : PlaneF32View{f.pixel_mask.data(), f.coding,
                                    f.coding.width}})))
      return false;
    // Lazily materialize after the maskless call, then repeat with no mask.
    if (options.resident_ac_strategy_inputs &&
        (!Ok(prepared->ComputeInitialQuantization(
             {}, {{f.field.data(), f.blocks, f.blocks.width},
                  {f.mask.data(), f.blocks, f.blocks.width},
                  {f.pixel_mask.data(), f.coding, f.coding.width}})) ||
         !Ok(prepared->ComputeInitialQuantization(
             {}, {{f.field.data(), f.blocks, f.blocks.width},
                  {f.mask.data(), f.blocks, f.blocks.width},
                  {}}))))
      return false;
  }
  for (size_t configuration = 0;
       configuration < (options.reconfigure ? 3u : 1u); ++configuration) {
    const auto &grid = configuration == 1 ? f.mixed : f.grid;
    if (options.reconfigure &&
        !Ok(prepared->Reconfigure(
            grid, {f.sharpness.data(), f.blocks, f.blocks.width})))
      return false;
    if (!Ok(prepared->SetInvariantColorCorrelation(
            {f.cfl.data(), f.tiles, f.tiles.width},
            {f.cfl.data(), f.tiles, f.tiles.width})))
      return false;
    if (options.resident_quantization) {
      if (!Ok(prepared->AdjustQuantFieldResident(
              1.0f, {f.field.data(), f.blocks, f.blocks.width},
              {f.adjusted.data(), f.blocks, f.blocks.width})))
        return false;
      adaptive_quantization_internal::ButteraugliPolicySetup setup;
      if (!Ok(adaptive_quantization_internal::PrepareButteraugliPolicy(
              {f.adjusted.data(), f.blocks, f.blocks.width}, 1.0f, &setup)))
        return false;
      std::vector<double> expected_scores, scores;
      AqResidentButteraugliPolicyInput input{
          .adjusted_initial_quant_field = {f.adjusted.data(), f.blocks,
                                           f.blocks.width},
          .quant_dc = setup.quant_dc,
          .butteraugli_target = 1.0f,
          .lower_bound = setup.lower_bound,
          .upper_bound = setup.upper_bound,
          .iterations = configuration == 0 ? 4u : 2u,
          .evaluate_final_field = configuration != 2};
      if (!Ok(prepared->EvaluateResidentButteraugliPolicy(
              input, {.score_history = &expected_scores, .frame = &owned})))
        return false;
      AqResidentButteraugliPolicyOutput output{
          .quant_field = {f.field.data(), f.blocks, f.blocks.width},
          .block_distance_map =
              input.evaluate_final_field
                  ? PlaneF32View{f.map.data(), f.blocks, f.blocks.width}
                  : PlaneF32View{},
          .score_history = &scores,
          .reconstructed_linear_rgb =
              input.evaluate_final_field ? f.diagnostic.view() : Image3FView{},
          .completed_frame = &completed};
      // Drop the old lease here; the separate failure/lifetime tests cover its
      // replacement overlap. This envelope contains one completed output.
      completed.reset();
      if (!Ok(prepared->EvaluateResidentButteraugliPolicy(input, output)) ||
          !Check(scores == expected_scores &&
                     Equal(BorrowFrame(owned), completed->view()),
                 "Owned and completed AQ output differ"))
        return false;
      const size_t completed_bytes =
          budget.snapshot()
              .classes[static_cast<size_t>(ResourceClass::kCompletedFrame)]
              .live_capacity_bytes;
      if (!Check(completed_bytes <=
                     envelope.owned.output.retained_bytes +
                         envelope.completed_host.output.retained_bytes +
                         envelope.completed_device.capacity_bytes,
                 "Completed output exceeded its host/device bound"))
        return false;
    } else if (options.frame_only) {
      if (!Ok(prepared->EncodeFrame(f.Input(), &owned)) ||
          !Ok(prepared->EncodeFrame(f.Input(), &repeat)) ||
          !Check(Equal(BorrowFrame(owned), BorrowFrame(repeat)),
                 "Repeated frame-only export changed coefficients"))
        return false;
    } else {
      double score = 0;
      MaximumErrorReduction maximum;
      AqEvaluationOutput::Final final{
          .reconstructed_linear_rgb = f.diagnostic.view(), .frame = &owned};
      AqEvaluationOutput output{
          .block_distance_map = {f.map.data(), f.blocks, f.blocks.width},
          .score = &score,
          .maximum_error = options.metric == AqEvaluationMetric::kMaximumError
                               ? &maximum
                               : nullptr,
          .final = &final};
      if (!Ok(prepared->Evaluate(f.Input(), output)))
        return false;
      // Exercise both exact-prefix paths and their fixed-size retained staging.
      auto exact = f.Input();
      exact.exact_coefficients = &owned;
      exact.raw_quant_field = owned.raw_quant_field();
      exact.quantizer = owned.quantizer().params();
      final.frame = &repeat;
      if (mode == 6)
        exact.exact_reconstructed_linear_rgb = f.source.const_view();
      if (!Ok(prepared->Evaluate(exact, output)) ||
          !Check(Equal(BorrowFrame(owned), BorrowFrame(repeat)),
                 "Exact coefficient export changed coefficients"))
        return false;
      if (options.metric == AqEvaluationMetric::kButteraugli) {
        exact.exact_reconstructed_linear_rgb = f.source.const_view();
        if (!Ok(prepared->Evaluate(exact, output)) ||
            !Check(Equal(BorrowFrame(owned), BorrowFrame(repeat)),
                   "Exact linear-prefix export changed coefficients"))
          return false;
      }
    }
    if (!Check(HostBytes(budget) <= envelope.host.retained.retained_bytes &&
                   budget.snapshot().peak_backing_bytes <= envelope.capacity,
               "AQ retained host backing exceeded the plan"))
      return false;
    // Avoid a third owned output during next iteration's atomic replacement.
    repeat = {};
  }
  prepared.reset();
  if (!Ok(gpu.TrimPreparationCache()) ||
      !Check(HostBytes(budget) == 0,
             "AQ host storage survived evaluator destruction"))
    return false;
  return !completed ||
         Check(Equal(BorrowFrame(owned), completed->view()),
               "Completed frame borrowed the destroyed evaluator");
}

bool CheckRuntime(GpuBackend &gpu) {
  size_t cases = 0;
  for (const Extent2D extent :
       {Extent2D{1, 1}, {8, 8}, {17, 9}, {65, 63}, {257, 9}, {89, 57}}) {
    Fixture f(extent);
    if (!f.Init())
      return false;
    for (size_t mode = 0; mode < 7; ++mode) {
      std::cout << "AQ runtime " << extent.width << 'x' << extent.height
                << " mode " << mode << std::endl;
      if (!Ok(gpu.TrimPreparationCache()))
        return false;
      Envelope envelope;
      if (!envelope.Init(f, mode))
        return false;
      ResourceBudget budget(envelope.capacity);
      ResourceReservation job;
      if (!Ok(budget.TryReserve(envelope.capacity, &job)))
        return false;
      {
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        std::unique_ptr<PreparedAqEvaluation> prepared;
        if (!Ok(PrepareAqEvaluation(gpu, f.Preparation(mode), &prepared)) ||
            !Check(HostBytes(budget) <= envelope.host.prepared.retained_bytes &&
                       budget.snapshot().peak_backing_bytes <=
                           envelope.device +
                               envelope.host.preparation.peak_bytes,
                   "Prepared AQ host peak exceeded the plan") ||
            !RunOperations(gpu, f, mode, budget, envelope, prepared))
          return false;
      }
      job.Reset();
      if (!Empty(budget))
        return false;
      ++cases;
    }
  }
  std::cout << "AQ host real-Metal lifetime cases: " << cases << '\n';
  return true;
}

bool CheckPrepareFailures(GpuBackend &gpu) {
  Fixture f({89, 57});
  if (!f.Init())
    return false;
  Envelope envelope;
  if (!envelope.Init(f, 1))
    return false;
  size_t failures = 0;
  for (size_t position = 0; position < 256; ++position) {
    if (!Ok(gpu.TrimPreparationCache()))
      return false;
    ResourceBudget budget(envelope.capacity);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(envelope.capacity, &job)))
      return false;
    bool injected = false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      std::unique_ptr<PreparedAqEvaluation> prepared;
      ArmManagedHostAllocationFailureAfterForTest(position);
      const Status status =
          PrepareAqEvaluation(gpu, f.Preparation(1), &prepared);
      injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (injected) {
        ++failures;
        if (!Check(status.code() == StatusCode::kOutOfMemory && !prepared &&
                       budget.snapshot().total.pending_count == 0,
                   "AQ preparation failure was not atomic") ||
            !Ok(gpu.TrimPreparationCache()) ||
            !Ok(PrepareAqEvaluation(gpu, f.Preparation(1), &prepared)))
          return false;
      } else if (!Ok(status))
        return false;
      prepared.reset();
      if (!Ok(gpu.TrimPreparationCache()))
        return false;
    }
    job.Reset();
    if (!Empty(budget))
      return false;
    if (!injected)
      break;
  }
  if (!Check(failures > 20 && failures < 256,
             "AQ host failure enumeration did not finish"))
    return false;
  // Explicit underplans must not be converted to a physical OOM, submit work,
  // or consume the still-armed physical allocation failure.
  for (bool late : {false, true}) {
    ResourceBudget budget(envelope.capacity);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(envelope.capacity, &job)))
      return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      std::unique_ptr<PreparedAqEvaluation> prepared;
      if (late && !Ok(PrepareAqEvaluation(gpu, f.Preparation(1), &prepared)))
        return false;
      const auto before = gpu.stats();
      if (!Ok(job.ReduceCapacity(budget.snapshot().total.live_capacity_bytes)))
        return false;
      ArmNextManagedHostAllocationFailureForTest();
      const Status status =
          late ? prepared->Reconfigure(
                     f.mixed, {f.sharpness.data(), f.blocks, f.blocks.width})
               : PrepareAqEvaluation(gpu, f.Preparation(1), &prepared);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (!Check(status.resource_plan_exceeded() && pending &&
                     gpu.stats().committed_submissions ==
                         before.committed_submissions &&
                     gpu.stats().successful_allocations ==
                         before.successful_allocations &&
                     budget.snapshot().total.pending_count == 0,
                 "AQ typed underplan escaped or submitted work"))
        return false;
      prepared.reset();
      if (!Ok(gpu.TrimPreparationCache()))
        return false;
    }
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  std::cout << "AQ preparation physical failure positions: " << failures
            << "; typed underplans: 2\n";
  return true;
}
bool CheckExactGroupBoundary(GpuBackend &gpu, bool only_underplan = false) {
  Fixture f({89, 57});
  Envelope e;
  if (!f.Init() || !e.Init(f, 0))
    return false;
  for (const bool underplan : {false, true}) {
    if (only_underplan && !underplan)
      continue;
    if (!Ok(gpu.TrimPreparationCache()))
      return false;
    ResourceBudget budget(e.capacity);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(e.capacity, &job)))
      return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      std::unique_ptr<PreparedAqEvaluation> prepared;
      if (!Ok(PrepareAqEvaluation(gpu, f.Preparation(0), &prepared)) ||
          !Ok(prepared->SetInvariantColorCorrelation(
              {f.cfl.data(), f.tiles, f.tiles.width},
              {f.cfl.data(), f.tiles, f.tiles.width})))
        return false;
      VarDctEncoderFrame oracle, result;
      double score = 0;
      AqEvaluationOutput::Final final{.frame = &oracle};
      AqEvaluationOutput output{
          .block_distance_map = {f.map.data(), f.blocks, f.blocks.width},
          .score = &score,
          .final = &final};
      if (!Ok(prepared->Evaluate(f.Input(), output)))
        return false;
      auto input = f.Input();
      input.exact_coefficients = &oracle;
      input.raw_quant_field = oracle.raw_quant_field();
      input.quantizer = oracle.quantizer().params();
      final.frame = &result;
      // Populate all fixed-size exact staging first. The next host allocation
      // is the fresh per-group offset array inside UploadInput, not C/DC data.
      if (!Ok(prepared->Evaluate(input, output)))
        return false;
      if (underplan &&
          !Ok(job.ReduceCapacity(budget.snapshot().total.live_capacity_bytes)))
        return false;
      score = -77;
      std::fill(f.map.begin(), f.map.end(), -77.0f);
      const auto before = gpu.stats();
      ArmManagedHostClassAllocationFailureAfterForTest(
          ResourceClass::kPreparation, 0);
      Status status;
      bool escaped = false;
      try {
        status = prepared->Evaluate(input, output);
      } catch (const ManagedAllocationFailure &) {
        std::cerr << "Exact group-offset underplan escaped the Status API\n";
        escaped = true;
      } catch (const std::bad_alloc &) {
        std::cerr << "Exact group-offset physical OOM escaped the Status API\n";
        escaped = true;
      }
      const bool hook_pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (!Check(!escaped && status.code() == StatusCode::kOutOfMemory &&
                     status.resource_plan_exceeded() == underplan &&
                     hook_pending == underplan && score == -77 &&
                     std::ranges::all_of(f.map,
                                         [](float v) { return v == -77.0f; }) &&
                     Equal(BorrowFrame(oracle), BorrowFrame(result)) &&
                     gpu.stats().committed_submissions ==
                         before.committed_submissions &&
                     budget.snapshot().total.pending_count == 0,
                 "Exact group-offset failure was not typed, atomic, or "
                 "pre-submission"))
        return false;
      prepared.reset();
      oracle = {};
      result = {};
      if (!Ok(gpu.TrimPreparationCache()))
        return false;
    }
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  std::cout << "Exact group-offset physical OOM and typed underplan stay "
               "inside the Status API\n";
  return true;
}

bool CheckOperationFailures(GpuBackend &gpu) {
  enum Phase {
    kReconfigure,
    kCfl,
    kMask,
    kAdjustment,
    kExact,
    kRgb,
    kCompleted
  };
  constexpr std::array<const char *, 7> names{
      "reconfigure",   "invariant CfL", "lazy mask",         "field adjustment",
      "exact staging", "RGB readback",  "completed snapshot"};
  Fixture f({89, 57});
  if (!f.Init())
    return false;
  for (size_t phase = 0; phase < names.size(); ++phase) {
    const size_t mode =
        phase == kMask ? 2 : (phase == kExact || phase == kRgb ? 0 : 1);
    Envelope e;
    if (!e.Init(f, mode))
      return false;
    // Preserve a previously exported lease while preparing its replacement.
    const size_t capacity = e.capacity + e.completed_device.capacity_bytes +
                            e.completed_host.working.peak_bytes;
    size_t failures = 0;
    for (size_t position = 0; position < 256; ++position) {
      if (!Ok(gpu.TrimPreparationCache()))
        return false;
      ResourceBudget budget(capacity);
      ResourceReservation job;
      if (!Ok(budget.TryReserve(capacity, &job)))
        return false;
      bool injected = false;
      {
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        std::unique_ptr<PreparedAqEvaluation> prepared;
        VarDctEncoderFrame oracle, result;
        std::unique_ptr<CompletedVarDctFrame> completed;
        std::vector<double> scores;
        double score = -77;
        AqResidentButteraugliPolicyInput policy;
        const auto prime = [&]() -> bool {
          std::fill(f.field.begin(), f.field.end(), 0.8f);
          if (!Ok(PrepareAqEvaluation(gpu, f.Preparation(mode), &prepared)) ||
              !Ok(prepared->Reconfigure(
                  f.grid, {f.sharpness.data(), f.blocks, f.blocks.width})) ||
              !Ok(prepared->SetInvariantColorCorrelation(
                  {f.cfl.data(), f.tiles, f.tiles.width},
                  {f.cfl.data(), f.tiles, f.tiles.width})))
            return false;
          if (phase == kMask &&
              !Ok(prepared->ComputeInitialQuantization(
                  {}, {{f.field.data(), f.blocks, f.blocks.width},
                       {f.mask.data(), f.blocks, f.blocks.width},
                       {}})))
            return false;
          if (phase == kExact || phase == kRgb) {
            AqEvaluationOutput::Final final{.frame = &oracle};
            if (!Ok(prepared->Evaluate(
                    f.Input(), {.block_distance_map = {f.map.data(), f.blocks,
                                                       f.blocks.width},
                                .score = &score,
                                .final = &final})))
              return false;
          }
          if (phase == kCompleted) {
            adaptive_quantization_internal::ButteraugliPolicySetup setup;
            if (!Ok(adaptive_quantization_internal::PrepareButteraugliPolicy(
                    {f.field.data(), f.blocks, f.blocks.width}, 1.0f, &setup)))
              return false;
            policy = {.adjusted_initial_quant_field = {f.field.data(), f.blocks,
                                                       f.blocks.width},
                      .quant_dc = setup.quant_dc,
                      .butteraugli_target = 1.0f,
                      .lower_bound = setup.lower_bound,
                      .upper_bound = setup.upper_bound,
                      .iterations = 0,
                      .evaluate_final_field = true};
            if (!Ok(prepared->EvaluateResidentButteraugliPolicy(
                    policy, {.score_history = &scores, .frame = &oracle})) ||
                !Ok(prepared->EvaluateResidentButteraugliPolicy(
                    policy,
                    {.score_history = &scores, .completed_frame = &completed})))
              return false;
          }
          score = -77;
          scores = {42};
          std::fill(f.map.begin(), f.map.end(), -77.0f);
          std::fill(f.adjusted.begin(), f.adjusted.end(), -77.0f);
          for (auto &plane : f.diagnostic.view().plane)
            for (size_t y = 0; y < f.source_extent.height; ++y)
              std::fill_n(plane.Row(y), f.source_extent.width, -77.0f);
          return true;
        };
        const auto operation = [&]() -> Status {
          if (phase == kReconfigure)
            return prepared->Reconfigure(
                f.mixed, {f.sharpness.data(), f.blocks, f.blocks.width});
          if (phase == kCfl)
            return prepared->SetInvariantColorCorrelation(
                {f.cfl.data(), f.tiles, f.tiles.width},
                {f.cfl.data(), f.tiles, f.tiles.width});
          if (phase == kMask)
            return prepared->ComputeInitialQuantization(
                {}, {{f.field.data(), f.blocks, f.blocks.width},
                     {f.mask.data(), f.blocks, f.blocks.width},
                     {f.pixel_mask.data(), f.coding, f.coding.width}});
          if (phase == kAdjustment)
            return prepared->AdjustQuantFieldResident(
                1.0f, {f.field.data(), f.blocks, f.blocks.width},
                {f.adjusted.data(), f.blocks, f.blocks.width});
          if (phase == kCompleted)
            return prepared->EvaluateResidentButteraugliPolicy(
                policy,
                {.score_history = &scores, .completed_frame = &completed});
          auto input = f.Input();
          if (phase == kExact) {
            input.exact_coefficients = &oracle;
            input.raw_quant_field = oracle.raw_quant_field();
            input.quantizer = oracle.quantizer().params();
          }
          AqEvaluationOutput::Final final{
              .reconstructed_linear_rgb =
                  phase == kRgb ? f.diagnostic.view() : Image3FView{},
              .frame = &result};
          return prepared->Evaluate(
              input,
              {.block_distance_map = {f.map.data(), f.blocks, f.blocks.width},
               .score = &score,
               .final = &final});
        };
        const auto clear = [&]() -> bool {
          prepared.reset();
          oracle = {};
          result = {};
          completed.reset();
          return Ok(gpu.TrimPreparationCache());
        };
        if (!prime())
          return false;
        auto *old_completed = completed.get();
        const auto old_field = f.field, old_mask = f.mask,
                   old_pixel_mask = f.pixel_mask;
        ArmManagedHostClassAllocationFailureAfterForTest(
            phase == kCompleted ? ResourceClass::kCompletedFrame
                                : ResourceClass::kPreparation,
            position);
        const Status status = operation();
        injected = !ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (injected) {
          ++failures;
          bool rgb_unchanged = true;
          for (const auto &plane : f.diagnostic.const_view().plane)
            for (size_t y = 0; y < f.source_extent.height; ++y)
              for (size_t x = 0; x < f.source_extent.width; ++x)
                rgb_unchanged &= plane.Row(y)[x] == -77.0f;
          if (!Check(status.code() == StatusCode::kOutOfMemory &&
                         !status.resource_plan_exceeded() && !result.valid() &&
                         completed.get() == old_completed && score == -77 &&
                         scores == std::vector<double>{42} &&
                         std::ranges::all_of(
                             f.map, [](float v) { return v == -77.0f; }) &&
                         std::ranges::all_of(
                             f.adjusted, [](float v) { return v == -77.0f; }) &&
                         rgb_unchanged &&
                         (phase != kMask ||
                          (f.field == old_field && f.mask == old_mask &&
                           f.pixel_mask == old_pixel_mask)) &&
                         (phase != kCompleted ||
                          Equal(BorrowFrame(oracle), completed->view())) &&
                         budget.snapshot().total.pending_count == 0,
                     "AQ operation physical failure changed output or leaked "
                     "pending storage")) {
            std::cerr << names[phase] << " allocation " << position << ": "
                      << status.message() << '\n';
            return false;
          }
          // Invalidation is terminal. Recovery creates a fresh evaluator with
          // the same backend and original envelope, without reservation growth.
          if (!clear() || !prime() || !Ok(operation()))
            return false;
        } else if (!Ok(status))
          return false;
        if (!clear())
          return false;
      }
      job.Reset();
      if (!Empty(budget))
        return false;
      if (!injected)
        break;
    }
    constexpr std::array<size_t, 7> minimum{10, 2, 1, 1, 4, 3, 8};
    if (!Check(failures >= minimum[phase] && failures < 256,
               "AQ operation failure enumeration was incomplete"))
      return false;
    std::cout << "AQ " << names[phase]
              << " physical failures and recoveries: " << failures << '\n';
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--exact-group-boundary" ||
                    std::string_view(argv[1]) == "--exact-group-underplan")) {
    std::unique_ptr<GpuBackend> gpu;
    return Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)) &&
                   CheckExactGroupBoundary(*gpu, std::string_view(argv[1]) ==
                                                     "--exact-group-underplan")
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
  }
  if (argc != 1)
    return EXIT_FAILURE;
  if (!CheckPlans())
    return EXIT_FAILURE;
  std::unique_ptr<GpuBackend> gpu;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)) ||
      !CheckRuntime(*gpu) || !CheckPrepareFailures(*gpu) ||
      !CheckOperationFailures(*gpu) || !CheckExactGroupBoundary(*gpu))
    return EXIT_FAILURE;
  gpu.reset();
  return Empty(DefaultResourceBudget()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
