// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "codec/chroma_from_luma_internal.h"
#include "codec/prepared_coefficients_internal.h"
#include "codec/quantization.h"
#include "codec/reconstruction.h"
#include "core/image_buffer.h"
#include "core/managed_allocator.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::prepared_coefficients_internal;

bool Check(bool good, const char* message) {
  if (!good) std::cerr << message << '\n';
  return good;
}
bool Ok(const Status& status) {
  if (!status.ok()) std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget& budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
    s.total.pending_count == 0 && s.open_reservations == 0,
    "Frontend storage charge leaked");
}

struct Fixture {
  Image3FBuffer opsin{{24, 16}};
  FrameGeometry geometry;
  AcStrategyGrid strategies;
  ColorCorrelationMap color;
  Quantizer quantizer;
  std::array<int32_t, 6> raw{29, 29, 29, 29, 29, 29};
  std::array<uint8_t, 6> sharpness{};
  bool Init() {
    for (size_t c = 0; c < 3; ++c)
      for (size_t i = 0; i < opsin.plane(c).size(); ++i)
        opsin.plane(c)[i] = 0.05f * c + 0.001f * (i % 57);
    const int8_t zero = 0;
    if (!Ok(FrameGeometry::Create({17, 9}, &geometry)) ||
        !Ok(AcStrategyGrid::Create({3, 2}, &strategies)) ||
        !Ok(Quantizer::Create({3541, 10}, &quantizer)) ||
        !Ok(chroma_from_luma_internal::CreateColorCorrelationMap(
          {&zero, {1, 1}, 1}, {&zero, {1, 1}, 1}, &color))) return false;
    strategies.fill_dct8();
    return true;
  }
  Status Frame(VarDctEncoderFrame* out) const {
    return ComputeQuantizedCoefficients(opsin.const_view(), {
      .geometry = geometry, .strategies = &strategies,
      .raw_quant_field = {raw.data(), {3, 2}, 3}, .quantizer = &quantizer,
      .color_correlation = &color,
      .epf_sharpness = {sharpness.data(), {3, 2}, 3}}, {}, out);
  }
};

bool CheckFrameOwnership(const Fixture& fixture) {
  ResourceBudget budget(2 * 1024 * 1024);
  ResourceReservation job;
  VarDctEncoderFrame retained;
  if (!Ok(budget.Reserve(2 * 1024 * 1024, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    if (!Ok(fixture.Frame(&retained))) return false;
  }
  // Fresh owning frame: strategy/raw/sharpness, three DC pairs, two CfL
  // planes, one group count, and one three-channel group-major AC allocation.
  constexpr size_t bytes = 6 * (1 + 4 + 1 + 3 * (4 + 4)) + 2 + sizeof(size_t) +
    3 * kVarDctAcGroupCoefficientCapacity * sizeof(int32_t);
  const auto s = budget.snapshot();
  const auto& owned = s.classes[static_cast<size_t>(ResourceClass::kCompletedFrame)];
  if (!Check(s.total.live_capacity_bytes == bytes && owned.live_capacity_bytes == bytes &&
      s.total.backing_count == 13 && s.total.pending_count == 0,
      "Completed-frame backing or owner class differs")) return false;
  const auto* original = retained.raw_quant_field().data;
  VarDctEncoderFrame moved = std::move(retained);
  job.Reset();
  if (!Check(moved.raw_quant_field().data == original &&
      budget.snapshot().committed_bytes() == bytes,
      "Frame move/closed producer lost its backing ownership")) return false;
  std::thread consumer([frame = std::move(moved)] {
    if (!frame.valid()) std::abort();
  });
  consumer.join();
  return Empty(budget);
}

bool CheckPreparedOwnership(const Fixture& fixture) {
  ResourceBudget budget(1024 * 1024);
  ResourceReservation job;
  if (!Ok(budget.Reserve(1024 * 1024, &job))) return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    PreparedForwardDctCoefficients prepared;
    if (!Ok(PrepareForwardDctCoefficients(fixture.opsin.const_view(),
        fixture.strategies, &prepared))) return false;
    size_t bytes = 0, count = 0;
    const auto add = [&](const auto& values) {
      bytes += values.capacity() * sizeof(values[0]);
      count += values.capacity() != 0;
    };
    add(prepared.transforms); add(prepared.color_tile_offsets);
    add(prepared.color_tile_transform_indices);
    for (const auto& plane : prepared.coefficients) add(plane);
    const auto s = budget.snapshot();
    if (!Check(prepared.valid() && s.total.live_capacity_bytes == bytes &&
        s.total.backing_count == count && s.total.pending_count == 0 &&
        s.classes[static_cast<size_t>(ResourceClass::kPreparation)].live_capacity_bytes == bytes,
        "Prepared coefficients omitted backing or retained tile scratch")) return false;
  }
  job.Reset();
  return Empty(budget);
}

bool CheckFailures(const Fixture& fixture) {
  for (bool prepare : {false, true}) {
    size_t fail_at = 0;
    for (; fail_at < 128; ++fail_at) {
      ResourceBudget budget(2 * 1024 * 1024);
      ResourceReservation job;
      if (!Ok(budget.Reserve(2 * 1024 * 1024, &job))) return false;
      bool injected;
      {
        VarDctEncoderFrame frame;
        PreparedForwardDctCoefficients coefficients;
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        ArmManagedHostAllocationFailureAfterForTest(fail_at);
        const Status status = prepare
          ? PrepareForwardDctCoefficients(fixture.opsin.const_view(), fixture.strategies, &coefficients)
          : fixture.Frame(&frame);
        injected = !ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (injected) {
          if (!Check(status.code() == StatusCode::kOutOfMemory && !frame.valid() &&
              !coefficients.valid() && budget.snapshot().total.backing_count == 0 &&
              budget.snapshot().total.pending_count == 0, "Frontend failure was not atomic")) return false;
        } else if (!Ok(status) || !Check(prepare ? coefficients.valid() : frame.valid(),
                    "Frontend did not recover")) return false;
      }
      job.Reset();
      if (!Empty(budget)) return false;
      if (!injected) break;
    }
    if (!Check(fail_at > 0 && fail_at < 128, "Incomplete frontend failure sweep")) return false;
    std::cout << (prepare ? "Prepared" : "Frame") << " failure positions: " << fail_at << '\n';
  }
  // Exact input-sized allowance cannot hold even the frame's AC allocation.
  ResourceBudget tiny(6);
  ResourceReservation job;
  VarDctEncoderFrame frame;
  if (!Ok(fixture.Frame(&frame)) || !Ok(tiny.Reserve(6, &job))) return false;
  const auto* sentinel = frame.raw_quant_field().data;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    const Status status = fixture.Frame(&frame);
    if (!Check(status.code() == StatusCode::kOutOfMemory &&
        status.message() == "Allocation exceeds admitted resource plan" &&
        frame.raw_quant_field().data == sentinel, "Underplan lost precise status or previous frame")) return false;
  }
  job.Reset();
  return Empty(tiny);
}

bool CheckQuantizerFailures() {
  const std::array<float, 6> field{0.5f, 1.0f, 0.7f, 1.1f, 0.8f, 0.6f};
  for (size_t fail_at = 0; fail_at < 3; ++fail_at) {
    ResourceBudget budget(1024);
    ResourceReservation job;
    if (!Ok(budget.Reserve(1024, &job))) return false;
    std::array<int32_t, 6> raw{-7, -7, -7, -7, -7, -7};
    const auto original = raw;
    Quantizer quantizer;
    if (!Ok(Quantizer::Create({3541, 10}, &quantizer))) return false;
    const auto params = quantizer.params();
    {
      ResourceContextScope context({&job, ResourceClass::kPreparation});
      ArmManagedHostAllocationFailureAfterForTest(fail_at);
      const auto status = CreateQuantizerFromField(1.0f,
        {field.data(), {3, 2}, 3}, {raw.data(), {3, 2}, 3}, &quantizer);
      const bool injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (fail_at < 2) {
        if (!Check(injected && status.code() == StatusCode::kOutOfMemory &&
            raw == original && quantizer.params().global_scale == params.global_scale &&
            quantizer.params().quant_dc == params.quant_dc,
            "Quantizer allocation failure changed caller fields")) return false;
      } else if (!Check(!injected, "Quantizer selection added an unexpected backing") ||
                 !Ok(status)) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;
  }
  return true;
}
}  // namespace

int main() {
  Fixture fixture;
  return fixture.Init() && CheckFrameOwnership(fixture) &&
    CheckPreparedOwnership(fixture) && CheckFailures(fixture) && CheckQuantizerFailures() &&
    Check(DefaultResourceBudget().snapshot().peak_backing_bytes == 0,
          "Frontend escaped the explicit domain") ? EXIT_SUCCESS : EXIT_FAILURE;
}
