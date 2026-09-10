// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#ifndef GJXL_FRONTEND_PLAN_ORACLE_ONLY
#include "codec/frontend_storage_plan.h"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include "codec/chroma_from_luma_internal.h"
#include "codec/reconstruction_internal.h"
#include "codec/vardct_frame_internal.h"
#include "core/image_buffer.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::prepared_coefficients_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::vardct_frame_internal;

bool Check(bool value, const char *message) {
  if (!value)
    std::cerr << message << '\n';
  return value;
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
               "Frontend plan leaked a charge");
}

constexpr std::array<AcStrategyType, 7> strategies{
    AcStrategyType::kDct8,     AcStrategyType::kDct16x16,
    AcStrategyType::kDct32x32, AcStrategyType::kDct16x8,
    AcStrategyType::kDct8x16,  AcStrategyType::kDct32x16,
    AcStrategyType::kDct16x32};
struct Case {
  Extent2D pixels;
  size_t family;
};
constexpr std::array<Case, 12> cases{{{{1, 1}, 0},
                                      {{17, 9}, 0},
                                      {{64, 64}, 2},
                                      {{63, 9}, 3},
                                      {{9, 63}, 4},
                                      {{63, 33}, 5},
                                      {{33, 63}, 6},
                                      {{263, 7}, 1},
                                      {{247, 263}, 7},
                                      {{256, 256}, 0},
                                      {{263, 263}, 7},
                                      {{2055, 7}, 0}}};

struct Fixture {
  FrameGeometry geometry;
  Image3FBuffer opsin;
  AcStrategyGrid grid;
  Quantizer quantizer;
  ColorCorrelationMap color;
  SimpleVarDctCodestreamProfile profile;
  std::vector<int32_t> raw;
  std::vector<uint8_t> sharpness;
  bool Create(Case c) {
    if (!Ok(FrameGeometry::Create(c.pixels, &geometry)))
      return false;
    opsin.resize(geometry.padded_frame());
    for (size_t channel = 0; channel < 3; ++channel)
      for (size_t i = 0; i < opsin.plane(channel).size(); ++i)
        opsin.plane(channel)[i] =
            0.04f * channel +
            0.003f * static_cast<float>(
                         static_cast<int>((i * 17 + channel * 13) % 113) - 56);
    const auto blocks = geometry.block_grid().blocks;
    if (!Ok(AcStrategyGrid::Create(blocks, &grid)) ||
        !Ok(Quantizer::Create({3541, 10}, &quantizer)))
      return false;
    size_t sequence = 0;
    for (size_t y = 0; y < blocks.height; ++y) {
      for (size_t x = 0; x < blocks.width; ++x) {
        if (grid.occupied(x, y))
          continue;
        auto strategy = strategies[c.family < 7 ? c.family : sequence++ % 7];
        const auto *info = GetAcStrategyInfo(strategy);
        // Prepared transforms cannot cross a 64-pixel color tile. This also
        // guarantees the larger 256-pixel AC-group boundary is respected.
        bool fits = info->covered_blocks.width <=
                        std::min(8 - x % 8, blocks.width - x) &&
                    info->covered_blocks.height <=
                        std::min(8 - y % 8, blocks.height - y);
        for (size_t dy = 0; fits && dy < info->covered_blocks.height; ++dy)
          for (size_t dx = 0; fits && dx < info->covered_blocks.width; ++dx)
            fits = !grid.occupied(x + dx, y + dy);
        if (!fits)
          strategy = AcStrategyType::kDct8;
        if (!Ok(grid.Set(x, y, strategy)))
          return false;
      }
    }
    raw.resize(blocks.width * blocks.height);
    sharpness.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
      raw[i] = 1 + i % 256;
      sharpness[i] = i % 8;
    }
    const auto tiles = ColorTileExtent(geometry.padded_frame());
    std::vector<int8_t> x(tiles.width * tiles.height), b(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = static_cast<int8_t>(i % 9) - 4;
      b[i] = static_cast<int8_t>(i % 7) - 3;
    }
    profile.x_qm_scale = c.family % 8;
    profile.b_qm_scale = 7 - profile.x_qm_scale;
    return Ok(chroma_from_luma_internal::CreateColorCorrelationMap(
        {x.data(), tiles, tiles.width}, {b.data(), tiles, tiles.width},
        &color));
  }
  VarDctFrameInput input() const {
    const auto blocks = geometry.block_grid().blocks;
    return {.geometry = geometry,
            .strategies = &grid,
            .raw_quant_field = {raw.data(), blocks, blocks.width},
            .quantizer = &quantizer,
            .color_correlation = &color,
            .epf_sharpness = {sharpness.data(), blocks, blocks.width}};
  }
  Status Frame(AcCoefficientDecisionMode mode, VarDctEncoderFrame *out) const {
    return ComputeQuantizedCoefficients(opsin.const_view(), input(), profile,
                                        out, mode);
  }
};

struct Bytes {
  std::vector<uint8_t> values;
  void Integer(uint64_t value) {
    for (size_t i = 0; i < 8; ++i)
      values.push_back((value >> (8 * i)) & 255);
  }
  template <class T> void Range(std::span<const T> range) {
    Integer(range.size());
    for (T value : range) {
      if constexpr (std::is_same_v<T, float>) {
        const auto bits = std::bit_cast<uint32_t>(value);
        for (size_t i = 0; i < 4; ++i)
          values.push_back((bits >> (8 * i)) & 255);
      } else {
        using U = std::make_unsigned_t<T>;
        const auto bits = static_cast<U>(value);
        for (size_t i = 0; i < sizeof(T); ++i)
          values.push_back((bits >> (8 * i)) & 255);
      }
    }
  }
  template <class T> void Plane(T view) {
    Integer(view.extent.width);
    Integer(view.extent.height);
    for (size_t y = 0; y < view.extent.height; ++y)
      Range(std::span(view.Row(y), view.extent.width));
  }
};
Bytes Snapshot(const Image3FBuffer &image) {
  Bytes result;
  for (size_t c = 0; c < 3; ++c)
    result.Range(image.plane(c));
  return result;
}
Bytes Snapshot(const PreparedForwardDctCoefficients &prepared) {
  Bytes result;
  result.Integer(prepared.pixel_extent.width);
  result.Integer(prepared.pixel_extent.height);
  result.Integer(prepared.transforms.size());
  for (const auto &t : prepared.transforms) {
    result.Integer(t.block_x);
    result.Integer(t.block_y);
    result.Integer(static_cast<uint8_t>(t.strategy));
    result.Integer(t.coefficient_offset);
    result.Integer(t.coefficient_count);
  }
  result.Range(std::span<const size_t>(prepared.color_tile_offsets));
  result.Range(std::span<const size_t>(prepared.color_tile_transform_indices));
  for (const auto &p : prepared.coefficients)
    result.Range(std::span<const float>(p));
  return result;
}
Bytes Snapshot(const VarDctEncoderFrame &frame) {
  Bytes result;
  result.Integer(frame.geometry().frame().width);
  result.Integer(frame.geometry().frame().height);
  result.Integer(frame.quantizer().params().global_scale);
  result.Integer(frame.quantizer().params().quant_dc);
  result.Integer(frame.profile().x_qm_scale);
  result.Integer(frame.profile().b_qm_scale);
  result.Plane(frame.raw_quant_field());
  result.Plane(frame.epf_sharpness());
  const auto blocks = frame.strategies().extent();
  for (size_t y = 0; y < blocks.height; ++y)
    for (size_t x = 0; x < blocks.width; ++x) {
      AcStrategyCell cell;
      if (!Ok(frame.strategies().Get(x, y, &cell)))
        std::abort();
      result.Integer(static_cast<uint8_t>(cell.strategy));
      result.Integer(cell.is_anchor);
    }
  result.Plane(frame.color_correlation().y_to_x_map());
  result.Plane(frame.color_correlation().y_to_b_map());
  for (size_t c = 0; c < 3; ++c) {
    result.Plane(frame.quantized_dc().plane[c]);
    result.Plane(frame.dc().plane[c]);
  }
  result.Integer(frame.ac_group_count());
  for (size_t g = 0; g < frame.ac_group_count(); ++g) {
    VarDctAcGroupView group;
    if (!Ok(frame.GetAcGroup(g, &group)))
      std::abort();
    result.Integer(group.used_coefficient_count);
    for (auto plane : group.coefficients)
      result.Range(plane);
  }
  return result;
}
void Emit(const Bytes &bytes) {
  Bytes prefix;
  prefix.Integer(bytes.values.size());
  std::cout.write(reinterpret_cast<const char *>(prefix.values.data()),
                  prefix.values.size());
  std::cout.write(reinterpret_cast<const char *>(bytes.values.data()),
                  bytes.values.size());
}
bool Oracle() {
  thread_budget_internal::EncodeScope serial(1);
  for (auto c : cases) {
    Fixture f;
    PreparedForwardDctCoefficients prepared;
    if (!f.Create(c) || !Ok(PrepareForwardDctCoefficients(f.opsin.const_view(),
                                                          f.grid, &prepared)))
      return false;
    Emit(Snapshot(prepared));
    for (auto mode : {AcCoefficientDecisionMode::kAdjustedSharedQuant,
                      AcCoefficientDecisionMode::kFixedRawQuant}) {
      VarDctEncoderFrame frame;
      Image3FBuffer reconstructed(f.geometry.padded_frame());
      if (!Ok(f.Frame(mode, &frame)) ||
          !Ok(ReconstructQuantizedCoefficients(frame, reconstructed.view())))
        return false;
      Emit(Snapshot(frame));
      Emit(Snapshot(reconstructed));
    }
  }
  return std::cout.good() && Empty(DefaultResourceBudget());
}

#ifndef GJXL_FRONTEND_PLAN_ORACLE_ONLY
using namespace gjxl::frontend_storage_internal;

struct Assembly {
  std::vector<int32_t> coefficients;
  std::vector<QuantizedAcTransformLayout> transforms;
  QuantizedFrameAssemblyInput input;
  bool Create(const VarDctEncoderFrame &frame) {
    constexpr size_t capacity = kVarDctAcGroupCoefficientCapacity;
    coefficients.reserve(frame.ac_group_count() * 3 * capacity);
    for (size_t g = 0; g < frame.ac_group_count(); ++g) {
      VarDctAcGroupView group;
      if (!Ok(frame.GetAcGroup(g, &group)))
        return false;
      for (auto plane : group.coefficients)
        coefficients.insert(coefficients.end(), plane.begin(), plane.end());
    }
    std::vector<size_t> offsets(frame.ac_group_count());
    if (!Ok(frame.strategies().ForEachAnchor([&](size_t x, size_t y,
                                                 AcStrategyType strategy) {
          const size_t group =
              (y / 32) * frame.ac_group_extent().width + x / 32;
          const size_t count = GetAcStrategyInfo(strategy)->coefficient_count();
          QuantizedAcTransformLayout t{x, y, strategy, count, {}};
          for (size_t c = 0; c < 3; ++c)
            t.coefficient_offsets[c] =
                (group * 3 + c) * capacity + offsets[group];
          transforms.push_back(t);
          offsets[group] += count;
          return Status::Ok();
        })))
      return false;
    input = {.geometry = frame.geometry(),
             .strategies = &frame.strategies(),
             .raw_quant_field = frame.raw_quant_field(),
             .quantizer = &frame.quantizer(),
             .y_to_x = frame.color_correlation().y_to_x_map(),
             .y_to_b = frame.color_correlation().y_to_b_map(),
             .epf_sharpness = frame.epf_sharpness(),
             .profile = frame.profile(),
             .quantized_dc = frame.quantized_dc(),
             .quantized_ac = coefficients,
             .transforms = transforms};
    return true;
  }
};

bool PurePlans() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.Reserve(1, &job)))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    ArmNextManagedHostAllocationFailureForTest();
    for (auto pixels :
         {Extent2D{1, 1}, {257, 259}, {3839, 2159}, {1ul << 24, 1}}) {
      FrameGeometry geometry;
      OwnedFrameStoragePlan frame;
      HostStorageBound image, reconstruction;
      if (!Ok(FrameGeometry::Create(pixels, &geometry)) ||
          !Ok(ComputeOwnedFrameStoragePlan(pixels, &frame)) ||
          !Ok(ComputeImage3FStorageBound(pixels, &image)) ||
          !Ok(ComputeCoefficientReconstructionStorageBound(pixels,
                                                           &reconstruction)))
        return false;
      const size_t b = geometry.block_grid().blocks.width *
                       geometry.block_grid().blocks.height;
      const size_t expected =
          30 * b + 2 * frame.color_tiles + frame.ac_groups * sizeof(size_t) +
          3 * frame.ac_groups * kVarDctAcGroupCoefficientCapacity *
              sizeof(int32_t);
      if (!Check(frame.blocks == b && frame.output.retained_bytes == expected &&
                     frame.output.peak_bytes == expected &&
                     image.peak_bytes == pixels.width * pixels.height * 12,
                 "Frontend representation counts differ"))
        return false;
      size_t previous = 0;
      for (size_t workers : {1ul, 2ul, 8ul, SIZE_MAX, 0ul}) {
        PreparedForwardStoragePlan plan;
        if (!Ok(ComputePreparedForwardStoragePlan(geometry.padded_frame(),
                                                  workers, &plan)) ||
            !Check(plan.maximum_transforms == b &&
                       plan.maximum_participants >= 1 &&
                       plan.maximum_participants <= 8 &&
                       plan.working.peak_bytes >= previous &&
                       plan.working.peak_bytes >= plan.output.peak_bytes,
                   "Forward participation or working envelope differs"))
          return false;
        previous = plan.working.peak_bytes;
      }
    }
    if (!Check(ManagedHostAllocationFailurePendingForTest() &&
                   budget.snapshot().peak_backing_bytes == 0,
               "Frontend planning allocated backing"))
      return false;
    DisarmManagedHostAllocationFailureForTest();
  }
  job.Reset();
  OwnedFrameStoragePlan frame;
  frame.blocks = 17;
  PreparedForwardStoragePlan forward;
  forward.maximum_transforms = 19;
  HostStorageBound bound{23, 29};
  const auto initial_frame = frame;
  const auto initial_forward = forward;
  const auto initial_bound = bound;
  for (Extent2D extent :
       {Extent2D{}, {1, 0}, {SIZE_MAX, 8}, {SIZE_MAX - 7, 8}}) {
    if (!Check(
            !ComputeOwnedFrameStoragePlan(extent, &frame).ok() &&
                frame == initial_frame &&
                !ComputePreparedForwardStoragePlan(extent, 0, &forward).ok() &&
                forward == initial_forward &&
                !ComputeImage3FStorageBound(extent, &bound).ok() &&
                bound == initial_bound &&
                !ComputeCoefficientReconstructionStorageBound(extent, &bound)
                     .ok() &&
                bound == initial_bound,
            "Invalid or overflowing frontend plan changed output"))
      return false;
  }
  return Check(
             !ComputePreparedForwardStoragePlan({9, 8}, 0, &forward).ok() &&
                 forward == initial_forward &&
                 !ComputeImage3FStorageBound({1, 1}, nullptr).ok() &&
                 !ComputeOwnedFrameStoragePlan({1, 1}, nullptr).ok() &&
                 !ComputePreparedForwardStoragePlan({8, 8}, 0, nullptr).ok() &&
                 !ComputeCoefficientReconstructionStorageBound({1, 1}, nullptr)
                      .ok(),
             "Frontend planner accepted null output or unpadded "
             "coefficients") &&
         Empty(budget);
}

template <class Owner, class Operation>
bool Within(HostStorageBound working, HostStorageBound retained,
            ResourceClass resource_class, const Bytes &expected,
            size_t backing_count, Operation operation) {
  ResourceBudget budget(working.peak_bytes);
  ResourceReservation job;
  if (!Ok(budget.Reserve(working.peak_bytes, &job)))
    return false;
  {
    Owner owner;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      if (!Ok(operation(&owner)))
        return false;
    }
    const auto s = budget.snapshot();
    if (!Check(Snapshot(owner).values == expected.values &&
                   s.peak_backing_bytes <= working.peak_bytes &&
                   s.total.live_capacity_bytes <= retained.retained_bytes &&
                   s.classes[static_cast<size_t>(resource_class)]
                           .live_capacity_bytes ==
                       s.total.live_capacity_bytes &&
                   s.total.backing_count == backing_count &&
                   s.total.pending_count == 0,
               "Frontend operation exceeded its plan or changed output"))
      return false;
    if constexpr (std::is_same_v<Owner, VarDctEncoderFrame>) {
      if (!Check(s.total.live_capacity_bytes == retained.retained_bytes &&
                     s.peak_backing_bytes == retained.peak_bytes,
                 "Fresh frame did not match exact backing recipe"))
        return false;
    }
    job.Reset();
    if (!Check(budget.snapshot().total.live_capacity_bytes ==
                   s.total.live_capacity_bytes,
               "Closing producer uncharged retained frontend output"))
      return false;
  }
  return Empty(budget);
}

bool RuntimeCases() {
  size_t frames = 0, forwards = 0, reconstructions = 0, images = 0;
  for (auto c : cases) {
    Fixture f;
    PreparedForwardDctCoefficients prepared;
    if (!f.Create(c))
      return false;
    {
      thread_budget_internal::EncodeScope serial(1);
      if (!Ok(PrepareForwardDctCoefficients(f.opsin.const_view(), f.grid,
                                            &prepared)))
        return false;
    }
    const auto expected_prepared = Snapshot(prepared);
    for (size_t threads : {0ul, 1ul, 2ul, 8ul}) {
      PreparedForwardStoragePlan plan;
      if (!Ok(ComputePreparedForwardStoragePlan(f.opsin.extent(), threads,
                                                &plan)))
        return false;
      thread_budget_internal::EncodeScope scope(threads);
      if (!Within<PreparedForwardDctCoefficients>(
              plan.working, plan.output, ResourceClass::kPreparation,
              expected_prepared, 6, [&](auto *out) {
                return PrepareForwardDctCoefficients(f.opsin.const_view(),
                                                     f.grid, out);
              }))
        return false;
      ++forwards;
    }
    HostStorageBound image_plan;
    if (!Ok(ComputeImage3FStorageBound(f.opsin.extent(), &image_plan)))
      return false;
    Image3FBuffer zero(f.opsin.extent());
    if (!Within<Image3FBuffer>(image_plan, image_plan,
                               ResourceClass::kPreparation, Snapshot(zero), 3,
                               [&](auto *out) {
                                 out->resize(f.opsin.extent());
                                 return Status::Ok();
                               }))
      return false;
    ++images;
    for (auto mode : {AcCoefficientDecisionMode::kAdjustedSharedQuant,
                      AcCoefficientDecisionMode::kFixedRawQuant}) {
      VarDctEncoderFrame oracle;
      if (!Ok(f.Frame(mode, &oracle)))
        return false;
      const auto expected = Snapshot(oracle);
      OwnedFrameStoragePlan plan;
      if (!Ok(ComputeOwnedFrameStoragePlan(c.pixels, &plan)))
        return false;
      Assembly assembly;
      if (!assembly.Create(oracle))
        return false;
      for (int method = 0; method < 3; ++method) {
        if (!Within<VarDctEncoderFrame>(
                plan.output, plan.output, ResourceClass::kCompletedFrame,
                expected, 13, [&](auto *out) {
                  if (method == 0)
                    return f.Frame(mode, out);
                  if (method == 1)
                    return ComputeQuantizedCoefficientsPrepared(
                        prepared, f.input(), f.profile, out, mode);
                  return AssembleVarDctEncoderFrame(assembly.input, out);
                }))
          return false;
        ++frames;
      }
      Image3FBuffer reference(f.opsin.extent()), actual(f.opsin.extent());
      if (!Ok(ReconstructQuantizedCoefficients(oracle, reference.view())))
        return false;
      HostStorageBound scratch;
      if (!Ok(ComputeCoefficientReconstructionStorageBound(c.pixels, &scratch)))
        return false;
      ResourceBudget budget(scratch.peak_bytes);
      ResourceReservation job;
      if (!Ok(budget.Reserve(scratch.peak_bytes, &job)))
        return false;
      {
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        if (!Ok(ReconstructQuantizedCoefficients(oracle, actual.view())))
          return false;
      }
      if (!Check(Snapshot(actual).values == Snapshot(reference).values &&
                     budget.snapshot().peak_backing_bytes <=
                         scratch.peak_bytes &&
                     budget.snapshot().total.backing_count == 0,
                 "Reconstruction plan or exact output differs"))
        return false;
      job.Reset();
      if (!Empty(budget))
        return false;
      ++reconstructions;
    }
  }
  std::cerr << "Frontend reservation cases: " << images << " images, "
            << forwards << " prepared transforms, " << frames << " frames, "
            << reconstructions << " reconstructions\n";
  return true;
}

bool Faults() {
  Fixture f;
  const auto mode = AcCoefficientDecisionMode::kAdjustedSharedQuant;
  VarDctEncoderFrame source;
  if (!f.Create({{17, 9}, 0}) || !Ok(f.Frame(mode, &source)))
    return false;
  Assembly assembly;
  if (!assembly.Create(source))
    return false;
  HostStorageBound image_bound, reconstruction_bound;
  OwnedFrameStoragePlan frame_plan;
  PreparedForwardStoragePlan prepared_plan;
  if (!Ok(ComputeImage3FStorageBound(f.opsin.extent(), &image_bound)) ||
      !Ok(ComputeOwnedFrameStoragePlan(f.geometry.frame(), &frame_plan)) ||
      !Ok(ComputePreparedForwardStoragePlan(f.opsin.extent(), 1,
                                            &prepared_plan)) ||
      !Ok(ComputeCoefficientReconstructionStorageBound(f.geometry.frame(),
                                                       &reconstruction_bound)))
    return false;
  const std::array<HostStorageBound, 5> bounds{
      image_bound, prepared_plan.working, frame_plan.output,
      reconstruction_bound, frame_plan.output};
  thread_budget_internal::EncodeScope serial(1);
  for (size_t operation = 0; operation < bounds.size(); ++operation) {
    bool complete = false, underplan_verified = false;
    // After exhaustive physical failures, test a zero-credit underplan too.
    for (size_t failure = 0; failure < 2048; ++failure) {
      const bool underplan = complete;
      ResourceBudget budget(bounds[operation].peak_bytes);
      ResourceReservation job;
      if (!Ok(budget.Reserve(bounds[operation].peak_bytes, &job)) ||
          (underplan && !Ok(job.ReduceCapacity(0))))
        return false;
      bool injected = false;
      {
        Image3FBuffer image(f.opsin.extent());
        for (size_t c = 0; c < 3; ++c)
          std::ranges::fill(image.plane(c), -17.0f);
        PreparedForwardDctCoefficients prepared;
        VarDctEncoderFrame frame;
        if (!Ok(PrepareForwardDctCoefficients(f.opsin.const_view(), f.grid,
                                              &prepared)) ||
            !Ok(f.Frame(mode, &frame)))
          return false;
        const auto before_image = Snapshot(image);
        const auto before_prepared = Snapshot(prepared);
        const auto before_frame = Snapshot(frame);
        {
          ResourceContextScope scope({&job, ResourceClass::kPreparation});
          ArmManagedHostAllocationFailureAfterForTest(underplan ? 0 : failure);
          Status status;
          try {
            if (operation == 0) {
              image.resize(f.opsin.extent());
              status = Status::Ok();
            } else if (operation == 1)
              status = PrepareForwardDctCoefficients(f.opsin.const_view(),
                                                     f.grid, &prepared);
            else if (operation == 2)
              status = f.Frame(mode, &frame);
            else if (operation == 3)
              status = ReconstructQuantizedCoefficients(source, image.view());
            else
              status = AssembleVarDctEncoderFrame(assembly.input, &frame);
          } catch (const ManagedAllocationFailure &error) {
            status = error.status();
          } catch (const std::bad_alloc &) {
            // Image3FBuffer throws allocation failures. The adjacent tests use
            // Status-returning frame, preparation and reconstruction APIs.
            status = Status::OutOfMemory("Image backing allocation failed");
          }
          injected = !ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          if (underplan || injected) {
            if (!Check(
                    status.code() == StatusCode::kOutOfMemory &&
                        status.resource_plan_exceeded() == underplan &&
                        injected != underplan &&
                        Snapshot(image).values == before_image.values &&
                        Snapshot(prepared).values == before_prepared.values &&
                        Snapshot(frame).values == before_frame.values &&
                        budget.snapshot().total.backing_count == 0 &&
                        budget.snapshot().total.pending_count == 0,
                    "Frontend fault changed output, escaped a plan, or leaked"))
              return false;
          } else if (!Ok(status))
            return false;
        }
      }
      job.Reset();
      if (!Empty(budget))
        return false;
      if (underplan) {
        underplan_verified = true;
        break;
      }
      if (!injected) {
        complete = true;
        std::cerr << "Frontend allocation failure positions " << operation
                  << ": " << failure << '\n';
      }
    }
    if (!Check(complete && underplan_verified,
               "Frontend fault sweep or underplan check did not finish"))
      return false;
  }
  return true;
}
#endif
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--oracle")
    return Oracle() ? EXIT_SUCCESS : EXIT_FAILURE;
#ifndef GJXL_FRONTEND_PLAN_ORACLE_ONLY
  return PurePlans() && RuntimeCases() && Faults() &&
                 Empty(DefaultResourceBudget()) &&
                 Check(DefaultResourceBudget().snapshot().peak_backing_bytes ==
                           0,
                       "Frontend test escaped into the default domain")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
#else
  return EXIT_FAILURE;
#endif
}
