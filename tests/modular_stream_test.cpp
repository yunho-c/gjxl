// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <algorithm>
#include <array>
#include <bit>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "codec/modular/prediction.h"
#include "codec/vardct/dc_prediction_internal.h"
#include "codestream/dc_group.h"
#include "codestream/modular/stream_encoder.h"
#include "codestream/modular/tree_codec.h"

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;

void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Ok(const Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
uint64_t Bits(const BitWriter& writer, size_t first, size_t count) {
  uint64_t value = 0;
  Check(first + count <= writer.bits_written(), "bit range exceeds stream");
  for (size_t i = 0; i < count; ++i)
    value |= uint64_t{(writer.padded_bytes()[(first + i) / 8] >>
                      ((first + i) % 8)) & 1u} << i;
  return value;
}
void CheckSentinel(const BitWriter& writer) {
  Check(writer.bits_written() == 3 && writer.padded_bytes()[0] == 5,
        "failure changed the destination");
}

void Headers() {
  // Check syntax at every starting bit alignment, including both adapter
  // prefixes. These are wire values, independent of the extracted writer.
  for (size_t offset = 0; offset < 8; ++offset) {
    for (uint8_t precision = 0; precision < 4; ++precision) {
      BitWriter dc;
      Ok(dc.WriteBits(offset, 0));
      Ok(WriteSimpleDcGroupModularHeader(&dc, precision));
      Check(dc.bits_written() == offset + 6 && Bits(dc, offset, 2) == precision &&
            Bits(dc, offset + 2, 1) == 1 && Bits(dc, offset + 3, 1) == 1 &&
            Bits(dc, offset + 4, 2) == 0, "DC prefix/suffix wire syntax changed");
    }
    for (Extent2D extent : {Extent2D{1, 1}, {3, 7}, {256, 256}}) {
      const size_t area = extent.width * extent.height;
      const size_t count_bits = std::bit_width(area - 1);
      for (size_t anchors : {size_t{1}, area}) {
        BitWriter metadata;
        Ok(metadata.WriteBits(offset, 0));
        Ok(WriteSimpleAcMetadataModularHeader(extent, anchors, &metadata));
        Check(metadata.bits_written() == offset + count_bits + 4 &&
              Bits(metadata, offset, count_bits) == anchors - 1 &&
              Bits(metadata, offset + count_bits, 4) == 3,
              "metadata anchor prefix or Modular suffix changed");
      }
    }
  }
  BitWriter writer;
  Ok(writer.WriteBits(3, 5));
  for (ModularStreamHeader unsupported : {
      ModularStreamHeader{.tree = TreeMode::kLocal},
      {.weighted = WeightedParameters::kCustom},
      {.transforms = TransformMode::kPresent},
      {.tree = static_cast<TreeMode>(255)},
      {.weighted = static_cast<WeightedParameters>(255)},
      {.transforms = static_cast<TransformMode>(255)}}) {
    Check(WriteStreamHeader(unsupported, &writer).code() ==
          StatusCode::kInvalidArgument, "unsupported header accepted");
    CheckSentinel(writer);
  }
  Check(!WriteStreamHeader({}, nullptr).ok(), "null header writer accepted");
  Check(!WriteSimpleDcGroupModularHeader(&writer, 4).ok(), "invalid precision accepted");
  Check(!WriteSimpleAcMetadataModularHeader({1, 1}, 2, &writer).ok(),
        "invalid anchor count accepted");
  CheckSentinel(writer);
  Check(!writer.WithMaxBits(3, [&] { return WriteStreamHeader({}, &writer); }).ok(),
        "header escaped enclosing allotment");
  CheckSentinel(writer);
  BitWriter failed;
  ArmNextManagedHostAllocationFailureForTest();
  const Status status = WriteStreamHeader({}, &failed);
  DisarmManagedHostAllocationFailureForTest();
  Check(status.code() == StatusCode::kOutOfMemory && failed.bits_written() == 0,
        "header allocation failure was not atomic");
  Ok(WriteStreamHeader({}, &failed));
}

template<ResourceClass Owner> void PredictorOwner() {
  constexpr size_t width = 17, bytes = 5 * (width + 2) * 2 * sizeof(uint32_t);
  constexpr auto charge = Owner == ResourceClass::kCount ? ResourceClass::kPreparation : Owner;
  constexpr auto context = Owner == ResourceClass::kCount ? ResourceClass::kPreparation : ResourceClass::kInput;
  ResourceBudget budget(bytes);
  ResourceReservation reservation;
  Ok(budget.Reserve(bytes, &reservation));
  for (size_t fail = 0; fail <= 5; ++fail) {
    ResourceContextScope scope({&reservation, context});
    ArmManagedHostClassAllocationFailureAfterForTest(charge, fail);
    bool complete = false;
    try {
      WeightedPredictor<Owner> predictor(width);
      const auto usage = budget.snapshot().classes[static_cast<size_t>(charge)];
      Check(usage.live_capacity_bytes == bytes && usage.backing_count == 5,
            "predictor owner or row storage changed");
      predictor.Reset();
      Check(predictor.Predict(0, 0, 0, 0, 0, 0, 0).first == 0 &&
            predictor.Update(17, 0, 0), "predictor initial state incorrect");
      predictor.Reset();
      Check(predictor.Predict(0, 0, 0, 0, 0, 0, 0) ==
            std::pair<int64_t, int64_t>{0, 0}, "predictor reset leaked state");
      complete = true;
    } catch (const std::bad_alloc&) {}
    DisarmManagedHostAllocationFailureForTest();
    Check(complete == (fail == 5), "predictor allocation count changed");
    Check(budget.snapshot().total.backing_count == 0 &&
          budget.snapshot().total.pending_count == 0,
          "partial predictor construction leaked backing");
  }
  reservation.Reset();
  Check(budget.snapshot().committed_bytes() == 0, "predictor reservation leaked");
}

void TreeStorageAndTransactions() {
  // A prescribed, one-leaf gradient tree with zero offset and unit multiplier.
  constexpr std::array<EntropyToken, 5> tree{{{1, 0}, {2, 5}, {3, 0}, {4, 0}, {5, 0}}};
  GlobalTreeStoragePlan plan;
  Ok(ComputeGlobalTreeStoragePlan(tree.size(), &plan));
  const auto sentinel = plan;
  Check(!ComputeGlobalTreeStoragePlan(0, &plan).ok() && plan == sentinel &&
        !ComputeGlobalTreeStoragePlan(std::numeric_limits<size_t>::max(), &plan).ok() &&
        plan == sentinel && !ComputeGlobalTreeStoragePlan(5, nullptr).ok(),
        "tree planner failure changed output");
  HostStorageBound bound = plan.scratch, writer_bound;
  Ok(codestream_internal::ComputeEntropyWriterStorageBound(plan.maximum_bits + 3,
                                                          &writer_bound));
  Check(bound.Add(writer_bound), "tree writer bound overflow");
  BitWriter oracle;
  Ok(oracle.WriteBits(3, 5));
  Ok(oracle.WithMaxBits(plan.maximum_bits, [&] {
    return WriteGlobalTreeInTransaction(tree, &oracle);
  }));
  Check(Bits(oracle, 3, 2) == 1, "global-tree presence/LZ77 flags changed");
  size_t failures = 0;
  for (size_t fail : {size_t{0}, size_t{1}, size_t{2}, size_t{10}, size_t{25}}) {
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    {
      ResourceContextScope scope({&reservation, ResourceClass::kSerializer});
      BitWriter writer;
      Ok(writer.WriteBits(3, 5));
      ArmManagedHostAllocationFailureAfterForTest(fail);
      const Status status = writer.WithMaxBits(plan.maximum_bits, [&] {
        return WriteGlobalTreeInTransaction(tree, &writer);
      });
      DisarmManagedHostAllocationFailureForTest();
      if (!status.ok()) {
        ++failures;
        Check(status.code() == StatusCode::kOutOfMemory,
              "tree failure lost allocation status");
        CheckSentinel(writer);
        Ok(writer.WithMaxBits(plan.maximum_bits, [&] {
          return WriteGlobalTreeInTransaction(tree, &writer);
        }));
      }
      Check(writer.bits_written() == oracle.bits_written() &&
            std::ranges::equal(writer.padded_bytes(), oracle.padded_bytes()),
            "tree recovery changed emitted bits");
      Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
            "tree exceeded composed plan");
    }
    reservation.Reset();
    Check(budget.snapshot().committed_bytes() == 0, "tree backing leaked");
  }
  Check(failures >= 3, "tree failure points were not reached");
}
}  // namespace

int main() {
  try {
    Headers();
    PredictorOwner<ResourceClass::kPreparation>();
    PredictorOwner<ResourceClass::kSerializer>();
    PredictorOwner<ResourceClass::kCount>();  // Quantization inherits its caller's scope.
    Check(vardct_internal::ModularDcPredictor(kDefaultDcPrediction) == Predictor::kWeighted &&
          vardct_internal::ModularDcPredictor(VarDctDcPrediction::kGradient) == Predictor::kGradient &&
          vardct_internal::ModularDcPredictor(static_cast<VarDctDcPrediction>(255)) == Predictor::kInvalid,
          "VarDCT predictor mapping changed");
    TreeStorageAndTransactions();
    std::cout << "Modular stream, predictor ownership and transaction checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    DisarmManagedHostAllocationFailureForTest();
    std::cerr << error.what() << '\n';
    return 1;
  }
}
