// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "gpu/metal/metal_aq_profile_storage_plan.h"

namespace {
using namespace gjxl;
using namespace gjxl::metal_internal;
using namespace gjxl::resource_budget_internal;
bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Counts() {
  for (const auto [count, passes] :
       std::array<std::pair<size_t, size_t>, 9>{{{0, 0},
                                                 {1, 1},
                                                 {255, 1},
                                                 {256, 1},
                                                 {257, 2},
                                                 {65536, 2},
                                                 {65537, 3},
                                                 {16777216, 3},
                                                 {16777217, 4}}})
    if (!Check(ButteraugliReductionDispatchCount(count) == passes,
               "Reduction depth differs at a boundary"))
      return false;
  for (size_t level = 0; level <= 31; ++level) {
    const size_t power = size_t{1} << level;
    for (size_t blocks : {power, std::max(size_t{1}, power - 1)}) {
      InitialQuantSortPlan sort;
      if (!Check(ComputeInitialQuantSortPlan(blocks, &sort).ok() &&
                     sort.count >= blocks && sort.count <= power &&
                     sort.dispatches == (blocks == 1 ? 0 : level * (level + 1)),
                 "Bitonic count differs at a boundary"))
        return false;
    }
  }
  size_t cases = 0;
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!budget.TryReserve(1, &job).ok() || !job.ReduceCapacity(0).ok())
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    ArmNextManagedHostAllocationFailureForTest();
    for (Extent2D source : {Extent2D{1, 1},
                            {8, 8},
                            {14, 15},
                            {15, 15},
                            {17, 9},
                            {257, 257},
                            {2049, 2049},
                            {3839, 2159}}) {
      const Extent2D coding{(source.width + 7) / 8 * 8,
                            (source.height + 7) / 8 * 8};
      const size_t blocks = coding.width * coding.height / 64;
      const size_t families = std::min(size_t{7}, blocks);
      const bool expanded = source.width < 8 || source.height < 8;
      const bool multiscale = source.width >= 15 && source.height >= 15;
      ButteraugliDispatchPlan butter;
      if (!Check(
              ComputeButteraugliDispatchPlan(source, blocks, families, &butter)
                      .ok() &&
                  butter.reference == (expanded     ? 19
                                       : multiscale ? 35
                                                    : 16) &&
                  butter.comparison == (expanded     ? 33
                                        : multiscale ? 62
                                                     : 29) +
                                           ButteraugliReductionDispatchCount(
                                               source.width * source.height) &&
                  butter.resident_comparison ==
                      (multiscale
                           ? 58 + families +
                                 ButteraugliReductionDispatchCount(blocks)
                           : 0),
              "Butteraugli dispatch formula differs"))
        return false;
      for (size_t flags = 0; flags < 16; ++flags) {
        AqAuxiliaryProfileStoragePlan auxiliary;
        const bool ac = flags & 1, cfl = flags & 2, quantizer = flags & 4,
                   gaborish = flags & 8;
        if (!Check(
                ComputeAqAuxiliaryProfileStoragePlan(
                    {source, coding, ac, cfl, quantizer, gaborish}, &auxiliary)
                        .ok() &&
                    auxiliary.reference_dispatches == butter.reference &&
                    auxiliary.initial_dispatches ==
                        5 + size_t(ac) + size_t(cfl) +
                            3 * size_t(ac && gaborish) +
                            (quantizer ? 5 + auxiliary.sort.dispatches : 0) &&
                    auxiliary.adjustment_dispatches == 1 + families,
                "Auxiliary AQ dispatch formula differs"))
          return false;
      }
      for (size_t iterations = 0; iterations <= 4; ++iterations)
        for (bool final : {false, true})
          for (bool gaborish : {false, true})
            for (size_t epf = 0; epf <= 3; ++epf) {
              if (iterations == 0 && !final)
                continue;
              ResidentAqProfileStoragePlan p;
              const size_t per_score =
                  23 + 4 * families + size_t(gaborish) + epf +
                  (multiscale ? butter.resident_comparison
                              : butter.comparison + families);
              if (!Check(ComputeResidentAqProfileStoragePlan(
                             source, coding,
                             {iterations, final, multiscale, gaborish, epf},
                             AqProfileFrameOutput::kCompleted, &p)
                                 .ok() &&
                             p.maximum_dispatches ==
                                 (iterations + size_t(final)) * per_score +
                                     2 * families + 2 +
                                     size_t(!final) * (20 + 2 * families) &&
                             p.working.peak_bytes >=
                                 p.metadata.input.peak_bytes +
                                     p.graph.recorded.peak_bytes &&
                             p.working.peak_bytes >=
                                 p.graph.resolution.peak_bytes,
                         "Resident AQ profile formula differs"))
                return false;
              ++cases;
            }
    }
    // Near the shared 32-bit coefficient ceiling, without allocating pixels.
    // This reaches four reduction passes and a 25-level initial sort.
    ResidentAqProfileStoragePlan largest;
    AqAuxiliaryProfileStoragePlan initial;
    const Extent2D huge{32760, 43680};
    if (!Check(ComputeResidentAqProfileStoragePlan(
                   huge, huge, {4, true, true, true, 3},
                   AqProfileFrameOutput::kCompleted, &largest)
                       .ok() &&
                   largest.maximum_dispatches == 636 &&
                   ComputeAqAuxiliaryProfileStoragePlan(
                       {huge, huge, true, true, true, true}, &initial)
                       .ok() &&
                   initial.initial_dispatches == 665,
               "Largest supported AQ profile count differs"))
      return false;
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(pending && budget.snapshot().peak_backing_bytes == 0,
               "AQ profile planning allocated backing"))
      return false;
  }
  job.Reset();
  std::cout
      << "Resident profile shapes: " << cases
      << "; auxiliary shapes: 128; reduction boundaries: 9; sort cases: 64\n";
  return budget.snapshot().committed_bytes() == 0;
}

bool Failures() {
  InitialQuantSortPlan sort{17, 19};
  for (size_t bad :
       {size_t{0}, (size_t{1} << 31) + 1, std::numeric_limits<size_t>::max()})
    if (!Check(!ComputeInitialQuantSortPlan(bad, &sort).ok() &&
                   sort == InitialQuantSortPlan{17, 19},
               "Invalid sort plan changed output"))
      return false;
  ResidentAqProfileStoragePlan sentinel;
  sentinel.maximum_dispatches = 123;
  const auto before = sentinel;
  for (const auto bad : {ResidentAqProfileInputOptions{0, false},
                         {5},
                         {0, true, true},
                         {0, true, false, false, 4}})
    if (!Check(
            !ComputeResidentAqProfileStoragePlan(
                 {8, 8}, {8, 8}, bad, AqProfileFrameOutput::kOwned, &sentinel)
                    .ok() &&
                sentinel == before,
            "Bad resident policy changed output"))
      return false;
  AqAuxiliaryProfileStoragePlan aux;
  aux.initial_dispatches = 123;
  const auto old_aux = aux;
  for (Extent2D bad : {Extent2D{},
                       {9, 8},
                       {65536, 65536},
                       {std::numeric_limits<size_t>::max(), 8}})
    if (!Check(
            !ComputeResidentAqProfileStoragePlan(
                 bad, bad, {}, AqProfileFrameOutput::kOwned, &sentinel)
                    .ok() &&
                sentinel == before &&
                !ComputeAqAuxiliaryProfileStoragePlan({bad, bad}, &aux).ok() &&
                aux == old_aux,
            "Bad geometry changed profile plans"))
      return false;
  ButteraugliDispatchPlan butter{.reference = 123};
  const auto old_butter = butter;
  for (const auto [anchors, families] :
       std::array<std::pair<size_t, size_t>, 4>{
           {{0, 1}, {1, 0}, {1, 2}, {65, 7}}})
    if (!Check(
            !ComputeButteraugliDispatchPlan({8, 8}, anchors, families, &butter)
                    .ok() &&
                butter == old_butter,
            "Bad sink counts changed output"))
      return false;
  return Check(
      !ComputeResidentAqProfileStoragePlan(
           {8, 8}, {8, 8}, {1, false}, AqProfileFrameOutput::kNone, &sentinel)
              .ok() &&
          sentinel == before &&
          !ComputeResidentAqProfileStoragePlan(
               {8, 8}, {8, 8}, {}, AqProfileFrameOutput(255), &sentinel)
               .ok() &&
          sentinel == before &&
          !ComputeResidentAqProfileStoragePlan(
               {8, 8}, {8, 8}, {}, AqProfileFrameOutput::kOwned, nullptr)
               .ok() &&
          !ComputeButteraugliDispatchPlan({8, 8}, 0, 0, nullptr).ok() &&
          !ComputeAqAuxiliaryProfileStoragePlan({}, nullptr).ok() &&
          !ComputeInitialQuantSortPlan(1, nullptr).ok(),
      "Invalid output shape accepted");
}
} // namespace

int main() { return Counts() && Failures() ? EXIT_SUCCESS : EXIT_FAILURE; }
