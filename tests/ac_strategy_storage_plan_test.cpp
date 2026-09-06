// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "core/managed_allocator.h"
#include "gpu/ops/ac_strategy_storage_plan.h"

namespace {
using namespace gjxl;
using namespace gjxl::ac_strategy_search_internal;
using namespace gjxl::resource_budget_internal;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}

bool CheckGeometry(Extent2D coding, bool resident) {
  // Frozen dd5cd54 policy dimensions and tile-by-tile capacity recipe. Keep
  // this independent of production stage constants and its separable count
  // formula. Strategy names are rows-by-columns, extents width-by-height.
  constexpr std::array<Extent2D, 7> covered{
      {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {2, 4}, {4, 2}, {4, 4}}};
  const size_t bw = coding.width / 8, bh = coding.height / 8;
  const size_t tw = (bw + 7) / 8, th = (bh + 7) / 8;
  StoragePlan plan;
  if (!Check(ComputeStoragePlan(coding, resident, &plan).ok(),
             "Valid AC geometry failed"))
    return false;
  if (!Check(plan.block_extent == Extent2D{bw, bh} &&
                 plan.tile_extent == Extent2D{tw, th} &&
                 plan.pixel_count == coding.width * coding.height &&
                 plan.block_count == bw * bh &&
                 plan.block_cost_bytes_per_stage == bw * bh * 4 &&
                 plan.opsin_bytes ==
                     (resident ? 0 : coding.width * coding.height * 12) &&
                 plan.mask_bytes ==
                     (resident ? 0 : coding.width * coding.height * 4),
             "AC geometry/input plan disagrees with frozen recipe"))
    return false;
  size_t device_bytes = resident ? 0 : coding.width * coding.height * 16;
  size_t packed = 0, rate = 0;
  for (size_t family = 0; family < covered.size(); ++family) {
    const size_t step = family < 4 ? 1 : 2;
    size_t count = 0;
    for (size_t ty = 0; ty < th; ++ty) {
      const size_t height = std::min(size_t{8}, bh - ty * 8);
      for (size_t tx = 0; tx < tw; ++tx) {
        const size_t width = std::min(size_t{8}, bw - tx * 8);
        if (width >= covered[family].width &&
            height >= covered[family].height) {
          count += ((width - covered[family].width) / step + 1) *
                   ((height - covered[family].height) / step + 1);
        }
      }
    }
    const size_t coefficients =
        covered[family].width * covered[family].height * 64;
    const auto &stage = plan.stages[family];
    if (!Check(
            stage.candidate_count == count &&
                stage.candidate_bytes == count * 24 &&
                stage.matrix_bytes == coefficients * 6 * 4 &&
                stage.cost_bytes == count * 4,
            "AC stage count/capacity differs from frozen tile-by-tile recipe")) {
      std::cerr << "coding=" << coding.width << 'x' << coding.height
                << " family=" << family << " resident=" << resident
                << " candidates=" << stage.candidate_count << " expected=" << count << '\n';
      return false;
    }
    if (count != 0)
      device_bytes += count * 28 + coefficients * 24;
    packed = std::max(packed, count * 3 * coefficients * 4);
    rate = std::max(rate, count * 3 * 8);
  }
  return Check(
      plan.maximum_packed_bytes == packed && plan.maximum_rate_bytes == rate &&
          plan.device_bytes == device_bytes + 2 * packed + rate,
      "AC maximum scratch/total device capacity differs from frozen recipe");
}

bool CheckFailuresAndNoBacking() {
  StoragePlan plan;
  plan.device_bytes = 123;
  plan.stages[2].candidate_count = 99;
  const auto previous = plan;
  for (Extent2D bad : {Extent2D{0, 8},
                       {7, 8},
                       {8, 9},
                       {65536, 65536},
                       {std::numeric_limits<size_t>::max(), 8},
                       {size_t{1} << 32, 8}}) {
    if (!Check(ComputeStoragePlan(bad, true, &plan).code() ==
                       StatusCode::kInvalidArgument &&
                   plan == previous,
               "Invalid AC storage plan changed output"))
      return false;
  }
  if (!Check(!ComputeStoragePlan({8, 8}, true, nullptr).ok(),
             "Null AC plan accepted"))
    return false;
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Check(budget.TryReserve(1, &job).ok(), "Test reservation failed"))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kAcSearch});
    ArmNextManagedHostAllocationFailureForTest();
    // Largest multiple-of-eight width below the accepted pixel-index limit.
    // This needs more scratch than this machine's RAM: planning must not visit pixels,
    // allocate that backing, or iterate all tiles to count candidates.
    const bool success = ComputeStoragePlan({536870904, 8}, false, &plan).ok();
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(success && pending &&
                   budget.snapshot().peak_backing_bytes == 0 &&
                   plan.stages[0].candidate_count == 67108863,
               "Large AC geometry planning allocated backing or counted "
               "incorrectly"))
      return false;
  }
  return Check(DefaultResourceBudget().snapshot().peak_backing_bytes == 0,
               "AC planner escaped to default domain");
}
} // namespace

int main() {
  for (size_t by = 1; by <= 64; ++by) {
    for (size_t bx = 1; bx <= 64; ++bx) {
      for (bool resident : {false, true}) {
        if (!CheckGeometry({bx * 8, by * 8}, resident))
          return EXIT_FAILURE;
      }
    }
  }
  for (Extent2D coding :
       {Extent2D{1920, 1080}, {3840, 2160}, {256, 4096}, {8192, 8}}) {
    for (bool resident : {false, true})
      if (!CheckGeometry(coding, resident))
        return EXIT_FAILURE;
  }
  if (!CheckFailuresAndNoBacking())
    return EXIT_FAILURE;
  std::cout << "Frozen AC geometry/capacity cases: 8200\n";
  return EXIT_SUCCESS;
}
