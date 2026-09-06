// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/perceptual_storage_plan.h"

#include <algorithm>
#include <limits>

#include "codec/butteraugli_distance_internal.h"

namespace gjxl::frontend_storage_internal {
namespace {
using namespace resource_budget_internal;
using enum VectorCapacityPolicy;
using namespace butteraugli_internal;

constexpr size_t Kernel(float sigma) {
  return 2 * std::max(size_t{1}, static_cast<size_t>(2.25f * sigma)) + 1;
}
// These allocation recipes depend on the pinned call order: Opsin uses the
// non-transposed five-tap path; frequency blur starts with its largest kernel.
static_assert(Kernel(kOpsinBlurSigma) == 5 &&
              Kernel(kLowFrequencyBlurSigma) == 33 &&
              Kernel(kHighFrequencyBlurSigma) == 15 &&
              Kernel(kUltraHighFrequencyBlurSigma) == 7 &&
              Kernel(kMaskBlurSigma) == 13);

Status Overflow() {
  return Status::OutOfMemory("Native perceptual storage bound overflows");
}
bool Planes(HostStorageBound *bound, size_t pixels, size_t channels,
            VectorCapacityPolicy policy, size_t owners = 1) {
  if (pixels > std::numeric_limits<size_t>::max() / channels)
    return false;
  // Validate the count of each CONTIGUOUS owner, not just individual planes.
  return bound->AddVector<float>(pixels * channels, policy, owners);
}
bool Kernels(HostStorageBound *bound) {
  return bound->AddVector<float>(Kernel(kOpsinBlurSigma), kFreshExact) &&
         bound->AddVector<float>(Kernel(kLowFrequencyBlurSigma), kFreshExact) &&
         bound->AddVector<float>(Kernel(kMaskBlurSigma), kFreshExact);
}
bool PreparedScale(size_t pixels, HostStorageBound *bound) {
  // XYB; Opsin blurred/result; frequency blurred/transposed; reference and
  // distorted psychoimages; difference seven planes and AC/DC images; mask
  // blur transpose; observable output plus preallocated atomic stage swap.
  return Planes(bound, pixels, 3, kFreshExact, 3) &&
         Planes(bound, pixels, 1, kFreshExact, 2) &&
         Planes(bound, pixels, 10, kFreshExact, 2) &&
         Planes(bound, pixels, 1, kFreshExact, 7) &&
         Planes(bound, pixels, 3, kFreshExact, 2) &&
         Planes(bound, pixels, 1, kFreshExact) &&
         Planes(bound, pixels, kDifferenceStageCount, kFreshExact, 2) &&
         Kernels(bound);
}
} // namespace

Status ComputeNativeButteraugliStoragePlan(Extent2D requested,
                                           NativeButteraugliStoragePlan *out) {
  if (out == nullptr || requested.empty())
    return Status::InvalidArgument(
        "Native perceptual extent or output is invalid");
  NativeButteraugliStoragePlan plan;
  plan.working_extent = {std::max(requested.width, size_t{8}),
                         std::max(requested.height, size_t{8})};
  const bool expanded = plan.working_extent != requested;
  if (!expanded && requested.width >= 15 && requested.height >= 15)
    plan.sub_extent = requested.ceil_div(2);
  size_t original = 0, main = 0, sub = 0;
  if (!requested.try_area(&original) || !plan.working_extent.try_area(&main) ||
      (!plan.sub_extent.empty() && !plan.sub_extent.try_area(&sub)))
    return Overflow();
  auto &prepared = plan.prepared;
  if (!PreparedScale(main, &prepared) ||
      (expanded && !Planes(&prepared, main, 3, kFreshExact)) ||
      (sub != 0 && (!PreparedScale(sub, &prepared) ||
                    !Planes(&prepared, sub, 3, kFreshExact))) ||
      !Planes(&prepared, original, 1, kFreshExact))
    return Overflow();
  plan.preparation = prepared;
  plan.comparison = prepared;
  // SeparateFrequencies builds a fresh ten-plane candidate before releasing
  // the old distorted psychoimage. Main and subscale replacements are serial.
  if (!Planes(&plan.comparison, main, 10, kFreshExact))
    return Overflow();

  auto &one = plan.one_shot;
  // This API shares scratch between scales. Extent-changing Resize is a fresh
  // replacement, never an in-place shrink; include old/new backing overlap.
  // Repeated calls also retain the stage-swap owner, even though it is empty
  // after a first successful call with initially empty outputs.
  if (!Planes(&one, main, 3, kReusedExact, 4) || // Two XYB, two Opsin.
      !Planes(&one, main, 10, kReusedExact, 2) ||
      !Planes(&one, main, 1, kReusedExact, 2) || // Frequency scratch.
      !Planes(&one, main, 1, kReusedExact, 7) ||
      !Planes(&one, main, 3, kReusedExact, 2) || // Difference AC/DC.
      !Planes(&one, main, 1, kReusedExact) ||    // Difference blur transpose.
      !Planes(&one, main, kDifferenceStageCount, kReusedExact) || // Stage swap.
      !Planes(&one, main, kDifferenceStageCount, kFreshExact) ||
      !Planes(&one, sub, kDifferenceStageCount, kFreshExact) ||
      !Planes(&one, original, 1, kFreshExact) ||
      (expanded && !Planes(&one, main, 3, kFreshExact, 2)) ||
      (sub != 0 && !Planes(&one, sub, 3, kFreshExact, 2)) || !Kernels(&one))
    return Overflow();
  // Native execution is serial. Summing every owner's replacement peak above
  // would assume all resizes happen together. At most one ordinary replacement
  // coexists with its old backing. SeparateFrequencies is the exception: its
  // ten-plane candidate survives a one-plane frequency scratch replacement.
  // The stage-swap candidate is already an owner in the retained inventory.
  HostStorageBound replacement;
  if (!Planes(&replacement, main, 10, kFreshExact) ||
      !Planes(&replacement, main, 1, kFreshExact))
    return Overflow();
  one.peak_bytes = one.retained_bytes;
  if (!one.Add({0, replacement.peak_bytes}))
    return Overflow();
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::frontend_storage_internal
