// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <cstddef>

namespace gjxl::frontend_dispatch_internal {

// Shared dispatch policy, not a storage/lifetime plan. Work is measured in
// pixels for color/initial quantization and coefficients for forward DCT.
struct Policy {
  size_t minimum_parallel_work;
  size_t maximum_participants;

  [[nodiscard]] constexpr size_t Participants(
      size_t tasks, size_t work, size_t cpu_limit, size_t hardware) const noexcept {
    if (tasks == 0) return 0;
    if (work < minimum_parallel_work) return 1;
    return std::min({tasks, maximum_participants, std::max(hardware, size_t{1}),
                     cpu_limit == 0 ? maximum_participants : cpu_limit});
  }

  // Geometry-only planning cannot rely on the executing machine's CPU count
  // or on how many additional participants its domain admits at runtime.
  [[nodiscard]] constexpr size_t MaximumParticipants(
      size_t tasks, size_t work, size_t cpu_limit) const noexcept {
    return Participants(tasks, work, cpu_limit, maximum_participants);
  }
};

inline constexpr Policy kColor{256 * 256, 12};
inline constexpr Policy kInitialQuant{256 * 256, 12};
inline constexpr Policy kForwardTransform{256 * 256, 8};

// Use the admitted participant count at runtime. Explicit CPU limits and
// managed participation both use the caller; unmanaged automatic work can
// spawn every participant. Serial execution does not allocate a thread vector.
[[nodiscard]] constexpr size_t SpawnedWorkers(
    size_t participants, bool caller_participates) noexcept {
  return participants <= 1 ? 0 : participants - size_t(caller_participates);
}

} // namespace gjxl::frontend_dispatch_internal
