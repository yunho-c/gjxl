// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

namespace gjxl::codestream_internal {
// Fault injection only: replace the next outer admission's computed capacity.
// It cannot grow a reservation after admission or affect a different thread.
inline thread_local std::optional<size_t> next_admission_capacity_for_testing;
inline void ArmNextWorkflowAdmissionCapacityForTest(size_t bytes) noexcept {
  next_admission_capacity_for_testing = bytes;
}
inline void DisarmWorkflowAdmissionCapacityForTest() noexcept {
  next_admission_capacity_for_testing.reset();
}
} // namespace gjxl::codestream_internal
