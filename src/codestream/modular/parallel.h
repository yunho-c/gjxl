// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/storage.h"
#include "core/parallel_work_internal.h"
#include <algorithm>
namespace gjxl::modular_internal {
inline size_t ModularParticipants(size_t requested) {
  return std::min(size_t{4},
                  requested ? requested
                            : std::max(size_t{1}, size_t{std::thread::hardware_concurrency()}));
}
template <class Function>
Status RunModularStreams(size_t count, size_t maximum,
                         thread_budget_internal::WorkerLaunchSite site, Function &&function) {
  using namespace thread_budget_internal;
  if (count < 2 || maximum == 1 || InExplicitParallelScope()) {
    for (size_t i = 0; i < count; ++i)
      if (auto s = function(i); !s.ok())
        return s;
    return Status::Ok();
  }
  CpuWorkerGroup group(std::min(count, maximum));
  if (group.participants() == 1) {
    for (size_t i = 0; i < count; ++i)
      if (auto s = function(i); !s.ok())
        return s;
    return Status::Ok();
  }
  constexpr ParallelWorkErrors errors{"Modular worker allocation failed",
                                      "Modular worker failed",
                                      StatusCode::kInvalidArgument,
                                      "Modular worker storage overflow",
                                      "Modular worker launch allocation failed",
                                      LaunchFailureAction::kReturnError,
                                      "Unable to start Modular workers"};
  return RunParallelWork<codestream_internal::Storage>(
      count, group, group.participants() - 1, site, errors,
      [&](size_t i, size_t) { return function(i); });
}
} // namespace gjxl::modular_internal
