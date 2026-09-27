// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cstdlib>
#include <cstring>
#include <optional>
#include "core/thread_budget.h"

namespace gjxl::codestream_internal {

// Single-image calls favor latency; batch workers retain the original schedule.
// Concurrent ordinary callers still share the existing resource budgets.
// Configure process-wide controls before encoding, not while calls are active.
inline unsigned EntropyReadinessMode() {
  if (const char* value = std::getenv("GJXL_EARLY_ENTROPY")) {
    if (std::strcmp(value, "0") == 0) return 0;
  }
#ifdef GJXL_TOKENIZATION_EXPERIMENT
  if (const char* value = std::getenv("GJXL_EXPERIMENT_EAGER_ENTROPY")) {
    if (value[0] >= '0' && value[0] <= '3' && value[1] == '\0')
      return static_cast<unsigned>(value[0] - '0');
  }
#endif
  return 3;  // Preserve the full DC grant before admitting early AC entropy.
}

// Batch workers install this marker on the thread that enters the serializer.
inline thread_local bool entropy_readiness_in_batch = false;
class EntropyReadinessBatchScope {
public:
  EntropyReadinessBatchScope() : previous_(entropy_readiness_in_batch) {
    entropy_readiness_in_batch = true;
  }
  ~EntropyReadinessBatchScope() { entropy_readiness_in_batch = previous_; }
  EntropyReadinessBatchScope(const EntropyReadinessBatchScope&) = delete;
  EntropyReadinessBatchScope& operator=(const EntropyReadinessBatchScope&) = delete;
private:
  bool previous_;
};

class EntropyReadinessAdmission {
public:
  EntropyReadinessAdmission(size_t dc_participants, size_t ceiling) {
    using namespace thread_budget_internal;
    if (entropy_readiness_in_batch || !HasCpuParticipation() ||
        InExplicitParallelScope() || dc_participants == 0 ||
        dc_participants >= ceiling) return;
    // Reserve the complete DC target plus AC before launching either branch.
    // A snapshot of free capacity followed by a one-worker reservation would
    // race other images and could leave DC with fewer participants.
    branches_.emplace(dc_participants + 1);
    if (branches_->participants() != dc_participants + 1) {
      branches_.reset();  // Release every partial grant before normal DC work.
      return;
    }
    dc_.emplace(branches_->SplitWorkers(dc_participants - 1));
  }
  [[nodiscard]] bool admitted() const noexcept { return branches_.has_value(); }
  [[nodiscard]] const thread_budget_internal::CpuWorkerGroup& branches() const {
    return *branches_;
  }
  [[nodiscard]] const thread_budget_internal::CpuWorkerGroup& dc() const { return *dc_; }
private:
  std::optional<thread_budget_internal::CpuWorkerGroup> branches_;
  std::optional<thread_budget_internal::CpuWorkerGroup> dc_;
};

}  // namespace gjxl::codestream_internal
