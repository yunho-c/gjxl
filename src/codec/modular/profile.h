// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/coding.h"
#include <array>
#include <chrono>
#include <cstddef>

namespace gjxl::modular_internal {
// Private, opt-in caller-wall-time instrumentation. Workers are deliberately
// not given the sink: tokenization/emission measure the caller's joined interval.
enum class ProfileStage {
  kInput,
  kTransforms,
  kTraining,
  kTokens,
  kModel,
  kEmission,
  kAssembly,
  kCount
};
struct ModularProfile {
  std::array<double, static_cast<size_t>(ProfileStage::kCount)> seconds{};
  ModularCodingPolicy resolved;
  size_t encoded_candidates = 0;
};
inline thread_local ModularProfile *active_modular_profile = nullptr;
class ProfileSession {
public:
  explicit ProfileSession(ModularProfile *profile) : previous_(active_modular_profile) {
    active_modular_profile = profile;
  }
  ~ProfileSession() { active_modular_profile = previous_; }
  ProfileSession(const ProfileSession &) = delete;
  ProfileSession &operator=(const ProfileSession &) = delete;

private:
  ModularProfile *previous_;
};
class ProfileScope {
  using Clock = std::chrono::steady_clock;

public:
  explicit ProfileScope(ProfileStage stage) : sink_(active_modular_profile), stage_(stage) {
    if (sink_) {
      start_ = Clock::now();
      parent_ = current_;
      current_ = this;
    }
  }
  ~ProfileScope() {
    if (sink_) {
      const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
      sink_->seconds[static_cast<size_t>(stage_)] += elapsed - children_;
      if (parent_ && parent_->sink_ == sink_)
        parent_->children_ += elapsed;
      current_ = parent_;
    }
  }
  ProfileScope(const ProfileScope &) = delete;
  ProfileScope &operator=(const ProfileScope &) = delete;

private:
  inline static thread_local ProfileScope *current_ = nullptr;
  ModularProfile *sink_;
  ProfileStage stage_;
  Clock::time_point start_{};
  ProfileScope *parent_ = nullptr;
  double children_ = 0;
};
} // namespace gjxl::modular_internal
