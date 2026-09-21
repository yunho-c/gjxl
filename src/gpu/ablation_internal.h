// SPDX-License-Identifier: Apache-2.0
// Experimental controls: absent from ordinary builds and the installed API.
#pragma once

#ifdef GJXL_ABLATION_EXPERIMENT
#include <array>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#endif

namespace gjxl::ablation_internal {
struct Config {
  const char* name = "production";
  bool sync_aq = false;
  bool ac_handoff = false;
  bool split_malta = false;
  bool split_epf = false;
  bool host_selector = false;
  bool split_ac = false;
  bool packed_dct = false;
  bool scalar_dct = false;
};

#ifdef GJXL_ABLATION_EXPERIMENT
inline constexpr std::array kConfigs{
  Config{},
  Config{.name = "aq-sync", .sync_aq = true},
  Config{.name = "ac-handoff", .ac_handoff = true},
  Config{.name = "split-malta", .split_malta = true},
  Config{.name = "split-epf", .split_epf = true},
  Config{.name = "host-fused", .host_selector = true},
  Config{.name = "host-split-ac", .host_selector = true, .split_ac = true},
  Config{.name = "packed-simd", .host_selector = true, .split_ac = true,
         .packed_dct = true},
  Config{.name = "packed-scalar", .host_selector = true, .split_ac = true,
         .packed_dct = true, .scalar_dct = true},
};

// One variant per process: production backend preparation is process-cached.
inline const Config& Get() {
  static const Config config = [] {
    const char* name = std::getenv("GJXL_ABLATION_VARIANT");
    if (!name) return kConfigs[0];
    for (const auto& candidate : kConfigs)
      if (std::string_view(name) == candidate.name) return candidate;
    throw std::runtime_error("Unknown GJXL_ABLATION_VARIANT");
  }();
  return config;
}

// Fixed storage, no allocation, no GPU timestamp profiling. Enabled only for
// a separate untimed validation call. Metal submission/encoding is synchronous
// on the public-call thread; CPU worker activity is outside this audit.
struct Audit {
  struct Entry { std::array<char, 192> name{}; uint64_t count = 0; };
  std::array<Entry, 512> entries{};
  size_t size = 0;
  bool overflow = false;
  std::array<char, 192> kernel{};
};
inline thread_local Audit* active_audit = nullptr;
inline bool Auditing() { return active_audit != nullptr; }
inline void Count(std::string_view name, uint64_t amount = 1) {
  if (!active_audit) return;
  auto& audit = *active_audit;
  for (size_t i = 0; i < audit.size; ++i) {
    if (name == audit.entries[i].name.data()) {
      audit.entries[i].count += amount;
      return;
    }
  }
  if (audit.size == audit.entries.size() || name.size() >= 192) {
    audit.overflow = true;
    return;
  }
  auto& entry = audit.entries[audit.size++];
  name.copy(entry.name.data(), name.size());
  entry.count = amount;
}
inline void Kernel(std::string_view name) {
  if (!active_audit) return;
  active_audit->kernel.fill(0);
  if (name.size() >= active_audit->kernel.size()) {
    active_audit->overflow = true;
    return;
  }
  name.copy(active_audit->kernel.data(), name.size());
}
inline void Dispatch() {
  if (!active_audit) return;
  Count("dispatches");
  Count(active_audit->kernel[0] ? active_audit->kernel.data() : "unknown_kernel");
}
#else
inline constexpr Config Get() { return {}; }
inline constexpr bool Auditing() { return false; }
inline void Count(const char*, unsigned long long = 1) {}
inline void Dispatch() {}
#endif
}  // namespace gjxl::ablation_internal
