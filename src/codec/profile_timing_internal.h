// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <chrono>
#include <cstdint>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

namespace gjxl::profile_internal {

struct HostInterval {
  uint64_t begin_nanoseconds = 0;
  uint64_t end_nanoseconds = 0;

  [[nodiscard]] bool available() const noexcept {
    return begin_nanoseconds != 0 && end_nanoseconds >= begin_nanoseconds;
  }

  [[nodiscard]] uint64_t duration_nanoseconds() const noexcept {
    return available() ? end_nanoseconds - begin_nanoseconds : 0;
  }
};

[[nodiscard]] inline uint64_t HostNowNanoseconds() noexcept {
#if defined(__APPLE__)
  static const mach_timebase_info_data_t timebase = [] {
    mach_timebase_info_data_t result{};
    (void)mach_timebase_info(&result);
    return result;
  }();
  const __uint128_t scaled =
    static_cast<__uint128_t>(mach_absolute_time()) * timebase.numer;
  return static_cast<uint64_t>(scaled / timebase.denom);
#else
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

inline void Begin(HostInterval* interval) noexcept {
  if (interval != nullptr) {
    interval->begin_nanoseconds = HostNowNanoseconds();
  }
}

inline void End(HostInterval* interval) noexcept {
  if (interval != nullptr) {
    interval->end_nanoseconds = HostNowNanoseconds();
  }
}

}  // namespace gjxl::profile_internal
