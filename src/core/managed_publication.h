// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>

#include "core/managed_allocator.h"

namespace gjxl::resource_budget_internal {

/// These helpers release a container's backing charge, not nested element
/// charges. Call only after successful publication; they never free/copy data.
template <typename T, ResourceClass Owner>
void ReleaseManagedBackingAfterPublication(ManagedVector<T, Owner>& values) noexcept {
  ManagedAllocator<T, Owner>::ReleaseChargeAfterPublication(values.data());
}

// Reviewed libc++ C++20 string contract: __get_short_pointer points into the
// string object's inline character array; __get_long_pointer is the allocator's
// original pointer, outside the string object. Never inspect an
// inline short string as if it had an allocation header. Review before porting.
#if !defined(_LIBCPP_VERSION) || _LIBCPP_STD_VER != 20
#error "Managed string publication requires the reviewed libc++ C++20 contract"
#endif
template <ResourceClass Owner>
void ReleaseManagedBackingAfterPublication(ManagedString<Owner>& value) noexcept {
  const auto object = reinterpret_cast<uintptr_t>(&value);
  const auto data = reinterpret_cast<uintptr_t>(value.data());
  if (data < object || data - object >= sizeof(value))
    ManagedAllocator<char, Owner>::ReleaseChargeAfterPublication(value.data());
}

}  // namespace gjxl::resource_budget_internal
