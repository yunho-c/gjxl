// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/managed_allocator.h"
#include "core/stdlib_storage_compat.h"

namespace gjxl::resource_budget_internal {

/// These helpers release a container's backing charge, not nested element
/// charges. Call only after successful publication; they never free/copy data.
template <typename T, ResourceClass Owner>
void ReleaseManagedBackingAfterPublication(ManagedVector<T, Owner>& values) noexcept {
  ManagedAllocator<T, Owner>::ReleaseChargeAfterPublication(values.data());
}

template <ResourceClass Owner>
void ReleaseManagedBackingAfterPublication(ManagedString<Owner>& value) noexcept {
  if (auto* data = stdlib_storage_internal::StringHeapData(value))
    ManagedAllocator<char, Owner>::ReleaseChargeAfterPublication(data);
}

}  // namespace gjxl::resource_budget_internal
