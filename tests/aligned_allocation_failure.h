// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdlib>
#include <limits>
#include <new>
#if defined(_WIN32)
#include <malloc.h>
#endif

// Included by standalone allocation-fault tests after their ShouldFail hook.
// Managed host backing uses aligned new; exercise it in the same exhaustive
// failure sequence as ordinary container/control allocations.
void* operator new(size_t size, std::align_val_t alignment) {
  if (allocation_failure::ShouldFail()) throw std::bad_alloc();
  const size_t align = static_cast<size_t>(alignment);
#if defined(_WIN32)
  if (void* p = _aligned_malloc(size == 0 ? 1 : size, align)) return p;
#else
  const size_t bytes = size == 0 ? 1 : size;
  if (bytes > std::numeric_limits<size_t>::max() - (align - 1))
    throw std::bad_alloc();
  if (void* p = std::aligned_alloc(align, (bytes + align - 1) / align * align))
    return p;
#endif
  throw std::bad_alloc();
}
void* operator new[](size_t size, std::align_val_t align) {
  return ::operator new(size, align);
}
void operator delete(void* p, std::align_val_t) noexcept {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t align) noexcept {
  ::operator delete(p, align);
}
void operator delete(void* p, size_t, std::align_val_t align) noexcept {
  ::operator delete(p, align);
}
void operator delete[](void* p, size_t, std::align_val_t align) noexcept {
  ::operator delete(p, align);
}
