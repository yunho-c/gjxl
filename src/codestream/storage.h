// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <iterator>
#include <stdexcept>
#include <vector>

#include "core/managed_allocator.h"

namespace gjxl::codestream_internal {

/// Owning serializer storage. Consumers should borrow spans, not require the
/// allocator type of a producer. Backing is charged only in managed scopes.
template <typename T>
using Storage = resource_budget_internal::ManagedVector<
  T, resource_budget_internal::ResourceClass::kSerializer>;

/// Explicit compatibility boundary for callers supplying an ordinary vector.
/// The resident serializer calls the managed overload directly, without this
/// publication copy. Build the caller's candidate before replacing its output.
template <typename T, typename Allocator, typename Function>
Status LegacyStorageOutput(std::vector<T, Allocator>* output, Function&& operation) {
  if (output == nullptr) return Status::InvalidArgument("Storage output is null");
  try {
    Storage<T> storage;
    Status status = operation(&storage);
    if (!status.ok()) return status;
    std::vector<T, Allocator> candidate(
      std::make_move_iterator(storage.begin()),
      std::make_move_iterator(storage.end()), output->get_allocator());
    *output = std::move(candidate);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Legacy storage publication allocation failed");
  } catch (const std::length_error&) {
    return Status::OutOfMemory("Legacy storage publication is too large");
  }
}

template <typename T, typename Allocator, typename Function>
Status LegacyNestedStorageOutput(
  std::vector<std::vector<T>, Allocator>* output, Function&& operation) {
  if (output == nullptr) return Status::InvalidArgument("Storage output is null");
  try {
    Storage<Storage<T>> storage;
    Status status = operation(&storage);
    if (!status.ok()) return status;
    std::vector<std::vector<T>, Allocator> candidate(output->get_allocator());
    candidate.reserve(storage.size());
    for (const auto& row : storage) candidate.emplace_back(row.begin(), row.end());
    *output = std::move(candidate);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Legacy nested storage publication allocation failed");
  } catch (const std::length_error&) {
    return Status::OutOfMemory("Legacy nested storage publication is too large");
  }
}

}  // namespace gjxl::codestream_internal
