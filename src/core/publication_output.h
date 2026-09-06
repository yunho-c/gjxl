// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/publication_vector.h"

namespace gjxl::resource_budget_internal {

/// A synchronous vector output whose producer need not know whether the
/// consumer is public or another internal backing owner. Commit only after
/// all fallible work succeeds, exactly as with the original output pointer.
template <typename T>
class PublicationOutput {
public:
  PublicationOutput(std::nullptr_t = nullptr) noexcept {}
  PublicationOutput(std::vector<T>* output) noexcept : public_(output) {}
  PublicationOutput(PublicationVector<T>* output) noexcept : owned_(output) {}

  [[nodiscard]] bool operator==(std::nullptr_t) const noexcept {
    return public_ == nullptr && owned_ == nullptr;
  }

  void Publish(PublicationVector<T>&& candidate) const noexcept {
    assert(*this != nullptr);
    if (owned_ != nullptr) *owned_ = std::move(candidate);
    else candidate.PublishTo(public_);
  }

private:
  std::vector<T>* public_ = nullptr;
  PublicationVector<T>* owned_ = nullptr;
};

}  // namespace gjxl::resource_budget_internal
