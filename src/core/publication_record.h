// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/publication_vector.h"

namespace gjxl::resource_budget_internal {

/// Internal owner for a public record with one vector field. Only SetField
/// may replace that field before publication; scalar metadata can be edited
/// through value(). Member order and replacement preserve backing-before-charge
/// destruction. Public record layout and default-allocator ownership stay intact.
template <typename Record, typename T, std::vector<T> Record::*Field>
class PublicationRecord {
public:
  PublicationRecord() = default;
  PublicationRecord(const PublicationRecord&) = delete;
  PublicationRecord& operator=(const PublicationRecord&) = delete;
  PublicationRecord(PublicationRecord&&) noexcept = default;
  PublicationRecord& operator=(PublicationRecord&& other) noexcept {
    if (this != &other) {
      PublicationRecord replacement(std::move(other));
      swap(replacement);
    }
    return *this;
  }

  [[nodiscard]] Record& value() noexcept { return value_; }
  [[nodiscard]] const Record& value() const noexcept { return value_; }

  void SetField(PublicationVector<T>&& values) noexcept {
    // MoveToPublication frees the old vector while allocation_ still owns its
    // charge. Then replace that charge with the new backing's ticket.
    auto charge = values.MoveToPublication(&(value_.*Field));
    allocation_ = std::move(charge);
  }

  void swap(PublicationRecord& other) noexcept {
    static_assert(std::is_nothrow_swappable_v<Record>);
    std::swap(value_, other.value_);
    std::swap(allocation_, other.allocation_);
  }
  void Reset() noexcept { PublicationRecord{}.swap(*this); }

  [[nodiscard]] ResourceAllocation MoveToPublication(Record* output) noexcept {
    static_assert(std::is_nothrow_move_assignable_v<Record>);
    assert(output != nullptr);
    *output = std::move(value_);
    return std::move(allocation_);
  }

  void PublishTo(Record* output) noexcept {
    auto charge = MoveToPublication(output);
  }

  [[nodiscard]] Status Reclassify(ResourceClass owner) {
    return allocation_.valid() ? allocation_.Reclassify(owner) :
      ((value_.*Field).capacity() == 0 ? Status::Ok() :
        Status::FailedPrecondition("Record backing is not managed"));
  }

  friend bool operator==(const PublicationRecord& a, const PublicationRecord& b) {
    return a.value_ == b.value_;
  }

private:
  ResourceAllocation allocation_;
  Record value_;
};

}  // namespace gjxl::resource_budget_internal
