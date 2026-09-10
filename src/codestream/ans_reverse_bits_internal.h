// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

#include "codestream/bit_writer.h"

namespace gjxl::codestream_internal {

// Accumulates reverse-ordered ANS chunks in full 56-bit words plus one tail.
// PushValidated requires count <= 31, bits confined to that width, and total
// payload length representable by size_t. The ANS caller validates these bounds
// and catches allocation failures from ReserveBits/PushValidated.
class AnsReverseBits {
public:
  void ReserveBits(size_t upper_bits) { words_.reserve(upper_bits / 56); }

  void PushValidated(uint32_t bits, unsigned count) {
    const unsigned combined = pending_bits_ + count;
    if (combined < 56) {
      pending_ = (pending_ << count) | bits;
      pending_bits_ = combined;
      return;
    }
    // pending_bits_ <= 55 and count <= 31, so spill <= 30. Flush the oldest
    // high 56 bits and retain the newest low spill bits. The combined value
    // can need 86 bits, but neither expression below does.
    const unsigned spill = combined - 56;
    words_.push_back((pending_ << (56 - pending_bits_)) |
                     (static_cast<uint64_t>(bits) >> spill));
    pending_ = bits & ((uint64_t{1} << spill) - 1);
    pending_bits_ = spill;
  }

  [[nodiscard]] size_t payload_bits() const {
    return words_.size() * 56 + pending_bits_;
  }
  [[nodiscard]] std::span<const uint64_t> words() const { return words_; }
  [[nodiscard]] uint64_t pending() const { return pending_; }
  [[nodiscard]] unsigned pending_bits() const { return pending_bits_; }

  // Append the terminal state and reverse payload atomically, without a
  // temporary byte buffer. All writes retain BitWriter's checked contract.
  [[nodiscard]] Status Append(uint32_t state, BitWriter* destination) const {
    if (destination == nullptr) {
      return Status::InvalidArgument("ANS reverse-bit output is null");
    }
    const size_t payload = payload_bits();
    if (payload > std::numeric_limits<size_t>::max() - 32) {
      return Status::OutOfMemory("ANS reverse-bit output size overflow");
    }
    // Keep the std::function target pointer-sized. Construction can still
    // allocate on another implementation, so retain the enclosing catches.
    struct Operation {
      const AnsReverseBits* source;
      uint32_t state;
      BitWriter* destination;
    };
    const Operation operation{this, state, destination};
    try {
      return destination->WithMaxBits(payload + 32, [&operation]() -> Status {
        const auto& source = *operation.source;
        auto* writer = operation.destination;
        if (source.pending_bits_ <= 24) {
          if (auto s = writer->WriteBits(
                32 + source.pending_bits_,
                operation.state | (source.pending_ << 32));
              !s.ok()) {
            return s;
          }
        } else {
          if (auto s = writer->WriteBits(32, operation.state); !s.ok()) return s;
          if (auto s = writer->WriteBits(source.pending_bits_, source.pending_);
              !s.ok()) {
            return s;
          }
        }
        for (auto word = source.words_.rbegin(); word != source.words_.rend();
             ++word) {
          if (auto s = writer->WriteBits(56, *word); !s.ok()) return s;
        }
        return Status::Ok();
      });
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("ANS reverse-bit append allocation failed");
    } catch (const std::length_error&) {
      return Status::OutOfMemory("ANS reverse-bit append allocation failed");
    }
  }

private:
  std::vector<uint64_t> words_;
  uint64_t pending_ = 0;
  unsigned pending_bits_ = 0;
};

}  // namespace gjxl::codestream_internal
