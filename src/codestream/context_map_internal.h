// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "codestream/bit_writer.h"
#include "codestream/storage.h"

namespace gjxl::codestream_internal {

/// Immutable serialized map plus the exact input used to produce it. Matching
/// the input keeps externally edited EntropyCodes from using stale bytes;
/// const writers never mutate a cache or introduce shared lazy state.
class ContextMapEncoding {
public:
  [[nodiscard]] bool Matches(std::span<const uint8_t> map) const noexcept;
  [[nodiscard]] Status AppendTo(BitWriter* writer) const;
  [[nodiscard]] size_t bits_written() const noexcept { return bits_; }
  [[nodiscard]] const Storage<uint8_t>& source() const noexcept { return source_; }
  [[nodiscard]] const Storage<uint8_t>& bytes() const noexcept { return bytes_; }
  bool operator==(const ContextMapEncoding&) const = default;

private:
  Storage<uint8_t> source_;
  Storage<uint8_t> bytes_;
  size_t bits_ = 0;
  friend Status EncodeContextMap(std::span<const uint8_t>, ContextMapEncoding*);
};

/// Searches raw/MTF, simple, Prefix, ANS and distance-one RLE representations.
/// The existing raw Prefix representation is the incumbent and wins ties.
/// Failure preserves *out. No coefficient or model-clustering policy changes.
[[nodiscard]] Status EncodeContextMap(std::span<const uint8_t> map,
                                      ContextMapEncoding* out);

/// The pre-search representation, retained as a size fallback and test oracle.
[[nodiscard]] Status WriteLegacyContextMap(std::span<const uint8_t> map,
                                          BitWriter* writer);

}  // namespace gjxl::codestream_internal
