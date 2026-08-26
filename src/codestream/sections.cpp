// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's frame-section assembly.

#include "codestream/sections.h"

#include <array>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace gjxl {
namespace {

constexpr std::array<size_t, 4> kTocBitWidths = {10, 14, 22, 30};

bool TryAdd(size_t left, size_t right, size_t* result) {
  if (result == nullptr ||
      left > std::numeric_limits<size_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

Status ValidateTocSizes(
  std::span<const size_t> section_sizes,
  size_t* maximum_bits) {

  if (maximum_bits == nullptr) {
    return Status::InvalidArgument("TOC bit-count output is null");
  }
  size_t bits = 8 + 7;
  for (size_t size : section_sizes) {
    if (size > kMaximumTocSectionSize) {
      return Status::InvalidArgument(
        "Section size cannot be represented by the JPEG XL TOC");
    }
    if (!TryAdd(bits, 32, &bits)) {
      return Status::OutOfMemory("TOC bit count overflow");
    }
  }
  *maximum_bits = bits;
  return Status::Ok();
}

Status WriteTocSizesInternal(
  std::span<const size_t> section_sizes,
  BitWriter* output) {

  if (Status status = output->WriteBits(1, 0); !status.ok()) {
    return status;
  }
  if (Status status = output->ZeroPadToByte(); !status.ok()) {
    return status;
  }
  for (size_t section_size : section_sizes) {
    size_t offset = 0;
    bool encoded = false;
    for (size_t selector = 0; selector < kTocBitWidths.size(); ++selector) {
      const size_t range = size_t{1} << kTocBitWidths[selector];
      if (section_size - offset < range) {
        if (Status status = output->WriteBits(2, selector); !status.ok()) {
          return status;
        }
        if (Status status = output->WriteBits(
              kTocBitWidths[selector], section_size - offset);
            !status.ok()) {
          return status;
        }
        encoded = true;
        break;
      }
      offset += range;
    }
    if (!encoded) {
      return Status::Internal("Validated TOC size was not encoded");
    }
  }
  return output->ZeroPadToByte();
}

Status SectionSizes(
  std::span<const BitWriter> sections,
  std::vector<size_t>* sizes,
  size_t* payload_bits) {

  if (sizes == nullptr || payload_bits == nullptr) {
    return Status::InvalidArgument("Section-size output is null");
  }
  sizes->clear();
  sizes->reserve(sections.size());
  size_t total_bytes = 0;
  for (const BitWriter& section : sections) {
    const size_t size = section.padded_size();
    if (size > kMaximumTocSectionSize) {
      return Status::InvalidArgument(
        "Section size cannot be represented by the JPEG XL TOC");
    }
    if (!TryAdd(total_bytes, size, &total_bytes)) {
      return Status::OutOfMemory("Section payload size overflow");
    }
    sizes->push_back(size);
  }
  if (total_bytes > std::numeric_limits<size_t>::max() / 8) {
    return Status::OutOfMemory("Section payload bit count overflow");
  }
  *payload_bits = total_bytes * 8;
  return Status::Ok();
}

}  // namespace

Status WriteTocSizes(
  std::span<const size_t> section_sizes,
  BitWriter* output) {

  if (output == nullptr) {
    return Status::InvalidArgument("TOC output is null");
  }
  size_t maximum_bits = 0;
  if (Status status = ValidateTocSizes(section_sizes, &maximum_bits);
      !status.ok()) {
    return status;
  }
  return output->WithMaxBits(maximum_bits, [&]() {
    return WriteTocSizesInternal(section_sizes, output);
  });
}

Status ConcatenateSections(
  std::span<const BitWriter> sections,
  BitWriter* output) {

  if (output == nullptr) {
    return Status::InvalidArgument("Section output is null");
  }
  return output->AppendByteAligned(sections);
}

Status WriteTocAndSections(
  std::span<const BitWriter> sections,
  BitWriter* output) {

  if (output == nullptr) {
    return Status::InvalidArgument("Section output is null");
  }
  std::vector<size_t> sizes;
  size_t payload_bits = 0;
  try {
    if (Status status = SectionSizes(sections, &sizes, &payload_bits);
        !status.ok()) {
      return status;
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Section-size allocation failed");
  } catch (const std::length_error&) {
    return Status::OutOfMemory("Section-size allocation is too large");
  }

  size_t toc_bits = 0;
  if (Status status = ValidateTocSizes(sizes, &toc_bits); !status.ok()) {
    return status;
  }
  size_t maximum_bits = 0;
  if (!TryAdd(toc_bits, payload_bits, &maximum_bits)) {
    return Status::OutOfMemory("Combined section bit count overflow");
  }
  return output->WithMaxBits(maximum_bits, [&]() -> Status {
    if (Status status = WriteTocSizesInternal(sizes, output); !status.ok()) {
      return status;
    }
    return output->AppendByteAligned(sections);
  });
}

}  // namespace gjxl
