// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Primitive field writing extracted from the libjxl-tiny-derived header writer.
#include "codestream/fields_internal.h"

#include <array>

namespace gjxl::codestream_internal {

Status WriteFields(BitWriter* writer, std::span<const BitField> fields) {
  for (const BitField field : fields) {
    if (Status status = writer->WriteBits(field.width, field.value);
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status AppendTemporary(BitWriter* writer, const BitWriter& temporary) {
  if (writer == nullptr) {
    return Status::InvalidArgument("Bit-writer output is null");
  }
  return writer->Append(temporary);
}

Status WriteSize(uint32_t size, BitWriter* writer) {
  if (size == 0 || size > kMaximumJxlDimension) {
    return Status::InvalidArgument("JPEG XL dimension is out of range");
  }
  const uint32_t encoded = size - 1;
  constexpr std::array<size_t, 4> kWidths = {9, 13, 18, 30};
  for (size_t selector = 0; selector < kWidths.size(); ++selector) {
    if (encoded < (uint32_t{1} << kWidths[selector])) {
      const std::array<BitField, 2> fields = {{
        {2, selector},
        {kWidths[selector], encoded},
      }};
      return WriteFields(writer, fields);
    }
  }
  return Status::Internal("Validated JPEG XL dimension was not encoded");
}

}  // namespace gjxl::codestream_internal
