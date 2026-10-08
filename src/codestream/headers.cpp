// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Primitive field writing extracted from the libjxl-tiny-derived header writer.
#include "codestream/fields_internal.h"

#include <array>
#include <limits>
#include "codestream/headers_internal.h"
#include "codestream/entropy_storage_plan.h"

namespace gjxl::codestream_internal {

Status WriteFields(BitWriter *writer, std::span<const BitField> fields) {
  for (const BitField field : fields) {
    if (Status status = writer->WriteBits(field.width, field.value); !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status AppendTemporary(BitWriter *writer, const BitWriter &temporary) {
  if (writer == nullptr) {
    return Status::InvalidArgument("Bit-writer output is null");
  }
  return writer->Append(temporary);
}

Status WriteSize(uint32_t size, BitWriter *writer) {
  if (size == 0 || size > kMaximumJxlDimension) {
    return Status::InvalidArgument("JPEG XL dimension is out of range");
  }
  const uint32_t encoded = size - 1;
  constexpr std::array<size_t, 4> kWidths = {9, 13, 18, 30};
  for (size_t selector = 0; selector < kWidths.size(); ++selector) {
    if (encoded < (uint32_t{1} << kWidths[selector])) {
      const std::array<BitField, 2> fields = {{
          BitField{2, selector},
          {kWidths[selector], encoded},
      }};
      return WriteFields(writer, fields);
    }
  }
  return Status::Internal("Validated JPEG XL dimension was not encoded");
}

namespace {
Status WriteIntegerDepth(uint8_t bits, BitWriter *writer) {
  if (auto s = writer->WriteBits(1, 0); !s.ok())
    return s;
  if (bits == 8)
    return writer->WriteBits(2, 0);
  return WriteFields(writer, std::array{BitField{2, 3}, BitField{6, 15}});
}
} // namespace

Status WriteImageHeader(const codec_internal::ImageMetadata &metadata,
                        BitWriter *writer) {
  if (!writer)
    return Status::InvalidArgument("Null image-header output");
  if (auto s = codec_internal::ValidateImageMetadata(metadata); !s.ok())
    return s;
  BitWriter temporary;
  const std::array<BitField, 3> prefix = {
      {BitField{8, 0xFF}, BitField{8, 0x0A}, BitField{1, 0}}};
  if (Status status = WriteFields(&temporary, prefix); !status.ok()) {
    return status;
  }
  if (Status status =
          WriteSize(static_cast<uint32_t>(metadata.extent.height), &temporary);
      !status.ok()) {
    return status;
  }
  if (Status status = temporary.WriteBits(3, 0); !status.ok()) {
    return status;
  }
  if (Status status =
          WriteSize(static_cast<uint32_t>(metadata.extent.width), &temporary);
      !status.ok()) {
    return status;
  }

  if (metadata.sample_format == codec_internal::SampleFormat::kFloat) {
    // Non-default metadata: float32 linear sRGB, XYB transform, no extras.
    const std::array<BitField, 19> fields = {{
        BitField{1, 0}, BitField{1, 0}, BitField{1, 1}, BitField{2, 0}, BitField{4, 7},
        BitField{1, 0}, BitField{2, 0}, BitField{1, 1}, BitField{1, 0}, BitField{1, 0},
        BitField{2, 0}, BitField{2, 1}, BitField{2, 1}, BitField{1, 0}, BitField{2, 2},
        BitField{4, 6}, BitField{2, 1}, BitField{2, 0}, BitField{1, 1},
    }};
    if (Status status = WriteFields(&temporary, fields); !status.ok()) {
      return status;
    }
  } else {
    // ImageMetadata: explicit fields, no extended metadata, integer bit depth.
    if (auto s = WriteFields(&temporary, std::array{BitField{1, 0}, BitField{1, 0}});
        !s.ok())
      return s;
    if (auto s = WriteIntegerDepth(metadata.bits, &temporary); !s.ok())
      return s;
    // A 32-bit channel buffer is allowed for every advertised source depth.
    if (auto s =
            WriteFields(&temporary, std::array{BitField{1, 0},
                                               BitField{2, metadata.alpha ? 1u : 0u}});
        !s.ok())
      return s;
    if (metadata.alpha) {
      if (metadata.bits == 8) {
        if (auto s = temporary.WriteBits(1, 1); !s.ok())
          return s; // default alpha
      } else {
        if (auto s =
                WriteFields(&temporary, std::array{BitField{1, 0}, BitField{2, 0}});
            !s.ok())
          return s;
        if (auto s = WriteIntegerDepth(metadata.alpha->bits, &temporary); !s.ok())
          return s;
        // dim_shift=0, empty name, unassociated.
        if (auto s = WriteFields(
                &temporary, std::array{BitField{2, 0}, BitField{2, 0}, BitField{1, 0}});
            !s.ok())
          return s;
      }
    }
    if (auto s = temporary.WriteBits(1, 0); !s.ok())
      return s; // no XYB
    if (metadata.color == codec_internal::SourceColor::kSrgb) {
      if (auto s = temporary.WriteBits(1, 1); !s.ok())
        return s; // default sRGB
    } else {
      // Explicit gray, D65, sRGB transfer, relative intent. Enums use
      // U32(Val(0), Val(1), BitsOffset(4,2), BitsOffset(6,18)).
      const std::array fields = {BitField{1, 0},  BitField{1, 0}, BitField{2, 1},
                                 BitField{2, 1},  BitField{1, 0}, BitField{2, 2},
                                 BitField{4, 11}, BitField{2, 1}};
      if (auto s = WriteFields(&temporary, fields); !s.ok())
        return s;
    }
    if (auto s = temporary.WriteBits(2, 0); !s.ok())
      return s; // metadata extensions
    if (auto s = temporary.WriteBits(1, 1); !s.ok())
      return s; // default transform data
  }
  if (Status status = temporary.ZeroPadToByte(); !status.ok()) {
    return status;
  }
  return AppendTemporary(writer, temporary);
}
Status WriteFrameHeader(const codec_internal::ImageMetadata &image,
                        const codec_internal::FrameMetadata &frame, BitWriter *writer) {
  if (!writer)
    return Status::InvalidArgument("Null frame-header output");
  if (auto s = codec_internal::ValidateFrameMetadata(image, frame); !s.ok())
    return s;
  if (frame.encoding == codec_internal::FrameEncoding::kModular) {
    BitWriter temporary;
    const std::array prefix = {BitField{1, 0}, BitField{2, 0}, BitField{1, 1},
                               BitField{2, 0}, BitField{1, 0}, BitField{2, 0}};
    if (auto s = WriteFields(&temporary, prefix); !s.ok())
      return s;
    if (image.alpha) {
      if (auto s = temporary.WriteBits(2, 0); !s.ok())
        return s; // extra upsampling
    }
    const std::array middle = {BitField{2, 1}, BitField{2, 0}, BitField{1, 0},
                               BitField{2, 0}};
    if (auto s = WriteFields(&temporary, middle); !s.ok())
      return s;
    if (image.alpha) {
      if (auto s = temporary.WriteBits(2, 0); !s.ok())
        return s; // replace alpha
    }
    // Final, empty name, explicit loop filter: Gaborish and EPF disabled.
    const std::array suffix = {BitField{1, 1}, BitField{2, 0}, BitField{1, 0},
                               BitField{1, 0}, BitField{2, 0}, BitField{2, 0},
                               BitField{2, 0}};
    if (auto s = WriteFields(&temporary, suffix); !s.ok())
      return s;
    return AppendTemporary(writer, temporary);
  }
  BitWriter temporary;
  const std::array<BitField, 21> fields = {{
      BitField{1, 0},                                            // not all default
      BitField{2, 0},                                            // regular frame
      BitField{1, 0},                                            // VarDCT
      BitField{2, frame.vardct.adaptive_dc_smoothing ? 0u : 2u}, // flags selector
      {frame.vardct.adaptive_dc_smoothing ? 0u : 8u,
       frame.vardct.adaptive_dc_smoothing ? 0u : 111u}, // kSkipAdaptiveDCSmoothing
      BitField{2, 0},                                   // no upsampling
      BitField{3, frame.vardct.x_qm_scale},
      BitField{3, frame.vardct.b_qm_scale},
      BitField{2, 0},                               // one pass
      BitField{1, 0},                               // no custom size or origin
      BitField{2, 0},                               // replace blend mode
      BitField{1, 1},                               // final frame
      BitField{2, 0},                               // no name
      BitField{1, frame.vardct.gaborish ? 1u : 0u}, // loop-filter all_default
      {frame.vardct.gaborish ? 0u : 1u, 0},         // gaborish off
      {frame.vardct.gaborish ? 0u : 2u,
       frame.vardct.gaborish ? 0u : 2u},    // two EPF passes
      {frame.vardct.gaborish ? 0u : 1u, 0}, // default sharpness
      {frame.vardct.gaborish ? 0u : 1u, 0}, // default weights
      {frame.vardct.gaborish ? 0u : 1u, 0}, // default sigma
      {frame.vardct.gaborish ? 0u : 2u, 0}, // loop-filter extensions
      BitField{2, 0},                       // no extensions
  }};
  if (Status status = WriteFields(&temporary, fields); !status.ok()) {
    return status;
  }
  return AppendTemporary(writer, temporary);
}

Status ComputeCommonHeaderStoragePlan(size_t prior_bits, CommonHeaderStoragePlan *out) {
  if (!out || prior_bits > std::numeric_limits<size_t>::max() - 384)
    return Status::InvalidArgument("Common header storage overflow");
  CommonHeaderStoragePlan plan;
  if (auto s = ComputeEntropyWriterStorageBound(256, &plan.temporary); !s.ok())
    return s;
  if (auto s = ComputeEntropyWriterStorageBound(prior_bits + 384, &plan.destination);
      !s.ok())
    return s;
  *out = plan;
  return Status::Ok();
}

} // namespace gjxl::codestream_internal
