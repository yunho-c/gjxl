// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder/enc_file.cc and
// encoder/enc_frame.cc.

#include "codestream/headers.h"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

#include "codestream/simple_ac_context.h"

namespace gjxl {
namespace {

constexpr size_t kMaximumJxlDimension = 0x3FFFFFFFu;
constexpr size_t kContextTreeContextCount = 6;

struct BitField {
  size_t width;
  uint64_t value;
};

// Static modular context tree for the DC and AC-metadata channels. Values are
// already HybridUint inputs; only token 1 depends on the number of DC groups.
constexpr EntropyToken kContextTreeTokens[] = {
  {1, 2},   {0, 4},  {1, 1},   {0, 2},  {1, 10},   {0, 0},  {1, 1},   {0, 4},
  {1, 1},   {0, 0},  {1, 10},  {0, 94}, {1, 10},   {0, 61}, {1, 0},   {2, 0},
  {3, 0},   {4, 0},  {5, 0},   {1, 3},  {0, 0},    {1, 0},  {2, 5},   {3, 0},
  {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 10},
  {0, 382}, {1, 10}, {0, 22},  {1, 10}, {0, 13},   {1, 10}, {0, 253}, {1, 8},
  {0, 10},  {1, 8},  {0, 10},  {1, 10}, {0, 784},  {1, 10}, {0, 190}, {1, 10},
  {0, 46},  {1, 10}, {0, 10},  {1, 10}, {0, 5},    {1, 10}, {0, 29},  {1, 10},
  {0, 125}, {1, 10}, {0, 509}, {1, 8},  {0, 22},   {1, 8},  {0, 6},   {1, 8},
  {0, 22},  {1, 8},  {0, 6},   {1, 10}, {0, 1000}, {1, 10}, {0, 510}, {1, 10},
  {0, 254}, {1, 10}, {0, 126}, {1, 10}, {0, 62},   {1, 10}, {0, 30},  {1, 10},
  {0, 14},  {1, 10}, {0, 6},   {1, 10}, {0, 1},    {1, 10}, {0, 7},   {1, 10},
  {0, 21},  {1, 10}, {0, 45},  {1, 10}, {0, 93},   {1, 10}, {0, 189}, {1, 10},
  {0, 381}, {1, 10}, {0, 783}, {1, 0},  {2, 1},    {3, 0},  {4, 0},   {5, 0},
  {1, 0},   {2, 1},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 1},   {3, 0},
  {4, 0},   {5, 0},  {1, 0},   {2, 1},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
  {2, 0},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 0},  {3, 0},   {4, 0},
  {5, 0},   {1, 0},  {2, 0},   {3, 0},  {4, 0},    {5, 0},  {1, 0},   {2, 0},
  {3, 0},   {4, 0},  {5, 0},   {1, 0},  {2, 5},    {3, 0},  {4, 0},   {5, 0},
  {1, 0},   {2, 5},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 5},   {3, 0},
  {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
  {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
  {5, 0},   {1, 0},  {2, 5},   {3, 0},  {4, 0},    {5, 0},  {1, 0},   {2, 5},
  {3, 0},   {4, 0},  {5, 0},   {1, 0},  {2, 5},    {3, 0},  {4, 0},   {5, 0},
  {1, 0},   {2, 5},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 5},   {3, 0},
  {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
  {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
  {5, 0},   {1, 0},  {2, 5},   {3, 0},  {4, 0},    {5, 0},  {1, 10},  {0, 2},
  {1, 0},   {2, 5},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 5},   {3, 0},
  {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
  {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
  {5, 0},   {1, 0},  {2, 5},   {3, 0},  {4, 0},    {5, 0},  {1, 0},   {2, 5},
  {3, 0},   {4, 0},  {5, 0},   {1, 0},  {2, 5},    {3, 0},  {4, 0},   {5, 0},
  {1, 0},   {2, 5},  {3, 0},   {4, 0},  {5, 0},    {1, 0},  {2, 5},   {3, 0},
  {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
  {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
  {5, 0},   {1, 0},  {2, 5},   {3, 0},  {4, 0},    {5, 0},  {1, 0},   {2, 5},
  {3, 0},   {4, 0},  {5, 0},   {1, 10}, {0, 999},  {1, 0},  {2, 5},   {3, 0},
  {4, 0},   {5, 0},  {1, 0},   {2, 5},  {3, 0},    {4, 0},  {5, 0},   {1, 0},
  {2, 5},   {3, 0},  {4, 0},   {5, 0},  {1, 0},    {2, 5},  {3, 0},   {4, 0},
  {5, 0},
};

static_assert(std::size(kContextTreeTokens) == 313);

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

Status WriteQuantizerInternal(QuantizerParams params, BitWriter* writer) {
  if (params.global_scale == 0 ||
      params.global_scale > kMaxEncoderGlobalScale ||
      params.quant_dc == 0 || params.quant_dc > kMaxQuantDc) {
    return Status::InvalidArgument("Quantizer parameters cannot be encoded");
  }

  Status status;
  if (params.global_scale < 2049) {
    const std::array fields = {
      BitField{2, 0}, BitField{11, params.global_scale - 1}};
    status = WriteFields(writer, fields);
  } else if (params.global_scale < 4097) {
    const std::array fields = {
      BitField{2, 1}, BitField{11, params.global_scale - 2049}};
    status = WriteFields(writer, fields);
  } else if (params.global_scale < 8193) {
    const std::array fields = {
      BitField{2, 2}, BitField{12, params.global_scale - 4097}};
    status = WriteFields(writer, fields);
  } else {
    const std::array fields = {
      BitField{2, 3}, BitField{16, params.global_scale - 8193}};
    status = WriteFields(writer, fields);
  }
  if (!status.ok()) {
    return status;
  }

  if (params.quant_dc == 16) {
    return writer->WriteBits(2, 0);
  }
  if (params.quant_dc < 33) {
    const std::array fields = {
      BitField{2, 1}, BitField{5, params.quant_dc - 1}};
    return WriteFields(writer, fields);
  }
  if (params.quant_dc < 257) {
    const std::array fields = {
      BitField{2, 2}, BitField{8, params.quant_dc - 1}};
    return WriteFields(writer, fields);
  }
  const std::array fields = {
    BitField{2, 3}, BitField{16, params.quant_dc - 1}};
  return WriteFields(writer, fields);
}

Status WriteCompactBlockContextMap(BitWriter* writer) {
  EntropyCode map;
  map.context_count = codestream_internal::kSimpleBlockContextMap.size();
  map.context_map.assign(
    codestream_internal::kSimpleBlockContextMap.begin(),
    codestream_internal::kSimpleBlockContextMap.end());
  // Prefix codes are irrelevant to context-map serialization, but retaining
  // the four referenced clusters keeps the EntropyCode structurally valid.
  map.uint_configs.resize(4, kDefaultHybridUintConfig);
  map.prefix_codes.resize(4);
  return WriteContextMap(map, writer);
}

Status WriteContextTree(size_t dc_group_count, BitWriter* writer) {
  if (dc_group_count == 0 ||
      dc_group_count >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return Status::InvalidArgument("DC-group count cannot be encoded");
  }

  std::vector<EntropyToken> tokens(
    std::begin(kContextTreeTokens), std::end(kContextTreeTokens));
  tokens[1].value = PackSigned(static_cast<int32_t>(1 + dc_group_count));
  const std::array<std::vector<EntropyToken>, 1> streams = {tokens};
  EntropyCode code;
  if (Status status = OptimizeEntropyCode(
        streams, {.context_count = kContextTreeContextCount}, &code);
      !status.ok()) {
    return status;
  }
  const std::array<BitField, 2> fields = {{{1, 1}, {1, 0}}};
  if (Status status = WriteFields(writer, fields); !status.ok()) {
    return status;
  }
  if (Status status = WriteEntropyCode(code, writer); !status.ok()) {
    return status;
  }
  return WriteTokenStream(tokens, code, writer);
}

Status AllocationFailure(const char* operation) {
  return Status::OutOfMemory(operation);
}

}  // namespace

Status WriteSimpleCodestreamHeader(Extent2D frame_extent, BitWriter* writer) {
  if (writer == nullptr) {
    return Status::InvalidArgument("Codestream-header output is null");
  }
  if (frame_extent.empty() || frame_extent.width > kMaximumJxlDimension ||
      frame_extent.height > kMaximumJxlDimension) {
    return Status::InvalidArgument("Frame dimensions cannot be encoded");
  }

  BitWriter temporary;
  const std::array<BitField, 3> prefix = {{{8, 0xFF}, {8, 0x0A}, {1, 0}}};
  if (Status status = WriteFields(&temporary, prefix); !status.ok()) {
    return status;
  }
  if (Status status = WriteSize(
        static_cast<uint32_t>(frame_extent.height), &temporary);
      !status.ok()) {
    return status;
  }
  if (Status status = temporary.WriteBits(3, 0); !status.ok()) {
    return status;
  }
  if (Status status = WriteSize(
        static_cast<uint32_t>(frame_extent.width), &temporary);
      !status.ok()) {
    return status;
  }

  // Non-default metadata: float32 linear sRGB, XYB transform, no extras.
  const std::array<BitField, 19> metadata = {{
    {1, 0}, {1, 0}, {1, 1}, {2, 0}, {4, 7}, {1, 0}, {2, 0},
    {1, 1}, {1, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 1}, {1, 0},
    {2, 2}, {4, 6}, {2, 1}, {2, 0}, {1, 1},
  }};
  if (Status status = WriteFields(&temporary, metadata); !status.ok()) {
    return status;
  }
  if (Status status = temporary.ZeroPadToByte(); !status.ok()) {
    return status;
  }
  return AppendTemporary(writer, temporary);
}

Status WriteSimpleFrameHeader(
  const SimpleVarDctCodestreamProfile& profile, BitWriter* writer) {

  if (writer == nullptr) {
    return Status::InvalidArgument("Frame-header output is null");
  }
  const SimpleVarDctCodestreamProfile defaults;
  SimpleVarDctCodestreamProfile normalized = profile;
  normalized.x_qm_scale = defaults.x_qm_scale;
  normalized.b_qm_scale = defaults.b_qm_scale;
  if (!profile.valid() || normalized != defaults) {
    return Status::InvalidArgument(
      "Profile cannot be represented by the simple frame header");
  }

  BitWriter temporary;
  const std::array<BitField, 15> fields = {{
    {1, 0},   // not all default
    {2, 0},   // regular frame
    {1, 0},   // VarDCT
    {2, 2},   // flags selector
    {8, 111}, // kSkipAdaptiveDCSmoothing
    {2, 0},   // no upsampling
    {3, profile.x_qm_scale},
    {3, profile.b_qm_scale},
    {2, 0},   // one pass
    {1, 0},   // no custom size or origin
    {2, 0},   // replace blend mode
    {1, 1},   // final frame
    {2, 0},   // no name
    {1, 1},   // default loop filter
    {2, 0},   // no extensions
  }};
  if (Status status = WriteFields(&temporary, fields); !status.ok()) {
    return status;
  }
  return AppendTemporary(writer, temporary);
}

Status WriteSimpleQuantizer(QuantizerParams params, BitWriter* writer) {
  if (writer == nullptr) {
    return Status::InvalidArgument("Quantizer output is null");
  }
  BitWriter temporary;
  if (Status status = WriteQuantizerInternal(params, &temporary); !status.ok()) {
    return status;
  }
  return AppendTemporary(writer, temporary);
}

Status WriteSimpleDcGlobal(
  QuantizerParams params, size_t dc_group_count,
  const EntropyCode& dc_code, BitWriter* writer) {

  if (writer == nullptr) {
    return Status::InvalidArgument("DC-global output is null");
  }
  try {
    BitWriter temporary;
    if (Status status = temporary.WriteBits(1, 1); !status.ok()) {
      return status;
    }
    if (Status status = WriteQuantizerInternal(params, &temporary);
        !status.ok()) {
      return status;
    }
    const std::array<BitField, 2> block_context = {{{1, 0}, {16, 0}}};
    if (Status status = WriteFields(&temporary, block_context); !status.ok()) {
      return status;
    }
    if (Status status = WriteCompactBlockContextMap(&temporary); !status.ok()) {
      return status;
    }
    if (Status status = temporary.WriteBits(1, 1); !status.ok()) {
      return status;
    }
    if (Status status = WriteContextTree(dc_group_count, &temporary);
        !status.ok()) {
      return status;
    }
    if (Status status = temporary.WriteBits(1, 0); !status.ok()) {
      return status;
    }
    if (Status status = WriteEntropyCode(dc_code, &temporary); !status.ok()) {
      return status;
    }
    return AppendTemporary(writer, temporary);
  } catch (const std::bad_alloc&) {
    return AllocationFailure("DC-global allocation failed");
  } catch (const std::length_error&) {
    return AllocationFailure("DC-global allocation is too large");
  }
}

Status WriteSimpleAcGlobal(
  size_t ac_group_count, const EntropyCode& ac_code, BitWriter* writer) {

  if (writer == nullptr) {
    return Status::InvalidArgument("AC-global output is null");
  }
  if (ac_group_count == 0) {
    return Status::InvalidArgument("AC-group count is zero");
  }

  BitWriter temporary;
  if (Status status = temporary.WriteBits(1, 1); !status.ok()) {
    return status;
  }
  const size_t histogram_bits = std::bit_width(ac_group_count - 1);
  if (histogram_bits > BitWriter::kMaxBitsPerWrite) {
    return Status::InvalidArgument("AC-group count cannot be encoded");
  }
  if (histogram_bits != 0) {
    if (Status status = temporary.WriteBits(histogram_bits, 0); !status.ok()) {
      return status;
    }
  }
  const std::array<BitField, 3> fields = {{{2, 3}, {13, 0}, {1, 0}}};
  if (Status status = WriteFields(&temporary, fields); !status.ok()) {
    return status;
  }
  if (Status status = WriteEntropyCode(ac_code, &temporary); !status.ok()) {
    return status;
  }
  return AppendTemporary(writer, temporary);
}

}  // namespace gjxl
