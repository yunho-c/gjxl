// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder/enc_file.cc and
// encoder/enc_frame.cc.

#include "codestream/headers.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>

#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/dc_group.h"
#include "codestream/simple_ac_context.h"
#include "codestream/serializer_storage_plan.h"

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

Status WriteCoefficientOrderMask(uint16_t mask, BitWriter* writer) {
  if (mask == 0x5F) {
    return writer->WriteBits(2, 0);
  }
  if (mask == 0x13) {
    return writer->WriteBits(2, 1);
  }
  if (mask == 0) {
    return writer->WriteBits(2, 2);
  }
  if (mask >=
      (uint16_t{1} << codestream_internal::kSimpleCoefficientOrderCount)) {
    return Status::InvalidArgument(
      "Coefficient-order mask cannot be encoded");
  }
  const std::array<BitField, 2> fields = {{{2, 3}, {13, mask}}};
  return WriteFields(writer, fields);
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

Status WriteQuantizationThreshold(uint32_t threshold, BitWriter* writer) {
  if (threshold == 0 || threshold > 255) {
    return Status::InvalidArgument(
      "Block-context quantization threshold is invalid");
  }
  const uint32_t value = threshold - 1;
  if (value < 4) {
    const std::array fields = {BitField{2, 0}, BitField{2, value}};
    return WriteFields(writer, fields);
  }
  if (value < 12) {
    const std::array fields = {BitField{2, 1}, BitField{3, value - 4}};
    return WriteFields(writer, fields);
  }
  if (value < 44) {
    const std::array fields = {BitField{2, 2}, BitField{5, value - 12}};
    return WriteFields(writer, fields);
  }
  const std::array fields = {BitField{2, 3}, BitField{8, value - 44}};
  return WriteFields(writer, fields);
}

Status WriteBlockContextMap(
  const SimpleBlockContextMap& block_context_map,
  BitWriter* writer) {

  Status status = ValidateSimpleBlockContextMap(block_context_map);
  if (!status.ok()) {
    return status;
  }
  // The JPEG XL default map has a dedicated one-bit representation. The
  // simple four-context map and adaptive maps use the general representation.
  if (codestream_internal::IsJxlDefaultBlockContextMap(block_context_map)) {
    return writer->WriteBits(1, 1);
  }
  if (Status write = writer->WriteBits(1, 0); !write.ok()) {
    return write;
  }
  // This profile does not split block contexts by quantized DC.
  const std::array<BitField, 3> dc_threshold_counts = {{
    {4, 0}, {4, 0}, {4, 0},
  }};
  if (Status write = WriteFields(writer, dc_threshold_counts); !write.ok()) {
    return write;
  }
  if (Status write = writer->WriteBits(
        4, block_context_map.qf_thresholds.size());
      !write.ok()) {
    return write;
  }
  for (uint32_t threshold : block_context_map.qf_thresholds) {
    if (Status write = WriteQuantizationThreshold(threshold, writer);
        !write.ok()) {
      return write;
    }
  }
  EntropyCode map;
  map.context_count = block_context_map.context_map.size();
  map.context_map = block_context_map.context_map;
  // Prefix codes are irrelevant to context-map serialization, but retaining
  // the referenced clusters keeps the EntropyCode structurally valid.
  map.uint_configs.resize(
    block_context_map.num_contexts, kDefaultHybridUintConfig);
  map.prefix_codes.resize(block_context_map.num_contexts);
  return WriteContextMap(map, writer);
}

Status WriteContextTree(size_t dc_group_count, BitWriter* writer) {
  if (dc_group_count == 0 ||
      dc_group_count >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return Status::InvalidArgument("DC-group count cannot be encoded");
  }

  auto tokens = std::to_array(kContextTreeTokens);
  tokens[1].value = PackSigned(static_cast<int32_t>(1 + dc_group_count));
  const std::array streams = {EntropyTokenStreamView::Interleaved(tokens)};
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

Status codestream_internal::ComputeSerializerHeaderStoragePlan(
  size_t ac_groups, size_t dc_groups, const BlockContextMapStoragePlan& maps,
  size_t order_tokens, SerializerHeaderStoragePlan* out) {
  if (out == nullptr || ac_groups == 0 || dc_groups == 0 ||
      dc_groups >= static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      maps.maximum_map_entries == 0 || maps.maximum_block_contexts == 0 ||
      maps.maximum_block_contexts > 16 || maps.maximum_thresholds > 1) {
    return Status::InvalidArgument("Serializer header storage inputs are invalid");
  }
  const auto overflow = [] {
    return Status::OutOfMemory("Serializer header storage overflows");
  };
  const auto add_bits = [](size_t n, size_t* total) {
    if (n > std::numeric_limits<size_t>::max() - *total) return false;
    *total += n;
    return true;
  };
  const auto writer = [](size_t bits, HostStorageBound* bound) {
    HostStorageBound scratch;
    const Status status = ComputeEntropyWriterStorageBound(bits, &scratch);
    return status.ok() && bound->Add(scratch);
  };
  const auto either_model = [](size_t contexts, EntropyModelStoragePlan* model) {
    EntropyModelStoragePlan prefix, ans;
    const size_t clusters = std::min(contexts, kMaximumPrefixClusters);
    Status status = ComputeEntropyModelStoragePlan(
      EntropyCodingMode::kPrefix, contexts, clusters, &prefix);
    if (!status.ok()) return status;
    status = ComputeEntropyModelStoragePlan(
      EntropyCodingMode::kAns, contexts, clusters, &ans);
    if (!status.ok()) return status;
    *model = {
      .maximum_bits = std::max(prefix.maximum_bits, ans.maximum_bits),
      .owned = {std::max(prefix.owned.retained_bytes, ans.owned.retained_bytes),
                std::max(prefix.owned.peak_bytes, ans.owned.peak_bytes)},
      .write_scratch = {
        std::max(prefix.write_scratch.retained_bytes, ans.write_scratch.retained_bytes),
        std::max(prefix.write_scratch.peak_bytes, ans.write_scratch.peak_bytes)},
    };
    return Status::Ok();
  };
  SerializerHeaderStoragePlan plan;
  // Maximum file header: 17 prefix + 32 height + 3 ratio + 32 width +
  // 33 metadata = 117 bits, padded to 120. The frame header is 33 bits.
  plan.frame_prefix_bits = 120 + 33;
  if (!writer(120, &plan.frame_scratch) ||
      !writer(33, &plan.frame_scratch)) return overflow();
  Status status = either_model(kSimpleDcContextCount, &plan.dc_model);
  if (!status.ok()) return status;
  status = either_model(maps.maximum_ac_contexts, &plan.ac_model);
  if (!status.ok()) return status;
  status = either_model(kSimplePermutationContextCount, &plan.order_model);
  if (!status.ok()) return status;

  EntropyModelStoragePlan map_model, tree_model;
  status = ComputeEntropyModelStoragePlan(
    EntropyCodingMode::kPrefix, maps.maximum_map_entries,
    maps.maximum_block_contexts, &map_model);
  if (!status.ok()) return status;
  status = ComputeEntropyModelStoragePlan(
    EntropyCodingMode::kPrefix, kContextTreeContextCount,
    kContextTreeContextCount, &tree_model);
  if (!status.ok()) return status;
  EntropyOptimizationStoragePlan tree_search;
  status = ComputeEntropyOptimizationStoragePlan(
    {.policy = EntropyStoragePolicy::kPrefix,
     .tokens = std::size(kContextTreeTokens), .contexts = kContextTreeContextCount,
     .sections = 1, .return_cost = false}, &tree_search);
  if (!status.ok()) return status;
  EntropyTokenEmissionStoragePlan tree_tokens, order_emission;
  status = ComputeEntropyTokenEmissionStoragePlan(
    EntropyCodingMode::kPrefix, std::size(kContextTreeTokens), &tree_tokens);
  if (!status.ok()) return status;
  status = ComputeEntropyTokenEmissionStoragePlan(
    EntropyCodingMode::kAns, order_tokens, &order_emission);
  if (!status.ok()) return status;
  // Quantizer <=36 bits; general block map <=17 flags/count bits plus 10 per
  // threshold. The full map-model bound safely includes its context-map part.
  plan.dc_global_bits = 1 + 36 + 17 + 10 * maps.maximum_thresholds + 1 + 2 + 1;
  for (size_t bits : {map_model.maximum_bits, tree_model.maximum_bits,
                      tree_tokens.maximum_bits, plan.dc_model.maximum_bits}) {
    if (!add_bits(bits, &plan.dc_global_bits)) return overflow();
  }
  if (!plan.dc_global_scratch.Add(map_model.owned) ||
      !plan.dc_global_scratch.Add(map_model.write_scratch) ||
      !plan.dc_global_scratch.Add(tree_search.working) ||
      !plan.dc_global_scratch.Add(tree_model.write_scratch) ||
      !plan.dc_global_scratch.Add(tree_tokens.scratch) ||
      !plan.dc_global_scratch.Add(plan.dc_model.write_scratch) ||
      !writer(plan.dc_global_bits, &plan.dc_global_scratch)) return overflow();

  const size_t histogram_bits = std::bit_width(ac_groups - 1);
  if (histogram_bits > BitWriter::kMaxBitsPerWrite) {
    return Status::InvalidArgument("AC histogram count cannot be serialized");
  }
  plan.ac_global_bits = 1 + histogram_bits + 15 + 1;
  if (!add_bits(plan.ac_model.maximum_bits, &plan.ac_global_bits) ||
      !plan.ac_global_scratch.Add(plan.ac_model.write_scratch)) return overflow();
  if (order_tokens != 0 &&
      (!add_bits(1, &plan.ac_global_bits) ||
       !add_bits(plan.order_model.maximum_bits, &plan.ac_global_bits) ||
       !add_bits(order_emission.maximum_bits, &plan.ac_global_bits) ||
       !plan.ac_global_scratch.Add(plan.order_model.write_scratch) ||
       !plan.ac_global_scratch.Add(order_emission.scratch))) return overflow();
  if (!writer(plan.ac_global_bits, &plan.ac_global_scratch)) return overflow();
  *out = plan;
  return Status::Ok();
}

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
  const SimpleBlockContextMap& block_context_map,
  const EntropyCode& dc_code,
  BitWriter* writer) {

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
    if (Status status = WriteBlockContextMap(
          block_context_map, &temporary);
        !status.ok()) {
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
  } catch (const resource_budget_internal::ManagedAllocationFailure& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return AllocationFailure("DC-global allocation failed");
  } catch (const std::length_error&) {
    return AllocationFailure("DC-global allocation is too large");
  }
}

Status WriteSimpleDcGlobal(
  QuantizerParams params, size_t dc_group_count,
  const EntropyCode& dc_code, BitWriter* writer) {
  return WriteSimpleDcGlobal(
    params, dc_group_count, DefaultSimpleBlockContextMap(), dc_code, writer);
}

Status WriteSimpleAcGlobal(
  size_t ac_group_count,
  uint16_t used_order_mask,
  std::span<const EntropyToken> order_tokens,
  const EntropyCode* order_code,
  const EntropyCode& ac_code,
  BitWriter* writer) {

  if (writer == nullptr) {
    return Status::InvalidArgument("AC-global output is null");
  }
  if (ac_group_count == 0) {
    return Status::InvalidArgument("AC-group count is zero");
  }
  if ((used_order_mask == 0) != order_tokens.empty() ||
      (used_order_mask == 0) != (order_code == nullptr)) {
    return Status::InvalidArgument(
      "Coefficient-order AC-global state is inconsistent");
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
  if (Status status = WriteCoefficientOrderMask(
        used_order_mask, &temporary);
      !status.ok()) {
    return status;
  }
  if (used_order_mask != 0) {
    if (Status status = temporary.WriteBits(1, 0); !status.ok()) {
      return status;
    }
    if (Status status = WriteEntropyCode(*order_code, &temporary);
        !status.ok()) {
      return status;
    }
    if (Status status = WriteTokenStream(
          order_tokens, *order_code, &temporary);
        !status.ok()) {
      return status;
    }
  }
  if (Status status = temporary.WriteBits(1, 0); !status.ok()) {
    return status;
  }
  if (Status status = WriteEntropyCode(ac_code, &temporary); !status.ok()) {
    return status;
  }
  return AppendTemporary(writer, temporary);
}

Status WriteSimpleAcGlobal(
  size_t ac_group_count, const EntropyCode& ac_code, BitWriter* writer) {
  return WriteSimpleAcGlobal(
    ac_group_count, 0, {}, nullptr, ac_code, writer);
}

}  // namespace gjxl
