// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Extracted from the libjxl-tiny-derived VarDCT header writer.
#include "codestream/modular/tree_codec.h"

#include <array>
#include <limits>

namespace gjxl::modular_internal {

Status ComputeGlobalTreeStoragePlan(size_t maximum_tokens,
                                  GlobalTreeStoragePlan* out) {
  using namespace codestream_internal;
  if (out == nullptr || maximum_tokens == 0) {
    return Status::InvalidArgument("Modular tree storage inputs are invalid");
  }
  EntropyModelStoragePlan model;
  Status status = ComputeEntropyModelStoragePlan(
      EntropyCodingMode::kPrefix, kTreeContextCount, kTreeContextCount, &model);
  if (!status.ok()) return status;
  EntropyOptimizationStoragePlan search;
  status = ComputeEntropyOptimizationStoragePlan(
      {.policy = EntropyStoragePolicy::kPrefix, .tokens = maximum_tokens,
       .contexts = kTreeContextCount, .sections = 1, .return_cost = false}, &search);
  if (!status.ok()) return status;
  EntropyTokenEmissionStoragePlan tokens;
  status = ComputeEntropyTokenEmissionStoragePlan(
      EntropyCodingMode::kPrefix, maximum_tokens, &tokens);
  if (!status.ok()) return status;
  GlobalTreeStoragePlan plan;
  plan.maximum_bits = model.maximum_bits;
  if (plan.maximum_bits > std::numeric_limits<size_t>::max() - 2 ||
      tokens.maximum_bits > std::numeric_limits<size_t>::max() -
                                (plan.maximum_bits + 2) ||
      !plan.scratch.Add(search.working) ||
      !plan.scratch.Add(model.write_scratch) ||
      !plan.scratch.Add(tokens.scratch)) {
    return Status::OutOfMemory("Modular tree storage overflows");
  }
  plan.maximum_bits += 2 + tokens.maximum_bits;
  *out = plan;
  return Status::Ok();
}

Status WriteGlobalTreeInTransaction(std::span<const EntropyToken> tokens,
                                    BitWriter* writer) {
  if (writer == nullptr || tokens.empty()) {
    return Status::InvalidArgument("Modular global-tree inputs are invalid");
  }
  const std::array streams = {EntropyTokenStreamView::Interleaved(tokens)};
  EntropyCode code;
  if (Status status = OptimizeEntropyCode(
          streams, {.context_count = kTreeContextCount}, &code); !status.ok()) {
    return status;
  }
  // Global tree present, followed by the tree entropy stream's no-LZ77 flag.
  if (Status status = writer->WriteBits(1, 1); !status.ok()) return status;
  if (Status status = writer->WriteBits(1, 0); !status.ok()) return status;
  if (Status status = WriteEntropyCode(code, writer); !status.ok()) return status;
  return WriteTokenStream(tokens, code, writer);
}

}  // namespace gjxl::modular_internal
