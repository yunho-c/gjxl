// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/stream_encoder.h"

#include "codestream/entropy_internal.h"

namespace gjxl::modular_internal {

Status WriteStreamHeader(const ModularStreamHeader& header, BitWriter* writer) {
  if (writer == nullptr || !header.supported()) {
    return Status::InvalidArgument("Unsupported Modular stream header");
  }
  // use_global_tree=1, default weighted parameters=1, transform count=0 (2 bits).
  return writer->WriteBits(kStreamHeaderBits, 3);
}

Status WriteGlobalModelInTransaction(const EntropyCode& code, BitWriter* writer) {
  if (writer == nullptr) {
    return Status::InvalidArgument("Modular global-model writer is null");
  }
  if (Status status = writer->WriteBits(1, 0); !status.ok()) return status;
  return WriteEntropyCode(code, writer);
}

Status WriteStreamTokensWithValidatedModel(
    EntropyTokenStreamView tokens, const EntropyCode& code, BitWriter* writer) {
  return codestream_internal::WriteValidatedTokenStream(tokens, code, writer);
}

}  // namespace gjxl::modular_internal
