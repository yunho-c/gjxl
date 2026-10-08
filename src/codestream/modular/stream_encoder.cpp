// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/stream_encoder.h"

#include "codestream/entropy_internal.h"

namespace gjxl::modular_internal {

Status WriteStreamHeader(const ModularStreamHeader &header, BitWriter *writer) {
  if (writer == nullptr || !header.supported()) {
    return Status::InvalidArgument("Unsupported Modular stream header");
  }
  // use_global_tree=1, default weighted parameters=1, transform count=0 (2 bits).
  return writer->WriteBits(kStreamHeaderBits, 3);
}

Status WriteCodingStreamHeader(const WeightedPredictorParameters &weighted, uint8_t rct,
                               BitWriter *writer) {
  if (!writer)
    return Status::InvalidArgument("Null Modular stream header output");
  if (!weighted.valid() || rct >= 42)
    return Status::InvalidArgument("Invalid Modular stream parameters");
  return writer->WithMaxBits(kMaximumCodingStreamHeaderBits, [&]() -> Status {
    if (auto s = writer->WriteBits(1, 1); !s.ok())
      return s;
    const bool defaults = weighted == WeightedPredictorParameters{};
    if (auto s = writer->WriteBits(1, defaults); !s.ok())
      return s;
    if (!defaults) {
      for (auto p : weighted.coefficients)
        if (auto s = writer->WriteBits(5, p); !s.ok())
          return s;
      for (auto w : weighted.weights)
        if (auto s = writer->WriteBits(4, w); !s.ok())
          return s;
    }
    if (auto s = writer->WriteBits(2, rct ? 1 : 0); !s.ok())
      return s;
    if (rct) {
      if (auto s = writer->WriteBits(2, 0); !s.ok())
        return s; // RCT ID
      if (auto s = writer->WriteBits(5, 0); !s.ok())
        return s; // begin_c=0
      const auto type = rct;
      if (type == 6)
        return writer->WriteBits(2, 0);
      if (type < 4)
        return writer->WriteBits(4, 1 | (uint64_t{type} << 2));
      if (type < 18)
        return writer->WriteBits(6, 2 | (uint64_t{type - 2u} << 2));
      return writer->WriteBits(8, 3 | (uint64_t{type - 10u} << 2));
    }
    return Status::Ok();
  });
}

Status WriteGlobalModelInTransaction(const EntropyCode &code, BitWriter *writer) {
  if (writer == nullptr) {
    return Status::InvalidArgument("Modular global-model writer is null");
  }
  if (Status status = writer->WriteBits(1, 0); !status.ok())
    return status;
  return WriteEntropyCode(code, writer);
}

Status WriteStreamTokensWithValidatedModel(EntropyTokenStreamView tokens, const EntropyCode &code,
                                           BitWriter *writer) {
  return codestream_internal::WriteValidatedTokenStream(tokens, code, writer);
}

} // namespace gjxl::modular_internal
