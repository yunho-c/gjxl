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
  return WriteCodingStreamHeader(weighted, rct, {}, writer);
}
Status WriteCodingStreamHeader(const WeightedPredictorParameters &weighted, uint8_t rct,
                               const TransformSequence &transforms, BitWriter *writer) {
  if (!writer)
    return Status::InvalidArgument("Null Modular stream header output");
  if (!weighted.valid() || rct >= 42)
    return Status::InvalidArgument("Invalid Modular stream parameters");
  if (auto s = ValidateTransforms(transforms); !s.ok())
    return s;
  return writer->WithMaxBits(MaximumCodingStreamHeaderBits(transforms), [&]() -> Status {
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
    const size_t count = transforms.size + (rct != 0);
    if (auto s =
            count < 2 ? writer->WriteBits(2, count) : writer->WriteBits(6, 2 | ((count - 2) << 2));
        !s.ok())
      return s;
    if (rct) {
      if (auto s = writer->WriteBits(2, 0); !s.ok())
        return s; // RCT ID
      if (auto s = writer->WriteBits(5, 0); !s.ok())
        return s; // begin_c=0
      const auto type = rct;
      Status status = type == 6   ? writer->WriteBits(2, 0)
                      : type < 4  ? writer->WriteBits(4, 1 | (uint64_t{type} << 2))
                      : type < 18 ? writer->WriteBits(6, 2 | (uint64_t{type - 2u} << 2))
                                  : writer->WriteBits(8, 3 | (uint64_t{type - 10u} << 2));
      if (!status.ok())
        return status;
    }
    auto begin = [&](uint8_t c) {
      return c < 8 ? writer->WriteBits(5, uint64_t{c} << 2)
                   : writer->WriteBits(8, 1 | (uint64_t{c - 8u} << 2));
    };
    for (size_t i = 0; i < transforms.size; ++i) {
      const auto &t = transforms.entries[i];
      if (t.kind == TransformKind::kPalette) {
        if (auto s = writer->WriteBits(2, 1); !s.ok())
          return s;
        if (auto s = begin(t.begin); !s.ok())
          return s;
        if (auto s = t.count == 1   ? writer->WriteBits(2, 0)
                     : t.count == 3 ? writer->WriteBits(2, 1)
                     : t.count == 4 ? writer->WriteBits(2, 2)
                                    : writer->WriteBits(15, 3 | (uint64_t{t.count - 1u} << 2));
            !s.ok())
          return s;
        if (auto s = t.colors < 256 ? writer->WriteBits(10, uint64_t{t.colors} << 2)
                                    : writer->WriteBits(12, 1);
            !s.ok())
          return s;
        if (auto s = writer->WriteBits(6, 0); !s.ok())
          return s; // no deltas, zero predictor
      } else {
        if (auto s = writer->WriteBits(2, 2); !s.ok())
          return s;
        if (auto s = writer->WriteBits(6, 1); !s.ok())
          return s; // one explicit squeeze
        if (auto s = writer->WriteBits(2, uint64_t{t.horizontal} | (uint64_t{t.in_place} << 1));
            !s.ok())
          return s;
        if (auto s = begin(t.begin); !s.ok())
          return s;
        if (auto s = t.count < 4 ? writer->WriteBits(2, t.count - 1)
                                 : writer->WriteBits(6, 3 | (uint64_t{t.count - 4u} << 2));
            !s.ok())
          return s;
      }
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
