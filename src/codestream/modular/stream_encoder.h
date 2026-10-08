// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "codestream/entropy.h"
#include "codec/modular/prediction.h"
#include "codec/modular/transform/transform.h"
#include "codestream/modular/stream_types.h"

namespace gjxl::modular_internal {

// Atomic, allocation bounded by the existing BitWriter. No adapter prefixes.
[[nodiscard]] Status WriteStreamHeader(const ModularStreamHeader &header, BitWriter *writer);

inline constexpr size_t kMaximumCodingStreamHeaderBits = 72;
[[nodiscard]] Status WriteCodingStreamHeader(const WeightedPredictorParameters &weighted,
                                             uint8_t rct, BitWriter *writer);
inline constexpr size_t MaximumCodingStreamHeaderBits(const TransformSequence &transforms) {
  return kMaximumCodingStreamHeaderBits + (transforms.size ? 8 + 64 * transforms.size : 0);
}
[[nodiscard]] Status WriteCodingStreamHeader(const WeightedPredictorParameters &weighted,
                                             uint8_t rct, const TransformSequence &transforms,
                                             BitWriter *writer);
// Internal composition helper: the caller owns an atomic temporary/allotment
// and handles allocation exceptions. Writes no-LZ77 plus the global model.
// Scratch is WriteEntropyCode's existing model-emission plan; no model copy.
[[nodiscard]] Status WriteGlobalModelInTransaction(const EntropyCode &code, BitWriter *writer);

// Both inputs are borrowed. The global model must have been serialized (and
// validated) before this call. No per-stream rebuilding or model-wide ANS
// revalidation. Emission uses the shared entropy token-emission storage plan.
[[nodiscard]] Status WriteStreamTokensWithValidatedModel(EntropyTokenStreamView tokens,
                                                         const EntropyCode &code,
                                                         BitWriter *writer);

} // namespace gjxl::modular_internal
