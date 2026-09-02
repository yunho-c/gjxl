// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/geometry.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

namespace codestream_internal {

struct LibjxlTailOptions {
  int effort = 7;
  float butteraugli_distance = 1.0f;
  size_t thread_count = 1;
};

struct LibjxlTailStateDigest {
  uint64_t dimensions = 0;
  uint64_t strategies = 0;
  uint64_t quantizer = 0;
  uint64_t raw_quant = 0;
  uint64_t epf = 0;
  uint64_t cfl = 0;
  uint64_t quantized_dc = 0;
  uint64_t dc = 0;
  uint64_t ac_used_counts = 0;
  uint64_t ac_coefficients = 0;

  friend bool operator==(
    const LibjxlTailStateDigest&,
    const LibjxlTailStateDigest&) = default;
};

struct LibjxlTailStateAudit {
  LibjxlTailStateDigest source;
  LibjxlTailStateDigest copied;

  friend bool operator==(
    const LibjxlTailStateAudit&,
    const LibjxlTailStateAudit&) = default;
};

struct LibjxlTailProfile {
  uint64_t adapter_validation_and_copy_nanoseconds = 0;
  uint64_t context_setup_nanoseconds = 0;
  uint64_t libjxl_validation_nanoseconds = 0;
  uint64_t state_initialization_nanoseconds = 0;
  uint64_t dc_metadata_nanoseconds = 0;
  uint64_t block_context_nanoseconds = 0;
  uint64_t coefficient_order_nanoseconds = 0;
  uint64_t coefficient_tokenization_nanoseconds = 0;
  uint64_t modular_model_nanoseconds = 0;
  uint64_t histogram_model_nanoseconds = 0;
  uint64_t section_writing_nanoseconds = 0;
  uint64_t assembly_nanoseconds = 0;
  uint64_t libjxl_internal_nanoseconds = 0;
  uint64_t output_copy_nanoseconds = 0;
  uint64_t total_nanoseconds = 0;
  size_t worker_count = 1;
  bool calling_thread_participates = true;

  friend bool operator==(const LibjxlTailProfile&,
                         const LibjxlTailProfile&) = default;
};

/// Reusable allocator and thread-runner context for warm tail measurements.
/// This private type never crosses the installed GJXL API boundary.
class LibjxlTailContext {
public:
  LibjxlTailContext(const LibjxlTailContext&) = delete;
  LibjxlTailContext& operator=(const LibjxlTailContext&) = delete;
  ~LibjxlTailContext();

private:
  LibjxlTailContext() = default;

  void* implementation_ = nullptr;

  friend Status CreateLibjxlTailContext(
    size_t,
    std::unique_ptr<LibjxlTailContext>*);
  friend Status EncodeVarDctCodestreamWithLibjxlContextProfiled(
    const VarDctEncoderFrame&,
    LibjxlTailOptions,
    LibjxlTailContext&,
    std::vector<uint8_t>*,
    LibjxlTailProfile*);
};

[[nodiscard]] bool LibjxlTailExperimentAvailable() noexcept;

/// Creates the reusable warm context for exactly `thread_count` workers.
/// Failure leaves `context` unchanged.
[[nodiscard]] Status CreateLibjxlTailContext(
  size_t thread_count,
  std::unique_ptr<LibjxlTailContext>* context);

/// Copies a completed frame into libjxl's ordinary encoder structures and
/// returns field-specific pre/post-copy digests. Failure leaves `audit`
/// unchanged.
[[nodiscard]] Status AuditVarDctStateWithLibjxl(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options,
  LibjxlTailStateAudit* audit);

/// Serializes a completed frame through the pinned internal libjxl bridge.
/// Failure leaves `output` unchanged.
[[nodiscard]] Status
EncodeVarDctCodestreamWithLibjxl(const VarDctEncoderFrame &frame,
                                 LibjxlTailOptions options,
                                 std::vector<uint8_t> *output);

/// Encodes identically while returning non-overlapping elapsed phase timings.
/// Failure leaves both caller-visible outputs unchanged.
[[nodiscard]] Status EncodeVarDctCodestreamWithLibjxlProfiled(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options,
  std::vector<uint8_t>* output,
  LibjxlTailProfile* profile);

/// Profiled warm-context variant. Context construction and destruction are
/// outside `total_nanoseconds`; all frame-dependent adapter work remains in it.
[[nodiscard]] Status EncodeVarDctCodestreamWithLibjxlContextProfiled(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options,
  LibjxlTailContext& context,
  std::vector<uint8_t>* output,
  LibjxlTailProfile* profile);

/// Untimed validation helper backed by the pinned libjxl decoder. Decodes a
/// raw codestream to interleaved RGB float samples and commits atomically.
[[nodiscard]] Status DecodeCodestreamPixelsWithLibjxl(
  const std::vector<uint8_t>& codestream,
  Extent2D expected_extent,
  std::vector<float>* pixels);

} // namespace codestream_internal
} // namespace gjxl
