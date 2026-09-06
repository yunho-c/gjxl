// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "codestream/workflow.h"

namespace gjxl {

struct VarDctBatchEncodingRequest {
  ConstImage3FView linear_rgb;
  VarDctEncodingOptions options;
};

struct VarDctBatchEncodingResult {
  Status status;
  std::vector<uint8_t> codestream;
  VarDctEncodingSummary summary;
  VarDctEncodingTiming timing;
};

/// Persistent bounded-concurrency driver for independent image encodes.
///
/// Each worker executes the existing single-image encoding workflow, preserving
/// its codec decisions and atomic output behavior. Metal requests share the
/// process-wide production backend while retaining independent per-image
/// preparation and scratch. This permits CPU preparation and serialization
/// for one image to overlap another image's GPU work; it does not fuse images
/// into one Metal dispatch.
/// Encoded bytes remain internally owned until the whole result array is
/// published; this ownership boundary alone does not impose a memory limit.
///
/// One Encode call runs every request and preserves request order in results.
/// Individual failures are reported in the matching result. A successful
/// scheduler call can therefore contain failed image results. Invalid driver
/// or output arguments leave caller-visible results unchanged. Concurrent
/// Encode calls on the same driver are serialized; use one call containing all
/// available requests to expose the configured in-flight parallelism.
class VarDctBatchEncoder {
public:
  ~VarDctBatchEncoder();

  VarDctBatchEncoder(const VarDctBatchEncoder&) = delete;
  VarDctBatchEncoder& operator=(const VarDctBatchEncoder&) = delete;

  [[nodiscard]] static Status Create(
    size_t max_in_flight,
    std::unique_ptr<VarDctBatchEncoder>* encoder);

  [[nodiscard]] size_t max_in_flight() const noexcept;

  [[nodiscard]] Status Encode(
    std::span<const VarDctBatchEncodingRequest> requests,
    std::vector<VarDctBatchEncodingResult>* results);

private:
  class Impl;

  explicit VarDctBatchEncoder(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace gjxl
