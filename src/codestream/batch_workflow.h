// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "codestream/workflow.h"

namespace gjxl {

struct VarDctBatchEncodingRequest {
  ConstImage3FView linear_rgb;
  VarDctEncodingOptions options;
};

/// Per-image calling-thread wall spans, not CPU time or GPU execution time.
/// Arrival is the beginning of this batch's Encode call, before driver locking.
/// Ready means the image result is internally retained and optional cache trim
/// has finished; the public result array is published only after all images.
struct VarDctBatchSchedulingTiming {
  /// Arrival through successful initial image CPU admission. Includes driver
  /// queueing, preflight, memory admission, work-slot wait and initial CPU wait.
  uint64_t queue_nanoseconds = 0;
  /// Initial image CPU admission through internal readiness, including GPU and
  /// worker waits, CPU resumption, prepared teardown and result retention/trim.
  uint64_t service_nanoseconds = 0;
  /// Arrival through internal readiness; exactly queue + service.
  uint64_t ready_nanoseconds = 0;
  /// False for errors resolved before image CPU admission. In that case service
  /// is zero and queue equals ready. Whole-batch failures publish no new timing.
  bool cpu_admitted = false;
};

struct VarDctBatchEncodingResult {
  Status status;
  std::vector<uint8_t> codestream;
  VarDctEncodingSummary summary;
  VarDctEncodingTiming timing;
  VarDctBatchSchedulingTiming scheduling;
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
/// All requests must share one execution domain (null means the shared default).
/// Admission reserves all retained results and at least the largest work slot
/// before allocating or encoding; a finite limit may reduce in-flight workers,
/// never effort or candidates. An infeasible batch or terminal planner violation
/// leaves the entire caller-visible result array unchanged.
class VarDctBatchEncoder {
public:
  ~VarDctBatchEncoder();

  VarDctBatchEncoder(const VarDctBatchEncoder&) = delete;
  VarDctBatchEncoder& operator=(const VarDctBatchEncoder&) = delete;

  [[nodiscard]] static Status Create(
    size_t max_in_flight,
    std::unique_ptr<VarDctBatchEncoder>* encoder);

  [[nodiscard]] size_t max_in_flight() const noexcept;

  /// Permanently close this driver, drain the active Encode call (including
  /// admission waits), and join its workers. New calls and calls queued behind
  /// the active call return kUnavailable without changing their result arrays.
  /// Idempotent and safe alongside Encode/Shutdown calls while the object stays
  /// alive. This does not cancel active work or release another caller's held
  /// resources: admission must still become possible for a drain to finish.
  /// All external calls must return before the object is destroyed. Do not
  /// invoke Shutdown from work whose completion that same driver is awaiting.
  void Shutdown() noexcept;

  [[nodiscard]] Status Encode(
    std::span<const VarDctBatchEncodingRequest> requests,
    std::vector<VarDctBatchEncodingResult>* results);

private:
  class Impl;

  explicit VarDctBatchEncoder(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace gjxl
