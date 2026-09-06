// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codestream/batch_workflow.h"
#include "codestream/storage.h"
#include "codestream/encoding_result_internal.h"

namespace gjxl::codestream_internal {

// Synchronous observation at the all-workers-complete, not-yet-published
// boundary. The caller owns the hook/context until Encode returns. This does
// not alter scheduling and is deliberately absent from the public batch API.
struct BatchPublicationObserverForTesting {
  void* context = nullptr;
  void (*observe)(void*, std::span<const VarDctBatchEncodingResult>,
                  std::span<const OwnedEncodingResult>) noexcept = nullptr;
};
inline thread_local BatchPublicationObserverForTesting batch_publication_observer_for_testing;

}  // namespace gjxl::codestream_internal
