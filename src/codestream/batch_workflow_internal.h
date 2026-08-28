// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "codestream/batch_workflow.h"
#include "codestream/workflow_internal.h"

namespace gjxl::codestream_internal {

struct VarDctBatchEncodingStageProfile {
  size_t worker_index = 0;
  VarDctEncodingStageProfile encoding;
};

class VarDctBatchProfileAccess {
public:
  [[nodiscard]] static Status Encode(
    VarDctBatchEncoder& encoder,
    std::span<const VarDctBatchEncodingRequest> requests,
    std::vector<VarDctBatchEncodingResult>* results,
    std::vector<VarDctBatchEncodingStageProfile>* profiles);
};

[[nodiscard]] Status EncodeVarDctBatchProfiled(
  VarDctBatchEncoder& encoder,
  std::span<const VarDctBatchEncodingRequest> requests,
  std::vector<VarDctBatchEncodingResult>* results,
  std::vector<VarDctBatchEncodingStageProfile>* profiles);

}  // namespace gjxl::codestream_internal
