// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"
namespace gjxl::modular_internal {
// Internal handoff from the RGB8 workflow: plan/tokens have canonical provenance.
// Consumes tokens; frees them before assembly. Caller owns admission/publication.
[[nodiscard]] Status EncodeModularFrame(const codec_internal::ImageMetadata &metadata,
                                        const ModularStreamPlan &layout,
                                        PreparedModularTokens &&tokens, EntropyCodingMode mode,
                                        const ModularWorkflowStoragePlan &storage,
                                        codestream_internal::CodestreamBuffer *out);
} // namespace gjxl::modular_internal
