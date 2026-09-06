// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codestream/storage.h"
#include "codestream/workflow.h"
#include "core/publication_record.h"

namespace gjxl::codestream_internal {

using OwnedEncodingSummary = resource_budget_internal::PublicationRecord<
  VarDctEncodingSummary, double, &VarDctEncodingSummary::score_history>;
using OwnedEncodingTiming = resource_budget_internal::PublicationRecord<
  VarDctEncodingTiming, VarDctEncodingAttemptTiming, &VarDctEncodingTiming::attempts>;

inline const VarDctEncodingSummary& SummaryValue(const VarDctEncodingSummary& summary) noexcept {
  return summary;
}
inline const VarDctEncodingSummary& SummaryValue(const OwnedEncodingSummary& summary) noexcept {
  return summary.value();
}

struct OwnedEncodingResult {
  CodestreamBuffer codestream;
  OwnedEncodingSummary summary;
  OwnedEncodingTiming timing;

  [[nodiscard]] Status Reclassify(resource_budget_internal::ResourceClass owner) {
    Status status = codestream.Reclassify(owner);
    if (status.ok()) status = summary.Reclassify(owner);
    if (status.ok()) status = timing.Reclassify(owner);
    return status;
  }
};

}  // namespace gjxl::codestream_internal
