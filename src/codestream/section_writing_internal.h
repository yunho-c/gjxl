// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cstdlib>
#include <cstring>
#include <limits>
#include "codestream/entropy_readiness_internal.h"
#include "codestream/parallel_sections_internal.h"

namespace gjxl::codestream_internal {

// Favor latency for eligible single-image calls. Configure process-wide
// controls before encoding, not while calls are active. Storage planning covers
// both schedules regardless of these controls.
inline bool SectionWritingOverlapRequested() {
  if (const char* value = std::getenv("GJXL_SECTION_WRITE_OVERLAP")) {
    if (std::strcmp(value, "0") == 0) return false;
  }
#ifdef GJXL_TOKENIZATION_EXPERIMENT
  if (const char* value = std::getenv("GJXL_EXPERIMENT_SECTION_WRITE")) {
    if (std::strcmp(value, "0") == 0) return false;
  }
#endif
  return true;
}

inline bool CanOverlapSectionWriting(size_t ac_groups) {
  return SectionWritingOverlapRequested() && !entropy_readiness_in_batch &&
      thread_budget_internal::HasCpuParticipation() &&
      !thread_budget_internal::InExplicitParallelScope() &&
      DesiredSectionParticipants(ac_groups) > 1;
}

// One dynamic queue preserves the previous maximum worker count. Global model
// headers must finish before entry; each job owns a distinct destination writer.
// DC jobs precede AC jobs for deterministic status selection after every join.
template <typename DcFunction, typename AcFunction>
Status RunSectionWritingTasks(size_t dc_groups, size_t ac_groups,
                             DcFunction&& dc, AcFunction&& ac) {
  if (dc_groups == 0 || ac_groups == 0 ||
      dc_groups > std::numeric_limits<size_t>::max() - ac_groups) {
    return Status::InvalidArgument("Section-writing task counts are invalid");
  }
  return RunParallelSections(dc_groups + ac_groups, [&](size_t index) {
    return index < dc_groups ? dc(index) : ac(index - dc_groups);
  }, DesiredSectionParticipants(std::max(dc_groups, ac_groups)));
}

}  // namespace gjxl::codestream_internal
