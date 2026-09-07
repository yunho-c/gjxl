// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/execution_domain.h"
#include "gjxl/gjxl.h"

namespace gjxl {
/// Optional C++ bridge (link gjxl::c as well as the C++ codec target). Both
/// handles retain the very same domain, not independent copies of its limit.
/// Null selects the shared default. Failure preserves the empty output handle.
[[nodiscard]] GJXL_API Status CreateCExecutionDomain(std::shared_ptr<const ExecutionDomain> domain,
                                                     GJXLExecutionDomain **out);
/// Retains a C handle's domain independently of that handle/context lifetime.
/// Null returns the process-wide default.
[[nodiscard]] GJXL_API std::shared_ptr<const ExecutionDomain>
RetainExecutionDomain(const GJXLExecutionDomain *domain);
} // namespace gjxl
