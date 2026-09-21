// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/resource_budget.h"

namespace gjxl::cuda_internal {
// Visits only existing pools, without initializing a device. A null budget
// releases all idle backing; otherwise only that domain's tickets are freed.
[[nodiscard]] Status TrimCudaPreparationCaches(
    const resource_budget_internal::ResourceBudget *budget = nullptr);
} // namespace gjxl::cuda_internal
