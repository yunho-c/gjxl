// SPDX-License-Identifier: Apache-2.0
// Diagnostic hook compiled only by GJXL_BUILD_FRONTIER_EXPERIMENT.
#pragma once
#include "codec/ac_strategy_search_internal.h"
namespace gjxl::frontier_experiment {
Status Capture(Extent2D, ConstPlaneF32View, const ColorCorrelationMap&,
               AcStrategySearchOptions,
               const ac_strategy_internal::CandidateCostTableView&,
               const AcStrategyGrid&);
Status SelectDiagnostic(AcStrategySearchOptions,
                       const ac_strategy_internal::CandidateCostTableView&,
                       AcStrategyGrid*);
}
