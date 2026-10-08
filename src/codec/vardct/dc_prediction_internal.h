// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "codec/dc_prediction.h"
#include "codec/modular/prediction.h"

namespace gjxl::vardct_internal {

// Public VarDCT options retain their identity and defaults. Callers validate
// them before selecting a kernel; an unknown mode never becomes weighted.
constexpr modular_internal::Predictor ModularDcPredictor(VarDctDcPrediction mode) {
  switch (mode) {
  case VarDctDcPrediction::kGradient:
    return modular_internal::Predictor::kGradient;
  case VarDctDcPrediction::kWeighted:
    return modular_internal::Predictor::kWeighted;
  }
  return modular_internal::Predictor::kInvalid;
}

}  // namespace gjxl::vardct_internal
