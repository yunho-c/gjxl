// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/histogram_tree_learning.h"
#include "reference_capture.h"

namespace reference_learning {
gjxl::modular_internal::HistogramTrainingSamples
NativeSamples(const jxl::TreeSamples &reference);
struct NativeFitResult {
  Timing time;
  size_t scratch_bound_bytes = 0;
  size_t scratch_peak_bytes = 0;
  size_t sample_backing_bytes = 0;
  gjxl::modular_internal::DecisionTree tree;
};
// Conversion/cloning precedes the timer, as in Replay's reference sample copy.
// Native admission/row validation and all fitting allocations are timed.
NativeFitResult FitNative(const Fixture &fixture, bool track_storage = false,
                         bool resumable = false);
// Deterministic independent sample generation, fitted by both implementations.
size_t RunNativeDifferentialTests();
} // namespace reference_learning
