// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <variant>

#include "gpu/image.h"

namespace gjxl {

struct PointwiseAffineCommand {
  /// Input and output must have matching float32 geometry. Exact in-place
  /// operation is supported; scale and bias must be finite.
  ConstDevicePlaneView input;
  DevicePlaneView output;
  float scale = 1.0f;
  float bias = 0.0f;
};

struct SeparableConvolutionCommand {
  /// Applies one odd 1-33 tap float32 kernel horizontally and vertically.
  /// Edges truncate and renormalize included weights. Kernel values must be
  /// finite and every included edge subset must have a nonzero sum. Exact
  /// input/output aliasing is supported through the distinct intermediate.
  ConstDevicePlaneView input;
  ConstDevicePlaneView kernel;
  DevicePlaneView intermediate;
  DevicePlaneView output;
};

struct MaximumReductionCommand {
  /// Returns the exact maximum of finite float32 logical input values.
  /// Scratch planes are contiguous one-row capacities for partial maxima.
  ConstDevicePlaneView input;
  DevicePlaneView scratch_a;
  DevicePlaneView scratch_b;
  DevicePlaneView output;
};

using PrimitiveCommand = std::variant<
  PointwiseAffineCommand,
  SeparableConvolutionCommand,
  MaximumReductionCommand>;

}  // namespace gjxl
