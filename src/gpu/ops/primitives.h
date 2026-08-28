// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>
#include <span>
#include <variant>

#include "gpu/backend.h"
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

struct Symmetric5ConvolutionWeights {
  float distance0 = 0.0f;
  float distance1 = 0.0f;
  float distance2 = 0.0f;
  float distance4 = 0.0f;
  float distance8 = 0.0f;
  float distance5 = 0.0f;
};

struct Symmetric5ConvolutionCommand {
  /// Applies the codec's mirrored-boundary symmetric 5x5 convolution.
  /// Input and output must be distinct float32 planes with equal geometry.
  ConstDevicePlaneView input;
  DevicePlaneView output;
  Symmetric5ConvolutionWeights weights;
};

struct MaximumReductionCommand {
  /// Returns the exact maximum of finite float32 logical input values.
  /// Scratch planes are contiguous one-row capacities for partial maxima.
  ConstDevicePlaneView input;
  DevicePlaneView scratch_a;
  DevicePlaneView scratch_b;
  DevicePlaneView output;
};

using ImagePrimitiveCommand = std::variant<
  PointwiseAffineCommand,
  SeparableConvolutionCommand,
  Symmetric5ConvolutionCommand,
  MaximumReductionCommand>;

/// Optional capability for a fixed set of reusable image primitives.
class GpuImagePrimitives {
public:
  virtual ~GpuImagePrimitives() = default;

  /// Validates and enqueues one non-empty dependent sequence. Success returns
  /// a non-null submission; failure resets the output and submits no work.
  virtual Status SubmitImagePrimitiveSequence(
    std::span<const ImagePrimitiveCommand> commands,
    std::unique_ptr<GpuSubmission>* submission) = 0;
};

[[nodiscard]] inline GpuImagePrimitives* QueryGpuImagePrimitives(
  GpuBackend& backend) noexcept {

  return dynamic_cast<GpuImagePrimitives*>(&backend);
}

}  // namespace gjxl
