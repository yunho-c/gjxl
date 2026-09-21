// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "codec/vardct_frame.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

/// Borrowed, target-invariant search inputs. Both use the padded source
/// geometry; the XYB reference is taken before inverse Gaborish.
struct EpfSharpnessSearchReference {
  ConstImage3FView original_opsin;
  ConstPlaneF32View pixel_mask;
};

struct EpfSharpnessSearchConfig {
  std::array<uint8_t, 3> candidates{};
  size_t count = 0;
  float no_smoothing_bias = 1.0f;
};

/// Candidate values and distortion bias from libjxl's ComputeARHeuristics.
/// The caller decides whether the effort/distance/EPF policy enables search.
[[nodiscard]] Status MakeEpfSharpnessSearchConfig(
    float butteraugli_target, EpfSharpnessSearchConfig* config);

/// Mask-squared, channel-weighted XYB error for each clipped 8x8 block.
/// Images cover the actual source extent, not its transform padding. Mask
/// and output may have padded row strides. Failure preserves output.
[[nodiscard]] Status ComputeEpfBlockErrors(
    ConstImage3FView original, ConstImage3FView reconstructed,
    ConstPlaneF32View pixel_mask, PlaneF32View errors);

/// Selects from the candidate error planes in configuration order. Matches
/// the effective second pass of the pinned libjxl selector, including its
/// integer histogram division and first-candidate tie breaking. No image
/// reconstruction or filtering occurs here. Failure preserves output.
[[nodiscard]] Status SelectEpfSharpnessFromErrors(
    const std::array<ConstPlaneF32View, 3>& errors, float butteraugli_target,
    PlaneU8View sharpness);

/// CPU reference search over one fixed coefficient frame. Reconstructs once,
/// shares Gaborish across candidates, and filters each candidate globally.
/// Inputs are the original padded XYB (before inverse Gaborish) and its initial
/// blurred pixel mask. Only actual source pixels contribute to block errors.
/// Disabled EPF or distance below 0.5 produces the default map (4).
/// Does not modify coefficients or the frame; failure preserves sharpness.
[[nodiscard]] Status SearchEpfSharpness(
    ConstImage3FView original_opsin, ConstPlaneF32View pixel_mask,
    const VarDctEncoderFrame& frame, float butteraugli_target,
    PlaneU8View sharpness);

/// Same search with an existing padded coefficient reconstruction. This is
/// the final-AQ integration boundary; no inverse transforms are repeated.
[[nodiscard]] Status SearchEpfSharpnessFromReconstruction(
    EpfSharpnessSearchReference reference, ConstImage3FView reconstruction,
    const VarDctEncoderFrame& frame, float butteraugli_target,
    PlaneU8View sharpness);

}  // namespace gjxl
