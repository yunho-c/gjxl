// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/epf_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "codec/epf.h"
#include "codec/gaborish.h"
#include "codec/reconstruction.h"
#include "core/block_grid.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"
#include "core/managed_allocator.h"

namespace gjxl {
namespace {

template <class T>
bool ValidPlane(PlaneView<T> plane) {
  return plane.valid() &&
         plane.extent.height - 1 <=
             (std::numeric_limits<size_t>::max() - plane.extent.width) /
                 plane.stride &&
         (plane.extent.height - 1) * plane.stride + plane.extent.width <=
             std::numeric_limits<size_t>::max() / sizeof(T);
}

bool ValidImage(ConstImage3FView image) {
  return image.valid() && std::ranges::all_of(image.plane, [](auto plane) {
    return ValidPlane(plane);
  });
}

ConstImage3FView Crop(ConstImage3FView image, Extent2D extent) {
  for (auto& plane : image.plane) plane.extent = extent;
  return image;
}

}  // namespace

Status MakeEpfSharpnessSearchConfig(float target,
                                  EpfSharpnessSearchConfig* config) {
  if (config == nullptr || !std::isfinite(target) || target <= 0.0f) {
    return Status::InvalidArgument("EPF sharpness search target is invalid");
  }
  EpfSharpnessSearchConfig result;
  if (target > 4.5f) {
    result.candidates = {0, 4, 0};
    result.count = 2;
  } else {
    result.candidates = {0, 2, 7};
    result.count = 3;
  }
  result.no_smoothing_bias = std::max(
      0.85970338919928291f,
      std::pow(0.98017198824148288f, std::min(5.0f, target)));
  *config = result;
  return Status::Ok();
}

Status ComputeEpfBlockErrors(ConstImage3FView original,
                             ConstImage3FView reconstructed,
                             ConstPlaneF32View pixel_mask,
                             PlaneF32View errors) {
  if (!ValidImage(original) || !ValidImage(reconstructed) ||
      original.extent() != reconstructed.extent() || !ValidPlane(pixel_mask) ||
      pixel_mask.extent != original.extent() || !ValidPlane(errors) ||
      errors.extent != Extent2D{(original.width() + 7) / 8,
                                (original.height() + 7) / 8}) {
    return Status::InvalidArgument("EPF block-error geometry is invalid");
  }
  size_t count = 0;
  if (!errors.extent.try_area(&count)) {
    return Status::InvalidArgument("EPF block-error dimensions are too large");
  }
  try {
    resource_budget_internal::ManagedVector<float> result(count);
    for (size_t by = 0; by < errors.extent.height; ++by) {
      for (size_t bx = 0; bx < errors.extent.width; ++bx) {
        float error[3]{};
        for (size_t y = 8 * by; y < std::min(8 * by + 8, original.height()); ++y) {
          for (size_t x = 8 * bx; x < std::min(8 * bx + 8, original.width()); ++x) {
            const float mask = pixel_mask.Row(y)[x];
            if (!std::isfinite(mask) || mask < 0.0f) {
              return Status::InvalidArgument("EPF pixel mask is invalid");
            }
            const float mask2 = mask * mask;
            for (size_t c = 0; c < 3; ++c) {
              const float difference = original.plane[c].Row(y)[x] -
                                       reconstructed.plane[c].Row(y)[x];
              error[c] += mask2 * difference * difference;
            }
          }
        }
        const float value = 12.339445295782363 * error[0] + error[1] +
                            0.2 * error[2];
        if (!std::isfinite(value) || value < 0.0f) {
          return Status::InvalidArgument("EPF block error is not finite");
        }
        result[by * errors.extent.width + bx] = value;
      }
    }
    CopyContiguousPlane(result, errors);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Unable to allocate EPF block errors");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("EPF block-error dimensions are too large");
  }
}

Status SelectEpfSharpnessFromErrors(
    const std::array<ConstPlaneF32View, 3>& errors, float target,
    PlaneU8View sharpness) {
  EpfSharpnessSearchConfig config;
  Status status = MakeEpfSharpnessSearchConfig(target, &config);
  if (!status.ok()) return status;
  if (!ValidPlane(sharpness)) {
    return Status::InvalidArgument("EPF sharpness output is invalid");
  }
  // Validate the complete input before publishing any output.
  for (size_t i = 0; i < config.count; ++i) {
    if (!ValidPlane(errors[i]) || errors[i].extent != sharpness.extent) {
      return Status::InvalidArgument("EPF candidate-error geometry is invalid");
    }
    for (size_t y = 0; y < sharpness.extent.height; ++y) {
      for (size_t x = 0; x < sharpness.extent.width; ++x) {
        const float value = errors[i].Row(y)[x];
        if (!std::isfinite(value) || value < 0.0f) {
          return Status::InvalidArgument("EPF candidate error is invalid");
        }
      }
    }
  }
  // libjxl 9e7fba5d: histo and totals are size_t, with totals initialized to 1.
  // Each histogram count is strictly below its total. Their integer quotient
  // is therefore zero, log1p(0) is zero, and the final multiplier depends only
  // on whether the candidate is zero. The first raster pass cannot affect the
  // final map. Preserve those semantics instead of silently changing the ratio
  // to floating point and introducing a different search policy.
  for (size_t y = 0; y < sharpness.extent.height; ++y) {
    for (size_t x = 0; x < sharpness.extent.width; ++x) {
      float best_error = errors[0].Row(y)[x] * config.no_smoothing_bias;
      uint8_t best = config.candidates[0];
      for (size_t i = 1; i < config.count; ++i) {
        const float error = errors[i].Row(y)[x];
        if (error < best_error) {
          best_error = error;
          best = config.candidates[i];
        }
      }
      sharpness.Row(y)[x] = best;
    }
  }
  return Status::Ok();
}

Status SearchEpfSharpness(ConstImage3FView original_opsin,
                         ConstPlaneF32View pixel_mask,
                         const VarDctEncoderFrame& frame, float target,
                         PlaneU8View sharpness) {
  EpfSharpnessSearchConfig config;
  Status status = MakeEpfSharpnessSearchConfig(target, &config);
  if (!status.ok()) return status;
  if (!frame.valid() || !ValidImage(original_opsin) || !ValidPlane(pixel_mask) ||
      original_opsin.extent() != frame.geometry().padded_frame() ||
      pixel_mask.extent != original_opsin.extent() || !ValidPlane(sharpness) ||
      sharpness.extent != frame.geometry().block_grid().blocks) {
    return Status::InvalidArgument("EPF sharpness search geometry is invalid");
  }
  if (target < 0.5f || frame.profile().loop_filter.epf_options.iterations == 0)
    return FillDefaultEpfSharpness(sharpness);
  try {
    Image3FBuffer reconstruction(frame.geometry().padded_frame());
    status = ReconstructQuantizedCoefficients(frame, reconstruction.view());
    if (!status.ok()) return status;
    return SearchEpfSharpnessFromReconstruction(
        {original_opsin, pixel_mask}, reconstruction.const_view(), frame,
        target, sharpness);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Unable to allocate EPF reconstruction");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("EPF reconstruction dimensions are too large");
  }
}

Status SearchEpfSharpnessFromReconstruction(
    EpfSharpnessSearchReference reference, ConstImage3FView reconstruction,
    const VarDctEncoderFrame& frame, float target, PlaneU8View sharpness) {
  const auto original_opsin = reference.original_opsin;
  auto pixel_mask = reference.pixel_mask;
  EpfSharpnessSearchConfig config;
  Status status = MakeEpfSharpnessSearchConfig(target, &config);
  if (!status.ok()) return status;
  if (!frame.valid() || !ValidImage(original_opsin) || !ValidPlane(pixel_mask) ||
      !ValidImage(reconstruction) ||
      reconstruction.extent() != frame.geometry().padded_frame() ||
      original_opsin.extent() != frame.geometry().padded_frame() ||
      pixel_mask.extent != original_opsin.extent() || !ValidPlane(sharpness) ||
      sharpness.extent != frame.geometry().block_grid().blocks) {
    return Status::InvalidArgument("EPF sharpness search geometry is invalid");
  }
  if (target < 0.5f || frame.profile().loop_filter.epf_options.iterations == 0) {
    return FillDefaultEpfSharpness(sharpness);
  }
  try {
    const auto source = frame.geometry().frame();
    const auto blocks = sharpness.extent;
    size_t block_count = 0;
    if (!blocks.try_area(&block_count)) {
      return Status::InvalidArgument("EPF search dimensions are too large");
    }
    Image3FBuffer base(source);
    if (frame.profile().loop_filter.gaborish) {
      status = ApplyGaborish(Crop(reconstruction, source),
                            frame.profile().loop_filter.gaborish_options,
                            base.view());
      if (!status.ok()) return status;
    } else {
      CopyImage(Crop(reconstruction, source), base.view());
    }
    Image3FBuffer filtered(source);
    resource_budget_internal::ManagedVector<uint8_t> candidate_map(block_count);
    resource_budget_internal::ManagedVector<float> sigma(block_count);
    std::array<resource_budget_internal::ManagedVector<float>, 3> error_storage;
    std::array<ConstPlaneF32View, 3> errors;
    pixel_mask.extent = source;
    const auto original = Crop(original_opsin, source);
    for (size_t i = 0; i < config.count; ++i) {
      std::fill(candidate_map.begin(), candidate_map.end(), config.candidates[i]);
      status = ComputeEpfInverseSigma(
          frame.strategies(), frame.raw_quant_field(), frame.quantizer(),
          {candidate_map.data(), blocks, blocks.width}, frame.profile().epf_sigma,
          {sigma.data(), blocks, blocks.width});
      if (!status.ok()) return status;
      status = ApplyEpf(base.const_view(), {sigma.data(), blocks, blocks.width},
                        frame.profile().loop_filter.epf_options, filtered.view());
      if (!status.ok()) return status;
      error_storage[i].resize(block_count);
      status = ComputeEpfBlockErrors(
          original, filtered.const_view(), pixel_mask,
          {error_storage[i].data(), blocks, blocks.width});
      if (!status.ok()) return status;
      errors[i] = {error_storage[i].data(), blocks, blocks.width};
    }
    return SelectEpfSharpnessFromErrors(errors, target, sharpness);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Unable to allocate EPF sharpness search");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("EPF sharpness search dimensions are too large");
  }
}

}  // namespace gjxl
