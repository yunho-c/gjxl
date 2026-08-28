// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/workflow.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "codec/color_transform.h"
#include "codec/quantization_pipeline.h"
#include "codec/vardct_frame.h"
#include "codestream/encoder.h"
#include "codestream/workflow_internal.h"
#include "core/frame_geometry.h"
#include "core/image_buffer.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/ac_strategy.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/quantization_pipeline.h"

namespace gjxl {
namespace {

constexpr float kInitialProfileIntensityTarget = 255.0f;

// Cold 64x48 ranges overlap on the qualified M4 Pro; 96x64 is the first
// non-overlapping win. Keep one measured step of headroom, so automatic
// selection begins at 128x96.
constexpr size_t kAutomaticMetalMinimumCodingPixels = 128 * 96;
constexpr size_t kAutomaticMetalMinimumCodingDimension = 96;
// Policy sweeps preserve exact frame decisions in this closed interval;
// threshold-sensitive fixtures outside it remain on the CPU in auto mode.
constexpr float kAutomaticMetalMinimumButteraugliTarget = 1.0f;
constexpr float kAutomaticMetalMaximumButteraugliTarget = 1.2f;
constexpr std::string_view kQualifiedMetalBackend = "Metal: Apple M4 Pro";

MetalBackendOptions ProductionMetalBackendOptions() {
  constexpr auto implementation =
    MetalDctImplementation::kSimdgroupMatmul;
  return {
    .forward_dct8 = implementation,
    .inverse_dct8 = implementation,
    .forward_dct16x16 = implementation,
    .inverse_dct16x16 = implementation,
    .forward_dct32x32 = implementation,
    .inverse_dct32x32 = implementation,
    .forward_dct16x8 = implementation,
    .inverse_dct16x8 = implementation,
    .forward_dct8x16 = implementation,
    .inverse_dct8x16 = implementation,
    .forward_dct32x16 = implementation,
    .inverse_dct32x16 = implementation,
    .forward_dct16x32 = implementation,
    .inverse_dct16x32 = implementation,
  };
}

Status ResolveProductionMetalBackend(GpuBackend** out) {
  if (out == nullptr) {
    return Status::InvalidArgument(
      "Production Metal backend output pointer is null");
  }
  *out = nullptr;
  struct Cache {
    std::once_flag once;
    std::unique_ptr<GpuBackend> backend;
    Status status = Status::Unavailable(
      "Production Metal backend has not been initialized");
  };
  static Cache cache;
  std::call_once(cache.once, [&] {
    cache.status = CreateEmbeddedMetalBackend(
      ProductionMetalBackendOptions(), &cache.backend);
    if (cache.status.ok() && cache.backend == nullptr) {
      cache.status = Status::Internal(
        "Embedded Metal factory returned no backend");
    }
  });
  if (!cache.status.ok()) {
    return cache.status;
  }
  *out = cache.backend.get();
  return Status::Ok();
}

bool HasCompleteGpuQuantizationCapabilities(GpuBackend& backend) {
  return QueryGpuAcStrategyEvaluation(backend) != nullptr &&
    QueryGpuAqEvaluation(backend) != nullptr;
}

struct PipelineStorage {
  PipelineStorage(Extent2D frame_extent, Extent2D padded_extent)
      : block_extent{
          padded_extent.width / kJxlBlockDimension,
          padded_extent.height / kJxlBlockDimension},
        padded_extent(padded_extent),
        initial_quant(BlockCount()),
        strategy_mask(BlockCount()),
        pixel_mask(PixelCount(padded_extent)),
        final_quant(BlockCount()),
        block_distance(BlockCount()),
        reconstructed(frame_extent) {}

  [[nodiscard]] CpuQuantizationPipelineOutput Output() {
    return {
      .initial_quantization = {
        .quant_field = {
          initial_quant.data(), block_extent, block_extent.width},
        .strategy_mask = {
          strategy_mask.data(), block_extent, block_extent.width},
        .pixel_mask = {
          pixel_mask.data(), padded_extent, padded_extent.width},
      },
      .adaptive_quantization = {
        .quant_field = {
          final_quant.data(), block_extent, block_extent.width},
        .block_distance_map = {
          block_distance.data(), block_extent, block_extent.width},
        .reconstructed_linear_rgb = reconstructed.view(),
        .frame = &frame,
        .score_history = &score_history,
      },
    };
  }

  [[nodiscard]] size_t BlockCount() const {
    size_t count = 0;
    if (!block_extent.try_area(&count)) {
      throw std::length_error("VarDCT block extent is too large");
    }
    return count;
  }

  [[nodiscard]] static size_t PixelCount(Extent2D extent) {
    size_t count = 0;
    if (!extent.try_area(&count)) {
      throw std::length_error("VarDCT padded extent is too large");
    }
    return count;
  }

  Extent2D block_extent;
  Extent2D padded_extent;
  std::vector<float> initial_quant;
  std::vector<float> strategy_mask;
  std::vector<float> pixel_mask;
  std::vector<float> final_quant;
  std::vector<float> block_distance;
  Image3FBuffer reconstructed;
  VarDctEncoderFrame frame;
  std::vector<double> score_history;
};

[[nodiscard]] Status EdgeExtend(
  ConstImage3FView source,
  Image3FView destination) {

  if (!source.valid() || !destination.valid() ||
      source.width() > destination.width() ||
      source.height() > destination.height()) {
    return Status::InvalidArgument(
      "Linear RGB source or padded destination is invalid");
  }
  for (size_t y = 0; y < destination.height(); ++y) {
    const size_t source_y = std::min(y, source.height() - 1);
    for (size_t x = 0; x < destination.width(); ++x) {
      const size_t source_x = std::min(x, source.width() - 1);
      for (size_t channel = 0; channel < 3; ++channel) {
        const float value = source.plane[channel].Row(source_y)[source_x];
        if (!std::isfinite(value)) {
          return Status::InvalidArgument(
            "Linear RGB input pixels must be finite");
        }
        destination.plane[channel].Row(y)[x] = value;
      }
    }
  }
  return Status::Ok();
}

}  // namespace

Status EncodeLinearRgbVarDctCodestreamImpl(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* supplied_backend,
  bool supplied_backend_is_qualified,
  bool resolve_production_backend,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary) {

  if (codestream == nullptr || !linear_rgb.valid() ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "VarDCT encoding input, target, or output is invalid");
  }

  switch (options.backend) {
    case VarDctBackendPreference::kAutomatic:
    case VarDctBackendPreference::kCpu:
    case VarDctBackendPreference::kMetal:
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT encoding backend preference is invalid");
  }

  try {
    FrameGeometry geometry;
    Status status = FrameGeometry::Create(linear_rgb.extent(), &geometry);
    if (!status.ok()) {
      return status;
    }

    Image3FBuffer padded_linear(geometry.padded_frame());
    status = EdgeExtend(linear_rgb, padded_linear.view());
    if (!status.ok()) {
      return status;
    }
    Image3FBuffer opsin(geometry.padded_frame());
    status = LinearRgbToOpsin(
      padded_linear.const_view(),
      kInitialProfileIntensityTarget,
      opsin.view());
    if (!status.ok()) {
      return status;
    }

    PipelineStorage pipeline(linear_rgb.extent(), geometry.padded_frame());
    CpuQuantizationPipelineOptions pipeline_options;
    pipeline_options.butteraugli_target = options.butteraugli_target;
    GpuBackend* selected_gpu = nullptr;
    bool selected_metal = false;
    if (options.backend != VarDctBackendPreference::kCpu) {
      const bool geometry_eligible =
        codestream_internal::IsAutomaticMetalGeometryEligible(
          geometry.padded_frame());
      const bool target_eligible =
        codestream_internal::IsAutomaticMetalTargetEligible(
          options.butteraugli_target);
      const bool should_resolve =
        options.backend == VarDctBackendPreference::kMetal ||
        (options.backend == VarDctBackendPreference::kAutomatic &&
         geometry_eligible && target_eligible);
      if (should_resolve) {
        selected_gpu = supplied_backend;
        bool qualified = supplied_backend_is_qualified;
        if (selected_gpu == nullptr && resolve_production_backend) {
          status = ResolveProductionMetalBackend(&selected_gpu);
          if (!status.ok()) {
            if (options.backend == VarDctBackendPreference::kAutomatic &&
                status.code() == StatusCode::kUnavailable) {
              selected_gpu = nullptr;
              status = Status::Ok();
            } else {
              return status;
            }
          }
          qualified = selected_gpu != nullptr &&
            codestream_internal::IsAutomaticMetalBackendQualified(
              *selected_gpu);
        }
        if (selected_gpu != nullptr) {
          if (!HasCompleteGpuQuantizationCapabilities(*selected_gpu)) {
            if (options.backend == VarDctBackendPreference::kMetal) {
              return Status::Unavailable(
                "Forced Metal workflow requires complete GPU quantization");
            }
            selected_gpu = nullptr;
          } else if (options.backend == VarDctBackendPreference::kMetal ||
                     qualified) {
            selected_metal = true;
          } else {
            selected_gpu = nullptr;
          }
        } else if (options.backend == VarDctBackendPreference::kMetal) {
          return Status::Unavailable(
            "Forced Metal workflow has no available backend");
        }
      }
    }
    status = selected_metal
      ? RunGpuQuantizationPipeline(
          *selected_gpu, linear_rgb, opsin.const_view(), pipeline_options,
          pipeline.Output())
      : RunCpuQuantizationPipeline(
          linear_rgb, opsin.const_view(), pipeline_options,
          pipeline.Output());
    if (!status.ok()) {
      return status;
    }

    std::vector<uint8_t> candidate;
    status = EncodeVarDctCodestream(pipeline.frame, &candidate);
    if (!status.ok()) {
      return status;
    }

    VarDctEncodingSummary candidate_summary;
    candidate_summary.extent = linear_rgb.extent();
    candidate_summary.encoded_bytes = candidate.size();
    candidate_summary.score_history = std::move(pipeline.score_history);
    candidate_summary.execution_backend = selected_metal
      ? VarDctExecutionBackend::kMetal
      : VarDctExecutionBackend::kCpu;
    status = pipeline.frame.strategies().ForEachAnchor(
      [&](size_t, size_t, AcStrategyType strategy) {
        const size_t index = static_cast<size_t>(strategy);
        if (index >= candidate_summary.strategy_counts.size()) {
          return Status::Internal(
            "Completed frame contains an unknown AC strategy");
        }
        ++candidate_summary.strategy_counts[index];
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }

    *codestream = std::move(candidate);
    if (summary != nullptr) {
      *summary = std::move(candidate_summary);
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate public VarDCT encoding storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Public VarDCT encoding dimensions are too large");
  }
  return Status::Ok();
}

Status EncodeLinearRgbVarDctCodestream(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary) {

  return EncodeLinearRgbVarDctCodestreamImpl(
    linear_rgb, options, nullptr, false, true, codestream, summary);
}

namespace codestream_internal {

bool IsAutomaticMetalGeometryEligible(Extent2D padded_extent) noexcept {
  size_t area = 0;
  return padded_extent.try_area(&area) &&
    area >= kAutomaticMetalMinimumCodingPixels &&
    std::min(padded_extent.width, padded_extent.height) >=
      kAutomaticMetalMinimumCodingDimension;
}

bool IsAutomaticMetalTargetEligible(float butteraugli_target) noexcept {
  return std::isfinite(butteraugli_target) &&
    butteraugli_target >= kAutomaticMetalMinimumButteraugliTarget &&
    butteraugli_target <= kAutomaticMetalMaximumButteraugliTarget;
}

bool IsAutomaticMetalBackendQualified(
  const GpuBackend& backend) noexcept {

  return backend.kind() == BackendKind::kMetal &&
    backend.name() == kQualifiedMetalBackend;
}

Status EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary) {

  return EncodeLinearRgbVarDctCodestreamImpl(
    linear_rgb, options, backend, backend_is_qualified_for_automatic, false,
    codestream, summary);
}

}  // namespace codestream_internal

}  // namespace gjxl
