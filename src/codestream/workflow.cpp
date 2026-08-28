// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/workflow.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
#include "codestream/rate_control_internal.h"
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

Status EffectiveTargetBytes(
  Extent2D source_extent,
  double bits_per_pixel,
  size_t* target_bytes) {

  if (target_bytes == nullptr || !std::isfinite(bits_per_pixel) ||
      bits_per_pixel <= 0.0) {
    return Status::InvalidArgument(
      "Target bits per pixel must be finite and positive");
  }
  size_t pixel_count = 0;
  if (!source_extent.try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "Target bits-per-pixel source dimensions are too large");
  }
  const long double bytes =
    static_cast<long double>(bits_per_pixel) *
    static_cast<long double>(pixel_count) / 8.0L;
  if (bytes < 1.0L ||
      bytes > static_cast<long double>(std::numeric_limits<size_t>::max())) {
    return Status::InvalidArgument(
      "Target bits per pixel does not produce a representable byte budget");
  }
  *target_bytes = static_cast<size_t>(bytes);
  return Status::Ok();
}

Status ValidateRateControlOptions(
  Extent2D source_extent,
  const VarDctEncodingOptions& options,
  size_t* effective_target_bytes,
  size_t* tolerance_bytes) {

  if (effective_target_bytes == nullptr || tolerance_bytes == nullptr) {
    return Status::InvalidArgument(
      "Rate-control derived-value output is null");
  }
  *effective_target_bytes = 0;
  *tolerance_bytes = 0;
  switch (options.rate_control_mode) {
    case VarDctRateControlMode::kButteraugliTarget:
      if (!std::isfinite(options.butteraugli_target) ||
          options.butteraugli_target <= 0.0f) {
        return Status::InvalidArgument(
          "Butteraugli target must be finite and positive");
      }
      return Status::Ok();

    case VarDctRateControlMode::kMaximumError:
      for (float maximum_error : options.maximum_error) {
        if (!std::isfinite(maximum_error) || maximum_error <= 0.0f) {
          return Status::InvalidArgument(
            "Maximum-error limits must be finite and positive");
        }
      }
      return Status::Ok();

    case VarDctRateControlMode::kTargetBytes:
      if (options.target_bytes == 0) {
        return Status::InvalidArgument(
          "Target byte count must be nonzero");
      }
      *effective_target_bytes = options.target_bytes;
      break;

    case VarDctRateControlMode::kTargetBitsPerPixel: {
      Status status = EffectiveTargetBytes(
        source_extent,
        options.target_bits_per_pixel,
        effective_target_bytes);
      if (!status.ok()) {
        return status;
      }
      break;
    }

    default:
      return Status::InvalidArgument(
        "VarDCT rate-control mode is invalid");
  }

  if (!std::isfinite(options.target_size_tolerance) ||
      options.target_size_tolerance < 0.0 ||
      options.target_size_tolerance > 1.0 ||
      options.target_size_maximum_attempts == 0 ||
      options.target_size_maximum_attempts >
        codestream_internal::kMaximumTargetSizeEncodeAttempts) {
    return Status::InvalidArgument(
      "Target-size tolerance or attempt limit is invalid");
  }
  const long double tolerance = std::ceil(
    static_cast<long double>(*effective_target_bytes) *
    static_cast<long double>(options.target_size_tolerance));
  if (tolerance >
      static_cast<long double>(std::numeric_limits<size_t>::max())) {
    return Status::InvalidArgument(
      "Target-size tolerance is not representable");
  }
  *tolerance_bytes = static_cast<size_t>(tolerance);
  return Status::Ok();
}

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
        .maximum_error_result = &maximum_error_result,
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
  MaximumErrorResult maximum_error_result;
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

  if (codestream == nullptr || !linear_rgb.valid()) {
    return Status::InvalidArgument(
      "VarDCT encoding input or output is invalid");
  }
  size_t effective_target_bytes = 0;
  size_t target_size_tolerance_bytes = 0;
  Status status = ValidateRateControlOptions(
    linear_rgb.extent(),
    options,
    &effective_target_bytes,
    &target_size_tolerance_bytes);
  if (!status.ok()) {
    return status;
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
  switch (options.metal_aq_mode) {
    case GpuAdaptiveQuantizationMode::kExactCoefficients:
      break;
    case GpuAdaptiveQuantizationMode::kFullyResident:
      if (options.backend != VarDctBackendPreference::kMetal) {
        return Status::InvalidArgument(
          "Fully resident AQ requires an explicitly forced Metal backend");
      }
      break;
    default:
      return Status::InvalidArgument(
        "VarDCT Metal AQ mode is invalid");
  }
  if (options.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
      options.rate_control_mode ==
        VarDctRateControlMode::kTargetBitsPerPixel) {
    try {
      codestream_internal::TargetSizeSearchResult search_result;
      status = codestream_internal::SearchTargetSize(
        {
          .target_bytes = effective_target_bytes,
          .tolerance_bytes = target_size_tolerance_bytes,
          .maximum_attempts = options.target_size_maximum_attempts,
        },
        [&](float butteraugli_target,
            std::vector<uint8_t>* attempt_codestream,
            VarDctEncodingSummary* attempt_summary) {
          VarDctEncodingOptions attempt_options = options;
          attempt_options.butteraugli_target = butteraugli_target;
          attempt_options.rate_control_mode =
            VarDctRateControlMode::kButteraugliTarget;
          return EncodeLinearRgbVarDctCodestreamImpl(
            linear_rgb,
            attempt_options,
            supplied_backend,
            supplied_backend_is_qualified,
            resolve_production_backend,
            attempt_codestream,
            attempt_summary);
        },
        &search_result);
      if (!status.ok()) {
        return status;
      }

      search_result.summary.rate_control_mode = options.rate_control_mode;
      search_result.summary.effective_target_bytes = effective_target_bytes;
      search_result.summary.target_size_tolerance_bytes =
        target_size_tolerance_bytes;
      search_result.summary.encode_attempt_count = search_result.attempt_count;
      search_result.summary.target_size_met = search_result.target_size_met;
      if (options.rate_control_mode == VarDctRateControlMode::kTargetBytes) {
        search_result.summary.requested_target_bytes = options.target_bytes;
      } else {
        search_result.summary.requested_target_bits_per_pixel =
          options.target_bits_per_pixel;
      }
      *codestream = std::move(search_result.codestream);
      if (summary != nullptr) {
        *summary = std::move(search_result.summary);
      }
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate target-size workflow storage");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Target-size workflow storage is too large");
    }
    return Status::Ok();
  }

  try {
    FrameGeometry geometry;
    status = FrameGeometry::Create(linear_rgb.extent(), &geometry);
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
    if (options.rate_control_mode == VarDctRateControlMode::kMaximumError) {
      pipeline_options.adaptive_quantization.control_mode =
        AdaptiveQuantizationControlMode::kMaximumError;
      pipeline_options.adaptive_quantization.maximum_error =
        options.maximum_error;
    }
    GpuBackend* selected_gpu = nullptr;
    bool selected_metal = false;
    if (options.backend != VarDctBackendPreference::kCpu &&
        (options.rate_control_mode != VarDctRateControlMode::kMaximumError ||
         options.backend == VarDctBackendPreference::kMetal)) {
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
          options.metal_aq_mode, pipeline.Output())
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
    candidate_summary.rate_control_mode = options.rate_control_mode;
    candidate_summary.effective_target_bytes = effective_target_bytes;
    candidate_summary.target_size_tolerance_bytes =
      target_size_tolerance_bytes;
    if (options.rate_control_mode == VarDctRateControlMode::kTargetBytes) {
      candidate_summary.requested_target_bytes = options.target_bytes;
    } else if (options.rate_control_mode ==
               VarDctRateControlMode::kTargetBitsPerPixel) {
      candidate_summary.requested_target_bits_per_pixel =
        options.target_bits_per_pixel;
    }
    size_t source_pixel_count = 0;
    if (!linear_rgb.extent().try_area(&source_pixel_count) ||
        source_pixel_count == 0) {
      return Status::Internal(
        "Validated VarDCT source has no representable pixels");
    }
    candidate_summary.achieved_bits_per_pixel =
      8.0 * static_cast<double>(candidate.size()) /
      static_cast<double>(source_pixel_count);
    candidate_summary.selected_butteraugli_target =
      options.rate_control_mode == VarDctRateControlMode::kMaximumError
        ? 0.0f
        : options.butteraugli_target;
    if (options.rate_control_mode == VarDctRateControlMode::kMaximumError) {
      candidate_summary.requested_maximum_error = options.maximum_error;
      candidate_summary.achieved_maximum_error =
        pipeline.maximum_error_result.achieved;
      candidate_summary.achieved_maximum_error_ratio =
        pipeline.maximum_error_result.normalized_maximum;
      candidate_summary.maximum_error_evaluation_count =
        pipeline.maximum_error_result.evaluation_count;
      candidate_summary.maximum_error_outcome =
        pipeline.maximum_error_result.outcome;
    }
    candidate_summary.encode_attempt_count = 1;
    candidate_summary.score_history = std::move(pipeline.score_history);
    candidate_summary.execution_backend = selected_metal
      ? VarDctExecutionBackend::kMetal
      : VarDctExecutionBackend::kCpu;
    candidate_summary.metal_aq_mode = options.metal_aq_mode;
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
