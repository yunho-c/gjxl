// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_aq_evaluation_internal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/dc_conversion.h"
#include "codec/quantization.h"
#include "codec/quantization_tables_generated.h"
#include "codec/vardct_frame_internal.h"
#include "core/image_buffer.h"
#include "core/quantizer.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_butteraugli_encoding.h"
#include "gpu/metal/metal_status.h"
#include "gpu/scratch.h"

#define setComputePipelineState(state)                                    \
  setComputePipelineState(state);                                         \
  ::gjxl::metal_internal::RecordMetalComputePipelineState(state)

namespace gjxl::metal_internal {
namespace {

using ProfileClock = std::chrono::steady_clock;

uint64_t ElapsedNanoseconds(ProfileClock::time_point begin) {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      ProfileClock::now() - begin).count());
}

inline constexpr size_t kBufferAlignment = 256;
inline constexpr NS::UInteger kBlockReductionThreadCount = 256;
inline constexpr NS::UInteger kAqThreadCount = 256;
inline constexpr size_t kQuantTableValueCount = 11904;
inline constexpr std::array<AcStrategyType, 7> kSupportedAqStrategies = {
    AcStrategyType::kDct8,     AcStrategyType::kDct16x16,
    AcStrategyType::kDct32x32, AcStrategyType::kDct16x8,
    AcStrategyType::kDct8x16,  AcStrategyType::kDct32x16,
    AcStrategyType::kDct16x32,
};

void BindPlane(MTL::ComputeCommandEncoder* encoder, DevicePlaneView plane,
               NS::UInteger index) {
  MetalBuffer* buffer = dynamic_cast<MetalBuffer*>(plane.buffer);
  encoder->setBuffer(buffer->handle(), plane.offset_bytes, index);
}

void DispatchThreads1d(MTL::ComputeCommandEncoder* encoder,
                       size_t thread_count) {
  DispatchMetalThreads(
    encoder,
    MTL::Size(static_cast<NS::UInteger>(thread_count), 1, 1),
    MTL::Size(kAqThreadCount, 1, 1));
}

static_assert(std::is_standard_layout_v<AqReconstructionParams>);
static_assert(std::is_trivially_copyable_v<AqReconstructionParams>);
static_assert(sizeof(AqReconstructionParams) == 136);
static_assert(sizeof(AqResetParams) == 32);
static_assert(sizeof(AqResidentPolicyInitializeParams) == 20);
static_assert(sizeof(AqResidentPolicyUpdateParams) == 44);
static_assert(std::is_standard_layout_v<AqInitialCflParams>);
static_assert(std::is_trivially_copyable_v<AqInitialCflParams>);
static_assert(sizeof(AqInitialCflParams) == 24);
static_assert(std::is_standard_layout_v<AqFinalCflParams>);
static_assert(std::is_trivially_copyable_v<AqFinalCflParams>);
static_assert(sizeof(AqFinalCflParams) == 16);
static_assert(std::is_standard_layout_v<AqInitialQuantGradientParams>);
static_assert(std::is_trivially_copyable_v<AqInitialQuantGradientParams>);
static_assert(sizeof(AqInitialQuantGradientParams) == 28);
static_assert(std::is_standard_layout_v<AqInitialQuantErosionParams>);
static_assert(std::is_trivially_copyable_v<AqInitialQuantErosionParams>);
static_assert(sizeof(AqInitialQuantErosionParams) == 44);
static_assert(std::is_standard_layout_v<AqInitialQuantModulationParams>);
static_assert(std::is_trivially_copyable_v<AqInitialQuantModulationParams>);
static_assert(sizeof(AqInitialQuantModulationParams) == 24);
static_assert(std::is_standard_layout_v<AqInitialQuantSelectionParams>);
static_assert(std::is_trivially_copyable_v<AqInitialQuantSelectionParams>);
static_assert(sizeof(AqInitialQuantSelectionParams) == 36);
static_assert(std::is_standard_layout_v<AqInitialQuantSortParams>);
static_assert(std::is_trivially_copyable_v<AqInitialQuantSortParams>);
static_assert(sizeof(AqInitialQuantSortParams) == 12);
static_assert(std::is_standard_layout_v<AqQuantFieldAdjustmentParams>);
static_assert(std::is_trivially_copyable_v<AqQuantFieldAdjustmentParams>);
static_assert(sizeof(AqQuantFieldAdjustmentParams) == 24);
static_assert(std::is_standard_layout_v<AqResidentQuantSelectionPass>);
static_assert(std::is_trivially_copyable_v<AqResidentQuantSelectionPass>);
static_assert(sizeof(AqResidentQuantSelectionPass) == 8);
static_assert(std::is_standard_layout_v<AqBlockReductionParams>);
static_assert(std::is_trivially_copyable_v<AqBlockReductionParams>);
static_assert(sizeof(AqBlockReductionParams) == 40);
static_assert(std::is_standard_layout_v<AqMaximumErrorReductionParams>);
static_assert(std::is_trivially_copyable_v<AqMaximumErrorReductionParams>);
static_assert(sizeof(AqMaximumErrorReductionParams) == 56);
static_assert(sizeof(AqQuantizationProbeParams) == 24);
static_assert(sizeof(AqAdjustmentProbeParams) == 32);
static_assert(std::is_standard_layout_v<AqGaborishParams>);
static_assert(std::is_trivially_copyable_v<AqGaborishParams>);
static_assert(sizeof(AqGaborishParams) == 52);
static_assert(std::is_standard_layout_v<AqEpfParams>);
static_assert(std::is_trivially_copyable_v<AqEpfParams>);
static_assert(sizeof(AqEpfParams) == 44);
static_assert(std::is_standard_layout_v<AqOpsinToLinearParams>);
static_assert(std::is_trivially_copyable_v<AqOpsinToLinearParams>);
static_assert(sizeof(AqOpsinToLinearParams) == 20);

[[nodiscard]] bool SupportedAqStrategy(AcStrategyType strategy) noexcept {
  switch (strategy) {
    case AcStrategyType::kDct8:
    case AcStrategyType::kDct16x16:
    case AcStrategyType::kDct32x32:
    case AcStrategyType::kDct16x8:
    case AcStrategyType::kDct8x16:
    case AcStrategyType::kDct32x16:
    case AcStrategyType::kDct16x32:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] size_t AqStrategyBatchIndex(AcStrategyType strategy) noexcept {

  const auto found = std::ranges::find(kSupportedAqStrategies, strategy);
  return found == kSupportedAqStrategies.end()
             ? kSupportedAqStrategies.size()
             : static_cast<size_t>(found - kSupportedAqStrategies.begin());
}

[[nodiscard]] Status BuildColorTransformMetadata(
  const std::vector<AqAnchor>& row_major_anchors,
  const std::array<AqStrategyBatch, 7>& batches,
  Extent2D block_extent,
  Extent2D tile_extent,
  std::vector<int32_t>* records,
  std::vector<int32_t>* tile_offsets) {

  if (records == nullptr || tile_offsets == nullptr) {
    return Status::Internal(
      "Prepared AQ color-transform metadata output is null");
  }
  size_t tile_count = 0;
  if (!tile_extent.try_area(&tile_count) ||
      row_major_anchors.size() >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return Status::InvalidArgument(
      "Prepared AQ color-transform metadata is too large");
  }
  try {
    records->assign(6 * row_major_anchors.size(), 0);
    tile_offsets->assign(tile_count + 1, 0);
    for (const AqAnchor& anchor : row_major_anchors) {
      const size_t tile_index =
        (anchor.block_y / 8) * tile_extent.width + anchor.block_x / 8;
      if (tile_index >= tile_count ||
          (*tile_offsets)[tile_index + 1] ==
            std::numeric_limits<int32_t>::max()) {
        return Status::InvalidArgument(
          "Prepared AQ color-transform tile count exceeds Metal limits");
      }
      ++(*tile_offsets)[tile_index + 1];
    }
    for (size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
      (*tile_offsets)[tile_index + 1] += (*tile_offsets)[tile_index];
    }
    std::vector<int32_t> positions = *tile_offsets;
    std::vector<size_t> tile_value_offsets(tile_count, 0);
    for (const AqAnchor& anchor : row_major_anchors) {
      const size_t tile_index =
        (anchor.block_y / 8) * tile_extent.width + anchor.block_x / 8;
      if (anchor.batch_index >= batches.size()) {
        return Status::Internal(
          "Prepared AQ color-transform batch disappeared");
      }
      const AqStrategyBatch& batch = batches[anchor.batch_index];
      const size_t channel_stride =
        batch.anchor_count * batch.coefficient_count;
      const size_t coefficient_offset = batch.coefficient_offset +
        anchor.index_in_batch * batch.coefficient_count;
      const size_t raw_quant_index =
        anchor.block_y * block_extent.width + anchor.block_x;
      const std::array<size_t, 6> values = {
        coefficient_offset,
        channel_stride,
        batch.coefficient_count,
        static_cast<size_t>(anchor.strategy),
        raw_quant_index,
        tile_value_offsets[tile_index],
      };
      const size_t record_index =
        static_cast<size_t>(positions[tile_index]++);
      for (size_t field = 0; field < values.size(); ++field) {
        if (values[field] > static_cast<size_t>(
              std::numeric_limits<int32_t>::max())) {
          return Status::InvalidArgument(
            "Prepared AQ color-transform record exceeds Metal limits");
        }
        (*records)[6 * record_index + field] =
          static_cast<int32_t>(values[field]);
      }
      tile_value_offsets[tile_index] += batch.coefficient_count;
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate prepared AQ color-transform metadata");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Prepared AQ color-transform metadata is too large");
  }
  return Status::Ok();
}

[[nodiscard]] const char* AqReconstructionProfileStageId(
    AcStrategyType strategy) noexcept {
  switch (strategy) {
    case AcStrategyType::kDct8:
      return "aq.reconstruction.dct8";
    case AcStrategyType::kDct16x8:
      return "aq.reconstruction.dct16x8";
    case AcStrategyType::kDct8x16:
      return "aq.reconstruction.dct8x16";
    case AcStrategyType::kDct16x16:
      return "aq.reconstruction.dct16";
    case AcStrategyType::kDct32x16:
      return "aq.reconstruction.dct32x16";
    case AcStrategyType::kDct16x32:
      return "aq.reconstruction.dct16x32";
    case AcStrategyType::kDct32x32:
      return "aq.reconstruction.dct32";
    default:
      return "aq.reconstruction.unsupported";
  }
}

[[nodiscard]] const char* AqForwardCoefficientProfileStageId(
    AcStrategyType strategy) noexcept {
  switch (strategy) {
    case AcStrategyType::kDct8:
      return "aq.reconstruction.forward.dct8";
    case AcStrategyType::kDct16x8:
      return "aq.reconstruction.forward.dct16x8";
    case AcStrategyType::kDct8x16:
      return "aq.reconstruction.forward.dct8x16";
    case AcStrategyType::kDct16x16:
      return "aq.reconstruction.forward.dct16";
    case AcStrategyType::kDct32x16:
      return "aq.reconstruction.forward.dct32x16";
    case AcStrategyType::kDct16x32:
      return "aq.reconstruction.forward.dct16x32";
    case AcStrategyType::kDct32x32:
      return "aq.reconstruction.forward.dct32";
    default:
      return "aq.reconstruction.forward.unsupported";
  }
}

[[nodiscard]] const char* AqFinalFrameProfileStageId(
    AcStrategyType strategy) noexcept {
  switch (strategy) {
    case AcStrategyType::kDct8:
      return "aq.final_frame.dct8";
    case AcStrategyType::kDct16x8:
      return "aq.final_frame.dct16x8";
    case AcStrategyType::kDct8x16:
      return "aq.final_frame.dct8x16";
    case AcStrategyType::kDct16x16:
      return "aq.final_frame.dct16";
    case AcStrategyType::kDct32x16:
      return "aq.final_frame.dct32x16";
    case AcStrategyType::kDct16x32:
      return "aq.final_frame.dct16x32";
    case AcStrategyType::kDct32x32:
      return "aq.final_frame.dct32";
    default:
      return "aq.final_frame.unsupported";
  }
}

[[nodiscard]] bool FinitePositive(float value) noexcept {
  return std::isfinite(value) && value > 0.0f;
}

template <typename T>
[[nodiscard]] bool ValidHostPlaneLayout(PlaneView<T> plane) noexcept {
  if (!plane.valid()) {
    return false;
  }
  if (plane.extent.height - 1 >
      (std::numeric_limits<size_t>::max() - plane.extent.width) /
        plane.stride) {
    return false;
  }
  const size_t elements =
    (plane.extent.height - 1) * plane.stride + plane.extent.width;
  using Value = std::remove_const_t<T>;
  return elements <= std::numeric_limits<size_t>::max() / sizeof(Value);
}

template <typename T>
[[nodiscard]] bool PlaneDescriptorSpecified(PlaneView<T> plane) noexcept {
  return plane.data != nullptr || plane.extent.width != 0 ||
         plane.extent.height != 0 || plane.stride != 0;
}

template <typename T>
[[nodiscard]] bool ImageDescriptorSpecified(
    Image3View<T> image) noexcept {
  return std::ranges::any_of(
      image.plane, [](PlaneView<T> plane) {
        return plane.data != nullptr || plane.extent.width != 0 ||
               plane.extent.height != 0 || plane.stride != 0;
      });
}

[[nodiscard]] Status ValidateOptions(const AqEvaluationOptions& options) {
  if (!options.profile.valid()) {
    return Status::InvalidArgument(
      "Prepared AQ scalar options must be finite and positive");
  }
  switch (options.metric) {
    case AqEvaluationMetric::kButteraugli:
      if (!FinitePositive(options.butteraugli.hf_asymmetry) ||
          !FinitePositive(options.butteraugli.x_multiplier) ||
          !FinitePositive(options.butteraugli.intensity_target)) {
        return Status::InvalidArgument(
          "Prepared AQ Butteraugli options must be finite and positive");
      }
      break;
    case AqEvaluationMetric::kMaximumError:
      if (!std::ranges::all_of(
            options.maximum_error,
            [](float value) { return FinitePositive(value); })) {
        return Status::InvalidArgument(
          "Prepared AQ maximum-error limits must be finite and positive");
      }
      break;
    default:
      return Status::InvalidArgument(
        "Prepared AQ evaluation metric is invalid");
  }

  if (options.profile.loop_filter.gaborish) {
    for (size_t channel = 0; channel < 3; ++channel) {
      const float weight1 =
        options.profile.loop_filter.gaborish_options.weight1[channel];
      const float weight2 =
        options.profile.loop_filter.gaborish_options.weight2[channel];
      const float divisor = 1.0f + 4.0f * (weight1 + weight2);
      if (!std::isfinite(weight1) || !std::isfinite(weight2) ||
          !std::isfinite(divisor) || std::abs(divisor) < 1.0e-8f) {
        return Status::InvalidArgument(
          "Prepared AQ Gaborish options are invalid");
      }
    }
  }

  const EpfFilterOptions& epf = options.profile.loop_filter.epf_options;
  if (epf.iterations > 3 ||
      !FinitePositive(epf.pass0_sigma_scale) ||
      !FinitePositive(epf.pass2_sigma_scale) ||
      !FinitePositive(epf.border_sad_multiplier)) {
    return Status::InvalidArgument(
      "Prepared AQ EPF options are invalid");
  }
  for (float scale : epf.channel_scale) {
    if (!std::isfinite(scale) || scale < 0.0f) {
      return Status::InvalidArgument(
        "Prepared AQ EPF channel scales are invalid");
    }
  }
  return Status::Ok();
}

[[nodiscard]] Status ValidateFiniteImage(
  ConstImage3FView image,
  std::string_view name) {

  if (!image.valid() ||
      !std::ranges::all_of(image.plane, [](ConstPlaneF32View plane) {
        return ValidHostPlaneLayout(plane);
      })) {
    return Status::InvalidArgument(
      std::string(name) + " image view is invalid");
  }
  for (const ConstPlaneF32View plane : image.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.Row(y)[x])) {
          return Status::InvalidArgument(
            std::string(name) + " image contains a non-finite sample");
        }
      }
    }
  }
  return Status::Ok();
}

[[nodiscard]] Status ValidateAqGeometry(
  Extent2D source,
  Extent2D coding) {

  if (source.empty() || coding.empty() ||
      coding.width % kJxlBlockDimension != 0 ||
      coding.height % kJxlBlockDimension != 0 ||
      source.width > coding.width || source.height > coding.height ||
      coding.width - source.width >= kJxlBlockDimension ||
      coding.height - source.height >= kJxlBlockDimension) {
    return Status::InvalidArgument(
      "Prepared AQ source and padded coding geometry are incompatible");
  }
  const Extent2D blocks{
    coding.width / kJxlBlockDimension,
    coding.height / kJxlBlockDimension,
  };
  constexpr size_t kShaderMaximum =
    static_cast<size_t>(std::numeric_limits<uint32_t>::max());
  size_t block_count = 0;
  if (source.width > kShaderMaximum || source.height > kShaderMaximum ||
      coding.width > kShaderMaximum || coding.height > kShaderMaximum ||
      blocks.width > kShaderMaximum / 2 || blocks.height > kShaderMaximum ||
      !blocks.try_area(&block_count) || block_count > kShaderMaximum) {
    return Status::InvalidArgument(
      "Prepared AQ geometry exceeds Metal shader limits");
  }
  return Status::Ok();
}

[[nodiscard]] Status AddPlannedPlane(
  DeviceElementType type,
  Extent2D extent,
  size_t row_stride,
  size_t* bytes) {

  if (bytes == nullptr || extent.empty() || row_stride < extent.width) {
    return Status::InvalidArgument("Prepared AQ plane plan is invalid");
  }
  const size_t element_size = DeviceElementSize(type);
  if (*bytes > std::numeric_limits<size_t>::max() - (kBufferAlignment - 1)) {
    return Status::InvalidArgument("Prepared AQ plane alignment overflows");
  }
  const size_t aligned =
    (*bytes + kBufferAlignment - 1) & ~(kBufferAlignment - 1);
  if (extent.height - 1 >
      (std::numeric_limits<size_t>::max() - extent.width) / row_stride) {
    return Status::InvalidArgument("Prepared AQ plane geometry overflows");
  }
  const size_t elements = (extent.height - 1) * row_stride + extent.width;
  if (elements > std::numeric_limits<size_t>::max() / element_size) {
    return Status::InvalidArgument("Prepared AQ plane byte size overflows");
  }
  const size_t plane_bytes = elements * element_size;
  if (aligned > std::numeric_limits<size_t>::max() - plane_bytes) {
    return Status::InvalidArgument("Prepared AQ arena size overflows");
  }
  *bytes = aligned + plane_bytes;
  return Status::Ok();
}

template <typename T>
[[nodiscard]] Status UploadPlane(
  MetalBackend& backend,
  PlaneView<const T> source,
  DevicePlaneView destination) {

  const size_t row_bytes = source.extent.width * sizeof(T);
  for (size_t y = 0; y < source.extent.height; ++y) {
    Status status = backend.CopyHostToDevice(
      *destination.buffer,
      source.Row(y),
      row_bytes,
      destination.offset_bytes + y * destination.row_stride * sizeof(T));
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

template <typename Range>
void AppendQuantTable(const Range& values, std::vector<float>* packed) {
  packed->insert(packed->end(), values.begin(), values.end());
}

[[nodiscard]] Status PackQuantTables(std::vector<float>* packed) {
  if (packed == nullptr) {
    return Status::Internal("Prepared AQ quantization table output is null");
  }
  try {
    packed->clear();
    packed->reserve(kQuantTableValueCount);
    AppendQuantTable(quantization_internal::kDct8Dequant, packed);
    AppendQuantTable(quantization_internal::kDct8InverseDequant, packed);
    AppendQuantTable(quantization_internal::kDct16Dequant, packed);
    AppendQuantTable(quantization_internal::kDct16InverseDequant, packed);
    AppendQuantTable(quantization_internal::kDct32Dequant, packed);
    AppendQuantTable(quantization_internal::kDct32InverseDequant, packed);
    AppendQuantTable(quantization_internal::kDct8x16Dequant, packed);
    AppendQuantTable(quantization_internal::kDct8x16InverseDequant, packed);
    AppendQuantTable(quantization_internal::kDct16x32Dequant, packed);
    AppendQuantTable(quantization_internal::kDct16x32InverseDequant, packed);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to pack prepared AQ quantization tables");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Prepared AQ quantization table pack is too large");
  }
  if (packed->size() != kQuantTableValueCount) {
    return Status::Internal("Prepared AQ quantization table layout changed");
  }
  return Status::Ok();
}

[[nodiscard]] Status CreateAqPipeline(
  MTL::Device* device,
  MTL::Library* library,
  std::string_view function_name,
  NS::SharedPtr<MTL::ComputePipelineState>* out) {

  if (device == nullptr || library == nullptr || function_name.empty() ||
      out == nullptr) {
    return Status::InvalidArgument(
      "CreateAqPipeline received invalid argument");
  }
  const std::string function_name_string(function_name);
  auto function = NS::TransferPtr(library->newFunction(NS::String::string(
    function_name_string.c_str(), NS::UTF8StringEncoding)));
  if (!function) {
    return Status::Internal(
      std::string("Metal function not found: ") + function_name_string);
  }
  if (function->functionType() != MTL::FunctionTypeKernel) {
    return Status::InvalidArgument(
      std::string("Metal function is not a kernel: ") + function_name_string);
  }
  NS::Error* error = nullptr;
  auto pipeline = NS::TransferPtr(
    device->newComputePipelineState(function.get(), &error));
  if (!pipeline) {
    return metal::ErrorToStatus(error, "newComputePipelineState");
  }
  RegisterMetalComputePipeline(pipeline.get(), function_name);
  *out = std::move(pipeline);
  return Status::Ok();
}

}  // namespace

MetalPreparedAqEvaluation::MetalPreparedAqEvaluation(MetalBackend &backend)
    : backend_(&backend) {}

MetalPreparedAqEvaluation::~MetalPreparedAqEvaluation() {
  std::unique_ptr<GpuSubmission> submission;
  bool *observer = nullptr;
  {
    std::lock_guard lock(mutex_);
    submission = std::move(submission_);
    observer = wait_observer_;
  }
  if (submission != nullptr) {
    (void)submission->Wait();
    if (observer != nullptr) {
      *observer = true;
    }
  }
}

Status MetalPreparedAqEvaluation::Prepare(
  const AqEvaluationPreparation& preparation,
  gpu_profile_internal::GpuProfilingMode profiling_mode,
  gpu_profile_internal::GpuExecutionProfile* profile) {

  const bool profiling =
    profiling_mode != gpu_profile_internal::GpuProfilingMode::kDisabled;
  if (profiling != (profile != nullptr)) {
    return Status::InvalidArgument(
      "Prepared AQ profiling request is invalid");
  }
  gpu_profile_internal::GpuExecutionProfile candidate_profile;
  if (profiling) {
    Status profile_status = InitializeGpuExecutionProfile(
      profiling_mode, &candidate_profile);
    if (!profile_status.ok()) return profile_status;
  }
  Status status = ValidatePreparation(preparation);
  if (!status.ok()) {
    return status;
  }

  source_extent_ = preparation.original_linear_rgb.extent();
  coding_extent_ = preparation.coding_opsin.extent();
  block_extent_ = {
      coding_extent_.width / kJxlBlockDimension,
      coding_extent_.height / kJxlBlockDimension,
  };
  tile_extent_ = {
      (coding_extent_.width + 63) / 64,
      (coding_extent_.height + 63) / 64,
  };
  options_ = preparation.options;
  coefficient_decision_mode_ = preparation.coefficient_decision_mode;
  frame_only_ = preparation.frame_only;
  frame_only_inverse_gaborish_ =
      preparation.frame_only_inverse_gaborish;
  resident_initial_cfl_ = preparation.resident_initial_cfl;
  frame_only_resident_initial_quant_ =
      preparation.frame_only_resident_initial_quant;
  resident_ac_strategy_inputs_ =
      preparation.resident_ac_strategy_inputs;
  frame_only_resident_quantizer_ =
      preparation.frame_only_resident_quantizer;
  resident_quantization_ = preparation.resident_quantization;
  borrowed_coding_opsin_ =
    preparation.resident_coding_opsin.plane[0].buffer != nullptr;
  const size_t filter_stage_count =
    (options_.profile.loop_filter.gaborish ? size_t{1} : size_t{0}) +
    options_.profile.loop_filter.epf_options.iterations;
  filter_scratch_image_count_ =
      frame_only_ ? 0 : std::min<size_t>(2, filter_stage_count);
  final_filter_scratch_index_ = frame_only_ || filter_stage_count == 0
      ? -1
      : static_cast<int>((filter_stage_count - 1) % 2);
  const bool needs_reconstructed = !frame_only_ ||
      frame_only_inverse_gaborish_ ||
      (resident_ac_strategy_inputs_ &&
       options_.profile.loop_filter.gaborish);
  (void)block_extent_.try_area(&block_count_);
  (void)coding_extent_.try_area(&pixel_count_);
  if (frame_only_resident_quantizer_) {
    initial_quant_sort_count_ = 1;
    while (initial_quant_sort_count_ < block_count_) {
      if (initial_quant_sort_count_ >
          std::numeric_limits<uint32_t>::max() / 2) {
        return Status::InvalidArgument(
            "Resident initial-quant sort dimensions are too large");
      }
      initial_quant_sort_count_ *= 2;
    }
  }
  if (pixel_count_ >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) / 3) {
    return Status::InvalidArgument(
        "Prepared AQ coefficient storage exceeds Metal shader limits");
  }
  coefficient_value_count_ = 3 * pixel_count_;

  std::vector<int32_t> strategy_records;
  std::vector<int32_t> anchor_records;
  std::vector<int32_t> color_transform_records;
  std::vector<int32_t> color_tile_offsets;
  std::vector<float> quant_tables;
  std::array<std::vector<std::array<int32_t, 2>>, 7> grouped_anchors;
  try {
    if (block_count_ > std::numeric_limits<size_t>::max() / 2) {
      return Status::InvalidArgument(
          "Prepared AQ strategy-record count overflows");
    }
    strategy_records.resize(block_count_ * 2);
    if (!frame_only_) {
      readback_.resize(block_count_);
    }
    strategies_host_ = *preparation.strategies;
    epf_sharpness_host_.resize(block_count_);
    last_raw_quant_.resize(block_count_);
    size_t tile_count = 0;
    if (!tile_extent_.try_area(&tile_count)) {
      return Status::InvalidArgument(
        "Prepared AQ color-tile extent is too large");
    }
    last_y_to_x_.resize(tile_count);
    last_y_to_b_.resize(tile_count);
    if (frame_only_resident_initial_quant_) {
      last_initial_quant_field_.resize(block_count_);
      last_initial_strategy_mask_.resize(block_count_);
      last_initial_pixel_mask_.resize(pixel_count_);
    }
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Unable to allocate prepared AQ host staging");
  } catch (const std::length_error &) {
    return Status::InvalidArgument("Prepared AQ host staging is too large");
  }

  for (size_t y = 0; y < block_extent_.height; ++y) {
    std::copy_n(
      preparation.epf_sharpness.Row(y), block_extent_.width,
      epf_sharpness_host_.data() + y * block_extent_.width);
  }

  for (size_t y = 0; y < block_extent_.height; ++y) {
    for (size_t x = 0; x < block_extent_.width; ++x) {
      AcStrategyCell cell;
      status = preparation.strategies->Get(x, y, &cell);
      if (!status.ok()) {
        return status;
      }
      const size_t index = 2 * (y * block_extent_.width + x);
      strategy_records[index] = static_cast<int32_t>(cell.strategy);
      strategy_records[index + 1] = cell.is_anchor ? 1 : 0;
      if (cell.is_anchor) {
        const size_t batch_index = AqStrategyBatchIndex(cell.strategy);
        if (batch_index >= grouped_anchors.size()) {
          return Status::Internal(
              "Validated AQ strategy has no reconstruction batch");
        }
        const size_t index_in_batch = grouped_anchors[batch_index].size();
        grouped_anchors[batch_index].push_back(
            {static_cast<int32_t>(x), static_cast<int32_t>(y)});
        row_major_anchors_.push_back(
            {x, y, cell.strategy, batch_index, index_in_batch});
      }
    }
  }
  anchor_count_ = row_major_anchors_.size();
  const size_t anchor_capacity_count =
      frame_only_ ? anchor_count_ : block_count_;
  if (!frame_only_) {
    try {
      transform_maximum_error_readback_.resize(3 * block_count_);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate maximum-error readback storage");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Maximum-error readback storage is too large");
    }
  }
  size_t coefficient_offset = 0;
  size_t anchor_offset = 0;
  for (size_t batch_index = 0; batch_index < batches_.size(); ++batch_index) {
    const AcStrategyType strategy = kSupportedAqStrategies[batch_index];
    const AcStrategyInfo *info = GetAcStrategyInfo(strategy);
    if (info == nullptr) {
      return Status::Internal("AQ reconstruction strategy disappeared");
    }
    const size_t coefficient_count = info->coefficient_count();
    const size_t count = grouped_anchors[batch_index].size();
    batches_[batch_index] = {strategy, anchor_offset, count, coefficient_offset,
                             coefficient_count};
    maximum_coefficient_count_ =
        std::max(maximum_coefficient_count_, coefficient_count);
    anchor_offset += count;
    coefficient_offset += 3 * count * coefficient_count;
    for (const auto &anchor : grouped_anchors[batch_index]) {
      anchor_records.push_back(anchor[0]);
      anchor_records.push_back(anchor[1]);
    }
  }
  if (anchor_offset != anchor_count_ ||
      coefficient_offset != coefficient_value_count_) {
    return Status::Internal(
        "Prepared AQ anchors do not cover the coding image exactly");
  }
  try {
    quantized_readback_.resize(coefficient_value_count_);
    quantized_dc_readback_.resize(3 * block_count_);
    final_transform_views_.reserve(block_count_);
    for (const AqAnchor& anchor : row_major_anchors_) {
      const AqStrategyBatch& batch = batches_[anchor.batch_index];
      const size_t channel_stride =
        batch.anchor_count * batch.coefficient_count;
      vardct_frame_internal::QuantizedAcTransformView transform{
        .block_x = anchor.block_x,
        .block_y = anchor.block_y,
        .strategy = anchor.strategy,
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        const size_t offset = batch.coefficient_offset +
          channel * channel_stride +
          anchor.index_in_batch * batch.coefficient_count;
        transform.coefficients[channel] = std::span<const int32_t>(
          quantized_readback_.data() + offset,
          batch.coefficient_count);
      }
      final_transform_views_.push_back(transform);
    }
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory(
        "Unable to allocate AQ reconstruction host readback");
  } catch (const std::length_error &) {
    return Status::InvalidArgument(
        "AQ reconstruction host readback is too large");
  }
  status = PackQuantTables(&quant_tables);
  if (!status.ok()) {
    return status;
  }
  if (resident_quantization_) {
    status = BuildColorTransformMetadata(
      row_major_anchors_, batches_, block_extent_, tile_extent_,
      &color_transform_records, &color_tile_offsets);
    if (!status.ok()) return status;
  }

  size_t persistent_bytes = 0;
  if (!frame_only_) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, source_extent_,
                               source_extent_.width, &persistent_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (!borrowed_coding_opsin_) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, coding_extent_,
                               coding_extent_.width, &persistent_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (needs_reconstructed) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, coding_extent_,
                               coding_extent_.width, &persistent_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (!frame_only_) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, source_extent_,
                               source_extent_.width, &persistent_bytes);
      if (!status.ok())
        return status;
    }
    status = AddPlannedPlane(DeviceElementType::kI32,
                             {block_extent_.width * 2, block_extent_.height},
                             block_extent_.width * 2, &persistent_bytes);
    if (!status.ok())
      return status;
  }
  status = AddPlannedPlane(
      DeviceElementType::kI32, {2 * anchor_capacity_count, 1},
      2 * anchor_capacity_count, &persistent_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kU8, block_extent_,
                           block_extent_.width, &persistent_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, {kQuantTableValueCount, 1},
                           kQuantTableValueCount, &persistent_bytes);
  if (!status.ok())
    return status;
  if (resident_quantization_) {
    status = AddPlannedPlane(
      DeviceElementType::kI32, {6 * block_count_, 1},
      6 * block_count_, &persistent_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(
      DeviceElementType::kI32, {color_tile_offsets.size(), 1},
      color_tile_offsets.size(), &persistent_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kI8, tile_extent_,
                             tile_extent_.width, &persistent_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kI8, tile_extent_,
                             tile_extent_.width, &persistent_bytes);
    if (!status.ok()) return status;
  }

  size_t staging_bytes = 0;
  status = AddPlannedPlane(DeviceElementType::kI32, block_extent_,
                           block_extent_.width, &staging_bytes);
  if (!status.ok())
    return status;
  if (!frame_only_) {
    status = AddPlannedPlane(DeviceElementType::kF32, block_extent_,
                             block_extent_.width, &staging_bytes);
    if (!status.ok())
      return status;
  }
  if (frame_only_resident_initial_quant_) {
    const Extent2D pre_erosion_extent{
      coding_extent_.width / 4, coding_extent_.height / 4};
    status = AddPlannedPlane(DeviceElementType::kF32, pre_erosion_extent,
                             pre_erosion_extent.width, &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kF32, coding_extent_,
                             coding_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kF32, block_extent_,
                             block_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kF32, block_extent_,
                             block_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kF32, coding_extent_,
                             coding_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    if (frame_only_resident_quantizer_) {
      status = AddPlannedPlane(
        DeviceElementType::kF32, {initial_quant_sort_count_, 1},
        initial_quant_sort_count_, &staging_bytes);
      if (!status.ok()) return status;
      status = AddPlannedPlane(
        DeviceElementType::kF32, {1, 1}, 1, &staging_bytes);
      if (!status.ok()) return status;
      status = AddPlannedPlane(
        DeviceElementType::kI32, {2, 1}, 2, &staging_bytes);
      if (!status.ok()) return status;
    }
  }
  if (resident_quantization_) {
    status = AddPlannedPlane(DeviceElementType::kF32, block_extent_,
                             block_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kF32, block_extent_,
                             block_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kF32, {5, 1}, 5,
                             &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kI32, {256, 1}, 256,
                             &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kI32, {3, 1}, 3,
                             &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kF32, {2, 1}, 2,
                             &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kI32, {2, 1}, 2,
                             &staging_bytes);
    if (!status.ok()) return status;
  }
  for (size_t image = 0; image < filter_scratch_image_count_; ++image) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, coding_extent_,
                               coding_extent_.width, &staging_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (!resident_quantization_) {
    status = AddPlannedPlane(DeviceElementType::kI8, tile_extent_,
                             tile_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
    status = AddPlannedPlane(DeviceElementType::kI8, tile_extent_,
                             tile_extent_.width, &staging_bytes);
    if (!status.ok()) return status;
  }
  if (!frame_only_) {
    status = AddPlannedPlane(DeviceElementType::kF32, block_extent_,
                             block_extent_.width, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32, source_extent_,
                             source_extent_.width, &staging_bytes);
    if (!status.ok())
      return status;
    status =
        AddPlannedPlane(DeviceElementType::kF32, {1, 1}, 1, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(
        DeviceElementType::kF32, {3 * block_count_, 1}, 3 * block_count_,
        &staging_bytes);
    if (!status.ok())
      return status;
  }
  status =
      AddPlannedPlane(DeviceElementType::kF32, {coefficient_value_count_, 1},
                      coefficient_value_count_, &staging_bytes);
  if (!status.ok())
    return status;
  status =
      AddPlannedPlane(DeviceElementType::kF32, {coefficient_value_count_, 1},
                      coefficient_value_count_, &staging_bytes);
  if (!status.ok())
    return status;
  status =
      AddPlannedPlane(DeviceElementType::kI32, {coefficient_value_count_, 1},
                      coefficient_value_count_, &staging_bytes);
  if (!status.ok())
    return status;
  if (!frame_only_) {
    status = AddPlannedPlane(
        DeviceElementType::kF32, {coefficient_value_count_, 1},
        coefficient_value_count_, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32,
                             {3 * block_count_, 1}, 3 * block_count_,
                             &staging_bytes);
    if (!status.ok())
      return status;
  }
  status = AddPlannedPlane(DeviceElementType::kI32, {3 * block_count_, 1},
                           3 * block_count_, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kI32, {1, 1}, 1, &staging_bytes);
  if (!status.ok())
    return status;
  if (!frame_only_) {
    status = AddPlannedPlane(
        DeviceElementType::kF32, {maximum_coefficient_count_, 1},
        maximum_coefficient_count_, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(
        DeviceElementType::kI32, {maximum_coefficient_count_, 1},
        maximum_coefficient_count_, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(
        DeviceElementType::kF32, {maximum_coefficient_count_, 1},
        maximum_coefficient_count_, &staging_bytes);
    if (!status.ok())
      return status;
  }

  status = persistent_.Prepare(*backend_, persistent_bytes);
  if (!status.ok())
    return status;
  status = staging_.Prepare(*backend_, staging_bytes);
  if (!status.ok())
    return status;

  if (!frame_only_) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = persistent_.AllocatePlane(
          DeviceElementType::kF32, source_extent_, source_extent_.width,
          kBufferAlignment, &original_[channel]);
      if (!status.ok())
        return status;
    }
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (borrowed_coding_opsin_) {
      const ConstDevicePlaneView plane =
        preparation.resident_coding_opsin.plane[channel];
      coding_[channel] = {
        const_cast<DeviceBuffer*>(plane.buffer), plane.offset_bytes,
        plane.element_type, plane.extent, plane.row_stride};
    } else {
      status = persistent_.AllocatePlane(
          DeviceElementType::kF32, coding_extent_, coding_extent_.width,
          kBufferAlignment, &coding_[channel]);
      if (!status.ok())
        return status;
    }
  }
  if (needs_reconstructed) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = persistent_.AllocatePlane(
          DeviceElementType::kF32, coding_extent_, coding_extent_.width,
          kBufferAlignment, &reconstructed_[channel]);
      if (!status.ok())
        return status;
    }
  }
  if (!frame_only_) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = persistent_.AllocatePlane(
          DeviceElementType::kF32, source_extent_, source_extent_.width,
          kBufferAlignment, &reconstructed_linear_[channel]);
      if (!status.ok())
        return status;
    }
    status = persistent_.AllocatePlane(
        DeviceElementType::kI32,
        {block_extent_.width * 2, block_extent_.height},
        block_extent_.width * 2, kBufferAlignment, &strategies_);
    if (!status.ok())
      return status;
  }
  status =
      persistent_.AllocatePlane(
          DeviceElementType::kI32, {2 * anchor_capacity_count, 1},
          2 * anchor_capacity_count, kBufferAlignment, &anchors_);
  if (!status.ok())
    return status;
  status = persistent_.AllocatePlane(
      DeviceElementType::kU8, block_extent_, block_extent_.width,
      kBufferAlignment, &epf_sharpness_);
  if (!status.ok())
    return status;
  status = persistent_.AllocatePlane(
      DeviceElementType::kF32, {kQuantTableValueCount, 1},
      kQuantTableValueCount, kBufferAlignment, &quant_tables_);
  if (!status.ok())
    return status;
  if (resident_quantization_) {
    status = persistent_.AllocatePlane(
      DeviceElementType::kI32, {6 * block_count_, 1},
      6 * block_count_, kBufferAlignment,
      &color_transform_records_);
    if (!status.ok()) return status;
    status = persistent_.AllocatePlane(
      DeviceElementType::kI32, {color_tile_offsets.size(), 1},
      color_tile_offsets.size(), kBufferAlignment, &color_tile_offsets_);
    if (!status.ok()) return status;
  }

  status = staging_.AllocatePlane(DeviceElementType::kI32, block_extent_,
                                  block_extent_.width, kBufferAlignment,
                                  &raw_quant_);
  if (!status.ok())
    return status;
  if (!frame_only_) {
    status = staging_.AllocatePlane(DeviceElementType::kF32, block_extent_,
                                    block_extent_.width, kBufferAlignment,
                                    &inverse_sigma_);
    if (!status.ok())
      return status;
  }
  if (frame_only_resident_initial_quant_) {
    const Extent2D pre_erosion_extent{
      coding_extent_.width / 4, coding_extent_.height / 4};
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, pre_erosion_extent, pre_erosion_extent.width,
      kBufferAlignment, &initial_quant_pre_erosion_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, coding_extent_, coding_extent_.width,
      kBufferAlignment, &initial_quant_unblurred_pixel_mask_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, block_extent_, block_extent_.width,
      kBufferAlignment, &initial_quant_field_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, block_extent_, block_extent_.width,
      kBufferAlignment, &initial_quant_strategy_mask_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, coding_extent_, coding_extent_.width,
      kBufferAlignment, &initial_quant_pixel_mask_);
    if (!status.ok()) return status;
    if (frame_only_resident_quantizer_) {
      status = staging_.AllocatePlane(
        DeviceElementType::kF32, {initial_quant_sort_count_, 1},
        initial_quant_sort_count_, kBufferAlignment, &initial_quant_sort_);
      if (!status.ok()) return status;
      status = staging_.AllocatePlane(
        DeviceElementType::kF32, {1, 1}, 1, kBufferAlignment,
        &initial_quant_median_);
      if (!status.ok()) return status;
      status = staging_.AllocatePlane(
        DeviceElementType::kI32, {2, 1}, 2, kBufferAlignment,
        &initial_quantizer_params_);
      if (!status.ok()) return status;
    }
  }
  if (resident_quantization_) {
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, block_extent_, block_extent_.width,
      kBufferAlignment, &resident_quant_field_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, block_extent_, block_extent_.width,
      kBufferAlignment, &resident_policy_initial_field_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, {5, 1}, 5, kBufferAlignment,
      &resident_policy_scores_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kI32, {256, 1}, 256, kBufferAlignment,
      &resident_quant_histogram_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kI32, {3, 1}, 3, kBufferAlignment,
      &resident_quant_selection_state_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kF32, {2, 1}, 2, kBufferAlignment,
      &resident_quant_statistics_);
    if (!status.ok()) return status;
    status = staging_.AllocatePlane(
      DeviceElementType::kI32, {2, 1}, 2, kBufferAlignment,
      &resident_quantizer_params_);
    if (!status.ok()) return status;
  }
  for (size_t image = 0; image < filter_scratch_image_count_; ++image) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = staging_.AllocatePlane(DeviceElementType::kF32, coding_extent_,
                                      coding_extent_.width, kBufferAlignment,
                                      &filter_scratch_[image][channel]);
      if (!status.ok())
        return status;
    }
  }
  DeviceScratchArena& color_arena =
      resident_quantization_ ? persistent_ : staging_;
  status =
      color_arena.AllocatePlane(DeviceElementType::kI8, tile_extent_,
                                tile_extent_.width, kBufferAlignment, &y_to_x_);
  if (!status.ok())
    return status;
  status =
      color_arena.AllocatePlane(DeviceElementType::kI8, tile_extent_,
                                tile_extent_.width, kBufferAlignment, &y_to_b_);
  if (!status.ok())
    return status;
  if (!frame_only_) {
    status = staging_.AllocatePlane(DeviceElementType::kF32, block_extent_,
                                    block_extent_.width, kBufferAlignment,
                                    &block_distance_);
    if (!status.ok())
      return status;
    status = staging_.AllocatePlane(DeviceElementType::kF32, source_extent_,
                                    source_extent_.width, kBufferAlignment,
                                    &distance_map_);
    if (!status.ok())
      return status;
    status = staging_.AllocatePlane(DeviceElementType::kF32, {1, 1}, 1,
                                    kBufferAlignment, &score_);
    if (!status.ok())
      return status;
    status = staging_.AllocatePlane(
        DeviceElementType::kF32, {3 * block_count_, 1}, 3 * block_count_,
        kBufferAlignment, &transform_maximum_error_);
    if (!status.ok())
      return status;
  }
  status = staging_.AllocatePlane(
      DeviceElementType::kF32, {coefficient_value_count_, 1},
      coefficient_value_count_, kBufferAlignment, &gathered_pixels_);
  if (!status.ok())
    return status;
  status = staging_.AllocatePlane(
      DeviceElementType::kF32, {coefficient_value_count_, 1},
      coefficient_value_count_, kBufferAlignment, &forward_coefficients_);
  if (!status.ok())
    return status;
  status = staging_.AllocatePlane(
      DeviceElementType::kI32, {coefficient_value_count_, 1},
      coefficient_value_count_, kBufferAlignment, &quantized_coefficients_);
  if (!status.ok())
    return status;
  if (!frame_only_) {
    status = staging_.AllocatePlane(DeviceElementType::kF32,
                                    {coefficient_value_count_, 1},
                                    coefficient_value_count_, kBufferAlignment,
                                    &reconstruction_coefficients_);
    if (!status.ok())
      return status;
    status = staging_.AllocatePlane(
        DeviceElementType::kF32, {3 * block_count_, 1}, 3 * block_count_,
        kBufferAlignment, &dc_);
    if (!status.ok())
      return status;
  }
  status = staging_.AllocatePlane(
      DeviceElementType::kI32, {3 * block_count_, 1}, 3 * block_count_,
      kBufferAlignment, &quantized_dc_);
  if (!status.ok())
    return status;
  status = staging_.AllocatePlane(DeviceElementType::kI32, {1, 1}, 1,
                                  kBufferAlignment, &reconstruction_error_);
  if (!status.ok())
    return status;
  if (!frame_only_) {
    status = staging_.AllocatePlane(
        DeviceElementType::kF32, {maximum_coefficient_count_, 1},
        maximum_coefficient_count_, kBufferAlignment, &quant_probe_input_);
    if (!status.ok())
      return status;
    status = staging_.AllocatePlane(
        DeviceElementType::kI32, {maximum_coefficient_count_, 1},
        maximum_coefficient_count_, kBufferAlignment, &quant_probe_quantized_);
    if (!status.ok())
      return status;
    status = staging_.AllocatePlane(
        DeviceElementType::kF32, {maximum_coefficient_count_, 1},
        maximum_coefficient_count_, kBufferAlignment,
        &quant_probe_dequantized_);
    if (!status.ok())
      return status;
  }

  for (size_t channel = 0; channel < 3; ++channel) {
    if (!frame_only_) {
      status = UploadPlane(
          *backend_, preparation.original_linear_rgb.plane[channel],
          original_[channel]);
      if (!status.ok())
        return status;
    }
    if (!borrowed_coding_opsin_) {
      status = UploadPlane(*backend_, preparation.coding_opsin.plane[channel],
                           coding_[channel]);
      if (!status.ok())
        return status;
    }
  }
  if (!frame_only_) {
    status = UploadPlane(
        *backend_,
        ConstPlaneI32View{strategy_records.data(), strategies_.extent,
                          strategies_.row_stride},
        strategies_);
    if (!status.ok())
      return status;
  }
  status = UploadPlane(*backend_,
                       ConstPlaneI32View{
                         anchor_records.data(), {2 * anchor_count_, 1},
                         2 * anchor_count_},
                       anchors_);
  if (!status.ok())
    return status;
  status = UploadPlane(*backend_, preparation.epf_sharpness, epf_sharpness_);
  if (!status.ok())
    return status;
  status =
      UploadPlane(*backend_,
                  ConstPlaneF32View{quant_tables.data(), quant_tables_.extent,
                                    quant_tables_.row_stride},
                  quant_tables_);
  if (!status.ok())
    return status;
  if (resident_quantization_) {
    status = UploadPlane(
      *backend_,
      ConstPlaneI32View{
        color_transform_records.data(), {color_transform_records.size(), 1},
        color_transform_records.size()},
      color_transform_records_);
    if (status.ok()) {
      status = UploadPlane(
        *backend_,
        ConstPlaneI32View{
          color_tile_offsets.data(), color_tile_offsets_.extent,
          color_tile_offsets_.row_stride},
        color_tile_offsets_);
    }
    if (!status.ok()) return status;
  }

  for (size_t batch_index = 0; batch_index < batches_.size(); ++batch_index) {
    const AqStrategyBatch &batch = batches_[batch_index];
    const AcStrategyInfo *info = GetAcStrategyInfo(batch.strategy);
    if (info == nullptr) {
      return Status::Internal("AQ reconstruction strategy disappeared");
    }
    reconstruction_params_[batch_index] = {
        static_cast<uint32_t>(coding_extent_.width),
        static_cast<uint32_t>(coding_extent_.height),
        static_cast<uint32_t>(coding_[0].row_stride),
        static_cast<uint32_t>(block_extent_.width),
        static_cast<uint32_t>(block_extent_.height),
        static_cast<uint32_t>(raw_quant_.row_stride),
        static_cast<uint32_t>(tile_extent_.width),
        static_cast<uint32_t>(y_to_x_.row_stride),
        static_cast<uint32_t>(batch.anchor_offset),
        static_cast<uint32_t>(batch.anchor_count),
        static_cast<uint32_t>(batch.coefficient_offset),
        static_cast<uint32_t>(batch.coefficient_count),
        static_cast<uint32_t>(info->pixel_extent().width),
        static_cast<uint32_t>(info->pixel_extent().height),
        static_cast<uint32_t>(info->covered_blocks.width),
        static_cast<uint32_t>(info->covered_blocks.height),
        static_cast<uint32_t>(batch.strategy),
        0,
        0,
        QuantizationMatrixMultiplier(options_.profile.x_qm_scale),
        QuantizationMatrixMultiplier(options_.profile.b_qm_scale),
        coefficient_decision_mode_ ==
            AcCoefficientDecisionMode::kAdjustedSharedQuant
          ? 1u
          : 0u,
        static_cast<uint32_t>(inverse_sigma_.row_stride),
        static_cast<uint32_t>(epf_sharpness_.row_stride),
        options_.profile.epf_sigma.quant_multiplier,
        options_.profile.epf_sigma.sharpness_lut,
    };
    if (!frame_only_) {
      block_reduction_params_[batch_index] = {
          static_cast<uint32_t>(source_extent_.width),
          static_cast<uint32_t>(source_extent_.height),
          static_cast<uint32_t>(distance_map_.row_stride),
          static_cast<uint32_t>(block_distance_.row_stride),
          static_cast<uint32_t>(batch.anchor_offset),
          static_cast<uint32_t>(batch.anchor_count),
          static_cast<uint32_t>(info->pixel_extent().width),
          static_cast<uint32_t>(info->pixel_extent().height),
          static_cast<uint32_t>(info->covered_blocks.width),
          static_cast<uint32_t>(info->covered_blocks.height),
      };
      maximum_error_reduction_params_[batch_index] = {
          static_cast<uint32_t>(source_extent_.width),
          static_cast<uint32_t>(source_extent_.height),
          static_cast<uint32_t>(coding_[0].row_stride),
          static_cast<uint32_t>(FinalFilteredImage()[0].row_stride),
          static_cast<uint32_t>(block_distance_.row_stride),
          static_cast<uint32_t>(batch.anchor_offset),
          static_cast<uint32_t>(batch.anchor_count),
          static_cast<uint32_t>(info->pixel_extent().width),
          static_cast<uint32_t>(info->pixel_extent().height),
          static_cast<uint32_t>(info->covered_blocks.width),
          static_cast<uint32_t>(info->covered_blocks.height),
          options_.maximum_error[0],
          options_.maximum_error[1],
          options_.maximum_error[2],
      };
    }
  }
  reset_params_ = {
      static_cast<uint32_t>(coefficient_value_count_),
      static_cast<uint32_t>(3 * block_count_),
      static_cast<uint32_t>(pixel_count_),
      static_cast<uint32_t>(block_count_),
      0,
  };
  initial_cfl_params_ = {
      static_cast<uint32_t>(coding_extent_.width),
      static_cast<uint32_t>(coding_extent_.height),
      static_cast<uint32_t>(coding_[0].row_stride),
      static_cast<uint32_t>(tile_extent_.width),
      static_cast<uint32_t>(tile_extent_.height),
      static_cast<uint32_t>(y_to_x_.row_stride),
  };
  final_cfl_params_ = {
      static_cast<uint32_t>(tile_extent_.width),
      static_cast<uint32_t>(tile_extent_.height),
      static_cast<uint32_t>(y_to_x_.row_stride),
      static_cast<uint32_t>(anchor_count_),
  };
  if (frame_only_resident_initial_quant_) {
    initial_quant_gradient_params_ = {
        static_cast<uint32_t>(coding_extent_.width),
        static_cast<uint32_t>(coding_extent_.height),
        static_cast<uint32_t>(coding_[0].row_stride),
        static_cast<uint32_t>(initial_quant_unblurred_pixel_mask_.row_stride),
        static_cast<uint32_t>(initial_quant_pre_erosion_.extent.width),
        static_cast<uint32_t>(initial_quant_pre_erosion_.row_stride),
        0,
    };
    initial_quant_erosion_params_ = {
        static_cast<uint32_t>(initial_quant_pre_erosion_.extent.width),
        static_cast<uint32_t>(initial_quant_pre_erosion_.extent.height),
        static_cast<uint32_t>(initial_quant_pre_erosion_.row_stride),
        static_cast<uint32_t>(block_extent_.width),
        static_cast<uint32_t>(block_extent_.height),
        static_cast<uint32_t>(initial_quant_field_.row_stride),
        static_cast<uint32_t>(initial_quant_strategy_mask_.row_stride),
        {},
    };
    initial_quant_modulation_params_ = {
        static_cast<uint32_t>(coding_[0].row_stride),
        static_cast<uint32_t>(block_extent_.width),
        static_cast<uint32_t>(block_extent_.height),
        static_cast<uint32_t>(initial_quant_field_.row_stride),
        0.0f,
        0.0f,
    };
    if (frame_only_resident_quantizer_) {
      initial_quant_selection_params_ = {
          static_cast<uint32_t>(block_count_),
          static_cast<uint32_t>(initial_quant_sort_count_),
          static_cast<uint32_t>(block_count_ / 2),
          static_cast<uint32_t>(block_extent_.width),
          static_cast<uint32_t>(block_extent_.height),
          static_cast<uint32_t>(initial_quant_field_.row_stride),
          static_cast<uint32_t>(raw_quant_.row_stride),
          0,
          0.0f,
      };
    }
  }
  if (resident_quantization_) {
    resident_quant_selection_params_ = {
        static_cast<uint32_t>(block_count_),
        static_cast<uint32_t>(block_count_),
        static_cast<uint32_t>(block_count_ / 2),
        static_cast<uint32_t>(block_extent_.width),
        static_cast<uint32_t>(block_extent_.height),
        static_cast<uint32_t>(resident_quant_field_.row_stride),
        static_cast<uint32_t>(raw_quant_.row_stride),
        0,
        0.0f,
    };
    resident_policy_initialize_params_ = {
        static_cast<uint32_t>(block_extent_.width),
        static_cast<uint32_t>(block_extent_.height),
        static_cast<uint32_t>(resident_quant_field_.row_stride),
        static_cast<uint32_t>(resident_policy_initial_field_.row_stride),
        5,
    };
    resident_policy_update_params_ = {
        .block_width = static_cast<uint32_t>(block_extent_.width),
        .block_height = static_cast<uint32_t>(block_extent_.height),
        .quant_stride =
          static_cast<uint32_t>(resident_quant_field_.row_stride),
        .initial_stride =
          static_cast<uint32_t>(resident_policy_initial_field_.row_stride),
        .block_distance_stride =
          static_cast<uint32_t>(block_distance_.row_stride),
    };
  }

  if (!frame_only_) {
    gaborish_params_ = {
        static_cast<uint32_t>(source_extent_.width),
        static_cast<uint32_t>(source_extent_.height),
        static_cast<uint32_t>(reconstructed_[0].row_stride),
        static_cast<uint32_t>(filter_scratch_[0][0].row_stride),
    };
    for (size_t channel = 0; channel < 3; ++channel) {
      const float weight1 =
        options_.profile.loop_filter.gaborish_options.weight1[channel];
      const float weight2 =
        options_.profile.loop_filter.gaborish_options.weight2[channel];
      const float divisor = 1.0f + 4.0f * (weight1 + weight2);
      gaborish_params_.center_weight[channel] = 1.0f / divisor;
      gaborish_params_.axis_weight[channel] = weight1 / divisor;
      gaborish_params_.diagonal_weight[channel] = weight2 / divisor;
    }
    constexpr std::array<uint32_t, 3> kEpfPasses = {0, 1, 2};
    for (size_t index = 0; index < epf_params_.size(); ++index) {
      const uint32_t pass = kEpfPasses[index];
      const float pass_scale = pass == 0
        ? options_.profile.loop_filter.epf_options.pass0_sigma_scale
        : pass == 2
          ? options_.profile.loop_filter.epf_options.pass2_sigma_scale
          : 1.0f;
      epf_params_[index] = {
          static_cast<uint32_t>(source_extent_.width),
          static_cast<uint32_t>(source_extent_.height),
          static_cast<uint32_t>(filter_scratch_[0][0].row_stride),
          static_cast<uint32_t>(filter_scratch_[0][0].row_stride),
          static_cast<uint32_t>(inverse_sigma_.row_stride),
          pass,
          1.65f * pass_scale,
          options_.profile.loop_filter.epf_options.border_sad_multiplier,
          options_.profile.loop_filter.epf_options.channel_scale,
      };
    }
    opsin_to_linear_params_ = {
        static_cast<uint32_t>(source_extent_.width),
        static_cast<uint32_t>(source_extent_.height),
        static_cast<uint32_t>(coding_extent_.width),
        static_cast<uint32_t>(reconstructed_linear_[0].row_stride),
        255.0f / options_.profile.intensity_target,
    };
  }

  if (frame_only_) {
    memory_stats_ = {
        persistent_.capacity_bytes(),
        staging_.capacity_bytes(),
        staging_.capacity_bytes(),
    };
    if (profiling) *profile = std::move(candidate_profile);
    return Status::Ok();
  }
  DeviceButteraugliMemoryStats butteraugli_memory;
  if (options_.metric == AqEvaluationMetric::kButteraugli) {
    const DeviceButteraugliPrepareDescriptor descriptor{
      .reference_linear_rgb = {{{original_[0], original_[1], original_[2]}}},
      .options = options_.butteraugli,
    };
    if (profiling) {
      gpu_profile_internal::GpuExecutionProfile reference_profile;
      status = backend_->PrepareDeviceButteraugliImpl(
        descriptor, profiling_mode, &butteraugli_, &reference_profile);
      if (status.ok()) {
        if (reference_profile.mode != profiling_mode ||
            reference_profile.capabilities !=
              candidate_profile.capabilities) {
          return Status::Internal(
            "Butteraugli preparation profile metadata is inconsistent");
        }
        candidate_profile.submissions = std::move(
          reference_profile.submissions);
      }
    } else {
      status = PrepareDeviceButteraugli(
        *backend_, descriptor, &butteraugli_);
    }
    if (!status.ok())
      return status;
    butteraugli_memory = butteraugli_->memory_stats();
  }
  if (butteraugli_memory.prepared_allocation_bytes >
          std::numeric_limits<size_t>::max() - staging_.capacity_bytes() ||
      butteraugli_memory.peak_comparison_scratch_bytes >
          std::numeric_limits<size_t>::max() - staging_.capacity_bytes()) {
    return Status::InvalidArgument(
        "Prepared AQ Butteraugli memory accounting overflows");
  }

  memory_stats_ = {
      persistent_.capacity_bytes(),
      staging_.capacity_bytes() +
          butteraugli_memory.prepared_allocation_bytes,
      staging_.capacity_bytes() +
          butteraugli_memory.peak_comparison_scratch_bytes,
  };
  if (profiling) *profile = std::move(candidate_profile);
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::Reconfigure(
  const AcStrategyGrid& strategies,
  ConstPlaneU8View epf_sharpness) {

  if (frame_only_) {
    return Status::FailedPrecondition(
      "Frame-only prepared AQ state cannot be reconfigured");
  }
  if (!strategies.complete() || strategies.extent() != block_extent_ ||
      !ValidHostPlaneLayout(epf_sharpness) ||
      epf_sharpness.extent != block_extent_) {
    return Status::InvalidArgument(
      "Prepared AQ reconfiguration geometry is invalid");
  }
  for (size_t y = 0; y < block_extent_.height; ++y) {
    for (size_t x = 0; x < block_extent_.width; ++x) {
      AcStrategyCell cell;
      const Status status = strategies.Get(x, y, &cell);
      if (!status.ok() || !SupportedAqStrategy(cell.strategy) ||
          epf_sharpness.Row(y)[x] >= 8) {
        return Status::InvalidArgument(
          "Prepared AQ reconfiguration metadata is invalid");
      }
    }
  }

  Status status = BeginOperation();
  if (!status.ok()) {
    return status;
  }
  try {
    std::vector<int32_t> strategy_records(2 * block_count_);
    std::vector<int32_t> anchor_records;
    anchor_records.reserve(2 * block_count_);
    std::array<std::vector<std::array<int32_t, 2>>, 7> grouped_anchors;
    std::vector<AqAnchor> row_major_anchors;
    row_major_anchors.reserve(block_count_);
    std::vector<uint8_t> sharpness(block_count_);
    for (size_t y = 0; y < block_extent_.height; ++y) {
      std::copy_n(
        epf_sharpness.Row(y), block_extent_.width,
        sharpness.data() + y * block_extent_.width);
      for (size_t x = 0; x < block_extent_.width; ++x) {
        AcStrategyCell cell;
        status = strategies.Get(x, y, &cell);
        if (!status.ok()) {
          Invalidate();
          return status;
        }
        const size_t record = 2 * (y * block_extent_.width + x);
        strategy_records[record] = static_cast<int32_t>(cell.strategy);
        strategy_records[record + 1] = cell.is_anchor ? 1 : 0;
        if (cell.is_anchor) {
          const size_t batch_index = AqStrategyBatchIndex(cell.strategy);
          const size_t index_in_batch = grouped_anchors[batch_index].size();
          grouped_anchors[batch_index].push_back(
            {static_cast<int32_t>(x), static_cast<int32_t>(y)});
          row_major_anchors.push_back(
            {x, y, cell.strategy, batch_index, index_in_batch});
        }
      }
    }

    std::array<AqStrategyBatch, 7> batches;
    std::array<AqReconstructionParams, 7> reconstruction_params;
    std::array<AqBlockReductionParams, 7> block_reduction_params;
    std::array<AqMaximumErrorReductionParams, 7>
      maximum_error_reduction_params;
    size_t coefficient_offset = 0;
    size_t anchor_offset = 0;
    for (size_t batch_index = 0; batch_index < batches.size(); ++batch_index) {
      const AcStrategyType strategy = kSupportedAqStrategies[batch_index];
      const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
      if (info == nullptr) {
        Invalidate();
        return Status::Internal(
          "AQ reconfiguration strategy disappeared");
      }
      const size_t coefficient_count = info->coefficient_count();
      const size_t count = grouped_anchors[batch_index].size();
      batches[batch_index] = {
        strategy, anchor_offset, count, coefficient_offset,
        coefficient_count};
      for (const auto& anchor : grouped_anchors[batch_index]) {
        anchor_records.push_back(anchor[0]);
        anchor_records.push_back(anchor[1]);
      }
      reconstruction_params[batch_index] = {
        static_cast<uint32_t>(coding_extent_.width),
        static_cast<uint32_t>(coding_extent_.height),
        static_cast<uint32_t>(coding_[0].row_stride),
        static_cast<uint32_t>(block_extent_.width),
        static_cast<uint32_t>(block_extent_.height),
        static_cast<uint32_t>(raw_quant_.row_stride),
        static_cast<uint32_t>(tile_extent_.width),
        static_cast<uint32_t>(y_to_x_.row_stride),
        static_cast<uint32_t>(anchor_offset),
        static_cast<uint32_t>(count),
        static_cast<uint32_t>(coefficient_offset),
        static_cast<uint32_t>(coefficient_count),
        static_cast<uint32_t>(info->pixel_extent().width),
        static_cast<uint32_t>(info->pixel_extent().height),
        static_cast<uint32_t>(info->covered_blocks.width),
        static_cast<uint32_t>(info->covered_blocks.height),
        static_cast<uint32_t>(strategy),
        0,
        0,
        QuantizationMatrixMultiplier(options_.profile.x_qm_scale),
        QuantizationMatrixMultiplier(options_.profile.b_qm_scale),
        coefficient_decision_mode_ ==
            AcCoefficientDecisionMode::kAdjustedSharedQuant
          ? 1u
          : 0u,
        static_cast<uint32_t>(inverse_sigma_.row_stride),
        static_cast<uint32_t>(epf_sharpness_.row_stride),
        options_.profile.epf_sigma.quant_multiplier,
        options_.profile.epf_sigma.sharpness_lut,
      };
      block_reduction_params[batch_index] = {
        static_cast<uint32_t>(source_extent_.width),
        static_cast<uint32_t>(source_extent_.height),
        static_cast<uint32_t>(distance_map_.row_stride),
        static_cast<uint32_t>(block_distance_.row_stride),
        static_cast<uint32_t>(anchor_offset),
        static_cast<uint32_t>(count),
        static_cast<uint32_t>(info->pixel_extent().width),
        static_cast<uint32_t>(info->pixel_extent().height),
        static_cast<uint32_t>(info->covered_blocks.width),
        static_cast<uint32_t>(info->covered_blocks.height),
      };
      maximum_error_reduction_params[batch_index] = {
        static_cast<uint32_t>(source_extent_.width),
        static_cast<uint32_t>(source_extent_.height),
        static_cast<uint32_t>(coding_[0].row_stride),
        static_cast<uint32_t>(FinalFilteredImage()[0].row_stride),
        static_cast<uint32_t>(block_distance_.row_stride),
        static_cast<uint32_t>(anchor_offset),
        static_cast<uint32_t>(count),
        static_cast<uint32_t>(info->pixel_extent().width),
        static_cast<uint32_t>(info->pixel_extent().height),
        static_cast<uint32_t>(info->covered_blocks.width),
        static_cast<uint32_t>(info->covered_blocks.height),
        options_.maximum_error[0],
        options_.maximum_error[1],
        options_.maximum_error[2],
      };
      anchor_offset += count;
      coefficient_offset += 3 * count * coefficient_count;
    }
    if (anchor_offset != row_major_anchors.size() ||
        coefficient_offset != coefficient_value_count_) {
      Invalidate();
      return Status::Internal(
        "Reconfigured AQ anchors do not cover the coding image exactly");
    }

    std::vector<int32_t> color_transform_records;
    std::vector<int32_t> color_tile_offsets;
    if (resident_quantization_) {
      status = BuildColorTransformMetadata(
        row_major_anchors, batches, block_extent_, tile_extent_,
        &color_transform_records, &color_tile_offsets);
      if (!status.ok()) {
        Invalidate();
        return status;
      }
    }

    std::vector<vardct_frame_internal::QuantizedAcTransformView>
      transform_views;
    transform_views.reserve(row_major_anchors.size());
    for (const AqAnchor& anchor : row_major_anchors) {
      const AqStrategyBatch& batch = batches[anchor.batch_index];
      const size_t channel_stride =
        batch.anchor_count * batch.coefficient_count;
      vardct_frame_internal::QuantizedAcTransformView transform{
        .block_x = anchor.block_x,
        .block_y = anchor.block_y,
        .strategy = anchor.strategy,
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        const size_t offset = batch.coefficient_offset +
          channel * channel_stride +
          anchor.index_in_batch * batch.coefficient_count;
        transform.coefficients[channel] = std::span<const int32_t>(
          quantized_readback_.data() + offset, batch.coefficient_count);
      }
      transform_views.push_back(transform);
    }

    status = UploadPlane(
      *backend_,
      ConstPlaneI32View{
        strategy_records.data(), strategies_.extent, strategies_.row_stride},
      strategies_);
    if (status.ok()) {
      status = UploadPlane(
        *backend_,
        ConstPlaneI32View{
          anchor_records.data(), {2 * anchor_offset, 1}, 2 * anchor_offset},
        anchors_);
    }
    if (status.ok()) {
      status = UploadPlane(*backend_, epf_sharpness, epf_sharpness_);
    }
    if (status.ok() && resident_quantization_) {
      status = UploadPlane(
        *backend_,
        ConstPlaneI32View{
          color_transform_records.data(),
          {color_transform_records.size(), 1},
          color_transform_records.size()},
        color_transform_records_);
    }
    if (status.ok() && resident_quantization_) {
      status = UploadPlane(
        *backend_,
        ConstPlaneI32View{
          color_tile_offsets.data(), {color_tile_offsets.size(), 1},
          color_tile_offsets.size()},
        color_tile_offsets_);
    }
    if (!status.ok()) {
      Invalidate();
      return status;
    }

    strategies_host_ = strategies;
    epf_sharpness_host_ = std::move(sharpness);
    batches_ = batches;
    reconstruction_params_ = reconstruction_params;
    block_reduction_params_ = block_reduction_params;
    maximum_error_reduction_params_ = maximum_error_reduction_params;
    row_major_anchors_ = std::move(row_major_anchors);
    final_transform_views_ = std::move(transform_views);
    anchor_count_ = anchor_offset;
    final_cfl_params_.transform_count =
      static_cast<uint32_t>(anchor_count_);
    invariant_color_correlation_ready_ = false;
    resident_forward_coefficients_ready_ = false;
    resident_color_correlation_pending_ = false;
    resident_color_correlation_readback_needed_ = false;
    CompleteOperation();
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    Invalidate();
    return Status::OutOfMemory(
      "Unable to allocate AQ reconfiguration storage");
  } catch (const std::length_error&) {
    Invalidate();
    return Status::InvalidArgument(
      "AQ reconfiguration storage is too large");
  }
}

Status MetalPreparedAqEvaluation::Evaluate(AqEvaluationInput input,
                                           AqEvaluationOutput output) {
  Status status = ValidateOutput(output);
  if (!status.ok()) {
    return status;
  }
  status = SubmitEvaluation(input);
  if (!status.ok()) {
    return status;
  }
  return FinishEvaluation(output);
}

Status MetalPreparedAqEvaluation::EvaluateResidentButteraugliPolicy(
    AqResidentButteraugliPolicyInput input,
    AqResidentButteraugliPolicyOutput output) {
  return EvaluateResidentButteraugliPolicyImpl(
    input, output, gpu_profile_internal::GpuProfilingMode::kDisabled,
    nullptr);
}

Status MetalPreparedAqEvaluation::EvaluateResidentButteraugliPolicyProfiled(
    AqResidentButteraugliPolicyInput input,
    AqResidentButteraugliPolicyOutput output,
    gpu_profile_internal::GpuProfilingMode mode,
    gpu_profile_internal::GpuExecutionProfile* profile) {
  if (profile == nullptr) {
    return Status::InvalidArgument(
      "Resident GPU execution profile output is null");
  }
  if (mode == gpu_profile_internal::GpuProfilingMode::kDisabled) {
    return Status::InvalidArgument(
      "Resident GPU execution profiling mode is disabled");
  }
  return EvaluateResidentButteraugliPolicyImpl(input, output, mode, profile);
}

Status MetalPreparedAqEvaluation::InitializeGpuExecutionProfile(
    gpu_profile_internal::GpuProfilingMode mode,
    gpu_profile_internal::GpuExecutionProfile* profile) const {
  if (profile == nullptr ||
      mode == gpu_profile_internal::GpuProfilingMode::kDisabled) {
    return Status::InvalidArgument(
      "Metal GPU execution profile request is invalid");
  }
  gpu_profile_internal::GpuExecutionProfile candidate;
  candidate.mode = mode;
  candidate.capabilities = backend_->ProfilingCapabilities();
  if (!candidate.capabilities.timestamp_counter ||
      !candidate.capabilities.stage_boundary) {
    return Status::Unavailable(
      "Metal stage-boundary timestamp sampling is unavailable");
  }
  if (mode == gpu_profile_internal::GpuProfilingMode::kDispatch &&
      !candidate.capabilities.dispatch_boundary) {
    return Status::Unavailable(
      "Metal dispatch-boundary timestamp sampling is unavailable");
  }
  *profile = std::move(candidate);
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::EvaluateResidentButteraugliPolicyImpl(
    AqResidentButteraugliPolicyInput input,
    AqResidentButteraugliPolicyOutput output,
    gpu_profile_internal::GpuProfilingMode profiling_mode,
    gpu_profile_internal::GpuExecutionProfile* profile) {
  const bool profiling =
    profiling_mode != gpu_profile_internal::GpuProfilingMode::kDisabled;
  gpu_profile_internal::GpuExecutionProfile candidate_profile;
  if (profiling) {
    Status profile_status = InitializeGpuExecutionProfile(
      profiling_mode, &candidate_profile);
    if (!profile_status.ok()) return profile_status;
  }
  if (!resident_quantization_ ||
      options_.metric != AqEvaluationMetric::kButteraugli) {
    return Status::Unavailable(
      "Resident Butteraugli policy was not prepared");
  }
  if (input.iterations > 4 ||
      !ValidHostPlaneLayout(input.adjusted_initial_quant_field) ||
      input.adjusted_initial_quant_field.extent != block_extent_ ||
      !std::isfinite(input.quant_dc) || input.quant_dc <= 0.0f ||
      !std::isfinite(input.butteraugli_target) ||
      input.butteraugli_target <= 0.0f ||
      !std::isfinite(input.lower_bound) || input.lower_bound <= 0.0f ||
      !std::isfinite(input.upper_bound) ||
      input.upper_bound < input.lower_bound ||
      input.upper_bound / input.lower_bound >= 253.0f) {
    return Status::InvalidArgument(
      "Resident Butteraugli policy input is invalid");
  }
  const bool quant_field_requested =
    PlaneDescriptorSpecified(output.quant_field);
  const bool block_map_requested =
    PlaneDescriptorSpecified(output.block_distance_map);
  const bool reconstruction_requested = std::ranges::any_of(
    output.reconstructed_linear_rgb.plane,
    [](PlaneF32View plane) { return PlaneDescriptorSpecified(plane); });
  if ((quant_field_requested &&
       (!ValidHostPlaneLayout(output.quant_field) ||
        output.quant_field.extent != block_extent_)) ||
      (block_map_requested &&
       (!ValidHostPlaneLayout(output.block_distance_map) ||
        output.block_distance_map.extent != block_extent_)) ||
      output.score_history == nullptr) {
    return Status::InvalidArgument(
      "Resident Butteraugli policy output is invalid");
  }
  if (reconstruction_requested &&
      (!output.reconstructed_linear_rgb.valid() ||
       output.reconstructed_linear_rgb.extent() != source_extent_ ||
       !std::ranges::all_of(
         output.reconstructed_linear_rgb.plane,
         [](PlaneF32View plane) { return ValidHostPlaneLayout(plane); }))) {
    return Status::InvalidArgument(
      "Resident Butteraugli final output is invalid");
  }
  if (!input.evaluate_final_field &&
      (input.iterations == 0 || output.frame == nullptr ||
       block_map_requested || reconstruction_requested)) {
    return Status::InvalidArgument(
      "Resident final-frame-only policy output is invalid");
  }

  const AqEvaluationInput evaluation_input{
      .quant_field = input.adjusted_initial_quant_field,
      .quant_dc = input.quant_dc,
  };
  Status status = ValidateInput(evaluation_input);
  if (!status.ok()) return status;
  if (butteraugli_ == nullptr) {
    return Status::FailedPrecondition(
      "Prepared AQ Butteraugli state is missing");
  }
  status = ValidatePreparedMetalButteraugliEncoding(
    *butteraugli_,
    {
      .distorted_linear_rgb = {{{reconstructed_linear_[0],
                                 reconstructed_linear_[1],
                                 reconstructed_linear_[2]}}},
      .distance_map = distance_map_,
      .score = score_,
    });
  if (!status.ok()) return status;

  status = BeginOperation();
  if (!status.ok()) return status;
  bool fail_staging = false;
  {
    std::lock_guard lock(mutex_);
    fail_staging = fail_next_resident_staging_;
    fail_next_resident_staging_ = false;
  }
  if (fail_staging) {
    CompleteOperation();
    return Status::OutOfMemory(
      "Injected resident Butteraugli host-staging failure");
  }

  std::vector<double> candidate_scores;
  VarDctEncoderFrame candidate_frame;
  const size_t score_count =
    input.iterations + static_cast<size_t>(input.evaluate_final_field);
  try {
    candidate_scores.resize(score_count);
    if (quant_field_requested) {
      resident_policy_quant_readback_.resize(block_count_);
    }
  } catch (const std::bad_alloc&) {
    CompleteOperation();
    return Status::OutOfMemory(
      "Unable to allocate resident Butteraugli policy readback");
  } catch (const std::length_error&) {
    CompleteOperation();
    return Status::InvalidArgument(
      "Resident Butteraugli policy dimensions are too large");
  }
  if (reconstruction_requested) {
    status = PrepareLinearReadback();
    if (!status.ok()) {
      CompleteOperation();
      return status;
    }
  }
  bool fail_upload = false;
  bool fail_numeric = false;
  {
    std::lock_guard lock(mutex_);
    fail_upload = fail_next_upload_;
    fail_numeric = fail_next_numeric_;
    fail_next_upload_ = false;
    fail_next_numeric_ = false;
  }
  if (fail_upload) {
    Invalidate();
    return Status::DeviceError("Injected Metal AQ upload failure");
  }
  reset_params_.test_error_mask = fail_numeric ? 512u : 0u;
  reset_params_.preserve_error = 0u;
  status = UploadInput(evaluation_input);
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  resident_quant_selection_params_.quant_dc = input.quant_dc;
  resident_quant_selection_params_.scaled_quant_dc =
    static_cast<uint32_t>(static_cast<int32_t>(
      static_cast<double>(input.quant_dc * 4096.0f) * 1.6));
  for (AqReconstructionParams& params : reconstruction_params_) {
    params.use_resident_quantizer = 1u;
  }
  resident_policy_iterations_ = input.iterations;
  resident_evaluate_final_field_ = input.evaluate_final_field;
  resident_policy_update_params_.butteraugli_target =
    input.butteraugli_target;
  resident_policy_update_params_.lower_bound = input.lower_bound;
  resident_policy_update_params_.upper_bound = input.upper_bound;

  std::unique_ptr<GpuSubmission> submission;
  if (profiling) {
    std::vector<ResidentProfileStageContext> contexts;
    std::vector<MetalProfiledComputeStage> stages;
    const uint32_t epf_iterations =
      options_.profile.loop_filter.epf_options.iterations;
    const bool butteraugli_multiscale =
      source_extent_.width >= 15 && source_extent_.height >= 15;
    const bool profile_forward_coefficients =
      !exact_coefficient_reconstruction_ &&
      !resident_forward_coefficients_ready_;
    const bool profile_final_color_correlation =
      !exact_coefficient_reconstruction_ &&
      resident_color_correlation_pending_;
    const size_t stages_per_iteration =
      12 + 2 * kSupportedAqStrategies.size() +
      static_cast<size_t>(butteraugli_multiscale) * 4 +
      static_cast<size_t>(options_.profile.loop_filter.gaborish) +
      epf_iterations;
    const size_t stage_count =
      score_count * stages_per_iteration +
      static_cast<size_t>(!resident_evaluate_final_field_) *
        (1 + kSupportedAqStrategies.size()) + 1;
    try {
      contexts.reserve(stage_count);
      stages.reserve(stage_count);
    } catch (const std::bad_alloc&) {
      Invalidate();
      return Status::OutOfMemory(
        "Unable to allocate resident GPU stage metadata");
    } catch (const std::length_error&) {
      Invalidate();
      return Status::InvalidArgument(
        "Resident GPU stage metadata is too large");
    }
    const auto append_stage = [&](const char* stage_id,
                                  ResidentProfileStage stage,
                                  uint32_t iteration,
                                  uint32_t epf_pass = 0,
                                  MetalButteraugliProfileStage butter_stage =
                                    MetalButteraugliProfileStage::
                                      kDistortedPsychoMain) {
      contexts.push_back({
        .self = this,
        .stage = stage,
        .iteration = iteration,
        .epf_pass = epf_pass,
        .butteraugli_stage = butter_stage,
      });
      stages.push_back({
        .stage_id = stage_id,
        .group_id = stage_id,
        .iteration = iteration,
        .invocation = iteration,
        .encode = &MetalPreparedAqEvaluation::EncodeResidentProfileStage,
        .context = &contexts.back(),
      });
    };
    const auto append_reconstruction_stage = [&](const char* stage_id,
          ReconstructionProfileStage reconstruction_stage,
          uint32_t iteration, size_t batch_index = 0,
          const char* group_id = "aq.reconstruction") {
      contexts.push_back({
        .self = this,
        .stage = ResidentProfileStage::kReconstruction,
        .iteration = iteration,
        .reconstruction_stage = reconstruction_stage,
        .reconstruction_batch_index = batch_index,
      });
      stages.push_back({
        .stage_id = stage_id,
        .group_id = group_id,
        .iteration = iteration,
        .invocation = iteration,
        .encode = &MetalPreparedAqEvaluation::EncodeResidentProfileStage,
        .context = &contexts.back(),
      });
    };
    try {
      const uint32_t first_epf_pass = epf_iterations == 3 ? 0 : 1;
      for (uint32_t iteration = 0;
           iteration < score_count; ++iteration) {
        append_reconstruction_stage(
          "aq.reconstruction.reset", ReconstructionProfileStage::kReset,
          iteration);
        append_reconstruction_stage(
          "aq.reconstruction.quantizer",
          ReconstructionProfileStage::kQuantizer, iteration);
        if (iteration == 0 && profile_forward_coefficients) {
          for (size_t batch_index = 0; batch_index < batches_.size();
               ++batch_index) {
            if (batches_[batch_index].anchor_count == 0) continue;
            append_reconstruction_stage(
              AqForwardCoefficientProfileStageId(
                batches_[batch_index].strategy),
              ReconstructionProfileStage::kForwardBatch, iteration,
              batch_index);
          }
        }
        if (iteration == 0 && profile_final_color_correlation) {
          append_reconstruction_stage(
            "aq.reconstruction.final_cfl",
            ReconstructionProfileStage::kFinalColorCorrelation, iteration);
        }
        for (size_t batch_index = 0; batch_index < batches_.size();
             ++batch_index) {
          if (batches_[batch_index].anchor_count == 0) continue;
          append_reconstruction_stage(
            AqReconstructionProfileStageId(batches_[batch_index].strategy),
            ReconstructionProfileStage::kBatch, iteration, batch_index);
        }
        if (iteration == 0) {
          append_stage(
            "aq.policy_initialize",
            ResidentProfileStage::kPolicyInitialize, iteration);
        }
        if (options_.profile.loop_filter.gaborish) {
          append_stage(
            "aq.gaborish", ResidentProfileStage::kGaborish, iteration);
        }
        for (uint32_t pass = first_epf_pass;
             pass < first_epf_pass + epf_iterations; ++pass) {
          const char* id = pass == 0 ? "aq.epf.pass_0"
                           : pass == 1 ? "aq.epf.pass_1"
                                       : "aq.epf.pass_2";
          append_stage(id, ResidentProfileStage::kEpf, iteration, pass);
        }
        append_stage(
          "aq.opsin_to_linear", ResidentProfileStage::kOpsinToLinear,
          iteration);
        append_stage(
          "butteraugli.psycho.main", ResidentProfileStage::kButteraugli,
          iteration, 0,
          MetalButteraugliProfileStage::kDistortedPsychoMain);
        append_stage(
          "butteraugli.malta.main", ResidentProfileStage::kButteraugli,
          iteration, 0, MetalButteraugliProfileStage::kMaltaMain);
        append_stage(
          "butteraugli.l2.main", ResidentProfileStage::kButteraugli,
          iteration, 0, MetalButteraugliProfileStage::kL2Main);
        append_stage(
          "butteraugli.mask_final.main",
          ResidentProfileStage::kButteraugli, iteration, 0,
          MetalButteraugliProfileStage::kMaskAndFinalMain);
        if (butteraugli_multiscale) {
          append_stage(
            "butteraugli.psycho.sub", ResidentProfileStage::kButteraugli,
            iteration, 0,
            MetalButteraugliProfileStage::kDistortedPsychoSub);
          append_stage(
            "butteraugli.malta.sub", ResidentProfileStage::kButteraugli,
            iteration, 0, MetalButteraugliProfileStage::kMaltaSub);
          append_stage(
            "butteraugli.l2.sub", ResidentProfileStage::kButteraugli,
            iteration, 0, MetalButteraugliProfileStage::kL2Sub);
          append_stage(
            "butteraugli.mask_final.sub",
            ResidentProfileStage::kButteraugli, iteration, 0,
            MetalButteraugliProfileStage::kMaskAndFinalSub);
        }
        append_stage(
          "butteraugli.score_reduction",
          ResidentProfileStage::kButteraugli, iteration, 0,
          MetalButteraugliProfileStage::kScoreReduction);
        append_stage(
          "aq.block_reduction", ResidentProfileStage::kBlockReduction,
          iteration);
        append_stage(
          "aq.policy_update", ResidentProfileStage::kPolicyUpdate,
          iteration);
      }
      if (!resident_evaluate_final_field_) {
        append_reconstruction_stage(
          "aq.final_frame.quantizer",
          ReconstructionProfileStage::kQuantizer,
          static_cast<uint32_t>(resident_policy_iterations_), 0,
          "aq.final_frame");
        for (size_t batch_index = 0; batch_index < batches_.size();
             ++batch_index) {
          if (batches_[batch_index].anchor_count == 0) continue;
          append_reconstruction_stage(
            AqFinalFrameProfileStageId(batches_[batch_index].strategy),
            ReconstructionProfileStage::kCoefficientBatch,
            static_cast<uint32_t>(resident_policy_iterations_), batch_index,
            "aq.final_frame");
        }
      }
    } catch (const std::bad_alloc&) {
      Invalidate();
      return Status::OutOfMemory(
        "Unable to allocate resident GPU stage metadata");
    } catch (const std::length_error&) {
      Invalidate();
      return Status::InvalidArgument(
        "Resident GPU stage metadata is too large");
    }
    status = backend_->SubmitComputeProfiled(
      "gjxl profiled resident Butteraugli policy", stages,
      profiling_mode, &submission);
    reset_params_.preserve_error = 0u;
    reset_params_.preserve_forward_coefficients = 0u;
  } else {
    status = backend_->SubmitCompute(
      "gjxl prepared resident Butteraugli policy",
      &MetalPreparedAqEvaluation::EncodeResidentButteraugliPolicySubmission,
      this, &submission);
  }
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
      ? Status::Internal(
          "Resident Butteraugli policy returned no submission")
      : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }

  gpu_profile_internal::GpuSubmissionProfile submission_profile;
  status = WaitForOperation(profiling ? &submission_profile : nullptr);
  if (!status.ok()) return status;
  if (profiling) {
    try {
      submission_profile.submission_id = "resident.aq";
      candidate_profile.submissions.push_back(std::move(submission_profile));
    } catch (const std::bad_alloc&) {
      Invalidate();
      return Status::OutOfMemory(
        "Unable to retain resident GPU execution profile");
    } catch (const std::length_error&) {
      Invalidate();
      return Status::InvalidArgument(
        "Resident GPU execution profile is too large");
    }
  }

  MetalAqReadbackStatsForTesting candidate_readback_stats;
  uint32_t device_error = 0;
  status = backend_->CopyDeviceToHost(
    *reconstruction_error_.buffer, &device_error, sizeof(device_error),
    reconstruction_error_.offset_bytes);
  if (!status.ok() || device_error != 0) {
    Invalidate();
    return status.ok()
      ? Status::DeviceError(
          "Metal resident AQ policy produced invalid device numerics (flag " +
          std::to_string(device_error) + ")")
      : status;
  }
  if (resident_color_correlation_readback_needed_) {
    status = ReadbackColorCorrelation();
    if (!status.ok()) {
      Invalidate();
      return status;
    }
    resident_color_correlation_readback_needed_ = false;
  }
  candidate_readback_stats.control_bytes = sizeof(device_error);

  status = backend_->CopyDeviceToHost(
    *resident_policy_scores_.buffer,
    resident_policy_score_readback_.data(),
    score_count * sizeof(float),
    resident_policy_scores_.offset_bytes);
  if (status.ok()) {
    candidate_readback_stats.score_history_bytes =
      score_count * sizeof(float);
  }
  const size_t row_bytes = block_extent_.width * sizeof(float);
  if (quant_field_requested) {
    for (size_t y = 0; status.ok() && y < block_extent_.height; ++y) {
      status = backend_->CopyDeviceToHost(
        *resident_quant_field_.buffer,
        resident_policy_quant_readback_.data() + y * block_extent_.width,
        row_bytes,
        resident_quant_field_.offset_bytes +
          y * resident_quant_field_.row_stride * sizeof(float));
    }
    if (status.ok()) {
      candidate_readback_stats.quant_field_bytes =
        block_count_ * sizeof(float);
    }
  }
  if (block_map_requested) {
    for (size_t y = 0; status.ok() && y < block_extent_.height; ++y) {
      status = backend_->CopyDeviceToHost(
        *block_distance_.buffer,
        readback_.data() + y * block_extent_.width,
        row_bytes,
        block_distance_.offset_bytes +
          y * block_distance_.row_stride * sizeof(float));
    }
    if (status.ok()) {
      candidate_readback_stats.block_distance_map_bytes =
        block_count_ * sizeof(float);
    }
  }

  QuantizerParams resident_quantizer;
  if (output.frame != nullptr) {
    status = backend_->CopyDeviceToHost(
      *resident_quantizer_params_.buffer, &resident_quantizer,
      sizeof(resident_quantizer), resident_quantizer_params_.offset_bytes);
    if (status.ok()) {
      status = Quantizer::Create(resident_quantizer, &last_quantizer_);
    }
    if (status.ok()) {
      status = backend_->CopyDeviceToHost(
        *quantized_coefficients_.buffer, quantized_readback_.data(),
        quantized_readback_.size() * sizeof(int32_t),
        quantized_coefficients_.offset_bytes);
    }
    if (status.ok()) {
      status = backend_->CopyDeviceToHost(
        *quantized_dc_.buffer, quantized_dc_readback_.data(),
        quantized_dc_readback_.size() * sizeof(int32_t),
        quantized_dc_.offset_bytes);
    }
    if (status.ok()) status = ReadbackRawQuant();
    if (status.ok()) {
      candidate_readback_stats.quantizer_bytes = sizeof(resident_quantizer);
      candidate_readback_stats.frame_bytes =
        quantized_readback_.size() * sizeof(int32_t) +
        quantized_dc_readback_.size() * sizeof(int32_t) +
        last_raw_quant_.size() * sizeof(int32_t);
    }
  }
  if (reconstruction_requested) {
    const size_t linear_row_bytes = source_extent_.width * sizeof(float);
    for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
      for (size_t y = 0; status.ok() && y < source_extent_.height; ++y) {
        status = backend_->CopyDeviceToHost(
          *reconstructed_linear_[channel].buffer,
          linear_readback_[channel].data() + y * source_extent_.width,
          linear_row_bytes,
          reconstructed_linear_[channel].offset_bytes +
            y * reconstructed_linear_[channel].row_stride * sizeof(float));
      }
    }
    if (status.ok()) {
      candidate_readback_stats.reconstructed_rgb_bytes =
        3 * source_extent_.width * source_extent_.height * sizeof(float);
    }
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  if (quant_field_requested && !std::ranges::all_of(
        resident_policy_quant_readback_, [](float value) {
          return std::isfinite(value) && value > 0.0f;
        })) {
    Invalidate();
    return Status::Internal(
      "Resident Butteraugli quant-field readback is invalid");
  }
  for (size_t index = 0; index < score_count; ++index) {
    const float score = resident_policy_score_readback_[index];
    if (!std::isfinite(score) || score < 0.0f) {
      Invalidate();
      return Status::Internal(
        "Resident Butteraugli score history is invalid");
    }
    candidate_scores[index] = static_cast<double>(score);
  }
  if (block_map_requested &&
      !std::ranges::all_of(readback_, [](float value) {
        return std::isfinite(value) && value >= 0.0f;
      })) {
    Invalidate();
    return Status::Internal(
      "Resident Butteraugli block-distance readback is invalid");
  }
  constexpr int32_t kQuantizedPoison =
    static_cast<int32_t>(0x81234567u);
  if (output.frame != nullptr &&
      (std::ranges::find(quantized_readback_, kQuantizedPoison) !=
         quantized_readback_.end() ||
       std::ranges::find(quantized_dc_readback_, kQuantizedPoison) !=
         quantized_dc_readback_.end())) {
    Invalidate();
    return Status::Internal(
      "Resident Butteraugli frame readback contains poison");
  }
  if (reconstruction_requested &&
      !std::ranges::all_of(
        linear_readback_, [](const std::vector<float>& plane) {
          return std::ranges::all_of(
            plane, [](float value) { return std::isfinite(value); });
        })) {
    Invalidate();
    return Status::Internal(
      "Resident Butteraugli reconstruction readback is invalid");
  }
  if (output.frame != nullptr) {
    status = AssembleFrameFromReadback(&candidate_frame);
    if (!status.ok()) {
      Invalidate();
      return status;
    }
  }

  for (size_t y = 0; y < block_extent_.height; ++y) {
    if (quant_field_requested) {
      std::copy_n(
        resident_policy_quant_readback_.data() + y * block_extent_.width,
        block_extent_.width, output.quant_field.Row(y));
    }
    if (block_map_requested) {
      std::copy_n(
        readback_.data() + y * block_extent_.width,
        block_extent_.width, output.block_distance_map.Row(y));
    }
  }
  *output.score_history = std::move(candidate_scores);
  if (reconstruction_requested) {
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < source_extent_.height; ++y) {
        std::copy_n(
          linear_readback_[channel].data() + y * source_extent_.width,
          source_extent_.width,
          output.reconstructed_linear_rgb.plane[channel].Row(y));
      }
    }
  }
  if (output.frame != nullptr) *output.frame = std::move(candidate_frame);
  last_readback_stats_ = candidate_readback_stats;
  resident_forward_coefficients_ready_ = true;
  if (profiling) *profile = std::move(candidate_profile);
  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::SetInvariantColorCorrelation(
    ConstPlaneI8View y_to_x, ConstPlaneI8View y_to_b) {
  if (resident_initial_cfl_) {
    return Status::FailedPrecondition(
        "Resident initial CfL owns the prepared color-correlation state");
  }
  if (!ValidHostPlaneLayout(y_to_x) || y_to_x.extent != tile_extent_ ||
      !ValidHostPlaneLayout(y_to_b) || y_to_b.extent != tile_extent_) {
    return Status::InvalidArgument(
        "Prepared invariant color-correlation geometry is invalid");
  }

  std::vector<int8_t> host_y_to_x;
  std::vector<int8_t> host_y_to_b;
  try {
    size_t tile_count = 0;
    if (!tile_extent_.try_area(&tile_count)) {
      return Status::InvalidArgument(
          "Prepared invariant color-correlation grid is too large");
    }
    host_y_to_x.resize(tile_count);
    host_y_to_b.resize(tile_count);
    for (size_t y = 0; y < tile_extent_.height; ++y) {
      std::copy_n(y_to_x.Row(y), tile_extent_.width,
                  host_y_to_x.data() + y * tile_extent_.width);
      std::copy_n(y_to_b.Row(y), tile_extent_.width,
                  host_y_to_b.data() + y * tile_extent_.width);
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
        "Unable to allocate invariant color-correlation staging");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
        "Invariant color-correlation dimensions are too large");
  }

  Status status = BeginOperation();
  if (!status.ok()) return status;
  status = UploadPlane(
      *backend_,
      ConstPlaneI8View{host_y_to_x.data(), tile_extent_, tile_extent_.width},
      y_to_x_);
  if (status.ok()) {
    status = UploadPlane(
        *backend_,
        ConstPlaneI8View{host_y_to_b.data(), tile_extent_, tile_extent_.width},
        y_to_b_);
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  last_y_to_x_ = std::move(host_y_to_x);
  last_y_to_b_ = std::move(host_y_to_b);
  invariant_color_correlation_ready_ = true;
  resident_color_correlation_pending_ = false;
  resident_color_correlation_readback_needed_ = false;
  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::PrepareInvariantColorCorrelationResident(
    ConstPlaneF32View quant_field, float quant_dc) {

  if (!resident_quantization_ || frame_only_) {
    return Status::FailedPrecondition(
      "Resident final color correlation was not prepared");
  }
  if (!ValidHostPlaneLayout(quant_field) ||
      quant_field.extent != block_extent_ || !std::isfinite(quant_dc) ||
      quant_dc <= 0.0f || quant_dc > static_cast<float>(kMaxQuantDc)) {
    return Status::InvalidArgument(
      "Resident final color-correlation input is invalid");
  }
  for (size_t y = 0; y < block_extent_.height; ++y) {
    for (size_t x = 0; x < block_extent_.width; ++x) {
      const float value = quant_field.Row(y)[x];
      if (!std::isfinite(value) || value <= 0.0f) {
        return Status::InvalidArgument(
          "Resident final color-correlation field is invalid");
      }
    }
  }

  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ != State::kReady) {
    return Status::FailedPrecondition(
      "Prepared resident color correlation requires ready state");
  }
  // The next resident evaluation already uploads this field and selects its
  // quantizer. Schedule final CfL in that same command buffer so no additional
  // submission or host synchronization is introduced.
  (void)quant_dc;
  invariant_color_correlation_ready_ = true;
  resident_forward_coefficients_ready_ = false;
  resident_color_correlation_pending_ = true;
  resident_color_correlation_readback_needed_ = false;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::EvaluateProfiled(
  AqEvaluationInput input,
  AqEvaluationOutput output,
  MetalAqEvaluationProfile* profile) {

  if (profile == nullptr) {
    return Status::InvalidArgument(
      "Metal AQ evaluation profile output is null");
  }
  *profile = {};
  {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || state_ != State::kReady ||
        active_profile_ != nullptr) {
      return Status::FailedPrecondition(
        "Metal AQ profiling requires a ready prepared evaluation");
    }
    active_profile_ = profile;
  }
  Status status = ValidateOutput(output);
  if (status.ok()) {
    status = SubmitEvaluation(input, true);
  }
  if (status.ok()) {
    status = FinishEvaluation(output);
  }
  {
    std::lock_guard lock(mutex_);
    active_profile_ = nullptr;
  }
  return status;
}

AqEvaluationMemoryStats
MetalPreparedAqEvaluation::memory_stats() const noexcept {
  return memory_stats_;
}

Status MetalPreparedAqEvaluation::ReadbackRawQuant() {
  const size_t row_bytes = block_extent_.width * sizeof(int32_t);
  Status status = Status::Ok();
  for (size_t y = 0; status.ok() && y < block_extent_.height; ++y) {
    status = backend_->CopyDeviceToHost(
      *raw_quant_.buffer,
      last_raw_quant_.data() + y * block_extent_.width,
      row_bytes,
      raw_quant_.offset_bytes + y * raw_quant_.row_stride * sizeof(int32_t));
  }
  if (!status.ok()) return status;
  if (!std::ranges::all_of(last_raw_quant_, [](int32_t raw_quant) {
        return raw_quant >= 1 && raw_quant <= kMaxRawQuant;
      })) {
    return Status::DeviceError(
      "Metal AQ adjusted raw-quant readback is invalid");
  }
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::ReadbackColorCorrelation() {
  size_t tile_count = 0;
  if (!tile_extent_.try_area(&tile_count)) {
    return Status::Internal("Prepared AQ color-tile extent changed");
  }
  Status status = backend_->CopyDeviceToHost(
      *y_to_x_.buffer, last_y_to_x_.data(), tile_count * sizeof(int8_t),
      y_to_x_.offset_bytes);
  if (status.ok()) {
    status = backend_->CopyDeviceToHost(
        *y_to_b_.buffer, last_y_to_b_.data(), tile_count * sizeof(int8_t),
        y_to_b_.offset_bytes);
  }
  return status;
}

Status MetalPreparedAqEvaluation::AssembleFrameFromReadback(
    VarDctEncoderFrame *frame) const {

  if (frame == nullptr) {
    return Status::InvalidArgument("AQ frame assembly output is null");
  }
  FrameGeometry geometry;
  Status status = FrameGeometry::Create(source_extent_, &geometry);
  if (!status.ok())
    return status;
  const ConstImage3I32View quantized_dc{{
      ConstPlaneI32View{quantized_dc_readback_.data(), block_extent_,
                        block_extent_.width},
      ConstPlaneI32View{quantized_dc_readback_.data() + block_count_,
                        block_extent_, block_extent_.width},
      ConstPlaneI32View{quantized_dc_readback_.data() + 2 * block_count_,
                        block_extent_, block_extent_.width},
  }};
  return vardct_frame_internal::AssembleVarDctEncoderFrame(
      {
          .geometry = geometry,
          .strategies = &strategies_host_,
          .raw_quant_field = {last_raw_quant_.data(), block_extent_,
                              block_extent_.width},
          .quantizer = &last_quantizer_,
          .y_to_x = {last_y_to_x_.data(), tile_extent_, tile_extent_.width},
          .y_to_b = {last_y_to_b_.data(), tile_extent_, tile_extent_.width},
          .epf_sharpness = {epf_sharpness_host_.data(), block_extent_,
                            block_extent_.width},
          .profile = options_.profile,
          .quantized_dc = quantized_dc,
          .transforms = final_transform_views_,
      },
      frame);
}

Status MetalPreparedAqEvaluation::SubmitEvaluation(
  AqEvaluationInput input,
  bool profiling_reserved) {
  Status status = ValidateInput(input);
  if (!status.ok()) {
    return status;
  }

  if (options_.metric == AqEvaluationMetric::kButteraugli &&
      butteraugli_ == nullptr) {
    return Status::FailedPrecondition(
      "Prepared AQ Butteraugli state is missing");
  }
  if (options_.metric == AqEvaluationMetric::kButteraugli) {
    status = ValidatePreparedMetalButteraugliEncoding(
      *butteraugli_,
      {
        .distorted_linear_rgb = {{{reconstructed_linear_[0],
                                   reconstructed_linear_[1],
                                   reconstructed_linear_[2]}}},
        .distance_map = distance_map_,
        .score = score_,
      });
    if (!status.ok()) {
      return status;
    }
  }

  status = BeginOperation(profiling_reserved);
  if (!status.ok())
    return status;
  status = PrepareExactCoefficientStaging(input);
  if (!status.ok()) {
    CompleteOperation();
    return status;
  }
  bool fail_upload = false;
  bool fail_numeric = false;
  {
    std::lock_guard lock(mutex_);
    fail_upload = fail_next_upload_;
    fail_numeric = fail_next_numeric_;
    fail_next_upload_ = false;
    fail_next_numeric_ = false;
  }
  if (fail_upload) {
    Invalidate();
    return Status::DeviceError("Injected Metal AQ upload failure");
  }
  reset_params_.test_error_mask = fail_numeric ? 512u : 0u;
  reset_params_.preserve_error = 0u;
  const ProfileClock::time_point upload_begin = active_profile_ != nullptr
    ? ProfileClock::now() : ProfileClock::time_point{};
  status = UploadInput(input);
  if (active_profile_ != nullptr) {
    active_profile_->input_upload_nanoseconds =
      ElapsedNanoseconds(upload_begin);
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  reset_params_.preserve_forward_coefficients =
    resident_quantization_active_ && resident_forward_coefficients_ready_
      ? 1u : 0u;

  if (resident_quantization_active_) {
    resident_quant_selection_params_.quant_dc = input.quant_dc;
    resident_quant_selection_params_.scaled_quant_dc =
      static_cast<uint32_t>(static_cast<int32_t>(
        static_cast<double>(input.quant_dc * 4096.0f) * 1.6));
  }
  for (AqReconstructionParams& params : reconstruction_params_) {
    params.use_resident_quantizer =
      resident_quantization_active_ ? 1u : 0u;
    if (!resident_quantization_active_) {
      params.global_scale = input.quantizer.global_scale;
      params.quant_dc = input.quantizer.quant_dc;
    }
  }

  std::unique_ptr<GpuSubmission> submission;
  const ProfileClock::time_point submission_begin = active_profile_ != nullptr
    ? ProfileClock::now() : ProfileClock::time_point{};
  status = backend_->SubmitCompute(
      "gjxl prepared AQ production evaluation",
      &MetalPreparedAqEvaluation::EncodeEvaluationSubmission, this,
      &submission);
  if (active_profile_ != nullptr) {
    active_profile_->submission_nanoseconds =
      ElapsedNanoseconds(submission_begin);
  }
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
               ? Status::Internal(
                   "Prepared AQ evaluation returned no submission")
               : status;
  }
  std::lock_guard lock(mutex_);
  submission_ = std::move(submission);
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::FinishEvaluation(
    AqEvaluationOutput output, bool complete_operation) {
  Status status = ValidateOutput(output);
  if (!status.ok()) {
    return status;
  }
  const bool frame_requested = output.final != nullptr;
  const bool reconstruction_requested = output.final != nullptr &&
    ImageDescriptorSpecified(output.final->reconstructed_linear_rgb);
  MetalAqReadbackStatsForTesting candidate_readback_stats;

  const ProfileClock::time_point wait_begin = active_profile_ != nullptr
    ? ProfileClock::now() : ProfileClock::time_point{};
  status = WaitForOperation();
  if (active_profile_ != nullptr) {
    active_profile_->completion_wait_nanoseconds =
      ElapsedNanoseconds(wait_begin);
  }
  if (!status.ok())
    return status;

  if (reconstruction_requested) {
    status = PrepareLinearReadback();
    if (!status.ok()) {
      CompleteOperation();
      return status;
    }
  }

  const ProfileClock::time_point bounded_begin = active_profile_ != nullptr
    ? ProfileClock::now() : ProfileClock::time_point{};
  uint32_t device_error = 0;
  status = backend_->CopyDeviceToHost(
    *reconstruction_error_.buffer, &device_error, sizeof(device_error),
    reconstruction_error_.offset_bytes);
  if (!status.ok() || device_error != 0) {
    Invalidate();
    return status.ok()
      ? Status::DeviceError(
          "Metal AQ evaluation produced invalid device numerics (flag " +
          std::to_string(device_error) + ")")
      : status;
  }
  if (resident_color_correlation_readback_needed_) {
    status = ReadbackColorCorrelation();
    if (!status.ok()) {
      Invalidate();
      return status;
    }
    resident_color_correlation_readback_needed_ = false;
  }
  candidate_readback_stats.control_bytes = sizeof(device_error);

  QuantizerParams resident_quantizer;
  if (resident_quantization_active_) {
    status = backend_->CopyDeviceToHost(
      *resident_quantizer_params_.buffer, &resident_quantizer,
      sizeof(resident_quantizer), resident_quantizer_params_.offset_bytes);
    if (status.ok()) {
      status = Quantizer::Create(resident_quantizer, &last_quantizer_);
    }
    if (!status.ok()) {
      Invalidate();
      return status;
    }
    candidate_readback_stats.quantizer_bytes = sizeof(resident_quantizer);
  }

  const size_t row_bytes = block_extent_.width * sizeof(float);
  for (size_t y = 0; status.ok() && y < block_extent_.height; ++y) {
    status = backend_->CopyDeviceToHost(
      *block_distance_.buffer,
      readback_.data() + y * block_extent_.width,
      row_bytes,
      block_distance_.offset_bytes +
        y * block_distance_.row_stride * sizeof(float));
  }
  if (status.ok()) {
    candidate_readback_stats.block_distance_map_bytes =
      block_count_ * sizeof(float);
  }
  float score = 0.0f;
  MaximumErrorReduction maximum_error;
  if (status.ok() &&
      options_.metric == AqEvaluationMetric::kButteraugli) {
    status = backend_->CopyDeviceToHost(
      *score_.buffer, &score, sizeof(score), score_.offset_bytes);
    if (status.ok()) {
      candidate_readback_stats.score_history_bytes = sizeof(score);
    }
  }
  if (status.ok() &&
      options_.metric == AqEvaluationMetric::kMaximumError) {
    status = backend_->CopyDeviceToHost(
      *transform_maximum_error_.buffer,
      transform_maximum_error_readback_.data(),
      3 * anchor_count_ * sizeof(float),
      transform_maximum_error_.offset_bytes);
    if (status.ok()) {
      candidate_readback_stats.maximum_error_bytes =
        3 * anchor_count_ * sizeof(float);
    }
    for (size_t anchor = 0; status.ok() && anchor < anchor_count_; ++anchor) {
      for (size_t channel = 0; channel < 3; ++channel) {
        const float value =
          transform_maximum_error_readback_[3 * anchor + channel];
        if (!std::isfinite(value) || value < 0.0f) {
          status = Status::Internal(
            "Metal AQ maximum-error readback is invalid");
          break;
        }
        maximum_error.channel_maximum[channel] = std::max(
          maximum_error.channel_maximum[channel], value);
      }
    }
    if (status.ok()) {
      maximum_error.normalized_maximum =
        *std::ranges::max_element(readback_);
      score = maximum_error.normalized_maximum;
    }
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  if (!std::ranges::all_of(readback_, [](float value) {
        return std::isfinite(value) && value >= 0.0f;
      }) ||
      !std::isfinite(score) || score < 0.0f) {
    Invalidate();
    return Status::Internal(
      "Metal AQ bounded readback is not finite and non-negative");
  }
  if (active_profile_ != nullptr) {
    active_profile_->bounded_readback_nanoseconds =
      ElapsedNanoseconds(bounded_begin);
  }

  VarDctEncoderFrame final_frame;
  const ProfileClock::time_point final_begin = active_profile_ != nullptr
    ? ProfileClock::now() : ProfileClock::time_point{};
  if (frame_requested) {
    if (!exact_coefficients_) {
      status = backend_->CopyDeviceToHost(
        *quantized_coefficients_.buffer,
        quantized_readback_.data(),
        quantized_readback_.size() * sizeof(int32_t),
        quantized_coefficients_.offset_bytes);
      if (status.ok()) {
        status = backend_->CopyDeviceToHost(
          *quantized_dc_.buffer,
          quantized_dc_readback_.data(),
          quantized_dc_readback_.size() * sizeof(int32_t),
          quantized_dc_.offset_bytes);
      }
      if (status.ok() && coefficient_decision_mode_ ==
            AcCoefficientDecisionMode::kAdjustedSharedQuant) {
        status = ReadbackRawQuant();
      }
      if (status.ok()) {
        candidate_readback_stats.frame_bytes =
          quantized_readback_.size() * sizeof(int32_t) +
          quantized_dc_readback_.size() * sizeof(int32_t) +
          (coefficient_decision_mode_ ==
               AcCoefficientDecisionMode::kAdjustedSharedQuant
             ? last_raw_quant_.size() * sizeof(int32_t)
             : 0);
      }
    }
  }
  if (reconstruction_requested) {
    const size_t linear_row_bytes = source_extent_.width * sizeof(float);
    for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
      for (size_t y = 0; status.ok() && y < source_extent_.height; ++y) {
        status = backend_->CopyDeviceToHost(
          *reconstructed_linear_[channel].buffer,
          linear_readback_[channel].data() + y * source_extent_.width,
          linear_row_bytes,
          reconstructed_linear_[channel].offset_bytes +
            y * reconstructed_linear_[channel].row_stride * sizeof(float));
      }
    }
    if (!status.ok()) {
      Invalidate();
      return status;
    }
    candidate_readback_stats.reconstructed_rgb_bytes =
      3 * source_extent_.width * source_extent_.height * sizeof(float);

    if (!std::ranges::all_of(
          linear_readback_,
          [](const std::vector<float>& plane) {
            return std::ranges::all_of(
              plane, [](float value) { return std::isfinite(value); });
          })) {
      Invalidate();
      return Status::Internal(
        "Metal AQ final readback contains non-finite pixels");
    }
  }

  if (frame_requested) {
    constexpr int32_t kQuantizedPoison =
      static_cast<int32_t>(0x81234567u);
    if (std::ranges::find(quantized_readback_, kQuantizedPoison) !=
          quantized_readback_.end() ||
        std::ranges::find(quantized_dc_readback_, kQuantizedPoison) !=
          quantized_dc_readback_.end()) {
      Invalidate();
      return Status::Internal(
        "Metal AQ final frame readback contains poison");
    }
    status = AssembleFrameFromReadback(&final_frame);
    if (!status.ok()) {
      Invalidate();
      return status;
    }
  }
  if (active_profile_ != nullptr &&
      (frame_requested || reconstruction_requested)) {
    active_profile_->final_readback_nanoseconds =
      ElapsedNanoseconds(final_begin);
  }

  const ProfileClock::time_point commit_begin = active_profile_ != nullptr
    ? ProfileClock::now() : ProfileClock::time_point{};
  for (size_t y = 0; y < block_extent_.height; ++y) {
    std::copy_n(readback_.data() + y * block_extent_.width, block_extent_.width,
                output.block_distance_map.Row(y));
  }
  *output.score = static_cast<double>(score);
  if (options_.metric == AqEvaluationMetric::kMaximumError) {
    *output.maximum_error = maximum_error;
  }
  if (resident_quantization_active_ && output.quantizer != nullptr) {
    *output.quantizer = resident_quantizer;
  }
  if (reconstruction_requested) {
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < source_extent_.height; ++y) {
        std::copy_n(
          linear_readback_[channel].data() + y * source_extent_.width,
          source_extent_.width,
          output.final->reconstructed_linear_rgb.plane[channel].Row(y));
      }
    }
  }
  if (frame_requested) *output.final->frame = std::move(final_frame);
  last_readback_stats_ = candidate_readback_stats;
  if (active_profile_ != nullptr) {
    active_profile_->output_commit_nanoseconds =
      ElapsedNanoseconds(commit_begin);
  }
  if (resident_quantization_active_) {
    resident_forward_coefficients_ready_ = true;
  }
  if (complete_operation) CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::RunBlockReduction(
    ConstPlaneF32View distance_map, PlaneF32View block_distance_map) {

  if (!ValidHostPlaneLayout(distance_map) ||
      distance_map.extent != source_extent_ ||
      !ValidHostPlaneLayout(block_distance_map) ||
      block_distance_map.extent != block_extent_) {
    return Status::InvalidArgument(
      "Metal AQ block-reduction diagnostic geometry is invalid");
  }
  for (size_t y = 0; y < distance_map.extent.height; ++y) {
    for (size_t x = 0; x < distance_map.extent.width; ++x) {
      const float value = distance_map.Row(y)[x];
      if (!std::isfinite(value) || value < 0.0f) {
        return Status::InvalidArgument(
          "Metal AQ block-reduction input is invalid");
      }
    }
  }

  Status status = BeginOperation();
  if (!status.ok()) {
    return status;
  }
  status = UploadPlane(*backend_, distance_map, distance_map_);
  std::fill(
    readback_.begin(), readback_.end(),
    std::numeric_limits<float>::quiet_NaN());
  if (status.ok()) {
    status = UploadPlane(
      *backend_,
      ConstPlaneF32View{
        readback_.data(), block_extent_, block_extent_.width},
      block_distance_);
  }
  const uint32_t zero = 0;
  if (status.ok()) {
    status = backend_->CopyHostToDevice(
      *reconstruction_error_.buffer, &zero, sizeof(zero),
      reconstruction_error_.offset_bytes);
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
    "gjxl AQ block-reduction diagnostic",
    &MetalPreparedAqEvaluation::EncodeBlockReductionSubmission,
    this, &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
      ? Status::Internal(
          "Metal AQ block-reduction diagnostic returned no submission")
      : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  status = WaitForOperation();
  if (!status.ok()) {
    return status;
  }

  uint32_t device_error = 0;
  status = backend_->CopyDeviceToHost(
    *reconstruction_error_.buffer, &device_error, sizeof(device_error),
    reconstruction_error_.offset_bytes);
  const size_t row_bytes = block_extent_.width * sizeof(float);
  for (size_t y = 0; status.ok() && y < block_extent_.height; ++y) {
    status = backend_->CopyDeviceToHost(
      *block_distance_.buffer,
      readback_.data() + y * block_extent_.width,
      row_bytes,
      block_distance_.offset_bytes +
        y * block_distance_.row_stride * sizeof(float));
  }
  if (!status.ok() || device_error != 0 ||
      !std::ranges::all_of(readback_, [](float value) {
        return std::isfinite(value) && value >= 0.0f;
      })) {
    Invalidate();
    return status.ok()
      ? Status::DeviceError(
          "Metal AQ block reduction produced invalid device output")
      : status;
  }

  for (size_t y = 0; y < block_extent_.height; ++y) {
    std::copy_n(
      readback_.data() + y * block_extent_.width,
      block_extent_.width, block_distance_map.Row(y));
  }
  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::FailNextUpload() {
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ != State::kReady) {
    return Status::FailedPrecondition(
        "Prepared AQ upload injection requires a ready object");
  }
  fail_next_upload_ = true;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::FailNextNumeric() {
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ != State::kReady) {
    return Status::FailedPrecondition(
        "Prepared AQ numeric injection requires a ready object");
  }
  fail_next_numeric_ = true;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::FailNextReadback() {
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ != State::kReady) {
    return Status::FailedPrecondition(
        "Prepared AQ readback injection requires a ready object");
  }
  fail_next_readback_ = true;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::FailNextResidentStaging() {
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ != State::kReady) {
    return Status::FailedPrecondition(
      "Prepared AQ staging injection requires a ready object");
  }
  fail_next_resident_staging_ = true;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::SetWaitObserver(bool *observed) {
  if (observed == nullptr) {
    return Status::InvalidArgument("Prepared AQ wait observer is null");
  }
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    return Status::FailedPrecondition(
        "Prepared AQ wait observer cannot be changed during use");
  }
  wait_observer_ = observed;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::GetReadbackStats(
    MetalAqReadbackStatsForTesting* stats) const {
  if (stats == nullptr) {
    return Status::InvalidArgument("Prepared AQ readback stats output is null");
  }
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ != State::kReady) {
    return Status::FailedPrecondition(
      "Prepared AQ readback stats require a ready object");
  }
  *stats = last_readback_stats_;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::ValidatePreparation(
    const AqEvaluationPreparation &preparation) const {

  if (!preparation.original_linear_rgb.valid() ||
      !preparation.coding_opsin.valid() ||
      !std::ranges::all_of(preparation.original_linear_rgb.plane,
                           [](ConstPlaneF32View plane) {
                             return ValidHostPlaneLayout(plane);
                           }) ||
      !std::ranges::all_of(preparation.coding_opsin.plane,
                           [](ConstPlaneF32View plane) {
                             return ValidHostPlaneLayout(plane);
                           })) {
    return Status::InvalidArgument(
        "Prepared AQ source image views are invalid");
  }
  Status status = ValidateAqGeometry(preparation.original_linear_rgb.extent(),
                                     preparation.coding_opsin.extent());
  if (!status.ok())
    return status;
  const bool resident_coding_specified = std::ranges::any_of(
    preparation.resident_coding_opsin.plane,
    [](ConstDevicePlaneView plane) { return plane.buffer != nullptr; });
  if (resident_coding_specified) {
    status = ValidateDeviceImage3View(
      preparation.resident_coding_opsin, backend_->id());
    if (!status.ok() ||
        preparation.resident_coding_opsin.plane[0].element_type !=
          DeviceElementType::kF32 ||
        preparation.resident_coding_opsin.plane[0].extent !=
          preparation.coding_opsin.extent()) {
      return Status::InvalidArgument(
        "Prepared AQ resident coding image is invalid");
    }
  }
  status = ValidateOptions(preparation.options);
  if (!status.ok())
    return status;
  if (preparation.frame_only_inverse_gaborish &&
      (!preparation.frame_only ||
       !preparation.options.profile.loop_filter.gaborish)) {
    return Status::InvalidArgument(
        "Frame-only inverse Gaborish preparation is inconsistent");
  }
  if (preparation.resident_initial_cfl &&
      !preparation.frame_only && !preparation.resident_quantization) {
    return Status::InvalidArgument(
        "Resident initial CfL requires a resident consumer");
  }
  if (preparation.frame_only_resident_initial_quant &&
      !preparation.frame_only &&
      !preparation.resident_ac_strategy_inputs) {
    return Status::InvalidArgument(
        "Resident initial quantization requires a resident consumer");
  }
  if (preparation.resident_ac_strategy_inputs &&
      !preparation.frame_only_resident_initial_quant) {
    return Status::InvalidArgument(
        "Resident AC-strategy inputs require initial quantization");
  }
  if (preparation.frame_only_resident_quantizer &&
      (!preparation.frame_only ||
       !preparation.frame_only_resident_initial_quant)) {
    return Status::InvalidArgument(
        "Resident quantizer requires resident frame-only initial quantization");
  }
  if (preparation.resident_quantization && preparation.frame_only) {
    return Status::InvalidArgument(
      "Resident AQ quantization requires complete evaluation preparation");
  }
  if (preparation.resident_quantization &&
      preparation.coefficient_decision_mode !=
        AcCoefficientDecisionMode::kAdjustedSharedQuant) {
    return Status::InvalidArgument(
      "Resident AQ quantization requires adjusted coefficient decisions");
  }
  switch (preparation.coefficient_decision_mode) {
    case AcCoefficientDecisionMode::kAdjustedSharedQuant:
    case AcCoefficientDecisionMode::kFixedRawQuant:
      break;
    default:
      return Status::InvalidArgument(
        "Prepared AQ coefficient decision mode is invalid");
  }

  const Extent2D coding = preparation.coding_opsin.extent();
  const Extent2D blocks{
      coding.width / kJxlBlockDimension,
      coding.height / kJxlBlockDimension,
  };
  if (preparation.strategies == nullptr ||
      !preparation.strategies->complete() ||
      preparation.strategies->extent() != blocks ||
      !ValidHostPlaneLayout(preparation.epf_sharpness) ||
      preparation.epf_sharpness.extent != blocks) {
    return Status::InvalidArgument(
        "Prepared AQ strategy or EPF grid is invalid or differently sized");
  }
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      AcStrategyCell cell;
      status = preparation.strategies->Get(x, y, &cell);
      if (!status.ok() || !SupportedAqStrategy(cell.strategy)) {
        return Status::InvalidArgument(
            "Prepared AQ strategy grid contains an unsupported strategy");
      }
      if (preparation.epf_sharpness.Row(y)[x] >= 8) {
        return Status::InvalidArgument(
            "Prepared AQ EPF sharpness is out of range");
      }
    }
  }
  status = ValidateFiniteImage(preparation.original_linear_rgb,
                               "Prepared AQ original");
  if (!status.ok())
    return status;
  if (resident_coding_specified) return Status::Ok();
  return ValidateFiniteImage(preparation.coding_opsin,
                             "Prepared AQ coding opsin");
}

Status MetalPreparedAqEvaluation::ValidateInput(AqEvaluationInput input) const {
  const bool resident_field = input.quant_field.valid();
  if (resident_field &&
      (!resident_quantization_ || input.quant_field.extent != block_extent_ ||
       !std::isfinite(input.quant_dc) || input.quant_dc <= 0.0f ||
       input.quant_dc > static_cast<float>(kMaxQuantDc) ||
       input.raw_quant_field.valid() || input.epf_inverse_sigma.valid() ||
       input.exact_coefficients != nullptr ||
       input.exact_reconstructed_linear_rgb.valid())) {
    return Status::InvalidArgument(
      "Resident AQ quant-field input is invalid or was not prepared");
  }
  const bool linear_specified =
      ImageDescriptorSpecified(input.exact_reconstructed_linear_rgb);
  const bool exact_linear = input.exact_reconstructed_linear_rgb.valid();
  if ((linear_specified && !exact_linear) ||
      (exact_linear && input.exact_coefficients == nullptr) ||
      (exact_linear &&
       input.exact_reconstructed_linear_rgb.extent() != source_extent_) ||
      (exact_linear &&
       options_.metric == AqEvaluationMetric::kMaximumError)) {
    return Status::InvalidArgument(
        "An exact AQ linear image must be correctly sized and accompanied "
        "by exact coefficients");
  }
  if (resident_initial_cfl_ &&
      input.exact_coefficients != nullptr) {
    return Status::InvalidArgument(
        "Resident initial CfL does not accept exact coefficients");
  }
  if (frame_only_resident_quantizer_ &&
      input.exact_coefficients != nullptr) {
    return Status::InvalidArgument(
        "Resident initial quantizer does not accept exact coefficients");
  }
  const bool valid_host_cfl =
      ValidHostPlaneLayout(input.y_to_x) &&
      input.y_to_x.extent == tile_extent_ &&
      ValidHostPlaneLayout(input.y_to_b) &&
      input.y_to_b.extent == tile_extent_;
  const bool host_cfl_specified =
      PlaneDescriptorSpecified(input.y_to_x) ||
      PlaneDescriptorSpecified(input.y_to_b);
  const bool valid_host_quant =
      ValidHostPlaneLayout(input.raw_quant_field) &&
      input.raw_quant_field.extent == block_extent_ &&
      ValidHostPlaneLayout(input.epf_inverse_sigma) &&
      input.epf_inverse_sigma.extent == block_extent_;
  if ((!resident_field && !frame_only_resident_quantizer_ &&
       !valid_host_quant) ||
      (!resident_initial_cfl_ &&
       ((invariant_color_correlation_ready_ && host_cfl_specified) ||
        (!invariant_color_correlation_ready_ && !valid_host_cfl)))) {
    return Status::InvalidArgument(
        "Prepared AQ evaluation input geometry is invalid");
  }
  Quantizer quantizer;
  Status status = Status::Ok();
  if (!resident_field) {
    status = Quantizer::Create(input.quantizer, &quantizer);
    if (!status.ok()) return status;
  }
  if (frame_only_resident_quantizer_) {
    if (!resident_quantizer_ready_) {
      return Status::FailedPrecondition(
          "Resident initial quantizer has not been computed");
    }
    if (input.quantizer.global_scale !=
            last_quantizer_.params().global_scale ||
        input.quantizer.quant_dc != last_quantizer_.params().quant_dc) {
      return Status::InvalidArgument(
          "Resident initial quantizer parameters do not match device state");
    }
  } else if (!resident_field) {
    for (size_t y = 0; y < block_extent_.height; ++y) {
      for (size_t x = 0; x < block_extent_.width; ++x) {
        const int32_t raw_quant = input.raw_quant_field.Row(y)[x];
        const float inverse_sigma = input.epf_inverse_sigma.Row(y)[x];
        if (raw_quant < 1 || raw_quant > kMaxRawQuant ||
            !std::isfinite(inverse_sigma) || inverse_sigma >= 0.0f) {
          return Status::InvalidArgument(
              "Prepared AQ quant or EPF input value is invalid");
        }
      }
    }
  }
  if (input.exact_coefficients != nullptr) {
    const VarDctEncoderFrame& frame = *input.exact_coefficients;
    if (!frame.valid() || frame.geometry().frame() != source_extent_ ||
        frame.geometry().padded_frame() != coding_extent_ ||
        frame.strategies().extent() != strategies_host_.extent() ||
        frame.raw_quant_field().extent != block_extent_ ||
        frame.epf_sharpness().extent != block_extent_ ||
        frame.color_correlation().tile_extent() != tile_extent_ ||
        frame.quantizer().params().global_scale != input.quantizer.global_scale ||
        frame.quantizer().params().quant_dc != input.quantizer.quant_dc ||
        frame.profile() != options_.profile) {
      return Status::InvalidArgument(
          "Exact AQ coefficient frame does not match prepared state");
    }
    for (size_t y = 0; y < block_extent_.height; ++y) {
      for (size_t x = 0; x < block_extent_.width; ++x) {
        AcStrategyCell frame_cell;
        AcStrategyCell prepared_cell;
        if (!frame.strategies().Get(x, y, &frame_cell).ok() ||
            !strategies_host_.Get(x, y, &prepared_cell).ok() ||
            frame_cell.strategy != prepared_cell.strategy ||
            frame_cell.is_anchor != prepared_cell.is_anchor ||
            frame.raw_quant_field().Row(y)[x] !=
                input.raw_quant_field.Row(y)[x] ||
            frame.epf_sharpness().Row(y)[x] !=
                epf_sharpness_host_[y * block_extent_.width + x]) {
          return Status::InvalidArgument(
              "Exact AQ coefficient decisions do not match evaluation input");
        }
      }
    }
    const ConstPlaneI8View frame_x = frame.color_correlation().y_to_x_map();
    const ConstPlaneI8View frame_b = frame.color_correlation().y_to_b_map();
    const ConstPlaneI8View input_x = invariant_color_correlation_ready_
        ? ConstPlaneI8View{last_y_to_x_.data(), tile_extent_,
                           tile_extent_.width}
        : input.y_to_x;
    const ConstPlaneI8View input_b = invariant_color_correlation_ready_
        ? ConstPlaneI8View{last_y_to_b_.data(), tile_extent_,
                           tile_extent_.width}
        : input.y_to_b;
    for (size_t y = 0; y < tile_extent_.height; ++y) {
      for (size_t x = 0; x < tile_extent_.width; ++x) {
        if (frame_x.Row(y)[x] != input_x.Row(y)[x] ||
            frame_b.Row(y)[x] != input_b.Row(y)[x]) {
          return Status::InvalidArgument(
              "Exact AQ coefficient color factors do not match input");
        }
      }
    }
    if (exact_linear) {
      status = ValidateFiniteImage(input.exact_reconstructed_linear_rgb,
                                   "Exact AQ reconstructed linear RGB");
      if (!status.ok()) return status;
    }
  }
  return Status::Ok();
}

Status
MetalPreparedAqEvaluation::ValidateOutput(AqEvaluationOutput output) const {
  if (!ValidHostPlaneLayout(output.block_distance_map) ||
      output.block_distance_map.extent != block_extent_ ||
      output.score == nullptr ||
      (options_.metric == AqEvaluationMetric::kMaximumError &&
       output.maximum_error == nullptr)) {
    return Status::InvalidArgument("Prepared AQ evaluation output is invalid");
  }
  if (output.final != nullptr) {
    const bool reconstruction_requested =
      ImageDescriptorSpecified(output.final->reconstructed_linear_rgb);
    if (output.final->frame == nullptr ||
        (reconstruction_requested &&
         (!output.final->reconstructed_linear_rgb.valid() ||
          output.final->reconstructed_linear_rgb.extent() != source_extent_ ||
          !std::ranges::all_of(
            output.final->reconstructed_linear_rgb.plane,
            [](PlaneF32View plane) { return ValidHostPlaneLayout(plane); })))) {
      return Status::InvalidArgument(
          "Prepared AQ final evaluation output is invalid");
    }
  }
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::BeginOperation(bool profiling_reserved) {
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ == State::kBusy) {
    return Status::FailedPrecondition(
        "Prepared AQ evaluation is already in use");
  }
  if (state_ == State::kInvalid) {
    return Status::FailedPrecondition("Prepared AQ evaluation was invalidated");
  }
  if ((active_profile_ != nullptr) != profiling_reserved) {
    return Status::FailedPrecondition(
        "Prepared AQ evaluation is reserved for profiling");
  }
  state_ = State::kBusy;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::UploadInput(AqEvaluationInput input) {
  Status status = Status::Ok();
  uint64_t upload_bytes = 0;
  resident_quantization_active_ = input.quant_field.valid();
  if (resident_quantization_active_) {
    status = UploadPlane(*backend_, input.quant_field,
                         resident_quant_field_);
    upload_bytes += block_count_ * sizeof(float);
  } else if (!frame_only_resident_quantizer_) {
    status = UploadPlane(*backend_, input.raw_quant_field, raw_quant_);
    upload_bytes += block_count_ * sizeof(int32_t);
  }
  if (status.ok() && !frame_only_ && !resident_quantization_active_) {
    status = UploadPlane(*backend_, input.epf_inverse_sigma, inverse_sigma_);
    upload_bytes += block_count_ * sizeof(float);
  }
  if (status.ok() && !resident_initial_cfl_ &&
      !invariant_color_correlation_ready_) {
    status = UploadPlane(*backend_, input.y_to_x, y_to_x_);
    upload_bytes += last_y_to_x_.size() * sizeof(int8_t);
  }
  if (status.ok() && !resident_initial_cfl_ &&
      !invariant_color_correlation_ready_) {
    status = UploadPlane(*backend_, input.y_to_b, y_to_b_);
    upload_bytes += last_y_to_b_.size() * sizeof(int8_t);
  }
  if (status.ok() && !resident_quantization_active_) {
    status = Quantizer::Create(input.quantizer, &last_quantizer_);
  }
  if (status.ok() && !frame_only_resident_quantizer_ &&
      !resident_quantization_active_) {
    for (size_t y = 0; y < block_extent_.height; ++y) {
      std::copy_n(
        input.raw_quant_field.Row(y), block_extent_.width,
        last_raw_quant_.data() + y * block_extent_.width);
    }
  }
  if (status.ok() && !resident_initial_cfl_ &&
      !invariant_color_correlation_ready_) {
    for (size_t y = 0; y < tile_extent_.height; ++y) {
      std::copy_n(
        input.y_to_x.Row(y), tile_extent_.width,
        last_y_to_x_.data() + y * tile_extent_.width);
      std::copy_n(
        input.y_to_b.Row(y), tile_extent_.width,
        last_y_to_b_.data() + y * tile_extent_.width);
    }
  }
  exact_coefficients_ = input.exact_coefficients != nullptr;
  exact_linear_reconstruction_ =
      input.exact_reconstructed_linear_rgb.valid();
  exact_coefficient_reconstruction_ = exact_coefficients_ &&
      !exact_linear_reconstruction_;
  if (status.ok() && exact_coefficients_) {
    const VarDctEncoderFrame& frame = *input.exact_coefficients;
    const ConstImage3I32View quantized_dc = frame.quantized_dc();
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < block_extent_.height; ++y) {
        std::copy_n(
            quantized_dc.plane[channel].Row(y), block_extent_.width,
            quantized_dc_readback_.data() + channel * block_count_ +
                y * block_extent_.width);
      }
    }

    std::vector<size_t> group_offsets(frame.ac_group_count(), 0);
    for (const AqAnchor& anchor : row_major_anchors_) {
      const AcStrategyInfo* info = GetAcStrategyInfo(anchor.strategy);
      if (info == nullptr) {
        return Status::Internal(
            "Exact AQ coefficient strategy disappeared during upload");
      }
      const size_t group_x =
          anchor.block_x / kVarDctAcGroupBlockDimension;
      const size_t group_y =
          anchor.block_y / kVarDctAcGroupBlockDimension;
      const size_t group_index =
          group_y * frame.ac_group_extent().width + group_x;
      VarDctAcGroupView group;
      status = frame.GetAcGroup(group_index, &group);
      if (!status.ok()) return status;
      const size_t source_offset = group_offsets[group_index];
      const AqStrategyBatch& batch = batches_[anchor.batch_index];
      const size_t channel_stride =
          batch.anchor_count * batch.coefficient_count;
      if (batch.coefficient_count != info->coefficient_count() ||
          source_offset > group.used_coefficient_count ||
          batch.coefficient_count >
              group.used_coefficient_count - source_offset) {
        return Status::InvalidArgument(
            "Exact AQ coefficient group layout does not match strategies");
      }
      for (size_t channel = 0; channel < 3; ++channel) {
        const size_t destination_offset = batch.coefficient_offset +
            channel * channel_stride +
            anchor.index_in_batch * batch.coefficient_count;
        std::copy_n(group.coefficients[channel].data() + source_offset,
                    batch.coefficient_count,
                    quantized_readback_.data() + destination_offset);
        if (exact_coefficient_reconstruction_) {
          constexpr std::array<XybChannel, 3> kChannels = {
              XybChannel::kX, XybChannel::kY, XybChannel::kB};
          const float matrix_multiplier = channel == 0
              ? QuantizationMatrixMultiplier(options_.profile.x_qm_scale)
              : (channel == 2
                     ? QuantizationMatrixMultiplier(
                           options_.profile.b_qm_scale)
                     : 1.0f);
          status = DequantizeAcBlock(
              anchor.strategy, frame.quantizer(),
              input.raw_quant_field.Row(anchor.block_y)[anchor.block_x],
              {.channel = kChannels[channel],
               .matrix_multiplier = matrix_multiplier},
              std::span<const int32_t>(
                  group.coefficients[channel].data() + source_offset,
                  batch.coefficient_count),
              std::span<float>(
                  exact_reconstruction_coefficients_.data() +
                      destination_offset,
                  batch.coefficient_count));
          if (!status.ok()) return status;
        }
      }
      if (exact_coefficient_reconstruction_) {
        // Preserve the CPU reference's double-accumulated coefficient-coding
        // decisions and its DC/LLF conversion. The Metal handoff starts at
        // inverse transforms, where the remaining float error is stable and
        // does not perturb the accepted raw-quant decisions.
        const std::array<float, 3> factors =
            frame.color_correlation().AcFactors(
                anchor.block_x /
                    (kColorTileDimension / kJxlBlockDimension),
                anchor.block_y /
                    (kColorTileDimension / kJxlBlockDimension));
        const size_t x_offset = batch.coefficient_offset +
            anchor.index_in_batch * batch.coefficient_count;
        const size_t y_offset = x_offset + channel_stride;
        const size_t b_offset = y_offset + channel_stride;
        for (size_t coefficient = 0;
             coefficient < batch.coefficient_count; ++coefficient) {
          const float reconstructed_y =
              exact_reconstruction_coefficients_[y_offset + coefficient];
          exact_reconstruction_coefficients_[x_offset + coefficient] +=
              factors[0] * reconstructed_y;
          exact_reconstruction_coefficients_[b_offset + coefficient] +=
              factors[2] * reconstructed_y;
        }

        const ConstImage3FView frame_dc = frame.dc();
        for (size_t channel = 0; channel < 3; ++channel) {
          // The largest supported strategy covers a 4x4 base-block region.
          std::array<float, 16> dc{};
          const size_t dc_count = info->covered_blocks.width *
              info->covered_blocks.height;
          if (dc_count > dc.size()) {
            return Status::Internal(
                "Exact AQ strategy exceeds the DC preparation workspace");
          }
          for (size_t y = 0; y < info->covered_blocks.height; ++y) {
            std::copy_n(
                frame_dc.plane[channel].Row(anchor.block_y + y) +
                    anchor.block_x,
                info->covered_blocks.width,
                dc.data() + y * info->covered_blocks.width);
          }
          const ConstPlaneF32View dc_view{
              dc.data(), info->covered_blocks, info->covered_blocks.width};
          const size_t destination_offset = batch.coefficient_offset +
              channel * channel_stride +
              anchor.index_in_batch * batch.coefficient_count;
          status = ConvertDcToLowFrequencies(
              anchor.strategy, dc_view,
              std::span<float>(
                  exact_reconstruction_coefficients_.data() +
                      destination_offset,
                  batch.coefficient_count));
          if (!status.ok()) return status;
        }
      }
      group_offsets[group_index] += batch.coefficient_count;
    }
    for (size_t group_index = 0; group_index < group_offsets.size();
         ++group_index) {
      VarDctAcGroupView group;
      status = frame.GetAcGroup(group_index, &group);
      if (!status.ok()) return status;
      if (group_offsets[group_index] != group.used_coefficient_count) {
        return Status::InvalidArgument(
            "Exact AQ coefficient group contains unconsumed values");
      }
    }
    if (exact_coefficient_reconstruction_) {
      status = backend_->CopyHostToDevice(
          *reconstruction_coefficients_.buffer,
          exact_reconstruction_coefficients_.data(),
          exact_reconstruction_coefficients_.size() * sizeof(float),
          reconstruction_coefficients_.offset_bytes);
      upload_bytes +=
          exact_reconstruction_coefficients_.size() * sizeof(float);
    }
  }
  if (status.ok() && exact_linear_reconstruction_) {
    for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
      status = UploadPlane(
          *backend_, input.exact_reconstructed_linear_rgb.plane[channel],
          reconstructed_linear_[channel]);
      upload_bytes += source_extent_.width * source_extent_.height *
          sizeof(float);
    }
  }
  if (active_profile_ != nullptr) {
    active_profile_->input_upload_bytes = upload_bytes;
  }
  return status;
}

Status MetalPreparedAqEvaluation::PrepareExactCoefficientStaging(
    AqEvaluationInput input) {
  if (input.exact_coefficients == nullptr ||
      input.exact_reconstructed_linear_rgb.valid()) {
    return Status::Ok();
  }
  try {
    exact_reconstruction_coefficients_.resize(coefficient_value_count_);
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory(
        "Unable to allocate exact AQ coefficient staging");
  } catch (const std::length_error &) {
    return Status::InvalidArgument(
        "Exact AQ coefficient staging is too large");
  }
}

Status MetalPreparedAqEvaluation::PrepareLinearReadback() {
  size_t source_pixel_count = 0;
  if (!source_extent_.try_area(&source_pixel_count)) {
    return Status::InvalidArgument(
      "Prepared AQ source image dimensions are too large");
  }
  try {
    for (std::vector<float>& plane : linear_readback_) {
      plane.resize(source_pixel_count);
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate AQ reconstruction host readback");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "AQ reconstruction host readback is too large");
  }
}

Status MetalPreparedAqEvaluation::WaitForOperation(
    gpu_profile_internal::GpuSubmissionProfile* gpu_profile) {
  std::unique_ptr<GpuSubmission> submission;
  bool fail_readback = false;
  bool *observer = nullptr;
  {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || state_ != State::kBusy || submission_ == nullptr) {
      return Status::FailedPrecondition(
          "Prepared AQ evaluation has no outstanding submission");
    }
    submission = std::move(submission_);
    fail_readback = fail_next_readback_;
    fail_next_readback_ = false;
    observer = wait_observer_;
  }
  Status status = submission->Wait();
  if (status.ok() && gpu_profile != nullptr) {
    status = GetMetalSubmissionGpuProfile(*submission, gpu_profile);
  }
  if (status.ok() && active_profile_ != nullptr) {
    uint64_t gpu_nanoseconds = 0;
    if (GetMetalSubmissionGpuDuration(
          *submission, &gpu_nanoseconds).ok()) {
      active_profile_->command_buffer_gpu_nanoseconds = gpu_nanoseconds;
    }
  }
  if (observer != nullptr) {
    *observer = true;
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  if (fail_readback) {
    Invalidate();
    return Status::DeviceError("Injected Metal AQ readback failure");
  }
  return Status::Ok();
}

void MetalPreparedAqEvaluation::CompleteOperation() {
  std::lock_guard lock(mutex_);
  state_ = State::kReady;
}

void MetalPreparedAqEvaluation::Invalidate() {
  std::lock_guard lock(mutex_);
  submission_.reset();
  state_ = State::kInvalid;
}

void MetalPreparedAqEvaluation::EncodeBlockReduction(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder) const {

  encoder->setComputePipelineState(backend.aq_pipelines_.block_reduction.get());
  const MetalBuffer* distance =
    MetalBackend::AsMetalBuffer(*distance_map_.buffer);
  const MetalBuffer* anchors = MetalBackend::AsMetalBuffer(*anchors_.buffer);
  MetalBuffer* block = MetalBackend::AsMetalBuffer(*block_distance_.buffer);
  MetalBuffer* error =
    MetalBackend::AsMetalBuffer(*reconstruction_error_.buffer);
  for (size_t batch_index = 0; batch_index < batches_.size(); ++batch_index) {
    const AqStrategyBatch& batch = batches_[batch_index];
    if (batch.anchor_count == 0) {
      continue;
    }
    encoder->setBuffer(distance->handle(), distance_map_.offset_bytes, 0);
    encoder->setBuffer(anchors->handle(), anchors_.offset_bytes, 1);
    encoder->setBuffer(block->handle(), block_distance_.offset_bytes, 2);
    encoder->setBuffer(
      error->handle(), reconstruction_error_.offset_bytes, 3);
    encoder->setBytes(
      &block_reduction_params_[batch_index],
      sizeof(block_reduction_params_[batch_index]), 4);
    DispatchMetalThreadgroups(
      encoder,
      MTL::Size(static_cast<NS::UInteger>(batch.anchor_count), 1, 1),
      MTL::Size(kBlockReductionThreadCount, 1, 1));
  }
}

void MetalPreparedAqEvaluation::EncodeMaximumErrorReduction(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder) const {

  encoder->setComputePipelineState(
    backend.aq_pipelines_.maximum_error_reduction.get());
  const auto reconstructed = FinalFilteredImage();
  const MetalBuffer* anchors = MetalBackend::AsMetalBuffer(*anchors_.buffer);
  MetalBuffer* block = MetalBackend::AsMetalBuffer(*block_distance_.buffer);
  MetalBuffer* maxima =
    MetalBackend::AsMetalBuffer(*transform_maximum_error_.buffer);
  MetalBuffer* error =
    MetalBackend::AsMetalBuffer(*reconstruction_error_.buffer);
  for (size_t batch_index = 0; batch_index < batches_.size(); ++batch_index) {
    const AqStrategyBatch& batch = batches_[batch_index];
    if (batch.anchor_count == 0) {
      continue;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      const MetalBuffer* reference =
        MetalBackend::AsMetalBuffer(*coding_[channel].buffer);
      const MetalBuffer* candidate =
        MetalBackend::AsMetalBuffer(*reconstructed[channel].buffer);
      encoder->setBuffer(
        reference->handle(), coding_[channel].offset_bytes, channel);
      encoder->setBuffer(
        candidate->handle(), reconstructed[channel].offset_bytes,
        channel + 3);
    }
    encoder->setBuffer(anchors->handle(), anchors_.offset_bytes, 6);
    encoder->setBuffer(block->handle(), block_distance_.offset_bytes, 7);
    encoder->setBuffer(
      maxima->handle(), transform_maximum_error_.offset_bytes, 8);
    encoder->setBuffer(
      error->handle(), reconstruction_error_.offset_bytes, 9);
    encoder->setBytes(
      &maximum_error_reduction_params_[batch_index],
      sizeof(maximum_error_reduction_params_[batch_index]), 10);
    DispatchMetalThreadgroups(
      encoder,
      MTL::Size(static_cast<NS::UInteger>(batch.anchor_count), 1, 1),
      MTL::Size(kBlockReductionThreadCount, 1, 1));
  }
}

void MetalPreparedAqEvaluation::EncodeBlockReductionSubmission(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
    const void* context) {

  const auto& self = *static_cast<const MetalPreparedAqEvaluation*>(context);
  self.EncodeBlockReduction(backend, encoder);
}

void MetalPreparedAqEvaluation::EncodeEvaluationSubmission(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
    const void* context) {

  auto& self = *static_cast<MetalPreparedAqEvaluation*>(
    const_cast<void*>(context));
  const bool prepared_color_correlation =
    self.resident_color_correlation_pending_;
  EncodeReconstructionSubmission(backend, encoder, &self);
  if (prepared_color_correlation) {
    self.resident_color_correlation_pending_ = false;
    self.resident_color_correlation_readback_needed_ = true;
    self.resident_forward_coefficients_ready_ = true;
  }
  if (!self.exact_linear_reconstruction_) {
    self.EncodePostprocess(backend, encoder);
  }
  if (self.options_.metric == AqEvaluationMetric::kButteraugli) {
    EncodePreparedMetalButteraugli(
      *self.butteraugli_, encoder,
      {
        .distorted_linear_rgb = {{{self.reconstructed_linear_[0],
                                   self.reconstructed_linear_[1],
                                   self.reconstructed_linear_[2]}}},
        .distance_map = self.distance_map_,
        .score = self.score_,
      });
    self.EncodeBlockReduction(backend, encoder);
  } else {
    self.EncodeMaximumErrorReduction(backend, encoder);
  }
}

void MetalPreparedAqEvaluation::EncodeResidentButteraugliPolicySubmission(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
    const void* context) {
  auto& self = *static_cast<MetalPreparedAqEvaluation*>(
    const_cast<void*>(context));
  const size_t evaluation_count =
    self.resident_policy_iterations_ +
    static_cast<size_t>(self.resident_evaluate_final_field_);
  for (size_t iteration = 0; iteration < evaluation_count; ++iteration) {
    self.EncodeResidentReconstruction(
      backend, encoder, static_cast<uint32_t>(iteration));

    if (iteration == 0) {
      self.EncodeResidentPolicyInitialize(backend, encoder);
    }

    self.EncodePostprocess(backend, encoder);
    EncodePreparedMetalButteraugli(
      *self.butteraugli_, encoder,
      {
        .distorted_linear_rgb = {{{self.reconstructed_linear_[0],
                                   self.reconstructed_linear_[1],
                                   self.reconstructed_linear_[2]}}},
        .distance_map = self.distance_map_,
        .score = self.score_,
      });
    self.EncodeBlockReduction(backend, encoder);

    self.EncodeResidentPolicyUpdate(
      backend, encoder, static_cast<uint32_t>(iteration));
  }
  if (!self.resident_evaluate_final_field_) {
    self.EncodeResidentFrame(backend, encoder);
  }
  self.reset_params_.preserve_error = 0u;
  self.reset_params_.preserve_forward_coefficients = 0u;
}

void MetalPreparedAqEvaluation::EncodeResidentReconstruction(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
    uint32_t iteration) {
  reset_params_.preserve_error = iteration == 0 ? 0u : 1u;
  reset_params_.preserve_forward_coefficients =
    iteration == 0 && !resident_forward_coefficients_ready_ ? 0u : 1u;
  const bool prepared_color_correlation =
    resident_color_correlation_pending_;
  EncodeReconstructionSubmission(backend, encoder, this);
  if (prepared_color_correlation) {
    resident_color_correlation_pending_ = false;
    resident_color_correlation_readback_needed_ = true;
    resident_forward_coefficients_ready_ = true;
  }
}

void MetalPreparedAqEvaluation::EncodeResidentFrame(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder) {
  reset_params_.preserve_error = 1u;
  reset_params_.preserve_forward_coefficients = 1u;
  EncodeResidentQuantizer(backend, encoder);
  for (size_t batch_index = 0; batch_index < batches_.size();
       ++batch_index) {
    EncodeReconstructionCoefficientBatch(backend, encoder, batch_index);
  }
}

void MetalPreparedAqEvaluation::EncodeResidentPolicyInitialize(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder) const {
  encoder->setComputePipelineState(
    backend.aq_pipelines_.resident_policy_initialize.get());
  BindPlane(encoder, resident_quant_field_, 0);
  BindPlane(encoder, resident_policy_initial_field_, 1);
  BindPlane(encoder, resident_policy_scores_, 2);
  BindPlane(encoder, reconstruction_error_, 3);
  encoder->setBytes(
    &resident_policy_initialize_params_,
    sizeof(resident_policy_initialize_params_), 4);
  DispatchThreads1d(encoder, std::max(block_count_, size_t{5}));
}

void MetalPreparedAqEvaluation::EncodeResidentPolicyUpdate(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
    uint32_t iteration) {
  resident_policy_update_params_.score_index = iteration;
  resident_policy_update_params_.iteration = iteration;
  resident_policy_update_params_.apply_update =
    iteration < resident_policy_iterations_ ? 1u : 0u;
  encoder->setComputePipelineState(
    backend.aq_pipelines_.resident_policy_update.get());
  BindPlane(encoder, resident_quant_field_, 0);
  BindPlane(encoder, resident_policy_initial_field_, 1);
  BindPlane(encoder, block_distance_, 2);
  BindPlane(encoder, score_, 3);
  BindPlane(encoder, resident_policy_scores_, 4);
  BindPlane(encoder, resident_quantizer_params_, 5);
  BindPlane(encoder, reconstruction_error_, 6);
  encoder->setBytes(
    &resident_policy_update_params_,
    sizeof(resident_policy_update_params_), 7);
  DispatchThreads1d(encoder, block_count_);
}

void MetalPreparedAqEvaluation::EncodeResidentProfileStage(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
    const void* context) {
  auto& stage = *static_cast<const ResidentProfileStageContext*>(context);
  MetalPreparedAqEvaluation& self = *stage.self;
  switch (stage.stage) {
    case ResidentProfileStage::kReconstruction:
      self.reset_params_.preserve_error = stage.iteration == 0 ? 0u : 1u;
      self.reset_params_.preserve_forward_coefficients =
        stage.iteration == 0 &&
            !self.resident_forward_coefficients_ready_
          ? 0u
          : 1u;
      self.EncodeReconstructionProfileStage(
        backend, encoder, stage.reconstruction_stage,
        stage.reconstruction_batch_index);
      if (stage.reconstruction_stage ==
            ReconstructionProfileStage::kFinalColorCorrelation &&
          self.resident_color_correlation_pending_) {
        self.resident_color_correlation_pending_ = false;
        self.resident_color_correlation_readback_needed_ = true;
        self.resident_forward_coefficients_ready_ = true;
      }
      break;
    case ResidentProfileStage::kPolicyInitialize:
      self.EncodeResidentPolicyInitialize(backend, encoder);
      break;
    case ResidentProfileStage::kGaborish:
      self.EncodeGaborish(backend, encoder);
      break;
    case ResidentProfileStage::kEpf:
      self.EncodeEpfPass(backend, encoder, stage.epf_pass);
      break;
    case ResidentProfileStage::kOpsinToLinear:
      self.EncodeOpsinToLinear(backend, encoder);
      break;
    case ResidentProfileStage::kButteraugli:
      EncodePreparedMetalButteraugliProfileStage(
        *self.butteraugli_, encoder,
        {
          .distorted_linear_rgb = {{{self.reconstructed_linear_[0],
                                     self.reconstructed_linear_[1],
                                     self.reconstructed_linear_[2]}}},
          .distance_map = self.distance_map_,
          .score = self.score_,
        },
        stage.butteraugli_stage);
      break;
    case ResidentProfileStage::kBlockReduction:
      self.EncodeBlockReduction(backend, encoder);
      break;
    case ResidentProfileStage::kPolicyUpdate:
      self.EncodeResidentPolicyUpdate(backend, encoder, stage.iteration);
      break;
  }
}

Status CreateAqPipelines(
  MTL::Device* device,
  MTL::Library* library,
  AqPipelines* out) {

  if (out == nullptr) {
    return Status::InvalidArgument("AQ pipeline output is null");
  }
  AqPipelines pipelines;
  Status status = CreateAqPipeline(
    device, library, "gjxl_aq_reduce_block_distance_f32",
    &pipelines.block_reduction);
  if (!status.ok()) {
    return {
      status.code(),
      std::string("Failed to create required AQ block-reduction pipeline: ") +
        std::string(status.message()),
    };
  }
  if (pipelines.block_reduction->maxTotalThreadsPerThreadgroup() <
      kBlockReductionThreadCount) {
    return Status::Unavailable(
      "Metal cannot launch the AQ block-reduction threadgroup");
  }
  status = CreateAqPipeline(
    device, library, "gjxl_aq_reduce_maximum_error_f32",
    &pipelines.maximum_error_reduction);
  if (!status.ok()) {
    return {
      status.code(),
      std::string(
        "Failed to create required AQ maximum-error pipeline: ") +
        std::string(status.message()),
    };
  }
  if (pipelines.maximum_error_reduction->maxTotalThreadsPerThreadgroup() <
      kBlockReductionThreadCount) {
    return Status::Unavailable(
      "Metal cannot launch the AQ maximum-error threadgroup");
  }
  const std::array<
      std::pair<std::string_view, NS::SharedPtr<MTL::ComputePipelineState> *>,
      29>
      reconstruction = {{
          {"gjxl_aq_reset_exact_evaluation",
           &pipelines.reset_exact_evaluation},
          {"gjxl_aq_reset_exact_coefficients",
           &pipelines.reset_exact_coefficients},
          {"gjxl_aq_reset_reconstruction", &pipelines.reset_reconstruction},
          {"gjxl_aq_reset_frame_encoding", &pipelines.reset_frame_encoding},
          {"gjxl_aq_initial_cfl", &pipelines.initial_cfl},
          {"gjxl_aq_final_cfl", &pipelines.final_cfl},
          {"gjxl_aq_reset_initial_quant", &pipelines.reset_initial_quant},
          {"gjxl_aq_initial_quant_gradient",
           &pipelines.initial_quant_gradient},
          {"gjxl_aq_initial_quant_fuzzy_erosion",
           &pipelines.initial_quant_fuzzy_erosion},
          {"gjxl_aq_initial_quant_modulation",
           &pipelines.initial_quant_modulation},
          {"gjxl_aq_initial_quant_sort_prepare",
           &pipelines.initial_quant_sort_prepare},
          {"gjxl_aq_initial_quant_sort_step",
           &pipelines.initial_quant_sort_step},
          {"gjxl_aq_initial_quant_capture_median",
           &pipelines.initial_quant_capture_median},
          {"gjxl_aq_initial_quant_deviation_prepare",
           &pipelines.initial_quant_deviation_prepare},
          {"gjxl_aq_initial_quant_finalize_quantizer",
           &pipelines.initial_quant_finalize_quantizer},
          {"gjxl_aq_initial_quant_raw_quant",
           &pipelines.initial_quant_raw_quant},
          {"gjxl_aq_adjust_quant_field", &pipelines.adjust_quant_field},
          {"gjxl_aq_resident_quant_select_initialize",
           &pipelines.resident_quant_select_initialize},
          {"gjxl_aq_resident_quant_histogram",
           &pipelines.resident_quant_histogram},
          {"gjxl_aq_resident_quant_select_bucket",
           &pipelines.resident_quant_select_bucket},
          {"gjxl_aq_resident_quant_finalize_quantizer",
           &pipelines.resident_quant_finalize_quantizer},
          {"gjxl_aq_resident_policy_initialize",
           &pipelines.resident_policy_initialize},
          {"gjxl_aq_resident_policy_update",
           &pipelines.resident_policy_update},
          {"gjxl_aq_gather_transform_pixels",
           &pipelines.gather_transform_pixels},
          {"gjxl_aq_encode_reconstruction_coefficients",
           &pipelines.encode_reconstruction_coefficients},
          {"gjxl_aq_encode_frame_coefficients",
           &pipelines.encode_frame_coefficients},
          {"gjxl_aq_scatter_reconstructed_pixels",
           &pipelines.scatter_reconstructed_pixels},
          {"gjxl_aq_quantization_probe", &pipelines.quantization_probe},
          {"gjxl_aq_adjustment_probe", &pipelines.adjustment_probe},
      }};
  for (const auto &[name, pipeline] : reconstruction) {
    status = CreateAqPipeline(device, library, name, pipeline);
    if (!status.ok()) {
      return {
          status.code(),
          std::string(
              "Failed to create required AQ reconstruction pipeline: ") +
              std::string(status.message()),
      };
    }
    constexpr NS::UInteger kReconstructionThreads = 256;
    if ((*pipeline)->maxTotalThreadsPerThreadgroup() < kReconstructionThreads) {
      return Status::Unavailable(
          "Metal cannot launch an AQ reconstruction threadgroup");
    }
  }
  const std::array<
      std::pair<std::string_view, NS::SharedPtr<MTL::ComputePipelineState> *>,
      3>
      postprocess = {{
          {"gjxl_aq_gaborish_f32", &pipelines.gaborish},
          {"gjxl_aq_epf_f32", &pipelines.epf},
          {"gjxl_aq_opsin_to_linear_rgb_f32", &pipelines.opsin_to_linear},
      }};
  for (const auto &[name, pipeline] : postprocess) {
    status = CreateAqPipeline(device, library, name, pipeline);
    if (!status.ok()) {
      return {
          status.code(),
          std::string(
            "Failed to create required AQ postprocess pipeline: ") +
            std::string(status.message()),
      };
    }
    constexpr NS::UInteger kPostprocessThreads = 8 * 8;
    if ((*pipeline)->maxTotalThreadsPerThreadgroup() <
        kPostprocessThreads) {
      return Status::Unavailable(
        "Metal cannot launch an AQ postprocess threadgroup");
    }
  }
  *out = std::move(pipelines);
  return Status::Ok();
}

Status MetalBackend::PrepareAqEvaluation(
  const AqEvaluationPreparation& preparation,
  std::unique_ptr<PreparedAqEvaluation>* prepared) {

  return PrepareAqEvaluationImpl(
    preparation, gpu_profile_internal::GpuProfilingMode::kDisabled,
    prepared, nullptr);
}

Status MetalBackend::PrepareAqEvaluationProfiled(
  const AqEvaluationPreparation& preparation,
  gpu_profile_internal::GpuProfilingMode mode,
  std::unique_ptr<PreparedAqEvaluation>* prepared,
  gpu_profile_internal::GpuExecutionProfile* profile) {

  if (mode == gpu_profile_internal::GpuProfilingMode::kDisabled) {
    return Status::InvalidArgument(
      "Profiled AQ preparation mode is disabled");
  }
  return PrepareAqEvaluationImpl(
    preparation, mode, prepared, profile);
}

Status MetalBackend::PrepareAqEvaluationImpl(
  const AqEvaluationPreparation& preparation,
  gpu_profile_internal::GpuProfilingMode mode,
  std::unique_ptr<PreparedAqEvaluation>* prepared,
  gpu_profile_internal::GpuExecutionProfile* profile) {

  const bool profiling =
    mode != gpu_profile_internal::GpuProfilingMode::kDisabled;
  if (prepared == nullptr || profiling != (profile != nullptr)) {
    return Status::InvalidArgument(
      "Prepared AQ evaluation output request is invalid");
  }
  prepared->reset();
  try {
    auto result = std::make_unique<MetalPreparedAqEvaluation>(*this);
    gpu_profile_internal::GpuExecutionProfile candidate_profile;
    Status status = result->Prepare(
      preparation, mode, profiling ? &candidate_profile : nullptr);
    if (!status.ok()) {
      return status;
    }
    *prepared = std::move(result);
    if (profiling) *profile = std::move(candidate_profile);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate Metal prepared AQ evaluation");
  }
}

namespace {

MetalPreparedAqEvaluation* AsMetalPrepared(
  PreparedAqEvaluation& prepared) noexcept {

  return dynamic_cast<MetalPreparedAqEvaluation*>(&prepared);
}

}  // namespace

Status EvaluateMetalAqProfiled(
  PreparedAqEvaluation& prepared,
  AqEvaluationInput input,
  AqEvaluationOutput output,
  MetalAqEvaluationProfile* profile) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ profiling requires a Metal prepared evaluation");
  }
  return metal->EvaluateProfiled(input, output, profile);
}

Status SubmitMetalAqEvaluationForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationInput input) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ evaluation seam requires a Metal prepared evaluation");
  }
  return metal->SubmitEvaluation(input);
}

Status FinishMetalAqEvaluationForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationOutput output) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ evaluation seam requires a Metal prepared evaluation");
  }
  return metal->FinishEvaluation(output);
}

Status RunMetalAqBlockReductionForTesting(
  PreparedAqEvaluation& prepared,
  ConstPlaneF32View distance_map,
  PlaneF32View block_distance_map) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ block reduction requires a Metal prepared evaluation");
  }
  return metal->RunBlockReduction(distance_map, block_distance_map);
}

Status FailNextMetalAqUploadForTesting(
  PreparedAqEvaluation& prepared) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ upload injection requires a Metal prepared evaluation");
  }
  return metal->FailNextUpload();
}

Status FailNextMetalAqNumericForTesting(
  PreparedAqEvaluation& prepared) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ numeric injection requires a Metal prepared evaluation");
  }
  return metal->FailNextNumeric();
}

Status FailNextMetalAqReadbackForTesting(
  PreparedAqEvaluation& prepared) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ readback injection requires a Metal prepared evaluation");
  }
  return metal->FailNextReadback();
}

Status FailNextMetalAqResidentStagingForTesting(
  PreparedAqEvaluation& prepared) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ staging injection requires a Metal prepared evaluation");
  }
  return metal->FailNextResidentStaging();
}

Status SetMetalAqWaitObserverForTesting(
  PreparedAqEvaluation& prepared,
  bool* observed) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ wait observation requires a Metal prepared evaluation");
  }
  return metal->SetWaitObserver(observed);
}

Status GetMetalAqReadbackStatsForTesting(
  PreparedAqEvaluation& prepared,
  MetalAqReadbackStatsForTesting* stats) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ readback stats require a Metal prepared evaluation");
  }
  return metal->GetReadbackStats(stats);
}

Status ValidateMetalAqGeometryForTesting(
  Extent2D source_extent,
  Extent2D coding_extent) {

  return ValidateAqGeometry(source_extent, coding_extent);
}

}  // namespace gjxl::metal_internal

#undef setComputePipelineState
