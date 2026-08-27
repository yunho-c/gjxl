// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_aq_evaluation_internal.h"

#include <algorithm>
#include <array>
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

#include "codec/quantization_tables_generated.h"
#include "core/quantizer.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_status.h"
#include "gpu/scratch.h"

namespace gjxl::metal_internal {
namespace {

inline constexpr size_t kBufferAlignment = 256;
inline constexpr size_t kReductionThreadCount = 256;
inline constexpr size_t kQuantTableValueCount = 11904;
inline constexpr std::array<AcStrategyType, 7> kSupportedAqStrategies = {
    AcStrategyType::kDct8,     AcStrategyType::kDct16x16,
    AcStrategyType::kDct32x32, AcStrategyType::kDct16x8,
    AcStrategyType::kDct8x16,  AcStrategyType::kDct32x16,
    AcStrategyType::kDct16x32,
};

static_assert(std::is_standard_layout_v<AqContractProbeParams>);
static_assert(std::is_trivially_copyable_v<AqContractProbeParams>);
static_assert(sizeof(AqContractProbeParams) == 72);
static_assert(std::is_standard_layout_v<AqReconstructionParams>);
static_assert(std::is_trivially_copyable_v<AqReconstructionParams>);
static_assert(sizeof(AqReconstructionParams) == 80);
static_assert(sizeof(AqResetParams) == 12);
static_assert(sizeof(AqQuantizationProbeParams) == 24);

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

[[nodiscard]] bool FinitePositive(float value) noexcept {
  return std::isfinite(value) && value > 0.0f;
}

[[nodiscard]] float OptionProbeValue(
  const AqEvaluationOptions& options) noexcept;

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

[[nodiscard]] Status ValidateOptions(const AqEvaluationOptions& options) {
  if (!FinitePositive(options.coefficient_coding.x_matrix_multiplier) ||
      !FinitePositive(options.coefficient_coding.b_matrix_multiplier) ||
      !FinitePositive(options.opsin_intensity_target) ||
      !FinitePositive(options.butteraugli.hf_asymmetry) ||
      !FinitePositive(options.butteraugli.x_multiplier) ||
      !FinitePositive(options.butteraugli.intensity_target)) {
    return Status::InvalidArgument(
      "Prepared AQ scalar options must be finite and positive");
  }

  if (options.loop_filter.gaborish) {
    for (size_t channel = 0; channel < 3; ++channel) {
      const float weight1 =
        options.loop_filter.gaborish_options.weight1[channel];
      const float weight2 =
        options.loop_filter.gaborish_options.weight2[channel];
      const float divisor = 1.0f + 4.0f * (weight1 + weight2);
      if (!std::isfinite(weight1) || !std::isfinite(weight2) ||
          !std::isfinite(divisor) || std::abs(divisor) < 1.0e-8f) {
        return Status::InvalidArgument(
          "Prepared AQ Gaborish options are invalid");
      }
    }
  }

  const EpfFilterOptions& epf = options.loop_filter.epf_options;
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
  if (!std::isfinite(OptionProbeValue(options))) {
    return Status::InvalidArgument(
      "Prepared AQ options exceed the contract-probe numeric range");
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

[[nodiscard]] float OptionProbeValue(
  const AqEvaluationOptions& options) noexcept {

  float value = options.coefficient_coding.x_matrix_multiplier * 0.0625f;
  value += options.coefficient_coding.b_matrix_multiplier * 0.03125f;
  value += options.opsin_intensity_target * (1.0f / 1024.0f);
  value += options.butteraugli.hf_asymmetry * (1.0f / 64.0f);
  value += options.butteraugli.x_multiplier * (1.0f / 128.0f);
  value += options.butteraugli.intensity_target * (1.0f / 2048.0f);
  value += options.loop_filter.gaborish ? 0.25f : 0.0f;
  value += static_cast<float>(options.loop_filter.epf_options.iterations) *
    (1.0f / 64.0f);
  return value;
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

Status
MetalPreparedAqEvaluation::Prepare(const AqEvaluationPreparation &preparation) {
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
  (void)block_extent_.try_area(&block_count_);
  (void)coding_extent_.try_area(&pixel_count_);
  if (pixel_count_ >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) / 3) {
    return Status::InvalidArgument(
        "Prepared AQ coefficient storage exceeds Metal shader limits");
  }
  coefficient_value_count_ = 3 * pixel_count_;

  std::vector<int32_t> strategy_records;
  std::vector<int32_t> anchor_records;
  std::vector<float> quant_tables;
  std::array<std::vector<std::array<int32_t, 2>>, 7> grouped_anchors;
  try {
    if (block_count_ > std::numeric_limits<size_t>::max() / 2) {
      return Status::InvalidArgument(
          "Prepared AQ strategy-record count overflows");
    }
    strategy_records.resize(block_count_ * 2);
    readback_.resize(block_count_);
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Unable to allocate prepared AQ host staging");
  } catch (const std::length_error &) {
    return Status::InvalidArgument("Prepared AQ host staging is too large");
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
    forward_readback_.resize(coefficient_value_count_);
    quantized_readback_.resize(coefficient_value_count_);
    dc_readback_.resize(3 * block_count_);
    for (std::vector<float> &plane : reconstructed_readback_) {
      plane.resize(pixel_count_);
    }
    quant_probe_quantized_readback_.resize(maximum_coefficient_count_);
    quant_probe_dequantized_readback_.resize(maximum_coefficient_count_);
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

  size_t persistent_bytes = 0;
  for (size_t channel = 0; channel < 3; ++channel) {
    status = AddPlannedPlane(DeviceElementType::kF32, source_extent_,
                             source_extent_.width, &persistent_bytes);
    if (!status.ok())
      return status;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    status = AddPlannedPlane(DeviceElementType::kF32, coding_extent_,
                             coding_extent_.width, &persistent_bytes);
    if (!status.ok())
      return status;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    status = AddPlannedPlane(DeviceElementType::kF32, coding_extent_,
                             coding_extent_.width, &persistent_bytes);
    if (!status.ok())
      return status;
  }
  status = AddPlannedPlane(DeviceElementType::kI32,
                           {block_extent_.width * 2, block_extent_.height},
                           block_extent_.width * 2, &persistent_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kI32, {2 * anchor_count_, 1},
                           2 * anchor_count_, &persistent_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, {kQuantTableValueCount, 1},
                           kQuantTableValueCount, &persistent_bytes);
  if (!status.ok())
    return status;

  const size_t partial_count = std::max<size_t>(
      1, (block_count_ + kReductionThreadCount - 1) / kReductionThreadCount);
  size_t staging_bytes = 0;
  status = AddPlannedPlane(DeviceElementType::kI32, block_extent_,
                           block_extent_.width, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, block_extent_,
                           block_extent_.width, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kI8, tile_extent_,
                           tile_extent_.width, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kI8, tile_extent_,
                           tile_extent_.width, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, block_extent_,
                           block_extent_.width, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, {partial_count, 1},
                           partial_count, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, {partial_count, 1},
                           partial_count, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, {1, 1}, 1, &staging_bytes);
  if (!status.ok())
    return status;
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
  status =
      AddPlannedPlane(DeviceElementType::kF32, {coefficient_value_count_, 1},
                      coefficient_value_count_, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, {3 * block_count_, 1},
                           3 * block_count_, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kI32, {1, 1}, 1, &staging_bytes);
  if (!status.ok())
    return status;
  status =
      AddPlannedPlane(DeviceElementType::kF32, {maximum_coefficient_count_, 1},
                      maximum_coefficient_count_, &staging_bytes);
  if (!status.ok())
    return status;
  status =
      AddPlannedPlane(DeviceElementType::kI32, {maximum_coefficient_count_, 1},
                      maximum_coefficient_count_, &staging_bytes);
  if (!status.ok())
    return status;
  status =
      AddPlannedPlane(DeviceElementType::kF32, {maximum_coefficient_count_, 1},
                      maximum_coefficient_count_, &staging_bytes);
  if (!status.ok())
    return status;

  status = persistent_.Prepare(*backend_, persistent_bytes);
  if (!status.ok())
    return status;
  status = staging_.Prepare(*backend_, staging_bytes);
  if (!status.ok())
    return status;

  for (size_t channel = 0; channel < 3; ++channel) {
    status = persistent_.AllocatePlane(DeviceElementType::kF32, source_extent_,
                                       source_extent_.width, kBufferAlignment,
                                       &original_[channel]);
    if (!status.ok())
      return status;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    status = persistent_.AllocatePlane(DeviceElementType::kF32, coding_extent_,
                                       coding_extent_.width, kBufferAlignment,
                                       &coding_[channel]);
    if (!status.ok())
      return status;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    status = persistent_.AllocatePlane(DeviceElementType::kF32, coding_extent_,
                                       coding_extent_.width, kBufferAlignment,
                                       &reconstructed_[channel]);
    if (!status.ok())
      return status;
  }
  status = persistent_.AllocatePlane(
      DeviceElementType::kI32, {block_extent_.width * 2, block_extent_.height},
      block_extent_.width * 2, kBufferAlignment, &strategies_);
  if (!status.ok())
    return status;
  status =
      persistent_.AllocatePlane(DeviceElementType::kI32, {2 * anchor_count_, 1},
                                2 * anchor_count_, kBufferAlignment, &anchors_);
  if (!status.ok())
    return status;
  status = persistent_.AllocatePlane(
      DeviceElementType::kF32, {kQuantTableValueCount, 1},
      kQuantTableValueCount, kBufferAlignment, &quant_tables_);
  if (!status.ok())
    return status;

  status = staging_.AllocatePlane(DeviceElementType::kI32, block_extent_,
                                  block_extent_.width, kBufferAlignment,
                                  &raw_quant_);
  if (!status.ok())
    return status;
  status = staging_.AllocatePlane(DeviceElementType::kF32, block_extent_,
                                  block_extent_.width, kBufferAlignment,
                                  &inverse_sigma_);
  if (!status.ok())
    return status;
  status =
      staging_.AllocatePlane(DeviceElementType::kI8, tile_extent_,
                             tile_extent_.width, kBufferAlignment, &y_to_x_);
  if (!status.ok())
    return status;
  status =
      staging_.AllocatePlane(DeviceElementType::kI8, tile_extent_,
                             tile_extent_.width, kBufferAlignment, &y_to_b_);
  if (!status.ok())
    return status;
  status = staging_.AllocatePlane(DeviceElementType::kF32, block_extent_,
                                  block_extent_.width, kBufferAlignment,
                                  &probe_output_);
  if (!status.ok())
    return status;
  status =
      staging_.AllocatePlane(DeviceElementType::kF32, {partial_count, 1},
                             partial_count, kBufferAlignment, &reduction_a_);
  if (!status.ok())
    return status;
  status =
      staging_.AllocatePlane(DeviceElementType::kF32, {partial_count, 1},
                             partial_count, kBufferAlignment, &reduction_b_);
  if (!status.ok())
    return status;
  status = staging_.AllocatePlane(DeviceElementType::kF32, {1, 1}, 1,
                                  kBufferAlignment, &score_);
  if (!status.ok())
    return status;
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
  status = staging_.AllocatePlane(DeviceElementType::kF32,
                                  {coefficient_value_count_, 1},
                                  coefficient_value_count_, kBufferAlignment,
                                  &reconstruction_coefficients_);
  if (!status.ok())
    return status;
  status =
      staging_.AllocatePlane(DeviceElementType::kF32, {3 * block_count_, 1},
                             3 * block_count_, kBufferAlignment, &dc_);
  if (!status.ok())
    return status;
  status = staging_.AllocatePlane(DeviceElementType::kI32, {1, 1}, 1,
                                  kBufferAlignment, &reconstruction_error_);
  if (!status.ok())
    return status;
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
      maximum_coefficient_count_, kBufferAlignment, &quant_probe_dequantized_);
  if (!status.ok())
    return status;

  for (size_t channel = 0; channel < 3; ++channel) {
    status =
        UploadPlane(*backend_, preparation.original_linear_rgb.plane[channel],
                    original_[channel]);
    if (!status.ok())
      return status;
    status = UploadPlane(*backend_, preparation.coding_opsin.plane[channel],
                         coding_[channel]);
    if (!status.ok())
      return status;
  }
  status =
      UploadPlane(*backend_,
                  ConstPlaneI32View{strategy_records.data(), strategies_.extent,
                                    strategies_.row_stride},
                  strategies_);
  if (!status.ok())
    return status;
  status = UploadPlane(*backend_,
                       ConstPlaneI32View{anchor_records.data(), anchors_.extent,
                                         anchors_.row_stride},
                       anchors_);
  if (!status.ok())
    return status;
  status =
      UploadPlane(*backend_,
                  ConstPlaneF32View{quant_tables.data(), quant_tables_.extent,
                                    quant_tables_.row_stride},
                  quant_tables_);
  if (!status.ok())
    return status;

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
        options_.coefficient_coding.x_matrix_multiplier,
        options_.coefficient_coding.b_matrix_multiplier,
    };
  }
  reset_params_ = {
      static_cast<uint32_t>(coefficient_value_count_),
      static_cast<uint32_t>(3 * block_count_),
      static_cast<uint32_t>(pixel_count_),
  };

  memory_stats_ = {
      persistent_.capacity_bytes(),
      staging_.capacity_bytes(),
      staging_.capacity_bytes(),
  };
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::Evaluate(AqEvaluationInput,
                                           AqEvaluationOutput) {

  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ == State::kBusy) {
    return Status::FailedPrecondition(
        "Prepared AQ evaluation is already in use");
  }
  if (state_ == State::kInvalid) {
    return Status::FailedPrecondition(
        "Prepared AQ evaluation was invalidated by an operational failure");
  }
  return Status::Unavailable(
      "Metal AQ production evaluation has remaining stages to connect");
}

AqEvaluationMemoryStats
MetalPreparedAqEvaluation::memory_stats() const noexcept {
  return memory_stats_;
}

Status MetalPreparedAqEvaluation::RunProbe(AqEvaluationInput input,
                                           AqEvaluationOutput output) {
  Status status = ValidateOutput(output);
  if (!status.ok())
    return status;
  status = SubmitProbe(input);
  if (!status.ok())
    return status;
  return FinishProbe(output);
}

Status MetalPreparedAqEvaluation::SubmitProbe(AqEvaluationInput input) {
  Status status = ValidateInput(input);
  if (!status.ok()) {
    return status;
  }

  status = BeginOperation();
  if (!status.ok())
    return status;
  status = UploadInput(input);
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  probe_params_ = {
      static_cast<uint32_t>(source_extent_.width),
      static_cast<uint32_t>(source_extent_.height),
      static_cast<uint32_t>(coding_extent_.width),
      static_cast<uint32_t>(coding_extent_.height),
      static_cast<uint32_t>(block_extent_.width),
      static_cast<uint32_t>(block_extent_.height),
      static_cast<uint32_t>(tile_extent_.width),
      static_cast<uint32_t>(tile_extent_.height),
      static_cast<uint32_t>(original_[0].row_stride),
      static_cast<uint32_t>(coding_[0].row_stride),
      static_cast<uint32_t>(strategies_.row_stride),
      static_cast<uint32_t>(raw_quant_.row_stride),
      static_cast<uint32_t>(inverse_sigma_.row_stride),
      static_cast<uint32_t>(y_to_x_.row_stride),
      static_cast<uint32_t>(probe_output_.row_stride),
      input.quantizer.global_scale,
      input.quantizer.quant_dc,
      OptionProbeValue(options_),
  };

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
      "gjxl prepared AQ contract probe",
      &MetalPreparedAqEvaluation::EncodeProbeSubmission, this, &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
               ? Status::Internal("Prepared AQ probe returned no submission")
               : status;
  }
  std::lock_guard lock(mutex_);
  submission_ = std::move(submission);
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::FinishProbe(AqEvaluationOutput output) {
  Status status = ValidateOutput(output);
  if (!status.ok()) {
    return status;
  }

  status = WaitForOperation();
  if (!status.ok())
    return status;

  const size_t output_bytes = block_count_ * sizeof(float);
  status = backend_->CopyDeviceToHost(*probe_output_.buffer, readback_.data(),
                                      output_bytes, probe_output_.offset_bytes);
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  float score = 0.0f;
  status = backend_->CopyDeviceToHost(*score_.buffer, &score, sizeof(score),
                                      score_.offset_bytes);
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  for (size_t y = 0; y < block_extent_.height; ++y) {
    std::copy_n(readback_.data() + y * block_extent_.width, block_extent_.width,
                output.block_distance_map.Row(y));
  }
  *output.score = static_cast<double>(score);
  CompleteOperation();
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
  status = ValidateOptions(preparation.options);
  if (!status.ok())
    return status;

  const Extent2D coding = preparation.coding_opsin.extent();
  const Extent2D blocks{
      coding.width / kJxlBlockDimension,
      coding.height / kJxlBlockDimension,
  };
  if (preparation.strategies == nullptr ||
      !preparation.strategies->complete() ||
      preparation.strategies->extent() != blocks) {
    return Status::InvalidArgument(
        "Prepared AQ strategy grid is incomplete or differently sized");
  }
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      AcStrategyCell cell;
      status = preparation.strategies->Get(x, y, &cell);
      if (!status.ok() || !SupportedAqStrategy(cell.strategy)) {
        return Status::InvalidArgument(
            "Prepared AQ strategy grid contains an unsupported strategy");
      }
    }
  }
  status = ValidateFiniteImage(preparation.original_linear_rgb,
                               "Prepared AQ original");
  if (!status.ok())
    return status;
  return ValidateFiniteImage(preparation.coding_opsin,
                             "Prepared AQ coding opsin");
}

Status MetalPreparedAqEvaluation::ValidateInput(AqEvaluationInput input) const {
  if (!ValidHostPlaneLayout(input.raw_quant_field) ||
      input.raw_quant_field.extent != block_extent_ ||
      !ValidHostPlaneLayout(input.epf_inverse_sigma) ||
      input.epf_inverse_sigma.extent != block_extent_ ||
      !ValidHostPlaneLayout(input.y_to_x) ||
      input.y_to_x.extent != tile_extent_ ||
      !ValidHostPlaneLayout(input.y_to_b) ||
      input.y_to_b.extent != tile_extent_) {
    return Status::InvalidArgument(
        "Prepared AQ evaluation input geometry is invalid");
  }
  Quantizer quantizer;
  Status status = Quantizer::Create(input.quantizer, &quantizer);
  if (!status.ok()) {
    return status;
  }
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
  return Status::Ok();
}

Status
MetalPreparedAqEvaluation::ValidateOutput(AqEvaluationOutput output) const {
  if (!ValidHostPlaneLayout(output.block_distance_map) ||
      output.block_distance_map.extent != block_extent_ ||
      output.score == nullptr) {
    return Status::InvalidArgument("Prepared AQ evaluation output is invalid");
  }
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::BeginOperation() {
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || state_ == State::kBusy) {
    return Status::FailedPrecondition(
        "Prepared AQ evaluation is already in use");
  }
  if (state_ == State::kInvalid) {
    return Status::FailedPrecondition("Prepared AQ evaluation was invalidated");
  }
  state_ = State::kBusy;
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::UploadInput(AqEvaluationInput input) {
  Status status = UploadPlane(*backend_, input.raw_quant_field, raw_quant_);
  if (status.ok()) {
    status = UploadPlane(*backend_, input.epf_inverse_sigma, inverse_sigma_);
  }
  if (status.ok()) {
    status = UploadPlane(*backend_, input.y_to_x, y_to_x_);
  }
  if (status.ok()) {
    status = UploadPlane(*backend_, input.y_to_b, y_to_b_);
  }
  return status;
}

Status MetalPreparedAqEvaluation::WaitForOperation() {
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

void MetalPreparedAqEvaluation::EncodeProbeSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {

  const auto &self = *static_cast<const MetalPreparedAqEvaluation *>(context);
  encoder->setComputePipelineState(backend.aq_pipelines_.contract_probe.get());
  for (size_t channel = 0; channel < 3; ++channel) {
    const MetalBuffer *buffer =
        MetalBackend::AsMetalBuffer(*self.original_[channel].buffer);
    encoder->setBuffer(buffer->handle(), self.original_[channel].offset_bytes,
                       channel);
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    const MetalBuffer *buffer =
        MetalBackend::AsMetalBuffer(*self.coding_[channel].buffer);
    encoder->setBuffer(buffer->handle(), self.coding_[channel].offset_bytes,
                       channel + 3);
  }
  const std::array<DevicePlaneView, 7> bindings = {
      self.strategies_,    self.quant_tables_, self.raw_quant_,
      self.inverse_sigma_, self.y_to_x_,       self.y_to_b_,
      self.probe_output_,
  };
  for (size_t i = 0; i < bindings.size(); ++i) {
    MetalBuffer *buffer = MetalBackend::AsMetalBuffer(*bindings[i].buffer);
    encoder->setBuffer(buffer->handle(), bindings[i].offset_bytes, i + 6);
  }
  encoder->setBytes(&self.probe_params_, sizeof(self.probe_params_), 13);
  MetalBackend::DispatchPlane(encoder, self.block_extent_);

  ConstDevicePlaneView input = self.probe_output_;
  size_t input_count = self.block_count_;
  bool use_a = true;
  while (true) {
    const size_t output_count =
        (input_count + kReductionThreadCount - 1) / kReductionThreadCount;
    DevicePlaneView destination =
        output_count == 1 ? self.score_
                          : (use_a ? self.reduction_a_ : self.reduction_b_);
    backend.EncodeReductionPass(encoder, input, input_count, destination);
    if (output_count == 1) {
      break;
    }
    input = destination;
    input.extent = {output_count, 1};
    input.row_stride = output_count;
    input_count = output_count;
    use_a = !use_a;
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
    device, library, "gjxl_aq_contract_probe", &pipelines.contract_probe);
  if (!status.ok()) {
    return {
      status.code(),
      std::string("Failed to create required AQ contract pipeline: ") +
        std::string(status.message()),
    };
  }
  constexpr NS::UInteger kProbeThreads = 8 * 8;
  if (pipelines.contract_probe->maxTotalThreadsPerThreadgroup() <
      kProbeThreads) {
    return Status::Unavailable(
      "Metal cannot launch the AQ contract-probe threadgroup");
  }
  const std::array<
      std::pair<std::string_view, NS::SharedPtr<MTL::ComputePipelineState> *>,
      5>
      reconstruction = {{
          {"gjxl_aq_reset_reconstruction", &pipelines.reset_reconstruction},
          {"gjxl_aq_gather_transform_pixels",
           &pipelines.gather_transform_pixels},
          {"gjxl_aq_encode_reconstruction_coefficients",
           &pipelines.encode_reconstruction_coefficients},
          {"gjxl_aq_scatter_reconstructed_pixels",
           &pipelines.scatter_reconstructed_pixels},
          {"gjxl_aq_quantization_probe", &pipelines.quantization_probe},
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
  *out = std::move(pipelines);
  return Status::Ok();
}

Status MetalBackend::PrepareAqEvaluation(
  const AqEvaluationPreparation& preparation,
  std::unique_ptr<PreparedAqEvaluation>* prepared) {

  if (prepared == nullptr) {
    return Status::InvalidArgument(
      "Prepared AQ evaluation output pointer is null");
  }
  prepared->reset();
  try {
    auto result = std::make_unique<MetalPreparedAqEvaluation>(*this);
    Status status = result->Prepare(preparation);
    if (!status.ok()) {
      return status;
    }
    *prepared = std::move(result);
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

Status RunMetalAqContractProbeForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationInput input,
  AqEvaluationOutput output) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ contract probe requires a Metal prepared evaluation");
  }
  return metal->RunProbe(input, output);
}

Status SubmitMetalAqContractProbeForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationInput input) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ contract probe requires a Metal prepared evaluation");
  }
  return metal->SubmitProbe(input);
}

Status FinishMetalAqContractProbeForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationOutput output) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "AQ contract probe requires a Metal prepared evaluation");
  }
  return metal->FinishProbe(output);
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

Status ValidateMetalAqGeometryForTesting(
  Extent2D source_extent,
  Extent2D coding_extent) {

  return ValidateAqGeometry(source_extent, coding_extent);
}

}  // namespace gjxl::metal_internal
