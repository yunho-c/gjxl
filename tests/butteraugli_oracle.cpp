// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "butteraugli_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hwy/highway.h>

#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/memory_manager_internal.h"

// These declarations deliberately pin the test adapter to libjxl's internal
// Butteraugli contract at the submodule revision recorded by the repository.
namespace jxl {

Status Blur(const ImageF &in, float sigma, const ButteraugliParams &params,
            BlurTemp *temp, ImageF *out);

namespace HWY_NAMESPACE {

Status OpsinDynamicsImage(const Image3F &rgb, const ButteraugliParams &params,
                          Image3F *blurred, BlurTemp *blur_temp, Image3F *xyb);
Status SeparateFrequencies(size_t xsize, size_t ysize,
                           const ButteraugliParams &params, BlurTemp *blur_temp,
                           const Image3F &xyb, PsychoImage &psycho_image);
Status MaltaDiffMap(const ImageF &lum0, const ImageF &lum1,
                    double weight_0_gt_1, double weight_0_lt_1, double norm1,
                    ImageF *diffs, ImageF *block_diff_ac);
Status MaltaDiffMapLF(const ImageF &lum0, const ImageF &lum1,
                      double weight_0_gt_1, double weight_0_lt_1, double norm1,
                      ImageF *diffs, ImageF *block_diff_ac);
Status MaskPsychoImage(const PsychoImage &psycho_image0,
                       const PsychoImage &psycho_image1, size_t xsize,
                       size_t ysize, const ButteraugliParams &params,
                       BlurTemp *blur_temp, ImageF *mask, ImageF *diff_ac);
Status CombineChannelsToDiffmap(const ImageF &mask,
                                const Image3F &block_diff_dc,
                                const Image3F &block_diff_ac,
                                float x_multiplier, ImageF *result);

} // namespace HWY_NAMESPACE
} // namespace jxl

namespace gjxl::butteraugli_test {
namespace {

constexpr std::array<double, 6> kMaltaWeights = {
    37.0819870399, 8246.75321353, 18.7237414387,
    6923.99476109, 1.10039032555, 173.5,
};
constexpr std::array<double, 6> kMaltaNorms = {
    130262059.556, 1009002.70582, 4498534.45232,
    8051.15833247, 71.7800275169, 5.0,
};
constexpr std::array<float, 9> kL2Weights = {
    400.0f,         1.50815703118f,  0.0f,
    2150.0f,        10.6195433239f,  16.2176043152f,
    29.2353797994f, 0.844626970982f, 0.703646627719f,
};

[[nodiscard]] bool ValidExtent(OracleExtent extent) {
  return extent.width != 0 && extent.height != 0 &&
         extent.width <= std::numeric_limits<size_t>::max() / extent.height;
}

template <typename Plane> [[nodiscard]] bool ValidPlane(const Plane &plane) {
  if (plane.data == nullptr || !ValidExtent(plane.extent) ||
      plane.stride < plane.extent.width) {
    return false;
  }
  if (plane.extent.height > 1 &&
      plane.stride > (std::numeric_limits<size_t>::max() - plane.extent.width) /
                         (plane.extent.height - 1)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool ValidImage(ConstOracleImage3 image) {
  if (!ValidPlane(image.plane[0])) {
    return false;
  }
  for (size_t channel = 1; channel < 3; ++channel) {
    if (!ValidPlane(image.plane[channel]) ||
        image.plane[channel].extent != image.plane[0].extent) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ValidOptions(OracleOptions options) {
  return std::isfinite(options.hf_asymmetry) && options.hf_asymmetry > 0.0f &&
         std::isfinite(options.x_multiplier) && options.x_multiplier > 0.0f &&
         std::isfinite(options.intensity_target) &&
         options.intensity_target > 0.0f;
}

[[nodiscard]] bool PixelsAreFinite(ConstOracleImage3 image) {
  for (const ConstOraclePlane &plane : image.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        if (!std::isfinite(plane.data[y * plane.stride + x])) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool PixelsAreFinite(ConstOraclePlane plane) {
  for (size_t y = 0; y < plane.extent.height; ++y) {
    for (size_t x = 0; x < plane.extent.width; ++x) {
      if (!std::isfinite(plane.data[y * plane.stride + x])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] jxl::ButteraugliParams ToLibjxl(OracleOptions options) {
  return {
      .hf_asymmetry = options.hf_asymmetry,
      .xmul = options.x_multiplier,
      .intensity_target = options.intensity_target,
  };
}

[[nodiscard]] bool InitMemoryManager(const JxlMemoryManager *requested,
                                     JxlMemoryManager *initialized) {
  return static_cast<bool>(jxl::MemoryManagerInit(initialized, requested));
}

[[nodiscard]] bool AllocateImage3(JxlMemoryManager *memory_manager,
                                  OracleExtent extent, jxl::Image3F *image) {
  auto image_or =
      jxl::Image3F::Create(memory_manager, extent.width, extent.height);
  if (!image_or.ok()) {
    return false;
  }
  *image = std::move(image_or).value_();
  return true;
}

[[nodiscard]] bool AllocatePlane(JxlMemoryManager *memory_manager,
                                 OracleExtent extent, jxl::ImageF *image) {
  auto image_or =
      jxl::ImageF::Create(memory_manager, extent.width, extent.height);
  if (!image_or.ok()) {
    return false;
  }
  *image = std::move(image_or).value_();
  return true;
}

[[nodiscard]] bool CopyToLibjxl(ConstOracleImage3 source,
                                JxlMemoryManager *memory_manager,
                                jxl::Image3F *image) {
  const OracleExtent extent = source.plane[0].extent;
  if (!AllocateImage3(memory_manager, extent, image)) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < extent.height; ++y) {
      std::copy_n(source.plane[channel].data + y * source.plane[channel].stride,
                  extent.width, image->PlaneRow(channel, y));
    }
  }
  return true;
}

[[nodiscard]] bool CopyToLibjxl(ConstOraclePlane source,
                                JxlMemoryManager *memory_manager,
                                jxl::ImageF *image) {
  if (!AllocatePlane(memory_manager, source.extent, image)) {
    return false;
  }
  for (size_t y = 0; y < source.extent.height; ++y) {
    std::copy_n(source.data + y * source.stride, source.extent.width,
                image->Row(y));
  }
  return true;
}

[[nodiscard]] bool ValidRequest(ConstOracleImage3 reference,
                                ConstOracleImage3 distorted,
                                OracleOptions options) {
  return ValidImage(reference) && ValidImage(distorted) &&
         reference.plane[0].extent == distorted.plane[0].extent &&
         ValidOptions(options) && PixelsAreFinite(reference) &&
         PixelsAreFinite(distorted);
}

[[nodiscard]] bool ValidOutput(OraclePlane distance_map,
                               OracleExtent expected_extent,
                               const double *score) {
  return score != nullptr && ValidPlane(distance_map) &&
         distance_map.extent == expected_extent;
}

[[nodiscard]] bool CopyResultAtomically(const jxl::ImageF &source,
                                        OraclePlane destination,
                                        double source_score,
                                        double *destination_score) {
  if (!std::isfinite(source_score)) {
    return false;
  }
  for (size_t y = 0; y < destination.extent.height; ++y) {
    const float *row = source.ConstRow(y);
    for (size_t x = 0; x < destination.extent.width; ++x) {
      if (!std::isfinite(row[x]) || row[x] < 0.0f) {
        return false;
      }
    }
  }
  for (size_t y = 0; y < destination.extent.height; ++y) {
    std::copy_n(source.ConstRow(y), destination.extent.width,
                destination.data + y * destination.stride);
  }
  *destination_score = source_score;
  return true;
}

[[nodiscard]] bool CopyPlaneAtomically(const jxl::ImageF &source,
                                       OraclePlane destination) {
  for (size_t y = 0; y < destination.extent.height; ++y) {
    const float *row = source.ConstRow(y);
    for (size_t x = 0; x < destination.extent.width; ++x) {
      if (!std::isfinite(row[x])) {
        return false;
      }
    }
  }
  for (size_t y = 0; y < destination.extent.height; ++y) {
    std::copy_n(source.ConstRow(y), destination.extent.width,
                destination.data + y * destination.stride);
  }
  return true;
}

void CopyStagePlane(const jxl::ImageF &source, IntermediateStage stage,
                    IntermediateStageOutput *output) {
  std::vector<float> &values = output->plane[static_cast<size_t>(stage)];
  values.resize(output->extent.width * output->extent.height);
  for (size_t y = 0; y < output->extent.height; ++y) {
    std::copy_n(source.ConstRow(y), output->extent.width,
                values.data() + y * output->extent.width);
  }
}

void CopyOpsinFrequencyPlane(const jxl::ImageF &source,
                             OpsinFrequencyStage stage,
                             OpsinFrequencyStageOutput *output) {
  std::vector<float> &values = output->plane[static_cast<size_t>(stage)];
  values.resize(output->extent.width * output->extent.height);
  for (size_t y = 0; y < output->extent.height; ++y) {
    std::copy_n(source.ConstRow(y), output->extent.width,
                values.data() + y * output->extent.width);
  }
}

void AddL2Diff(const jxl::ImageF &image0, const jxl::ImageF &image1,
               float weight, jxl::ImageF *diffmap) {
  if (weight == 0.0f) {
    return;
  }
  for (size_t y = 0; y < image0.ysize(); ++y) {
    for (size_t x = 0; x < image0.xsize(); ++x) {
      const float difference = image0.ConstRow(y)[x] - image1.ConstRow(y)[x];
      diffmap->Row(y)[x] =
          difference * difference * weight + diffmap->Row(y)[x];
    }
  }
}

void SetL2Diff(const jxl::ImageF &image0, const jxl::ImageF &image1,
               float weight, jxl::ImageF *diffmap) {
  if (weight == 0.0f) {
    return;
  }
  for (size_t y = 0; y < image0.ysize(); ++y) {
    for (size_t x = 0; x < image0.xsize(); ++x) {
      const float difference = image0.ConstRow(y)[x] - image1.ConstRow(y)[x];
      diffmap->Row(y)[x] = difference * difference * weight;
    }
  }
}

void AddAsymmetricL2Diff(const jxl::ImageF &image0, const jxl::ImageF &image1,
                         float weight_0_gt_1, float weight_0_lt_1,
                         jxl::ImageF *diffmap) {
  const float primary_weight = weight_0_gt_1 * 0.8f;
  const float secondary_weight = weight_0_lt_1 * 0.8f;
  for (size_t y = 0; y < image0.ysize(); ++y) {
    for (size_t x = 0; x < image0.xsize(); ++x) {
      const float value0 = image0.ConstRow(y)[x];
      const float value1 = image1.ConstRow(y)[x];
      const float difference = value0 - value1;
      float total =
          difference * difference * primary_weight + diffmap->Row(y)[x];
      const float magnitude = std::abs(value0);
      const float too_small = 0.4f * magnitude;
      float secondary = 0.0f;
      if (value0 < 0.0f) {
        if (value1 > -too_small) {
          secondary = value1 + too_small;
        } else if (value1 < -magnitude) {
          secondary = -value1 - magnitude;
        }
      } else if (value1 < too_small) {
        secondary = too_small - value1;
      } else if (value1 > magnitude) {
        secondary = value1 - magnitude;
      }
      total = secondary_weight * (secondary * secondary) + total;
      diffmap->Row(y)[x] = total;
    }
  }
}

[[nodiscard]] bool RunMalta(const jxl::ImageF &image0,
                            const jxl::ImageF &image1, bool low_frequency,
                            double weight_0_gt_1, double weight_0_lt_1,
                            double norm, JxlMemoryManager *memory_manager,
                            jxl::ImageF *output) {
  const OracleExtent extent{image0.xsize(), image0.ysize()};
  jxl::ImageF diffs;
  if (!AllocatePlane(memory_manager, extent, &diffs) ||
      !AllocatePlane(memory_manager, extent, output)) {
    return false;
  }
  jxl::ZeroFillImage(output);
  if (low_frequency) {
    return static_cast<bool>(jxl::HWY_NAMESPACE::MaltaDiffMapLF(
        image0, image1, weight_0_gt_1, weight_0_lt_1, norm, &diffs, output));
  }
  return static_cast<bool>(jxl::HWY_NAMESPACE::MaltaDiffMap(
      image0, image1, weight_0_gt_1, weight_0_lt_1, norm, &diffs, output));
}

} // namespace

struct PreparedReference::Impl {
  JxlMemoryManager memory_manager{};
  OracleExtent extent;
  jxl::ButteraugliParams params;
  std::unique_ptr<jxl::ButteraugliComparator> comparator;
};

PreparedReference::PreparedReference() = default;
PreparedReference::~PreparedReference() = default;
PreparedReference::PreparedReference(PreparedReference &&) noexcept = default;
PreparedReference &
PreparedReference::operator=(PreparedReference &&) noexcept = default;

bool PreparedReference::valid() const {
  return impl_ != nullptr && impl_->comparator != nullptr;
}

bool ComputeLiveGaussianBlur(ConstOraclePlane input, float sigma,
                             OraclePlane output,
                             const JxlMemoryManager *requested_memory_manager) {
  const float scaled_radius = 2.25f * std::abs(sigma);
  const int radius = std::isfinite(scaled_radius) && scaled_radius < 17.0f
                         ? std::max(1, static_cast<int>(scaled_radius))
                         : 0;
  const size_t kernel_size = 2 * static_cast<size_t>(radius) + 1;
  if (!ValidPlane(input) || !PixelsAreFinite(input) || !ValidPlane(output) ||
      input.extent != output.extent || !std::isfinite(sigma) || sigma <= 0.0f ||
      (kernel_size != 5 && kernel_size != 7 && kernel_size != 13 &&
       kernel_size != 15 && kernel_size != 33)) {
    return false;
  }

  try {
    JxlMemoryManager memory_manager{};
    if (!InitMemoryManager(requested_memory_manager, &memory_manager)) {
      return false;
    }
    jxl::ImageF input_image;
    jxl::ImageF result;
    if (!CopyToLibjxl(input, &memory_manager, &input_image) ||
        !AllocatePlane(&memory_manager, input.extent, &result)) {
      return false;
    }
    jxl::BlurTemp blur_temp;
    const jxl::ButteraugliParams params;
    if (!jxl::Blur(input_image, sigma, params, &blur_temp, &result)) {
      return false;
    }
    return CopyPlaneAtomically(result, output);
  } catch (const std::length_error &) {
    return false;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

const char *OpsinFrequencyStageName(OpsinFrequencyStage stage) {
  constexpr std::array<const char *, kOpsinFrequencyStageCount> kNames = {
      "opsin_x", "opsin_y", "opsin_b", "lf_x", "lf_y",  "lf_b",  "mf_x",
      "mf_y",    "mf_b",    "hf_x",    "hf_y", "uhf_x", "uhf_y",
  };
  const size_t index = static_cast<size_t>(stage);
  return index < kNames.size() ? kNames[index] : "unknown";
}

bool ComputeLiveOpsinAndFrequencies(
    ConstOracleImage3 input, float intensity_target,
    OpsinFrequencyStageOutput *output,
    const JxlMemoryManager *requested_memory_manager) {
  if (output == nullptr || !ValidImage(input) || !PixelsAreFinite(input) ||
      !std::isfinite(intensity_target) || intensity_target <= 0.0f) {
    return false;
  }

  try {
    OpsinFrequencyStageOutput local;
    local.extent = input.plane[0].extent;
    JxlMemoryManager memory_manager{};
    if (!InitMemoryManager(requested_memory_manager, &memory_manager)) {
      return false;
    }
    jxl::Image3F image;
    jxl::Image3F xyb;
    jxl::Image3F opsin_temp;
    if (!CopyToLibjxl(input, &memory_manager, &image) ||
        !AllocateImage3(&memory_manager, local.extent, &xyb) ||
        !AllocateImage3(&memory_manager, local.extent, &opsin_temp)) {
      return false;
    }
    jxl::ButteraugliParams params;
    params.intensity_target = intensity_target;
    jxl::BlurTemp blur_temp;
    if (!jxl::HWY_NAMESPACE::OpsinDynamicsImage(image, params, &opsin_temp,
                                                &blur_temp, &xyb)) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      CopyOpsinFrequencyPlane(
          xyb.Plane(channel),
          static_cast<OpsinFrequencyStage>(
              static_cast<size_t>(OpsinFrequencyStage::kOpsinX) + channel),
          &local);
    }

    jxl::PsychoImage psycho;
    if (!jxl::HWY_NAMESPACE::SeparateFrequencies(local.extent.width,
                                                 local.extent.height, params,
                                                 &blur_temp, xyb, psycho)) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      CopyOpsinFrequencyPlane(
          psycho.lf.Plane(channel),
          static_cast<OpsinFrequencyStage>(
              static_cast<size_t>(OpsinFrequencyStage::kLowFrequencyX) +
              channel),
          &local);
      CopyOpsinFrequencyPlane(
          psycho.mf.Plane(channel),
          static_cast<OpsinFrequencyStage>(
              static_cast<size_t>(OpsinFrequencyStage::kMediumFrequencyX) +
              channel),
          &local);
    }
    for (size_t channel = 0; channel < 2; ++channel) {
      CopyOpsinFrequencyPlane(
          psycho.hf[channel],
          static_cast<OpsinFrequencyStage>(
              static_cast<size_t>(OpsinFrequencyStage::kHighFrequencyX) +
              channel),
          &local);
      CopyOpsinFrequencyPlane(
          psycho.uhf[channel],
          static_cast<OpsinFrequencyStage>(
              static_cast<size_t>(OpsinFrequencyStage::kUltraHighFrequencyX) +
              channel),
          &local);
    }
    for (const std::vector<float> &plane : local.plane) {
      if (!std::ranges::all_of(
              plane, [](float value) { return std::isfinite(value); })) {
        return false;
      }
    }
    *output = std::move(local);
    return true;
  } catch (const std::length_error &) {
    return false;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool ComputeLiveButteraugli(ConstOracleImage3 reference,
                            ConstOracleImage3 distorted, OracleOptions options,
                            OraclePlane distance_map, double *score,
                            const JxlMemoryManager *requested_memory_manager) {
  if (!ValidRequest(reference, distorted, options) ||
      !ValidOutput(distance_map, reference.plane[0].extent, score)) {
    return false;
  }

  JxlMemoryManager memory_manager{};
  if (!InitMemoryManager(requested_memory_manager, &memory_manager)) {
    return false;
  }
  jxl::Image3F reference_image;
  jxl::Image3F distorted_image;
  if (!CopyToLibjxl(reference, &memory_manager, &reference_image) ||
      !CopyToLibjxl(distorted, &memory_manager, &distorted_image)) {
    return false;
  }
  jxl::ImageF result;
  const jxl::ButteraugliParams params = ToLibjxl(options);
  if (!jxl::ButteraugliDiffmap(reference_image, distorted_image, params,
                               result)) {
    return false;
  }
  const double result_score = jxl::ButteraugliScoreFromDiffmap(result, &params);
  return CopyResultAtomically(result, distance_map, result_score, score);
}

bool PrepareLiveButteraugliReference(
    ConstOracleImage3 reference, OracleOptions options,
    PreparedReference *prepared,
    const JxlMemoryManager *requested_memory_manager) {
  if (prepared == nullptr || !ValidImage(reference) || !ValidOptions(options) ||
      !PixelsAreFinite(reference)) {
    return false;
  }
  try {
    auto local = std::make_unique<PreparedReference::Impl>();
    if (!InitMemoryManager(requested_memory_manager, &local->memory_manager)) {
      return false;
    }
    jxl::Image3F reference_image;
    if (!CopyToLibjxl(reference, &local->memory_manager, &reference_image)) {
      return false;
    }
    local->extent = reference.plane[0].extent;
    local->params = ToLibjxl(options);
    auto comparator_or =
        jxl::ButteraugliComparator::Make(reference_image, local->params);
    if (!comparator_or.ok()) {
      return false;
    }
    local->comparator = std::move(comparator_or).value_();
    prepared->impl_ = std::move(local);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool CompareLiveButteraugliPrepared(const PreparedReference &prepared,
                                    ConstOracleImage3 distorted,
                                    OraclePlane distance_map, double *score) {
  if (!prepared.valid() || !ValidImage(distorted) ||
      distorted.plane[0].extent != prepared.impl_->extent ||
      !PixelsAreFinite(distorted) ||
      !ValidOutput(distance_map, prepared.impl_->extent, score)) {
    return false;
  }
  jxl::Image3F distorted_image;
  if (!CopyToLibjxl(distorted, &prepared.impl_->memory_manager,
                    &distorted_image)) {
    return false;
  }
  jxl::ImageF result;
  if (!prepared.impl_->comparator->Diffmap(distorted_image, result)) {
    return false;
  }
  const double result_score =
      jxl::ButteraugliScoreFromDiffmap(result, &prepared.impl_->params);
  return CopyResultAtomically(result, distance_map, result_score, score);
}

const char *IntermediateStageName(IntermediateStage stage) {
  constexpr std::array<const char *, kIntermediateStageCount> kNames = {
      "blur_sigma_1.2",
      "blur_sigma_7.15593339443",
      "blur_sigma_3.22489901262",
      "blur_sigma_1.56416327805",
      "blur_sigma_2.7",
      "opsin_x",
      "opsin_y",
      "opsin_b",
      "lf_x",
      "lf_y",
      "lf_b",
      "mf_x",
      "mf_y",
      "mf_b",
      "hf_x",
      "hf_y",
      "uhf_x",
      "uhf_y",
      "malta_mf_y",
      "malta_mf_x",
      "malta_hf_y",
      "malta_hf_x",
      "malta_uhf_y",
      "malta_uhf_x",
      "mask",
      "masked_ac_y",
      "final_composition",
  };
  const size_t index = static_cast<size_t>(stage);
  return index < kNames.size() ? kNames[index] : "unknown";
}

bool ComputePinnedIntermediateStages(
    ConstOracleImage3 reference, ConstOracleImage3 distorted,
    OracleOptions options, IntermediateStageOutput *output,
    const JxlMemoryManager *requested_memory_manager) {
  if (output == nullptr || !ValidRequest(reference, distorted, options)) {
    return false;
  }
  try {
    IntermediateStageOutput local;
    local.extent = reference.plane[0].extent;
    JxlMemoryManager memory_manager{};
    if (!InitMemoryManager(requested_memory_manager, &memory_manager)) {
      return false;
    }
    jxl::Image3F image0;
    jxl::Image3F image1;
    if (!CopyToLibjxl(reference, &memory_manager, &image0) ||
        !CopyToLibjxl(distorted, &memory_manager, &image1)) {
      return false;
    }
    const jxl::ButteraugliParams params = ToLibjxl(options);

    constexpr std::array<float, 5> kBlurSigmas = {
        1.2f, 7.15593339443f, 3.22489901262f, 1.56416327805f, 2.7f,
    };
    for (size_t index = 0; index < kBlurSigmas.size(); ++index) {
      jxl::BlurTemp blur_temp;
      jxl::ImageF blurred;
      if (!AllocatePlane(&memory_manager, local.extent, &blurred) ||
          !jxl::Blur(image0.Plane(0), kBlurSigmas[index], params, &blur_temp,
                     &blurred)) {
        return false;
      }
      CopyStagePlane(blurred, static_cast<IntermediateStage>(index), &local);
    }

    jxl::Image3F xyb0;
    jxl::Image3F xyb1;
    jxl::Image3F opsin_temp;
    if (!AllocateImage3(&memory_manager, local.extent, &xyb0) ||
        !AllocateImage3(&memory_manager, local.extent, &xyb1) ||
        !AllocateImage3(&memory_manager, local.extent, &opsin_temp)) {
      return false;
    }
    jxl::BlurTemp opsin_blur_temp;
    if (!jxl::HWY_NAMESPACE::OpsinDynamicsImage(image0, params, &opsin_temp,
                                                &opsin_blur_temp, &xyb0) ||
        !jxl::HWY_NAMESPACE::OpsinDynamicsImage(image1, params, &opsin_temp,
                                                &opsin_blur_temp, &xyb1)) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      CopyStagePlane(
          xyb0.Plane(channel),
          static_cast<IntermediateStage>(
              static_cast<size_t>(IntermediateStage::kOpsinX) + channel),
          &local);
    }

    jxl::PsychoImage psycho0;
    jxl::PsychoImage psycho1;
    jxl::BlurTemp frequency_blur_temp;
    if (!jxl::HWY_NAMESPACE::SeparateFrequencies(
            local.extent.width, local.extent.height, params,
            &frequency_blur_temp, xyb0, psycho0) ||
        !jxl::HWY_NAMESPACE::SeparateFrequencies(
            local.extent.width, local.extent.height, params,
            &frequency_blur_temp, xyb1, psycho1)) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      CopyStagePlane(
          psycho0.lf.Plane(channel),
          static_cast<IntermediateStage>(
              static_cast<size_t>(IntermediateStage::kLowFrequencyX) + channel),
          &local);
      CopyStagePlane(
          psycho0.mf.Plane(channel),
          static_cast<IntermediateStage>(
              static_cast<size_t>(IntermediateStage::kMediumFrequencyX) +
              channel),
          &local);
    }
    for (size_t channel = 0; channel < 2; ++channel) {
      CopyStagePlane(
          psycho0.hf[channel],
          static_cast<IntermediateStage>(
              static_cast<size_t>(IntermediateStage::kHighFrequencyX) +
              channel),
          &local);
      CopyStagePlane(
          psycho0.uhf[channel],
          static_cast<IntermediateStage>(
              static_cast<size_t>(IntermediateStage::kUltraHighFrequencyX) +
              channel),
          &local);
    }

    const float asymmetry = params.hf_asymmetry;
    const float sqrt_asymmetry = std::sqrt(asymmetry);
    const std::array<const jxl::ImageF *, 6> malta0 = {
        &psycho0.mf.Plane(1), &psycho0.mf.Plane(0), &psycho0.hf[1],
        &psycho0.hf[0],       &psycho0.uhf[1],      &psycho0.uhf[0],
    };
    const std::array<const jxl::ImageF *, 6> malta1 = {
        &psycho1.mf.Plane(1), &psycho1.mf.Plane(0), &psycho1.hf[1],
        &psycho1.hf[0],       &psycho1.uhf[1],      &psycho1.uhf[0],
    };
    const std::array<double, 6> weight_up = {
        kMaltaWeights[0],
        kMaltaWeights[1],
        kMaltaWeights[2] * sqrt_asymmetry,
        kMaltaWeights[3] * sqrt_asymmetry,
        kMaltaWeights[4] * asymmetry,
        kMaltaWeights[5] * asymmetry,
    };
    const std::array<double, 6> weight_down = {
        kMaltaWeights[0],
        kMaltaWeights[1],
        kMaltaWeights[2] / sqrt_asymmetry,
        kMaltaWeights[3] / sqrt_asymmetry,
        kMaltaWeights[4] / asymmetry,
        kMaltaWeights[5] / asymmetry,
    };
    std::array<jxl::ImageF, 6> malta_output;
    for (size_t index = 0; index < malta_output.size(); ++index) {
      if (!RunMalta(*malta0[index], *malta1[index], index < 4, weight_up[index],
                    weight_down[index], kMaltaNorms[index], &memory_manager,
                    &malta_output[index])) {
        return false;
      }
      CopyStagePlane(
          malta_output[index],
          static_cast<IntermediateStage>(
              static_cast<size_t>(IntermediateStage::kMaltaMediumFrequencyY) +
              index),
          &local);
    }

    jxl::Image3F block_diff_ac;
    jxl::Image3F block_diff_dc;
    if (!AllocateImage3(&memory_manager, local.extent, &block_diff_ac) ||
        !AllocateImage3(&memory_manager, local.extent, &block_diff_dc)) {
      return false;
    }
    jxl::ZeroFillImage(&block_diff_ac);
    jxl::ZeroFillImage(&block_diff_dc);
    // Preserve libjxl's UHF, HF, MF accumulation order. Reordering these
    // positive terms changes the final float bits at some pixels.
    constexpr std::array<size_t, 6> kMaltaAccumulationOrder = {4, 5, 2,
                                                               3, 0, 1};
    for (size_t index : kMaltaAccumulationOrder) {
      const size_t channel = index % 2 == 0 ? 1 : 0;
      for (size_t y = 0; y < local.extent.height; ++y) {
        float *destination = block_diff_ac.PlaneRow(channel, y);
        const float *source = malta_output[index].ConstRow(y);
        for (size_t x = 0; x < local.extent.width; ++x) {
          destination[x] += source[x];
        }
      }
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      if (channel < 2) {
        AddAsymmetricL2Diff(psycho0.hf[channel], psycho1.hf[channel],
                            kL2Weights[channel] * asymmetry,
                            kL2Weights[channel] / asymmetry,
                            &block_diff_ac.Plane(channel));
      }
      AddL2Diff(psycho0.mf.Plane(channel), psycho1.mf.Plane(channel),
                kL2Weights[3 + channel], &block_diff_ac.Plane(channel));
      SetL2Diff(psycho0.lf.Plane(channel), psycho1.lf.Plane(channel),
                kL2Weights[6 + channel], &block_diff_dc.Plane(channel));
    }

    jxl::ImageF mask;
    jxl::BlurTemp mask_blur_temp;
    if (!jxl::HWY_NAMESPACE::MaskPsychoImage(
            psycho0, psycho1, local.extent.width, local.extent.height, params,
            &mask_blur_temp, &mask, &block_diff_ac.Plane(1))) {
      return false;
    }
    CopyStagePlane(mask, IntermediateStage::kMask, &local);
    CopyStagePlane(block_diff_ac.Plane(1), IntermediateStage::kMaskedAcY,
                   &local);

    jxl::ImageF final_map;
    if (!AllocatePlane(&memory_manager, local.extent, &final_map) ||
        !jxl::HWY_NAMESPACE::CombineChannelsToDiffmap(
            mask, block_diff_dc, block_diff_ac, params.xmul, &final_map)) {
      return false;
    }
    CopyStagePlane(final_map, IntermediateStage::kFinalComposition, &local);
    *output = std::move(local);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

} // namespace gjxl::butteraugli_test
