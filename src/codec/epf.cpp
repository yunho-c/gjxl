// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/epf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace gjxl {
namespace {

constexpr float kInvSigmaNumerator = -1.1715728752538099024f;
constexpr float kMinInverseSigma = kInvSigmaNumerator / 0.3f;

enum class EpfPass {
  kPass0,
  kPass1,
  kPass2,
};

size_t Mirror(ptrdiff_t coordinate, size_t size) {
  ptrdiff_t mirrored = coordinate;
  const ptrdiff_t signed_size = static_cast<ptrdiff_t>(size);
  while (mirrored < 0 || mirrored >= signed_size) {
    if (mirrored < 0) {
      mirrored = -mirrored - 1;
    } else {
      mirrored = 2 * signed_size - mirrored - 1;
    }
  }
  return static_cast<size_t>(mirrored);
}

float Sample(ConstPlaneF32View plane, ptrdiff_t x, ptrdiff_t y) {
  return plane.Row(Mirror(y, plane.extent.height))[
    Mirror(x, plane.extent.width)];
}

float EpfWeight(float sad, float scaled_inverse_sigma) {
  return std::max(0.0f, std::fma(sad, scaled_inverse_sigma, 1.0f));
}

float PatchSad(
  ConstImage3FView input,
  ptrdiff_t x,
  ptrdiff_t y,
  ptrdiff_t dx,
  ptrdiff_t dy,
  const std::array<float, 3>& channel_scale) {

  constexpr std::array<std::array<ptrdiff_t, 2>, 5> kPlusOffsets = {{
    {0, 0}, {0, -1}, {-1, 0}, {0, 1}, {1, 0},
  }};
  float sad = 0.0f;
  for (size_t channel = 0; channel < 3; ++channel) {
    float channel_sad = 0.0f;
    for (const auto& offset : kPlusOffsets) {
      channel_sad += std::abs(
        Sample(
          input.plane[channel],
          x + offset[0],
          y + offset[1]) -
        Sample(
          input.plane[channel],
          x + dx + offset[0],
          y + dy + offset[1]));
    }
    sad = std::fma(channel_sad, channel_scale[channel], sad);
  }
  return sad;
}

float PixelSad(
  ConstImage3FView input,
  ptrdiff_t x,
  ptrdiff_t y,
  ptrdiff_t dx,
  ptrdiff_t dy,
  const std::array<float, 3>& channel_scale) {

  float sad = 0.0f;
  for (size_t channel = 0; channel < 3; ++channel) {
    sad = std::fma(
      std::abs(
        Sample(input.plane[channel], x, y) -
        Sample(input.plane[channel], x + dx, y + dy)),
      channel_scale[channel],
      sad);
  }
  return sad;
}

void ApplyEpfPass(
  ConstImage3FView input,
  ConstPlaneF32View inverse_sigma,
  EpfFilterOptions options,
  EpfPass pass,
  Image3FView output) {

  constexpr std::array<std::array<ptrdiff_t, 2>, 12> kPass0Offsets = {{
    {0, -2}, {-1, -1}, {0, -1}, {1, -1},
    {-2, 0}, {-1, 0}, {1, 0}, {2, 0},
    {-1, 1}, {0, 1}, {1, 1}, {0, 2},
  }};
  constexpr std::array<std::array<ptrdiff_t, 2>, 4> kCardinalOffsets = {{
    {0, -1}, {-1, 0}, {1, 0}, {0, 1},
  }};

  const float sigma_scale = 1.65f * (
    pass == EpfPass::kPass0
      ? options.pass0_sigma_scale
      : pass == EpfPass::kPass2
        ? options.pass2_sigma_scale
        : 1.0f);

  for (size_t y = 0; y < input.height(); ++y) {
    for (size_t x = 0; x < input.width(); ++x) {
      const float block_inverse_sigma =
        inverse_sigma.Row(y / kJxlBlockDimension)[x / kJxlBlockDimension];
      if (block_inverse_sigma < kMinInverseSigma) {
        for (size_t channel = 0; channel < 3; ++channel) {
          output.plane[channel].Row(y)[x] = input.plane[channel].Row(y)[x];
        }
        continue;
      }

      const bool block_border =
        x % kJxlBlockDimension == 0 ||
        x % kJxlBlockDimension == kJxlBlockDimension - 1 ||
        y % kJxlBlockDimension == 0 ||
        y % kJxlBlockDimension == kJxlBlockDimension - 1;
      const float scaled_inverse_sigma = block_inverse_sigma * sigma_scale *
        (block_border ? options.border_sad_multiplier : 1.0f);
      const ptrdiff_t sx = static_cast<ptrdiff_t>(x);
      const ptrdiff_t sy = static_cast<ptrdiff_t>(y);
      std::array<float, 3> sum = {
        input.plane[0].Row(y)[x],
        input.plane[1].Row(y)[x],
        input.plane[2].Row(y)[x],
      };
      float weight_sum = 1.0f;

      const auto add_candidate = [&](ptrdiff_t dx, ptrdiff_t dy, float sad) {
        const float weight = EpfWeight(sad, scaled_inverse_sigma);
        weight_sum += weight;
        for (size_t channel = 0; channel < 3; ++channel) {
          sum[channel] = std::fma(
            weight,
            Sample(input.plane[channel], sx + dx, sy + dy),
            sum[channel]);
        }
      };

      if (pass == EpfPass::kPass0) {
        for (const auto& offset : kPass0Offsets) {
          add_candidate(
            offset[0],
            offset[1],
            PatchSad(
              input,
              sx,
              sy,
              offset[0],
              offset[1],
              options.channel_scale));
        }
      } else {
        for (const auto& offset : kCardinalOffsets) {
          const float sad = pass == EpfPass::kPass1
            ? PatchSad(
                input,
                sx,
                sy,
                offset[0],
                offset[1],
                options.channel_scale)
            : PixelSad(
                input,
                sx,
                sy,
                offset[0],
                offset[1],
                options.channel_scale);
          add_candidate(offset[0], offset[1], sad);
        }
      }

      for (size_t channel = 0; channel < 3; ++channel) {
        output.plane[channel].Row(y)[x] = sum[channel] / weight_sum;
      }
    }
  }
}

}  // namespace

Status FillDefaultEpfSharpness(PlaneU8View sharpness) {
  if (!sharpness.valid()) {
    return Status::InvalidArgument(
      "EPF sharpness field is invalid");
  }
  for (size_t y = 0; y < sharpness.extent.height; ++y) {
    std::fill_n(sharpness.Row(y), sharpness.extent.width, uint8_t{4});
  }
  return Status::Ok();
}

Status ComputeEpfInverseSigma(
  const AcStrategyGrid& strategies,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  ConstPlaneU8View sharpness,
  EpfSigmaOptions options,
  PlaneF32View inverse_sigma) {

  if (!strategies.complete() ||
      !raw_quant_field.valid() ||
      !sharpness.valid() ||
      !inverse_sigma.valid() ||
      raw_quant_field.extent != strategies.extent() ||
      sharpness.extent != strategies.extent() ||
      inverse_sigma.extent != strategies.extent() ||
      !quantizer.valid() ||
      !std::isfinite(options.quant_multiplier) ||
      options.quant_multiplier <= 0.0f) {
    return Status::InvalidArgument(
      "EPF sigma inputs are invalid");
  }
  for (float value : options.sharpness_lut) {
    if (!std::isfinite(value) || value < 0.0f) {
      return Status::InvalidArgument(
        "EPF sharpness lookup table is invalid");
    }
  }

  size_t value_count = 0;
  if (!strategies.extent().try_area(&value_count)) {
    return Status::InvalidArgument(
      "EPF sigma field dimensions are too large");
  }

  try {
    for (size_t y = 0; y < strategies.extent().height; ++y) {
      for (size_t x = 0; x < strategies.extent().width; ++x) {
        if (raw_quant_field.Row(y)[x] < 1 ||
            raw_quant_field.Row(y)[x] > kMaxRawQuant ||
            sharpness.Row(y)[x] >= options.sharpness_lut.size()) {
          return Status::InvalidArgument(
            "EPF raw quantization or sharpness is out of range");
        }
      }
    }

    std::vector<float> result(value_count);
    const Status status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const Extent2D covered =
          GetAcStrategyInfo(strategy)->covered_blocks;
        const float sigma_quant = options.quant_multiplier /
          (quantizer.scale() *
           static_cast<float>(raw_quant_field.Row(block_y)[block_x]) *
           kInvSigmaNumerator);
        for (size_t dy = 0; dy < covered.height; ++dy) {
          for (size_t dx = 0; dx < covered.width; ++dx) {
            const uint8_t sharpness_value =
              sharpness.Row(block_y + dy)[block_x + dx];
            float sigma = sigma_quant *
              options.sharpness_lut[sharpness_value];
            sigma = std::min(-1.0e-4f, sigma);
            result[(block_y + dy) * strategies.extent().width +
                   block_x + dx] = 1.0f / sigma;
          }
        }
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }

    for (size_t y = 0; y < inverse_sigma.extent.height; ++y) {
      std::copy_n(
        result.data() + y * inverse_sigma.extent.width,
        inverse_sigma.extent.width,
        inverse_sigma.Row(y));
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate EPF sigma storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "EPF sigma field dimensions are too large");
  }
  return Status::Ok();
}

Status ApplyEpf(
  ConstImage3FView input,
  ConstPlaneF32View inverse_sigma,
  EpfFilterOptions options,
  Image3FView output) {

  if (!input.valid() ||
      !output.valid() ||
      !inverse_sigma.valid() ||
      input.extent() != output.extent() ||
      input.width() % kJxlBlockDimension != 0 ||
      input.height() % kJxlBlockDimension != 0 ||
      inverse_sigma.extent != Extent2D{
        input.width() / kJxlBlockDimension,
        input.height() / kJxlBlockDimension} ||
      options.iterations > 3 ||
      !std::isfinite(options.pass0_sigma_scale) ||
      options.pass0_sigma_scale <= 0.0f ||
      !std::isfinite(options.pass2_sigma_scale) ||
      options.pass2_sigma_scale <= 0.0f ||
      !std::isfinite(options.border_sad_multiplier) ||
      options.border_sad_multiplier <= 0.0f) {
    return Status::InvalidArgument(
      "EPF image, sigma field, or filter options are invalid");
  }
  for (float channel_scale : options.channel_scale) {
    if (!std::isfinite(channel_scale) || channel_scale < 0.0f) {
      return Status::InvalidArgument(
        "EPF channel scales are invalid");
    }
  }
  for (size_t y = 0; y < inverse_sigma.extent.height; ++y) {
    for (size_t x = 0; x < inverse_sigma.extent.width; ++x) {
      if (!std::isfinite(inverse_sigma.Row(y)[x]) ||
          inverse_sigma.Row(y)[x] >= 0.0f) {
        return Status::InvalidArgument(
          "EPF inverse sigma values must be finite and negative");
      }
    }
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < input.height(); ++y) {
      for (size_t x = 0; x < input.width(); ++x) {
        if (!std::isfinite(input.plane[channel].Row(y)[x])) {
          return Status::InvalidArgument(
            "EPF input pixels must be finite");
        }
      }
    }
  }

  try {
    size_t pixel_count = 0;
    if (!input.extent().try_area(&pixel_count)) {
      return Status::InvalidArgument(
        "EPF image dimensions are too large");
    }
    std::array<std::vector<float>, 3> scratch_a;
    std::array<std::vector<float>, 3> scratch_b;
    for (size_t channel = 0; channel < 3; ++channel) {
      scratch_a[channel].resize(pixel_count);
      if (options.iterations >= 2) {
        scratch_b[channel].resize(pixel_count);
      }
      for (size_t y = 0; y < input.height(); ++y) {
        std::copy_n(
          input.plane[channel].Row(y),
          input.width(),
          scratch_a[channel].data() + y * input.width());
      }
    }

    const auto view = [&](std::array<std::vector<float>, 3>& storage) {
      return Image3FView{{
        PlaneF32View{storage[0].data(), input.extent(), input.width()},
        PlaneF32View{storage[1].data(), input.extent(), input.width()},
        PlaneF32View{storage[2].data(), input.extent(), input.width()},
      }};
    };
    const auto const_view = [&](
      const std::array<std::vector<float>, 3>& storage) {
      return ConstImage3FView{{
        ConstPlaneF32View{
          storage[0].data(), input.extent(), input.width()},
        ConstPlaneF32View{
          storage[1].data(), input.extent(), input.width()},
        ConstPlaneF32View{
          storage[2].data(), input.extent(), input.width()},
      }};
    };

    if (options.iterations == 1) {
      // Preserve the original input in scratch B only when it is needed as
      // the source of a single pass.
      for (size_t channel = 0; channel < 3; ++channel) {
        scratch_b[channel] = scratch_a[channel];
      }
      ApplyEpfPass(
        const_view(scratch_b),
        inverse_sigma,
        options,
        EpfPass::kPass1,
        view(scratch_a));
    } else if (options.iterations == 2) {
      ApplyEpfPass(
        const_view(scratch_a),
        inverse_sigma,
        options,
        EpfPass::kPass1,
        view(scratch_b));
      ApplyEpfPass(
        const_view(scratch_b),
        inverse_sigma,
        options,
        EpfPass::kPass2,
        view(scratch_a));
    } else if (options.iterations == 3) {
      ApplyEpfPass(
        const_view(scratch_a),
        inverse_sigma,
        options,
        EpfPass::kPass0,
        view(scratch_b));
      ApplyEpfPass(
        const_view(scratch_b),
        inverse_sigma,
        options,
        EpfPass::kPass1,
        view(scratch_a));
      ApplyEpfPass(
        const_view(scratch_a),
        inverse_sigma,
        options,
        EpfPass::kPass2,
        view(scratch_b));
      scratch_a.swap(scratch_b);
    }

    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < input.height(); ++y) {
        std::copy_n(
          scratch_a[channel].data() + y * input.width(),
          input.width(),
          output.plane[channel].Row(y));
      }
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate EPF scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "EPF image dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace gjxl
