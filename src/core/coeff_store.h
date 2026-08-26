// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace gjxl {

/// Quantized AC data for one complete transform, in canonical coefficient
/// layout. The transform's LLF entries are zero; DC is stored separately.
struct QuantizedTransform {
  size_t block_x = 0;
  size_t block_y = 0;
  AcStrategyType strategy = AcStrategyType::kDct8;
  int32_t raw_quant = 0;
  std::array<std::vector<int32_t>, 3> ac;
};

/// Owns the CPU reference representation between coefficient coding and
/// reconstruction. DC has one floating-point value per base block and channel.
class QuantizedCoefficientFrame {
public:
  QuantizedCoefficientFrame() = default;

  [[nodiscard]] static Status Create(
    Extent2D block_extent,
    QuantizedCoefficientFrame* out) {

    if (out == nullptr) {
      return Status::InvalidArgument(
        "Quantized coefficient frame output is null");
    }

    size_t block_count = 0;
    if (block_extent.empty() || !block_extent.try_area(&block_count)) {
      return Status::InvalidArgument(
        "Quantized coefficient frame extent is invalid");
    }

    try {
      QuantizedCoefficientFrame result;
      result.block_extent_ = block_extent;
      for (std::vector<float>& dc : result.dc_) {
        dc.assign(block_count, 0.0f);
      }
      result.occupied_.assign(block_count, false);
      *out = std::move(result);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate quantized coefficient frame");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Quantized coefficient frame extent is too large");
    }

    return Status::Ok();
  }

  [[nodiscard]] bool valid() const noexcept {
    size_t block_count = 0;
    if (block_extent_.empty() ||
        !block_extent_.try_area(&block_count) ||
        occupied_.size() != block_count) {
      return false;
    }

    for (const std::vector<float>& dc : dc_) {
      if (dc.size() != block_count) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool complete() const noexcept {
    if (!valid()) {
      return false;
    }
    for (bool occupied : occupied_) {
      if (!occupied) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] Extent2D block_extent() const noexcept {
    return block_extent_;
  }

  [[nodiscard]] const std::vector<QuantizedTransform>& transforms()
    const noexcept {
    return transforms_;
  }

  [[nodiscard]] float dc(
    size_t channel,
    size_t block_x,
    size_t block_y) const noexcept {

    return dc_[channel][block_y * block_extent_.width + block_x];
  }

  [[nodiscard]] Status SetDc(
    size_t channel,
    size_t block_x,
    size_t block_y,
    float value) {

    if (!valid() ||
        channel >= dc_.size() ||
        block_x >= block_extent_.width ||
        block_y >= block_extent_.height ||
        !std::isfinite(value)) {
      return Status::InvalidArgument(
        "Quantized coefficient DC assignment is invalid");
    }

    dc_[channel][block_y * block_extent_.width + block_x] = value;
    return Status::Ok();
  }

  [[nodiscard]] Status AddTransform(QuantizedTransform transform) {
    if (!valid() ||
        transform.raw_quant < 1 ||
        transform.raw_quant > kMaxRawQuant) {
      return Status::InvalidArgument(
        "Quantized transform metadata is invalid");
    }

    const AcStrategyInfo* info = GetAcStrategyInfo(transform.strategy);
    if (info == nullptr ||
        transform.block_x >= block_extent_.width ||
        transform.block_y >= block_extent_.height ||
        info->covered_blocks.width >
          block_extent_.width - transform.block_x ||
        info->covered_blocks.height >
          block_extent_.height - transform.block_y) {
      return Status::InvalidArgument(
        "Quantized transform does not fit the frame");
    }

    for (const std::vector<int32_t>& ac : transform.ac) {
      if (ac.size() != info->coefficient_count()) {
        return Status::InvalidArgument(
          "Quantized transform coefficient count is invalid");
      }
    }

    for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
      for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
        const size_t index =
          (transform.block_y + dy) * block_extent_.width +
          transform.block_x + dx;
        if (occupied_[index]) {
          return Status::InvalidArgument(
            "Quantized transform overlaps existing coefficients");
        }
      }
    }

    try {
      transforms_.push_back(std::move(transform));
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to store quantized transform");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Too many quantized transforms");
    }

    const QuantizedTransform& stored = transforms_.back();
    for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
      for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
        occupied_[
          (stored.block_y + dy) * block_extent_.width +
          stored.block_x + dx] = true;
      }
    }

    return Status::Ok();
  }

private:
  Extent2D block_extent_;
  std::array<std::vector<float>, 3> dc_;
  std::vector<QuantizedTransform> transforms_;
  std::vector<bool> occupied_;
};

}  // namespace gjxl
