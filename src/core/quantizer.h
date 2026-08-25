// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "core/status.h"

namespace gjxl {

enum class XybChannel : uint8_t {
  kX = 0,
  kY = 1,
  kB = 2,
};

inline constexpr int32_t kMaxRawQuant = 256;
inline constexpr uint32_t kMaxEncoderGlobalScale = 1u << 15;
inline constexpr uint32_t kMaxQuantDc = 1u << 16;
inline constexpr uint32_t kQuantGlobalScaleDenominator = 1u << 16;

struct QuantizerParams {
  uint32_t global_scale = 1024;
  uint32_t quant_dc = 64;
};

/// Small, backend-independent quantizer state derived from serialized values.
class Quantizer {
public:
  Quantizer() = default;

  [[nodiscard]] static Status Create(
    QuantizerParams params,
    Quantizer* out) {

    if (out == nullptr) {
      return Status::InvalidArgument(
        "Quantizer output is null");
    }

    if (params.global_scale == 0 ||
        params.global_scale > kMaxEncoderGlobalScale ||
        params.quant_dc == 0 ||
        params.quant_dc > kMaxQuantDc) {

      return Status::InvalidArgument(
        "Quantizer parameters are outside encoder ranges");
    }

    Quantizer result;
    result.params_ = params;
    result.scale_ =
      static_cast<float>(params.global_scale) /
      static_cast<float>(kQuantGlobalScaleDenominator);
    result.inverse_global_scale_ =
      static_cast<float>(kQuantGlobalScaleDenominator) /
      static_cast<float>(params.global_scale);

    const float inverse_quant_dc =
      result.inverse_global_scale_ /
      static_cast<float>(params.quant_dc);

    constexpr std::array<float, 3> kInverseDcQuant = {
      4096.0f,
      512.0f,
      256.0f,
    };

    for (size_t channel = 0; channel < 3; ++channel) {
      result.dc_steps_[channel] =
        inverse_quant_dc / kInverseDcQuant[channel];
      result.inverse_dc_steps_[channel] =
        kInverseDcQuant[channel] *
        result.scale_ *
        static_cast<float>(params.quant_dc);
    }

    result.valid_ = true;
    *out = result;
    return Status::Ok();
  }

  [[nodiscard]] bool valid() const noexcept {
    return valid_;
  }

  [[nodiscard]] QuantizerParams params() const noexcept {
    return params_;
  }

  [[nodiscard]] float scale() const noexcept {
    return scale_;
  }

  [[nodiscard]] float inverse_global_scale() const noexcept {
    return inverse_global_scale_;
  }

  [[nodiscard]] float inverse_quant_ac(
    int32_t raw_quant) const noexcept {

    return inverse_global_scale_ /
      static_cast<float>(raw_quant);
  }

  [[nodiscard]] const std::array<float, 3>& dc_steps() const noexcept {
    return dc_steps_;
  }

  [[nodiscard]] const std::array<float, 3>& inverse_dc_steps() const noexcept {
    return inverse_dc_steps_;
  }

private:
  QuantizerParams params_;
  float scale_ = 0.0f;
  float inverse_global_scale_ = 0.0f;
  std::array<float, 3> dc_steps_{};
  std::array<float, 3> inverse_dc_steps_{};
  bool valid_ = false;
};

}  // namespace gjxl
