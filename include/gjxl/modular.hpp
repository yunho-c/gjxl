// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include "core/geometry.h"
#include "core/execution_domain.h"
#include "core/status.h"
namespace gjxl {
// Samples are source-space unsigned integers. RGBA alpha is unassociated;
// invisible RGB is preserved. RGB uses sRGB; gray uses the sRGB transfer curve.
enum class ModularPixelFormat : uint8_t { kGray8, kRgb8, kRgba8, kGray16, kRgb16, kRgba16 };
enum class ModularByteOrder : uint8_t { kLittleEndian, kBigEndian };
enum class ModularBackend : uint8_t { kAutomatic, kCpu, kMetal, kCuda };
enum class ModularEntropy : uint8_t { kPrefix, kAns };
struct ModularImageView {
  std::span<const uint8_t> bytes;
  Extent2D extent;
  size_t row_stride_bytes = 0;
  ModularPixelFormat format = ModularPixelFormat::kRgb8;
  ModularByteOrder byte_order = ModularByteOrder::kLittleEndian;
};
struct ModularEncodingOptions {
  ModularBackend backend = ModularBackend::kAutomatic; // Automatic resolves to CPU.
  size_t cpu_thread_count = 0; // 0 automatic, 1 serial oracle, 2..256 upper bound.
  std::shared_ptr<const ExecutionDomain> execution_domain;
  ModularEntropy entropy = ModularEntropy::kPrefix;
  bool search = false; // Bounded lossless policy search; never changes samples.
};
struct ModularEncodingSummary {
  Extent2D extent;
  size_t encoded_bytes = 0;
  ModularBackend backend = ModularBackend::kCpu;
  double total_seconds = 0;
};
// Atomic output and summary replacement. Input layout and format are validated
// before admission. GPU requests are unsupported; no float conversion occurs.
[[nodiscard]] Status EncodeModularImage(ModularImageView input, ModularEncodingOptions options,
                                        std::vector<uint8_t> *output,
                                        ModularEncodingSummary *summary = nullptr);
} // namespace gjxl
