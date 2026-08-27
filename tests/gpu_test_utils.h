// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/geometry.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl::test {

class GuardedDevicePlane {
public:
  [[nodiscard]] Status Prepare(
    GpuBackend& backend,
    Extent2D extent,
    size_t row_stride,
    size_t prefix_words = 7,
    size_t suffix_words = 11) {

    if (extent.empty() || row_stride < extent.width) {
      return Status::InvalidArgument(
        "Guarded plane geometry is invalid");
    }
    backend_ = &backend;
    extent_ = extent;
    row_stride_ = row_stride;
    prefix_words_ = prefix_words;
    const size_t plane_words = row_stride * extent.height;
    words_.assign(prefix_words + plane_words + suffix_words, kGuardBits);
    Status status = backend.Allocate(
      words_.size() * sizeof(uint32_t), &buffer_);
    if (!status.ok()) {
      return status;
    }
    return Upload();
  }

  void SetLogical(std::span<const float> values) {
    size_t index = 0;
    for (size_t y = 0; y < extent_.height; ++y) {
      for (size_t x = 0; x < extent_.width; ++x) {
        words_[prefix_words_ + y * row_stride_ + x] =
          std::bit_cast<uint32_t>(values[index++]);
      }
    }
  }

  void PoisonLogical() {
    for (size_t y = 0; y < extent_.height; ++y) {
      for (size_t x = 0; x < extent_.width; ++x) {
        words_[prefix_words_ + y * row_stride_ + x] = kPoisonBits;
      }
    }
  }

  [[nodiscard]] Status Upload() {
    return backend_->CopyHostToDevice(
      *buffer_, words_.data(), words_.size() * sizeof(uint32_t));
  }

  [[nodiscard]] Status Download() {
    return backend_->CopyDeviceToHost(
      *buffer_, words_.data(), words_.size() * sizeof(uint32_t));
  }

  [[nodiscard]] std::vector<float> Logical() const {
    std::vector<float> result;
    result.reserve(extent_.width * extent_.height);
    for (size_t y = 0; y < extent_.height; ++y) {
      for (size_t x = 0; x < extent_.width; ++x) {
        result.push_back(std::bit_cast<float>(
          words_[prefix_words_ + y * row_stride_ + x]));
      }
    }
    return result;
  }

  [[nodiscard]] bool GuardsIntact() const noexcept {
    for (size_t index = 0; index < words_.size(); ++index) {
      bool logical = false;
      if (index >= prefix_words_) {
        const size_t local = index - prefix_words_;
        const size_t y = local / row_stride_;
        const size_t x = local % row_stride_;
        logical = y < extent_.height && x < extent_.width;
      }
      if (!logical && words_[index] != kGuardBits) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] DevicePlaneView View() noexcept {
    return {
      buffer_.get(),
      prefix_words_ * sizeof(uint32_t),
      DeviceElementType::kF32,
      extent_,
      row_stride_,
    };
  }

  [[nodiscard]] ConstDevicePlaneView ConstView() const noexcept {
    return {
      buffer_.get(),
      prefix_words_ * sizeof(uint32_t),
      DeviceElementType::kF32,
      extent_,
      row_stride_,
    };
  }

  [[nodiscard]] DeviceBuffer& buffer() noexcept {
    return *buffer_;
  }

private:
  static constexpr uint32_t kGuardBits = 0x7fa12345u;
  static constexpr uint32_t kPoisonBits = 0x7fc00001u;

  GpuBackend* backend_ = nullptr;
  std::unique_ptr<DeviceBuffer> buffer_;
  Extent2D extent_;
  size_t row_stride_ = 0;
  size_t prefix_words_ = 0;
  std::vector<uint32_t> words_;
};

}  // namespace gjxl::test
