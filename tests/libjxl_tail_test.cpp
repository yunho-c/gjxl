// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/codestream.h"
#include "codec/reconstruction.h"
#include "codec/vardct_frame.h"
#include "codestream/libjxl_tail_internal.h"
#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/status.h"

namespace {

template <typename T>
gjxl::PlaneView<const T> View(const std::vector<T> &values,
                              gjxl::Extent2D extent) {
  return {values.data(), extent, extent.width};
}

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D extent) : extent(extent) {
    for (std::vector<float> &values : planes) {
      values.resize(extent.width * extent.height);
    }
  }

  gjxl::ConstImage3FView ConstView() const {
    return gjxl::ConstImage3FView{{
        gjxl::ConstPlaneF32View{planes[0].data(), extent, extent.width},
        gjxl::ConstPlaneF32View{planes[1].data(), extent, extent.width},
        gjxl::ConstPlaneF32View{planes[2].data(), extent, extent.width},
    }};
  }

  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> planes;
};

gjxl::Status MakeFrame(gjxl::VarDctEncoderFrame *frame) {
  gjxl::FrameGeometry geometry;
  gjxl::Status status = gjxl::FrameGeometry::Create(17, 13, &geometry);
  if (!status.ok()) {
    return status;
  }

  ImageStorage opsin(geometry.padded_frame());
  for (size_t y = 0; y < opsin.extent.height; ++y) {
    for (size_t x = 0; x < opsin.extent.width; ++x) {
      const size_t index = y * opsin.extent.width + x;
      const float value =
          static_cast<float>((x * 17 + y * 11 + 5) % 97) * 0.002f;
      opsin.planes[0][index] = value - 0.01f;
      opsin.planes[1][index] = value + 0.02f;
      opsin.planes[2][index] = 1.2f * value - 0.03f;
    }
  }

  const gjxl::Extent2D blocks = geometry.block_grid().blocks;
  size_t block_count = 0;
  if (!blocks.try_area(&block_count)) {
    return gjxl::Status::InvalidArgument("Test block count overflow");
  }
  gjxl::AcStrategyGrid strategies;
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  if (!(status = gjxl::AcStrategyGrid::Create(blocks, &strategies)).ok()) {
    return status;
  }
  strategies.fill_dct8();
  if (!(status = gjxl::Quantizer::Create({3541, 10}, &quantizer)).ok()) {
    return status;
  }
  if (!(status = gjxl::ComputeInitialColorCorrelationMap(opsin.ConstView(),
                                                         &color_correlation))
           .ok()) {
    return status;
  }
  const std::vector<int32_t> raw_quant(block_count, 29);
  std::vector<uint8_t> sharpness(block_count);
  for (size_t index = 0; index < block_count; ++index) {
    sharpness[index] = static_cast<uint8_t>(index % 8);
  }
  return gjxl::ComputeQuantizedCoefficients(
      opsin.ConstView(),
      {
          .geometry = geometry,
          .strategies = &strategies,
          .raw_quant_field = View(raw_quant, blocks),
          .quantizer = &quantizer,
          .color_correlation = &color_correlation,
          .epf_sharpness = View(sharpness, blocks),
      },
      {}, frame);
}

bool CheckStatus(const gjxl::Status &status, gjxl::StatusCode expected,
                 const char *operation) {
  if (status.code() == expected) {
    return true;
  }
  std::cerr << operation << " returned status "
            << static_cast<int>(status.code()) << ": " << status.message()
            << '\n';
  return false;
}

} // namespace

int main() {
  gjxl::VarDctEncoderFrame frame;
  const gjxl::Status frame_status = MakeFrame(&frame);
  if (!frame_status.ok()) {
    std::cerr << "Could not construct the bridge fixture: "
              << frame_status.message() << '\n';
    return 1;
  }

  const bool available =
      gjxl::codestream_internal::LibjxlTailExperimentAvailable();
  if (available != static_cast<bool>(GJXL_TEST_LIBJXL_TAIL_ENABLED)) {
    std::cerr << "Libjxl tail availability does not match the build option\n";
    return 1;
  }

  const std::vector<uint8_t> sentinel{0xA5, 0x5A, 0xC3};
  std::vector<uint8_t> output = sentinel;
  const gjxl::Status status =
      gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxl(frame, {},
                                                                  &output);
#if GJXL_TEST_LIBJXL_TAIL_ENABLED
  constexpr gjxl::StatusCode kExpected = gjxl::StatusCode::kUnsupported;
#else
  constexpr gjxl::StatusCode kExpected = gjxl::StatusCode::kUnavailable;
#endif
  if (!CheckStatus(status, kExpected, "Valid bridge request") ||
      output != sentinel) {
    std::cerr << "A failed bridge request changed caller-visible output\n";
    return 1;
  }

  if (!CheckStatus(gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxl(
                       frame, {}, nullptr),
                   gjxl::StatusCode::kInvalidArgument,
                   "Null-output bridge request")) {
    return 1;
  }

  output = sentinel;
  if (!CheckStatus(gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxl(
                       frame, {.effort = 0}, &output),
                   gjxl::StatusCode::kInvalidArgument,
                   "Invalid-effort bridge request") ||
      output != sentinel) {
    std::cerr << "Invalid options changed caller-visible output\n";
    return 1;
  }
  return 0;
}
