// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#if GJXL_TEST_LIBJXL_TAIL_ENABLED
#include <jxl/decode.h>
#include <jxl/encode.h>
#endif

#include "codec/chroma_from_luma.h"
#include "codec/codestream.h"
#include "codec/reconstruction.h"
#include "codec/vardct_frame.h"
#include "codestream/libjxl_tail_internal.h"
#include "codestream/encoder.h"
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

gjxl::Status MakeFrame(gjxl::Extent2D frame_extent, bool mixed,
                       gjxl::VarDctEncoderFrame *frame) {
  gjxl::FrameGeometry geometry;
  gjxl::Status status = gjxl::FrameGeometry::Create(frame_extent, &geometry);
  if (!status.ok()) {
    return status;
  }

  ImageStorage opsin(geometry.padded_frame());
  for (size_t y = 0; y < opsin.extent.height; ++y) {
    for (size_t x = 0; x < opsin.extent.width; ++x) {
      const size_t index = y * opsin.extent.width + x;
      const float value =
          0.22f * std::sin(0.031f * static_cast<float>(x + 3 * y)) +
          0.13f * std::cos(0.047f * static_cast<float>(2 * x + y));
      opsin.planes[0][index] = 0.7f * value + 0.01f;
      opsin.planes[1][index] = value;
      opsin.planes[2][index] = 1.2f * value - 0.02f;
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
  if (mixed) {
    if (!(status = strategies.Set(0, 0, gjxl::AcStrategyType::kDct32x32))
             .ok() ||
        !(status = strategies.Set(4, 0,
                                  gjxl::AcStrategyType::kDct32x16))
             .ok() ||
        !(status = strategies.Set(6, 0,
                                  gjxl::AcStrategyType::kDct16x32))
             .ok() ||
        !(status = strategies.Set(10, 0,
                                  gjxl::AcStrategyType::kDct16x16))
             .ok() ||
        !(status = strategies.Set(6, 2,
                                  gjxl::AcStrategyType::kDct16x8))
             .ok() ||
        !(status = strategies.Set(7, 2,
                                  gjxl::AcStrategyType::kDct8x16))
             .ok()) {
      return status;
    }
    strategies.fill_empty_dct8();
  } else {
    strategies.fill_dct8();
  }
  if (!(status = gjxl::Quantizer::Create({3541, 10}, &quantizer)).ok()) {
    return status;
  }
  if (mixed) {
    const gjxl::Extent2D tiles = gjxl::ColorTileExtent(geometry.padded_frame());
    size_t tile_count = 0;
    if (!tiles.try_area(&tile_count)) {
      return gjxl::Status::InvalidArgument("Test color-tile count overflow");
    }
    std::vector<int8_t> y_to_x(tile_count);
    std::vector<int8_t> y_to_b(tile_count);
    for (size_t index = 0; index < tile_count; ++index) {
      y_to_x[index] = static_cast<int8_t>(static_cast<int>(index % 13) - 6);
      y_to_b[index] = static_cast<int8_t>(7 - static_cast<int>(index % 17));
    }
    status = gjxl::chroma_from_luma_internal::CreateColorCorrelationMap(
        View(y_to_x, tiles), View(y_to_b, tiles), &color_correlation);
  } else {
    status = gjxl::ComputeInitialColorCorrelationMap(opsin.ConstView(),
                                                     &color_correlation);
  }
  if (!status.ok()) {
    return status;
  }
  std::vector<int32_t> raw_quant(block_count, 29);
  std::vector<uint8_t> sharpness(block_count);
  for (size_t index = 0; index < block_count; ++index) {
    if (mixed) {
      raw_quant[index] = 1 + static_cast<int32_t>((index * 17) % 255);
    }
    sharpness[index] = static_cast<uint8_t>(index % 8);
  }
  gjxl::SimpleVarDctCodestreamProfile profile;
  if (mixed) {
    profile.x_qm_scale = 0;
    profile.b_qm_scale = 7;
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
      profile, frame);
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

bool EveryDigestFieldIsNonzero(
    const gjxl::codestream_internal::LibjxlTailStateDigest &digest) {
  return digest.dimensions != 0 && digest.strategies != 0 &&
         digest.quantizer != 0 && digest.raw_quant != 0 && digest.epf != 0 &&
         digest.cfl != 0 && digest.quantized_dc != 0 && digest.dc != 0 &&
         digest.ac_used_counts != 0 && digest.ac_coefficients != 0;
}

bool CheckAudit(const gjxl::VarDctEncoderFrame &frame, bool available) {
  const gjxl::codestream_internal::LibjxlTailStateAudit sentinel{
      .source = {.dimensions = 0xA5A5A5A5A5A5A5A5u},
      .copied = {.dimensions = 0x5A5A5A5A5A5A5A5Au},
  };
  gjxl::codestream_internal::LibjxlTailStateAudit audit = sentinel;
  const gjxl::Status status =
      gjxl::codestream_internal::AuditVarDctStateWithLibjxl(frame, {}, &audit);
  if (available) {
    return CheckStatus(status, gjxl::StatusCode::kOk,
                       "Valid state-audit request") &&
           audit.source == audit.copied &&
           EveryDigestFieldIsNonzero(audit.source);
  }
  return CheckStatus(status, gjxl::StatusCode::kUnavailable,
                     "Valid state-audit request") &&
         audit == sentinel;
}

bool CheckStressCoverage(const gjxl::VarDctEncoderFrame &frame) {
  const std::array<gjxl::AcStrategyType, 7> expected{
      gjxl::AcStrategyType::kDct8,     gjxl::AcStrategyType::kDct16x16,
      gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
      gjxl::AcStrategyType::kDct8x16,  gjxl::AcStrategyType::kDct32x16,
      gjxl::AcStrategyType::kDct16x32,
  };
  std::array<bool, expected.size()> seen{};
  const gjxl::Extent2D blocks = frame.geometry().block_grid().blocks;
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      gjxl::AcStrategyCell cell;
      if (!frame.strategies().Get(x, y, &cell).ok() || !cell.is_anchor) {
        continue;
      }
      const auto found = std::find(expected.begin(), expected.end(),
                                   cell.strategy);
      if (found != expected.end()) {
        seen[static_cast<size_t>(found - expected.begin())] = true;
      }
    }
  }

  bool edge_group = false;
  bool ac_negative = false;
  bool ac_zero = false;
  bool ac_positive = false;
  for (size_t group_index = 0; group_index < frame.ac_group_count();
       ++group_index) {
    gjxl::VarDctAcGroupView group;
    if (!frame.GetAcGroup(group_index, &group).ok()) {
      return false;
    }
    edge_group |=
        group.used_coefficient_count < gjxl::kVarDctAcGroupCoefficientCapacity;
    for (std::span<const int32_t> coefficients : group.coefficients) {
      for (int32_t coefficient :
           coefficients.first(group.used_coefficient_count)) {
        ac_negative |= coefficient < 0;
        ac_zero |= coefficient == 0;
        ac_positive |= coefficient > 0;
      }
      if (!std::all_of(
              coefficients.begin() + group.used_coefficient_count,
              coefficients.end(), [](int32_t value) { return value == 0; })) {
        return false;
      }
    }
  }

  bool dc_negative = false;
  bool dc_zero = false;
  bool dc_positive = false;
  const gjxl::ConstImage3I32View quantized_dc = frame.quantized_dc();
  for (const gjxl::ConstPlaneI32View plane : quantized_dc.plane) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      for (size_t x = 0; x < plane.extent.width; ++x) {
        const int32_t value = plane.Row(y)[x];
        dc_negative |= value < 0;
        dc_zero |= value == 0;
        dc_positive |= value > 0;
      }
    }
  }
  return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }) &&
         frame.ac_group_count() > 1 && edge_group && ac_negative && ac_zero &&
         ac_positive && dc_negative && dc_zero && dc_positive &&
         frame.profile().x_qm_scale == 0 && frame.profile().b_qm_scale == 7;
}

#if GJXL_TEST_LIBJXL_TAIL_ENABLED
bool DecodePixels(const std::vector<uint8_t> &codestream, size_t expected_width,
                  size_t expected_height, std::vector<float> *pixels) {
  JxlDecoder *decoder = JxlDecoderCreate(nullptr);
  if (decoder == nullptr) {
    return false;
  }
  const auto destroy = [&] { JxlDecoderDestroy(decoder); };
  if (JxlDecoderSubscribeEvents(decoder,
                                JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
          JXL_DEC_SUCCESS ||
      JxlDecoderSetInput(decoder, codestream.data(), codestream.size()) !=
          JXL_DEC_SUCCESS) {
    destroy();
    return false;
  }
  JxlDecoderCloseInput(decoder);
  const JxlPixelFormat format{
      3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0,
  };
  bool saw_basic_info = false;
  bool saw_full_image = false;
  while (true) {
    const JxlDecoderStatus status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      if (JxlDecoderGetBasicInfo(decoder, &info) != JXL_DEC_SUCCESS ||
          info.xsize != expected_width || info.ysize != expected_height) {
        destroy();
        return false;
      }
      saw_basic_info = true;
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t byte_count = 0;
      if (JxlDecoderImageOutBufferSize(decoder, &format, &byte_count) !=
              JXL_DEC_SUCCESS ||
          byte_count % sizeof(float) != 0) {
        destroy();
        return false;
      }
      pixels->resize(byte_count / sizeof(float));
      if (JxlDecoderSetImageOutBuffer(decoder, &format, pixels->data(),
                                      byte_count) != JXL_DEC_SUCCESS) {
        destroy();
        return false;
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      saw_full_image = true;
    } else if (status == JXL_DEC_SUCCESS) {
      destroy();
      return saw_basic_info && saw_full_image;
    } else {
      destroy();
      return false;
    }
  }
}

bool CheckOrdinaryLibjxlEncode() {
  JxlEncoder *encoder = JxlEncoderCreate(nullptr);
  if (encoder == nullptr) {
    return false;
  }
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 4;
  info.ysize = 3;
  info.bits_per_sample = 8;
  info.num_color_channels = 3;
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
  const JxlPixelFormat format{
      3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0,
  };
  std::array<uint8_t, 4 * 3 * 3> pixels{};
  for (size_t index = 0; index < pixels.size(); ++index) {
    pixels[index] = static_cast<uint8_t>(index * 7);
  }
  JxlEncoderFrameSettings *settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
  if (settings == nullptr ||
      JxlEncoderSetBasicInfo(encoder, &info) != JXL_ENC_SUCCESS ||
      JxlEncoderSetColorEncoding(encoder, &color) != JXL_ENC_SUCCESS ||
      JxlEncoderAddImageFrame(settings, &format, pixels.data(), pixels.size()) !=
          JXL_ENC_SUCCESS) {
    JxlEncoderDestroy(encoder);
    return false;
  }
  JxlEncoderCloseInput(encoder);
  std::vector<uint8_t> output(1024);
  uint8_t *next = output.data();
  size_t available = output.size();
  while (true) {
    const JxlEncoderStatus status =
        JxlEncoderProcessOutput(encoder, &next, &available);
    if (status == JXL_ENC_NEED_MORE_OUTPUT) {
      const size_t used = static_cast<size_t>(next - output.data());
      output.resize(output.size() * 2);
      next = output.data() + used;
      available = output.size() - used;
    } else if (status == JXL_ENC_SUCCESS) {
      output.resize(static_cast<size_t>(next - output.data()));
      JxlEncoderDestroy(encoder);
      std::vector<float> decoded;
      return DecodePixels(output, info.xsize, info.ysize, &decoded);
    } else {
      JxlEncoderDestroy(encoder);
      return false;
    }
  }
}
#endif

} // namespace

int main() {
  gjxl::VarDctEncoderFrame frame;
  const gjxl::Status frame_status = MakeFrame({17, 13}, false, &frame);
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

  if (!CheckAudit(frame, available)) {
    std::cerr << "Libjxl state copy did not preserve every audited field\n";
    return 1;
  }

  gjxl::VarDctEncoderFrame stress_frame;
  const gjxl::Status stress_status =
      MakeFrame({2057, 257}, true, &stress_frame);
  if (!stress_status.ok() || !CheckStressCoverage(stress_frame) ||
      !CheckAudit(stress_frame, available)) {
    std::cerr << "The mixed edge/multi-group fixture failed: "
              << stress_status.message() << '\n';
    return 1;
  }

  if (!CheckStatus(gjxl::codestream_internal::AuditVarDctStateWithLibjxl(
                       frame, {}, nullptr),
                   gjxl::StatusCode::kInvalidArgument,
                   "Null-output state-audit request")) {
    return 1;
  }

  const gjxl::codestream_internal::LibjxlTailStateAudit audit_sentinel{
      .source = {.dimensions = 0xA5A5A5A5A5A5A5A5u},
      .copied = {.dimensions = 0x5A5A5A5A5A5A5A5Au},
  };
  gjxl::codestream_internal::LibjxlTailStateAudit failed_audit =
      audit_sentinel;
  if (!CheckStatus(gjxl::codestream_internal::AuditVarDctStateWithLibjxl(
                       frame, {.effort = 0}, &failed_audit),
                   gjxl::StatusCode::kInvalidArgument,
                   "Invalid-effort state-audit request") ||
      failed_audit != audit_sentinel) {
    std::cerr << "Invalid audit options changed caller-visible output\n";
    return 1;
  }
  const gjxl::VarDctEncoderFrame invalid_frame;
  if (!CheckStatus(gjxl::codestream_internal::AuditVarDctStateWithLibjxl(
                       invalid_frame, {}, &failed_audit),
                   gjxl::StatusCode::kInvalidArgument,
                   "Invalid-frame state-audit request") ||
      failed_audit != audit_sentinel) {
    std::cerr << "Invalid frame changed caller-visible audit output\n";
    return 1;
  }

  const std::vector<uint8_t> sentinel{0xA5, 0x5A, 0xC3};
  std::vector<uint8_t> output = sentinel;
  const gjxl::Status status =
      gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxl(frame, {},
                                                                  &output);
#if GJXL_TEST_LIBJXL_TAIL_ENABLED
  if (!CheckOrdinaryLibjxlEncode()) {
    std::cerr << "The refactored ordinary libjxl one-shot path failed\n";
    return 1;
  }
  if (!CheckStatus(status, gjxl::StatusCode::kOk, "Valid bridge request") ||
      output.empty() || output == sentinel) {
    std::cerr << "The enabled libjxl tail did not produce a codestream\n";
    return 1;
  }
  const std::vector<uint8_t> single_thread_output = output;
  const gjxl::codestream_internal::LibjxlTailProfile profile_sentinel{
      .total_nanoseconds = 0xA5A5A5A5A5A5A5A5u,
  };
  gjxl::codestream_internal::LibjxlTailProfile tail_profile = profile_sentinel;
  output.clear();
  if (!CheckStatus(
          gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxlProfiled(
              frame, {}, &output, &tail_profile),
          gjxl::StatusCode::kOk, "Profiled bridge request") ||
      output != single_thread_output ||
      tail_profile.adapter_validation_and_copy_nanoseconds == 0 ||
      tail_profile.libjxl_internal_nanoseconds == 0 ||
      tail_profile.section_writing_nanoseconds == 0 ||
      tail_profile.total_nanoseconds <
          tail_profile.adapter_validation_and_copy_nanoseconds +
              tail_profile.context_setup_nanoseconds +
              tail_profile.libjxl_internal_nanoseconds +
              tail_profile.output_copy_nanoseconds ||
      tail_profile.worker_count != 1 ||
      !tail_profile.calling_thread_participates) {
    std::cerr << "Libjxl-tail phase profiling is inconsistent\n";
    return 1;
  }
  output.clear();
  if (!CheckStatus(gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxl(
                       frame, {}, &output),
                   gjxl::StatusCode::kOk, "Repeated bridge request") ||
      output != single_thread_output) {
    std::cerr << "Repeated libjxl-tail output was not deterministic\n";
    return 1;
  }
  output.clear();
  if (!CheckStatus(gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxl(
                       frame, {.thread_count = 8}, &output),
                   gjxl::StatusCode::kOk, "Eight-thread bridge request") ||
      output != single_thread_output) {
    std::cerr << "Libjxl-tail output changed with the thread count\n";
    return 1;
  }

  std::vector<uint8_t> native_output;
  if (!gjxl::EncodeVarDctCodestream(frame, &native_output).ok()) {
    std::cerr << "Could not produce the same-frame native codestream\n";
    return 1;
  }
  std::vector<float> libjxl_pixels;
  std::vector<float> native_pixels;
  const gjxl::Extent2D frame_extent = frame.geometry().frame();
  if (!DecodePixels(single_thread_output, frame_extent.width,
                    frame_extent.height, &libjxl_pixels) ||
      !DecodePixels(native_output, frame_extent.width, frame_extent.height,
                    &native_pixels) ||
      libjxl_pixels != native_pixels) {
    std::cerr << "Same-frame tails did not decode to identical pixels\n";
    return 1;
  }

  std::vector<uint8_t> stress_single_thread;
  std::vector<uint8_t> stress_eight_threads;
  std::vector<uint8_t> stress_native;
  if (!gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxl(
           stress_frame, {}, &stress_single_thread)
           .ok() ||
      !gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxl(
           stress_frame, {.thread_count = 8}, &stress_eight_threads)
           .ok() ||
      stress_single_thread != stress_eight_threads ||
      !gjxl::EncodeVarDctCodestream(stress_frame, &stress_native).ok()) {
    std::cerr << "The mixed multi-group fixture did not encode deterministically\n";
    return 1;
  }
  const gjxl::Extent2D stress_extent = stress_frame.geometry().frame();
  libjxl_pixels.clear();
  native_pixels.clear();
  if (!DecodePixels(stress_single_thread, stress_extent.width,
                    stress_extent.height, &libjxl_pixels) ||
      !DecodePixels(stress_native, stress_extent.width, stress_extent.height,
                    &native_pixels) ||
      libjxl_pixels != native_pixels) {
    std::cerr << "Mixed same-frame tails did not decode to identical pixels\n";
    return 1;
  }
#else
  const gjxl::codestream_internal::LibjxlTailProfile profile_sentinel{
      .total_nanoseconds = 0xA5A5A5A5A5A5A5A5u,
  };
  gjxl::codestream_internal::LibjxlTailProfile tail_profile = profile_sentinel;
  std::vector<uint8_t> profiled_output = sentinel;
  if (!CheckStatus(
          gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxlProfiled(
              frame, {}, &profiled_output, &tail_profile),
          gjxl::StatusCode::kUnavailable, "Profiled bridge request") ||
      profiled_output != sentinel || tail_profile != profile_sentinel) {
    std::cerr << "Unavailable profiled request changed caller-visible output\n";
    return 1;
  }
  if (!CheckStatus(status, gjxl::StatusCode::kUnavailable,
                   "Valid bridge request") ||
      output != sentinel) {
    std::cerr << "A failed bridge request changed caller-visible output\n";
    return 1;
  }
#endif

  if (!CheckStatus(
          gjxl::codestream_internal::EncodeVarDctCodestreamWithLibjxlProfiled(
              frame, {}, &output, nullptr),
          gjxl::StatusCode::kInvalidArgument,
          "Null-profile bridge request")) {
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
