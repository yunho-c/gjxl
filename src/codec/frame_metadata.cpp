// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/frame_metadata.h"

namespace gjxl::codec_internal {
Status ValidateImageMetadata(const ImageMetadata &m) {
  if (m.extent.empty() || m.extent.width > kMaximumImageDimension ||
      m.extent.height > kMaximumImageDimension)
    return Status::InvalidArgument("Image metadata dimensions are invalid");
  if (m.sample_format == SampleFormat::kFloat && m.bits == 32 &&
      m.color == SourceColor::kLinearSrgb && m.xyb && !m.alpha)
    return Status::Ok();
  if (m.sample_format != SampleFormat::kUnsigned || (m.bits != 8 && m.bits != 16) ||
      (m.color != SourceColor::kSrgb && m.color != SourceColor::kGraySrgb) || m.xyb)
    return Status::Unsupported("Unsupported image sample or color metadata");
  if (m.alpha && (m.color != SourceColor::kSrgb || m.alpha->bits != m.bits ||
                  m.alpha->associated || m.alpha->dimension_shift != 0))
    return Status::Unsupported("Unsupported alpha metadata");
  return Status::Ok();
}
Status ValidateFrameMetadata(const ImageMetadata &m, const FrameMetadata &f) {
  if (auto s = ValidateImageMetadata(m); !s.ok())
    return s;
  if (f.encoding == FrameEncoding::kModular &&
      m.sample_format == SampleFormat::kUnsigned && f.vardct == VarDctFrameFields{})
    return Status::Ok();
  if (f.encoding == FrameEncoding::kVarDct && m.sample_format == SampleFormat::kFloat &&
      f.vardct.x_qm_scale < 8 && f.vardct.b_qm_scale < 8)
    return Status::Ok();
  return Status::Unsupported("Unsupported frame metadata combination");
}
} // namespace gjxl::codec_internal
