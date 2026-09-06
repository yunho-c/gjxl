// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/vardct_frame.h"

namespace gjxl::vardct_frame_internal {

/// Borrowed storage for the existing native group-major frame layout. Small
/// value metadata is copied; all pointers, planes, and spans borrow storage.
/// Plane backing must cover every addressed row, just like core PlaneView.
struct VarDctFrameViewData {
  VarDctFrameInput input;
  SimpleVarDctCodestreamProfile profile;
  ConstImage3I32View quantized_dc;
  ConstImage3FView dc;
  Extent2D ac_group_extent;
  std::span<const size_t> group_used_coefficient_count;
  std::span<const int32_t> ac_coefficients;
};

/// Read-only, non-owning completed frame. No allocation or materialization.
///
/// The caller must keep every backing object alive, at a stable address, and
/// immutable until all consumers return (including their parallel workers).
/// For device-backed storage, the producer must complete before borrowing;
/// its output lease must prevent reuse, purging, and writes throughout the
/// borrow. This view neither waits for completion nor retains a device lease.
/// Copying a view does not extend the backing lifetime. Consumers do not retain
/// the view after returning. Temporary AQ scratch need not share that lifetime.
///
/// Accessors require a valid view, except valid() and checked GetAcGroup().
/// As with other core views, valid() cannot detect dangling backing pointers.
class VarDctFrameView {
 public:
  VarDctFrameView() = default;
  explicit VarDctFrameView(VarDctFrameViewData data) noexcept : data_(data) {}

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const FrameGeometry& geometry() const noexcept {
    return data_.input.geometry;
  }
  [[nodiscard]] const AcStrategyGrid& strategies() const noexcept {
    return *data_.input.strategies;
  }
  [[nodiscard]] ConstPlaneI32View raw_quant_field() const noexcept {
    return data_.input.raw_quant_field;
  }
  [[nodiscard]] const Quantizer& quantizer() const noexcept {
    return *data_.input.quantizer;
  }
  [[nodiscard]] const ColorCorrelationMap& color_correlation() const noexcept {
    return *data_.input.color_correlation;
  }
  [[nodiscard]] ConstPlaneU8View epf_sharpness() const noexcept {
    return data_.input.epf_sharpness;
  }
  [[nodiscard]] const SimpleVarDctCodestreamProfile& profile() const noexcept {
    return data_.profile;
  }
  [[nodiscard]] ConstImage3I32View quantized_dc() const noexcept {
    return data_.quantized_dc;
  }
  [[nodiscard]] ConstImage3FView dc() const noexcept { return data_.dc; }
  [[nodiscard]] Extent2D ac_group_extent() const noexcept {
    return data_.ac_group_extent;
  }
  [[nodiscard]] size_t ac_group_count() const noexcept {
    return data_.group_used_coefficient_count.size();
  }
  [[nodiscard]] Status GetAcGroup(size_t index, VarDctAcGroupView* out) const;

 private:
  VarDctFrameViewData data_;
};

/// Exclusive completed-output lease, independent of a producer's scratch.
/// Destruction releases the backing only after all synchronous readers finish.
/// Implementations publish a view only after device completion and must not
/// retain a prepared evaluator or return live storage to a reuse/purge pool.
class CompletedVarDctFrame {
 public:
  virtual ~CompletedVarDctFrame() = default;
  [[nodiscard]] virtual VarDctFrameView view() const noexcept = 0;
};

/// Borrows without validating coefficient values; an invalid owner yields an
/// invalid view. Destroying, assigning, or moving the owner ends the borrow.
[[nodiscard]] VarDctFrameView BorrowFrame(
    const VarDctEncoderFrame& frame) noexcept;
VarDctFrameView BorrowFrame(VarDctEncoderFrame&&) = delete;
VarDctFrameView BorrowFrame(const VarDctEncoderFrame&&) = delete;

/// Same structural and initial-profile gates as the owned public API.
[[nodiscard]] Status ValidateSimpleCodestreamFrame(
    const VarDctFrameView& frame);

}  // namespace gjxl::vardct_frame_internal
