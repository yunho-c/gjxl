// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <atomic>
#include <memory>

#include "codec/butteraugli.h"
#include "core/geometry.h"
#include "core/image.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl {

struct DeviceButteraugliPrepareDescriptor {
  ConstDeviceImage3View reference_linear_rgb;
  ButteraugliOptions options;
};

struct DeviceButteraugliComparisonDescriptor {
  ConstDeviceImage3View distorted_linear_rgb;
  DevicePlaneView distance_map;
  DevicePlaneView score;
};

/// Validates the device-resident reference image and immutable options.
[[nodiscard]] Status ValidateDeviceButteraugliPrepareDescriptor(
  const GpuBackend& backend,
  const DeviceButteraugliPrepareDescriptor& descriptor);

/// Validates one comparison and rejects every output/input overlap.
[[nodiscard]] Status ValidateDeviceButteraugliComparisonDescriptor(
  const GpuBackend& backend,
  ConstDeviceImage3View reference_linear_rgb,
  Extent2D prepared_extent,
  const DeviceButteraugliComparisonDescriptor& descriptor);

/// Backend-neutral prepared operation contract. Compare is synchronous: one
/// implementation-owned submission has completed when it returns success.
/// It does not read either device output back to the host. The bound backend
/// and reference buffers must outlive this object. One object is non-reentrant;
/// independent prepared objects may execute concurrently.
class PreparedDeviceButteraugli {
public:
  virtual ~PreparedDeviceButteraugli() = default;

  PreparedDeviceButteraugli(const PreparedDeviceButteraugli&) = delete;
  PreparedDeviceButteraugli& operator=(const PreparedDeviceButteraugli&) =
    delete;

  [[nodiscard]] BackendId backend_id() const noexcept;
  [[nodiscard]] Extent2D extent() const noexcept;
  [[nodiscard]] ButteraugliOptions options() const noexcept;
  [[nodiscard]] bool valid() const noexcept;

  /// Validates and executes one comparison. Reference, distorted, and output
  /// buffers must remain alive until this call returns. The output buffers for
  /// the most recent successful comparison must additionally remain alive
  /// through any requested readback.
  [[nodiscard]] Status Compare(
    const DeviceButteraugliComparisonDescriptor& descriptor);

  /// Reads and atomically commits the most recent device score.
  [[nodiscard]] Status ReadScore(double* score);

  /// Optionally reads and atomically commits the most recent full distance map.
  [[nodiscard]] Status ReadDistanceMap(PlaneF32View distance_map);

protected:
  PreparedDeviceButteraugli(
    GpuBackend& backend,
    DeviceButteraugliPrepareDescriptor descriptor);

  [[nodiscard]] GpuBackend& backend() const noexcept;
  [[nodiscard]] ConstDeviceImage3View reference_linear_rgb() const noexcept;

private:
  [[nodiscard]] virtual Status CompareValidated(
    const DeviceButteraugliComparisonDescriptor& descriptor) = 0;

  [[nodiscard]] bool BeginCall() noexcept;
  void EndCall() noexcept;
  void Invalidate() noexcept;

  GpuBackend* backend_ = nullptr;
  ConstDeviceImage3View reference_linear_rgb_;
  Extent2D extent_;
  ButteraugliOptions options_;
  DevicePlaneView last_distance_map_;
  DevicePlaneView last_score_;
  bool has_result_ = false;
  std::atomic<bool> valid_{true};
  std::atomic_flag active_ = ATOMIC_FLAG_INIT;
};

/// Factory boundary implemented by a concrete device backend. Preparation
/// clears output on failure and returns no partially initialized state.
class DeviceButteraugliOperation {
public:
  virtual ~DeviceButteraugliOperation() = default;

  DeviceButteraugliOperation(const DeviceButteraugliOperation&) = delete;
  DeviceButteraugliOperation& operator=(const DeviceButteraugliOperation&) =
    delete;

  [[nodiscard]] virtual Status Prepare(
    GpuBackend& backend,
    const DeviceButteraugliPrepareDescriptor& descriptor,
    std::unique_ptr<PreparedDeviceButteraugli>* prepared) = 0;

protected:
  DeviceButteraugliOperation() = default;
};

}  // namespace gjxl
