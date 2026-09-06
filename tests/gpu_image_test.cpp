// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "gpu/backend.h"
#include "gpu/image.h"
#include "gpu/ops/primitives.h"
#include "gpu/scratch.h"

namespace {

class FakeBuffer final : public gjxl::DeviceBuffer {
public:
  FakeBuffer(gjxl::BackendId backend_id, size_t size_bytes)
    : DeviceBuffer(gjxl::BackendKind::kCuda, backend_id, size_bytes) {}
};

class FakeBackend final : public gjxl::GpuBackend {
public:
  [[nodiscard]] gjxl::BackendKind kind() const noexcept override {
    return gjxl::BackendKind::kCuda;
  }

  [[nodiscard]] std::string_view name() const noexcept override {
    return "range-test backend";
  }

  gjxl::Status Allocate(
    size_t size_bytes,
    std::unique_ptr<gjxl::DeviceBuffer>* out) override {

    if (out == nullptr || size_bytes == 0) {
      return gjxl::Status::InvalidArgument("Invalid fake allocation");
    }
    out->reset(new FakeBuffer(id(), size_bytes));
    RecordSuccessfulAllocation();
    return gjxl::Status::Ok();
  }

  gjxl::Status CopyHostToDevice(
    gjxl::DeviceBuffer&,
    const void*,
    size_t,
    size_t) override {
    return gjxl::Status::Ok();
  }

  gjxl::Status CopyDeviceToHost(
    const gjxl::DeviceBuffer&,
    void*,
    size_t,
    size_t) override {
    return gjxl::Status::Ok();
  }

  gjxl::Status ForwardTransform(
    const gjxl::TransformBatch&,
    std::unique_ptr<gjxl::GpuSubmission>* submission) override {
    if (submission != nullptr) submission->reset();
    return gjxl::Status::Unavailable("Not implemented by range-test backend");
  }

  gjxl::Status InverseTransform(
    const gjxl::TransformBatch&,
    std::unique_ptr<gjxl::GpuSubmission>* submission) override {
    if (submission != nullptr) submission->reset();
    return gjxl::Status::Unavailable("Not implemented by range-test backend");
  }
};

bool IsInvalid(const gjxl::Status& status) {
  return status.code() == gjxl::StatusCode::kInvalidArgument;
}

bool CheckPlaneRanges() {
  FakeBackend backend;
  FakeBackend other;
  std::unique_ptr<gjxl::DeviceBuffer> buffer;
  if (!backend.Allocate(512, &buffer).ok()) return false;

  const gjxl::ConstDevicePlaneView valid{
    buffer.get(), 16, gjxl::DeviceElementType::kF32, {7, 5}, 11};
  gjxl::DeviceMemoryRange range;
  if (!gjxl::ComputeDevicePlaneRange(valid, backend.id(), &range).ok() ||
      range.offset_bytes != 16 || range.size_bytes != 51 * sizeof(float)) {
    std::cerr << "Valid device plane range was computed incorrectly\n";
    return false;
  }

  for (gjxl::DeviceElementType type : {
         gjxl::DeviceElementType::kF32,
         gjxl::DeviceElementType::kI32,
         gjxl::DeviceElementType::kI8,
         gjxl::DeviceElementType::kU8}) {
    const gjxl::ConstDevicePlaneView typed{
      buffer.get(), 8, type, {3, 2}, 5};
    if (!gjxl::ComputeDevicePlaneRange(
          typed, backend.id(), &range).ok()) {
      std::cerr << "A supported device element type was rejected\n";
      return false;
    }
  }

  std::array<gjxl::ConstDevicePlaneView, 8> invalid{{
    {nullptr, 0, gjxl::DeviceElementType::kF32, {1, 1}, 1},
    {buffer.get(), 0, gjxl::DeviceElementType::kF32, {0, 1}, 1},
    {buffer.get(), 0, gjxl::DeviceElementType::kF32, {4, 2}, 3},
    {buffer.get(), 2, gjxl::DeviceElementType::kF32, {1, 1}, 1},
    {buffer.get(), 508, gjxl::DeviceElementType::kF32, {2, 1}, 2},
    {buffer.get(), 0, gjxl::DeviceElementType::kF32,
      {2, std::numeric_limits<size_t>::max()}, 2},
    {buffer.get(), 0, gjxl::DeviceElementType::kF32,
      {std::numeric_limits<size_t>::max(), 2},
      std::numeric_limits<size_t>::max()},
    {buffer.get(), std::numeric_limits<size_t>::max() - 3,
      gjxl::DeviceElementType::kF32, {1, 1}, 1},
  }};
  for (const auto& view : invalid) {
    if (!IsInvalid(gjxl::ComputeDevicePlaneRange(
          view, backend.id(), &range))) {
      std::cerr << "Invalid device plane range was accepted\n";
      return false;
    }
  }
  if (!IsInvalid(gjxl::ComputeDevicePlaneRange(
        valid, other.id(), &range))) {
    std::cerr << "Foreign backend ownership was accepted\n";
    return false;
  }
  return true;
}

bool CheckImageAndOverlap() {
  FakeBackend backend;
  std::unique_ptr<gjxl::DeviceBuffer> buffer;
  if (!backend.Allocate(1024, &buffer).ok()) return false;
  const gjxl::ConstDevicePlaneView plane{
    buffer.get(), 0, gjxl::DeviceElementType::kF32, {5, 3}, 8};
  const gjxl::ConstDeviceImage3View image{{{
    plane,
    {buffer.get(), 128, gjxl::DeviceElementType::kF32, {5, 3}, 8},
    {buffer.get(), 256, gjxl::DeviceElementType::kF32, {5, 3}, 8},
  }}};
  if (!gjxl::ValidateDeviceImage3View(image, backend.id()).ok()) {
    std::cerr << "Valid three-plane device image was rejected\n";
    return false;
  }
  gjxl::ConstDeviceImage3View mismatch = image;
  mismatch.plane[2].extent.width = 4;
  if (!IsInvalid(gjxl::ValidateDeviceImage3View(
        mismatch, backend.id()))) {
    std::cerr << "Mismatched device image was accepted\n";
    return false;
  }

  const gjxl::DeviceMemoryRange first{buffer.get(), 32, 64};
  const gjxl::DeviceMemoryRange same{buffer.get(), 32, 64};
  const gjxl::DeviceMemoryRange overlap{buffer.get(), 80, 32};
  const gjxl::DeviceMemoryRange separate{buffer.get(), 96, 32};
  return gjxl::DeviceRangesEqual(first, same) &&
         gjxl::DeviceRangesOverlap(first, overlap) &&
         !gjxl::DeviceRangesOverlap(first, separate);
}

bool CheckScratchArena() {
  FakeBackend backend;
  FakeBackend other;
  gjxl::DeviceScratchArena arena;
  if (!arena.Prepare(backend, 1024).ok() ||
      backend.stats().successful_allocations != 1 ||
      gjxl::QueryGpuImagePrimitives(backend) != nullptr) {
    return false;
  }
  gjxl::DevicePlaneView first;
  gjxl::DevicePlaneView second;
  if (!arena.AllocatePlane(
        gjxl::DeviceElementType::kF32, {7, 5}, 9, 64, &first).ok() ||
      !arena.AllocatePlane(
        gjxl::DeviceElementType::kU8, {13, 3}, 16, 64, &second).ok() ||
      first.offset_bytes % 64 != 0 || second.offset_bytes % 64 != 0 ||
      arena.layout_bytes() == 0 ||
      arena.peak_layout_bytes() != arena.layout_bytes()) {
    std::cerr << "Device scratch layout is invalid\n";
    return false;
  }
  const size_t peak = arena.peak_layout_bytes();
  if (!arena.Prepare(backend, 512).ok() ||
      backend.stats().successful_allocations != 1 ||
      arena.layout_bytes() != 0 || arena.peak_layout_bytes() != peak) {
    std::cerr << "Device scratch reuse allocated or lost its peak\n";
    return false;
  }
  gjxl::DevicePlaneView oversized;
  if (!IsInvalid(arena.AllocatePlane(
        gjxl::DeviceElementType::kF32, {1024, 2}, 1024, 64,
        &oversized)) ||
      !IsInvalid(arena.Prepare(other, 1024))) {
    std::cerr << "Invalid device scratch request was accepted\n";
    return false;
  }
  return true;
}

bool CheckLayoutPlanning() {
  using namespace gjxl;
  constexpr size_t maximum = std::numeric_limits<size_t>::max();
  std::mt19937_64 rng(0x37a19);
  const std::array<size_t, 11> values{0, 1, 2, 3, 4, 63, 64, 255, 256, maximum - 1, maximum};
  for (size_t iteration = 0; iteration < 20000; ++iteration) {
    const auto type = static_cast<DeviceElementType>(rng() % 5);
    const size_t element_size = static_cast<size_t>(type) < 2 ? 4 :
      static_cast<size_t>(type) < 4 ? 1 : 0;
    const auto choose = [&] { return values[rng() % values.size()]; };
    const Extent2D extent{choose(), choose()};
    const size_t stride = choose(), initial = choose(), alignment = choose();
    bool valid = element_size != 0 && extent.width != 0 && extent.height != 0 &&
      stride >= extent.width && alignment >= element_size && (alignment & (alignment - 1)) == 0;
    size_t expected_offset = 0, expected_bytes = 0, expected_end = 0;
    using Wide = __uint128_t;
    if (valid) {
      const Wide elements = (Wide{extent.height} - 1) * stride + extent.width;
      const Wide offset = ((Wide{initial} + alignment - 1) / alignment) * alignment;
      valid = elements <= maximum / element_size && offset <= maximum;
      if (valid) {
        const Wide bytes = elements * element_size;
        valid = offset + bytes <= maximum;
        expected_offset = static_cast<size_t>(offset);
        expected_bytes = static_cast<size_t>(bytes);
        expected_end = static_cast<size_t>(offset + bytes);
      }
    }
    DeviceScratchLayoutPlan plan(initial);
    DevicePlaneLayout plane{DeviceElementType::kI8, {5, 3}, 7, 9, 11};
    const auto before = plane;
    const Status status = plan.AddPlane(type, extent, stride, alignment, &plane);
    if (status.ok() != valid || (!valid && (plane != before || plan.capacity_bytes() != initial)) ||
        (valid && (plane.element_type != type || plane.extent != extent || plane.row_stride != stride ||
          plane.offset_bytes != expected_offset || plane.size_bytes != expected_bytes ||
          plan.capacity_bytes() != expected_end))) {
      std::cerr << "Scratch layout differs from 128-bit reference at " << iteration << '\n';
      return false;
    }
    if (!valid) continue;
    FakeBackend backend;
    DeviceScratchArena arena;
    DevicePlaneView view;
    if (!arena.Prepare(backend, expected_end).ok() || !arena.BindPlane(plane, &view).ok() ||
        view.offset_bytes != expected_offset || arena.layout_bytes() != expected_end) return false;
    const auto* buffer = view.buffer;
    const auto peak = arena.peak_layout_bytes();
    ++plane.size_bytes;
    if (!IsInvalid(arena.BindPlane(plane, &view)) || view.buffer != buffer ||
        view.offset_bytes != expected_offset || arena.peak_layout_bytes() != peak) return false;
  }
  DeviceScratchLayoutPlan plan;
  if (!IsInvalid(plan.AddPlane(DeviceElementType::kF32, {1, 1}, 1, 4, nullptr)) ||
      plan.capacity_bytes() != 0) return false;
  size_t size = 123;
  if (!IsInvalid(ComputeDevicePlaneSizeBytes(static_cast<DeviceElementType>(99), {1, 1}, 1, &size)) ||
      size != 123 || !IsInvalid(ComputeDevicePlaneSizeBytes(DeviceElementType::kF32, {1, 1}, 1, nullptr))) return false;
  return true;
}

}  // namespace

int main() {
  if (!CheckPlaneRanges() || !CheckImageAndOverlap() ||
      !CheckScratchArena() || !CheckLayoutPlanning()) {
    return EXIT_FAILURE;
  }
  std::cout << "All device-image and scratch tests passed.\n";
  return EXIT_SUCCESS;
}
