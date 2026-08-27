// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "codec/butteraugli.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_butteraugli_test.h"
#include "gpu/ops/butteraugli.h"
#include "gpu_test_utils.h"

namespace {

constexpr float kHostPoison = -12345.0f;
constexpr float kMetalTolerance = 1.5e-3f;

[[nodiscard]] bool CheckStatus(
  const gjxl::Status& status,
  std::string_view context) {

  if (status.ok()) return true;
  std::cerr << context << ": " << status.message() << '\n';
  return false;
}

[[nodiscard]] bool IsCode(
  const gjxl::Status& status,
  gjxl::StatusCode code) {

  return status.code() == code;
}

struct GuardedImage3 {
  [[nodiscard]] bool Prepare(
    gjxl::GpuBackend& backend,
    gjxl::Extent2D extent,
    size_t row_stride) {

    extent_ = extent;
    for (auto& channel : plane_) {
      if (!channel.Prepare(backend, extent, row_stride).ok()) {
        return false;
      }
    }
    return true;
  }

  void Fill(float base) {
    for (size_t channel = 0; channel < plane_.size(); ++channel) {
      std::vector<float> values;
      values.reserve(extent_.width * extent_.height);
      for (size_t y = 0; y < extent_.height; ++y) {
        for (size_t x = 0; x < extent_.width; ++x) {
          values.push_back(
            base + 0.1f * static_cast<float>(channel) +
            0.003f * static_cast<float>(x + 5 * y));
        }
      }
      plane_[channel].SetLogical(values);
    }
  }

  void FillAll(float value) {
    std::vector<float> values(extent_.width * extent_.height, value);
    for (auto& channel : plane_) channel.SetLogical(values);
  }

  [[nodiscard]] bool Upload() {
    for (auto& channel : plane_) {
      if (!channel.Upload().ok()) return false;
    }
    return true;
  }

  [[nodiscard]] bool DownloadAndCheckGuards() {
    for (auto& channel : plane_) {
      if (!channel.Download().ok() || !channel.GuardsIntact()) return false;
    }
    return true;
  }

  [[nodiscard]] gjxl::ConstDeviceImage3View ConstView() const noexcept {
    return {{{plane_[0].ConstView(), plane_[1].ConstView(),
              plane_[2].ConstView()}}};
  }

  [[nodiscard]] gjxl::DevicePlaneView MutablePlane(size_t channel) noexcept {
    return plane_[channel].View();
  }

  [[nodiscard]] std::vector<float> Logical(size_t channel) const {
    return plane_[channel].Logical();
  }

private:
  gjxl::Extent2D extent_;
  std::array<gjxl::test::GuardedDevicePlane, 3> plane_;
};

class FakeBuffer final : public gjxl::DeviceBuffer {
public:
  FakeBuffer(gjxl::BackendId backend_id, size_t size_bytes)
    : DeviceBuffer(gjxl::BackendKind::kCuda, backend_id, size_bytes) {}
};

class FakeBackendBase : public gjxl::GpuBackend {
public:
  [[nodiscard]] gjxl::BackendKind kind() const noexcept override {
    return gjxl::BackendKind::kCuda;
  }

  [[nodiscard]] std::string_view name() const noexcept override {
    return "Butteraugli contract fake backend";
  }

  void set_fail_readback(bool fail) noexcept {
    fail_readback_ = fail;
  }

  gjxl::Status Allocate(
    size_t size_bytes,
    std::unique_ptr<gjxl::DeviceBuffer>* out) override {

    if (out == nullptr || size_bytes == 0) {
      return gjxl::Status::InvalidArgument("Invalid fake allocation");
    }
    out->reset();
    out->reset(new FakeBuffer(id(), size_bytes));
    RecordSuccessfulAllocation();
    return gjxl::Status::Ok();
  }

  gjxl::Status CopyHostToDevice(
    gjxl::DeviceBuffer&, const void*, size_t, size_t) override {
    return gjxl::Status::Ok();
  }

  gjxl::Status CopyDeviceToHost(
    const gjxl::DeviceBuffer&, void*, size_t, size_t) override {
    if (fail_readback_) {
      return gjxl::Status::DeviceError("Injected fake readback failure");
    }
    return gjxl::Status::Ok();
  }

  gjxl::Status ForwardTransform(
    const gjxl::TransformBatch&,
    std::unique_ptr<gjxl::GpuSubmission>* submission) override {
    if (submission != nullptr) submission->reset();
    return gjxl::Status::Unavailable("Fake transform unavailable");
  }

  gjxl::Status InverseTransform(
    const gjxl::TransformBatch&,
    std::unique_ptr<gjxl::GpuSubmission>* submission) override {
    if (submission != nullptr) submission->reset();
    return gjxl::Status::Unavailable("Fake transform unavailable");
  }

private:
  bool fail_readback_ = false;
};

class PlainFakeBackend final : public FakeBackendBase {};

class OutOfMemoryFakeBackend final
    : public FakeBackendBase,
      public gjxl::DeviceButteraugliOperation {
public:
  gjxl::Status Prepare(
    gjxl::GpuBackend&,
    const gjxl::DeviceButteraugliPrepareDescriptor&,
    std::unique_ptr<gjxl::PreparedDeviceButteraugli>* prepared) override {

    if (prepared != nullptr) prepared->reset();
    return gjxl::Status::OutOfMemory("Injected preparation failure");
  }
};

struct FakeImage {
  [[nodiscard]] bool Prepare(
    gjxl::GpuBackend& backend,
    gjxl::Extent2D extent,
    size_t stride) {

    for (auto& buffer : buffer_) {
      if (!backend.Allocate(
            stride * extent.height * sizeof(float), &buffer).ok()) {
        return false;
      }
    }
    extent_ = extent;
    stride_ = stride;
    return true;
  }

  [[nodiscard]] gjxl::ConstDeviceImage3View View() const noexcept {
    return {{{
      {buffer_[0].get(), 0, gjxl::DeviceElementType::kF32, extent_, stride_},
      {buffer_[1].get(), 0, gjxl::DeviceElementType::kF32, extent_, stride_},
      {buffer_[2].get(), 0, gjxl::DeviceElementType::kF32, extent_, stride_},
    }}};
  }

private:
  std::array<std::unique_ptr<gjxl::DeviceBuffer>, 3> buffer_;
  gjxl::Extent2D extent_;
  size_t stride_ = 0;
};

class BlockingPrepared final : public gjxl::PreparedDeviceButteraugli {
public:
  BlockingPrepared(
    gjxl::GpuBackend& backend,
    gjxl::DeviceButteraugliPrepareDescriptor descriptor)
    : PreparedDeviceButteraugli(backend, descriptor) {}

  void WaitUntilEntered() {
    std::unique_lock lock(mutex_);
    entered_cv_.wait(lock, [this] { return entered_; });
  }

  void Release() {
    {
      std::lock_guard lock(mutex_);
      released_ = true;
    }
    released_cv_.notify_all();
  }

private:
  [[nodiscard]] gjxl::Status CompareValidated(
    const gjxl::DeviceButteraugliComparisonDescriptor&) override {

    std::unique_lock lock(mutex_);
    entered_ = true;
    entered_cv_.notify_all();
    released_cv_.wait(lock, [this] { return released_; });
    return gjxl::Status::Ok();
  }

  std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable released_cv_;
  bool entered_ = false;
  bool released_ = false;
};

class ImmediatePrepared final : public gjxl::PreparedDeviceButteraugli {
public:
  ImmediatePrepared(
    gjxl::GpuBackend& backend,
    gjxl::DeviceButteraugliPrepareDescriptor descriptor)
    : PreparedDeviceButteraugli(backend, descriptor) {}

private:
  [[nodiscard]] gjxl::Status CompareValidated(
    const gjxl::DeviceButteraugliComparisonDescriptor&) override {
    return gjxl::Status::Ok();
  }
};

[[nodiscard]] bool PrepareMetalCase(
  const gjxl::MetalBackendOptions& options,
  std::unique_ptr<gjxl::GpuBackend>* backend,
  GuardedImage3* reference,
  GuardedImage3* distorted,
  gjxl::test::GuardedDevicePlane* distance_map,
  gjxl::test::GuardedDevicePlane* score) {

  constexpr gjxl::Extent2D kExtent{17, 11};
  if (!gjxl::CreateMetalBackend(
        GJXL_METALLIB_PATH, options, backend).ok() ||
      !reference->Prepare(**backend, kExtent, 23) ||
      !distorted->Prepare(**backend, kExtent, 25) ||
      !distance_map->Prepare(**backend, kExtent, 27).ok() ||
      !score->Prepare(**backend, {1, 1}, 3).ok()) {
    return false;
  }
  reference->Fill(0.2f);
  distorted->Fill(0.4f);
  distance_map->PoisonLogical();
  score->PoisonLogical();
  return reference->Upload() && distorted->Upload() &&
         distance_map->Upload().ok() && score->Upload().ok();
}

[[nodiscard]] gjxl::DeviceButteraugliComparisonDescriptor MakeComparison(
  GuardedImage3& distorted,
  gjxl::test::GuardedDevicePlane& distance_map,
  gjxl::test::GuardedDevicePlane& score) {

  return {distorted.ConstView(), distance_map.View(), score.View()};
}

[[nodiscard]] bool ComputeExpected(
  const GuardedImage3& reference,
  const GuardedImage3& distorted,
  gjxl::Extent2D extent,
  gjxl::ButteraugliOptions options,
  std::vector<float>* distance_map,
  double* score) {

  std::array<std::vector<float>, 3> reference_values;
  std::array<std::vector<float>, 3> distorted_values;
  for (size_t channel = 0; channel < 3; ++channel) {
    reference_values[channel] = reference.Logical(channel);
    distorted_values[channel] = distorted.Logical(channel);
  }
  const gjxl::ConstImage3FView reference_view{{{
    {reference_values[0].data(), extent, extent.width},
    {reference_values[1].data(), extent, extent.width},
    {reference_values[2].data(), extent, extent.width},
  }}};
  const gjxl::ConstImage3FView distorted_view{{{
    {distorted_values[0].data(), extent, extent.width},
    {distorted_values[1].data(), extent, extent.width},
    {distorted_values[2].data(), extent, extent.width},
  }}};
  distance_map->assign(extent.width * extent.height, kHostPoison);
  return gjxl::ComputeButteraugliDistance(
    reference_view, distorted_view, options,
    {distance_map->data(), extent, extent.width}, score).ok();
}

[[nodiscard]] bool CheckValidRepeatedAndReadback() {
  std::unique_ptr<gjxl::GpuBackend> backend;
  GuardedImage3 reference;
  GuardedImage3 distorted;
  gjxl::test::GuardedDevicePlane distance_map;
  gjxl::test::GuardedDevicePlane score;
  if (!PrepareMetalCase({}, &backend, &reference, &distorted,
                        &distance_map, &score)) {
    return false;
  }

  const gjxl::ButteraugliOptions options{
    .hf_asymmetry = 1.25f,
    .x_multiplier = 0.75f,
    .intensity_target = 120.0f,
  };
  const gjxl::DeviceButteraugliPrepareDescriptor prepare{
    reference.ConstView(), options};
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  if (!CheckStatus(
        gjxl::PrepareDeviceButteraugli(*backend, prepare, &prepared),
        "Metal preparation")) {
    return false;
  }
  if (prepared == nullptr || !prepared->valid() ||
      prepared->backend_id() != backend->id() ||
      prepared->extent() != gjxl::Extent2D{17, 11} ||
      prepared->options().hf_asymmetry != options.hf_asymmetry) {
    return false;
  }

  double unread_score = -9.0;
  if (!IsCode(prepared->ReadScore(&unread_score),
              gjxl::StatusCode::kInvalidArgument) ||
      unread_score != -9.0 || !prepared->valid()) {
    return false;
  }

  const gjxl::GpuBackendStats before = backend->stats();
  const auto comparison = MakeComparison(distorted, distance_map, score);
  for (size_t iteration = 0; iteration < 3; ++iteration) {
    if (!CheckStatus(prepared->Compare(comparison),
                     "repeated Metal comparison")) {
      return false;
    }
  }
  const gjxl::GpuBackendStats after = backend->stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 3) {
    std::cerr << "Metal comparison allocated or submitted incorrectly\n";
    return false;
  }

  std::vector<float> expected;
  double expected_score = -1.0;
  if (!ComputeExpected(
        reference, distorted, {17, 11}, options, &expected,
        &expected_score)) {
    return false;
  }
  double actual_score = -1.0;
  if (!CheckStatus(prepared->ReadScore(&actual_score), "score readback") ||
      std::abs(actual_score - expected_score) > kMetalTolerance ||
      !IsCode(prepared->ReadScore(nullptr),
              gjxl::StatusCode::kInvalidArgument) ||
      !prepared->valid()) {
    return false;
  }

  constexpr size_t kHostStride = 22;
  std::vector<float> host_map(kHostStride * 11, kHostPoison);
  gjxl::PlaneF32View wrong_map{
    host_map.data(), {16, 11}, kHostStride};
  if (!IsCode(prepared->ReadDistanceMap(wrong_map),
              gjxl::StatusCode::kInvalidArgument) ||
      !prepared->valid()) {
    return false;
  }
  const gjxl::PlaneF32View map_view{
    host_map.data(), {17, 11}, kHostStride};
  if (!CheckStatus(prepared->ReadDistanceMap(map_view),
                   "distance-map readback")) {
    return false;
  }
  size_t index = 0;
  for (size_t y = 0; y < 11; ++y) {
    for (size_t x = 0; x < 17; ++x) {
      if (std::abs(host_map[y * kHostStride + x] - expected[index++]) >
          kMetalTolerance) {
        return false;
      }
    }
    for (size_t x = 17; x < kHostStride; ++x) {
      if (host_map[y * kHostStride + x] != kHostPoison) return false;
    }
  }

  const gjxl::DeviceButteraugliComparisonDescriptor identity{
    reference.ConstView(), distance_map.View(), score.View()};
  if (!CheckStatus(prepared->Compare(identity),
                   "exact input alias comparison") ||
      backend->stats().successful_allocations !=
        after.successful_allocations ||
      backend->stats().committed_submissions !=
        after.committed_submissions + 1 ||
      !reference.DownloadAndCheckGuards() ||
      !distorted.DownloadAndCheckGuards() ||
      !distance_map.Download().ok() || !distance_map.GuardsIntact() ||
      !score.Download().ok() || !score.GuardsIntact()) {
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckDescriptorValidation() {
  std::unique_ptr<gjxl::GpuBackend> backend;
  GuardedImage3 reference;
  GuardedImage3 distorted;
  gjxl::test::GuardedDevicePlane distance_map;
  gjxl::test::GuardedDevicePlane score;
  if (!PrepareMetalCase({}, &backend, &reference, &distorted,
                        &distance_map, &score)) {
    return false;
  }
  const gjxl::DeviceButteraugliPrepareDescriptor prepare{
    reference.ConstView(), {}};
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  if (!gjxl::PrepareDeviceButteraugli(
        *backend, prepare, &prepared).ok()) return false;
  const auto valid = MakeComparison(distorted, distance_map, score);
  const uint64_t submissions = backend->stats().committed_submissions;

  std::vector<gjxl::DeviceButteraugliComparisonDescriptor> invalid;
  auto mismatch = valid;
  mismatch.distorted_linear_rgb.plane[2].extent.width = 16;
  invalid.push_back(mismatch);
  auto wrong_input_type = valid;
  wrong_input_type.distorted_linear_rgb.plane[0].element_type =
    gjxl::DeviceElementType::kI32;
  invalid.push_back(wrong_input_type);
  auto wrong_map_extent = valid;
  wrong_map_extent.distance_map.extent.width = 16;
  invalid.push_back(wrong_map_extent);
  auto wrong_map_type = valid;
  wrong_map_type.distance_map.element_type = gjxl::DeviceElementType::kI32;
  invalid.push_back(wrong_map_type);
  auto short_map_stride = valid;
  short_map_stride.distance_map.row_stride = 16;
  invalid.push_back(short_map_stride);
  auto wrong_score_extent = valid;
  wrong_score_extent.score.extent = {2, 1};
  invalid.push_back(wrong_score_extent);
  auto map_overlaps_input = valid;
  map_overlaps_input.distance_map = distorted.MutablePlane(0);
  invalid.push_back(map_overlaps_input);
  auto map_overlaps_reference = valid;
  map_overlaps_reference.distance_map = reference.MutablePlane(1);
  invalid.push_back(map_overlaps_reference);
  auto score_overlaps_map = valid;
  score_overlaps_map.score = {
    valid.distance_map.buffer,
    valid.distance_map.offset_bytes,
    gjxl::DeviceElementType::kF32,
    {1, 1},
    valid.distance_map.row_stride,
  };
  invalid.push_back(score_overlaps_map);
  auto map_out_of_bounds = valid;
  map_out_of_bounds.distance_map.offset_bytes =
    map_out_of_bounds.distance_map.buffer->size_bytes();
  invalid.push_back(map_out_of_bounds);

  std::unique_ptr<gjxl::GpuBackend> other;
  GuardedImage3 foreign;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &other).ok() ||
      !foreign.Prepare(*other, {17, 11}, 23)) {
    return false;
  }
  auto foreign_input = valid;
  foreign_input.distorted_linear_rgb = foreign.ConstView();
  invalid.push_back(foreign_input);

  for (const auto& descriptor : invalid) {
    if (!IsCode(prepared->Compare(descriptor),
                gjxl::StatusCode::kInvalidArgument) ||
        !prepared->valid() ||
        backend->stats().committed_submissions != submissions) {
      std::cerr << "Invalid comparison changed prepared state\n";
      return false;
    }
  }

  std::unique_ptr<gjxl::PreparedDeviceButteraugli> output;
  if (!IsCode(gjxl::PrepareDeviceButteraugli(*backend, prepare, nullptr),
              gjxl::StatusCode::kInvalidArgument)) {
    return false;
  }
  for (gjxl::ButteraugliOptions bad_options : {
         gjxl::ButteraugliOptions{.hf_asymmetry = 0.0f},
         gjxl::ButteraugliOptions{
           .x_multiplier = std::numeric_limits<float>::infinity()},
         gjxl::ButteraugliOptions{
           .intensity_target = std::numeric_limits<float>::quiet_NaN()}}) {
    output = std::move(prepared);
    const gjxl::DeviceButteraugliPrepareDescriptor bad{
      reference.ConstView(), bad_options};
    if (!IsCode(gjxl::PrepareDeviceButteraugli(*backend, bad, &output),
                gjxl::StatusCode::kInvalidArgument) || output != nullptr) {
      return false;
    }
    if (!gjxl::PrepareDeviceButteraugli(
          *backend, prepare, &prepared).ok()) return false;
  }

  for (size_t variant = 0; variant < 2; ++variant) {
    auto bad_reference = prepare;
    if (variant == 0) {
      bad_reference.reference_linear_rgb.plane[0].element_type =
        gjxl::DeviceElementType::kI32;
    } else {
      bad_reference.reference_linear_rgb.plane[2].extent.width = 16;
    }
    output = std::move(prepared);
    if (!IsCode(gjxl::PrepareDeviceButteraugli(
                  *backend, bad_reference, &output),
                gjxl::StatusCode::kInvalidArgument) || output != nullptr) {
      return false;
    }
    if (!gjxl::PrepareDeviceButteraugli(
          *backend, prepare, &prepared).ok()) return false;
  }
  return true;
}

[[nodiscard]] bool CheckSmallStridedCase() {
  std::unique_ptr<gjxl::GpuBackend> backend;
  if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &backend).ok()) {
    return false;
  }
  GuardedImage3 reference;
  GuardedImage3 distorted;
  gjxl::test::GuardedDevicePlane distance_map;
  gjxl::test::GuardedDevicePlane score;
  if (!reference.Prepare(*backend, {1, 1}, 3) ||
      !distorted.Prepare(*backend, {1, 1}, 4) ||
      !distance_map.Prepare(*backend, {1, 1}, 2).ok() ||
      !score.Prepare(*backend, {1, 1}, 3).ok()) {
    return false;
  }
  reference.FillAll(0.25f);
  distorted.FillAll(0.75f);
  distance_map.PoisonLogical();
  score.PoisonLogical();
  if (!reference.Upload() || !distorted.Upload() ||
      !distance_map.Upload().ok() || !score.Upload().ok()) {
    return false;
  }
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  const gjxl::DeviceButteraugliPrepareDescriptor prepare{
    reference.ConstView(), {}};
  if (!gjxl::PrepareDeviceButteraugli(
        *backend, prepare, &prepared).ok() ||
      !prepared->Compare(
        MakeComparison(distorted, distance_map, score)).ok()) {
    return false;
  }
  std::vector<float> expected_map;
  double expected_score = -1.0;
  if (!ComputeExpected(
        reference, distorted, {1, 1}, {}, &expected_map,
        &expected_score)) {
    return false;
  }
  double actual_score = -1.0;
  std::array<float, 3> host_map{
    kHostPoison, kHostPoison, kHostPoison};
  return prepared->ReadScore(&actual_score).ok() &&
         std::abs(actual_score - expected_score) <= kMetalTolerance &&
         prepared->ReadDistanceMap(
           {host_map.data(), {1, 1}, host_map.size()}).ok() &&
         std::abs(host_map[0] - expected_map[0]) <= kMetalTolerance &&
         host_map[1] == kHostPoison &&
         host_map[2] == kHostPoison;
}

[[nodiscard]] bool CheckUnavailableAndAllocationFailure() {
  PlainFakeBackend plain;
  FakeImage plain_image;
  if (!plain_image.Prepare(plain, {5, 3}, 7)) return false;
  const gjxl::DeviceButteraugliPrepareDescriptor plain_prepare{
    plain_image.View(), {}};
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
  if (!IsCode(gjxl::PrepareDeviceButteraugli(
                plain, plain_prepare, &prepared),
              gjxl::StatusCode::kUnavailable) || prepared != nullptr) {
    return false;
  }

  OutOfMemoryFakeBackend failing;
  FakeImage failing_image;
  if (!failing_image.Prepare(failing, {5, 3}, 7)) return false;
  const gjxl::DeviceButteraugliPrepareDescriptor failing_prepare{
    failing_image.View(), {}};
  return IsCode(gjxl::PrepareDeviceButteraugli(
                  failing, failing_prepare, &prepared),
                gjxl::StatusCode::kOutOfMemory) &&
         prepared == nullptr;
}

[[nodiscard]] bool CheckPreparationFailures() {
  const auto check = [](gjxl::MetalBackendOptions options,
                        gjxl::StatusCode expected,
                        uint64_t expected_submissions) {
    std::unique_ptr<gjxl::GpuBackend> backend;
    GuardedImage3 reference;
    GuardedImage3 distorted;
    gjxl::test::GuardedDevicePlane distance_map;
    gjxl::test::GuardedDevicePlane score;
    if (!PrepareMetalCase(options, &backend, &reference, &distorted,
                          &distance_map, &score)) {
      return false;
    }
    std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
    const gjxl::DeviceButteraugliPrepareDescriptor prepare{
      reference.ConstView(), {}};
    const uint64_t before = backend->stats().committed_submissions;
    const gjxl::Status status = gjxl::PrepareDeviceButteraugli(
      *backend, prepare, &prepared);
    return IsCode(status, expected) && prepared == nullptr &&
           backend->stats().committed_submissions ==
             before + expected_submissions;
  };

  gjxl::MetalBackendOptions submission_failure;
  submission_failure.test_fail_submission = true;
  gjxl::MetalBackendOptions completion_failure;
  completion_failure.test_fail_completion = true;
  return check(submission_failure, gjxl::StatusCode::kSubmissionFailed, 0) &&
         check(completion_failure, gjxl::StatusCode::kDeviceError, 1);
}

[[nodiscard]] bool CheckComparisonFailures() {
  const auto check = [](bool fail_submission,
                        bool fail_completion,
                        gjxl::StatusCode expected,
                        uint64_t expected_submissions) {
    std::unique_ptr<gjxl::GpuBackend> backend;
    GuardedImage3 reference;
    GuardedImage3 distorted;
    gjxl::test::GuardedDevicePlane distance_map;
    gjxl::test::GuardedDevicePlane score;
    if (!PrepareMetalCase({}, &backend, &reference, &distorted,
                          &distance_map, &score)) {
      return false;
    }
    std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
    const gjxl::DeviceButteraugliPrepareDescriptor prepare{
      reference.ConstView(), {}};
    if (!gjxl::PrepareDeviceButteraugli(
          *backend, prepare, &prepared).ok()) return false;
    if (!gjxl::ArmNextMetalSubmissionFailureForTest(
          *backend, fail_submission, fail_completion).ok()) {
      return false;
    }
    const uint64_t before = backend->stats().committed_submissions;
    const gjxl::Status status = prepared->Compare(
      MakeComparison(distorted, distance_map, score));
    double host_score = -7.0;
    std::vector<float> host_map(17 * 11, kHostPoison);
    return IsCode(status, expected) && !prepared->valid() &&
           backend->stats().committed_submissions ==
             before + expected_submissions &&
           IsCode(prepared->ReadScore(&host_score),
                  gjxl::StatusCode::kInvalidArgument) &&
           host_score == -7.0 &&
           IsCode(prepared->ReadDistanceMap(
                    {host_map.data(), {17, 11}, 17}),
                  gjxl::StatusCode::kInvalidArgument) &&
           std::ranges::all_of(host_map, [](float value) {
             return value == kHostPoison;
           });
  };

  return check(true, false, gjxl::StatusCode::kSubmissionFailed, 0) &&
         check(false, true, gjxl::StatusCode::kDeviceError, 1);
}

[[nodiscard]] bool CheckInvalidComputedReadback() {
  const auto check = [](bool score_readback) {
    std::unique_ptr<gjxl::GpuBackend> backend;
    GuardedImage3 reference;
    GuardedImage3 distorted;
    gjxl::test::GuardedDevicePlane distance_map;
    gjxl::test::GuardedDevicePlane score;
    if (!PrepareMetalCase({}, &backend, &reference, &distorted,
                          &distance_map, &score)) {
      return false;
    }
    distorted.FillAll(std::numeric_limits<float>::quiet_NaN());
    if (!distorted.Upload()) return false;
    std::unique_ptr<gjxl::PreparedDeviceButteraugli> prepared;
    const gjxl::DeviceButteraugliPrepareDescriptor prepare{
      reference.ConstView(), {}};
    const gjxl::Status prepare_status = gjxl::PrepareDeviceButteraugli(
      *backend, prepare, &prepared);
    if (!CheckStatus(prepare_status, "non-finite preparation")) {
      return false;
    }
    const gjxl::Status compare_status = prepared->Compare(
      MakeComparison(distorted, distance_map, score));
    if (!CheckStatus(compare_status, "non-finite comparison")) {
      return false;
    }
    if (score_readback) {
      double host_score = -3.0;
      const gjxl::Status read_status = prepared->ReadScore(&host_score);
      if (!IsCode(read_status, gjxl::StatusCode::kInternal)) {
        std::cerr << "Unexpected non-finite score status: "
                  << static_cast<int>(read_status.code()) << ' '
                  << read_status.message() << '\n';
        return false;
      }
      return host_score == -3.0 && !prepared->valid();
    }
    std::vector<float> host_map(20 * 11, kHostPoison);
    const gjxl::PlaneF32View view{
      host_map.data(), {17, 11}, 20};
    const gjxl::Status read_status = prepared->ReadDistanceMap(view);
    if (!IsCode(read_status, gjxl::StatusCode::kInternal)) {
      std::cerr << "Unexpected non-finite map status: "
                << static_cast<int>(read_status.code()) << ' '
                << read_status.message() << '\n';
      return false;
    }
    return std::ranges::all_of(host_map, [](float value) {
             return value == kHostPoison;
           }) && !prepared->valid();
  };
  return check(true) && check(false);
}

[[nodiscard]] bool CheckInjectedReadbackFailure() {
  const auto check = [](bool score_readback) {
    PlainFakeBackend backend;
    FakeImage reference;
    FakeImage distorted;
    if (!reference.Prepare(backend, {5, 3}, 7) ||
        !distorted.Prepare(backend, {5, 3}, 7)) {
      return false;
    }
    std::unique_ptr<gjxl::DeviceBuffer> distance_buffer;
    std::unique_ptr<gjxl::DeviceBuffer> score_buffer;
    if (!backend.Allocate(7 * 3 * sizeof(float), &distance_buffer).ok() ||
        !backend.Allocate(3 * sizeof(float), &score_buffer).ok()) {
      return false;
    }
    const gjxl::DeviceButteraugliPrepareDescriptor prepare{
      reference.View(), {}};
    ImmediatePrepared prepared(backend, prepare);
    const gjxl::DeviceButteraugliComparisonDescriptor comparison{
      distorted.View(),
      {distance_buffer.get(), 0, gjxl::DeviceElementType::kF32,
       {5, 3}, 7},
      {score_buffer.get(), 0, gjxl::DeviceElementType::kF32,
       {1, 1}, 3},
    };
    if (!prepared.Compare(comparison).ok()) return false;
    backend.set_fail_readback(true);
    if (score_readback) {
      double host_score = -11.0;
      return IsCode(prepared.ReadScore(&host_score),
                    gjxl::StatusCode::kDeviceError) &&
             host_score == -11.0 && !prepared.valid();
    }
    std::vector<float> host_map(8 * 3, kHostPoison);
    return IsCode(prepared.ReadDistanceMap(
                    {host_map.data(), {5, 3}, 8}),
                  gjxl::StatusCode::kDeviceError) &&
           std::ranges::all_of(host_map, [](float value) {
             return value == kHostPoison;
           }) &&
           !prepared.valid();
  };
  return check(true) && check(false);
}

[[nodiscard]] bool CheckPreparedConcurrency() {
  std::unique_ptr<gjxl::GpuBackend> backend;
  GuardedImage3 reference;
  GuardedImage3 distorted;
  gjxl::test::GuardedDevicePlane distance_map_a;
  gjxl::test::GuardedDevicePlane score_a;
  if (!PrepareMetalCase({}, &backend, &reference, &distorted,
                        &distance_map_a, &score_a)) {
    return false;
  }
  gjxl::test::GuardedDevicePlane distance_map_b;
  gjxl::test::GuardedDevicePlane score_b;
  if (!distance_map_b.Prepare(*backend, {17, 11}, 29).ok() ||
      !score_b.Prepare(*backend, {1, 1}, 5).ok()) {
    return false;
  }
  const gjxl::DeviceButteraugliPrepareDescriptor prepare{
    reference.ConstView(), {}};
  BlockingPrepared first(*backend, prepare);
  gjxl::Status first_status;
  std::thread first_thread([&] {
    first_status = first.Compare(
      MakeComparison(distorted, distance_map_a, score_a));
  });
  first.WaitUntilEntered();
  const gjxl::Status overlapping = first.Compare(
    MakeComparison(distorted, distance_map_a, score_a));
  first.Release();
  first_thread.join();
  if (!first_status.ok() ||
      !IsCode(overlapping, gjxl::StatusCode::kInvalidArgument) ||
      !first.valid()) {
    return false;
  }

  BlockingPrepared independent_a(*backend, prepare);
  BlockingPrepared independent_b(*backend, prepare);
  gjxl::Status status_a;
  gjxl::Status status_b;
  std::thread thread_a([&] {
    status_a = independent_a.Compare(
      MakeComparison(distorted, distance_map_a, score_a));
  });
  std::thread thread_b([&] {
    status_b = independent_b.Compare(
      MakeComparison(distorted, distance_map_b, score_b));
  });
  independent_a.WaitUntilEntered();
  independent_b.WaitUntilEntered();
  independent_a.Release();
  independent_b.Release();
  thread_a.join();
  thread_b.join();
  if (!status_a.ok() || !status_b.ok() ||
      !independent_a.valid() || !independent_b.valid()) {
    return false;
  }

  std::unique_ptr<gjxl::PreparedDeviceButteraugli> metal_a;
  std::unique_ptr<gjxl::PreparedDeviceButteraugli> metal_b;
  if (!gjxl::PrepareDeviceButteraugli(
        *backend, prepare, &metal_a).ok() ||
      !gjxl::PrepareDeviceButteraugli(
        *backend, prepare, &metal_b).ok()) {
    return false;
  }
  gjxl::Status metal_status_a;
  gjxl::Status metal_status_b;
  std::thread metal_thread_a([&] {
    metal_status_a = metal_a->Compare(
      MakeComparison(distorted, distance_map_a, score_a));
  });
  std::thread metal_thread_b([&] {
    metal_status_b = metal_b->Compare(
      MakeComparison(distorted, distance_map_b, score_b));
  });
  metal_thread_a.join();
  metal_thread_b.join();
  return metal_status_a.ok() && metal_status_b.ok() &&
         metal_a->valid() && metal_b->valid();
}

}  // namespace

int main() {
  const std::array checks{
    std::pair{"valid repeated comparison", &CheckValidRepeatedAndReadback},
    std::pair{"descriptor validation", &CheckDescriptorValidation},
    std::pair{"small strided comparison", &CheckSmallStridedCase},
    std::pair{"availability and allocation failure",
              &CheckUnavailableAndAllocationFailure},
    std::pair{"preparation failures", &CheckPreparationFailures},
    std::pair{"comparison failures", &CheckComparisonFailures},
    std::pair{"invalid computed readback", &CheckInvalidComputedReadback},
    std::pair{"injected readback failure", &CheckInjectedReadbackFailure},
    std::pair{"prepared concurrency", &CheckPreparedConcurrency},
  };
  for (const auto& [name, check] : checks) {
    if (!check()) {
      std::cerr << "Device Butteraugli contract failure: " << name << '\n';
      return EXIT_FAILURE;
    }
  }
  std::cout << "All device Butteraugli operation contract tests passed.\n";
  return EXIT_SUCCESS;
}
