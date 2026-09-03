// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>

#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/cuda/cuda_kernels.h"
#include "gpu/image.h"
#include "gpu/ops/ac_strategy.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/butteraugli.h"
#include "gpu/ops/primitives.h"

namespace gjxl::cuda_internal {

[[nodiscard]] Status CudaStatus(cudaError_t error, std::string_view operation,
                                StatusCode code = StatusCode::kDeviceError);

[[nodiscard]] Status CudaRuntimeStatus(cudaError_t error,
                                       std::string_view operation);

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int ordinal);
  ~ScopedCudaDevice();

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  [[nodiscard]] cudaError_t status() const noexcept { return status_; }

 private:
  int previous_ = -1;
  cudaError_t status_ = cudaSuccess;
  bool changed_ = false;
};

struct CudaDeviceState {
  int ordinal = 0;
  cudaStream_t stream = nullptr;
  size_t maximum_grid_x = 0;
  size_t maximum_threads_per_block = 0;
  std::mutex submission_mutex;

  ~CudaDeviceState();
};

class CudaBuffer final : public DeviceBuffer {
 public:
  CudaBuffer(std::shared_ptr<CudaDeviceState> state, BackendId backend_id,
             size_t size_bytes, void* pointer);
  ~CudaBuffer() override;

  [[nodiscard]] void* pointer() noexcept { return pointer_; }
  [[nodiscard]] const void* pointer() const noexcept { return pointer_; }
  [[nodiscard]] const CudaDeviceState* state() const noexcept {
    return state_.get();
  }

 private:
  std::shared_ptr<CudaDeviceState> state_;
  void* pointer_ = nullptr;
};

class CudaSubmission final : public GpuSubmission {
 public:
  CudaSubmission(std::shared_ptr<CudaDeviceState> state, cudaEvent_t event,
                 bool fail_completion);
  ~CudaSubmission() override;

  Status Wait() override;

 private:
  std::shared_ptr<CudaDeviceState> state_;
  cudaEvent_t event_ = nullptr;
  bool fail_completion_ = false;
  std::once_flag wait_once_;
  Status completion_status_;
};

class CudaPreparedAqEvaluation;
class CudaPreparedExactAqEvaluation;
class CudaPreparedResidentAqEvaluation;
class CudaPreparedDeviceButteraugli;

class CudaBackend final : public GpuBackend,
                          public GpuImagePrimitives,
                          public GpuAcStrategyEvaluation,
                          public DeviceButteraugliOperation,
                          public GpuAqEvaluation {
 public:
  using EncodeCallback = cudaError_t (*)(CudaBackend&, const void*);

  CudaBackend(std::shared_ptr<CudaDeviceState> state, std::string name,
              bool fail_submission, bool fail_completion);

  [[nodiscard]] BackendKind kind() const noexcept override;
  [[nodiscard]] std::string_view name() const noexcept override;

  Status Allocate(size_t size_bytes,
                  std::unique_ptr<DeviceBuffer>* out) override;
  Status CopyHostToDevice(DeviceBuffer& dst, const void* src, size_t size_bytes,
                          size_t dst_offset_bytes) override;
  Status CopyDeviceToHost(const DeviceBuffer& src, void* dst, size_t size_bytes,
                          size_t src_offset_bytes) override;
  Status ForwardTransform(const TransformBatch& batch,
                          std::unique_ptr<GpuSubmission>* submission) override;
  Status InverseTransform(const TransformBatch& batch,
                          std::unique_ptr<GpuSubmission>* submission) override;

  Status SubmitImagePrimitiveSequence(
      std::span<const ImagePrimitiveCommand> commands,
      std::unique_ptr<GpuSubmission>* submission) override;
  Status EvaluateAcStrategyCandidateBatches(
      std::span<const AcStrategyCandidateBatch> batches,
      std::unique_ptr<GpuSubmission>* submission) override;
  Status Prepare(GpuBackend& backend,
                 const DeviceButteraugliPrepareDescriptor& descriptor,
                 std::unique_ptr<PreparedDeviceButteraugli>* prepared) override;
  Status PrepareAqEvaluation(
      const AqEvaluationPreparation& preparation,
      std::unique_ptr<PreparedAqEvaluation>* prepared) override;

  void ArmNextSubmissionFailureForTest(bool fail_submission,
                                       bool fail_completion) noexcept;

 private:
  friend class CudaPreparedAqEvaluation;
  friend class CudaPreparedExactAqEvaluation;
  friend class CudaPreparedResidentAqEvaluation;
  friend class CudaPreparedDeviceButteraugli;

  struct ResolvedConstPlane {
    ConstDevicePlaneView view;
    DeviceMemoryRange range;
    const CudaBuffer* buffer = nullptr;
  };

  struct ResolvedPlane {
    DevicePlaneView view;
    DeviceMemoryRange range;
    CudaBuffer* buffer = nullptr;
  };

  struct TransformContext {
    bool forward = true;
    const CudaBuffer* input = nullptr;
    CudaBuffer* output = nullptr;
    size_t transform_count = 0;
    unsigned int width = 0;
    unsigned int height = 0;
  };

  struct ValidatedAcStrategyBatch {
    AcStrategyType strategy = AcStrategyType::kCount;
    std::array<const float*, 3> opsin{};
    const float* pixel_mask = nullptr;
    const float* quant_field = nullptr;
    const float* matrices = nullptr;
    const void* candidates = nullptr;
    float* scratch_a = nullptr;
    float* scratch_b = nullptr;
    void* rate_scratch = nullptr;
    float* costs = nullptr;
    CudaAcStrategyBatchParams params{};
  };

  struct AcStrategyEncodeContext {
    std::span<const ValidatedAcStrategyBatch> batches;
  };

  [[nodiscard]] static CudaBuffer* AsCudaBuffer(DeviceBuffer& buffer);
  [[nodiscard]] static const CudaBuffer* AsCudaBuffer(
      const DeviceBuffer& buffer);
  [[nodiscard]] static bool IsSupportedDct(AcStrategyType strategy) noexcept;
  [[nodiscard]] static cudaError_t EncodeTransform(CudaBackend& backend,
                                                   const void* context);
  [[nodiscard]] static bool IsSupportedAcStrategy(
      AcStrategyType strategy) noexcept;
  [[nodiscard]] static cudaError_t EncodeAcStrategySubmission(
      CudaBackend& backend, const void* context);

  Status SubmitCompute(EncodeCallback encode, const void* context,
                       std::unique_ptr<GpuSubmission>* submission);
  Status SubmitTransform(bool forward, const TransformBatch& batch,
                         std::unique_ptr<GpuSubmission>* submission);
  Status ValidateAcStrategyCandidateBatch(const AcStrategyCandidateBatch& batch,
                                          ValidatedAcStrategyBatch* out) const;
  Status RequireCudaBuffer(const DeviceBuffer* buffer, size_t required_bytes,
                           std::string_view role, const CudaBuffer** out) const;
  Status RequireCudaBuffer(DeviceBuffer* buffer, size_t required_bytes,
                           std::string_view role, CudaBuffer** out) const;

  Status ResolvePlane(ConstDevicePlaneView view, ResolvedConstPlane* out) const;
  Status ResolvePlane(DevicePlaneView view, ResolvedPlane* out) const;
  [[nodiscard]] static bool SamePlaneLayout(
      ConstDevicePlaneView left, ConstDevicePlaneView right) noexcept;
  [[nodiscard]] static Status RejectOverlap(DeviceMemoryRange left,
                                            DeviceMemoryRange right,
                                            std::string_view message);
  Status ValidatePrimitive(const PointwiseAffineCommand& command) const;
  Status ValidatePrimitive(const SeparableConvolutionCommand& command) const;
  Status ValidatePrimitive(const Symmetric5ConvolutionCommand& command) const;
  Status ValidatePrimitive(const MaximumReductionCommand& command) const;
  Status ValidatePrimitiveCommand(const ImagePrimitiveCommand& command) const;
  [[nodiscard]] cudaError_t EncodePrimitive(
      const PointwiseAffineCommand& command);
  [[nodiscard]] cudaError_t EncodePrimitive(
      const SeparableConvolutionCommand& command);
  [[nodiscard]] cudaError_t EncodePrimitive(
      const Symmetric5ConvolutionCommand& command);
  [[nodiscard]] cudaError_t EncodePrimitive(
      const MaximumReductionCommand& command);
  [[nodiscard]] cudaError_t EncodePrimitiveCommand(
      const ImagePrimitiveCommand& command);
  [[nodiscard]] static cudaError_t EncodePrimitiveSequence(CudaBackend& backend,
                                                           const void* context);

  std::shared_ptr<CudaDeviceState> state_;
  std::string name_;
  bool test_fail_submission_ = false;
  bool test_fail_completion_ = false;
  std::atomic<bool> fail_next_submission_{false};
  std::atomic<bool> fail_next_completion_{false};
};

[[nodiscard]] Status PrepareCudaExactAqEvaluation(
    CudaBackend& backend, const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared);

[[nodiscard]] Status PrepareCudaResidentAqEvaluation(
    CudaBackend& backend, const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared);

}  // namespace gjxl::cuda_internal
