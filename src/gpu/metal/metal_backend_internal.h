// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"
#include "gpu/ops/ac_strategy.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/aq_evaluation_internal.h"
#include "gpu/ops/butteraugli.h"
#include "gpu/ops/gpu_execution_profile_internal.h"
#include "gpu/ops/primitives.h"
#include "gpu/scratch.h"

namespace gjxl::metal_internal {

class MetalBackend;

using MetalComputeEncodeCallback = void (*)(
  MetalBackend&,
  MTL::ComputeCommandEncoder*,
  const void*);

struct MetalProfiledComputeStage {
  const char* stage_id = nullptr;
  const char* group_id = nullptr;
  uint32_t iteration = 0;
  uint32_t invocation = 0;
  MetalComputeEncodeCallback encode = nullptr;
  const void* context = nullptr;
};

void DispatchMetalThreads(
  MTL::ComputeCommandEncoder* encoder,
  MTL::Size threads_per_grid,
  MTL::Size threads_per_threadgroup);

void DispatchMetalThreadgroups(
  MTL::ComputeCommandEncoder* encoder,
  MTL::Size threadgroups_per_grid,
  MTL::Size threads_per_threadgroup);

void RegisterMetalComputePipeline(
  MTL::ComputePipelineState* pipeline,
  std::string_view kernel_id);

void RecordMetalComputePipelineState(MTL::ComputePipelineState* pipeline);

enum class TransformDirection {
  kForward,
  kInverse,
};

struct TransformPipeline {
  NS::SharedPtr<MTL::ComputePipelineState> state;
  NS::UInteger threads_per_threadgroup = 0;
  size_t transforms_per_threadgroup = 1;
  bool uses_device_basis = false;
  AcStrategyType strategy = AcStrategyType::kCount;
  std::string label;
};

struct TransformPipelinePair {
  TransformPipeline forward;
  TransformPipeline inverse;
};

using TransformPipelineRegistry =
  std::array<TransformPipelinePair, kAcStrategyCount>;

struct PrimitivePipelines {
  NS::SharedPtr<MTL::ComputePipelineState> affine;
  NS::SharedPtr<MTL::ComputePipelineState> convolution_horizontal;
  NS::SharedPtr<MTL::ComputePipelineState> convolution_vertical;
  NS::SharedPtr<MTL::ComputePipelineState> symmetric5_convolution;
  NS::SharedPtr<MTL::ComputePipelineState> maximum_reduction;
};

struct AqPipelines {
  NS::SharedPtr<MTL::ComputePipelineState> block_reduction;
  NS::SharedPtr<MTL::ComputePipelineState> maximum_error_reduction;
  NS::SharedPtr<MTL::ComputePipelineState> reset_exact_evaluation;
  NS::SharedPtr<MTL::ComputePipelineState> reset_exact_coefficients;
  NS::SharedPtr<MTL::ComputePipelineState> reset_reconstruction;
  NS::SharedPtr<MTL::ComputePipelineState> reset_frame_encoding;
  NS::SharedPtr<MTL::ComputePipelineState> initial_cfl;
  NS::SharedPtr<MTL::ComputePipelineState> final_cfl;
  NS::SharedPtr<MTL::ComputePipelineState> reset_initial_quant;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_gradient;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_fuzzy_erosion;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_modulation;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_sort_prepare;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_sort_step;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_capture_median;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_deviation_prepare;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_finalize_quantizer;
  NS::SharedPtr<MTL::ComputePipelineState> initial_quant_raw_quant;
  NS::SharedPtr<MTL::ComputePipelineState> adjust_quant_field;
  NS::SharedPtr<MTL::ComputePipelineState> resident_quant_select_initialize;
  NS::SharedPtr<MTL::ComputePipelineState> resident_quant_histogram;
  NS::SharedPtr<MTL::ComputePipelineState> resident_quant_select_bucket;
  NS::SharedPtr<MTL::ComputePipelineState> resident_quant_finalize_quantizer;
  NS::SharedPtr<MTL::ComputePipelineState> resident_policy_initialize;
  NS::SharedPtr<MTL::ComputePipelineState> resident_policy_update;
  NS::SharedPtr<MTL::ComputePipelineState> gather_transform_pixels;
  NS::SharedPtr<MTL::ComputePipelineState> select_adjusted_quantization;
  NS::SharedPtr<MTL::ComputePipelineState> encode_reconstruction_coefficients;
  NS::SharedPtr<MTL::ComputePipelineState> encode_frame_coefficients;
  NS::SharedPtr<MTL::ComputePipelineState> scatter_reconstructed_pixels;
  NS::SharedPtr<MTL::ComputePipelineState> quantization_probe;
  NS::SharedPtr<MTL::ComputePipelineState> adjustment_probe;
  NS::SharedPtr<MTL::ComputePipelineState> gaborish;
  NS::SharedPtr<MTL::ComputePipelineState> epf;
  NS::SharedPtr<MTL::ComputePipelineState> opsin_to_linear;
};

struct AcStrategyPipelines {
  struct FusedStages {
    NS::SharedPtr<MTL::ComputePipelineState> forward;
    NS::SharedPtr<MTL::ComputePipelineState> residual_inverse;
    NS::UInteger forward_threads_per_threadgroup = 0;
  };

  NS::SharedPtr<MTL::ComputePipelineState> gather;
  NS::SharedPtr<MTL::ComputePipelineState> residual;
  NS::SharedPtr<MTL::ComputePipelineState> cost;
  std::array<FusedStages, kAcStrategyCount> fused;
  NS::UInteger gather_threads_per_threadgroup = 0;
};

struct MetalAcStrategyBatchParams {
  uint32_t pixel_width;
  uint32_t pixel_height;
  uint32_t opsin_row_stride;
  uint32_t pixel_mask_row_stride;
  uint32_t quant_field_row_stride;
  uint32_t candidate_count;
  uint32_t coefficient_count;
  uint32_t transform_width;
  uint32_t transform_height;
  uint32_t covered_block_width;
  uint32_t covered_block_height;
  uint32_t covered_block_count;
  uint32_t use_device_quant_norm;
  float info_loss_multiplier;
  float zeros_multiplier;
  float cost_delta;
};

static_assert(std::is_standard_layout_v<MetalAcStrategyBatchParams>);
static_assert(sizeof(MetalAcStrategyBatchParams) == 16 * sizeof(uint32_t));

class MetalPreparedAqEvaluation;

enum class MetalAqScratchArena : uint8_t {
  kPersistent,
  kStaging,
  kCount,
};

struct ButteraugliPipelines {
  NS::SharedPtr<MTL::ComputePipelineState> copy;
  NS::SharedPtr<MTL::ComputePipelineState> expand;
  NS::SharedPtr<MTL::ComputePipelineState> subsample;
  NS::SharedPtr<MTL::ComputePipelineState> blur5_horizontal;
  NS::SharedPtr<MTL::ComputePipelineState> blur5_vertical;
  NS::SharedPtr<MTL::ComputePipelineState> convolution_transpose;
  NS::SharedPtr<MTL::ComputePipelineState> opsin_blur5;
  NS::SharedPtr<MTL::ComputePipelineState> frequency_low_medium_convolve;
  NS::SharedPtr<MTL::ComputePipelineState> frequency_high_convolve;
  NS::SharedPtr<MTL::ComputePipelineState> frequency_suppress_x;
  NS::SharedPtr<MTL::ComputePipelineState> frequency_ultra_convolve;
  NS::SharedPtr<MTL::ComputePipelineState> malta_scale;
  NS::SharedPtr<MTL::ComputePipelineState> malta_response;
  NS::SharedPtr<MTL::ComputePipelineState> malta_fused;
  NS::SharedPtr<MTL::ComputePipelineState> l2;
  NS::SharedPtr<MTL::ComputePipelineState> mask_precompute;
  NS::SharedPtr<MTL::ComputePipelineState> fuzzy_erosion;
  NS::SharedPtr<MTL::ComputePipelineState> masked_ac;
  NS::SharedPtr<MTL::ComputePipelineState> final;
  NS::SharedPtr<MTL::ComputePipelineState> final_masked_ac;
  NS::SharedPtr<MTL::ComputePipelineState> crop;
  NS::SharedPtr<MTL::ComputePipelineState> compose;
  NS::SharedPtr<MTL::ComputePipelineState> maximum_reduction;
};

class MetalPreparedDeviceButteraugli;

class MetalBuffer final : public DeviceBuffer {
public:
  MetalBuffer(
    NS::SharedPtr<MTL::Buffer> buffer,
    BackendId backend_id,
    size_t size_bytes)
    : DeviceBuffer(BackendKind::kMetal, backend_id, size_bytes),
      buffer_(std::move(buffer)) {}

  ~MetalBuffer() override = default;

  [[nodiscard]] MTL::Buffer* handle() const noexcept {
    return buffer_.get();
  }

  [[nodiscard]] MTL::Device* device() const noexcept {
    return buffer_->device();
  }

  [[nodiscard]] void* contents() noexcept {
    return buffer_->contents();
  }

  [[nodiscard]] const void* contents() const noexcept {
    return buffer_->contents();
  }

private:
  NS::SharedPtr<MTL::Buffer> buffer_;
};

class MetalBackend final
  : public GpuBackend,
    public GpuAcStrategyEvaluation,
    public gpu_profile_internal::GpuAcStrategyEvaluationProfiler,
    public gpu_profile_internal::GpuSubmissionProfiler,
    public GpuImagePrimitives,
    public gpu_profile_internal::GpuImagePrimitivesProfiler,
    public GpuAqEvaluation,
    public gpu_profile_internal::GpuAqEvaluationProfiler,
    public aq_evaluation_internal::GpuValidatedAqEvaluation,
    public DeviceButteraugliOperation {
public:
  MetalBackend(
    NS::SharedPtr<MTL::Device> device,
    NS::SharedPtr<MTL::CommandQueue> command_queue,
    NS::SharedPtr<MTL::Library> library,
    NS::SharedPtr<MTL::Buffer> dct_basis_buffer,
    TransformPipelineRegistry transform_pipelines,
    AcStrategyPipelines ac_strategy_pipelines,
    PrimitivePipelines primitive_pipelines,
    AqPipelines aq_pipelines,
    ButteraugliPipelines butteraugli_pipelines,
    bool test_fail_submission,
    bool test_fail_completion);

  ~MetalBackend() override = default;

  [[nodiscard]] BackendKind kind() const noexcept override;
  [[nodiscard]] std::string_view name() const noexcept override;

  Status Allocate(
    size_t size_bytes,
    std::unique_ptr<DeviceBuffer>* out) override;

  Status CopyHostToDevice(
    DeviceBuffer& dst,
    const void* src,
    size_t size_bytes,
    size_t dst_offset_bytes) override;

  Status CopyDeviceToHost(
    const DeviceBuffer& src,
    void* dst,
    size_t size_bytes,
    size_t src_offset_bytes) override;

  /// Borrows completed shared-buffer storage for synchronous host reads.
  /// The caller must wait for every GPU command that can write the range and
  /// must not retain the span beyond the buffer's lifetime.
  Status BorrowCompletedReadOnly(
    const DeviceBuffer& src,
    size_t size_bytes,
    size_t src_offset_bytes,
    std::span<const std::byte>* out) const;

  Status ForwardTransform(
    const TransformBatch& batch,
    std::unique_ptr<GpuSubmission>* submission) override;

  Status InverseTransform(
    const TransformBatch& batch,
    std::unique_ptr<GpuSubmission>* submission) override;

  Status EvaluateAcStrategyCandidateBatches(
    std::span<const AcStrategyCandidateBatch> batches,
    std::unique_ptr<GpuSubmission>* submission) override;

  Status EvaluateAcStrategyCandidateBatchesProfiled(
    std::span<const AcStrategyCandidateBatch> batches,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<GpuSubmission>* submission) override;

  [[nodiscard]] gpu_profile_internal::GpuProfilingCapabilities
  QueryGpuProfilingCapabilities() const override;

  Status ResolveGpuSubmissionProfile(
    GpuSubmission& submission,
    std::string_view submission_id,
    gpu_profile_internal::GpuProfilingMode mode,
    gpu_profile_internal::GpuExecutionProfile* profile) override;

  Status SubmitImagePrimitiveSequence(
    std::span<const ImagePrimitiveCommand> commands,
    std::unique_ptr<GpuSubmission>* submission) override;

  Status SubmitImagePrimitiveSequenceProfiled(
    std::span<const ImagePrimitiveCommand> commands,
    std::string_view stage_id,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<GpuSubmission>* submission) override;

  Status PrepareAqEvaluation(
    const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared) override;

  Status PrepareAqEvaluationProfiled(
    const AqEvaluationPreparation& preparation,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<PreparedAqEvaluation>* prepared,
    gpu_profile_internal::GpuExecutionProfile* profile) override;

  Status PrepareValidatedAqEvaluation(
    const AqEvaluationPreparation& preparation,
    std::unique_ptr<PreparedAqEvaluation>* prepared) override;

  Status PrepareValidatedAqEvaluationProfiled(
    const AqEvaluationPreparation& preparation,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<PreparedAqEvaluation>* prepared,
    gpu_profile_internal::GpuExecutionProfile* profile) override;

  Status Prepare(
    GpuBackend& backend,
    const DeviceButteraugliPrepareDescriptor& descriptor,
    std::unique_ptr<PreparedDeviceButteraugli>* prepared) override;

  void ArmNextSubmissionFailureForTest(
    bool fail_submission,
    bool fail_completion) noexcept;

private:
  friend class MetalPreparedAqEvaluation;
  friend class MetalPreparedDeviceButteraugli;
  friend Status EmptyMetalAqScratchArenasForTesting(GpuBackend& backend);

  Status PrepareDeviceButteraugliImpl(
    const DeviceButteraugliPrepareDescriptor& descriptor,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<PreparedDeviceButteraugli>* prepared,
    gpu_profile_internal::GpuExecutionProfile* profile);

  Status PrepareAqEvaluationImpl(
    const AqEvaluationPreparation& preparation,
    bool host_images_are_finite,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<PreparedAqEvaluation>* prepared,
    gpu_profile_internal::GpuExecutionProfile* profile);

  Status AcquireAqScratchArena(
    MetalAqScratchArena kind,
    size_t required_capacity_bytes,
    DeviceScratchArena* arena);

  void ReleaseAqScratchArena(
    MetalAqScratchArena kind,
    DeviceScratchArena arena,
    bool reusable) noexcept;

  Status EmptyAqScratchArenasForTesting();
  struct TransformEncodeContext {
    const TransformPipeline* pipeline = nullptr;
    TransformDirection direction = TransformDirection::kForward;
    const MetalBuffer* input = nullptr;
    MetalBuffer* output = nullptr;
    size_t input_offset_bytes = 0;
    size_t output_offset_bytes = 0;
    size_t transform_count = 0;
  };

  struct ResolvedConstPlane {
    ConstDevicePlaneView view;
    DeviceMemoryRange range;
    const MetalBuffer* buffer = nullptr;
  };

  struct ResolvedPlane {
    DevicePlaneView view;
    DeviceMemoryRange range;
    MetalBuffer* buffer = nullptr;
  };

  struct ValidatedAcStrategyBatch {
    AcStrategyType strategy = AcStrategyType::kCount;
    std::array<const MetalBuffer*, 3> opsin{};
    std::array<size_t, 3> opsin_offset_bytes{};
    const MetalBuffer* pixel_mask = nullptr;
    size_t pixel_mask_offset_bytes = 0;
    const MetalBuffer* quant_field = nullptr;
    size_t quant_field_offset_bytes = 0;
    const MetalBuffer* matrices = nullptr;
    const MetalBuffer* candidates = nullptr;
    MetalBuffer* scratch_a = nullptr;
    MetalBuffer* scratch_b = nullptr;
    MetalBuffer* rate_scratch = nullptr;
    MetalBuffer* costs = nullptr;
    const TransformPipeline* forward = nullptr;
    const TransformPipeline* inverse = nullptr;
    MetalAcStrategyBatchParams params{};
    size_t transform_count = 0;
    size_t packed_element_count = 0;
  };

  struct AcStrategyEncodeContext {
    std::span<const ValidatedAcStrategyBatch> batches;
  };

  struct AcStrategyProfileContext {
    const ValidatedAcStrategyBatch* batch = nullptr;
  };

  using ComputeEncodeCallback = MetalComputeEncodeCallback;

  static MetalBuffer* AsMetalBuffer(DeviceBuffer& buffer);
  static const MetalBuffer* AsMetalBuffer(const DeviceBuffer& buffer);

  [[nodiscard]] static bool TryMultiply(
    size_t left,
    size_t right,
    size_t* result) noexcept;

  Status RequireMetalBuffer(
    const DeviceBuffer* buffer,
    size_t required_bytes,
    std::string_view role,
    const MetalBuffer** out) const;

  Status RequireMetalBuffer(
    DeviceBuffer* buffer,
    size_t required_bytes,
    std::string_view role,
    MetalBuffer** out) const;

  Status ValidateAcStrategyCandidateBatch(
    const AcStrategyCandidateBatch& batch,
    ValidatedAcStrategyBatch* out) const;

  static void EncodeAcStrategySubmission(
    MetalBackend& backend,
    MTL::ComputeCommandEncoder* encoder,
    const void* context);

  static void EncodeAcStrategyProfileStage(
    MetalBackend& backend,
    MTL::ComputeCommandEncoder* encoder,
    const void* context);

  void EncodeAcStrategyCandidateBatch(
    MTL::ComputeCommandEncoder* encoder,
    const ValidatedAcStrategyBatch& validated);

  Status SubmitAcStrategyCandidatesImpl(
    std::span<const AcStrategyCandidateBatch> batches,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<GpuSubmission>* submission);

  Status SubmitCompute(
    const char* label,
    ComputeEncodeCallback encode,
    const void* context,
    std::unique_ptr<GpuSubmission>* submission);

  [[nodiscard]] gpu_profile_internal::GpuProfilingCapabilities
  ProfilingCapabilities() const;

  Status SubmitComputeProfiled(
    const char* label,
    std::span<const MetalProfiledComputeStage> stages,
    gpu_profile_internal::GpuProfilingMode mode,
    std::unique_ptr<GpuSubmission>* submission);

  Status ValidateTransformBatch(
    const AcStrategyInfo& strategy_info,
    const TransformBatch& batch,
    const MetalBuffer** input,
    MetalBuffer** output) const;

  Status SubmitTransform(
    TransformDirection direction,
    const TransformBatch& batch,
    std::unique_ptr<GpuSubmission>* submission);

  static void EncodeTransformSubmission(
    MetalBackend& backend,
    MTL::ComputeCommandEncoder* encoder,
    const void* context);

  void EncodeTransformBatch(
    MTL::ComputeCommandEncoder* encoder,
    TransformDirection direction,
    AcStrategyType strategy,
    const MetalBuffer& input,
    size_t input_offset_bytes,
    MetalBuffer& output,
    size_t output_offset_bytes,
    size_t transform_count) const;

  Status ResolvePlane(
    ConstDevicePlaneView view,
    ResolvedConstPlane* out) const;

  Status ResolvePlane(
    DevicePlaneView view,
    ResolvedPlane* out) const;

  [[nodiscard]] static bool SamePlaneLayout(
    ConstDevicePlaneView left,
    ConstDevicePlaneView right) noexcept;

  [[nodiscard]] static Status RejectOverlap(
    DeviceMemoryRange left,
    DeviceMemoryRange right,
    std::string_view message);

  Status ValidatePrimitive(const PointwiseAffineCommand& command) const;
  Status ValidatePrimitive(const SeparableConvolutionCommand& command) const;
  Status ValidatePrimitive(const Symmetric5ConvolutionCommand& command) const;
  Status ValidatePrimitive(const MaximumReductionCommand& command) const;
  Status ValidatePrimitiveCommand(
    const ImagePrimitiveCommand& command) const;

  static void DispatchPlane(
    MTL::ComputeCommandEncoder* encoder,
    Extent2D extent);

  void EncodePrimitive(
    MTL::ComputeCommandEncoder* encoder,
    const PointwiseAffineCommand& command);

  void EncodeConvolutionPass(
    MTL::ComputeCommandEncoder* encoder,
    MTL::ComputePipelineState* pipeline,
    ConstDevicePlaneView input_view,
    ConstDevicePlaneView kernel_view,
    DevicePlaneView output_view);

  void EncodePrimitive(
    MTL::ComputeCommandEncoder* encoder,
    const SeparableConvolutionCommand& command);

  void EncodePrimitive(
    MTL::ComputeCommandEncoder* encoder,
    const Symmetric5ConvolutionCommand& command);

  void EncodeReductionPass(
    MTL::ComputeCommandEncoder* encoder,
    ConstDevicePlaneView input_view,
    size_t input_count,
    DevicePlaneView output_view);

  void EncodePrimitive(
    MTL::ComputeCommandEncoder* encoder,
    const MaximumReductionCommand& command);

  void EncodePrimitiveCommand(
    MTL::ComputeCommandEncoder* encoder,
    const ImagePrimitiveCommand& command);

  static void EncodePrimitiveSubmission(
    MetalBackend& backend,
    MTL::ComputeCommandEncoder* encoder,
    const void* context);

  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
  NS::SharedPtr<MTL::Library> library_;
  NS::SharedPtr<MTL::Buffer> dct_basis_buffer_;
  TransformPipelineRegistry transform_pipelines_;
  AcStrategyPipelines ac_strategy_pipelines_;
  PrimitivePipelines primitive_pipelines_;
  AqPipelines aq_pipelines_;
  ButteraugliPipelines butteraugli_pipelines_;
  bool test_fail_submission_ = false;
  bool test_fail_completion_ = false;
  std::atomic<bool> test_fail_next_submission_{false};
  std::atomic<bool> test_fail_next_completion_{false};
  std::mutex aq_scratch_pool_mutex_;
  std::array<
    std::optional<DeviceScratchArena>,
    static_cast<size_t>(MetalAqScratchArena::kCount)> idle_aq_scratch_;
  std::string name_;
};

Status CreateAcStrategyPipelines(
  MTL::Device* device,
  MTL::Library* library,
  const std::array<bool, kAcStrategyCount>& fused_forward_enabled,
  const std::array<bool, kAcStrategyCount>& fused_inverse_enabled,
  AcStrategyPipelines* out);

Status CreatePrimitivePipelines(
  MTL::Device* device,
  MTL::Library* library,
  PrimitivePipelines* out);

Status CreateAqPipelines(
  MTL::Device* device,
  MTL::Library* library,
  AqPipelines* out);

[[nodiscard]] Status GetMetalSubmissionGpuDuration(
  GpuSubmission& submission,
  uint64_t* nanoseconds);

[[nodiscard]] Status GetMetalSubmissionGpuProfile(
  GpuSubmission& submission,
  gpu_profile_internal::GpuSubmissionProfile* profile);

Status CreateButteraugliPipelines(
  MTL::Device* device,
  MTL::Library* library,
  ButteraugliPipelines* out);

}  // namespace gjxl::metal_internal
