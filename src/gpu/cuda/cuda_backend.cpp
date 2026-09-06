// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_backend.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/ac_strategy.h"
#include "gpu/buffer.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_kernels.h"
#include "gpu/ops/transform.h"
#include "gpu/submission.h"

namespace gjxl::cuda_internal {

namespace {

struct PoolRegistry {
  std::mutex mutex;
  // Weak ownership prevents the registry from extending backend lifetimes.
  std::vector<std::weak_ptr<CudaMemoryPoolState>> pools;
};

PoolRegistry& MemoryPools() {
  static PoolRegistry registry;
  return registry;
}

cudaError_t AllocateDeviceMemory(CudaDeviceState& state, size_t size_bytes,
                                 void** pointer) {
#if CUDART_VERSION >= 11020
  if (state.memory_pool != nullptr) {
    std::lock_guard lock(state.submission_mutex);
    return cudaMallocFromPoolAsync(pointer, size_bytes, state.memory_pool->pool,
                                   state.stream);
  }
#endif
  return cudaMalloc(pointer, size_bytes);
}

void FreeDeviceMemory(CudaDeviceState& state, void* pointer) {
#if CUDART_VERSION >= 11020
  if (state.memory_pool != nullptr) {
    std::lock_guard lock(state.submission_mutex);
    if (cudaFreeAsync(pointer, state.stream) == cudaSuccess) return;
    // cudaFree does not synchronize pool allocations. If enqueueing the free
    // fails, drain our stream before attempting synchronous cleanup.
    (void)cudaStreamSynchronize(state.stream);
  }
#endif
  (void)cudaFree(pointer);
}

}  // namespace

Status AcquireMemoryPool(int ordinal, uint64_t release_threshold_bytes,
                         std::shared_ptr<CudaMemoryPoolState>* out) {
#if CUDART_VERSION >= 11020
  int supported = 0;
  cudaError_t error = cudaDeviceGetAttribute(
      &supported, cudaDevAttrMemoryPoolsSupported, ordinal);
  if (error == cudaErrorInvalidValue || error == cudaErrorNotSupported) {
    // Older drivers may not recognize this attribute. Pooling is optional.
    (void)cudaGetLastError();
    return Status::Ok();
  }
  if (error != cudaSuccess)
    return CudaRuntimeStatus(error, "Query CUDA memory-pool support");
  if (supported == 0) return Status::Ok();

  PoolRegistry& registry = MemoryPools();
  std::lock_guard lock(registry.mutex);
  for (auto it = registry.pools.begin(); it != registry.pools.end();) {
    auto existing = it->lock();
    if (existing == nullptr) {
      it = registry.pools.erase(it);
    } else {
      if (existing->ordinal == ordinal &&
          existing->release_threshold_bytes == release_threshold_bytes) {
        *out = std::move(existing);
        return Status::Ok();
      }
      ++it;
    }
  }
  auto pool = std::make_shared<CudaMemoryPoolState>();
  pool->ordinal = ordinal;
  pool->release_threshold_bytes = release_threshold_bytes;
  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = ordinal;
  cudaMemPool_t handle = nullptr;
  error = cudaMemPoolCreate(&handle, &properties);
  if (error == cudaErrorNotSupported) {
    (void)cudaGetLastError();
    return Status::Ok();
  }
  if (error != cudaSuccess)
    return CudaRuntimeStatus(error, "Create private CUDA memory pool");
  pool->pool = handle;
  error = cudaMemPoolSetAttribute(pool->pool, cudaMemPoolAttrReleaseThreshold,
                                  &release_threshold_bytes);
  if (error != cudaSuccess)
    return CudaRuntimeStatus(error, "Configure CUDA memory-pool retention");
  // Sharing the pool must not inject dependencies between independent lanes
  // just to reuse memory whose free has not completed yet.
  int allow_internal_dependencies = 0;
  error = cudaMemPoolSetAttribute(
      pool->pool, cudaMemPoolReuseAllowInternalDependencies,
      &allow_internal_dependencies);
  if (error != cudaSuccess)
    return CudaRuntimeStatus(error, "Configure CUDA memory-pool ordering");
  registry.pools.push_back(pool);
  *out = std::move(pool);
#else
  (void)ordinal;
  (void)release_threshold_bytes;
  (void)out;
#endif
  return Status::Ok();
}

Status TrimMemoryPools(int ordinal) {
  std::vector<std::shared_ptr<CudaMemoryPoolState>> pools;
  try {
    PoolRegistry& registry = MemoryPools();
    std::lock_guard lock(registry.mutex);
    for (auto it = registry.pools.begin(); it != registry.pools.end();) {
      auto pool = it->lock();
      if (pool == nullptr) {
        it = registry.pools.erase(it);
      } else {
        if (pool->ordinal == ordinal) pools.push_back(std::move(pool));
        ++it;
      }
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Snapshot CUDA memory pools for trimming");
  }
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
    return CudaRuntimeStatus(error, "Wait before trimming CUDA memory pools");
#if CUDART_VERSION >= 11020
  for (const auto& pool : pools) {
    error = cudaMemPoolTrimTo(pool->pool, 0);
    if (error != cudaSuccess)
      return CudaRuntimeStatus(error, "Trim private CUDA memory pool");
  }
#endif
  return Status::Ok();
}

CudaMemoryPoolState::~CudaMemoryPoolState() {
#if CUDART_VERSION >= 11020
  ScopedCudaDevice device(ordinal);
  if (device.status() == cudaSuccess && pool != nullptr)
    (void)cudaMemPoolDestroy(pool);
#endif
}

Status CudaStatus(
  cudaError_t error,
  std::string_view operation,
  StatusCode code) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  std::string message(operation);
  message += ": ";
  const char* name = cudaGetErrorName(error);
  const char* description = cudaGetErrorString(error);
  message += name == nullptr ? "unknown CUDA error" : name;
  if (description != nullptr) {
    message += " (";
    message += description;
    message += ')';
  }
  switch (code) {
    case StatusCode::kOutOfMemory:
      return Status::OutOfMemory(std::move(message));
    case StatusCode::kUnavailable:
      return Status::Unavailable(std::move(message));
    case StatusCode::kSubmissionFailed:
      return Status::SubmissionFailed(std::move(message));
    case StatusCode::kDeviceError:
      return Status::DeviceError(std::move(message));
    default:
      return Status(code, std::move(message));
  }
}

Status CudaRuntimeStatus(cudaError_t error, std::string_view operation) {
  if (error == cudaErrorMemoryAllocation) {
    return CudaStatus(error, operation, StatusCode::kOutOfMemory);
  }
  if (error == cudaErrorNoDevice ||
      error == cudaErrorInsufficientDriver ||
      error == cudaErrorInitializationError ||
      error == cudaErrorSystemDriverMismatch) {
    return CudaStatus(error, operation, StatusCode::kUnavailable);
  }
  return CudaStatus(error, operation);
}

ScopedCudaDevice::ScopedCudaDevice(int ordinal) {
  if (cudaGetDevice(&previous_) != cudaSuccess) {
    previous_ = -1;
  }
  status_ = previous_ == ordinal ? cudaSuccess : cudaSetDevice(ordinal);
  changed_ = status_ == cudaSuccess && previous_ >= 0 && previous_ != ordinal;
}

ScopedCudaDevice::~ScopedCudaDevice() {
  if (changed_) {
    (void)cudaSetDevice(previous_);
  }
}

CudaDeviceState::~CudaDeviceState() {
  ScopedCudaDevice device(ordinal);
  if (device.status() == cudaSuccess && stream != nullptr) {
    // Buffers and submissions share this state. At its final release all
    // pooled frees have been queued; complete them before destroying the lane.
    if (memory_pool != nullptr) (void)cudaStreamSynchronize(stream);
    (void)cudaStreamDestroy(stream);
  }
}

CudaBuffer::CudaBuffer(
  std::shared_ptr<CudaDeviceState> state,
  BackendId backend_id,
  size_t size_bytes,
  void* pointer)
  : DeviceBuffer(BackendKind::kCuda, backend_id, size_bytes),
    state_(std::move(state)), pointer_(pointer) {}

CudaBuffer::~CudaBuffer() {
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() == cudaSuccess && pointer_ != nullptr) {
    FreeDeviceMemory(*state_, pointer_);
  }
}

CudaSubmission::CudaSubmission(
  std::shared_ptr<CudaDeviceState> state,
  cudaEvent_t event,
  bool fail_completion)
  : state_(std::move(state)), event_(event),
    fail_completion_(fail_completion) {}

CudaSubmission::~CudaSubmission() {
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() == cudaSuccess && event_ != nullptr) {
    (void)cudaEventDestroy(event_);
  }
}

Status CudaSubmission::Wait() {
  std::call_once(wait_once_, [this] {
    ScopedCudaDevice device(state_->ordinal);
    if (device.status() != cudaSuccess) {
      completion_status_ =
        CudaRuntimeStatus(device.status(), "Select CUDA submission device");
      return;
    }
    const cudaError_t error = cudaEventSynchronize(event_);
    if (error != cudaSuccess) {
      completion_status_ =
        CudaRuntimeStatus(error, "CUDA submission completion");
    } else if (fail_completion_) {
      completion_status_ = Status::DeviceError(
        "Injected CUDA submission completion failure");
    }
  });
  return completion_status_;
}

CudaBackend::CudaBackend(
  std::shared_ptr<CudaDeviceState> state,
  std::string name,
  bool fail_submission,
  bool fail_completion)
  : state_(std::move(state)), name_(std::move(name)),
    test_fail_submission_(fail_submission),
    test_fail_completion_(fail_completion) {}

BackendKind CudaBackend::kind() const noexcept {
  return BackendKind::kCuda;
}

std::string_view CudaBackend::name() const noexcept {
  return name_;
}

Status CudaBackend::Allocate(
  size_t size_bytes,
  std::unique_ptr<DeviceBuffer>* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("Allocate output pointer is null");
  }
  if (size_bytes == 0) {
    return Status::InvalidArgument("Cannot allocate zero-sized CUDA buffer");
  }
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() != cudaSuccess) {
    return CudaRuntimeStatus(device.status(), "Select CUDA allocation device");
  }
  void* pointer = nullptr;
  const cudaError_t error = AllocateDeviceMemory(*state_, size_bytes, &pointer);
  if (error != cudaSuccess) {
    return CudaRuntimeStatus(error, "Allocate CUDA device buffer");
  }
  try {
    out->reset(new CudaBuffer(state_, id(), size_bytes, pointer));
  } catch (const std::bad_alloc&) {
    FreeDeviceMemory(*state_, pointer);
    return Status::OutOfMemory("Allocate CUDA buffer owner");
  }
  RecordSuccessfulAllocation();
  return Status::Ok();
}

CudaBuffer* CudaBackend::AsCudaBuffer(DeviceBuffer& buffer) {
  return buffer.backend() == BackendKind::kCuda
    ? dynamic_cast<CudaBuffer*>(&buffer)
    : nullptr;
}

const CudaBuffer* CudaBackend::AsCudaBuffer(const DeviceBuffer& buffer) {
  return buffer.backend() == BackendKind::kCuda
    ? dynamic_cast<const CudaBuffer*>(&buffer)
    : nullptr;
}

Status CudaBackend::CopyHostToDevice(
  DeviceBuffer& dst,
  const void* src,
  size_t size_bytes,
  size_t dst_offset_bytes) {
  if (src == nullptr && size_bytes != 0) {
    return Status::InvalidArgument("Host source pointer is null");
  }
  CudaBuffer* cuda_dst = AsCudaBuffer(dst);
  if (cuda_dst == nullptr) {
    return Status::InvalidArgument("Destination is not a CUDA buffer");
  }
  if (!owns(dst) || cuda_dst->state() != state_.get()) {
    return Status::InvalidArgument(
      "Destination belongs to another CUDA backend");
  }
  if (dst_offset_bytes > dst.size_bytes() ||
      size_bytes > dst.size_bytes() - dst_offset_bytes) {
    return Status::InvalidArgument(
      "Host to device copy exceeds destination buffer");
  }
  if (size_bytes == 0) {
    return Status::Ok();
  }
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() != cudaSuccess) {
    return CudaRuntimeStatus(device.status(), "Select CUDA copy device");
  }
  auto* pointer = static_cast<std::byte*>(cuda_dst->pointer()) +
    dst_offset_bytes;
  std::lock_guard lock(state_->submission_mutex);
  cudaError_t error = cudaMemcpyAsync(
    pointer, src, size_bytes, cudaMemcpyHostToDevice, state_->stream);
  if (error == cudaSuccess) {
    error = cudaStreamSynchronize(state_->stream);
  }
  return CudaRuntimeStatus(error, "Copy host data to CUDA buffer");
}

Status CudaBackend::CopyHostToDevice2D(
  DeviceBuffer& dst,
  const void* src,
  size_t src_row_stride_bytes,
  size_t row_bytes,
  size_t row_count,
  size_t dst_row_stride_bytes,
  size_t dst_offset_bytes) {
  const bool empty = row_bytes == 0 || row_count == 0;
  if (src == nullptr && !empty) {
    return Status::InvalidArgument("Host 2D source pointer is null");
  }
  CudaBuffer* cuda_dst = AsCudaBuffer(dst);
  if (cuda_dst == nullptr) {
    return Status::InvalidArgument("2D destination is not a CUDA buffer");
  }
  if (!owns(dst) || cuda_dst->state() != state_.get()) {
    return Status::InvalidArgument(
      "2D destination belongs to another CUDA backend");
  }
  if (dst_offset_bytes > dst.size_bytes()) {
    return Status::InvalidArgument(
      "Host 2D copy exceeds destination buffer");
  }
  if (empty) {
    return Status::Ok();
  }
  if (src_row_stride_bytes < row_bytes ||
      dst_row_stride_bytes < row_bytes) {
    return Status::InvalidArgument("CUDA 2D copy row stride is too small");
  }
  if (row_count - 1 >
      (std::numeric_limits<size_t>::max() - row_bytes) /
        src_row_stride_bytes) {
    return Status::InvalidArgument("Host 2D copy geometry overflows");
  }
  if (row_bytes > dst.size_bytes() - dst_offset_bytes ||
      row_count - 1 >
        (dst.size_bytes() - dst_offset_bytes - row_bytes) /
          dst_row_stride_bytes) {
    return Status::InvalidArgument(
      "Host 2D copy exceeds destination buffer");
  }
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() != cudaSuccess) {
    return CudaRuntimeStatus(device.status(), "Select CUDA 2D copy device");
  }
  auto* pointer = static_cast<std::byte*>(cuda_dst->pointer()) +
    dst_offset_bytes;
  std::lock_guard lock(state_->submission_mutex);
  cudaError_t error =
    src_row_stride_bytes == row_bytes && dst_row_stride_bytes == row_bytes
      ? cudaMemcpyAsync(
          pointer, src, row_count * row_bytes, cudaMemcpyHostToDevice,
          state_->stream)
      : cudaMemcpy2DAsync(
          pointer, dst_row_stride_bytes, src, src_row_stride_bytes,
          row_bytes, row_count, cudaMemcpyHostToDevice, state_->stream);
  if (error == cudaSuccess) {
    error = cudaStreamSynchronize(state_->stream);
  }
  return CudaRuntimeStatus(error, "Copy 2D host data to CUDA buffer");
}

Status CudaBackend::CopyHostToDeviceBatch(
    std::span<const CudaHostToDeviceCopy> copies) {
  for (const CudaHostToDeviceCopy& copy : copies) {
    if (copy.destination == nullptr ||
        (copy.source == nullptr && copy.size_bytes != 0)) {
      return Status::InvalidArgument(
          "CUDA host-to-device batch contains a null pointer");
    }
    CudaBuffer* destination = AsCudaBuffer(*copy.destination);
    if (destination == nullptr || !owns(*copy.destination) ||
        destination->state() != state_.get()) {
      return Status::InvalidArgument(
          "CUDA host-to-device batch destination is not owned");
    }
    if (copy.destination_offset_bytes > copy.destination->size_bytes() ||
        copy.size_bytes > copy.destination->size_bytes() -
                            copy.destination_offset_bytes) {
      return Status::InvalidArgument(
          "CUDA host-to-device batch exceeds a destination buffer");
    }
  }
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() != cudaSuccess) {
    return CudaRuntimeStatus(device.status(), "Select CUDA batch copy device");
  }
  std::lock_guard lock(state_->submission_mutex);
  cudaError_t error = cudaSuccess;
  bool enqueued = false;
  for (const CudaHostToDeviceCopy& copy : copies) {
    if (copy.size_bytes == 0) continue;
    CudaBuffer* destination = AsCudaBuffer(*copy.destination);
    auto* pointer = static_cast<std::byte*>(destination->pointer()) +
                    copy.destination_offset_bytes;
    error = cudaMemcpyAsync(pointer, copy.source, copy.size_bytes,
                            cudaMemcpyHostToDevice, state_->stream);
    if (error != cudaSuccess) break;
    enqueued = true;
  }
  const cudaError_t enqueue_error = error;
  if (enqueued) {
    const cudaError_t completion_error = cudaStreamSynchronize(state_->stream);
    if (error == cudaSuccess) error = completion_error;
  }
  return CudaRuntimeStatus(
      enqueue_error != cudaSuccess ? enqueue_error : error,
      "Copy host batch to CUDA buffers");
}

Status CudaBackend::CopyDeviceToHost(
  const DeviceBuffer& src,
  void* dst,
  size_t size_bytes,
  size_t src_offset_bytes) {
  if (dst == nullptr && size_bytes != 0) {
    return Status::InvalidArgument("Host destination pointer is null");
  }
  const CudaBuffer* cuda_src = AsCudaBuffer(src);
  if (cuda_src == nullptr) {
    return Status::InvalidArgument("Source is not a CUDA buffer");
  }
  if (!owns(src) || cuda_src->state() != state_.get()) {
    return Status::InvalidArgument("Source belongs to another CUDA backend");
  }
  if (src_offset_bytes > src.size_bytes() ||
      size_bytes > src.size_bytes() - src_offset_bytes) {
    return Status::InvalidArgument(
      "Device to host copy exceeds source buffer");
  }
  if (size_bytes == 0) {
    return Status::Ok();
  }
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() != cudaSuccess) {
    return CudaRuntimeStatus(device.status(), "Select CUDA copy device");
  }
  const auto* pointer = static_cast<const std::byte*>(cuda_src->pointer()) +
    src_offset_bytes;
  std::lock_guard lock(state_->submission_mutex);
  cudaError_t error = cudaMemcpyAsync(
    dst, pointer, size_bytes, cudaMemcpyDeviceToHost, state_->stream);
  if (error == cudaSuccess) {
    error = cudaStreamSynchronize(state_->stream);
  }
  return CudaRuntimeStatus(error, "Copy CUDA buffer to host");
}

Status CudaBackend::CopyDeviceToHostBatch(
    std::span<const CudaDeviceToHostCopy> copies) {
  for (const CudaDeviceToHostCopy& copy : copies) {
    if (copy.source == nullptr ||
        (copy.destination == nullptr && copy.size_bytes != 0)) {
      return Status::InvalidArgument(
          "CUDA device-to-host batch contains a null pointer");
    }
    const CudaBuffer* source = AsCudaBuffer(*copy.source);
    if (source == nullptr || !owns(*copy.source) ||
        source->state() != state_.get()) {
      return Status::InvalidArgument(
          "CUDA device-to-host batch source is not owned");
    }
    if (copy.source_offset_bytes > copy.source->size_bytes() ||
        copy.size_bytes >
            copy.source->size_bytes() - copy.source_offset_bytes) {
      return Status::InvalidArgument(
          "CUDA device-to-host batch exceeds a source buffer");
    }
    if (copy.size_bytes == 0 || copy.row_count == 0) continue;
    const size_t source_pitch = copy.source_row_stride_bytes == 0
      ? copy.size_bytes : copy.source_row_stride_bytes;
    const size_t destination_pitch = copy.destination_row_stride_bytes == 0
      ? copy.size_bytes : copy.destination_row_stride_bytes;
    if (source_pitch < copy.size_bytes || destination_pitch < copy.size_bytes ||
        copy.row_count - 1 >
          (copy.source->size_bytes() - copy.source_offset_bytes - copy.size_bytes) /
            source_pitch ||
        copy.row_count - 1 >
          (std::numeric_limits<size_t>::max() - copy.size_bytes) /
            destination_pitch) {
      return Status::InvalidArgument(
          "CUDA device-to-host batch row layout is invalid");
    }
  }
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() != cudaSuccess) {
    return CudaRuntimeStatus(device.status(), "Select CUDA batch copy device");
  }
  std::lock_guard lock(state_->submission_mutex);
  cudaError_t error = cudaSuccess;
  bool enqueued = false;
  for (const CudaDeviceToHostCopy& copy : copies) {
    if (copy.size_bytes == 0 || copy.row_count == 0) continue;
    const CudaBuffer* source = AsCudaBuffer(*copy.source);
    const auto* pointer = static_cast<const std::byte*>(source->pointer()) +
                          copy.source_offset_bytes;
    const size_t source_pitch = copy.source_row_stride_bytes == 0
      ? copy.size_bytes : copy.source_row_stride_bytes;
    const size_t destination_pitch = copy.destination_row_stride_bytes == 0
      ? copy.size_bytes : copy.destination_row_stride_bytes;
    error = copy.row_count == 1 ||
              (source_pitch == copy.size_bytes && destination_pitch == copy.size_bytes)
      ? cudaMemcpyAsync(copy.destination, pointer, copy.row_count * copy.size_bytes,
                        cudaMemcpyDeviceToHost, state_->stream)
      : cudaMemcpy2DAsync(copy.destination, destination_pitch, pointer, source_pitch,
                          copy.size_bytes, copy.row_count, cudaMemcpyDeviceToHost,
                          state_->stream);
    if (error != cudaSuccess) break;
    enqueued = true;
  }
  const cudaError_t enqueue_error = error;
  if (enqueued) {
    const cudaError_t completion_error = cudaStreamSynchronize(state_->stream);
    if (error == cudaSuccess) error = completion_error;
  }
  return CudaRuntimeStatus(
      enqueue_error != cudaSuccess ? enqueue_error : error,
      "Copy CUDA buffer batch to host");
}

Status CudaBackend::SubmitCompute(
  EncodeCallback encode,
  const void* context,
  std::unique_ptr<GpuSubmission>* submission) {
  if (submission == nullptr) {
    return Status::InvalidArgument("CUDA submission output pointer is null");
  }
  submission->reset();
  if (encode == nullptr) {
    return Status::Internal("CUDA submission callback is null");
  }
  if (test_fail_submission_ ||
      fail_next_submission_.exchange(false, std::memory_order_relaxed)) {
    return Status::SubmissionFailed("Injected CUDA submission failure");
  }

  ScopedCudaDevice device(state_->ordinal);
  if (device.status() != cudaSuccess) {
    return CudaRuntimeStatus(device.status(), "Select CUDA submission device");
  }
  cudaEvent_t event = nullptr;
  cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (error != cudaSuccess) {
    return CudaRuntimeStatus(error, "Create CUDA completion event");
  }
  std::unique_ptr<CudaSubmission> pending;
  try {
    pending.reset(new CudaSubmission(
      state_, event,
      test_fail_completion_ ||
        fail_next_completion_.exchange(false, std::memory_order_relaxed)));
  } catch (const std::bad_alloc&) {
    (void)cudaEventDestroy(event);
    return Status::OutOfMemory("Allocate CUDA submission owner");
  }

  {
    std::lock_guard lock(state_->submission_mutex);
    error = encode(*this, context);
    if (error == cudaSuccess) {
      error = cudaEventRecord(event, state_->stream);
    }
  }
  if (error != cudaSuccess) {
    return CudaStatus(
      error, "Submit CUDA compute sequence", StatusCode::kSubmissionFailed);
  }
  RecordCommittedSubmission();
  *submission = std::move(pending);
  return Status::Ok();
}

bool CudaBackend::IsSupportedDct(AcStrategyType strategy) noexcept {
  switch (strategy) {
    case AcStrategyType::kDct8:
    case AcStrategyType::kDct16x16:
    case AcStrategyType::kDct32x32:
    case AcStrategyType::kDct16x8:
    case AcStrategyType::kDct8x16:
    case AcStrategyType::kDct32x16:
    case AcStrategyType::kDct16x32:
    case AcStrategyType::kDct64x32:
    case AcStrategyType::kDct32x64:
      return true;
    default:
      return false;
  }
}

cudaError_t CudaBackend::EncodeTransform(
  CudaBackend& backend,
  const void* context) {
  const auto& transform = *static_cast<const TransformContext*>(context);
  return LaunchCudaDct(
    transform.forward,
    static_cast<const float*>(transform.input->pointer()),
    static_cast<float*>(transform.output->pointer()),
    transform.transform_count,
    transform.width,
    transform.height,
    backend.state_->stream);
}

Status CudaBackend::SubmitTransform(
  bool forward,
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {
  if (submission == nullptr) {
    return Status::InvalidArgument(
      "Transform submission output pointer is null");
  }
  submission->reset();
  const AcStrategyInfo* info = GetAcStrategyInfo(batch.strategy);
  if (info == nullptr) {
    return Status::InvalidArgument("Unknown JPEG XL AC strategy");
  }
  if (!IsSupportedDct(batch.strategy)) {
    return Status::Unavailable(
      std::string("CUDA backend does not support ") +
      std::string(info->name));
  }
  if (batch.transform_count == 0) {
    return Status::Ok();
  }
  if (batch.input == nullptr || batch.output == nullptr) {
    return Status::InvalidArgument("Transform input/output buffer is null");
  }
  if (batch.input == batch.output) {
    return Status::InvalidArgument("In-place transforms are not supported yet");
  }
  const size_t elements = info->coefficient_count();
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return Status::Internal("Transform coefficient count is too large");
  }
  const size_t bytes_per_transform = elements * sizeof(float);
  if (batch.transform_count >
      std::numeric_limits<size_t>::max() / bytes_per_transform) {
    return Status::InvalidArgument("Transform batch is too large");
  }
  const size_t required_bytes = batch.transform_count * bytes_per_transform;
  if (batch.input->size_bytes() < required_bytes ||
      batch.output->size_bytes() < required_bytes) {
    return Status::InvalidArgument("Transform buffer is too small");
  }
  const CudaBuffer* input = AsCudaBuffer(*batch.input);
  CudaBuffer* output = AsCudaBuffer(*batch.output);
  if (input == nullptr || output == nullptr) {
    return Status::InvalidArgument("Transform buffers are not CUDA buffers");
  }
  if (!owns(*batch.input) || !owns(*batch.output) ||
      input->state() != state_.get() || output->state() != state_.get()) {
    return Status::InvalidArgument(
      "Transform buffer belongs to another CUDA backend");
  }
  if (batch.transform_count > state_->maximum_grid_x ||
      batch.transform_count > std::numeric_limits<unsigned int>::max()) {
    return Status::InvalidArgument("Transform batch exceeds CUDA grid range");
  }

  const Extent2D extent = info->pixel_extent();
  const TransformContext context{
    forward,
    input,
    output,
    batch.transform_count,
    static_cast<unsigned int>(extent.width),
    static_cast<unsigned int>(extent.height),
  };
  return SubmitCompute(&CudaBackend::EncodeTransform, &context, submission);
}

Status CudaBackend::ForwardTransform(
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {
  return SubmitTransform(true, batch, submission);
}

Status CudaBackend::InverseTransform(
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {
  return SubmitTransform(false, batch, submission);
}

void CudaBackend::ArmNextSubmissionFailureForTest(
  bool fail_submission,
  bool fail_completion) noexcept {
  fail_next_submission_.store(fail_submission, std::memory_order_relaxed);
  fail_next_completion_.store(fail_completion, std::memory_order_relaxed);
}

}  // namespace gjxl::cuda_internal

namespace gjxl {

Status CreateCudaBackend(
  const CudaBackendOptions& options,
  std::unique_ptr<GpuBackend>* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("CreateCudaBackend output pointer is null");
  }
  out->reset();
  if (options.device_ordinal < 0) {
    return Status::InvalidArgument("CUDA device ordinal is negative");
  }

  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    return cuda_internal::CudaRuntimeStatus(error, "Enumerate CUDA devices");
  }
  if (device_count == 0) {
    return Status::Unavailable("No CUDA device is available");
  }
  if (options.device_ordinal >= device_count) {
    return Status::InvalidArgument("CUDA device ordinal is out of range");
  }

  cuda_internal::ScopedCudaDevice selected(options.device_ordinal);
  if (selected.status() != cudaSuccess) {
    return cuda_internal::CudaRuntimeStatus(
      selected.status(), "Select CUDA backend device");
  }
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, options.device_ordinal);
  if (error != cudaSuccess) {
    return cuda_internal::CudaRuntimeStatus(
      error, "Query CUDA device properties");
  }
  cudaStream_t stream = nullptr;
  error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    return cuda_internal::CudaRuntimeStatus(error, "Create CUDA stream");
  }
  error = cuda_internal::InitializeCudaDctBasis();
  if (error != cudaSuccess) {
    (void)cudaStreamDestroy(stream);
    return cuda_internal::CudaRuntimeStatus(error, "Initialize CUDA DCT basis");
  }

  try {
    auto state = std::make_shared<cuda_internal::CudaDeviceState>();
    state->ordinal = options.device_ordinal;
    state->stream = std::exchange(stream, nullptr);
    state->maximum_grid_x = static_cast<size_t>(properties.maxGridSize[0]);
    state->maximum_threads_per_block =
      static_cast<size_t>(properties.maxThreadsPerBlock);
    if (options.use_stream_ordered_allocation) {
      const uint64_t threshold =
          options.memory_pool_release_threshold_bytes.value_or(
              std::min<uint64_t>(properties.totalGlobalMem / 2,
                                 uint64_t{4} << 30));
      const Status status = cuda_internal::AcquireMemoryPool(
          options.device_ordinal, threshold, &state->memory_pool);
      if (!status.ok()) return status;
    }
    std::string name = "CUDA";
    if (properties.name[0] != '\0') {
      name += ": ";
      name += properties.name;
    }
    out->reset(new cuda_internal::CudaBackend(
      std::move(state), std::move(name),
      options.test_fail_submission, options.test_fail_completion));
  } catch (const std::bad_alloc&) {
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
    return Status::OutOfMemory("Allocate CUDA backend");
  }
  return Status::Ok();
}

Status TrimCudaDeviceMemory(int device_ordinal) {
  if (device_ordinal < 0)
    return Status::InvalidArgument("CUDA trim device ordinal is negative");
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess)
    return cuda_internal::CudaRuntimeStatus(error, "Enumerate CUDA trim devices");
  if (device_ordinal >= count)
    return Status::InvalidArgument("CUDA trim device ordinal is out of range");
  cuda_internal::ScopedCudaDevice selected(device_ordinal);
  if (selected.status() != cudaSuccess)
    return cuda_internal::CudaRuntimeStatus(selected.status(),
                                            "Select CUDA trim device");
  return cuda_internal::TrimMemoryPools(device_ordinal);
}

Status ArmNextCudaSubmissionFailureForTest(
  GpuBackend& backend,
  bool fail_submission,
  bool fail_completion) {
  auto* cuda = dynamic_cast<cuda_internal::CudaBackend*>(&backend);
  if (cuda == nullptr) {
    return Status::InvalidArgument(
      "Submission failure injection requires a CUDA backend");
  }
  cuda->ArmNextSubmissionFailureForTest(fail_submission, fail_completion);
  return Status::Ok();
}

}  // namespace gjxl
