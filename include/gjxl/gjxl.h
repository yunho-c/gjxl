// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#ifndef GJXL_GJXL_H_
#define GJXL_GJXL_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(GJXL_BUILDING_SHARED_LIBRARY)
#define GJXL_API __declspec(dllexport)
#elif defined(GJXL_USING_SHARED_LIBRARY)
#define GJXL_API __declspec(dllimport)
#else
#define GJXL_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define GJXL_API __attribute__((visibility("default")))
#else
#define GJXL_API
#endif

#ifdef __cplusplus
#define GJXL_NOEXCEPT noexcept
extern "C" {
#else
#define GJXL_NOEXCEPT
#endif

typedef struct GJXLContext GJXLContext;
typedef struct GJXLExecutionDomain GJXLExecutionDomain;

typedef int32_t GJXLResult;
enum {
  GJXL_OK = 0,
  GJXL_ERROR_INVALID_ARGUMENT = 1,
  GJXL_ERROR_UNSUPPORTED = 2,
  GJXL_ERROR_UNAVAILABLE = 3,
  GJXL_ERROR_OUT_OF_MEMORY = 4,
  GJXL_ERROR_BACKEND = 5,
  GJXL_ERROR_INTERNAL = 6,
  /// An admitted plan was exceeded; terminal, not a retryable allocation miss.
  GJXL_ERROR_RESOURCE_PLAN_EXCEEDED = 7,
};

typedef int32_t GJXLBackend;
enum {
  GJXL_BACKEND_AUTO = 0,
  GJXL_BACKEND_CPU = 1,
  GJXL_BACKEND_METAL = 2,
};

enum {
  GJXL_MAX_CPU_THREADS = 256,
};

typedef struct {
  uint32_t struct_size;
  /// Zero is unlimited but accounted. Managed capacity is not process RSS.
  uint64_t managed_memory_bytes;
  /// Aggregate executing callers and workers across this domain. Zero selects
  /// hardware concurrency (1..GJXL_MAX_CPU_THREADS); positive values must not
  /// exceed GJXL_MAX_CPU_THREADS. Per-encode limits remain upper bounds.
  uint32_t cpu_participant_limit;
} GJXLExecutionDomainOptions;

typedef struct {
  uint32_t struct_size;
  uint64_t live_requested_bytes;
  uint64_t live_capacity_bytes;
  uint64_t idle_capacity_bytes;
  uint64_t reserved_unbacked_bytes;
  uint64_t peak_backing_bytes;
  uint64_t peak_committed_bytes;
  uint64_t active_reservations;
  uint64_t waiting_requests;
  /// Memory and CPU sets are sampled separately, not as one atomic snapshot.
  uint64_t effective_cpu_participant_limit;
  uint64_t active_cpu_participants;
  /// Capacity granted before workers start, not additional active threads.
  uint64_t reserved_cpu_workers;
  /// Created workers retain protected capacity while blocked, preventing nested
  /// joins from multiplying dormant worker threads. Not active participants.
  uint64_t suspended_cpu_workers;
  uint64_t waiting_cpu_callers;
  uint64_t peak_cpu_participants;
  /// Peak active plus reserved plus suspended capacity; at most the limit.
  uint64_t peak_cpu_protected_slots;
} GJXLExecutionDomainSnapshot;

typedef struct {
  uint32_t struct_size;
  GJXLBackend backend;
  /// Maximum participating CPU threads per encode. Zero selects automatic;
  /// positive values must not exceed GJXL_MAX_CPU_THREADS.
  uint32_t num_cpu_threads;
  /// Shared immutable allowance; null selects one process-wide default.
  /// Context creation retains the domain independently of this handle.
  const GJXLExecutionDomain *execution_domain;
} GJXLContextOptions;

typedef int32_t GJXLCompressionMode;
enum {
  GJXL_COMPRESSION_AUTOMATIC = 0,
  GJXL_COMPRESSION_MAXIMUM = 1,
};

typedef struct {
  uint32_t struct_size;
  float distance;
  /// Speed/refinement intent in [1, 10]. Efforts 1-4 use DCT8 only;
  /// efforts 5-10 enable mixed-transform AC-strategy search.
  int32_t effort;
  /// Selects the entropy/codestream search policy independently of effort.
  /// Callers using the previous struct size implicitly select AUTOMATIC.
  GJXLCompressionMode compression_mode;
} GJXLEncoderOptions;

typedef int32_t GJXLPixelFormat;
enum {
  GJXL_PIXEL_FORMAT_RGB8_SRGB = 1,
  GJXL_PIXEL_FORMAT_RGBA8_SRGB = 2,
};

typedef struct {
  uint32_t struct_size;
  uint32_t width;
  uint32_t height;
  GJXLPixelFormat pixel_format;
  const void* pixels;
  size_t pixels_size;
  size_t row_stride_bytes;
} GJXLImageView;

typedef struct {
  uint8_t* data;
  size_t size;
} GJXLBuffer;

/// Initializes context options and defaults to GJXL_BACKEND_AUTO.
/// caller_size must describe the complete caller allocation and fit uint32_t.
GJXL_API GJXLResult gjxl_context_options_init(
  GJXLContextOptions* options, size_t caller_size) GJXL_NOEXCEPT;

GJXL_API GJXLResult gjxl_execution_domain_options_init(GJXLExecutionDomainOptions *options,
                                                       size_t caller_size) GJXL_NOEXCEPT;
/// Creates a domain; null options select a new unlimited explicit domain.
GJXL_API GJXLResult gjxl_execution_domain_create(const GJXLExecutionDomainOptions *options,
                                                 GJXLExecutionDomain **domain) GJXL_NOEXCEPT;
/// Releases only this handle. Contexts and active/retained storage remain valid.
GJXL_API void gjxl_execution_domain_destroy(GJXLExecutionDomain *domain) GJXL_NOEXCEPT;
/// Null domain queries the shared default. caller_size bounds the output write.
GJXL_API GJXLResult gjxl_execution_domain_snapshot(const GJXLExecutionDomain *domain,
                                                   GJXLExecutionDomainSnapshot *snapshot,
                                                   size_t caller_size) GJXL_NOEXCEPT;

/// Initializes encoder options with distance 1.0, effort 7, and automatic
/// compression behavior.
/// caller_size must describe the complete caller allocation and fit uint32_t.
GJXL_API GJXLResult gjxl_encoder_options_init(
  GJXLEncoderOptions* options, size_t caller_size) GJXL_NOEXCEPT;

/// Creates a reusable execution context. Null options select AUTO.
GJXL_API GJXLResult gjxl_context_create(
  const GJXLContextOptions* options,
  GJXLContext** context) GJXL_NOEXCEPT;

/// Releases idle AQ/resident-input and Butteraugli capacity shared by contexts and
/// batch encoders. Safe alongside encoding; does not wait for active work or
/// initialize Metal. Pre-trim active leases cannot refill these caches; leases
/// acquired afterward may cache again. Context destruction does not trim these
/// process-wide caches. Useful when the application becomes idle.
GJXL_API GJXLResult gjxl_trim_preparation_cache(void) GJXL_NOEXCEPT;

GJXL_API void gjxl_context_destroy(
  GJXLContext* context) GJXL_NOEXCEPT;

/// Encodes one packed sRGB image to a library-owned raw JPEG XL codestream.
GJXL_API GJXLResult gjxl_encode(
  GJXLContext* context,
  const GJXLImageView* image,
  const GJXLEncoderOptions* options,
  GJXLBuffer* output) GJXL_NOEXCEPT;

/// Releases an encoded buffer and resets both fields to zero.
GJXL_API void gjxl_buffer_free(GJXLBuffer* buffer) GJXL_NOEXCEPT;

/// Returns this thread's diagnostic from the most recent failing call.
GJXL_API const char* gjxl_get_last_error(void) GJXL_NOEXCEPT;

/// Maps quality in the documented range [0, 100] to JPEG XL distance.
GJXL_API float gjxl_distance_from_quality(float quality) GJXL_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GJXL_GJXL_H_
