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

typedef int32_t GJXLResult;
enum {
  GJXL_OK = 0,
  GJXL_ERROR_INVALID_ARGUMENT = 1,
  GJXL_ERROR_UNSUPPORTED = 2,
  GJXL_ERROR_UNAVAILABLE = 3,
  GJXL_ERROR_OUT_OF_MEMORY = 4,
  GJXL_ERROR_BACKEND = 5,
  GJXL_ERROR_INTERNAL = 6,
};

typedef int32_t GJXLBackend;
enum {
  GJXL_BACKEND_AUTO = 0,
  GJXL_BACKEND_CPU = 1,
  GJXL_BACKEND_METAL = 2,
};

typedef struct {
  uint32_t struct_size;
  GJXLBackend backend;
} GJXLContextOptions;

typedef struct {
  uint32_t struct_size;
  float distance;
  int32_t effort;
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

/// Initializes encoder options with distance 1.0 and effort 7.
/// caller_size must describe the complete caller allocation and fit uint32_t.
GJXL_API GJXLResult gjxl_encoder_options_init(
  GJXLEncoderOptions* options, size_t caller_size) GJXL_NOEXCEPT;

/// Creates a reusable execution context. Null options select AUTO.
GJXL_API GJXLResult gjxl_context_create(
  const GJXLContextOptions* options,
  GJXLContext** context) GJXL_NOEXCEPT;

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
