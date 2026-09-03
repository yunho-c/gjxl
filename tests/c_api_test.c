// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gjxl/gjxl.h"

#define CHECK(condition, message)                    \
  do {                                               \
    if (!(condition)) {                              \
      fprintf(stderr, "C API test: %s\n", message); \
      return 0;                                      \
    }                                                \
  } while (0)

static int ExpectError(GJXLResult actual, GJXLResult expected,
                       const char* name) {
  if (actual != expected) {
    fprintf(stderr, "%s: expected result %d, got %d (%s)\n", name,
            (int)expected, (int)actual, gjxl_get_last_error());
    return 0;
  }
  if (gjxl_get_last_error()[0] == '\0') {
    fprintf(stderr, "%s: missing diagnostic\n", name);
    return 0;
  }
  return 1;
}

static int CheckInitializers(void) {
  CHECK(GJXL_MAX_CPU_THREADS == 256,
        "maximum CPU thread count is incorrect");
  GJXLContextOptions context_options;
  CHECK(gjxl_context_options_init(&context_options,
                                  sizeof(context_options)) == GJXL_OK,
        "context initializer failed");
  CHECK(context_options.struct_size == sizeof(context_options) &&
          context_options.backend == GJXL_BACKEND_AUTO &&
          context_options.num_cpu_threads == 0,
        "context defaults are incorrect");

  GJXLEncoderOptions encoder_options;
  CHECK(gjxl_encoder_options_init(&encoder_options,
                                  sizeof(encoder_options)) == GJXL_OK,
        "encoder initializer failed");
  CHECK(encoder_options.struct_size == sizeof(encoder_options) &&
          encoder_options.distance == 1.0f && encoder_options.effort == 7 &&
          encoder_options.compression_mode == GJXL_COMPRESSION_AUTOMATIC,
        "encoder defaults are incorrect");

  GJXLEncoderOptions legacy_encoder;
  memset(&legacy_encoder, 0xa5, sizeof(legacy_encoder));
  CHECK(gjxl_encoder_options_init(
          &legacy_encoder,
          offsetof(GJXLEncoderOptions, compression_mode)) == GJXL_OK,
        "legacy encoder initializer failed");
  CHECK(legacy_encoder.struct_size ==
          offsetof(GJXLEncoderOptions, compression_mode) &&
          legacy_encoder.distance == 1.0f && legacy_encoder.effort == 7 &&
          legacy_encoder.compression_mode == (int32_t)0xa5a5a5a5,
        "legacy encoder defaults or trailing storage are incorrect");

  GJXLContextOptions small_context;
  GJXLContextOptions original_context;
  memset(&small_context, 0xa5, sizeof(small_context));
  memcpy(&original_context, &small_context, sizeof(small_context));
  CHECK(ExpectError(
          gjxl_context_options_init(&small_context,
                                    offsetof(GJXLContextOptions,
                                             num_cpu_threads) - 1),
          GJXL_ERROR_INVALID_ARGUMENT, "small context initializer"),
        "small context initializer result is incorrect");
  CHECK(memcmp(&small_context, &original_context, sizeof(small_context)) == 0,
        "small context initializer modified storage");

  GJXLEncoderOptions small_encoder;
  GJXLEncoderOptions original_encoder;
  memset(&small_encoder, 0x5a, sizeof(small_encoder));
  memcpy(&original_encoder, &small_encoder, sizeof(small_encoder));
  CHECK(ExpectError(
          gjxl_encoder_options_init(&small_encoder,
                                    offsetof(GJXLEncoderOptions,
                                             compression_mode) - 1),
          GJXL_ERROR_INVALID_ARGUMENT, "small encoder initializer"),
        "small encoder initializer result is incorrect");
  CHECK(memcmp(&small_encoder, &original_encoder, sizeof(small_encoder)) == 0,
        "small encoder initializer modified storage");

  struct LargerContextOptions {
    GJXLContextOptions options;
    uint64_t future_field;
  } larger_context;
  memset(&larger_context, 0xa5, sizeof(larger_context));
  CHECK(gjxl_context_options_init(&larger_context.options,
                                  sizeof(larger_context)) == GJXL_OK,
        "larger context initializer failed");
  CHECK(larger_context.options.struct_size == sizeof(larger_context) &&
          larger_context.options.backend == GJXL_BACKEND_AUTO &&
          larger_context.options.num_cpu_threads == 0 &&
          larger_context.future_field == 0,
        "larger context initializer did not zero future storage");

  struct LargerEncoderOptions {
    GJXLEncoderOptions options;
    uint64_t future_field;
  } larger_encoder;
  memset(&larger_encoder, 0xa5, sizeof(larger_encoder));
  CHECK(gjxl_encoder_options_init(&larger_encoder.options,
                                  sizeof(larger_encoder)) == GJXL_OK,
        "larger encoder initializer failed");
  CHECK(larger_encoder.options.struct_size == sizeof(larger_encoder) &&
          larger_encoder.options.distance == 1.0f &&
          larger_encoder.options.effort == 7 &&
          larger_encoder.options.compression_mode ==
            GJXL_COMPRESSION_AUTOMATIC &&
          larger_encoder.future_field == 0,
        "larger encoder initializer did not zero future storage");

  CHECK(ExpectError(gjxl_context_options_init(NULL, 0),
                    GJXL_ERROR_INVALID_ARGUMENT,
                    "null context initializer"),
        "null context initializer result is incorrect");
  CHECK(ExpectError(gjxl_encoder_options_init(NULL, 0),
                    GJXL_ERROR_INVALID_ARGUMENT,
                    "null encoder initializer"),
        "null encoder initializer result is incorrect");

#if SIZE_MAX > UINT32_MAX
  memset(&context_options, 0xa5, sizeof(context_options));
  memcpy(&original_context, &context_options, sizeof(context_options));
  CHECK(ExpectError(
          gjxl_context_options_init(
            &context_options, (size_t)UINT32_MAX + (size_t)1),
          GJXL_ERROR_INVALID_ARGUMENT, "unrepresentable caller size"),
        "unrepresentable caller size result is incorrect");
  CHECK(memcmp(&context_options, &original_context,
               sizeof(context_options)) == 0,
        "unrepresentable caller size modified storage");
#endif

  CHECK(gjxl_encoder_options_init(&encoder_options,
                                  sizeof(encoder_options)) == GJXL_OK &&
          gjxl_get_last_error()[0] == '\0',
        "successful call did not clear the diagnostic");
  return 1;
}

static int CheckQualityHelper(void) {
  CHECK(gjxl_distance_from_quality(100.0f) == 0.0f,
        "quality 100 mapping is incorrect");
  CHECK(fabsf(gjxl_distance_from_quality(90.0f) - 1.0f) < 1.0e-6f,
        "quality 90 mapping is incorrect");
  CHECK(fabsf(gjxl_distance_from_quality(80.0f) - 1.9f) < 1.0e-6f,
        "quality 80 mapping is incorrect");
  CHECK(fabsf(gjxl_distance_from_quality(30.0f) - 6.4f) < 1.0e-6f,
        "quality 30 mapping is incorrect");
  CHECK(fabsf(gjxl_distance_from_quality(0.0f) - 25.0f) < 1.0e-6f,
        "quality 0 mapping is incorrect");
  CHECK(isnan(gjxl_distance_from_quality(NAN)),
        "NaN quality mapping is incorrect");
  return 1;
}

static int CheckContexts(GJXLContext** cpu_context) {
  GJXLContext* automatic = NULL;
  CHECK(gjxl_context_create(NULL, &automatic) == GJXL_OK &&
          automatic != NULL,
        "default automatic context creation failed");
  gjxl_context_destroy(automatic);

  CHECK(ExpectError(gjxl_context_create(NULL, NULL),
                    GJXL_ERROR_INVALID_ARGUMENT,
                    "null context output"),
        "null context output result is incorrect");

  GJXLContext* occupied = (GJXLContext*)(uintptr_t)1;
  CHECK(ExpectError(gjxl_context_create(NULL, &occupied),
                    GJXL_ERROR_INVALID_ARGUMENT,
                    "occupied context output"),
        "occupied context output result is incorrect");
  CHECK(occupied == (GJXLContext*)(uintptr_t)1,
        "occupied context output was modified");

  GJXLContextOptions options;
  memset(&options, 0xa5, sizeof(options));
  CHECK(gjxl_context_options_init(
          &options, offsetof(GJXLContextOptions, num_cpu_threads)) == GJXL_OK,
        "legacy context options initialization failed");
  CHECK(options.struct_size == offsetof(GJXLContextOptions, num_cpu_threads) &&
          options.backend == GJXL_BACKEND_AUTO,
        "legacy context options defaults are incorrect");
  GJXLContext* legacy = NULL;
  CHECK(gjxl_context_create(&options, &legacy) == GJXL_OK && legacy != NULL,
        "legacy context options creation failed");
  gjxl_context_destroy(legacy);

  CHECK(gjxl_context_options_init(&options, sizeof(options)) == GJXL_OK,
        "context options initialization failed");
  options.struct_size = offsetof(GJXLContextOptions, num_cpu_threads) - 1;
  CHECK(ExpectError(gjxl_context_create(&options, cpu_context),
                    GJXL_ERROR_INVALID_ARGUMENT,
                    "small context options"),
        "small context options result is incorrect");
  CHECK(*cpu_context == NULL, "small context options modified output");

  options.struct_size = sizeof(options);
  options.backend = 99;
  CHECK(ExpectError(gjxl_context_create(&options, cpu_context),
                    GJXL_ERROR_INVALID_ARGUMENT,
                    "unknown context backend"),
        "unknown context backend result is incorrect");
  CHECK(*cpu_context == NULL, "unknown backend modified context output");

  options.backend = GJXL_BACKEND_CPU;
  options.num_cpu_threads = 257;
  CHECK(ExpectError(gjxl_context_create(&options, cpu_context),
                    GJXL_ERROR_INVALID_ARGUMENT,
                    "large CPU thread count"),
        "large CPU thread count result is incorrect");
  CHECK(*cpu_context == NULL, "large CPU thread count modified context output");
  options.num_cpu_threads = 0;

  options.backend = GJXL_BACKEND_METAL;
  GJXLContext* metal = NULL;
  const GJXLResult metal_result = gjxl_context_create(&options, &metal);
  CHECK(metal_result == GJXL_OK ||
          metal_result == GJXL_ERROR_UNAVAILABLE,
        "forced Metal returned the wrong result category");
  if (metal_result == GJXL_OK) {
    CHECK(metal != NULL, "forced Metal returned no context");
    gjxl_context_destroy(metal);
  } else {
    CHECK(metal == NULL && gjxl_get_last_error()[0] != '\0',
          "unavailable Metal modified output or omitted a diagnostic");
  }

  options.backend = GJXL_BACKEND_CUDA;
  GJXLContext* cuda = NULL;
  const GJXLResult cuda_result = gjxl_context_create(&options, &cuda);
  CHECK(cuda_result == GJXL_OK || cuda_result == GJXL_ERROR_UNAVAILABLE,
        "forced CUDA returned the wrong result category");
  if (cuda_result == GJXL_OK) {
    CHECK(cuda != NULL, "forced CUDA returned no context");
    gjxl_context_destroy(cuda);
  } else {
    CHECK(cuda == NULL && gjxl_get_last_error()[0] != '\0',
          "unavailable CUDA modified output or omitted a diagnostic");
  }

  struct LargerContextOptions {
    GJXLContextOptions options;
    uint64_t future_field;
  } larger_options;
  CHECK(gjxl_context_options_init(&larger_options.options,
                                  sizeof(larger_options)) == GJXL_OK,
        "larger CPU context options initialization failed");
  larger_options.options.backend = GJXL_BACKEND_CPU;
  larger_options.options.num_cpu_threads = 2;
  larger_options.future_field = UINT64_MAX;
  CHECK(gjxl_context_create(&larger_options.options, cpu_context) == GJXL_OK &&
          *cpu_context != NULL,
        "larger CPU context creation failed");
  return 1;
}

static int ExpectEncodeError(GJXLContext* context,
                             const GJXLImageView* image,
                             const GJXLEncoderOptions* options,
                             GJXLResult expected,
                             const char* name) {
  GJXLBuffer output = {NULL, 0};
  const GJXLResult result = gjxl_encode(context, image, options, &output);
  if (!ExpectError(result, expected, name)) {
    return 0;
  }
  if (output.data != NULL || output.size != 0) {
    fprintf(stderr, "%s: failure modified output\n", name);
    return 0;
  }
  return 1;
}

static int CheckEncoding(GJXLContext* context) {
  enum {
    kWidth = 17,
    kHeight = 9,
    kRgbStride = kWidth * 3 + 5,
    kRgbaStride = kWidth * 4 + 3,
  };
  uint8_t rgb[kRgbStride * kHeight];
  uint8_t rgba[kRgbaStride * kHeight];
  memset(rgb, 0xa5, sizeof(rgb));
  memset(rgba, 0, sizeof(rgba));
  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      const uint8_t red = (uint8_t)((17 * x + 3 * y) & 0xff);
      const uint8_t green = (uint8_t)((5 * x + 29 * y) & 0xff);
      const uint8_t blue = (uint8_t)((11 * x + 7 * y + 31) & 0xff);
      rgb[y * kRgbStride + 3 * x] = red;
      rgb[y * kRgbStride + 3 * x + 1] = green;
      rgb[y * kRgbStride + 3 * x + 2] = blue;
      rgba[y * kRgbaStride + 4 * x] = red;
      rgba[y * kRgbaStride + 4 * x + 1] = green;
      rgba[y * kRgbaStride + 4 * x + 2] = blue;
      rgba[y * kRgbaStride + 4 * x + 3] = 255;
    }
  }

  GJXLEncoderOptions options;
  CHECK(gjxl_encoder_options_init(&options, sizeof(options)) == GJXL_OK,
        "encoder options initialization failed");
  GJXLImageView rgb_view = {
    sizeof(GJXLImageView), kWidth, kHeight,
    GJXL_PIXEL_FORMAT_RGB8_SRGB, rgb, sizeof(rgb), kRgbStride,
  };
  GJXLImageView rgba_view = {
    sizeof(GJXLImageView), kWidth, kHeight,
    GJXL_PIXEL_FORMAT_RGBA8_SRGB, rgba, sizeof(rgba), kRgbaStride,
  };

  GJXLBuffer rgb_output = {NULL, 0};
  CHECK(gjxl_encode(context, &rgb_view, &options, &rgb_output) == GJXL_OK,
        "RGB encoding failed");
  CHECK(rgb_output.data != NULL && rgb_output.size >= 2 &&
          rgb_output.data[0] == 0xff && rgb_output.data[1] == 0x0a &&
          gjxl_get_last_error()[0] == '\0',
        "RGB encoding returned an invalid codestream");

  GJXLEncoderOptions legacy_options = options;
  legacy_options.struct_size =
    offsetof(GJXLEncoderOptions, compression_mode);
  legacy_options.compression_mode = 99;
  GJXLBuffer legacy_output = {NULL, 0};
  CHECK(gjxl_encode(context, &rgb_view, &legacy_options,
                    &legacy_output) == GJXL_OK,
        "legacy encoder options were not accepted");
  CHECK(legacy_output.size == rgb_output.size &&
          memcmp(legacy_output.data, rgb_output.data, rgb_output.size) == 0,
        "legacy encoder options did not use automatic compression");
  gjxl_buffer_free(&legacy_output);

  options.compression_mode = GJXL_COMPRESSION_MAXIMUM;
  GJXLBuffer maximum_output = {NULL, 0};
  CHECK(gjxl_encode(context, &rgb_view, &options, &maximum_output) == GJXL_OK,
        "maximum-compression encoding failed");
  CHECK(maximum_output.data != NULL && maximum_output.size >= 2 &&
          maximum_output.data[0] == 0xff && maximum_output.data[1] == 0x0a,
        "maximum-compression encoding returned an invalid codestream");
  gjxl_buffer_free(&maximum_output);
  options.compression_mode = GJXL_COMPRESSION_AUTOMATIC;

  struct LargerEncoderOptions {
    GJXLEncoderOptions options;
    uint64_t future_field;
  } larger_options;
  CHECK(gjxl_encoder_options_init(&larger_options.options,
                                  sizeof(larger_options)) == GJXL_OK,
        "larger encoder options initialization failed");
  larger_options.future_field = UINT64_MAX;
  struct LargerImageView {
    GJXLImageView image;
    uint64_t future_field;
  } larger_image;
  larger_image.image = rgba_view;
  larger_image.image.struct_size = sizeof(larger_image);
  larger_image.future_field = UINT64_MAX;
  GJXLBuffer rgba_output = {NULL, 0};
  CHECK(gjxl_encode(context, &larger_image.image, &larger_options.options,
                    &rgba_output) == GJXL_OK,
        "larger opaque RGBA encoding failed");
  CHECK(rgba_output.size == rgb_output.size &&
          memcmp(rgba_output.data, rgb_output.data, rgb_output.size) == 0,
        "RGB and opaque RGBA did not encode identically");
  gjxl_buffer_free(&rgb_output);
  gjxl_buffer_free(&rgba_output);
  CHECK(rgb_output.data == NULL && rgb_output.size == 0 &&
          rgba_output.data == NULL && rgba_output.size == 0,
        "buffer free did not reset outputs");
  gjxl_buffer_free(&rgb_output);
  gjxl_buffer_free(NULL);

  uint8_t marker = 7;
  GJXLBuffer occupied = {&marker, 1};
  CHECK(ExpectError(gjxl_encode(context, &rgb_view, &options, &occupied),
                    GJXL_ERROR_INVALID_ARGUMENT,
                    "occupied encode output"),
        "occupied encode output result is incorrect");
  CHECK(occupied.data == &marker && occupied.size == 1,
        "occupied encode output was modified");

  rgba[4 * 5 + 3] = 254;
  CHECK(ExpectEncodeError(context, &rgba_view, &options,
                          GJXL_ERROR_UNSUPPORTED,
                          "non-opaque alpha"),
        "non-opaque alpha result is incorrect");
  rgba[4 * 5 + 3] = 255;

  options.distance = 0.0f;
  CHECK(ExpectEncodeError(context, &rgb_view, &options,
                          GJXL_ERROR_UNSUPPORTED, "distance zero"),
        "distance-zero result is incorrect");
  options.distance = -1.0f;
  CHECK(ExpectEncodeError(context, &rgb_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "negative distance"),
        "negative-distance result is incorrect");
  options.distance = NAN;
  CHECK(ExpectEncodeError(context, &rgb_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "NaN distance"),
        "NaN-distance result is incorrect");
  options.distance = 25.1f;
  CHECK(ExpectEncodeError(context, &rgb_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "large distance"),
        "large-distance result is incorrect");
  options.distance = 1.0f;
  options.effort = 0;
  CHECK(ExpectEncodeError(context, &rgb_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "effort zero"),
        "effort-zero result is incorrect");
  options.effort = 11;
  CHECK(ExpectEncodeError(context, &rgb_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "effort eleven"),
        "effort-eleven result is incorrect");
  options.effort = 7;
  options.compression_mode = 99;
  CHECK(ExpectEncodeError(context, &rgb_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT,
                          "unknown compression mode"),
        "unknown-compression-mode result is incorrect");
  options.compression_mode = GJXL_COMPRESSION_AUTOMATIC;

  GJXLImageView invalid_view = rgb_view;
  invalid_view.struct_size = sizeof(invalid_view) - 1;
  CHECK(ExpectEncodeError(context, &invalid_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "small image view"),
        "small-image-view result is incorrect");
  invalid_view = rgb_view;
  invalid_view.width = 0;
  CHECK(ExpectEncodeError(context, &invalid_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "zero width"),
        "zero-width result is incorrect");
  invalid_view = rgb_view;
  invalid_view.pixel_format = 99;
  CHECK(ExpectEncodeError(context, &invalid_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT,
                          "unknown pixel format"),
        "unknown-pixel-format result is incorrect");
  invalid_view = rgb_view;
  invalid_view.pixels = NULL;
  CHECK(ExpectEncodeError(context, &invalid_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "null pixels"),
        "null-pixels result is incorrect");
  invalid_view = rgb_view;
  invalid_view.row_stride_bytes = kWidth * 3 - 1;
  CHECK(ExpectEncodeError(context, &invalid_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "short stride"),
        "short-stride result is incorrect");
  invalid_view = rgb_view;
  invalid_view.pixels_size = kWidth * 3 - 1;
  CHECK(ExpectEncodeError(context, &invalid_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "short buffer"),
        "short-buffer result is incorrect");
  invalid_view = rgb_view;
  invalid_view.width = 1;
  invalid_view.height = 2;
  invalid_view.row_stride_bytes = SIZE_MAX;
  invalid_view.pixels_size = SIZE_MAX;
  CHECK(ExpectEncodeError(context, &invalid_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT,
                          "image size overflow"),
        "image-overflow result is incorrect");

  GJXLEncoderOptions small_options = options;
  small_options.struct_size =
    offsetof(GJXLEncoderOptions, compression_mode) - 1;
  CHECK(ExpectEncodeError(context, &rgb_view, &small_options,
                          GJXL_ERROR_INVALID_ARGUMENT,
                          "small encoder options"),
        "small-encoder-options result is incorrect");

  CHECK(ExpectEncodeError(NULL, &rgb_view, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "null context"),
        "null-context result is incorrect");
  CHECK(ExpectEncodeError(context, NULL, &options,
                          GJXL_ERROR_INVALID_ARGUMENT, "null image"),
        "null-image result is incorrect");
  CHECK(ExpectEncodeError(context, &rgb_view, NULL,
                          GJXL_ERROR_INVALID_ARGUMENT, "null options"),
        "null-options result is incorrect");
  CHECK(ExpectError(gjxl_encode(context, &rgb_view, &options, NULL),
                    GJXL_ERROR_INVALID_ARGUMENT, "null output"),
        "null-output result is incorrect");
  return 1;
}

int main(void) {
  if (!CheckInitializers() || !CheckQualityHelper()) {
    return EXIT_FAILURE;
  }
  GJXLContext* cpu_context = NULL;
  if (!CheckContexts(&cpu_context) || !CheckEncoding(cpu_context)) {
    gjxl_context_destroy(cpu_context);
    return EXIT_FAILURE;
  }
  gjxl_context_destroy(cpu_context);
  return EXIT_SUCCESS;
}
