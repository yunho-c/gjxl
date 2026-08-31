// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <gjxl/gjxl.h>

int main(void) {
  GJXLContextOptions context_options;
  if (gjxl_context_options_init(
        &context_options, sizeof(context_options)) != GJXL_OK) {
    return EXIT_FAILURE;
  }
  context_options.backend = GJXL_BACKEND_CPU;

  GJXLContext* context = NULL;
  if (gjxl_context_create(&context_options, &context) != GJXL_OK ||
      context == NULL) {
    return EXIT_FAILURE;
  }

  GJXLEncoderOptions encoder_options;
  if (gjxl_encoder_options_init(
        &encoder_options, sizeof(encoder_options)) != GJXL_OK) {
    gjxl_context_destroy(context);
    return EXIT_FAILURE;
  }
  encoder_options.distance = gjxl_distance_from_quality(90.0f);

  const uint8_t pixels[] = {32, 96, 224};
  const GJXLImageView image = {
    sizeof(GJXLImageView),
    1,
    1,
    GJXL_PIXEL_FORMAT_RGB8_SRGB,
    pixels,
    sizeof(pixels),
    sizeof(pixels),
  };
  GJXLBuffer output = {NULL, 0};
  const GJXLResult result =
    gjxl_encode(context, &image, &encoder_options, &output);
  const int valid = result == GJXL_OK && output.data != NULL &&
    output.size >= 2 && output.data[0] == 0xff && output.data[1] == 0x0a;

  gjxl_buffer_free(&output);
  gjxl_context_destroy(context);
  return valid && output.data == NULL && output.size == 0
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
}
