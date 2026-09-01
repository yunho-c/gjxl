// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <type_traits>

#include "gjxl/gjxl.h"

static_assert(std::is_same_v<GJXLResult, int32_t>);
static_assert(std::is_same_v<GJXLBackend, int32_t>);
static_assert(std::is_same_v<GJXLPixelFormat, int32_t>);
static_assert(std::is_standard_layout_v<GJXLContextOptions>);
static_assert(std::is_standard_layout_v<GJXLEncoderOptions>);
static_assert(std::is_standard_layout_v<GJXLImageView>);
static_assert(std::is_standard_layout_v<GJXLBuffer>);
static_assert(GJXL_MAX_CPU_THREADS == 256);

static_assert(noexcept(gjxl_context_options_init(nullptr, 0)));
static_assert(noexcept(gjxl_encoder_options_init(nullptr, 0)));
static_assert(noexcept(gjxl_context_create(nullptr, nullptr)));
static_assert(noexcept(gjxl_context_destroy(nullptr)));
static_assert(noexcept(gjxl_encode(nullptr, nullptr, nullptr, nullptr)));
static_assert(noexcept(gjxl_buffer_free(nullptr)));
static_assert(noexcept(gjxl_get_last_error()));
static_assert(noexcept(gjxl_distance_from_quality(90.0f)));

using ContextInitializer =
  GJXLResult (*)(GJXLContextOptions*, size_t) noexcept;
using EncodeFunction = GJXLResult (*)(
  GJXLContext*, const GJXLImageView*, const GJXLEncoderOptions*,
  GJXLBuffer*) noexcept;
static_assert(std::is_same_v<decltype(&gjxl_context_options_init),
                             ContextInitializer>);
static_assert(std::is_same_v<decltype(&gjxl_encode), EncodeFunction>);

int main() {
  GJXLContextOptions options;
  if (gjxl_context_options_init(&options, sizeof(options)) != GJXL_OK) {
    return 1;
  }
  options.backend = GJXL_BACKEND_CPU;
  GJXLContext* context = nullptr;
  if (gjxl_context_create(&options, &context) != GJXL_OK ||
      context == nullptr) {
    return 1;
  }
  gjxl_context_destroy(context);

  if (gjxl_context_options_init(
        &options, offsetof(GJXLContextOptions, num_cpu_threads) - 1) !=
      GJXL_ERROR_INVALID_ARGUMENT) {
    return 1;
  }
  const std::string main_diagnostic = gjxl_get_last_error();
  std::atomic<bool> thread_isolated{false};
  std::thread worker([&] {
    const bool initially_empty = gjxl_get_last_error()[0] == '\0';
    const bool received_error =
      gjxl_encoder_options_init(nullptr, 0) == GJXL_ERROR_INVALID_ARGUMENT &&
      gjxl_get_last_error()[0] != '\0';
    thread_isolated = initially_empty && received_error;
  });
  worker.join();
  const bool main_preserved = main_diagnostic == gjxl_get_last_error();
  return gjxl_distance_from_quality(90.0f) == 1.0f && thread_isolated &&
      main_preserved &&
      main_diagnostic == "Options allocation size is invalid"
    ? 0
    : 1;
}
