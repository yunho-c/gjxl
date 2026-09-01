// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gjxl/gjxl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

#include "c_api/image_conversion.h"
#include "codestream/workflow.h"
#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"
#include "core/status.h"

struct GJXLContext {
  gjxl::VarDctBackendPreference backend =
    gjxl::VarDctBackendPreference::kAutomatic;
  size_t cpu_thread_count = 0;
};

namespace {

constexpr size_t kDiagnosticCapacity = 1024;
constexpr size_t kContextOptionsV1Size =
  offsetof(GJXLContextOptions, num_cpu_threads);
constexpr size_t kContextOptionsCpuThreadsSize =
  offsetof(GJXLContextOptions, num_cpu_threads) + sizeof(uint32_t);
static_assert(gjxl::kMaximumCpuThreadCount == GJXL_MAX_CPU_THREADS);
thread_local std::array<char, kDiagnosticCapacity> last_error{};

void ClearLastError() noexcept {
  last_error[0] = '\0';
}

void SetLastError(std::string_view message) noexcept {
  const size_t length = std::min(message.size(), last_error.size() - 1);
  if (length != 0) {
    std::memcpy(last_error.data(), message.data(), length);
  }
  last_error[length] = '\0';
}

GJXLResult Fail(GJXLResult result, std::string_view message) noexcept {
  SetLastError(message);
  return result;
}

GJXLResult TranslateStatus(const gjxl::Status& status) noexcept {
  if (status.ok()) {
    return GJXL_OK;
  }
  SetLastError(status.message());
  switch (status.code()) {
    case gjxl::StatusCode::kInvalidArgument:
      return GJXL_ERROR_INVALID_ARGUMENT;
    case gjxl::StatusCode::kUnsupported:
      return GJXL_ERROR_UNSUPPORTED;
    case gjxl::StatusCode::kUnavailable:
      return GJXL_ERROR_UNAVAILABLE;
    case gjxl::StatusCode::kOutOfMemory:
      return GJXL_ERROR_OUT_OF_MEMORY;
    case gjxl::StatusCode::kSubmissionFailed:
    case gjxl::StatusCode::kDeviceError:
      return GJXL_ERROR_BACKEND;
    case gjxl::StatusCode::kFailedPrecondition:
    case gjxl::StatusCode::kInternal:
      return GJXL_ERROR_INTERNAL;
    case gjxl::StatusCode::kOk:
      break;
  }
  return Fail(GJXL_ERROR_INTERNAL, "Unknown internal status code");
}

template <typename Function>
GJXLResult Guard(Function&& function) noexcept {
  ClearLastError();
  try {
    return std::forward<Function>(function)();
  } catch (const std::bad_alloc&) {
    return Fail(GJXL_ERROR_OUT_OF_MEMORY, "C API allocation failed");
  } catch (const std::exception& exception) {
    return Fail(GJXL_ERROR_INTERNAL, exception.what());
  } catch (...) {
    return Fail(GJXL_ERROR_INTERNAL, "Unknown C++ exception");
  }
}

template <typename Options>
GJXLResult InitializeOptions(
  Options* options,
  size_t caller_size,
  size_t required_size = sizeof(Options)) noexcept {
  if (options == nullptr) {
    return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                "Options output pointer must not be null");
  }
  if (caller_size < required_size ||
      caller_size > std::numeric_limits<uint32_t>::max()) {
    return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                "Options allocation size is invalid");
  }
  std::memset(options, 0, caller_size);
  options->struct_size = static_cast<uint32_t>(caller_size);
  return GJXL_OK;
}

GJXLResult ValidateSizedStruct(uint32_t struct_size, size_t required_size,
                               std::string_view name) noexcept {
  if (struct_size < required_size) {
    SetLastError(name);
    return GJXL_ERROR_INVALID_ARGUMENT;
  }
  return GJXL_OK;
}

GJXLResult ParseBackend(GJXLBackend backend,
                        gjxl::VarDctBackendPreference* preference) noexcept {
  if (preference == nullptr) {
    return Fail(GJXL_ERROR_INTERNAL, "Backend destination is null");
  }
  switch (backend) {
    case GJXL_BACKEND_AUTO:
      *preference = gjxl::VarDctBackendPreference::kAutomatic;
      return GJXL_OK;
    case GJXL_BACKEND_CPU:
      *preference = gjxl::VarDctBackendPreference::kCpu;
      return GJXL_OK;
    case GJXL_BACKEND_METAL:
      *preference = gjxl::VarDctBackendPreference::kMetal;
      return GJXL_OK;
    default:
      return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                  "Context backend is not recognized");
  }
}

GJXLResult ParsePixelFormat(
  GJXLPixelFormat format,
  gjxl::c_api_internal::PackedPixelFormat* packed_format) noexcept {
  if (packed_format == nullptr) {
    return Fail(GJXL_ERROR_INTERNAL, "Pixel-format destination is null");
  }
  switch (format) {
    case GJXL_PIXEL_FORMAT_RGB8_SRGB:
      *packed_format =
        gjxl::c_api_internal::PackedPixelFormat::kRgb8Srgb;
      return GJXL_OK;
    case GJXL_PIXEL_FORMAT_RGBA8_SRGB:
      *packed_format =
        gjxl::c_api_internal::PackedPixelFormat::kRgba8Srgb;
      return GJXL_OK;
    default:
      return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                  "Image pixel format is not recognized");
  }
}

GJXLResult ValidateEncoderOptions(
  const GJXLEncoderOptions& options) noexcept {
  if (options.distance == 0.0f) {
    return Fail(GJXL_ERROR_UNSUPPORTED,
                "Lossless distance zero is not supported");
  }
  if (!std::isfinite(options.distance) || options.distance < 0.0f ||
      options.distance > 25.0f) {
    return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                "Distance must be finite and in (0, 25]");
  }
  if (options.effort < 1 || options.effort > 10) {
    return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                "Effort must be in [1, 10]");
  }
  return GJXL_OK;
}

}  // namespace

extern "C" {

GJXLResult gjxl_context_options_init(
  GJXLContextOptions* options, size_t caller_size) noexcept {
  return Guard([&]() -> GJXLResult {
    const GJXLResult result = InitializeOptions(
      options, caller_size, kContextOptionsV1Size);
    if (result != GJXL_OK) {
      return result;
    }
    options->backend = GJXL_BACKEND_AUTO;
    return GJXL_OK;
  });
}

GJXLResult gjxl_encoder_options_init(
  GJXLEncoderOptions* options, size_t caller_size) noexcept {
  return Guard([&]() -> GJXLResult {
    const GJXLResult result = InitializeOptions(options, caller_size);
    if (result != GJXL_OK) {
      return result;
    }
    options->distance = 1.0f;
    options->effort = 7;
    return GJXL_OK;
  });
}

GJXLResult gjxl_context_create(
  const GJXLContextOptions* options, GJXLContext** context) noexcept {
  return Guard([&]() -> GJXLResult {
    if (context == nullptr) {
      return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                  "Context output pointer must not be null");
    }
    if (*context != nullptr) {
      return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                  "Context output must be empty");
    }

    gjxl::VarDctBackendPreference backend =
      gjxl::VarDctBackendPreference::kAutomatic;
    size_t cpu_thread_count = 0;
    if (options != nullptr) {
      GJXLResult result = ValidateSizedStruct(
        options->struct_size, kContextOptionsV1Size,
        "Context options struct is too small");
      if (result != GJXL_OK) {
        return result;
      }
      if (options->struct_size >= kContextOptionsCpuThreadsSize) {
        cpu_thread_count = options->num_cpu_threads;
        if (cpu_thread_count > GJXL_MAX_CPU_THREADS) {
          return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                      "CPU thread count must be zero or at most 256");
        }
      }
      result = ParseBackend(options->backend, &backend);
      if (result != GJXL_OK) {
        return result;
      }
    }

    if (backend == gjxl::VarDctBackendPreference::kMetal) {
      const GJXLResult result = TranslateStatus(
        gjxl::codestream_internal::EnsureProductionMetalBackendAvailable());
      if (result != GJXL_OK) {
        return result;
      }
    }

    auto candidate = std::make_unique<GJXLContext>();
    candidate->backend = backend;
    candidate->cpu_thread_count = cpu_thread_count;
    *context = candidate.release();
    return GJXL_OK;
  });
}

void gjxl_context_destroy(GJXLContext* context) noexcept {
  ClearLastError();
  delete context;
}

GJXLResult gjxl_encode(
  GJXLContext* context,
  const GJXLImageView* image,
  const GJXLEncoderOptions* options,
  GJXLBuffer* output) noexcept {
  return Guard([&]() -> GJXLResult {
    if (context == nullptr || image == nullptr || options == nullptr ||
        output == nullptr) {
      return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                  "Encode pointer argument must not be null");
    }
    if (output->data != nullptr || output->size != 0) {
      return Fail(GJXL_ERROR_INVALID_ARGUMENT,
                  "Encode output buffer must be empty");
    }

    GJXLResult result = ValidateSizedStruct(
      image->struct_size, sizeof(*image), "Image view struct is too small");
    if (result != GJXL_OK) {
      return result;
    }
    result = ValidateSizedStruct(
      options->struct_size, sizeof(*options),
      "Encoder options struct is too small");
    if (result != GJXL_OK) {
      return result;
    }
    result = ValidateEncoderOptions(*options);
    if (result != GJXL_OK) {
      return result;
    }

    gjxl::c_api_internal::PackedPixelFormat packed_format;
    result = ParsePixelFormat(image->pixel_format, &packed_format);
    if (result != GJXL_OK) {
      return result;
    }
    const gjxl::c_api_internal::PackedSrgbImageView packed_image{
      .pixels = static_cast<const uint8_t*>(image->pixels),
      .pixels_size = image->pixels_size,
      .width = image->width,
      .height = image->height,
      .row_stride_bytes = image->row_stride_bytes,
      .format = packed_format,
    };
    gjxl::Image3FBuffer linear_rgb;
    result = TranslateStatus(
      gjxl::c_api_internal::ConvertPackedSrgbToLinearRgb(
        packed_image, &linear_rgb));
    if (result != GJXL_OK) {
      return result;
    }

    gjxl::VarDctEncodingOptions encoding_options;
    encoding_options.butteraugli_target = options->distance;
    encoding_options.effort = options->effort;
    encoding_options.backend = context->backend;
    encoding_options.cpu_thread_count = context->cpu_thread_count;
    std::vector<uint8_t> codestream;
    result = TranslateStatus(gjxl::EncodeLinearRgbVarDctCodestream(
      linear_rgb.const_view(), encoding_options, &codestream));
    if (result != GJXL_OK) {
      return result;
    }
    if (codestream.empty()) {
      return Fail(GJXL_ERROR_INTERNAL,
                  "Encoder returned an empty codestream");
    }

    auto data = std::make_unique<uint8_t[]>(codestream.size());
    std::memcpy(data.get(), codestream.data(), codestream.size());
    output->data = data.release();
    output->size = codestream.size();
    return GJXL_OK;
  });
}

void gjxl_buffer_free(GJXLBuffer* buffer) noexcept {
  ClearLastError();
  if (buffer == nullptr) {
    return;
  }
  delete[] buffer->data;
  buffer->data = nullptr;
  buffer->size = 0;
}

const char* gjxl_get_last_error(void) noexcept {
  return last_error.data();
}

float gjxl_distance_from_quality(float quality) noexcept {
  ClearLastError();
  return quality >= 100.0 ? 0.0
       : quality >= 30.0
         ? 0.1 + (100.0 - quality) * 0.09
         : 53.0 / 3000.0 * quality * quality -
           23.0 / 20.0 * quality + 25.0;
}

}  // extern "C"
