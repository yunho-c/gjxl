// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "codestream/workflow.h"
#include "gjxl/execution_domain.hpp"

extern "C" int check_legacy_contexts(void);

namespace {
bool CheckCValidation() {
  GJXLExecutionDomainOptions options{};
  if (gjxl_execution_domain_options_init(nullptr, sizeof(options)) != GJXL_ERROR_INVALID_ARGUMENT)
    return false;
  std::memset(&options, 0xa5, sizeof(options));
  std::array<unsigned char, sizeof(options)> before{};
  std::memcpy(before.data(), &options, sizeof(options));
  for (size_t size : {size_t{0}, sizeof(options) - 1,
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1}) {
    if (gjxl_execution_domain_options_init(&options, size) != GJXL_ERROR_INVALID_ARGUMENT ||
        std::memcmp(before.data(), &options, sizeof(options)) != 0)
      return false;
  }
  if (gjxl_execution_domain_options_init(&options, sizeof(options)) != GJXL_OK ||
      options.managed_memory_bytes != 0)
    return false;
  GJXLExecutionDomain *raw = nullptr;
  options.struct_size = sizeof(options) - 1;
  if (gjxl_execution_domain_create(&options, &raw) != GJXL_ERROR_INVALID_ARGUMENT ||
      raw != nullptr ||
      gjxl_execution_domain_create(nullptr, nullptr) != GJXL_ERROR_INVALID_ARGUMENT)
    return false;
  if (gjxl_execution_domain_create(nullptr, &raw) != GJXL_OK)
    return false;
  const std::unique_ptr<GJXLExecutionDomain, decltype(&gjxl_execution_domain_destroy)> domain(
      raw, gjxl_execution_domain_destroy);
  if (gjxl_execution_domain_create(nullptr, &raw) != GJXL_ERROR_INVALID_ARGUMENT ||
      raw != domain.get())
    return false;
  const auto retained = gjxl::RetainExecutionDomain(raw);
  if (!retained || retained == gjxl::ExecutionDomain::Default())
    return false;
  GJXLExecutionDomainSnapshot snapshot{};
  std::memset(&snapshot, 0xa5, sizeof(snapshot));
  std::array<unsigned char, sizeof(snapshot)> old_snapshot{};
  std::memcpy(old_snapshot.data(), &snapshot, sizeof(snapshot));
  for (size_t size : {size_t{0}, sizeof(snapshot) - 1,
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1}) {
    if (gjxl_execution_domain_snapshot(raw, &snapshot, size) != GJXL_ERROR_INVALID_ARGUMENT ||
        std::memcmp(old_snapshot.data(), &snapshot, sizeof(snapshot)) != 0)
      return false;
  }
  return gjxl_execution_domain_snapshot(raw, nullptr, sizeof(snapshot)) ==
             GJXL_ERROR_INVALID_ARGUMENT &&
         gjxl_execution_domain_snapshot(raw, &snapshot, sizeof(snapshot)) == GJXL_OK &&
         snapshot.struct_size == sizeof(snapshot) && snapshot.peak_committed_bytes == 0;
}

bool CheckSharedLimit() {
  std::shared_ptr<const gjxl::ExecutionDomain> domain;
  if (!gjxl::ExecutionDomain::Create({1}, &domain).ok())
    return false;
  GJXLExecutionDomain *raw = nullptr;
  if (!gjxl::CreateCExecutionDomain(domain, &raw).ok())
    return false;
  std::unique_ptr<GJXLExecutionDomain, decltype(&gjxl_execution_domain_destroy)> c_domain(
      raw, gjxl_execution_domain_destroy);
  if (gjxl::RetainExecutionDomain(raw) != domain ||
      gjxl::CreateCExecutionDomain(domain, &raw).ok() || raw != c_domain.get())
    return false;
  GJXLContextOptions co{};
  GJXLEncoderOptions eo{};
  if (gjxl_context_options_init(&co, sizeof(co)) != GJXL_OK ||
      gjxl_encoder_options_init(&eo, sizeof(eo)) != GJXL_OK || co.execution_domain != nullptr)
    return false;
  co.backend = GJXL_BACKEND_CPU;
  co.num_cpu_threads = 1;
  co.execution_domain = c_domain.get();
  eo.effort = 1;
  GJXLContext *context_raw = nullptr;
  if (gjxl_context_create(&co, &context_raw) != GJXL_OK)
    return false;
  const std::unique_ptr<GJXLContext, decltype(&gjxl_context_destroy)> context(context_raw,
                                                                              gjxl_context_destroy);
  c_domain.reset(); // Context retains the domain, not the opaque wrapper.
  const std::array<uint8_t, 3> pixel{20, 30, 40};
  const GJXLImageView image{
      sizeof(GJXLImageView), 1,           1, GJXL_PIXEL_FORMAT_RGB8_SRGB, pixel.data(),
      pixel.size(),          pixel.size()};
  GJXLBuffer output{};
  const auto result = gjxl_encode(context.get(), &image, &eo, &output);
  const bool c_good =
      result == GJXL_ERROR_OUT_OF_MEMORY && output.data == nullptr && output.size == 0;
  gjxl_buffer_free(&output);
  constexpr gjxl::Extent2D extent{1, 1};
  const std::array<float, 3> linear{0.2f, 0.3f, 0.4f};
  const gjxl::ConstImage3FView view{{
      gjxl::ConstPlaneF32View{&linear[0], extent, 1},
      gjxl::ConstPlaneF32View{&linear[1], extent, 1},
      gjxl::ConstPlaneF32View{&linear[2], extent, 1}}};
  gjxl::VarDctEncodingOptions options;
  options.backend = gjxl::VarDctBackendPreference::kCpu;
  options.execution_domain = domain;
  options.effort = 1;
  options.cpu_thread_count = 1;
  std::vector<uint8_t> bytes{19};
  const auto status = gjxl::EncodeLinearRgbVarDctCodestream(view, options, &bytes);
  return c_good && status.code() == gjxl::StatusCode::kOutOfMemory &&
         !status.resource_plan_exceeded() && bytes == std::vector<uint8_t>{19} &&
         domain->snapshot().peak_committed_bytes == 0 &&
         gjxl::RetainExecutionDomain(nullptr) == gjxl::ExecutionDomain::Default();
}
} // namespace

int main() {
  return check_legacy_contexts() && CheckCValidation() && CheckSharedLimit() ? EXIT_SUCCESS
                                                                             : EXIT_FAILURE;
}
