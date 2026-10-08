// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gjxl/modular.hpp"
#include "codestream/modular/workflow.h"
#include <chrono>
namespace gjxl {
Status EncodeModularImage(ModularImageView input, ModularEncodingOptions options,
                          std::vector<uint8_t> *output, ModularEncodingSummary *summary) {
  if (!output)
    return Status::InvalidArgument("Null Modular output");
  switch (options.backend) {
  case ModularBackend::kAutomatic:
  case ModularBackend::kCpu:
    break;
  case ModularBackend::kMetal:
  case ModularBackend::kCuda:
    return Status::Unsupported("Modular GPU encoding is not implemented");
  default:
    return Status::InvalidArgument("Invalid Modular backend");
  }
  if (options.entropy != ModularEntropy::kPrefix && options.entropy != ModularEntropy::kAns)
    return Status::InvalidArgument("Invalid Modular entropy mode");
  const auto start = std::chrono::steady_clock::now();
  modular_internal::PackedModularImageView packed{
      input.bytes, input.extent, input.row_stride_bytes,
      static_cast<modular_internal::PackedModularFormat>(input.format),
      static_cast<modular_internal::SampleByteOrder>(input.byte_order)};
  modular_internal::ModularEncodingOptions resolved;
  resolved.execution_domain = options.execution_domain;
  resolved.cpu_thread_count = options.cpu_thread_count;
  resolved.entropy = options.entropy == ModularEntropy::kPrefix ? EntropyCodingMode::kPrefix
                                                                : EntropyCodingMode::kAns;
  resolved.search = options.search;
  if (auto s = modular_internal::EncodeModularImage(packed, resolved, output); !s.ok())
    return s;
  if (summary)
    *summary = {input.extent, output->size(), ModularBackend::kCpu,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
  return Status::Ok();
}
} // namespace gjxl
