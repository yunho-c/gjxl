// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/frame.h"
#include "codestream/entropy.h"
#include "core/execution_domain.h"

namespace gjxl::modular_internal {
struct ModularEncodingOptions {
  std::shared_ptr<const ExecutionDomain> execution_domain;
  EntropyCodingMode entropy = EntropyCodingMode::kPrefix;
};
[[nodiscard]] Status EncodeRgb8ModularOwned(Rgb8View input, ModularEncodingOptions options,
                                            codestream_internal::CodestreamBuffer *out);
[[nodiscard]] Status EncodeRgb8Modular(Rgb8View input, ModularEncodingOptions options,
                                       std::vector<uint8_t> *out);
[[nodiscard]] Status EncodeModularImageOwned(PackedModularImageView input,
                                             ModularEncodingOptions options,
                                             codestream_internal::CodestreamBuffer *out);
[[nodiscard]] Status EncodeModularImage(PackedModularImageView input,
                                        ModularEncodingOptions options, std::vector<uint8_t> *out);
} // namespace gjxl::modular_internal
