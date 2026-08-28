// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "codestream/batch_workflow.h"
#include "codestream/workflow.h"
#include "gpu/metal/metal_backend.h"

int main() {
  std::unique_ptr<gjxl::GpuBackend> embedded_backend;
  if (!gjxl::CreateEmbeddedMetalBackend({}, &embedded_backend).ok() ||
      embedded_backend == nullptr) {
    return EXIT_FAILURE;
  }
  constexpr gjxl::Extent2D kExtent{1, 1};
  const std::array<float, 3> pixel = {0.2f, 0.3f, 0.4f};
  const gjxl::ConstImage3FView image{{
    gjxl::ConstPlaneF32View{&pixel[0], kExtent, 1},
    gjxl::ConstPlaneF32View{&pixel[1], kExtent, 1},
    gjxl::ConstPlaneF32View{&pixel[2], kExtent, 1},
  }};
  std::vector<uint8_t> codestream;
  gjxl::VarDctEncodingSummary summary;
  const gjxl::Status status = gjxl::EncodeLinearRgbVarDctCodestream(
    image, {}, &codestream, &summary);
  std::unique_ptr<gjxl::VarDctBatchEncoder> batch_encoder;
  if (!status.ok() ||
      !gjxl::VarDctBatchEncoder::Create(2, &batch_encoder).ok() ||
      batch_encoder == nullptr) {
    return EXIT_FAILURE;
  }
  const std::array<gjxl::VarDctBatchEncodingRequest, 2> requests = {{
    {.linear_rgb = image, .options = {}},
    {.linear_rgb = image, .options = {}},
  }};
  std::vector<gjxl::VarDctBatchEncodingResult> batch_results;
  const gjxl::Status batch_status =
    batch_encoder->Encode(requests, &batch_results);
  return batch_status.ok() && batch_results.size() == 2 &&
      batch_results[0].status.ok() && batch_results[1].status.ok() &&
      batch_results[0].codestream == codestream &&
      batch_results[1].codestream == codestream &&
      codestream.size() >= 2 &&
      codestream[0] == 0xff && codestream[1] == 0x0a &&
      summary.extent == kExtent &&
      summary.execution_backend == gjxl::VarDctExecutionBackend::kCpu
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
}
