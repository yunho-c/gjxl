// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "codec/color_transform.h"
#include "core/frame_geometry.h"
#include "core/image_buffer.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/ops/resident_input.h"

namespace {
using namespace gjxl;
void Check(bool good, const char *message) {
  if (!good)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Source {
  Image3FBuffer storage;
  Extent2D extent;
  size_t calls = 0;
  bool fail = false;
  ConstImage3FView View() const { return storage.cropped_view(extent); }
};
Status Fill(const void *opaque, Image3FView output) {
  auto &source = *static_cast<Source *>(const_cast<void *>(opaque));
  ++source.calls;
  if (source.fail)
    return Status::Internal("Injected input generator failure");
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < source.extent.height; ++y)
      std::copy_n(source.View().plane[c].Row(y), source.extent.width,
                  output.plane[c].Row(y));
  return Status::Ok();
}
void Run(GpuBackend &gpu, Extent2D extent) {
  Source source{Image3FBuffer({extent.width + 3, extent.height}), extent};
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        source.storage.view().plane[c].Row(y)[x] =
            0.001f * float((x * 13 + y * 47 + c * 19) % 997);
  FrameGeometry geometry;
  Ok(FrameGeometry::Create(extent, &geometry));
  const auto padded = geometry.padded_frame();
  Image3FBuffer cpu_opsin(extent);
  Ok(LinearRgbToOpsin(source.View(), 255.0f, cpu_opsin.view()));
  std::array<std::unique_ptr<PreparedResidentInput>, 2> prepared;
  Ok(PrepareResidentInput(gpu,
                          {.original_linear_rgb = source.View(),
                           .coding_extent = padded,
                           .compute_matrix_scale_statistics = true},
                          &prepared[0]));
  Ok(PrepareResidentInput(gpu,
                          {.coding_extent = padded,
                           .compute_matrix_scale_statistics = true,
                           .fill_extent = extent,
                           .fill_original = Fill,
                           .fill_context = &source},
                          &prepared[1]));
  Check(source.calls == 1, "Resident generator call count changed");
  const auto host = prepared[1]->original_linear_rgb_host();
  Check(host.valid() && host.extent() == extent,
        "Generated input lost its owned host view");
  for (size_t c = 0; c < 3; ++c) {
    std::array<std::vector<float>, 2> coding;
    for (size_t route = 0; route < 2; ++route) {
      std::vector<float> rgb(extent.width * extent.height);
      coding[route].resize(padded.width * padded.height);
      const auto original = prepared[route]->original_linear_rgb().plane[c];
      const auto opsin = prepared[route]->coding_opsin().plane[c];
      Ok(gpu.CopyDeviceToHost(*original.buffer, rgb.data(),
                              rgb.size() * sizeof(float),
                              original.offset_bytes));
      Ok(gpu.CopyDeviceToHost(*opsin.buffer, coding[route].data(),
                              coding[route].size() * sizeof(float),
                              opsin.offset_bytes));
      for (size_t y = 0; y < padded.height; ++y)
        for (size_t x = 0; x < padded.width; ++x) {
          const auto sx = std::min(x, extent.width - 1);
          const auto sy = std::min(y, extent.height - 1);
          Check(std::abs(coding[route][y * padded.width + x] -
                         cpu_opsin.const_view().plane[c].Row(sy)[sx]) < 2e-6f,
                "Resident Opsin differs from padded CPU conversion");
          if (x < extent.width && y < extent.height) {
            Check(rgb[y * extent.width + x] ==
                          source.View().plane[c].Row(y)[x] &&
                      host.plane[c].Row(y)[x] == rgb[y * extent.width + x],
                  "Resident original upload or retained host view differs");
          }
        }
    }
    Check(coding[0] == coding[1],
          "Generated and supplied resident inputs differ");
  }
  const auto a = prepared[0]->statistics(), b = prepared[1]->statistics();
  Check(a.x_edge == b.x_edge && a.b_edge == b.b_edge &&
            a.exposed_blue == b.exposed_blue,
        "Resident input statistics changed between source routes");
  // Invalid/mixed sources and callback failure clear the output; they never
  // publish partially initialized device or host views.
  auto invalid = ResidentInputPreparation{.original_linear_rgb = source.View(),
                                          .coding_extent = padded,
                                          .fill_extent = extent,
                                          .fill_original = Fill,
                                          .fill_context = &source};
  Check(PrepareResidentInput(gpu, invalid, &prepared[0]).code() ==
                StatusCode::kInvalidArgument &&
            prepared[0] == nullptr && source.calls == 1,
        "Mixed source was accepted");
  source.fail = true;
  invalid.original_linear_rgb = {};
  Check(PrepareResidentInput(gpu, invalid, &prepared[1]).code() ==
                StatusCode::kInternal &&
            prepared[1] == nullptr && source.calls == 2,
        "Generator failure was not atomic");
  source.storage.view().plane[0].Row(0)[0] =
      std::numeric_limits<float>::quiet_NaN();
  Check(PrepareResidentInput(
            gpu,
            {.original_linear_rgb = source.View(), .coding_extent = padded},
            &prepared[0])
                    .code() == StatusCode::kInvalidArgument &&
            prepared[0] == nullptr,
        "Nonfinite source was published");
}
} // namespace

int main() {
  std::unique_ptr<GpuBackend> gpu;
  const auto status = CreateCudaBackend(&gpu);
  if (status.code() == StatusCode::kUnavailable)
    return 77;
  try {
    Ok(status);
    for (auto extent : {Extent2D{1, 1}, {17, 13}, {65, 33}, {1, 257}})
      Run(*gpu, extent);
    std::cout << "CUDA supplied/generated resident inputs and failure "
                 "publication passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
