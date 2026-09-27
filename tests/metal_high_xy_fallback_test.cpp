// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/workflow_internal.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void RequireStatus(const char *what, const gjxl::Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(what) + ": " +
                             std::string(status.message()));
}
struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D e) : extent(e) {
    for (auto &p : plane)
      p.resize(e.width * e.height);
  }
  gjxl::ConstImage3FView ConstView() const {
    return {{gjxl::ConstPlaneF32View{plane[0].data(), extent, extent.width},
             gjxl::ConstPlaneF32View{plane[1].data(), extent, extent.width},
             gjxl::ConstPlaneF32View{plane[2].data(), extent, extent.width}}};
  }
  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> plane;
};
} // namespace
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/metal/metal_butteraugli_high_xy.h"
namespace gjxl::metal_internal {
struct MetalButteraugliTrafficTestAccess {
  static bool Traffic(GpuBackend &b) {
    return static_cast<MetalBackend &>(b)
        .butteraugli_pipelines_.traffic.enabled;
  }
  static bool Fused(GpuBackend &b) {
    return static_cast<MetalBackend &>(b)
               .butteraugli_pipelines_.traffic.high_xy_suppress.get() !=
           nullptr;
  }
  static MTL::Device *Device(GpuBackend &b) {
    return static_cast<MetalBackend &>(b).device_.get();
  }
  static NS::SharedPtr<MTL::ComputePipelineState>
  ReplaceScalar(GpuBackend &b, NS::SharedPtr<MTL::ComputePipelineState> p) {
    auto &slot = static_cast<MetalBackend &>(b)
                     .butteraugli_pipelines_.traffic.high_reuse;
    return std::exchange(slot, std::move(p));
  }
  static void DisableFused(GpuBackend &b) {
    static_cast<MetalBackend &>(b)
        .butteraugli_pipelines_.traffic.high_xy_suppress.reset();
  }
};
} // namespace gjxl::metal_internal
int main(int argc, char **argv) try {
  using namespace gjxl;
  using namespace gjxl::metal_internal;
  using namespace gjxl::codestream_internal;
  using namespace gjxl::gpu_profile_internal;
  if (argc != 2 && argc != 3)
    throw std::runtime_error("candidate.metallib [baseline.metallib] required");
  static_assert(HighXYDispatchFits(256, 0, 9984));
  static_assert(!HighXYDispatchFits(255, 0, 32768));
  static_assert(!HighXYDispatchFits(256, 1, 9984));
  static_assert(!HighXYDispatchFits(256, SIZE_MAX, 32768));
  static_assert(HighXYDispatchFits(256, 32768 - 9984, 32768));
  std::unique_ptr<GpuBackend> fused, fallback;
  RequireStatus("Candidate", CreateMetalBackend(argv[1], &fused));
  if (fused->name() != "Metal: Apple M4 Pro") {
    if (MetalButteraugliTrafficTestAccess::Traffic(*fused) ||
        MetalButteraugliTrafficTestAccess::Fused(*fused))
      throw std::runtime_error("Fusion enabled on an unqualified GPU");
    std::cout << "SKIP: fusion qualification is restricted to M4 Pro\n";
    return 0;
  }
  RequireStatus("Missing optional shader",
                CreateMetalBackend(argc == 3 ? argv[2] : argv[1], &fallback));
  if (argc == 2)
    MetalButteraugliTrafficTestAccess::DisableFused(*fallback);
  if (!MetalButteraugliTrafficTestAccess::Traffic(*fused) ||
      !MetalButteraugliTrafficTestAccess::Fused(*fused) ||
      !MetalButteraugliTrafficTestAccess::Traffic(*fallback) ||
      MetalButteraugliTrafficTestAccess::Fused(*fallback))
    throw std::runtime_error(
        "Optional pipeline fallback disabled original bundle");
  ImageStorage im({193, 131});
  for (size_t c = 0; c < 3; ++c)
    for (size_t i = 0; i < im.plane[c].size(); ++i)
      im.plane[c][i] = float((i * 13 + c * 41) % 257) / 256.f;
  VarDctEncodingOptions options{.butteraugli_target = 1.0f,
                                .effort = 7,
                                .cpu_thread_count = 8,
                                .backend = VarDctBackendPreference::kMetal};
  std::vector<uint8_t> a, b;
  VarDctEncodingSummary sa, sb;
  auto run = [&](GpuBackend &gpu, std::vector<uint8_t> &bytes,
                 VarDctEncodingSummary &summary) {
    RequireStatus("Encode",
                  EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
                      im.ConstView(), options, &gpu, true, &bytes, &summary));
  };
  run(*fused, a, sa);
  run(*fallback, b, sb);
  if (a != b || sa != sb)
    throw std::runtime_error("Missing-shader output changed");
  run(*fused, a, sa);
  const auto before = fused->stats();
  run(*fused, a, sa);
  const auto after = fused->stats();
  GpuExecutionProfile profile;
  VarDctEncodingProfile host;
  RequireStatus("Diagnostic profile",
                EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
                    im.ConstView(), options, fused.get(), true,
                    GpuProfilingMode::kStage, &b, &sb, &host, &profile));
  if (a != b || sa != sb)
    throw std::runtime_error("Diagnostic output changed");
  size_t fused_dispatches = 0, scalar_dispatches = 0;
  for (const auto &sub : profile.submissions)
    for (const auto &stage : sub.stages)
      for (const auto &d : stage.dispatches) {
        if (d.kernel_id == "gjxl_butteraugli_high_xy_suppress_f32")
          ++fused_dispatches;
        if (d.kernel_id == "gjxl_butteraugli_high_shared_f32")
          ++scalar_dispatches;
      }
  if (fused_dispatches != 0 || scalar_dispatches != 12)
    throw std::runtime_error(
        "Individual stage profiling did not preserve scalar dispatches");
  // A valid, deliberately wrong scalar shader witnesses dispatch selection.
  // The normal fused path must ignore it; disabling fusion must expose it.
  const char *witness_source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct P {uint w,h,in_stride,medium_stride,high_stride,channel;};
kernel void witness(device float* medium [[buffer(2)]],
                    device float* high [[buffer(3)]],constant P& p [[buffer(4)]],
                    uint2 local [[thread_position_in_threadgroup]],
                    uint2 group [[threadgroup_position_in_grid]]) {
  uint x=group.x*16+local.x;
  for(uint k=0;k<4;++k){uint y=group.y*64+local.y+k*16;
    if(x<p.w&&y<p.h){medium[y*p.medium_stride+x]=0;high[y*p.high_stride+x]=0;}}
})metal";
  NS::Error *error = nullptr;
  auto *device = MetalButteraugliTrafficTestAccess::Device(*fused);
  auto library = NS::TransferPtr(device->newLibrary(
      NS::String::string(witness_source, NS::UTF8StringEncoding), nullptr,
      &error));
  if (!library)
    throw std::runtime_error("Witness library failed");
  auto function = NS::TransferPtr(library->newFunction(
      NS::String::string("witness", NS::UTF8StringEncoding)));
  auto witness =
      NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  if (!witness)
    throw std::runtime_error("Witness pipeline failed");
  auto scalar = MetalButteraugliTrafficTestAccess::ReplaceScalar(
      *fused, std::move(witness));
  run(*fused, b, sb);
  if (a != b || sa != sb)
    throw std::runtime_error("Ordinary fusion unexpectedly used scalar High");
  MetalButteraugliTrafficTestAccess::DisableFused(*fused);
  const auto witnessed = EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      im.ConstView(), options, fused.get(), true, &b, &sb);
  if (witnessed.ok() && a == b && sa == sb)
    throw std::runtime_error("Scalar dispatch witness did not fire");
  MetalButteraugliTrafficTestAccess::ReplaceScalar(*fused, std::move(scalar));
  run(*fused, b, sb);
  auto disabled_before = fused->stats();
  run(*fused, b, sb);
  auto disabled_after = fused->stats();
  if (a != b || sa != sb)
    throw std::runtime_error("Disabled-fusion output changed");
  if (after.committed_submissions - before.committed_submissions !=
          disabled_after.committed_submissions -
              disabled_before.committed_submissions ||
      after.successful_allocations - before.successful_allocations !=
          disabled_after.successful_allocations -
              disabled_before.successful_allocations)
    throw std::runtime_error("Allocation/submission counts changed");
  std::cout
      << "PASS optional pipeline fallback retains traffic bundle; disabled "
         "fusion exact bytes/full summary; allocation/submission parity; "
         "resource boundaries; ordinary fusion bypasses scalar witness; "
         "diagnostic fused_dispatches="
      << fused_dispatches << " scalar_dispatches=" << scalar_dispatches << '\n';
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
