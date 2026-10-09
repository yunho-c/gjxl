// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <array>
#include <iostream>
#include <string_view>
#include <vector>

#include "codec/vardct_frame_view_internal.h"
#include "codestream/encoder_internal.h"
#include "cuda_sparse_resident_fixture.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_tokenization_request.h"

namespace {
using namespace resident_sparse_test;
using namespace gjxl::cuda_internal;
using Lease = std::unique_ptr<vardct_frame_internal::CompletedVarDctFrame>;

std::vector<int32_t> Packed(
    const vardct_frame_internal::VarDctFrameView& frame) {
  std::vector<int32_t> result;
  for (size_t g = 0; g < frame.ac_group_count(); ++g) {
    VarDctNativeAcGroupView native;
    Check(frame.GetNativeAcGroup(g, &native));
    std::visit(
        [&](const auto& group) {
          for (size_t c = 0; c < 3; ++c)
            for (size_t i = 0; i < group.used_coefficient_count; ++i)
              result.push_back(group.coefficients[c][i]);
        },
        native);
  }
  return result;
}

std::vector<int32_t> Read(const Lease& lease) {
  size_t offset = 99;
  const auto* buffer = lease->resident_ac_buffer(&offset);
  Require(buffer && offset == 0, "Resident coefficient owner missing");
  const auto* cuda = dynamic_cast<const CudaBuffer*>(buffer);
  Require(cuda && buffer->size_bytes() % sizeof(int32_t) == 0,
          "Resident coefficient buffer type/length differs");
  ScopedCudaDevice device(cuda->state()->ordinal);
  Check(CudaRuntimeStatus(device.status(), "test device selection"));
  std::vector<int32_t> output(buffer->size_bytes() / sizeof(int32_t));
  Check(CudaRuntimeStatus(
      cudaMemcpy(output.data(), cuda->pointer(), buffer->size_bytes(),
                 cudaMemcpyDeviceToHost),
      "test retained readback"));
  return output;
}

std::vector<uint8_t> Bytes(const Lease& lease) {
  std::vector<uint8_t> bytes;
  Check(codestream_internal::EncodeVarDctCodestreamFromView(lease->view(), {},
                                                            &bytes));
  return bytes;
}

void Case(Extent2D extent, float amplitude, bool evaluate_final, bool mixed) {
  std::array<Lease, 2> retained;
  std::array<std::vector<int32_t>, 2> coefficients;
  std::array<std::vector<uint8_t>, 2> bytes;
  {
    std::unique_ptr<GpuBackend> gpu;
    Check(CreateCudaBackend(&gpu));
    Fixture fixture(extent, amplitude);
    if (mixed) fixture.strategies = gjxl_test::MakeFrame(7, 0, 36).strategies();
    auto prepared = fixture.Prepare(*gpu);
    std::vector<double> scores;
    const auto run = [&](Lease* out) {
      return prepared->EvaluateResidentButteraugliPolicy(
          {.adjusted_initial_quant_field = {fixture.field.data(),
                                            fixture.blocks,
                                            fixture.blocks.width},
           .quant_dc = fixture.quant_dc,
           .butteraugli_target = 1.1f,
           .lower_bound = 0.1f,
           .upper_bound = 10.0f,
           .iterations = 1,
           .evaluate_final_field = evaluate_final},
          {.score_history = &scores, .completed_frame = out});
    };
    for (size_t generation = 0; generation < retained.size(); ++generation) {
      std::fill(fixture.field.begin(), fixture.field.end(),
                generation ? 1.1f : .8f);
      Lease ordinary;
      Check(run(&ordinary));
      Require(ordinary->resident_ac_buffer(nullptr) == nullptr,
              "Disabled handoff retained device storage");
      const auto expected = Bytes(ordinary);
      const auto previous_scores = scores;
      {
        CudaTokenCoefficientScope request(true);
        Check(run(&retained[generation]));
      }
      Require(
          scores == previous_scores && Bytes(retained[generation]) == expected,
          "Device handoff changed ordinary codestream/score");
      coefficients[generation] = Packed(retained[generation]->view());
      Require(Read(retained[generation]) == coefficients[generation],
              "Packed device coefficients differ from native CPU owner");
      bytes[generation] = expected;
    }
    Require(retained[0]->resident_ac_buffer(nullptr) !=
                retained[1]->resident_ac_buffer(nullptr),
            "Repeated completion reused a borrowed owner");
    // A failed replacement must preserve the prior complete output and scores.
    const auto* before = retained[1].get();
    const auto old_scores = scores;
    auto* cuda = dynamic_cast<CudaBackend*>(gpu.get());
    Require(cuda != nullptr, "CUDA backend type differs");
    cuda->ArmNextAllocationFailureForTest();
    {
      CudaTokenCoefficientScope request(true);
      Require(!run(&retained[1]).ok(),
              "Injected completed allocation did not fail");
    }
    Require(retained[1].get() == before && scores == old_scores,
            "Failed device owner publication changed previous output");
    // Destroy producer, backend, input and failed state while both owners live.
  }
  for (size_t generation = 0; generation < retained.size(); ++generation) {
    Require(Read(retained[generation]) == coefficients[generation],
            "Resident coefficients depended on producer/backend lifetime");
    Require(Packed(retained[generation]->view()) == coefficients[generation] &&
                Bytes(retained[generation]) == bytes[generation],
            "Native completed frame changed after producer destruction");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
    Require(argc == 1 || smoke, "Unknown test arguments");
    const auto mixed = gjxl_test::MakeFrame(7, 0, 36).geometry().frame();
    const std::array<Extent2D, 4> extents{
        {{1, 1}, {17, 33}, {257, 263}, mixed}};
    size_t cases = 0;
    for (size_t shape = 0; shape < extents.size(); ++shape) {
      if (smoke && shape != 1) continue;
      for (float amplitude : {0.0f, 16.0f})
        for (bool evaluate_final : {false, true}) {
          Case(extents[shape], amplitude, evaluate_final, shape == 3);
          ++cases;
        }
    }
    std::cout << "Verified " << cases << " CUDA coefficient handoff cases, "
              << 2 * cases << " independent retained owners, " << cases
              << " atomic allocation failures.\n"
              << std::flush;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n' << std::flush;
    return 1;
  }
}
