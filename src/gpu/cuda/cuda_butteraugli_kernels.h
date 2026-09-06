// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace gjxl::cuda_internal {

// 20 psycho planes, one cached reference mask, and six reusable work planes.
inline constexpr size_t kCudaButteraugliWorkingPlaneCount = 27;
inline constexpr size_t kCudaButteraugliPsychoPlaneCount = 10;
inline constexpr size_t kCudaButteraugliKernelCount = 5;

struct CudaButteraugliPlan {
  std::array<const float*, 3> reference{};
  std::array<uint32_t, 3> reference_stride{};
  std::array<float*, kCudaButteraugliWorkingPlaneCount> planes{};
  std::array<float*, kCudaButteraugliPsychoPlaneCount> reference_sub{};
  std::array<const float*, kCudaButteraugliKernelCount> kernels{};
  std::array<float*, 2> reduction{};
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t working_width = 0;
  uint32_t working_height = 0;
  uint32_t sub_width = 0;
  uint32_t sub_height = 0;
  uint32_t xborder = 0;
  uint32_t yborder = 0;
  uint32_t expanded = 0;
  uint32_t multiscale = 0;
  float hf_asymmetry = 1.0f;
  float x_multiplier = 1.0f;
  float intensity_target = 255.0f;
};

// One scaled Malta response, including the caller's initialization/addition
// policy. The separate-pass entry point is retained as a test oracle.
struct CudaButteraugliMaltaParams {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t reference_stride = 0;
  uint32_t distorted_stride = 0;
  uint32_t accumulation_stride = 0;
  uint32_t low_frequency = 0;
  uint32_t initialize_accumulation = 0;
  float norm2_0_gt_1 = 0.0f;
  float norm2_0_lt_1 = 0.0f;
  float norm = 0.0f;
};

[[nodiscard]] cudaError_t LaunchCudaButteraugliMalta(
    const float* reference, const float* distorted, float* accumulation,
    CudaButteraugliMaltaParams params, cudaStream_t stream);

// Internal differential/sanitizer entry: exercise each production specialization
// on small guarded inputs, independently of the size-based launch policy.
[[nodiscard]] cudaError_t LaunchCudaButteraugliMaltaForTesting(
    const float* reference, const float* distorted, float* accumulation,
    CudaButteraugliMaltaParams params, unsigned int tile_height, bool flat_grid,
    cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaButteraugliMaltaReference(
    const float* reference, const float* distorted, float* scaled,
    uint32_t scaled_stride, float* accumulation,
    CudaButteraugliMaltaParams params, cudaStream_t stream);

// Blur followed by the in-place low/high frequency split. Channels 0/1 use
// 15 taps; 3/4 use 7 taps. Intermediate storage is tightly packed width*height.
// Input and high output must not overlap it or each other. The reference
// additionally materializes the blurred plane for differential qualification.
struct CudaButteraugliFrequencyParams {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t input_stride = 0;
  uint32_t output_stride = 0;
  uint32_t channel = 0;
};

[[nodiscard]] cudaError_t LaunchCudaButteraugliBlurAndSplit(
    float* input, const float* weights, float* intermediate, float* output,
    CudaButteraugliFrequencyParams params, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaButteraugliBlurAndSplitReference(
    float* input, const float* weights, float* intermediate, float* blurred,
    uint32_t blurred_stride, float* output,
    CudaButteraugliFrequencyParams params, cudaStream_t stream);

// Mirrored five-tap RGB blur followed by pointwise Opsin conversion.
// Fused intermediates are three disjoint packed width*height planes. RGB
// inputs, intermediates and XYB outputs must be mutually disjoint. The
// separate-pass oracle may reuse one horizontal intermediate and may alias
// each blurred plane with its output. Blurred/output strides are identical;
// fused blurred pointers are ignored and may be null.
struct CudaButteraugliOpsinPlan {
  std::array<const float*, 3> input{};
  std::array<uint32_t, 3> input_stride{};
  std::array<float*, 3> intermediate{};
  std::array<float*, 3> blurred{};
  std::array<float*, 3> output{};
  const float* weights = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t output_stride = 0;
  float intensity_target = 255.0f;
};

[[nodiscard]] cudaError_t LaunchCudaButteraugliOpsin(
    const CudaButteraugliOpsinPlan& plan, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaButteraugliOpsinReference(
    const CudaButteraugliOpsinPlan& plan, cudaStream_t stream);

// 33-tap separable blur followed by low/medium-frequency construction.
// Each horizontal intermediate is tightly packed width*height. Inputs,
// intermediates, and outputs must be mutually disjoint. Only the reference
// entry writes blurred planes; those pointers may be null for the fused path.
struct CudaButteraugliLowMediumPlan {
  std::array<const float*, 3> input{};
  std::array<float*, 3> intermediate{};
  std::array<float*, 3> blurred{};
  std::array<float*, 3> low{};
  std::array<float*, 3> medium{};
  const float* weights = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t input_stride = 0;
  uint32_t blurred_stride = 0;
  uint32_t output_stride = 0;
};

[[nodiscard]] cudaError_t LaunchCudaButteraugliLowMedium(
    const CudaButteraugliLowMediumPlan& plan, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaButteraugliLowMediumReference(
    const CudaButteraugliLowMediumPlan& plan, cudaStream_t stream);

// Pointwise L2 difference followed by final masking. The fused entry reads
// only the first two Malta AC accumulations and leaves all AC/DC planes
// unchanged. The separate-pass oracle overwrites all six scratch planes.
// Output must not overlap any input or scratch plane; inputs must not overlap
// scratch. The unused fused ac[2] and dc pointers may be null.
struct CudaButteraugliL2FinalPlan {
  std::array<const float*, 8> reference{};
  std::array<const float*, 8> distorted{};
  std::array<float*, 3> ac{};
  std::array<float*, 3> dc{};
  const float* mask = nullptr;
  const float* mask_reference = nullptr;
  const float* mask_distorted = nullptr;
  float* output = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t reference_stride = 0;
  uint32_t distorted_stride = 0;
  uint32_t work_stride = 0;
  uint32_t output_stride = 0;
  float asymmetry = 1.0f;
  float x_multiplier = 1.0f;
};

[[nodiscard]] cudaError_t LaunchCudaButteraugliL2Final(
    const CudaButteraugliL2FinalPlan& plan, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaButteraugliL2FinalReference(
    const CudaButteraugliL2FinalPlan& plan, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaButteraugliPrepare(
    const CudaButteraugliPlan& plan, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaButteraugliCompare(
    const CudaButteraugliPlan& plan, std::array<const float*, 3> distorted,
    std::array<uint32_t, 3> distorted_stride, float* distance_map,
    uint32_t distance_stride, float* score, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
