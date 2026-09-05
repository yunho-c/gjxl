// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gjxl::cuda_internal {

struct CudaAqGeometry {
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int block_width = 0;
  unsigned int block_height = 0;
  unsigned int tile_width = 0;
  unsigned int tile_height = 0;
};

struct CudaAcStrategyBatchParams {
  uint32_t pixel_width = 0;
  uint32_t pixel_height = 0;
  uint32_t opsin_row_stride = 0;
  uint32_t pixel_mask_row_stride = 0;
  uint32_t quant_field_row_stride = 0;
  uint32_t candidate_count = 0;
  uint32_t coefficient_count = 0;
  uint32_t transform_width = 0;
  uint32_t transform_height = 0;
  uint32_t covered_block_width = 0;
  uint32_t covered_block_height = 0;
  uint32_t covered_block_count = 0;
  uint32_t use_device_quant_norm = 0;
  uint32_t color_tile_row_stride = 0;
  uint32_t use_device_cfl = 0;
  float info_loss_multiplier = 0.0f;
  float zeros_multiplier = 0.0f;
  float cost_delta = 0.0f;
};

[[nodiscard]] cudaError_t InitializeCudaDctBasis();

[[nodiscard]] cudaError_t LaunchCudaDct(bool forward, const float* input,
                                        float* output, size_t transform_count,
                                        unsigned int width, unsigned int height,
                                        cudaStream_t stream);

// Gather candidate rectangles into forward-DCT shared memory, without a
// materialized packed-pixel buffer. Shapes/strides are validated by the batch.
[[nodiscard]] cudaError_t LaunchCudaAcStrategyForward(
    const float* opsin_x, const float* opsin_y, const float* opsin_b,
    const void* candidates, float* output, CudaAcStrategyBatchParams params,
    cudaStream_t stream);

// One weighted residual-loss sum per candidate/channel, without materializing
// inverse-transform pixels. Shape and input ranges are validated by the batch.
[[nodiscard]] cudaError_t LaunchCudaAcStrategyInverseLoss(
    const float* input, const float* pixel_mask, const void* candidates,
    float* losses, CudaAcStrategyBatchParams params, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAcStrategyBatch(
    const float* opsin_x, const float* opsin_y, const float* opsin_b,
    const float* pixel_mask, const float* quant_field,
    const signed char* y_to_x, const signed char* y_to_b,
    const float* matrices, const void* candidates, float* scratch_a,
    float* scratch_b, void* rate_scratch, float* costs,
    CudaAcStrategyBatchParams params, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaPointwiseAffine(
    const float* input, float* output, unsigned int width, unsigned int height,
    unsigned int input_stride, unsigned int output_stride, float scale,
    float bias, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaSeparableConvolutionPass(
    bool horizontal, const float* input, const float* kernel, float* output,
    unsigned int width, unsigned int height, unsigned int input_stride,
    unsigned int output_stride, unsigned int kernel_size, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaSymmetric5Convolution(
    const float* input, float* output, unsigned int width, unsigned int height,
    unsigned int input_stride, unsigned int output_stride, float distance0,
    float distance1, float distance2, float distance4, float distance8,
    float distance5, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaMaximumReduction(
    const float* input, float* output, unsigned int width,
    unsigned int input_stride, unsigned int input_count, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqInitialQuantization(
    const float* coding_x, const float* coding_y, const float* coding_b,
    float* unblurred_pixel_mask, float* pixel_mask, float* pre_erosion,
    float* quant_field, float* strategy_mask, unsigned int* selection_state,
    unsigned int* histogram, float* statistics, unsigned int* quantizer_params,
    int* raw_quant, unsigned int* error, CudaAqGeometry geometry,
    float butteraugli_target, float rescale, float quant_dc,
    cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqInitialField(
    const float* coding_x, const float* coding_y, const float* coding_b,
    float* unblurred_pixel_mask, float* pixel_mask, float* pre_erosion,
    float* quant_field, float* strategy_mask, unsigned int* error,
    CudaAqGeometry geometry, float butteraugli_target, float rescale,
    cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqInitialCfl(
    const float* coding_x, const float* coding_y, const float* coding_b,
    signed char* y_to_x, signed char* y_to_b, unsigned int* error,
    CudaAqGeometry geometry, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqSelectResidentQuantizer(
    const float* quant_field, unsigned int block_count,
    unsigned int* selection_state, unsigned int* histogram, float* statistics,
    unsigned int* quantizer_params, int* raw_quant, unsigned int* error,
    float quant_dc, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqEncodeFrame(
    const float* coding_x, const float* coding_y, const float* coding_b,
    const float* transform_x, const float* transform_y,
    const float* transform_b, float* gathered, float* forward_coefficients,
    const float* quant_tables, int* raw_quant,
    const unsigned int* quantizer_params, signed char* y_to_x,
    signed char* y_to_b, float* y_thresholds, int* quantized_ac,
    int* quantized_dc, unsigned int* error, CudaAqGeometry geometry,
    float x_matrix_multiplier, float b_matrix_multiplier, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
