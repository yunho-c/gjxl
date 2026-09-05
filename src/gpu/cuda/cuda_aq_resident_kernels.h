// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "gpu/cuda/cuda_aq_exact_kernels.h"

namespace gjxl::cuda_internal {

struct CudaAqResidentParams {
  uint32_t coding_stride = 0;
  uint32_t block_width = 0;
  uint32_t block_height = 0;
  uint32_t color_stride = 0;
  uint32_t strategy = 0;
  float x_matrix_multiplier = 1.0f;
  float b_matrix_multiplier = 1.0f;
  uint32_t adjust_ac_quant = 0;
  float epf_quant_multiplier = 0.0f;
  float epf_sharpness_lut[8]{};
};

struct CudaAqColorTransformRecord {
  uint32_t coefficient_offset = 0;
  uint32_t channel_stride = 0;
  uint32_t coefficient_count = 0;
  uint32_t strategy = 0;
  uint32_t raw_quant_index = 0;
  uint32_t tile_value_offset = 0;
};

struct CudaAqResidentPolicyParams {
  uint32_t block_count = 0;
  uint32_t score_count = 0;
  uint32_t score_index = 0;
  uint32_t iteration = 0;
  uint32_t apply_update = 0;
  float butteraugli_target = 0.0f;
  float lower_bound = 0.0f;
  float upper_bound = 0.0f;
};

[[nodiscard]] cudaError_t LaunchCudaAqAdjustQuantField(
    const CudaAqAnchor* anchors, float* quant_field, unsigned int* error,
    uint32_t quant_stride, CudaAqExactBatch batch, float mean_max_mixer,
    cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqPositiveRange(
    const float* values, uint32_t count, float* range, unsigned int* error,
    cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqGatherTransformPixels(
    const float* coding_x, const float* coding_y, const float* coding_b,
    const CudaAqAnchor* anchors, float* gathered, CudaAqExactBatch batch,
    uint32_t coding_stride, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqFinalColorCorrelation(
    const CudaAqColorTransformRecord* transforms, const uint32_t* tile_offsets,
    const float* quant_tables, const float* forward_coefficients,
    const int* raw_quant, const unsigned int* quantizer, signed char* y_to_x,
    signed char* y_to_b, unsigned int* error, uint32_t tile_count,
    cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqSelectAdjustedQuantization(
    const CudaAqAnchor* anchors, const float* quant_tables, int* raw_quant,
    const float* forward_coefficients, float* adjustment_thresholds,
    const unsigned int* quantizer, unsigned int* error, CudaAqExactBatch batch,
    CudaAqResidentParams params, cudaStream_t stream);

// Retained serial arithmetic/layout oracle for cooperative quantization tests.
[[nodiscard]] cudaError_t LaunchCudaAqSelectAdjustedQuantizationScalar(
    const CudaAqAnchor* anchors, const float* quant_tables, int* raw_quant,
    const float* forward_coefficients, float* adjustment_thresholds,
    const unsigned int* quantizer, unsigned int* error, CudaAqExactBatch batch,
    CudaAqResidentParams params, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqEncodeResidentCoefficients(
    const CudaAqAnchor* anchors, const float* quant_tables,
    const int* raw_quant, const signed char* y_to_x, const signed char* y_to_b,
    const float* forward_coefficients, int* quantized_coefficients,
    float* reconstruction_coefficients, float* dc, int* quantized_dc,
    float* inverse_sigma, const unsigned char* epf_sharpness,
    const unsigned int* quantizer, const float* adjustment_thresholds,
    unsigned int* error, CudaAqExactBatch batch, CudaAqResidentParams params,
    cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqResidentPolicyInitialize(
    const float* quant_field, float* initial_quant_field, float* scores,
    unsigned int* error, CudaAqResidentPolicyParams params,
    cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqResidentPolicyUpdate(
    float* quant_field, const float* initial_quant_field,
    const float* block_distance, const float* score, float* scores,
    const unsigned int* quantizer, unsigned int* error,
    CudaAqResidentPolicyParams params, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
