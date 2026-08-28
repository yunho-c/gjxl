// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_aq_evaluation_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/gaborish_internal.h"
#include "core/image_ops.h"
#include "core/quantizer.h"
#include "gpu/ops/primitives.h"

namespace gjxl::metal_internal {
namespace {

inline constexpr NS::UInteger kAqThreadCount = 256;

template <typename T>
Status UploadContiguous(MetalBackend &backend, std::span<const T> source,
                        DevicePlaneView destination) {

  return backend.CopyHostToDevice(*destination.buffer, source.data(),
                                  source.size_bytes(),
                                  destination.offset_bytes);
}

void BindPlane(MTL::ComputeCommandEncoder *encoder, DevicePlaneView plane,
               NS::UInteger index) {

  MetalBuffer *buffer = dynamic_cast<MetalBuffer *>(plane.buffer);
  encoder->setBuffer(buffer->handle(), plane.offset_bytes, index);
}

void DispatchThreads1d(MTL::ComputeCommandEncoder *encoder,
                       size_t thread_count) {

  encoder->dispatchThreads(
      MTL::Size(static_cast<NS::UInteger>(thread_count), 1, 1),
      MTL::Size(kAqThreadCount, 1, 1));
}

void DispatchThreads2d(MTL::ComputeCommandEncoder* encoder, Extent2D extent) {
  encoder->dispatchThreads(
      MTL::Size(static_cast<NS::UInteger>(extent.width),
                static_cast<NS::UInteger>(extent.height), 1),
      MTL::Size(8, 8, 1));
}

Status CopyReadback(MetalBackend &backend, DevicePlaneView source,
                    void *destination, size_t size_bytes) {

  return backend.CopyDeviceToHost(*source.buffer, destination, size_bytes,
                                  source.offset_bytes);
}

MetalPreparedAqEvaluation *
AsMetalPrepared(PreparedAqEvaluation &prepared) noexcept {

  return dynamic_cast<MetalPreparedAqEvaluation *>(&prepared);
}

} // namespace

void MetalPreparedAqEvaluation::EncodeReconstructionSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {

  const auto &self = *static_cast<const MetalPreparedAqEvaluation *>(context);
  if (self.exact_linear_reconstruction_) {
    encoder->setComputePipelineState(
        backend.aq_pipelines_.reset_exact_evaluation.get());
    BindPlane(encoder, self.reconstruction_error_, 0);
    BindPlane(encoder, self.block_distance_, 1);
    encoder->setBytes(&self.reset_params_, sizeof(self.reset_params_), 2);
    DispatchThreads1d(encoder, std::max<size_t>(self.block_count_, 1));
    return;
  }

  if (self.exact_coefficient_reconstruction_) {
    encoder->setComputePipelineState(
        backend.aq_pipelines_.reset_exact_coefficients.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      BindPlane(encoder, self.reconstructed_[channel], channel);
    }
    BindPlane(encoder, self.reconstruction_error_, 3);
    BindPlane(encoder, self.block_distance_, 4);
    encoder->setBytes(&self.reset_params_, sizeof(self.reset_params_), 5);
    DispatchThreads1d(
        encoder, std::max({self.pixel_count_, self.block_count_, size_t{1}}));
  } else {
    encoder->setComputePipelineState(
        backend.aq_pipelines_.reset_reconstruction.get());
    BindPlane(encoder, self.gathered_pixels_, 0);
    BindPlane(encoder, self.forward_coefficients_, 1);
    BindPlane(encoder, self.quantized_coefficients_, 2);
    BindPlane(encoder, self.reconstruction_coefficients_, 3);
    BindPlane(encoder, self.dc_, 4);
    BindPlane(encoder, self.quantized_dc_, 5);
    for (size_t channel = 0; channel < 3; ++channel) {
      BindPlane(encoder, self.reconstructed_[channel], channel + 6);
    }
    BindPlane(encoder, self.reconstruction_error_, 9);
    BindPlane(encoder, self.block_distance_, 10);
    encoder->setBytes(&self.reset_params_, sizeof(self.reset_params_), 11);
    DispatchThreads1d(encoder,
                      std::max({self.coefficient_value_count_,
                                3 * self.block_count_, self.pixel_count_,
                                self.block_count_}));
  }

  const MetalBuffer *anchors =
      MetalBackend::AsMetalBuffer(*self.anchors_.buffer);
  const MetalBuffer *gathered =
      MetalBackend::AsMetalBuffer(*self.gathered_pixels_.buffer);
  MetalBuffer *forward =
      MetalBackend::AsMetalBuffer(*self.forward_coefficients_.buffer);
  const MetalBuffer *reconstruction =
      MetalBackend::AsMetalBuffer(*self.reconstruction_coefficients_.buffer);
  MetalBuffer *inverse_output =
      MetalBackend::AsMetalBuffer(*self.gathered_pixels_.buffer);

  for (size_t batch_index = 0; batch_index < self.batches_.size();
       ++batch_index) {
    const AqStrategyBatch &batch = self.batches_[batch_index];
    if (batch.anchor_count == 0)
      continue;
    const AqReconstructionParams &params =
        self.reconstruction_params_[batch_index];
    const size_t batch_value_count =
        3 * batch.anchor_count * batch.coefficient_count;
    const size_t coefficient_offset_bytes =
        batch.coefficient_offset * sizeof(float);

    if (!self.exact_coefficient_reconstruction_) {
      encoder->setComputePipelineState(
          backend.aq_pipelines_.gather_transform_pixels.get());
      for (size_t channel = 0; channel < 3; ++channel) {
        BindPlane(encoder, self.coding_[channel], channel);
      }
      encoder->setBuffer(anchors->handle(), self.anchors_.offset_bytes, 3);
      BindPlane(encoder, self.gathered_pixels_, 4);
      encoder->setBytes(&params, sizeof(params), 5);
      DispatchThreads1d(encoder, batch_value_count);

      backend.EncodeTransformBatch(
          encoder, TransformDirection::kForward, batch.strategy, *gathered,
          self.gathered_pixels_.offset_bytes + coefficient_offset_bytes,
          *forward,
          self.forward_coefficients_.offset_bytes + coefficient_offset_bytes,
          3 * batch.anchor_count);

      encoder->setComputePipelineState(
          backend.aq_pipelines_.encode_reconstruction_coefficients.get());
      BindPlane(encoder, self.anchors_, 0);
      BindPlane(encoder, self.quant_tables_, 1);
      BindPlane(encoder, self.raw_quant_, 2);
      BindPlane(encoder, self.y_to_x_, 3);
      BindPlane(encoder, self.y_to_b_, 4);
      BindPlane(encoder, self.forward_coefficients_, 5);
      BindPlane(encoder, self.quantized_coefficients_, 6);
      BindPlane(encoder, self.reconstruction_coefficients_, 7);
      BindPlane(encoder, self.dc_, 8);
      BindPlane(encoder, self.quantized_dc_, 9);
      BindPlane(encoder, self.reconstruction_error_, 10);
      encoder->setBytes(&params, sizeof(params), 11);
      BindPlane(encoder, self.inverse_sigma_, 12);
      BindPlane(encoder, self.epf_sharpness_, 13);
      encoder->dispatchThreadgroups(
          MTL::Size(static_cast<NS::UInteger>(batch.anchor_count), 1, 1),
          MTL::Size(kAqThreadCount, 1, 1));
    }

    backend.EncodeTransformBatch(
        encoder, TransformDirection::kInverse, batch.strategy, *reconstruction,
        self.reconstruction_coefficients_.offset_bytes +
            coefficient_offset_bytes,
        *inverse_output,
        self.gathered_pixels_.offset_bytes + coefficient_offset_bytes,
        3 * batch.anchor_count);

    encoder->setComputePipelineState(
        backend.aq_pipelines_.scatter_reconstructed_pixels.get());
    BindPlane(encoder, self.anchors_, 0);
    BindPlane(encoder, self.gathered_pixels_, 1);
    for (size_t channel = 0; channel < 3; ++channel) {
      BindPlane(encoder, self.reconstructed_[channel], channel + 2);
    }
    encoder->setBytes(&params, sizeof(params), 5);
    DispatchThreads1d(encoder, batch_value_count);
  }
}

void MetalPreparedAqEvaluation::EncodeFrameSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {

  const auto &self = *static_cast<const MetalPreparedAqEvaluation *>(context);
  encoder->setComputePipelineState(
      backend.aq_pipelines_.reset_frame_encoding.get());
  BindPlane(encoder, self.quantized_coefficients_, 0);
  BindPlane(encoder, self.quantized_dc_, 1);
  BindPlane(encoder, self.reconstruction_error_, 2);
  encoder->setBytes(&self.reset_params_, sizeof(self.reset_params_), 3);
  DispatchThreads1d(encoder,
                    std::max(self.coefficient_value_count_,
                             3 * self.block_count_));

  if (self.frame_only_resident_initial_cfl_) {
    encoder->setComputePipelineState(backend.aq_pipelines_.initial_cfl.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      BindPlane(encoder, self.coding_[channel], channel);
    }
    BindPlane(encoder, self.y_to_x_, 3);
    BindPlane(encoder, self.y_to_b_, 4);
    BindPlane(encoder, self.reconstruction_error_, 5);
    encoder->setBytes(&self.initial_cfl_params_,
                      sizeof(self.initial_cfl_params_), 6);
    DispatchThreads1d(encoder, self.tile_extent_.width *
                                   self.tile_extent_.height);
  }

  if (self.frame_only_inverse_gaborish_) {
    for (size_t channel = 0; channel < 3; ++channel) {
      const Symmetric5Weights weights =
          gaborish_internal::GaborishInverseWeights(
              self.options_.profile.gaborish_inverse_multipliers[channel]);
      backend.EncodePrimitive(
          encoder,
          Symmetric5ConvolutionCommand{
              .input = self.coding_[channel],
              .output = self.reconstructed_[channel],
              .weights = {
                  weights.distance0,
                  weights.distance1,
                  weights.distance2,
                  weights.distance4,
                  weights.distance8,
                  weights.distance5,
              },
          });
    }
  }

  const MetalBuffer *anchors =
      MetalBackend::AsMetalBuffer(*self.anchors_.buffer);
  const MetalBuffer *gathered =
      MetalBackend::AsMetalBuffer(*self.gathered_pixels_.buffer);
  MetalBuffer *forward =
      MetalBackend::AsMetalBuffer(*self.forward_coefficients_.buffer);
  for (size_t batch_index = 0; batch_index < self.batches_.size();
       ++batch_index) {
    const AqStrategyBatch &batch = self.batches_[batch_index];
    if (batch.anchor_count == 0)
      continue;
    const AqReconstructionParams &params =
        self.reconstruction_params_[batch_index];
    const size_t batch_value_count =
        3 * batch.anchor_count * batch.coefficient_count;
    const size_t coefficient_offset_bytes =
        batch.coefficient_offset * sizeof(float);

    encoder->setComputePipelineState(
        backend.aq_pipelines_.gather_transform_pixels.get());
    for (size_t channel = 0; channel < 3; ++channel) {
      BindPlane(encoder,
                self.frame_only_inverse_gaborish_
                    ? self.reconstructed_[channel]
                    : self.coding_[channel],
                channel);
    }
    encoder->setBuffer(anchors->handle(), self.anchors_.offset_bytes, 3);
    BindPlane(encoder, self.gathered_pixels_, 4);
    encoder->setBytes(&params, sizeof(params), 5);
    DispatchThreads1d(encoder, batch_value_count);

    backend.EncodeTransformBatch(
        encoder, TransformDirection::kForward, batch.strategy, *gathered,
        self.gathered_pixels_.offset_bytes + coefficient_offset_bytes,
        *forward,
        self.forward_coefficients_.offset_bytes + coefficient_offset_bytes,
        3 * batch.anchor_count);

    encoder->setComputePipelineState(
        backend.aq_pipelines_.encode_frame_coefficients.get());
    BindPlane(encoder, self.anchors_, 0);
    BindPlane(encoder, self.quant_tables_, 1);
    BindPlane(encoder, self.raw_quant_, 2);
    BindPlane(encoder, self.y_to_x_, 3);
    BindPlane(encoder, self.y_to_b_, 4);
    BindPlane(encoder, self.forward_coefficients_, 5);
    BindPlane(encoder, self.quantized_coefficients_, 6);
    BindPlane(encoder, self.quantized_dc_, 7);
    BindPlane(encoder, self.reconstruction_error_, 8);
    encoder->setBytes(&params, sizeof(params), 9);
    encoder->dispatchThreadgroups(
        MTL::Size(static_cast<NS::UInteger>(batch.anchor_count), 1, 1),
        MTL::Size(kAqThreadCount, 1, 1));
  }
}

void MetalPreparedAqEvaluation::EncodeInitialQuantizationSubmission(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
    const void* context) {

  const auto& self =
      *static_cast<const MetalPreparedAqEvaluation*>(context);
  encoder->setComputePipelineState(
      backend.aq_pipelines_.reset_initial_quant.get());
  BindPlane(encoder, self.reconstruction_error_, 0);
  encoder->setBytes(&self.initial_quant_gradient_params_,
                    sizeof(self.initial_quant_gradient_params_), 1);
  DispatchThreads1d(encoder, 1);

  if (self.resident_ac_strategy_inputs_ &&
      self.options_.profile.loop_filter.gaborish) {
    for (size_t channel = 0; channel < 3; ++channel) {
      const Symmetric5Weights weights =
          gaborish_internal::GaborishInverseWeights(
              self.options_.profile.gaborish_inverse_multipliers[channel]);
      backend.EncodePrimitive(
          encoder,
          Symmetric5ConvolutionCommand{
              .input = self.coding_[channel],
              .output = self.reconstructed_[channel],
              .weights = {
                  weights.distance0,
                  weights.distance1,
                  weights.distance2,
                  weights.distance4,
                  weights.distance8,
                  weights.distance5,
              },
          });
    }
  }

  encoder->setComputePipelineState(
      backend.aq_pipelines_.initial_quant_gradient.get());
  BindPlane(encoder, self.coding_[1], 0);
  BindPlane(encoder, self.initial_quant_unblurred_pixel_mask_, 1);
  BindPlane(encoder, self.initial_quant_pre_erosion_, 2);
  BindPlane(encoder, self.reconstruction_error_, 3);
  encoder->setBytes(&self.initial_quant_gradient_params_,
                    sizeof(self.initial_quant_gradient_params_), 4);
  DispatchThreads2d(encoder, self.initial_quant_pre_erosion_.extent);

  encoder->setComputePipelineState(
      backend.aq_pipelines_.initial_quant_fuzzy_erosion.get());
  BindPlane(encoder, self.initial_quant_pre_erosion_, 0);
  BindPlane(encoder, self.initial_quant_field_, 1);
  BindPlane(encoder, self.initial_quant_strategy_mask_, 2);
  BindPlane(encoder, self.reconstruction_error_, 3);
  encoder->setBytes(&self.initial_quant_erosion_params_,
                    sizeof(self.initial_quant_erosion_params_), 4);
  DispatchThreads2d(encoder, self.block_extent_);

  encoder->setComputePipelineState(
      backend.aq_pipelines_.initial_quant_modulation.get());
  for (size_t channel = 0; channel < 3; ++channel) {
    BindPlane(encoder, self.coding_[channel], channel);
  }
  BindPlane(encoder, self.initial_quant_field_, 3);
  BindPlane(encoder, self.reconstruction_error_, 4);
  encoder->setBytes(&self.initial_quant_modulation_params_,
                    sizeof(self.initial_quant_modulation_params_), 5);
  DispatchThreads2d(encoder, self.block_extent_);

  constexpr std::array<float, 5> kFilter = {
      0.364911248f, 0.05f, 0.1688888021f, 0.221069183f, 0.306563504f};
  constexpr double kWeightSum =
      1.0 + 4.0 * (kFilter[0] + kFilter[1] + kFilter[2] + kFilter[4] +
                   2.0 * kFilter[3]);
  constexpr float kNormalize = static_cast<float>(1.0 / kWeightSum);
  backend.EncodePrimitive(
      encoder,
      Symmetric5ConvolutionCommand{
          .input = self.initial_quant_unblurred_pixel_mask_,
          .output = self.initial_quant_pixel_mask_,
          .weights = {
              kNormalize,
              kNormalize * kFilter[0],
              kNormalize * kFilter[2],
              kNormalize * kFilter[1],
              kNormalize * kFilter[4],
              kNormalize * kFilter[3],
          },
      });

  if (!self.frame_only_resident_quantizer_) return;
  encoder->setComputePipelineState(
      backend.aq_pipelines_.initial_quant_sort_prepare.get());
  BindPlane(encoder, self.initial_quant_field_, 0);
  BindPlane(encoder, self.initial_quant_sort_, 1);
  encoder->setBytes(&self.initial_quant_selection_params_,
                    sizeof(self.initial_quant_selection_params_), 2);
  DispatchThreads1d(encoder, self.initial_quant_sort_count_);

  const auto encode_sort = [&] {
    encoder->setComputePipelineState(
        backend.aq_pipelines_.initial_quant_sort_step.get());
    for (size_t sequence = 2; sequence <= self.initial_quant_sort_count_;
         sequence *= 2) {
      for (size_t distance = sequence / 2; distance != 0; distance /= 2) {
        const AqInitialQuantSortParams params{
            static_cast<uint32_t>(distance),
            static_cast<uint32_t>(sequence),
            static_cast<uint32_t>(self.initial_quant_sort_count_),
        };
        BindPlane(encoder, self.initial_quant_sort_, 0);
        encoder->setBytes(&params, sizeof(params), 1);
        DispatchThreads1d(encoder, self.initial_quant_sort_count_);
      }
    }
  };
  encode_sort();

  encoder->setComputePipelineState(
      backend.aq_pipelines_.initial_quant_capture_median.get());
  BindPlane(encoder, self.initial_quant_sort_, 0);
  BindPlane(encoder, self.initial_quant_median_, 1);
  encoder->setBytes(&self.initial_quant_selection_params_,
                    sizeof(self.initial_quant_selection_params_), 2);
  DispatchThreads1d(encoder, 1);

  encoder->setComputePipelineState(
      backend.aq_pipelines_.initial_quant_deviation_prepare.get());
  BindPlane(encoder, self.initial_quant_field_, 0);
  BindPlane(encoder, self.initial_quant_median_, 1);
  BindPlane(encoder, self.initial_quant_sort_, 2);
  encoder->setBytes(&self.initial_quant_selection_params_,
                    sizeof(self.initial_quant_selection_params_), 3);
  DispatchThreads1d(encoder, self.initial_quant_sort_count_);
  encode_sort();

  encoder->setComputePipelineState(
      backend.aq_pipelines_.initial_quant_finalize_quantizer.get());
  BindPlane(encoder, self.initial_quant_median_, 0);
  BindPlane(encoder, self.initial_quant_sort_, 1);
  BindPlane(encoder, self.initial_quantizer_params_, 2);
  BindPlane(encoder, self.reconstruction_error_, 3);
  encoder->setBytes(&self.initial_quant_selection_params_,
                    sizeof(self.initial_quant_selection_params_), 4);
  DispatchThreads1d(encoder, 1);

  encoder->setComputePipelineState(
      backend.aq_pipelines_.initial_quant_raw_quant.get());
  BindPlane(encoder, self.initial_quant_field_, 0);
  BindPlane(encoder, self.initial_quantizer_params_, 1);
  BindPlane(encoder, self.raw_quant_, 2);
  BindPlane(encoder, self.reconstruction_error_, 3);
  encoder->setBytes(&self.initial_quant_selection_params_,
                    sizeof(self.initial_quant_selection_params_), 4);
  DispatchThreads2d(encoder, self.block_extent_);
}

Status MetalPreparedAqEvaluation::ComputeInitialQuantization(
    InitialQuantizationOptions options, InitialQuantFieldOutput output,
    QuantizerParams* quantizer, float quant_dc) {

  return ComputeInitialQuantizationImpl(
    options, output, quantizer, quant_dc, nullptr);
}

Status MetalPreparedAqEvaluation::ComputeInitialQuantizationProfiled(
    InitialQuantizationOptions options, InitialQuantFieldOutput output,
    QuantizerParams* quantizer, float quant_dc,
    gpu_profile_internal::FrameEncodingProfile* profile) {

  if (profile == nullptr) {
    return Status::InvalidArgument(
      "Resident initial-quantization profile output is null");
  }
  gpu_profile_internal::FrameEncodingProfile candidate;
  Status status = ComputeInitialQuantizationImpl(
    options, output, quantizer, quant_dc, &candidate);
  if (status.ok()) {
    *profile = candidate;
  }
  return status;
}

Status MetalPreparedAqEvaluation::ComputeInitialQuantizationImpl(
    InitialQuantizationOptions options, InitialQuantFieldOutput output,
    QuantizerParams* quantizer, float quant_dc,
    gpu_profile_internal::FrameEncodingProfile* profile) {

  if (!frame_only_resident_initial_quant_) {
    return Status::FailedPrecondition(
        "Resident initial quantization was not prepared");
  }
  if (!std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f || !std::isfinite(options.rescale) ||
      options.rescale <= 0.0f || !output.quant_field.valid() ||
      output.quant_field.extent != block_extent_ ||
      !output.strategy_mask.valid() ||
      output.strategy_mask.extent != block_extent_ ||
      !output.pixel_mask.valid() ||
      output.pixel_mask.extent != coding_extent_) {
    return Status::InvalidArgument(
        "Resident initial quantization inputs or outputs are invalid");
  }
  if (frame_only_resident_quantizer_ &&
      (!std::isfinite(quant_dc) || quant_dc <= 0.0f ||
       quant_dc > static_cast<float>(kMaxQuantDc))) {
    return Status::InvalidArgument(
        "Resident initial quantizer DC input is invalid");
  }

  constexpr std::array<float, 4> kMulBase = {0.125f, 0.1f, 0.09f, 0.06f};
  constexpr std::array<float, 4> kMulAdd = {0.0f, -0.1f, -0.09f, -0.06f};
  constexpr float kTotal = 0.29959705784054957f;
  const float target_mix = options.butteraugli_target < 2.0f
      ? (2.0f - options.butteraugli_target) * 0.5f
      : 0.0f;
  float weight_sum = 0.0f;
  for (size_t index = 0; index < 4; ++index) {
    initial_quant_erosion_params_.weights[index] =
        kMulBase[index] + target_mix * kMulAdd[index];
    weight_sum += initial_quant_erosion_params_.weights[index];
  }
  for (float& weight : initial_quant_erosion_params_.weights) {
    weight *= kTotal / weight_sum;
  }
  constexpr float kAcQuant = 0.765f;
  const float scale =
      kAcQuant / options.butteraugli_target * options.rescale;
  const float base_level = 0.48f * scale;
  float dampen = 1.0f;
  if (options.butteraugli_target >= 2.0f) {
    dampen = 1.0f - (options.butteraugli_target - 2.0f) / 12.0f;
    dampen = std::max(dampen, 0.0f);
  }
  initial_quant_modulation_params_.multiplier = scale * dampen;
  initial_quant_modulation_params_.addend = (1.0f - dampen) * base_level;
  if (frame_only_resident_quantizer_) {
    initial_quant_selection_params_.quant_dc = quant_dc;
    initial_quant_selection_params_.scaled_quant_dc =
        static_cast<uint32_t>(static_cast<int32_t>(
            static_cast<double>(quant_dc * 4096.0f) * 1.6));
  }

  Status status = BeginOperation();
  if (!status.ok()) return status;
  bool fail_upload = false;
  bool fail_numeric = false;
  {
    std::lock_guard lock(mutex_);
    fail_upload = fail_next_upload_;
    fail_numeric = fail_next_numeric_;
    fail_next_upload_ = false;
    fail_next_numeric_ = false;
  }
  resident_initial_quant_ready_ = false;
  resident_quantizer_ready_ = false;
  if (fail_upload) {
    Invalidate();
    return Status::DeviceError(
        "Injected Metal initial-quantization upload failure");
  }
  initial_quant_gradient_params_.test_error_mask = fail_numeric ? 16384u : 0u;
  std::unique_ptr<GpuSubmission> submission;
  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->submission);
  status = backend_->SubmitCompute(
      "gjxl prepared initial quantization",
      &MetalPreparedAqEvaluation::EncodeInitialQuantizationSubmission, this,
      &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
        ? Status::Internal("Initial quantization returned no submission")
        : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->submission);
  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->completion_wait);
  status = WaitForOperation(
    profile == nullptr ? nullptr : &profile->command_buffer);
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->completion_wait);
  if (!status.ok()) return status;

  if (profile != nullptr) {
    profile->command_buffer.host_envelope_aligned =
      profile->command_buffer.start_host_nanoseconds >=
        profile->submission.begin_nanoseconds &&
      profile->command_buffer.end_host_nanoseconds <=
        profile->completion_wait.end_nanoseconds;
  }

  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->readback);
  uint32_t device_error = 0;
  status = CopyReadback(*backend_, reconstruction_error_, &device_error,
                        sizeof(device_error));
  if (status.ok() && device_error != 0) {
    status = Status::DeviceError(
        "Metal initial quantization detected an invalid numeric result");
  }
  if (status.ok()) {
    status = CopyReadback(
        *backend_, initial_quant_field_, last_initial_quant_field_.data(),
        last_initial_quant_field_.size() * sizeof(float));
  }
  if (status.ok()) {
    status = CopyReadback(
        *backend_, initial_quant_strategy_mask_,
        last_initial_strategy_mask_.data(),
        last_initial_strategy_mask_.size() * sizeof(float));
  }
  if (status.ok()) {
    status = CopyReadback(
        *backend_, initial_quant_pixel_mask_, last_initial_pixel_mask_.data(),
        last_initial_pixel_mask_.size() * sizeof(float));
  }
  QuantizerParams device_quantizer;
  if (status.ok() && frame_only_resident_quantizer_) {
    status = CopyReadback(*backend_, initial_quantizer_params_,
                          &device_quantizer, sizeof(device_quantizer));
    if (status.ok()) {
      status = Quantizer::Create(device_quantizer, &last_quantizer_);
    }
  }
  const auto valid_values = [](const std::vector<float>& values) {
    return std::ranges::all_of(values, [](float value) {
      return std::isfinite(value) && value > 0.0f;
    });
  };
  if (status.ok() &&
      (!valid_values(last_initial_quant_field_) ||
       !valid_values(last_initial_strategy_mask_) ||
       !valid_values(last_initial_pixel_mask_))) {
    status = Status::DeviceError(
        "Metal initial quantization readback is invalid");
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->readback);
  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->frame_assembly);
  CopyContiguousPlane(last_initial_quant_field_, output.quant_field);
  CopyContiguousPlane(last_initial_strategy_mask_, output.strategy_mask);
  CopyContiguousPlane(last_initial_pixel_mask_, output.pixel_mask);
  resident_initial_quant_ready_ = true;
  if (frame_only_resident_quantizer_) {
    resident_quantizer_ready_ = true;
    if (quantizer != nullptr) *quantizer = device_quantizer;
  }
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->frame_assembly);
  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::GetResidentAcStrategyInputs(
    ResidentAcStrategyInputs* inputs) {
  if (inputs == nullptr) {
    return Status::InvalidArgument(
        "Resident AC-strategy input output is null");
  }
  std::lock_guard lock(mutex_);
  if (!resident_ac_strategy_inputs_) {
    return Status::FailedPrecondition(
        "Resident AC-strategy inputs were not prepared");
  }
  if (state_ == State::kInvalid) {
    return Status::FailedPrecondition(
        "Prepared AQ evaluation is invalid");
  }
  if (state_ == State::kBusy) {
    return Status::FailedPrecondition(
        "Prepared AQ evaluation is already in use");
  }
  if (!resident_initial_quant_ready_) {
    return Status::FailedPrecondition(
        "Resident initial quantization has not been computed");
  }
  const std::array<DevicePlaneView, 3>& search_opsin =
      options_.profile.loop_filter.gaborish ? reconstructed_ : coding_;
  *inputs = {
      .opsin = {{{search_opsin[0], search_opsin[1], search_opsin[2]}}},
      .quant_field = initial_quant_field_,
      .pixel_mask = initial_quant_pixel_mask_,
  };
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::EncodeFrame(
    AqEvaluationInput input, VarDctEncoderFrame *frame) {

  return EncodeFrameImpl(input, frame, nullptr);
}

Status MetalPreparedAqEvaluation::EncodeFrameProfiled(
    AqEvaluationInput input, VarDctEncoderFrame* frame,
    gpu_profile_internal::FrameEncodingProfile* profile) {

  if (profile == nullptr) {
    return Status::InvalidArgument(
      "AQ frame-only profile output is null");
  }
  gpu_profile_internal::FrameEncodingProfile candidate;
  Status status = EncodeFrameImpl(input, frame, &candidate);
  if (status.ok()) {
    *profile = candidate;
  }
  return status;
}

Status MetalPreparedAqEvaluation::EncodeFrameImpl(
    AqEvaluationInput input, VarDctEncoderFrame* frame,
    gpu_profile_internal::FrameEncodingProfile* profile) {

  if (frame == nullptr) {
    return Status::InvalidArgument("AQ frame-only output is null");
  }
  if (!frame_only_) {
    return Status::FailedPrecondition(
        "AQ frame-only encoding requires frame-only preparation");
  }
  Status status = ValidateInput(input);
  if (!status.ok())
    return status;
  if (input.exact_coefficients != nullptr ||
      input.exact_reconstructed_linear_rgb.valid()) {
    return Status::InvalidArgument(
        "AQ frame-only encoding requires resident coefficients");
  }
  status = BeginOperation();
  if (!status.ok())
    return status;
  bool fail_upload = false;
  bool fail_numeric = false;
  {
    std::lock_guard lock(mutex_);
    fail_upload = fail_next_upload_;
    fail_numeric = fail_next_numeric_;
    fail_next_upload_ = false;
    fail_next_numeric_ = false;
  }
  if (fail_upload) {
    Invalidate();
    return Status::DeviceError("Injected Metal AQ upload failure");
  }
  reset_params_.test_error_mask = fail_numeric ? 512u : 0u;
  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->input_upload);
  status = UploadInput(input);
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  for (AqReconstructionParams &params : reconstruction_params_) {
    params.global_scale = input.quantizer.global_scale;
    params.quant_dc = input.quantizer.quant_dc;
  }
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->input_upload);

  std::unique_ptr<GpuSubmission> submission;
  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->submission);
  status = backend_->SubmitCompute(
      "gjxl prepared AQ frame encoding",
      &MetalPreparedAqEvaluation::EncodeFrameSubmission, this, &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
               ? Status::Internal("AQ frame encoding returned no submission")
               : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->submission);
  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->completion_wait);
  status = WaitForOperation(
    profile == nullptr ? nullptr : &profile->command_buffer);
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->completion_wait);
  if (!status.ok())
    return status;

  if (profile != nullptr) {
    profile->command_buffer.host_envelope_aligned =
      profile->command_buffer.start_host_nanoseconds >=
        profile->submission.begin_nanoseconds &&
      profile->command_buffer.end_host_nanoseconds <=
        profile->completion_wait.end_nanoseconds;
  }

  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->readback);
  uint32_t device_error = 0;
  status = CopyReadback(*backend_, reconstruction_error_, &device_error,
                        sizeof(device_error));
  if (status.ok() && device_error != 0) {
    status = Status::DeviceError(
        "Metal AQ frame encoding detected invalid numeric input");
  }
  if (status.ok()) {
    status = CopyReadback(*backend_, quantized_coefficients_,
                          quantized_readback_.data(),
                          quantized_readback_.size() * sizeof(int32_t));
  }
  if (status.ok()) {
    status = CopyReadback(*backend_, quantized_dc_,
                          quantized_dc_readback_.data(),
                          quantized_dc_readback_.size() * sizeof(int32_t));
  }
  if (status.ok() &&
      (coefficient_decision_mode_ ==
           AcCoefficientDecisionMode::kAdjustedSharedQuant ||
       frame_only_resident_quantizer_)) {
    status = ReadbackRawQuant();
  }
  if (status.ok() && frame_only_resident_initial_cfl_) {
    status = ReadbackColorCorrelation();
  }
  constexpr int32_t kQuantizedPoison =
      static_cast<int32_t>(0x81234567u);
  if (status.ok() &&
      (std::ranges::find(quantized_readback_, kQuantizedPoison) !=
           quantized_readback_.end() ||
       std::ranges::find(quantized_dc_readback_, kQuantizedPoison) !=
           quantized_dc_readback_.end())) {
    status = Status::Internal(
        "Metal AQ frame encoding readback contains poison");
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->readback);

  profile_internal::Begin(
    profile == nullptr ? nullptr : &profile->frame_assembly);
  VarDctEncoderFrame candidate;
  status = AssembleFrameFromReadback(&candidate);
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  *frame = std::move(candidate);
  profile_internal::End(
    profile == nullptr ? nullptr : &profile->frame_assembly);
  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::RunReconstruction(
    AqEvaluationInput input,
    MetalAqReconstructionSnapshotForTesting *snapshot) {

  if (snapshot == nullptr) {
    return Status::InvalidArgument("AQ reconstruction snapshot output is null");
  }
  Status status = ValidateInput(input);
  if (!status.ok())
    return status;
  status = BeginOperation();
  if (!status.ok())
    return status;
  status = PrepareExactCoefficientStaging(input);
  if (status.ok()) {
    status = PrepareReconstructionDiagnosticReadback();
  }
  if (!status.ok()) {
    CompleteOperation();
    return status;
  }
  status = UploadInput(input);
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  for (AqReconstructionParams &params : reconstruction_params_) {
    params.global_scale = input.quantizer.global_scale;
    params.quant_dc = input.quantizer.quant_dc;
  }

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
      "gjxl prepared AQ coefficient reconstruction",
      &MetalPreparedAqEvaluation::EncodeReconstructionSubmission, this,
      &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
               ? Status::Internal("AQ reconstruction returned no submission")
               : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  status = WaitForOperation();
  if (!status.ok())
    return status;

  uint32_t device_error = 0;
  status = CopyReadback(*backend_, reconstruction_error_, &device_error,
                        sizeof(device_error));
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  if (device_error != 0) {
    Invalidate();
    return Status::DeviceError(
        "Metal AQ coefficient reconstruction detected invalid numeric input");
  }
  status =
      CopyReadback(*backend_, forward_coefficients_, forward_readback_.data(),
                   forward_readback_.size() * sizeof(float));
  if (status.ok()) {
    status = CopyReadback(*backend_, quantized_coefficients_,
                          quantized_readback_.data(),
                          quantized_readback_.size() * sizeof(int32_t));
  }
  if (status.ok()) {
    status = CopyReadback(*backend_, dc_, dc_readback_.data(),
                          dc_readback_.size() * sizeof(float));
  }
  if (status.ok()) {
    status = ReadbackRawQuant();
  }
  const size_t inverse_sigma_row_bytes =
      block_extent_.width * sizeof(float);
  for (size_t y = 0; status.ok() && y < block_extent_.height; ++y) {
    status = backend_->CopyDeviceToHost(
        *inverse_sigma_.buffer,
        readback_.data() + y * block_extent_.width,
        inverse_sigma_row_bytes,
        inverse_sigma_.offset_bytes +
            y * inverse_sigma_.row_stride * sizeof(float));
  }
  for (size_t channel = 0; status.ok() && channel < 3; ++channel) {
    status =
        CopyReadback(*backend_, reconstructed_[channel],
                     reconstructed_readback_[channel].data(),
                     reconstructed_readback_[channel].size() * sizeof(float));
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }

  try {
    MetalAqReconstructionSnapshotForTesting result;
    result.block_extent = block_extent_;
    result.pixel_extent = coding_extent_;
    result.raw_quant = last_raw_quant_;
    result.epf_inverse_sigma = readback_;
    result.transforms.reserve(row_major_anchors_.size());
    for (const AqAnchor &anchor : row_major_anchors_) {
      const AqStrategyBatch &batch = batches_[anchor.batch_index];
      MetalAqTransformSnapshotForTesting transform{
          .block_x = anchor.block_x,
          .block_y = anchor.block_y,
          .strategy = anchor.strategy,
      };
      const size_t channel_stride =
          batch.anchor_count * batch.coefficient_count;
      for (size_t channel = 0; channel < 3; ++channel) {
        const size_t offset = batch.coefficient_offset +
                              channel * channel_stride +
                              anchor.index_in_batch * batch.coefficient_count;
        transform.forward_coefficients[channel].assign(
            forward_readback_.begin() + static_cast<std::ptrdiff_t>(offset),
            forward_readback_.begin() +
                static_cast<std::ptrdiff_t>(offset + batch.coefficient_count));
        transform.quantized_coefficients[channel].assign(
            quantized_readback_.begin() + static_cast<std::ptrdiff_t>(offset),
            quantized_readback_.begin() +
                static_cast<std::ptrdiff_t>(offset + batch.coefficient_count));
      }
      result.transforms.push_back(std::move(transform));
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      result.dc[channel].assign(
          dc_readback_.begin() +
              static_cast<std::ptrdiff_t>(channel * block_count_),
          dc_readback_.begin() +
              static_cast<std::ptrdiff_t>((channel + 1) * block_count_));
      result.reconstructed_opsin[channel] = reconstructed_readback_[channel];
    }
    *snapshot = std::move(result);
  } catch (const std::bad_alloc &) {
    CompleteOperation();
    return Status::OutOfMemory(
        "Unable to allocate AQ reconstruction diagnostic snapshot");
  } catch (const std::length_error &) {
    CompleteOperation();
    return Status::InvalidArgument(
        "AQ reconstruction diagnostic snapshot is too large");
  }

  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::PrepareReconstructionDiagnosticReadback() {
  try {
    forward_readback_.resize(coefficient_value_count_);
    dc_readback_.resize(3 * block_count_);
    for (std::vector<float> &plane : reconstructed_readback_) {
      plane.resize(pixel_count_);
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory(
        "Unable to allocate AQ reconstruction diagnostic readback");
  } catch (const std::length_error &) {
    return Status::InvalidArgument(
        "AQ reconstruction diagnostic readback is too large");
  }
}

void MetalPreparedAqEvaluation::EncodeQuantizationProbeSubmission(
    MetalBackend &backend, MTL::ComputeCommandEncoder *encoder,
    const void *context) {

  const auto &self = *static_cast<const MetalPreparedAqEvaluation *>(context);
  encoder->setComputePipelineState(
      backend.aq_pipelines_.quantization_probe.get());
  BindPlane(encoder, self.quant_probe_input_, 0);
  BindPlane(encoder, self.quant_tables_, 1);
  BindPlane(encoder, self.quant_probe_quantized_, 2);
  BindPlane(encoder, self.quant_probe_dequantized_, 3);
  BindPlane(encoder, self.reconstruction_error_, 4);
  encoder->setBytes(&self.quant_probe_params_, sizeof(self.quant_probe_params_),
                    5);
  DispatchThreads1d(encoder, self.quant_probe_params_.coefficient_count);
}

Status MetalPreparedAqEvaluation::RunQuantizationProbe(
    const MetalAqQuantizationProbeForTesting &probe,
    std::vector<int32_t> *quantized, std::vector<float> *dequantized) {

  if (quantized == nullptr || dequantized == nullptr) {
    return Status::InvalidArgument("AQ quantization probe output is null");
  }
  const AcStrategyInfo *info = GetAcStrategyInfo(probe.strategy);
  if (info == nullptr || info->coefficient_count() == 0 ||
      info->coefficient_count() > maximum_coefficient_count_ ||
      probe.coefficients.size() != info->coefficient_count()) {
    return Status::InvalidArgument(
        "AQ quantization probe strategy or coefficient count is invalid");
  }
  if (static_cast<uint8_t>(probe.channel) >
          static_cast<uint8_t>(XybChannel::kB) ||
      probe.raw_quant < 1 || probe.raw_quant > kMaxRawQuant ||
      !std::isfinite(probe.matrix_multiplier) ||
      probe.matrix_multiplier <= 0.0f) {
    return Status::InvalidArgument(
        "AQ quantization probe parameters are invalid");
  }
  Quantizer quantizer;
  Status status = Quantizer::Create(probe.quantizer, &quantizer);
  if (!status.ok())
    return status;
  status = BeginOperation();
  if (!status.ok())
    return status;
  status = PrepareQuantizationProbeReadback();
  if (!status.ok()) {
    CompleteOperation();
    return status;
  }
  status = UploadContiguous(*backend_, probe.coefficients, quant_probe_input_);
  const uint32_t zero = 0;
  if (status.ok()) {
    status = backend_->CopyHostToDevice(*reconstruction_error_.buffer, &zero,
                                        sizeof(zero),
                                        reconstruction_error_.offset_bytes);
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  quant_probe_params_ = {
      static_cast<uint32_t>(info->coefficient_count()),
      static_cast<uint32_t>(probe.strategy),
      static_cast<uint32_t>(probe.channel),
      probe.raw_quant,
      probe.quantizer.global_scale,
      probe.matrix_multiplier,
  };

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
      "gjxl AQ quantization probe",
      &MetalPreparedAqEvaluation::EncodeQuantizationProbeSubmission, this,
      &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok() ? Status::Internal(
                             "AQ quantization probe returned no submission")
                       : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  status = WaitForOperation();
  if (!status.ok())
    return status;

  uint32_t device_error = 0;
  status = CopyReadback(*backend_, reconstruction_error_, &device_error,
                        sizeof(device_error));
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  if (device_error != 0) {
    Invalidate();
    return Status::DeviceError(
        "Metal AQ quantization probe detected non-finite or overflowing input");
  }
  const size_t count = info->coefficient_count();
  status = CopyReadback(*backend_, quant_probe_quantized_,
                        quant_probe_quantized_readback_.data(),
                        count * sizeof(int32_t));
  if (status.ok()) {
    status = CopyReadback(*backend_, quant_probe_dequantized_,
                          quant_probe_dequantized_readback_.data(),
                          count * sizeof(float));
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  try {
    std::vector<int32_t> quantized_result(
        quant_probe_quantized_readback_.begin(),
        quant_probe_quantized_readback_.begin() +
            static_cast<std::ptrdiff_t>(count));
    std::vector<float> dequantized_result(
        quant_probe_dequantized_readback_.begin(),
        quant_probe_dequantized_readback_.begin() +
            static_cast<std::ptrdiff_t>(count));
    *quantized = std::move(quantized_result);
    *dequantized = std::move(dequantized_result);
  } catch (const std::bad_alloc &) {
    CompleteOperation();
    return Status::OutOfMemory(
        "Unable to allocate AQ quantization probe output");
  } catch (const std::length_error &) {
    CompleteOperation();
    return Status::InvalidArgument("AQ quantization probe output is too large");
  }
  CompleteOperation();
  return Status::Ok();
}

void MetalPreparedAqEvaluation::EncodeAdjustmentProbeSubmission(
    MetalBackend& backend, MTL::ComputeCommandEncoder* encoder,
    const void* context) {

  const auto& self = *static_cast<const MetalPreparedAqEvaluation*>(context);
  encoder->setComputePipelineState(backend.aq_pipelines_.adjustment_probe.get());
  BindPlane(encoder, self.forward_coefficients_, 0);
  BindPlane(encoder, self.quant_tables_, 1);
  BindPlane(encoder, self.quantized_coefficients_, 2);
  BindPlane(encoder, self.quant_probe_quantized_, 3);
  BindPlane(encoder, self.quant_probe_dequantized_, 4);
  BindPlane(encoder, self.reconstruction_error_, 5);
  encoder->setBytes(
      &self.adjustment_probe_params_, sizeof(self.adjustment_probe_params_), 6);
  DispatchThreads1d(encoder, 1);
}

Status MetalPreparedAqEvaluation::RunAdjustmentProbe(
    const MetalAqAdjustmentProbeForTesting& probe,
    MetalAqAdjustmentResultForTesting* result) {

  if (result == nullptr) {
    return Status::InvalidArgument("AQ adjustment probe output is null");
  }
  const AcStrategyInfo* info = GetAcStrategyInfo(probe.strategy);
  QuantizationMatrixView matrix;
  if (info == nullptr || info->coefficient_count() == 0 ||
      info->coefficient_count() > maximum_coefficient_count_ ||
      3 * info->coefficient_count() > coefficient_value_count_ ||
      !GetDefaultQuantizationMatrix(
          probe.strategy, XybChannel::kY, &matrix).ok() ||
      probe.initial_raw_quant < 1 || probe.initial_raw_quant > kMaxRawQuant ||
      probe.matrix_multipliers[1] != 1.0f) {
    return Status::InvalidArgument(
        "AQ adjustment probe strategy or quantization is invalid");
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (probe.coefficients[channel].size() != info->coefficient_count() ||
        !std::isfinite(probe.matrix_multipliers[channel]) ||
        probe.matrix_multipliers[channel] <= 0.0f) {
      return Status::InvalidArgument(
          "AQ adjustment probe coefficients or multipliers are invalid");
    }
  }
  Quantizer quantizer;
  Status status = Quantizer::Create(probe.quantizer, &quantizer);
  if (!status.ok()) return status;

  std::vector<float> packed_coefficients;
  try {
    packed_coefficients.reserve(3 * info->coefficient_count());
    for (std::span<const float> channel : probe.coefficients) {
      packed_coefficients.insert(
          packed_coefficients.end(), channel.begin(), channel.end());
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
        "Unable to allocate AQ adjustment probe input");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("AQ adjustment probe input is too large");
  }

  status = BeginOperation();
  if (!status.ok()) return status;
  status = UploadContiguous(
      *backend_, std::span<const float>(packed_coefficients),
      forward_coefficients_);
  const uint32_t zero = 0;
  if (status.ok()) {
    status = backend_->CopyHostToDevice(
        *reconstruction_error_.buffer, &zero, sizeof(zero),
        reconstruction_error_.offset_bytes);
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  adjustment_probe_params_ = {
      static_cast<uint32_t>(info->coefficient_count()),
      static_cast<uint32_t>(matrix.coefficient_extent.width),
      static_cast<uint32_t>(matrix.coefficient_extent.height),
      static_cast<uint32_t>(probe.strategy),
      probe.initial_raw_quant,
      probe.quantizer.global_scale,
      probe.matrix_multipliers[0],
      probe.matrix_multipliers[2],
  };

  std::unique_ptr<GpuSubmission> submission;
  status = backend_->SubmitCompute(
      "gjxl AQ adjustment probe",
      &MetalPreparedAqEvaluation::EncodeAdjustmentProbeSubmission, this,
      &submission);
  if (!status.ok() || submission == nullptr) {
    Invalidate();
    return status.ok()
        ? Status::Internal("AQ adjustment probe returned no submission")
        : status;
  }
  {
    std::lock_guard lock(mutex_);
    submission_ = std::move(submission);
  }
  status = WaitForOperation();
  if (!status.ok()) return status;

  uint32_t device_error = 0;
  int32_t adjusted_raw_quant = 0;
  std::array<float, 4> adjusted_y_thresholds{};
  const size_t coefficient_count = info->coefficient_count();
  status = CopyReadback(
      *backend_, reconstruction_error_, &device_error, sizeof(device_error));
  if (status.ok()) {
    status = CopyReadback(
        *backend_, quant_probe_quantized_, &adjusted_raw_quant,
        sizeof(adjusted_raw_quant));
  }
  if (status.ok()) {
    status = CopyReadback(
        *backend_, quant_probe_dequantized_, adjusted_y_thresholds.data(),
        adjusted_y_thresholds.size() * sizeof(float));
  }
  if (status.ok()) {
    status = CopyReadback(
        *backend_, quantized_coefficients_, quantized_readback_.data(),
        coefficient_count * sizeof(int32_t));
  }
  if (!status.ok()) {
    Invalidate();
    return status;
  }
  if (device_error != 0 || adjusted_raw_quant < 1 ||
      adjusted_raw_quant > kMaxRawQuant ||
      !std::ranges::all_of(adjusted_y_thresholds, [](float threshold) {
        return std::isfinite(threshold) && threshold >= 0.0f;
      })) {
    Invalidate();
    return Status::DeviceError(
        "Metal AQ adjustment probe produced invalid output");
  }
  try {
    MetalAqAdjustmentResultForTesting candidate{
        .decision = {
            .raw_quant = adjusted_raw_quant,
            .y_thresholds = adjusted_y_thresholds,
        },
        .quantized_y = std::vector<int32_t>(
            quantized_readback_.begin(),
            quantized_readback_.begin() +
                static_cast<std::ptrdiff_t>(coefficient_count)),
    };
    *result = std::move(candidate);
  } catch (const std::bad_alloc&) {
    CompleteOperation();
    return Status::OutOfMemory(
        "Unable to allocate AQ adjustment probe output");
  } catch (const std::length_error&) {
    CompleteOperation();
    return Status::InvalidArgument("AQ adjustment probe output is too large");
  }
  CompleteOperation();
  return Status::Ok();
}

Status MetalPreparedAqEvaluation::PrepareQuantizationProbeReadback() {
  try {
    quant_probe_quantized_readback_.resize(maximum_coefficient_count_);
    quant_probe_dequantized_readback_.resize(maximum_coefficient_count_);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate AQ quantization-probe readback");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "AQ quantization-probe readback is too large");
  }
}

Status RunMetalAqReconstructionForTesting(
    PreparedAqEvaluation &prepared, AqEvaluationInput input,
    MetalAqReconstructionSnapshotForTesting *snapshot) {

  MetalPreparedAqEvaluation *metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ reconstruction requires a Metal prepared evaluation");
  }
  return metal->RunReconstruction(input, snapshot);
}

Status RunMetalAqQuantizationProbeForTesting(
    PreparedAqEvaluation &prepared,
    const MetalAqQuantizationProbeForTesting &probe,
    std::vector<int32_t> *quantized, std::vector<float> *dequantized) {

  MetalPreparedAqEvaluation *metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ quantization probe requires a Metal prepared evaluation");
  }
  return metal->RunQuantizationProbe(probe, quantized, dequantized);
}

Status RunMetalAqAdjustmentProbeForTesting(
    PreparedAqEvaluation& prepared,
    const MetalAqAdjustmentProbeForTesting& probe,
    MetalAqAdjustmentResultForTesting* result) {

  MetalPreparedAqEvaluation* metal = AsMetalPrepared(prepared);
  if (metal == nullptr) {
    return Status::InvalidArgument(
        "AQ adjustment probe requires a Metal prepared evaluation");
  }
  return metal->RunAdjustmentProbe(probe, result);
}

} // namespace gjxl::metal_internal
