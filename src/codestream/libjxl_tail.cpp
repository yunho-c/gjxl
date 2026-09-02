// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/libjxl_tail_internal.h"

#include <jxl/memory_manager.h>
#include <jxl/thread_parallel_runner.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/codestream.h"
#include "codec/vardct_frame.h"
#include "core/ac_strategy.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_precomputed_vardct.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/padded_bytes.h"

namespace gjxl::codestream_internal {
namespace {

using TailClock = std::chrono::steady_clock;

uint64_t ElapsedNanoseconds(TailClock::time_point begin) {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      TailClock::now() - begin).count());
}

struct BridgeFrameStorage {
  std::vector<uint8_t> strategies;
  std::vector<jxl::PrecomputedVarDctAcGroup> ac_groups;
  jxl::PrecomputedVarDctFrame frame;
};

class ThreadRunnerHandle {
public:
  ThreadRunnerHandle() = default;
  ThreadRunnerHandle(const ThreadRunnerHandle&) = delete;
  ThreadRunnerHandle& operator=(const ThreadRunnerHandle&) = delete;

  ~ThreadRunnerHandle() {
    if (opaque_ != nullptr) {
      JxlThreadParallelRunnerDestroy(opaque_);
    }
  }

  Status Create(const JxlMemoryManager& memory_manager, size_t thread_count) {
    if (thread_count == 1) {
      return Status::Ok();
    }
    opaque_ = JxlThreadParallelRunnerCreate(&memory_manager, thread_count);
    if (opaque_ == nullptr) {
      return Status::OutOfMemory("Could not create the libjxl thread runner");
    }
    return Status::Ok();
  }

  [[nodiscard]] JxlParallelRunner runner() const noexcept {
    return opaque_ == nullptr ? nullptr : JxlThreadParallelRunner;
  }

  [[nodiscard]] void* opaque() const noexcept { return opaque_; }

private:
  void* opaque_ = nullptr;
};

template <typename T>
jxl::PrecomputedPlaneView<T> BridgePlane(PlaneView<const T> plane) {
  return {
    .data = plane.data,
    .xsize = plane.extent.width,
    .ysize = plane.extent.height,
    .stride = plane.stride,
  };
}

LibjxlTailStateDigest ConvertDigest(
  const jxl::PrecomputedVarDctStateDigest& digest) {
  return {
    .dimensions = digest.dimensions,
    .strategies = digest.strategies,
    .quantizer = digest.quantizer,
    .raw_quant = digest.raw_quant,
    .epf = digest.epf,
    .cfl = digest.cfl,
    .quantized_dc = digest.quantized_dc,
    .dc = digest.dc,
    .ac_used_counts = digest.ac_used_counts,
    .ac_coefficients = digest.ac_coefficients,
  };
}

Status BuildBridgeFrame(
  const VarDctEncoderFrame& source,
  BridgeFrameStorage* storage) {

  const Extent2D pixels = source.geometry().frame();
  const Extent2D blocks = source.geometry().block_grid().blocks;
  size_t block_count = 0;
  if (!blocks.try_area(&block_count)) {
    return Status::InvalidArgument("Libjxl tail block grid is too large");
  }

  try {
    storage->strategies.resize(block_count);
    for (size_t y = 0; y < blocks.height; ++y) {
      for (size_t x = 0; x < blocks.width; ++x) {
        AcStrategyCell cell;
        const Status status = source.strategies().Get(x, y, &cell);
        if (!status.ok()) {
          return status;
        }
        storage->strategies[y * blocks.width + x] =
          (static_cast<uint8_t>(cell.strategy) << 1) |
          static_cast<uint8_t>(cell.is_anchor);
      }
    }

    storage->ac_groups.resize(source.ac_group_count());
    for (size_t group_index = 0;
         group_index < source.ac_group_count();
         ++group_index) {
      VarDctAcGroupView group;
      const Status status = source.GetAcGroup(group_index, &group);
      if (!status.ok()) {
        return status;
      }
      storage->ac_groups[group_index] = {
        .coefficients = {
          group.coefficients[0].data(),
          group.coefficients[1].data(),
          group.coefficients[2].data(),
        },
        .row_size = group.coefficients[0].size(),
        .used_coefficient_count = group.used_coefficient_count,
      };
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Could not allocate the libjxl state adapter");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("Libjxl state adapter is too large");
  }

  const SimpleVarDctCodestreamProfile& profile = source.profile();
  const QuantizerParams quantizer = source.quantizer().params();
  const ConstImage3I32View quantized_dc = source.quantized_dc();
  const ConstImage3FView dc = source.dc();
  storage->frame = {
    .xsize = pixels.width,
    .ysize = pixels.height,
    .xsize_blocks = blocks.width,
    .ysize_blocks = blocks.height,
    .color_channel_count = profile.color_channel_count,
    .extra_channel_count = profile.extra_channel_count,
    .pass_count = profile.pass_count,
    .xyb_encoded = true,
    .global_scale = quantizer.global_scale,
    .quant_dc = quantizer.quant_dc,
    .x_qm_scale = profile.x_qm_scale,
    .b_qm_scale = profile.b_qm_scale,
    .ac_strategy = {
      .data = storage->strategies.data(),
      .xsize = blocks.width,
      .ysize = blocks.height,
      .stride = blocks.width,
    },
    .raw_quant_field = BridgePlane(source.raw_quant_field()),
    .epf_sharpness = BridgePlane(source.epf_sharpness()),
    .ytox_map = BridgePlane(source.color_correlation().y_to_x_map()),
    .ytob_map = BridgePlane(source.color_correlation().y_to_b_map()),
    .quantized_dc = {
      BridgePlane(quantized_dc.plane[0]),
      BridgePlane(quantized_dc.plane[1]),
      BridgePlane(quantized_dc.plane[2]),
    },
    .dc = {
      BridgePlane(dc.plane[0]),
      BridgePlane(dc.plane[1]),
      BridgePlane(dc.plane[2]),
    },
    .ac_groups = storage->ac_groups.data(),
    .ac_group_count = storage->ac_groups.size(),
  };
  return Status::Ok();
}

Status ValidateRequest(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options) {

  if (options.effort < 1 || options.effort > 10 ||
      !std::isfinite(options.butteraugli_distance) ||
      options.butteraugli_distance <= 0.0f ||
      options.thread_count == 0) {
    return Status::InvalidArgument("Libjxl tail options are invalid");
  }
  return ValidateSimpleCodestreamFrame(frame);
}

Status MapBridgeFailure(jxl::Status status) {
  if (status.code() == jxl::StatusCode::kUnsupported) {
    return Status::Unsupported(
      "Pinned libjxl precomputed VarDCT tail is not implemented");
  }
  return Status::Internal("Pinned libjxl VarDCT tail bridge failed");
}

Status EncodeVarDctCodestreamWithLibjxlImpl(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options,
  std::vector<uint8_t>* output,
  LibjxlTailProfile* profile) {

  if (output == nullptr) {
    return Status::InvalidArgument("Libjxl tail output is null");
  }
  LibjxlTailProfile candidate_profile;
  candidate_profile.worker_count = options.thread_count;
  candidate_profile.calling_thread_participates = options.thread_count == 1;
  const TailClock::time_point total_begin = TailClock::now();
  std::vector<uint8_t> candidate_output;
  {
    const TailClock::time_point adapter_begin = TailClock::now();
    const Status validation = ValidateRequest(frame, options);
    if (!validation.ok()) {
      return validation;
    }
    BridgeFrameStorage storage;
    const Status adapter = BuildBridgeFrame(frame, &storage);
    if (!adapter.ok()) {
      return adapter;
    }
    candidate_profile.adapter_validation_and_copy_nanoseconds =
      ElapsedNanoseconds(adapter_begin);

    const TailClock::time_point context_begin = TailClock::now();
    JxlMemoryManager memory_manager;
    if (!jxl::MemoryManagerInit(&memory_manager, nullptr)) {
      return Status::OutOfMemory(
        "Could not initialize the libjxl tail memory manager");
    }
    ThreadRunnerHandle runner;
    const Status runner_status =
      runner.Create(memory_manager, options.thread_count);
    if (!runner_status.ok()) {
      return runner_status;
    }
    candidate_profile.context_setup_nanoseconds =
      ElapsedNanoseconds(context_begin);

    const jxl::PrecomputedVarDctEncodeOptions bridge_options{
      .effort = options.effort,
      .butteraugli_distance = options.butteraugli_distance,
      .num_threads = options.thread_count,
      .runner = runner.runner(),
      .runner_opaque = runner.opaque(),
    };
    jxl::PaddedBytes bridge_output(&memory_manager);
    jxl::PrecomputedVarDctEncodeProfile bridge_profile;
    const jxl::Status bridge_status = jxl::EncodePrecomputedVarDctFrame(
      &memory_manager, storage.frame, bridge_options, &bridge_output,
      &bridge_profile);
    if (!bridge_status) {
      return MapBridgeFailure(bridge_status);
    }
    if (bridge_output.empty()) {
      return Status::Internal(
        "Pinned libjxl tail produced an empty codestream");
    }
    candidate_profile.libjxl_validation_nanoseconds =
      bridge_profile.validation_nanoseconds;
    candidate_profile.state_initialization_nanoseconds =
      bridge_profile.state_initialization_nanoseconds;
    candidate_profile.dc_metadata_nanoseconds =
      bridge_profile.dc_metadata_nanoseconds;
    candidate_profile.block_context_nanoseconds =
      bridge_profile.block_context_nanoseconds;
    candidate_profile.coefficient_order_nanoseconds =
      bridge_profile.coefficient_order_nanoseconds;
    candidate_profile.coefficient_tokenization_nanoseconds =
      bridge_profile.coefficient_tokenization_nanoseconds;
    candidate_profile.modular_model_nanoseconds =
      bridge_profile.modular_model_nanoseconds;
    candidate_profile.histogram_model_nanoseconds =
      bridge_profile.histogram_model_nanoseconds;
    candidate_profile.section_writing_nanoseconds =
      bridge_profile.section_writing_nanoseconds;
    candidate_profile.assembly_nanoseconds =
      bridge_profile.assembly_nanoseconds;
    candidate_profile.libjxl_internal_nanoseconds =
      bridge_profile.total_nanoseconds;

    const TailClock::time_point copy_begin = TailClock::now();
    try {
      candidate_output.assign(bridge_output.begin(), bridge_output.end());
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("Could not copy the libjxl tail output");
    } catch (const std::length_error&) {
      return Status::OutOfMemory("Libjxl tail output is too large");
    }
    candidate_profile.output_copy_nanoseconds =
      ElapsedNanoseconds(copy_begin);
  }
  candidate_profile.total_nanoseconds = ElapsedNanoseconds(total_begin);
  *output = std::move(candidate_output);
  if (profile != nullptr) {
    *profile = candidate_profile;
  }
  return Status::Ok();
}

}  // namespace

bool LibjxlTailExperimentAvailable() noexcept {
  return true;
}

Status AuditVarDctStateWithLibjxl(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options,
  LibjxlTailStateAudit* audit) {

  if (audit == nullptr) {
    return Status::InvalidArgument("Libjxl state-audit output is null");
  }
  const Status validation = ValidateRequest(frame, options);
  if (!validation.ok()) {
    return validation;
  }

  BridgeFrameStorage storage;
  const Status adapter = BuildBridgeFrame(frame, &storage);
  if (!adapter.ok()) {
    return adapter;
  }
  JxlMemoryManager memory_manager;
  if (!jxl::MemoryManagerInit(&memory_manager, nullptr)) {
    return Status::OutOfMemory(
      "Could not initialize the libjxl tail memory manager");
  }
  const jxl::PrecomputedVarDctEncodeOptions bridge_options{
    .effort = options.effort,
    .butteraugli_distance = options.butteraugli_distance,
    .num_threads = options.thread_count,
  };
  jxl::PrecomputedVarDctStateAudit bridge_audit;
  const jxl::Status bridge_status = jxl::AuditPrecomputedVarDctFrame(
    &memory_manager, storage.frame, bridge_options, &bridge_audit);
  if (!bridge_status) {
    return MapBridgeFailure(bridge_status);
  }
  LibjxlTailStateAudit candidate{
    .source = ConvertDigest(bridge_audit.source),
    .copied = ConvertDigest(bridge_audit.copied),
  };
  if (candidate.source != candidate.copied) {
    return Status::Internal("Pinned libjxl state audit differs after copy");
  }
  *audit = candidate;
  return Status::Ok();
}

Status EncodeVarDctCodestreamWithLibjxl(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options,
  std::vector<uint8_t>* output) {

  return EncodeVarDctCodestreamWithLibjxlImpl(
    frame, options, output, nullptr);
}

Status EncodeVarDctCodestreamWithLibjxlProfiled(
  const VarDctEncoderFrame& frame,
  LibjxlTailOptions options,
  std::vector<uint8_t>* output,
  LibjxlTailProfile* profile) {

  if (profile == nullptr) {
    return Status::InvalidArgument("Libjxl tail profile output is null");
  }
  return EncodeVarDctCodestreamWithLibjxlImpl(
    frame, options, output, profile);
}

}  // namespace gjxl::codestream_internal
