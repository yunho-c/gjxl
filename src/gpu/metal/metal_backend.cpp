#include "gpu/metal/metal_backend.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/ac_strategy.h"
#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/buffer.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/metal/metal_embedded_library_internal.h"
#include "gpu/metal/metal_status.h"
#include "gpu/ops/ac_strategy.h"

#define setComputePipelineState(state)                                    \
  setComputePipelineState(state);                                         \
  ::gjxl::metal_internal::RecordMetalComputePipelineState(state)

namespace gjxl {
namespace {

using metal_internal::ButteraugliPipelines;
using metal_internal::MetalBackend;
using metal_internal::MetalBuffer;
using metal_internal::AcStrategyPipelines;
using metal_internal::AqPipelines;
using metal_internal::PrimitivePipelines;
using metal_internal::TransformDirection;
using metal_internal::TransformPipeline;
using metal_internal::TransformPipelinePair;
using metal_internal::TransformPipelineRegistry;

enum class TransformDispatchMode {
  kOneThreadPerElement,
  kFixedThreadCount,
  kFixedSimdgroupCount,
};

struct DctImplementationSpec {
  AcStrategyType strategy;
  MetalDctImplementation implementation;
  std::string_view display_name;
  std::string_view forward_function_name;
  std::string_view inverse_function_name;
  TransformDispatchMode dispatch_mode;
  size_t fixed_threads_per_threadgroup = 0;
  size_t simdgroups_per_threadgroup = 0;
  size_t transforms_per_threadgroup = 1;
  bool forward_uses_device_basis = false;
  bool inverse_uses_device_basis = false;
};

struct DctSelection {
  AcStrategyType strategy;
  MetalDctImplementation forward;
  MetalDctImplementation inverse;
};

struct MetalLibrarySource {
  std::string_view path;
  std::span<const uint8_t> bytes;
};

constexpr std::array<DctImplementationSpec, 27>
kDctImplementationSpecs{{
  {
    .strategy = AcStrategyType::kDct8,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name =
      "gjxl_dct8_forward_scalar_2d_matmul",
    .inverse_function_name =
      "gjxl_dct8_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct8,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name =
      "gjxl_dct8_forward_simdgroup_2d_matmul",
    .inverse_function_name =
      "gjxl_dct8_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct8,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct8_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct8_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
    .transforms_per_threadgroup = 4,
  },
  {
    .strategy = AcStrategyType::kDct16x16,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name =
      "gjxl_dct16_forward_scalar_2d_matmul",
    .inverse_function_name =
      "gjxl_dct16_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct16x16,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name =
      "gjxl_dct16_forward_simdgroup_2d_matmul",
    .inverse_function_name =
      "gjxl_dct16_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 2,
  },
  {
    .strategy = AcStrategyType::kDct16x16,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct16_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct16_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct32x32,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name =
      "gjxl_dct32_forward_scalar_2d_matmul",
    .inverse_function_name =
      "gjxl_dct32_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct32x32,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name =
      "gjxl_dct32_forward_simdgroup_2d_matmul",
    .inverse_function_name =
      "gjxl_dct32_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 4,
  },
  {
    .strategy = AcStrategyType::kDct32x32,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct32_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct32_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct16x8,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct16x8_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct16x8_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct16x8,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct16x8_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct16x8_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 2,
  },
  {
    .strategy = AcStrategyType::kDct16x8,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct16x8_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct16x8_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct8x16,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct8x16_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct8x16_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct8x16,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct8x16_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct8x16_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
    .inverse_uses_device_basis = true,
  },
  {
    .strategy = AcStrategyType::kDct8x16,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct8x16_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct8x16_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct32x16,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct32x16_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct32x16_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct32x16,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct32x16_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct32x16_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 4,
  },
  {
    .strategy = AcStrategyType::kDct32x16,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct32x16_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct32x16_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct16x32,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct16x32_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct16x32_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kOneThreadPerElement,
  },
  {
    .strategy = AcStrategyType::kDct16x32,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct16x32_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct16x32_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 2,
    .inverse_uses_device_basis = true,
  },
  {
    .strategy = AcStrategyType::kDct16x32,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name =
      "gjxl_dct16x32_forward_factored_radix2",
    .inverse_function_name =
      "gjxl_dct16x32_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct64x32,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct64x32_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct64x32_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedThreadCount,
    .fixed_threads_per_threadgroup = 512,
    .forward_uses_device_basis = true,
    .inverse_uses_device_basis = true,
  },
  {
    .strategy = AcStrategyType::kDct64x32,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct64x32_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct64x32_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 8,
    .forward_uses_device_basis = true,
    .inverse_uses_device_basis = true,
  },
  {
    .strategy = AcStrategyType::kDct64x32,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name = "gjxl_dct64x32_forward_factored_radix2",
    .inverse_function_name = "gjxl_dct64x32_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
  {
    .strategy = AcStrategyType::kDct32x64,
    .implementation = MetalDctImplementation::kScalarMatmul,
    .display_name = "scalar matmul",
    .forward_function_name = "gjxl_dct32x64_forward_scalar_2d_matmul",
    .inverse_function_name = "gjxl_dct32x64_inverse_scalar_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedThreadCount,
    .fixed_threads_per_threadgroup = 512,
    .forward_uses_device_basis = true,
    .inverse_uses_device_basis = true,
  },
  {
    .strategy = AcStrategyType::kDct32x64,
    .implementation = MetalDctImplementation::kSimdgroupMatmul,
    .display_name = "simdgroup matmul",
    .forward_function_name = "gjxl_dct32x64_forward_simdgroup_2d_matmul",
    .inverse_function_name = "gjxl_dct32x64_inverse_simdgroup_2d_matmul",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 4,
    .forward_uses_device_basis = true,
    .inverse_uses_device_basis = true,
  },
  {
    .strategy = AcStrategyType::kDct32x64,
    .implementation = MetalDctImplementation::kFactoredRadix2,
    .display_name = "factored radix-2",
    .forward_function_name = "gjxl_dct32x64_forward_factored_radix2",
    .inverse_function_name = "gjxl_dct32x64_inverse_factored_radix2",
    .dispatch_mode = TransformDispatchMode::kFixedSimdgroupCount,
    .simdgroups_per_threadgroup = 1,
  },
}};

const DctImplementationSpec* FindDctImplementationSpec(
  AcStrategyType strategy,
  MetalDctImplementation implementation) {

  for (const DctImplementationSpec& spec : kDctImplementationSpecs) {
    if (spec.strategy == strategy &&
        spec.implementation == implementation) {
      return &spec;
    }
  }

  return nullptr;
}

constexpr size_t kDct8BasisOffsetFloats = 0;
constexpr size_t kDct16BasisOffsetFloats = 8 * 8;
constexpr size_t kDct32BasisOffsetFloats =
  kDct16BasisOffsetFloats + 16 * 16;
constexpr size_t kDct64BasisOffsetFloats =
  kDct32BasisOffsetFloats + 32 * 32;
constexpr size_t kDctBasisFloatCount =
  kDct64BasisOffsetFloats + 64 * 64;

[[nodiscard]] constexpr size_t DctBasisOffsetBytes(size_t length) {
  switch (length) {
    case 8:
      return kDct8BasisOffsetFloats * sizeof(float);
    case 16:
      return kDct16BasisOffsetFloats * sizeof(float);
    case 32:
      return kDct32BasisOffsetFloats * sizeof(float);
    case 64:
      return kDct64BasisOffsetFloats * sizeof(float);
    default:
      return std::numeric_limits<size_t>::max();
  }
}

static_assert(DctBasisOffsetBytes(8) % 256 == 0);
static_assert(DctBasisOffsetBytes(16) % 256 == 0);
static_assert(DctBasisOffsetBytes(32) % 256 == 0);
static_assert(DctBasisOffsetBytes(64) % 256 == 0);

// Mirrors the formula used to generate the Metal constant bases. Keeping the
// buffer values identical avoids changing the transform's numerical behavior.
void FillOrthonormalDctBasis(
  size_t length,
  float* basis) {

  const double scale =
    std::sqrt(2.0 / static_cast<double>(length));

  for (size_t frequency = 0; frequency < length; ++frequency) {
    const double alpha =
      frequency == 0 ? 1.0 / std::sqrt(2.0) : 1.0;

    for (size_t sample = 0; sample < length; ++sample) {
      const double angle =
        (static_cast<double>(sample) + 0.5) *
        static_cast<double>(frequency) *
        std::numbers::pi_v<double> /
        static_cast<double>(length);

      basis[frequency * length + sample] =
        static_cast<float>(alpha * scale * std::cos(angle));
    }
  }
}

Status CreateDctBasisBuffer(
  MTL::Device* device,
  NS::SharedPtr<MTL::Buffer>* out) {

  if (device == nullptr || out == nullptr) {
    return Status::InvalidArgument(
      "CreateDctBasisBuffer received invalid argument");
  }

  std::array<float, kDctBasisFloatCount> basis{};

  FillOrthonormalDctBasis(
    8,
    basis.data() + kDct8BasisOffsetFloats);
  FillOrthonormalDctBasis(
    16,
    basis.data() + kDct16BasisOffsetFloats);
  FillOrthonormalDctBasis(
    32,
    basis.data() + kDct32BasisOffsetFloats);
  FillOrthonormalDctBasis(
    64,
    basis.data() + kDct64BasisOffsetFloats);

  auto buffer =
    NS::TransferPtr(
      device->newBuffer(
        basis.data(),
        static_cast<NS::UInteger>(basis.size() * sizeof(float)),
        MTL::ResourceStorageModeShared));

  if (!buffer) {
    return Status::OutOfMemory(
      "Metal failed to allocate the DCT basis buffer");
  }

  *out = std::move(buffer);
  return Status::Ok();
}

[[nodiscard]] constexpr size_t StrategyIndex(
  AcStrategyType strategy) noexcept {

  return static_cast<size_t>(strategy);
}

// Pipeline creation
Status CreatePipeline(
  MTL::Device* device,
  MTL::Library* library,
  std::string_view function_name,
  NS::SharedPtr<MTL::ComputePipelineState>* out) {

  if (device == nullptr ||
      library == nullptr ||
      function_name.empty() ||
      out == nullptr) {

    return Status::InvalidArgument(
      "CreatePipeline received invalid argument");
  }

  const std::string function_name_string(function_name);

  NS::String* name =
    NS::String::string(
      function_name_string.c_str(),
      NS::UTF8StringEncoding);

  auto function =
    NS::TransferPtr(
      library->newFunction(name));

  if (!function) {
    return Status::Internal(
      std::string("Metal function not found: ") +
      function_name_string);
  }

  if (function->functionType() != MTL::FunctionTypeKernel) {
    return Status::InvalidArgument(
      std::string("Metal function is not a kernel: ") +
      function_name_string);
  }

  NS::Error* error = nullptr;

  auto pipeline =
    NS::TransferPtr(
      device->newComputePipelineState(
        function.get(),
        &error));

  if (!pipeline) {
    return metal::ErrorToStatus(
      error,
      "newComputePipelineState");
  }

  metal_internal::RegisterMetalComputePipeline(
    pipeline.get(), function_name);

  *out = std::move(pipeline);

  return Status::Ok();
}

Status CreateTransformPipeline(
  MTL::Device* device,
  MTL::Library* library,
  AcStrategyType strategy,
  std::string_view implementation_name,
  TransformDispatchMode dispatch_mode,
  size_t fixed_threads_per_threadgroup,
  size_t simdgroups_per_threadgroup,
  size_t transforms_per_threadgroup,
  bool uses_device_basis,
  std::string_view function_name,
  std::string_view operation,
  TransformPipeline* out) {

  const AcStrategyInfo* strategy_info =
    GetAcStrategyInfo(strategy);

  if (strategy_info == nullptr || transforms_per_threadgroup == 0 ||
      out == nullptr) {
    return Status::InvalidArgument(
      "CreateTransformPipeline received invalid argument");
  }

  const size_t coefficient_count =
    strategy_info->coefficient_count();

  NS::SharedPtr<MTL::ComputePipelineState> state;

  Status status =
    CreatePipeline(
      device,
      library,
      function_name,
      &state);

  if (!status.ok()) {
    return {
      status.code(),
      std::string("Failed to create ") +
      std::string(operation) +
      " " +
      std::string(strategy_info->name) +
      " pipeline for " +
      std::string(implementation_name) +
      ": " +
      std::string(status.message()),
    };
  }

  NS::UInteger threads_per_threadgroup = 0;

  switch (dispatch_mode) {
    case TransformDispatchMode::kOneThreadPerElement:
      threads_per_threadgroup =
        static_cast<NS::UInteger>(coefficient_count);
      break;

    case TransformDispatchMode::kFixedThreadCount:
      threads_per_threadgroup =
        static_cast<NS::UInteger>(fixed_threads_per_threadgroup);
      break;

    case TransformDispatchMode::kFixedSimdgroupCount: {
      const NS::UInteger simd_width = state->threadExecutionWidth();

      if (simd_width == 0 || simdgroups_per_threadgroup == 0 ||
          simdgroups_per_threadgroup >
            std::numeric_limits<NS::UInteger>::max() / simd_width) {
        return Status::Unavailable(
          std::string("Metal reported invalid SIMD-group dispatch data for ") +
          std::string(implementation_name));
      }

      threads_per_threadgroup =
        simd_width *
        static_cast<NS::UInteger>(simdgroups_per_threadgroup);
      break;
    }
  }

  if (threads_per_threadgroup == 0) {
    return Status::Unavailable(
      std::string("Metal reported an invalid threadgroup size for ") +
      std::string(implementation_name));
  }

  if (state->maxTotalThreadsPerThreadgroup() <
      threads_per_threadgroup) {

    return Status::Unavailable(
      std::string("Metal GPU cannot launch the required threadgroup for ") +
      std::string(implementation_name));
  }

  out->state = std::move(state);
  out->threads_per_threadgroup = threads_per_threadgroup;
  out->transforms_per_threadgroup = transforms_per_threadgroup;
  out->uses_device_basis = uses_device_basis;
  out->strategy = strategy;
  out->label =
    std::string("gjxl ") +
    std::string(operation) +
    " " +
    std::string(strategy_info->name) +
    " (" +
    std::string(implementation_name) +
    ")";

  return Status::Ok();
}


}  // namespace

namespace metal_internal {
namespace {

// One idle arena of each class is enough to accelerate sequential encodes
// without multiplying the retained capacity by the number of concurrent
// callers. Larger forced-Metal workloads remain supported, but their arenas
// are released instead of becoming a permanent backend high-water mark.
constexpr size_t kMaximumRetainedAqScratchArenaBytes =
  size_t{1024} * 1024 * 1024;

}  // namespace

MetalBackend::MetalBackend(
  NS::SharedPtr<MTL::Device> device,
  NS::SharedPtr<MTL::CommandQueue> command_queue,
  NS::SharedPtr<MTL::Library> library,
  NS::SharedPtr<MTL::Buffer> dct_basis_buffer,
  TransformPipelineRegistry transform_pipelines,
  AcStrategyPipelines ac_strategy_pipelines,
  PrimitivePipelines primitive_pipelines,
  AqPipelines aq_pipelines,
  ButteraugliPipelines butteraugli_pipelines,
  bool test_fail_submission,
  bool test_fail_completion)
  : device_(std::move(device)),
    command_queue_(std::move(command_queue)),
    library_(std::move(library)),
    dct_basis_buffer_(std::move(dct_basis_buffer)),
    transform_pipelines_(std::move(transform_pipelines)),
    ac_strategy_pipelines_(std::move(ac_strategy_pipelines)),
    primitive_pipelines_(std::move(primitive_pipelines)),
    aq_pipelines_(std::move(aq_pipelines)),
    butteraugli_pipelines_(std::move(butteraugli_pipelines)),
    test_fail_submission_(test_fail_submission),
    test_fail_completion_(test_fail_completion) {

  NS::String* device_name = device_->name();
  if (device_name != nullptr) {
    const char* utf8 = device_name->utf8String();
    if (utf8 != nullptr) {
      name_ = std::string("Metal: ") + utf8;
    }
  }
  if (name_.empty()) {
    name_ = "Metal";
  }
}

BackendKind MetalBackend::kind() const noexcept {
  return BackendKind::kMetal;
}

std::string_view MetalBackend::name() const noexcept {
  return name_;
}

Status MetalBackend::Allocate(
  size_t size_bytes,
  std::unique_ptr<DeviceBuffer>* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Allocate output pointer is null");
  }
  if (size_bytes == 0) {
    return Status::InvalidArgument(
      "Cannot allocate zero-sized Metal buffer");
  }
  if (size_bytes > std::numeric_limits<NS::UInteger>::max()) {
    return Status::InvalidArgument(
      "Requested Metal buffer is too large");
  }

  auto buffer = NS::TransferPtr(
    device_->newBuffer(
      static_cast<NS::UInteger>(size_bytes),
      MTL::ResourceStorageModeShared));
  if (!buffer) {
    return Status::OutOfMemory(
      "Metal failed to allocate MTL::Buffer");
  }

  out->reset(new MetalBuffer(std::move(buffer), id(), size_bytes));
  RecordSuccessfulAllocation();
  return Status::Ok();
}

Status MetalBackend::AcquireAqScratchArena(
  MetalAqScratchArena kind,
  size_t required_capacity_bytes,
  DeviceScratchArena* arena) {

  const size_t index = static_cast<size_t>(kind);
  if (arena == nullptr || required_capacity_bytes == 0 ||
      index >= idle_aq_scratch_.size() || arena->capacity_bytes() != 0) {
    return Status::InvalidArgument(
      "Metal AQ scratch lease request is invalid");
  }

  DeviceScratchArena candidate;
  {
    std::lock_guard lock(aq_scratch_pool_mutex_);
    std::optional<DeviceScratchArena>& idle = idle_aq_scratch_[index];
    if (idle.has_value()) {
      candidate = std::move(*idle);
      idle.reset();
    }
  }
  if (candidate.capacity_bytes() != 0 &&
      candidate.capacity_bytes() != required_capacity_bytes) {
    candidate = DeviceScratchArena{};
  }
  if (candidate.capacity_bytes() != 0) {
    MetalBuffer* buffer = AsMetalBuffer(*candidate.backing_buffer());
    // Lock the purgeable resource before reuse. Empty means Metal discarded
    // its contents under pressure, so retain neither the bytes nor the object.
    if (buffer == nullptr ||
        buffer->handle()->setPurgeableState(
          MTL::PurgeableStateNonVolatile) == MTL::PurgeableStateEmpty) {
      candidate = DeviceScratchArena{};
    }
  }
  Status status = candidate.Prepare(*this, required_capacity_bytes);
  if (!status.ok()) {
    return status;
  }
  *arena = std::move(candidate);
  return Status::Ok();
}

void MetalBackend::ReleaseAqScratchArena(
  MetalAqScratchArena kind,
  DeviceScratchArena arena,
  bool reusable) noexcept {

  const size_t index = static_cast<size_t>(kind);
  if (!reusable || index >= idle_aq_scratch_.size() ||
      arena.capacity_bytes() == 0 ||
      arena.capacity_bytes() > kMaximumRetainedAqScratchArenaBytes) {
    return;
  }
  arena.ResetLayout();
  MetalBuffer* buffer = AsMetalBuffer(*arena.backing_buffer());
  if (buffer == nullptr) {
    return;
  }
  // Keep the allocation as a latency cache without preventing Metal from
  // reclaiming its backing storage under memory pressure.
  (void)buffer->handle()->setPurgeableState(MTL::PurgeableStateVolatile);
  try {
    std::lock_guard lock(aq_scratch_pool_mutex_);
    std::optional<DeviceScratchArena>& idle = idle_aq_scratch_[index];
    if (!idle.has_value() ||
        arena.capacity_bytes() < idle->capacity_bytes()) {
      idle = std::move(arena);
    }
  } catch (...) {
    // Pooling is opportunistic. Destruction must remain noexcept even if the
    // platform mutex reports an exceptional failure.
  }
}

Status MetalBackend::EmptyAqScratchArenasForTesting() {
  std::lock_guard lock(aq_scratch_pool_mutex_);
  for (std::optional<DeviceScratchArena>& arena : idle_aq_scratch_) {
    if (!arena.has_value()) {
      continue;
    }
    MetalBuffer* buffer = AsMetalBuffer(*arena->backing_buffer());
    if (buffer == nullptr) {
      return Status::Internal("Idle AQ scratch is not a Metal buffer");
    }
    (void)buffer->handle()->setPurgeableState(MTL::PurgeableStateEmpty);
  }
  return Status::Ok();
}

Status MetalBackend::CopyHostToDevice(
  DeviceBuffer& dst,
  const void* src,
  size_t size_bytes,
  size_t dst_offset_bytes) {

  if (src == nullptr && size_bytes != 0) {
    return Status::InvalidArgument(
      "Host source pointer is null");
  }
  MetalBuffer* metal_dst = AsMetalBuffer(dst);
  if (metal_dst == nullptr) {
    return Status::InvalidArgument(
      "Destination is not a Metal buffer");
  }
  if (!owns(dst) || metal_dst->device() != device_.get()) {
    return Status::InvalidArgument(
      "Destination belongs to another Metal backend");
  }
  if (dst_offset_bytes > dst.size_bytes() ||
      size_bytes > dst.size_bytes() - dst_offset_bytes) {
    return Status::InvalidArgument(
      "Host to device copy exceeds destination buffer");
  }

  auto* destination =
    static_cast<std::byte*>(metal_dst->contents()) + dst_offset_bytes;
  std::memcpy(destination, src, size_bytes);
  return Status::Ok();
}

Status MetalBackend::CopyDeviceToHost(
  const DeviceBuffer& src,
  void* dst,
  size_t size_bytes,
  size_t src_offset_bytes) {

  if (dst == nullptr && size_bytes != 0) {
    return Status::InvalidArgument(
      "Host destination pointer is null");
  }
  std::span<const std::byte> source;
  Status status = BorrowCompletedReadOnly(
    src, size_bytes, src_offset_bytes, &source);
  if (!status.ok()) {
    return status;
  }
  if (source.empty()) {
    return Status::Ok();
  }
  std::memcpy(dst, source.data(), source.size());
  return Status::Ok();
}

Status MetalBackend::BorrowCompletedReadOnly(
  const DeviceBuffer& src,
  size_t size_bytes,
  size_t src_offset_bytes,
  std::span<const std::byte>* out) const {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Mapped Metal range output is null");
  }
  *out = {};
  const MetalBuffer* metal_src = AsMetalBuffer(src);
  if (metal_src == nullptr) {
    return Status::InvalidArgument(
      "Source is not a Metal buffer");
  }
  if (!owns(src) || metal_src->device() != device_.get()) {
    return Status::InvalidArgument(
      "Source belongs to another Metal backend");
  }
  if (src_offset_bytes > src.size_bytes() ||
      size_bytes > src.size_bytes() - src_offset_bytes) {
    return Status::InvalidArgument(
      "Device-to-host copy exceeds source buffer");
  }

  const auto* source =
    static_cast<const std::byte*>(metal_src->contents()) + src_offset_bytes;
  *out = {source, size_bytes};
  return Status::Ok();
}

Status MetalBackend::ForwardTransform(
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {

  return SubmitTransform(
    TransformDirection::kForward, batch, submission);
}

Status MetalBackend::InverseTransform(
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {

  return SubmitTransform(
    TransformDirection::kInverse, batch, submission);
}

MetalBuffer* MetalBackend::AsMetalBuffer(DeviceBuffer& buffer) {
  if (buffer.backend() != BackendKind::kMetal) {
    return nullptr;
  }
  return dynamic_cast<MetalBuffer*>(&buffer);
}

const MetalBuffer* MetalBackend::AsMetalBuffer(
  const DeviceBuffer& buffer) {

  if (buffer.backend() != BackendKind::kMetal) {
    return nullptr;
  }
  return dynamic_cast<const MetalBuffer*>(&buffer);
}

Status MetalBackend::ValidateTransformBatch(
  const AcStrategyInfo& strategy_info,
  const TransformBatch& batch,
  const MetalBuffer** input,
  MetalBuffer** output) const {

  if (input == nullptr || output == nullptr) {
    return Status::Internal(
      "ValidateTransformBatch output is null");
  }
  if (batch.transform_count == 0) {
    *input = nullptr;
    *output = nullptr;
    return Status::Ok();
  }
  if (batch.input == nullptr || batch.output == nullptr) {
    return Status::InvalidArgument(
      "Transform input/output buffer is null");
  }
  if (batch.input == batch.output) {
    return Status::InvalidArgument(
      "In-place transforms are not supported yet");
  }

  const size_t coefficient_count = strategy_info.coefficient_count();
  if (coefficient_count >
      std::numeric_limits<size_t>::max() / sizeof(float)) {
    return Status::Internal(
      "Transform coefficient count is too large");
  }
  const size_t bytes_per_transform =
    coefficient_count * sizeof(float);
  if (batch.transform_count >
      std::numeric_limits<size_t>::max() / bytes_per_transform) {
    return Status::InvalidArgument(
      "Transform batch is too large");
  }
  const size_t required_bytes =
    batch.transform_count * bytes_per_transform;
  if (batch.input->size_bytes() < required_bytes ||
      batch.output->size_bytes() < required_bytes) {
    return Status::InvalidArgument(
      "Transform buffer is too small");
  }

  *input = AsMetalBuffer(*batch.input);
  *output = AsMetalBuffer(*batch.output);
  if (*input == nullptr || *output == nullptr) {
    return Status::InvalidArgument(
      "Transform buffers are not Metal buffers");
  }
  if (!owns(*batch.input) || !owns(*batch.output) ||
      (*input)->device() != device_.get() ||
      (*output)->device() != device_.get()) {
    return Status::InvalidArgument(
      "Transform buffer belongs to another Metal backend");
  }
  return Status::Ok();
}

void MetalBackend::EncodeTransformSubmission(
  MetalBackend& backend,
  MTL::ComputeCommandEncoder* encoder,
  const void* context) {

  const auto& transform =
    *static_cast<const TransformEncodeContext*>(context);
  backend.EncodeTransformBatch(
    encoder,
    transform.direction,
    transform.pipeline->strategy,
    *transform.input,
    transform.input_offset_bytes,
    *transform.output,
    transform.output_offset_bytes,
    transform.transform_count);
}

void MetalBackend::EncodeTransformBatch(
  MTL::ComputeCommandEncoder* encoder,
  TransformDirection direction,
  AcStrategyType strategy,
  const MetalBuffer& input,
  size_t input_offset_bytes,
  MetalBuffer& output,
  size_t output_offset_bytes,
  size_t transform_count) const {

  const TransformPipelinePair& pair =
    transform_pipelines_[StrategyIndex(strategy)];
  const TransformPipeline& pipeline =
    direction == TransformDirection::kForward
      ? pair.forward
      : pair.inverse;
  encoder->setComputePipelineState(pipeline.state.get());
  encoder->setBuffer(input.handle(), input_offset_bytes, 0);
  encoder->setBuffer(output.handle(), output_offset_bytes, 1);

  if (pipeline.uses_device_basis) {
    const AcStrategyInfo* strategy_info = GetAcStrategyInfo(strategy);
    const Extent2D extent = strategy_info->pixel_extent();
    encoder->setBuffer(
      dct_basis_buffer_.get(),
      static_cast<NS::UInteger>(DctBasisOffsetBytes(extent.height)),
      2);
    if (extent.width != extent.height) {
      encoder->setBuffer(
        dct_basis_buffer_.get(),
        static_cast<NS::UInteger>(DctBasisOffsetBytes(extent.width)),
        3);
    }
  } else if (pipeline.transforms_per_threadgroup > 1) {
    const uint32_t packed_transform_count =
      static_cast<uint32_t>(transform_count);
    encoder->setBytes(
      &packed_transform_count, sizeof(packed_transform_count), 2);
  }

  const size_t threadgroup_count = transform_count == 0
    ? 0
    : 1 + (transform_count - 1) / pipeline.transforms_per_threadgroup;
  DispatchMetalThreadgroups(
    encoder,
    MTL::Size(static_cast<NS::UInteger>(threadgroup_count), 1, 1),
    MTL::Size(pipeline.threads_per_threadgroup, 1, 1));
}

Status MetalBackend::SubmitTransform(
  TransformDirection direction,
  const TransformBatch& batch,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr) {
    return Status::InvalidArgument(
      "Transform submission output pointer is null");
  }
  submission->reset();

  const AcStrategyInfo* strategy_info =
    GetAcStrategyInfo(batch.strategy);
  if (strategy_info == nullptr) {
    return Status::InvalidArgument(
      "Unknown JPEG XL AC strategy");
  }

  const TransformPipelinePair& pair =
    transform_pipelines_[StrategyIndex(batch.strategy)];
  const TransformPipeline& pipeline =
    direction == TransformDirection::kForward
      ? pair.forward
      : pair.inverse;
  if (!pipeline.state) {
    return Status::Unavailable(
      std::string("Metal backend does not support ") +
      std::string(strategy_info->name));
  }
  if (pipeline.strategy != batch.strategy) {
    return Status::Internal(
      "Transform pipeline strategy does not match batch strategy");
  }

  const MetalBuffer* input = nullptr;
  MetalBuffer* output = nullptr;
  Status status = ValidateTransformBatch(
    *strategy_info, batch, &input, &output);
  if (!status.ok()) {
    return status;
  }
  if (batch.transform_count == 0) {
    return Status::Ok();
  }
  const size_t threadgroup_count = batch.transform_count == 0
    ? 0
    : 1 + (batch.transform_count - 1) /
        pipeline.transforms_per_threadgroup;
  if ((pipeline.transforms_per_threadgroup > 1 &&
       batch.transform_count > std::numeric_limits<uint32_t>::max()) ||
      threadgroup_count > std::numeric_limits<NS::UInteger>::max()) {
    return Status::InvalidArgument(
      "Transform batch exceeds Metal grid range");
  }

  const TransformEncodeContext context{
    &pipeline, direction, input, output, 0, 0, batch.transform_count};
  return SubmitCompute(
    pipeline.label.c_str(),
    &MetalBackend::EncodeTransformSubmission,
    &context,
    submission);
}

}  // namespace metal_internal

// Factory

Status CreateMetalBackend(
  std::string_view metallib_path,
  std::unique_ptr<GpuBackend>* out) {

  return CreateMetalBackend(
    metallib_path,
    MetalBackendOptions{},
    out);
}

Status CreateMetalBackendImpl(
  MetalLibrarySource metallib,
  const MetalBackendOptions& options,
  std::unique_ptr<GpuBackend>* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "CreateMetalBackend output pointer is null");
  }
  out->reset();

  if (metallib.path.empty() == metallib.bytes.empty()) {
    return Status::InvalidArgument(
      "Exactly one Metal library source is required");
  }

  const std::array<DctSelection, 9> dct_selections{{
    {
      .strategy = AcStrategyType::kDct8,
      .forward = options.forward_dct8,
      .inverse = options.inverse_dct8,
    },
    {
      .strategy = AcStrategyType::kDct16x16,
      .forward = options.forward_dct16x16,
      .inverse = options.inverse_dct16x16,
    },
    {
      .strategy = AcStrategyType::kDct32x32,
      .forward = options.forward_dct32x32,
      .inverse = options.inverse_dct32x32,
    },
    {
      .strategy = AcStrategyType::kDct16x8,
      .forward = options.forward_dct16x8,
      .inverse = options.inverse_dct16x8,
    },
    {
      .strategy = AcStrategyType::kDct8x16,
      .forward = options.forward_dct8x16,
      .inverse = options.inverse_dct8x16,
    },
    {
      .strategy = AcStrategyType::kDct32x16,
      .forward = options.forward_dct32x16,
      .inverse = options.inverse_dct32x16,
    },
    {
      .strategy = AcStrategyType::kDct16x32,
      .forward = options.forward_dct16x32,
      .inverse = options.inverse_dct16x32,
    },
    {
      .strategy = AcStrategyType::kDct64x32,
      .forward = options.forward_dct64x32,
      .inverse = options.inverse_dct64x32,
    },
    {
      .strategy = AcStrategyType::kDct32x64,
      .forward = options.forward_dct32x64,
      .inverse = options.inverse_dct32x64,
    },
  }};

  for (const DctSelection& selection : dct_selections) {
    const AcStrategyInfo* strategy_info =
      GetAcStrategyInfo(selection.strategy);

    if (strategy_info == nullptr) {
      return Status::Internal(
        "Metal DCT selection has an invalid strategy");
    }

    if (FindDctImplementationSpec(
          selection.strategy,
          selection.forward) == nullptr) {
      return Status::InvalidArgument(
        std::string("Unknown forward Metal ") +
        std::string(strategy_info->name) +
        " implementation");
    }

    if (FindDctImplementationSpec(
          selection.strategy,
          selection.inverse) == nullptr) {
      return Status::InvalidArgument(
        std::string("Unknown inverse Metal ") +
        std::string(strategy_info->name) +
        " implementation");
    }
  }

  auto pool =
    NS::TransferPtr(
      NS::AutoreleasePool::alloc()->init());

  auto device =
    NS::TransferPtr(
      MTL::CreateSystemDefaultDevice());

  if (!device) {
    return Status::Unavailable(
      "No Metal device is available");
  }

  auto command_queue =
    NS::TransferPtr(
      device->newCommandQueue());

  if (!command_queue) {
    return Status::Internal(
      "Failed to create Metal command queue");
  }

  NS::Error* error = nullptr;
  MTL::Library* raw_library = nullptr;
  if (!metallib.path.empty()) {
    const std::string path(metallib.path);
    NS::String* ns_path = NS::String::string(
      path.c_str(), NS::UTF8StringEncoding);
    raw_library = device->newLibrary(ns_path, &error);
  } else {
    dispatch_data_t library_data = dispatch_data_create(
      metallib.bytes.data(), metallib.bytes.size(), nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (library_data == nullptr) {
      return Status::OutOfMemory(
        "Unable to wrap the embedded Metal library");
    }
    raw_library = device->newLibrary(library_data, &error);
    dispatch_release(library_data);
  }
  auto library = NS::TransferPtr(raw_library);

  if (!library) {
    return metal::ErrorToStatus(
      error,
      metallib.path.empty()
        ? "Loading embedded gjxl.metallib"
        : "Loading gjxl.metallib");
  }

  TransformPipelineRegistry transform_pipelines;
  NS::SharedPtr<MTL::Buffer> dct_basis_buffer;
  Status status = Status::Ok();

  for (const DctSelection& selection : dct_selections) {
    const DctImplementationSpec* forward_spec =
      FindDctImplementationSpec(
        selection.strategy,
        selection.forward);
    const DctImplementationSpec* inverse_spec =
      FindDctImplementationSpec(
        selection.strategy,
        selection.inverse);

    if (forward_spec == nullptr || inverse_spec == nullptr) {
      return Status::Internal(
        "Validated Metal DCT implementation disappeared");
    }

    if ((forward_spec->forward_uses_device_basis ||
         inverse_spec->inverse_uses_device_basis) &&
        !dct_basis_buffer) {

      status =
        CreateDctBasisBuffer(
          device.get(),
          &dct_basis_buffer);

      if (!status.ok()) {
        return status;
      }
    }

    TransformPipelinePair& pipelines =
      transform_pipelines[StrategyIndex(selection.strategy)];

    status =
      CreateTransformPipeline(
        device.get(),
        library.get(),
        selection.strategy,
        forward_spec->display_name,
        forward_spec->dispatch_mode,
        forward_spec->fixed_threads_per_threadgroup,
        forward_spec->simdgroups_per_threadgroup,
        forward_spec->transforms_per_threadgroup,
        forward_spec->forward_uses_device_basis,
        forward_spec->forward_function_name,
        "forward",
        &pipelines.forward);

    if (!status.ok()) {
      return status;
    }

    status =
      CreateTransformPipeline(
        device.get(),
        library.get(),
        selection.strategy,
        inverse_spec->display_name,
        inverse_spec->dispatch_mode,
        inverse_spec->fixed_threads_per_threadgroup,
        inverse_spec->simdgroups_per_threadgroup,
        inverse_spec->transforms_per_threadgroup,
        inverse_spec->inverse_uses_device_basis,
        inverse_spec->inverse_function_name,
        "inverse",
        &pipelines.inverse);

    if (!status.ok()) {
      return status;
    }
  }

  AcStrategyPipelines ac_strategy_pipelines;
  std::array<bool, kAcStrategyCount> fused_ac_forward_enabled{};
  std::array<bool, kAcStrategyCount> fused_ac_inverse_enabled{};
  for (const DctSelection& selection : dct_selections) {
    fused_ac_forward_enabled[static_cast<size_t>(selection.strategy)] =
      selection.forward == MetalDctImplementation::kSimdgroupMatmul;
    fused_ac_inverse_enabled[static_cast<size_t>(selection.strategy)] =
      selection.inverse == MetalDctImplementation::kSimdgroupMatmul;
  }
  status = CreateAcStrategyPipelines(
    device.get(),
    library.get(),
    fused_ac_forward_enabled,
    fused_ac_inverse_enabled,
    &ac_strategy_pipelines);
  if (!status.ok()) {
    return {
      status.code(),
      std::string("Failed to create AC-strategy pipelines: ") +
        std::string(status.message()),
    };
  }

  PrimitivePipelines primitive_pipelines;
  status = CreatePrimitivePipelines(
    device.get(), library.get(), &primitive_pipelines);
  if (!status.ok()) {
    return status;
  }

  AqPipelines aq_pipelines;
  status = CreateAqPipelines(device.get(), library.get(), &aq_pipelines);
  if (!status.ok()) {
    return status;
  }

  ButteraugliPipelines butteraugli_pipelines;
  status = CreateButteraugliPipelines(
    device.get(), library.get(), &butteraugli_pipelines);
  if (!status.ok()) {
    return status;
  }

  out->reset(
    new MetalBackend(
      std::move(device),
      std::move(command_queue),
      std::move(library),
      std::move(dct_basis_buffer),
      std::move(transform_pipelines),
      std::move(ac_strategy_pipelines),
      std::move(primitive_pipelines),
      std::move(aq_pipelines),
      std::move(butteraugli_pipelines),
      options.test_fail_submission,
      options.test_fail_completion));

  return Status::Ok();
}

Status CreateMetalBackend(
  std::string_view metallib_path,
  const MetalBackendOptions& options,
  std::unique_ptr<GpuBackend>* out) {

  return CreateMetalBackendImpl(
    {.path = metallib_path}, options, out);
}

Status CreateMetalBackend(
  std::span<const uint8_t> metallib,
  std::unique_ptr<GpuBackend>* out) {

  return CreateMetalBackend(metallib, MetalBackendOptions{}, out);
}

Status CreateMetalBackend(
  std::span<const uint8_t> metallib,
  const MetalBackendOptions& options,
  std::unique_ptr<GpuBackend>* out) {

  return CreateMetalBackendImpl(
    {.bytes = metallib}, options, out);
}

Status CreateEmbeddedMetalBackend(
  const MetalBackendOptions& options,
  std::unique_ptr<GpuBackend>* out) {

  return CreateMetalBackend(
    metal_internal::EmbeddedMetalLibrary(), options, out);
}

}  // namespace gjxl

#undef setComputePipelineState
