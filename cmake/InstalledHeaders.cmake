# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

# Explicit installed interface, using paths relative to the include directory.
# gjxl/ headers come from include/; all others come from src/. Adding a source
# header does not publish it. See docs/installed-interface.md before extending
# either list, and run codec_install_consumer to check dependency completeness.
set(GJXL_PUBLIC_C_HEADERS
  gjxl/gjxl.h
)

set(GJXL_PUBLIC_CXX_HEADERS
  gjxl/execution_domain.hpp

  core/ac_strategy.h
  core/block_grid.h
  core/execution_domain.h
  core/frame_geometry.h
  core/geometry.h
  core/image.h
  core/image_buffer.h
  core/image_ops.h
  core/quantizer.h
  core/status.h

  codec/ac_strategy.h
  codec/adaptive_quantization.h
  codec/butteraugli.h
  codec/chroma_from_luma.h
  codec/codestream.h
  codec/color_transform.h
  codec/convolution.h
  codec/dc_conversion.h
  codec/dc_quantization.h
  codec/dct.h
  codec/epf.h
  codec/gaborish.h
  codec/loop_filter.h
  codec/maximum_error.h
  codec/quantization.h
  codec/quantization_pipeline.h
  codec/reconstruction.h
  codec/vardct_frame.h

  codestream/ac_group.h
  codestream/batch_workflow.h
  codestream/bit_writer.h
  codestream/block_context_map.h
  codestream/coefficient_order.h
  codestream/dc_group.h
  codestream/encoder.h
  codestream/entropy.h
  codestream/entropy_behavior.h
  codestream/headers.h
  codestream/huffman.h
  codestream/sections.h
  codestream/simple_ac_context.h
  codestream/workflow.h

  gpu/backend.h
  gpu/buffer.h
  gpu/image.h
  gpu/scratch.h
  gpu/submission.h
  gpu/metal/metal_backend.h
  gpu/ops/ac_strategy.h
  gpu/ops/ac_strategy_search.h
  gpu/ops/adaptive_quantization.h
  gpu/ops/aq_evaluation.h
  gpu/ops/butteraugli.h
  gpu/ops/gaborish.h
  gpu/ops/primitives.h
  gpu/ops/quantization_pipeline.h
  gpu/ops/resident_input.h
  gpu/ops/transform.h
)

# Required by public templates, inline functions and object layouts. These
# implementation dependencies are shipped, but are not additional public APIs.
# Keep their definitions identical in source builds and installed consumers.
set(GJXL_INSTALLED_SUPPORT_HEADERS
  # Public serializer containers and compatibility output overloads.
  codestream/storage.h
  # ExecutionDomain's inline CPU budget and memory accounting.
  core/cpu_budget.h
  core/resource_budget.h
  core/resource_context.h
  # Owning image/frame/serializer types and adaptive-quantization outputs.
  core/managed_allocator.h
  core/publication_output.h
  core/publication_vector.h
  # PublicationVector and the installed toolchain configuration probe.
  core/stdlib_storage_compat.h
)

set(GJXL_INSTALLED_HEADERS
  ${GJXL_PUBLIC_C_HEADERS}
  ${GJXL_PUBLIC_CXX_HEADERS}
  ${GJXL_INSTALLED_SUPPORT_HEADERS}
)
