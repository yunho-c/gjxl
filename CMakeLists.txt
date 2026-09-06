# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

cmake_minimum_required(VERSION 3.25)

project(
  gjxl
  VERSION 0.0.1
  LANGUAGES C CXX
)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(
  GJXL_BUILD_TESTS
  "Build tests"
  ON
)

option(
  GJXL_BUILD_BENCHMARKS
  "Build benchmarks"
  OFF
)

option(
  GJXL_ENABLE_LIBJXL_REFERENCE
  "Build the pinned libjxl Butteraugli reference tests and benchmarks"
  OFF
)

option(
  GJXL_ENABLE_METAL_PROFILING
  "Emit line tables, recorded sources, and a flat symbol companion for Metal shaders"
  OFF
)

option(
  GJXL_ENABLE_METAL
  "Build the Metal GPU backend"
  ${APPLE}
)

option(
  GJXL_ENABLE_CUDA
  "Build the CUDA GPU backend"
  OFF
)

if(GJXL_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "GJXL_ENABLE_METAL requires an Apple platform")
  endif()
  enable_language(OBJCXX)
elseif(GJXL_ENABLE_METAL_PROFILING)
  message(FATAL_ERROR
    "GJXL_ENABLE_METAL_PROFILING requires GJXL_ENABLE_METAL")
endif()

if(GJXL_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(GJXL_ENABLE_LIBJXL_REFERENCE AND
   (GJXL_BUILD_TESTS OR GJXL_BUILD_BENCHMARKS))
  if(NOT EXISTS
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/third_party/highway/CMakeLists.txt")
    message(FATAL_ERROR
      "Pinned libjxl dependencies are missing. Run: git submodule update --init --recursive")
  endif()

  # Butteraugli is the perceptual oracle used by iterative AQ. Build only its
  # required libjxl translation units plus Highway, rather than the full
  # encoder/decoder libraries. Highway resets CMP0077 to OLD, so cache entries
  # are required to keep its bundled tests and examples disabled on CMake 4.
  set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
  add_subdirectory(
    third_party/libjxl/third_party/highway
    third_party/highway
    EXCLUDE_FROM_ALL
  )

  set(
    GJXL_BUTTERAUGLI_REFERENCE_SOURCES
    third_party/libjxl/lib/jxl/butteraugli/butteraugli.cc
    third_party/libjxl/lib/jxl/convolve_slow.cc
    third_party/libjxl/lib/jxl/enc_convolve_separable5.cc
    third_party/libjxl/lib/jxl/image.cc
    third_party/libjxl/lib/jxl/memory_manager_internal.cc
    third_party/libjxl/lib/jxl/simd_util.cc
  )

  function(gjxl_add_butteraugli_reference target_name)
    add_library(
      "${target_name}"
      STATIC
      ${GJXL_BUTTERAUGLI_REFERENCE_SOURCES}
    )

    target_include_directories(
      "${target_name}"
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/lib/include"
    )

    target_compile_features(
      "${target_name}"
      PRIVATE
        cxx_std_20
    )

    target_link_libraries(
      "${target_name}"
      PRIVATE
        hwy
    )
  endfunction()

  gjxl_add_butteraugli_reference(gjxl_butteraugli_reference)

  if(GJXL_BUILD_TESTS)
    gjxl_add_butteraugli_reference(gjxl_butteraugli_reference_scalar)
    target_compile_definitions(
      gjxl_butteraugli_reference_scalar
      PRIVATE
        HWY_COMPILE_ONLY_SCALAR=1
    )
  endif()

  if(GJXL_BUILD_TESTS OR GJXL_BUILD_BENCHMARKS)
    add_library(
      gjxl_butteraugli_oracle
      STATIC
      tests/butteraugli_oracle.cpp
    )
    target_include_directories(
      gjxl_butteraugli_oracle
      PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/lib/include"
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl"
    )
    target_compile_features(
      gjxl_butteraugli_oracle
      PUBLIC
        cxx_std_20
    )
    target_link_libraries(
      gjxl_butteraugli_oracle
      PRIVATE
        gjxl_butteraugli_reference
        hwy
    )
  endif()

  if(GJXL_BUILD_TESTS)
    add_library(
      gjxl_butteraugli_oracle_scalar
      STATIC
      tests/butteraugli_oracle.cpp
    )
    target_include_directories(
      gjxl_butteraugli_oracle_scalar
      PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/lib/include"
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl"
    )
    target_compile_features(
      gjxl_butteraugli_oracle_scalar
      PUBLIC
        cxx_std_20
    )
    target_compile_definitions(
      gjxl_butteraugli_oracle_scalar
      PRIVATE
        HWY_COMPILE_ONLY_SCALAR=1
    )
    target_link_libraries(
      gjxl_butteraugli_oracle_scalar
      PRIVATE
        gjxl_butteraugli_reference_scalar
        hwy
    )
  endif()
endif()

if(GJXL_BUILD_TESTS OR GJXL_BUILD_BENCHMARKS)
  add_library(
    gjxl_butteraugli_fixtures
    STATIC
    tests/butteraugli_fixtures.cpp
  )
  target_include_directories(
    gjxl_butteraugli_fixtures
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  )
  target_compile_features(
    gjxl_butteraugli_fixtures
    PUBLIC
      cxx_std_20
  )
endif()

# C++ common interface
add_library(
  gjxl_core
  INTERFACE
)

add_library(gjxl::core ALIAS gjxl_core)

set_target_properties(
  gjxl_core
  PROPERTIES
    EXPORT_NAME core
)

target_compile_features(
  gjxl_core
  INTERFACE
    cxx_std_20
)

# Some external build drivers replace CMAKE_CXX_FLAGS instead of appending to
# them. Keep exception unwinding explicit for MSVC because GJXL's public error
# boundary relies on C++ exceptions being caught before they cross the C API.
if(MSVC)
  target_compile_options(
    gjxl_core
    INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
  )
endif()

target_include_directories(
  gjxl_core
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Codec algorithms
add_library(
  gjxl_codec
  STATIC

  src/codec/ac_strategy.cpp
  src/codec/adaptive_quantization.cpp
  src/codec/butteraugli.cpp
  src/codec/butteraugli_distance_internal.cpp
  src/codec/butteraugli_internal.cpp
  src/codec/chroma_from_luma.cpp
  src/codec/color_transform.cpp
  src/codec/convolution.cpp
  src/codec/dc_conversion.cpp
  src/codec/dc_quantization.cpp
  src/codec/dct.cpp
  src/codec/epf.cpp
  src/codec/gaborish.cpp
  src/codec/loop_filter.cpp
  src/codec/maximum_error.cpp
  src/codec/quantization.cpp
  src/codec/quantization_pipeline.cpp
  src/codec/reconstruction.cpp
  src/codec/codestream.cpp
  src/codec/vardct_frame.cpp
)

add_library(gjxl::codec ALIAS gjxl_codec)

set_target_properties(
  gjxl_codec
  PROPERTIES
    EXPORT_NAME codec
)

target_link_libraries(
  gjxl_codec
  PUBLIC
    gjxl_core
)

# Backend-independent codestream serialization primitives
add_library(
  gjxl_codestream
  STATIC

  src/codestream/ac_group.cpp
  src/codestream/ans.cpp
  src/codestream/batch_workflow.cpp
  src/codestream/bit_writer.cpp
  src/codestream/block_context_map.cpp
  src/codestream/coefficient_order.cpp
  src/codestream/dc_group.cpp
  src/codestream/entropy.cpp
  src/codestream/encoder.cpp
  src/codestream/headers.cpp
  src/codestream/huffman.cpp
  src/codestream/rate_control.cpp
  src/codestream/sections.cpp
  src/codestream/workflow.cpp
)

add_library(gjxl::codestream ALIAS gjxl_codestream)

set_target_properties(
  gjxl_codestream
  PROPERTIES
    EXPORT_NAME codestream
)

target_link_libraries(
  gjxl_codestream
  PUBLIC
    gjxl_codec
)

# Private adapters shared by C API and tests.
add_library(
  gjxl_c_api_internal
  OBJECT
  src/c_api/image_conversion.cpp
)

set_target_properties(
  gjxl_c_api_internal
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

target_link_libraries(
  gjxl_c_api_internal
  PUBLIC
    gjxl_core
)

# High-level C interface
add_library(
  gjxl_c
  STATIC
  src/c_api/gjxl.cpp
  $<TARGET_OBJECTS:gjxl_c_api_internal>
)

add_library(gjxl::c ALIAS gjxl_c)

set_target_properties(
  gjxl_c
  PROPERTIES
    EXPORT_NAME c
    OUTPUT_NAME gjxl
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

target_include_directories(
  gjxl_c
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

if(MSVC)
  target_compile_options(gjxl_c PRIVATE /W4 /WX)
else()
  target_compile_options(
    gjxl_c
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Werror
  )
endif()

target_link_libraries(
  gjxl_c
  PRIVATE
    gjxl_codestream
)

# Floating-point interchange support used by the command-line frontend.
add_library(
  gjxl_pfm_io
  STATIC
  src/io/pfm.cpp
)

target_link_libraries(
  gjxl_pfm_io
  PUBLIC
    gjxl_core
)

add_executable(
  gjxl_encode
  tools/gjxl_encode.cpp
)

target_link_libraries(
  gjxl_encode
  PRIVATE
    gjxl_codestream
    gjxl_pfm_io
)

add_executable(
  gjxl_rate_control_probe
  tools/rate_control_probe.cpp
)

target_link_libraries(
  gjxl_rate_control_probe
  PRIVATE
    gjxl_codestream
    gjxl_pfm_io
)

install(
  TARGETS
    gjxl_core
    gjxl_codec
    gjxl_codestream
    gjxl_c
  EXPORT gjxlTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(
  FILES include/gjxl/gjxl.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gjxl"
)

install(
  DIRECTORY src/core
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.h"
)

install(
  DIRECTORY src/codec
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*_internal.h" EXCLUDE
    PATTERN "quantization_tables_generated.h" EXCLUDE
)

install(
  DIRECTORY src/codestream
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*_internal.h" EXCLUDE
)

install(
  TARGETS gjxl_encode
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

configure_package_config_file(
  cmake/gjxlConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/gjxlConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gjxl"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gjxlConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(
  EXPORT gjxlTargets
  FILE gjxlTargets.cmake
  NAMESPACE gjxl::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gjxl"
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gjxlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/gjxlConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gjxl"
)

# Backend-neutral GPU infrastructure
add_library(
  gjxl_gpu
  STATIC

  src/gpu/image.cpp
  src/gpu/scratch.cpp
)

target_link_libraries(
  gjxl_gpu
  PUBLIC
    gjxl_core
)

# Codec-specific GPU operation contracts
add_library(
  gjxl_gpu_butteraugli
  STATIC

  src/gpu/ops/butteraugli.cpp
)

target_link_libraries(
  gjxl_gpu_butteraugli
  PUBLIC
    gjxl_gpu
)

# Backend-independent GPU codec operations
add_library(
  gjxl_gpu_ops
  STATIC

  src/gpu/ops/ac_strategy_search.cpp
  src/gpu/ops/adaptive_quantization.cpp
  src/gpu/ops/gaborish.cpp
  src/gpu/ops/quantization_pipeline.cpp
)

target_link_libraries(
  gjxl_gpu_ops
  PUBLIC
    gjxl_codec
    gjxl_gpu
)

target_link_libraries(
  gjxl_codestream
  PUBLIC
    gjxl_gpu_ops
)

install(
  TARGETS
    gjxl_gpu
    gjxl_gpu_butteraugli
    gjxl_gpu_ops
  EXPORT gjxlTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(
  DIRECTORY src/gpu
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*_internal.h" EXCLUDE
    PATTERN "*_profile.h" EXCLUDE
    PATTERN "*_test.h" EXCLUDE
    PATTERN "metal/kernels" EXCLUDE
)

if(GJXL_ENABLE_CUDA)
  add_library(
    gjxl_cuda
    STATIC

    src/gpu/cuda/cuda_backend.cpp
    src/gpu/cuda/cuda_ac_strategy.cpp
    src/gpu/cuda/cuda_ac_strategy_kernels.cu
    src/gpu/cuda/cuda_ac_group_kernels.cu
    src/gpu/cuda/cuda_aq.cpp
    src/gpu/cuda/cuda_aq_kernels.cu
    src/gpu/cuda/cuda_aq_exact.cpp
    src/gpu/cuda/cuda_aq_exact_kernels.cu
    src/gpu/cuda/cuda_input_preparation.cpp
    src/gpu/cuda/cuda_aq_resident.cpp
    src/gpu/cuda/cuda_aq_resident_kernels.cu
    src/gpu/cuda/cuda_butteraugli.cpp
    src/gpu/cuda/cuda_butteraugli_kernels.cu
    src/gpu/cuda/cuda_kernels.cu
    src/gpu/cuda/cuda_primitives.cpp
  )

  add_library(gjxl::cuda ALIAS gjxl_cuda)

  set_target_properties(
    gjxl_cuda
    PROPERTIES
      EXPORT_NAME cuda
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_EXTENSIONS NO
  )

  target_compile_features(
    gjxl_cuda
    PRIVATE
      cxx_std_20
  )

  target_link_libraries(
    gjxl_cuda
    PUBLIC
      gjxl_gpu_butteraugli
      gjxl_gpu_ops
    PRIVATE
      CUDA::cudart
  )

  target_link_libraries(
    gjxl_codestream
    PUBLIC
      gjxl_cuda
  )

  target_compile_definitions(
    gjxl_codestream
    PRIVATE
      GJXL_ENABLE_CUDA=1
  )

  install(
    TARGETS gjxl_cuda
    EXPORT gjxlTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
endif()

if(GJXL_ENABLE_METAL)

# metal-cpp
add_library(
  metal_cpp
  INTERFACE
)

target_include_directories(
  metal_cpp
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp"
)
# Apple frameworks
find_library(
  METAL_FRAMEWORK
  Metal
  REQUIRED
)

find_library(
  FOUNDATION_FRAMEWORK
  Foundation
  REQUIRED
)

find_library(
  COREGRAPHICS_FRAMEWORK
  CoreGraphics
  REQUIRED
)

# Metal shader compilation
find_program(
  XCRUN_EXECUTABLE
  xcrun
  REQUIRED
)

find_program(
  XXD_EXECUTABLE
  xxd
  REQUIRED
)

set(
  GJXL_METAL_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/metal"
)

set(
  GJXL_DCT_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/dct.metal"
)

set(
  GJXL_AC_STRATEGY_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/ac_strategy.metal"
)

set(
  GJXL_PRIMITIVES_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/primitives.metal"
)

set(
  GJXL_AQ_REDUCTION_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/aq_reduction.metal"
)

set(
  GJXL_AQ_RECONSTRUCTION_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/aq_reconstruction.metal"
)

set(
  GJXL_AQ_POSTPROCESS_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/aq_postprocess.metal"
)

set(
  GJXL_AQ_QUANTIZATION_HEADER
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/aq_quantization.h"
)

set(
  GJXL_BUTTERAUGLI_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/butteraugli.metal"
)

set(
  GJXL_DCT_BASIS_HEADER
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/dct_basis.h"
)

set(
  GJXL_DCT_METAL_IR
  "${GJXL_METAL_DIR}/dct.ir"
)

set(
  GJXL_AC_STRATEGY_METAL_IR
  "${GJXL_METAL_DIR}/ac_strategy.ir"
)

set(
  GJXL_PRIMITIVES_METAL_IR
  "${GJXL_METAL_DIR}/primitives.ir"
)

set(
  GJXL_AQ_REDUCTION_METAL_IR
  "${GJXL_METAL_DIR}/aq_reduction.ir"
)

set(
  GJXL_AQ_RECONSTRUCTION_METAL_IR
  "${GJXL_METAL_DIR}/aq_reconstruction.ir"
)

set(
  GJXL_AQ_POSTPROCESS_METAL_IR
  "${GJXL_METAL_DIR}/aq_postprocess.ir"
)

set(
  GJXL_BUTTERAUGLI_METAL_IR
  "${GJXL_METAL_DIR}/butteraugli.ir"
)

set(
  GJXL_METALLIB
  "${GJXL_METAL_DIR}/gjxl.metallib"
)

set(
  GJXL_EMBEDDED_METALLIB_INCLUDE
  "${GJXL_METAL_DIR}/gjxl_embedded_metallib.inc"
)

set(GJXL_METAL_PROFILING_FLAGS)
if(GJXL_ENABLE_METAL_PROFILING)
  list(APPEND GJXL_METAL_PROFILING_FLAGS
    -gline-tables-only
    -frecord-sources=yes
  )
endif()

# .metal -> .ir

add_custom_command(
  OUTPUT
    "${GJXL_DCT_METAL_IR}"

  COMMAND
    "${CMAKE_COMMAND}"
    -E make_directory
    "${GJXL_METAL_DIR}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -c
    ${GJXL_METAL_PROFILING_FLAGS}
    "${GJXL_DCT_METAL_SOURCE}"
    -o
    "${GJXL_DCT_METAL_IR}"

  DEPENDS
    "${GJXL_DCT_METAL_SOURCE}"
    "${GJXL_DCT_BASIS_HEADER}"

  COMMENT
    "Compiling Metal DCT shaders"

  VERBATIM
)

add_custom_command(
  OUTPUT
    "${GJXL_AC_STRATEGY_METAL_IR}"

  COMMAND
    "${CMAKE_COMMAND}"
    -E make_directory
    "${GJXL_METAL_DIR}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -c
    ${GJXL_METAL_PROFILING_FLAGS}
    "${GJXL_AC_STRATEGY_METAL_SOURCE}"
    -o
    "${GJXL_AC_STRATEGY_METAL_IR}"

  DEPENDS
    "${GJXL_AC_STRATEGY_METAL_SOURCE}"

  COMMENT
    "Compiling Metal AC-strategy shaders"

  VERBATIM
)

add_custom_command(
  OUTPUT
    "${GJXL_PRIMITIVES_METAL_IR}"

  COMMAND
    "${CMAKE_COMMAND}"
    -E make_directory
    "${GJXL_METAL_DIR}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -c
    ${GJXL_METAL_PROFILING_FLAGS}
    "${GJXL_PRIMITIVES_METAL_SOURCE}"
    -o
    "${GJXL_PRIMITIVES_METAL_IR}"

  DEPENDS
    "${GJXL_PRIMITIVES_METAL_SOURCE}"

  COMMENT
    "Compiling Metal primitive shaders"

  VERBATIM
)

add_custom_command(
  OUTPUT
    "${GJXL_AQ_REDUCTION_METAL_IR}"

  COMMAND
    "${CMAKE_COMMAND}"
    -E make_directory
    "${GJXL_METAL_DIR}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -c
    ${GJXL_METAL_PROFILING_FLAGS}
    -fmetal-math-mode=safe
    -fmetal-math-fp32-functions=precise
    -ffp-contract=off
    "${GJXL_AQ_REDUCTION_METAL_SOURCE}"
    -o
    "${GJXL_AQ_REDUCTION_METAL_IR}"

  DEPENDS
    "${GJXL_AQ_REDUCTION_METAL_SOURCE}"

  COMMENT
    "Compiling Metal AQ block-reduction shader"

  VERBATIM
)

add_custom_command(
  OUTPUT
    "${GJXL_AQ_RECONSTRUCTION_METAL_IR}"

  COMMAND
    "${CMAKE_COMMAND}"
    -E make_directory
    "${GJXL_METAL_DIR}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -c
    ${GJXL_METAL_PROFILING_FLAGS}
    -fmetal-math-mode=safe
    -ffp-contract=off
    "${GJXL_AQ_RECONSTRUCTION_METAL_SOURCE}"
    -o
    "${GJXL_AQ_RECONSTRUCTION_METAL_IR}"

  DEPENDS
    "${GJXL_AQ_RECONSTRUCTION_METAL_SOURCE}"
    "${GJXL_AQ_QUANTIZATION_HEADER}"

  COMMENT
    "Compiling Metal AQ reconstruction shaders"

  VERBATIM
)

add_custom_command(
  OUTPUT
    "${GJXL_AQ_POSTPROCESS_METAL_IR}"

  COMMAND
    "${CMAKE_COMMAND}"
    -E make_directory
    "${GJXL_METAL_DIR}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -c
    ${GJXL_METAL_PROFILING_FLAGS}
    -fmetal-math-mode=safe
    -fmetal-math-fp32-functions=precise
    -ffp-contract=off
    "${GJXL_AQ_POSTPROCESS_METAL_SOURCE}"
    -o
    "${GJXL_AQ_POSTPROCESS_METAL_IR}"

  DEPENDS
    "${GJXL_AQ_POSTPROCESS_METAL_SOURCE}"

  COMMENT
    "Compiling Metal AQ postprocess shaders"

  VERBATIM
)

add_custom_command(
  OUTPUT
    "${GJXL_BUTTERAUGLI_METAL_IR}"

  COMMAND
    "${CMAKE_COMMAND}"
    -E make_directory
    "${GJXL_METAL_DIR}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -c
    ${GJXL_METAL_PROFILING_FLAGS}
    -fmetal-math-mode=safe
    -fmetal-math-fp32-functions=precise
    -ffp-contract=off
    "${GJXL_BUTTERAUGLI_METAL_SOURCE}"
    -o
    "${GJXL_BUTTERAUGLI_METAL_IR}"

  DEPENDS
    "${GJXL_BUTTERAUGLI_METAL_SOURCE}"

  COMMENT
    "Compiling strict-math Metal Butteraugli shaders"

  VERBATIM
)

# .ir -> .metallib

add_custom_command(
  OUTPUT
    "${GJXL_METALLIB}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -o
    "${GJXL_METALLIB}"
    "${GJXL_DCT_METAL_IR}"
    "${GJXL_AC_STRATEGY_METAL_IR}"
    "${GJXL_PRIMITIVES_METAL_IR}"
    "${GJXL_AQ_REDUCTION_METAL_IR}"
    "${GJXL_AQ_RECONSTRUCTION_METAL_IR}"
    "${GJXL_AQ_POSTPROCESS_METAL_IR}"
    "${GJXL_BUTTERAUGLI_METAL_IR}"

  DEPENDS
    "${GJXL_DCT_METAL_IR}"
    "${GJXL_AC_STRATEGY_METAL_IR}"
    "${GJXL_PRIMITIVES_METAL_IR}"
    "${GJXL_AQ_REDUCTION_METAL_IR}"
    "${GJXL_AQ_RECONSTRUCTION_METAL_IR}"
    "${GJXL_AQ_POSTPROCESS_METAL_IR}"
    "${GJXL_BUTTERAUGLI_METAL_IR}"

  COMMENT
    "Linking gjxl.metallib"

  VERBATIM
)


add_custom_target(
  gjxl_metal_shaders
  DEPENDS
    "${GJXL_METALLIB}"
)

if(GJXL_ENABLE_METAL_PROFILING)
  set(
    GJXL_METALLIB_SYMBOLS
    "${GJXL_METAL_DIR}/gjxl.metallibsym"
  )

  add_custom_command(
    OUTPUT
      "${GJXL_METALLIB_SYMBOLS}"

    COMMAND
      "${XCRUN_EXECUTABLE}"
      -sdk macosx
      metal-dsymutil
      --flat
      -o
      "${GJXL_METALLIB_SYMBOLS}"
      "${GJXL_METALLIB}"

    DEPENDS
      "${GJXL_METALLIB}"

    COMMENT
      "Generating gjxl.metallib shader symbols"

    VERBATIM
  )

  add_custom_target(
    gjxl_metal_profile_symbols
    DEPENDS
      "${GJXL_METALLIB_SYMBOLS}"
  )
endif()

add_custom_command(
  OUTPUT
    "${GJXL_EMBEDDED_METALLIB_INCLUDE}"

  COMMAND
    "${XXD_EXECUTABLE}"
    -i
    -n gjxl_embedded_metallib
    "${GJXL_METALLIB}"
    "${GJXL_EMBEDDED_METALLIB_INCLUDE}"

  DEPENDS
    "${GJXL_METALLIB}"

  COMMENT
    "Embedding gjxl.metallib"

  VERBATIM
)

add_custom_target(
  gjxl_embedded_metal_shaders
  DEPENDS
    "${GJXL_EMBEDDED_METALLIB_INCLUDE}"
)

# Metal backend

add_library(
  gjxl_metal
  STATIC

  src/gpu/metal/metal_cpp_impl.cpp
  src/gpu/metal/metal_ac_strategy.cpp
  src/gpu/metal/metal_backend.cpp
  src/gpu/metal/metal_embedded_library.cpp
  src/gpu/metal/metal_aq_butteraugli.cpp
  src/gpu/metal/metal_aq_evaluation.cpp
  src/gpu/metal/metal_aq_postprocess.cpp
  src/gpu/metal/metal_aq_reconstruction.cpp
  src/gpu/metal/metal_butteraugli.cpp
  src/gpu/metal/metal_primitives.cpp
  src/gpu/metal/metal_submission.cpp
)

add_dependencies(
  gjxl_metal
  gjxl_embedded_metal_shaders
)

set_source_files_properties(
  src/gpu/metal/metal_embedded_library.cpp
  PROPERTIES
    OBJECT_DEPENDS "${GJXL_EMBEDDED_METALLIB_INCLUDE}"
)

target_include_directories(
  gjxl_metal
  PRIVATE
    "${GJXL_METAL_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp"
)

target_link_libraries(
  gjxl_metal

  PUBLIC
    gjxl_gpu
    gjxl_gpu_butteraugli
    gjxl_gpu_ops

  PRIVATE
    "${METAL_FRAMEWORK}"
    "${FOUNDATION_FRAMEWORK}"
    "${COREGRAPHICS_FRAMEWORK}"
)

target_link_libraries(
  gjxl_codestream
  PUBLIC
    gjxl_metal
)

install(
  TARGETS
    gjxl_metal
  EXPORT gjxlTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

target_compile_features(
  gjxl_metal
  PRIVATE
    cxx_std_20
)

target_compile_definitions(
  gjxl_codestream
  PRIVATE
    GJXL_ENABLE_METAL=1
)

endif()

if(GJXL_BUILD_TESTS)

  enable_testing()

  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )

  if(GJXL_ENABLE_METAL)
    add_test(
      NAME metal_profile_driver
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/metal_profile_test.py"
    )
    set_tests_properties(metal_profile_driver PROPERTIES LABELS cli)
  endif()

  find_program(GJXL_IMAGEMAGICK_EXECUTABLE NAMES magick)
  if(GJXL_IMAGEMAGICK_EXECUTABLE)
    add_test(
      NAME encode_image
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/encode_image_test.py"
        --wrapper "${CMAKE_CURRENT_SOURCE_DIR}/tools/encode_image.py"
        --encoder $<TARGET_FILE:gjxl_encode>
        --magick "${GJXL_IMAGEMAGICK_EXECUTABLE}"
    )
    set_tests_properties(encode_image PROPERTIES LABELS cli)
  else()
    message(STATUS
      "ImageMagick not found; optional normal-image wrapper test is unavailable")
  endif()

  add_executable(
    gjxl_codestream_conformance_test
    tests/codestream_conformance_test.cpp
  )

  target_link_libraries(
    gjxl_codestream_conformance_test
    PRIVATE
      gjxl_codestream
  )

  set(
    GJXL_PINNED_LIBJXL_REVISION
    e8ff09762481785938d8e4e01333ed3917571161
  )
  set(
    GJXL_PINNED_LIBJXL_BUILD
    "${CMAKE_CURRENT_BINARY_DIR}/pinned-libjxl"
  )
  set(GJXL_PINNED_DECODER_COMPILER_ARGUMENTS)
  if(WIN32)
    # The pinned libjxl revision references an MSVC overflow intrinsic that is
    # absent from some otherwise supported VS 2022 toolsets. clang-cl uses the
    # revision's builtin-overflow path and remains ABI-compatible with MSVC.
    find_program(GJXL_PINNED_CLANG_CL clang-cl)
    if(GJXL_PINNED_CLANG_CL)
      set(GJXL_PINNED_LIBJXL_BUILD
          "${CMAKE_CURRENT_BINARY_DIR}/pinned-libjxl-clang")
      list(APPEND GJXL_PINNED_DECODER_COMPILER_ARGUMENTS
        "-DGJXL_C_COMPILER=${GJXL_PINNED_CLANG_CL}"
        "-DGJXL_CXX_COMPILER=${GJXL_PINNED_CLANG_CL}")
    endif()
  endif()
  set(
    GJXL_PINNED_DJXL
    "${GJXL_PINNED_LIBJXL_BUILD}/tools/djxl${CMAKE_EXECUTABLE_SUFFIX}"
  )
  set(
    GJXL_PINNED_JXLINFO
    "${GJXL_PINNED_LIBJXL_BUILD}/tools/jxlinfo${CMAKE_EXECUTABLE_SUFFIX}"
  )

  add_custom_target(
    gjxl_pinned_decoder
    COMMAND
      "${CMAKE_COMMAND}"
      "-DGJXL_LIBJXL_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl"
      "-DGJXL_LIBJXL_BUILD=${GJXL_PINNED_LIBJXL_BUILD}"
      "-DGJXL_LIBJXL_REVISION=${GJXL_PINNED_LIBJXL_REVISION}"
      "-DGJXL_GENERATOR=${CMAKE_GENERATOR}"
      "-DGJXL_EXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}"
      ${GJXL_PINNED_DECODER_COMPILER_ARGUMENTS}
      -P
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildPinnedDjxl.cmake"
    COMMENT
      "Building pinned libjxl decoder at ${GJXL_PINNED_LIBJXL_REVISION}"
    VERBATIM
  )

  add_custom_target(
    codestream-conformance
    COMMAND
      gjxl_codestream_conformance_test
      --decoder "${GJXL_PINNED_DJXL}"
      --info "${GJXL_PINNED_JXLINFO}"
      --encoder $<TARGET_FILE:gjxl_encode>
      --sample "${CMAKE_CURRENT_SOURCE_DIR}/testdata/codestream_sample.pfm"
      --artifacts
        "${CMAKE_CURRENT_BINARY_DIR}/codestream-conformance-artifacts"
    DEPENDS
      gjxl_codestream_conformance_test
      gjxl_pinned_decoder
      gjxl_encode
    USES_TERMINAL
    VERBATIM
  )

  find_program(GJXL_SYSTEM_DJXL djxl)
  find_program(GJXL_SYSTEM_JXLINFO jxlinfo)
  if(GJXL_SYSTEM_DJXL AND GJXL_SYSTEM_JXLINFO)
    add_custom_target(
      codestream-smoke
      COMMAND
        gjxl_codestream_conformance_test
        --decoder "${GJXL_SYSTEM_DJXL}"
        --info "${GJXL_SYSTEM_JXLINFO}"
        --encoder $<TARGET_FILE:gjxl_encode>
        --sample "${CMAKE_CURRENT_SOURCE_DIR}/testdata/codestream_sample.pfm"
        --artifacts "${CMAKE_CURRENT_BINARY_DIR}/codestream-smoke-artifacts"
        --smoke
      DEPENDS
        gjxl_codestream_conformance_test
        gjxl_encode
      USES_TERMINAL
      VERBATIM
    )

    add_test(
      NAME codestream_conformance_smoke
      COMMAND
        gjxl_codestream_conformance_test
        --decoder "${GJXL_SYSTEM_DJXL}"
        --info "${GJXL_SYSTEM_JXLINFO}"
        --encoder $<TARGET_FILE:gjxl_encode>
        --sample "${CMAKE_CURRENT_SOURCE_DIR}/testdata/codestream_sample.pfm"
        --artifacts "${CMAKE_CURRENT_BINARY_DIR}/codestream-smoke-artifacts"
        --smoke
    )
    set_tests_properties(
      codestream_conformance_smoke
      PROPERTIES LABELS conformance
    )
  else()
    message(STATUS
      "System djxl/jxlinfo not found; codestream-smoke target is unavailable")
  endif()

  add_executable(
    gjxl_bit_writer_test
    tests/bit_writer_test.cpp
  )

  target_link_libraries(
    gjxl_bit_writer_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME bit_writer
    COMMAND gjxl_bit_writer_test
  )

  add_executable(
    gjxl_entropy_test
    tests/entropy_test.cpp
  )

  target_link_libraries(
    gjxl_entropy_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME entropy
    COMMAND gjxl_entropy_test
  )

  add_executable(
    gjxl_sections_test
    tests/sections_test.cpp
  )

  target_link_libraries(
    gjxl_sections_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME sections
    COMMAND gjxl_sections_test
  )

  add_executable(
    gjxl_dc_group_test
    tests/dc_group_test.cpp
  )

  target_link_libraries(
    gjxl_dc_group_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME dc_group
    COMMAND gjxl_dc_group_test
  )

  add_executable(
    gjxl_ac_group_test
    tests/ac_group_test.cpp
  )

  target_link_libraries(
    gjxl_ac_group_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME ac_group
    COMMAND gjxl_ac_group_test
  )

  add_executable(gjxl_coefficient_order_test tests/coefficient_order_test.cpp)
  target_link_libraries(gjxl_coefficient_order_test PRIVATE gjxl_codestream)
  add_test(NAME coefficient_order COMMAND gjxl_coefficient_order_test)

  add_executable(gjxl_ac_direct_tokenization_test tests/ac_direct_tokenization_test.cpp)
  target_link_libraries(gjxl_ac_direct_tokenization_test PRIVATE gjxl_codestream)
  add_test(NAME ac_direct_tokenization COMMAND gjxl_ac_direct_tokenization_test)

  add_executable(
    gjxl_block_context_map_test
    tests/block_context_map_test.cpp
  )

  target_link_libraries(
    gjxl_block_context_map_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME block_context_map
    COMMAND gjxl_block_context_map_test
  )

  add_executable(
    gjxl_codestream_encoder_test
    tests/codestream_encoder_test.cpp
  )

  target_link_libraries(
    gjxl_codestream_encoder_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME codestream_encoder
    COMMAND gjxl_codestream_encoder_test
  )

  add_executable(
    gjxl_codestream_workflow_test
    tests/codestream_workflow_test.cpp
  )

  target_link_libraries(
    gjxl_codestream_workflow_test
    PRIVATE
      gjxl_codestream
  )

  target_compile_definitions(
    gjxl_codestream_workflow_test
    PRIVATE
      GJXL_TEST_HAS_METAL=$<BOOL:${GJXL_ENABLE_METAL}>
  )

  add_test(
    NAME codestream_workflow
    COMMAND gjxl_codestream_workflow_test
  )

  add_executable(
    gjxl_codestream_batch_workflow_test
    tests/codestream_batch_workflow_test.cpp
  )

  target_link_libraries(
    gjxl_codestream_batch_workflow_test
    PRIVATE
      gjxl_codestream
  )

  target_compile_definitions(
    gjxl_codestream_batch_workflow_test
    PRIVATE
      GJXL_TEST_HAS_CUDA=$<BOOL:${GJXL_ENABLE_CUDA}>
  )

  add_test(
    NAME codestream_batch_workflow
    COMMAND gjxl_codestream_batch_workflow_test
  )

  add_executable(
    gjxl_rate_control_search_test
    tests/rate_control_search_test.cpp
  )

  target_link_libraries(
    gjxl_rate_control_search_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME rate_control_search
    COMMAND gjxl_rate_control_search_test
  )

  add_test(
    NAME rate_control_probe
    COMMAND
      gjxl_rate_control_probe
      --backend cpu
      --targets 1.0,1.2
      "${CMAKE_CURRENT_SOURCE_DIR}/testdata/codestream_sample.pfm"
  )

  add_test(
    NAME codestream_cli
    COMMAND
      "${CMAKE_COMMAND}"
      "-DGJXL_ENCODER=$<TARGET_FILE:gjxl_encode>"
      "-DGJXL_SAMPLE=${CMAKE_CURRENT_SOURCE_DIR}/testdata/codestream_sample.pfm"
      "-DGJXL_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/codestream-cli-test"
      "-DGJXL_HAS_METAL=${GJXL_ENABLE_METAL}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunCodestreamCliTest.cmake"
  )

  add_test(
    NAME codestream_sample_generated
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_codestream_sample.py"
      --output
      "${CMAKE_CURRENT_SOURCE_DIR}/testdata/codestream_sample.pfm"
      --check
  )

  add_executable(
    gjxl_c_api_image_conversion_test
    tests/c_api_image_conversion_test.cpp
  )

  target_link_libraries(
    gjxl_c_api_image_conversion_test
    PRIVATE
      gjxl_c_api_internal
  )

  add_test(
    NAME c_api_image_conversion
    COMMAND gjxl_c_api_image_conversion_test
  )

  add_executable(
    gjxl_c_api_test
    tests/c_api_test.c
  )

  set_target_properties(
    gjxl_c_api_test
    PROPERTIES
      LINKER_LANGUAGE CXX
  )

  target_compile_features(
    gjxl_c_api_test
    PRIVATE
      c_std_11
  )

  if(MSVC)
    target_compile_options(gjxl_c_api_test PRIVATE /W4 /WX /wd4127)
  else()
    target_compile_options(
      gjxl_c_api_test
      PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
  endif()

  target_link_libraries(
    gjxl_c_api_test
    PRIVATE
      gjxl_c
  )

  add_test(
    NAME c_api
    COMMAND gjxl_c_api_test
  )

  add_executable(
    gjxl_c_api_cpp_test
    tests/c_api_cpp_test.cpp
  )

  target_compile_features(
    gjxl_c_api_cpp_test
    PRIVATE
      cxx_std_20
  )

  if(MSVC)
    target_compile_options(gjxl_c_api_cpp_test PRIVATE /W4 /WX)
  else()
    target_compile_options(
      gjxl_c_api_cpp_test
      PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
  endif()

  target_link_libraries(
    gjxl_c_api_cpp_test
    PRIVATE
      gjxl_c
  )

  add_test(
    NAME c_api_cpp
    COMMAND gjxl_c_api_cpp_test
  )

  add_executable(
    gjxl_pfm_test
    tests/pfm_test.cpp
  )

  target_link_libraries(
    gjxl_pfm_test
    PRIVATE
      gjxl_pfm_io
  )

  add_test(
    NAME pfm
    COMMAND gjxl_pfm_test
  )

  add_executable(
    gjxl_butteraugli_primitives_test
    tests/butteraugli_primitives_test.cpp
  )

  target_link_libraries(
    gjxl_butteraugli_primitives_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME butteraugli_primitives
    COMMAND gjxl_butteraugli_primitives_test
  )

  add_executable(
    gjxl_butteraugli_stages_test
    tests/butteraugli_stages_test.cpp
  )

  target_link_libraries(
    gjxl_butteraugli_stages_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME butteraugli_stages
    COMMAND gjxl_butteraugli_stages_test
  )

  add_executable(
    gjxl_butteraugli_distance_test
    tests/butteraugli_distance_test.cpp
  )

  target_link_libraries(
    gjxl_butteraugli_distance_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME butteraugli_distance
    COMMAND gjxl_butteraugli_distance_test
  )

  add_executable(
    gjxl_core_geometry_test
    tests/core_geometry_test.cpp
  )

  target_link_libraries(
    gjxl_core_geometry_test
    PRIVATE
      gjxl_core
  )

  add_test(
    NAME core_geometry
    COMMAND gjxl_core_geometry_test
  )

  add_executable(
    gjxl_ac_strategy_grid_test
    tests/ac_strategy_grid_test.cpp
  )

  target_link_libraries(
    gjxl_ac_strategy_grid_test
    PRIVATE
      gjxl_core
  )

  add_test(
    NAME ac_strategy_grid
    COMMAND gjxl_ac_strategy_grid_test
  )

  add_executable(
    gjxl_fast_math_test
    tests/fast_math_test.cpp
  )

  target_link_libraries(
    gjxl_fast_math_test
    PRIVATE
      gjxl_core
  )

  add_test(
    NAME fast_math
    COMMAND gjxl_fast_math_test
  )

  add_executable(
    gjxl_dct_reference_test
    tests/dct_reference.cpp
    tests/dct_reference_test.cpp
  )

  target_link_libraries(
    gjxl_dct_reference_test
    PRIVATE
      gjxl_core
  )

  add_test(
    NAME dct_reference
    COMMAND gjxl_dct_reference_test
  )

  add_executable(
    gjxl_cpu_dct_test
    tests/cpu_dct_test.cpp
    tests/dct_reference.cpp
  )

  target_link_libraries(
    gjxl_cpu_dct_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME cpu_dct
    COMMAND gjxl_cpu_dct_test
  )

  add_executable(
    gjxl_ac_strategy_cost_test
    tests/ac_strategy_cost_test.cpp
  )

  target_link_libraries(
    gjxl_ac_strategy_cost_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME ac_strategy_cost
    COMMAND gjxl_ac_strategy_cost_test
  )

  add_executable(
    gjxl_ac_strategy_search_test
    tests/ac_strategy_search_test.cpp
  )

  target_link_libraries(
    gjxl_ac_strategy_search_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME ac_strategy_search
    COMMAND gjxl_ac_strategy_search_test
  )

  add_executable(
    gjxl_quantization_test
    tests/quantization_test.cpp
  )

  target_link_libraries(
    gjxl_quantization_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME quantization
    COMMAND gjxl_quantization_test
  )

  add_executable(
    gjxl_quantization_adjustment_test
    tests/quantization_adjustment_test.cpp
  )

  target_link_libraries(
    gjxl_quantization_adjustment_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME quantization_adjustment
    COMMAND gjxl_quantization_adjustment_test
  )

  add_executable(
    gjxl_maximum_error_test
    tests/maximum_error_test.cpp
  )

  target_link_libraries(
    gjxl_maximum_error_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME maximum_error
    COMMAND gjxl_maximum_error_test
  )

  add_executable(
    gjxl_dc_conversion_test
    tests/dct_reference.cpp
    tests/dc_conversion_test.cpp
  )

  target_link_libraries(
    gjxl_dc_conversion_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME dc_conversion
    COMMAND gjxl_dc_conversion_test
  )

  add_executable(
    gjxl_dc_quantization_test
    tests/dc_quantization_test.cpp
  )

  target_link_libraries(
    gjxl_dc_quantization_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME dc_quantization
    COMMAND gjxl_dc_quantization_test
  )

  add_executable(
    gjxl_adaptive_quantization_test
    tests/adaptive_quantization_test.cpp
  )

  target_link_libraries(
    gjxl_adaptive_quantization_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME adaptive_quantization
    COMMAND gjxl_adaptive_quantization_test
  )

  add_executable(
    gjxl_adaptive_quantization_loop_test
    tests/adaptive_quantization_loop_test.cpp
  )

  target_link_libraries(
    gjxl_adaptive_quantization_loop_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME adaptive_quantization_loop
    COMMAND gjxl_adaptive_quantization_loop_test
  )

  add_executable(
    gjxl_color_transform_test
    tests/color_transform_test.cpp
  )

  target_link_libraries(
    gjxl_color_transform_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME color_transform
    COMMAND gjxl_color_transform_test
  )

  add_executable(
    gjxl_quantization_pipeline_test
    tests/quantization_pipeline_test.cpp
  )

  target_link_libraries(
    gjxl_quantization_pipeline_test
    PRIVATE
      gjxl_codestream
  )

  add_test(
    NAME quantization_pipeline
    COMMAND gjxl_quantization_pipeline_test
  )

  add_executable(
    gjxl_gaborish_test
    tests/gaborish_test.cpp
  )

  target_link_libraries(
    gjxl_gaborish_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME gaborish
    COMMAND gjxl_gaborish_test
  )

  add_executable(
    gjxl_chroma_from_luma_test
    tests/chroma_from_luma_test.cpp
  )

  target_link_libraries(
    gjxl_chroma_from_luma_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME chroma_from_luma
    COMMAND gjxl_chroma_from_luma_test
  )

  add_executable(
    gjxl_epf_test
    tests/epf_test.cpp
  )

  target_link_libraries(
    gjxl_epf_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME epf
    COMMAND gjxl_epf_test
  )

  add_executable(
    gjxl_reconstruction_test
    tests/reconstruction_test.cpp
  )

  target_link_libraries(
    gjxl_reconstruction_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME reconstruction
    COMMAND gjxl_reconstruction_test
  )

  add_executable(
    gjxl_vardct_frame_test
    tests/vardct_frame_test.cpp
  )

  add_executable(gjxl_overwrite_array_test tests/overwrite_array_test.cpp)
  target_link_libraries(gjxl_overwrite_array_test PRIVATE gjxl_codec)
  add_test(NAME overwrite_array COMMAND gjxl_overwrite_array_test)

  target_link_libraries(
    gjxl_vardct_frame_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME vardct_frame
    COMMAND gjxl_vardct_frame_test
  )

  add_executable(
    gjxl_simple_codestream_profile_test
    tests/simple_codestream_profile_test.cpp
  )

  target_link_libraries(
    gjxl_simple_codestream_profile_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME simple_codestream_profile
    COMMAND gjxl_simple_codestream_profile_test
  )

  add_executable(
    gjxl_loop_filter_test
    tests/loop_filter_test.cpp
  )

  target_link_libraries(
    gjxl_loop_filter_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME loop_filter
    COMMAND gjxl_loop_filter_test
  )

  add_executable(
    gjxl_butteraugli_test
    tests/butteraugli_facade_test.cpp
  )

  target_link_libraries(
    gjxl_butteraugli_test
    PRIVATE
      gjxl_codec
  )

  add_test(
    NAME butteraugli
    COMMAND gjxl_butteraugli_test
  )

  if(GJXL_ENABLE_LIBJXL_REFERENCE)
    add_executable(
      gjxl_butteraugli_reference_test
      tests/butteraugli_test.cpp
    )

    target_link_libraries(
      gjxl_butteraugli_reference_test
      PRIVATE
        gjxl_codec
        gjxl_butteraugli_fixtures
        gjxl_butteraugli_oracle
    )

    target_compile_definitions(
      gjxl_butteraugli_reference_test
      PRIVATE
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
    )

    add_test(
      NAME butteraugli_reference
      COMMAND gjxl_butteraugli_reference_test
    )

    add_executable(
      gjxl_butteraugli_blur_differential_test
      tests/butteraugli_blur_differential_test.cpp
    )

    target_link_libraries(
      gjxl_butteraugli_blur_differential_test
      PRIVATE
        gjxl_codec
        gjxl_butteraugli_fixtures
        gjxl_butteraugli_oracle
    )

    target_compile_definitions(
      gjxl_butteraugli_blur_differential_test
      PRIVATE
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
    )

    add_test(
      NAME butteraugli_blur_differential
      COMMAND gjxl_butteraugli_blur_differential_test
    )

    add_executable(
      gjxl_butteraugli_frequency_differential_test
      tests/butteraugli_frequency_differential_test.cpp
    )

    target_link_libraries(
      gjxl_butteraugli_frequency_differential_test
      PRIVATE
        gjxl_codec
        gjxl_butteraugli_fixtures
        gjxl_butteraugli_oracle
    )

    target_compile_definitions(
      gjxl_butteraugli_frequency_differential_test
      PRIVATE
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
    )

    add_test(
      NAME butteraugli_frequency_differential
      COMMAND gjxl_butteraugli_frequency_differential_test
    )

    add_executable(
      gjxl_butteraugli_distance_differential_test
      tests/butteraugli_distance_differential_test.cpp
    )

    target_link_libraries(
      gjxl_butteraugli_distance_differential_test
      PRIVATE
        gjxl_codec
        gjxl_butteraugli_fixtures
        gjxl_butteraugli_oracle
    )

    target_compile_definitions(
      gjxl_butteraugli_distance_differential_test
      PRIVATE
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
        GJXL_NATIVE_DISTANCE_ORACLE_SCALAR=0
    )

    add_test(
      NAME butteraugli_distance_differential
      COMMAND gjxl_butteraugli_distance_differential_test
    )

    # Compile the internal scalar implementation directly so this executable
    # links only the pinned scalar libjxl oracle, without the production
    # facade's dispatched reference dependency.
    add_executable(
      gjxl_butteraugli_distance_scalar_differential_test
      tests/butteraugli_distance_differential_test.cpp
      src/codec/butteraugli_distance_internal.cpp
      src/codec/butteraugli_internal.cpp
    )

    target_link_libraries(
      gjxl_butteraugli_distance_scalar_differential_test
      PRIVATE
        gjxl_core
        gjxl_butteraugli_fixtures
        gjxl_butteraugli_oracle_scalar
    )

    target_compile_definitions(
      gjxl_butteraugli_distance_scalar_differential_test
      PRIVATE
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
        GJXL_NATIVE_DISTANCE_ORACLE_SCALAR=1
    )

    add_test(
      NAME butteraugli_distance_scalar_differential
      COMMAND gjxl_butteraugli_distance_scalar_differential_test
    )
  endif()

  if(GJXL_ENABLE_LIBJXL_REFERENCE)
    add_executable(
      gjxl_butteraugli_golden_generator
      tools/generate_butteraugli_goldens.cpp
    )
    target_link_libraries(
      gjxl_butteraugli_golden_generator
      PRIVATE
        gjxl_butteraugli_fixtures
        gjxl_butteraugli_oracle_scalar
    )

    add_test(
      NAME butteraugli_goldens_generated
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_butteraugli_goldens.py"
        --generator $<TARGET_FILE:gjxl_butteraugli_golden_generator>
        --output "${CMAKE_CURRENT_SOURCE_DIR}/tests/butteraugli_goldens_generated.h"
        --check
    )
  endif()

  add_test(
    NAME quantization_tables_generated
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_quantization_tables.py"
      --check
  )

  add_executable(
    gjxl_gpu_image_test
    tests/gpu_image_test.cpp
  )

  target_link_libraries(
    gjxl_gpu_image_test
    PRIVATE
      gjxl_gpu
  )

  add_test(
    NAME gpu_image
    COMMAND gjxl_gpu_image_test
  )

  if(GJXL_ENABLE_CUDA)
    add_executable(
      gjxl_cuda_backend_test
      tests/cuda_backend_test.cpp
      tests/dct_reference.cpp
    )

    target_link_libraries(
      gjxl_cuda_backend_test
      PRIVATE
        gjxl_cuda
    )

    target_include_directories(
      gjxl_cuda_backend_test
      PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    add_test(
      NAME cuda_backend
      COMMAND gjxl_cuda_backend_test
    )
    set_tests_properties(cuda_backend PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(
      gjxl_gpu_primitives_cuda_test
      tests/gpu_primitives_metal_test.cpp
    )

    target_compile_definitions(
      gjxl_gpu_primitives_cuda_test
      PRIVATE
        GJXL_TEST_CUDA=1
    )

    target_link_libraries(
      gjxl_gpu_primitives_cuda_test
      PRIVATE
        gjxl_codec
        gjxl_cuda
    )

    add_test(
      NAME cuda_gpu_primitives
      COMMAND gjxl_gpu_primitives_cuda_test
    )
    set_tests_properties(cuda_gpu_primitives PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(
      gjxl_cuda_aq_test
      tests/cuda_aq_test.cpp
    )

    target_link_libraries(
      gjxl_cuda_aq_test
      PRIVATE
        gjxl_codestream
        gjxl_cuda
    )

    add_test(
      NAME cuda_aq
      COMMAND gjxl_cuda_aq_test
    )
    target_include_directories(gjxl_cuda_aq_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_tests_properties(cuda_aq PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_quantization_adjustment_test
      tests/cuda_quantization_adjustment_test.cpp)
    target_link_libraries(gjxl_cuda_quantization_adjustment_test PRIVATE gjxl_codec gjxl_cuda)
    target_include_directories(gjxl_cuda_quantization_adjustment_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_quantization_adjustment COMMAND gjxl_cuda_quantization_adjustment_test)
    set_tests_properties(cuda_quantization_adjustment PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_resident_dct_test tests/cuda_resident_dct_test.cpp)
    target_link_libraries(gjxl_cuda_resident_dct_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_resident_dct_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_resident_dct COMMAND gjxl_cuda_resident_dct_test)
    set_tests_properties(cuda_resident_dct PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_resident_coefficients_test
      tests/cuda_resident_coefficients_test.cpp)
    target_link_libraries(gjxl_cuda_resident_coefficients_test PRIVATE gjxl_codec gjxl_cuda)
    target_include_directories(gjxl_cuda_resident_coefficients_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_resident_coefficients COMMAND gjxl_cuda_resident_coefficients_test)
    set_tests_properties(cuda_resident_coefficients PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_epf_test tests/cuda_epf_test.cpp)
    target_link_libraries(gjxl_cuda_epf_test PRIVATE gjxl_codec gjxl_cuda)
    target_include_directories(gjxl_cuda_epf_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_epf COMMAND gjxl_cuda_epf_test)
    set_tests_properties(cuda_epf PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(
      gjxl_ac_strategy_cuda_test
      tests/ac_strategy_metal_test.cpp
    )

    target_compile_definitions(
      gjxl_ac_strategy_cuda_test
      PRIVATE
        GJXL_TEST_CUDA=1
    )

    target_link_libraries(
      gjxl_ac_strategy_cuda_test
      PRIVATE
        gjxl_codec
        gjxl_cuda
    )

    add_test(
      NAME cuda_ac_strategy
      COMMAND gjxl_ac_strategy_cuda_test
    )
    set_tests_properties(cuda_ac_strategy PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_ac_strategy_cuda_search_test
      tests/ac_strategy_gpu_search_test.cpp)
    target_compile_definitions(gjxl_ac_strategy_cuda_search_test PRIVATE
      GJXL_TEST_CUDA=1)
    target_link_libraries(gjxl_ac_strategy_cuda_search_test PRIVATE
      gjxl_gpu_ops gjxl_cuda)
    add_test(NAME cuda_ac_strategy_search COMMAND gjxl_ac_strategy_cuda_search_test)
    set_tests_properties(cuda_ac_strategy_search PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_memory_pool_test tests/cuda_memory_pool_test.cpp)
    target_link_libraries(gjxl_cuda_memory_pool_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_memory_pool_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_memory_pool COMMAND gjxl_cuda_memory_pool_test)
    set_tests_properties(cuda_memory_pool PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_ac_packing_test tests/cuda_ac_packing_test.cpp)
    target_link_libraries(gjxl_cuda_ac_packing_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_ac_packing_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_ac_packing COMMAND gjxl_cuda_ac_packing_test)
    set_tests_properties(cuda_ac_packing PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_opsin_test tests/cuda_opsin_test.cpp)
    target_link_libraries(gjxl_cuda_opsin_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_opsin_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_opsin COMMAND gjxl_cuda_opsin_test)
    add_test(NAME cuda_opsin_tall COMMAND gjxl_cuda_opsin_test --tall-only)
    set_tests_properties(cuda_opsin cuda_opsin_tall PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_low_medium_test tests/cuda_low_medium_test.cpp)
    target_link_libraries(gjxl_cuda_low_medium_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_low_medium_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_low_medium COMMAND gjxl_cuda_low_medium_test)
    add_test(NAME cuda_low_medium_tall COMMAND gjxl_cuda_low_medium_test --tall-only)
    set_tests_properties(cuda_low_medium cuda_low_medium_tall PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_l2_final_test tests/cuda_l2_final_test.cpp)
    target_link_libraries(gjxl_cuda_l2_final_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_l2_final_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_l2_final COMMAND gjxl_cuda_l2_final_test)
    set_tests_properties(cuda_l2_final PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_final_cfl_test tests/cuda_final_cfl_test.cpp)
    target_link_libraries(gjxl_cuda_final_cfl_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_final_cfl_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_final_cfl COMMAND gjxl_cuda_final_cfl_test)
    set_tests_properties(cuda_final_cfl PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_initial_cfl_test tests/cuda_initial_cfl_test.cpp)
    target_link_libraries(gjxl_cuda_initial_cfl_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_initial_cfl_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_initial_cfl COMMAND gjxl_cuda_initial_cfl_test)
    set_tests_properties(cuda_initial_cfl PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_frequency_split_test tests/cuda_frequency_split_test.cpp)
    target_link_libraries(gjxl_cuda_frequency_split_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_frequency_split_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_frequency_split COMMAND gjxl_cuda_frequency_split_test)
    set_tests_properties(cuda_frequency_split PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gjxl_cuda_malta_test tests/cuda_malta_test.cpp)
    target_link_libraries(gjxl_cuda_malta_test PRIVATE gjxl_cuda)
    target_include_directories(gjxl_cuda_malta_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    add_test(NAME cuda_malta COMMAND gjxl_cuda_malta_test)
    add_test(NAME cuda_malta_tall COMMAND gjxl_cuda_malta_test --tall-only)
    set_tests_properties(cuda_malta cuda_malta_tall PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(
      gjxl_cuda_butteraugli_test
      tests/cuda_butteraugli_test.cpp
    )

    target_link_libraries(
      gjxl_cuda_butteraugli_test
      PRIVATE
        gjxl_codec
        gjxl_cuda
    )

    add_test(
      NAME cuda_butteraugli
      COMMAND gjxl_cuda_butteraugli_test
    )
    set_tests_properties(cuda_butteraugli PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  if(GJXL_ENABLE_METAL)

  add_executable(
    gjxl_gpu_primitives_metal_test
    tests/gpu_primitives_metal_test.cpp
  )

  target_link_libraries(
    gjxl_gpu_primitives_metal_test
    PRIVATE
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_gpu_primitives_metal_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_gpu_primitives_metal_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_gpu_primitives
    COMMAND gjxl_gpu_primitives_metal_test
  )

  add_executable(
    gjxl_butteraugli_device_operation_test
    tests/butteraugli_device_operation_test.cpp
  )

  target_link_libraries(
    gjxl_butteraugli_device_operation_test
    PRIVATE
      gjxl_codec
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_butteraugli_device_operation_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_butteraugli_device_operation_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_butteraugli_operation_contract
    COMMAND gjxl_butteraugli_device_operation_test
  )

  add_executable(
    gjxl_metal_butteraugli_test
    tests/metal_butteraugli_test.cpp
  )

  target_link_libraries(
    gjxl_metal_butteraugli_test
    PRIVATE
      gjxl_codec
      gjxl_butteraugli_fixtures
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_metal_butteraugli_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
      GJXL_METAL_BUTTERAUGLI_REFERENCE_FIXTURES=$<BOOL:${GJXL_ENABLE_LIBJXL_REFERENCE}>
  )

  if(GJXL_ENABLE_LIBJXL_REFERENCE)
    target_compile_definitions(
      gjxl_metal_butteraugli_test
      PRIVATE
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
    )
  endif()

  add_dependencies(
    gjxl_metal_butteraugli_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_butteraugli
    COMMAND gjxl_metal_butteraugli_test
  )

  add_executable(
    gjxl_metal_aq_evaluation_test
    tests/metal_aq_evaluation_test.cpp
  )

  target_link_libraries(
    gjxl_metal_aq_evaluation_test
    PRIVATE
      gjxl_codec
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_metal_aq_evaluation_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_metal_aq_evaluation_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_aq_evaluation
    COMMAND gjxl_metal_aq_evaluation_test
  )

  add_executable(
    gjxl_adaptive_quantization_gpu_policy_test
    tests/adaptive_quantization_gpu_policy_test.cpp
  )

  target_link_libraries(
    gjxl_adaptive_quantization_gpu_policy_test
    PRIVATE
      gjxl_codec
      gjxl_codestream
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_adaptive_quantization_gpu_policy_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_adaptive_quantization_gpu_policy_test
    gjxl_metal_shaders
  )

  add_test(
    NAME adaptive_quantization_gpu_policy
    COMMAND gjxl_adaptive_quantization_gpu_policy_test
  )

  add_executable(
    gjxl_metal_aq_reconstruction_test
    tests/dct_reference.cpp
    tests/metal_aq_reconstruction_test.cpp
  )

  target_link_libraries(
    gjxl_metal_aq_reconstruction_test
    PRIVATE
      gjxl_codec
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_metal_aq_reconstruction_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_metal_aq_reconstruction_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_aq_reconstruction
    COMMAND gjxl_metal_aq_reconstruction_test
  )

  add_executable(
    gjxl_metal_aq_postprocess_test
    tests/metal_aq_postprocess_test.cpp
  )

  target_link_libraries(
    gjxl_metal_aq_postprocess_test
    PRIVATE
      gjxl_codec
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_metal_aq_postprocess_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_metal_aq_postprocess_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_aq_postprocess
    COMMAND gjxl_metal_aq_postprocess_test
  )

  add_executable(
    gjxl_dct_test
    tests/dct_reference.cpp
    tests/dct_test.cpp
  )

  target_link_libraries(
    gjxl_dct_test
    PRIVATE
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_dct_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_dct_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_dct
    COMMAND gjxl_dct_test
  )

  add_executable(
    gjxl_ac_strategy_metal_test
    tests/ac_strategy_metal_test.cpp
  )

  target_link_libraries(
    gjxl_ac_strategy_metal_test
    PRIVATE
      gjxl_codec
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_ac_strategy_metal_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_ac_strategy_metal_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_ac_strategy
    COMMAND gjxl_ac_strategy_metal_test
  )

  add_executable(
    gjxl_ac_strategy_gpu_search_test
    tests/ac_strategy_gpu_search_test.cpp
  )

  target_link_libraries(
    gjxl_ac_strategy_gpu_search_test
    PRIVATE
      gjxl_gpu_ops
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_ac_strategy_gpu_search_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_ac_strategy_gpu_search_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_ac_strategy_search
    COMMAND gjxl_ac_strategy_gpu_search_test
  )

  add_executable(
    gjxl_quantization_gpu_pipeline_test
    tests/quantization_gpu_pipeline_test.cpp
  )

  target_link_libraries(
    gjxl_quantization_gpu_pipeline_test
    PRIVATE
      gjxl_codestream
      gjxl_gpu_ops
      gjxl_metal
  )

  target_compile_definitions(
    gjxl_quantization_gpu_pipeline_test
    PRIVATE
      GJXL_METALLIB_PATH="${GJXL_METALLIB}"
  )

  add_dependencies(
    gjxl_quantization_gpu_pipeline_test
    gjxl_metal_shaders
  )

  add_test(
    NAME metal_quantization_pipeline
    COMMAND gjxl_quantization_gpu_pipeline_test
  )

  endif()

  add_test(
    NAME codec_install_consumer
    COMMAND
      "${CMAKE_COMMAND}"
      "-DGJXL_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
      "-DGJXL_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DGJXL_TEST_CONFIG=$<CONFIG>"
      "-DGJXL_TEST_GENERATOR=${CMAKE_GENERATOR}"
      "-DGJXL_TEST_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
      "-DGJXL_TEST_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DGJXL_TEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DGJXL_TEST_RC_COMPILER=${CMAKE_RC_COMPILER}"
      "-DGJXL_TEST_MT=${CMAKE_MT}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_installed_consumer.cmake"
  )
endif()

if(GJXL_BUILD_BENCHMARKS)

  add_executable(
    gjxl_pfm_benchmark
    benchmarks/pfm_benchmark.cpp
  )

  target_link_libraries(
    gjxl_pfm_benchmark
    PRIVATE
      gjxl_pfm_io
  )

  if(GJXL_ENABLE_CUDA)
    add_executable(
      gjxl_cuda_encoding_benchmark
      benchmarks/cuda_encoding_benchmark.cpp
    )

    target_link_libraries(
      gjxl_cuda_encoding_benchmark
      PRIVATE
        gjxl_codestream
        gjxl_cuda
        gjxl_pfm_io
        CUDA::cudart
    )

    if(GJXL_BUILD_TESTS)
      add_test(
        NAME cuda_encoding_benchmark_cli
        COMMAND
          gjxl_cuda_encoding_benchmark
          --workload synthetic_128x96
          --gpu-aq maximum-throughput
          --gpu-only
          --warmups 0
          --samples 1
      )
      set_tests_properties(
        cuda_encoding_benchmark_cli
        PROPERTIES
          LABELS "benchmark;cli;cuda"
          SKIP_RETURN_CODE 77
      )
    endif()
  endif()

  if(GJXL_ENABLE_METAL)
    add_executable(
      gjxl_metal_butteraugli_benchmark
      benchmarks/metal_butteraugli_benchmark.cpp
    )

    target_link_libraries(
      gjxl_metal_butteraugli_benchmark
      PRIVATE
        gjxl_codec
        gjxl_butteraugli_fixtures
        gjxl_metal
    )

    target_compile_definitions(
      gjxl_metal_butteraugli_benchmark
      PRIVATE
        GJXL_METALLIB_PATH="${GJXL_METALLIB}"
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
    )

    add_dependencies(
      gjxl_metal_butteraugli_benchmark
      gjxl_metal_shaders
    )
  endif()

  if(GJXL_ENABLE_LIBJXL_REFERENCE)
    add_executable(
      gjxl_butteraugli_benchmark
      benchmarks/butteraugli_benchmark.cpp
    )

    target_link_libraries(
      gjxl_butteraugli_benchmark
      PRIVATE
        gjxl_codec
        gjxl_butteraugli_fixtures
        gjxl_butteraugli_oracle
    )

    target_compile_definitions(
      gjxl_butteraugli_benchmark
      PRIVATE
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
    )
  endif()

  if(GJXL_ENABLE_METAL)
    add_executable(
      gjxl_encoding_benchmark
      benchmarks/encoding_benchmark.cpp
    )

    target_link_libraries(
      gjxl_encoding_benchmark
      PRIVATE
        gjxl_codestream
        gjxl_metal
        gjxl_pfm_io
    )

    target_compile_definitions(
      gjxl_encoding_benchmark
      PRIVATE
        GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
    )

    add_dependencies(
      gjxl_encoding_benchmark
      gjxl_embedded_metal_shaders
    )

    if(GJXL_BUILD_TESTS)
      add_test(
        NAME encoding_benchmark_cli
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/encoding_benchmark_cli_test.py"
          --benchmark $<TARGET_FILE:gjxl_encoding_benchmark>
          --metallib "${GJXL_METALLIB}"
      )
      set_tests_properties(
        encoding_benchmark_cli
        PROPERTIES LABELS "benchmark;cli;metal"
      )
    endif()
  endif()

  add_executable(
    gjxl_image_batch_benchmark
    benchmarks/image_batch_benchmark.cpp
  )

  target_link_libraries(
    gjxl_image_batch_benchmark
    PRIVATE
      gjxl_codestream
  )

  target_compile_definitions(
    gjxl_image_batch_benchmark
    PRIVATE
      GJXL_BENCHMARK_HAS_METAL=$<BOOL:${GJXL_ENABLE_METAL}>
  )

  if(GJXL_ENABLE_METAL)
    add_dependencies(
      gjxl_image_batch_benchmark
      gjxl_embedded_metal_shaders
    )

    add_executable(
      gjxl_dct_benchmark
      benchmarks/dct_benchmark.cpp
    )

    target_link_libraries(
      gjxl_dct_benchmark
      PRIVATE
        gjxl_metal
    )

    target_compile_definitions(
      gjxl_dct_benchmark
      PRIVATE
        GJXL_METALLIB_PATH="${GJXL_METALLIB}"
    )

    add_dependencies(
      gjxl_dct_benchmark
      gjxl_metal_shaders
    )

    add_executable(
      gjxl_ac_strategy_benchmark
      benchmarks/ac_strategy_benchmark.cpp
    )

    target_link_libraries(
      gjxl_ac_strategy_benchmark
      PRIVATE
        gjxl_codec
        gjxl_metal
    )

    target_compile_definitions(
      gjxl_ac_strategy_benchmark
      PRIVATE
        GJXL_METALLIB_PATH="${GJXL_METALLIB}"
    )

    add_dependencies(
      gjxl_ac_strategy_benchmark
      gjxl_metal_shaders
    )

    add_executable(
      gjxl_ac_strategy_search_benchmark
      benchmarks/ac_strategy_search_benchmark.cpp
    )

    target_link_libraries(
      gjxl_ac_strategy_search_benchmark
      PRIVATE
        gjxl_gpu_ops
        gjxl_metal
    )

    target_compile_definitions(
      gjxl_ac_strategy_search_benchmark
      PRIVATE
        GJXL_METALLIB_PATH="${GJXL_METALLIB}"
    )

    add_dependencies(
      gjxl_ac_strategy_search_benchmark
      gjxl_metal_shaders
    )
  endif()

endif()
