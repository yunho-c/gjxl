# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

cmake_minimum_required(VERSION 3.25)

project(
  gjxl
  VERSION 0.0.1
  LANGUAGES CXX
)

# TEMP
if(NOT APPLE)
  message(FATAL_ERROR
    "gjxl currently requires macOS")
endif()

enable_language(OBJCXX)

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
  ON
)

if(GJXL_ENABLE_LIBJXL_REFERENCE)
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
      gjxl_butteraugli_fixtures
      STATIC
      tests/butteraugli_fixtures.cpp
    )
    target_include_directories(
      gjxl_butteraugli_fixtures
      PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/lib/include"
    )
    target_compile_features(
      gjxl_butteraugli_fixtures
      PUBLIC
        cxx_std_20
    )

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

# C++ common interface
add_library(
  gjxl_core
  INTERFACE
)

target_compile_features(
  gjxl_core
  INTERFACE
    cxx_std_20
)

target_include_directories(
  gjxl_core
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
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
  src/codec/dct.cpp
  src/codec/epf.cpp
  src/codec/gaborish.cpp
  src/codec/loop_filter.cpp
  src/codec/quantization.cpp
  src/codec/quantization_pipeline.cpp
  src/codec/reconstruction.cpp
)

target_link_libraries(
  gjxl_codec
  PUBLIC
    gjxl_core
)

if(GJXL_ENABLE_LIBJXL_REFERENCE)
  target_compile_definitions(
    gjxl_codec
    PRIVATE
      GJXL_ENABLE_LIBJXL_REFERENCE=1
  )

  target_include_directories(
    gjxl_codec
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl"
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/lib/include"
  )

  target_link_libraries(
    gjxl_codec
    PRIVATE
      gjxl_butteraugli_reference
  )
else()
  target_compile_definitions(
    gjxl_codec
    PRIVATE
      GJXL_ENABLE_LIBJXL_REFERENCE=0
  )
endif()

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

set(
  GJXL_METAL_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/metal"
)

set(
  GJXL_DCT_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/dct.metal"
)

set(
  GJXL_PRIMITIVES_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/primitives.metal"
)

set(
  GJXL_AQ_CONTRACT_METAL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/aq_contract.metal"
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
  GJXL_DCT_BASIS_HEADER
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/dct_basis.h"
)

set(
  GJXL_DCT_METAL_IR
  "${GJXL_METAL_DIR}/dct.ir"
)

set(
  GJXL_PRIMITIVES_METAL_IR
  "${GJXL_METAL_DIR}/primitives.ir"
)

set(
  GJXL_AQ_CONTRACT_METAL_IR
  "${GJXL_METAL_DIR}/aq_contract.ir"
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
  GJXL_METALLIB
  "${GJXL_METAL_DIR}/gjxl.metallib"
)

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
    "${GJXL_AQ_CONTRACT_METAL_IR}"

  COMMAND
    "${CMAKE_COMMAND}"
    -E make_directory
    "${GJXL_METAL_DIR}"

  COMMAND
    "${XCRUN_EXECUTABLE}"
    -sdk macosx
    metal
    -c
    "${GJXL_AQ_CONTRACT_METAL_SOURCE}"
    -o
    "${GJXL_AQ_CONTRACT_METAL_IR}"

  DEPENDS
    "${GJXL_AQ_CONTRACT_METAL_SOURCE}"
    "${GJXL_AQ_QUANTIZATION_HEADER}"

  COMMENT
    "Compiling Metal AQ contract probe"

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
    "${GJXL_PRIMITIVES_METAL_IR}"
    "${GJXL_AQ_CONTRACT_METAL_IR}"
    "${GJXL_AQ_RECONSTRUCTION_METAL_IR}"
    "${GJXL_AQ_POSTPROCESS_METAL_IR}"

  DEPENDS
    "${GJXL_DCT_METAL_IR}"
    "${GJXL_PRIMITIVES_METAL_IR}"
    "${GJXL_AQ_CONTRACT_METAL_IR}"
    "${GJXL_AQ_RECONSTRUCTION_METAL_IR}"
    "${GJXL_AQ_POSTPROCESS_METAL_IR}"

  COMMENT
    "Linking gjxl.metallib"

  VERBATIM
)


add_custom_target(
  gjxl_metal_shaders
  DEPENDS
    "${GJXL_METALLIB}"
)

# Metal backend

add_library(
  gjxl_metal
  STATIC

  src/gpu/metal/metal_cpp_impl.cpp
  src/gpu/metal/metal_backend.cpp
  src/gpu/metal/metal_aq_evaluation.cpp
  src/gpu/metal/metal_aq_postprocess.cpp
  src/gpu/metal/metal_aq_reconstruction.cpp
  src/gpu/metal/metal_primitives.cpp
  src/gpu/metal/metal_submission.cpp
)

add_dependencies(
  gjxl_metal
  gjxl_metal_shaders
)

target_link_libraries(
  gjxl_metal

  PUBLIC
    gjxl_gpu

  PRIVATE
    metal_cpp
    "${METAL_FRAMEWORK}"
    "${FOUNDATION_FRAMEWORK}"
    "${COREGRAPHICS_FRAMEWORK}"
)

target_compile_features(
  gjxl_metal
  PRIVATE
    cxx_std_20
)

if(GJXL_BUILD_TESTS)

  enable_testing()

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
      gjxl_codec
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

  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )

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

endif()

if(GJXL_BUILD_BENCHMARKS)

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

  add_executable(
    gjxl_quantization_benchmark
    benchmarks/quantization_benchmark.cpp
  )

  target_link_libraries(
    gjxl_quantization_benchmark
    PRIVATE
      gjxl_codec
  )

  target_compile_definitions(
    gjxl_quantization_benchmark
    PRIVATE
      GJXL_FLOWER_PPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl/testdata/jxl/flower/flower_small.rgb.depth8.ppm"
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

endif()
