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
  src/codec/chroma_from_luma.cpp
  src/codec/convolution.cpp
  src/codec/dc_conversion.cpp
  src/codec/dct.cpp
  src/codec/epf.cpp
  src/codec/gaborish.cpp
  src/codec/loop_filter.cpp
  src/codec/quantization.cpp
  src/codec/reconstruction.cpp
)

target_link_libraries(
  gjxl_codec
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
  GJXL_DCT_BASIS_HEADER
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels/dct_basis.h"
)

set(
  GJXL_DCT_METAL_IR
  "${GJXL_METAL_DIR}/dct.ir"
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

  DEPENDS
    "${GJXL_DCT_METAL_IR}"

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
)

add_dependencies(
  gjxl_metal
  gjxl_metal_shaders
)

target_link_libraries(
  gjxl_metal

  PUBLIC
    gjxl_core

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

  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )

  add_test(
    NAME quantization_tables_generated
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_quantization_tables.py"
      --check
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
