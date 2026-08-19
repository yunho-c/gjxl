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
  ON
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

