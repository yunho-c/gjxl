# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

cmake_minimum_required(VERSION 3.24)
project(gjxl_libjxl_comparison LANGUAGES CXX)

set(GJXL_LIBJXL_SOURCE "" CACHE PATH "Pinned libjxl source directory")
set(GJXL_LIBJXL_BUILD "" CACHE PATH "Pinned libjxl build directory")
set(GJXL_LIBJXL_REVISION "" CACHE STRING "Expected pinned libjxl revision")

foreach(required IN ITEMS GJXL_LIBJXL_SOURCE GJXL_LIBJXL_BUILD
                          GJXL_LIBJXL_REVISION)
  if("${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required ${required}")
  endif()
endforeach()

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${GJXL_LIBJXL_SOURCE}" rev-parse HEAD
  RESULT_VARIABLE revision_result
  OUTPUT_VARIABLE actual_revision
  ERROR_VARIABLE revision_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT revision_result EQUAL 0)
  message(FATAL_ERROR "Unable to inspect pinned libjxl: ${revision_error}")
endif()
if(NOT actual_revision STREQUAL GJXL_LIBJXL_REVISION)
  message(FATAL_ERROR
    "libjxl revision ${actual_revision} does not match ${GJXL_LIBJXL_REVISION}")
endif()

find_library(
  GJXL_LIBJXL_LIBRARY
  NAMES jxl
  PATHS "${GJXL_LIBJXL_BUILD}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(
  GJXL_LIBJXL_THREADS_LIBRARY
  NAMES jxl_threads
  PATHS "${GJXL_LIBJXL_BUILD}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)

add_executable(
  gjxl_libjxl_comparison_benchmark
  libjxl_comparison_benchmark.cpp
)
target_compile_features(gjxl_libjxl_comparison_benchmark PRIVATE cxx_std_20)
target_compile_options(
  gjxl_libjxl_comparison_benchmark
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
target_compile_definitions(
  gjxl_libjxl_comparison_benchmark
  PRIVATE GJXL_LIBJXL_REVISION="${GJXL_LIBJXL_REVISION}"
)
target_include_directories(
  gjxl_libjxl_comparison_benchmark
  PRIVATE
    "${GJXL_LIBJXL_SOURCE}/lib/include"
    "${GJXL_LIBJXL_BUILD}/lib/include"
)
target_link_libraries(
  gjxl_libjxl_comparison_benchmark
  PRIVATE
    "${GJXL_LIBJXL_LIBRARY}"
    "${GJXL_LIBJXL_THREADS_LIBRARY}"
)
set_target_properties(
  gjxl_libjxl_comparison_benchmark
  PROPERTIES
    BUILD_RPATH "${GJXL_LIBJXL_BUILD}/lib"
)
