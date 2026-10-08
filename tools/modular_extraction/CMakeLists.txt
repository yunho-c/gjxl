# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)
project(gjxl_modular_extraction LANGUAGES C CXX)
find_package(gjxl CONFIG REQUIRED COMPONENTS codestream)
file(TO_CMAKE_PATH "${GJXL_SOURCE_DIR}" GJXL_SOURCE_DIR)
if(NOT IS_DIRECTORY "${GJXL_SOURCE_DIR}/tests")
  message(FATAL_ERROR "GJXL_SOURCE_DIR must name the matching source checkout")
endif()
add_executable(modular_extraction_capture capture.cpp "${GJXL_SOURCE_DIR}/src/io/pfm.cpp")
target_compile_features(modular_extraction_capture PRIVATE cxx_std_20)
target_include_directories(modular_extraction_capture PRIVATE
  "${GJXL_SOURCE_DIR}/src" "${GJXL_SOURCE_DIR}/tests")
target_link_libraries(modular_extraction_capture PRIVATE gjxl::codestream)
