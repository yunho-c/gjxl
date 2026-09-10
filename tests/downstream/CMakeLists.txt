# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

cmake_minimum_required(VERSION 3.25)

project(gjxl_downstream_consumer LANGUAGES C CXX)

if(GJXL_TEST_C_ABI_ONLY)
  find_package(gjxl CONFIG REQUIRED COMPONENTS c)
  # The same C ABI encode/free fixture is also valid C++.
  set_source_files_properties(c_api_consumer.c PROPERTIES LANGUAGE CXX)
  add_executable(gjxl_c_abi_consumer c_api_consumer.c)
  target_link_libraries(gjxl_c_abi_consumer PRIVATE gjxl::c)
  return()
endif()

find_package(gjxl CONFIG REQUIRED COMPONENTS c codec codestream)

# Installed templates and inline methods must work without preceding includes
# or access to src/, generated build headers, or metal-cpp. The install test
# copies the reviewed manifest alongside this fixture before configuring it.
if(GJXL_CHECK_INSTALLED_HEADERS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/InstalledHeaders.cmake")
  set(header_sources)
  foreach(header IN LISTS GJXL_INSTALLED_HEADERS)
    set(header_source "${CMAKE_CURRENT_BINARY_DIR}/header-checks/${header}.cpp")
    file(WRITE "${header_source}" "#include <${header}>\n")
    list(APPEND header_sources "${header_source}")
  endforeach()
  foreach(header IN LISTS GJXL_PUBLIC_C_HEADERS)
    set(header_source "${CMAKE_CURRENT_BINARY_DIR}/header-checks/${header}.c")
    file(WRITE "${header_source}" "#include <${header}>\n")
    list(APPEND header_sources "${header_source}")
  endforeach()
  add_library(gjxl_installed_headers OBJECT ${header_sources})
  target_link_libraries(gjxl_installed_headers PRIVATE gjxl::c gjxl::codestream)
  target_compile_features(gjxl_installed_headers PRIVATE c_std_11)
  target_compile_options(gjxl_installed_headers PRIVATE -Wall -Wextra -Wpedantic -Werror)
  # Combining translation units or pre-including other headers would hide a
  # missing dependency in the installed interface.
  set_target_properties(gjxl_installed_headers PROPERTIES
    UNITY_BUILD OFF DISABLE_PRECOMPILE_HEADERS ON
    NO_SYSTEM_FROM_IMPORTED ON
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
endif()

add_executable(
  gjxl_codec_consumer
  butteraugli_consumer.cpp
)

target_link_libraries(
  gjxl_codec_consumer
  PRIVATE
    gjxl::codec
)

add_executable(
  gjxl_codestream_consumer
  codestream_consumer.cpp
)

target_link_libraries(
  gjxl_codestream_consumer
  PRIVATE
    gjxl::codestream
)

add_executable(
  gjxl_c_consumer
  c_api_consumer.c
)

set_target_properties(
  gjxl_c_consumer
  PROPERTIES
    LINKER_LANGUAGE CXX
)

target_compile_features(
  gjxl_c_consumer
  PRIVATE
    c_std_11
)

target_compile_options(
  gjxl_c_consumer
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(
  gjxl_c_consumer
  PRIVATE
    gjxl::c
)

add_executable(gjxl_domain_consumer execution_domain_consumer.cpp legacy_context_consumer.c legacy_domain_consumer.c)
target_link_libraries(gjxl_domain_consumer PRIVATE gjxl::c gjxl::codestream)
target_compile_features(gjxl_domain_consumer PRIVATE c_std_11)
target_compile_options(gjxl_domain_consumer PRIVATE -Wall -Wextra -Wpedantic -Werror)
