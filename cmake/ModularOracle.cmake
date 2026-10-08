# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
# Included only under GJXL_BUILD_TESTS. No target here is installed/exported.
set(GJXL_MODULAR_ORACLE_BUILD "" CACHE PATH "Pinned libjxl build for Modular oracle tests")
if(NOT GJXL_MODULAR_ORACLE_BUILD)
  return()
endif()
file(STRINGS "${GJXL_MODULAR_ORACLE_BUILD}/CMakeCache.txt" modular_oracle_home
  REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" modular_oracle_source "${modular_oracle_home}")
execute_process(COMMAND git -C "${modular_oracle_source}" rev-parse HEAD
  OUTPUT_VARIABLE modular_oracle_revision OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE modular_oracle_revision_result)
execute_process(COMMAND git -C "${modular_oracle_source}" diff --quiet HEAD --
  RESULT_VARIABLE modular_oracle_dirty)
if(NOT modular_oracle_revision_result EQUAL 0 OR NOT modular_oracle_dirty EQUAL 0 OR
   NOT modular_oracle_revision STREQUAL GJXL_PINNED_LIBJXL_REVISION)
  message(FATAL_ERROR "Modular oracle requires a clean pinned libjxl source/build")
endif()
if(WIN32)
  set(modular_oracle_prefix "")
  set(modular_oracle_archive ".lib")
  set(modular_oracle_link ".lib")
else()
  set(modular_oracle_prefix "lib")
  set(modular_oracle_archive ".a")
  set(modular_oracle_link "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
add_library(gjxl_modular_reference STATIC tests/modular_reference.cpp)
target_include_directories(gjxl_modular_reference PUBLIC
  "${modular_oracle_source}" "${modular_oracle_source}/lib/include"
  "${GJXL_MODULAR_ORACLE_BUILD}/lib/include"
  "${modular_oracle_source}/third_party/highway")
target_compile_features(gjxl_modular_reference PUBLIC cxx_std_20)
target_link_libraries(gjxl_modular_reference PUBLIC
  "${GJXL_MODULAR_ORACLE_BUILD}/lib/${modular_oracle_prefix}jxl-internal${modular_oracle_archive}"
  "${GJXL_MODULAR_ORACLE_BUILD}/lib/${modular_oracle_prefix}jxl${modular_oracle_link}"
  "${GJXL_MODULAR_ORACLE_BUILD}/third_party/highway/${modular_oracle_prefix}hwy${modular_oracle_archive}"
  "${GJXL_MODULAR_ORACLE_BUILD}/lib/${modular_oracle_prefix}jxl_cms${modular_oracle_link}"
  "${GJXL_MODULAR_ORACLE_BUILD}/third_party/brotli/${modular_oracle_prefix}brotlienc${modular_oracle_link}"
  "${GJXL_MODULAR_ORACLE_BUILD}/third_party/brotli/${modular_oracle_prefix}brotlidec${modular_oracle_link}"
  "${GJXL_MODULAR_ORACLE_BUILD}/third_party/brotli/${modular_oracle_prefix}brotlicommon${modular_oracle_link}")
add_executable(gjxl_modular_oracle_test tests/modular_oracle_test.cpp)
target_link_libraries(gjxl_modular_oracle_test PRIVATE gjxl_codestream gjxl_modular_reference)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # As for the DC oracle: pinned internal archives omit RTTI.
  target_compile_options(gjxl_modular_reference PRIVATE -fno-sanitize=vptr)
  target_compile_options(gjxl_modular_oracle_test PRIVATE -fno-sanitize=vptr)
endif()
add_test(NAME modular_oracle COMMAND gjxl_modular_oracle_test)
add_executable(gjxl_modular_conformance_test tests/modular_conformance_test.cpp)
target_link_libraries(gjxl_modular_conformance_test PRIVATE gjxl_codestream gjxl_modular_reference)
add_test(NAME modular_conformance COMMAND gjxl_modular_conformance_test
  "${CMAKE_CURRENT_BINARY_DIR}/modular-conformance")
if(WIN32)
  set_tests_properties(modular_oracle modular_conformance PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
endif()
